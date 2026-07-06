import os

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

NUM_CHECKPOINTS = 8
CHECKPOINT_STATE_START = 0
CHECKPOINT_STATE_DIM = 3 * NUM_CHECKPOINTS
CHECKPOINT_ACTION_DIM = 3 * NUM_CHECKPOINTS
PATH_GLOBAL_ACTION_DIM = 6
WIND_ACTION_START = CHECKPOINT_ACTION_DIM + PATH_GLOBAL_ACTION_DIM
WIND_ACTION_DIM = 17
GLOBAL_ACTION_START = CHECKPOINT_ACTION_DIM
IDX_ACT_LENGTH = GLOBAL_ACTION_START
IDX_ACT_RADIUS = IDX_ACT_LENGTH + 1
IDX_ACT_VERTICAL = IDX_ACT_RADIUS + 1
IDX_ACT_WAVE = IDX_ACT_VERTICAL + 1
IDX_ACT_CLOSED = IDX_ACT_WAVE + 1
IDX_ACT_PHASE = IDX_ACT_CLOSED + 1
IDX_ACT_WIND_MEAN = WIND_ACTION_START
IDX_ACT_WIND_GUST_AMP = IDX_ACT_WIND_MEAN + 3
IDX_ACT_WIND_GUST_FREQ = IDX_ACT_WIND_GUST_AMP + 3
IDX_ACT_WIND_GUST_PHASE = IDX_ACT_WIND_GUST_FREQ + 3
IDX_ACT_WIND_BURST_AMP = IDX_ACT_WIND_GUST_PHASE + 3
IDX_ACT_WIND_BURST_START = IDX_ACT_WIND_BURST_AMP + 3
IDX_ACT_WIND_BURST_DURATION = IDX_ACT_WIND_BURST_START + 1
PASS_ERROR_START = CHECKPOINT_STATE_START + CHECKPOINT_STATE_DIM
PASS_ERROR_DIM = NUM_CHECKPOINTS
IDX_PATH_ERROR = PASS_ERROR_START
IDX_LOOKAHEAD_ERROR = PASS_ERROR_START + 1
IDX_PROGRESS = PASS_ERROR_START + PASS_ERROR_DIM
IDX_COMPLETED = IDX_PROGRESS + 1
IDX_CRASHED = IDX_COMPLETED + 1
IDX_ENV_SCORE = IDX_CRASHED + 1
IDX_INFEASIBLE_COUNT = IDX_ENV_SCORE + 1
WIND_STATE_START = IDX_INFEASIBLE_COUNT + 1
WIND_STATE_DIM = WIND_ACTION_DIM


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=200000, seed=0):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.capacity = int(capacity)
        self.size = 0
        self.pos = 0
        self.random_state = np.random.RandomState(seed)

    def add_batch(self, obs, actions, rewards, next_obs, dones):
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        for i in range(obs.shape[0]):
            self.obs[self.pos] = obs[i]
            self.actions[self.pos] = actions[i]
            self.rewards[self.pos] = rewards[i]
            self.next_obs[self.pos] = next_obs[i]
            self.dones[self.pos] = dones[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.random_state.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
        }


class LearnableEnvironmentPolicy:
    """SAC-style environment generator for adaptive path-following curricula."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        learning_rate=1e-4,
        entropy_coef=1.0,
        hidden_sizes=(256, 256),
        rank_reward_scale=1.0,
        action_clip=0.25,
        action_smoothing=0.70,
        max_grad_norm=1.0,
        gamma=0.95,
        tau=0.01,
        replay_size=200000,
        batch_size=256,
        min_replay_size=100,
        gradient_steps=100,
        target_progress_low=0.35,
        target_progress_high=0.90,
        hard_progress_threshold=0.20,
        hard_error_threshold=0.60,
        ready_progress_threshold=0.55,
        ready_error_threshold=0.25,
        task_bank_size=256,
        task_bank_mix_prob=0.20,
        task_bank_noise=0.05,
        seed=0,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.learning_rate = float(learning_rate)
        self.alpha = float(entropy_coef)
        self.hidden_sizes = tuple(hidden_sizes)
        self.rank_reward_scale = float(rank_reward_scale)
        self.action_clip = float(action_clip)
        self.action_smoothing = float(action_smoothing)
        self.max_grad_norm = float(max_grad_norm)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.min_replay_size = int(min_replay_size)
        self.gradient_steps = int(gradient_steps)
        self.target_progress_low = float(target_progress_low)
        self.target_progress_high = float(target_progress_high)
        self.hard_progress_threshold = float(hard_progress_threshold)
        self.hard_error_threshold = float(hard_error_threshold)
        self.ready_progress_threshold = float(ready_progress_threshold)
        self.ready_error_threshold = float(ready_error_threshold)
        self.task_bank_size = int(task_bank_size)
        self.task_bank_mix_prob = float(task_bank_mix_prob)
        self.task_bank_noise = float(task_bank_noise)
        self.random_state = np.random.RandomState(seed)

        self.replay = ReplayBuffer(
            self.obs_dim, self.act_dim, capacity=replay_size, seed=seed)
        self._last_obs = None
        self._last_actions = None
        self._last_applied_actions = None
        self._task_bank_actions = np.zeros(
            (self.task_bank_size, self.act_dim), dtype=np.float32)
        self._task_bank_rewards = np.full(
            self.task_bank_size, -np.inf, dtype=np.float32)
        self._task_bank_size = 0
        self._updates = 0

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.compat.v1.set_random_seed(seed)
            self._build_graph()
            self.saver = tf.compat.v1.train.Saver()
            self.param_saver = tf.compat.v1.train.Saver(
                var_list=tf.compat.v1.trainable_variables())
            self.sess = tf.compat.v1.Session(graph=self.graph)
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.sess.run(self.target_init_op)

    def _build_graph(self):
        self.obs_ph = tf.compat.v1.placeholder(
            tf.float32, [None, self.obs_dim], name="env_policy_obs")
        self.act_ph = tf.compat.v1.placeholder(
            tf.float32, [None, self.act_dim], name="env_policy_act")
        self.rew_ph = tf.compat.v1.placeholder(
            tf.float32, [None], name="env_policy_reward")
        self.next_obs_ph = tf.compat.v1.placeholder(
            tf.float32, [None, self.obs_dim], name="env_policy_next_obs")
        self.done_ph = tf.compat.v1.placeholder(
            tf.float32, [None], name="env_policy_done")

        policy_action, log_prob, deterministic_action, log_std = self._actor(
            self.obs_ph, scope="actor", reuse=False)
        next_action, next_log_prob, _, _ = self._actor(
            self.next_obs_ph, scope="actor", reuse=True)

        q1 = self._critic(self.obs_ph, self.act_ph, scope="q1", reuse=False)
        q2 = self._critic(self.obs_ph, self.act_ph, scope="q2", reuse=False)
        target_q1 = self._critic(
            self.next_obs_ph, next_action, scope="target_q1", reuse=False)
        target_q2 = self._critic(
            self.next_obs_ph, next_action, scope="target_q2", reuse=False)
        target_v = tf.minimum(target_q1, target_q2) - self.alpha * next_log_prob
        target_q = tf.stop_gradient(
            self.rew_ph + self.gamma * (1.0 - self.done_ph) * target_v)

        self.critic_loss = tf.reduce_mean(
            tf.square(q1 - target_q) + tf.square(q2 - target_q))
        actor_q1 = self._critic(
            self.obs_ph, policy_action, scope="q1", reuse=True)
        actor_q2 = self._critic(
            self.obs_ph, policy_action, scope="q2", reuse=True)
        self.actor_loss = tf.reduce_mean(
            self.alpha * log_prob - tf.minimum(actor_q1, actor_q2))

        actor_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="actor")
        critic_vars = (
            tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="q1") +
            tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="q2")
        )
        q1_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="q1")
        q2_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="q2")
        target_q1_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="target_q1")
        target_q2_vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="target_q2")

        self.critic_train_op = self._minimize(
            self.critic_loss, critic_vars, name="critic_adam")
        with tf.control_dependencies([self.critic_train_op]):
            self.actor_train_op = self._minimize(
                self.actor_loss, actor_vars, name="actor_adam")
        with tf.control_dependencies([self.actor_train_op]):
            self.train_op = tf.group(*self._target_update_ops(
                q1_vars, q2_vars, target_q1_vars, target_q2_vars, self.tau))

        self.target_init_op = tf.group(*self._target_update_ops(
            q1_vars, q2_vars, target_q1_vars, target_q2_vars, tau=1.0))
        self.sample_action = policy_action
        self.deterministic_action = deterministic_action
        self.log_std = log_std
        self.q_mean = 0.5 * (tf.reduce_mean(q1) + tf.reduce_mean(q2))

    def _actor(self, obs, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            latent = self._mlp(obs, self.hidden_sizes)
            mean = tf.compat.v1.layers.dense(latent, self.act_dim, name="mean")
            log_std = tf.compat.v1.layers.dense(
                latent, self.act_dim, name="log_std")
            log_std = tf.clip_by_value(log_std, -5.0, -1.0)
            std = tf.exp(log_std)
            noise = tf.random.normal(tf.shape(mean))
            raw_action = mean + std * noise
            squashed = tf.tanh(raw_action)
            action = self.action_clip * squashed

            normal_log_prob = -0.5 * tf.reduce_sum(
                tf.square((raw_action - mean) / (std + 1e-8)) +
                2.0 * log_std + np.log(2.0 * np.pi),
                axis=1,
            )
            squash_correction = tf.reduce_sum(
                tf.math.log(
                    self.action_clip * (1.0 - tf.square(squashed)) + 1e-6),
                axis=1,
            )
            log_prob = normal_log_prob - squash_correction
            deterministic_action = self.action_clip * tf.tanh(mean)
            return action, log_prob, deterministic_action, log_std

    def _critic(self, obs, action, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x = tf.concat([obs, action], axis=1)
            latent = self._mlp(x, self.hidden_sizes)
            q = tf.compat.v1.layers.dense(latent, 1, name="q")
            return tf.squeeze(q, axis=1)

    def _mlp(self, x, hidden_sizes):
        latent = x
        for idx, hidden_size in enumerate(hidden_sizes):
            latent = tf.compat.v1.layers.dense(
                latent,
                hidden_size,
                activation=tf.nn.relu,
                name="fc{}".format(idx),
            )
        return latent

    def _minimize(self, loss, variables, name):
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.learning_rate, name=name)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=variables)
        clipped = []
        for grad, var in grads_and_vars:
            if grad is not None:
                grad = tf.clip_by_norm(grad, self.max_grad_norm)
            clipped.append((grad, var))
        return optimizer.apply_gradients(clipped)

    def _target_update_ops(self, q1_vars, q2_vars, target_q1_vars,
                           target_q2_vars, tau):
        source_vars = sorted(q1_vars, key=lambda v: v.name) + sorted(
            q2_vars, key=lambda v: v.name)
        target_vars = sorted(target_q1_vars, key=lambda v: v.name) + sorted(
            target_q2_vars, key=lambda v: v.name)
        return [
            target.assign((1.0 - tau) * target + tau * source)
            for source, target in zip(source_vars, target_vars)
        ]

    def sample(self, obs, deterministic=False):
        obs = np.asarray(obs, dtype=np.float32)
        tensor = self.deterministic_action if deterministic else self.sample_action
        actions = self.sess.run(tensor, {self.obs_ph: obs})
        return np.clip(actions, -self.action_clip,
                       self.action_clip).astype(np.float32)

    def update_and_apply(self, env, logger=None):
        obs = env.get_env_policy_obs()
        rewards = self._curriculum_rewards(obs)
        train_stats = None

        if self._last_obs is not None and self._last_actions is not None:
            self._update_task_bank(self._last_actions, rewards)
            self.replay.add_batch(
                self._last_obs,
                self._last_actions,
                rewards,
                obs,
                np.zeros_like(rewards, dtype=np.float32),
            )
            if self.replay.size >= self.min_replay_size:
                train_stats = self._train_sac()

        raw_actions = self.sample(obs, deterministic=False)
        raw_actions = self._mix_task_bank_actions(raw_actions)
        actions = self._condition_actions(obs, raw_actions)
        env.apply_env_policy_actions(actions)

        self._last_obs = obs
        self._last_actions = actions
        self._last_applied_actions = actions

        if logger is not None:
            self._log(logger, obs, actions, rewards, train_stats)

        return {
            "reward_mean": float(np.mean(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "loss": None if train_stats is None else train_stats["actor_loss"],
            "updates": self._updates,
        }

    def _train_sac(self):
        actor_losses = []
        critic_losses = []
        q_means = []
        for _ in range(self.gradient_steps):
            batch = self.replay.sample(self.batch_size)
            feed = {
                self.obs_ph: batch["obs"],
                self.act_ph: batch["actions"],
                self.rew_ph: batch["rewards"],
                self.next_obs_ph: batch["next_obs"],
                self.done_ph: batch["dones"],
            }
            _, actor_loss, critic_loss, q_mean = self.sess.run(
                [self.train_op, self.actor_loss, self.critic_loss, self.q_mean],
                feed,
            )
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            q_means.append(q_mean)
            self._updates += 1

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "q_mean": float(np.mean(q_means)),
        }

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with self.graph.as_default():
            self.saver.save(self.sess, os.path.join(path, "model"))

    def load(self, path, reset_optimizer=True):
        checkpoint = self._resolve_checkpoint(path)
        if checkpoint is None:
            raise RuntimeError(
                "No environment-policy checkpoint found at {}".format(path))

        saver = self.param_saver if reset_optimizer else self.saver
        with self.graph.as_default():
            saver.restore(self.sess, checkpoint)
        print("[EnvPolicy] loaded {} ({})".format(
            checkpoint,
            "parameters only" if reset_optimizer else "full checkpoint",
        ))

    def _resolve_checkpoint(self, path):
        if not path:
            return None
        if os.path.isdir(path):
            fallback = os.path.join(path, "model")
            if os.path.exists(fallback + ".index"):
                return fallback
            checkpoint = tf.train.latest_checkpoint(path)
            if checkpoint is not None and os.path.exists(checkpoint + ".index"):
                return checkpoint
            return None
        if os.path.exists(path + ".index"):
            return path
        if os.path.exists(path):
            return path
        return None

    def _curriculum_rewards(self, obs):
        crashed = obs[:, IDX_CRASHED]
        infeasible = obs[:, IDX_INFEASIBLE_COUNT]
        frontier_score, _, _, _ = self._frontier_score(obs)
        path_quality, _, _ = self._path_quality(obs)
        wind_quality = self._wind_quality(obs)
        path_failure = self._path_failure(obs)
        wind_failure = self._wind_failure(obs)

        difficulty_quality = self._difficulty_quality(path_quality,
                                                      wind_quality)
        gated_quality = frontier_score * (0.20 + 0.80 * difficulty_quality)
        failure_penalty = (
            0.80 * crashed +
            0.45 * infeasible +
            0.35 * path_failure +
            0.45 * wind_failure
        )
        rewards = 1.25 * gated_quality - failure_penalty
        return np.clip(rewards, -1.5, 1.5).astype(np.float32)

    def _condition_actions(self, obs, actions):
        actions = np.asarray(actions, dtype=np.float32)
        guarded = np.clip(actions, -self.action_clip, self.action_clip)

        if self._last_applied_actions is not None:
            guarded = (
                self.action_smoothing * self._last_applied_actions +
                (1.0 - self.action_smoothing) * guarded
            )

        return np.clip(guarded, -self.action_clip,
                       self.action_clip).astype(np.float32)

    def _gaussian_score(self, value, target, sigma):
        sigma = max(float(sigma), 1e-6)
        return np.exp(-0.5 * np.square((value - target) / sigma)).astype(
            np.float32)

    def _clipped_under_score(self, value, target, tolerance=0.08, width=0.35):
        deficit = np.asarray(target - value, dtype=np.float32)
        score = 1.0 - np.clip((deficit - tolerance) / width, 0.0, 1.0)
        return np.clip(score, 0.0, 1.0).astype(np.float32)

    def _clipped_band_score(self, value, target, tolerance=0.08, width=0.35):
        deviation = np.abs(np.asarray(value - target, dtype=np.float32))
        score = 1.0 - np.clip((deviation - tolerance) / width, 0.0, 1.0)
        return np.clip(score, 0.0, 1.0).astype(np.float32)

    def _difficulty_quality(self, path_quality, wind_quality):
        joint_quality = np.sqrt(np.maximum(path_quality * wind_quality, 1e-6))
        quality = (
            0.50 * path_quality +
            0.35 * wind_quality +
            0.15 * joint_quality
        )
        return np.clip(quality, 0.0, 1.0).astype(np.float32)

    def _relative_rank_score(self, performance):
        performance = np.asarray(performance, dtype=np.float32)
        if performance.shape[0] <= 1:
            return np.ones_like(performance, dtype=np.float32)
        order = np.argsort(performance)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(order.shape[0], dtype=np.float32)
        center = 0.5 * float(order.shape[0] - 1)
        rank_score = 1.0 - np.abs(ranks - center) / max(center, 1.0)
        return np.clip(rank_score, 0.0, 1.0).astype(np.float32)

    def _learnability_score(self, obs):
        progress = obs[:, IDX_PROGRESS]
        error = self._tracking_error(obs)
        completed = obs[:, IDX_COMPLETED]
        crashed = obs[:, IDX_CRASHED]
        infeasible = obs[:, IDX_INFEASIBLE_COUNT]
        progress_score = np.clip(
            progress / max(self.target_progress_low, 1e-6), 0.0, 1.0)
        tracking_score = 1.0 - np.clip(error, 0.0, 1.0)
        learnability = (
            0.50 * tracking_score +
            0.35 * progress_score +
            0.15 * completed -
            0.35 * crashed -
            0.20 * infeasible
        )
        return np.clip(learnability, 0.0, 1.0).astype(np.float32)

    def _frontier_score(self, obs):
        learnability = self._learnability_score(obs)
        crashed = obs[:, IDX_CRASHED]
        infeasible = obs[:, IDX_INFEASIBLE_COUNT]
        path_failure = self._path_failure(obs)
        wind_failure = self._wind_failure(obs)

        target = np.clip(
            0.5 * (self.ready_progress_threshold +
                   self.target_progress_high),
            0.55,
            0.85,
        )
        hard_sigma = max(0.12, 0.85 *
                         (target - self.ready_progress_threshold))
        easy_sigma = max(0.18, 1.25 *
                         (self.target_progress_high - target))
        sigma = np.where(learnability < target, hard_sigma, easy_sigma)
        smooth_frontier = np.exp(
            -0.5 * np.square((learnability - target) / sigma)).astype(
                np.float32)

        performance = (
            learnability -
            0.35 * crashed -
            0.20 * infeasible -
            0.25 * path_failure -
            0.30 * wind_failure
        )
        rank_score = self._relative_rank_score(performance)
        rank_mix = np.clip(0.15 * self.rank_reward_scale, 0.0, 0.30)
        frontier = smooth_frontier * ((1.0 - rank_mix) +
                                      rank_mix * rank_score)
        return (
            np.clip(frontier, 0.0, 1.0).astype(np.float32),
            smooth_frontier,
            rank_score,
            learnability,
        )

    def _path_quality(self, obs):
        difficulty = self._path_difficulty(obs)
        target_difficulty = self._target_difficulty(obs)
        closedness = self._closedness(obs)
        target_closedness = self._target_closedness(obs)
        difficulty_score = self._clipped_under_score(
            difficulty, target_difficulty, tolerance=0.08, width=0.35)
        closedness_score = self._clipped_under_score(
            closedness, target_closedness, tolerance=0.08, width=0.35)
        joint_score = np.sqrt(
            np.maximum(difficulty_score * closedness_score, 1e-6))
        quality = (
            0.55 * difficulty_score +
            0.35 * closedness_score +
            0.10 * joint_score
        )
        return (
            np.clip(quality, 0.0, 1.0).astype(np.float32),
            difficulty_score,
            closedness_score,
        )

    def _wind_quality(self, obs):
        wind_difficulty = self._wind_difficulty(obs)
        target_wind_difficulty = self._target_wind_difficulty(obs)
        return self._clipped_band_score(
            wind_difficulty,
            target_wind_difficulty,
            tolerance=0.08,
            width=0.35,
        )

    def _ability(self, obs):
        progress = obs[:, IDX_PROGRESS]
        error = self._tracking_error(obs)
        completed = obs[:, IDX_COMPLETED]
        crashed = obs[:, IDX_CRASHED]
        infeasible = obs[:, IDX_INFEASIBLE_COUNT]
        progress_score = np.clip(
            progress / max(self.ready_progress_threshold, 1e-6), 0.0, 1.0)
        tracking_score = 1.0 - np.clip(
            error / max(self.hard_error_threshold, 1e-6), 0.0, 1.0)
        ability = (
            0.08 + 0.45 * tracking_score + 0.25 * progress_score +
            0.25 * completed - 0.35 * crashed - 0.20 * infeasible
        )
        return np.clip(ability, 0.05, 1.0).astype(np.float32)

    def _update_task_bank(self, actions, rewards):
        if self.task_bank_size <= 0:
            return
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        for action, reward in zip(actions, rewards):
            if not np.isfinite(reward) or reward < 0.35:
                continue
            if self._task_bank_size < self.task_bank_size:
                idx = self._task_bank_size
                self._task_bank_size += 1
            else:
                idx = int(np.argmin(self._task_bank_rewards))
                if reward <= self._task_bank_rewards[idx]:
                    continue
            self._task_bank_actions[idx] = action
            self._task_bank_rewards[idx] = reward

    def _mix_task_bank_actions(self, actions):
        if self._task_bank_size <= 0 or self.task_bank_mix_prob <= 0.0:
            return actions
        mixed = np.asarray(actions, dtype=np.float32).copy()
        use_bank = self.random_state.rand(mixed.shape[0]) < self.task_bank_mix_prob
        if not np.any(use_bank):
            return mixed
        bank_indices = self.random_state.randint(
            0, self._task_bank_size, size=int(np.sum(use_bank)))
        noise = self.task_bank_noise * self.action_clip * self.random_state.randn(
            int(np.sum(use_bank)), self.act_dim)
        mixed[use_bank] = self._task_bank_actions[bank_indices] + noise
        return np.clip(mixed, -self.action_clip,
                       self.action_clip).astype(np.float32)

    def _readiness(self, obs):
        progress = obs[:, IDX_PROGRESS]
        error = self._tracking_error(obs)
        progress_score = np.clip(
            progress / max(self.target_progress_low, 1e-6), 0.0, 1.0)
        tracking_score = 1.0 - np.clip(error, 0.0, 1.0)
        learnability = np.clip(
            0.65 * tracking_score + 0.35 * progress_score, 0.0, 1.0)
        learnability_ready = np.clip(
            (learnability - self.ready_progress_threshold) /
            max(self.target_progress_high - self.ready_progress_threshold,
                1e-6), 0.0, 1.0)
        error_ready = 1.0 - np.clip(
            error / max(self.ready_error_threshold, 1e-6), 0.0, 1.0)
        crash_ready = 1.0 - obs[:, IDX_CRASHED]
        return np.clip(
            learnability_ready * error_ready * crash_ready, 0.0, 1.0)

    def _path_readiness(self, obs):
        return self._readiness(obs).astype(np.float32)

    def _wind_readiness(self, obs):
        progress = obs[:, IDX_PROGRESS]
        error = self._tracking_error(obs)
        completed = obs[:, IDX_COMPLETED]
        crashed = obs[:, IDX_CRASHED]
        infeasible = obs[:, IDX_INFEASIBLE_COUNT]
        path_stress = self._path_failure(obs)
        progress_score = np.clip(
            progress / max(self.ready_progress_threshold, 1e-6), 0.0, 1.0)
        tracking_score = 1.0 - np.clip(
            error / max(self.ready_error_threshold, 1e-6), 0.0, 1.0)
        stable = (
            0.35 * progress_score +
            0.30 * tracking_score +
            0.25 * completed +
            0.10 * (1.0 - crashed)
        )
        readiness = stable * (1.0 - crashed) * (1.0 - infeasible)
        readiness *= (1.0 - 0.60 * path_stress)
        return np.clip(readiness, 0.0, 1.0).astype(np.float32)

    def _path_failure(self, obs):
        error = self._tracking_error(obs)
        progress = obs[:, IDX_PROGRESS]
        low_progress = 1.0 - np.clip(
            progress / max(self.target_progress_low, 1e-6), 0.0, 1.0)
        path_error_pressure = np.clip(
            error / max(self.ready_error_threshold, 1e-6), 0.0, 1.0)
        return np.clip(
            0.65 * path_error_pressure + 0.35 * low_progress,
            0.0,
            1.0,
        ).astype(np.float32)

    def _wind_failure(self, obs):
        error = self._tracking_error(obs)
        progress = obs[:, IDX_PROGRESS]
        crashed = obs[:, IDX_CRASHED]
        wind_difficulty = self._wind_difficulty(obs)
        high_wind_gate = np.clip((wind_difficulty - 0.18) / 0.45, 0.0, 1.0)
        low_progress = 1.0 - np.clip(
            progress / max(self.target_progress_low, 1e-6), 0.0, 1.0)
        error_pressure = np.clip(
            error / max(self.hard_error_threshold, 1e-6), 0.0, 1.0)
        return np.clip(
            high_wind_gate *
            (0.55 * crashed + 0.25 * error_pressure + 0.20 * low_progress),
            0.0,
            1.0,
        ).astype(np.float32)

    def _path_difficulty(self, obs):
        points = self._checkpoint_points(obs)
        diffs = points[:, 1:, :] - points[:, :-1, :]
        x_span = np.max(points[:, :, 0], axis=1) - np.min(points[:, :, 0], axis=1)
        y_span = np.max(points[:, :, 1], axis=1) - np.min(points[:, :, 1], axis=1)
        z_span = np.max(points[:, :, 2], axis=1) - np.min(points[:, :, 2], axis=1)
        seg_norm = np.linalg.norm(diffs, axis=2) + 1e-6
        direction = diffs / seg_norm[:, :, None]
        turn = np.linalg.norm(direction[:, 1:, :] - direction[:, :-1, :],
                              axis=2)
        mean_turn = np.mean(turn, axis=1) if turn.shape[1] > 0 else 0.0
        reverse = np.mean((diffs[:, :, 0] < -0.02).astype(np.float32), axis=1)
        closedness = self._closedness(obs)
        difficulty = (
            0.06 * np.clip(np.abs(x_span) / 0.35, 0.0, 1.0) +
            0.14 * np.clip(np.abs(y_span), 0.0, 1.0) +
            0.16 * np.clip(np.abs(z_span), 0.0, 1.0) +
            0.24 * np.clip(mean_turn / 0.75, 0.0, 1.0) +
            0.18 * np.clip(reverse / 0.35, 0.0, 1.0) +
            0.22 * closedness
        )
        return np.clip(difficulty, 0.0, 1.0).astype(np.float32)

    def _closedness(self, obs):
        points = self._checkpoint_points(obs)
        diffs = points[:, 1:, :] - points[:, :-1, :]
        reverse = np.mean((diffs[:, :, 0] < -0.02).astype(np.float32), axis=1)
        x_span = np.max(points[:, :, 0], axis=1) - np.min(points[:, :, 0], axis=1)
        last_point = points[:, -1, :]
        endpoint_return = 1.0 - np.clip(
            np.linalg.norm(last_point, axis=1) / 0.65, 0.0, 1.0)
        closedness = (
            0.25 * np.clip(reverse / 0.30, 0.0, 1.0) +
            0.55 * endpoint_return +
            0.20 * np.clip(x_span / 0.20, 0.0, 1.0)
        )
        return np.clip(closedness, 0.0, 1.0).astype(np.float32)

    def _target_closedness(self, obs):
        ability = self._ability(obs)
        readiness = self._path_readiness(obs)
        ability_gate = np.clip((ability - 0.40) / 0.45, 0.0, 1.0)
        target = readiness * (0.30 + 0.65 * ability_gate)
        return np.clip(target, 0.0, 1.0).astype(np.float32)

    def _target_difficulty(self, obs):
        ability = self._ability(obs)
        readiness = self._path_readiness(obs)
        guarded_target = 0.08 + 0.42 * ability
        ready_target = 0.25 + 0.65 * ability
        target = (1.0 - readiness) * guarded_target + readiness * ready_target
        return np.clip(target, 0.08, 0.90).astype(np.float32)

    def _wind_state(self, obs):
        end = WIND_STATE_START + WIND_STATE_DIM
        if obs.shape[1] < end:
            return np.zeros((obs.shape[0], WIND_STATE_DIM), dtype=np.float32)
        return obs[:, WIND_STATE_START:end].astype(np.float32)

    def _wind_difficulty(self, obs):
        wind = self._wind_state(obs)
        if wind.shape[1] < WIND_STATE_DIM:
            return np.zeros(obs.shape[0], dtype=np.float32)

        mean_score = np.linalg.norm(wind[:, 0:3], axis=1)
        gust_score = np.linalg.norm(np.clip(wind[:, 3:6], 0.0, 1.0),
                                    axis=1)
        freq_score = np.mean(np.clip(wind[:, 6:9], 0.0, 1.0), axis=1)
        burst_score = np.linalg.norm(np.abs(wind[:, 12:15]),
                                     axis=1)
        duration_score = np.clip(wind[:, 16], 0.0, 1.0)
        difficulty = (
            0.35 * mean_score +
            0.20 * gust_score +
            0.10 * freq_score +
            0.30 * burst_score +
            0.05 * duration_score
        )
        return np.clip(difficulty, 0.0, 1.0).astype(np.float32)

    def _target_wind_difficulty(self, obs):
        ability = self._ability(obs)
        readiness = self._wind_readiness(obs)
        completed = obs[:, IDX_COMPLETED]
        stable = np.clip(
            1.0 - self._tracking_error(obs) /
            max(self.ready_error_threshold, 1e-6),
            0.0,
            1.0,
        )
        base_target = 0.08 + 0.58 * ability
        completion_bonus = 0.08 * completed * stable
        target = readiness * (base_target + completion_bonus)
        return np.clip(target, 0.0, 1.0).astype(np.float32)

    def _hard_mask(self, obs):
        progress = obs[:, IDX_PROGRESS]
        error = self._tracking_error(obs)
        crashed = obs[:, IDX_CRASHED]
        infeasible = obs[:, IDX_INFEASIBLE_COUNT]
        return np.logical_or.reduce((
            crashed > 0.5,
            infeasible > 0.0,
            np.logical_and(progress < 0.05,
                           error > self.hard_error_threshold),
        ))

    def _ready_mask(self, obs):
        progress = obs[:, IDX_PROGRESS]
        error = self._tracking_error(obs)
        progress_score = np.clip(
            progress / max(self.target_progress_low, 1e-6), 0.0, 1.0)
        tracking_score = 1.0 - np.clip(error, 0.0, 1.0)
        learnability = np.clip(
            0.65 * tracking_score + 0.35 * progress_score, 0.0, 1.0)
        crashed = obs[:, IDX_CRASHED]
        infeasible = obs[:, IDX_INFEASIBLE_COUNT]
        return np.logical_and.reduce((
            crashed <= 0.5,
            infeasible <= 0.0,
            learnability >= self.ready_progress_threshold,
            error <= self.ready_error_threshold,
        ))

    def _too_easy_mask(self, obs):
        progress = obs[:, IDX_PROGRESS]
        error = self._tracking_error(obs)
        difficulty = self._path_difficulty(obs)
        target_difficulty = self._target_difficulty(obs)
        closedness = self._closedness(obs)
        target_closedness = self._target_closedness(obs)
        wind_difficulty = self._wind_difficulty(obs)
        target_wind_difficulty = self._target_wind_difficulty(obs)
        return np.logical_and.reduce((
            self._ready_mask(obs),
            progress >= self.target_progress_low,
            error <= self.ready_error_threshold,
            np.logical_or.reduce((
                difficulty < target_difficulty - 0.08,
                closedness < target_closedness - 0.08,
                wind_difficulty < target_wind_difficulty - 0.08,
            )),
        ))

    def _checkpoint_points(self, obs):
        return obs[:, CHECKPOINT_STATE_START:
                   CHECKPOINT_STATE_START + CHECKPOINT_STATE_DIM].reshape(
                       (-1, NUM_CHECKPOINTS, 3))

    def _mean_pass_error(self, obs):
        return self._tracking_error(obs)

    def _tracking_error(self, obs):
        path_error = np.clip(obs[:, IDX_PATH_ERROR], 0.0, 1.0)
        lookahead_error = np.clip(obs[:, IDX_LOOKAHEAD_ERROR], 0.0, 1.0)
        return (0.65 * path_error + 0.35 * lookahead_error).astype(np.float32)

    def _legacy_mean_pass_error(self, obs):
        errors = obs[:, PASS_ERROR_START:PASS_ERROR_START + PASS_ERROR_DIM]
        progress = obs[:, IDX_PROGRESS]
        passed = np.maximum(
            np.round(progress * NUM_CHECKPOINTS).astype(np.int32), 0)
        mean_errors = np.ones(obs.shape[0], dtype=np.float32)
        for i in range(obs.shape[0]):
            n = min(max(passed[i], 1), NUM_CHECKPOINTS)
            mean_errors[i] = float(np.mean(errors[i, :n]))
        return mean_errors

    def _log(self, logger, obs, actions, rewards, train_stats):
        log_std = self.sess.run(self.log_std, {self.obs_ph: obs})
        difficulty = self._path_difficulty(obs)
        target_difficulty = self._target_difficulty(obs)
        closedness = self._closedness(obs)
        target_closedness = self._target_closedness(obs)
        wind_difficulty = self._wind_difficulty(obs)
        target_wind_difficulty = self._target_wind_difficulty(obs)
        frontier_score, smooth_frontier, rank_score, learnability = (
            self._frontier_score(obs))
        path_quality, difficulty_score, closedness_score = (
            self._path_quality(obs))
        wind_quality = self._wind_quality(obs)
        difficulty_quality = self._difficulty_quality(path_quality,
                                                      wind_quality)
        logger.logkv("env_policy_algo", 1.0)
        logger.logkv("env_policy_reward_mean", float(np.mean(rewards)))
        logger.logkv("env_policy_reward_min", float(np.min(rewards)))
        logger.logkv("env_policy_reward_max", float(np.max(rewards)))
        logger.logkv("env_policy_frontier_mean",
                     float(np.mean(frontier_score)))
        logger.logkv("env_policy_smooth_frontier_mean",
                     float(np.mean(smooth_frontier)))
        logger.logkv("env_policy_rank_frontier_mean",
                     float(np.mean(rank_score)))
        logger.logkv("env_policy_learnability_mean",
                     float(np.mean(learnability)))
        logger.logkv("env_policy_path_quality_mean",
                     float(np.mean(path_quality)))
        logger.logkv("env_policy_path_difficulty_score_mean",
                     float(np.mean(difficulty_score)))
        logger.logkv("env_policy_path_closedness_score_mean",
                     float(np.mean(closedness_score)))
        logger.logkv("env_policy_wind_quality_mean",
                     float(np.mean(wind_quality)))
        logger.logkv("env_policy_difficulty_quality_mean",
                     float(np.mean(difficulty_quality)))
        logger.logkv("env_policy_score_mean",
                     float(np.mean(obs[:, IDX_ENV_SCORE])))
        logger.logkv("env_policy_progress_mean",
                     float(np.mean(obs[:, IDX_PROGRESS])))
        logger.logkv("env_policy_error_mean",
                     float(np.mean(self._mean_pass_error(obs))))
        logger.logkv("env_policy_crash_rate",
                     float(np.mean(obs[:, IDX_CRASHED])))
        logger.logkv("env_policy_action_abs_mean",
                     float(np.mean(np.abs(actions))))
        if actions.shape[1] > CHECKPOINT_ACTION_DIM:
            path_action_end = min(actions.shape[1], WIND_ACTION_START)
            logger.logkv(
                "env_policy_global_action_abs_mean",
                float(np.mean(np.abs(actions[:, CHECKPOINT_ACTION_DIM:
                                             path_action_end]))))
        if actions.shape[1] > WIND_ACTION_START:
            wind_actions = actions[:, WIND_ACTION_START:]
            logger.logkv("env_policy_wind_action_abs_mean",
                         float(np.mean(np.abs(wind_actions))))
        checkpoint_actions = actions[:, :CHECKPOINT_ACTION_DIM]
        logger.logkv("env_policy_xy_action_mean",
                     float(np.mean(checkpoint_actions[:, 0::3])))
        logger.logkv("env_policy_y_action_mean",
                     float(np.mean(checkpoint_actions[:, 1::3])))
        logger.logkv("env_policy_z_action_mean",
                     float(np.mean(checkpoint_actions[:, 2::3])))
        logger.logkv("env_policy_action_std", float(np.mean(np.exp(log_std))))
        logger.logkv("env_policy_difficulty_mean",
                     float(np.mean(difficulty)))
        logger.logkv("env_policy_target_difficulty_mean",
                     float(np.mean(target_difficulty)))
        logger.logkv("env_policy_difficulty_gap_mean",
                     float(np.mean(target_difficulty - difficulty)))
        logger.logkv("env_policy_closedness_mean",
                     float(np.mean(closedness)))
        logger.logkv("env_policy_target_closedness_mean",
                     float(np.mean(target_closedness)))
        logger.logkv("env_policy_closedness_gap_mean",
                     float(np.mean(target_closedness - closedness)))
        logger.logkv("env_policy_wind_difficulty_mean",
                     float(np.mean(wind_difficulty)))
        logger.logkv("env_policy_target_wind_difficulty_mean",
                     float(np.mean(target_wind_difficulty)))
        logger.logkv("env_policy_wind_difficulty_gap_mean",
                     float(np.mean(target_wind_difficulty -
                                   wind_difficulty)))
        logger.logkv("env_policy_path_readiness_mean",
                     float(np.mean(self._path_readiness(obs))))
        logger.logkv("env_policy_wind_readiness_mean",
                     float(np.mean(self._wind_readiness(obs))))
        logger.logkv("env_policy_path_failure_mean",
                     float(np.mean(self._path_failure(obs))))
        logger.logkv("env_policy_wind_failure_mean",
                     float(np.mean(self._wind_failure(obs))))
        logger.logkv("env_policy_too_easy_rate",
                     float(np.mean(self._too_easy_mask(obs))))
        logger.logkv("env_policy_hard_rate",
                     float(np.mean(self._hard_mask(obs))))
        logger.logkv("env_policy_ready_rate",
                     float(np.mean(self._ready_mask(obs))))
        logger.logkv("env_policy_ability_mean",
                     float(np.mean(self._ability(obs))))
        logger.logkv("env_policy_task_bank_size",
                     float(self._task_bank_size))
        points = self._checkpoint_points(obs)
        logger.logkv("env_policy_x_span_mean",
                     float(np.mean(np.max(points[:, :, 0], axis=1) -
                                   np.min(points[:, :, 0], axis=1))))
        logger.logkv("env_policy_y_span_mean",
                     float(np.mean(np.max(points[:, :, 1], axis=1) -
                                   np.min(points[:, :, 1], axis=1))))
        logger.logkv("env_policy_z_span_mean",
                     float(np.mean(np.max(points[:, :, 2], axis=1) -
                                   np.min(points[:, :, 2], axis=1))))
        wind = self._wind_state(obs)
        logger.logkv("env_policy_wind_mean_norm",
                     float(np.mean(np.linalg.norm(wind[:, 0:3], axis=1))))
        logger.logkv("env_policy_wind_gust_norm",
                     float(np.mean(np.linalg.norm(wind[:, 3:6], axis=1))))
        logger.logkv("env_policy_wind_burst_norm",
                     float(np.mean(np.linalg.norm(wind[:, 12:15], axis=1))))
        logger.logkv("env_policy_buffer_size", self.replay.size)
        logger.logkv("env_policy_updates", self._updates)
        if train_stats is not None:
            logger.logkv("env_policy_actor_loss", train_stats["actor_loss"])
            logger.logkv("env_policy_critic_loss", train_stats["critic_loss"])
            logger.logkv("env_policy_q_mean", train_stats["q_mean"])

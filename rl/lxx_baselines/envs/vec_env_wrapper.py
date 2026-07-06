import os

import numpy as np
from gym import spaces
from stable_baselines.common.vec_env import VecEnv


class FlightEnvVec(VecEnv):
    #
    def __init__(self, impl, obs_history_len=1):
        self.wrapper = impl
        self.base_num_obs = self.wrapper.getObsDim()
        self.obs_history_len = max(int(obs_history_len), 1)
        self.num_obs = self.base_num_obs * self.obs_history_len
        self.num_acts = self.wrapper.getActDim()
        if os.environ.get("FlightLxx_VERBOSE_ENV", "0") == "1":
            print("obs_dim={}, base_obs_dim={}, history_len={}, act_dim={}".format(
                self.num_obs, self.base_num_obs, self.obs_history_len,
                self.num_acts))
        self._observation_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf,
            np.ones(self.num_obs) * np.Inf, dtype=np.float32)
        self._action_space = spaces.Box(
            low=np.ones(self.num_acts) * -1.,
            high=np.ones(self.num_acts) * 1.,
            dtype=np.float32)
        self._base_observation = np.zeros([self.num_envs, self.base_num_obs],
                                          dtype=np.float32)
        self._obs_history = np.zeros(
            [self.num_envs, self.obs_history_len, self.base_num_obs],
            dtype=np.float32)
        self._observation = np.zeros([self.num_envs, self.num_obs],
                                     dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=bool)
        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self._extraInfo = np.zeros([self.num_envs,
                                    len(self._extraInfoNames)], dtype=np.float32)
        self.rewards = [[] for _ in range(self.num_envs)]

        if hasattr(self.wrapper, "getEpisodeLength"):
            self.max_episode_steps = int(self.wrapper.getEpisodeLength())
        else:
            self.max_episode_steps = 500
        if hasattr(self.wrapper, "getEnvPolicyObsDim"):
            self.env_policy_obs_dim = int(self.wrapper.getEnvPolicyObsDim())
            self.env_policy_act_dim = int(self.wrapper.getEnvPolicyActDim())
        else:
            self.env_policy_obs_dim = 0
            self.env_policy_act_dim = 0

    def _history_to_observation(self):
        self._observation = self._obs_history.reshape(
            self.num_envs, self.num_obs).astype(np.float32)
        return self._observation

    def _reset_history_rows(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return
        self._obs_history[rows, :, :] = self._base_observation[rows, None, :]

    def _append_history(self, done=None):
        if self.obs_history_len <= 1:
            self._obs_history[:, 0, :] = self._base_observation
        else:
            self._obs_history[:, :-1, :] = self._obs_history[:, 1:, :]
            self._obs_history[:, -1, :] = self._base_observation
            if done is not None:
                done_rows = np.nonzero(done)[0]
                self._reset_history_rows(done_rows)
        return self._history_to_observation()

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def step(self, action):
        self.wrapper.step(action, self._base_observation,
                          self._reward, self._done, self._extraInfo)
        self._append_history(done=self._done)

        if len(self._extraInfoNames) != 0:
            info = [{'extra_info': {
                self._extraInfoNames[j]: self._extraInfo[i, j] for j in range(0, len(self._extraInfoNames))
            }} for i in range(self.num_envs)]
        else:
            info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i])
            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                epinfo.update(info[i].get('extra_info', {}))
                epinfo['rollout_cut'] = 0.0
                info[i]['episode'] = epinfo
                self.rewards[i].clear()

        return self._observation.copy(), self._reward.copy(), \
            self._done.copy(), info.copy()

    def stepUnity(self, action, send_id):
        receive_id = self.wrapper.stepUnity(action, self._base_observation,
                                            self._reward, self._done, self._extraInfo, send_id)
        self._append_history(done=self._done)

        return receive_id

    def sample_actions(self):
        actions = []
        for i in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float32)

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset(self._base_observation)
        self._obs_history[:, :, :] = self._base_observation[:, None, :]
        self._history_to_observation()
        return self._observation.copy()

    def reset_and_update_info(self):
        info = self._update_epi_info(clear=True)
        return self.reset(), info

    def rollout_info(self, clear=False):
        """Return unfinished rollout fragments without resetting environments."""
        return self._update_epi_info(clear=clear)

    def _update_epi_info(self, clear):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            if eplen == 0:
                continue
            epinfo = {"r": eprew, "l": eplen}
            if len(self._extraInfoNames) != 0:
                epinfo.update({
                    self._extraInfoNames[j]: self._extraInfo[i, j]
                    for j in range(0, len(self._extraInfoNames))
                })
            epinfo['rollout_cut'] = 1.0
            info[i]['episode'] = epinfo
            if clear:
                self.rewards[i].clear()
        return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def connectUnity(self):
        self.wrapper.connectUnity()

    def disconnectUnity(self):
        self.wrapper.disconnectUnity()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    def start_recording_video(self, file_name):
        raise RuntimeError('This method is not implemented')

    def stop_recording_video(self):
        raise RuntimeError('This method is not implemented')

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def get_env_policy_obs(self):
        obs = np.zeros([self.num_envs, self.env_policy_obs_dim],
                       dtype=np.float32)
        self.wrapper.getEnvPolicyObs(obs)
        return obs

    def apply_env_policy_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        self.wrapper.applyEnvPolicyActions(actions)

    def step_async(self):
        raise RuntimeError('This method is not implemented')

    def step_wait(self):
        raise RuntimeError('This method is not implemented')

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError('This method is not implemented')

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError('This method is not implemented')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError('This method is not implemented')

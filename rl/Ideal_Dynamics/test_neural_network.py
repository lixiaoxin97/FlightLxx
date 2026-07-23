import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


def test_model(env, model_1, model_2, model_3, render=False):
    """Run 3 trained models in sequence and plot trajectory comparison."""
    max_ep_length = 250

    # ---- Build figure layout ----
    fig = plt.figure(figsize=(18, 12), tight_layout=True)
    gs = gridspec.GridSpec(5, 12)

    axes = {
        'pos_x':      fig.add_subplot(gs[0, 0:4]),
        'pos_y':      fig.add_subplot(gs[0, 4:8]),
        'pos_z':      fig.add_subplot(gs[0, 8:12]),
        'vel_x':      fig.add_subplot(gs[1, 0:4]),
        'vel_y':      fig.add_subplot(gs[1, 4:8]),
        'vel_z':      fig.add_subplot(gs[1, 8:12]),
        'euler_x':    fig.add_subplot(gs[2, 0:4]),
        'euler_y':    fig.add_subplot(gs[2, 4:8]),
        'euler_z':    fig.add_subplot(gs[2, 8:12]),
        'omega_x':    fig.add_subplot(gs[3, 0:4]),
        'omega_y':    fig.add_subplot(gs[3, 4:8]),
        'omega_z':    fig.add_subplot(gs[3, 8:12]),
        'act_1':      fig.add_subplot(gs[4, 0:3]),
        'act_2':      fig.add_subplot(gs[4, 3:6]),
        'act_3':      fig.add_subplot(gs[4, 6:9]),
        'act_4':      fig.add_subplot(gs[4, 9:12]),
    }

    _label_axes(axes)

    if render:
        env.connectUnity()

    # ---- Run each model and plot ----
    models = [model_1, model_2, model_3]
    names = ["CTBR", "CTBR+DR", "CTBR+ID"]
    colors = [1, 2, 3]

    for model, name, color_idx in zip(models, names, colors):
        pos, dpos, euler, deuler, actions = _run_rollout(env, model, max_ep_length)
        t = np.arange(0, pos.shape[0])
        _plot_trajectory(axes, t, pos, dpos, euler, deuler, actions, color_idx, name)

    if render:
        env.disconnectUnity()

    # ---- Legends ----
    for key in ['pos_z', 'vel_z', 'euler_z', 'omega_z', 'act_4']:
        axes[key].legend(loc='lower right')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------

def _label_axes(axes):
    """Attach axis labels."""
    axes['pos_x'].set(xlabel='t(s)', ylabel='pos_x(m)')
    axes['pos_y'].set(xlabel='t(s)', ylabel='pos_y(m)')
    axes['pos_z'].set(xlabel='t(s)', ylabel='pos_z(m)')

    axes['vel_x'].set(xlabel='t(s)', ylabel='vel_x(m/s)')
    axes['vel_y'].set(xlabel='t(s)', ylabel='vel_y(m/s)')
    axes['vel_z'].set(xlabel='t(s)', ylabel='vel_z(m/s)')

    axes['euler_x'].set(xlabel='t(s)', ylabel='euler_x(rad)')
    axes['euler_y'].set(xlabel='t(s)', ylabel='euler_y(rad)')
    axes['euler_z'].set(xlabel='t(s)', ylabel='euler_z(rad)')

    axes['omega_x'].set(xlabel='t(s)', ylabel='omega_x(rad/s)')
    axes['omega_y'].set(xlabel='t(s)', ylabel='omega_y(rad/s)')
    axes['omega_z'].set(xlabel='t(s)', ylabel='omega_z(rad/s)')

    axes['act_1'].set(xlabel='t(s)', ylabel='act_1(rad/s)')
    axes['act_2'].set(xlabel='t(s)', ylabel='act_2(rad/s)')
    axes['act_3'].set(xlabel='t(s)', ylabel='act_3(rad/s)')
    axes['act_4'].set(xlabel='t(s)', ylabel='act_4(N)')


def _run_rollout(env, model, max_ep_length):
    """Run a single deterministic rollout. Returns trajectory arrays."""
    pos, dpos, euler, deuler, actions = [], [], [], [], []
    obs, done, ep_len = env.reset(), False, 0

    while not (done or (ep_len >= max_ep_length)):
        act, _ = model.predict(obs, deterministic=True)
        obs, _rew, done, _infos = env.step(act)

        ep_len += 1
        pos.append(obs[0, 0:3].tolist())
        dpos.append(obs[0, 6:9].tolist())
        euler.append(obs[0, 3:6].tolist())
        deuler.append(obs[0, 9:12].tolist())
        actions.append(act[0, :].tolist())

    return (np.asarray(arr) for arr in (pos, dpos, euler, deuler, actions))


def _plot_trajectory(axes, t, pos, dpos, euler, deuler, actions, color_idx, name):
    """Plot a single model's trajectory onto the shared axes."""
    dt = 0.02
    color = f"C{color_idx}"

    # Position
    axes['pos_x'].step(t * dt, pos[:, 0],  color, label=f"trail: {color_idx}")
    axes['pos_y'].step(t * dt, pos[:, 1],  color, label=f"trail: {color_idx}")
    axes['pos_z'].step(t * dt, pos[:, 2],  color, label=name)

    # Velocity
    axes['vel_x'].step(t * dt, dpos[:, 0], color, label=f"trail: {color_idx}")
    axes['vel_y'].step(t * dt, dpos[:, 1], color, label=f"trail: {color_idx}")
    axes['vel_z'].step(t * dt, dpos[:, 2], color, label=name)

    # Euler angles
    axes['euler_x'].step(t * dt, euler[:, -1], color, label=f"trail: {color_idx}")
    axes['euler_y'].step(t * dt, euler[:, 0],  color, label=f"trail: {color_idx}")
    axes['euler_z'].step(t * dt, euler[:, 1],  color, label=name)

    # Angular velocity (omega)
    axes['omega_x'].step(t * dt, deuler[:, 0], color, label=f"trail: {color_idx}")
    axes['omega_y'].step(t * dt, deuler[:, 1], color, label=f"trail: {color_idx}")
    axes['omega_z'].step(t * dt, deuler[:, 2], color, label=name)

    # Actions (un-normalize from [-1, 1] to physical units)
    two_pi = 2 * np.pi
    axes['act_1'].step(t * dt, actions[:, 0] * two_pi,      color, label=f"trail: {color_idx}")
    axes['act_2'].step(t * dt, actions[:, 1] * two_pi,      color, label=f"trail: {color_idx}")
    axes['act_3'].step(t * dt, actions[:, 2] * np.pi,       color, label=f"trail: {color_idx}")
    axes['act_4'].step(t * dt, actions[:, 3] * 9.81 + 9.81, color, label=name)

import os

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


def test_model(env, model, render=False):
    #
    fig = plt.figure(figsize=(18, 12), tight_layout=True)
    gs = gridspec.GridSpec(5, 12)
    #
    ax_x = fig.add_subplot(gs[0, 0:4])
    ax_y = fig.add_subplot(gs[0, 4:8])
    ax_z = fig.add_subplot(gs[0, 8:12])
    #
    ax_dx = fig.add_subplot(gs[1, 0:4])
    ax_dy = fig.add_subplot(gs[1, 4:8])
    ax_dz = fig.add_subplot(gs[1, 8:12])
    #
    ax_euler_x = fig.add_subplot(gs[2, 0:4])
    ax_euler_y = fig.add_subplot(gs[2, 4:8])
    ax_euler_z = fig.add_subplot(gs[2, 8:12])
    #
    ax_euler_vx = fig.add_subplot(gs[3, 0:4])
    ax_euler_vy = fig.add_subplot(gs[3, 4:8])
    ax_euler_vz = fig.add_subplot(gs[3, 8:12])
    #
    ax_action0 = fig.add_subplot(gs[4, 0:3])
    ax_action1 = fig.add_subplot(gs[4, 3:6])
    ax_action2 = fig.add_subplot(gs[4, 6:9])
    ax_action3 = fig.add_subplot(gs[4, 9:12])

    wind_names = set(getattr(env, "extra_info_names", []))
    plot_wind = {"wind_x", "wind_y", "wind_z"}.issubset(wind_names)
    if plot_wind:
        wind_fig, wind_axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        wind_fig.suptitle("Wind field during evaluation")
    else:
        wind_fig, wind_axes = None, None
    wind_summaries = []

    max_ep_length = env.max_episode_steps
    num_rollouts = 5
    
    if render:
        env.connectUnity()
    for n_roll in range(num_rollouts):
        pos, euler, dpos, deuler = [], [], [], []
        actions = []
        wind_rows = []
        obs, done, ep_len = env.reset(), False, 0
        while not (done or (ep_len >= max_ep_length)):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, infos = env.step(act)
            #
            ep_len += 1
            #
            pos.append(obs[0, 0:3].tolist())
            dpos.append(obs[0, 6:9].tolist())
            euler.append(obs[0, 3:6].tolist())
            deuler.append(obs[0, 9:12].tolist())
            #
            actions.append(act[0, :].tolist())
            if plot_wind:
                extra = infos[0].get("extra_info", {})
                wind_rows.append([
                    extra.get("wind_x", 0.0),
                    extra.get("wind_y", 0.0),
                    extra.get("wind_z", 0.0),
                    extra.get("wind_mean_x", 0.0),
                    extra.get("wind_mean_y", 0.0),
                    extra.get("wind_mean_z", 0.0),
                    extra.get("wind_gust_x", 0.0),
                    extra.get("wind_gust_y", 0.0),
                    extra.get("wind_gust_z", 0.0),
                    extra.get("wind_burst_x", 0.0),
                    extra.get("wind_burst_y", 0.0),
                    extra.get("wind_burst_z", 0.0),
                    extra.get("wind_acc_norm", 0.0),
                ])
        pos = np.asarray(pos)
        dpos = np.asarray(dpos)
        euler = np.asarray(euler)
        deuler = np.asarray(deuler)
        actions = np.asarray(actions)
        wind_rows = np.asarray(wind_rows)
        #
        t = np.arange(0, pos.shape[0])
        ax_x.step(t, pos[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_y.step(t, pos[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_z.step(t, pos[:, 2], color="C{0}".format(
            n_roll), label="pos [x, y, z] -- trail: {0}".format(n_roll))
        #
        ax_dx.step(t, dpos[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_dy.step(t, dpos[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_dz.step(t, dpos[:, 2], color="C{0}".format(
            n_roll), label="vel [x, y, z] -- trail: {0}".format(n_roll))
        #
        ax_euler_x.step(t, euler[:, -1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_euler_y.step(t, euler[:, 0], color="C{0}".format(
            n_roll), label="trail :{0}".format(n_roll))
        ax_euler_z.step(t, euler[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        #
        ax_euler_vx.step(t, deuler[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_euler_vy.step(t, deuler[:, 1], color="C{0}".format(
            n_roll), label="trail :{0}".format(n_roll))
        ax_euler_vz.step(t, deuler[:, 2], color="C{0}".format(
            n_roll), label=r"$\theta$ [x, y, z] -- trail: {0}".format(n_roll))
        # NP2G
        ax_action0.step(t, actions[:, 0] * 0.78 * 9.81 / 2, color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action1.step(t, actions[:, 1] * 0.78 * 9.81 / 2, color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action2.step(t, actions[:, 2] * 0.78 * 9.81 / 2, color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action3.step(t, actions[:, 3] * 0.78 * 9.81 / 2, color="C{0}".format(
            n_roll), label="act [0, 1, 2, 3] -- trail: {0}".format(n_roll))
        if plot_wind and wind_rows.size > 0:
            wind = wind_rows[:, 0:3]
            mean = wind_rows[:, 3:6]
            gust = wind_rows[:, 6:9]
            burst = wind_rows[:, 9:12]
            wind_acc = wind_rows[:, 12]
            wind_norm = np.linalg.norm(wind, axis=1)
            wind_summaries.append((wind_norm.mean(), wind_norm.max(), wind_acc.mean()))

            wind_axes[0].plot(t, wind[:, 0], color="C0", alpha=0.75)
            wind_axes[0].plot(t, wind[:, 1], color="C1", alpha=0.75)
            wind_axes[0].plot(t, wind[:, 2], color="C2", alpha=0.75)
            wind_axes[1].plot(t, mean[:, 0], color="C0", alpha=0.45, linestyle="--")
            wind_axes[1].plot(t, mean[:, 1], color="C1", alpha=0.45, linestyle="--")
            wind_axes[1].plot(t, mean[:, 2], color="C2", alpha=0.45, linestyle="--")
            wind_axes[1].plot(t, gust[:, 0], color="C0", alpha=0.80)
            wind_axes[1].plot(t, gust[:, 1], color="C1", alpha=0.80)
            wind_axes[1].plot(t, gust[:, 2], color="C2", alpha=0.80)
            wind_axes[2].plot(t, burst[:, 0], color="C0", alpha=0.75)
            wind_axes[2].plot(t, burst[:, 1], color="C1", alpha=0.75)
            wind_axes[2].plot(t, burst[:, 2], color="C2", alpha=0.75)
            wind_axes[3].plot(t, wind_norm, color="C3", alpha=0.80)
            wind_axes[3].plot(t, wind_acc, color="C4", alpha=0.80)
        # P2G
        # ax_action0.step(t, actions[:, 0] * 0.78 * 9.81 / 4 + 0.78 * 9.81 / 4, color="C{0}".format(
        #     n_roll), label="trail: {0}".format(n_roll))
        # ax_action1.step(t, actions[:, 1] * 0.78 * 9.81 / 4 + 0.78 * 9.81 / 4, color="C{0}".format(
        #     n_roll), label="trail: {0}".format(n_roll))
        # ax_action2.step(t, actions[:, 2] * 0.78 * 9.81 / 4 + 0.78 * 9.81 / 4, color="C{0}".format(
        #     n_roll), label="trail: {0}".format(n_roll))
        # ax_action3.step(t, actions[:, 3] * 0.78 * 9.81 / 4 + 0.78 * 9.81 / 4, color="C{0}".format(
        #     n_roll), label="act [0, 1, 2, 3] -- trail: {0}".format(n_roll))
    #
    if render:
        env.disconnectUnity()
    ax_z.legend()
    ax_dz.legend()
    ax_euler_z.legend()
    ax_euler_vz.legend()
    ax_action3.legend()
    if plot_wind:
        wind_axes[0].set_ylabel("wind xyz [m/s]")
        wind_axes[1].set_ylabel("mean -- / gust [m/s]")
        wind_axes[2].set_ylabel("burst xyz [m/s]")
        wind_axes[3].set_ylabel("|wind|, |wind acc|")
        wind_axes[3].set_xlabel("step")
        wind_axes[0].legend(["wind x", "wind y", "wind z"], loc="upper right")
        wind_axes[1].legend(["mean x", "mean y", "mean z",
                             "gust x", "gust y", "gust z"], loc="upper right")
        wind_axes[2].legend(["burst x", "burst y", "burst z"], loc="upper right")
        wind_axes[3].legend(["|wind| [m/s]", "|wind acc| [m/s^2]"],
                            loc="upper right")
        wind_fig.tight_layout()
        if wind_summaries:
            summaries = np.asarray(wind_summaries)
            print("[WindTest] mean |wind|={:.2f} m/s, peak |wind|={:.2f} m/s, mean |wind acc|={:.2f} m/s^2".format(
                summaries[:, 0].mean(), summaries[:, 1].max(), summaries[:, 2].mean()))
    #
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import csv
from scipy.spatial.transform import Rotation as R


def btc_test_model(env, model_1, model_2, render=False):
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

    max_ep_length = env.max_episode_steps
    num_rollouts = 1
    
    
    if render:
        env.connectUnity()

    with open("trajectory_BTC.csv","w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['','t','px','py','pz','qw','qx','qy','qz','vx','vy','vz','omex','omey','omez','accx','accy','accz','taux','tauy','tauz','jerkx','jerky','jerkz','snapx','snapy','snapz','bomex','bomey','bomez','baccx','baccy','baccz','mot1','mot2','mot3','mot4','motdex1','motdex2','motdex3','motdex4','f1','f2','f3','f4'])

    for n_roll in range(num_rollouts):
        pos, euler, dpos, deuler = [], [], [], []
        actions = []
        obs, done, ep_len = env.reset(), False, 0
        while not (done or (ep_len >= max_ep_length)):

            euler_angles = [obs[0, 3],obs[0, 4],obs[0, 5]]
            r = R.from_euler('ZYX', euler_angles, degrees=False)
            quaternion = r.as_quat()
                    
            with open("trajectory_BTC.csv","a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([ep_len, ep_len*0.02, obs[0, 0],obs[0, 1],obs[0, 2]+5, quaternion[3], quaternion[0], quaternion[1], quaternion[2], obs[0, 6], obs[0, 7],obs[0, 8],obs[0, 9],obs[0, 10],obs[0, 11],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

            act, _ = model_1.predict(obs, deterministic=True)
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
        pos = np.asarray(pos)
        dpos = np.asarray(dpos)
        euler = np.asarray(euler)
        deuler = np.asarray(deuler)
        actions = np.asarray(actions)
        #
        t = np.arange(0, pos.shape[0])
        ax_x.step(t * 0.02, pos[:, 0], color="C{0}".format(
            11), label="trail: {0}".format(n_roll))
        ax_y.step(t * 0.02, pos[:, 1], color="C{0}".format(
            11), label="trail: {0}".format(n_roll))
        ax_z.step(t * 0.02, pos[:, 2] + 5 , color="C{0}".format(
            11), label="pos [x, y, z] -- BTC".format(n_roll))
        #
        ax_dx.step(t * 0.02, dpos[:, 0], color="C{0}".format(
            11), label="trail: {0}".format(n_roll))
        ax_dy.step(t * 0.02, dpos[:, 1], color="C{0}".format(
            11), label="trail: {0}".format(n_roll))
        ax_dz.step(t * 0.02, dpos[:, 2], color="C{0}".format(
            11), label="vel [x, y, z] -- BTC".format(n_roll))
        #
        ax_euler_x.step(t * 0.02, euler[:, -1], color="C{0}".format(
            11), label="trail: {0}".format(n_roll))
        ax_euler_y.step(t * 0.02, euler[:, 0], color="C{0}".format(
            11), label="trail :{0}".format(n_roll))
        ax_euler_z.step(t * 0.02, euler[:, 1], color="C{0}".format(
            11), label="euler [x, y, z] -- BTC".format(n_roll))
        #
        ax_euler_vx.step(t * 0.02, deuler[:, 0], color="C{0}".format(
            11), label="trail: {0}".format(n_roll))
        ax_euler_vy.step(t * 0.02, deuler[:, 1], color="C{0}".format(
            11), label="trail :{0}".format(n_roll))
        ax_euler_vz.step(t * 0.02, deuler[:, 2], color="C{0}".format(
            11), label="omgea [x, y, z] -- BTC".format(n_roll))
        # NP2G
        ax_action0.step(t * 0.02, actions[:, 0] * 0.78 * 9.81 / 2, color="C{0}".format(
            11), label="trail: {0}".format(n_roll))
        ax_action1.step(t * 0.02, actions[:, 1] * 0.78 * 9.81 / 2, color="C{0}".format(
            11), label="trail: {0}".format(n_roll))
        ax_action2.step(t * 0.02, actions[:, 2] * 0.78 * 9.81 / 2, color="C{0}".format(
            11), label="trail: {0}".format(n_roll))
        ax_action3.step(t * 0.02, actions[:, 3] * 0.78 * 9.81 / 2, color="C{0}".format(
            11), label="act [1, 2, 3, 4] -- BTC".format(n_roll))







    with open("trajectory_OPTC.csv","w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['','t','px','py','pz','qw','qx','qy','qz','vx','vy','vz','omex','omey','omez','accx','accy','accz','taux','tauy','tauz','jerkx','jerky','jerkz','snapx','snapy','snapz','bomex','bomey','bomez','baccx','baccy','baccz','mot1','mot2','mot3','mot4','motdex1','motdex2','motdex3','motdex4','f1','f2','f3','f4'])

    for n_roll in range(num_rollouts):
        pos, euler, dpos, deuler = [], [], [], []
        actions = []
        obs, done, ep_len = env.reset(), False, 0
        while not (done or (ep_len >= max_ep_length)):
                                
            euler_angles = [obs[0, 3],obs[0, 4],obs[0, 5]]
            r = R.from_euler('ZYX', euler_angles, degrees=False)
            quaternion = r.as_quat()
                    
            with open("trajectory_OPTC.csv","a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([ep_len, ep_len*0.02, obs[0, 0],obs[0, 1],obs[0, 2]+5, quaternion[3], quaternion[0], quaternion[1], quaternion[2], obs[0, 6], obs[0, 7],obs[0, 8],obs[0, 9],obs[0, 10],obs[0, 11],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

            # act, _ = model_1.predict(obs, deterministic=True)
            # obs, rew, done, infos = env.step(act)
            act1, _ = model_2.predict(obs, deterministic=True)
            act = (act1 + 1)/2
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
        pos = np.asarray(pos)
        dpos = np.asarray(dpos)
        euler = np.asarray(euler)
        deuler = np.asarray(deuler)
        actions = np.asarray(actions)
        #
        t = np.arange(0, pos.shape[0])
        ax_x.step(t * 0.02, pos[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_y.step(t * 0.02, pos[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_z.step(t * 0.02, pos[:, 2] + 5 , color="C{0}".format(
            n_roll), label="pos [x, y, z] -- OPTC".format(n_roll))
        #
        ax_dx.step(t * 0.02, dpos[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_dy.step(t * 0.02, dpos[:, 1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_dz.step(t * 0.02, dpos[:, 2], color="C{0}".format(
            n_roll), label="vel [x, y, z] -- OPTC".format(n_roll))
        #
        ax_euler_x.step(t * 0.02, euler[:, -1], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_euler_y.step(t * 0.02, euler[:, 0], color="C{0}".format(
            n_roll), label="trail :{0}".format(n_roll))
        ax_euler_z.step(t * 0.02, euler[:, 1], color="C{0}".format(
            n_roll), label="euler [x, y, z] -- OPTC".format(n_roll))
        #
        ax_euler_vx.step(t * 0.02, deuler[:, 0], color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_euler_vy.step(t * 0.02, deuler[:, 1], color="C{0}".format(
            n_roll), label="trail :{0}".format(n_roll))
        ax_euler_vz.step(t * 0.02, deuler[:, 2], color="C{0}".format(
            n_roll), label="omgea [x, y, z] -- OPTC".format(n_roll))
        #
        ax_action0.step(t * 0.02, actions[:, 0] * 0.78 * 9.81 / 2, color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action1.step(t * 0.02, actions[:, 1] * 0.78 * 9.81 / 2, color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action2.step(t * 0.02, actions[:, 2] * 0.78 * 9.81 / 2, color="C{0}".format(
            n_roll), label="trail: {0}".format(n_roll))
        ax_action3.step(t * 0.02, actions[:, 3] * 0.78 * 9.81 / 2, color="C{0}".format(
            n_roll), label="act [1, 2, 3, 4] -- OPTC".format(n_roll))

    




    if render:
        env.disconnectUnity()
    ax_z.legend()
    ax_dz.legend()
    ax_euler_z.legend()
    ax_euler_vz.legend()
    ax_action3.legend()
    #
    plt.tight_layout()
    plt.show()

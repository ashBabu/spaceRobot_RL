import gym
# import spacecraftRobot
import time
from spacerobot_env import SpaceRobotEnv
import numpy as np
np.set_printoptions(precision=3)

# actions_true = np.load('opt_actions_true_dyn.npy', allow_pickle=True)
# actions = np.load('opt_actions.npy', allow_pickle=True)
# actions_lab = np.load('opt_actions_frm_labsys.npy', allow_pickle=True)
# actions = np.load('actions.npy', allow_pickle=True)
actions = np.load('actions_server.npy', allow_pickle=True)
"""
# The actions below are found out using MPC & env.step()
# dynamics_true = actions_trueDyn_batch.npy but very slow
# dynamics = None (fast)
actions = np.load('actions_trueDyn.npy', allow_pickle=True)
# actions = np.load('actions_trueDyn_batch.npy', allow_pickle=True)
"""
env = SpaceRobotEnv()
env.reset()
for a in actions:
    env.step(a)
    print(env.sim.get_state())
    env.render()
    # time.sleep(0.1)

print('done')

# R2 metric


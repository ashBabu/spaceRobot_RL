import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ddpg_agent import ddpg
from actor_network import Actor
import gym
import spacecraftRobot

from ddpg_agent import replaybuffer

env = gym.make('SpaceRobotReach-v0')
# env = gym.make('FetchReach-v1')
a_dim = env.action_space.shape[0]
s_dim = env.observation_space

replaybuffer = replaybuffer.ReplayBuffer(1e06)
actor = Actor(s_dim, a_dim).model()



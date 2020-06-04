"""
A fast Implementation of Deep Deterministic Policy Gradient in Tensorflow 2.

Original Paper:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: Ash Babu
"""

import numpy as np
import gym
import tensorflow as tf
import os
from .replaybuffer import ReplayBuffer
from .protect_loop import Protect_loop
from .actor_critic import Actor, Critic
tf.keras.backend.set_floatx('float64')


class DDPG:
    def __init__(self, env=gym.make('Pendulum-v0'), s_dim=2, a_dim=1, gamma=0.99, tau=0.001, buffer_size=1e06,
                 minibatch_size=64, actor_lr=0.001, critic_lr=0.001):
        self.env = env
        self.EPSILON = 0.4
        self.EPISODES = 100
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.GAMMA = gamma
        self.TAU = tau
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor = Actor(self.s_dim, self.a_dim).model()
        self.target_actor = Actor(self.s_dim, self.a_dim).model()
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.target_actor.set_weights(self.actor.get_weights())

        self.critic = Critic(self.s_dim, self.a_dim).model()
        self.critic.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(self.critic_lr))
        self.target_critic = Critic(self.s_dim, self.a_dim).model()
        self.target_critic.set_weights(self.critic.get_weights())

        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def update_target(self):
        self.target_actor.set_weights(np.array(self.actor.get_weights()) * self.TAU + np.array(
            self.target_actor.get_weights()) * (1 - self.TAU))
        self.target_critic.set_weights(np.array(self.critic.get_weights()) * self.TAU + np.array(
            self.target_critic.get_weights()) * (1 - self.TAU))

    def train_step(self):
        s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample_batch(
            self.minibatch_size)
        mu_prime = self.target_actor(s2_batch)  # predictions by target actor
        Q_prime = self.target_critic([s2_batch, mu_prime])  # predictions by target critic
        y = np.zeros_like(Q_prime)
        for k in range(self.minibatch_size):
            if d_batch[k]:
                y[k] = r_batch[k]
            else:
                y[k] = r_batch[k] + self.GAMMA * Q_prime[k]
        checkpoint_path = "training/cp_critic.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # Create a callback that saves the model's weights
        cp_callback1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                          save_weights_only=True,
                                                          verbose=1)
        self.critic.fit([s_batch, a_batch], y, verbose=0, steps_per_epoch=8, callbacks=[cp_callback1])

        with tf.GradientTape(persistent=True) as tape:
            a = self.actor(s_batch)
            tape.watch(a)
            theta = self.actor.trainable_variables
            q = self.critic([s_batch, a])
        dq_da = tape.gradient(q, a)
        da_dtheta = tape.gradient(a, theta, output_gradients=-dq_da)
        self.actor_opt.apply_gradients(zip(da_dtheta, self.actor.trainable_variables))
        self.update_target()

    def train(self, render=False):
        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # log_dir = 'logs/dqn/' + current_time
        # summary_writer = tf.summary.create_file_writer(log_dir)
        with Protect_loop():
            for i in range(self.EPISODES):
                R = 0
                s = self.env.reset()
                for _ in range(50):
                    if render:
                        self.env.render()
                    a = self.actor(s[None, :])
                    s_, r, d, _ = self.env.step(a)
                    self.replay_buffer.add(np.reshape(s, (self.s_dim,)), np.reshape(a, (self.a_dim,)), r,
                                           d, np.reshape(s_, (self.s_dim,)))
                    R += r
                    if self.replay_buffer.size() > self.minibatch_size:
                        self.train_step()
                    # with summary_writer.as_default():
                    #     tf.summary.scalar('episode reward', R, step=5)
                    # tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
                    # tf.summary.scalar('average loss)', losses, step=n)
                    # print('Reward per episode :', R)
            self.actor.save('training/actor.h5')


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    s_dim, a_dim = 3, 1

    # env = Pendulum()
    train = True
    if train:
        ddpg_train = DDPG(env, s_dim=s_dim, a_dim=a_dim)
        ddpg_train.train()
        ddpg_train.target_actor.save_weights('training/target_actor_weights', save_format='tf')
        ddpg_train.target_critic.save_weights('training/target_critic_weights', save_format='tf')
        print('Training completed and network weights saved')
    else:
        pass
    #     # tgt_critic = ddpg.target_critic.load_weights('training_ash/target_critic_weights')
    #     actor_trained = Actor(1, 0.0001)
    #     actor_trained.load_weights('training_ash/target_actor_weights')

    #     test_env = Pendulum(180, store_data=True)
    #     ddpg = DDPG(test_env, actor=actor_trained)
    #     s = test_env.get_obs()
    #     for t in range(100):
    #         # a = actor_trained.predict(s[None, :])[0]
    #         a = ddpg.get_action(s)
    #         s, _, done, _ = test_env.step(a)
    #         if done:
    #             break
    #     print('Trial result: ', end='')
    #     test_env.print_eval()
    #     test_env.plot3D(actor_trained)
    #     # test_env.plot3D(ddpg.get_action)
    #     # test_env.plot3D(actor_trained, tgt_critic)
    #     test_env.plot_trial()
    #     test_env.show()
    # print('hi')
        






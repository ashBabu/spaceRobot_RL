import gym
import multiprocessing as mp
import numpy as np
import numpy.matlib as nm
import time
import copy
np.set_printoptions(precision=3, suppress=True)
# from utils_a import do_env_rollout
# import types


class MPPI:
    def __init__(self, env, H=16, rollouts=1, dynamics=None, reward=None,
                 num_cpu=1,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None,
                 filter_coefs=None,
                 default_act='repeat',
                 warmstart=True,
                 seed=123,
                 ):

        self.env, self.seed = env, seed
        self.env_cpy = copy.deepcopy(env)
        self.env_cpy.reset()
        if not dynamics:
            self.step = True
        else:
            self.step = False
            self.fwd_dyn = dynamics
        if not reward:
            self.reward_fn = self.env.reward
        else:
            self.reward_fn = reward
        self.nx, self.nu = env.observation_space.shape[0], env.action_dim
        self.a_low, self.a_high = self.env.action_space.low, self.env.action_space.high
        self.H, self.rollouts, self.num_cpu = H, rollouts, num_cpu
        self.warmstart = warmstart
        # self.do_env_rollout = types.MethodType(do_env_rollout, self)

        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.nu)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.nu), 1.0, 0.0, 0.0]
        self.default_act = default_act

        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []

        self.env.reset()
        # self.sol_state.append(self.env.get_env_state().copy())
        self.act_sequence = np.ones((self.H, self.nu)) * self.mean
        self.init_act_sequence = self.act_sequence.copy()

    def update(self, act, rewards):
        R = self.score_trajectory(rewards)
        S = np.exp(self.kappa*(R-np.max(R)))
        # blend the action sequence
        weighted_seq = S*act.T
        act_sequence = np.sum(weighted_seq.T, axis=0)/(np.sum(S) + 1e-6)
        self.act_sequence = act_sequence
        print('action updated')

    def advance_time(self, rew, act_sequence=None):
        act_sequence = self.act_sequence if act_sequence is None else act_sequence
        # accept first action and step
        action = act_sequence[0].copy()
        self.sol_act.append(action)
        # self.sol_state.append(self.env.get_env_state().copy())
        self.sol_reward.append(rew)

        # get updated action sequence
        if self.warmstart:
            self.act_sequence[:-1] = act_sequence[1:]
            if self.default_act == 'repeat':
                self.act_sequence[-1] = self.act_sequence[-2]
            else:
                self.act_sequence[-1] = self.mean.copy()
        else:
            self.act_sequence = self.init_act_sequence.copy()

    def score_trajectory(self, rewards):
        rr, rc = rewards.shape
        scores = np.zeros(rr)
        for i in range(rr):
            scores[i] = 0.0
            for t in range(rc):
                scores[i] += (self.gamma**t)*rewards[i, t]
        return scores

    def control(self, state):
        act, rewards = self.generate_paths(state,
                                           self.act_sequence,
                                           self.filter_coefs,
                                           base_seed=2134,)
        self.update(act, rewards)
        return self.act_sequence[0]

    def do_env_rollout(self, start_state, act):
        """
            1) Construct env with env_id and set it to start_state.
            2) Generate rollouts using act_list.
               act_list is a list with each element having size (H,m).
               Length of act_list is the number of desired rollouts.
        """
        N, H, nu = act.shape
        rewards = np.zeros((N, H))
        qp, qv = start_state[:14], start_state[14:]
        if self.step:
            self.env_cpy.reset_model()
            for i in range(N):
                self.env_cpy.set_env_state(qp, qv)
                rews = np.zeros(H)
                for k in range(H):
                    s, r, d, ifo = self.env_cpy.step(act[i][k])
                    rews[k] = r
                rewards[i] = rews
        else:
            next_states = np.zeros((N, H, self.nx))  # N x H x nx
            s0_batch = nm.repmat(start_state, N, 1)

            for t in range(H):
                # t_rew = time.time()
                reward = self.reward_fn(s0_batch, act[:, t, :])
                # print('rew time:', time.time() - t_rew)
                rewards[:, t] = reward
                # t_nxt = time.time()
                next_state = self.fwd_dyn(s0_batch, act[:, t, :])
                # print('next state time:', time.time() - t_nxt)
                next_states[:, t, :] = next_state
                s0_batch = next_state

        return rewards

    def generate_perturbed_actions(self, base_act, filter_coefs):
        """
        Generate perturbed actions around a base action sequence
        """
        sigma, beta_0, beta_1, beta_2 = filter_coefs
        eps = np.random.normal(loc=0, scale=1.0, size=base_act.shape) * sigma
        for i in range(2, eps.shape[0]):
            eps[i] = beta_0 * eps[i] + beta_1 * eps[i - 1] + beta_2 * eps[i - 2]
        return base_act + eps

    def generate_paths(self, start_state, base_act, filter_coefs, base_seed):
        """
        first generate enough perturbed actions
        then do rollouts with generated actions
        set seed inside this function for multiprocessing
        """
        N = self.rollouts
        np.random.seed(base_seed)
        act = np.zeros((N, *base_act.shape))
        for i in range(N):
            act[i, :, :] = self.generate_perturbed_actions(base_act, filter_coefs)
        rewards = self.do_env_rollout(start_state, act)
        return act, rewards


def run_mppi(mppi, env, retrain_dynamics=None, retrain_after_iter=50, iter=200, render=True):
    dataset = np.zeros((retrain_after_iter, mppi.nx + mppi.nu))
    total_reward = 0
    states, actions = [], []
    nn = 0
    for i in range(iter):
        # state = env.env.state.copy()
        # state = env.get_env_state().copy()
        state = env.state_vector()
        command_start = time.perf_counter()
        action = mppi.control(state)
        action = np.clip(action, mppi.a_low, mppi.a_high)
        actions.append(action)
        print('action:', action)
        # elapsed = time.perf_counter() - command_start
        # print('Elaspsed time:', elapsed)
        s, r, _, _ = env.step(action)
        mppi.advance_time(rew=r)
        print(i)
        # time.sleep(.2)
        total_reward += r
        if render:
            env.render()
        di = i % retrain_after_iter
        # if nn <= 7:
        if retrain_dynamics and di == 0 and i > 0:
            retrain_dynamics(dataset)
            nn += 1
                # don't have to clear dataset since it'll be overridden, but useful for debugging
        dataset[di, :mppi.nx] = env.state_vector()
        dataset[di, mppi.nx:] = action
    return total_reward, dataset, actions


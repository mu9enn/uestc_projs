import numpy as np
import gym
from kaggle_environments import make, evaluate
from gym import spaces

class ConnectFourGym(gym.Env):
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        # Define action and observation spaces.
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=(1, self.rows, self.columns),
                                            dtype=int)
        # Define reward range
        self.reward_range = (-10, 1)
        self.spec = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs['board']).reshape(1, self.rows, self.columns)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # Agent wins
            return 1
        elif done:         # Opponent wins
            return -1
        else:
            return 1 / (self.rows * self.columns)

    def step(self, action):
        # Validate move: check if the chosen column is empty at the top.
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:
            self.obs, old_reward, done, info = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:
            reward, done, info = -10, True, {}
        return np.array(self.obs['board']).reshape(1, self.rows, self.columns), reward, done, info

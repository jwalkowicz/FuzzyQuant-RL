import gymnasium as gym


class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(9)

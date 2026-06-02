import numpy as np


class QLearningAgent:
    def __init__(
        self,
        action_size: int,
        observation_size: int,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            action_size: Number of discrete actions
            observation_size: Number of discrete observation states
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.action_size = action_size
        self.observation_size = observation_size

        # Q-table: maps (state, action) to expected reward
        self.q_values = np.zeros((observation_size, action_size))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: int) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0: Hold, 1: Buy, 2: Sell
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return int(np.argmax(self.q_values[obs]))

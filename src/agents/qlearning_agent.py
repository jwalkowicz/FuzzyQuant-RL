import numpy as np


class QLearningAgent:
    """
    Standard Q-Learning agent with epsilon-greedy exploration.
    
    Uses a Q-table to store and update state-action values based on 
    the Temporal Difference (TD) learning algorithm.
    """

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
        """
        Initialize a Q-Learning agent.

        Args:
            action_size: Number of discrete actions.
            observation_size: Number of discrete observation states.
            learning_rate: How quickly to update Q-values.
            initial_epsilon: Starting exploration rate.
            epsilon_decay: How much to reduce epsilon each episode.
            final_epsilon: Minimum exploration rate.
            discount_factor: How much to value future rewards.
        """
        self.action_size = action_size
        self.observation_size = observation_size
        self.q_values = np.zeros((observation_size, action_size))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs: int) -> int:
        """
        Choose an action using epsilon-greedy strategy.

        Args:
            obs: The current discrete observation state.

        Returns:
            int: Action index (0: Hold, 1: Buy, 2: Sell).
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: int,
    ) -> None:
        """
        Update the Q-table using the Bellman equation.

        Args:
            obs: Current state.
            action: Action taken.
            reward: Reward received.
            terminated: Whether the episode ended.
            next_obs: Resulting next state.
        """
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs, action]

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self) -> None:
        """
        Reduce exploration rate (epsilon) according to the decay schedule.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

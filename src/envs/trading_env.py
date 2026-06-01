import gymnasium as gym
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from src.fuzzy.mock_converter import FuzzyStateConverter


class TradingEnv(gym.Env):
    """
    Gymnasium environment for simulated stock trading.

    Provides a discrete action space (Hold, Buy, Sell) and observation space
    based on fuzzy logic states.
    """

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0) -> None:
        """
        Initialize the trading environment.

        Args:
            data (pd.DataFrame): Historical price and technical data.
            initial_balance (float): Starting cash balance.
        """
        self.data: pd.DataFrame = data
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(
            3
        )  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space: gym.spaces.Discrete = gym.spaces.Discrete(
            9
        )  # States 0-8 from fuzzy module
        self.initial_balance: float = initial_balance
        self.balance: float = initial_balance
        self.shares_held: int = 0
        self.current_step: int = 0

        self.fuzzy_converter: FuzzyStateConverter = FuzzyStateConverter()

    def _get_obs(self) -> int:
        """
        Convert current market data to a fuzzy logic state observation.

        Returns:
            int: Discrete fuzzy state index (0-8).
        """
        current_row: pd.Series = self.data.iloc[self.current_step]
        return self.fuzzy_converter.get_state(current_row)

    def _get_info(self) -> Dict[str, Any]:
        """
        Compute auxiliary environment metrics for debugging and tracking.

        Returns:
            dict: Current portfolio performance metrics (net worth, balance, shares).
        """
        current_price: float = float(self.data.iloc[self.current_step]["Close"])
        net_worth: float = self.balance + (self.shares_held * current_price)
        return {
            "net_worth": net_worth,
            "balance": self.balance,
            "shares": self.shares_held,
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Reset the environment state to start a new episode.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[dict]): Additional configuration.

        Returns:
            tuple: Initial (observation, info) pair.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.

        Args:
            action (int): Action to take (0: Hold, 1: Buy, 2: Sell).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        prev_net_worth = self._get_info()["net_worth"]
        current_price = self.data.iloc[self.current_step]["Close"]

        if action == 1:  # Buy
            if self.balance > 0:
                shares_bought = self.balance // current_price
                self.shares_held += shares_bought
                self.balance -= shares_bought * current_price

        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1

        if not terminated:
            obs = self._get_obs()
            info = self._get_info()
            reward = info["net_worth"] - prev_net_worth
        else:
            current_net_worth = self.balance + (self.shares_held * current_price)
            reward = current_net_worth - prev_net_worth
            obs = 0
            info = {
                "net_worth": current_net_worth,
                "balance": self.balance,
                "shares": self.shares_held,
            }

        return obs, reward, terminated, False, info

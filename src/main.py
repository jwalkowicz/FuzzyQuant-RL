import typer
from src.core.logger import logger
from src.core.config import config
from src.data.indicators import get_technical_data
from src.envs.trading_env import TradingEnv
from src.agents.qlearning_agent import QLearningAgent
from tqdm import tqdm
import numpy as np

app = typer.Typer(help="fuzzyQuant-RL")


@app.command(name="run")
def run():
    logger.info("Application starting...")

    data = get_technical_data(
        ticker=config.yfinance.ticker,
        period=config.yfinance.period,
        interval=config.yfinance.interval,
        rsi_window=config.indicators.rsi_window,
    )

    train_data, test_data = _train_test_split_data(data)

    train_env = TradingEnv(
        data=train_data,
        initial_balance=config.trading.initial_balance,
        action_size=config.trading.action_size,
        observation_size=config.trading.observation_size,
    )

    test_env = TradingEnv(
        data=test_data,
        initial_balance=config.trading.initial_balance,
        action_size=config.trading.action_size,
        observation_size=config.trading.observation_size,
    )

    agent = QLearningAgent(
        action_size=config.trading.action_size,
        observation_size=config.trading.observation_size,
        learning_rate=config.agent.learning_rate,
        initial_epsilon=config.agent.initial_epsilon,
        epsilon_decay=config.agent.epsilon_decay,
        final_epsilon=config.agent.final_epsilon,
        discount_factor=config.agent.discount_factor,
    )
    train_agent(train_env, agent)
    evaluate_agent(env, agent, test_data)


def _train_test_split_data(data):
    return (
        data.loc[data["Date"] < "2025-01-01"].copy(),
        data.loc[data["Date"] >= "2025-01-01"].copy(),
    )


def train_agent(env, agent):
    rewards_history = []
    for _ in tqdm(range(config.training.n_episodes)):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)

            episode_reward += reward

            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

    np.save("q_table_fuzzy_quant.npy", agent.q_values)
    avg_reward = np.mean(rewards_history[-100:])
    logger.info(f"Average reward over last 100 episodes: {avg_reward:.2f}")


def evaluate_agent(env, agent, data):
    try:
        agent.q_values = np.load("q_table_fuzzy_quant.npy")
        logger.info("Successfully loaded pre-trained Q-table.")
    except FileNotFoundError:
        logger.error("No Q-table found. Train the agent first.")
        return

    history = []
    agent.epsilon = 0.0
    obs, info = env.reset()
    done = False

    for _ in tqdm(range(len(data))):
        while not done:
            action = agent.get_action(obs)
            next_obs, _, terminated, truncated, info = env.step(action)

            history.append(
                {
                    "step": env.current_step,
                    "action": action,
                    "balance": info.get("balance", 0),
                }
            )

            done = terminated or truncated
            obs = next_obs

    final_balance = history[-1]["balance"]
    logger.info(f"Final Balance in 2025: {final_balance}")


if __name__ == "__main__":
    app()

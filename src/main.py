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

    if not data.empty:
        logger.info(f"Processing successful. Latest data point:\n{data.tail(1)}")

        env = TradingEnv(
            data=data,
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

        logger.info("Environment and Agent initialized successfully.")
    else:
        logger.error("Data acquisition failed. Check logs for details.")

    train_agent(env, agent)


def train_agent(env, agent):
    rewards_history = []
    for episode in tqdm(range(config.training.n_episodes)):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)

            episode_reward += reward

            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    rewards_history.append(episode_reward)
    np.save("q_table_fuzzy_quant.npy", agent.q_values)
    avg_reward = np.mean(rewards_history[-100:])
    logger.info(f"Average reward over last 100 episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    app()

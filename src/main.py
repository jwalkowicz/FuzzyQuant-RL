import typer
from src.core.logger import logger
from src.core.config import config
from src.data.indicators import get_technical_data
from src.envs.trading_env import TradingEnv
from src.agents.qlearning_agent import QLearningAgent
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    train_history = train_agent(train_env, agent)
    draw_training_chart(train_history, agent)
    test_history = evaluate_agent(test_env, agent, test_data)
    compare_with_hold_strategy(test_history, test_data)


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
    return rewards_history


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
                    "net_worth": info.get("net_worth", 0),
                }
            )

            done = terminated or truncated
            obs = next_obs

    final_balance = history[-1]["net_worth"]
    logger.info(f"Final Balance in 2025: {final_balance}")

    return history


def calculate_hold_strategy(data):
    entry_price = data.iloc[0]["Close"]
    shares_held = int(config.trading.initial_balance // entry_price)

    leftover_cash = config.trading.initial_balance - (shares_held * entry_price)

    return (data["Close"] * shares_held) + leftover_cash


def compare_with_hold_strategy(agent_history, test_data):
    dates = pd.to_datetime(test_data["Date"])
    initial_cash = config.trading.initial_balance
    agent_balances = [initial_cash] + [h["net_worth"] for h in agent_history]
    hold_balances = calculate_hold_strategy(test_data)

    plt.figure(figsize=(12, 6))

    plt.plot(dates, agent_balances, label="RL Agent", color="blue", linewidth=2)
    plt.plot(
        dates,
        hold_balances,
        label="Buy & Hold Baseline",
        color="orange",
        linestyle="--",
    )

    plt.title("Trading Bot vs Buy & Hold (2025 Test OOS)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()


def draw_training_chart(rewards_history, agent):
    def _get_moving_avgs(arr, window, convolution_mode):
        return (
            np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_mode)
            / window
        )

    rolling_length = 5
    _, axs = plt.subplots(ncols=2, figsize=(12, 5))

    axs[0].set_title("Episode Rewards (Learning Curve)")
    reward_moving_average = _get_moving_avgs(rewards_history, rolling_length, "valid")
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average, color="green")
    axs[0].set_ylabel("Total Reward")
    axs[0].set_xlabel("Episode")

    axs[1].set_title("TD Error (Network Convergence)")
    training_error_moving_average = _get_moving_avgs(
        agent.training_error, rolling_length, "same"
    )
    axs[1].plot(
        range(len(training_error_moving_average)),
        training_error_moving_average,
        color="red",
        alpha=0.6,
    )
    axs[1].set_ylabel("Temporal Difference Error")
    axs[1].set_xlabel("Step (Total Env Steps)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()

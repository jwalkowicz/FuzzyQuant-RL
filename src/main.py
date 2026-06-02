import typer
from src.core.logger import logger
from src.core.config import config
from src.data.indicators import get_technical_data
from src.envs.trading_env import TradingEnv
from src.agents.qlearning_agent import QLearningAgent

app = typer.Typer(help="fuzzyQuant-RL")

@app.command(name="run")
def run():
    logger.info("Application starting...")
    
    data = get_technical_data(
        ticker=config.yfinance.ticker,
        period=config.yfinance.period,
        interval=config.yfinance.interval,
        rsi_window=config.indicators.rsi_window
    )
    
    if not data.empty:
        logger.info(f"Processing successful. Latest data point:\n{data.tail(1)}")
        
        # Instantiate environment with config values
        env = TradingEnv(
            data=data,
            initial_balance=config.trading.initial_balance,
            action_size=config.trading.action_size,
            observation_size=config.trading.observation_size
        )
        
        # Instantiate agent with config values
        agent = QLearningAgent(
            action_size=config.trading.action_size,
            observation_size=config.trading.observation_size,
            learning_rate=config.agent.learning_rate,
            initial_epsilon=config.agent.initial_epsilon,
            epsilon_decay=config.agent.epsilon_decay,
            final_epsilon=config.agent.final_epsilon,
            discount_factor=config.agent.discount_factor
        )
        
        logger.info("Environment and Agent initialized successfully.")
    else:
        logger.error("Data acquisition failed. Check logs for details.")

if __name__ == "__main__":
    app()

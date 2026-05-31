import typer
from src.core.logger import logger
from src.core.config import config
from src.data.indicators import get_technical_data

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
    else:
        logger.error("Data acquisition failed. Check logs for details.")

if __name__ == "__main__":
    app()

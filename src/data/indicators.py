import yfinance as yf
import pandas as pd
import ta
from src.core.logger import logger


def get_technical_data(
    ticker: str, period: str, interval: str, rsi_window: int
) -> pd.DataFrame:
    """
    Downloads historical price data and computes technical indicators.

    Args:
        ticker: The stock ticker symbol.
        period: The data period.
        interval: The data interval.
        rsi_window: The window size for RSI calculation.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing Date, Close, RSI, and MACD.
    """
    logger.info(
        f"Initiating data download for {ticker} (Period: {period}, Interval: {interval})"
    )

    df = yf.download(ticker, period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns.name = None

    rsi_indicator = ta.momentum.RSIIndicator(close=df["Close"], window=rsi_window)
    df["RSI"] = rsi_indicator.rsi()

    macd_indicator = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd_indicator.macd()

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else "Datetime"

    df = df[[date_col, "Close", "RSI", "MACD"]]
    df.rename(columns={date_col: "Date"}, inplace=True)

    df_clean = df.dropna()
    return df_clean

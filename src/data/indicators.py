import yfinance as yf
import pandas as pd
import ta


def get_spy_technical_data():
    df = yf.download("SPY", period="10y", interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    rsi_indicator = ta.momentum.RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()

    macd_indicator = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd_indicator.macd()

    df = df.reset_index()
    df.rename(columns={"index": "Date"}, inplace=True)
    df = df[["Date", "Close", "RSI", "MACD"]]
    df_clean = df.dropna()
    return df_clean

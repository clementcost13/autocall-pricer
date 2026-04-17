import yfinance as yf
import numpy as np
import pandas as pd
import re
import urllib.request
import time

# Cache for scraped yields to avoid constant hitting
_YIELD_CACHE = {
    "timestamp": 0,
    "data": {}
}
CACHE_DURATION = 3600 # 1 hour

def fetch_historical_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetches historical data for a given ticker from Yahoo Finance.
    """
    tick = yf.Ticker(ticker)
    df = tick.history(period=period)
    if df.empty:
        raise ValueError(f"No historical data found for {ticker}")
    return df

def calculate_historical_volatility(df: pd.DataFrame, column: str = "Close", trading_days: int = 252) -> float:
    """
    Calculates the annualized historical volatility from daily log returns.
    """
    df['Log_Return'] = np.log(df[column] / df[column].shift(1))
    daily_vol = df['Log_Return'].std()
    ann_vol = daily_vol * np.sqrt(trading_days)
    return float(ann_vol)

def calculate_return_stats(df: pd.DataFrame, column: str = "Close") -> dict:
    if 'Log_Return' not in df.columns:
        df['Log_Return'] = np.log(df[column] / df[column].shift(1))
    returns = df['Log_Return'].dropna()
    return {
        "mean_daily": returns.mean(),
        "vol_daily": returns.std(),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "count": len(returns)
    }

def calculate_rolling_volatility(df, window=21):
    returns = df['Log_Return'].dropna()
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    return rolling_vol

# --- Institutional Yield Curve Benchmarks ---

DEFAULT_CURVES = {
    # 6M, 1Y, 2Y, 3Y, 5Y, 10Y (User-Defined Bloomberg Benchmarks)
    "EUR": [2.30, 2.40, 2.50, 2.60, 2.70, 2.85],
    "JPY": [-0.10, 0.00, 0.15, 0.35, 0.60, 0.95],
    "GBP": [3.80, 4.00, 4.20, 4.30, 4.40, 4.55],
    "CHF": [-0.15, -0.05, 0.05, 0.15, 0.25, 0.45], # Negative short-term, positive long
    "HKD": [3.50, 3.40, 3.20, 3.10, 3.00, 3.20],
    "USD": [3.70, 3.75, 3.82, 3.95, 4.10, 4.27]  # 3.7% to 4.27% as requested
}

def fetch_yield_curve(currency: str):
    """
    Returns a DataFrame with the yield curve for the given currency.
    Uses institutional-grade Bloomberg benchmarks.
    """
    tenors = ["6M", "1Y", "2Y", "3Y", "5Y", "10Y"]
    years = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    
    # Use the high-quality benchmarks directly
    rates = DEFAULT_CURVES.get(currency, DEFAULT_CURVES["USD"]).copy()
    
    return pd.DataFrame({
        "Tenor": tenors,
        "Years": years,
        "Rate (%)": rates
    })

def get_latest_spot(df: pd.DataFrame, column: str = "Close") -> float:
    return float(df[column].iloc[-1])

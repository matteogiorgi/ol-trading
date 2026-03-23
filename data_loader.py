from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily price data for a ticker and return a clean DataFrame
    with a single 'price' column.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data downloaded for ticker {ticker!r}.")

    # yfinance may return columns like Open, High, Low, Close, Volume.
    # With auto_adjust=True, Close is already adjusted.
    if "Close" not in df.columns:
        raise ValueError("Downloaded data does not contain a 'Close' column.")

    out = df[["Close"]].copy()
    out.columns = ["price"]
    out.dropna(inplace=True)

    if out.empty:
        raise ValueError("Price series is empty after cleaning.")

    return out


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily simple returns to the DataFrame.
    """
    out = df.copy()
    out["return"] = out["price"].pct_change()
    out.dropna(inplace=True)

    if out.empty:
        raise ValueError("Return series is empty after pct_change().")

    return out

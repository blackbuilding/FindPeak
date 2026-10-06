from datetime import datetime, timedelta
import os

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ACCOUNT = 5056221297
SERVER = "MetaQuotes-Demo"

SYMBOL = "AAPL"
TIMEFRAME = mt5.TIMEFRAME_D1
YEARS_BACK = 2
MA_WINDOWS = (10, 20, 50)


def get_asset_data(symbol: str, timeframe: int, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    password = os.environ.get("MT5_PASSWORD")
    if not password:
        raise RuntimeError("Set the MT5_PASSWORD environment variable before connecting to MetaTrader 5")

    if not mt5.initialize(login=ACCOUNT, password=password, server=SERVER):
        error = mt5.last_error()
        raise RuntimeError(f"MT5 initialize() failed: {error}")

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol {symbol}: {mt5.last_error()}")

        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned for {symbol}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
    finally:
        mt5.shutdown()


def detect_peaks(df: pd.DataFrame, column: str = "close", **kwargs):
    peaks, properties = find_peaks(df[column].to_numpy(),**kwargs)
    return peaks, properties


def add_moving_averages(df: pd.DataFrame, windows=MA_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"MA{w}"] = df["close"].rolling(window=w).mean()
    return df


def plot_asset_data(df: pd.DataFrame, symbol: str, peaks: np.ndarray = None, out_path: str = "aapl_d1.png") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["time"], df["close"], label=f"{symbol} close")
    if peaks is not None:
        ax.plot(df["time"].iloc[peaks], df["close"].iloc[peaks], "rx", label="peaks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{symbol} D1 close price - last {YEARS_BACK} years")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")


def plot_ma_peaks(df: pd.DataFrame, symbol: str, windows=MA_WINDOWS, out_path: str = "aapl_ma_peaks.png") -> None:
    fig, axes = plt.subplots(len(windows), 1, figsize=(12, 4 * len(windows)), sharex=True)
    for ax, w in zip(axes, windows):
        col = f"MA{w}"
        valid = df.dropna(subset=[col])
        peaks, _ = detect_peaks(valid, column=col)
        peak_idx = valid.index[peaks]

        ax.plot(df["time"], df[col], label=col)
        ax.plot(df.loc[peak_idx, "time"], df.loc[peak_idx, col], "rx", label="peaks")
        ax.set_ylabel("Price")
        ax.set_title(f"{symbol} {col} with peaks")
        ax.legend()

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved MA peaks plot to {out_path}")


if __name__ == "__main__":
    date_to = datetime.now()
    date_from = date_to - timedelta(days=365 * YEARS_BACK)

    data = get_asset_data(SYMBOL, TIMEFRAME, date_from, date_to)
    print(data)

    peaks, _ = detect_peaks(data)
    print(f"Found {len(peaks)} peaks")

    plot_asset_data(data, SYMBOL, peaks=peaks)

    data = add_moving_averages(data)
    plot_ma_peaks(data, SYMBOL)

from datetime import datetime, timedelta
import os

import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ACCOUNT = 5056221297
SERVER = "MetaQuotes-Demo"

SYMBOL = "AAPL"
TIMEFRAME = mt5.TIMEFRAME_D1
YEARS_BACK = 2
MA_WINDOW = 10
RSI_PERIOD = 14


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


def detect_peaks_and_valleys(df: pd.DataFrame, column: str, **kwargs):
    values = df[column].to_numpy()
    peaks, peak_props = find_peaks(values, **kwargs)
    valleys, valley_props = find_peaks(-values, **kwargs)
    return peaks, valleys, peak_props, valley_props


def add_ma10(df: pd.DataFrame, window: int = MA_WINDOW) -> pd.DataFrame:
    df = df.copy()
    df[f"MA{window}"] = df["close"].rolling(window=window).mean()
    return df


def add_rsi(df: pd.DataFrame, column: str, period: int = RSI_PERIOD, out_col: str = None) -> pd.DataFrame:
    df = df.copy()
    out_col = out_col or f"RSI{period}"

    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df[out_col] = 100 - (100 / (1 + rs))
    return df


def _plot_series_with_peaks_valleys(ax, df: pd.DataFrame, col: str, color: str):
    valid = df.dropna(subset=[col])
    peaks, valleys, _, _ = detect_peaks_and_valleys(valid, column=col)
    peak_idx = valid.index[peaks]
    valley_idx = valid.index[valleys]

    ax.plot(df["time"], df[col], label=col, color=color)
    ax.plot(df.loc[peak_idx, "time"], df.loc[peak_idx, col], "rx", label="peaks")
    ax.plot(df.loc[valley_idx, "time"], df.loc[valley_idx, col], "go", label="valleys")


def plot_ma10_rsi_sequence(df: pd.DataFrame, symbol: str, out_path: str = "RSI_MA10.pdf") -> None:
    ma_col = f"MA{MA_WINDOW}"
    rsi_close_col = f"RSI{RSI_PERIOD}_close"
    rsi_ma10_col = f"RSI{RSI_PERIOD}_MA{MA_WINDOW}"

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    _plot_series_with_peaks_valleys(axes[0], df, ma_col, "tab:blue")
    axes[0].set_ylabel(ma_col)
    axes[0].set_title(f"{symbol} {ma_col} with peaks and valleys")
    axes[0].legend()

    _plot_series_with_peaks_valleys(axes[1], df, rsi_close_col, "tab:purple")
    axes[1].axhline(70, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(30, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel(rsi_close_col)
    axes[1].set_title(f"{symbol} RSI{RSI_PERIOD} (input: close) with peaks and valleys")
    axes[1].legend()

    _plot_series_with_peaks_valleys(axes[2], df, rsi_ma10_col, "tab:orange")
    axes[2].axhline(70, color="gray", linestyle="--", linewidth=0.8)
    axes[2].axhline(30, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel(rsi_ma10_col)
    axes[2].set_title(f"{symbol} RSI{RSI_PERIOD} (input: {ma_col}) with peaks and valleys")
    axes[2].legend()

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved RSI/MA10 plot to {out_path}")


if __name__ == "__main__":
    date_to = datetime.now()
    date_from = date_to - timedelta(days=365 * YEARS_BACK)

    data = get_asset_data(SYMBOL, TIMEFRAME, date_from, date_to)
    print(data)

    data = add_ma10(data)
    data = add_rsi(data, column="close", period=RSI_PERIOD, out_col=f"RSI{RSI_PERIOD}_close")
    data = add_rsi(data, column=f"MA{MA_WINDOW}", period=RSI_PERIOD, out_col=f"RSI{RSI_PERIOD}_MA{MA_WINDOW}")

    plot_ma10_rsi_sequence(data, SYMBOL)

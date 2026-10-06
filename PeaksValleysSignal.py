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
RSI_PERIOD = 14
RSI_PERIODS = (14, 20, 50)


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


def detect_peaks_and_valleys(df: pd.DataFrame, column: str, **kwargs):
    values = df[column].to_numpy()
    peaks, peak_props = find_peaks(values, **kwargs)
    valleys, valley_props = find_peaks(-values, **kwargs)
    return peaks, valleys, peak_props, valley_props


def add_moving_averages(df: pd.DataFrame, windows=MA_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"MA{w}"] = df["close"].rolling(window=w).mean()
    return df


def add_rsi(df: pd.DataFrame, column: str = "close", period: int = RSI_PERIOD) -> pd.DataFrame:
    df = df.copy()
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df[f"RSI{period}"] = 100 - (100 / (1 + rs))
    return df


def add_rsi_multi(df: pd.DataFrame, column: str = "close", periods=MA_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    for period in periods:
        df = add_rsi(df, column=column, period=period)
    return df


def find_entry_signals(
    df: pd.DataFrame,
    ma_column: str = "MA10",
    rsi_column: str = "RSI14",
    max_gap: int = 3,
) -> pd.DataFrame:
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")

    valid_positions = np.flatnonzero(df[ma_column].notna() & df[rsi_column].notna())
    valid = df.iloc[valid_positions]
    if len(valid) < 3:
        return pd.DataFrame()

    ma_values = valid[ma_column].to_numpy()
    rsi_values = valid[rsi_column].to_numpy()
    ma_peaks, _ = find_peaks(ma_values)
    ma_valleys, _ = find_peaks(-ma_values)
    rsi_peaks, _ = find_peaks(rsi_values)
    rsi_valleys, _ = find_peaks(-rsi_values)

    entries = []
    for side, ma_extrema, rsi_extrema in (
        ("LONG", ma_valleys, rsi_valleys),
        ("SHORT", ma_peaks, rsi_peaks),
    ):
        used_rsi_extrema = set()
        for ma_extreme in ma_extrema:
            ma_position = valid_positions[ma_extreme]
            candidates = [
                rsi_extreme
                for rsi_extreme in rsi_extrema
                if rsi_extreme not in used_rsi_extrema
                and abs(valid_positions[rsi_extreme] - ma_position) <= max_gap
            ]
            if not candidates:
                continue

            rsi_extreme = min(
                candidates,
                key=lambda position: abs(valid_positions[position] - ma_position),
            )
            used_rsi_extrema.add(rsi_extreme)
            rsi_position = valid_positions[rsi_extreme]
            pivot_position = max(ma_position, rsi_position)
            entry_position = pivot_position + 2
            if entry_position >= len(df):
                continue

            entry_row = df.iloc[entry_position]
            ma_row = df.iloc[ma_position]
            rsi_row = df.iloc[rsi_position]
            entries.append(
                {
                    "side": side,
                    "ma10_pivot_date": ma_row["time"] if "time" in df else df.index[ma_position],
                    "rsi14_pivot_date": rsi_row["time"] if "time" in df else df.index[rsi_position],
                    "entry_date": entry_row["time"] if "time" in df else df.index[entry_position],
                    "entry_price": entry_row["open"] if "open" in df else entry_row["close"],
                    "entry_ma10": entry_row[ma_column],
                    "MA10": ma_row[ma_column],
                    "RSI14": rsi_row[rsi_column],
                }
            )

    return pd.DataFrame(entries).sort_values("entry_date", ignore_index=True) if entries else pd.DataFrame()


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


def _plot_series_with_peaks_valleys(ax, df: pd.DataFrame, col: str, color: str, marker_peak: str, marker_valley: str):
    valid = df.dropna(subset=[col])
    peaks, valleys, _, _ = detect_peaks_and_valleys(valid, column=col)
    peak_idx = valid.index[peaks]
    valley_idx = valid.index[valleys]

    line, = ax.plot(df["time"], df[col], label=col, color=color)
    ax.plot(df.loc[peak_idx, "time"], df.loc[peak_idx, col], marker_peak, label=f"{col} peaks")
    ax.plot(df.loc[valley_idx, "time"], df.loc[valley_idx, col], marker_valley, label=f"{col} valleys")
    return line


def plot_ma_rsi_sequence(
    df: pd.DataFrame,
    symbol: str,
    ma_windows=MA_WINDOWS,
    rsi_periods=RSI_PERIODS,
    out_path: str = "aapl_ma_rsi_sequence.pdf",
) -> None:
    pairs = list(zip(ma_windows, rsi_periods))
    fig, axes = plt.subplots(2 * len(pairs), 1, figsize=(12, 4 * 2 * len(pairs)), sharex=True)

    ax_idx = 0
    for ma_w, rsi_p in pairs:
        ma_col = f"MA{ma_w}"
        ax_ma = axes[ax_idx]
        _plot_series_with_peaks_valleys(ax_ma, df, ma_col, "tab:blue", "rx", "go")
        ax_ma.set_ylabel(ma_col)
        ax_ma.set_title(f"{symbol} {ma_col} with peaks and valleys")
        ax_ma.legend(loc="upper left", fontsize="small")
        ax_idx += 1

        rsi_col = f"RSI{rsi_p}"
        ax_rsi = axes[ax_idx]
        _plot_series_with_peaks_valleys(ax_rsi, df, rsi_col, "tab:purple", "rx", "go")
        ax_rsi.axhline(70, color="gray", linestyle="--", linewidth=0.8)
        ax_rsi.axhline(30, color="gray", linestyle="--", linewidth=0.8)
        ax_rsi.set_ylabel(rsi_col)
        ax_rsi.set_title(f"{symbol} {rsi_col} with peaks and valleys")
        ax_rsi.legend(loc="upper left", fontsize="small")
        ax_idx += 1

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved MA/RSI sequence plot to {out_path}")


def plot_ma10_entry_signals(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    symbol: str,
    out_path: str = "aapl_ma10_signals.png",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["time"], df["MA10"], label="MA10", color="tab:blue")

    for side, color, marker in (("LONG", "tab:green", "^"), ("SHORT", "tab:red", "v")):
        signals = entries[entries["side"] == side]
        if not signals.empty:
            ax.scatter(
                signals["entry_date"],
                signals["entry_ma10"],
                color=color,
                marker=marker,
                label=f"{side} entry",
                zorder=3,
            )

    ax.set_xlabel("Date")
    ax.set_ylabel("MA10")
    ax.set_title(f"{symbol} MA10 with long and short entries")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved MA10 entry signal plot to {out_path}")


if __name__ == "__main__":
    date_to = datetime.now()
    date_from = date_to - timedelta(days=365 * YEARS_BACK)

    data = get_asset_data(SYMBOL, TIMEFRAME, date_from, date_to)
    print(data)

    peaks, _ = detect_peaks(data)
    print(f"Found {len(peaks)} peaks")

    plot_asset_data(data, SYMBOL, peaks=peaks)

    data = add_moving_averages(data)
    data = add_rsi_multi(data, periods=RSI_PERIODS)
    entries = find_entry_signals(data)
    print("MA10/RSI14 confluence entries (entry at next available open):")
    print(entries.to_string(index=False) if not entries.empty else "No matching entries found")
    plot_ma_rsi_sequence(data, SYMBOL)
    plot_ma10_entry_signals(data, entries, SYMBOL)

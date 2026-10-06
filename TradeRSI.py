from datetime import datetime, timedelta
import os

import MetaTrader5 as mt5
import numpy as np
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

STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.30
INITIAL_CAPITAL = 10000.0

MA_COL = f"MA{MA_WINDOW}"
RSI_CLOSE_COL = f"RSI{RSI_PERIOD}_close"
RSI_MA10_COL = f"RSI{RSI_PERIOD}_MA{MA_WINDOW}"


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


def compute_portfolio(trades: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    trades = trades.copy()
    trades["equity"] = initial_capital * (1 + trades["pnl_pct"] / 100).cumprod()
    prev_equity = trades["equity"].shift(1)
    prev_equity.iloc[0] = initial_capital
    trades["trade_pnl_dollar"] = trades["equity"] - prev_equity
    return trades


def plot_trade_rsi_report(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str,
    initial_capital: float = INITIAL_CAPITAL,
    out_path: str = "TradeRIS.pdf",
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=True)

    _plot_series_with_peaks_valleys(axes[0], df, MA_COL, "tab:blue")
    axes[0].set_ylabel(MA_COL)
    axes[0].set_title(f"{symbol} {MA_COL} with peaks and valleys")
    axes[0].legend()

    _plot_series_with_peaks_valleys(axes[1], df, RSI_CLOSE_COL, "tab:purple")
    axes[1].axhline(70, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(30, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel(RSI_CLOSE_COL)
    axes[1].set_title(f"{symbol} RSI{RSI_PERIOD} (input: close) with peaks and valleys")
    axes[1].legend()

    _plot_series_with_peaks_valleys(axes[2], df, RSI_MA10_COL, "tab:orange")
    axes[2].axhline(70, color="gray", linestyle="--", linewidth=0.8)
    axes[2].axhline(30, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel(RSI_MA10_COL)
    axes[2].set_title(f"{symbol} RSI{RSI_PERIOD} (input: {MA_COL}) with peaks and valleys")
    axes[2].legend()

    ax_portfolio, ax_pnl = axes[3], axes[4]
    if trades.empty:
        ax_portfolio.set_title(f"{symbol} Portfolio Value - no trades")
        ax_pnl.set_title(f"{symbol} Trade Profit/Loss - no trades")
    else:
        portfolio = compute_portfolio(trades, initial_capital)

        equity_dates = [df["time"].iloc[0]] + list(portfolio["exit_date"])
        equity_values = [initial_capital] + list(portfolio["equity"])
        ax_portfolio.plot(equity_dates, equity_values, marker="o", color="tab:green", label="Portfolio value")
        ax_portfolio.axhline(initial_capital, color="gray", linestyle="--", linewidth=0.8, label="Initial capital")
        ax_portfolio.set_ylabel("Portfolio Value ($)")
        ax_portfolio.set_title(f"{symbol} Portfolio Value (start ${initial_capital:,.0f})")
        ax_portfolio.legend()

        colors = ["tab:green" if pnl >= 0 else "tab:red" for pnl in portfolio["trade_pnl_dollar"]]
        ax_pnl.bar(portfolio["exit_date"], portfolio["trade_pnl_dollar"], color=colors, width=3)
        ax_pnl.axhline(0, color="black", linewidth=0.8)
        ax_pnl.set_ylabel("Trade P/L ($)")
        ax_pnl.set_title(f"{symbol} Trade Profit/Loss per Exit")

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved TradeRSI report to {out_path}")


def find_entry_signals(
    df: pd.DataFrame,
    ma_column: str = MA_COL,
    rsi_column: str = RSI_CLOSE_COL,
    max_gap: int = 3,
) -> pd.DataFrame:
    """LONG/SHORT signals where an MA10 pivot and an RSI14(close) pivot line up
    within max_gap bars: MA10 valley + RSI14 valley -> LONG, MA10 peak + RSI14 peak -> SHORT.
    Entry is taken 2 bars after the later of the two pivots, at that bar's open."""
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
            entries.append(
                {
                    "side": side,
                    "entry_position": entry_position,
                    "entry_date": entry_row["time"],
                    "entry_price": entry_row["open"],
                    "entry_ma10": entry_row[ma_column],
                    "pivot_ma10": ma_row[ma_column],
                }
            )

    if not entries:
        return pd.DataFrame()
    return pd.DataFrame(entries).sort_values("entry_date", ignore_index=True)


def print_ma10_long_short_table(entries: pd.DataFrame) -> None:
    print("Date         MA10 long    MA10 Short")
    if entries.empty:
        print("No long/short opportunities found")
        return

    for _, row in entries.iterrows():
        long_val = f"{row['entry_ma10']:.3f}" if row["side"] == "LONG" else ""
        short_val = f"{row['entry_ma10']:.3f}" if row["side"] == "SHORT" else ""
        print(f"{row['entry_date'].date()}   {long_val:<12} {short_val}")


def simulate_trades(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
) -> pd.DataFrame:
    """For each entry, exit on whichever comes first:
    - stop loss: price moves stop_loss_pct against the position
    - take profit: price moves take_profit_pct in favor of the position
    - opposite entry signal: the next opposite-side entry (LONG exits on a SHORT
      entry, SHORT exits on a LONG entry)
    """
    if entries.empty:
        return pd.DataFrame()

    trades = []
    for _, entry in entries.iterrows():
        side = entry["side"]
        entry_price = entry["entry_price"]
        entry_position = int(entry["entry_position"])

        if side == "LONG":
            stop_price = entry_price * (1 - stop_loss_pct)
            target_price = entry_price * (1 + take_profit_pct)
        else:
            stop_price = entry_price * (1 + stop_loss_pct)
            target_price = entry_price * (1 - take_profit_pct)

        opposite_side = "SHORT" if side == "LONG" else "LONG"
        opposite_signals = entries[
            (entries["side"] == opposite_side) & (entries["entry_date"] > entry["entry_date"])
        ]
        opposite_entry = opposite_signals.iloc[0] if not opposite_signals.empty else None

        exit_price = None
        exit_date = None
        exit_reason = None

        for position in range(entry_position + 1, len(df)):
            row = df.iloc[position]

            if opposite_entry is not None and row["time"] >= opposite_entry["entry_date"]:
                exit_price = opposite_entry["entry_price"]
                exit_date = opposite_entry["entry_date"]
                exit_reason = "opposite_signal"
                break

            if side == "LONG":
                if row["low"] <= stop_price:
                    exit_price, exit_date, exit_reason = stop_price, row["time"], "stop_loss"
                    break
                if row["high"] >= target_price:
                    exit_price, exit_date, exit_reason = target_price, row["time"], "take_profit"
                    break
            else:
                if row["high"] >= stop_price:
                    exit_price, exit_date, exit_reason = stop_price, row["time"], "stop_loss"
                    break
                if row["low"] <= target_price:
                    exit_price, exit_date, exit_reason = target_price, row["time"], "take_profit"
                    break

        if exit_price is None:
            last_row = df.iloc[-1]
            exit_price, exit_date, exit_reason = last_row["close"], last_row["time"], "end_of_data"

        pnl_pct = (exit_price - entry_price) / entry_price * 100
        if side == "SHORT":
            pnl_pct = -pnl_pct

        trades.append(
            {
                "side": side,
                "entry_date": entry["entry_date"],
                "entry_price": entry_price,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_pct": pnl_pct,
            }
        )

    return pd.DataFrame(trades)


def plot_ma10_entry_signals(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    symbol: str,
    out_path: str = "aapl_ma10_signals.png",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["time"], df[MA_COL], label=MA_COL, color="tab:blue")

    for side, color, marker in (("LONG", "tab:green", "^"), ("SHORT", "tab:red", "v")):
        signals = entries[entries["side"] == side] if not entries.empty else entries
        if signals is not None and not signals.empty:
            ax.scatter(
                signals["entry_date"],
                signals["entry_ma10"],
                color=color,
                marker=marker,
                label=f"{side} entry",
                zorder=3,
            )

    ax.set_xlabel("Date")
    ax.set_ylabel(MA_COL)
    ax.set_title(f"{symbol} {MA_COL} with long and short entries")
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

    data = add_ma10(data)
    data = add_rsi(data, column="close", period=RSI_PERIOD, out_col=RSI_CLOSE_COL)
    data = add_rsi(data, column=MA_COL, period=RSI_PERIOD, out_col=RSI_MA10_COL)

    entries = find_entry_signals(data)
    print()
    print_ma10_long_short_table(entries)

    plot_ma10_entry_signals(data, entries, SYMBOL)

    trades = simulate_trades(data, entries)
    print()
    print(f"Trades (stop loss {STOP_LOSS_PCT:.0%}, take profit {TAKE_PROFIT_PCT:.0%}, or opposite-signal exit):")
    if trades.empty:
        print("No trades")
    else:
        print(trades.to_string(index=False))
        print(f"\nTotal trades: {len(trades)}  Win rate: {(trades['pnl_pct'] > 0).mean():.1%}  "
              f"Avg pnl: {trades['pnl_pct'].mean():.2f}%  Cumulative pnl: {trades['pnl_pct'].sum():.2f}%")

    plot_trade_rsi_report(data, trades, SYMBOL)

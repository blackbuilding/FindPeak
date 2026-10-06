# Recommended remedies, roughly in order of expected impact:
# 1. Filter pivots with find_peaks(..., prominence=..., distance=...) so only
#    meaningful MA10/RSI turns count, cutting the whipsaw entries.
#    [implemented in v1]
# 2. Add a minimum holding period (e.g., ignore opposite signals for N bars
#    after entry) so a trade can't be closed by noise the day after opening.
#    [implemented in v1]
# 3. Add a trend filter (e.g., only take SHORT signals when price is below a
#    longer MA like MA50/MA200) or drop shorts entirely in this regime.
#    [implemented in v2: ADX14 trend-strength filter + +DI/-DI direction
#    confirmation - a LONG needs ADX>=threshold and +DI>-DI at the pivot bar,
#    a SHORT needs ADX>=threshold and -DI>+DI. Signals in flat/choppy ADX
#    regimes, or fighting the DI-implied trend, are dropped.]
# 4. Lower the take-profit to something the strategy actually reaches (e.g.,
#    8-12%) or switch to a trailing stop once a trade is in profit, to lock in
#    gains on the big trend trades instead of relying solely on opposite-signal
#    exits. [implemented in v2: fixed %/% stop+take-profit replaced with an
#    ATR-based initial stop (entry +/- STOP_ATR_MULT*ATR14) plus a chandelier-
#    style trailing stop (TRAIL_ATR_MULT*ATR14 off the best price since entry),
#    so risk adapts to actual volatility and winners are trailed instead of
#    capped at a fixed target.]
# 5. Before trusting the P&L number, backtest across more symbols/periods - 55
#    trades on one uptrending stock isn't enough to confirm the edge is real
#    rather than lucky. [not yet implemented]
#
# Source techniques (see conversation for full citations): ADX trend-strength
# filtering and ATR-based/trailing stops are standard fixes for MA/RSI
# crossover whipsaw, e.g. "ADX Trading Strategy: How to Filter Weak Trends"
# (quant-signals.com) and "ATR Based Stop Loss" (alphaexcapital.com).

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

# --- v1 signal-quality fixes (carried into v2) --------------------------------
MA_PIVOT_PROMINENCE_PCT = 0.015   # require a swing of >=1.5% of the MA10 level
RSI_PIVOT_PROMINENCE = 8.0        # require a swing of >=8 RSI points
PIVOT_DISTANCE = 3                # pivots must be >=3 bars apart
MIN_HOLD_BARS = 3                 # ignore opposite signals within 3 bars of entry
# ------------------------------------------------------------------------------

# --- v2 trend filter + ATR-based risk management -------------------------------
ATR_PERIOD = 14
ADX_PERIOD = 14
ADX_THRESHOLD = 20.0   # only trade when ADX14 >= this (trending, not choppy)
STOP_ATR_MULT = 2.0    # initial stop = entry price -/+ 2 * ATR14
TRAIL_ATR_MULT = 3.0   # trailing stop = 3 * ATR14 off the best price since entry
# ------------------------------------------------------------------------------

MA_COL = f"MA{MA_WINDOW}"
RSI_CLOSE_COL = f"RSI{RSI_PERIOD}_close"
RSI_MA10_COL = f"RSI{RSI_PERIOD}_MA{MA_WINDOW}"
ATR_COL = f"ATR{ATR_PERIOD}"
ADX_COL = f"ADX{ADX_PERIOD}"
PLUS_DI_COL = f"PLUS_DI{ADX_PERIOD}"
MINUS_DI_COL = f"MINUS_DI{ADX_PERIOD}"


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


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD, out_col: str = None) -> pd.DataFrame:
    df = df.copy()
    out_col = out_col or f"ATR{period}"
    tr = _true_range(df)
    df[out_col] = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return df


def add_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Wilder's ADX/+DI/-DI: measures trend strength (ADX) and direction (+DI vs -DI)."""
    df = df.copy()

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = _true_range(df)
    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    df[f"PLUS_DI{period}"] = plus_di
    df[f"MINUS_DI{period}"] = minus_di
    df[f"ADX{period}"] = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
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
    out_path: str = "TradeRSI_v2.pdf",
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(12, 24), sharex=True)

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

    ax_adx = axes[3]
    ax_adx.plot(df["time"], df[ADX_COL], label=ADX_COL, color="black")
    ax_adx.plot(df["time"], df[PLUS_DI_COL], label=PLUS_DI_COL, color="tab:green", linewidth=0.8)
    ax_adx.plot(df["time"], df[MINUS_DI_COL], label=MINUS_DI_COL, color="tab:red", linewidth=0.8)
    ax_adx.axhline(ADX_THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
    ax_adx.set_ylabel(ADX_COL)
    ax_adx.set_title(f"{symbol} ADX{ADX_PERIOD} trend strength (threshold {ADX_THRESHOLD:.0f})")
    ax_adx.legend()

    ax_portfolio, ax_pnl = axes[4], axes[5]
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
    print(f"Saved TradeRSI_v2 report to {out_path}")


def find_entry_signals(
    df: pd.DataFrame,
    ma_column: str = MA_COL,
    rsi_column: str = RSI_CLOSE_COL,
    max_gap: int = 3,
    ma_prominence_pct: float = None,
    rsi_prominence: float = None,
    distance: int = None,
) -> pd.DataFrame:
    """LONG/SHORT signals where an MA10 pivot and an RSI14(close) pivot line up
    within max_gap bars: MA10 valley + RSI14 valley -> LONG, MA10 peak + RSI14 peak -> SHORT.
    Entry is taken 2 bars after the later of the two pivots, at that bar's open.

    ma_prominence_pct / rsi_prominence / distance filter out insignificant pivots
    (minor wiggles) so entries only fire on real swings. Leave them None to
    reproduce the unfiltered v0 behavior.
    """
    valid_positions = np.flatnonzero(df[ma_column].notna() & df[rsi_column].notna())
    valid = df.iloc[valid_positions]
    if len(valid) < 3:
        return pd.DataFrame()

    ma_values = valid[ma_column].to_numpy()
    rsi_values = valid[rsi_column].to_numpy()

    ma_prominence = ma_prominence_pct * np.nanmedian(ma_values) if ma_prominence_pct else None
    ma_kwargs = {"prominence": ma_prominence, "distance": distance}
    rsi_kwargs = {"prominence": rsi_prominence, "distance": distance}

    ma_peaks, _ = find_peaks(ma_values, **ma_kwargs)
    ma_valleys, _ = find_peaks(-ma_values, **ma_kwargs)
    rsi_peaks, _ = find_peaks(rsi_values, **rsi_kwargs)
    rsi_valleys, _ = find_peaks(-rsi_values, **rsi_kwargs)

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


def filter_entries_by_trend(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    adx_col: str = ADX_COL,
    plus_di_col: str = PLUS_DI_COL,
    minus_di_col: str = MINUS_DI_COL,
    adx_threshold: float = ADX_THRESHOLD,
) -> pd.DataFrame:
    """Keep only entries where, at the entry bar, ADX confirms a strong enough
    trend (adx_threshold) AND the DI direction agrees with the trade side
    (+DI>-DI for LONG, -DI>+DI for SHORT). Drops counter-trend / choppy-market
    signals that v1 would still have taken."""
    if entries.empty:
        return entries

    keep_rows = []
    for _, row in entries.iterrows():
        pos = int(row["entry_position"])
        adx_val = df[adx_col].iloc[pos]
        plus_di = df[plus_di_col].iloc[pos]
        minus_di = df[minus_di_col].iloc[pos]

        if pd.isna(adx_val) or pd.isna(plus_di) or pd.isna(minus_di):
            continue
        if adx_val < adx_threshold:
            continue
        if row["side"] == "LONG" and plus_di <= minus_di:
            continue
        if row["side"] == "SHORT" and minus_di <= plus_di:
            continue

        keep_rows.append(row)

    if not keep_rows:
        return pd.DataFrame(columns=entries.columns)
    return pd.DataFrame(keep_rows).reset_index(drop=True)


def print_ma10_long_short_table(entries: pd.DataFrame) -> None:
    print("Date         MA10 long    MA10 Short")
    if entries.empty:
        print("No long/short opportunities found")
        return

    for _, row in entries.iterrows():
        long_val = f"{row['entry_ma10']:.3f}" if row["side"] == "LONG" else ""
        short_val = f"{row['entry_ma10']:.3f}" if row["side"] == "SHORT" else ""
        print(f"{row['entry_date'].date()}   {long_val:<12} {short_val}")


def simulate_trades_pct(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    min_hold_bars: int = 0,
) -> pd.DataFrame:
    """v0/v1 money management: fixed % stop loss and fixed % take profit, or an
    opposite-side entry signal (after min_hold_bars) - whichever comes first."""
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
            (entries["side"] == opposite_side)
            & (entries["entry_position"] >= entry_position + min_hold_bars)
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


def simulate_trades_atr(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    atr_col: str = ATR_COL,
    stop_atr_mult: float = STOP_ATR_MULT,
    trail_atr_mult: float = TRAIL_ATR_MULT,
    min_hold_bars: int = MIN_HOLD_BARS,
) -> pd.DataFrame:
    """v2 money management: initial stop = entry +/- stop_atr_mult*ATR14 (fixed
    risk sized to actual volatility). As the trade moves favorably, a chandelier
    trailing stop (trail_atr_mult*ATR14 off the best price seen since entry)
    ratchets in the trade's favor, locking in gains on big trend moves instead
    of capping them at a fixed take-profit. Still exits early on an opposite
    entry signal that occurs at least min_hold_bars after entry."""
    if entries.empty:
        return pd.DataFrame()

    trades = []
    for _, entry in entries.iterrows():
        side = entry["side"]
        entry_price = entry["entry_price"]
        entry_position = int(entry["entry_position"])

        atr = df[atr_col].iloc[entry_position]
        if pd.isna(atr) or atr <= 0:
            continue

        if side == "LONG":
            initial_stop = entry_price - stop_atr_mult * atr
        else:
            initial_stop = entry_price + stop_atr_mult * atr
        extreme_price = entry_price
        current_stop = initial_stop

        opposite_side = "SHORT" if side == "LONG" else "LONG"
        opposite_signals = entries[
            (entries["side"] == opposite_side)
            & (entries["entry_position"] >= entry_position + min_hold_bars)
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
                extreme_price = max(extreme_price, row["high"])
                trailing_stop = extreme_price - trail_atr_mult * atr
                new_stop = max(current_stop, trailing_stop)
                if row["low"] <= new_stop:
                    exit_price = new_stop
                    exit_date = row["time"]
                    exit_reason = "trailing_stop" if new_stop > initial_stop else "stop_loss"
                    break
                current_stop = new_stop
            else:
                extreme_price = min(extreme_price, row["low"])
                trailing_stop = extreme_price + trail_atr_mult * atr
                new_stop = min(current_stop, trailing_stop)
                if row["high"] >= new_stop:
                    exit_price = new_stop
                    exit_date = row["time"]
                    exit_reason = "trailing_stop" if new_stop < initial_stop else "stop_loss"
                    break
                current_stop = new_stop

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
    out_path: str = "aapl_ma10_signals_v2.png",
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
    ax.set_title(f"{symbol} {MA_COL} with long and short entries (v2)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved MA10 entry signal plot to {out_path}")


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "n_trades": 0, "win_rate": np.nan, "avg_win_pct": np.nan, "avg_loss_pct": np.nan,
            "median_pnl_pct": np.nan, "sum_pnl_pct": np.nan, "stop_loss_n": 0, "whipsaw_n": 0,
        }

    t = trades.copy()
    t["hold_days"] = (t["exit_date"] - t["entry_date"]).dt.days
    wins = t["pnl_pct"] > 0
    stop_reasons = {"stop_loss", "trailing_stop"}
    return {
        "n_trades": len(t),
        "win_rate": wins.mean(),
        "avg_win_pct": t.loc[wins, "pnl_pct"].mean() if wins.any() else np.nan,
        "avg_loss_pct": t.loc[~wins, "pnl_pct"].mean() if (~wins).any() else np.nan,
        "median_pnl_pct": t["pnl_pct"].median(),
        "sum_pnl_pct": t["pnl_pct"].sum(),
        "stop_loss_n": int(t["exit_reason"].isin(stop_reasons).sum()),
        "whipsaw_n": int(((t["hold_days"] <= 3) & (~wins)).sum()),
    }


def print_backtest_comparison(trades_by_label: "dict[str, pd.DataFrame]") -> None:
    summaries = {label: summarize_trades(trades) for label, trades in trades_by_label.items()}

    def fmt(key, val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "n/a"
        if key == "n_trades" or key.endswith("_n"):
            return f"{val:d}"
        if key == "win_rate":
            return f"{val:.1%}"
        return f"{val:+.2f}%"

    rows = [
        ("Trades", "n_trades"),
        ("Win rate", "win_rate"),
        ("Avg win", "avg_win_pct"),
        ("Avg loss", "avg_loss_pct"),
        ("Median trade P&L", "median_pnl_pct"),
        ("Cumulative P&L", "sum_pnl_pct"),
        ("Stop exits", "stop_loss_n"),
        ("Whipsaws (<=3d losers)", "whipsaw_n"),
    ]

    label_w = 24
    col_w = 16
    header = f"{'Metric':<{label_w}}" + "".join(f"{label:>{col_w}}" for label in trades_by_label)
    print(header)
    for label, key in rows:
        line = f"{label:<{label_w}}" + "".join(
            f"{fmt(key, summaries[version][key]):>{col_w}}" for version in trades_by_label
        )
        print(line)


if __name__ == "__main__":
    date_to = datetime.now()
    date_from = date_to - timedelta(days=365 * YEARS_BACK)

    data = get_asset_data(SYMBOL, TIMEFRAME, date_from, date_to)
    print(data)

    data = add_ma10(data)
    data = add_rsi(data, column="close", period=RSI_PERIOD, out_col=RSI_CLOSE_COL)
    data = add_rsi(data, column=MA_COL, period=RSI_PERIOD, out_col=RSI_MA10_COL)
    data = add_atr(data, period=ATR_PERIOD)
    data = add_adx(data, period=ADX_PERIOD)

    # v0: unfiltered pivots, fixed %/% stop and take profit (reproduces TradeRSI.py).
    entries_v0 = find_entry_signals(data)
    trades_v0 = simulate_trades_pct(data, entries_v0, min_hold_bars=0)

    # v1: prominence/distance-filtered pivots + min-hold, still fixed %/% stop/TP.
    entries_v1 = find_entry_signals(
        data,
        ma_prominence_pct=MA_PIVOT_PROMINENCE_PCT,
        rsi_prominence=RSI_PIVOT_PROMINENCE,
        distance=PIVOT_DISTANCE,
    )
    trades_v1 = simulate_trades_pct(data, entries_v1, min_hold_bars=MIN_HOLD_BARS)

    # v2: v1 entries further filtered by ADX/DI trend confirmation, exits switched
    # to an ATR-based initial stop + chandelier trailing stop.
    entries_v2 = filter_entries_by_trend(data, entries_v1, adx_threshold=ADX_THRESHOLD)
    print()
    print_ma10_long_short_table(entries_v2)

    plot_ma10_entry_signals(data, entries_v2, SYMBOL)

    trades_v2 = simulate_trades_atr(
        data, entries_v2,
        stop_atr_mult=STOP_ATR_MULT,
        trail_atr_mult=TRAIL_ATR_MULT,
        min_hold_bars=MIN_HOLD_BARS,
    )
    print()
    print(f"Trades (ADX>={ADX_THRESHOLD:.0f} trend filter, "
          f"{STOP_ATR_MULT:.0f}xATR{ATR_PERIOD} stop, {TRAIL_ATR_MULT:.0f}xATR{ATR_PERIOD} trailing stop, "
          f"opposite-signal exit after >= {MIN_HOLD_BARS} bars):")
    if trades_v2.empty:
        print("No trades")
    else:
        print(trades_v2.to_string(index=False))
        print(f"\nTotal trades: {len(trades_v2)}  Win rate: {(trades_v2['pnl_pct'] > 0).mean():.1%}  "
              f"Avg pnl: {trades_v2['pnl_pct'].mean():.2f}%  Cumulative pnl: {trades_v2['pnl_pct'].sum():.2f}%")

    print()
    print("=== Backtest comparison: v0 (unfiltered) vs v1 (prominence+min-hold) vs "
          "v2 (+ADX trend filter, ATR stops) ===")
    print_backtest_comparison({"v0": trades_v0, "v1": trades_v1, "v2": trades_v2})

    plot_trade_rsi_report(data, trades_v2, SYMBOL)

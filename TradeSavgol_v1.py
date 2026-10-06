# Variant of TradeSavgol.py with a different entry/exit rule: instead of an
# MA10-pivot + Savgol-pivot confluence with fixed %/% stop-loss/take-profit,
# this version trades purely off the peaks/valleys of the Savitzky-Golay
# filter applied to MA10 (SAVGOL_MA10):
#   - a valley signal enters LONG (or closes an open SHORT)
#   - a peak signal enters SHORT (or closes an open LONG)
# After closing a position, the strategy stays flat for at least
# MIN_FLAT_BARS bars before opening the opposite position - it does not
# reverse directly from LONG to SHORT (or vice versa) on the same signal.
# The same 5-panel report (MA10, Savgol(close), Savgol(MA10), portfolio,
# P/L) is kept.

from datetime import datetime, timedelta
import os

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks, savgol_filter

ACCOUNT = 5056221297
SERVER = "MetaQuotes-Demo"

SYMBOL = "AAPL"
TIMEFRAME = mt5.TIMEFRAME_D1
YEARS_BACK = 2
MA_WINDOW = 10
SAVGOL_WINDOW = 15    # must be odd and > SAVGOL_POLYORDER
SAVGOL_POLYORDER = 3

INITIAL_CAPITAL = 10000.0

PIVOT_CONFIRM_BARS = 2   # a pivot is only confirmed 2 bars later, same convention as TradeSavgol.py
MIN_FLAT_BARS = 2        # minimum bars to stay flat after closing, before entering the opposite side

MA_COL = f"MA{MA_WINDOW}"
SAVGOL_CLOSE_COL = f"SAVGOL{SAVGOL_WINDOW}_close"
SAVGOL_MA10_COL = f"SAVGOL{SAVGOL_WINDOW}_MA{MA_WINDOW}"


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


def add_savgol(
    df: pd.DataFrame,
    column: str,
    window: int = SAVGOL_WINDOW,
    polyorder: int = SAVGOL_POLYORDER,
    out_col: str = None,
) -> pd.DataFrame:
    """Savitzky-Golay smoothed version of df[column]. Leading NaNs (e.g. from an
    MA's rolling window) are skipped and left as NaN; the filter is applied to
    the contiguous valid tail."""
    df = df.copy()
    out_col = out_col or f"SAVGOL{window}_{column}"

    values = df[column].to_numpy(dtype=float)
    valid_mask = ~np.isnan(values)  # savgol_filter can't handle NaNs (e.g. MA10's rolling-window prefix)
    result = np.full_like(values, np.nan)

    valid_values = values[valid_mask]
    if len(valid_values) >= window:  # window_length must not exceed the data being filtered
        result[valid_mask] = savgol_filter(valid_values, window_length=window, polyorder=polyorder)

    df[out_col] = result
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
    # Trades are compounded sequentially (each trade risks the full current equity),
    # not sized as a fixed fraction of the starting capital.
    trades = trades.copy()
    trades["equity"] = initial_capital * (1 + trades["pnl_pct"] / 100).cumprod()
    prev_equity = trades["equity"].shift(1)
    prev_equity.iloc[0] = initial_capital
    trades["trade_pnl_dollar"] = trades["equity"] - prev_equity
    return trades


def compute_daily_equity(df: pd.DataFrame, trades: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> pd.Series:
    """Mark-to-market equity for every bar in df (not just trade exits), so the
    Sharpe ratio can be computed from daily returns like a standard backtest,
    instead of only from the (irregularly spaced) per-trade returns."""
    equity = pd.Series(float(initial_capital), index=df.index)
    if trades.empty:
        return equity

    time_to_pos = pd.Series(df.index, index=df["time"]).to_dict()
    current_equity = initial_capital

    for _, trade in trades.sort_values("entry_date").iterrows():
        entry_pos = time_to_pos[trade["entry_date"]]
        exit_pos = time_to_pos[trade["exit_date"]]
        side_sign = 1 if trade["side"] == "LONG" else -1
        entry_price = trade["entry_price"]

        for pos in range(entry_pos, exit_pos):
            price = df["close"].iloc[pos]
            equity.iloc[pos] = current_equity * (1 + side_sign * (price - entry_price) / entry_price)

        current_equity = current_equity * (1 + side_sign * (trade["exit_price"] - entry_price) / entry_price)
        equity.iloc[exit_pos:] = current_equity  # flat while out of the market, until the next trade overwrites it

    return equity


def summarize_performance(df: pd.DataFrame, trades: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> dict:
    if trades.empty:
        return {
            "sharpe_ratio": float("nan"), "net_profit_dollar": 0.0, "net_profit_pct_compounded": 0.0,
            "net_profit_pct_summed": 0.0, "n_trades": 0,
            "pct_winning_trades": float("nan"), "pct_losing_trades": float("nan"),
        }

    portfolio = compute_portfolio(trades, initial_capital)

    # Standard (annualized) Sharpe ratio: computed from the daily equity curve's
    # returns, not per-trade returns, so it reflects actual time spent in the
    # market and is comparable across strategies with different trade frequency.
    daily_equity = compute_daily_equity(df, trades, initial_capital)
    daily_returns = daily_equity.pct_change().dropna()
    daily_std = daily_returns.std(ddof=1)
    sharpe_ratio = (daily_returns.mean() / daily_std) * np.sqrt(252) if daily_std > 0 else float("nan")

    wins = trades["pnl_pct"] > 0
    return {
        "sharpe_ratio": sharpe_ratio,
        "net_profit_dollar": portfolio["equity"].iloc[-1] - initial_capital,
        "net_profit_pct_compounded": (portfolio["equity"].iloc[-1] / initial_capital - 1) * 100,
        "net_profit_pct_summed": trades["pnl_pct"].sum(),
        "n_trades": len(trades),
        "pct_winning_trades": wins.mean() * 100,
        "pct_losing_trades": (~wins).mean() * 100,
    }


def format_performance_summary_lines(summary: dict) -> list:
    # One number for the whole strategy (not one per trade): mean daily return /
    # std of daily return * sqrt(252), from the daily equity curve in
    # compute_daily_equity - the standard, annualized definition.
    sharpe_line = (f"Sharpe ratio (annualized): {summary['sharpe_ratio']:.3f}"
                   if not pd.isna(summary["sharpe_ratio"]) else "Sharpe ratio (annualized): n/a")
    pct_win_line = (f"Percent profitable trades: {summary['pct_winning_trades']:.1f}%"
                     if summary["n_trades"] else "Percent profitable trades: n/a")
    pct_loss_line = (f"Percent losing trades:     {summary['pct_losing_trades']:.1f}%"
                      if summary["n_trades"] else "Percent losing trades:     n/a")
    return [
        sharpe_line,
        f"Net profit (compounded):  ${summary['net_profit_dollar']:,.2f} ({summary['net_profit_pct_compounded']:+.2f}%)",
        f"Net profit (summed P&L):  {summary['net_profit_pct_summed']:+.2f}%",
        f"Number of trades:          {summary['n_trades']}",
        pct_win_line,
        pct_loss_line,
    ]


def print_performance_summary(df: pd.DataFrame, trades: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> None:
    summary = summarize_performance(df, trades, initial_capital)
    print("=== Performance summary ===")
    for line in format_performance_summary_lines(summary):
        print(line)


def _draw_ma10_panel(ax, df: pd.DataFrame, symbol: str) -> None:
    _plot_series_with_peaks_valleys(ax, df, MA_COL, "tab:blue")
    ax.set_ylabel(MA_COL)
    ax.set_title(f"{symbol} {MA_COL} with peaks and valleys")
    ax.legend()


def _draw_savgol_close_panel(ax, df: pd.DataFrame, symbol: str) -> None:
    _plot_series_with_peaks_valleys(ax, df, SAVGOL_CLOSE_COL, "tab:purple")
    ax.set_ylabel(SAVGOL_CLOSE_COL)
    ax.set_title(f"{symbol} Savitzky-Golay (input: close) with peaks and valleys")
    ax.legend()


def _draw_savgol_ma10_panel(ax, df: pd.DataFrame, symbol: str) -> None:
    _plot_series_with_peaks_valleys(ax, df, SAVGOL_MA10_COL, "tab:orange")
    ax.set_ylabel(SAVGOL_MA10_COL)
    ax.set_title(f"{symbol} Savitzky-Golay (input: {MA_COL}) with peaks and valleys")
    ax.legend()


def _draw_portfolio_panel(ax, df: pd.DataFrame, portfolio: pd.DataFrame, initial_capital: float, symbol: str) -> None:
    if portfolio is None:
        ax.set_title(f"{symbol} Portfolio Value - no trades")
        return
    equity_dates = [df["time"].iloc[0]] + list(portfolio["exit_date"])
    equity_values = [initial_capital] + list(portfolio["equity"])
    ax.plot(equity_dates, equity_values, marker="o", color="tab:green", label="Portfolio value")
    ax.axhline(initial_capital, color="gray", linestyle="--", linewidth=0.8, label="Initial capital")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(f"{symbol} Portfolio Value (start ${initial_capital:,.0f})")
    ax.legend()


def _draw_pnl_panel(ax, portfolio: pd.DataFrame, symbol: str) -> None:
    if portfolio is None:
        ax.set_title(f"{symbol} Trade Profit/Loss - no trades")
        return
    colors = ["tab:green" if pnl >= 0 else "tab:red" for pnl in portfolio["trade_pnl_dollar"]]
    ax.bar(portfolio["exit_date"], portfolio["trade_pnl_dollar"], color=colors, width=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Trade P/L ($)")
    ax.set_title(f"{symbol} Trade Profit/Loss per Exit")


def plot_trade_savgol_report(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str,
    initial_capital: float = INITIAL_CAPITAL,
    out_path: str = "TradeSavgol_v1.pdf",
) -> None:
    portfolio = compute_portfolio(trades, initial_capital) if not trades.empty else None

    # Each page draws up to 2 panels; a page with only 1 panel hides the empty second axis.
    panel_pages = [
        [
            lambda ax: _draw_ma10_panel(ax, df, symbol),
            lambda ax: _draw_savgol_close_panel(ax, df, symbol),
        ],
        [
            lambda ax: _draw_savgol_ma10_panel(ax, df, symbol),
            lambda ax: _draw_portfolio_panel(ax, df, portfolio, initial_capital, symbol),
        ],
        [
            lambda ax: _draw_pnl_panel(ax, portfolio, symbol),
        ],
    ]

    summary = summarize_performance(df, trades, initial_capital)
    summary_lines = ["Performance summary", ""] + format_performance_summary_lines(summary)

    with PdfPages(out_path) as pdf:
        for panels in panel_pages:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            for ax, draw in zip(axes, panels):
                draw(ax)
            for ax in axes[len(panels):]:
                ax.axis("off")

            axes[len(panels) - 1].set_xlabel("Date")
            fig.autofmt_xdate()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        summary_fig, summary_ax = plt.subplots(figsize=(8.5, 4))
        summary_ax.axis("off")
        summary_ax.text(
            0.02, 0.95, "\n".join(summary_lines),
            transform=summary_ax.transAxes, fontsize=12, family="monospace",
            verticalalignment="top",
        )
        pdf.savefig(summary_fig)
        plt.close(summary_fig)

    print(f"Saved TradeSavgol_v1 report to {out_path}")


def find_pivot_events(df: pd.DataFrame, signal_column: str = SAVGOL_MA10_COL) -> list:
    """All peaks and valleys of df[signal_column], as a chronological list of
    (bar_position, 'peak' | 'valley') tuples."""
    valid_positions = np.flatnonzero(df[signal_column].notna())
    values = df[signal_column].to_numpy()[valid_positions]

    peak_idx, _ = find_peaks(values)
    valley_idx, _ = find_peaks(-values)

    events = [(int(valid_positions[i]), "peak") for i in peak_idx]
    events += [(int(valid_positions[i]), "valley") for i in valley_idx]
    events.sort(key=lambda e: e[0])
    return events


def simulate_flip_trades(
    df: pd.DataFrame,
    events: list,
    ma_column: str = MA_COL,
    confirm_bars: int = PIVOT_CONFIRM_BARS,
    min_flat_bars: int = MIN_FLAT_BARS,
) -> pd.DataFrame:
    """Walk the peak/valley events in order, flipping position on each signal:
    a valley opens LONG (closing an open SHORT first), a peak opens SHORT
    (closing an open LONG first). After a close, entry into the opposite side
    is delayed until at least min_flat_bars after the close."""
    trades = []
    position = None       # None, "LONG" or "SHORT"
    open_trade = None
    cooldown_until = 0     # earliest bar position a new (opposite) trade may open

    def close(exit_position, exit_reason):
        exit_row = df.iloc[exit_position]
        entry_price = open_trade["entry_price"]
        exit_price = exit_row["open"]
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        if open_trade["side"] == "SHORT":
            pnl_pct = -pnl_pct
        trades.append(
            {
                "side": open_trade["side"],
                "entry_date": open_trade["entry_date"],
                "entry_price": entry_price,
                "entry_ma10": open_trade["entry_ma10"],
                "exit_date": exit_row["time"],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_pct": pnl_pct,
            }
        )

    for pivot_position, pivot_type in events:
        signal_side = "LONG" if pivot_type == "valley" else "SHORT"
        opposite_side = "SHORT" if signal_side == "LONG" else "LONG"
        confirm_position = pivot_position + confirm_bars
        if confirm_position >= len(df):
            continue

        if position == opposite_side:
            close(confirm_position, f"{pivot_type}_signal")
            position = None
            open_trade = None
            cooldown_until = confirm_position + min_flat_bars

        if position is None:
            entry_position = max(confirm_position, cooldown_until)
            if entry_position >= len(df):
                continue
            entry_row = df.iloc[entry_position]
            open_trade = {
                "side": signal_side,
                "entry_date": entry_row["time"],
                "entry_price": entry_row["open"],
                "entry_ma10": entry_row[ma_column],
            }
            position = signal_side
        # position == signal_side: already in this position, nothing to do

    if open_trade is not None:
        close(len(df) - 1, "end_of_data")

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)


def print_ma10_long_short_table(entries: pd.DataFrame) -> None:
    print("Date         MA10 long    MA10 Short")
    if entries.empty:
        print("No long/short opportunities found")
        return

    for _, row in entries.iterrows():
        long_val = f"{row['entry_ma10']:.3f}" if row["side"] == "LONG" else ""
        short_val = f"{row['entry_ma10']:.3f}" if row["side"] == "SHORT" else ""
        print(f"{row['entry_date'].date()}   {long_val:<12} {short_val}")


def plot_ma10_entry_signals(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    symbol: str,
    out_path: str = "aapl_ma10_signals_savgol_v1.png",
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
    ax.set_title(f"{symbol} {MA_COL} with long and short entries (Savgol v1)")
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
    data = add_savgol(data, column="close", out_col=SAVGOL_CLOSE_COL)
    data = add_savgol(data, column=MA_COL, out_col=SAVGOL_MA10_COL)

    events = find_pivot_events(data, signal_column=SAVGOL_MA10_COL)
    trades = simulate_flip_trades(data, events)

    entries = trades[["side", "entry_date", "entry_price", "entry_ma10"]] if not trades.empty else pd.DataFrame()
    print()
    print_ma10_long_short_table(entries)

    plot_ma10_entry_signals(data, entries, SYMBOL)

    print()
    print(f"Trades (enter/exit on {SAVGOL_MA10_COL} peak/valley signals, "
          f"stay flat >= {MIN_FLAT_BARS} bars before reversing):")
    if trades.empty:
        print("No trades")
    else:
        print(trades.to_string(index=False))
        print(f"\nTotal trades: {len(trades)}  Win rate: {(trades['pnl_pct'] > 0).mean():.1%}  "
              f"Avg pnl: {trades['pnl_pct'].mean():.2f}%  Cumulative pnl: {trades['pnl_pct'].sum():.2f}%")

    plot_trade_savgol_report(data, trades, SYMBOL)

    print()
    print_performance_summary(data, trades)

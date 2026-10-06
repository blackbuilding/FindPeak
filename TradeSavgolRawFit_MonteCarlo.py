# Multi-asset/multi-timeframe sweep of the raw-fit pivot refinement from
# TradeSavgolRawFit.py, over a fixed historical window (2020-01-01 to
# 2026-01-01) instead of "N years back from today" like the other scripts.
# "Monte Carlo" here means varying the asset and bar frequency to get many
# independent runs of the same strategy, not randomized resampling - each
# (symbol, timeframe) pair is one run.
#
# For each of the 7 assets (AAPL, MSFT, GOOGL - stocks; EURUSD, GBPUSD,
# USDJPY - currencies; TSLA - the highly-volatile pick, since this MT5 demo
# account has no crypto symbols) x 2 timeframes (D1, H1) = 14 runs, this
# reproduces the same per-run output TradeSavgol.py produces (long/short
# table, entry-signal PNG, multi-panel PDF report, performance summary),
# saved under monte_carlo_output/. It then overlays every run's profit/loss
# (%) vs. calendar time on one chart, with the single highest-return run
# highlighted.
#
# Stocks are long-only here (no shorting individual names); currency pairs
# trade both directions, since going short one currency is just going long
# the other side of the pair. Signals on the disallowed side are dropped
# before simulation, so e.g. a stock's SHORT pivots never become trades.
#
# TradeSavgol.py and TradeSavgolRawFit.py are both left unmodified.

import os
from datetime import datetime

import MetaTrader5 as mt5
import matplotlib.pyplot as plt

from TradeSavgol import (
    MA_COL, SAVGOL_CLOSE_COL, SAVGOL_MA10_COL, INITIAL_CAPITAL,
    get_asset_data, add_ma10, add_savgol, simulate_trades, compute_daily_equity,
    summarize_performance, print_performance_summary, print_ma10_long_short_table,
    plot_trade_savgol_report, plot_ma10_entry_signals,
)
from TradeSavgolRawFit import find_entry_signals_rawfit

DATE_FROM = datetime(2020, 1, 1)
DATE_TO = datetime(2026, 1, 1)

# (symbol, asset class, display note) - asset class decides which trade
# sides are allowed (see ALLOWED_SIDES below); the note is just for the
# asset grid / labeling and doesn't affect behavior.
ASSETS = [
    ("AAPL", "stock", "stock"),
    ("MSFT", "stock", "stock"),
    ("GOOGL", "stock", "stock"),
    ("EURUSD", "currency", "currency"),
    ("GBPUSD", "currency", "currency"),
    ("USDJPY", "currency", "currency"),
    ("TSLA", "stock", "highly volatile"),
]
TIMEFRAMES = [("D1", mt5.TIMEFRAME_D1), ("H1", mt5.TIMEFRAME_H1)]

ALLOWED_SIDES = {
    "stock": ("LONG",),
    "currency": ("LONG", "SHORT"),
}

OUTPUT_DIR = "monte_carlo_output"


def filter_entries_by_side(entries, allowed_sides: tuple):
    if entries.empty:
        return entries
    return entries[entries["side"].isin(allowed_sides)].reset_index(drop=True)


def run_combo(symbol: str, asset_class: str, tf_label: str, tf_const: int, output_dir: str) -> dict | None:
    label = f"{symbol} {tf_label}"
    print(f"--- {label} ({asset_class}, {'/'.join(ALLOWED_SIDES[asset_class])} only) ---")

    try:
        data = get_asset_data(symbol, tf_const, DATE_FROM, DATE_TO)
    except RuntimeError as exc:
        print(f"{label}: skipped ({exc})\n")
        return None

    data = add_ma10(data)
    data = add_savgol(data, column="close", out_col=SAVGOL_CLOSE_COL)
    data = add_savgol(data, column=MA_COL, out_col=SAVGOL_MA10_COL)

    entries = find_entry_signals_rawfit(data)
    entries = filter_entries_by_side(entries, ALLOWED_SIDES[asset_class])
    trades = simulate_trades(data, entries)

    print_ma10_long_short_table(entries)
    plot_ma10_entry_signals(
        data, entries, symbol,
        out_path=os.path.join(output_dir, f"{symbol.lower()}_{tf_label.lower()}_ma10_signals_savgol_rawfit.png"),
    )
    plot_trade_savgol_report(
        data, trades, symbol, INITIAL_CAPITAL,
        out_path=os.path.join(output_dir, f"TradeSavgolRawFit_{symbol}_{tf_label}.pdf"),
    )
    print()
    print_performance_summary(data, trades, INITIAL_CAPITAL)
    print()

    summary = summarize_performance(data, trades, INITIAL_CAPITAL)
    daily_equity = compute_daily_equity(data, trades, INITIAL_CAPITAL)
    pct_return = (daily_equity / INITIAL_CAPITAL - 1) * 100

    return {
        "label": label,
        "timeframe": tf_label,
        "time": data["time"],
        "pct_return": pct_return,
        "net_profit_pct": summary["net_profit_pct_compounded"],
        "n_trades": summary["n_trades"],
    }


def plot_monte_carlo_overlay(runs: list, out_path: str, title_suffix: str) -> None:
    # Split per timeframe rather than one combined chart: compounded return at
    # H1's trade counts (hundreds-to-thousands of trades, full-equity
    # reinvestment per trade) is orders of magnitude larger than D1's, so a
    # combined chart flattens every D1 line to ~0. Each group gets its own
    # axis scale and its own highest-return highlight.
    fig, ax = plt.subplots(figsize=(14, 8))

    best = max(runs, key=lambda r: r["net_profit_pct"] if r["net_profit_pct"] == r["net_profit_pct"] else float("-inf"))

    for run in runs:
        is_best = run is best
        ax.plot(
            run["time"], run["pct_return"],
            label=f"{run['label']} ({run['net_profit_pct']:+.1f}%)" + ("  <- highest return" if is_best else ""),
            linewidth=3.0 if is_best else 1.2,
            color="black" if is_best else None,
            zorder=3 if is_best else 2,
            alpha=1.0 if is_best else 0.8,
        )

    # Compounded return under full reinvestment reliably produces multi-tier
    # outliers here (e.g. GOOGL H1 +454,794,545% vs. TSLA H1 +49,342,087% vs.
    # everything else in the thousands) - a single "is the top run >Nx the
    # next one" check misses cases where the 2nd-place run is ALSO an outlier
    # relative to the rest. Always use a symlog axis instead of trying to
    # detect the outlier pattern: below linthresh it reads as a normal linear
    # plot (so well-behaved groups like D1 aren't hurt), and above it every
    # run stays visible regardless of how many outlier tiers there are.
    ax.set_yscale("symlog", linthresh=50.0)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Profit / loss (%, compounded)")
    ax.set_title(f"TradeSavgolRawFit: profit/loss vs. time, {title_suffix} "
                  f"({DATE_FROM.date()} - {DATE_TO.date()})")
    ax.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved Monte Carlo overlay plot to {out_path}")
    print(f"Highest-return run ({title_suffix}): {best['label']} ({best['net_profit_pct']:+.2f}%, {best['n_trades']} trades)")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    runs = []
    for symbol, asset_class, _note in ASSETS:
        for tf_label, tf_const in TIMEFRAMES:
            result = run_combo(symbol, asset_class, tf_label, tf_const, OUTPUT_DIR)
            if result is not None:
                runs.append(result)

    print("=== Summary across all runs ===")
    for run in sorted(runs, key=lambda r: r["net_profit_pct"], reverse=True):
        print(f"{run['label']:<14} net profit {run['net_profit_pct']:+8.2f}%   trades: {run['n_trades']}")
    print()

    for tf_label, _tf_const in TIMEFRAMES:
        tf_runs = [r for r in runs if r["timeframe"] == tf_label]
        if tf_runs:
            plot_monte_carlo_overlay(
                tf_runs,
                os.path.join(OUTPUT_DIR, f"TradeSavgolRawFit_MonteCarlo_{tf_label}.png"),
                title_suffix=f"{tf_label} runs",
            )
        else:
            print(f"No successful {tf_label} runs - nothing to plot.")

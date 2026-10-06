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

ASSETS = [
    ("AAPL", "stock"),
    ("MSFT", "stock"),
    ("GOOGL", "stock"),
    ("EURUSD", "currency"),
    ("GBPUSD", "currency"),
    ("USDJPY", "currency"),
    ("TSLA", "highly volatile"),
]
TIMEFRAMES = [("D1", mt5.TIMEFRAME_D1), ("H1", mt5.TIMEFRAME_H1)]

OUTPUT_DIR = "monte_carlo_output"


def run_combo(symbol: str, tf_label: str, tf_const: int, output_dir: str) -> dict | None:
    label = f"{symbol} {tf_label}"
    print(f"--- {label} ---")

    try:
        data = get_asset_data(symbol, tf_const, DATE_FROM, DATE_TO)
    except RuntimeError as exc:
        print(f"{label}: skipped ({exc})\n")
        return None

    data = add_ma10(data)
    data = add_savgol(data, column="close", out_col=SAVGOL_CLOSE_COL)
    data = add_savgol(data, column=MA_COL, out_col=SAVGOL_MA10_COL)

    entries = find_entry_signals_rawfit(data)
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

    # If one run's return dwarfs the rest (e.g. an outlier like TSLA H1 at
    # +9,054,904% vs. the next-highest H1 run at +10,444%), a linear axis
    # flattens every other line to ~0. Auto-detect that case and switch to a
    # symlog axis so every run stays visible; leave well-behaved groups (like
    # D1 here) on a normal linear axis.
    abs_returns = [abs(r["net_profit_pct"]) for r in runs if r["net_profit_pct"] == r["net_profit_pct"]]
    others_max = max((v for v in abs_returns if v != abs(best["net_profit_pct"])), default=0)
    if others_max > 0 and abs(best["net_profit_pct"]) / others_max > 20:
        ax.set_yscale("symlog", linthresh=max(1.0, others_max))

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
    for symbol, _category in ASSETS:
        for tf_label, tf_const in TIMEFRAMES:
            result = run_combo(symbol, tf_label, tf_const, OUTPUT_DIR)
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

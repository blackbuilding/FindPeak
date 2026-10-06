# Robustness check for the raw-fit pivot refinement from TradeSavgolRawFit.py:
# that experiment showed a sharpe/net-profit improvement on AAPL D1, but on a
# single symbol/timeframe/15-trade sample that could easily be noise. This
# reruns the same baseline-vs-raw-fit comparison on a different symbol AND a
# different bar frequency (EURUSD H4, vs the original AAPL D1) to see whether
# the improvement holds outside that one sample.
#
# TradeSavgol.py is still not modified - this only imports its machinery plus
# the raw-fit refinement.

from datetime import datetime, timedelta

import MetaTrader5 as mt5

from TradeSavgol import (
    MA_COL, SAVGOL_CLOSE_COL, SAVGOL_MA10_COL,
    get_asset_data, add_ma10, add_savgol,
    find_entry_signals, simulate_trades,
)
from TradeSavgolThresholds import _sweep_row, print_sweep_table
from TradeSavgolRawFit import find_entry_signals_rawfit

# (symbol, mt5 timeframe, years of history, label) - EURUSD H4 differs from
# the original AAPL D1 test in both asset class and bar frequency at once.
COMBOS = [
    ("AAPL", mt5.TIMEFRAME_D1, 2, "AAPL D1 (original)"),
    ("EURUSD", mt5.TIMEFRAME_H4, 1, "EURUSD H4"),
]


def run_combo(symbol: str, timeframe: int, years_back: float, label: str) -> None:
    date_to = datetime.now()
    date_from = date_to - timedelta(days=365 * years_back)

    try:
        data = get_asset_data(symbol, timeframe, date_from, date_to)
    except RuntimeError as exc:
        print(f"{label}: skipped ({exc})\n")
        return

    data = add_ma10(data)
    data = add_savgol(data, column="close", out_col=SAVGOL_CLOSE_COL)
    data = add_savgol(data, column=MA_COL, out_col=SAVGOL_MA10_COL)

    baseline_entries = find_entry_signals(data)
    baseline_trades = simulate_trades(data, baseline_entries)

    rawfit_entries = find_entry_signals_rawfit(data)
    rawfit_trades = simulate_trades(data, rawfit_entries)

    print_sweep_table(
        f"{label}: {len(data)} bars\n",
        [
            _sweep_row("baseline (smoothed pivot)", data, baseline_entries, baseline_trades),
            _sweep_row("raw-fit refined pivot", data, rawfit_entries, rawfit_trades),
        ],
    )


if __name__ == "__main__":
    for symbol, timeframe, years_back, label in COMBOS:
        run_combo(symbol, timeframe, years_back, label)

# Experiment: adds scipy find_peaks noise-rejection thresholds (prominence,
# distance) to the MA10/Savgol confluence signal from TradeSavgol.py, and
# compares the resulting signals/trades/performance against the untouched
# baseline. TradeSavgol.py itself is not modified - this script imports its
# data/indicator/backtest machinery and only reimplements find_entry_signals
# with thresholds added, so the two can be run side by side.
#
# Motivated by https://terpconnect.umd.edu/~toh/spectrum/PeakFindingandMeasurement.htm:
# that page's derivative zero-crossing peak detector always pairs detection
# with an AmplitudeThreshold/SlopeThreshold to reject noise-induced false
# peaks. TradeSavgol.py's find_peaks calls currently have no such thresholds,
# so any single-bar wiggle in MA10 or the Savgol curve counts as a pivot.
# scipy's find_peaks exposes `prominence` (~AmplitudeThreshold) and `distance`
# (minimum bar spacing, guards against noise-adjacent double-pivots) as the
# equivalent knobs.

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from TradeSavgol import (
    SYMBOL, TIMEFRAME, YEARS_BACK, MA_COL, SAVGOL_CLOSE_COL, SAVGOL_MA10_COL,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, INITIAL_CAPITAL,
    get_asset_data, add_ma10, add_savgol,
    find_entry_signals, simulate_trades,
    summarize_performance,
)

# Prominence expressed as a fraction of each series' full range rather than an
# absolute price level, so the same factor is reusable across symbols/columns
# with very different scales (e.g. MA10 in price units vs a Savgol-smoothed
# derivative-like series).
PEAK_PROMINENCE_PCT = 0.01   # reject peaks/valleys with prominence < 1% of series range
PEAK_MIN_DISTANCE = 3        # minimum bars between two accepted peaks (or two valleys)


def _prominence_value(values: np.ndarray, pct: float) -> float:
    series_range = np.nanmax(values) - np.nanmin(values)
    return pct * series_range


def find_entry_signals_thresholded(
    df: pd.DataFrame,
    ma_column: str = MA_COL,
    signal_column: str = SAVGOL_CLOSE_COL,
    max_gap: int = 3,
    prominence_pct: float = PEAK_PROMINENCE_PCT,
    distance: int = PEAK_MIN_DISTANCE,
) -> pd.DataFrame:
    """Same confluence logic as TradeSavgol.find_entry_signals, but each
    find_peaks call is given a prominence (scaled to that series' range) and
    a minimum distance, so single-bar noise wiggles no longer count as
    pivots."""
    valid_positions = np.flatnonzero(df[ma_column].notna() & df[signal_column].notna())
    valid = df.iloc[valid_positions]
    if len(valid) < 3:
        return pd.DataFrame()

    ma_values = valid[ma_column].to_numpy()
    signal_values = valid[signal_column].to_numpy()

    ma_prominence = _prominence_value(ma_values, prominence_pct)
    signal_prominence = _prominence_value(signal_values, prominence_pct)

    ma_peaks, _ = find_peaks(ma_values, prominence=ma_prominence, distance=distance)
    ma_valleys, _ = find_peaks(-ma_values, prominence=ma_prominence, distance=distance)
    signal_peaks, _ = find_peaks(signal_values, prominence=signal_prominence, distance=distance)
    signal_valleys, _ = find_peaks(-signal_values, prominence=signal_prominence, distance=distance)

    entries = []
    for side, ma_extrema, signal_extrema in (
        ("LONG", ma_valleys, signal_valleys),
        ("SHORT", ma_peaks, signal_peaks),
    ):
        used_signal_extrema = set()
        for ma_extreme in ma_extrema:
            ma_position = valid_positions[ma_extreme]
            candidates = [
                signal_extreme
                for signal_extreme in signal_extrema
                if signal_extreme not in used_signal_extrema
                and abs(valid_positions[signal_extreme] - ma_position) <= max_gap
            ]
            if not candidates:
                continue

            signal_extreme = min(
                candidates,
                key=lambda position: abs(valid_positions[position] - ma_position),
            )
            used_signal_extrema.add(signal_extreme)
            signal_position = valid_positions[signal_extreme]
            pivot_position = max(ma_position, signal_position)
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


def _sweep_row(label: str, df: pd.DataFrame, entries: pd.DataFrame, trades: pd.DataFrame) -> dict:
    summary = summarize_performance(df, trades, INITIAL_CAPITAL)
    return {
        "setting": label,
        "entries": len(entries),
        "trades": len(trades),
        "sharpe": summary["sharpe_ratio"],
        "net_profit_pct": summary["net_profit_pct_compounded"],
        "pct_win": summary["pct_winning_trades"],
    }


_TABLE_HEADER = f"{'setting':28}{'entries':>9}{'trades':>8}{'sharpe':>9}{'net %':>10}{'win %':>8}"


def print_sweep_table(title: str, rows: list) -> None:
    print(title)
    print(_TABLE_HEADER)
    for row in rows:
        sharpe_str = f"{row['sharpe']:.3f}" if not pd.isna(row["sharpe"]) else "n/a"
        win_str = f"{row['pct_win']:.1f}" if not pd.isna(row["pct_win"]) else "n/a"
        print(f"{row['setting']:28}{row['entries']:>9}{row['trades']:>8}{sharpe_str:>9}"
              f"{row['net_profit_pct']:>+9.2f}{win_str:>8}")
    print()


if __name__ == "__main__":
    date_to = datetime.now()
    date_from = date_to - timedelta(days=365 * YEARS_BACK)

    data = get_asset_data(SYMBOL, TIMEFRAME, date_from, date_to)
    data = add_ma10(data)
    data = add_savgol(data, column="close", out_col=SAVGOL_CLOSE_COL)
    data = add_savgol(data, column=MA_COL, out_col=SAVGOL_MA10_COL)

    baseline_entries = find_entry_signals(data)
    baseline_trades = simulate_trades(data, baseline_entries)

    rows = [_sweep_row("baseline (no threshold)", data, baseline_entries, baseline_trades)]

    prominence_levels = [0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01]
    for prominence_pct in prominence_levels:
        entries = find_entry_signals_thresholded(data, prominence_pct=prominence_pct, distance=PEAK_MIN_DISTANCE)
        trades = simulate_trades(data, entries)
        rows.append(_sweep_row(f"prominence {prominence_pct:.2%}", data, entries, trades))

    print_sweep_table(f"{SYMBOL}, {YEARS_BACK}y, prominence sweep (distance={PEAK_MIN_DISTANCE} bars)\n", rows)

    # Distance alone, no prominence (prominence_pct=0.0 is effectively a no-op
    # filter in scipy's find_peaks - isolates the distance parameter's effect).
    distance_rows = [rows[0]]  # baseline (no threshold) also means distance=1, no prominence
    for distance in [2, 3, 4, 5, 7, 10]:
        entries = find_entry_signals_thresholded(data, prominence_pct=0.0, distance=distance)
        trades = simulate_trades(data, entries)
        distance_rows.append(_sweep_row(f"distance={distance} bars", data, entries, trades))

    print_sweep_table(f"{SYMBOL}, {YEARS_BACK}y, distance-only sweep (no prominence)\n", distance_rows)

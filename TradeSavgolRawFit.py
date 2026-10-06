# Experiment: applies the UMD peak-finding page's measurement refinement -
# "use the smoothed curve only to *locate* a candidate peak; measure its
# actual position/height by curve-fitting the *unsmoothed* data near that
# point" - to the Savgol-close pivot in TradeSavgol.py's confluence signal.
# TradeSavgol.py is not modified; this imports its machinery and only adds a
# raw-data refinement step on top of the existing (unmodified) pivot
# detection.
#
# https://terpconnect.umd.edu/~toh/spectrum/PeakFindingandMeasurement.htm
# rationale: smoothing systematically flattens peak amplitude and can shift
# the apparent peak position by a point or two. A local least-squares
# quadratic fit to the nearby raw samples recovers a less-biased estimate of
# both.
#
# Two questions this answers:
# 1. How large is the smoothed-vs-raw bias in practice for this Savgol
#    window/polyorder, on this symbol? (a measurement-accuracy question)
# 2. Does refining the pivot's *bar position* this way ever move it to a
#    different integer bar than the smoothed series' own argmax/argmin -
#    which is the only way this refinement could actually change which
#    trades get taken? (a strategy-impact question)
#
# Results (AAPL D1, 2y; and EURUSD H4, 1y, via TradeSavgolRobustness.py):
#
#   AAPL D1 (492 bars)
#   setting                       entries  trades   sharpe     net %   win %
#   baseline (smoothed pivot)          15      15    1.187   +73.13    60.0
#   raw-fit refined pivot              14      14    1.473  +102.55    71.4
#
#   EURUSD H4 (1554 bars)
#   setting                       entries  trades   sharpe     net %   win %
#   baseline (smoothed pivot)          87      87    0.993   +16.22    56.3
#   raw-fit refined pivot              87      87    1.184   +20.23    66.7
#
# The raw-fit refinement improved sharpe, net profit, and win rate in both
# cases - a different asset class (equity vs. forex) and bar frequency
# (daily vs. 4-hour) at once - which is some evidence the effect isn't just
# AAPL/D1 noise, though both samples (15 and 87 trades) are still small.

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from TradeSavgol import (
    SYMBOL, TIMEFRAME, YEARS_BACK, MA_COL, SAVGOL_CLOSE_COL, SAVGOL_MA10_COL,
    INITIAL_CAPITAL,
    get_asset_data, add_ma10, add_savgol,
    find_entry_signals, simulate_trades,
    plot_trade_savgol_report, plot_ma10_entry_signals,
    print_ma10_long_short_table, print_performance_summary,
)
from TradeSavgolThresholds import _sweep_row, print_sweep_table

FIT_HALF_WIDTH = 3  # raw-data fit window: FIT_HALF_WIDTH bars on each side of the pivot


def _refine_with_raw_fit(df: pd.DataFrame, raw_column: str, center_position: int, fit_half_width: int):
    """Fit a quadratic to raw_column in a window around center_position and
    return (refined_position, refined_value, fit_found_extremum). Falls back
    to (center_position, raw value at center_position, False) if the window
    is too small or the fit has no interior extremum (i.e. isn't concave the
    expected way - a sign the window is on a slope, not a real peak)."""
    lo = max(0, center_position - fit_half_width)
    hi = min(len(df) - 1, center_position + fit_half_width)
    positions = np.arange(lo, hi + 1)
    raw_at_center = float(df[raw_column].iloc[center_position])

    if len(positions) < 3:
        return center_position, raw_at_center, False

    x = (positions - center_position).astype(float)
    y = df[raw_column].to_numpy(dtype=float)[positions]
    a, b, c = np.polyfit(x, y, 2)

    if abs(a) < 1e-9:
        return center_position, raw_at_center, False

    vertex_x = -b / (2 * a)
    if vertex_x < x.min() or vertex_x > x.max():
        # Fit's extremum falls outside the window - the raw data is still
        # trending here, not peaking, so trust the original detection instead.
        return center_position, raw_at_center, False

    vertex_value = a * vertex_x**2 + b * vertex_x + c
    refined_position = int(np.clip(round(center_position + vertex_x), lo, hi))
    return refined_position, vertex_value, True


def find_entry_signals_rawfit(
    df: pd.DataFrame,
    ma_column: str = MA_COL,
    signal_column: str = SAVGOL_CLOSE_COL,
    raw_column: str = "close",
    max_gap: int = 3,
    fit_half_width: int = FIT_HALF_WIDTH,
    diagnostics: list = None,
) -> pd.DataFrame:
    """Same confluence logic as TradeSavgol.find_entry_signals, except the
    Savgol-series pivot's bar position is refined by fitting a quadratic to
    raw `close` around it before being used for max_gap matching and the
    2-bar entry delay. If `diagnostics` is given, one dict per matched pivot
    (smoothed vs. refined position/value) is appended to it."""
    valid_positions = np.flatnonzero(df[ma_column].notna() & df[signal_column].notna())
    valid = df.iloc[valid_positions]
    if len(valid) < 3:
        return pd.DataFrame()

    ma_values = valid[ma_column].to_numpy()
    signal_values = valid[signal_column].to_numpy()
    ma_peaks, _ = find_peaks(ma_values)
    ma_valleys, _ = find_peaks(-ma_values)
    signal_peaks, _ = find_peaks(signal_values)
    signal_valleys, _ = find_peaks(-signal_values)

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
            smoothed_position = valid_positions[signal_extreme]
            smoothed_value = float(df[signal_column].iloc[smoothed_position])

            refined_position, refined_value, found = _refine_with_raw_fit(
                df, raw_column, smoothed_position, fit_half_width,
            )
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "side": side,
                        "smoothed_position": smoothed_position,
                        "refined_position": refined_position,
                        "position_shift_bars": refined_position - smoothed_position,
                        "smoothed_value": smoothed_value,
                        "raw_value_at_smoothed_pos": float(df[raw_column].iloc[smoothed_position]),
                        "refined_value": refined_value,
                        "fit_found_extremum": found,
                    }
                )

            pivot_position = max(ma_position, refined_position)
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


if __name__ == "__main__":
    date_to = datetime.now()
    date_from = date_to - timedelta(days=365 * YEARS_BACK)

    data = get_asset_data(SYMBOL, TIMEFRAME, date_from, date_to)
    data = add_ma10(data)
    data = add_savgol(data, column="close", out_col=SAVGOL_CLOSE_COL)
    data = add_savgol(data, column=MA_COL, out_col=SAVGOL_MA10_COL)

    baseline_entries = find_entry_signals(data)
    baseline_trades = simulate_trades(data, baseline_entries)

    diagnostics = []
    rawfit_entries = find_entry_signals_rawfit(data, diagnostics=diagnostics)
    rawfit_trades = simulate_trades(data, rawfit_entries)

    # --- Question 1: how big is the smoothed-vs-raw bias? ---
    diag_df = pd.DataFrame(diagnostics)
    print(f"Matched Savgol-close pivots: {len(diag_df)}  "
          f"(fit found an interior extremum for {diag_df['fit_found_extremum'].sum()} of them)\n")

    bias = diag_df["smoothed_value"] - diag_df["refined_value"]
    shift = diag_df["position_shift_bars"]
    print("Smoothed vs. raw-fit measurement bias at matched pivots:")
    print(f"  mean |smoothed - refined| value bias: {bias.abs().mean():.4f} "
          f"({(bias.abs() / diag_df['raw_value_at_smoothed_pos']).mean():.3%} of price)")
    print(f"  max  |smoothed - refined| value bias: {bias.abs().max():.4f}")
    print(f"  mean |position shift|: {shift.abs().mean():.2f} bars, "
          f"max |position shift|: {shift.abs().max()} bars, "
          f"shifted-by->=1-bar count: {(shift != 0).sum()} / {len(shift)}\n")

    # --- Question 2: did any shift actually change which trades get taken? ---
    print_sweep_table(
        f"{SYMBOL}, {YEARS_BACK}y, raw-fit pivot refinement vs baseline\n",
        [
            _sweep_row("baseline (smoothed pivot)", data, baseline_entries, baseline_trades),
            _sweep_row("raw-fit refined pivot", data, rawfit_entries, rawfit_trades),
        ],
    )

    # --- Report/plot/table for the raw-fit variant, same style as TradeSavgol.py ---
    print()
    print_ma10_long_short_table(rawfit_entries)

    plot_ma10_entry_signals(
        data, rawfit_entries, SYMBOL, out_path="aapl_ma10_signals_savgol_rawfit.png",
    )

    plot_trade_savgol_report(
        data, rawfit_trades, SYMBOL, INITIAL_CAPITAL, out_path="TradeSavgolRawFit.pdf",
    )

    print()
    print_performance_summary(data, rawfit_trades, INITIAL_CAPITAL)

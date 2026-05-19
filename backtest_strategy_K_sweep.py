"""
Strategy K Parameter Sweep: Dual Timeframe Momentum + Dip Buying
================================================================
Sweep key parameters (3888 combinations) focused on drawdown control.
"""

from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass, field
from typing import Optional
import time

# ─────────────────────────────────────────────────────────────────────────────
# Fixed Configuration
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "/Users/heyu11/Code/finance/data_layer/backtest_fixed.db"
BENCHMARK = "510300"
FEE_RATE = 0.0005  # 0.05% per side
INITIAL_CAPITAL = 1_000_000.0

# Fixed params (not swept)
W1 = 0.6
W2 = 0.4
MA_SHORT = 10
MA_MEDIUM_TREND = 20  # MA for trend filter (always 20)
ATR_PERIOD = 14
TREND_BREAK_DAYS = 2
PORTFOLIO_COOLDOWN = 5
VOLUME_BOOST_THRESHOLD = 1.5
VOLUME_BONUS = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# Sweep Parameters
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "MOM_MEDIUM": [15, 20, 25],
    "DIP_SHORT": [2, 3, 5],
    "TOP_N": [1, 2],
    "REBAL_FREQ": [3, 5],
    "PORTFOLIO_DD_STOP": [0.05, 0.06, 0.07, 0.08],
    "ATR_MULT": [1.5, 2.0, 2.5],
    "HARD_STOP": [0.03, 0.04, 0.05],
    "BREADTH_MIN": [0.35, 0.40, 0.45],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading (once)
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    """Load all ETF data from SQLite into a dict of code -> DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM etf_daily ORDER BY code, date", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    data = {}
    for code, group in df.groupby("code"):
        g = group.set_index("date").sort_index()
        data[code] = g[["open", "high", "low", "close", "volume"]].copy()
    return data


def precompute_indicators(data: dict[str, pd.DataFrame]):
    """Precompute indicators that don't depend on swept parameters."""
    for code, df in data.items():
        # Moving averages (fixed)
        df["ma10"] = df["close"].rolling(MA_SHORT).mean()
        df["ma20"] = df["close"].rolling(MA_MEDIUM_TREND).mean()
        # ATR
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["tr"] = tr
        df["atr14"] = tr.rolling(ATR_PERIOD).mean()
        # Volume
        df["vol_5d"] = df["volume"].rolling(5).mean()
        df["vol_20d"] = df["volume"].rolling(20).mean()
        # Above MA20 (for breadth)
        df["above_ma20"] = (df["close"] > df["ma20"]).astype(float)
        # Pre-compute various return lookbacks
        for lb in [2, 3, 5, 15, 20, 25]:
            df[f"ret_{lb}"] = df["close"].pct_change(lb)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Engine (streamlined for sweep)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Position:
    code: str
    entry_price: float
    shares: float
    highest_close: float
    days_below_ma20: int = 0
    pnl_pct: float = 0.0


def run_backtest(data: dict[str, pd.DataFrame], params: dict) -> dict:
    """
    Run one backtest with given parameters, return metrics dict.
    Minimized overhead for sweep speed.
    """
    mom_medium = params["MOM_MEDIUM"]
    dip_short = params["DIP_SHORT"]
    top_n = params["TOP_N"]
    rebal_freq = params["REBAL_FREQ"]
    portfolio_dd_stop = params["PORTFOLIO_DD_STOP"]
    atr_mult = params["ATR_MULT"]
    hard_stop = params["HARD_STOP"]
    breadth_min = params["BREADTH_MIN"]

    etf_codes = [c for c in data.keys() if c != BENCHMARK]
    all_dates = sorted(data[BENCHMARK].index)

    cash = INITIAL_CAPITAL
    positions: list[Position] = []
    peak_equity = INITIAL_CAPITAL
    portfolio_stop_active = False
    cooldown_remaining = 0
    days_since_rebal = rebal_freq  # trigger first day

    equity_values = []
    trade_pnls = []
    n_trades = 0

    warmup = max(mom_medium, 25) + 5  # ensure enough data

    # Breadth helper
    breadth_full = breadth_min + 0.05  # slightly above min for "full"

    def get_breadth_mult(breadth):
        if breadth > 0.60:
            return 1.0
        elif breadth > breadth_full:
            return 0.8
        elif breadth >= breadth_min:
            return 0.4
        else:
            return 0.0

    def get_max_pos(breadth):
        if breadth > breadth_full:
            return top_n
        elif breadth >= breadth_min:
            return 1
        else:
            return 0

    for i in range(warmup, len(all_dates)):
        date = all_dates[i]
        prev_date = all_dates[i - 1]

        # Cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # ── Check exits ──
        new_positions = []
        for pos in positions:
            df = data[pos.code]
            if date not in df.index:
                new_positions.append(pos)
                continue
            close = df.loc[date, "close"]
            ma20 = df.loc[date, "ma20"]
            atr = df.loc[date, "atr14"]

            # Update trailing high
            if close > pos.highest_close:
                pos.highest_close = close

            exit_flag = False

            # Hard stop
            if (close - pos.entry_price) / pos.entry_price <= -hard_stop:
                exit_flag = True

            # ATR trailing stop
            if not exit_flag and not np.isnan(atr) and pos.highest_close > 0:
                trail_level = pos.highest_close - atr_mult * atr
                if close < trail_level:
                    exit_flag = True

            # Trend break
            if not exit_flag and not np.isnan(ma20):
                if close < ma20:
                    pos.days_below_ma20 += 1
                else:
                    pos.days_below_ma20 = 0
                if pos.days_below_ma20 >= TREND_BREAK_DAYS:
                    exit_flag = True

            if exit_flag:
                # Sell at close
                proceeds = pos.shares * close * (1 - FEE_RATE)
                cost_basis = pos.shares * pos.entry_price * (1 + FEE_RATE)
                pnl_pct = (close * (1 - FEE_RATE)) / (pos.entry_price * (1 + FEE_RATE)) - 1
                cash += proceeds
                trade_pnls.append(pnl_pct)
                n_trades += 1
            else:
                new_positions.append(pos)

        positions = new_positions

        # ── Portfolio stop ──
        current_equity = cash
        for pos in positions:
            df = data[pos.code]
            if date in df.index:
                current_equity += pos.shares * df.loc[date, "close"]
        if current_equity > peak_equity:
            peak_equity = current_equity
        dd_from_peak = (peak_equity - current_equity) / peak_equity
        if dd_from_peak >= portfolio_dd_stop and not portfolio_stop_active:
            portfolio_stop_active = True
            cooldown_remaining = PORTFOLIO_COOLDOWN
            for pos in positions:
                df = data[pos.code]
                if date in df.index:
                    close = df.loc[date, "close"]
                    proceeds = pos.shares * close * (1 - FEE_RATE)
                    pnl_pct = (close * (1 - FEE_RATE)) / (pos.entry_price * (1 + FEE_RATE)) - 1
                    cash += proceeds
                    trade_pnls.append(pnl_pct)
                    n_trades += 1
            positions = []
            current_equity = cash

        if cooldown_remaining == 0 and portfolio_stop_active:
            portfolio_stop_active = False

        # ── Rebalance ──
        days_since_rebal += 1
        if (days_since_rebal >= rebal_freq and
                not portfolio_stop_active and cooldown_remaining == 0):

            # Breadth on prev_date
            above_count = 0
            total_count = 0
            for code in etf_codes:
                df = data[code]
                if prev_date in df.index:
                    val = df.loc[prev_date, "above_ma20"]
                    if not np.isnan(val):
                        above_count += val
                        total_count += 1
            breadth = above_count / total_count if total_count > 0 else 0.0

            max_pos = get_max_pos(breadth)
            size_mult = get_breadth_mult(breadth)

            if max_pos > 0 and size_mult > 0:
                # Compute scores
                r_med_col = f"ret_{mom_medium}"
                r_dip_col = f"ret_{dip_short}"

                r_med_vals = {}
                r_dip_vals = {}
                vol_boost_flags = {}

                for code in etf_codes:
                    df = data[code]
                    if prev_date not in df.index:
                        continue
                    r_med = df.loc[prev_date, r_med_col]
                    r_dip = df.loc[prev_date, r_dip_col]
                    if np.isnan(r_med) or np.isnan(r_dip):
                        continue
                    r_med_vals[code] = r_med
                    r_dip_vals[code] = r_dip

                    vol5 = df.loc[prev_date, "vol_5d"]
                    vol20 = df.loc[prev_date, "vol_20d"]
                    if not np.isnan(vol5) and not np.isnan(vol20) and vol20 > 0:
                        vol_boost_flags[code] = (vol5 / vol20) > VOLUME_BOOST_THRESHOLD
                    else:
                        vol_boost_flags[code] = False

                if len(r_med_vals) >= 2:
                    codes = list(r_med_vals.keys())
                    n = len(codes)

                    # Rank medium momentum (higher = better)
                    r_med_sorted = sorted(codes, key=lambda c: r_med_vals[c])
                    r_med_rank = {c: idx / (n - 1) for idx, c in enumerate(r_med_sorted)}

                    # Rank -R_dip (more negative dip = higher rank = bigger pullback)
                    r_dip_sorted = sorted(codes, key=lambda c: -r_dip_vals[c])
                    r_dip_rank = {c: idx / (n - 1) for idx, c in enumerate(r_dip_sorted)}

                    # Composite scores
                    scores = {}
                    for c in codes:
                        score = W1 * r_med_rank[c] + W2 * r_dip_rank[c]
                        if vol_boost_flags.get(c, False):
                            score += VOLUME_BONUS
                        scores[c] = score

                    # Trend filter
                    candidates = {}
                    for c, s in scores.items():
                        df = data[c]
                        if prev_date not in df.index:
                            continue
                        close = df.loc[prev_date, "close"]
                        ma10 = df.loc[prev_date, "ma10"]
                        ma20 = df.loc[prev_date, "ma20"]
                        if np.isnan(ma10) or np.isnan(ma20):
                            continue
                        if close > ma10 and close > ma20:
                            candidates[c] = s

                    # Exclude held
                    held_codes = {p.code for p in positions}
                    candidates = {c: s for c, s in candidates.items() if c not in held_codes}

                    slots = max_pos - len(positions)
                    if slots > 0 and candidates:
                        ranked = sorted(candidates.items(), key=lambda x: -x[1])
                        to_buy = ranked[:slots]

                        per_position = (size_mult * current_equity) / max_pos

                        for code, score in to_buy:
                            df = data[code]
                            if date not in df.index:
                                continue
                            buy_price = df.loc[date, "open"]
                            if np.isnan(buy_price) or buy_price <= 0:
                                continue
                            shares = per_position / (buy_price * (1 + FEE_RATE))
                            cost = shares * buy_price * (1 + FEE_RATE)
                            if cost > cash:
                                shares = cash / (buy_price * (1 + FEE_RATE))
                                cost = shares * buy_price * (1 + FEE_RATE)
                            if shares <= 0:
                                continue
                            cash -= cost
                            positions.append(Position(
                                code=code,
                                entry_price=buy_price,
                                shares=shares,
                                highest_close=buy_price,
                            ))

            days_since_rebal = 0

        # ── Record equity ──
        equity = cash
        for pos in positions:
            df = data[pos.code]
            if date in df.index:
                equity += pos.shares * df.loc[date, "close"]
        equity_values.append(equity)
        current_equity = equity
        if equity > peak_equity:
            peak_equity = equity

    # Close remaining positions
    if positions and len(all_dates) > warmup:
        last_date = all_dates[-1]
        for pos in positions:
            df = data[pos.code]
            if last_date in df.index:
                close = df.loc[last_date, "close"]
                pnl_pct = (close * (1 - FEE_RATE)) / (pos.entry_price * (1 + FEE_RATE)) - 1
                trade_pnls.append(pnl_pct)
                n_trades += 1

    # ── Compute metrics ──
    if len(equity_values) < 10:
        return None

    eq = np.array(equity_values)
    total_days = len(eq)
    years = total_days / 252

    total_return = eq[-1] / eq[0] - 1
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Daily returns
    daily_ret = np.diff(eq) / eq[:-1]
    std = np.std(daily_ret)
    sharpe = (np.mean(daily_ret) / std) * np.sqrt(252) if std > 0 else 0

    # Max drawdown
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd = np.min(drawdowns)

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = len([p for p in trade_pnls if p > 0]) / n_trades if n_trades > 0 else 0

    # Trades per week
    trades_per_week = n_trades / (total_days / 5) if total_days > 0 else 0

    return {
        "ann_return": ann_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
        "win_rate": win_rate,
        "trades_per_week": trades_per_week,
        "n_trades": n_trades,
        "total_return": total_return,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Sweep
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  STRATEGY K PARAMETER SWEEP")
    print("  Dual Timeframe Momentum + Dip Buying - DD Control Focus")
    print("=" * 70)

    print("\nLoading data...")
    data = load_data()
    print(f"Loaded {len(data)} ETFs")

    print("Precomputing indicators...")
    precompute_indicators(data)
    print("Indicators ready.\n")

    # Generate all combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combos = list(product(*param_values))
    total = len(all_combos)
    print(f"Total combinations to sweep: {total}\n")

    results = []
    start_time = time.time()

    for idx, combo in enumerate(all_combos):
        params = dict(zip(param_names, combo))
        metrics = run_backtest(data, params)

        if metrics is not None:
            row = {**params, **metrics}
            results.append(row)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (total - idx - 1) / rate
            print(f"  Progress: {idx+1}/{total} ({(idx+1)/total*100:.1f}%) | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s | "
                  f"Rate: {rate:.1f} combos/s")

    elapsed_total = time.time() - start_time
    print(f"\nSweep complete: {len(results)} valid results in {elapsed_total:.1f}s")
    print(f"Average speed: {len(all_combos)/elapsed_total:.1f} combos/sec\n")

    if not results:
        print("No valid results!")
        return

    df = pd.DataFrame(results)

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT 1: Top 20 by Calmar Ratio
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 90)
    print("  TOP 20 BY CALMAR RATIO")
    print("=" * 90)
    top_calmar = df.nlargest(20, "calmar")
    print(f"\n{'#':<3} {'MOM':>4} {'DIP':>4} {'N':>2} {'REB':>4} {'DDST':>5} {'ATR':>4} "
          f"{'HARD':>5} {'BMIN':>5} | {'Ann%':>6} {'MaxDD%':>7} {'Sharpe':>7} "
          f"{'Calmar':>7} {'WinR%':>6} {'T/wk':>5}")
    print("-" * 90)
    for rank, (_, row) in enumerate(top_calmar.iterrows(), 1):
        print(f"{rank:<3} {int(row['MOM_MEDIUM']):>4} {int(row['DIP_SHORT']):>4} "
              f"{int(row['TOP_N']):>2} {int(row['REBAL_FREQ']):>4} "
              f"{row['PORTFOLIO_DD_STOP']:>5.2f} {row['ATR_MULT']:>4.1f} "
              f"{row['HARD_STOP']:>5.2f} {row['BREADTH_MIN']:>5.2f} | "
              f"{row['ann_return']*100:>6.2f} {row['max_dd']*100:>7.2f} "
              f"{row['sharpe']:>7.3f} {row['calmar']:>7.3f} "
              f"{row['win_rate']*100:>6.1f} {row['trades_per_week']:>5.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT 2: Top 20 by Sharpe Ratio
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  TOP 20 BY SHARPE RATIO")
    print("=" * 90)
    top_sharpe = df.nlargest(20, "sharpe")
    print(f"\n{'#':<3} {'MOM':>4} {'DIP':>4} {'N':>2} {'REB':>4} {'DDST':>5} {'ATR':>4} "
          f"{'HARD':>5} {'BMIN':>5} | {'Ann%':>6} {'MaxDD%':>7} {'Sharpe':>7} "
          f"{'Calmar':>7} {'WinR%':>6} {'T/wk':>5}")
    print("-" * 90)
    for rank, (_, row) in enumerate(top_sharpe.iterrows(), 1):
        print(f"{rank:<3} {int(row['MOM_MEDIUM']):>4} {int(row['DIP_SHORT']):>4} "
              f"{int(row['TOP_N']):>2} {int(row['REBAL_FREQ']):>4} "
              f"{row['PORTFOLIO_DD_STOP']:>5.2f} {row['ATR_MULT']:>4.1f} "
              f"{row['HARD_STOP']:>5.2f} {row['BREADTH_MIN']:>5.2f} | "
              f"{row['ann_return']*100:>6.2f} {row['max_dd']*100:>7.2f} "
              f"{row['sharpe']:>7.3f} {row['calmar']:>7.3f} "
              f"{row['win_rate']*100:>6.1f} {row['trades_per_week']:>5.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT 3: Ann > 15% AND MaxDD > -12% AND trades/week < 2
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  BALANCED: Annual > 15% AND MaxDD > -12% AND Trades/Week < 2")
    print("=" * 90)
    balanced = df[(df["ann_return"] > 0.15) & (df["max_dd"] > -0.12) & (df["trades_per_week"] < 2)]
    balanced = balanced.sort_values("calmar", ascending=False)
    if len(balanced) == 0:
        print("\n  No combinations meet all three criteria.")
        # Relax criteria slightly to show closest
        close = df[(df["ann_return"] > 0.12) & (df["max_dd"] > -0.13) & (df["trades_per_week"] < 2.5)]
        close = close.sort_values("calmar", ascending=False)
        if len(close) > 0:
            print(f"  (Relaxed to Ann>12%, MaxDD>-13%, T/wk<2.5: {len(close)} found)")
            balanced = close.head(20)
    else:
        print(f"\n  Found {len(balanced)} combinations meeting all criteria.")
        balanced = balanced.head(20)

    if len(balanced) > 0:
        print(f"\n{'#':<3} {'MOM':>4} {'DIP':>4} {'N':>2} {'REB':>4} {'DDST':>5} {'ATR':>4} "
              f"{'HARD':>5} {'BMIN':>5} | {'Ann%':>6} {'MaxDD%':>7} {'Sharpe':>7} "
              f"{'Calmar':>7} {'WinR%':>6} {'T/wk':>5}")
        print("-" * 90)
        for rank, (_, row) in enumerate(balanced.iterrows(), 1):
            print(f"{rank:<3} {int(row['MOM_MEDIUM']):>4} {int(row['DIP_SHORT']):>4} "
                  f"{int(row['TOP_N']):>2} {int(row['REBAL_FREQ']):>4} "
                  f"{row['PORTFOLIO_DD_STOP']:>5.2f} {row['ATR_MULT']:>4.1f} "
                  f"{row['HARD_STOP']:>5.2f} {row['BREADTH_MIN']:>5.2f} | "
                  f"{row['ann_return']*100:>6.2f} {row['max_dd']*100:>7.2f} "
                  f"{row['sharpe']:>7.3f} {row['calmar']:>7.3f} "
                  f"{row['win_rate']*100:>6.1f} {row['trades_per_week']:>5.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT 4: Ann > 25% AND MaxDD > -15%
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  HIGH RETURN: Annual > 25% AND MaxDD > -15%")
    print("=" * 90)
    high_ret = df[(df["ann_return"] > 0.25) & (df["max_dd"] > -0.15)]
    high_ret = high_ret.sort_values("calmar", ascending=False)
    if len(high_ret) == 0:
        print("\n  No combinations meet both criteria.")
        # Relax
        close2 = df[(df["ann_return"] > 0.20) & (df["max_dd"] > -0.16)]
        close2 = close2.sort_values("calmar", ascending=False)
        if len(close2) > 0:
            print(f"  (Relaxed to Ann>20%, MaxDD>-16%: {len(close2)} found)")
            high_ret = close2.head(20)
    else:
        print(f"\n  Found {len(high_ret)} combinations meeting both criteria.")
        high_ret = high_ret.head(20)

    if len(high_ret) > 0:
        print(f"\n{'#':<3} {'MOM':>4} {'DIP':>4} {'N':>2} {'REB':>4} {'DDST':>5} {'ATR':>4} "
              f"{'HARD':>5} {'BMIN':>5} | {'Ann%':>6} {'MaxDD%':>7} {'Sharpe':>7} "
              f"{'Calmar':>7} {'WinR%':>6} {'T/wk':>5}")
        print("-" * 90)
        for rank, (_, row) in enumerate(high_ret.iterrows(), 1):
            print(f"{rank:<3} {int(row['MOM_MEDIUM']):>4} {int(row['DIP_SHORT']):>4} "
                  f"{int(row['TOP_N']):>2} {int(row['REBAL_FREQ']):>4} "
                  f"{row['PORTFOLIO_DD_STOP']:>5.2f} {row['ATR_MULT']:>4.1f} "
                  f"{row['HARD_STOP']:>5.2f} {row['BREADTH_MIN']:>5.2f} | "
                  f"{row['ann_return']*100:>6.2f} {row['max_dd']*100:>7.2f} "
                  f"{row['sharpe']:>7.3f} {row['calmar']:>7.3f} "
                  f"{row['win_rate']*100:>6.1f} {row['trades_per_week']:>5.2f}")

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT 5: Parameter Sensitivity for Best Calmar
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  PARAMETER SENSITIVITY (Best Calmar combination)")
    print("=" * 90)

    best_row = df.loc[df["calmar"].idxmax()]
    best_params = {k: best_row[k] for k in param_names}
    print(f"\n  Best Calmar params: {best_params}")
    print(f"  Ann: {best_row['ann_return']*100:.2f}% | MaxDD: {best_row['max_dd']*100:.2f}% | "
          f"Sharpe: {best_row['sharpe']:.3f} | Calmar: {best_row['calmar']:.3f}")
    print()

    for param in param_names:
        values = PARAM_GRID[param]
        best_val = best_params[param]

        print(f"  {param} sensitivity (best={best_val}):")
        print(f"    {'Value':<8} {'Ann%':>7} {'MaxDD%':>8} {'Sharpe':>8} {'Calmar':>8} {'WinR%':>7} {'T/wk':>6}")
        print(f"    {'─'*8} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*6}")

        for val in values:
            # Filter: all other params same as best, only this one varies
            mask = pd.Series([True] * len(df))
            for other_param in param_names:
                if other_param == param:
                    mask &= (df[other_param] == val)
                else:
                    mask &= (df[other_param] == best_params[other_param])
            subset = df[mask]
            if len(subset) == 1:
                r = subset.iloc[0]
                marker = " <-- best" if val == best_val else ""
                print(f"    {val:<8} {r['ann_return']*100:>7.2f} {r['max_dd']*100:>8.2f} "
                      f"{r['sharpe']:>8.3f} {r['calmar']:>8.3f} "
                      f"{r['win_rate']*100:>7.1f} {r['trades_per_week']:>6.2f}{marker}")
            elif len(subset) == 0:
                print(f"    {val:<8}  (no data)")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # Summary statistics
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 90)
    print("  SUMMARY STATISTICS ACROSS ALL COMBINATIONS")
    print("=" * 90)
    print(f"\n  {'Metric':<18} {'Min':>9} {'25%':>9} {'Median':>9} {'75%':>9} {'Max':>9}")
    print(f"  {'─'*18} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")
    for col, label, mult in [
        ("ann_return", "Annual Return %", 100),
        ("max_dd", "Max Drawdown %", 100),
        ("sharpe", "Sharpe Ratio", 1),
        ("calmar", "Calmar Ratio", 1),
        ("win_rate", "Win Rate %", 100),
        ("trades_per_week", "Trades/Week", 1),
    ]:
        desc = df[col].describe()
        print(f"  {label:<18} {desc['min']*mult:>9.2f} {desc['25%']*mult:>9.2f} "
              f"{desc['50%']*mult:>9.2f} {desc['75%']*mult:>9.2f} {desc['max']*mult:>9.2f}")

    print("\n" + "=" * 90)
    print("  SWEEP COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()

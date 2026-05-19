"""
Backtest Strategy N: Rank-Weighted Ensemble with Adaptive Regime
================================================================
ETF rotation strategy using 5-signal ensemble scoring with regime-dependent
weights, aggressive drawdown control, and anti-whipsaw filters.
"""

import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# Configuration
# ============================================================
DB_PATH = "data_layer/backtest_fixed.db"
BENCHMARK_CODE = "510300"
FEE_RATE = 0.0005  # 0.05% per side
START_DATE = "2021-05-14"
END_DATE = "2025-12-31"
INITIAL_CAPITAL = 1_000_000.0

# Regime thresholds
BULL_BREADTH = 0.55
BEAR_BREADTH = 0.30
REGIME_CONFIRM_DAYS = 2

# Position sizing
BULL_POSITION_PCT = 0.45  # per position in bull (2 positions = 90%)
NEUTRAL_POSITION_PCT = 0.60  # single position in neutral

# Rebalance frequency
BULL_REBAL_FREQ = 3
NEUTRAL_REBAL_FREQ = 5

# Exit parameters
ATR_TRAIL_MULT = 2.0
ATR_PERIOD = 14
HARD_STOP_PCT = -0.04

# Drawdown control
DD_REDUCE_THRESHOLD = 0.05  # 5% dd -> half size
DD_CASH_THRESHOLD = 0.07   # 7% dd -> all cash
DD_COOLDOWN_DAYS = 5

# Anti-whipsaw
MIN_HOLD_DAYS = 2
POST_STOP_WAIT_DAYS = 2


# ============================================================
# Data Loading
# ============================================================
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT code, date, open, high, low, close, volume FROM etf_daily ORDER BY date, code",
        conn
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def prepare_price_data(df):
    """Pivot data into per-ETF DataFrames."""
    trading_codes = [c for c in df['code'].unique() if c != BENCHMARK_CODE]
    all_dates = sorted(df['date'].unique())

    # Create pivot tables
    close_df = df.pivot(index='date', columns='code', values='close')
    open_df = df.pivot(index='date', columns='code', values='open')
    high_df = df.pivot(index='date', columns='code', values='high')
    low_df = df.pivot(index='date', columns='code', values='low')
    volume_df = df.pivot(index='date', columns='code', values='volume')

    return trading_codes, all_dates, close_df, open_df, high_df, low_df, volume_df


# ============================================================
# Signal Computation (no look-ahead)
# ============================================================
def compute_signals(close_df, high_df, low_df, volume_df, trading_codes, benchmark_code):
    """
    Compute all 5 signals for each date. Returns DataFrames.
    """
    # Signal 1: 5-day momentum
    sig1 = close_df[trading_codes].pct_change(5)

    # Signal 2: 20-day momentum
    sig2 = close_df[trading_codes].pct_change(20)

    # Signal 3: Volume breakout (5d avg vol / 20d avg vol)
    vol_5d = volume_df[trading_codes].rolling(5).mean()
    vol_20d = volume_df[trading_codes].rolling(20).mean()
    sig3 = vol_5d / vol_20d

    # Signal 4: Relative strength vs benchmark (ETF 10d return - benchmark 10d return)
    etf_10d_ret = close_df[trading_codes].pct_change(10)
    bench_10d_ret = close_df[benchmark_code].pct_change(10)
    sig4 = etf_10d_ret.sub(bench_10d_ret, axis=0)

    # Signal 5: Trend quality: (close - MA20) / ATR(20)
    ma20 = close_df[trading_codes].rolling(20).mean()
    # ATR(20)
    tr_frames = {}
    for code in trading_codes:
        h = high_df[code]
        l = low_df[code]
        c_prev = close_df[code].shift(1)
        tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
        tr_frames[code] = tr
    tr_df = pd.DataFrame(tr_frames, index=close_df.index)
    atr20 = tr_df.rolling(20).mean()
    sig5 = (close_df[trading_codes] - ma20) / atr20.replace(0, np.nan)

    return sig1, sig2, sig3, sig4, sig5


def compute_breadth(close_df, trading_codes):
    """Market breadth: fraction of ETFs above their 20-day MA."""
    ma20 = close_df[trading_codes].rolling(20).mean()
    above_ma = (close_df[trading_codes] > ma20).sum(axis=1)
    breadth = above_ma / len(trading_codes)
    return breadth


def compute_ma10(close_df, trading_codes):
    """10-day MA for each ETF (used as trend filter)."""
    return close_df[trading_codes].rolling(10).mean()


def compute_atr14(high_df, low_df, close_df, trading_codes):
    """ATR(14) for trailing stop."""
    tr_frames = {}
    for code in trading_codes:
        h = high_df[code]
        l = low_df[code]
        c_prev = close_df[code].shift(1)
        tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
        tr_frames[code] = tr
    tr_df = pd.DataFrame(tr_frames, index=close_df.index)
    return tr_df.rolling(ATR_PERIOD).mean()


# ============================================================
# Position & Trade Tracking
# ============================================================
@dataclass
class Position:
    code: str
    entry_price: float
    entry_date: pd.Timestamp
    shares: float
    highest_close: float
    target_weight: float
    entry_idx: int  # index in date array


@dataclass
class Trade:
    code: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    regime_at_entry: str
    holding_days: int
    signals_at_entry: dict = field(default_factory=dict)


# ============================================================
# Backtest Engine
# ============================================================
def run_backtest():
    print("Loading data...")
    df = load_data()
    trading_codes, all_dates, close_df, open_df, high_df, low_df, volume_df = prepare_price_data(df)

    print(f"Trading universe: {len(trading_codes)} ETFs")
    print(f"Period: {START_DATE} to {END_DATE}, {len(all_dates)} trading days")
    print(f"Benchmark: {BENCHMARK_CODE}")
    print()

    # Pre-compute all signals
    print("Computing signals...")
    sig1, sig2, sig3, sig4, sig5 = compute_signals(close_df, high_df, low_df, volume_df, trading_codes, BENCHMARK_CODE)
    breadth = compute_breadth(close_df, trading_codes)
    ma10 = compute_ma10(close_df, trading_codes)
    atr14 = compute_atr14(high_df, low_df, close_df, trading_codes)

    # Date index mapping
    date_idx = {d: i for i, d in enumerate(all_dates)}

    # State variables
    equity = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    positions: dict[str, Position] = {}  # code -> Position
    trades: list[Trade] = []
    equity_curve = []
    benchmark_curve = []

    # Regime tracking
    confirmed_regime = "neutral"
    regime_counter = 0
    pending_regime = "neutral"

    # Drawdown control state
    peak_equity = INITIAL_CAPITAL
    dd_reduction_active = False
    dd_cash_forced = False
    dd_cooldown_remaining = 0
    post_cooldown_half_size = False

    # Rebalance tracking
    days_since_rebal = 999  # force immediate first rebalance
    last_rebal_regime = None

    # Anti-whipsaw state
    last_stop_exit_date_per_code: dict[str, int] = {}  # code -> day_idx of last stop exit
    position_entry_day: dict[str, int] = {}  # code -> day_idx of entry

    # Equity curve MA tracking
    equity_values = []

    # Signal contribution tracking
    signal_scores_at_entry = []

    # Regime time tracking
    regime_days = {"bull": 0, "neutral": 0, "bear": 0}
    regime_returns = {"bull": [], "neutral": [], "bear": []}

    # Get benchmark starting price
    bench_start_price = None
    for d in all_dates:
        if BENCHMARK_CODE in close_df.columns and not pd.isna(close_df.loc[d, BENCHMARK_CODE]):
            bench_start_price = close_df.loc[d, BENCHMARK_CODE]
            break

    print("Running backtest...")
    warmup_days = 25  # need at least 20 days for signals + some buffer

    for day_i, date in enumerate(all_dates):
        # ---- Compute daily portfolio value ----
        port_value = cash
        for code, pos in positions.items():
            if code in close_df.columns and not pd.isna(close_df.loc[date, code]):
                port_value += pos.shares * close_df.loc[date, code]
            else:
                port_value += pos.shares * pos.entry_price  # fallback

        equity = port_value
        equity_values.append(equity)
        equity_curve.append(equity)

        # Benchmark
        if BENCHMARK_CODE in close_df.columns and not pd.isna(close_df.loc[date, BENCHMARK_CODE]):
            bench_price = close_df.loc[date, BENCHMARK_CODE]
            benchmark_curve.append(INITIAL_CAPITAL * bench_price / bench_start_price if bench_start_price else INITIAL_CAPITAL)
        else:
            benchmark_curve.append(benchmark_curve[-1] if benchmark_curve else INITIAL_CAPITAL)

        # Skip warmup period
        if day_i < warmup_days:
            regime_days["neutral"] += 1
            continue

        # ---- Determine raw regime ----
        if date in breadth.index and not pd.isna(breadth.loc[date]):
            b = breadth.loc[date]
            if b > BULL_BREADTH:
                raw_reg = "bull"
            elif b < BEAR_BREADTH:
                raw_reg = "bear"
            else:
                raw_reg = "neutral"
        else:
            raw_reg = "neutral"

        # Regime confirmation (2 consecutive days)
        if raw_reg == pending_regime:
            regime_counter += 1
        else:
            pending_regime = raw_reg
            regime_counter = 1

        prev_regime = confirmed_regime
        if regime_counter >= REGIME_CONFIRM_DAYS and pending_regime != confirmed_regime:
            confirmed_regime = pending_regime

            # If regime changes to bear -> immediate liquidation
            if confirmed_regime == "bear" and positions:
                for code in list(positions.keys()):
                    pos = positions[code]
                    exit_price = close_df.loc[date, code] if not pd.isna(close_df.loc[date, code]) else pos.entry_price
                    pnl = (exit_price - pos.entry_price) * pos.shares
                    fee = exit_price * pos.shares * FEE_RATE
                    pnl -= fee
                    cash += pos.shares * exit_price - fee
                    trades.append(Trade(
                        code=code, entry_date=pos.entry_date, exit_date=date,
                        entry_price=pos.entry_price, exit_price=exit_price,
                        shares=pos.shares, pnl=pnl,
                        pnl_pct=(exit_price / pos.entry_price - 1),
                        exit_reason="regime_bear",
                        regime_at_entry=prev_regime,
                        holding_days=day_i - pos.entry_idx,
                        signals_at_entry={}
                    ))
                positions.clear()
                position_entry_day.clear()

        regime_days[confirmed_regime] += 1

        # Track regime returns
        if day_i > 0:
            daily_ret = (equity_values[-1] / equity_values[-2] - 1) if equity_values[-2] > 0 else 0
            regime_returns[confirmed_regime].append(daily_ret)

        # ---- Peak equity and drawdown control ----
        if equity > peak_equity:
            peak_equity = equity
            dd_reduction_active = False
            if post_cooldown_half_size:
                post_cooldown_half_size = False  # profitable after cooldown, restore full size

        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

        # Cooldown management
        if dd_cooldown_remaining > 0:
            dd_cooldown_remaining -= 1
            if dd_cooldown_remaining == 0:
                dd_cash_forced = False
                post_cooldown_half_size = True
                # CRITICAL: Reset peak to current equity after cooldown to prevent
                # perpetual drawdown state
                peak_equity = equity
            continue  # skip all trading during cooldown

        # 7% drawdown from peak -> forced cash + cooldown
        if current_dd >= DD_CASH_THRESHOLD and not dd_cash_forced:
            dd_cash_forced = True
            dd_cooldown_remaining = DD_COOLDOWN_DAYS
            # Liquidate all
            for code in list(positions.keys()):
                pos = positions[code]
                exit_price = close_df.loc[date, code] if not pd.isna(close_df.loc[date, code]) else pos.entry_price
                pnl = (exit_price - pos.entry_price) * pos.shares
                fee = exit_price * pos.shares * FEE_RATE
                pnl -= fee
                cash += pos.shares * exit_price - fee
                trades.append(Trade(
                    code=code, entry_date=pos.entry_date, exit_date=date,
                    entry_price=pos.entry_price, exit_price=exit_price,
                    shares=pos.shares, pnl=pnl,
                    pnl_pct=(exit_price / pos.entry_price - 1),
                    exit_reason="dd_7pct_cash",
                    regime_at_entry=confirmed_regime,
                    holding_days=day_i - pos.entry_idx,
                    signals_at_entry={}
                ))
            positions.clear()
            position_entry_day.clear()
            continue

        # 5% drawdown -> reduce size flag
        if current_dd >= DD_REDUCE_THRESHOLD:
            dd_reduction_active = True
        else:
            dd_reduction_active = False

        # ---- Equity curve MA check ----
        equity_ma_reduce = False
        if len(equity_values) >= 10:
            eq_ma10 = np.mean(equity_values[-10:])
            if equity < eq_ma10:
                equity_ma_reduce = True

        # ---- Bear regime: stay in cash ----
        if confirmed_regime == "bear":
            days_since_rebal += 1
            continue

        # ---- Check exits for existing positions ----
        codes_to_exit = []
        for code, pos in positions.items():
            if code not in close_df.columns or pd.isna(close_df.loc[date, code]):
                continue

            current_close = close_df.loc[date, code]
            # Update highest close
            if current_close > pos.highest_close:
                pos.highest_close = current_close

            exit_reason = None

            # Hard stop: -4% from entry
            if (current_close / pos.entry_price - 1) <= HARD_STOP_PCT:
                exit_reason = "hard_stop"

            # ATR trailing stop
            if exit_reason is None and date in atr14.index and code in atr14.columns:
                atr_val = atr14.loc[date, code]
                if not pd.isna(atr_val) and atr_val > 0:
                    trail_stop = pos.highest_close - ATR_TRAIL_MULT * atr_val
                    if current_close < trail_stop:
                        exit_reason = "atr_trailing_stop"

            # Minimum hold check - only hard_stop can override
            hold_days = day_i - pos.entry_idx
            if exit_reason and hold_days < MIN_HOLD_DAYS:
                if exit_reason != "hard_stop":
                    exit_reason = None

            if exit_reason:
                codes_to_exit.append((code, exit_reason))

        # Execute exits
        for code, reason in codes_to_exit:
            pos = positions[code]
            exit_price = close_df.loc[date, code]
            pnl = (exit_price - pos.entry_price) * pos.shares
            fee = exit_price * pos.shares * FEE_RATE
            pnl -= fee
            cash += pos.shares * exit_price - fee
            trades.append(Trade(
                code=code, entry_date=pos.entry_date, exit_date=date,
                entry_price=pos.entry_price, exit_price=exit_price,
                shares=pos.shares, pnl=pnl,
                pnl_pct=(exit_price / pos.entry_price - 1),
                exit_reason=reason,
                regime_at_entry=confirmed_regime,
                holding_days=day_i - pos.entry_idx,
                signals_at_entry={}
            ))
            del positions[code]
            if code in position_entry_day:
                del position_entry_day[code]
            # Record stop exit for anti-whipsaw
            if reason in ("hard_stop", "atr_trailing_stop"):
                last_stop_exit_date_per_code[code] = day_i

        # ---- Equity curve reduction: reduce all positions by 50% ----
        if equity_ma_reduce and positions:
            for code in list(positions.keys()):
                pos = positions[code]
                reduce_shares = pos.shares * 0.5
                if reduce_shares > 0 and not pd.isna(close_df.loc[date, code]):
                    exit_price = close_df.loc[date, code]
                    fee = exit_price * reduce_shares * FEE_RATE
                    cash += reduce_shares * exit_price - fee
                    pos.shares -= reduce_shares
                    if pos.shares <= 0:
                        del positions[code]
                        if code in position_entry_day:
                            del position_entry_day[code]

        # ---- Rebalance check ----
        rebal_freq = BULL_REBAL_FREQ if confirmed_regime == "bull" else NEUTRAL_REBAL_FREQ
        days_since_rebal += 1

        # Force rebalance if regime changed from bear to non-bear
        regime_changed = (last_rebal_regime is not None and last_rebal_regime != confirmed_regime)

        if days_since_rebal < rebal_freq and not regime_changed and not (not positions and days_since_rebal >= 1):
            continue

        # ---- Compute ensemble scores ----
        if date not in sig1.index:
            continue

        # Get signal values for this date
        s1_vals = sig1.loc[date, trading_codes]
        s2_vals = sig2.loc[date, trading_codes]
        s3_vals = sig3.loc[date, trading_codes]
        s4_vals = sig4.loc[date, trading_codes]
        s5_vals = sig5.loc[date, trading_codes]

        # Rank each signal (1=worst, N=best for valid entries)
        def rank_signal(s):
            valid = s.dropna()
            if len(valid) == 0:
                return pd.Series(0, index=s.index)
            ranks = valid.rank(method='average')
            return ranks.reindex(s.index, fill_value=0)

        r1 = rank_signal(s1_vals)
        r2 = rank_signal(s2_vals)
        r3 = rank_signal(s3_vals)
        r4 = rank_signal(s4_vals)
        r5 = rank_signal(s5_vals)

        # Regime-dependent weights
        if confirmed_regime == "bull":
            composite = 0.35 * r1 + 0.20 * r2 + 0.20 * r3 + 0.15 * r4 + 0.10 * r5
        else:  # neutral
            composite = 0.20 * r1 + 0.25 * r2 + 0.15 * r3 + 0.20 * r4 + 0.20 * r5

        # ---- Selection + Filters ----
        n_select = 2 if confirmed_regime == "bull" else 1

        # Sort by composite score descending
        sorted_codes = composite.sort_values(ascending=False)

        selected = []
        for code in sorted_codes.index:
            if len(selected) >= n_select:
                break
            if composite[code] <= 0:
                continue
            # Filter: close > 10d MA
            if date in ma10.index and code in ma10.columns:
                ma_val = ma10.loc[date, code]
                close_val = close_df.loc[date, code]
                if pd.isna(ma_val) or pd.isna(close_val) or close_val <= ma_val:
                    continue
            else:
                continue
            # Anti-whipsaw: check post-stop wait
            if code in last_stop_exit_date_per_code:
                days_since_stop = day_i - last_stop_exit_date_per_code[code]
                if days_since_stop < POST_STOP_WAIT_DAYS:
                    continue
            selected.append(code)

        # ---- Determine target positions ----
        if confirmed_regime == "bull":
            target_weight = BULL_POSITION_PCT
        else:
            target_weight = NEUTRAL_POSITION_PCT

        # Apply drawdown reduction
        size_mult = 1.0
        if dd_reduction_active:
            size_mult = 0.5
        if post_cooldown_half_size:
            size_mult = 0.5

        target_weight *= size_mult

        # ---- Execute rebalance ----
        # Close positions not in selected (respecting min hold)
        for code in list(positions.keys()):
            if code not in selected:
                hold_days = day_i - positions[code].entry_idx
                if hold_days < MIN_HOLD_DAYS:
                    continue  # keep position due to min hold
                pos = positions[code]
                exit_price = close_df.loc[date, code]
                if pd.isna(exit_price):
                    continue
                pnl = (exit_price - pos.entry_price) * pos.shares
                fee = exit_price * pos.shares * FEE_RATE
                pnl -= fee
                cash += pos.shares * exit_price - fee
                trades.append(Trade(
                    code=code, entry_date=pos.entry_date, exit_date=date,
                    entry_price=pos.entry_price, exit_price=exit_price,
                    shares=pos.shares, pnl=pnl,
                    pnl_pct=(exit_price / pos.entry_price - 1),
                    exit_reason="rebalance_out",
                    regime_at_entry=confirmed_regime,
                    holding_days=hold_days,
                    signals_at_entry={}
                ))
                del positions[code]
                if code in position_entry_day:
                    del position_entry_day[code]

        # Recalculate portfolio value after exits
        port_value = cash
        for code, pos in positions.items():
            c = close_df.loc[date, code] if not pd.isna(close_df.loc[date, code]) else pos.entry_price
            port_value += pos.shares * c

        # Open new positions
        for code in selected:
            if code in positions:
                continue  # already holding
            # New position
            target_val = port_value * target_weight
            entry_price = close_df.loc[date, code]
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            shares = target_val / entry_price
            fee = target_val * FEE_RATE
            if cash < target_val + fee:
                # Use available cash
                available = cash * 0.99  # leave tiny buffer
                if available <= 0:
                    continue
                target_val = available / (1 + FEE_RATE)
                shares = target_val / entry_price
                fee = target_val * FEE_RATE
            if shares <= 0:
                continue
            cash -= (shares * entry_price + fee)
            positions[code] = Position(
                code=code, entry_price=entry_price, entry_date=date,
                shares=shares, highest_close=entry_price,
                target_weight=target_weight, entry_idx=day_i
            )
            position_entry_day[code] = day_i

            # Record signal ranks at entry for analysis
            signal_scores_at_entry.append({
                'code': code, 'date': date,
                'r1': r1[code], 'r2': r2[code], 'r3': r3[code],
                'r4': r4[code], 'r5': r5[code],
                'composite': composite[code],
                'regime': confirmed_regime
            })

        days_since_rebal = 0
        last_rebal_regime = confirmed_regime

    # ---- Close remaining positions at end ----
    final_date = all_dates[-1]
    for code in list(positions.keys()):
        pos = positions[code]
        exit_price = close_df.loc[final_date, code] if not pd.isna(close_df.loc[final_date, code]) else pos.entry_price
        pnl = (exit_price - pos.entry_price) * pos.shares
        fee = exit_price * pos.shares * FEE_RATE
        pnl -= fee
        cash += pos.shares * exit_price - fee
        trades.append(Trade(
            code=code, entry_date=pos.entry_date, exit_date=final_date,
            entry_price=pos.entry_price, exit_price=exit_price,
            shares=pos.shares, pnl=pnl,
            pnl_pct=(exit_price / pos.entry_price - 1),
            exit_reason="end_of_backtest",
            regime_at_entry=confirmed_regime,
            holding_days=len(all_dates) - 1 - pos.entry_idx,
            signals_at_entry={}
        ))
    positions.clear()

    # ============================================================
    # Results Computation
    # ============================================================
    equity_arr = np.array(equity_curve)
    bench_arr = np.array(benchmark_curve)
    dates_arr = np.array(all_dates)

    # Daily returns
    daily_returns = np.diff(equity_arr) / equity_arr[:-1]
    bench_daily_returns = np.diff(bench_arr) / bench_arr[:-1]

    # Total return
    total_return = (equity_arr[-1] / equity_arr[0]) - 1
    bench_total_return = (bench_arr[-1] / bench_arr[0]) - 1

    # Annualized return
    n_years = len(all_dates) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    bench_annual_return = (1 + bench_total_return) ** (1 / n_years) - 1

    # Max drawdown
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max
    max_dd = drawdowns.min()

    bench_running_max = np.maximum.accumulate(bench_arr)
    bench_drawdowns = (bench_arr - bench_running_max) / bench_running_max
    bench_max_dd = bench_drawdowns.min()

    # Sharpe ratio (annualized, rf=0)
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0

    # Calmar ratio
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # Sortino ratio
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
    sortino = np.mean(daily_returns) / downside_std * np.sqrt(252)

    # Trade statistics
    n_trades = len(trades)
    if n_trades > 0:
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning_trades) / n_trades
        avg_win = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_hold = np.mean([t.holding_days for t in trades])
        trades_per_week = n_trades / (len(all_dates) / 5)
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_hold = trades_per_week = 0

    # ============================================================
    # Print Results
    # ============================================================
    print("\n" + "=" * 70)
    print("STRATEGY N: Rank-Weighted Ensemble with Adaptive Regime")
    print("=" * 70)

    print(f"\n{'PERFORMANCE SUMMARY':=^70}")
    print(f"  Period:              {START_DATE} to {END_DATE} ({len(all_dates)} days)")
    print(f"  Initial Capital:     ${INITIAL_CAPITAL:,.0f}")
    print(f"  Final Equity:        ${equity_arr[-1]:,.0f}")
    print(f"  Total Return:        {total_return*100:.2f}%")
    print(f"  Annual Return:       {annual_return*100:.2f}%")
    print(f"  Max Drawdown:        {max_dd*100:.2f}%")
    print(f"  Sharpe Ratio:        {sharpe:.3f}")
    print(f"  Sortino Ratio:       {sortino:.3f}")
    print(f"  Calmar Ratio:        {calmar:.3f}")
    print()
    print(f"  Benchmark Return:    {bench_total_return*100:.2f}%")
    print(f"  Benchmark Annual:    {bench_annual_return*100:.2f}%")
    print(f"  Benchmark Max DD:    {bench_max_dd*100:.2f}%")
    print(f"  Alpha (annual):      {(annual_return - bench_annual_return)*100:.2f}%")

    print(f"\n{'TRADE STATISTICS':=^70}")
    print(f"  Total Trades:        {n_trades}")
    print(f"  Trades/Week:         {trades_per_week:.2f}")
    print(f"  Win Rate:            {win_rate*100:.1f}%")
    print(f"  Avg Win:             {avg_win*100:.2f}%")
    print(f"  Avg Loss:            {avg_loss*100:.2f}%")
    print(f"  Profit Factor:       {profit_factor:.2f}")
    print(f"  Avg Holding Days:    {avg_hold:.1f}")

    # ---- Yearly Breakdown ----
    print(f"\n{'YEARLY BREAKDOWN':=^70}")
    print(f"  {'Year':<6} {'Return':>10} {'MaxDD':>10} {'Sharpe':>8} {'Trades':>8} {'BenchRet':>10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

    for year in range(2021, 2026):
        year_mask = np.array([d.year == year for d in all_dates])
        if not year_mask.any():
            continue
        year_eq = equity_arr[year_mask]
        year_bench = bench_arr[year_mask]
        if len(year_eq) < 2:
            continue

        yr_ret = year_eq[-1] / year_eq[0] - 1
        yr_running_max = np.maximum.accumulate(year_eq)
        yr_dd = ((year_eq - yr_running_max) / yr_running_max).min()
        yr_daily_ret = np.diff(year_eq) / year_eq[:-1]
        yr_sharpe = np.mean(yr_daily_ret) / np.std(yr_daily_ret) * np.sqrt(252) if np.std(yr_daily_ret) > 0 else 0
        yr_bench_ret = year_bench[-1] / year_bench[0] - 1

        yr_trades = [t for t in trades if t.entry_date.year == year]

        print(f"  {year:<6} {yr_ret*100:>9.2f}% {yr_dd*100:>9.2f}% {yr_sharpe:>8.3f} {len(yr_trades):>8} {yr_bench_ret*100:>9.2f}%")

    # ---- Regime Distribution ----
    print(f"\n{'REGIME TIME DISTRIBUTION':=^70}")
    total_regime_days = sum(regime_days.values())
    for regime in ["bull", "neutral", "bear"]:
        pct = regime_days[regime] / total_regime_days * 100 if total_regime_days > 0 else 0
        rets = regime_returns[regime]
        avg_ret = np.mean(rets) * 252 * 100 if rets else 0  # annualized
        cum_ret = (np.prod([1 + r for r in rets]) - 1) * 100 if rets else 0
        print(f"  {regime.upper():<10} {regime_days[regime]:>5} days ({pct:>5.1f}%)  "
              f"Ann.Ret: {avg_ret:>7.2f}%  Cum.Ret: {cum_ret:>7.2f}%")

    # ---- Per-Regime Trade Stats ----
    print(f"\n{'PER-REGIME TRADE PERFORMANCE':=^70}")
    for regime in ["bull", "neutral", "bear"]:
        regime_trades = [t for t in trades if t.regime_at_entry == regime]
        if not regime_trades:
            print(f"  {regime.upper():<10} No trades")
            continue
        r_wins = [t for t in regime_trades if t.pnl > 0]
        r_wr = len(r_wins) / len(regime_trades) * 100
        r_avg_pnl = np.mean([t.pnl_pct for t in regime_trades]) * 100
        r_total_pnl = sum(t.pnl for t in regime_trades)
        print(f"  {regime.upper():<10} {len(regime_trades):>4} trades  "
              f"WinRate: {r_wr:>5.1f}%  AvgReturn: {r_avg_pnl:>6.2f}%  TotalPnL: ${r_total_pnl:>,.0f}")

    # ---- Exit Reason Breakdown ----
    print(f"\n{'EXIT REASON BREAKDOWN':=^70}")
    exit_reasons = {}
    for t in trades:
        r = t.exit_reason
        if r not in exit_reasons:
            exit_reasons[r] = {'count': 0, 'pnl': 0, 'wins': 0}
        exit_reasons[r]['count'] += 1
        exit_reasons[r]['pnl'] += t.pnl
        if t.pnl > 0:
            exit_reasons[r]['wins'] += 1

    print(f"  {'Reason':<22} {'Count':>6} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*22} {'-'*6} {'-'*8} {'-'*12} {'-'*10}")
    for reason, stats in sorted(exit_reasons.items(), key=lambda x: -x[1]['count']):
        wr = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
        avg_pnl = stats['pnl'] / stats['count']
        print(f"  {reason:<22} {stats['count']:>6} {wr:>7.1f}% ${stats['pnl']:>11,.0f} ${avg_pnl:>9,.0f}")

    # ---- Signal Contribution Analysis ----
    print(f"\n{'SIGNAL CONTRIBUTION ANALYSIS':=^70}")
    if signal_scores_at_entry:
        sig_df = pd.DataFrame(signal_scores_at_entry)
        # Match with trade outcomes
        entry_outcomes = []
        for _, row in sig_df.iterrows():
            matching = [t for t in trades if t.code == row['code']
                       and abs((t.entry_date - row['date']).total_seconds()) < 86400]
            if matching:
                entry_outcomes.append({
                    'r1': row['r1'], 'r2': row['r2'], 'r3': row['r3'],
                    'r4': row['r4'], 'r5': row['r5'],
                    'composite': row['composite'],
                    'pnl_pct': matching[0].pnl_pct,
                    'win': 1 if matching[0].pnl > 0 else 0
                })

        if entry_outcomes:
            out_df = pd.DataFrame(entry_outcomes)
            signal_names = ['5d Momentum', '20d Momentum', 'Volume Breakout', 'Rel Strength', 'Trend Quality']

            print(f"  {'Signal':<18} {'Corr w/ Return':>14} {'Avg Rank (Win)':>14} {'Avg Rank (Loss)':>15}")
            print(f"  {'-'*18} {'-'*14} {'-'*14} {'-'*15}")
            for i, (col, name) in enumerate(zip(['r1', 'r2', 'r3', 'r4', 'r5'], signal_names)):
                corr = out_df[col].corr(out_df['pnl_pct'])
                if pd.isna(corr):
                    corr = 0.0
                avg_rank_win = out_df[out_df['win'] == 1][col].mean() if out_df['win'].sum() > 0 else 0
                avg_rank_loss = out_df[out_df['win'] == 0][col].mean() if (out_df['win'] == 0).sum() > 0 else 0
                print(f"  {name:<18} {corr:>14.4f} {avg_rank_win:>14.1f} {avg_rank_loss:>15.1f}")

            # Overall composite correlation
            comp_corr = out_df['composite'].corr(out_df['pnl_pct'])
            if pd.isna(comp_corr):
                comp_corr = 0.0
            print(f"\n  Composite score correlation with trade return: {comp_corr:.4f}")
            print(f"  Total entries with signal data: {len(out_df)}")
            print(f"  Avg composite (winners): {out_df[out_df['win']==1]['composite'].mean():.1f}" if out_df['win'].sum() > 0 else "")
            print(f"  Avg composite (losers):  {out_df[out_df['win']==0]['composite'].mean():.1f}" if (out_df['win']==0).sum() > 0 else "")
    else:
        print("  No signal data recorded.")

    # ---- Top Traded ETFs ----
    print(f"\n{'TOP TRADED ETFs':=^70}")
    etf_trade_stats = {}
    for t in trades:
        if t.code not in etf_trade_stats:
            etf_trade_stats[t.code] = {'count': 0, 'pnl': 0, 'wins': 0}
        etf_trade_stats[t.code]['count'] += 1
        etf_trade_stats[t.code]['pnl'] += t.pnl
        if t.pnl > 0:
            etf_trade_stats[t.code]['wins'] += 1

    print(f"  {'Code':<10} {'Trades':>7} {'WinRate':>8} {'TotalPnL':>12} {'AvgPnL':>10}")
    print(f"  {'-'*10} {'-'*7} {'-'*8} {'-'*12} {'-'*10}")
    for code, stats in sorted(etf_trade_stats.items(), key=lambda x: -x[1]['pnl'])[:15]:
        wr = stats['wins'] / stats['count'] * 100
        avg = stats['pnl'] / stats['count']
        print(f"  {code:<10} {stats['count']:>7} {wr:>7.1f}% ${stats['pnl']:>11,.0f} ${avg:>9,.0f}")

    # ---- Monthly Returns Heatmap (text) ----
    print(f"\n{'MONTHLY RETURNS (%)':=^70}")
    print(f"  {'':>6}", end="")
    for m in range(1, 13):
        print(f"  {m:>5}", end="")
    print()

    for year in range(2021, 2026):
        print(f"  {year:>4}  ", end="")
        for month in range(1, 13):
            month_mask = np.array([d.year == year and d.month == month for d in all_dates])
            if not month_mask.any():
                print(f"  {'--':>5}", end="")
                continue
            month_eq = equity_arr[month_mask]
            if len(month_eq) < 2:
                print(f"  {'--':>5}", end="")
                continue
            m_ret = (month_eq[-1] / month_eq[0] - 1) * 100
            print(f"  {m_ret:>5.1f}", end="")
        print()

    # ---- Summary Statistics ----
    print(f"\n{'RISK-ADJUSTED METRICS':=^70}")
    # Max consecutive wins/losses
    if trades:
        results = [1 if t.pnl > 0 else 0 for t in trades]
        max_consec_win = max_consec_loss = 0
        cur_win = cur_loss = 0
        for r in results:
            if r == 1:
                cur_win += 1
                cur_loss = 0
            else:
                cur_loss += 1
                cur_win = 0
            max_consec_win = max(max_consec_win, cur_win)
            max_consec_loss = max(max_consec_loss, cur_loss)

        print(f"  Max Consecutive Wins:   {max_consec_win}")
        print(f"  Max Consecutive Losses: {max_consec_loss}")
        print(f"  Best Trade:             {max(t.pnl_pct for t in trades)*100:.2f}%")
        print(f"  Worst Trade:            {min(t.pnl_pct for t in trades)*100:.2f}%")
        print(f"  Avg Trade Duration:     {avg_hold:.1f} days")

    # Time in market
    days_with_positions = sum(1 for i in range(warmup_days, len(all_dates))
                             if i < len(equity_values) and i > 0
                             and abs(equity_values[i] - equity_values[i-1]) > 0.01)
    # Simplified: estimate from trades
    total_position_days = sum(t.holding_days for t in trades)
    time_in_market = total_position_days / len(all_dates) * 100
    print(f"  Time in Market:         {time_in_market:.1f}%")
    print(f"  Exposure-adjusted Ret:  {annual_return / (time_in_market/100) * 100:.2f}% (if {time_in_market:.0f}% exposed)" if time_in_market > 0 else "")

    print(f"\n{'=' * 70}")
    print("Backtest complete.")

    return equity_arr, bench_arr, trades


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    run_backtest()

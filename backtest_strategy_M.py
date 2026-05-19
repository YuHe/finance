"""
Backtest: Concentrated Breakout with Multi-Condition Confirmation ETF Rotation Strategy
Database: data_layer/backtest_fixed.db
Period: 2021-05-14 to 2025-12-31
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ─── Configuration ───────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "data_layer" / "backtest_fixed.db"
BENCHMARK_CODE = "510300"
FEE_RATE = 0.0005  # 0.05% per side
POSITION_SIZE = 0.80  # 80% of equity
REDUCED_POSITION_SIZE = 0.40  # after 3% drawdown
CASH_BUFFER = 0.20

# Entry pre-conditions
BREADTH_THRESHOLD = 0.50
BREADTH_EXIT_THRESHOLD = 0.35
BENCH_MA_SHORT = 20
BENCH_MA_LONG = 60
BENCH_RETURN_DAYS = 5

# ETF selection
TOP_N_MOMENTUM = 5
BREAKOUT_PERIOD = 20
VOLUME_SURGE_RATIO = 1.5
VOLUME_SHORT = 5
VOLUME_LONG = 20
MA_5 = 5
MA_10 = 10
MA_20 = 20
RETURN_PERIOD = 10

# Exit parameters
HARD_STOP_PCT = -0.03
ATR_TRAIL_TRIGGER = 0.04
ATR_TRAIL_MULT = 1.8
ATR_PERIOD = 10
MA_EXIT_DAYS = 2  # consecutive closes below 5d MA
PROFIT_TARGET = 0.20
MAX_HOLDING_DAYS = 15

# Portfolio circuit breaker
CIRCUIT_BREAKER_FULL = 0.05  # 5% from peak → all cash
CIRCUIT_BREAKER_REDUCE = 0.03  # 3% from peak → reduce
CIRCUIT_BREAKER_WAIT = 10  # trading days

# Cooldown
COOLDOWN_DAYS = 2

# Top N to hold
TOP_HOLDINGS = 1


# ─── Data Loading ────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(str(DB_PATH))

    # Load ETF data
    df = pd.read_sql_query(
        "SELECT code, date, open, high, low, close, volume FROM etf_daily ORDER BY code, date",
        conn,
    )
    df["date"] = pd.to_datetime(df["date"])

    # Load northbound data
    nb = pd.read_sql_query(
        "SELECT date, total_deal FROM northbound_deal ORDER BY date", conn
    )
    nb["date"] = pd.to_datetime(nb["date"])
    nb = nb.set_index("date")["total_deal"]

    conn.close()
    return df, nb


def compute_indicators(df):
    """Compute all technical indicators per ETF."""
    results = {}
    codes = df["code"].unique()

    for code in codes:
        etf = df[df["code"] == code].copy().reset_index(drop=True)
        etf = etf.set_index("date").sort_index()

        etf["ma5"] = etf["close"].rolling(5).mean()
        etf["ma10"] = etf["close"].rolling(10).mean()
        etf["ma20"] = etf["close"].rolling(20).mean()
        etf["ma60"] = etf["close"].rolling(60).mean()
        etf["ret_5d"] = etf["close"].pct_change(5)
        etf["ret_10d"] = etf["close"].pct_change(10)
        etf["high_20d"] = etf["high"].rolling(20).max()
        etf["vol_avg_5"] = etf["volume"].rolling(5).mean()
        etf["vol_avg_20"] = etf["volume"].rolling(20).mean()
        etf["volatility_20d"] = etf["close"].pct_change().rolling(20).std()

        # ATR
        tr = pd.DataFrame(index=etf.index)
        tr["hl"] = etf["high"] - etf["low"]
        tr["hc"] = (etf["high"] - etf["close"].shift(1)).abs()
        tr["lc"] = (etf["low"] - etf["close"].shift(1)).abs()
        etf["tr"] = tr.max(axis=1)
        etf["atr10"] = etf["tr"].rolling(ATR_PERIOD).mean()

        results[code] = etf

    return results


# ─── Backtest Engine ─────────────────────────────────────────────────────────
@dataclass
class Trade:
    code: str
    entry_date: object
    entry_price: float
    shares: float
    position_value: float
    exit_date: object = None
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    holding_days: int = 0


@dataclass
class Position:
    code: str
    entry_date: object
    entry_price: float
    shares: float
    highest_close: float = 0.0
    trailing_stop: float = 0.0
    trail_active: bool = False
    days_below_ma5: int = 0
    holding_days: int = 0


def run_backtest():
    print("Loading data...")
    df, nb_data = load_data()
    print("Computing indicators...")
    indicators = compute_indicators(df)

    # Get tradeable codes (exclude benchmark)
    trade_codes = [c for c in indicators.keys() if c != BENCHMARK_CODE]
    bench = indicators[BENCHMARK_CODE]

    # Get all trading dates from benchmark
    all_dates = bench.index.tolist()
    print(f"Trading days: {len(all_dates)}, from {all_dates[0].date()} to {all_dates[-1].date()}")

    # Northbound indicators
    nb_ma5 = nb_data.rolling(5).mean()
    nb_ma20 = nb_data.rolling(20).mean()

    # State variables
    equity = 1_000_000.0
    cash = equity
    peak_equity = equity
    position: Optional[Position] = None
    trades: list[Trade] = []
    equity_curve = []
    cooldown_remaining = 0
    circuit_breaker_wait = 0
    reduced_mode = False

    # Condition blocking counters
    block_counts = {
        "breadth": 0,
        "bench_ma20": 0,
        "bench_ma60": 0,
        "bench_5d_ret": 0,
        "northbound": 0,
        "no_candidates": 0,
        "cooldown": 0,
        "circuit_breaker": 0,
        "already_holding": 0,
    }

    # We need at least 60 days of warmup for MA60
    warmup = 60

    for i, date in enumerate(all_dates):
        # Record equity at start of day (before any action)
        if position is not None:
            code = position.code
            if date in indicators[code].index:
                current_close = indicators[code].loc[date, "close"]
                pos_value = position.shares * current_close
                daily_equity = cash + pos_value
            else:
                daily_equity = cash + position.shares * position.entry_price
        else:
            daily_equity = cash

        equity = daily_equity
        equity_curve.append({"date": date, "equity": equity})

        if i < warmup:
            continue

        # Update peak and check circuit breaker
        if equity > peak_equity:
            peak_equity = equity

        drawdown_from_peak = (peak_equity - equity) / peak_equity

        if circuit_breaker_wait > 0:
            circuit_breaker_wait -= 1
            block_counts["circuit_breaker"] += 1
            # If in circuit breaker and holding, we should have already exited
            continue

        if drawdown_from_peak >= CIRCUIT_BREAKER_FULL:
            # Exit everything and wait
            if position is not None:
                exit_price = indicators[position.code].loc[date, "close"]
                pnl = (exit_price - position.entry_price) * position.shares
                fee = exit_price * position.shares * FEE_RATE
                pnl -= fee
                cash += position.shares * exit_price - fee
                t = Trade(
                    code=position.code,
                    entry_date=position.entry_date,
                    entry_price=position.entry_price,
                    shares=position.shares,
                    position_value=position.shares * position.entry_price,
                    exit_date=date,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_pct=(exit_price / position.entry_price - 1),
                    exit_reason="circuit_breaker_5pct",
                    holding_days=position.holding_days,
                )
                trades.append(t)
                position = None
            circuit_breaker_wait = CIRCUIT_BREAKER_WAIT
            reduced_mode = False
            continue
        elif drawdown_from_peak >= CIRCUIT_BREAKER_REDUCE:
            reduced_mode = True
        else:
            reduced_mode = False

        # ─── Exit checks (if holding) ─────────────────────────────────────
        if position is not None:
            code = position.code
            if date not in indicators[code].index:
                continue

            etf = indicators[code]
            current_close = etf.loc[date, "close"]
            current_open = etf.loc[date, "open"]
            position.holding_days += 1

            # Update highest close
            if current_close > position.highest_close:
                position.highest_close = current_close

            exit_reason = ""
            exit_price = current_close

            # Check exit conditions
            gain_pct = (current_close - position.entry_price) / position.entry_price

            # 1. Hard stop: -3% from entry
            if gain_pct <= HARD_STOP_PCT:
                exit_reason = "hard_stop_3pct"

            # 2. ATR trailing stop
            if not exit_reason and gain_pct > ATR_TRAIL_TRIGGER:
                atr_val = etf.loc[date, "atr10"]
                if not np.isnan(atr_val):
                    new_trail = position.highest_close - ATR_TRAIL_MULT * atr_val
                    if not position.trail_active or new_trail > position.trailing_stop:
                        position.trailing_stop = new_trail
                        position.trail_active = True
            if not exit_reason and position.trail_active and current_close < position.trailing_stop:
                exit_reason = "atr_trailing_stop"

            # 3. Break below 5d MA for 2 consecutive closes
            if not exit_reason:
                ma5_val = etf.loc[date, "ma5"]
                if not np.isnan(ma5_val):
                    if current_close < ma5_val:
                        position.days_below_ma5 += 1
                    else:
                        position.days_below_ma5 = 0
                    if position.days_below_ma5 >= MA_EXIT_DAYS:
                        exit_reason = "below_ma5_2days"

            # 4. Market breadth exit
            if not exit_reason:
                breadth = compute_breadth(indicators, trade_codes, date)
                if breadth is not None and breadth < BREADTH_EXIT_THRESHOLD:
                    exit_reason = "breadth_deterioration"

            # 5. Profit target
            if not exit_reason and gain_pct >= PROFIT_TARGET:
                exit_reason = "profit_target_20pct"

            # 6. Time stop
            if not exit_reason and position.holding_days >= MAX_HOLDING_DAYS:
                exit_reason = "time_stop_15d"

            # Execute exit
            if exit_reason:
                fee = exit_price * position.shares * FEE_RATE
                pnl = (exit_price - position.entry_price) * position.shares - fee
                cash += position.shares * exit_price - fee
                t = Trade(
                    code=position.code,
                    entry_date=position.entry_date,
                    entry_price=position.entry_price,
                    shares=position.shares,
                    position_value=position.shares * position.entry_price,
                    exit_date=date,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_pct=(exit_price / position.entry_price - 1) - 2 * FEE_RATE,
                    exit_reason=exit_reason,
                    holding_days=position.holding_days,
                )
                trades.append(t)
                position = None
                cooldown_remaining = COOLDOWN_DAYS
                continue

            # Check if position needs to be reduced due to circuit breaker
            if reduced_mode:
                target_value = equity * REDUCED_POSITION_SIZE
                current_value = position.shares * current_close
                if current_value > target_value * 1.05:
                    # Sell excess
                    sell_shares = (current_value - target_value) / current_close
                    fee = sell_shares * current_close * FEE_RATE
                    cash += sell_shares * current_close - fee
                    position.shares -= sell_shares

        # ─── Entry logic (if not holding) ─────────────────────────────────
        if position is not None:
            block_counts["already_holding"] += 1
            continue

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            block_counts["cooldown"] += 1
            continue

        # Check pre-conditions
        # 1. Market breadth
        breadth = compute_breadth(indicators, trade_codes, date)
        if breadth is None or breadth <= BREADTH_THRESHOLD:
            block_counts["breadth"] += 1
            continue

        # 2. Benchmark close > 20d MA AND > 60d MA
        if date not in bench.index:
            continue
        bench_close = bench.loc[date, "close"]
        bench_ma20 = bench.loc[date, "ma20"]
        bench_ma60 = bench.loc[date, "ma60"]
        bench_ret5 = bench.loc[date, "ret_5d"]

        if np.isnan(bench_ma20) or bench_close <= bench_ma20:
            block_counts["bench_ma20"] += 1
            continue
        if np.isnan(bench_ma60) or bench_close <= bench_ma60:
            block_counts["bench_ma60"] += 1
            continue

        # 3. Benchmark 5d return > 0
        if np.isnan(bench_ret5) or bench_ret5 <= 0:
            block_counts["bench_5d_ret"] += 1
            continue

        # 4. Northbound check (optional)
        if date in nb_data.index:
            nb_today = nb_data.loc[date]
            nb_5 = nb_ma5.loc[date] if date in nb_ma5.index else np.nan
            nb_20 = nb_ma20.loc[date] if date in nb_ma20.index else np.nan
            if not np.isnan(nb_5) and not np.isnan(nb_20):
                if not (nb_today > nb_5 and nb_today > nb_20):
                    block_counts["northbound"] += 1
                    continue

        # ─── ETF Selection ────────────────────────────────────────────────
        # Condition A: 10d return in top 5
        returns_10d = {}
        for code in trade_codes:
            if date in indicators[code].index:
                ret = indicators[code].loc[date, "ret_10d"]
                if not np.isnan(ret):
                    returns_10d[code] = ret

        if len(returns_10d) < TOP_N_MOMENTUM:
            block_counts["no_candidates"] += 1
            continue

        sorted_by_ret = sorted(returns_10d.items(), key=lambda x: x[1], reverse=True)
        top5_codes = set(c for c, _ in sorted_by_ret[:TOP_N_MOMENTUM])

        # Check all 4 conditions for each candidate
        candidates = []
        for code in top5_codes:
            etf = indicators[code]
            if date not in etf.index:
                continue

            row = etf.loc[date]

            # Condition B: Close > 20-day high (of previous days, not including today)
            # Use the rolling high which includes today's high; for breakout we check close > prev 20d high
            # Actually high_20d includes today's high. We want close > max of previous 20 days' high.
            # Let's compute it properly: get the index position
            date_idx = etf.index.get_loc(date)
            if date_idx < 20:
                continue
            prev_20d_high = etf["high"].iloc[date_idx - 20 : date_idx].max()
            if row["close"] <= prev_20d_high:
                continue

            # Condition C: 5d avg volume > 1.5x 20d avg volume
            vol5 = row["vol_avg_5"]
            vol20 = row["vol_avg_20"]
            if np.isnan(vol5) or np.isnan(vol20) or vol20 == 0:
                continue
            if vol5 <= VOLUME_SURGE_RATIO * vol20:
                continue

            # Condition D: close > ma5 > ma10 > ma20 (perfect alignment)
            ma5_v = row["ma5"]
            ma10_v = row["ma10"]
            ma20_v = row["ma20"]
            if np.isnan(ma5_v) or np.isnan(ma10_v) or np.isnan(ma20_v):
                continue
            if not (row["close"] > ma5_v > ma10_v > ma20_v):
                continue

            # Score: 10d return / 20d volatility
            vol_20 = row["volatility_20d"]
            if np.isnan(vol_20) or vol_20 == 0:
                continue
            score = returns_10d[code] / vol_20
            candidates.append((code, score))

        if not candidates:
            block_counts["no_candidates"] += 1
            continue

        # Rank and select top 1
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_code = candidates[0][0]

        # ─── Execute Entry at T+1 Open ────────────────────────────────────
        # Find next trading day
        if i + 1 >= len(all_dates):
            continue
        next_date = all_dates[i + 1]
        etf_next = indicators[selected_code]
        if next_date not in etf_next.index:
            continue

        entry_price = etf_next.loc[next_date, "open"]
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        # Position sizing
        pos_pct = REDUCED_POSITION_SIZE if reduced_mode else POSITION_SIZE
        position_value = equity * pos_pct
        fee = position_value * FEE_RATE
        shares = (position_value - fee) / entry_price
        cash = equity - shares * entry_price - fee

        position = Position(
            code=selected_code,
            entry_date=next_date,
            entry_price=entry_price,
            shares=shares,
            highest_close=entry_price,
            trailing_stop=0.0,
            trail_active=False,
            days_below_ma5=0,
            holding_days=0,
        )

    # ─── Final equity ─────────────────────────────────────────────────────
    # Close any open position at last day's close
    if position is not None:
        last_date = all_dates[-1]
        if last_date in indicators[position.code].index:
            exit_price = indicators[position.code].loc[last_date, "close"]
            fee = exit_price * position.shares * FEE_RATE
            pnl = (exit_price - position.entry_price) * position.shares - fee
            cash += position.shares * exit_price - fee
            t = Trade(
                code=position.code,
                entry_date=position.entry_date,
                entry_price=position.entry_price,
                shares=position.shares,
                position_value=position.shares * position.entry_price,
                exit_date=last_date,
                exit_price=exit_price,
                pnl=pnl,
                pnl_pct=(exit_price / position.entry_price - 1) - 2 * FEE_RATE,
                exit_reason="end_of_backtest",
                holding_days=position.holding_days,
            )
            trades.append(t)
            position = None
        equity = cash

    return equity_curve, trades, block_counts, indicators, all_dates


def compute_breadth(indicators, trade_codes, date):
    """Compute market breadth: % of ETFs with close > 20d MA."""
    above = 0
    total = 0
    for code in trade_codes:
        etf = indicators[code]
        if date in etf.index:
            row = etf.loc[date]
            if not np.isnan(row["ma20"]):
                total += 1
                if row["close"] > row["ma20"]:
                    above += 1
    if total == 0:
        return None
    return above / total


# ─── Analytics ───────────────────────────────────────────────────────────────
def print_results(equity_curve, trades, block_counts, indicators, all_dates):
    ec = pd.DataFrame(equity_curve)
    ec = ec.set_index("date")
    initial_equity = ec["equity"].iloc[0]
    final_equity = ec["equity"].iloc[-1]

    # Total return
    total_return = (final_equity / initial_equity) - 1

    # Annual return (CAGR)
    days = (ec.index[-1] - ec.index[0]).days
    years = days / 365.25
    cagr = (final_equity / initial_equity) ** (1 / years) - 1

    # Max drawdown
    rolling_max = ec["equity"].cummax()
    drawdown = (ec["equity"] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    # Sharpe ratio (annualized, risk-free = 0)
    daily_returns = ec["equity"].pct_change().dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trade statistics
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / n_trades
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
        avg_win_loss = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        total_holding_days = sum(t.holding_days for t in trades)
        time_in_market = total_holding_days / len(all_dates) * 100
        weeks = days / 7
        trades_per_week = n_trades / weeks
    else:
        win_rate = avg_win = avg_loss = avg_win_loss = 0
        profit_factor = 0
        time_in_market = 0
        trades_per_week = 0

    # Benchmark return
    bench = indicators[BENCHMARK_CODE]
    bench_start = bench["close"].iloc[60]  # after warmup
    bench_end = bench["close"].iloc[-1]
    bench_return = (bench_end / bench_start) - 1

    print("\n" + "=" * 80)
    print("  CONCENTRATED BREAKOUT WITH MULTI-CONDITION CONFIRMATION - BACKTEST RESULTS")
    print("=" * 80)

    print(f"\n{'─' * 40}")
    print("  PERFORMANCE SUMMARY")
    print(f"{'─' * 40}")
    print(f"  Period:              {ec.index[0].date()} to {ec.index[-1].date()} ({days} days)")
    print(f"  Initial Equity:      ${initial_equity:,.0f}")
    print(f"  Final Equity:        ${final_equity:,.0f}")
    print(f"  Total Return:        {total_return * 100:.2f}%")
    print(f"  CAGR:                {cagr * 100:.2f}%")
    print(f"  Benchmark Return:    {bench_return * 100:.2f}% (510300)")
    print(f"  Max Drawdown:        {max_dd * 100:.2f}% (on {max_dd_date.date()})")
    print(f"  Sharpe Ratio:        {sharpe:.3f}")
    print(f"  Calmar Ratio:        {calmar:.3f}")

    print(f"\n{'─' * 40}")
    print("  TRADE STATISTICS")
    print(f"{'─' * 40}")
    print(f"  Total Trades:        {n_trades}")
    print(f"  Win Rate:            {win_rate * 100:.1f}%")
    print(f"  Avg Win:             {avg_win * 100:.2f}%")
    print(f"  Avg Loss:            {avg_loss * 100:.2f}%")
    print(f"  Avg Win / Avg Loss:  {avg_win_loss:.2f}")
    print(f"  Profit Factor:       {profit_factor:.2f}")
    print(f"  Trades per Week:     {trades_per_week:.3f}")
    print(f"  Time in Market:      {time_in_market:.1f}%")
    if n_trades > 0:
        print(f"  Avg Holding Days:    {np.mean([t.holding_days for t in trades]):.1f}")
        print(f"  Max Holding Days:    {max(t.holding_days for t in trades)}")

    # ─── Exit Reason Breakdown ────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  EXIT REASON BREAKDOWN")
    print(f"{'─' * 40}")
    if trades:
        reasons = {}
        for t in trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            avg_pnl = np.mean([t.pnl_pct for t in trades if t.exit_reason == reason])
            print(f"  {reason:<30s} {count:>3d} trades  avg pnl: {avg_pnl * 100:>+6.2f}%")

    # ─── Condition Blocking Analysis ──────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  CONDITION BLOCKING ANALYSIS (days each condition blocked entry)")
    print(f"{'─' * 40}")
    for cond, count in sorted(block_counts.items(), key=lambda x: -x[1]):
        print(f"  {cond:<25s} {count:>5d} days")

    # ─── Yearly Breakdown ─────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  YEARLY BREAKDOWN")
    print(f"{'─' * 40}")
    ec["year"] = ec.index.year
    years_list = sorted(ec["year"].unique())
    print(f"  {'Year':<6s} {'Return':>10s} {'MaxDD':>10s} {'Sharpe':>8s} {'Trades':>8s}")
    print(f"  {'----':<6s} {'------':>10s} {'-----':>10s} {'------':>8s} {'------':>8s}")

    for year in years_list:
        year_data = ec[ec["year"] == year]["equity"]
        if len(year_data) < 2:
            continue
        yr_ret = (year_data.iloc[-1] / year_data.iloc[0]) - 1
        yr_rm = year_data.cummax()
        yr_dd = ((year_data - yr_rm) / yr_rm).min()
        yr_daily = year_data.pct_change().dropna()
        yr_sharpe = yr_daily.mean() / yr_daily.std() * np.sqrt(252) if yr_daily.std() > 0 else 0
        yr_trades = len([t for t in trades if t.entry_date.year == year])
        print(f"  {year:<6d} {yr_ret * 100:>+9.2f}% {yr_dd * 100:>+9.2f}% {yr_sharpe:>8.3f} {yr_trades:>8d}")

    # ─── Monthly Equity Curve ─────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  MONTHLY EQUITY CURVE")
    print(f"{'─' * 40}")
    monthly = ec["equity"].resample("M").last()
    print(f"  {'Month':<12s} {'Equity':>14s} {'Monthly Ret':>12s} {'Cum Ret':>10s}")
    print(f"  {'-----':<12s} {'------':>14s} {'-----------':>12s} {'-------':>10s}")
    prev_eq = initial_equity
    for date_m, eq_val in monthly.items():
        m_ret = (eq_val / prev_eq) - 1
        c_ret = (eq_val / initial_equity) - 1
        print(f"  {date_m.strftime('%Y-%m'):<12s} ${eq_val:>12,.0f} {m_ret * 100:>+10.2f}% {c_ret * 100:>+8.2f}%")
        prev_eq = eq_val

    # ─── All Trades Detail ────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  ALL TRADES")
    print(f"{'─' * 40}")
    if trades:
        print(f"  {'#':<4s} {'Code':<8s} {'Entry Date':<12s} {'Exit Date':<12s} {'Entry$':>8s} {'Exit$':>8s} {'PnL%':>8s} {'Days':>5s} {'Reason':<25s}")
        print(f"  {'─' * 100}")
        for i, t in enumerate(trades, 1):
            print(
                f"  {i:<4d} {t.code:<8s} {t.entry_date.strftime('%Y-%m-%d'):<12s} "
                f"{t.exit_date.strftime('%Y-%m-%d'):<12s} {t.entry_price:>8.4f} "
                f"{t.exit_price:>8.4f} {t.pnl_pct * 100:>+7.2f}% {t.holding_days:>5d} {t.exit_reason:<25s}"
            )

    print("\n" + "=" * 80)
    print("  BACKTEST COMPLETE")
    print("=" * 80)


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    equity_curve, trades, block_counts, indicators, all_dates = run_backtest()
    print_results(equity_curve, trades, block_counts, indicators, all_dates)

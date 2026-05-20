"""
Strategy W: Multi-Timeframe Trend Ensemble (低过拟合)
Inspired by AQR/Man Group trend-following research.

Key design:
- 4 timeframe EMA trend signals averaged → trend_score
- Entry: trend_score >= 0.75, benchmark positive, breadth > 40%
- Exit: trend_score < 0.25, 2.5x ATR trailing stop, benchmark turning
- Position sizing: 70-100% based on agreement level
- Only 3 real parameters: entry threshold, exit threshold, ATR multiplier
- Rebalance every 5 days
"""

import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DB_PATH = "data_layer/backtest_fixed.db"
BENCHMARK = "510300"
FEE = 0.0005  # per side
ENTRY_THRESHOLD = 0.75
EXIT_THRESHOLD = 0.25
ATR_MULTIPLIER = 2.5
REBALANCE_INTERVAL = 5
WARMUP_DAYS = 65
INITIAL_CAPITAL = 1_000_000


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT code, date, open, high, low, close, volume FROM etf_daily ORDER BY code, date", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators for a single ETF."""
    df = df.sort_values("date").reset_index(drop=True)

    # EMAs
    for span in [5, 10, 20, 40, 60]:
        df[f"ema{span}"] = compute_ema(df["close"], span)

    # Timeframe signals
    df["tf1"] = np.sign(df["ema5"] - df["ema10"])
    df["tf2"] = np.sign(df["ema10"] - df["ema20"])
    df["tf3"] = np.sign(df["ema20"] - df["ema40"])
    df["tf4"] = np.sign(df["ema40"] - df["ema60"])

    # Trend score
    df["trend_score"] = (df["tf1"] + df["tf2"] + df["tf3"] + df["tf4"]) / 4.0

    # ATR
    df["atr14"] = compute_atr(df["high"], df["low"], df["close"], 14)

    # 10-day return (momentum)
    df["r10"] = df["close"].pct_change(10)

    return df


# ─── TRADE TRACKING ───────────────────────────────────────────────────────────
@dataclass
class Trade:
    code: str
    entry_date: str
    entry_price: float
    entry_score: float
    shares: int
    position_pct: float  # 0.7 or 1.0
    trail_high: float = 0.0
    trail_stop: float = 0.0
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────
def run_backtest():
    print("=" * 80)
    print("Strategy W: Multi-Timeframe Trend Ensemble")
    print("=" * 80)

    # Load and prepare data
    raw = load_data()
    codes = sorted(raw[raw["code"] != BENCHMARK]["code"].unique())
    print(f"\nUniverse: {len(codes)} ETFs + benchmark {BENCHMARK}")
    print(f"Date range: {raw['date'].min().date()} to {raw['date'].max().date()}")

    # Compute indicators per ETF
    etf_data = {}
    for code in codes + [BENCHMARK]:
        sub = raw[raw["code"] == code].copy()
        if len(sub) < WARMUP_DAYS:
            continue
        sub = compute_indicators(sub)
        etf_data[code] = sub.set_index("date")

    # Get trading dates (benchmark dates after warmup)
    bench = etf_data[BENCHMARK]
    all_dates = bench.index.tolist()
    trading_dates = all_dates[WARMUP_DAYS:]
    print(f"Trading period: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)")

    # State
    capital = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    position: Optional[Trade] = None
    trades: list[Trade] = []
    equity_curve = []
    days_since_rebalance = 0

    for i, today in enumerate(trading_dates):
        # We use T-1 signals, execute at T open
        # today = execution day, signals from previous day
        prev_idx = all_dates.index(today) - 1
        prev_date = all_dates[prev_idx]

        # Get today's open prices for execution
        # Get previous day's signals for decision

        # ── BENCHMARK SIGNALS (T-1) ──
        if prev_date not in bench.index:
            # Mark to market and continue
            nav = cash
            if position:
                if today in etf_data[position.code].index:
                    nav += position.shares * etf_data[position.code].loc[today, "close"]
            equity_curve.append({"date": today, "equity": nav})
            continue

        bench_prev = bench.loc[prev_date]
        bench_trend_score = bench_prev["trend_score"]

        # ── BREADTH (T-1) ──
        breadth_count = 0
        breadth_total = 0
        for code in codes:
            if code not in etf_data:
                continue
            if prev_date not in etf_data[code].index:
                continue
            breadth_total += 1
            row = etf_data[code].loc[prev_date]
            if row["ema20"] > row["ema40"]:
                breadth_count += 1
        breadth = breadth_count / breadth_total if breadth_total > 0 else 0

        # ── POSITION MANAGEMENT ──
        if position:
            code = position.code
            if today not in etf_data[code].index:
                # Can't trade this ETF today, just hold
                nav = cash
                if position:
                    # Use last known close
                    nav += position.shares * position.entry_price  # approximate
                equity_curve.append({"date": today, "equity": nav})
                days_since_rebalance += 1
                continue

            today_data = etf_data[code].loc[today]
            today_open = today_data["open"]
            today_close = today_data["close"]
            today_high = today_data["high"]
            today_low = today_data["low"]

            # Get T-1 indicators for the held ETF
            if prev_date in etf_data[code].index:
                prev_data = etf_data[code].loc[prev_date]
                etf_trend_score = prev_data["trend_score"]
                etf_atr = prev_data["atr14"]
            else:
                etf_trend_score = 0
                etf_atr = 0

            # Update trailing stop
            position.trail_high = max(position.trail_high, today_high)
            if etf_atr > 0:
                position.trail_stop = position.trail_high - ATR_MULTIPLIER * etf_atr

            # Check exit conditions (signal on T-1, execute at T open)
            exit_reason = None

            # 1. Trend score exit
            if etf_trend_score < EXIT_THRESHOLD:
                exit_reason = "trend_score_exit"

            # 2. Benchmark turning
            elif bench_trend_score < -0.25:
                exit_reason = "benchmark_exit"

            # 3. Trailing stop (intraday check on today's low)
            elif position.trail_stop > 0 and today_low <= position.trail_stop:
                exit_reason = "trailing_stop"

            if exit_reason:
                # Exit at today's open (signal-based) or trail_stop (stop-based)
                if exit_reason == "trailing_stop":
                    exit_price = min(today_open, position.trail_stop)  # gap down
                else:
                    exit_price = today_open

                proceeds = position.shares * exit_price * (1 - FEE)
                cash += proceeds

                position.exit_date = str(today.date())
                position.exit_price = exit_price
                position.exit_reason = exit_reason
                position.pnl = proceeds - position.shares * position.entry_price * (1 + FEE)
                position.pnl_pct = (exit_price * (1 - FEE)) / (position.entry_price * (1 + FEE)) - 1
                entry_dt = pd.Timestamp(position.entry_date)
                position.holding_days = (today - entry_dt).days

                trades.append(position)
                position = None
                days_since_rebalance = 0

        # ── ENTRY / REBALANCE LOGIC ──
        days_since_rebalance += 1

        if position is None or (days_since_rebalance >= REBALANCE_INTERVAL and position is not None):
            # Check if we should swap positions (rebalance)
            if position is not None and days_since_rebalance >= REBALANCE_INTERVAL:
                # Find best candidate, swap if significantly better
                candidates = []
                for code in codes:
                    if code not in etf_data:
                        continue
                    if prev_date not in etf_data[code].index:
                        continue
                    row = etf_data[code].loc[prev_date]
                    ts = row["trend_score"]
                    r10 = row["r10"]
                    if ts >= ENTRY_THRESHOLD and not np.isnan(r10):
                        candidates.append((code, ts, ts * r10))

                if candidates:
                    candidates.sort(key=lambda x: x[2], reverse=True)
                    best_code, best_score, best_rank = candidates[0]

                    # Only swap if the best candidate is different and significantly better
                    if best_code != position.code:
                        # Check current position's trend score
                        if prev_date in etf_data[position.code].index:
                            curr_ts = etf_data[position.code].loc[prev_date]["trend_score"]
                            curr_r10 = etf_data[position.code].loc[prev_date]["r10"]
                            curr_rank = curr_ts * curr_r10 if not np.isnan(curr_r10) else 0
                        else:
                            curr_rank = 0

                        # Swap if new candidate ranks > 50% better
                        if best_rank > curr_rank * 1.5 and best_rank > 0:
                            # Exit current
                            if today in etf_data[position.code].index:
                                exit_price = etf_data[position.code].loc[today, "open"]
                                proceeds = position.shares * exit_price * (1 - FEE)
                                cash += proceeds

                                position.exit_date = str(today.date())
                                position.exit_price = exit_price
                                position.exit_reason = "rebalance_swap"
                                position.pnl = proceeds - position.shares * position.entry_price * (1 + FEE)
                                position.pnl_pct = (exit_price * (1 - FEE)) / (position.entry_price * (1 + FEE)) - 1
                                entry_dt = pd.Timestamp(position.entry_date)
                                position.holding_days = (today - entry_dt).days
                                trades.append(position)
                                position = None

                days_since_rebalance = 0

            # Try to enter new position
            if position is None:
                # Check regime filters
                if bench_trend_score > 0 and breadth > 0.40:
                    # Find candidates
                    candidates = []
                    for code in codes:
                        if code not in etf_data:
                            continue
                        if prev_date not in etf_data[code].index:
                            continue
                        if today not in etf_data[code].index:
                            continue
                        row = etf_data[code].loc[prev_date]
                        ts = row["trend_score"]
                        r10 = row["r10"]
                        if ts >= ENTRY_THRESHOLD and not np.isnan(r10):
                            candidates.append((code, ts, ts * r10))

                    if candidates:
                        candidates.sort(key=lambda x: x[2], reverse=True)
                        best_code, best_score, _ = candidates[0]

                        # Position sizing
                        if best_score >= 1.0:
                            pos_pct = 1.0
                        else:
                            pos_pct = 0.7

                        # Benchmark bonus
                        if bench_trend_score >= 1.0:
                            pos_pct = min(1.0, pos_pct + 0.2)

                        # Execute at today's open
                        entry_price = etf_data[best_code].loc[today, "open"]
                        invest_amount = cash * pos_pct
                        shares = int(invest_amount / (entry_price * (1 + FEE)))

                        if shares > 0:
                            cost = shares * entry_price * (1 + FEE)
                            cash -= cost

                            position = Trade(
                                code=best_code,
                                entry_date=str(today.date()),
                                entry_price=entry_price,
                                entry_score=best_score,
                                shares=shares,
                                position_pct=pos_pct,
                                trail_high=etf_data[best_code].loc[today, "high"],
                            )
                            # Set initial trail stop
                            if prev_date in etf_data[best_code].index:
                                atr = etf_data[best_code].loc[prev_date, "atr14"]
                                position.trail_stop = position.trail_high - ATR_MULTIPLIER * atr

        # ── MARK TO MARKET ──
        nav = cash
        if position:
            code = position.code
            if today in etf_data[code].index:
                nav += position.shares * etf_data[code].loc[today, "close"]
            else:
                nav += position.shares * position.entry_price  # fallback
        equity_curve.append({"date": today, "equity": nav})

    # Close any open position at end
    if position:
        last_date = trading_dates[-1]
        code = position.code
        if last_date in etf_data[code].index:
            exit_price = etf_data[code].loc[last_date, "close"]
        else:
            exit_price = position.entry_price
        proceeds = position.shares * exit_price * (1 - FEE)
        cash += proceeds
        position.exit_date = str(last_date.date())
        position.exit_price = exit_price
        position.exit_reason = "end_of_backtest"
        position.pnl = proceeds - position.shares * position.entry_price * (1 + FEE)
        position.pnl_pct = (exit_price * (1 - FEE)) / (position.entry_price * (1 + FEE)) - 1
        entry_dt = pd.Timestamp(position.entry_date)
        position.holding_days = (last_date - entry_dt).days
        trades.append(position)
        position = None

    # ── RESULTS ───────────────────────────────────────────────────────────────
    eq = pd.DataFrame(equity_curve)
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date")
    eq["returns"] = eq["equity"].pct_change()

    # Benchmark equity
    bench_prices = bench.loc[bench.index.isin(eq.index), "close"]
    bench_ret = bench_prices.pct_change().dropna()

    # ── METRICS ───────────────────────────────────────────────────────────────
    total_return = eq["equity"].iloc[-1] / INITIAL_CAPITAL - 1
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1

    # Max drawdown
    rolling_max = eq["equity"].cummax()
    drawdown = eq["equity"] / rolling_max - 1
    max_dd = drawdown.min()
    max_dd_end = drawdown.idxmin()
    # Find start of max drawdown
    peak_before_dd = eq["equity"][:max_dd_end].idxmax()

    # Sharpe & Sortino
    daily_rf = 0.0
    excess = eq["returns"].dropna() - daily_rf
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    downside = excess[excess < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Benchmark metrics
    bench_total = bench_prices.iloc[-1] / bench_prices.iloc[0] - 1 if len(bench_prices) > 1 else 0
    bench_cagr = (1 + bench_total) ** (1 / years) - 1
    bench_rolling_max = bench_prices.cummax()
    bench_dd = bench_prices / bench_rolling_max - 1
    bench_max_dd = bench_dd.min()

    # Win rate
    if trades:
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        win_rate = len(winners) / len(trades)
        avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0
        profit_factor = (sum(t.pnl for t in winners) / abs(sum(t.pnl for t in losers))
                         if losers and sum(t.pnl for t in losers) != 0 else float('inf'))
        avg_hold = np.mean([t.holding_days for t in trades])
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_hold = 0

    # Exposure
    invested_days = sum(1 for _, row in eq.iterrows() if row["equity"] != cash)
    # Better: count days we had a position
    position_days = 0
    temp_pos = None
    for t in trades:
        position_days += t.holding_days

    exposure = position_days / days if days > 0 else 0

    print("\n" + "─" * 80)
    print("PERFORMANCE METRICS")
    print("─" * 80)
    print(f"  Total Return:        {total_return:>10.2%}")
    print(f"  CAGR:                {cagr:>10.2%}")
    print(f"  Max Drawdown:        {max_dd:>10.2%}")
    print(f"    DD period:         {peak_before_dd.date()} → {max_dd_end.date()}")
    print(f"  Sharpe Ratio:        {sharpe:>10.2f}")
    print(f"  Sortino Ratio:       {sortino:>10.2f}")
    print(f"  Calmar Ratio:        {calmar:>10.2f}")
    print(f"  Profit Factor:       {profit_factor:>10.2f}")
    print(f"  Exposure:            {exposure:>10.1%}")
    print(f"")
    print(f"  Total Trades:        {len(trades):>10d}")
    print(f"  Win Rate:            {win_rate:>10.2%}")
    print(f"  Avg Win:             {avg_win:>10.2%}")
    print(f"  Avg Loss:            {avg_loss:>10.2%}")
    print(f"  Avg Holding Days:    {avg_hold:>10.1f}")
    print(f"")
    print(f"  ── Benchmark ({BENCHMARK}) ──")
    print(f"  Benchmark Return:    {bench_total:>10.2%}")
    print(f"  Benchmark CAGR:      {bench_cagr:>10.2%}")
    print(f"  Benchmark Max DD:    {bench_max_dd:>10.2%}")
    print(f"  Excess Return:       {total_return - bench_total:>10.2%}")

    # ── YEARLY BREAKDOWN ──────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("YEARLY BREAKDOWN")
    print("─" * 80)
    print(f"  {'Year':<6} {'Return':>9} {'MaxDD':>9} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8} {'Bench':>9}")
    print(f"  {'─'*6} {'─'*9} {'─'*9} {'─'*8} {'─'*7} {'─'*8} {'─'*9}")

    for year in sorted(eq.index.year.unique()):
        yr_eq = eq[eq.index.year == year]
        yr_ret = yr_eq["equity"].iloc[-1] / yr_eq["equity"].iloc[0] - 1
        yr_rm = yr_eq["equity"].cummax()
        yr_dd = (yr_eq["equity"] / yr_rm - 1).min()
        yr_rets = yr_eq["returns"].dropna()
        yr_sharpe = yr_rets.mean() / yr_rets.std() * np.sqrt(252) if yr_rets.std() > 0 else 0
        yr_trades = [t for t in trades if t.entry_date.startswith(str(year))]
        yr_wins = [t for t in yr_trades if t.pnl > 0]
        yr_wr = len(yr_wins) / len(yr_trades) if yr_trades else 0

        # Benchmark yearly
        yr_bench = bench_prices[bench_prices.index.year == year]
        yr_bench_ret = yr_bench.iloc[-1] / yr_bench.iloc[0] - 1 if len(yr_bench) > 1 else 0

        print(f"  {year:<6} {yr_ret:>8.2%} {yr_dd:>9.2%} {yr_sharpe:>8.2f} {len(yr_trades):>7d} {yr_wr:>8.1%} {yr_bench_ret:>9.2%}")

    # ── EXIT REASON BREAKDOWN ─────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("EXIT REASON BREAKDOWN")
    print("─" * 80)
    exit_reasons = {}
    for t in trades:
        r = t.exit_reason or "unknown"
        if r not in exit_reasons:
            exit_reasons[r] = {"count": 0, "total_pnl": 0, "wins": 0}
        exit_reasons[r]["count"] += 1
        exit_reasons[r]["total_pnl"] += t.pnl
        if t.pnl > 0:
            exit_reasons[r]["wins"] += 1

    print(f"  {'Reason':<20} {'Count':>6} {'WinRate':>8} {'Total PnL':>12} {'Avg PnL%':>9}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*12} {'─'*9}")
    for reason, stats in sorted(exit_reasons.items(), key=lambda x: x[1]["count"], reverse=True):
        reason_trades = [t for t in trades if t.exit_reason == reason]
        avg_pnl_pct = np.mean([t.pnl_pct for t in reason_trades])
        wr = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {reason:<20} {stats['count']:>6d} {wr:>8.1%} {stats['total_pnl']:>12,.0f} {avg_pnl_pct:>8.2%}")

    # ── TIMEFRAME AGREEMENT DISTRIBUTION ──────────────────────────────────────
    print("\n" + "─" * 80)
    print("TIMEFRAME AGREEMENT DISTRIBUTION (at entry)")
    print("─" * 80)
    score_dist = {}
    for t in trades:
        s = t.entry_score
        if s not in score_dist:
            score_dist[s] = {"count": 0, "wins": 0, "total_pnl": 0}
        score_dist[s]["count"] += 1
        score_dist[s]["total_pnl"] += t.pnl
        if t.pnl > 0:
            score_dist[s]["wins"] += 1

    print(f"  {'Score':<8} {'Count':>6} {'% of Total':>10} {'WinRate':>8} {'Avg PnL%':>9}")
    print(f"  {'─'*8} {'─'*6} {'─'*10} {'─'*8} {'─'*9}")
    for score in sorted(score_dist.keys(), reverse=True):
        stats = score_dist[score]
        pct = stats["count"] / len(trades) * 100 if trades else 0
        wr = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        score_trades = [t for t in trades if t.entry_score == score]
        avg_pnl = np.mean([t.pnl_pct for t in score_trades])
        agreement = int(score * 4)
        label = f"{score:.2f} ({agreement}/4)"
        print(f"  {label:<8} {stats['count']:>6d} {pct:>9.1f}% {wr:>8.1%} {avg_pnl:>8.2%}")

    # ── HOLDING PERIOD DISTRIBUTION ───────────────────────────────────────────
    print("\n" + "─" * 80)
    print("HOLDING PERIOD DISTRIBUTION")
    print("─" * 80)
    if trades:
        hold_days = [t.holding_days for t in trades]
        bins = [(0, 5, "1-5 days"), (5, 10, "6-10 days"), (10, 20, "11-20 days"),
                (20, 40, "21-40 days"), (40, 80, "41-80 days"), (80, 9999, "80+ days")]
        print(f"  {'Period':<12} {'Count':>6} {'% of Total':>10} {'Avg PnL%':>9} {'WinRate':>8}")
        print(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*9} {'─'*8}")
        for lo, hi, label in bins:
            bucket = [t for t in trades if lo < t.holding_days <= hi]
            if bucket:
                pct = len(bucket) / len(trades) * 100
                avg_pnl = np.mean([t.pnl_pct for t in bucket])
                wr = len([t for t in bucket if t.pnl > 0]) / len(bucket)
                print(f"  {label:<12} {len(bucket):>6d} {pct:>9.1f}% {avg_pnl:>8.2%} {wr:>8.1%}")

        print(f"\n  Median holding: {np.median(hold_days):.0f} days")
        print(f"  Mean holding:   {np.mean(hold_days):.1f} days")
        print(f"  Max holding:    {max(hold_days)} days")
        print(f"  Min holding:    {min(hold_days)} days")

    # ── TOP/BOTTOM 5 TRADES ───────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("TOP 5 TRADES")
    print("─" * 80)
    sorted_trades = sorted(trades, key=lambda t: t.pnl_pct, reverse=True)
    print(f"  {'Code':<8} {'Entry':<12} {'Exit':<12} {'PnL%':>8} {'Days':>5} {'Score':>6} {'Reason':<18}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*8} {'─'*5} {'─'*6} {'─'*18}")
    for t in sorted_trades[:5]:
        print(f"  {t.code:<8} {t.entry_date:<12} {t.exit_date:<12} {t.pnl_pct:>7.2%} {t.holding_days:>5d} {t.entry_score:>5.2f} {t.exit_reason:<18}")

    print("\n" + "─" * 80)
    print("BOTTOM 5 TRADES")
    print("─" * 80)
    print(f"  {'Code':<8} {'Entry':<12} {'Exit':<12} {'PnL%':>8} {'Days':>5} {'Score':>6} {'Reason':<18}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*8} {'─'*5} {'─'*6} {'─'*18}")
    for t in sorted_trades[-5:]:
        print(f"  {t.code:<8} {t.entry_date:<12} {t.exit_date:<12} {t.pnl_pct:>7.2%} {t.holding_days:>5d} {t.entry_score:>5.2f} {t.exit_reason:<18}")

    # ── MONTHLY RETURNS ───────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("MONTHLY RETURNS")
    print("─" * 80)
    monthly = eq["equity"].resample("M").last().pct_change().dropna()
    print(f"  Best month:  {monthly.max():>8.2%}")
    print(f"  Worst month: {monthly.min():>8.2%}")
    print(f"  % positive:  {(monthly > 0).mean():>8.1%}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)

    return eq, trades


if __name__ == "__main__":
    run_backtest()

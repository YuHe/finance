"""
Strategy O: Momentum Crash Protection with Defensive Allocation
===============================================================
ETF rotation strategy focused on minimizing drawdowns (target <= 8% MaxDD)
through aggressive crash detection and risk-budgeted position sizing.

Core: Risk-adjusted momentum (10d Sharpe-like score), top 3 ETFs,
with multiple crash protection layers and ATR-based risk budgeting.
"""

import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────
DB_PATH = "data_layer/backtest_fixed.db"
BENCHMARK = "510300"
FEE_RATE = 0.0005  # 0.05% per side

# Strategy parameters
MOMENTUM_WINDOW = 10
TOP_N = 3
REBALANCE_DAYS = 5

# Entry conditions
BREADTH_ENTRY_THRESHOLD = 0.45
BENCH_MA_PERIOD = 20
BENCH_VOL_ENTRY_MAX = 0.30  # 10d annualized vol < 30%
ETF_MA_SHORT = 10
ETF_MA_LONG = 20
EQUITY_MA_PERIOD = 15

# Position sizing
RISK_BUDGET_PCT = 0.01  # 1% risk per trade
ATR_PERIOD = 10
ATR_MULTIPLIER = 2.0
MAX_POSITION_PCT = 0.40  # 40% max single position
MAX_EXPOSURE_PCT = 0.80  # 80% max total exposure (20% cash min)

# Exit rules
BREADTH_EXIT_THRESHOLD = 0.30
BENCH_VOL_EXIT_MAX = 0.35  # 10d annualized vol > 35% → exit
EQUITY_DD_EXIT = 0.05  # 5% equity drawdown → exit all
EQUITY_DD_WAIT = 7  # wait 7 days after equity DD exit

# Crash detection
CRASH_BREADTH_DROP = 0.15  # breadth dropped 0.15 in 3 days
CRASH_BENCH_DROP = 0.03  # benchmark dropped 3% in 1 day
CRASH_PORTFOLIO_DROP = 0.02  # portfolio lost 2% in 1 day
CRASH_STOPS_HIT = 3  # 3+ positions hit stop same day
CRASH_BENCH_VOL_5D = 0.40  # 5d annualized vol > 40%
CRASH_SIGNALS_NEEDED = 3  # 3 out of 5 signals → crash exit
CRASH_WAIT_DAYS = 10  # wait 10 days after crash exit

# Re-entry after crash
REENTRY_SIZE_FACTOR = 0.50  # start at 50% sizing
REENTRY_FAIL_WAIT = 5  # wait 5 more days if first trade hits stop

INITIAL_CAPITAL = 1_000_000.0


# ─── Data Structures ─────────────────────────────────────────────────────────
@dataclass
class Position:
    code: str
    entry_date: str
    entry_price: float
    shares: int
    weight: float  # target weight at entry
    stop_price: float  # hard stop at entry
    trailing_stop: float  # ATR trailing stop
    highest_close: float  # for trailing stop tracking
    risk_budget_used: float  # actual risk % used


@dataclass
class Trade:
    code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    weight: float
    risk_budget: float
    exit_reason: str


@dataclass
class CrashEvent:
    date: str
    signals_fired: list
    action: str


# ─── Load Data ───────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM etf_daily ORDER BY code, date", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def prepare_data(df):
    """Prepare price panels and indicators."""
    codes = sorted(df['code'].unique())
    trading_codes = [c for c in codes if c != BENCHMARK]

    close_panel = df.pivot(index='date', columns='code', values='close').sort_index()
    high_panel = df.pivot(index='date', columns='code', values='high').sort_index()
    low_panel = df.pivot(index='date', columns='code', values='low').sort_index()

    return close_panel, high_panel, low_panel, trading_codes


def compute_indicators(close_panel, high_panel, low_panel, trading_codes):
    """Compute all indicators needed for the strategy."""
    indicators = {}

    # Benchmark indicators
    bench_close = close_panel[BENCHMARK]
    indicators['bench_close'] = bench_close
    indicators['bench_ma20'] = bench_close.rolling(BENCH_MA_PERIOD).mean()
    bench_ret = bench_close.pct_change()
    indicators['bench_vol_10d'] = bench_ret.rolling(10).std() * np.sqrt(252)
    indicators['bench_vol_5d'] = bench_ret.rolling(5).std() * np.sqrt(252)
    indicators['bench_daily_ret'] = bench_ret

    # ETF indicators
    for code in trading_codes:
        c = close_panel[code]
        h = high_panel[code]
        l = low_panel[code]

        # Returns and vol for momentum scoring
        ret_10d = c.pct_change(MOMENTUM_WINDOW)
        vol_10d = c.pct_change().rolling(MOMENTUM_WINDOW).std()

        # MAs for entry filter
        ma_short = c.rolling(ETF_MA_SHORT).mean()
        ma_long = c.rolling(ETF_MA_LONG).mean()

        # ATR(10)
        prev_close = c.shift(1)
        tr = pd.concat([
            h - l,
            (h - prev_close).abs(),
            (l - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(ATR_PERIOD).mean()

        indicators[f'{code}_ret10d'] = ret_10d
        indicators[f'{code}_vol10d'] = vol_10d
        indicators[f'{code}_ma_short'] = ma_short
        indicators[f'{code}_ma_long'] = ma_long
        indicators[f'{code}_atr'] = atr
        indicators[f'{code}_close'] = c

    # Market breadth: fraction of trading ETFs above their 20d MA
    breadth_components = pd.DataFrame(index=close_panel.index)
    for code in trading_codes:
        breadth_components[code] = (close_panel[code] > close_panel[code].rolling(20).mean()).astype(float)
    indicators['breadth'] = breadth_components.mean(axis=1)

    return indicators


# ─── Helper: Exit a position ────────────────────────────────────────────────
def exit_position(positions, code, exit_price, today_str, reason, trades, exit_reasons, capital):
    """Exit a single position and record the trade. Returns updated capital."""
    pos = positions[code]
    fee = pos.shares * exit_price * FEE_RATE
    proceeds = pos.shares * exit_price - fee
    entry_cost = pos.shares * pos.entry_price * (1 + FEE_RATE)
    pnl = proceeds - entry_cost
    pnl_pct = (exit_price / pos.entry_price - 1) - 2 * FEE_RATE  # approx

    trades.append(Trade(
        code=code, entry_date=pos.entry_date, exit_date=today_str,
        entry_price=pos.entry_price, exit_price=exit_price,
        shares=pos.shares, pnl=pnl, pnl_pct=pnl_pct,
        weight=pos.weight, risk_budget=pos.risk_budget_used,
        exit_reason=reason
    ))
    exit_reasons[reason] += 1
    capital += proceeds
    del positions[code]
    return capital


def exit_all_positions(positions, close_panel, i, today_str, reason, trades, exit_reasons, capital):
    """Exit ALL positions. Returns updated capital."""
    for code in list(positions.keys()):
        exit_price = close_panel[code].iloc[i]
        if np.isnan(exit_price):
            exit_price = positions[code].entry_price
        capital = exit_position(positions, code, exit_price, today_str, reason, trades, exit_reasons, capital)
    return capital


# ─── Backtest Engine ─────────────────────────────────────────────────────────
def run_backtest():
    print("Loading data...")
    df = load_data()
    close_panel, high_panel, low_panel, trading_codes = prepare_data(df)
    date_index = close_panel.index
    print(f"Trading codes: {len(trading_codes)}, Dates: {len(date_index)}")
    print(f"Period: {date_index[0].strftime('%Y-%m-%d')} to {date_index[-1].strftime('%Y-%m-%d')}")

    print("Computing indicators...")
    indicators = compute_indicators(close_panel, high_panel, low_panel, trading_codes)

    # State variables
    capital = INITIAL_CAPITAL
    positions = {}  # code -> Position
    equity_curve = []
    trades = []
    crash_events = []
    exit_reasons = defaultdict(int)

    days_since_rebalance = REBALANCE_DAYS  # start ready to rebalance
    crash_cooldown = 0
    equity_dd_cooldown = 0
    post_crash_mode = False
    post_crash_first_trade_codes = set()
    post_crash_profitable = None

    peak_equity = INITIAL_CAPITAL
    equity_history = []

    warmup = 25  # need at least 20 days for longest indicator

    print("Running backtest...")

    for i in range(len(date_index)):
        today = date_index[i]
        today_str = today.strftime('%Y-%m-%d')

        if i < warmup:
            equity_curve.append((today_str, INITIAL_CAPITAL))
            equity_history.append(INITIAL_CAPITAL)
            continue

        # ─── Calculate current portfolio value ───────────────────────────
        portfolio_value = capital
        for code, pos in positions.items():
            price = close_panel[code].iloc[i]
            portfolio_value += pos.shares * (price if not np.isnan(price) else pos.entry_price)

        equity_history.append(portfolio_value)
        equity_curve.append((today_str, portfolio_value))

        if portfolio_value > peak_equity:
            peak_equity = portfolio_value

        # Decrement cooldowns
        if crash_cooldown > 0:
            crash_cooldown -= 1
        if equity_dd_cooldown > 0:
            equity_dd_cooldown -= 1

        # Increment rebalance counter
        days_since_rebalance += 1

        # ─── Update trailing stops and check stop exits ──────────────────
        stops_hit_today = 0
        codes_to_exit_stop = []

        for code, pos in list(positions.items()):
            price = close_panel[code].iloc[i]
            if np.isnan(price):
                continue

            # Update highest close for trailing stop
            if price > pos.highest_close:
                pos.highest_close = price

            # Update trailing stop (only moves up)
            atr_val = indicators[f'{code}_atr'].iloc[i]
            if not np.isnan(atr_val) and atr_val > 0:
                new_trailing = pos.highest_close - ATR_MULTIPLIER * atr_val
                if new_trailing > pos.trailing_stop:
                    pos.trailing_stop = new_trailing

            # Check stops
            if price <= pos.stop_price:
                codes_to_exit_stop.append((code, 'hard_stop'))
                stops_hit_today += 1
            elif price <= pos.trailing_stop:
                codes_to_exit_stop.append((code, 'trailing_stop'))
                stops_hit_today += 1

        # Execute stop exits
        for code, reason in codes_to_exit_stop:
            if code in positions:
                exit_price = close_panel[code].iloc[i]
                capital = exit_position(positions, code, exit_price, today_str, reason,
                                        trades, exit_reasons, capital)
                # Track post-crash first trades
                if post_crash_mode and code in post_crash_first_trade_codes:
                    post_crash_first_trade_codes.discard(code)
                    post_crash_profitable = False

        # ─── Crash Detection (fires if 3+ of 5 signals) ─────────────────
        crash_signals = []

        # Signal a: breadth dropped > 0.15 in last 3 days
        breadth_now = indicators['breadth'].iloc[i]
        if i >= 3:
            breadth_3ago = indicators['breadth'].iloc[i - 3]
            if not np.isnan(breadth_now) and not np.isnan(breadth_3ago):
                if breadth_3ago - breadth_now > CRASH_BREADTH_DROP:
                    crash_signals.append('breadth_drop_3d')

        # Signal b: benchmark dropped > 3% in 1 day
        bench_ret_today = indicators['bench_daily_ret'].iloc[i]
        if not np.isnan(bench_ret_today) and bench_ret_today < -CRASH_BENCH_DROP:
            crash_signals.append('bench_drop_1d')

        # Signal c: portfolio lost > 2% in 1 day
        if len(equity_history) >= 2 and equity_history[-2] > 0:
            port_ret = (equity_history[-1] - equity_history[-2]) / equity_history[-2]
            if port_ret < -CRASH_PORTFOLIO_DROP:
                crash_signals.append('portfolio_drop_1d')

        # Signal d: 3+ positions hit stop same day
        if stops_hit_today >= CRASH_STOPS_HIT:
            crash_signals.append('multi_stop_hit')

        # Signal e: 5d annualized vol > 40%
        bench_vol5d = indicators['bench_vol_5d'].iloc[i]
        if not np.isnan(bench_vol5d) and bench_vol5d > CRASH_BENCH_VOL_5D:
            crash_signals.append('bench_vol_5d_high')

        # Crash exit if 3+ signals AND we have positions
        if len(crash_signals) >= CRASH_SIGNALS_NEEDED and len(positions) > 0:
            crash_events.append(CrashEvent(
                date=today_str, signals_fired=crash_signals, action='exit_all'
            ))
            capital = exit_all_positions(positions, close_panel, i, today_str,
                                         'crash_exit', trades, exit_reasons, capital)
            crash_cooldown = CRASH_WAIT_DAYS
            post_crash_mode = True
            post_crash_first_trade_codes = set()
            post_crash_profitable = None
            continue  # skip rest of day

        # ─── Breadth exit check ──────────────────────────────────────────
        if not np.isnan(breadth_now) and breadth_now < BREADTH_EXIT_THRESHOLD and len(positions) > 0:
            capital = exit_all_positions(positions, close_panel, i, today_str,
                                         'breadth_exit', trades, exit_reasons, capital)
            continue

        # ─── Benchmark vol exit check ────────────────────────────────────
        bench_vol10d = indicators['bench_vol_10d'].iloc[i]
        if not np.isnan(bench_vol10d) and bench_vol10d > BENCH_VOL_EXIT_MAX and len(positions) > 0:
            capital = exit_all_positions(positions, close_panel, i, today_str,
                                         'bench_vol_exit', trades, exit_reasons, capital)
            continue

        # ─── Equity drawdown exit ────────────────────────────────────────
        if peak_equity > 0:
            current_dd = (peak_equity - portfolio_value) / peak_equity
            if current_dd > EQUITY_DD_EXIT and len(positions) > 0:
                capital = exit_all_positions(positions, close_panel, i, today_str,
                                             'equity_dd_exit', trades, exit_reasons, capital)
                equity_dd_cooldown = EQUITY_DD_WAIT
                continue

        # ─── Rebalance Check (only enter/rebalance every N days) ─────────
        if days_since_rebalance < REBALANCE_DAYS:
            continue  # not a rebalance day

        # On any stop trigger, also allow rebalance (reset counter)
        # Otherwise normal 5-day cycle
        days_since_rebalance = 0

        # In cooldown → skip entry
        if crash_cooldown > 0 or equity_dd_cooldown > 0:
            continue

        # Post-crash re-entry failure → additional wait
        if post_crash_mode and post_crash_profitable is False:
            crash_cooldown = REENTRY_FAIL_WAIT
            post_crash_mode = False
            post_crash_profitable = None
            continue

        # ─── Entry Conditions ────────────────────────────────────────────
        # Condition 1: Market breadth > 0.45
        if np.isnan(breadth_now) or breadth_now <= BREADTH_ENTRY_THRESHOLD:
            continue

        # Condition 2: Benchmark close > 20d MA
        bench_close_val = indicators['bench_close'].iloc[i]
        bench_ma20_val = indicators['bench_ma20'].iloc[i]
        if np.isnan(bench_close_val) or np.isnan(bench_ma20_val):
            continue
        if bench_close_val <= bench_ma20_val:
            continue

        # Condition 3: Benchmark 10d vol < 30%
        if np.isnan(bench_vol10d) or bench_vol10d >= BENCH_VOL_ENTRY_MAX:
            continue

        # Condition 4: Equity curve filter (portfolio > 15d MA of equity)
        if len(equity_history) >= EQUITY_MA_PERIOD:
            equity_ma = np.mean(equity_history[-EQUITY_MA_PERIOD:])
            if portfolio_value < equity_ma * 0.9999:  # small tolerance for float precision
                continue

        # ─── Momentum Scoring ────────────────────────────────────────────
        scores = {}
        for code in trading_codes:
            ret10 = indicators[f'{code}_ret10d'].iloc[i]
            vol10 = indicators[f'{code}_vol10d'].iloc[i]
            if np.isnan(ret10) or np.isnan(vol10) or vol10 <= 0:
                continue
            scores[code] = ret10 / vol10  # risk-adjusted return

        if len(scores) == 0:
            continue

        # Select top N by score
        sorted_codes = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        candidates = sorted_codes[:TOP_N]

        # ─── Filter candidates by individual ETF conditions ──────────────
        valid_candidates = []
        for code in candidates:
            c_val = indicators[f'{code}_close'].iloc[i]
            ma_s = indicators[f'{code}_ma_short'].iloc[i]
            ma_l = indicators[f'{code}_ma_long'].iloc[i]
            if np.isnan(c_val) or np.isnan(ma_s) or np.isnan(ma_l):
                continue
            if c_val > ma_s and c_val > ma_l:
                valid_candidates.append(code)

        if len(valid_candidates) == 0:
            continue

        # ─── Exit positions not in new selection ─────────────────────────
        for code in list(positions.keys()):
            if code not in valid_candidates:
                exit_price = close_panel[code].iloc[i]
                if np.isnan(exit_price):
                    continue
                # Track post-crash
                pnl_check = exit_price - positions[code].entry_price
                capital = exit_position(positions, code, exit_price, today_str,
                                        'rebalance', trades, exit_reasons, capital)
                if post_crash_mode and code in post_crash_first_trade_codes:
                    post_crash_first_trade_codes.discard(code)
                    if pnl_check > 0:
                        if post_crash_profitable is None:
                            post_crash_profitable = True
                    else:
                        post_crash_profitable = False

        # ─── Recalculate portfolio value after exits ─────────────────────
        portfolio_value = capital
        for code, pos in positions.items():
            price = close_panel[code].iloc[i]
            portfolio_value += pos.shares * (price if not np.isnan(price) else pos.entry_price)

        # Size factor for post-crash mode
        size_factor = REENTRY_SIZE_FACTOR if post_crash_mode else 1.0

        # Calculate current exposure
        current_exposure = 0.0
        for code, pos in positions.items():
            price = close_panel[code].iloc[i]
            if not np.isnan(price):
                current_exposure += pos.shares * price / portfolio_value

        # ─── Enter/adjust positions for valid candidates ─────────────────
        for code in valid_candidates:
            if code in positions:
                continue  # already holding

            price = close_panel[code].iloc[i]
            atr_val = indicators[f'{code}_atr'].iloc[i]
            if np.isnan(price) or np.isnan(atr_val) or atr_val <= 0 or price <= 0:
                continue

            # Risk budget calculation
            stop_distance_pct = (ATR_MULTIPLIER * atr_val) / price
            if stop_distance_pct <= 0.001:
                continue

            raw_weight = (RISK_BUDGET_PCT * size_factor) / stop_distance_pct
            weight = min(raw_weight, MAX_POSITION_PCT)

            # Cap total exposure
            if current_exposure + weight > MAX_EXPOSURE_PCT:
                weight = MAX_EXPOSURE_PCT - current_exposure
                if weight <= 0.01:
                    break  # no more room

            # Calculate shares
            position_value = portfolio_value * weight
            shares = int(position_value / price)
            if shares <= 0:
                continue

            # Check capital
            entry_cost = shares * price * (1 + FEE_RATE)
            if entry_cost > capital:
                shares = int(capital * 0.99 / (price * (1 + FEE_RATE)))
                if shares <= 0:
                    continue
                entry_cost = shares * price * (1 + FEE_RATE)
                weight = (shares * price) / portfolio_value

            capital -= entry_cost
            stop_price = price - ATR_MULTIPLIER * atr_val

            positions[code] = Position(
                code=code,
                entry_date=today_str,
                entry_price=price,
                shares=shares,
                weight=weight,
                stop_price=stop_price,
                trailing_stop=stop_price,
                highest_close=price,
                risk_budget_used=RISK_BUDGET_PCT * size_factor
            )
            current_exposure += weight

            # Track post-crash first trades
            if post_crash_mode:
                post_crash_first_trade_codes.add(code)

        # If post-crash and first trade was profitable, return to full sizing
        if post_crash_mode and post_crash_profitable is True:
            post_crash_mode = False
            post_crash_profitable = None

    # ─── Final: close all remaining positions at last price ──────────────
    last_i = len(date_index) - 1
    last_date_str = date_index[last_i].strftime('%Y-%m-%d')
    if positions:
        capital = exit_all_positions(positions, close_panel, last_i, last_date_str,
                                     'end_of_backtest', trades, exit_reasons, capital)

    return equity_curve, trades, crash_events, exit_reasons, close_panel


# ─── Analytics ───────────────────────────────────────────────────────────────
def compute_metrics(equity_curve, trades, crash_events, exit_reasons, close_panel):
    """Compute and print comprehensive backtest results."""
    dates = [e[0] for e in equity_curve]
    equities = np.array([e[1] for e in equity_curve])

    total_days = len(equities)
    years = total_days / 252.0
    total_return = (equities[-1] / equities[0]) - 1
    annual_return = (1 + total_return) ** (1 / years) - 1

    # Daily returns
    daily_rets = np.diff(equities) / equities[:-1]
    daily_rets = daily_rets[~np.isnan(daily_rets)]
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equities)
    drawdowns = (peak - equities) / peak
    max_dd = np.max(drawdowns)
    max_dd_idx = np.argmax(drawdowns)
    max_dd_date = dates[max_dd_idx]
    dd_start_idx = np.argmax(equities[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
    dd_start_date = dates[dd_start_idx]

    calmar = annual_return / max_dd if max_dd > 0 else 0

    # Trade metrics
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / n_trades
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_risk = np.mean([t.risk_budget for t in trades])
        avg_weight = np.mean([t.weight for t in trades])
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_risk = avg_weight = 0

    trades_per_week = n_trades / (total_days / 5) if total_days > 0 else 0

    # Time in market (count days with positions)
    invested_days = set()
    for t in trades:
        entry_d = pd.Timestamp(t.entry_date)
        exit_d = pd.Timestamp(t.exit_date)
        mask = (close_panel.index >= entry_d) & (close_panel.index <= exit_d)
        for d in close_panel.index[mask]:
            invested_days.add(d)
    time_in_market = len(invested_days) / total_days * 100 if total_days > 0 else 0

    # Benchmark comparison
    bench_series = close_panel[BENCHMARK]
    bench_start = bench_series.iloc[0]
    bench_end = bench_series.iloc[-1]
    bench_return = (bench_end / bench_start) - 1
    bench_annual = (1 + bench_return) ** (1 / years) - 1
    bench_rets = bench_series.pct_change().dropna().values
    bench_peak = np.maximum.accumulate(bench_series.values)
    bench_dd = (bench_peak - bench_series.values) / bench_peak
    bench_max_dd = np.max(bench_dd)
    bench_sharpe = np.mean(bench_rets) / np.std(bench_rets) * np.sqrt(252) if np.std(bench_rets) > 0 else 0

    # ─── Print Results ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  STRATEGY O: MOMENTUM CRASH PROTECTION WITH DEFENSIVE ALLOCATION")
    print("=" * 80)

    print(f"\n{'─' * 60}")
    print("  PERFORMANCE SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Period:              {dates[0]} to {dates[-1]}")
    print(f"  Trading Days:        {total_days}")
    print(f"  Initial Capital:     ¥{INITIAL_CAPITAL:,.0f}")
    print(f"  Final Equity:        ¥{equities[-1]:,.0f}")
    print(f"  Total Return:        {total_return * 100:.2f}%")
    print(f"  Annual Return:       {annual_return * 100:.2f}%")
    print(f"  Max Drawdown:        {max_dd * 100:.2f}%")
    print(f"    DD Peak Date:      {dd_start_date}")
    print(f"    DD Trough Date:    {max_dd_date}")
    print(f"  Sharpe Ratio:        {sharpe:.3f}")
    print(f"  Calmar Ratio:        {calmar:.3f}")
    print(f"  Time in Market:      {time_in_market:.1f}%")

    print(f"\n{'─' * 60}")
    print("  BENCHMARK COMPARISON (510300)")
    print(f"{'─' * 60}")
    print(f"  Benchmark Return:    {bench_return * 100:.2f}%")
    print(f"  Benchmark Annual:    {bench_annual * 100:.2f}%")
    print(f"  Benchmark MaxDD:     {bench_max_dd * 100:.2f}%")
    print(f"  Benchmark Sharpe:    {bench_sharpe:.3f}")
    print(f"  Alpha (annual):      {(annual_return - bench_annual) * 100:.2f}%")

    print(f"\n{'─' * 60}")
    print("  TRADE STATISTICS")
    print(f"{'─' * 60}")
    print(f"  Total Trades:        {n_trades}")
    print(f"  Win Rate:            {win_rate * 100:.1f}%")
    print(f"  Avg Win:             {avg_win * 100:.2f}%")
    print(f"  Avg Loss:            {avg_loss * 100:.2f}%")
    print(f"  Profit Factor:       {profit_factor:.2f}")
    print(f"  Avg Risk/Trade:      {avg_risk * 100:.2f}%")
    print(f"  Avg Position Size:   {avg_weight * 100:.1f}%")
    print(f"  Trades/Week:         {trades_per_week:.2f}")

    print(f"\n{'─' * 60}")
    print("  EXIT REASON BREAKDOWN")
    print(f"{'─' * 60}")
    total_exits = sum(exit_reasons.values())
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        pct = count / total_exits * 100 if total_exits > 0 else 0
        print(f"  {reason:22s}: {count:4d} ({pct:5.1f}%)")

    print(f"\n{'─' * 60}")
    print("  POSITION SIZING DISTRIBUTION")
    print(f"{'─' * 60}")
    if n_trades > 0:
        weights = [t.weight * 100 for t in trades]
        risks = [t.risk_budget * 100 for t in trades]
        print(f"  --- Position Weights ---")
        print(f"  Min:                 {np.min(weights):.1f}%")
        print(f"  25th Percentile:     {np.percentile(weights, 25):.1f}%")
        print(f"  Median:              {np.median(weights):.1f}%")
        print(f"  75th Percentile:     {np.percentile(weights, 75):.1f}%")
        print(f"  Max:                 {np.max(weights):.1f}%")
        print(f"  Mean:                {np.mean(weights):.1f}%")
        print(f"  Std Dev:             {np.std(weights):.1f}%")
        print(f"  --- Risk Budget per Trade ---")
        print(f"  Mean Risk:           {np.mean(risks):.2f}%")
        print(f"  Max Risk:            {np.max(risks):.2f}%")

    print(f"\n{'─' * 60}")
    print("  CRASH DETECTION EVENTS")
    print(f"{'─' * 60}")
    print(f"  Total Crash Events:  {len(crash_events)}")
    if crash_events:
        for evt in crash_events:
            signals_str = ", ".join(evt.signals_fired)
            print(f"    {evt.date} | {evt.action} | Signals: {signals_str}")
    else:
        print("    (none triggered)")

    # ─── Yearly Breakdown ────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  YEARLY BREAKDOWN")
    print(f"{'─' * 60}")
    header = f"  {'Year':<6} {'Return':>8} {'MaxDD':>8} {'Sharpe':>8} {'Trades':>7} {'WinRate':>8}"
    print(header)
    print(f"  {'─' * 52}")

    eq_df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df.set_index('date', inplace=True)
    eq_df['year'] = eq_df.index.year

    for year in sorted(eq_df['year'].unique()):
        year_eq = eq_df[eq_df['year'] == year]['equity'].values
        if len(year_eq) < 2:
            continue
        yr_ret = (year_eq[-1] / year_eq[0]) - 1
        yr_peak = np.maximum.accumulate(year_eq)
        yr_dd = np.max((yr_peak - year_eq) / yr_peak) if len(yr_peak) > 0 else 0
        yr_daily = np.diff(year_eq) / year_eq[:-1]
        yr_sharpe = np.mean(yr_daily) / np.std(yr_daily) * np.sqrt(252) if np.std(yr_daily) > 0 else 0
        yr_trades = [t for t in trades if t.entry_date.startswith(str(year))]
        yr_wins = [t for t in yr_trades if t.pnl > 0]
        yr_wr = len(yr_wins) / len(yr_trades) * 100 if yr_trades else 0
        print(f"  {year:<6} {yr_ret * 100:>7.2f}% {yr_dd * 100:>7.2f}% {yr_sharpe:>8.3f} {len(yr_trades):>7} {yr_wr:>7.1f}%")

    # ─── Monthly Returns Table ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  MONTHLY RETURNS (%)")
    print(f"{'─' * 60}")

    # Use 'M' for older pandas compatibility
    monthly = eq_df['equity'].resample('M').last()
    monthly_rets = monthly.pct_change().dropna()

    months_header = "       " + "".join(f"{m:>6}" for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) + "   Year"
    print(months_header)

    for year in sorted(monthly_rets.index.year.unique()):
        row = f"  {year} "
        yr_total = 0
        for month in range(1, 13):
            mask = (monthly_rets.index.year == year) & (monthly_rets.index.month == month)
            vals = monthly_rets[mask]
            if len(vals) > 0:
                val = vals.iloc[0] * 100
                yr_total += val
                row += f"{val:>6.1f}"
            else:
                row += f"{'--':>6}"
        row += f"{yr_total:>7.1f}"
        print(row)

    # ─── Top 10 Trades ───────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  TOP 10 WINNING TRADES")
    print(f"{'─' * 60}")
    sorted_trades = sorted(trades, key=lambda t: t.pnl, reverse=True)
    print(f"  {'Code':<8} {'Entry':>10} {'Exit':>10} {'PnL%':>7} {'Weight':>7} {'Reason':<15}")
    for t in sorted_trades[:10]:
        print(f"  {t.code:<8} {t.entry_date:>10} {t.exit_date:>10} {t.pnl_pct*100:>6.2f}% {t.weight*100:>6.1f}% {t.exit_reason:<15}")

    print(f"\n{'─' * 60}")
    print("  TOP 10 LOSING TRADES")
    print(f"{'─' * 60}")
    print(f"  {'Code':<8} {'Entry':>10} {'Exit':>10} {'PnL%':>7} {'Weight':>7} {'Reason':<15}")
    for t in sorted_trades[-10:]:
        print(f"  {t.code:<8} {t.entry_date:>10} {t.exit_date:>10} {t.pnl_pct*100:>6.2f}% {t.weight*100:>6.1f}% {t.exit_reason:<15}")

    print("\n" + "=" * 80)
    print("  BACKTEST COMPLETE")
    print("=" * 80)

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'n_trades': n_trades,
    }


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    equity_curve, trades, crash_events, exit_reasons, close_panel = run_backtest()
    metrics = compute_metrics(equity_curve, trades, crash_events, exit_reasons, close_panel)

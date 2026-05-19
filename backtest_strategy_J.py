"""
Strategy J: Adaptive Lookback with Layered Risk Management
==========================================================
ETF rotation strategy that adapts its momentum lookback period based on
Information Coefficient (IC) analysis, combined with 4-layer risk management:
  Layer 1: Regime Filter (benchmark MA + market breadth)
  Layer 2: Volatility Targeting (15% annualized target)
  Layer 3: Trailing Stop (2.5x ATR + hard stop)
  Layer 4: Portfolio Drawdown Control (liquidation + cooldown)
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "data_layer/backtest_fixed.db"
BENCHMARK = "510300"
FEE_RATE = 0.0005  # 0.05% per side
REBALANCE_FREQ = 5  # every 5 trading days
TOP_N = 2  # select top 2 ETFs

# Adaptive lookback
CANDIDATE_LOOKBACKS = [5, 10, 15, 20, 25, 30]
IC_WINDOW = 60  # rolling window for IC calculation
IC_THRESHOLD = 0.03  # min IC to trust momentum

# Regime filter
REGIME_MA_PERIOD = 20
BREADTH_THRESHOLD = 0.40
REGIME_FAIL_EXPOSURE = 0.30

# Volatility targeting
TARGET_VOL = 0.15  # 15% annualized
VOL_LOOKBACK = 20

# Trailing stop
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.5
HARD_STOP_PCT = 0.05  # -5% from entry

# Drawdown control
DD_LIQUIDATE = 0.07  # 7% drawdown -> liquidate all
DD_REDUCE = 0.05  # 5% drawdown -> reduce 50%
COOLDOWN_DAYS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM etf_daily ORDER BY code, date", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_matrices(df):
    """Build price/volume matrices indexed by date, columns by ETF code."""
    pivot_close = df.pivot(index="date", columns="code", values="close")
    pivot_open = df.pivot(index="date", columns="code", values="open")
    pivot_high = df.pivot(index="date", columns="code", values="high")
    pivot_low = df.pivot(index="date", columns="code", values="low")
    pivot_close = pivot_close.sort_index()
    pivot_open = pivot_open.sort_index()
    pivot_high = pivot_high.sort_index()
    pivot_low = pivot_low.sort_index()
    return pivot_close, pivot_open, pivot_high, pivot_low


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────
def compute_atr(high, low, close, period=14):
    """Compute ATR for each ETF. Returns DataFrame."""
    tr = pd.DataFrame(index=high.index, columns=high.columns, dtype=float)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
    # Recompute properly
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = tr.rolling(period, min_periods=period).mean()
    return atr


def compute_returns(close, period):
    """Compute period returns."""
    return close.pct_change(period)


def compute_volatility(close, period):
    """Compute rolling volatility (annualized)."""
    daily_ret = close.pct_change()
    vol = daily_ret.rolling(period, min_periods=period).std() * np.sqrt(252)
    return vol


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Lookback via IC
# ─────────────────────────────────────────────────────────────────────────────
def compute_ic_for_lookback(close, etf_codes, lookback, dates, idx, ic_window=60, forward=5):
    """
    Compute IC: Spearman correlation between (lookback-return rank at t-5)
    and (forward 5-day return) over past ic_window days.
    We evaluate at time points separated by forward days within the window.
    """
    # We need data points from idx - ic_window - lookback - forward to idx
    # At each evaluation point t in [idx - ic_window, ..., idx - forward]:
    #   signal_t = lookback-day return rank at t (using data up to t)
    #   outcome_t = forward 5-day return from t to t+forward
    # IC = spearman(signal ranks, outcome) over all evaluation points

    end_idx = idx  # current position in dates array
    # We need at least ic_window evaluation points
    # Each eval point t needs: lookback days before t, and forward days after t

    eval_points = []
    # Go back from idx-forward (to ensure forward return is known without lookahead)
    # The most recent eval point is at idx - forward (so forward return ends at idx)
    for t in range(end_idx - forward, end_idx - forward - ic_window, -1):
        if t - lookback >= 0:
            eval_points.append(t)

    if len(eval_points) < 20:
        return np.nan

    signals = []
    outcomes = []

    for t in eval_points:
        # Signal: lookback-day return at time t
        sig_returns = (close.iloc[t][etf_codes] / close.iloc[t - lookback][etf_codes]) - 1.0
        # Outcome: forward 5-day return from t
        out_returns = (close.iloc[t + forward][etf_codes] / close.iloc[t][etf_codes]) - 1.0

        # Drop NaN
        valid = sig_returns.notna() & out_returns.notna()
        if valid.sum() < 5:
            continue
        signals.append(sig_returns[valid].values)
        outcomes.append(out_returns[valid].values)

    if len(signals) < 10:
        return np.nan

    # Flatten and compute rank correlation
    all_signals = np.concatenate(signals)
    all_outcomes = np.concatenate(outcomes)

    ic, _ = spearmanr(all_signals, all_outcomes)
    return ic


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────────────────────
class Position:
    def __init__(self, code, entry_price, shares, weight):
        self.code = code
        self.entry_price = entry_price
        self.shares = shares
        self.weight = weight
        self.highest_close = entry_price  # for trailing stop
        self.entry_date = None


class Trade:
    def __init__(self, code, entry_date, entry_price, exit_date, exit_price, shares, side):
        self.code = code
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.shares = shares
        self.side = side  # 'long'
        self.pnl = (exit_price - entry_price) * shares
        self.pnl_pct = (exit_price / entry_price) - 1.0
        self.fee = (entry_price * shares + exit_price * shares) * FEE_RATE


def run_backtest():
    print("=" * 80)
    print("Strategy J: Adaptive Lookback with Layered Risk Management")
    print("=" * 80)
    print()

    # Load data
    df = load_data()
    close_df, open_df, high_df, low_df = build_matrices(df)
    dates = close_df.index.tolist()
    n_days = len(dates)

    # ETF universe (exclude benchmark)
    etf_codes = [c for c in close_df.columns if c != BENCHMARK]
    all_codes = list(close_df.columns)

    # Precompute ATR
    atr_df = compute_atr(high_df, low_df, close_df, ATR_PERIOD)

    # Precompute benchmark MA
    bench_close = close_df[BENCHMARK]
    bench_ma20 = bench_close.rolling(REGIME_MA_PERIOD, min_periods=REGIME_MA_PERIOD).mean()

    # Precompute 20d MA for all ETFs (for breadth)
    ma20_all = close_df[etf_codes].rolling(REGIME_MA_PERIOD, min_periods=REGIME_MA_PERIOD).mean()

    # ─── State Variables ─────────────────────────────────────────────────────
    initial_capital = 1_000_000.0
    cash = initial_capital
    positions = {}  # code -> Position
    equity_curve = []
    trades = []

    # Drawdown control
    peak_equity = initial_capital
    cooldown_remaining = 0
    dd_reduced = False  # flag to avoid repeated 50% cuts in same drawdown

    # Tracking
    lookback_usage = defaultdict(int)
    ic_history = []  # (date, best_lookback, best_ic)
    last_rebalance_idx = -REBALANCE_FREQ  # force first rebalance

    # ─── Main Loop ───────────────────────────────────────────────────────────
    # Start from day max(30, IC_WINDOW+max_lookback+REBALANCE_FREQ) to have enough history
    start_idx = IC_WINDOW + max(CANDIDATE_LOOKBACKS) + REBALANCE_FREQ + 5

    # Initialize equity curve for days before start
    for i in range(start_idx):
        equity_curve.append(initial_capital)

    for i in range(start_idx, n_days):
        today = dates[i]

        # ─── Update position tracking (highest close for trailing stop) ──────
        for code, pos in list(positions.items()):
            current_close = close_df.iloc[i][code]
            if not np.isnan(current_close):
                pos.highest_close = max(pos.highest_close, current_close)

        # ─── Layer 4: Portfolio Drawdown Control ─────────────────────────────
        # Compute current equity
        port_value = cash
        for code, pos in positions.items():
            current_close = close_df.iloc[i][code]
            if not np.isnan(current_close):
                port_value += pos.shares * current_close

        peak_equity = max(peak_equity, port_value)
        current_dd = (peak_equity - port_value) / peak_equity if peak_equity > 0 else 0

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            equity_curve.append(port_value)
            continue

        # Drawdown > 7%: liquidate all, enter cooldown
        if current_dd > DD_LIQUIDATE and len(positions) > 0:
            for code, pos in list(positions.items()):
                exit_price = close_df.iloc[i][code]
                if np.isnan(exit_price):
                    continue
                proceeds = pos.shares * exit_price * (1 - FEE_RATE)
                cash += proceeds
                trades.append(Trade(code, pos.entry_date, pos.entry_price,
                                    today, exit_price, pos.shares, 'long'))
            positions = {}
            cooldown_remaining = COOLDOWN_DAYS
            dd_reduced = False
            port_value = cash
            equity_curve.append(port_value)
            continue

        # Drawdown > 5%: reduce positions by 50% (once per drawdown episode)
        if current_dd > DD_REDUCE and len(positions) > 0 and not dd_reduced:
            for code, pos in list(positions.items()):
                exit_price = close_df.iloc[i][code]
                if np.isnan(exit_price):
                    continue
                sell_shares = pos.shares // 2
                if sell_shares > 0:
                    proceeds = sell_shares * exit_price * (1 - FEE_RATE)
                    cash += proceeds
                    trades.append(Trade(code, pos.entry_date, pos.entry_price,
                                        today, exit_price, sell_shares, 'long'))
                    pos.shares -= sell_shares
                    if pos.shares <= 0:
                        del positions[code]
            dd_reduced = True
            port_value = cash
            for code, pos in positions.items():
                port_value += pos.shares * close_df.iloc[i][code]
            equity_curve.append(port_value)
            # Don't continue; still check stops below

        # Reset dd_reduced flag when drawdown recovers
        if current_dd < DD_REDUCE:
            dd_reduced = False

        # ─── Layer 3: Trailing Stop Check ────────────────────────────────────
        stopped_out = []
        for code, pos in list(positions.items()):
            current_close = close_df.iloc[i][code]
            if np.isnan(current_close):
                continue

            # ATR trailing stop
            atr_val = atr_df.iloc[i][code] if not np.isnan(atr_df.iloc[i][code]) else 0
            trail_stop = pos.highest_close - ATR_MULTIPLIER * atr_val

            # Hard stop
            hard_stop = pos.entry_price * (1 - HARD_STOP_PCT)

            stop_price = max(trail_stop, hard_stop)

            if current_close <= stop_price:
                # Stop triggered - exit at current close (intraday assumption)
                proceeds = pos.shares * current_close * (1 - FEE_RATE)
                cash += proceeds
                trades.append(Trade(code, pos.entry_date, pos.entry_price,
                                    today, current_close, pos.shares, 'long'))
                stopped_out.append(code)

        for code in stopped_out:
            del positions[code]

        # ─── Rebalance Check ─────────────────────────────────────────────────
        if i - last_rebalance_idx >= REBALANCE_FREQ:
            last_rebalance_idx = i

            # ─── Step 1: Adaptive Lookback Selection ─────────────────────────
            best_ic = -np.inf
            best_lookback = CANDIDATE_LOOKBACKS[0]

            for lb in CANDIDATE_LOOKBACKS:
                ic = compute_ic_for_lookback(close_df, etf_codes, lb, dates, i,
                                            IC_WINDOW, REBALANCE_FREQ)
                if not np.isnan(ic) and ic > best_ic:
                    best_ic = ic
                    best_lookback = lb

            lookback_usage[best_lookback] += 1
            ic_history.append((today, best_lookback, best_ic))

            # If IC too low, momentum not working → go to cash
            if best_ic < IC_THRESHOLD:
                # Exit all positions
                for code, pos in list(positions.items()):
                    exit_price = close_df.iloc[i][code]
                    if np.isnan(exit_price):
                        continue
                    proceeds = pos.shares * exit_price * (1 - FEE_RATE)
                    cash += proceeds
                    trades.append(Trade(code, pos.entry_date, pos.entry_price,
                                        today, exit_price, pos.shares, 'long'))
                positions = {}
                # Update equity
                port_value = cash
                equity_curve.append(port_value)
                continue

            # ─── Step 2: Signal Generation ───────────────────────────────────
            L = best_lookback
            ret_L = compute_returns(close_df[etf_codes], L).iloc[i]
            vol_L = close_df[etf_codes].pct_change().rolling(L, min_periods=L).std().iloc[i] * np.sqrt(252)

            # Risk-adjusted momentum score
            score = ret_L / vol_L.replace(0, np.nan)
            score = score.dropna()

            if len(score) == 0:
                equity_curve.append(port_value if 'port_value' in dir() else cash)
                continue

            # Rank and select top N
            score_ranked = score.sort_values(ascending=False)
            selected = score_ranked.head(TOP_N).index.tolist()

            # ─── Layer 1: Regime Filter ──────────────────────────────────────
            bench_above_ma = bench_close.iloc[i] > bench_ma20.iloc[i]
            # Market breadth: % of ETFs above their 20d MA
            above_ma = (close_df[etf_codes].iloc[i] > ma20_all.iloc[i]).sum()
            breadth = above_ma / len(etf_codes)

            regime_ok = bench_above_ma and (breadth > BREADTH_THRESHOLD)

            if regime_ok:
                max_exposure = 1.0
                n_positions = TOP_N
            else:
                max_exposure = REGIME_FAIL_EXPOSURE
                n_positions = 1
                selected = selected[:1]

            # ─── Layer 2: Volatility Targeting ───────────────────────────────
            # Compute realized vol of selected ETFs
            sel_vols = []
            for code in selected:
                v = close_df[code].pct_change().rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().iloc[i]
                if not np.isnan(v):
                    sel_vols.append(v * np.sqrt(252))

            if len(sel_vols) > 0:
                avg_vol = np.mean(sel_vols)
                if avg_vol > 0:
                    vol_scale = min(TARGET_VOL / avg_vol, 1.0)
                else:
                    vol_scale = 1.0
            else:
                vol_scale = 1.0

            # Final exposure
            total_exposure = min(max_exposure * vol_scale, 1.0)
            per_position_weight = total_exposure / n_positions

            # ─── Execute Rebalance (signal on T close, execute T+1 open) ─────
            # We'll use next day's open for execution
            if i + 1 >= n_days:
                equity_curve.append(port_value)
                continue

            next_open = open_df.iloc[i + 1]

            # Exit positions not in selected
            for code in list(positions.keys()):
                if code not in selected:
                    exit_price = next_open[code]
                    if np.isnan(exit_price):
                        exit_price = close_df.iloc[i][code]
                    pos = positions[code]
                    proceeds = pos.shares * exit_price * (1 - FEE_RATE)
                    cash += proceeds
                    trades.append(Trade(code, pos.entry_date, pos.entry_price,
                                        today, exit_price, pos.shares, 'long'))
                    del positions[code]

            # Compute target portfolio value
            port_value_now = cash
            for code, pos in positions.items():
                port_value_now += pos.shares * close_df.iloc[i][code]

            # Enter/adjust positions
            for code in selected:
                target_value = port_value_now * per_position_weight
                entry_price = next_open[code]
                if np.isnan(entry_price) or entry_price <= 0:
                    continue

                if code in positions:
                    # Already holding - adjust size
                    current_value = positions[code].shares * entry_price
                    diff = target_value - current_value
                    if abs(diff) / port_value_now > 0.05:  # Only adjust if > 5% difference
                        if diff > 0:
                            # Buy more
                            buy_shares = int(diff / (entry_price * (1 + FEE_RATE)))
                            cost = buy_shares * entry_price * (1 + FEE_RATE)
                            if cost <= cash and buy_shares > 0:
                                cash -= cost
                                positions[code].shares += buy_shares
                        else:
                            # Sell some
                            sell_shares = min(int(-diff / entry_price), positions[code].shares)
                            if sell_shares > 0:
                                proceeds = sell_shares * entry_price * (1 - FEE_RATE)
                                cash += proceeds
                                trades.append(Trade(code, positions[code].entry_date,
                                                    positions[code].entry_price,
                                                    today, entry_price, sell_shares, 'long'))
                                positions[code].shares -= sell_shares
                                if positions[code].shares <= 0:
                                    del positions[code]
                else:
                    # New position
                    buy_shares = int(target_value / (entry_price * (1 + FEE_RATE)))
                    cost = buy_shares * entry_price * (1 + FEE_RATE)
                    if cost <= cash and buy_shares > 0:
                        cash -= cost
                        pos = Position(code, entry_price, buy_shares, per_position_weight)
                        pos.entry_date = today
                        pos.highest_close = entry_price
                        positions[code] = pos

        # ─── End of Day: Compute Equity ──────────────────────────────────────
        if len(equity_curve) <= i:
            port_value = cash
            for code, pos in positions.items():
                c = close_df.iloc[i][code]
                if not np.isnan(c):
                    port_value += pos.shares * c
            equity_curve.append(port_value)

    # Ensure equity curve length matches
    while len(equity_curve) < n_days:
        port_value = cash
        for code, pos in positions.items():
            c = close_df.iloc[-1][code]
            if not np.isnan(c):
                port_value += pos.shares * c
        equity_curve.append(port_value)

    # ─────────────────────────────────────────────────────────────────────────
    # Results Analysis
    # ─────────────────────────────────────────────────────────────────────────
    equity = np.array(equity_curve[:n_days])
    daily_returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] / equity[0]) - 1.0
    n_years = n_days / 252.0
    annual_return = (1 + total_return) ** (1 / n_years) - 1.0

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    max_dd = drawdowns.max()

    # Sharpe
    rf_daily = 0.02 / 252  # 2% risk-free
    sharpe = (np.mean(daily_returns) - rf_daily) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0

    # Calmar
    calmar = annual_return / max_dd if max_dd > 0 else 0

    # Win rate
    if len(trades) > 0:
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = wins / len(trades)
        avg_win = np.mean([t.pnl_pct for t in trades if t.pnl > 0]) if wins > 0 else 0
        avg_loss = np.mean([t.pnl_pct for t in trades if t.pnl <= 0]) if (len(trades) - wins) > 0 else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0

    # Avg trades per week
    trading_weeks = n_days / 5.0
    avg_trades_week = len(trades) / trading_weeks if trading_weeks > 0 else 0

    # Total fees
    total_fees = sum(t.fee for t in trades)

    # ─── Print Results ───────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'─' * 80}")
    print(f"  Period:              {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({n_days} days)")
    print(f"  Initial Capital:     ¥{initial_capital:,.0f}")
    print(f"  Final Equity:        ¥{equity[-1]:,.0f}")
    print(f"  Total Return:        {total_return * 100:.2f}%")
    print(f"  Annual Return:       {annual_return * 100:.2f}%")
    print(f"  Max Drawdown:        {max_dd * 100:.2f}%")
    print(f"  Sharpe Ratio:        {sharpe:.3f}")
    print(f"  Calmar Ratio:        {calmar:.3f}")
    print(f"  Win Rate:            {win_rate * 100:.1f}%")
    print(f"  Avg Win:             {avg_win * 100:.2f}%")
    print(f"  Avg Loss:            {avg_loss * 100:.2f}%")
    print(f"  Total Trades:        {len(trades)}")
    print(f"  Avg Trades/Week:     {avg_trades_week:.2f}")
    print(f"  Total Fees:          ¥{total_fees:,.0f}")
    print(f"  Profit/Loss Ratio:   {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  Profit/Loss Ratio:   N/A")

    # ─── Yearly Breakdown ────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("YEARLY BREAKDOWN")
    print(f"{'─' * 80}")
    print(f"  {'Year':<6} {'Return':>10} {'MaxDD':>10} {'Sharpe':>8} {'Trades':>8}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

    equity_series = pd.Series(equity, index=dates[:n_days])
    for year in sorted(set(d.year for d in dates)):
        year_mask = [d.year == year for d in dates[:n_days]]
        year_equity = equity[year_mask]
        if len(year_equity) < 2:
            continue
        yr_ret = (year_equity[-1] / year_equity[0]) - 1.0
        yr_running_max = np.maximum.accumulate(year_equity)
        yr_dd = ((yr_running_max - year_equity) / yr_running_max).max()
        yr_daily = np.diff(year_equity) / year_equity[:-1]
        yr_sharpe = (np.mean(yr_daily) - rf_daily) / np.std(yr_daily) * np.sqrt(252) if np.std(yr_daily) > 0 else 0
        yr_trades = sum(1 for t in trades if t.exit_date.year == year)
        print(f"  {year:<6} {yr_ret*100:>9.2f}% {yr_dd*100:>9.2f}% {yr_sharpe:>8.3f} {yr_trades:>8}")

    # ─── Adaptive Lookback Usage Stats ───────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("ADAPTIVE LOOKBACK USAGE")
    print(f"{'─' * 80}")
    total_selections = sum(lookback_usage.values())
    print(f"  {'Lookback':<12} {'Count':>8} {'Percentage':>12}")
    print(f"  {'─'*12} {'─'*8} {'─'*12}")
    for lb in CANDIDATE_LOOKBACKS:
        count = lookback_usage.get(lb, 0)
        pct = count / total_selections * 100 if total_selections > 0 else 0
        print(f"  {lb:<12} {count:>8} {pct:>11.1f}%")

    # Count times IC was below threshold (went to cash)
    ic_below = sum(1 for _, _, ic in ic_history if ic < IC_THRESHOLD)
    print(f"\n  Times IC < {IC_THRESHOLD} (cash):  {ic_below}/{len(ic_history)} ({ic_below/len(ic_history)*100:.1f}%)")

    # ─── IC History (sampled) ────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("IC HISTORY (sampled every 10 rebalance points)")
    print(f"{'─' * 80}")
    print(f"  {'Date':<12} {'Best LB':>8} {'IC':>10}")
    print(f"  {'─'*12} {'─'*8} {'─'*10}")
    step = max(1, len(ic_history) // 20)
    for idx in range(0, len(ic_history), step):
        dt, lb, ic = ic_history[idx]
        ic_str = f"{ic:.4f}" if not np.isnan(ic) else "NaN"
        print(f"  {dt.strftime('%Y-%m-%d'):<12} {lb:>8} {ic_str:>10}")

    # ─── Benchmark Comparison ────────────────────────────────────────────────
    bench_prices = close_df[BENCHMARK].values
    bench_ret = (bench_prices[-1] / bench_prices[0]) - 1.0
    bench_annual = (1 + bench_ret) ** (1 / n_years) - 1.0
    bench_running_max = np.maximum.accumulate(bench_prices)
    bench_dd = ((bench_running_max - bench_prices) / bench_running_max).max()
    bench_daily = np.diff(bench_prices) / bench_prices[:-1]
    bench_sharpe = (np.mean(bench_daily) - rf_daily) / np.std(bench_daily) * np.sqrt(252)

    print(f"\n{'─' * 80}")
    print("BENCHMARK COMPARISON (510300 - CSI 300 ETF)")
    print(f"{'─' * 80}")
    print(f"  {'Metric':<20} {'Strategy':>12} {'Benchmark':>12}")
    print(f"  {'─'*20} {'─'*12} {'─'*12}")
    print(f"  {'Annual Return':<20} {annual_return*100:>11.2f}% {bench_annual*100:>11.2f}%")
    print(f"  {'Max Drawdown':<20} {max_dd*100:>11.2f}% {bench_dd*100:>11.2f}%")
    print(f"  {'Sharpe Ratio':<20} {sharpe:>12.3f} {bench_sharpe:>12.3f}")
    print(f"  {'Calmar Ratio':<20} {calmar:>12.3f} {(bench_annual/bench_dd if bench_dd > 0 else 0):>12.3f}")

    # ─── Monthly Returns Table ───────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("MONTHLY RETURNS (%)")
    print(f"{'─' * 80}")
    monthly_equity = equity_series.resample('M').last()
    monthly_ret = monthly_equity.pct_change().dropna()

    years = sorted(set(d.year for d in monthly_ret.index))
    print(f"  {'Year':<6}", end="")
    for m in range(1, 13):
        print(f" {'JFMAMJJASOND'[m-1]:>6}", end="")
    print(f" {'Total':>8}")
    print(f"  {'─'*6}", end="")
    for m in range(1, 13):
        print(f" {'─'*6}", end="")
    print(f" {'─'*8}")

    for year in years:
        print(f"  {year:<6}", end="")
        year_total = 0
        for month in range(1, 13):
            mask = (monthly_ret.index.year == year) & (monthly_ret.index.month == month)
            vals = monthly_ret[mask]
            if len(vals) > 0:
                r = vals.iloc[0] * 100
                year_total += r
                print(f" {r:>6.1f}", end="")
            else:
                print(f" {'--':>6}", end="")
        # Year total from equity
        yr_mask = [d.year == year for d in dates[:n_days]]
        yr_eq = equity[yr_mask]
        if len(yr_eq) > 1:
            yr_total = (yr_eq[-1] / yr_eq[0] - 1) * 100
            print(f" {yr_total:>7.1f}%")
        else:
            print(f" {'--':>8}")

    print(f"\n{'=' * 80}")
    print("Backtest Complete.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_backtest()

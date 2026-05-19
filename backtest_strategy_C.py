"""
Strategy C: Breakout-Rotation Hybrid with Adaptive TopN
========================================================
- Market breadth master switch (fraction of ETFs with close > MA20)
- Combined breakout + momentum + trend signal
- Adaptive TopN based on market regime
- Trailing stop, trend break, time stop, rebalance exits
- Weekly rebalance (every 5 trading days)
- Execution on T+1 open, fee 0.05% per side
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────────
DB_PATH = "/Users/heyu11/Code/finance/data_layer/backtest_fixed.db"
BENCHMARK = "510300"
FEE_RATE = 0.0005  # 0.05% per side
REBALANCE_FREQ = 5  # every 5 trading days
INITIAL_CAPITAL = 1_000_000

# Strategy parameters
MA20_PERIOD = 20
MA10_PERIOD = 10
BREAKOUT_PERIOD = 20
MOM_10 = 10
MOM_20 = 20
ATR_PERIOD = 14
TRAILING_ATR_MULT = 2.0
TIME_STOP_DAYS = 15
TIME_STOP_MIN_GAIN = 0.03  # 3%

# ─── Load Data ───────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM etf_daily ORDER BY code, date", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def compute_indicators(df):
    """Compute all indicators per ETF. Returns dict of code -> DataFrame."""
    etf_data = {}
    codes = [c for c in df['code'].unique() if c != BENCHMARK]

    for code in codes:
        d = df[df['code'] == code].copy().reset_index(drop=True)
        d.sort_values('date', inplace=True)
        d.reset_index(drop=True, inplace=True)

        # Moving averages
        d['ma10'] = d['close'].rolling(MA10_PERIOD).mean()
        d['ma20'] = d['close'].rolling(MA20_PERIOD).mean()

        # Breakout: close >= rolling max of close over PRIOR 20 days (not including today)
        d['rolling_high_20'] = d['close'].shift(1).rolling(BREAKOUT_PERIOD).max()

        # Momentum
        d['ret_10'] = d['close'].pct_change(MOM_10)
        d['ret_20'] = d['close'].pct_change(MOM_20)

        # 20-day volatility (std of daily returns)
        d['vol_20'] = d['close'].pct_change().rolling(MOM_20).std()

        # Risk-adjusted momentum
        d['risk_adj_mom'] = d['ret_20'] / d['vol_20']

        # ATR14
        d['prev_close'] = d['close'].shift(1)
        d['tr'] = np.maximum(
            d['high'] - d['low'],
            np.maximum(
                abs(d['high'] - d['prev_close']),
                abs(d['low'] - d['prev_close'])
            )
        )
        d['atr14'] = d['tr'].rolling(ATR_PERIOD).mean()

        # Close > MA20 flag (for breadth)
        d['above_ma20'] = (d['close'] > d['ma20']).astype(int)

        etf_data[code] = d

    return etf_data


def compute_benchmark(df):
    """Get benchmark series."""
    bench = df[df['code'] == BENCHMARK].copy().reset_index(drop=True)
    bench.sort_values('date', inplace=True)
    bench.reset_index(drop=True, inplace=True)
    return bench


def run_backtest(etf_data, bench_df):
    """Run the backtest."""
    # Get all trading dates
    dates = bench_df['date'].tolist()
    n_dates = len(dates)

    # Build date-indexed lookup for each ETF
    etf_by_date = {}
    for code, d in etf_data.items():
        etf_by_date[code] = d.set_index('date')

    codes = list(etf_data.keys())

    # Portfolio state
    capital = INITIAL_CAPITAL
    positions = {}  # code -> {shares, entry_price, entry_date, entry_idx, highest_since_entry}
    portfolio_values = []

    # Trade log
    trades = []

    # Track rebalance days
    last_rebalance_idx = -REBALANCE_FREQ  # force first rebalance

    # We need at least MA20_PERIOD + BREAKOUT_PERIOD days of warmup
    warmup = max(MA20_PERIOD, BREAKOUT_PERIOD, MOM_20, ATR_PERIOD) + 5

    for i in range(n_dates):
        today = dates[i]

        # Calculate portfolio value at today's close
        port_value = capital
        for code, pos in positions.items():
            if today in etf_by_date[code].index:
                close_price = etf_by_date[code].loc[today, 'close']
                port_value += pos['shares'] * close_price
        portfolio_values.append({'date': today, 'value': port_value})

        # Skip warmup period
        if i < warmup:
            continue

        # ─── Check exits on every day ───────────────────────────────────────
        if i + 1 < n_dates:
            next_date = dates[i + 1]
            exits_to_process = []

            for code in list(positions.keys()):
                if today not in etf_by_date[code].index:
                    continue
                row = etf_by_date[code].loc[today]
                pos = positions[code]

                # Update highest close since entry
                current_close = row['close']
                if current_close > pos['highest_since_entry']:
                    pos['highest_since_entry'] = current_close

                # Exit check 1: Trailing stop
                atr14 = row['atr14'] if not np.isnan(row['atr14']) else 0
                trail_stop = pos['highest_since_entry'] - TRAILING_ATR_MULT * atr14
                if current_close < trail_stop:
                    exits_to_process.append((code, 'trailing_stop'))
                    continue

                # Exit check 2: Trend break (close < MA20) - on weekly check
                if (i - last_rebalance_idx) % REBALANCE_FREQ == 0 or i == last_rebalance_idx:
                    ma20 = row['ma20'] if not np.isnan(row['ma20']) else 0
                    if current_close < ma20 and ma20 > 0:
                        exits_to_process.append((code, 'trend_break'))
                        continue

                # Exit check 3: Time stop
                days_held = i - pos['entry_idx']
                if days_held >= TIME_STOP_DAYS:
                    gain = (current_close - pos['entry_price']) / pos['entry_price']
                    if gain < TIME_STOP_MIN_GAIN:
                        exits_to_process.append((code, 'time_stop'))
                        continue

            # Execute exits at next day open
            for code, reason in exits_to_process:
                if next_date in etf_by_date[code].index:
                    exec_price = etf_by_date[code].loc[next_date, 'open']
                    pos = positions[code]
                    proceeds = pos['shares'] * exec_price * (1 - FEE_RATE)
                    capital += proceeds

                    trades.append({
                        'code': code,
                        'entry_date': pos['entry_date'],
                        'exit_date': next_date,
                        'entry_price': pos['entry_price'],
                        'exit_price': exec_price,
                        'shares': pos['shares'],
                        'pnl': proceeds - pos['shares'] * pos['entry_price'] * (1 + FEE_RATE),
                        'return': (exec_price * (1 - FEE_RATE)) / (pos['entry_price'] * (1 + FEE_RATE)) - 1,
                        'reason': reason,
                        'days_held': i - pos['entry_idx']
                    })
                    del positions[code]

        # ─── Rebalance check (every 5 trading days) ─────────────────────────
        if (i - warmup) % REBALANCE_FREQ != 0:
            continue

        last_rebalance_idx = i

        if i + 1 >= n_dates:
            continue
        next_date = dates[i + 1]

        # ─── Market Breadth ──────────────────────────────────────────────────
        above_count = 0
        total_count = 0
        for code in codes:
            if today in etf_by_date[code].index:
                row = etf_by_date[code].loc[today]
                if not np.isnan(row['ma20']):
                    total_count += 1
                    if row['close'] > row['ma20']:
                        above_count += 1

        breadth = above_count / total_count if total_count > 0 else 0

        # Determine regime
        if breadth > 0.5:
            regime = 'strong'
            exposure_cap = 1.0
            top_n = 3
        elif breadth > 0.3:
            regime = 'normal'
            exposure_cap = 0.6
            top_n = 2
        else:
            regime = 'weak'
            exposure_cap = 0.25
            top_n = 1

        # ─── Signal Generation ───────────────────────────────────────────────
        candidates = []
        for code in codes:
            if today not in etf_by_date[code].index:
                continue
            row = etf_by_date[code].loc[today]

            # Check all conditions
            # 1. Breakout: close >= rolling max of prior 20 days
            if np.isnan(row['rolling_high_20']):
                continue
            breakout = row['close'] >= row['rolling_high_20']

            # 2. Momentum confirmation: 10-day return > 0 AND 20-day return > 0
            if np.isnan(row['ret_10']) or np.isnan(row['ret_20']):
                continue
            momentum = (row['ret_10'] > 0) and (row['ret_20'] > 0)

            # 3. Trend: close > MA10 AND MA10 > MA20
            if np.isnan(row['ma10']) or np.isnan(row['ma20']):
                continue
            trend = (row['close'] > row['ma10']) and (row['ma10'] > row['ma20'])

            if breakout and momentum and trend:
                risk_adj = row['risk_adj_mom'] if not np.isnan(row['risk_adj_mom']) else 0
                candidates.append((code, risk_adj))

        # Rank by risk-adjusted momentum
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Select top N
        selected = [c[0] for c in candidates[:top_n]]

        # ─── Rebalance: exit positions not in selected ───────────────────────
        for code in list(positions.keys()):
            if code not in selected:
                if next_date in etf_by_date[code].index:
                    exec_price = etf_by_date[code].loc[next_date, 'open']
                    pos = positions[code]
                    proceeds = pos['shares'] * exec_price * (1 - FEE_RATE)
                    capital += proceeds

                    trades.append({
                        'code': code,
                        'entry_date': pos['entry_date'],
                        'exit_date': next_date,
                        'entry_price': pos['entry_price'],
                        'exit_price': exec_price,
                        'shares': pos['shares'],
                        'pnl': proceeds - pos['shares'] * pos['entry_price'] * (1 + FEE_RATE),
                        'return': (exec_price * (1 - FEE_RATE)) / (pos['entry_price'] * (1 + FEE_RATE)) - 1,
                        'reason': 'rebalance',
                        'days_held': i - pos['entry_idx']
                    })
                    del positions[code]

        # ─── Enter new positions ─────────────────────────────────────────────
        # Calculate current portfolio value for sizing
        current_port_value = capital
        for code, pos in positions.items():
            if today in etf_by_date[code].index:
                current_port_value += pos['shares'] * etf_by_date[code].loc[today, 'close']

        # Positions to fill
        n_current = len(positions)
        n_to_fill = top_n - n_current

        if n_to_fill > 0 and selected:
            new_entries = [c for c in selected if c not in positions][:n_to_fill]

            if new_entries:
                n_total = len(positions) + len(new_entries)
                size_per_position = (1.0 / n_total) * exposure_cap * current_port_value

                for code in new_entries:
                    if next_date in etf_by_date[code].index:
                        exec_price = etf_by_date[code].loc[next_date, 'open']
                        if exec_price <= 0:
                            continue
                        # Account for fee
                        shares = int(size_per_position / (exec_price * (1 + FEE_RATE)))
                        if shares <= 0:
                            continue
                        cost = shares * exec_price * (1 + FEE_RATE)
                        if cost > capital:
                            shares = int(capital / (exec_price * (1 + FEE_RATE)))
                            if shares <= 0:
                                continue
                            cost = shares * exec_price * (1 + FEE_RATE)

                        capital -= cost
                        positions[code] = {
                            'shares': shares,
                            'entry_price': exec_price,
                            'entry_date': next_date,
                            'entry_idx': i + 1,
                            'highest_since_entry': exec_price
                        }

    # Close remaining positions at last day's close
    last_date = dates[-1]
    for code in list(positions.keys()):
        if last_date in etf_by_date[code].index:
            close_price = etf_by_date[code].loc[last_date, 'close']
            pos = positions[code]
            proceeds = pos['shares'] * close_price * (1 - FEE_RATE)
            capital += proceeds
            trades.append({
                'code': code,
                'entry_date': pos['entry_date'],
                'exit_date': last_date,
                'entry_price': pos['entry_price'],
                'exit_price': close_price,
                'shares': pos['shares'],
                'pnl': proceeds - pos['shares'] * pos['entry_price'] * (1 + FEE_RATE),
                'return': (close_price * (1 - FEE_RATE)) / (pos['entry_price'] * (1 + FEE_RATE)) - 1,
                'reason': 'end_of_backtest',
                'days_held': n_dates - 1 - pos['entry_idx']
            })
            del positions[code]

    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades)

    return portfolio_df, trades_df


def compute_metrics(portfolio_df, trades_df, bench_df):
    """Compute comprehensive performance metrics."""
    portfolio_df = portfolio_df.copy()
    portfolio_df['returns'] = portfolio_df['value'].pct_change()

    total_return = (portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0]) - 1
    n_years = (portfolio_df['date'].iloc[-1] - portfolio_df['date'].iloc[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / n_years) - 1

    # Sharpe (annualized, 252 days)
    daily_rf = 0.0  # assume 0 risk-free
    excess_returns = portfolio_df['returns'].dropna()
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

    # Sortino
    downside = excess_returns[excess_returns < 0]
    sortino = (excess_returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

    # Max drawdown
    cummax = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - cummax) / cummax
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    if len(trades_df) > 0:
        win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df)
        avg_win = trades_df[trades_df['pnl'] > 0]['return'].mean() if (trades_df['pnl'] > 0).any() else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['return'].mean() if (trades_df['pnl'] <= 0).any() else 0
        profit_factor = (trades_df[trades_df['pnl'] > 0]['pnl'].sum() /
                        abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())) if (trades_df['pnl'] <= 0).any() and trades_df[trades_df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
    else:
        win_rate = avg_win = avg_loss = 0
        profit_factor = 0

    # Benchmark metrics
    bench_aligned = bench_df[bench_df['date'].isin(portfolio_df['date'])].copy()
    bench_aligned = bench_aligned.sort_values('date').reset_index(drop=True)
    bench_total_return = (bench_aligned['close'].iloc[-1] / bench_aligned['close'].iloc[0]) - 1
    bench_cagr = (1 + bench_total_return) ** (1 / n_years) - 1
    bench_returns = bench_aligned['close'].pct_change().dropna()
    bench_sharpe = (bench_returns.mean() / bench_returns.std()) * np.sqrt(252) if bench_returns.std() > 0 else 0
    bench_cummax = bench_aligned['close'].cummax()
    bench_dd = ((bench_aligned['close'] - bench_cummax) / bench_cummax).min()

    metrics = {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_trades': len(trades_df),
        'avg_days_held': trades_df['days_held'].mean() if len(trades_df) > 0 else 0,
        'bench_total_return': bench_total_return,
        'bench_cagr': bench_cagr,
        'bench_sharpe': bench_sharpe,
        'bench_max_dd': bench_dd,
        'n_years': n_years,
    }
    return metrics


def yearly_breakdown(portfolio_df):
    """Compute yearly performance."""
    portfolio_df = portfolio_df.copy()
    portfolio_df['year'] = portfolio_df['date'].dt.year
    yearly = []

    for year, group in portfolio_df.groupby('year'):
        if len(group) < 2:
            continue
        year_return = (group['value'].iloc[-1] / group['value'].iloc[0]) - 1
        year_returns = group['value'].pct_change().dropna()
        year_sharpe = (year_returns.mean() / year_returns.std()) * np.sqrt(252) if year_returns.std() > 0 else 0
        cummax = group['value'].cummax()
        year_dd = ((group['value'].values - cummax.values) / cummax.values).min()
        yearly.append({
            'year': year,
            'return': year_return,
            'sharpe': year_sharpe,
            'max_dd': year_dd
        })

    return pd.DataFrame(yearly)


def print_results(metrics, yearly_df, trades_df):
    """Print comprehensive results."""
    print("=" * 70)
    print("STRATEGY C: Breakout-Rotation Hybrid with Adaptive TopN")
    print("=" * 70)
    print(f"\nBacktest Period: ~{metrics['n_years']:.2f} years")
    print(f"Initial Capital: {INITIAL_CAPITAL:,.0f}")
    print(f"Fee: {FEE_RATE*100:.2f}% per side")
    print(f"Rebalance: Every {REBALANCE_FREQ} trading days")

    print("\n" + "─" * 70)
    print("PERFORMANCE METRICS")
    print("─" * 70)
    print(f"{'Metric':<30} {'Strategy':>15} {'Benchmark (510300)':>20}")
    print(f"{'─'*30} {'─'*15} {'─'*20}")
    print(f"{'Total Return':<30} {metrics['total_return']*100:>14.2f}% {metrics['bench_total_return']*100:>19.2f}%")
    print(f"{'CAGR':<30} {metrics['cagr']*100:>14.2f}% {metrics['bench_cagr']*100:>19.2f}%")
    print(f"{'Sharpe Ratio':<30} {metrics['sharpe']:>15.3f} {metrics['bench_sharpe']:>20.3f}")
    print(f"{'Sortino Ratio':<30} {metrics['sortino']:>15.3f} {'':>20}")
    print(f"{'Max Drawdown':<30} {metrics['max_drawdown']*100:>14.2f}% {metrics['bench_max_dd']*100:>19.2f}%")
    print(f"{'Calmar Ratio':<30} {metrics['calmar']:>15.3f} {'':>20}")

    print("\n" + "─" * 70)
    print("TRADE STATISTICS")
    print("─" * 70)
    print(f"{'Total Trades:':<30} {metrics['total_trades']}")
    print(f"{'Win Rate:':<30} {metrics['win_rate']*100:.1f}%")
    print(f"{'Avg Win:':<30} {metrics['avg_win']*100:.2f}%")
    print(f"{'Avg Loss:':<30} {metrics['avg_loss']*100:.2f}%")
    print(f"{'Profit Factor:':<30} {metrics['profit_factor']:.2f}")
    print(f"{'Avg Days Held:':<30} {metrics['avg_days_held']:.1f}")

    print("\n" + "─" * 70)
    print("YEARLY BREAKDOWN")
    print("─" * 70)
    print(f"{'Year':<8} {'Return':>10} {'Sharpe':>10} {'Max DD':>10}")
    print(f"{'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    for _, row in yearly_df.iterrows():
        print(f"{int(row['year']):<8} {row['return']*100:>9.2f}% {row['sharpe']:>10.3f} {row['max_dd']*100:>9.2f}%")

    print("\n" + "─" * 70)
    print("EXIT REASON DISTRIBUTION")
    print("─" * 70)
    if len(trades_df) > 0:
        exit_reasons = trades_df['reason'].value_counts()
        for reason, count in exit_reasons.items():
            pct = count / len(trades_df) * 100
            avg_ret = trades_df[trades_df['reason'] == reason]['return'].mean() * 100
            print(f"  {reason:<20} {count:>5} trades ({pct:>5.1f}%)  avg return: {avg_ret:>+6.2f}%")

    print("\n" + "─" * 70)
    print("TOP 10 BEST TRADES")
    print("─" * 70)
    if len(trades_df) > 0:
        top_trades = trades_df.nlargest(10, 'pnl')
        for _, t in top_trades.iterrows():
            print(f"  {t['code']}  {str(t['entry_date'])[:10]} -> {str(t['exit_date'])[:10]}  "
                  f"return: {t['return']*100:>+6.2f}%  PnL: {t['pnl']:>+12,.0f}  ({t['reason']})")

    print("\n" + "─" * 70)
    print("TOP 10 WORST TRADES")
    print("─" * 70)
    if len(trades_df) > 0:
        worst_trades = trades_df.nsmallest(10, 'pnl')
        for _, t in worst_trades.iterrows():
            print(f"  {t['code']}  {str(t['entry_date'])[:10]} -> {str(t['exit_date'])[:10]}  "
                  f"return: {t['return']*100:>+6.2f}%  PnL: {t['pnl']:>+12,.0f}  ({t['reason']})")

    print("\n" + "=" * 70)


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Computing indicators...")
    etf_data = compute_indicators(df)
    bench_df = compute_benchmark(df)

    print(f"Running backtest ({len(etf_data)} ETFs, {len(bench_df)} trading days)...")
    portfolio_df, trades_df = run_backtest(etf_data, bench_df)

    print("Computing metrics...")
    metrics = compute_metrics(portfolio_df, trades_df, bench_df)
    yearly_df = yearly_breakdown(portfolio_df)

    print_results(metrics, yearly_df, trades_df)

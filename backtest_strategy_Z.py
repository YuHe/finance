"""
Strategy Z: Adaptive Volatility-Scaled Risk-Adjusted Momentum
=============================================================
Core idea: Scale momentum by inverse volatility (Barroso & Santa-Clara 2015).
When trend is strong AND volatility is low → high conviction, large position.
When trend is strong BUT volatility is high → reduce position (likely reversal).

Database: data_layer/backtest_fixed.db
Table: etf_daily (code, date, open, high, low, close, volume)
Benchmark: 510300
Fee: 0.0005 per side
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
DB_PATH = Path(__file__).parent / "data_layer" / "backtest_fixed.db"
BENCHMARK = "510300"
FEE = 0.0005  # per side

# Strategy parameters (only 4 real ones)
RISK_ADJ_MOM_THRESHOLD = 0.5   # min risk-adj momentum to enter
TARGET_VOL = 0.15              # 15% annualized target vol
ATR_MULTIPLIER = 2.0           # trailing stop
BREADTH_THRESHOLD = 0.35       # min breadth to enter

# Derived / secondary
LOOKBACK = 20                  # days for momentum and vol calc
MA_PERIOD = 20                 # MA filter
BASE_SIZE = 0.8                # base position
VOL_SCALAR_CAP = 1.5           # max vol scalar
REBALANCE_DAYS = 5             # rebalance frequency
MA_BREAK_DAYS = 2              # days below MA to exit
CONSEC_DOWN_DAYS = 3           # consecutive down days to exit
WARMUP = 25                    # days before vol scaling kicks in
MIN_RISK_ADJ_MOM_HOLD = 0.0   # exit if risk_adj_mom drops below this
REBALANCE_SWAP_THRESHOLD = 0.3 # swap if current holding's risk_adj_mom < this


# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT code, date, open, high, low, close, volume FROM etf_daily ORDER BY date, code", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def prepare_data(df):
    """Pre-compute all signals for all ETFs."""
    all_codes = df['code'].unique().tolist()
    tradeable_codes = [c for c in all_codes if c != BENCHMARK]

    # Pivot to get price matrices
    close_pivot = df.pivot(index='date', columns='code', values='close')
    high_pivot = df.pivot(index='date', columns='code', values='high')
    low_pivot = df.pivot(index='date', columns='code', values='low')
    open_pivot = df.pivot(index='date', columns='code', values='open')

    dates = close_pivot.index.tolist()

    # Daily returns
    returns = close_pivot.pct_change()

    # 20-day return (momentum)
    mom20 = close_pivot / close_pivot.shift(LOOKBACK) - 1

    # 20-day realized vol (annualized)
    vol20 = returns.rolling(LOOKBACK).std() * np.sqrt(252)

    # Risk-adjusted momentum
    risk_adj_mom = mom20 / vol20

    # MA20
    ma20 = close_pivot.rolling(MA_PERIOD).mean()

    # ATR (14-day by convention, but we use 20 for consistency)
    tr = pd.DataFrame(index=close_pivot.index, columns=close_pivot.columns, dtype=float)
    for code in close_pivot.columns:
        h = high_pivot[code]
        l = low_pivot[code]
        c_prev = close_pivot[code].shift(1)
        tr[code] = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    atr20 = tr.rolling(LOOKBACK).mean()

    # Breadth: fraction of tradeable ETFs with close > MA20
    above_ma = (close_pivot[tradeable_codes] > ma20[tradeable_codes]).sum(axis=1) / len(tradeable_codes)

    return {
        'dates': dates,
        'tradeable_codes': tradeable_codes,
        'close': close_pivot,
        'open': open_pivot,
        'high': high_pivot,
        'low': low_pivot,
        'returns': returns,
        'mom20': mom20,
        'vol20': vol20,
        'risk_adj_mom': risk_adj_mom,
        'ma20': ma20,
        'atr20': atr20,
        'breadth': above_ma,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(data):
    dates = data['dates']
    tradeable = data['tradeable_codes']
    close = data['close']
    open_prices = data['open']
    risk_adj_mom = data['risk_adj_mom']
    ma20 = data['ma20']
    atr20 = data['atr20']
    breadth = data['breadth']

    # Portfolio state
    cash = 1.0
    position = None       # {'code': str, 'shares': float, 'entry_price': float, 'entry_date': date, 'trail_stop': float}
    portfolio_values = []
    portfolio_returns_list = []

    # Tracking
    trades = []
    exit_reasons = defaultdict(int)
    vol_scalars_by_year = defaultdict(list)
    regime_days = {'cash': 0, 'full': 0, 'reduced': 0}
    days_since_rebalance = 0
    days_below_ma = 0
    consec_down = 0

    # We need at least WARMUP days before trading
    start_idx = LOOKBACK + 1  # need LOOKBACK days for signals

    for i in range(len(dates)):
        date = dates[i]
        year = date.year

        if i < start_idx:
            portfolio_values.append(1.0)
            portfolio_returns_list.append(0.0)
            continue

        # Current portfolio value before today's action
        if position is not None:
            code = position['code']
            prev_close = close[code].iloc[i - 1]
            curr_close = close[code].iloc[i]
            # Mark to market using previous close (position value at start of day)
            pos_value = position['shares'] * prev_close
            total_value = cash + pos_value
        else:
            total_value = cash

        # ---- COMPUTE VOL SCALAR (portfolio level) ----
        if i >= WARMUP and len(portfolio_returns_list) >= LOOKBACK:
            recent_rets = np.array(portfolio_returns_list[-LOOKBACK:])
            port_vol = np.std(recent_rets, ddof=1) * np.sqrt(252)
            if port_vol > 0.001:
                vol_scalar = min(TARGET_VOL / port_vol, VOL_SCALAR_CAP)
            else:
                vol_scalar = 1.0
        else:
            vol_scalar = 1.0

        vol_scalars_by_year[year].append(vol_scalar)

        # ---- CHECK EXIT CONDITIONS (using T-1 data, execute at T open) ----
        exit_signal = None
        if position is not None:
            code = position['code']
            # Get T-1 values (signal day)
            ram_val = risk_adj_mom[code].iloc[i - 1]
            close_t1 = close[code].iloc[i - 1]
            ma_t1 = ma20[code].iloc[i - 1]
            atr_val = atr20[code].iloc[i - 1]

            # Update trailing stop using T-1 high
            high_t1 = data['high'][code].iloc[i - 1]
            new_trail = high_t1 - ATR_MULTIPLIER * atr_val
            if not np.isnan(new_trail) and new_trail > position['trail_stop']:
                position['trail_stop'] = new_trail

            # Check consecutive down days
            if i >= 2:
                today_ret = close[code].iloc[i-1] / close[code].iloc[i-2] - 1
                if today_ret < 0:
                    consec_down += 1
                else:
                    consec_down = 0

            # Check days below MA
            if close_t1 < ma_t1:
                days_below_ma += 1
            else:
                days_below_ma = 0

            # Exit conditions
            if ram_val < MIN_RISK_ADJ_MOM_HOLD:
                exit_signal = 'risk_adj_mom_negative'
            elif close_t1 < position['trail_stop']:
                exit_signal = 'atr_trailing_stop'
            elif days_below_ma >= MA_BREAK_DAYS:
                exit_signal = 'ma_break'
            elif consec_down >= CONSEC_DOWN_DAYS:
                exit_signal = 'consec_down'

        # ---- EXECUTE EXIT at today's open ----
        if exit_signal and position is not None:
            code = position['code']
            exit_price = open_prices[code].iloc[i]
            if not np.isnan(exit_price) and exit_price > 0:
                proceeds = position['shares'] * exit_price * (1 - FEE)
                cash += proceeds
                pnl = (exit_price / position['entry_price'] - 1)
                trades.append({
                    'code': code,
                    'entry_date': position['entry_date'],
                    'exit_date': date,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': exit_signal,
                })
                exit_reasons[exit_signal] += 1
                position = None
                days_below_ma = 0
                consec_down = 0
                days_since_rebalance = 0

        # ---- CHECK ENTRY / REBALANCE (using T-1 signals, execute at T open) ----
        days_since_rebalance += 1

        if position is None:
            # Look for entry
            # Get T-1 signals
            candidates = []
            for code in tradeable:
                ram = risk_adj_mom[code].iloc[i - 1]
                c = close[code].iloc[i - 1]
                ma = ma20[code].iloc[i - 1]
                if (not np.isnan(ram) and ram > RISK_ADJ_MOM_THRESHOLD
                    and not np.isnan(c) and not np.isnan(ma) and c > ma):
                    candidates.append((code, ram))

            # Check breadth and benchmark
            b = breadth.iloc[i - 1]
            bench_ram = risk_adj_mom[BENCHMARK].iloc[i - 1] if BENCHMARK in risk_adj_mom.columns else 0

            if (candidates and b > BREADTH_THRESHOLD
                and not np.isnan(bench_ram) and bench_ram > 0):
                # Sort by risk_adj_mom descending, pick top 1
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_code = candidates[0][0]

                # Position sizing with vol scalar
                actual_size = min(BASE_SIZE * vol_scalar, 1.0)
                entry_price = open_prices[best_code].iloc[i]

                if not np.isnan(entry_price) and entry_price > 0:
                    invest = total_value * actual_size
                    shares = invest * (1 - FEE) / entry_price
                    cash = total_value - invest
                    atr_val = atr20[best_code].iloc[i - 1]
                    trail_stop = entry_price - ATR_MULTIPLIER * atr_val if not np.isnan(atr_val) else entry_price * 0.9

                    position = {
                        'code': best_code,
                        'shares': shares,
                        'entry_price': entry_price,
                        'entry_date': date,
                        'trail_stop': trail_stop,
                    }
                    days_below_ma = 0
                    consec_down = 0
                    days_since_rebalance = 0

                    if actual_size >= 0.75:
                        regime_days['full'] += 1
                    else:
                        regime_days['reduced'] += 1
                else:
                    regime_days['cash'] += 1
            else:
                regime_days['cash'] += 1

        elif days_since_rebalance >= REBALANCE_DAYS:
            # Rebalance check: should we swap?
            code = position['code']
            curr_ram = risk_adj_mom[code].iloc[i - 1]

            # Find top-1 candidate
            candidates = []
            for c in tradeable:
                ram = risk_adj_mom[c].iloc[i - 1]
                cl = close[c].iloc[i - 1]
                ma = ma20[c].iloc[i - 1]
                if (not np.isnan(ram) and ram > RISK_ADJ_MOM_THRESHOLD
                    and not np.isnan(cl) and not np.isnan(ma) and cl > ma):
                    candidates.append((c, ram))

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                top_code = candidates[0][0]

                if (top_code != code
                    and not np.isnan(curr_ram) and curr_ram < REBALANCE_SWAP_THRESHOLD):
                    # Swap: exit current, enter new
                    exit_price = open_prices[code].iloc[i]
                    entry_price_new = open_prices[top_code].iloc[i]

                    if (not np.isnan(exit_price) and exit_price > 0
                        and not np.isnan(entry_price_new) and entry_price_new > 0):
                        # Exit
                        proceeds = position['shares'] * exit_price * (1 - FEE)
                        cash += proceeds
                        pnl = (exit_price / position['entry_price'] - 1)
                        trades.append({
                            'code': code,
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'exit_reason': 'rebalance_swap',
                        })
                        exit_reasons['rebalance_swap'] += 1

                        # Enter new
                        total_val_now = cash
                        actual_size = min(BASE_SIZE * vol_scalar, 1.0)
                        invest = total_val_now * actual_size
                        shares = invest * (1 - FEE) / entry_price_new
                        cash = total_val_now - invest
                        atr_val = atr20[top_code].iloc[i - 1]
                        trail_stop = entry_price_new - ATR_MULTIPLIER * atr_val if not np.isnan(atr_val) else entry_price_new * 0.9

                        position = {
                            'code': top_code,
                            'shares': shares,
                            'entry_price': entry_price_new,
                            'entry_date': date,
                            'trail_stop': trail_stop,
                        }
                        days_below_ma = 0
                        consec_down = 0

                    days_since_rebalance = 0
            else:
                days_since_rebalance = 0

            # Track regime
            if position is not None:
                actual_size = min(BASE_SIZE * vol_scalar, 1.0)
                if actual_size >= 0.75:
                    regime_days['full'] += 1
                else:
                    regime_days['reduced'] += 1
            else:
                regime_days['cash'] += 1
        else:
            # Holding, track regime
            actual_size = min(BASE_SIZE * vol_scalar, 1.0)
            if actual_size >= 0.75:
                regime_days['full'] += 1
            else:
                regime_days['reduced'] += 1

        # ---- COMPUTE END-OF-DAY PORTFOLIO VALUE ----
        if position is not None:
            code = position['code']
            curr_close = close[code].iloc[i]
            if not np.isnan(curr_close):
                pos_value = position['shares'] * curr_close
                eod_value = cash + pos_value
            else:
                eod_value = portfolio_values[-1] if portfolio_values else 1.0
        else:
            eod_value = cash

        portfolio_values.append(eod_value)

        # Daily return
        if len(portfolio_values) >= 2 and portfolio_values[-2] > 0:
            daily_ret = portfolio_values[-1] / portfolio_values[-2] - 1
        else:
            daily_ret = 0.0
        portfolio_returns_list.append(daily_ret)

    # Close any remaining position at last close
    if position is not None:
        code = position['code']
        last_close = close[code].iloc[-1]
        if not np.isnan(last_close):
            proceeds = position['shares'] * last_close * (1 - FEE)
            cash += proceeds
            pnl = (last_close / position['entry_price'] - 1)
            trades.append({
                'code': code,
                'entry_date': position['entry_date'],
                'exit_date': dates[-1],
                'entry_price': position['entry_price'],
                'exit_price': last_close,
                'pnl': pnl,
                'exit_reason': 'end_of_backtest',
            })
            position = None
            portfolio_values[-1] = cash

    return {
        'dates': dates,
        'portfolio_values': portfolio_values,
        'portfolio_returns': portfolio_returns_list,
        'trades': trades,
        'exit_reasons': dict(exit_reasons),
        'vol_scalars_by_year': dict(vol_scalars_by_year),
        'regime_days': regime_days,
    }


# ============================================================
# BENCHMARK
# ============================================================
def compute_benchmark(data):
    close = data['close']
    if BENCHMARK not in close.columns:
        return None
    bench = close[BENCHMARK].dropna()
    bench_ret = bench.pct_change().fillna(0)
    bench_values = (1 + bench_ret).cumprod()
    return bench_values


# ============================================================
# METRICS
# ============================================================
def compute_metrics(values, returns_list, dates, trades):
    values = np.array(values)
    returns_arr = np.array(returns_list)

    total_return = values[-1] / values[0] - 1
    n_years = (dates[-1] - dates[0]).days / 365.25
    cagr = (values[-1] / values[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Annualized vol
    ann_vol = np.std(returns_arr[1:], ddof=1) * np.sqrt(252) if len(returns_arr) > 1 else 0

    # Sharpe
    sharpe = cagr / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    max_dd = np.min(dd)

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    if trades:
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = wins / len(trades)
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = (sum(t['pnl'] for t in trades if t['pnl'] > 0) /
                        abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))) if losses else float('inf')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    # Sortino
    neg_rets = returns_arr[returns_arr < 0]
    downside_vol = np.std(neg_rets, ddof=1) * np.sqrt(252) if len(neg_rets) > 1 else 1
    sortino = cagr / downside_vol if downside_vol > 0 else 0

    return {
        'total_return': total_return,
        'cagr': cagr,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
    }


def yearly_breakdown(dates, values, vol_scalars_by_year):
    """Compute yearly returns and avg vol scalar."""
    df = pd.DataFrame({'date': dates, 'value': values})
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    yearly = []
    for year, group in df.groupby('year'):
        if len(group) < 2:
            continue
        yr_ret = group['value'].iloc[-1] / group['value'].iloc[0] - 1
        # Max DD within year
        vals = group['value'].values
        peak = np.maximum.accumulate(vals)
        dd = (vals - peak) / peak
        yr_mdd = np.min(dd)
        # Avg vol scalar
        avg_vs = np.mean(vol_scalars_by_year.get(year, [1.0]))
        yearly.append({
            'year': year,
            'return': yr_ret,
            'max_dd': yr_mdd,
            'avg_vol_scalar': avg_vs,
        })
    return yearly


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("STRATEGY Z: Adaptive Volatility-Scaled Risk-Adjusted Momentum")
    print("=" * 70)
    print()

    # Load and prepare
    print("Loading data...")
    df = load_data()
    print(f"  Records: {len(df):,}, ETFs: {df['code'].nunique()}, "
          f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print()

    print("Computing signals...")
    data = prepare_data(df)
    print(f"  Tradeable ETFs: {len(data['tradeable_codes'])}")
    print()

    # Run backtest
    print("Running backtest...")
    results = run_backtest(data)
    print()

    # Compute benchmark
    bench_values = compute_benchmark(data)

    # Metrics
    metrics = compute_metrics(
        results['portfolio_values'],
        results['portfolio_returns'],
        results['dates'],
        results['trades']
    )

    # Benchmark metrics
    if bench_values is not None:
        bench_vals = bench_values.values
        bench_rets = bench_values.pct_change().fillna(0).values
        bench_dates = bench_values.index.tolist()
        bench_metrics = compute_metrics(
            bench_vals, bench_rets, bench_dates, []
        )
    else:
        bench_metrics = None

    # ---- PRINT RESULTS ----
    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} {'Strategy':>12} {'Benchmark (510300)':>20}")
    print("-" * 60)
    print(f"{'Total Return':<25} {metrics['total_return']:>11.2%} {bench_metrics['total_return']:>19.2%}" if bench_metrics else f"{'Total Return':<25} {metrics['total_return']:>11.2%}")
    print(f"{'CAGR':<25} {metrics['cagr']:>11.2%} {bench_metrics['cagr']:>19.2%}" if bench_metrics else f"{'CAGR':<25} {metrics['cagr']:>11.2%}")
    print(f"{'Annualized Vol':<25} {metrics['ann_vol']:>11.2%} {bench_metrics['ann_vol']:>19.2%}" if bench_metrics else f"{'Annualized Vol':<25} {metrics['ann_vol']:>11.2%}")
    print(f"{'Sharpe Ratio':<25} {metrics['sharpe']:>11.2f} {bench_metrics['sharpe']:>19.2f}" if bench_metrics else f"{'Sharpe Ratio':<25} {metrics['sharpe']:>11.2f}")
    print(f"{'Sortino Ratio':<25} {metrics['sortino']:>11.2f} {bench_metrics['sortino']:>19.2f}" if bench_metrics else f"{'Sortino Ratio':<25} {metrics['sortino']:>11.2f}")
    print(f"{'Max Drawdown':<25} {metrics['max_dd']:>11.2%} {bench_metrics['max_dd']:>19.2%}" if bench_metrics else f"{'Max Drawdown':<25} {metrics['max_dd']:>11.2%}")
    print(f"{'Calmar Ratio':<25} {metrics['calmar']:>11.2f} {bench_metrics['calmar']:>19.2f}" if bench_metrics else f"{'Calmar Ratio':<25} {metrics['calmar']:>11.2f}")
    print()
    print(f"{'Num Trades':<25} {metrics['num_trades']:>11}")
    print(f"{'Win Rate':<25} {metrics['win_rate']:>11.1%}")
    print(f"{'Avg Win':<25} {metrics['avg_win']:>11.2%}")
    print(f"{'Avg Loss':<25} {metrics['avg_loss']:>11.2%}")
    print(f"{'Profit Factor':<25} {metrics['profit_factor']:>11.2f}")
    print()

    # ---- YEARLY BREAKDOWN ----
    print("=" * 70)
    print("YEARLY BREAKDOWN")
    print("=" * 70)
    yearly = yearly_breakdown(results['dates'], results['portfolio_values'],
                             results['vol_scalars_by_year'])
    print(f"{'Year':<8} {'Return':>10} {'MaxDD':>10} {'Avg VolScalar':>15}")
    print("-" * 45)
    for y in yearly:
        print(f"{y['year']:<8} {y['return']:>9.2%} {y['max_dd']:>9.2%} {y['avg_vol_scalar']:>14.2f}")
    print()

    # ---- EXIT REASON BREAKDOWN ----
    print("=" * 70)
    print("EXIT REASON BREAKDOWN")
    print("=" * 70)
    total_exits = sum(results['exit_reasons'].values())
    for reason, count in sorted(results['exit_reasons'].items(), key=lambda x: -x[1]):
        pct = count / total_exits if total_exits > 0 else 0
        # Avg PnL for this exit reason
        pnls = [t['pnl'] for t in results['trades'] if t['exit_reason'] == reason]
        avg_pnl = np.mean(pnls) if pnls else 0
        print(f"  {reason:<25} {count:>4} ({pct:>5.1%})  avg PnL: {avg_pnl:>7.2%}")
    print()

    # ---- REGIME TIME DISTRIBUTION ----
    print("=" * 70)
    print("REGIME TIME DISTRIBUTION")
    print("=" * 70)
    total_days = sum(results['regime_days'].values())
    if total_days > 0:
        print(f"  Cash (no position):    {results['regime_days']['cash']:>5} days ({results['regime_days']['cash']/total_days:>5.1%})")
        print(f"  Full position (>=75%): {results['regime_days']['full']:>5} days ({results['regime_days']['full']/total_days:>5.1%})")
        print(f"  Reduced position:      {results['regime_days']['reduced']:>5} days ({results['regime_days']['reduced']/total_days:>5.1%})")
    print()

    # ---- TOP/BOTTOM 5 TRADES ----
    if results['trades']:
        print("=" * 70)
        print("TOP 5 TRADES")
        print("=" * 70)
        sorted_trades = sorted(results['trades'], key=lambda x: x['pnl'], reverse=True)
        print(f"  {'Code':<8} {'Entry Date':<12} {'Exit Date':<12} {'PnL':>8} {'Exit Reason':<20}")
        print("  " + "-" * 62)
        for t in sorted_trades[:5]:
            entry_d = t['entry_date'].strftime('%Y-%m-%d') if hasattr(t['entry_date'], 'strftime') else str(t['entry_date'])[:10]
            exit_d = t['exit_date'].strftime('%Y-%m-%d') if hasattr(t['exit_date'], 'strftime') else str(t['exit_date'])[:10]
            print(f"  {t['code']:<8} {entry_d:<12} {exit_d:<12} {t['pnl']:>7.2%} {t['exit_reason']:<20}")
        print()

        print("=" * 70)
        print("BOTTOM 5 TRADES")
        print("=" * 70)
        print(f"  {'Code':<8} {'Entry Date':<12} {'Exit Date':<12} {'PnL':>8} {'Exit Reason':<20}")
        print("  " + "-" * 62)
        for t in sorted_trades[-5:]:
            entry_d = t['entry_date'].strftime('%Y-%m-%d') if hasattr(t['entry_date'], 'strftime') else str(t['entry_date'])[:10]
            exit_d = t['exit_date'].strftime('%Y-%m-%d') if hasattr(t['exit_date'], 'strftime') else str(t['exit_date'])[:10]
            print(f"  {t['code']:<8} {entry_d:<12} {exit_d:<12} {t['pnl']:>7.2%} {t['exit_reason']:<20}")
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

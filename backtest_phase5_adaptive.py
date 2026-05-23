"""Phase 5: Adaptive Parameters Backtest for A-share Sector ETF Rotation."""
import sqlite3
import numpy as np
import pandas as pd
from itertools import product

DB_PATH = "/Users/heyu11/Code/finance/data_layer/backtest_adjusted.db"
BENCHMARK = "510300"
FEE = 0.001  # 0.1% per side
START = "2021-08-16"
END = "2025-12-31"

# --- Data Loading ---
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT code, date, open, high, low, close, volume FROM etf_daily", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    close = df.pivot(index='date', columns='code', values='close')
    opn = df.pivot(index='date', columns='code', values='open')
    return close, opn

# --- Signal Computation ---
def compute_signals(close, benchmark_col=BENCHMARK):
    etf_cols = [c for c in close.columns if c != benchmark_col]
    ret = close.pct_change()
    # Sharpe 20d
    ret20 = close[etf_cols].pct_change(20)
    std20 = ret[etf_cols].rolling(20).std()
    sharpe20 = ret20 / (std20 * np.sqrt(20) + 1e-10)
    # RS 10d
    ret10_etf = close[etf_cols].pct_change(10)
    ret10_bench = close[benchmark_col].pct_change(10)
    rs10 = ret10_etf.sub(ret10_bench, axis=0)
    # Reversal 5d (negative return = reversal signal)
    rev5 = -close[etf_cols].pct_change(5)
    # Z-score normalization cross-sectionally
    def zscore(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-10, axis=0)
    return zscore(sharpe20), zscore(rs10), zscore(rev5), etf_cols

# --- Market Regime ---
def compute_regime(close, benchmark_col=BENCHMARK):
    bench = close[benchmark_col]
    ma20 = bench.rolling(20).mean()
    ma60 = bench.rolling(60).mean()
    regime = pd.Series('sideways', index=close.index)
    bull = (bench > ma20) & (ma20 > ma60)
    bear = (bench < ma20) & (ma20 < ma60)
    regime[bull] = 'bull'
    regime[bear] = 'bear'
    return regime

# --- Market Breadth ---
def compute_breadth(close, etf_cols):
    ma20 = close[etf_cols].rolling(20).mean()
    breadth = (close[etf_cols] > ma20).sum(axis=1) / len(etf_cols)
    return breadth

# --- Volatility of benchmark ---
def compute_vol(close, benchmark_col=BENCHMARK):
    ret = close[benchmark_col].pct_change()
    vol = ret.rolling(20).std() * np.sqrt(252)
    return vol

# --- Backtest Engine ---
def run_backtest(close, opn, etf_cols, config):
    """
    config keys:
      adaptive_hold: bool
      dynamic_topn: bool
      regime_weights: bool
      drawdown_control: bool
    """
    z_sharpe, z_rs, z_rev, _ = compute_signals(close)
    regime = compute_regime(close)
    breadth = compute_breadth(close, etf_cols)
    vol = compute_vol(close)

    dates = close.loc[START:END].index.tolist()
    if not dates:
        return None

    # Pre-compute score cross-sectional std for dynamic TOP-N
    # Default score (0.5/0.5) for dispersion measurement
    default_score = 0.5 * z_rs + 0.5 * z_sharpe
    score_std = default_score.std(axis=1)
    # Rolling percentiles for dispersion thresholds
    score_std_p25 = score_std.rolling(60, min_periods=20).quantile(0.25)
    score_std_p75 = score_std.rolling(60, min_periods=20).quantile(0.75)

    equity = 1.0
    peak = 1.0
    holdings = {}  # code -> weight
    hold_counter = 0
    hold_period = 7
    in_cash = False
    dd_cash = False  # drawdown-triggered cash
    dd_peak = 1.0  # peak at time of drawdown trigger
    half_position = False
    turnover_count = 0
    equity_curve = []

    for i, date in enumerate(dates):
        if date not in close.index:
            continue
        # Daily return from holdings
        if holdings and i > 0:
            prev_date = dates[i-1]
            day_ret = 0.0
            for code, w in holdings.items():
                if prev_date in close.index and date in close.index:
                    p0 = close.loc[prev_date, code]
                    p1 = close.loc[date, code]
                    if p0 > 0:
                        day_ret += w * (p1 / p0 - 1)
            equity *= (1 + day_ret)

        # Update peak and drawdown
        if equity > peak:
            peak = equity
        dd = 1 - equity / peak

        # Drawdown control (D)
        if config.get('drawdown_control'):
            b_val = breadth.loc[date] if date in breadth.index else 0
            if dd_cash:
                # Re-entry: breadth recovers above 50% (market timing clears)
                if b_val > 0.5:
                    dd_cash = False
                    half_position = False
                    peak = equity  # reset peak on re-entry
                else:
                    holdings = {}
                    equity_curve.append((date, equity))
                    hold_counter += 1
                    continue
            if dd > 0.08:
                if holdings:
                    turnover_count += 1
                holdings = {}
                dd_cash = True
                dd_peak = peak
                equity_curve.append((date, equity))
                hold_counter += 1
                continue
            elif dd > 0.05:
                half_position = True
            else:
                half_position = False

        # Check if rebalance needed
        hold_counter += 1
        rebalance = False
        if not holdings:
            rebalance = True
        else:
            # Determine current hold period
            if config.get('adaptive_hold'):
                v = vol.loc[date] if date in vol.index else 0.20
                if v < 0.15:
                    hold_period = 12
                elif v > 0.25:
                    hold_period = 4
                else:
                    hold_period = 7
            else:
                hold_period = 7
            if hold_counter >= hold_period:
                rebalance = True

        if not rebalance:
            equity_curve.append((date, equity))
            continue

        hold_counter = 0

        # Timing: breadth > 50%
        b = breadth.loc[date] if date in breadth.index else 0
        r = regime.loc[date] if date in regime.index else 'sideways'

        # Bear regime -> cash (if regime_weights enabled)
        go_cash = False
        if b <= 0.5:
            go_cash = True
        if config.get('regime_weights') and r == 'bear':
            go_cash = True

        if go_cash:
            if holdings:
                turnover_count += 1
            holdings = {}
            equity_curve.append((date, equity))
            continue

        # Compute score based on regime
        if config.get('regime_weights'):
            if r == 'bull':
                score = 0.7 * z_rs.loc[date] + 0.3 * z_sharpe.loc[date]
            else:  # sideways
                score = 0.6 * z_sharpe.loc[date] + 0.4 * z_rev.loc[date]
        else:
            score = 0.5 * z_rs.loc[date] + 0.5 * z_sharpe.loc[date]

        # Determine TOP-N
        top_n = 3
        if config.get('dynamic_topn'):
            s_std = score_std.loc[date] if date in score_std.index else None
            p25 = score_std_p25.loc[date] if date in score_std_p25.index else None
            p75 = score_std_p75.loc[date] if date in score_std_p75.index else None
            if s_std is not None and p75 is not None and s_std > p75:
                top_n = 1
            elif s_std is not None and p25 is not None and s_std < p25:
                top_n = 5
            else:
                top_n = 3

        # Select top ETFs
        valid_scores = score[etf_cols].dropna()
        if len(valid_scores) == 0:
            holdings = {}
            equity_curve.append((date, equity))
            continue
        top = valid_scores.nlargest(top_n).index.tolist()

        # T+1: use next day open for execution (fee applied to equity)
        new_holdings = {c: 1.0/len(top) for c in top}

        # Position scaling for drawdown control
        if config.get('drawdown_control') and half_position:
            new_holdings = {c: w * 0.5 for c, w in new_holdings.items()}

        # Count turnover if holdings changed
        old_set = set(holdings.keys())
        new_set = set(new_holdings.keys())
        if old_set != new_set:
            # Fee: apply to changed portion
            changed = old_set.symmetric_difference(new_set)
            fee_frac = len(changed) / max(len(old_set | new_set), 1)
            equity *= (1 - FEE * 2 * fee_frac)
            turnover_count += 1

        holdings = new_holdings
        equity_curve.append((date, equity))

    return pd.DataFrame(equity_curve, columns=['date', 'equity']).set_index('date'), turnover_count

# --- Metrics ---
def calc_metrics(eq_df, turnover_count):
    if eq_df is None or len(eq_df) < 2:
        return {}
    eq = eq_df['equity']
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (1/years) - 1
    daily_ret = eq.pct_change().dropna()
    sharpe = daily_ret.mean() / (daily_ret.std() + 1e-10) * np.sqrt(252)
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max
    max_dd = -dd.min()
    calmar = ann_ret / (max_dd + 1e-10)
    # Per-year returns
    yearly = {}
    for y in range(eq.index[0].year, eq.index[-1].year + 1):
        yr_eq = eq[eq.index.year == y]
        if len(yr_eq) >= 2:
            yearly[y] = yr_eq.iloc[-1] / yr_eq.iloc[0] - 1
    return {
        'AnnRet': ann_ret, 'Sharpe': sharpe, 'MaxDD': max_dd,
        'Calmar': calmar, 'Turnover': turnover_count, 'Yearly': yearly
    }

# --- Main ---
def main():
    print("Loading data...")
    close, opn = load_data()
    etf_cols = [c for c in close.columns if c != BENCHMARK]
    print(f"ETFs: {len(etf_cols)}, Date range: {close.index[0].date()} to {close.index[-1].date()}")

    variants = {
        'Baseline (3b)':     dict(adaptive_hold=False, dynamic_topn=False, regime_weights=False, drawdown_control=False),
        'A: Adaptive Hold':  dict(adaptive_hold=True,  dynamic_topn=False, regime_weights=False, drawdown_control=False),
        'B: Dynamic TOP-N':  dict(adaptive_hold=False, dynamic_topn=True,  regime_weights=False, drawdown_control=False),
        'C: Regime Weights': dict(adaptive_hold=False, dynamic_topn=False, regime_weights=True,  drawdown_control=False),
        'D: DD Control':     dict(adaptive_hold=False, dynamic_topn=False, regime_weights=False, drawdown_control=True),
        'A+B: Hold+TopN':    dict(adaptive_hold=True,  dynamic_topn=True,  regime_weights=False, drawdown_control=False),
        'C+D: Regime+DD':    dict(adaptive_hold=False, dynamic_topn=False, regime_weights=True,  drawdown_control=True),
        'E: All Combined':   dict(adaptive_hold=True,  dynamic_topn=True,  regime_weights=True,  drawdown_control=True),
    }

    results = {}
    for name, cfg in variants.items():
        print(f"  Running {name}...")
        eq_df, turnover = run_backtest(close, opn, etf_cols, cfg)
        metrics = calc_metrics(eq_df, turnover)
        results[name] = metrics

    # Benchmark buy-and-hold
    bench_eq = close[BENCHMARK].loc[START:END]
    bench_eq = bench_eq / bench_eq.iloc[0]
    bench_df = pd.DataFrame({'equity': bench_eq})
    bench_metrics = calc_metrics(bench_df, 0)
    results['Benchmark (510300)'] = bench_metrics

    # Print comparison table
    print("\n" + "="*120)
    print(f"{'Variant':<22} {'AnnRet':>8} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8} {'Turns':>6}", end="")
    all_years = sorted(set(y for m in results.values() if m for y in m.get('Yearly', {}).keys()))
    for y in all_years:
        print(f" {y:>7}", end="")
    print()
    print("-"*120)

    for name, m in results.items():
        if not m:
            print(f"{name:<22} {'N/A':>8}")
            continue
        print(f"{name:<22} {m['AnnRet']:>7.1%} {m['Sharpe']:>8.2f} {m['MaxDD']:>7.1%} {m['Calmar']:>8.2f} {m['Turnover']:>6}", end="")
        for y in all_years:
            yr = m.get('Yearly', {}).get(y, float('nan'))
            if np.isnan(yr):
                print(f" {'N/A':>7}", end="")
            else:
                print(f" {yr:>6.1%}", end="")
        print()
    print("="*120)

if __name__ == "__main__":
    main()

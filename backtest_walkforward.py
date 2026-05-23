"""
Walk-Forward Validation: Phase 5 best strategy (Adaptive Credit + Premium signal)
Tests whether optimized weights 0.4/0.3/0.3 are overfit by rolling train/test splits.
"""
import sqlite3, struct
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data_layer"
ETF_DB = DATA_DIR / "backtest_adjusted.db"
SIG_DB = DATA_DIR / "signals.db"
BENCHMARK = "510300"
FEE = 0.001
WARMUP = 80
TOP_N = 3
HOLD_DAYS = 7

def load_data():
    conn = sqlite3.connect(ETF_DB)
    df = pd.read_sql("SELECT * FROM etf_daily ORDER BY date, code", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    conn = sqlite3.connect(SIG_DB)
    shibor = pd.read_sql("SELECT * FROM shibor_daily", conn)
    shibor['date'] = pd.to_datetime(shibor['date'])
    macro = pd.read_sql("SELECT month, social_financing FROM macro_monthly", conn)
    nav_df = pd.read_sql("SELECT date, code, nav FROM etf_nav", conn)
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    conn.close()
    def decode_sf(v):
        if isinstance(v, bytes):
            return struct.unpack('<q', v)[0]
        try:
            return float(v)
        except:
            return np.nan
    macro['social_financing'] = macro['social_financing'].apply(decode_sf)
    macro['month'] = pd.to_datetime(macro['month'] + '-01')
    return df, shibor, macro, nav_df

def prepare_signals(df, shibor, macro, nav_df):
    price = df.pivot(index='date', columns='code', values='close').sort_index()
    codes = sorted([c for c in price.columns if c != BENCHMARK])
    bench = price[BENCHMARK]
    dates = price.index
    ret = price[codes].pct_change()

    # Timing
    ma20 = price[codes].rolling(20).mean()
    breadth = (price[codes] > ma20).mean(axis=1)
    breadth_delta = breadth - breadth.shift(5)
    sig_3b = breadth > 0.5
    sig_bt = (breadth_delta > 0.15) & (breadth > 0.4)

    sf = macro.set_index('month')['social_financing'].sort_index()
    sf_12m = sf.pct_change(12)
    ci = sf_12m.diff(1)
    ci_daily = ci.reindex(dates, method='ffill').shift(20)
    credit_expanding = ci_daily > 0

    timing_on = pd.Series(False, index=dates)
    for i in range(len(dates)):
        if credit_expanding.iloc[i] if i < len(credit_expanding) else False:
            timing_on.iloc[i] = sig_3b.iloc[i]
        else:
            timing_on.iloc[i] = sig_bt.iloc[i]

    # Signals
    sharpe_20d = ret.rolling(20).mean() / (ret.rolling(20).std() + 1e-8)
    rs_10d = price[codes].pct_change(10).sub(bench.pct_change(10), axis=0)

    nav_pivot = nav_df.pivot(index='date', columns='code', values='nav').sort_index()
    premium = pd.DataFrame(0.0, index=dates, columns=codes)
    for c in codes:
        if c in nav_pivot.columns:
            nav_aligned = nav_pivot[c].reindex(dates)
            prem = (price[c] - nav_aligned) / (nav_aligned + 1e-8)
            premium[c] = prem.rolling(5).mean()

    # Z-scores
    z_sharpe = row_zscore(sharpe_20d)
    z_rs = row_zscore(rs_10d)
    z_premium_adj = -row_zscore(premium)

    return price, codes, bench, dates, timing_on, z_sharpe, z_rs, z_premium_adj

def row_zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)

def run_backtest_period(price, codes, dates, timing_on, composite, start_date, end_date):
    """Run backtest on a specific date range, return final NAV (starting from 1.0)."""
    mask = (dates >= start_date) & (dates <= end_date)
    period_idx = np.where(mask)[0]
    if len(period_idx) == 0:
        return 1.0

    nav = 1.0
    portfolio = {}
    hold_timer = 0

    for i in period_idx:
        today = dates[i]
        if portfolio:
            daily_ret = sum((price.loc[today, c] / portfolio[c] - 1) for c in portfolio) / len(portfolio)
            nav *= (1 + daily_ret)
            portfolio = {c: price.loc[today, c] for c in portfolio}

        hold_timer += 1
        risk_on = bool(timing_on.iloc[i])

        if portfolio and not risk_on:
            nav *= (1 - FEE)
            portfolio = {}
            hold_timer = 0
            continue

        need_rebal = (not portfolio and risk_on) or (hold_timer >= HOLD_DAYS)
        if not need_rebal or not risk_on:
            continue

        sig = composite.iloc[i]
        valid = sig.dropna()
        if len(valid) < TOP_N:
            continue
        top = valid.nlargest(TOP_N).index.tolist()

        if set(top) == set(portfolio.keys()):
            continue

        if portfolio:
            nav *= (1 - FEE)
        portfolio = {c: price.loc[today, c] for c in top}
        nav *= (1 - FEE)
        hold_timer = 0

    return nav

def generate_windows(dates, warmup_end_date):
    """Generate train/test windows: 12m train, 6m test, rolling 6m."""
    windows = []
    # First possible test start: after warmup + 12 months training
    start = warmup_end_date
    # Use 6-month increments
    from dateutil.relativedelta import relativedelta
    train_start = start
    while True:
        train_end = train_start + relativedelta(months=12) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + relativedelta(months=6) - pd.Timedelta(days=1)
        if test_start > dates[-1]:
            break
        # Clip test_end to data range
        test_end = min(test_end, dates[-1])
        windows.append((train_start, train_end, test_start, test_end))
        train_start = train_start + relativedelta(months=6)
    return windows

def grid_search_weights(price, codes, dates, timing_on, z_sharpe, z_rs, z_premium_adj,
                        train_start, train_end):
    """Grid search over weights, return best (w_sharpe, w_rs, w_prem) maximizing return."""
    best_nav = -np.inf
    best_w = (0.4, 0.3, 0.3)
    step = 0.1
    w_sharpe_range = np.arange(0.2, 0.61, step)
    w_rs_range = np.arange(0.1, 0.41, step)
    w_prem_range = np.arange(0.1, 0.51, step)

    for ws in w_sharpe_range:
        for wr in w_rs_range:
            for wp in w_prem_range:
                if abs(ws + wr + wp - 1.0) > 0.01:
                    continue
                composite = ws * z_sharpe + wr * z_rs + wp * z_premium_adj
                nav_val = run_backtest_period(price, codes, dates, timing_on,
                                             composite, train_start, train_end)
                if nav_val > best_nav:
                    best_nav = nav_val
                    best_w = (round(ws, 1), round(wr, 1), round(wp, 1))
    return best_w, best_nav

def main():
    print("Loading data...")
    df, shibor, macro, nav_df = load_data()
    print("Preparing signals...")
    price, codes, bench, dates, timing_on, z_sharpe, z_rs, z_premium_adj = \
        prepare_signals(df, shibor, macro, nav_df)

    warmup_end_date = dates[WARMUP]
    windows = generate_windows(dates, warmup_end_date)

    print(f"\nData range: {dates[0].date()} to {dates[-1].date()}")
    print(f"Warmup ends: {warmup_end_date.date()}")
    print(f"Number of walk-forward windows: {len(windows)}")

    # Walk-forward with optimized weights
    print("\n" + "="*90)
    print("WALK-FORWARD VALIDATION (optimized weights per window)")
    print("="*90)
    print(f"{'Window':<6} {'Train Period':<25} {'Test Period':<25} {'Opt Weights':<18} {'Train Ret':>10} {'Test Ret':>10}")
    print("-"*90)

    wf_navs_opt = []
    for idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        best_w, train_nav = grid_search_weights(price, codes, dates, timing_on,
                                                 z_sharpe, z_rs, z_premium_adj, tr_s, tr_e)
        # Apply optimized weights to test window
        composite_test = best_w[0]*z_sharpe + best_w[1]*z_rs + best_w[2]*z_premium_adj
        test_nav = run_backtest_period(price, codes, dates, timing_on,
                                       composite_test, te_s, te_e)
        wf_navs_opt.append(test_nav)
        w_str = f"{best_w[0]}/{best_w[1]}/{best_w[2]}"
        print(f"  {idx+1:<4} {str(tr_s.date())+' - '+str(tr_e.date()):<25} "
              f"{str(te_s.date())+' - '+str(te_e.date()):<25} {w_str:<18} "
              f"{(train_nav-1)*100:>9.2f}% {(test_nav-1)*100:>9.2f}%")

    wf_compound_opt = 1.0
    for n in wf_navs_opt:
        wf_compound_opt *= n

    # Walk-forward with fixed weights (0.4/0.3/0.3)
    print("\n" + "="*90)
    print("WALK-FORWARD WITH FIXED WEIGHTS (0.4/0.3/0.3)")
    print("="*90)
    print(f"{'Window':<6} {'Test Period':<25} {'Test Ret':>10}")
    print("-"*90)

    composite_fixed = 0.4*z_sharpe + 0.3*z_rs + 0.3*z_premium_adj
    wf_navs_fixed = []
    for idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        test_nav = run_backtest_period(price, codes, dates, timing_on,
                                       composite_fixed, te_s, te_e)
        wf_navs_fixed.append(test_nav)
        print(f"  {idx+1:<4} {str(te_s.date())+' - '+str(te_e.date()):<25} {(test_nav-1)*100:>9.2f}%")

    wf_compound_fixed = 1.0
    for n in wf_navs_fixed:
        wf_compound_fixed *= n

    # Full in-sample return
    full_start = dates[WARMUP]
    full_end = dates[-1]
    insample_nav = run_backtest_period(price, codes, dates, timing_on,
                                       composite_fixed, full_start, full_end)

    # Summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"  Full in-sample return (0.4/0.3/0.3):     {(insample_nav-1)*100:.2f}%")
    print(f"  Walk-forward compound (optimized):        {(wf_compound_opt-1)*100:.2f}%")
    print(f"  Walk-forward compound (fixed 0.4/0.3/0.3): {(wf_compound_fixed-1)*100:.2f}%")
    print(f"  Overfit ratio (WF_opt / in-sample):       {wf_compound_opt/insample_nav:.3f}")
    print(f"  Fixed vs in-sample ratio:                 {wf_compound_fixed/insample_nav:.3f}")
    print()
    if wf_compound_fixed > 1.0:
        print("  CONCLUSION: Fixed weights produce positive OOS returns -> NOT overfit")
    else:
        print("  CONCLUSION: Fixed weights produce negative OOS returns -> POSSIBLE overfit")
    if wf_compound_opt >= wf_compound_fixed * 0.9:
        print("  Optimized WF close to or better than fixed -> weights are robust")
    else:
        print("  Optimized WF significantly worse than fixed -> in-sample optimization unstable")

if __name__ == "__main__":
    main()

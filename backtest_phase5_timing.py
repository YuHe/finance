"""
Phase 5: Advanced Timing Backtest for A-share Sector ETF Rotation.
Compares multiple timing signals against Phase 3b baseline.
"""
import sqlite3, struct
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data_layer"
ETF_DB = DATA_DIR / "backtest_adjusted.db"
SIG_DB = DATA_DIR / "signals.db"

BENCHMARK = "510300"
FEE = 0.001  # per side
TOP_N = 3
HOLD_DAYS = 7
WARMUP = 65


def load_data():
    conn = sqlite3.connect(ETF_DB)
    df = pd.read_sql("SELECT * FROM etf_daily ORDER BY date, code", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])

    conn = sqlite3.connect(SIG_DB)
    shibor = pd.read_sql("SELECT date, overnight FROM shibor_daily", conn)
    shibor['date'] = pd.to_datetime(shibor['date'])

    nb = pd.read_sql("SELECT date, net_buy FROM northbound_daily", conn)
    nb['date'] = pd.to_datetime(nb['date'])

    macro = pd.read_sql("SELECT month, social_financing FROM macro_monthly", conn)
    def decode_sf(v):
        if isinstance(v, bytes):
            return struct.unpack('<q', v)[0]
        try:
            return float(v)
        except:
            return np.nan
    macro['social_financing'] = macro['social_financing'].apply(decode_sf)
    macro['month'] = pd.to_datetime(macro['month'] + '-01')
    conn.close()
    return df, shibor, nb, macro


def prepare(df, shibor, nb, macro):
    close = df.pivot(index='date', columns='code', values='close').sort_index()
    dates = close.index
    etf_codes = [c for c in close.columns if c != BENCHMARK]
    bench = close[BENCHMARK]

    # --- Selection signals (same as Phase 3b) ---
    rets_1d = close[etf_codes].pct_change(1)
    rets_10d = close[etf_codes].pct_change(10)
    rets_20d = close[etf_codes].pct_change(20)
    vol_20d = rets_1d.rolling(20).std()
    sharpe_20d = rets_20d / (vol_20d * np.sqrt(20) + 1e-8)
    bench_ret_10d = bench.pct_change(10)
    rs_10d = rets_10d.sub(bench_ret_10d, axis=0)

    # Northbound acceleration
    nb_s = nb.set_index('date')['net_buy'].reindex(dates).fillna(0)
    nb_5d = nb_s.rolling(5).sum()
    nb_20d = nb_s.rolling(20).sum()
    nb_accel = nb_5d - nb_20d / 4

    # --- Timing signals ---
    etf_ma20 = close[etf_codes].rolling(20).mean()
    breadth = (close[etf_codes] > etf_ma20).mean(axis=1)

    # Phase 3b baseline: breadth > 0.5
    sig_3b = breadth > 0.5

    # Breadth Thrust: breadth delta > 0.15 AND breadth > 0.4
    breadth_delta = breadth - breadth.shift(5)
    sig_bt = (breadth_delta > 0.15) & (breadth > 0.4)

    # Vol-of-Vol: risk-off when vov > 75th pct of trailing 120d
    bench_ret = bench.pct_change()
    rvol20 = bench_ret.rolling(20).std() * np.sqrt(252)
    vov = rvol20.rolling(20).std()
    vov_75 = vov.rolling(120, min_periods=60).quantile(0.75)
    sig_vov = vov <= vov_75

    # Shibor Stress: risk-off when overnight/ma20 - 1 > 0.3
    shib = shibor.set_index('date')['overnight'].reindex(dates).ffill()
    shib_ma20 = shib.rolling(20).mean()
    shib_stress = shib / shib_ma20 - 1
    sig_shib = shib_stress <= 0.3

    # Credit Impulse: social_financing 12m pct_change diff > 0
    sf = macro.set_index('month')['social_financing'].sort_index()
    sf_12m = sf.pct_change(12)
    ci = sf_12m.diff(1)
    ci_daily = ci.reindex(dates, method='ffill').shift(20)
    sig_ci = ci_daily > 0

    timing = pd.DataFrame({
        'phase3b': sig_3b,
        'breadth_thrust': sig_bt,
        'vov': sig_vov,
        'shibor': sig_shib,
        'credit': sig_ci,
    }, index=dates).fillna(False)

    vote = timing[['breadth_thrust', 'vov', 'shibor', 'credit']].sum(axis=1)
    timing['composite_2of4'] = vote >= 2
    timing['composite_3of4'] = vote >= 3
    timing['bt_vov'] = timing['breadth_thrust'] & timing['vov']

    # Hybrid: Phase3b AND NOT shibor_stress (use 3b but cut when shibor spikes)
    timing['3b_no_shibor_stress'] = timing['phase3b'] & timing['shibor']
    # Hybrid: Phase3b OR breadth_thrust (enter on either)
    timing['3b_or_bt'] = timing['phase3b'] | timing['breadth_thrust']
    # Hybrid: Breadth_thrust OR (phase3b AND vov_ok)
    timing['bt_or_3b_vov'] = timing['breadth_thrust'] | (timing['phase3b'] & timing['vov'])
    # Adaptive: phase3b when credit expanding, breadth_thrust otherwise
    timing['adaptive_credit'] = np.where(timing['credit'], timing['phase3b'], timing['breadth_thrust'])
    timing['adaptive_credit'] = timing['adaptive_credit'].astype(bool)

    return close, etf_codes, bench, sharpe_20d, rs_10d, nb_accel, timing, dates


def zscore_cross(s):
    s = s.dropna()
    if len(s) < 3 or s.std() < 1e-8:
        return s
    return (s - s.mean()) / s.std()


def run_backtest(timing_col, close, etf_codes, bench, sharpe_20d, rs_10d, nb_accel, timing, dates):
    """Backtest engine matching Phase 3b logic exactly, only changing timing signal."""
    n = len(dates)
    nav = 1.0
    portfolio = {}
    hold_timer = 0
    trades = 0
    nav_hist = [1.0]
    date_hist = [dates[WARMUP - 1]]

    is_on = timing[timing_col]

    for i in range(WARMUP, n):
        today = dates[i]

        # Mark-to-market
        if portfolio:
            daily_ret = 0
            for code in portfolio:
                curr = close.loc[today, code]
                prev = portfolio[code]
                if prev > 0:
                    daily_ret += (curr / prev - 1) / len(portfolio)
            nav *= (1 + daily_ret)
            portfolio = {code: close.loc[today, code] for code in portfolio}

        nav_hist.append(nav)
        date_hist.append(today)
        hold_timer += 1

        # Timing check
        risk_on = bool(is_on.iloc[i]) if i < len(is_on) else False

        # Risk-off -> exit
        if portfolio and not risk_on:
            nav *= (1 - FEE)
            trades += len(portfolio)
            portfolio = {}
            hold_timer = 0
            continue

        # Rebalance decision
        need_rebal = (not portfolio and risk_on) or (hold_timer >= HOLD_DAYS)
        if not need_rebal or not risk_on:
            continue

        # Compute signal (Phase 3b: 0.5*z(sharpe) + 0.5*z(rs) + nb boost)
        s_sharpe = sharpe_20d.iloc[i]
        s_rs = rs_10d.iloc[i]
        z_s = zscore_cross(s_sharpe)
        z_r = zscore_cross(s_rs)
        base = 0.5 * z_s + 0.5 * z_r

        # Northbound boost
        nb_val = nb_accel.iloc[i] if i < len(nb_accel) else 0
        if not np.isnan(nb_val):
            if nb_val > 0:
                base = base * 1.1
            elif nb_val < -30:
                base = base * 0.9

        sig_clean = base.dropna()
        if len(sig_clean) < TOP_N:
            continue
        top = sig_clean.nlargest(TOP_N).index.tolist()

        if set(top) == set(portfolio.keys()):
            continue

        # Execute trade
        if portfolio:
            nav *= (1 - FEE)
            trades += len(portfolio)
        portfolio = {code: close.loc[today, code] for code in top}
        nav *= (1 - FEE)
        trades += len(top)
        hold_timer = 0

    return np.array(nav_hist), date_hist, trades


def calc_metrics(nav_hist, date_hist, bench):
    equity = np.array(nav_hist)
    n_days = len(equity) - 1
    years = n_days / 252

    total_ret = equity[-1] / equity[0] - 1
    ann_ret = (1 + total_ret) ** (1 / years) - 1

    daily_r = np.diff(equity) / equity[:-1]
    sharpe = np.mean(daily_r) / (np.std(daily_r) + 1e-8) * np.sqrt(252)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Benchmark
    bench_vals = bench.reindex([d for d in date_hist]).dropna()
    if len(bench_vals) >= 2:
        bench_total = bench_vals.iloc[-1] / bench_vals.iloc[0] - 1
        bench_ann = (1 + bench_total) ** (1 / years) - 1
    else:
        bench_ann = 0
    alpha = ann_ret - bench_ann

    # Per-year
    yearly = {}
    for idx in range(1, len(date_hist)):
        yr = date_hist[idx].year
        if yr not in yearly:
            yearly[yr] = []
        yearly[yr].append(equity[idx] / equity[idx-1] - 1)
    yr_rets = {yr: np.prod([1+r for r in rets]) - 1 for yr, rets in yearly.items()}

    return ann_ret, sharpe, max_dd, calmar, alpha, yr_rets


def main():
    print("Loading data...")
    df, shibor, nb, macro = load_data()
    print("Preparing signals...")
    close, etf_codes, bench, sharpe_20d, rs_10d, nb_accel, timing, dates = \
        prepare(df, shibor, nb, macro)

    strategies = [
        ('Phase 3b (breadth>50%)', 'phase3b'),
        ('Breadth Thrust only', 'breadth_thrust'),
        ('Vol-of-Vol only', 'vov'),
        ('Shibor Stress only', 'shibor'),
        ('Credit Impulse only', 'credit'),
        ('Composite 2-of-4', 'composite_2of4'),
        ('Composite 3-of-4', 'composite_3of4'),
        ('BT + VoV combo', 'bt_vov'),
        ('3b + Shibor filter', '3b_no_shibor_stress'),
        ('3b OR BreadthThrust', '3b_or_bt'),
        ('BT OR (3b AND VoV)', 'bt_or_3b_vov'),
        ('Adaptive (credit)', 'adaptive_credit'),
    ]

    results = {}
    print(f"\n{'='*95}")
    print(f"{'Strategy':<26} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'Alpha':>7} {'Trades':>7}")
    print(f"{'='*95}")

    for name, col in strategies:
        nav_hist, date_hist, trades = run_backtest(
            col, close, etf_codes, bench, sharpe_20d, rs_10d, nb_accel, timing, dates)
        ann_ret, sharpe, max_dd, calmar, alpha, yr_rets = calc_metrics(nav_hist, date_hist, bench)
        results[name] = (ann_ret, sharpe, max_dd, calmar, alpha, yr_rets, trades)
        print(f"{name:<26} {ann_ret*100:>7.1f}% {sharpe:>7.2f} {max_dd*100:>7.1f}% {calmar:>7.2f} {alpha*100:>6.1f}% {trades:>7}")

    # Benchmark
    bench_start = bench.iloc[WARMUP]
    bench_end = bench.iloc[-1]
    b_total = bench_end / bench_start - 1
    b_years = (len(dates) - WARMUP) / 252
    b_ann = (1 + b_total) ** (1/b_years) - 1
    b_daily = bench.pct_change().iloc[WARMUP:]
    b_sharpe = b_daily.mean() / (b_daily.std() + 1e-8) * np.sqrt(252)
    b_peak = bench.iloc[WARMUP:].cummax()
    b_dd = ((bench.iloc[WARMUP:] - b_peak) / b_peak).min()
    b_calmar = b_ann / abs(b_dd) if b_dd != 0 else 0
    print(f"{'510300 Buy&Hold':<26} {b_ann*100:>7.1f}% {b_sharpe:>7.2f} {b_dd*100:>7.1f}% {b_calmar:>7.2f} {'0.0%':>7} {'N/A':>7}")
    print(f"{'='*95}")

    # Per-year breakdown
    years = sorted(set(yr for _, _, _, _, _, yr_rets, _ in results.values() for yr in yr_rets))
    print(f"\n\nPER-YEAR RETURNS:")
    print(f"{'Strategy':<26}", end="")
    for yr in years:
        print(f" {yr:>8}", end="")
    print()
    print("-" * (26 + 9 * len(years)))
    for name, col in strategies:
        ann_ret, sharpe, max_dd, calmar, alpha, yr_rets, trades = results[name]
        print(f"{name:<26}", end="")
        for yr in years:
            v = yr_rets.get(yr, np.nan)
            print(f" {v*100:>7.1f}%", end="")
        print()

    # Benchmark per year
    print(f"{'510300 Buy&Hold':<26}", end="")
    for yr in years:
        yr_bench = bench[bench.index.year == yr]
        if len(yr_bench) > 1:
            r = yr_bench.iloc[-1] / yr_bench.iloc[0] - 1
            print(f" {r*100:>7.1f}%", end="")
        else:
            print(f" {'N/A':>8}", end="")
    print()

    # Signal activity
    bt_dates = dates[WARMUP:]
    print(f"\n\nSIGNAL ACTIVITY (% days risk-on):")
    for name, col in strategies:
        pct = timing.loc[bt_dates, col].sum() / len(bt_dates) * 100
        print(f"  {name:<26}: {pct:.1f}%")

    # Win rate analysis for best strategies
    print(f"\n\nTRADE ANALYSIS:")
    for name, col in strategies:
        nav_hist, date_hist, trades = run_backtest(
            col, close, etf_codes, bench, sharpe_20d, rs_10d, nb_accel, timing, dates)
        # Approximate: total return / number of round trips
        total_ret = nav_hist[-1] / nav_hist[0] - 1
        rt = trades // (TOP_N * 2) if trades > 0 else 1
        print(f"  {name:<26}: {trades:>4} legs, ~{rt:>3} round-trips, total={total_ret*100:.1f}%")


if __name__ == "__main__":
    main()

"""
Phase 5 Combined: Best timing (Adaptive Credit) + Best new signals (Premium + Shibor_Tilt)
+ Drawdown control. Target: 25-30% annual.
"""
import sqlite3, struct
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data_layer"
ETF_DB = DATA_DIR / "backtest_adjusted.db"
SIG_DB = DATA_DIR / "signals.db"
BENCHMARK = "510300"
FEE = 0.001
WARMUP = 80

def load_data():
    conn = sqlite3.connect(ETF_DB)
    df = pd.read_sql("SELECT * FROM etf_daily ORDER BY date, code", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])

    conn = sqlite3.connect(SIG_DB)
    shibor = pd.read_sql("SELECT * FROM shibor_daily", conn)
    shibor['date'] = pd.to_datetime(shibor['date'])
    macro = pd.read_sql("SELECT month, social_financing FROM macro_monthly", conn)
    nav = pd.read_sql("SELECT date, code, nav FROM etf_nav", conn)
    nav['date'] = pd.to_datetime(nav['date'])
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
    return df, shibor, macro, nav

def prepare(df, shibor, macro, nav):
    price = df.pivot(index='date', columns='code', values='close').sort_index()
    volume = df.pivot(index='date', columns='code', values='volume').sort_index()
    codes = sorted([c for c in price.columns if c != BENCHMARK])
    bench = price[BENCHMARK]
    dates = price.index

    ret = price[codes].pct_change()
    bench_ret = bench.pct_change()

    # === TIMING: Adaptive Credit ===
    # Breadth
    ma20 = price[codes].rolling(20).mean()
    breadth = (price[codes] > ma20).mean(axis=1)
    breadth_delta = breadth - breadth.shift(5)
    sig_3b = breadth > 0.5
    sig_bt = (breadth_delta > 0.15) & (breadth > 0.4)

    # Credit impulse
    sf = macro.set_index('month')['social_financing'].sort_index()
    sf_12m = sf.pct_change(12)
    ci = sf_12m.diff(1)
    ci_daily = ci.reindex(dates, method='ffill').shift(20)
    credit_expanding = ci_daily > 0

    # Adaptive: Phase3b when credit expanding, BT when contracting
    timing_on = pd.Series(False, index=dates)
    for i in range(len(dates)):
        if credit_expanding.iloc[i] if i < len(credit_expanding) else False:
            timing_on.iloc[i] = sig_3b.iloc[i]
        else:
            timing_on.iloc[i] = sig_bt.iloc[i]

    # === SELECTION SIGNALS ===
    # Sharpe_20d + RS_10d (baseline)
    sharpe_20d = ret.rolling(20).mean() / (ret.rolling(20).std() + 1e-8)
    rs_10d = price[codes].pct_change(10).sub(bench.pct_change(10), axis=0)

    # Shibor Term Spread tilt
    shib = shibor.set_index('date').sort_index()
    if 'three_month' in shib.columns and 'overnight' in shib.columns:
        term_spread = shib['three_month'] - shib['overnight']
    elif '3m' in shib.columns and 'overnight' in shib.columns:
        term_spread = shib['3m'] - shib['overnight']
    else:
        cols = shib.columns.tolist()
        term_spread = shib[cols[-1]] - shib[cols[0]] if len(cols) >= 2 else pd.Series(0, index=shib.index)
    term_z = (term_spread - term_spread.rolling(60).mean()) / (term_spread.rolling(60).std() + 1e-8)
    term_z_daily = term_z.reindex(dates).ffill()

    # Beta for tilt signals
    beta_60 = pd.DataFrame(index=dates, columns=codes, dtype=float)
    bx = bench_ret.values
    for c in codes:
        y = ret[c].values
        ci_idx = codes.index(c)
        for i in range(59, len(dates)):
            x60 = bx[i-59:i+1]
            y60 = y[i-59:i+1]
            mask = ~(np.isnan(x60) | np.isnan(y60))
            if mask.sum() < 20:
                continue
            xm, ym = x60[mask], y60[mask]
            cov_val = np.mean((xm - xm.mean()) * (ym - ym.mean()))
            var_val = np.var(xm)
            if var_val > 1e-15:
                beta_60.iat[i, ci_idx] = cov_val / var_val

    shibor_tilt = pd.DataFrame(0.0, index=dates, columns=codes)
    for i, d in enumerate(dates):
        if d in term_z_daily.index:
            tz = term_z_daily.loc[d]
            if not np.isnan(tz):
                betas = beta_60.iloc[i]
                b_mean = betas.mean()
                b_std = betas.std()
                if b_std > 1e-8:
                    shibor_tilt.iloc[i] = tz * (betas - b_mean) / b_std

    # ETF Premium/Discount
    nav_pivot = nav.pivot(index='date', columns='code', values='nav').sort_index()
    premium = pd.DataFrame(0.0, index=dates, columns=codes)
    for c in codes:
        if c in nav_pivot.columns:
            nav_aligned = nav_pivot[c].reindex(dates)
            prem = (price[c] - nav_aligned) / (nav_aligned + 1e-8)
            premium[c] = prem.rolling(5).mean()

    return price, codes, bench, dates, timing_on, sharpe_20d, rs_10d, shibor_tilt, premium, breadth

def row_zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)

def run_backtest(price, codes, bench, dates, timing_on, composite, top_n=3, hold_days=7, dd_control=False):
    nav = 1.0
    peak = 1.0
    portfolio = {}
    hold_timer = 0
    nav_hist = []
    dd_cash = False

    for i in range(WARMUP, len(dates)):
        today = dates[i]
        # Mark-to-market
        if portfolio:
            daily_ret = sum((price.loc[today, c] / portfolio[c] - 1) for c in portfolio) / len(portfolio)
            nav *= (1 + daily_ret)
            portfolio = {c: price.loc[today, c] for c in portfolio}

        # Drawdown control
        if nav > peak:
            peak = nav
        dd = (nav - peak) / peak
        if dd_control:
            if dd < -0.08:
                if portfolio:
                    nav *= (1 - FEE)
                    portfolio = {}
                dd_cash = True
                nav_hist.append(nav)
                hold_timer += 1
                continue
            if dd_cash and dd > -0.03 and timing_on.iloc[i]:
                dd_cash = False
            if dd_cash:
                nav_hist.append(nav)
                hold_timer += 1
                continue
            # Half position at 5% DD handled via top_n reduction
            effective_top_n = max(1, top_n // 2) if dd < -0.05 else top_n
        else:
            effective_top_n = top_n

        nav_hist.append(nav)
        hold_timer += 1
        risk_on = bool(timing_on.iloc[i])

        if portfolio and not risk_on:
            nav *= (1 - FEE)
            portfolio = {}
            hold_timer = 0
            continue

        need_rebal = (not portfolio and risk_on) or (hold_timer >= hold_days)
        if not need_rebal or not risk_on:
            continue

        sig = composite.iloc[i]
        valid = sig.dropna()
        if len(valid) < effective_top_n:
            continue
        top = valid.nlargest(effective_top_n).index.tolist()

        if set(top) == set(portfolio.keys()):
            continue

        if portfolio:
            nav *= (1 - FEE)
        portfolio = {c: price.loc[today, c] for c in top}
        nav *= (1 - FEE)
        hold_timer = 0

    return np.array(nav_hist)

def metrics(nav_hist, dates_sub, bench):
    eq = np.array(nav_hist)
    n = len(eq)
    years = n / 252
    total = eq[-1] / eq[0] - 1
    ann = (1 + total) ** (1/years) - 1
    daily_r = np.diff(eq) / eq[:-1]
    sharpe = np.mean(daily_r) / (np.std(daily_r) + 1e-8) * np.sqrt(252)
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    calmar = ann / abs(dd) if dd != 0 else 0

    # Per year
    yr_rets = {}
    start_idx = 0
    for yr in range(2021, 2026):
        yr_indices = [j for j in range(n) if dates_sub[j].year == yr]
        if len(yr_indices) >= 2:
            yr_ret = eq[yr_indices[-1]] / eq[yr_indices[0]] - 1
            yr_rets[yr] = yr_ret
    return ann, sharpe, dd, calmar, yr_rets

def main():
    print("Loading data...")
    df, shibor, macro, nav = load_data()
    print("Preparing signals (beta computation ~1min)...")
    price, codes, bench, dates, timing_on, sharpe_20d, rs_10d, shibor_tilt, premium, breadth = \
        prepare(df, shibor, macro, nav)

    # Z-score signals
    z_sharpe = row_zscore(sharpe_20d[codes])
    z_rs = row_zscore(rs_10d[codes])
    z_shibor = row_zscore(shibor_tilt[codes])
    z_premium = row_zscore(premium[codes])
    z_premium_adj = -z_premium  # discount is bullish

    # Strategy variants
    strategies = {
        'Phase3b baseline': (0.5*z_sharpe + 0.5*z_rs, breadth > 0.5, 3, 7, False),
        'Adaptive timing only': (0.5*z_sharpe + 0.5*z_rs, timing_on, 3, 7, False),
        'New signals only': (0.3*z_sharpe + 0.2*z_rs + 0.25*z_shibor + 0.25*z_premium_adj, breadth > 0.5, 3, 7, False),
        'Combined (no DD)': (0.3*z_sharpe + 0.2*z_rs + 0.25*z_shibor + 0.25*z_premium_adj, timing_on, 3, 7, False),
        'Combined + DD ctrl': (0.3*z_sharpe + 0.2*z_rs + 0.25*z_shibor + 0.25*z_premium_adj, timing_on, 3, 7, True),
        'Combined TOP-2': (0.3*z_sharpe + 0.2*z_rs + 0.25*z_shibor + 0.25*z_premium_adj, timing_on, 2, 7, False),
        'Combined hold-5d': (0.3*z_sharpe + 0.2*z_rs + 0.25*z_shibor + 0.25*z_premium_adj, timing_on, 3, 5, False),
        'Adaptive + Shibor': (0.4*z_sharpe + 0.3*z_rs + 0.3*z_shibor, timing_on, 3, 7, False),
        'Adaptive + Premium': (0.4*z_sharpe + 0.3*z_rs + 0.3*z_premium_adj, timing_on, 3, 7, False),
        'Adpt+Prem TOP-2': (0.4*z_sharpe + 0.3*z_rs + 0.3*z_premium_adj, timing_on, 2, 7, False),
        'Adpt+Prem TOP-1': (0.4*z_sharpe + 0.3*z_rs + 0.3*z_premium_adj, timing_on, 1, 7, False),
        'Adpt+Prem hold-5': (0.4*z_sharpe + 0.3*z_rs + 0.3*z_premium_adj, timing_on, 3, 5, False),
        'Adpt+Prem 0.5w': (0.3*z_sharpe + 0.2*z_rs + 0.5*z_premium_adj, timing_on, 3, 7, False),
        'Adpt+Prem+Shib': (0.3*z_sharpe + 0.2*z_rs + 0.25*z_premium_adj + 0.25*z_shibor, timing_on, 3, 7, False),
        'Adpt+Prem TOP2 h5': (0.4*z_sharpe + 0.3*z_rs + 0.3*z_premium_adj, timing_on, 2, 5, False),
    }

    dates_sub = dates[WARMUP:]
    print(f"\n{'='*90}")
    print(f"{'Strategy':<24} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} | 2021  2022  2023  2024  2025")
    print(f"{'='*90}")

    for name, (composite, timing_sig, top_n, hold_d, dd_ctrl) in strategies.items():
        # Override timing
        t_on = timing_sig if isinstance(timing_sig, pd.Series) else timing_sig
        nav_hist = run_backtest(price, codes, bench, dates, t_on, composite, top_n, hold_d, dd_ctrl)
        ann, sharpe, dd, calmar, yr_rets = metrics(nav_hist, dates_sub, bench)
        yr_str = " ".join(f"{yr_rets.get(y, 0)*100:>5.1f}" for y in range(2021, 2026))
        print(f"{name:<24} {ann*100:>7.1f}% {sharpe:>7.2f} {dd*100:>7.1f}% {calmar:>7.2f} | {yr_str}")

    print(f"{'='*90}")
    # Benchmark
    b_sub = bench.iloc[WARMUP:]
    b_total = b_sub.iloc[-1] / b_sub.iloc[0] - 1
    b_ann = (1 + b_total) ** (1/(len(b_sub)/252)) - 1
    print(f"{'510300 B&H':<24} {b_ann*100:>7.1f}%")

if __name__ == "__main__":
    main()

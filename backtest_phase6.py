"""
Phase 6: PCR (Put-Call Ratio) overlay on best Phase 5 strategy (Adaptive Credit + Premium).
PCR as market-level timing overlay: fear=concentrate, greed=diversify.
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
    pcr = pd.read_sql("SELECT date, underlying, pcr FROM option_pcr_daily", conn)
    pcr['date'] = pd.to_datetime(pcr['date'])
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
    return df, shibor, macro, nav, pcr

def prepare(df, shibor, macro, nav, pcr_raw):
    price = df.pivot(index='date', columns='code', values='close').sort_index()
    codes = sorted([c for c in price.columns if c != BENCHMARK])
    bench = price[BENCHMARK]
    dates = price.index
    ret = price[codes].pct_change()
    bench_ret = bench.pct_change()

    # === TIMING: Adaptive Credit ===
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

    # === SELECTION SIGNALS ===
    sharpe_20d = ret.rolling(20).mean() / (ret.rolling(20).std() + 1e-8)
    rs_10d = price[codes].pct_change(10).sub(bench.pct_change(10), axis=0)

    # ETF Premium/Discount
    nav_pivot = nav.pivot(index='date', columns='code', values='nav').sort_index()
    premium = pd.DataFrame(0.0, index=dates, columns=codes)
    for c in codes:
        if c in nav_pivot.columns:
            nav_aligned = nav_pivot[c].reindex(dates)
            prem = (price[c] - nav_aligned) / (nav_aligned + 1e-8)
            premium[c] = prem.rolling(5).mean()

    # === PCR Z-SCORE ===
    # Combined PCR from 50ETF and 300ETF
    pcr_50 = pcr_raw[pcr_raw['underlying'] == '50ETF'].set_index('date')['pcr']
    pcr_300 = pcr_raw[pcr_raw['underlying'] == '300ETF'].set_index('date')['pcr']
    combined_pcr = pd.DataFrame({'p50': pcr_50, 'p300': pcr_300}).mean(axis=1)
    combined_pcr = combined_pcr.reindex(dates).ffill()
    pcr_z = (combined_pcr - combined_pcr.rolling(60, min_periods=30).mean()) / \
            (combined_pcr.rolling(60, min_periods=30).std() + 1e-8)

    return price, codes, bench, dates, timing_on, sharpe_20d, rs_10d, premium, pcr_z

def row_zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)


def metrics(nav_hist, dates_sub):
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
    yr_rets = {}
    for yr in range(2021, 2026):
        yr_indices = [j for j in range(n) if dates_sub[j].year == yr]
        if len(yr_indices) >= 2:
            yr_rets[yr] = eq[yr_indices[-1]] / eq[yr_indices[0]] - 1
    return ann, sharpe, dd, calmar, yr_rets

def run_backtest_v2(price, codes, dates, timing_on, composite, pcr_z,
                    top_n=3, hold_days=7, pcr_sizing=False, pcr_filter=False,
                    fear_thresh=1.0, greed_thresh=-1.0, filter_thresh=-1.5,
                    pcr_momentum=False):
    """Extended version with configurable thresholds and PCR momentum."""
    nav_val = 1.0
    portfolio = {}
    hold_timer = 0
    nav_hist = []

    for i in range(WARMUP, len(dates)):
        today = dates[i]
        if portfolio:
            daily_ret = sum((price.loc[today, c] / portfolio[c] - 1) for c in portfolio) / len(portfolio)
            nav_val *= (1 + daily_ret)
            portfolio = {c: price.loc[today, c] for c in portfolio}

        nav_hist.append(nav_val)
        hold_timer += 1
        risk_on = bool(timing_on.iloc[i])

        pz = pcr_z.iloc[i] if i < len(pcr_z) and not np.isnan(pcr_z.iloc[i]) else 0.0

        # PCR momentum: rising PCR (5d delta > 0) is bullish signal
        if pcr_momentum and i >= 5:
            pz_prev = pcr_z.iloc[i-5] if not np.isnan(pcr_z.iloc[i-5]) else pz
            pcr_rising = pz > pz_prev
        else:
            pcr_rising = False

        if pcr_filter and pz < filter_thresh:
            risk_on = False
        # PCR momentum boost: if PCR rising fast, override timing off
        if pcr_momentum and pcr_rising and pz > 0.5 and not risk_on:
            risk_on = bool(timing_on.iloc[i])  # don't force, just don't block

        if portfolio and not risk_on:
            nav_val *= (1 - FEE)
            portfolio = {}
            hold_timer = 0
            continue

        need_rebal = (not portfolio and risk_on) or (hold_timer >= hold_days)
        if not need_rebal or not risk_on:
            continue

        effective_top_n = top_n
        if pcr_sizing and not np.isnan(pz):
            if pz > fear_thresh:
                effective_top_n = max(1, top_n - 1)
            elif pz < greed_thresh:
                effective_top_n = min(top_n + 2, len(codes))

        sig = composite.iloc[i]
        valid = sig.dropna()
        if len(valid) < effective_top_n:
            continue
        top = valid.nlargest(effective_top_n).index.tolist()

        if set(top) == set(portfolio.keys()):
            continue

        if portfolio:
            nav_val *= (1 - FEE)
        portfolio = {c: price.loc[today, c] for c in top}
        nav_val *= (1 - FEE)
        hold_timer = 0

    return np.array(nav_hist)

def main():
    print("Loading data...")
    df, shibor, macro, nav, pcr_raw = load_data()
    print("Preparing signals...")
    price, codes, bench, dates, timing_on, sharpe_20d, rs_10d, premium, pcr_z = \
        prepare(df, shibor, macro, nav, pcr_raw)

    z_sharpe = row_zscore(sharpe_20d[codes])
    z_rs = row_zscore(rs_10d[codes])
    z_premium_adj = -row_zscore(premium[codes])

    composite = 0.4*z_sharpe + 0.3*z_rs + 0.3*z_premium_adj

    # PCR stats
    valid_pcr = pcr_z.dropna()
    print(f"\nPCR z-score stats: mean={valid_pcr.mean():.3f}, std={valid_pcr.std():.3f}")
    print(f"  PCR_z > 1.0 (fear) days: {(valid_pcr > 1.0).sum()}")
    print(f"  PCR_z > 0.5 days: {(valid_pcr > 0.5).sum()}")
    print(f"  PCR_z < -0.5 days: {(valid_pcr < -0.5).sum()}")
    print(f"  PCR_z < -1.0 (greed) days: {(valid_pcr < -1.0).sum()}")
    print(f"  PCR_z < -1.5 (extreme greed) days: {(valid_pcr < -1.5).sum()}")

    dates_sub = dates[WARMUP:]

    # === PART 1: Original thresholds ===
    print(f"\n{'='*95}")
    print("PART 1: Standard thresholds (fear>1.0, greed<-1.0, filter<-1.5)")
    print(f"{'='*95}")
    print(f"{'Strategy':<28} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} | 2021  2022  2023  2024  2025")
    print(f"{'-'*95}")

    configs_1 = [
        ("Baseline TOP-3", 3, False, False, 1.0, -1.0, -1.5, False),
        ("+PCR sizing TOP-3", 3, True, False, 1.0, -1.0, -1.5, False),
        ("+PCR filter TOP-3", 3, False, True, 1.0, -1.0, -1.5, False),
        ("+PCR both TOP-3", 3, True, True, 1.0, -1.0, -1.5, False),
        ("Baseline TOP-2", 2, False, False, 1.0, -1.0, -1.5, False),
        ("+PCR sizing TOP-2", 2, True, False, 1.0, -1.0, -1.5, False),
        ("+PCR filter TOP-2", 2, False, True, 1.0, -1.0, -1.5, False),
        ("+PCR both TOP-2", 2, True, True, 1.0, -1.0, -1.5, False),
    ]

    for name, top_n, sizing, filt, ft, gt, filt_t, mom in configs_1:
        nav_hist = run_backtest_v2(price, codes, dates, timing_on, composite, pcr_z,
                                   top_n=top_n, hold_days=7, pcr_sizing=sizing,
                                   pcr_filter=filt, fear_thresh=ft, greed_thresh=gt,
                                   filter_thresh=filt_t, pcr_momentum=mom)
        ann, sharpe, dd, calmar, yr_rets = metrics(nav_hist, dates_sub)
        yr_str = " ".join(f"{yr_rets.get(y, 0)*100:>5.1f}" for y in range(2021, 2026))
        print(f"{name:<28} {ann*100:>7.1f}% {sharpe:>7.2f} {dd*100:>7.1f}% {calmar:>7.2f} | {yr_str}")

    # === PART 2: Softer thresholds ===
    print(f"\n{'='*95}")
    print("PART 2: Softer thresholds (fear>0.5, greed<-0.5, filter<-1.0)")
    print(f"{'='*95}")
    print(f"{'Strategy':<28} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} | 2021  2022  2023  2024  2025")
    print(f"{'-'*95}")

    configs_2 = [
        ("Baseline TOP-3", 3, False, False, 0.5, -0.5, -1.0, False),
        ("+PCR sizing TOP-3", 3, True, False, 0.5, -0.5, -1.0, False),
        ("+PCR filter TOP-3", 3, False, True, 0.5, -0.5, -1.0, False),
        ("+PCR both TOP-3", 3, True, True, 0.5, -0.5, -1.0, False),
        ("Baseline TOP-2", 2, False, False, 0.5, -0.5, -1.0, False),
        ("+PCR sizing TOP-2", 2, True, False, 0.5, -0.5, -1.0, False),
        ("+PCR filter TOP-2", 2, False, True, 0.5, -0.5, -1.0, False),
        ("+PCR both TOP-2", 2, True, True, 0.5, -0.5, -1.0, False),
    ]

    for name, top_n, sizing, filt, ft, gt, filt_t, mom in configs_2:
        nav_hist = run_backtest_v2(price, codes, dates, timing_on, composite, pcr_z,
                                   top_n=top_n, hold_days=7, pcr_sizing=sizing,
                                   pcr_filter=filt, fear_thresh=ft, greed_thresh=gt,
                                   filter_thresh=filt_t, pcr_momentum=mom)
        ann, sharpe, dd, calmar, yr_rets = metrics(nav_hist, dates_sub)
        yr_str = " ".join(f"{yr_rets.get(y, 0)*100:>5.1f}" for y in range(2021, 2026))
        print(f"{name:<28} {ann*100:>7.1f}% {sharpe:>7.2f} {dd*100:>7.1f}% {calmar:>7.2f} | {yr_str}")

    # === PART 3: PCR momentum ===
    print(f"\n{'='*95}")
    print("PART 3: PCR momentum (rising PCR = contrarian bullish)")
    print(f"{'='*95}")
    print(f"{'Strategy':<28} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} | 2021  2022  2023  2024  2025")
    print(f"{'-'*95}")

    configs_3 = [
        ("+PCR momentum TOP-3", 3, False, False, 1.0, -1.0, -1.5, True),
        ("+PCR mom+sizing TOP-3", 3, True, False, 0.5, -0.5, -1.0, True),
        ("+PCR mom+filter TOP-3", 3, False, True, 0.5, -0.5, -1.0, True),
        ("+PCR all TOP-3", 3, True, True, 0.5, -0.5, -1.0, True),
        ("+PCR momentum TOP-2", 2, False, False, 1.0, -1.0, -1.5, True),
        ("+PCR all TOP-2", 2, True, True, 0.5, -0.5, -1.0, True),
    ]

    for name, top_n, sizing, filt, ft, gt, filt_t, mom in configs_3:
        nav_hist = run_backtest_v2(price, codes, dates, timing_on, composite, pcr_z,
                                   top_n=top_n, hold_days=7, pcr_sizing=sizing,
                                   pcr_filter=filt, fear_thresh=ft, greed_thresh=gt,
                                   filter_thresh=filt_t, pcr_momentum=mom)
        ann, sharpe, dd, calmar, yr_rets = metrics(nav_hist, dates_sub)
        yr_str = " ".join(f"{yr_rets.get(y, 0)*100:>5.1f}" for y in range(2021, 2026))
        print(f"{name:<28} {ann*100:>7.1f}% {sharpe:>7.2f} {dd*100:>7.1f}% {calmar:>7.2f} | {yr_str}")

    print(f"\n{'='*95}")
    b_sub = bench.iloc[WARMUP:]
    b_total = b_sub.iloc[-1] / b_sub.iloc[0] - 1
    b_ann = (1 + b_total) ** (1/(len(b_sub)/252)) - 1
    print(f"{'510300 B&H':<28} {b_ann*100:>7.1f}%")

if __name__ == "__main__":
    main()

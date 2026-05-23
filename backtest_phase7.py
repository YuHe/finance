"""
Phase 7: Build on Phase 5 best (Adaptive Credit + Premium) with new signals:
1. ETF-level margin rank (cross-sectional from semi-annual snapshots)
2. Margin concentration (favor ETFs with rising margin share)
3. Aggregate margin position sizing (soft scaling, not binary)
4. Momentum decay filter (penalize overextended ETFs)
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
    margin_sector = pd.read_sql("SELECT * FROM sector_margin_snapshot", conn)
    margin_agg = pd.read_sql("SELECT * FROM margin_daily", conn)
    margin_agg['date'] = pd.to_datetime(margin_agg['date'])
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
    return df, shibor, macro, nav, margin_sector, margin_agg


def prepare(df, shibor, macro, nav, margin_sector, margin_agg):
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

    # === SELECTION SIGNALS (Phase 5 baseline) ===
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

    # === NEW SIGNAL 1: ETF-level margin rank ===
    # Semi-annual snapshots -> cross-sectional z-score of margin balance, forward-filled
    margin_sector['date'] = pd.to_datetime(margin_sector['信用交易日期'], format='%Y%m%d')
    margin_rank = pd.DataFrame(0.0, index=dates, columns=codes)
    snapshot_dates = sorted(margin_sector['date'].unique())
    for sd in snapshot_dates:
        snap = margin_sector[margin_sector['date'] == sd]
        snap_codes = snap[snap['标的证券代码'].isin(codes)]
        if len(snap_codes) < 3:
            continue
        bal = snap_codes.set_index('标的证券代码')['融资余额'].astype(float)
        # z-score across available ETFs
        z = (bal - bal.mean()) / (bal.std() + 1e-8)
        for c in z.index:
            if c in codes:
                margin_rank.loc[sd:, c] = z[c]  # forward fill from snapshot

    # === NEW SIGNAL 2: Margin concentration (rising share) ===
    # Change in margin share between consecutive snapshots
    margin_conc = pd.DataFrame(0.0, index=dates, columns=codes)
    prev_snap = None
    for sd in snapshot_dates:
        snap = margin_sector[margin_sector['date'] == sd]
        snap_codes = snap[snap['标的证券代码'].isin(codes)]
        if len(snap_codes) < 3:
            prev_snap = snap_codes
            continue
        bal = snap_codes.set_index('标的证券代码')['融资余额'].astype(float)
        share = bal / (bal.sum() + 1e-8)
        if prev_snap is not None and len(prev_snap) > 0:
            prev_bal = prev_snap.set_index('标的证券代码')['融资余额'].astype(float)
            prev_share = prev_bal / (prev_bal.sum() + 1e-8)
            common = share.index.intersection(prev_share.index)
            if len(common) >= 3:
                delta_share = share[common] - prev_share[common]
                z = (delta_share - delta_share.mean()) / (delta_share.std() + 1e-8)
                for c in z.index:
                    margin_conc.loc[sd:, c] = z[c]
        prev_snap = snap_codes

    # === NEW SIGNAL 3: Aggregate margin position sizing ===
    # z-score of 60d margin balance change
    ma = margin_agg.set_index('date')['margin_balance'].sort_index()
    ma_z = (ma - ma.rolling(60).mean()) / (ma.rolling(60).std() + 1e-8)
    ma_z_daily = ma_z.reindex(dates).ffill()
    # Position scale: 1.0 when z<0.5, 0.7 when 0.5<z<1.5, 0.5 when z>1.5
    pos_scale = pd.Series(1.0, index=dates)
    pos_scale[ma_z_daily > 0.5] = 0.7
    pos_scale[ma_z_daily > 1.5] = 0.5

    # Soft position scale: 1.0 when z<0.5, 0.85 when 0.5<z<1.5, 0.7 when z>1.5
    pos_scale_soft = pd.Series(1.0, index=dates)
    pos_scale_soft[ma_z_daily > 0.5] = 0.85
    pos_scale_soft[ma_z_daily > 1.5] = 0.7

    # === NEW SIGNAL 4: Momentum decay filter ===
    # Penalize ETFs whose 5d return > 3x their 20d avg daily return
    ret_5d = price[codes].pct_change(5)
    avg_daily_20d = ret.rolling(20).mean()
    # overextended = 5d return > 3 * (20d avg daily * 5)
    overextended = ret_5d > 3 * (avg_daily_20d * 5)
    decay_penalty = pd.DataFrame(0.0, index=dates, columns=codes)
    decay_penalty[overextended] = -1.0  # penalty z-score units

    return (price, codes, bench, dates, timing_on, sharpe_20d, rs_10d,
            premium, margin_rank, margin_conc, pos_scale, pos_scale_soft, decay_penalty, breadth)


def row_zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)


def run_backtest(price, codes, bench, dates, timing_on, composite, top_n=3,
                 hold_days=7, pos_scale=None):
    nav = 1.0
    portfolio = {}
    hold_timer = 0
    nav_hist = []

    for i in range(WARMUP, len(dates)):
        today = dates[i]
        if portfolio:
            daily_ret = sum((price.loc[today, c] / portfolio[c] - 1) for c in portfolio) / len(portfolio)
            # Apply position scaling
            scale = pos_scale.iloc[i] if pos_scale is not None else 1.0
            nav *= (1 + daily_ret * scale)
            portfolio = {c: price.loc[today, c] for c in portfolio}

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
        if len(valid) < top_n:
            continue
        top = valid.nlargest(top_n).index.tolist()

        if set(top) == set(portfolio.keys()):
            continue

        if portfolio:
            nav *= (1 - FEE)
        portfolio = {c: price.loc[today, c] for c in top}
        nav *= (1 - FEE)
        hold_timer = 0

    return np.array(nav_hist)


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


def main():
    print("Loading data...")
    df, shibor, macro, nav_data, margin_sector, margin_agg = load_data()
    print("Preparing signals...")
    (price, codes, bench, dates, timing_on, sharpe_20d, rs_10d,
     premium, margin_rank, margin_conc, pos_scale, pos_scale_soft, decay_penalty, breadth) = \
        prepare(df, shibor, macro, nav_data, margin_sector, margin_agg)

    # Z-score signals
    z_sharpe = row_zscore(sharpe_20d[codes])
    z_rs = row_zscore(rs_10d[codes])
    z_premium_adj = -row_zscore(premium[codes])
    z_margin_rank = row_zscore(margin_rank[codes])
    z_margin_conc = row_zscore(margin_conc[codes])
    z_decay = decay_penalty[codes]  # already in z-score-like units

    # Phase 5 baseline composite
    base_composite = 0.4*z_sharpe + 0.3*z_rs + 0.3*z_premium_adj

    # Strategy variants
    strategies = [
        # (name, composite, top_n, hold_days, use_pos_scale)
        # --- Phase 5 baselines ---
        ("P5 Baseline TOP-3", base_composite, 3, 7, False),
        ("P5 Baseline TOP-2", base_composite, 2, 7, False),
        ("P5 Baseline TOP-1", base_composite, 1, 7, False),
        # --- Margin Concentration (best new signal from round 1) ---
        ("+MC 0.1 TOP-3", 0.35*z_sharpe + 0.25*z_rs + 0.3*z_premium_adj + 0.1*z_margin_conc, 3, 7, False),
        ("+MC 0.15 TOP-3", 0.33*z_sharpe + 0.22*z_rs + 0.3*z_premium_adj + 0.15*z_margin_conc, 3, 7, False),
        ("+MC 0.2 TOP-3", 0.3*z_sharpe + 0.2*z_rs + 0.3*z_premium_adj + 0.2*z_margin_conc, 3, 7, False),
        ("+MC 0.15 TOP-2", 0.33*z_sharpe + 0.22*z_rs + 0.3*z_premium_adj + 0.15*z_margin_conc, 2, 7, False),
        ("+MC 0.15 TOP-1", 0.33*z_sharpe + 0.22*z_rs + 0.3*z_premium_adj + 0.15*z_margin_conc, 1, 7, False),
        ("+MC 0.2 TOP-2", 0.3*z_sharpe + 0.2*z_rs + 0.3*z_premium_adj + 0.2*z_margin_conc, 2, 7, False),
        ("+MC 0.2 TOP-1", 0.3*z_sharpe + 0.2*z_rs + 0.3*z_premium_adj + 0.2*z_margin_conc, 1, 7, False),
        # --- Decay filter variants ---
        ("+Decay 0.15 TOP-3", base_composite + 0.15*z_decay, 3, 7, False),
        ("+Decay 0.15 TOP-2", base_composite + 0.15*z_decay, 2, 7, False),
        ("+Decay 0.15 TOP-1", base_composite + 0.15*z_decay, 1, 7, False),
        # --- MC + Decay combined ---
        ("MC+Decay TOP-3", 0.33*z_sharpe + 0.22*z_rs + 0.25*z_premium_adj + 0.1*z_margin_conc + 0.1*z_decay, 3, 7, False),
        ("MC+Decay TOP-2", 0.33*z_sharpe + 0.22*z_rs + 0.25*z_premium_adj + 0.1*z_margin_conc + 0.1*z_decay, 2, 7, False),
        ("MC+Decay TOP-1", 0.33*z_sharpe + 0.22*z_rs + 0.25*z_premium_adj + 0.1*z_margin_conc + 0.1*z_decay, 1, 7, False),
        # --- Soft position scaling (lighter: 1.0/0.85/0.7) ---
        ("P5 + SoftPS TOP-3", base_composite, 3, 7, "soft"),
        ("P5 + SoftPS TOP-2", base_composite, 2, 7, "soft"),
        ("P5 + SoftPS TOP-1", base_composite, 1, 7, "soft"),
        # --- Best combo + soft PS ---
        ("MC+Decay+SPS TOP-3", 0.33*z_sharpe + 0.22*z_rs + 0.25*z_premium_adj + 0.1*z_margin_conc + 0.1*z_decay, 3, 7, "soft"),
        ("MC+Decay+SPS TOP-2", 0.33*z_sharpe + 0.22*z_rs + 0.25*z_premium_adj + 0.1*z_margin_conc + 0.1*z_decay, 2, 7, "soft"),
        ("MC+Decay+SPS TOP-1", 0.33*z_sharpe + 0.22*z_rs + 0.25*z_premium_adj + 0.1*z_margin_conc + 0.1*z_decay, 1, 7, "soft"),
        # --- Hold period 5d with MC ---
        ("+MC 0.15 TOP-2 h5", 0.33*z_sharpe + 0.22*z_rs + 0.3*z_premium_adj + 0.15*z_margin_conc, 2, 5, False),
        ("+MC 0.15 TOP-1 h5", 0.33*z_sharpe + 0.22*z_rs + 0.3*z_premium_adj + 0.15*z_margin_conc, 1, 5, False),
    ]

    dates_sub = dates[WARMUP:]
    print(f"\n{'='*100}")
    print(f"{'Strategy':<22} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} | 2021   2022   2023   2024   2025")
    print(f"{'='*100}")

    for name, composite, top_n, hold_d, use_ps in strategies:
        ps = pos_scale_soft if use_ps == "soft" else (pos_scale if use_ps else None)
        nav_hist = run_backtest(price, codes, bench, dates, timing_on, composite, top_n, hold_d, ps)
        ann, sharpe, dd, calmar, yr_rets = metrics(nav_hist, dates_sub)
        yr_str = " ".join(f"{yr_rets.get(y, 0)*100:>6.1f}" for y in range(2021, 2026))
        print(f"{name:<22} {ann*100:>7.1f}% {sharpe:>7.2f} {dd*100:>7.1f}% {calmar:>7.2f} | {yr_str}")

    print(f"{'='*100}")
    b_sub = bench.iloc[WARMUP:]
    b_total = b_sub.iloc[-1] / b_sub.iloc[0] - 1
    b_ann = (1 + b_total) ** (1/(len(b_sub)/252)) - 1
    print(f"{'510300 B&H':<22} {b_ann*100:>7.1f}%")


if __name__ == "__main__":
    main()

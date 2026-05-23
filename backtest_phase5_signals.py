"""
Phase 5: Orthogonal Cross-Sectional Signals for A-share Sector ETF Rotation
New signals: Residual Momentum, Volume-Price Divergence, QVIX Fear Spread,
             Shibor Term Spread, ETF Premium/Discount
"""
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats

# ─── Data Loading ───────────────────────────────────────────────────────────
def load_data():
    conn_bt = sqlite3.connect('/Users/heyu11/Code/finance/data_layer/backtest_adjusted.db')
    etf = pd.read_sql("SELECT code, date, open, high, low, close, volume FROM etf_daily", conn_bt)
    conn_bt.close()

    conn_sig = sqlite3.connect('/Users/heyu11/Code/finance/data_layer/signals.db')
    shibor = pd.read_sql("SELECT * FROM shibor_daily", conn_sig)
    qvix = pd.read_sql("SELECT date, symbol, qvix FROM option_qvix_daily", conn_sig)
    nav = pd.read_sql("SELECT date, code, nav FROM etf_nav", conn_sig)
    conn_sig.close()

    etf['date'] = pd.to_datetime(etf['date'])
    shibor['date'] = pd.to_datetime(shibor['date'])
    qvix['date'] = pd.to_datetime(qvix['date'])
    nav['date'] = pd.to_datetime(nav['date'])
    return etf, shibor, qvix, nav

# ─── Signal Computation ─────────────────────────────────────────────────────
def compute_signals(etf, shibor, qvix, nav):
    codes = sorted([c for c in etf['code'].unique() if c != '510300'])
    dates = sorted(etf[etf['code'] == '510300']['date'].unique())

    # Pivot price/volume
    price = etf.pivot(index='date', columns='code', values='close').sort_index()
    volume = etf.pivot(index='date', columns='code', values='volume').sort_index()

    bench_ret = price['510300'].pct_change()
    ret = price[codes].pct_change()

    # --- Sharpe_20d & RS_10d (baseline) ---
    sharpe_20d = ret.rolling(20).mean() / ret.rolling(20).std()
    rs_10d = ret.rolling(10).sum()

    # --- 1. Residual Momentum (vectorized via rolling OLS) ---
    resid_mom = pd.DataFrame(index=price.index, columns=codes, dtype=float)
    bx_arr = bench_ret.values
    for c in codes:
        y_arr = ret[c].values
        n = len(y_arr)
        for i in range(79, n):
            x60 = bx_arr[i-59:i+1]
            y60 = y_arr[i-59:i+1]
            mask = ~(np.isnan(x60) | np.isnan(y60))
            if mask.sum() < 20:
                continue
            xm, ym = x60[mask], y60[mask]
            sx, sy, sxx, sxy = xm.sum(), ym.sum(), (xm*xm).sum(), (xm*ym).sum()
            nm = mask.sum()
            denom = nm*sxx - sx*sx
            if abs(denom) < 1e-15:
                continue
            slope = (nm*sxy - sx*sy) / denom
            intercept = (sy - slope*sx) / nm
            y20 = y_arr[i-19:i+1]
            x20 = bx_arr[i-19:i+1]
            residuals = y20 - (slope * x20 + intercept)
            resid_mom.iat[i, codes.index(c)] = np.nansum(residuals)

    # --- 2. Volume-Price Divergence ---
    price_ret_20 = price[codes].pct_change(20)
    vol_chg_20 = volume[codes].pct_change(20)
    # Cross-sectional z-score
    def cs_zscore(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    vol_price_div = cs_zscore(price_ret_20) - cs_zscore(vol_chg_20)

    # --- 3. QVIX Fear Spread ---
    qvix_50 = qvix[qvix['symbol'] == '50ETF'].set_index('date')['qvix'].sort_index()
    qvix_300 = qvix[qvix['symbol'] == '300ETF'].set_index('date')['qvix'].sort_index()
    fear_spread_raw = qvix_50 - qvix_300
    fear_spread_z = (fear_spread_raw - fear_spread_raw.rolling(60).mean()) / fear_spread_raw.rolling(60).std()

    # Compute trailing 60d beta for each ETF (vectorized)
    beta_60 = pd.DataFrame(index=price.index, columns=codes, dtype=float)
    bx_full = bench_ret.values
    for c in codes:
        y_full = ret[c].values
        ci = codes.index(c)
        for i in range(59, len(price)):
            x60 = bx_full[i-59:i+1]
            y60 = y_full[i-59:i+1]
            mask = ~(np.isnan(x60) | np.isnan(y60))
            if mask.sum() < 20:
                continue
            xm, ym = x60[mask], y60[mask]
            cov = np.mean((xm - xm.mean()) * (ym - ym.mean()))
            var = np.var(xm)
            if var > 1e-15:
                beta_60.iat[i, ci] = cov / var

    # QVIX tilt: when fear_spread_z high, penalize high-beta ETFs
    qvix_tilt = pd.DataFrame(index=price.index, columns=codes, dtype=float)
    for d in price.index:
        if d in fear_spread_z.index and not np.isnan(fear_spread_z.get(d, np.nan)):
            fz = fear_spread_z[d]
            betas = beta_60.loc[d]
            # Negative score for high-beta when fear is high
            beta_z = (betas - betas.mean()) / (betas.std() + 1e-8)
            qvix_tilt.loc[d] = -fz * beta_z

    # --- 4. Shibor Term Spread ---
    shibor_ts = shibor.set_index('date').sort_index()
    term_spread = shibor_ts['three_month'] - shibor_ts['overnight']
    term_spread_z = (term_spread - term_spread.rolling(60).mean()) / term_spread.rolling(60).std()

    # Shibor tilt: steep (positive z) favors high-beta
    shibor_tilt = pd.DataFrame(index=price.index, columns=codes, dtype=float)
    for d in price.index:
        if d in term_spread_z.index and not np.isnan(term_spread_z.get(d, np.nan)):
            tz = term_spread_z[d]
            betas = beta_60.loc[d]
            beta_z = (betas - betas.mean()) / (betas.std() + 1e-8)
            shibor_tilt.loc[d] = tz * beta_z

    # --- 5. ETF Premium/Discount ---
    nav_pivot = nav.pivot(index='date', columns='code', values='nav').sort_index()
    common_codes = [c for c in codes if c in nav_pivot.columns]
    premium = pd.DataFrame(index=price.index, columns=codes, dtype=float)
    for c in common_codes:
        nav_aligned = nav_pivot[c].reindex(price.index)
        prem = (price[c] - nav_aligned) / nav_aligned
        premium[c] = prem.rolling(5).mean()

    # Market breadth timing (>50% ETFs above 20d MA)
    ma20 = price[codes].rolling(20).mean()
    breadth = (price[codes] > ma20).sum(axis=1) / len(codes)

    return {
        'sharpe_20d': sharpe_20d, 'rs_10d': rs_10d,
        'resid_mom': resid_mom, 'vol_price_div': vol_price_div,
        'qvix_tilt': qvix_tilt, 'shibor_tilt': shibor_tilt,
        'premium': premium, 'breadth': breadth,
        'price': price, 'ret': ret, 'codes': codes
    }

# ─── Cross-sectional z-score (per row) ──────────────────────────────────────
def row_zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)

# ─── IC Computation ─────────────────────────────────────────────────────────
def compute_ic(signal_df, fwd_ret_df, codes):
    """Spearman rank IC of signal vs 7-day forward return."""
    common_dates = signal_df.dropna(how='all').index.intersection(fwd_ret_df.dropna(how='all').index)
    ics = []
    for d in common_dates:
        s = signal_df.loc[d, codes]
        f = fwd_ret_df.loc[d, codes]
        mask = s.notna() & f.notna()
        if mask.sum() >= 5:
            ic, _ = stats.spearmanr(s[mask], f[mask])
            ics.append(ic)
    ics = np.array(ics)
    mean_ic = np.nanmean(ics)
    t_stat = mean_ic / (np.nanstd(ics) / np.sqrt(len(ics))) if len(ics) > 1 else 0
    return mean_ic, t_stat, ics

# ─── Backtest Engine ────────────────────────────────────────────────────────
def run_backtest(composite_signal, breadth, price, codes, start='2021-08-16', end='2025-12-31',
                 hold_days=7, top_n=3, fee=0.001):
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    dates = price.loc[start_dt:end_dt].index.tolist()

    portfolio_value = 1.0
    values = []
    hold_counter = 0
    current_holdings = []
    prev_breadth_ok = True

    for i, d in enumerate(dates):
        if i == 0:
            values.append(portfolio_value)
            continue

        # Daily return from holdings
        if current_holdings:
            day_ret = price[current_holdings].loc[d] / price[current_holdings].loc[dates[i-1]] - 1
            portfolio_value *= (1 + day_ret.mean())

        hold_counter += 1
        breadth_ok = breadth.get(d, 0.5) > 0.5 if d in breadth.index else prev_breadth_ok

        # Rebalance conditions
        rebalance = False
        if hold_counter >= hold_days:
            rebalance = True
        if breadth_ok != prev_breadth_ok:
            rebalance = True

        if rebalance and i < len(dates) - 1:
            # T+1: use signal from today, execute tomorrow
            sig_today = composite_signal.loc[d, codes] if d in composite_signal.index else pd.Series(dtype=float)
            valid = sig_today.dropna()
            if len(valid) >= top_n and breadth_ok:
                new_holdings = valid.nlargest(top_n).index.tolist()
            elif not breadth_ok:
                new_holdings = []
            else:
                new_holdings = current_holdings

            if set(new_holdings) != set(current_holdings):
                # Fee for turnover
                turnover = len(set(new_holdings) - set(current_holdings)) / max(top_n, 1)
                portfolio_value *= (1 - 2 * fee * turnover)
                current_holdings = new_holdings
                hold_counter = 0

        prev_breadth_ok = breadth_ok
        values.append(portfolio_value)

    result = pd.Series(values, index=dates)
    return result

# ─── Performance Metrics ────────────────────────────────────────────────────
def calc_metrics(equity_curve):
    rets = equity_curve.pct_change().dropna()
    n_years = len(rets) / 242
    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = rets.std() * np.sqrt(242)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = drawdown.min()
    return {'AnnRet': f"{ann_ret:.1%}", 'Sharpe': f"{sharpe:.2f}", 'MaxDD': f"{max_dd:.1%}",
            'ann_ret_val': ann_ret, 'sharpe_val': sharpe}

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    etf, shibor, qvix, nav = load_data()

    print("Computing signals (this may take a few minutes)...")
    sig = compute_signals(etf, shibor, qvix, nav)

    codes = sig['codes']
    price = sig['price']
    breadth = sig['breadth']

    # 7-day forward return
    fwd_ret_7d = price[codes].pct_change(7).shift(-7)

    # Z-score all signals cross-sectionally
    z_sharpe = row_zscore(sig['sharpe_20d'][codes])
    z_rs = row_zscore(sig['rs_10d'][codes])
    z_resid = row_zscore(sig['resid_mom'][codes].astype(float))
    z_volprice = row_zscore(sig['vol_price_div'][codes])
    z_qvix = row_zscore(sig['qvix_tilt'][codes].astype(float))
    z_shibor = row_zscore(sig['shibor_tilt'][codes].astype(float))
    z_premium = row_zscore(sig['premium'][codes].astype(float))

    # ─── IC Analysis ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SIGNAL IC ANALYSIS (Spearman rank IC vs 7d forward return)")
    print("="*70)
    signal_names = ['Sharpe_20d', 'RS_10d', 'ResidMom', 'VolPriceDiverg', 'QVIX_Tilt', 'Shibor_Tilt', 'Premium']
    signal_dfs = [z_sharpe, z_rs, z_resid, z_volprice, z_qvix, z_shibor, z_premium]

    ic_results = {}
    print(f"{'Signal':<16} {'IC':>8} {'t-stat':>8} {'Corr w/ Sharpe':>15} {'Corr w/ RS':>12}")
    print("-"*60)
    for name, sdf in zip(signal_names, signal_dfs):
        ic, t, _ = compute_ic(sdf, fwd_ret_7d, codes)
        # Correlation with baseline signals (time-series of cross-sectional means)
        flat_s = sdf.stack()
        flat_sharpe = z_sharpe.stack()
        flat_rs = z_rs.stack()
        common = flat_s.dropna().index.intersection(flat_sharpe.dropna().index)
        corr_sharpe = flat_s[common].corr(flat_sharpe[common]) if len(common) > 100 else np.nan
        common2 = flat_s.dropna().index.intersection(flat_rs.dropna().index)
        corr_rs = flat_s[common2].corr(flat_rs[common2]) if len(common2) > 100 else np.nan
        ic_results[name] = ic
        print(f"{name:<16} {ic:>8.4f} {t:>8.2f} {corr_sharpe:>15.3f} {corr_rs:>12.3f}")

    # ─── Composite Signals ──────────────────────────────────────────────────
    # Flip signals with negative IC so higher = better expected return
    z_premium_adj = -z_premium  # discount (negative premium) is bullish
    z_qvix_adj = -z_qvix       # negative QVIX tilt is bullish

    composites = {
        'Baseline (3b)': 0.5*z_sharpe + 0.5*z_rs,
        '+ResidMom': 0.4*z_sharpe + 0.3*z_rs + 0.3*z_resid,
        '+VolPrice': 0.4*z_sharpe + 0.3*z_rs + 0.3*z_volprice,
        '+Shibor': 0.4*z_sharpe + 0.3*z_rs + 0.3*z_shibor,
        '+Premium(-)': 0.4*z_sharpe + 0.3*z_rs + 0.3*z_premium_adj,
        '+QVIX(-)': 0.4*z_sharpe + 0.3*z_rs + 0.3*z_qvix_adj,
        'All New': 0.3*z_sharpe + 0.2*z_rs + 0.15*z_volprice + 0.15*z_shibor + 0.1*z_premium_adj + 0.1*z_qvix_adj,
        'Shibor+Prem': 0.35*z_sharpe + 0.25*z_rs + 0.2*z_shibor + 0.2*z_premium_adj,
    }

    # Best 3 by |IC| with sign correction (use adjusted signals directly)
    new_signals = {k: v for k, v in ic_results.items() if k not in ['Sharpe_20d', 'RS_10d']}
    top3_names = sorted(new_signals, key=lambda x: abs(new_signals[x]), reverse=True)[:3]
    print(f"\nTop 3 new signals by |IC|: {top3_names}")
    # All adjusted signals already have correct sign (higher = better)
    top3_map = {'ResidMom': z_resid, 'VolPriceDiverg': z_volprice, 'QVIX_Tilt': z_qvix_adj,
                'Shibor_Tilt': z_shibor, 'Premium': z_premium_adj}
    best3_composite = sum(top3_map[n] for n in top3_names if n in top3_map) / 3
    composites['Best3 New'] = 0.4*z_sharpe + 0.2*z_rs + 0.4*best3_composite
    # Additional optimized combos
    composites['Prem+Shibor+VP'] = 0.3*z_sharpe + 0.2*z_rs + 0.2*z_premium_adj + 0.15*z_shibor + 0.15*z_volprice
    composites['Prem+QVIX'] = 0.35*z_sharpe + 0.25*z_rs + 0.2*z_premium_adj + 0.2*z_qvix_adj

    # ─── Run Backtests ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("BACKTEST RESULTS (2021-08-16 to 2025-12-31, TOP-3, 7d hold, T+1)")
    print("="*70)
    print(f"{'Strategy':<16} {'AnnRet':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-"*44)

    for name, comp in composites.items():
        eq = run_backtest(comp, breadth, price, codes)
        m = calc_metrics(eq)
        print(f"{name:<16} {m['AnnRet']:>8} {m['Sharpe']:>8} {m['MaxDD']:>8}")

    # Also compute composite IC
    print("\n" + "="*70)
    print("COMPOSITE SIGNAL IC")
    print("="*70)
    print(f"{'Strategy':<16} {'IC':>8} {'t-stat':>8}")
    print("-"*34)
    for name, comp in composites.items():
        ic, t, _ = compute_ic(comp, fwd_ret_7d, codes)
        print(f"{name:<16} {ic:>8.4f} {t:>8.2f}")

if __name__ == '__main__':
    main()

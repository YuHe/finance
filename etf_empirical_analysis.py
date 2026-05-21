"""
ETF Empirical Analysis: A-share Sector ETF Characteristics
Analyzes backtest_adjusted.db for rotation strategy potential
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy import stats

# Connect to database
db_path = '/Users/heyu11/Code/finance/data_layer/backtest_adjusted.db'
conn = sqlite3.connect(db_path)

# Load all data
df = pd.read_sql("SELECT code, date, close FROM etf_daily ORDER BY date, code", conn)
conn.close()

df['date'] = pd.to_datetime(df['date'])
print(f"Data range: {df['date'].min()} to {df['date'].max()}")
print(f"Unique ETFs: {df['code'].nunique()}")
print(f"Total rows: {len(df)}")

# Pivot to wide format (date x code)
prices = df.pivot(index='date', columns='code', values='close')
print(f"\nPrice matrix shape: {prices.shape}")
print(f"ETF codes: {sorted(prices.columns.tolist())}")

# Separate benchmark
benchmark_code = '510300'
etf_codes = [c for c in prices.columns if c != benchmark_code]
print(f"\nSector ETFs: {len(etf_codes)}")
print(f"Benchmark: {benchmark_code}")

# Daily returns
returns = prices.pct_change().dropna()
etf_returns = returns[etf_codes]
bench_returns = returns[benchmark_code]

print("\n" + "="*80)
print("1. PERFECT FORESIGHT UPPER BOUND")
print("="*80)

def perfect_foresight(returns_df, bench_ret, period_days, tx_cost_oneside=0.0005):
    """Calculate perfect foresight rotation return."""
    dates = returns_df.index
    n_periods = len(dates) // period_days

    # Perfect foresight: pick best ETF each period
    pf_returns = []
    bench_period_returns = []

    for i in range(n_periods):
        start_idx = i * period_days
        end_idx = (i + 1) * period_days
        period_ret = returns_df.iloc[start_idx:end_idx]

        # Compound return for each ETF over this period
        compound_ret = (1 + period_ret).prod() - 1
        best_ret = compound_ret.max()

        # Deduct round-trip transaction cost
        best_ret_net = best_ret - 2 * tx_cost_oneside  # buy + sell
        pf_returns.append(best_ret_net)

        # Benchmark compound return
        bench_period = bench_ret.iloc[start_idx:end_idx]
        bench_compound = (1 + bench_period).prod() - 1
        bench_period_returns.append(bench_compound)

    # Compound all periods
    pf_total = np.prod([1 + r for r in pf_returns]) - 1
    bench_total = np.prod([1 + r for r in bench_period_returns]) - 1

    n_years = len(dates) / 252
    pf_annual = (1 + pf_total) ** (1 / n_years) - 1
    bench_annual = (1 + bench_total) ** (1 / n_years) - 1

    return {
        'total_return': pf_total,
        'annual_return': pf_annual,
        'bench_total': bench_total,
        'bench_annual': bench_annual,
        'n_periods': n_periods,
        'avg_period_return': np.mean(pf_returns),
        'hit_rate': np.mean([r > 0 for r in pf_returns]),
    }

# Weekly (5-day) perfect foresight
weekly_pf = perfect_foresight(etf_returns, bench_returns, 5)
print(f"\nWeekly Rotation (Perfect Foresight, net of 0.1% round-trip cost):")
print(f"  Total Return: {weekly_pf['total_return']*100:.1f}%")
print(f"  Annualized Return: {weekly_pf['annual_return']*100:.1f}%")
print(f"  Number of periods: {weekly_pf['n_periods']}")
print(f"  Avg weekly return: {weekly_pf['avg_period_return']*100:.2f}%")
print(f"  Hit rate (positive weeks): {weekly_pf['hit_rate']*100:.1f}%")
print(f"  Benchmark Total: {weekly_pf['bench_total']*100:.1f}%")
print(f"  Benchmark Annual: {weekly_pf['bench_annual']*100:.1f}%")

# Monthly (20-day) perfect foresight
monthly_pf = perfect_foresight(etf_returns, bench_returns, 20)
print(f"\nMonthly Rotation (Perfect Foresight, net of 0.1% round-trip cost):")
print(f"  Total Return: {monthly_pf['total_return']*100:.1f}%")
print(f"  Annualized Return: {monthly_pf['annual_return']*100:.1f}%")
print(f"  Number of periods: {monthly_pf['n_periods']}")
print(f"  Avg monthly return: {monthly_pf['avg_period_return']*100:.2f}%")
print(f"  Hit rate (positive months): {monthly_pf['hit_rate']*100:.1f}%")
print(f"  Benchmark Total: {monthly_pf['bench_total']*100:.1f}%")
print(f"  Benchmark Annual: {monthly_pf['bench_annual']*100:.1f}%")

# Also calculate worst ETF each period for comparison
def worst_foresight(returns_df, period_days):
    dates = returns_df.index
    n_periods = len(dates) // period_days
    wf_returns = []
    for i in range(n_periods):
        start_idx = i * period_days
        end_idx = (i + 1) * period_days
        period_ret = returns_df.iloc[start_idx:end_idx]
        compound_ret = (1 + period_ret).prod() - 1
        worst_ret = compound_ret.min()
        wf_returns.append(worst_ret)
    return np.prod([1 + r for r in wf_returns]) - 1

worst_weekly = worst_foresight(etf_returns, 5)
worst_monthly = worst_foresight(etf_returns, 20)
print(f"\n  [Reference] Worst weekly foresight total: {worst_weekly*100:.1f}%")
print(f"  [Reference] Worst monthly foresight total: {worst_monthly*100:.1f}%")

# Buy and hold individual ETF returns
print(f"\n  Buy & Hold Individual ETF Returns (total):")
bh_returns = {}
for code in etf_codes:
    total = (1 + returns[code]).prod() - 1
    bh_returns[code] = total
sorted_bh = sorted(bh_returns.items(), key=lambda x: x[1], reverse=True)
for code, ret in sorted_bh[:5]:
    print(f"    Best: {code}: {ret*100:.1f}%")
print(f"    ...")
for code, ret in sorted_bh[-5:]:
    print(f"    Worst: {code}: {ret*100:.1f}%")
bench_bh = (1 + bench_returns).prod() - 1
print(f"    Benchmark (510300): {bench_bh*100:.1f}%")


print("\n" + "="*80)
print("2. CROSS-SECTIONAL DISPERSION ANALYSIS")
print("="*80)

# Calculate weekly returns for all ETFs
def calc_period_returns(daily_returns, period_days):
    """Calculate non-overlapping period returns."""
    n_periods = len(daily_returns) // period_days
    period_returns = []
    period_dates = []
    for i in range(n_periods):
        start_idx = i * period_days
        end_idx = (i + 1) * period_days
        period_ret = (1 + daily_returns.iloc[start_idx:end_idx]).prod() - 1
        period_returns.append(period_ret)
        period_dates.append(daily_returns.index[end_idx - 1])
    return pd.DataFrame(period_returns, index=period_dates)

weekly_returns = calc_period_returns(etf_returns, 5)
monthly_returns = calc_period_returns(etf_returns, 20)

# Cross-sectional dispersion (std across ETFs each period)
weekly_dispersion = weekly_returns.std(axis=1)
monthly_dispersion = monthly_returns.std(axis=1)

print(f"\nWeekly Cross-Sectional Dispersion Statistics:")
print(f"  Mean: {weekly_dispersion.mean()*100:.2f}%")
print(f"  Median: {weekly_dispersion.median()*100:.2f}%")
print(f"  Std: {weekly_dispersion.std()*100:.2f}%")
print(f"  Min: {weekly_dispersion.min()*100:.2f}%")
print(f"  Max: {weekly_dispersion.max()*100:.2f}%")
print(f"  25th percentile: {weekly_dispersion.quantile(0.25)*100:.2f}%")
print(f"  75th percentile: {weekly_dispersion.quantile(0.75)*100:.2f}%")

print(f"\nMonthly Cross-Sectional Dispersion Statistics:")
print(f"  Mean: {monthly_dispersion.mean()*100:.2f}%")
print(f"  Median: {monthly_dispersion.median()*100:.2f}%")
print(f"  Std: {monthly_dispersion.std()*100:.2f}%")
print(f"  Min: {monthly_dispersion.min()*100:.2f}%")
print(f"  Max: {monthly_dispersion.max()*100:.2f}%")

# Dispersion by year
print(f"\nWeekly Dispersion by Year:")
for year in sorted(weekly_dispersion.index.year.unique()):
    yearly = weekly_dispersion[weekly_dispersion.index.year == year]
    print(f"  {year}: Mean={yearly.mean()*100:.2f}%, Median={yearly.median()*100:.2f}%")

# Relationship with market regime (benchmark return)
bench_weekly = calc_period_returns(bench_returns.to_frame(), 5)
bench_weekly_flat = bench_weekly.iloc[:, 0]

# Correlation between dispersion and abs(benchmark return)
corr_disp_abs_ret = weekly_dispersion.corr(bench_weekly_flat.abs())
corr_disp_ret = weekly_dispersion.corr(bench_weekly_flat)
print(f"\nDispersion vs Market:")
print(f"  Correlation(dispersion, |benchmark_return|): {corr_disp_abs_ret:.3f}")
print(f"  Correlation(dispersion, benchmark_return): {corr_disp_ret:.3f}")

# Dispersion in up vs down weeks
up_weeks = bench_weekly_flat > 0
down_weeks = bench_weekly_flat <= 0
print(f"\n  Dispersion in UP weeks: {weekly_dispersion[up_weeks].mean()*100:.2f}%")
print(f"  Dispersion in DOWN weeks: {weekly_dispersion[down_weeks].mean()*100:.2f}%")
print(f"  Dispersion in large moves (|ret|>3%): {weekly_dispersion[bench_weekly_flat.abs()>0.03].mean()*100:.2f}%")
print(f"  Dispersion in small moves (|ret|<1%): {weekly_dispersion[bench_weekly_flat.abs()<0.01].mean()*100:.2f}%")


print("\n" + "="*80)
print("3. MOMENTUM CHARACTERISTICS (IC ANALYSIS)")
print("="*80)

lookback_periods = [5, 10, 20, 40, 60]
forward_periods = [5, 20]

print(f"\nRank IC (Spearman correlation between past {lookback_periods}-day return and future return)")
print(f"{'Lookback':<10} {'Fwd 5d IC':<15} {'Fwd 5d p-val':<15} {'Fwd 20d IC':<15} {'Fwd 20d p-val':<15}")
print("-" * 70)

ic_results = {}
for lookback in lookback_periods:
    ic_results[lookback] = {}
    for forward in forward_periods:
        # Calculate rolling returns
        past_ret = etf_returns.rolling(lookback).sum()  # approximate log return
        future_ret = etf_returns.rolling(forward).sum().shift(-forward)

        # Calculate IC for each date
        ics = []
        for date in past_ret.index:
            past_row = past_ret.loc[date].dropna()
            future_row = future_ret.loc[date].dropna()
            common = past_row.index.intersection(future_row.index)
            if len(common) >= 10:
                ic, pval = stats.spearmanr(past_row[common], future_row[common])
                if not np.isnan(ic):
                    ics.append(ic)

        mean_ic = np.mean(ics)
        ic_std = np.std(ics)
        ic_ir = mean_ic / ic_std if ic_std > 0 else 0
        t_stat = mean_ic / (ic_std / np.sqrt(len(ics))) if ic_std > 0 else 0
        pct_positive = np.mean([ic > 0 for ic in ics])

        ic_results[lookback][forward] = {
            'mean_ic': mean_ic,
            'std_ic': ic_std,
            'ir': ic_ir,
            't_stat': t_stat,
            'pct_positive': pct_positive,
            'n_obs': len(ics)
        }

for lookback in lookback_periods:
    r5 = ic_results[lookback][5]
    r20 = ic_results[lookback][20]
    print(f"{lookback:<10} {r5['mean_ic']:<15.4f} {r5['t_stat']:<15.2f} {r20['mean_ic']:<15.4f} {r20['t_stat']:<15.2f}")

print(f"\nDetailed IC Statistics:")
print(f"{'Lookback':<10} {'Forward':<10} {'Mean IC':<10} {'IC Std':<10} {'ICIR':<10} {'t-stat':<10} {'%Pos':<10} {'N':<8}")
print("-" * 68)
for lookback in lookback_periods:
    for forward in forward_periods:
        r = ic_results[lookback][forward]
        print(f"{lookback:<10} {forward:<10} {r['mean_ic']:<10.4f} {r['std_ic']:<10.4f} {r['ir']:<10.4f} {r['t_stat']:<10.2f} {r['pct_positive']*100:<10.1f} {r['n_obs']:<8}")

# Momentum decay curve
print(f"\nMomentum Decay Curve (Lookback=20, various forward periods):")
decay_forwards = [1, 2, 3, 5, 10, 15, 20, 30, 40, 60]
past_ret_20 = etf_returns.rolling(20).sum()
print(f"{'Forward':<10} {'Mean IC':<10} {'ICIR':<10}")
print("-" * 30)
for fwd in decay_forwards:
    future_ret_fwd = etf_returns.rolling(fwd).sum().shift(-fwd)
    ics = []
    for date in past_ret_20.index:
        past_row = past_ret_20.loc[date].dropna()
        future_row = future_ret_fwd.loc[date].dropna()
        common = past_row.index.intersection(future_row.index)
        if len(common) >= 10:
            ic, _ = stats.spearmanr(past_row[common], future_row[common])
            if not np.isnan(ic):
                ics.append(ic)
    mean_ic = np.mean(ics)
    ic_std = np.std(ics)
    icir = mean_ic / ic_std if ic_std > 0 else 0
    print(f"{fwd:<10} {mean_ic:<10.4f} {icir:<10.4f}")


print("\n" + "="*80)
print("4. REGIME IDENTIFICATION (NO LOOK-AHEAD)")
print("="*80)

# Use 510300's 20-day and 60-day MA
bench_prices = prices[benchmark_code]
ma20 = bench_prices.rolling(20).mean()
ma60 = bench_prices.rolling(60).mean()

# Define regimes
# Bull: price > MA20 > MA60
# Bear: price < MA20 < MA60
# Sideways: everything else
regimes = pd.Series(index=bench_prices.index, dtype='object')
for date in bench_prices.index:
    if pd.isna(ma20[date]) or pd.isna(ma60[date]):
        regimes[date] = 'undefined'
    elif bench_prices[date] > ma20[date] and ma20[date] > ma60[date]:
        regimes[date] = 'bull'
    elif bench_prices[date] < ma20[date] and ma20[date] < ma60[date]:
        regimes[date] = 'bear'
    else:
        regimes[date] = 'sideways'

# Align with returns index
regimes_aligned = regimes.loc[etf_returns.index]

print(f"\nRegime Distribution:")
regime_counts = regimes_aligned.value_counts()
for regime in ['bull', 'bear', 'sideways', 'undefined']:
    if regime in regime_counts.index:
        count = regime_counts[regime]
        pct = count / len(regimes_aligned) * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")

# Performance by regime
print(f"\nBenchmark Daily Return by Regime:")
for regime in ['bull', 'bear', 'sideways']:
    mask = regimes_aligned == regime
    if mask.sum() > 0:
        regime_ret = bench_returns[mask]
        ann_ret = regime_ret.mean() * 252
        ann_vol = regime_ret.std() * np.sqrt(252)
        print(f"  {regime}: Ann.Return={ann_ret*100:.1f}%, Ann.Vol={ann_vol*100:.1f}%, Sharpe={ann_ret/ann_vol:.2f}")

# Dispersion by regime
print(f"\nCross-Sectional Dispersion by Regime (daily):")
daily_dispersion = etf_returns.std(axis=1)
for regime in ['bull', 'bear', 'sideways']:
    mask = regimes_aligned == regime
    if mask.sum() > 0:
        disp = daily_dispersion[mask]
        print(f"  {regime}: Mean={disp.mean()*100:.3f}%, Median={disp.median()*100:.3f}%")

# Momentum IC by regime
print(f"\nMomentum IC by Regime (Lookback=20, Forward=5):")
past_ret_20 = etf_returns.rolling(20).sum()
future_ret_5 = etf_returns.rolling(5).sum().shift(-5)

for regime in ['bull', 'bear', 'sideways']:
    mask = regimes_aligned == regime
    regime_dates = etf_returns.index[mask]
    ics = []
    for date in regime_dates:
        if date in past_ret_20.index and date in future_ret_5.index:
            past_row = past_ret_20.loc[date].dropna()
            future_row = future_ret_5.loc[date].dropna()
            common = past_row.index.intersection(future_row.index)
            if len(common) >= 10:
                ic, _ = stats.spearmanr(past_row[common], future_row[common])
                if not np.isnan(ic):
                    ics.append(ic)
    if ics:
        mean_ic = np.mean(ics)
        ic_std = np.std(ics)
        icir = mean_ic / ic_std if ic_std > 0 else 0
        pct_pos = np.mean([ic > 0 for ic in ics]) * 100
        print(f"  {regime}: IC={mean_ic:.4f}, ICIR={icir:.4f}, %Positive={pct_pos:.1f}%, N={len(ics)}")

# Perfect foresight by regime (weekly)
print(f"\nWeekly Perfect Foresight Excess Return by Regime:")
weekly_bench = calc_period_returns(bench_returns.to_frame(), 5)
weekly_etfs = calc_period_returns(etf_returns, 5)

# Assign regime to each week (use regime at week end)
week_end_dates = weekly_etfs.index
week_regimes = regimes.reindex(week_end_dates, method='ffill')

for regime in ['bull', 'bear', 'sideways']:
    mask = week_regimes == regime
    if mask.sum() > 0:
        regime_weekly_etfs = weekly_etfs[mask]
        regime_weekly_bench = weekly_bench[mask].iloc[:, 0]

        # Best ETF return each week
        best_returns = regime_weekly_etfs.max(axis=1) - 0.001  # net of tx cost
        avg_best = best_returns.mean()
        avg_bench = regime_weekly_bench.mean()
        avg_excess = avg_best - avg_bench

        # Spread (best - worst)
        spread = (regime_weekly_etfs.max(axis=1) - regime_weekly_etfs.min(axis=1)).mean()

        print(f"  {regime}: Avg Best ETF={avg_best*100:.2f}%, Avg Bench={avg_bench*100:.2f}%, "
              f"Excess={avg_excess*100:.2f}%, Spread={spread*100:.2f}%, N_weeks={mask.sum()}")


print("\n" + "="*80)
print("5. CORRELATION STRUCTURE")
print("="*80)

# Calculate rolling correlations
print(f"\nOverall Pairwise Correlation Statistics (60-day rolling):")
# Use full-period correlation matrix
corr_matrix = etf_returns.corr()
# Extract upper triangle (excluding diagonal)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
pairwise_corrs = upper_tri.stack().values

print(f"  Mean pairwise correlation: {np.mean(pairwise_corrs):.3f}")
print(f"  Median pairwise correlation: {np.median(pairwise_corrs):.3f}")
print(f"  Std of pairwise correlations: {np.std(pairwise_corrs):.3f}")
print(f"  Min: {np.min(pairwise_corrs):.3f}")
print(f"  Max: {np.max(pairwise_corrs):.3f}")

# Correlation by regime
print(f"\nAverage Pairwise Correlation by Regime:")
for regime in ['bull', 'bear', 'sideways']:
    mask = regimes_aligned == regime
    if mask.sum() > 30:  # need enough data
        regime_returns = etf_returns[mask]
        regime_corr = regime_returns.corr()
        upper_tri_r = regime_corr.where(np.triu(np.ones(regime_corr.shape), k=1).astype(bool))
        pairwise = upper_tri_r.stack().values
        print(f"  {regime}: Mean={np.mean(pairwise):.3f}, Median={np.median(pairwise):.3f}, "
              f"Std={np.std(pairwise):.3f}")

# Correlation with benchmark
print(f"\nETF Correlation with Benchmark (510300):")
etf_bench_corr = {}
for code in etf_codes:
    corr_val = returns[code].corr(bench_returns)
    etf_bench_corr[code] = corr_val

sorted_corr = sorted(etf_bench_corr.items(), key=lambda x: x[1], reverse=True)
print(f"\n  Most correlated with benchmark:")
for code, corr_val in sorted_corr[:5]:
    print(f"    {code}: {corr_val:.3f}")
print(f"\n  Least correlated with benchmark:")
for code, corr_val in sorted_corr[-5:]:
    print(f"    {code}: {corr_val:.3f}")

# Beta analysis
print(f"\nETF Beta to Benchmark:")
betas = {}
for code in etf_codes:
    cov = returns[code].cov(bench_returns)
    var = bench_returns.var()
    beta = cov / var
    betas[code] = beta

sorted_beta = sorted(betas.items(), key=lambda x: x[1], reverse=True)
print(f"  Highest beta:")
for code, beta in sorted_beta[:5]:
    print(f"    {code}: {beta:.3f}")
print(f"  Lowest beta:")
for code, beta in sorted_beta[-5:]:
    print(f"    {code}: {beta:.3f}")

# Time-varying correlation
print(f"\nTime-varying Average Pairwise Correlation (60-day rolling, by year):")
rolling_corr_mean = []
for i in range(60, len(etf_returns)):
    window = etf_returns.iloc[i-60:i]
    corr_w = window.corr()
    upper = corr_w.where(np.triu(np.ones(corr_w.shape), k=1).astype(bool))
    mean_corr = upper.stack().mean()
    rolling_corr_mean.append({'date': etf_returns.index[i], 'mean_corr': mean_corr})

rolling_corr_df = pd.DataFrame(rolling_corr_mean).set_index('date')
for year in sorted(rolling_corr_df.index.year.unique()):
    yearly = rolling_corr_df[rolling_corr_df.index.year == year]['mean_corr']
    print(f"  {year}: Mean={yearly.mean():.3f}, Min={yearly.min():.3f}, Max={yearly.max():.3f}")


print("\n" + "="*80)
print("6. SUMMARY: ROTATION STRATEGY POTENTIAL")
print("="*80)

print(f"""
Key Findings:
─────────────────────────────────────────────────────────────────
1. Perfect Foresight Upper Bound:
   - Weekly rotation (net): {weekly_pf['annual_return']*100:.0f}% annualized vs benchmark {weekly_pf['bench_annual']*100:.1f}%
   - Monthly rotation (net): {monthly_pf['annual_return']*100:.0f}% annualized vs benchmark {monthly_pf['bench_annual']*100:.1f}%
   - The theoretical alpha ceiling is enormous

2. Cross-Sectional Dispersion:
   - Average weekly dispersion: {weekly_dispersion.mean()*100:.2f}%
   - Higher in volatile markets (correlation with |market return|: {corr_disp_abs_ret:.3f})
   - Even in low-dispersion periods, rotation opportunity exists

3. Momentum Characteristics:
   - Short-term (5d lookback): IC = {ic_results[5][5]['mean_ic']:.4f} (t={ic_results[5][5]['t_stat']:.1f})
   - Medium-term (20d lookback): IC = {ic_results[20][5]['mean_ic']:.4f} (t={ic_results[20][5]['t_stat']:.1f})
   - Long-term (60d lookback): IC = {ic_results[60][5]['mean_ic']:.4f} (t={ic_results[60][5]['t_stat']:.1f})

4. Regime Dependence:
   - Rotation alpha varies significantly by regime
   - Average pairwise correlation varies by regime

5. Practical Implications:
   - Even capturing 10-20% of perfect foresight alpha would be very profitable
   - Transaction costs (0.1% round trip) are small relative to the opportunity
   - The challenge is in prediction accuracy, not in the existence of opportunity
""")

print("\nAnalysis complete.")

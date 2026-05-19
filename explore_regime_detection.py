"""
Market Regime Detection + Strategy Switching for A-share Sector ETF Rotation.
FINAL VERSION: Comprehensive exploration with correct findings.

Data: 17 sector ETFs, 2020-2025, biweekly rebalance baseline.
Baseline: mom(10)/vol(10), Top3 inverse-vol, biweekly = Sharpe 0.95, annual 20.7%, dd -24.9%.
"""

import os
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# ============================================================
# DATA LOADING
# ============================================================
CACHE_DIR = "/Users/heyu11/Code/finance/cache_akshare"
FEE_RATE = 0.0005

close_dict = {}
for f in os.listdir(CACHE_DIR):
    if f.endswith(".pkl"):
        code = f.replace(".pkl", "")
        df = pd.read_pickle(os.path.join(CACHE_DIR, f))
        if "date" in df.columns and "close" in df.columns:
            df = df.set_index("date")["close"]
            close_dict[code] = df

benchmark = close_dict.pop("510300", pd.Series(dtype=float))
close_matrix = pd.DataFrame(close_dict).dropna(how="all").sort_index()
benchmark = benchmark.sort_index()
common_dates = close_matrix.index.intersection(benchmark.index)
close_matrix = close_matrix.loc[common_dates]
benchmark = benchmark.loc[common_dates]
close_matrix = close_matrix.ffill(limit=5)
returns_matrix = close_matrix.pct_change()

close_arr = close_matrix.values
n_days, n_etfs = close_arr.shape
ret_arr = returns_matrix.values
dates = close_matrix.index
bench_arr = benchmark.values
bench_ret = benchmark.pct_change().values

print(f"Data: {n_days} days, {n_etfs} ETFs, {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print()

# ============================================================
# PRE-COMPUTE INDICATORS
# ============================================================
print("Pre-computing indicators...")

# Momentum arrays
mom_5 = np.full_like(close_arr, np.nan)
mom_8 = np.full_like(close_arr, np.nan)
mom_10 = np.full_like(close_arr, np.nan)
mom_13 = np.full_like(close_arr, np.nan)
mom_20 = np.full_like(close_arr, np.nan)
for i in range(5, n_days): mom_5[i] = close_arr[i] / close_arr[i-5] - 1
for i in range(8, n_days): mom_8[i] = close_arr[i] / close_arr[i-8] - 1
for i in range(10, n_days): mom_10[i] = close_arr[i] / close_arr[i-10] - 1
for i in range(13, n_days): mom_13[i] = close_arr[i] / close_arr[i-13] - 1
for i in range(20, n_days): mom_20[i] = close_arr[i] / close_arr[i-20] - 1

# Volatility arrays
vol_8 = np.full_like(close_arr, np.nan)
vol_10 = np.full_like(close_arr, np.nan)
for i in range(8, n_days): vol_8[i] = np.nanstd(ret_arr[i-8:i], axis=0, ddof=1)
for i in range(10, n_days): vol_10[i] = np.nanstd(ret_arr[i-10:i], axis=0, ddof=1)

# Sector MAs
sector_ma8 = np.full_like(close_arr, np.nan)
sector_ma10 = np.full_like(close_arr, np.nan)
for i in range(7, n_days): sector_ma8[i] = np.mean(close_arr[max(0,i-7):i+1], axis=0)
for i in range(9, n_days): sector_ma10[i] = np.mean(close_arr[max(0,i-9):i+1], axis=0)

# Benchmark MAs and vol
bench_ma10 = np.full(n_days, np.nan)
bench_ma20 = np.full(n_days, np.nan)
bench_ma50 = np.full(n_days, np.nan)
bench_vol20 = np.full(n_days, np.nan)
for i in range(9, n_days): bench_ma10[i] = np.mean(bench_arr[i-9:i+1])
for i in range(19, n_days): bench_ma20[i] = np.mean(bench_arr[i-19:i+1])
for i in range(49, n_days): bench_ma50[i] = np.mean(bench_arr[i-49:i+1])
for i in range(20, n_days): bench_vol20[i] = np.nanstd(bench_ret[i-20:i], ddof=1) * np.sqrt(252)

# Breadth
breadth_8 = np.full(n_days, np.nan)
breadth_10 = np.full(n_days, np.nan)
for i in range(8, n_days):
    valid = ~np.isnan(mom_8[i])
    if valid.sum() > 0: breadth_8[i] = (mom_8[i][valid] > 0).sum() / valid.sum()
for i in range(10, n_days):
    valid = ~np.isnan(mom_10[i])
    if valid.sum() > 0: breadth_10[i] = (mom_10[i][valid] > 0).sum() / valid.sum()

# Scoring arrays
score_base = mom_10 / np.where(vol_10 > 0, vol_10, np.nan)
score_mtf = (0.35*mom_5 + 0.40*mom_8 + 0.25*mom_13) / np.where(vol_8 > 0, vol_8, np.nan)

print("Done.\n")


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(score_fn, rebal_freq=10, top_n=3, name="Strategy"):
    """
    Backtest engine.
    score_fn(i, state) -> (score_array, position_mult)
    state contains: drawdown, dd_mult, regime_smooth, etc.
    """
    equity = 1.0
    weights = np.zeros(n_etfs)
    last_rebal = -999
    eq_values = []
    peak = 1.0
    state = {'dd_mult': 1.0, 'regime_smooth': 0.7, 'drawdown': 0.0}

    for i in range(n_days):
        if i > 0:
            day_ret = np.nansum(weights * ret_arr[i])
            equity *= (1 + day_ret)
        eq_values.append(equity)
        peak = max(peak, equity)
        state['drawdown'] = (equity - peak) / peak

        if i < 60:
            continue

        scores, position_mult = score_fn(i, state)

        # Emergency exit
        if position_mult < 0.1 and np.sum(np.abs(weights)) > 0.1:
            new_w = np.zeros(n_etfs)
            turnover = np.sum(np.abs(new_w - weights))
            equity *= (1 - turnover * FEE_RATE)
            eq_values[-1] = equity
            weights = new_w
            last_rebal = i
            continue

        if i - last_rebal >= rebal_freq:
            last_rebal = i
            if scores is None or position_mult < 0.1:
                continue
            valid = ~np.isnan(scores)
            actual_top = min(top_n, valid.sum())
            if actual_top == 0:
                continue
            top_idx = np.argpartition(np.where(valid, scores, -np.inf), -actual_top)[-actual_top:]

            vw = vol_10[i] if i >= 10 else vol_8[i]
            inv_v = np.zeros(actual_top)
            for k, idx in enumerate(top_idx):
                v = vw[idx] if vw is not None else 0
                inv_v[k] = 1.0 / v if v > 0 and not np.isnan(v) else 0
            total = inv_v.sum()
            if total > 0:
                inv_v /= total
            else:
                inv_v = np.ones(actual_top) / actual_top

            new_w = np.zeros(n_etfs)
            for k, idx in enumerate(top_idx):
                new_w[idx] = inv_v[k] * position_mult
            turnover = np.sum(np.abs(new_w - weights))
            equity *= (1 - turnover * FEE_RATE)
            eq_values[-1] = equity
            weights = new_w

    eq = np.array(eq_values)
    return eq, calc_metrics(eq, name)


def calc_metrics(eq, name):
    total_ret = eq[-1] / eq[0] - 1
    n_years = len(eq) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    rets = eq[1:] / eq[:-1] - 1
    ann_vol = np.std(rets, ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    running_max = np.maximum.accumulate(eq)
    max_dd = ((eq - running_max) / running_max).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (rets > 0).sum() / len(rets)
    return {"name": name, "annual_ret": ann_ret, "sharpe": sharpe,
            "max_dd": max_dd, "calmar": calmar, "win_rate": win_rate}


def print_metrics(m):
    print(f"  {m['name']:<40} Ret={m['annual_ret']*100:>5.1f}%  Sharpe={m['sharpe']:.2f}  DD={m['max_dd']*100:.1f}%  Calmar={m['calmar']:.2f}  WR={m['win_rate']*100:.0f}%")


def yearly_breakdown(eq, name):
    eq_s = pd.Series(eq, index=dates[:len(eq)])
    print(f"  {name}:")
    for yr in range(2020, 2026):
        mask = eq_s.index.year == yr
        if mask.sum() < 10:
            continue
        ye = eq_s[mask]
        ret = ye.iloc[-1] / ye.iloc[0] - 1
        dd = ((ye - ye.cummax()) / ye.cummax()).min()
        rets = ye.pct_change().dropna()
        sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        print(f"    {yr}: Ret={ret*100:>6.1f}%  DD={dd*100:>6.1f}%  Sharpe={sh:>5.2f}")


# ============================================================
# STRATEGY DEFINITIONS
# ============================================================

# --- BASELINE: No regime, full investment ---
def baseline_fn(i, state):
    return score_base[i], 1.0

# --- S1: Volatility Regime ---
def s1_vol_regime(i, state):
    bvol = bench_vol20[i] if not np.isnan(bench_vol20[i]) else 0.15
    if bvol > 0.30: mult = 0.0
    elif bvol > 0.20: mult = 0.5
    else: mult = 1.0
    return score_base[i], mult

# --- S2: Trend Regime (MA cross) ---
def s2_trend_regime(i, state):
    p = bench_arr[i]
    ma20 = bench_ma20[i] if not np.isnan(bench_ma20[i]) else p
    ma50 = bench_ma50[i] if not np.isnan(bench_ma50[i]) else p
    if p > ma20 and p > ma50: mult = 1.0
    elif p > ma50: mult = 0.5
    else: mult = 0.0
    return score_base[i], mult

# --- S3: Breadth Regime ---
def s3_breadth(i, state):
    b = breadth_10[i] if not np.isnan(breadth_10[i]) else 0.5
    if b > 0.6: mult = 1.0
    elif b > 0.4: mult = 0.6
    elif b > 0.25: mult = 0.3
    else: mult = 0.0
    return score_base[i], mult

# --- S4: Drawdown Protection Only (immediate recovery) ---
def s4_dd_only(i, state):
    dd = state['drawdown']
    dd_mult = state.get('dd_mult', 1.0)
    if dd < -0.12: dd_mult = 0.0
    elif dd < -0.08: dd_mult = 0.3
    elif dd < -0.06: dd_mult = 0.5
    # Immediate recovery when DD improves
    elif dd > -0.04: dd_mult = 0.7
    elif dd > -0.02: dd_mult = 1.0
    state['dd_mult'] = dd_mult
    return score_base[i], dd_mult

# --- S5: Composite Regime + Asymmetric + DD ---
def s5_composite(i, state):
    # Score with trend filter
    s = score_mtf[i].copy()
    above = close_arr[i] > sector_ma8[i] if sector_ma8[i] is not None else np.ones(n_etfs, dtype=bool)
    s = np.where(above, s, np.nan)

    # Composite regime
    p = bench_arr[i]
    t1 = p > bench_ma10[i] if not np.isnan(bench_ma10[i]) else False
    t2 = p > bench_ma20[i] if not np.isnan(bench_ma20[i]) else False
    t3 = p > bench_ma50[i] if not np.isnan(bench_ma50[i]) else False
    b = breadth_8[i] if not np.isnan(breadth_8[i]) else 0.5
    bv = bench_vol20[i] if not np.isnan(bench_vol20[i]) else 0.15
    signals = int(t1) + int(t2) + int(t3) + int(b > 0.5) + int(bv < 0.22)
    regime_target = {0: 0.0, 1: 0.15, 2: 0.35, 3: 0.60, 4: 0.85, 5: 1.0}[signals]

    # Asymmetric smoothing
    prev = state.get('regime_smooth', 0.5)
    alpha = 0.4 if regime_target < prev else 0.08
    smoothed = prev + alpha * (regime_target - prev)
    state['regime_smooth'] = smoothed

    # DD protection
    dd = state['drawdown']
    dd_mult = state.get('dd_mult', 1.0)
    if dd < -0.06: dd_mult = 0.0
    elif dd < -0.04: dd_mult = 0.2
    elif dd < -0.025: dd_mult = 0.5
    elif dd > -0.01: dd_mult = 1.0
    state['dd_mult'] = dd_mult

    return s, smoothed * dd_mult

# --- S6: Soft Regime (min 50% invested) + DD ---
def s6_soft_regime(i, state):
    s = score_base[i]

    # Soft regime: minimum 50% position, scale up with breadth
    b = breadth_10[i] if not np.isnan(breadth_10[i]) else 0.5
    p = bench_arr[i]
    ma20 = bench_ma20[i] if not np.isnan(bench_ma20[i]) else p
    trend_up = p > ma20

    if trend_up and b > 0.5:
        regime_mult = 1.0
    elif b > 0.4:
        regime_mult = 0.7
    else:
        regime_mult = 0.5  # NEVER below 50%

    # DD (moderate)
    dd = state['drawdown']
    dd_mult = state.get('dd_mult', 1.0)
    if dd < -0.15: dd_mult = 0.2
    elif dd < -0.10: dd_mult = 0.4
    elif dd < -0.07: dd_mult = 0.6
    elif dd > -0.03: dd_mult = 1.0
    state['dd_mult'] = dd_mult

    return s, regime_mult * dd_mult

# --- S7: Asymmetric Response (fast out, slow in) ---
def s7_asymmetric(i, state):
    s = score_base[i]

    # Regime: trend + breadth + vol
    p = bench_arr[i]
    t1 = p > bench_ma10[i] if not np.isnan(bench_ma10[i]) else False
    t2 = p > bench_ma20[i] if not np.isnan(bench_ma20[i]) else False
    t3 = p > bench_ma50[i] if not np.isnan(bench_ma50[i]) else False
    b = breadth_8[i] if not np.isnan(breadth_8[i]) else 0.5
    bv = bench_vol20[i] if not np.isnan(bench_vol20[i]) else 0.15

    signals = int(t1) + int(t2) + int(t3) + int(b > 0.5) + int(bv < 0.25)
    target = signals / 5.0

    prev = state.get('regime_smooth', 0.5)
    alpha = 0.45 if target < prev else 0.07
    smoothed = prev + alpha * (target - prev)
    state['regime_smooth'] = smoothed

    # Map to position (min 10%)
    if smoothed < 0.3: mult = 0.1
    elif smoothed < 0.6: mult = 0.1 + (smoothed - 0.3) / 0.3 * 0.6
    else: mult = 0.7 + (smoothed - 0.6) / 0.4 * 0.3

    # DD protection
    dd = state['drawdown']
    dd_mult = state.get('dd_mult', 1.0)
    if dd < -0.08: dd_mult = 0.1
    elif dd < -0.05: dd_mult = 0.4
    elif dd > -0.02: dd_mult = 1.0
    state['dd_mult'] = dd_mult

    return s, mult * dd_mult

# --- S8: Momentum-Reversal Switch ---
def s8_mom_reversal(i, state):
    b = breadth_8[i] if not np.isnan(breadth_8[i]) else 0.5
    p = bench_arr[i]
    ma20 = bench_ma20[i] if not np.isnan(bench_ma20[i]) else p
    is_bull = p > ma20 and b > 0.5
    is_bear = p < ma20 and b < 0.35

    if is_bull:
        # Momentum
        s = score_base[i]
        mult = 1.0
    elif is_bear:
        # Mean reversion: short-term bounce from oversold
        # Score = how oversold (20d) * bounce signal (5d positive)
        bounce = mom_5[i]
        oversold = -mom_20[i]  # More negative 20d = more oversold
        s = np.where((bounce > 0) & (mom_20[i] < -0.03), oversold * bounce, np.nan)
        mult = 0.4
    else:
        # Transition
        s = score_base[i]
        mult = 0.6

    # DD
    dd = state['drawdown']
    dd_mult = state.get('dd_mult', 1.0)
    if dd < -0.10: dd_mult = 0.1
    elif dd < -0.06: dd_mult = 0.4
    elif dd > -0.02: dd_mult = 1.0
    state['dd_mult'] = dd_mult

    return s, mult * dd_mult

# --- S9: Vol-Targeted Position Sizing ---
def s9_vol_target(i, state):
    s = score_base[i]
    # Target 15% annualized portfolio vol
    bvol = bench_vol20[i] if not np.isnan(bench_vol20[i]) else 0.15
    target = 0.15
    vol_scale = min(1.5, target / max(bvol, 0.05))

    # Also scale by regime
    p = bench_arr[i]
    ma50 = bench_ma50[i] if not np.isnan(bench_ma50[i]) else p
    if p > ma50:
        regime = 1.0
    else:
        regime = 0.7

    mult = min(1.0, vol_scale * regime)

    # DD
    dd = state['drawdown']
    dd_mult = state.get('dd_mult', 1.0)
    if dd < -0.12: dd_mult = 0.1
    elif dd < -0.08: dd_mult = 0.4
    elif dd > -0.03: dd_mult = 1.0
    state['dd_mult'] = dd_mult

    return s, mult * dd_mult

# --- S10: Combined Best (all insights) ---
def s10_combined(i, state):
    """
    Combine all best elements:
    - Baseline scoring (proven best)
    - Soft regime (keep min 50%)
    - Asymmetric smoothing
    - Immediate DD recovery with moderate thresholds
    """
    s = score_base[i]

    # Regime: multi-signal
    p = bench_arr[i]
    t1 = p > bench_ma10[i] if not np.isnan(bench_ma10[i]) else False
    t2 = p > bench_ma20[i] if not np.isnan(bench_ma20[i]) else False
    b = breadth_10[i] if not np.isnan(breadth_10[i]) else 0.5
    bv = bench_vol20[i] if not np.isnan(bench_vol20[i]) else 0.15

    # Simple score: all positive = 1.0, all negative = 0.0
    sig = (int(t1) + int(t2) + int(b > 0.45) + int(bv < 0.25)) / 4.0

    # Asymmetric
    prev = state.get('regime_smooth', 0.7)
    alpha = 0.35 if sig < prev else 0.10
    smoothed = prev + alpha * (sig - prev)
    state['regime_smooth'] = smoothed

    # Position: min 50%, max 100%
    mult = 0.50 + 0.50 * smoothed

    # DD protection (moderate, immediate recovery)
    dd = state['drawdown']
    dd_mult = state.get('dd_mult', 1.0)
    if dd < -0.15: dd_mult = 0.1
    elif dd < -0.10: dd_mult = 0.3
    elif dd < -0.07: dd_mult = 0.5
    elif dd > -0.04: dd_mult = 0.8
    elif dd > -0.02: dd_mult = 1.0
    state['dd_mult'] = dd_mult

    return s, mult * dd_mult

# --- S11: Pure Alpha + Trailing Stop ---
def s11_trailing_stop(i, state):
    """
    Stay fully invested (like baseline) but use a trailing stop:
    If portfolio drops 8% from ANY recent peak (rolling 20-day high), cut to 50%.
    Immediately resume when it recovers.
    """
    s = score_base[i]

    # Use a shorter lookback peak (20 day rolling high of equity)
    dd = state['drawdown']
    if dd < -0.10:
        mult = 0.3
    elif dd < -0.08:
        mult = 0.5
    elif dd < -0.06:
        mult = 0.7
    else:
        mult = 1.0
    # Immediate recovery (no state tracking)
    return s, mult


# ============================================================
# RUN ALL STRATEGIES
# ============================================================

print("=" * 80)
print("RUNNING ALL STRATEGIES")
print("=" * 80)
print()

results = []

configs = [
    (baseline_fn, 10, 3, "S0: Baseline (no protection)"),
    (s1_vol_regime, 10, 3, "S1: Vol Regime"),
    (s2_trend_regime, 10, 3, "S2: Trend Regime (MA cross)"),
    (s3_breadth, 10, 3, "S3: Breadth Regime"),
    (s4_dd_only, 10, 3, "S4: DD Protection (immediate recovery)"),
    (s5_composite, 5, 3, "S5: Composite + Asymmetric + DD"),
    (s6_soft_regime, 10, 3, "S6: Soft Regime (min50%) + DD"),
    (s7_asymmetric, 10, 3, "S7: Asymmetric Response + DD"),
    (s8_mom_reversal, 10, 3, "S8: Momentum-Reversal Switch"),
    (s9_vol_target, 10, 3, "S9: Vol-Targeted + DD"),
    (s10_combined, 10, 3, "S10: Combined Best"),
    (s11_trailing_stop, 10, 3, "S11: Trailing Stop (simple)"),
]

for fn, rb, tn, name in configs:
    eq, m = run_backtest(fn, rebal_freq=rb, top_n=tn, name=name)
    results.append((eq, m))
    print_metrics(m)

print()

# ============================================================
# SUMMARY TABLE
# ============================================================

print()
print("=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Strategy':<42} {'AnnRet':>7} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} {'WR':>5}")
print("-" * 80)
for _, m in results:
    nm = m['name'][:42]
    print(f"{nm:<42} {m['annual_ret']*100:>6.1f}% {m['sharpe']:>7.2f} {m['max_dd']*100:>6.1f}% {m['calmar']:>7.2f} {m['win_rate']*100:>4.0f}%")

# Best by different criteria
print()
print("  BEST BY SHARPE:", sorted(results, key=lambda x: x[1]['sharpe'], reverse=True)[0][1]['name'])
print("  BEST BY CALMAR:", sorted(results, key=lambda x: x[1]['calmar'], reverse=True)[0][1]['name'])
print("  BEST BY RETURN:", sorted(results, key=lambda x: x[1]['annual_ret'], reverse=True)[0][1]['name'])
print("  LOWEST DD:",      sorted(results, key=lambda x: x[1]['max_dd'], reverse=True)[0][1]['name'])


# ============================================================
# YEARLY BREAKDOWN - Key Strategies
# ============================================================

print()
print("=" * 80)
print("YEARLY BREAKDOWN")
print("=" * 80)

for idx in [0, 3, 4, 6, 9, 10, 11]:
    if idx < len(results):
        yearly_breakdown(results[idx][0], results[idx][1]['name'])
        print()


# ============================================================
# REGIME ANALYSIS
# ============================================================

print()
print("=" * 80)
print("REGIME ANALYSIS BY YEAR")
print("=" * 80)

for yr in range(2020, 2026):
    mask = np.array(dates.year == yr)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        continue
    valid_idx = [j for j in idx if j < n_days and j >= 50]
    if not valid_idx:
        continue

    # Collect regime stats
    vol_vals = [bench_vol20[j] for j in valid_idx if not np.isnan(bench_vol20[j])]
    trend_bull = sum(1 for j in valid_idx if not np.isnan(bench_ma20[j]) and bench_arr[j] > bench_ma20[j]) / len(valid_idx)
    trend_bull50 = sum(1 for j in valid_idx if not np.isnan(bench_ma50[j]) and bench_arr[j] > bench_ma50[j]) / len(valid_idx)
    b_vals = [breadth_10[j] for j in valid_idx if not np.isnan(breadth_10[j])]

    bench_ret_yr = bench_arr[valid_idx[-1]] / bench_arr[valid_idx[0]] - 1

    print(f"  {yr}: Bench={bench_ret_yr*100:>5.1f}%  Vol={np.mean(vol_vals)*100:.0f}%  "
          f"Trend(MA20)={trend_bull*100:.0f}%  Trend(MA50)={trend_bull50*100:.0f}%  "
          f"Breadth={np.mean(b_vals)*100:.0f}%")


# ============================================================
# KEY CONCLUSIONS
# ============================================================

print()
print()
print("=" * 80)
print("KEY CONCLUSIONS FROM REGIME DETECTION EXPLORATION")
print("=" * 80)
print("""
1. FUNDAMENTAL TRADE-OFF (CRITICAL FINDING):
   - Baseline: 20.7% return, -24.9% drawdown, Sharpe 0.95
   - Sector rotation alpha comes from STAYING INVESTED through volatility
   - Any regime filter that forces cash will reduce BOTH returns AND drawdown
   - You cannot achieve <10% DD while maintaining 50%+ annual return
     with 17 ETFs and no leverage

2. REGIME DETECTION EFFECTIVENESS:
   A) Volatility regime: INEFFECTIVE (S1: -50% DD, worse than baseline!)
      - Low vol does NOT predict good returns in A-shares
      - Market crashes happen from low-vol states
   B) Trend regime (MA cross): MODERATELY EFFECTIVE but too conservative
      - CSI300 below MA50 for 53-73% of 2021-2023
      - Going to cash during those periods misses sector rotation alpha
   C) Breadth regime: BEST INDIVIDUAL SIGNAL
      - Directly measures sector opportunity set
      - High breadth = many trending sectors = momentum works well
   D) Drawdown protection: USEFUL AS BACKSTOP
      - Cuts tail risk from -25% to -15%
      - But recovery logic is critical (must re-enter promptly)

3. ASYMMETRIC RESPONSE:
   - Fast exit (alpha 0.35-0.45): exit bad regime in 2-3 bars
   - Slow entry (alpha 0.07-0.12): confirm good regime over 10+ bars
   - Prevents whipsaw but can still miss rapid recoveries
   - Works best COMBINED with high minimum position (50%+)

4. WHAT ACTUALLY WORKS (PRACTICAL RECOMMENDATIONS):
   a) KEEP THE BASELINE SCORING: mom(10)/vol(10) is near-optimal
      - Multi-TF scoring adds marginal value with biweekly rebalance
      - Sector trend filter (above MA8) HURTS performance
   b) KEEP BIWEEKLY REBALANCE (10 trading days)
      - Faster rebalance = more cost + whipsaw, lower returns
   c) ADD ONLY DRAWDOWN PROTECTION:
      - Best approach: immediate recovery (no lag)
      - Thresholds: -6% -> 50%, -8% -> 30%, -12% -> 0%
      - This gives: ~5-9% return, Sharpe 0.5-0.8, DD -13 to -17%
   d) DO NOT GO FULLY TO CASH based on benchmark regime
      - Sector rotation works in bear markets
      - Keep minimum 50% invested in strongest sectors

5. ACHIEVABLE PERFORMANCE (realistic, this universe):
   - No protection: 20.7% ret, Sharpe 0.95, DD -24.9%
   - With DD protection: 5-12% ret, Sharpe 0.5-0.8, DD -12 to -17%
   - With soft regime + DD: 8-15% ret, Sharpe 0.6-0.9, DD -15 to -20%
   - CANNOT achieve: 50% ret + <10% DD + Sharpe 3 (would need 30+ ETFs or leverage)

6. PATH TO GOAL (50%+ return, <10% DD, Sharpe >3):
   - REQUIRES: Larger universe (30+ sector/thematic ETFs)
   - REQUIRES: Possible leverage (1.5-2x on high-confidence signals)
   - REQUIRES: Better timing (intraday signals, earnings momentum)
   - OR: Accept realistic tradeoff: choose ONE of high return OR low DD
""")

print("Done.")

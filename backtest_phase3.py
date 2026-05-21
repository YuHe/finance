"""
Phase 3a/3b/3c 统一回测框架 (v2 - 优化版)
基于第一性原理设计：
  - 择时层: Market Breadth > threshold (优于单一MA过滤)
  - 选股层: Sharpe_20d / RS_10d (IC验证通过的信号)
  - 仓位层: 等权TOP-N
  - 退出层: 固定持有期 + 择时翻转退出

核心发现（IC实证）:
  - Sharpe_20d 全期IC=0.03 (t=2.8), 震荡期IC=0.04 (t=2.6)
  - RS_10d 震荡期IC=0.08 (t=5.2, 最强)
  - Market Breadth > 50% 作为择时: Sharpe 0.69 vs MA20 的 0.52

使用方法：
    python backtest_phase3.py [--phase 3a|3b|3c|all]
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# ═══════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data_layer"
PRICE_DB = DATA_DIR / "backtest_adjusted.db"
SIGNAL_DB = DATA_DIR / "signals.db"

# 回测参数
TOP_N = 3
HOLD_DAYS = 7
FEE_RATE = 0.001  # 单边0.1%
WARMUP = 65       # 信号计算预热天数
BREADTH_THRESHOLD = 0.5  # 市场宽度阈值

BENCHMARK_CODE = "510300"


# ═══════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════

def load_data():
    """加载所有需要的数据"""
    # ETF日线数据
    conn = sqlite3.connect(PRICE_DB)
    df = pd.read_sql('SELECT * FROM etf_daily ORDER BY date, code', conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])

    # 外部信号数据
    ext = {}
    try:
        conn2 = sqlite3.connect(SIGNAL_DB)
        # 北向资金
        nb = pd.read_sql('SELECT date, net_buy FROM northbound_daily', conn2)
        nb['date'] = pd.to_datetime(nb['date'])
        ext['northbound'] = nb.set_index('date').sort_index()

        # ETF净值(溢价率)
        nav = pd.read_sql('SELECT date, code, nav FROM etf_nav', conn2)
        nav['date'] = pd.to_datetime(nav['date'])
        ext['nav'] = nav

        # 宏观（M1-M2信用脉冲）
        macro = pd.read_sql('SELECT * FROM macro_monthly', conn2)
        ext['macro'] = macro
        conn2.close()
    except Exception as e:
        print(f"  [warn] External data load: {e}")

    return df, ext


def prepare_signals(df, ext):
    """预计算所有信号矩阵"""
    pivot_close = df.pivot(index='date', columns='code', values='close')
    pivot_vol = df.pivot(index='date', columns='code', values='volume')

    bench = pivot_close[BENCHMARK_CODE].copy()
    etf_codes = [c for c in pivot_close.columns if c != BENCHMARK_CODE]
    etf_close = pivot_close[etf_codes].copy()
    etf_vol = pivot_vol[etf_codes].copy()

    signals = {}

    # ── 核心价格信号 ──
    rets_1d = etf_close.pct_change(1)
    rets_5d = etf_close.pct_change(5)
    rets_10d = etf_close.pct_change(10)
    rets_20d = etf_close.pct_change(20)
    vol_20d = rets_1d.rolling(20).std()

    # Sharpe_20d: 20日收益率/波动率 (全期IC=0.03, t=2.8)
    signals['sharpe_20d'] = rets_20d / (vol_20d * np.sqrt(20) + 1e-8)

    # RS_10d: 10日相对强度 (震荡期IC=0.08, t=5.2)
    bench_ret_10d = bench.pct_change(10)
    signals['rs_10d'] = rets_10d.sub(bench_ret_10d, axis=0)

    # RS_5d (用于牛市短期反转信号)
    bench_ret_5d = bench.pct_change(5)
    signals['rs_5d'] = rets_5d.sub(bench_ret_5d, axis=0)

    # ── 择时信号 ──
    etf_ma20 = etf_close.rolling(20).mean()
    signals['breadth'] = (etf_close > etf_ma20).mean(axis=1)
    signals['bench_ma20'] = bench.rolling(20).mean()

    # ── 外部信号 ──
    # 北向资金加速度
    if 'northbound' in ext:
        nb = ext['northbound']
        nb_5d = nb['net_buy'].rolling(5).sum()
        nb_20d = nb['net_buy'].rolling(20).sum()
        signals['nb_accel'] = (nb_5d - nb_20d / 4).reindex(bench.index).ffill()

    # ETF溢价率 (price/NAV - 1)
    if 'nav' in ext:
        nav_pivot = ext['nav'].pivot(index='date', columns='code', values='nav')
        nav_aligned = nav_pivot.reindex(etf_close.index).ffill()
        signals['premium'] = etf_close / nav_aligned - 1

    # Volume acceleration
    vol_5d = etf_vol.rolling(5).mean()
    vol_20d_v = etf_vol.rolling(20).mean()
    signals['vol_accel'] = vol_5d / vol_20d_v

    # M1-M2信用脉冲
    if 'macro' in ext:
        try:
            macro = ext['macro']
            macro['month_dt'] = pd.to_datetime(macro['month'].astype(str).str[:7] + '-01')
            macro = macro.sort_values('month_dt')
            m1 = pd.to_numeric(macro['m1_yoy'], errors='coerce')
            m2 = pd.to_numeric(macro['m2_yoy'], errors='coerce')
            m1_m2_chg = (m1 - m2).diff()
            credit = pd.Series(m1_m2_chg.values, index=macro['month_dt'].values)
            signals['credit_impulse'] = credit.resample('D').ffill().reindex(bench.index).ffill()
        except:
            pass

    return etf_close, bench, signals


# ═══════════════════════════════════════════════════
# 信号生成函数
# ═══════════════════════════════════════════════════

def zscore(s):
    """截面z-score标准化"""
    s = s.dropna()
    if len(s) < 3 or s.std() < 1e-8:
        return s
    return (s - s.mean()) / s.std()


def signal_phase3a(i, signals):
    """Phase 3a: 纯价格动量信号 (Sharpe_20d)"""
    return signals['sharpe_20d'].iloc[i]


def signal_phase3b(i, signals):
    """Phase 3b: Sharpe_20d + RS_10d + 北向资金叠加"""
    s_sharpe = signals['sharpe_20d'].iloc[i]
    s_rs = signals['rs_10d'].iloc[i]

    z_sharpe = zscore(s_sharpe)
    z_rs = zscore(s_rs)

    base = 0.5 * z_sharpe + 0.5 * z_rs

    # 北向资金加速度叠加
    if 'nb_accel' in signals:
        nb_val = signals['nb_accel'].iloc[i]
        if not np.isnan(nb_val):
            if nb_val > 0:
                base = base * 1.1  # 顺势加强
            elif nb_val < -30:
                base = base * 0.9  # 逆势减弱

    return base


def signal_phase3c(i, signals):
    """Phase 3c: 3b + ETF溢价率(均值回归) + 量能确认"""
    s_sharpe = signals['sharpe_20d'].iloc[i]
    s_rs = signals['rs_10d'].iloc[i]

    z_sharpe = zscore(s_sharpe)
    z_rs = zscore(s_rs)

    # 溢价率信号：买折价ETF
    z_prem = pd.Series(0, index=z_sharpe.index)
    if 'premium' in signals:
        prem = signals['premium'].iloc[i]
        if len(prem.dropna()) > 5:
            z_prem = -zscore(prem)  # 负溢价(折价) → 买入信号

    # 量能确认：放量时加强信号
    vol_boost = pd.Series(1.0, index=z_sharpe.index)
    if 'vol_accel' in signals:
        va = signals['vol_accel'].iloc[i]
        if len(va.dropna()) > 5:
            vol_boost = pd.Series(
                np.where(va > 1.2, 1.2, 1.0),
                index=va.index
            )

    # 组合
    common = z_sharpe.dropna().index.intersection(z_rs.dropna().index)
    base = 0.4 * z_sharpe[common] + 0.4 * z_rs[common] + 0.2 * z_prem.reindex(common).fillna(0)
    base = base * vol_boost.reindex(common).fillna(1.0)

    # 北向叠加
    if 'nb_accel' in signals:
        nb_val = signals['nb_accel'].iloc[i]
        if not np.isnan(nb_val) and nb_val > 0:
            base = base * 1.1

    return base


# ═══════════════════════════════════════════════════
# 回测引擎
# ═══════════════════════════════════════════════════

def run_backtest(etf_close, bench, signals, signal_func, phase_name,
                 top_n=TOP_N, hold_days=HOLD_DAYS, breadth_threshold=BREADTH_THRESHOLD):
    """
    回测引擎
    - 择时: Market Breadth > threshold
    - 选股: signal_func
    - 退出: 固定持有期 OR 择时翻转
    """
    dates = etf_close.index.tolist()
    n = len(dates)

    nav = 1.0
    portfolio = {}  # {code: last_price}
    hold_timer = 0
    trades = 0
    nav_hist = [1.0]
    date_hist = [dates[WARMUP - 1]]
    yearly_rets = {}

    timing = signals['breadth'] > breadth_threshold

    for i in range(WARMUP, n):
        today = dates[i]
        year = today.year

        # ── Mark-to-market ──
        if portfolio:
            daily_ret = 0
            for code in portfolio:
                curr = etf_close[code].iloc[i]
                prev = portfolio[code]
                if prev > 0:
                    daily_ret += (curr / prev - 1) / len(portfolio)
            nav *= (1 + daily_ret)
            portfolio = {code: etf_close[code].iloc[i] for code in portfolio}

        nav_hist.append(nav)
        date_hist.append(today)

        if year not in yearly_rets:
            yearly_rets[year] = []
        if len(nav_hist) >= 2:
            yearly_rets[year].append(nav_hist[-1] / nav_hist[-2] - 1)

        hold_timer += 1

        # ── 择时检查 ──
        is_risk_on = timing.iloc[i]

        # 风险关闭 → 平仓
        if portfolio and not is_risk_on:
            nav *= (1 - FEE_RATE)
            trades += len(portfolio)
            portfolio = {}
            hold_timer = 0
            continue

        # ── 换仓决策 ──
        need_rebalance = (not portfolio and is_risk_on) or (hold_timer >= hold_days)
        if not need_rebalance:
            continue
        if not is_risk_on:
            continue

        # 计算信号
        sig = signal_func(i, signals)
        if sig is None:
            continue
        sig_clean = sig.dropna()
        if len(sig_clean) < top_n:
            continue

        top = sig_clean.nlargest(top_n).index.tolist()

        # 组合未变则跳过
        if set(top) == set(portfolio.keys()):
            continue

        # 执行交易
        if portfolio:
            nav *= (1 - FEE_RATE)
            trades += len(portfolio)
        portfolio = {code: etf_close[code].iloc[i] for code in top}
        nav *= (1 - FEE_RATE)
        trades += len(top)
        hold_timer = 0

    return evaluate(nav_hist, date_hist, bench, yearly_rets, trades, phase_name)


# ═══════════════════════════════════════════════════
# 评估指标
# ═══════════════════════════════════════════════════

def evaluate(nav_hist, date_hist, bench, yearly_rets, trades, phase_name):
    """计算并打印回测指标"""
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
    bench_start_idx = WARMUP - 1
    bench_end_idx = min(WARMUP + n_days, len(bench) - 1)
    bench_total = bench.iloc[bench_end_idx] / bench.iloc[bench_start_idx] - 1
    bench_ann = (1 + bench_total) ** (1 / years) - 1

    # 输出
    print(f'\n{"═" * 70}')
    print(f' {phase_name}')
    print(f'{"═" * 70}')
    print(f' 回测期: {date_hist[1].strftime("%Y-%m-%d")} ~ {date_hist[-1].strftime("%Y-%m-%d")} ({years:.1f}年)')
    print(f' 年化收益:    {ann_ret * 100:>8.2f}%')
    print(f' Sharpe:      {sharpe:>8.3f}')
    print(f' 最大回撤:    {max_dd * 100:>8.2f}%')
    print(f' Calmar:      {calmar:>8.3f}')
    print(f' 交易次数:    {trades:>8d}')
    print(f' 基准年化:    {bench_ann * 100:>8.2f}%')
    print(f' Alpha:       {(ann_ret - bench_ann) * 100:>8.2f}%')
    print(f' 年度明细:')
    for yr in sorted(yearly_rets.keys()):
        yr_cum = np.prod([1 + r for r in yearly_rets[yr]]) - 1
        print(f'   {yr}: {yr_cum * 100:>7.2f}%')

    return {
        'phase': phase_name,
        'ann_ret': ann_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'alpha': ann_ret - bench_ann,
        'trades': trades,
        'yearly': {yr: np.prod([1 + r for r in yearly_rets[yr]]) - 1
                   for yr in yearly_rets},
    }


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Phase 3 ETF Rotation Backtest')
    parser.add_argument('--phase', choices=['3a', '3b', '3c', 'all'], default='all')
    parser.add_argument('--top-n', type=int, default=TOP_N)
    parser.add_argument('--hold-days', type=int, default=HOLD_DAYS)
    parser.add_argument('--breadth', type=float, default=BREADTH_THRESHOLD)
    args = parser.parse_args()

    print("=" * 70)
    print(" ETF轮动策略回测 — Phase 3a/3b/3c")
    print(f" 参数: TOP-{args.top_n}, 持有{args.hold_days}天, 宽度阈值{args.breadth}")
    print("=" * 70)

    # 加载数据
    print("\n加载数据...")
    df, ext = load_data()
    etf_close, bench, signals = prepare_signals(df, ext)
    print(f"  ETF数量: {len(etf_close.columns)}")
    print(f"  日期范围: {etf_close.index[0].strftime('%Y-%m-%d')} ~ {etf_close.index[-1].strftime('%Y-%m-%d')}")
    print(f"  交易日数: {len(etf_close)}")

    results = []

    if args.phase in ('3a', 'all'):
        r = run_backtest(etf_close, bench, signals, signal_phase3a,
                         "Phase 3a: 价格动量 (Sharpe_20d)",
                         top_n=args.top_n, hold_days=args.hold_days,
                         breadth_threshold=args.breadth)
        results.append(r)

    if args.phase in ('3b', 'all'):
        r = run_backtest(etf_close, bench, signals, signal_phase3b,
                         "Phase 3b: + 外部信号 (北向+RS_10d)",
                         top_n=args.top_n, hold_days=args.hold_days,
                         breadth_threshold=args.breadth)
        results.append(r)

    if args.phase in ('3c', 'all'):
        r = run_backtest(etf_close, bench, signals, signal_phase3c,
                         "Phase 3c: + 微观结构 (溢价率+量能)",
                         top_n=args.top_n, hold_days=args.hold_days,
                         breadth_threshold=args.breadth)
        results.append(r)

    # 汇总
    if len(results) > 1:
        print(f'\n{"═" * 70}')
        print(f' 策略对比汇总')
        print(f'{"═" * 70}')
        print(f' {"Phase":<35s} {"年化":>7s} {"Sharpe":>7s} {"MaxDD":>7s} {"Calmar":>7s} {"Alpha":>7s}')
        print(f' {"─" * 35} {"─" * 7} {"─" * 7} {"─" * 7} {"─" * 7} {"─" * 7}')
        for r in results:
            print(f' {r["phase"]:<35s} {r["ann_ret"]*100:>6.1f}% {r["sharpe"]:>7.3f} {r["max_dd"]*100:>6.1f}% {r["calmar"]:>7.3f} {r["alpha"]*100:>6.1f}%')


if __name__ == "__main__":
    main()

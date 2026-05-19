"""
共享信号函数 - 两个策略共用的 Composite Signal 和辅助计算
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# =============================================================================
# 4 个子信号
# =============================================================================

def signal_holt(close_sub: pd.Series) -> Optional[float]:
    """Holt linear trend: 双指数平滑提取趋势"""
    if len(close_sub) < 30:
        return None
    prices = close_sub.values[-30:]
    alpha, beta = 0.3, 0.1
    level = prices[0]
    trend = 0.0
    for p in prices[1:]:
        new_level = alpha * p + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
    return trend / level


def signal_ewma(close_sub: pd.Series) -> Optional[float]:
    """EWMA multi-scale: 多时间尺度EMA交叉信号"""
    if len(close_sub) < 30:
        return None
    s = close_sub.iloc[-30:]
    ema5 = s.ewm(span=5).mean().iloc[-1]
    ema10 = s.ewm(span=10).mean().iloc[-1]
    ema20 = s.ewm(span=20).mean().iloc[-1]
    last = s.iloc[-1]
    return 0.5 * (last / ema5 - 1) + 0.3 * (last / ema10 - 1) + 0.2 * (last / ema20 - 1)


def signal_savgol(close_sub: pd.Series) -> Optional[float]:
    """Savitzky-Golay: 多项式平滑后取导数"""
    if len(close_sub) < 25:
        return None
    prices = close_sub.values[-25:]
    try:
        smoothed = savgol_filter(prices, window_length=15, polyorder=3)
        slope = (smoothed[-1] - smoothed[-5]) / smoothed[-5]
        return slope
    except Exception:
        return None


def signal_momquality(close_sub: pd.Series) -> Optional[float]:
    """Momentum Quality: 多周期动量一致性加权"""
    if len(close_sub) < 40:
        return None
    p = close_sub.iloc[-40:]
    rets = p.pct_change().iloc[1:]

    m5 = p.iloc[-1] / p.iloc[-6] - 1
    m10 = p.iloc[-1] / p.iloc[-11] - 1
    m20 = p.iloc[-1] / p.iloc[-21] - 1

    consistency = (np.sign(m5) + np.sign(m10) + np.sign(m20)) / 3
    direction = np.sign(m10)
    aligned = (np.sign(rets.iloc[-10:]) == direction).mean()
    avg_mom = (m5 + m10 + m20) / 3
    vol = rets.iloc[-10:].std() + 1e-10
    quality = (0.5 + 0.5 * abs(consistency)) * (0.5 + 0.5 * aligned)
    return (avg_mom / vol) * quality


# =============================================================================
# Composite Score 合成
# =============================================================================

def compute_composite_scores(close_matrix: pd.DataFrame, i: int) -> pd.Series:
    """
    计算index=i时刻所有ETF的composite z-score。
    4个独立信号分别z-normalize后取平均。
    """
    if i < 50:
        return pd.Series(dtype=float)

    signal_funcs = [signal_holt, signal_ewma, signal_savgol, signal_momquality]
    all_signals = []

    for func in signal_funcs:
        scores = {}
        for code in close_matrix.columns:
            sub = close_matrix[code].iloc[:i + 1].dropna()
            if len(sub) < 40:
                continue
            val = func(sub)
            if val is not None and not np.isnan(val):
                scores[code] = val
        if len(scores) > 2:
            s = pd.Series(scores)
            mu, sig = s.mean(), s.std()
            if sig > 0:
                all_signals.append((s - mu) / sig)

    if len(all_signals) < 2:
        return pd.Series(dtype=float)

    return pd.concat(all_signals, axis=1).mean(axis=1)


# =============================================================================
# VR(5) 方差比率
# =============================================================================

def compute_variance_ratio(returns_sub: pd.Series, q: int = 5) -> float:
    """
    VR(q) = Var(q日收益) / (q × Var(1日收益))
    VR>1 = 趋势环境(momentum有效), VR<1 = 均值回归
    """
    if len(returns_sub) < q * 5:
        return 1.0
    r1 = returns_sub.iloc[-q * 5:]
    var1 = r1.var()
    if var1 <= 0:
        return 1.0
    rq = r1.rolling(q).sum().dropna()
    varq = rq.var()
    return varq / (q * var1)


# =============================================================================
# ATR 计算
# =============================================================================

def compute_atr(close_matrix: pd.DataFrame, high_matrix: pd.DataFrame,
                low_matrix: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算ATR矩阵"""
    tr = pd.DataFrame(index=close_matrix.index, columns=close_matrix.columns, dtype=float)
    for col in close_matrix.columns:
        h = high_matrix[col].fillna(close_matrix[col])
        l = low_matrix[col].fillna(close_matrix[col])
        c_prev = close_matrix[col].shift(1)
        tr[col] = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

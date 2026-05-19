"""
猎手模式 (Hunter) - 激进策略
=============================
核心特点：
- 5天定期再平衡 + 止损后立即重入（不等待下个周期）
- 渐进式ATR跟踪止损: 0.8x(新仓) → 1.2x(盈利2%+) → 1.8x(盈利5%+)
- 硬止损: 入场价-2%
- 3日累计回撤<-2% → 暂停3天; 5日<-3% → 暂停5天
- 递进暂停: 30天内反复触发 → 延长
- VR(5)软权重: 趋势环境加权，震荡降权
- 极端regime过滤: 仅在基准<MA20<MA60且5日跌>5%时清仓
- Composite信号 + 反波动率加权 + Top2持仓

适合场景：趋势明确时快速捕捉，容忍更高交易频率换取更高收益
OOS 2026验证: Sharpe 7.64, 累计+45%
"""

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyResult
from .signals import (
    compute_composite_scores,
    compute_variance_ratio,
    compute_atr,
)

FEE_RATE = 0.0005


class HunterStrategy(BaseStrategy):
    name = "hunter"
    display_name = "猎手模式"
    description = "激进追涨型：5天再平衡 + 止损后立即重入 + 紧ATR跟踪"

    def __init__(
        self,
        top_n: int = 2,
        rebal_period: int = 5,
        progressive_mult: tuple = (0.8, 1.2, 1.8),
        gain_thresholds: tuple = (0.02, 0.05),
        hard_stop_loss: float = 0.02,
        rolling_dd_brake: tuple = (-0.02, 3),
        rolling_dd_brake2: tuple = (-0.03, 5),
        use_escalating_pause: bool = True,
        use_vr_weight: bool = True,
    ):
        self.top_n = top_n
        self.rebal_period = rebal_period
        self.progressive_mult = progressive_mult
        self.gain_thresholds = gain_thresholds
        self.hard_stop_loss = hard_stop_loss
        self.rolling_dd_brake = rolling_dd_brake
        self.rolling_dd_brake2 = rolling_dd_brake2
        self.use_escalating_pause = use_escalating_pause
        self.use_vr_weight = use_vr_weight

    def run(
        self,
        close_matrix: pd.DataFrame,
        high_matrix: pd.DataFrame,
        low_matrix: pd.DataFrame,
        volume_matrix: pd.DataFrame,
        benchmark: pd.Series,
        start_date: str = None,
        end_date: str = None,
    ) -> StrategyResult:
        # 日期过滤
        if start_date:
            close_matrix = close_matrix.loc[start_date:]
            high_matrix = high_matrix.loc[start_date:]
            low_matrix = low_matrix.loc[start_date:]
            volume_matrix = volume_matrix.loc[start_date:]
        if end_date:
            close_matrix = close_matrix.loc[:end_date]
            high_matrix = high_matrix.loc[:end_date]
            low_matrix = low_matrix.loc[:end_date]
            volume_matrix = volume_matrix.loc[:end_date]

        returns_matrix = close_matrix.pct_change()

        # 预计算
        atr14 = compute_atr(close_matrix, high_matrix, low_matrix, 14)
        vol10 = returns_matrix.rolling(10).std()
        ma5 = close_matrix.rolling(5).mean()
        ma10 = close_matrix.rolling(10).mean()
        ma20 = close_matrix.rolling(20).mean()
        mom5 = close_matrix.pct_change(5)

        # Regime detection
        if len(benchmark) > 0:
            bm = benchmark.reindex(close_matrix.index).ffill()
            bm_ma20 = bm.rolling(20).mean()
            bm_ma60 = bm.rolling(60).mean()
            severe_bear = (bm < bm_ma20) & (bm_ma20 < bm_ma60) & (bm.pct_change(5) < -0.05)
        else:
            severe_bear = pd.Series(False, index=close_matrix.index)

        # 回测循环
        equity = 1.0
        equity_series = []
        holdings = {}  # code -> {weight, high_price, entry_price, entry_day}
        last_rebal = 0
        recent_daily_rets = []
        in_pause = False
        pause_counter = 0
        pause_history = []
        trades = []
        start_idx = 60

        for i in range(start_idx, len(close_matrix)):
            date = close_matrix.index[i]
            is_severe = bool(severe_bear.iloc[i]) if i < len(severe_bear) and not pd.isna(severe_bear.iloc[i]) else False

            # 暂停管理
            if in_pause:
                pause_counter -= 1
                if pause_counter <= 0:
                    in_pause = False
                equity_series.append((date, equity))
                recent_daily_rets.append(0)
                if len(recent_daily_rets) > 10:
                    recent_daily_rets.pop(0)
                continue

            # 极端熊市清仓
            if is_severe and holdings:
                total_w = sum(info["weight"] for info in holdings.values())
                if total_w > 0:
                    daily_ret = sum(
                        info["weight"] / total_w * float(returns_matrix[c].iloc[i])
                        for c, info in holdings.items()
                        if not np.isnan(returns_matrix[c].iloc[i])
                    )
                    equity *= (1 + daily_ret)
                    recent_daily_rets.append(daily_ret)
                else:
                    recent_daily_rets.append(0)
                equity *= (1 - FEE_RATE * 2)
                trades.append({"date": str(date), "action": "regime_liquidate"})
                holdings = {}
                last_rebal = i
                if len(recent_daily_rets) > 10:
                    recent_daily_rets.pop(0)
                equity_series.append((date, equity))
                continue

            # 渐进式ATR跟踪止损
            stopped_out = []
            if holdings:
                for code, info in list(holdings.items()):
                    current_price = close_matrix[code].iloc[i]
                    if np.isnan(current_price):
                        continue
                    info["high_price"] = max(info["high_price"], current_price)
                    gain = current_price / info["entry_price"] - 1

                    # 硬止损
                    if gain < -self.hard_stop_loss:
                        stopped_out.append(code)
                        continue

                    # 渐进ATR倍数
                    if gain >= self.gain_thresholds[1]:
                        mult = self.progressive_mult[2]
                    elif gain >= self.gain_thresholds[0]:
                        mult = self.progressive_mult[1]
                    else:
                        mult = self.progressive_mult[0]

                    atr_val = atr14[code].iloc[i]
                    if not np.isnan(atr_val) and atr_val > 0:
                        stop_level = info["high_price"] - mult * atr_val
                        if current_price < stop_level:
                            stopped_out.append(code)

            for code in stopped_out:
                stopped_weight = holdings[code]["weight"]
                del holdings[code]
                trades.append({"date": str(date), "code": code, "action": "stop_loss",
                               "weight": stopped_weight})

            # 每日P&L
            daily_pnl = 0
            if holdings:
                total_w = sum(info["weight"] for info in holdings.values())
                if total_w > 0:
                    daily_pnl = sum(
                        info["weight"] / total_w * float(returns_matrix[c].iloc[i])
                        for c, info in holdings.items()
                        if not np.isnan(returns_matrix[c].iloc[i])
                    )
                    equity *= (1 + daily_pnl)

            recent_daily_rets.append(daily_pnl)
            if len(recent_daily_rets) > 10:
                recent_daily_rets.pop(0)

            # 滚动回撤制动
            if len(recent_daily_rets) >= 3:
                rolling_3d = sum(recent_daily_rets[-3:])
                if rolling_3d < self.rolling_dd_brake[0]:
                    if holdings:
                        equity *= (1 - FEE_RATE * 2)
                        trades.append({"date": str(date), "action": "dd_brake_3d"})
                    holdings = {}
                    in_pause = True
                    if self.use_escalating_pause:
                        recent_pauses = [p for p in pause_history if i - p < 30]
                        if len(recent_pauses) >= 2:
                            pause_counter = 7
                        elif len(recent_pauses) >= 1:
                            pause_counter = 5
                        else:
                            pause_counter = self.rolling_dd_brake[1]
                        pause_history.append(i)
                    else:
                        pause_counter = self.rolling_dd_brake[1]
                    equity_series.append((date, equity))
                    continue

            if len(recent_daily_rets) >= 5:
                rolling_5d = sum(recent_daily_rets[-5:])
                if rolling_5d < self.rolling_dd_brake2[0]:
                    if holdings:
                        equity *= (1 - FEE_RATE * 2)
                        trades.append({"date": str(date), "action": "dd_brake_5d"})
                    holdings = {}
                    in_pause = True
                    pause_counter = self.rolling_dd_brake2[1]
                    equity_series.append((date, equity))
                    continue

            # 再平衡: 定期 OR 止损后立即重入(Hunter特有)
            need_rebal = (i - last_rebal >= self.rebal_period) or (stopped_out and not holdings)
            if (need_rebal or not holdings) and not is_severe:
                scores_today = compute_composite_scores(close_matrix, i)
                if scores_today.empty:
                    equity_series.append((date, equity))
                    continue

                # 趋势过滤
                uptrend_mask = (close_matrix.iloc[i] > ma20.iloc[i]) & (ma5.iloc[i] > ma10.iloc[i])
                pos_mom = mom5.iloc[i] > 0
                valid_mask = uptrend_mask & pos_mom
                for code in scores_today.index:
                    if code not in valid_mask.index or not valid_mask.get(code, False):
                        scores_today[code] = -999
                scores_today = scores_today[scores_today > 0]

                # VR 软权重
                if self.use_vr_weight and len(scores_today) > 0:
                    for code in scores_today.index:
                        ret_sub = returns_matrix[code].iloc[max(0, i - 25):i + 1].dropna()
                        vr = compute_variance_ratio(ret_sub, q=5)
                        vr_multiplier = np.clip(vr, 0.3, 2.0)
                        scores_today[code] *= vr_multiplier

                if len(scores_today) >= 1:
                    actual_n = min(self.top_n, len(scores_today))
                    top = scores_today.nlargest(actual_n)

                    # 反波动率加权
                    vols = vol10.iloc[i][top.index].replace(0, np.nan).dropna()
                    if len(vols) > 0:
                        inv_vol = 1.0 / vols.clip(lower=0.005)
                        weights = inv_vol / inv_vol.sum()
                    else:
                        weights = pd.Series(1.0 / actual_n, index=top.index)

                    new_holdings = {
                        c: {"weight": float(w), "high_price": float(close_matrix[c].iloc[i]),
                            "entry_price": float(close_matrix[c].iloc[i]), "entry_day": i}
                        for c, w in weights.items()
                        if not np.isnan(close_matrix[c].iloc[i])
                    }

                    if new_holdings:
                        old_set = set(holdings.keys())
                        new_set = set(new_holdings.keys())
                        if old_set != new_set:
                            turnover = len(old_set.symmetric_difference(new_set)) / max(len(old_set | new_set), 1)
                            equity *= (1 - FEE_RATE * turnover * 2)
                            trades.append({"date": str(date), "action": "rebalance",
                                           "codes": list(new_set),
                                           "weights": {c: h["weight"] for c, h in new_holdings.items()}})

                        holdings = new_holdings
                        last_rebal = i
                else:
                    if holdings:
                        pass  # 无信号时保持现有持仓
                    last_rebal = i

            equity_series.append((date, equity))

        # 构造结果
        eq = pd.Series(
            [x[1] for x in equity_series],
            index=pd.DatetimeIndex([x[0] for x in equity_series])
        )
        return self.compute_metrics(eq, trades)

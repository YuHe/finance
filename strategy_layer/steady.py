"""
稳健模式 (Steady) - 低频策略
=============================
核心特点：
- 7天定期再平衡 + 止损后排除该ETF等待下个周期（不立即重入）
- 渐进式ATR跟踪止损: 0.8x(新仓) → 1.2x(盈利2%+) → 1.8x(盈利5%+)
- 硬止损: 入场价-2%
- 3日累计回撤<-2% → 暂停3天; 5日<-3% → 暂停5天
- 递进暂停: 30天内反复触发 → 延长
- VR(5)软权重: 趋势环境加权，震荡降权
- 止损ETF排除: 止损后不在同一周期内重买该ETF（关键频率降低机制）
- 极端regime过滤: 仅在基准<MA20<MA60且5日跌>5%时清仓
- Composite信号 + 反波动率加权 + Top2持仓

适合场景：希望更少人工关注、更低交易频率，适度牺牲收益换取稳定性
OOS 2026验证: Sharpe 6.94, 累计+43.3%
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


class SteadyStrategy(BaseStrategy):
    name = "steady"
    display_name = "稳健模式"
    description = "低频稳健型：7天再平衡 + 止损后排除等待下周期"

    def __init__(
        self,
        top_n: int = 2,
        rebal_period: int = 7,
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
        holdings = {}
        last_rebal = 0
        recent_daily_rets = []
        in_pause = False
        pause_counter = 0
        pause_history = []
        trades = []
        excluded_codes = set()  # 止损排除集合（本周期内不重买）
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
                valid_items = [
                    (c, info) for c, info in holdings.items()
                    if not np.isnan(returns_matrix[c].iloc[i])
                ]
                total_w = sum(info["weight"] for _, info in valid_items)
                if total_w > 0:
                    daily_ret = sum(
                        info["weight"] / total_w * float(returns_matrix[c].iloc[i])
                        for c, info in valid_items
                    )
                    equity *= (1 + daily_ret)
                    recent_daily_rets.append(daily_ret)
                else:
                    recent_daily_rets.append(0)
                equity *= (1 - FEE_RATE * 2)
                trades.append({"date": str(date), "action": "regime_liquidate"})
                holdings = {}
                excluded_codes.clear()
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

            # 每日P&L（必须在删除止损仓位之前计算，否则止损日亏损会丢失）
            daily_pnl = 0
            if holdings:
                # 只对有有效收益的持仓计算加权收益（排除停牌/NaN）
                valid_items = [
                    (c, info) for c, info in holdings.items()
                    if not np.isnan(returns_matrix[c].iloc[i])
                ]
                total_w = sum(info["weight"] for _, info in valid_items)
                if total_w > 0:
                    daily_pnl = sum(
                        info["weight"] / total_w * float(returns_matrix[c].iloc[i])
                        for c, info in valid_items
                    )
                    equity *= (1 + daily_pnl)

            for code in stopped_out:
                stopped_weight = holdings[code]["weight"]
                del holdings[code]
                excluded_codes.add(code)  # Steady特有: 加入排除集合
                trades.append({"date": str(date), "code": code, "action": "stop_loss",
                               "weight": stopped_weight})

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

            # 再平衡: 仅在定期周期到达时（Steady特有: 止损后不立即重入）
            # 信号基于 i-1 (昨日收盘) 计算，入场用 i (今日收盘) 执行 → 消除同Bar偏差
            need_rebal = (i - last_rebal >= self.rebal_period)
            if need_rebal and not is_severe:
                # 新周期开始：清除上周期排除项，但保留今天止损的ETF
                today_stopped = set(stopped_out)
                excluded_codes = today_stopped.copy()

                scores_today = compute_composite_scores(close_matrix, i - 1)
                if scores_today.empty:
                    equity_series.append((date, equity))
                    last_rebal = i
                    continue

                # 趋势过滤（基于昨日数据）
                uptrend_mask = (close_matrix.iloc[i - 1] > ma20.iloc[i - 1]) & (ma5.iloc[i - 1] > ma10.iloc[i - 1])
                pos_mom = mom5.iloc[i - 1] > 0
                valid_mask = uptrend_mask & pos_mom
                for code in scores_today.index:
                    if code not in valid_mask.index or not valid_mask.get(code, False):
                        scores_today[code] = -999
                scores_today = scores_today[scores_today > 0]

                # 排除本周期止损的ETF（Steady特有）
                scores_today = scores_today.drop(
                    labels=[c for c in excluded_codes if c in scores_today.index],
                    errors='ignore'
                )

                # VR 软权重（基于昨日及之前数据）
                if self.use_vr_weight and len(scores_today) > 0:
                    for code in scores_today.index:
                        ret_sub = returns_matrix[code].iloc[max(0, i - 26):i].dropna()
                        vr = compute_variance_ratio(ret_sub, q=5)
                        vr_multiplier = np.clip(vr, 0.3, 2.0)
                        scores_today[code] *= vr_multiplier

                if len(scores_today) >= 1:
                    actual_n = min(self.top_n, len(scores_today))
                    top = scores_today.nlargest(actual_n)

                    # 反波动率加权（基于昨日波动率）
                    vols = vol10.iloc[i - 1][top.index].replace(0, np.nan).dropna()
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
                        # 计算真实换手率：基于权重差异而非仅代码变化
                        old_weights = {c: info["weight"] for c, info in holdings.items()}
                        new_weights = {c: info["weight"] for c, info in new_holdings.items()}
                        all_codes = set(old_weights.keys()) | set(new_weights.keys())
                        turnover = sum(abs(new_weights.get(c, 0) - old_weights.get(c, 0)) for c in all_codes) / 2
                        if turnover > 0.01:  # 忽略极小权重调整
                            equity *= (1 - FEE_RATE * turnover * 2)
                            trades.append({"date": str(date), "action": "rebalance",
                                           "codes": list(new_holdings.keys()),
                                           "weights": {c: h["weight"] for c, h in new_holdings.items()}})

                        holdings = new_holdings
                        last_rebal = i
                else:
                    if holdings:
                        pass
                    last_rebal = i

            equity_series.append((date, equity))

        # 构造结果
        eq = pd.Series(
            [x[1] for x in equity_series],
            index=pd.DatetimeIndex([x[0] for x in equity_series])
        )
        return self.compute_metrics(eq, trades)

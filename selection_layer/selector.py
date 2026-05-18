"""选股层 - 基于因子评分选Top N，生成目标权重"""

import pandas as pd
import numpy as np
from enum import Enum


class WeightMethod(str, Enum):
    EQUAL = "equal"                  # 等权
    INVERSE_VOL = "inverse_vol"     # 逆波动率加权
    MOMENTUM_WEIGHTED = "momentum_weighted"  # 动量加权


class Selector:
    """
    选股器：从因子评分中选Top N，分配权重

    参数:
        top_n: 持仓数（默认3）
        weight_method: 加权方式
        vol_window: 逆波动率窗口（天数）
    """

    def __init__(self, top_n: int = 3, weight_method: WeightMethod = WeightMethod.EQUAL, vol_window: int = 20):
        self.top_n = top_n
        self.weight_method = weight_method
        self.vol_window = vol_window

    def select(
        self,
        factor_scores: pd.DataFrame,
        cash_ratio: float,
        close_matrix: pd.DataFrame = None,
    ) -> dict[str, float]:
        """
        选股并分配权重

        参数:
            factor_scores: 因子评分 DataFrame (columns: code, score, trend, ...)
            cash_ratio: 现金比例 (0~1)
            close_matrix: 收盘价矩阵（逆波动率加权时需要）

        返回:
            {code: weight} 目标权重字典（含 "CASH" 键）
        """
        if factor_scores.empty or cash_ratio >= 1.0:
            return {"CASH": 1.0}

        # 过滤：趋势不符合的剔除（MA5 < MA20）
        eligible = factor_scores[factor_scores["trend"] == 1].copy()
        if eligible.empty:
            # 无符合趋势条件的，退化为取动量最高的
            eligible = factor_scores.head(self.top_n).copy()

        # 取Top N
        selected = eligible.head(self.top_n)
        codes = selected["code"].tolist()

        # 分配权重
        equity_ratio = 1.0 - cash_ratio

        if self.weight_method == WeightMethod.INVERSE_VOL and close_matrix is not None:
            weights = self._inverse_vol_weights(codes, close_matrix)
        elif self.weight_method == WeightMethod.MOMENTUM_WEIGHTED:
            weights = self._momentum_weights(codes, selected)
        else:
            weights = {code: 1.0 / len(codes) for code in codes}

        # 乘以权益比例
        result = {code: w * equity_ratio for code, w in weights.items()}
        result["CASH"] = cash_ratio

        return result

    def _inverse_vol_weights(self, codes: list[str], close_matrix: pd.DataFrame) -> dict[str, float]:
        """逆波动率加权：波动越低权重越高"""
        vols = {}
        for code in codes:
            if code in close_matrix.columns:
                returns = close_matrix[code].pct_change().dropna().iloc[-self.vol_window:]
                vol = returns.std()
                vols[code] = vol if vol > 0 else 1e-6
            else:
                vols[code] = 1e-6

        # 逆波动率
        inv_vols = {code: 1.0 / vol for code, vol in vols.items()}
        total = sum(inv_vols.values())
        return {code: iv / total for code, iv in inv_vols.items()}

    def _momentum_weights(self, codes: list[str], selected: pd.DataFrame) -> dict[str, float]:
        """动量加权：动量值越高权重越高（softmax归一化）"""
        momentums = {}
        for code in codes:
            row = selected[selected["code"] == code]
            if not row.empty:
                m = max(row.iloc[0]["momentum"], 0.001)  # 确保正值
                momentums[code] = m
            else:
                momentums[code] = 0.001
        total = sum(momentums.values())
        return {code: m / total for code, m in momentums.items()}

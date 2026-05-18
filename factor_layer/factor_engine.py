"""因子层 - 精简为：动量 + 趋势 + 量价确认"""

import pandas as pd
import numpy as np


class FactorEngine:
    """计算三类核心因子，输出综合评分"""

    def __init__(self, momentum_window: int = 20, ma_short: int = 5, ma_mid: int = 20, ma_long: int = 60):
        self.momentum_window = momentum_window
        self.ma_short = ma_short
        self.ma_mid = ma_mid
        self.ma_long = ma_long

    def compute_factors(self, close_matrix: pd.DataFrame, amount_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有ETF在最新日期的因子评分。

        参数:
            close_matrix: 收盘价矩阵 (date x code)
            amount_matrix: 成交额矩阵 (date x code)

        返回:
            DataFrame with columns: [code, momentum, trend, volume_confirm, score]
        """
        if close_matrix.empty or len(close_matrix) < self.ma_long:
            return pd.DataFrame(columns=["code", "momentum", "trend", "volume_confirm", "score"])

        results = []
        for code in close_matrix.columns:
            close = close_matrix[code].dropna()
            if len(close) < self.ma_long:
                continue

            # 动量：momentum_window 日涨幅（从 -window-1 到 -1，共 window 个交易日）
            momentum = close.iloc[-1] / close.iloc[-(self.momentum_window + 1)] - 1 if len(close) > self.momentum_window else 0

            # 趋势：MA5 > MA20 > MA60
            ma_short = close.rolling(self.ma_short).mean().iloc[-1]
            ma_mid = close.rolling(self.ma_mid).mean().iloc[-1]
            ma_long = close.rolling(self.ma_long).mean().iloc[-1]
            trend = 1 if (ma_short > ma_mid > ma_long) else 0

            # 量价确认：5日均成交额 > 20日均成交额
            volume_confirm = 0
            if code in amount_matrix.columns:
                amt = amount_matrix[code].dropna()
                if len(amt) >= self.ma_mid:
                    amt_5 = amt.iloc[-5:].mean()
                    amt_20 = amt.iloc[-20:].mean()
                    volume_confirm = 1 if amt_5 > amt_20 else 0

            results.append({
                "code": code,
                "momentum": momentum,
                "trend": trend,
                "volume_confirm": volume_confirm,
            })

        if not results:
            return pd.DataFrame(columns=["code", "momentum", "trend", "volume_confirm", "score"])

        df = pd.DataFrame(results)

        # 综合评分：动量排名分 + 趋势加分 + 量价加分
        df["momentum_rank"] = df["momentum"].rank(pct=True)  # 0-1
        df["score"] = df["momentum_rank"] + df["trend"] * 0.2 + df["volume_confirm"] * 0.1
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        return df

    def compute_factors_at_date(
        self, close_matrix: pd.DataFrame, amount_matrix: pd.DataFrame, date: str
    ) -> pd.DataFrame:
        """计算指定日期截面的因子评分（用于回测）"""
        # 截取到指定日期的数据
        close_sub = close_matrix.loc[:date]
        amount_sub = amount_matrix.loc[:date]
        return self.compute_factors(close_sub, amount_sub)

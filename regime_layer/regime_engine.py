"""环境层 - 基于沪深300判断牛/震荡/熊/极端熊，映射现金比例"""

import pandas as pd
import numpy as np
from enum import Enum


class MarketRegime(str, Enum):
    BULL = "bull"          # 牛市
    NEUTRAL = "neutral"    # 震荡
    BEAR = "bear"          # 熊市
    EXTREME_BEAR = "extreme_bear"  # 极端熊市


# 环境 → 现金比例
REGIME_CASH_RATIO = {
    MarketRegime.BULL: 0.0,
    MarketRegime.NEUTRAL: 0.3,
    MarketRegime.BEAR: 0.6,
    MarketRegime.EXTREME_BEAR: 1.0,
}


class RegimeEngine:
    """
    市场环境判断（基于沪深300 ETF）

    规则：
    - 牛市: price > MA60 且 MA20斜率 > 0
    - 震荡: price > MA60 但 MA20斜率 <= 0
    - 熊市: price < MA60
    - 极端熊市: price < MA60 且 20日跌幅 > 10%
    """

    def __init__(self, ma_mid: int = 20, ma_long: int = 60, extreme_threshold: float = -0.10):
        self.ma_mid = ma_mid
        self.ma_long = ma_long
        self.extreme_threshold = extreme_threshold

    def detect(self, benchmark_close: pd.Series) -> MarketRegime:
        """
        判断当前市场环境

        参数:
            benchmark_close: 沪深300ETF收盘价序列（需至少60个数据点）

        返回:
            MarketRegime 枚举值
        """
        if len(benchmark_close) < self.ma_long:
            return MarketRegime.NEUTRAL  # 数据不足默认震荡

        price = benchmark_close.iloc[-1]
        ma60 = benchmark_close.rolling(self.ma_long).mean().iloc[-1]
        ma20 = benchmark_close.rolling(self.ma_mid).mean()

        # MA20斜率：最近5日MA20的变化
        ma20_slope = (ma20.iloc[-1] - ma20.iloc[-6]) / ma20.iloc[-6] if len(ma20) >= 6 else 0

        # 20日涨跌幅
        ret_20 = price / benchmark_close.iloc[-self.ma_mid] - 1 if len(benchmark_close) >= self.ma_mid else 0

        if price < ma60:
            if ret_20 <= self.extreme_threshold:
                return MarketRegime.EXTREME_BEAR
            return MarketRegime.BEAR
        else:
            if ma20_slope > 0:
                return MarketRegime.BULL
            return MarketRegime.NEUTRAL

    def detect_at_date(self, benchmark_close: pd.Series, date: str) -> MarketRegime:
        """计算指定日期的环境（用于回测）"""
        sub = benchmark_close.loc[:date]
        return self.detect(sub)

    def get_cash_ratio(self, regime: MarketRegime) -> float:
        """获取对应环境的现金比例"""
        return REGIME_CASH_RATIO[regime]

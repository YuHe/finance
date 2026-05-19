"""策略基类和结果结构"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class StrategyResult:
    """策略回测结果"""
    nav_series: Optional[pd.Series] = None
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    trades: list = field(default_factory=list)


class BaseStrategy(ABC):
    """
    所有策略的基类。子类需实现 run() 方法。
    策略接收统一的价格数据，返回 StrategyResult。
    """

    # 策略元信息（子类覆盖）
    name: str = "base"
    display_name: str = "基础策略"
    description: str = ""

    @abstractmethod
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
        """执行策略回测，返回结果"""
        ...

    def compute_metrics(self, equity_series: pd.Series, trades: list) -> StrategyResult:
        """通用指标计算"""
        if equity_series is None or len(equity_series) < 20:
            return StrategyResult()

        total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
        n_years = len(equity_series) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0.1 else 0

        returns = equity_series.pct_change().dropna()
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        cummax = equity_series.cummax()
        drawdowns = (cummax - equity_series) / cummax
        max_dd = drawdowns.max()
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # 周胜率
        weekly = equity_series.resample("W").last().pct_change().dropna()
        win_rate = (weekly > 0).sum() / len(weekly) if len(weekly) > 0 else 0

        return StrategyResult(
            nav_series=equity_series / equity_series.iloc[0],  # 归一化为1起始
            total_return=total_return,
            annual_return=annual_return,
            annual_volatility=annual_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            total_trades=len(trades),
            trades=trades,
        )

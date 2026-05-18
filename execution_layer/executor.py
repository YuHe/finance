"""执行层 - T+1建模、涨跌停检查、成交量约束、交易成本"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class TradeOrder:
    """单笔交易订单"""
    code: str
    direction: str  # "buy" / "sell"
    target_weight: float
    actual_weight: float
    price: float  # 执行价（次日开盘价）
    shares: int
    amount: float
    fee: float
    skipped: bool = False  # 因涨跌停/流动性跳过
    skip_reason: str = ""


@dataclass
class ExecutionResult:
    """一次调仓的执行结果"""
    signal_date: str       # 信号产生日（计算日）
    execution_date: str    # 实际执行日（T+1）
    orders: list[TradeOrder] = field(default_factory=list)
    total_fee: float = 0.0
    actual_weights: dict = field(default_factory=dict)  # 实际达成的权重


class ExecutionEngine:
    """
    执行引擎：模拟真实交易约束

    - T+1: 信号日计算 → 次交易日开盘执行
    - 涨停不可买、跌停不可卖
    - 单只ETF单日买入 ≤ 前5日日均成交额的5%
    - 最小交易单位100份
    - 佣金0.05%（买卖各一）
    """

    def __init__(
        self,
        fee_rate: float = 0.0005,
        volume_limit_pct: float = 0.05,
        min_lot: int = 100,
    ):
        self.fee_rate = fee_rate
        self.volume_limit_pct = volume_limit_pct
        self.min_lot = min_lot

    def execute(
        self,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        execution_date_data: dict[str, dict],
        total_value: float,
        amount_5d_avg: dict[str, float],
    ) -> ExecutionResult:
        """
        模拟执行一次调仓

        参数:
            target_weights: 目标权重 {code: weight}
            current_weights: 当前权重 {code: weight}
            execution_date_data: 执行日各ETF数据 {code: {open, is_limit_up, is_limit_down}}
            total_value: 当前组合总价值
            amount_5d_avg: 前5日日均成交额 {code: avg_amount}

        返回:
            ExecutionResult
        """
        result = ExecutionResult(signal_date="", execution_date="")
        actual_weights = dict(current_weights)

        all_codes = sorted(set(list(target_weights.keys()) + list(current_weights.keys())) - {"CASH"})

        # 先卖后买：确保卖出释放资金后再买入，避免顺序影响结果
        sell_codes = [c for c in all_codes if target_weights.get(c, 0) < current_weights.get(c, 0)]
        buy_codes = [c for c in all_codes if target_weights.get(c, 0) >= current_weights.get(c, 0)]

        for code in sell_codes + buy_codes:
            target_w = target_weights.get(code, 0.0)
            current_w = current_weights.get(code, 0.0)
            delta_w = target_w - current_w

            if abs(delta_w) < 0.001:  # 变化太小不调
                actual_weights[code] = current_w
                continue

            data = execution_date_data.get(code, {})
            open_price = data.get("open", 0)
            is_limit_up = data.get("is_limit_up", False)
            is_limit_down = data.get("is_limit_down", False)

            if open_price <= 0:
                actual_weights[code] = current_w
                continue

            direction = "buy" if delta_w > 0 else "sell"

            # 涨停不可买
            if direction == "buy" and is_limit_up:
                result.orders.append(TradeOrder(
                    code=code, direction=direction,
                    target_weight=target_w, actual_weight=current_w,
                    price=open_price, shares=0, amount=0, fee=0,
                    skipped=True, skip_reason="涨停无法买入"
                ))
                actual_weights[code] = current_w
                continue

            # 跌停不可卖
            if direction == "sell" and is_limit_down:
                result.orders.append(TradeOrder(
                    code=code, direction=direction,
                    target_weight=target_w, actual_weight=current_w,
                    price=open_price, shares=0, amount=0, fee=0,
                    skipped=True, skip_reason="跌停无法卖出"
                ))
                actual_weights[code] = current_w
                continue

            # 计算目标金额
            target_amount = abs(delta_w) * total_value

            # 成交量约束（仅限买入）
            if direction == "buy":
                max_amount = amount_5d_avg.get(code, float("inf")) * self.volume_limit_pct
                if target_amount > max_amount:
                    target_amount = max_amount

            # 最小交易单位
            shares = int(target_amount / open_price / self.min_lot) * self.min_lot
            if shares == 0:
                actual_weights[code] = current_w
                continue

            actual_amount = shares * open_price
            fee = actual_amount * self.fee_rate
            actual_delta_w = actual_amount / total_value * (1 if direction == "buy" else -1)

            actual_weights[code] = current_w + actual_delta_w

            result.orders.append(TradeOrder(
                code=code, direction=direction,
                target_weight=target_w,
                actual_weight=actual_weights[code],
                price=open_price, shares=shares,
                amount=actual_amount, fee=fee,
            ))
            result.total_fee += fee

        # 现金 = 1 - 所有持仓权重之和
        equity_sum = sum(w for c, w in actual_weights.items() if c != "CASH")
        actual_weights["CASH"] = max(0, 1.0 - equity_sum)
        result.actual_weights = actual_weights

        return result

    def compute_daily_pnl(
        self,
        weights: dict[str, float],
        daily_returns: dict[str, float],
    ) -> float:
        """计算单日组合收益"""
        pnl = 0.0
        for code, w in weights.items():
            if code == "CASH":
                continue
            ret = daily_returns.get(code, 0.0)
            pnl += w * ret
        return pnl

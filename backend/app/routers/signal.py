"""信号 API - 每周计算轮动信号"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_layer import DataManager, BENCHMARK_CODE
from factor_layer import FactorEngine
from regime_layer import RegimeEngine
from selection_layer import Selector

router = APIRouter()

# 简易信号存储
_signals: list = []
_adopted_signals: list = []


class AdoptRequest(BaseModel):
    signal_id: int


@router.get("/latest")
def get_latest_signal():
    """计算最新信号"""
    dm = DataManager()
    factor_engine = FactorEngine()
    regime_engine = RegimeEngine()
    selector = Selector(top_n=3)

    close_matrix = dm.get_close_matrix()
    amount_matrix = dm.get_amount_matrix()
    benchmark_df = dm.get_daily(BENCHMARK_CODE)

    if close_matrix.empty or benchmark_df.empty:
        return {"error": "数据不足，请先更新数据"}

    benchmark_close = benchmark_df.set_index("date")["close"]

    # 因子评分
    factors = factor_engine.compute_factors(close_matrix, amount_matrix)

    # 环境
    regime = regime_engine.detect(benchmark_close)
    cash_ratio = regime_engine.get_cash_ratio(regime)

    # 选股
    target_weights = selector.select(factors, cash_ratio, close_matrix)

    signal = {
        "id": len(_signals),
        "regime": regime.value,
        "cash_ratio": cash_ratio,
        "target_weights": target_weights,
        "factors": factors.to_dict(orient="records") if not factors.empty else [],
        "status": "pending",
    }
    _signals.append(signal)

    return signal


@router.get("/history")
def get_signal_history():
    return _signals[-50:]  # 最近50条


@router.post("/adopt/{signal_id}")
def adopt_signal(signal_id: int):
    """采纳信号"""
    if signal_id >= len(_signals):
        return {"error": "signal not found"}
    _signals[signal_id]["status"] = "adopted"
    _adopted_signals.append(_signals[signal_id])
    return {"success": True, "signal": _signals[signal_id]}

"""信号 API - 每周计算轮动信号（含情绪过滤器）"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_layer import DataManager, BENCHMARK_CODE
from factor_layer import FactorEngine
from regime_layer import RegimeEngine
from selection_layer import Selector
from llm_layer.llm_engine import get_cached_sentiments, apply_sentiment_filter

from app.database import get_db
from app.deps import get_current_user
from app.models.user import User

router = APIRouter()

# 简易信号存储
_signals: list = []
_adopted_signals: list = []


class AdoptRequest(BaseModel):
    signal_id: int


@router.get("/latest")
def get_latest_signal(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """计算最新信号（含情绪过滤）"""
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

    # 选股（原始 Top N）
    target_weights = selector.select(factors, cash_ratio, close_matrix)

    # === 情绪过滤器 V1 ===
    sentiment_data = get_cached_sentiments(user.id, db)
    sentiment_warnings = []
    if sentiment_data:
        equity_codes = [c for c in target_weights if c != "CASH"]
        filtered_codes, sentiment_warnings = apply_sentiment_filter(
            equity_codes, factors, sentiment_data
        )
        # 如果有剔除/降权，重新分配权重
        if set(filtered_codes) != set(equity_codes):
            equity_ratio = 1.0 - cash_ratio
            warn_codes = {w["code"] for w in sentiment_warnings if w["action"] == "warn"}
            n = len(filtered_codes)
            if n > 0:
                target_weights = {}
                for code in filtered_codes:
                    base_w = equity_ratio / n
                    if code in warn_codes:
                        base_w *= 0.5  # 降权 50%
                    target_weights[code] = base_w
                # 降权剩余的归入现金
                used = sum(target_weights.values())
                target_weights["CASH"] = 1.0 - used
            else:
                target_weights = {"CASH": 1.0}

    signal = {
        "id": len(_signals),
        "regime": regime.value,
        "cash_ratio": cash_ratio,
        "target_weights": target_weights,
        "factors": factors.to_dict(orient="records") if not factors.empty else [],
        "sentiment_warnings": sentiment_warnings,
        "sentiment_data": {k: v for k, v in sentiment_data.items()} if sentiment_data else {},
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


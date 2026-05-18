"""模拟盘 API - 持仓管理 / 交易记录 / 收益追踪"""

from fastapi import APIRouter
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

router = APIRouter()

# 简易模拟盘状态（生产应持久化到DB）
_portfolio_state = {
    "positions": {},  # {code: {shares, cost_price, current_price}}
    "cash": 1_000_000.0,
    "initial_capital": 1_000_000.0,
    "trades": [],
    "nav_history": [],  # [{date, nav, benchmark_nav}]
}


@router.get("/positions")
def get_positions():
    """当前持仓"""
    positions = []
    for code, info in _portfolio_state["positions"].items():
        pnl = (info["current_price"] - info["cost_price"]) / info["cost_price"] if info["cost_price"] > 0 else 0
        positions.append({
            "code": code,
            "shares": info["shares"],
            "cost_price": info["cost_price"],
            "current_price": info["current_price"],
            "market_value": info["shares"] * info["current_price"],
            "pnl_pct": pnl,
        })
    total_value = _portfolio_state["cash"] + sum(
        p["market_value"] for p in positions
    )
    return {
        "positions": positions,
        "cash": _portfolio_state["cash"],
        "total_value": total_value,
        "total_pnl": (total_value / _portfolio_state["initial_capital"]) - 1,
    }


@router.get("/trades")
def get_trades():
    """交易记录"""
    return _portfolio_state["trades"][-100:]


@router.get("/performance")
def get_performance():
    """收益表现"""
    return {
        "nav_history": _portfolio_state["nav_history"],
        "total_return": 0.0,  # TODO: compute from nav_history
    }

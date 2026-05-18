"""回测 API"""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import uuid
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtest import BacktestConfig, BacktestEngine

router = APIRouter()

# 简单内存存储回测结果（生产应换DB）
_results: dict = {}


class BacktestRequest(BaseModel):
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 1_000_000
    top_n: int = 3
    # 前端枚举: equal / momentum_weighted / inverse_volatility
    weight_method: str = "equal"
    rebalance_freq: str = "weekly"
    momentum_window: int = 20
    # 止损参数（前端传来，映射到 BacktestConfig）
    stop_loss_enabled: bool = True
    stop_loss_threshold: float = 0.08
    trailing_stop: bool = False
    trailing_stop_threshold: float = 0.05


def _weight_method_map(wm: str) -> str:
    """前端枚举 → 引擎枚举"""
    return {"equal": "equal", "momentum_weighted": "equal", "inverse_volatility": "inverse_vol"}.get(wm, "equal")


def _run_backtest(task_id: str, req: BacktestRequest):
    try:
        config = BacktestConfig(
            start_date=req.start_date,
            end_date=req.end_date,
            initial_capital=req.initial_capital,
            top_n=req.top_n,
            weight_method=_weight_method_map(req.weight_method),
            rebalance_freq=req.rebalance_freq,
            momentum_window=req.momentum_window,
            stop_loss_single=req.stop_loss_threshold if req.stop_loss_enabled else 1.0,
            stop_loss_portfolio=req.stop_loss_threshold * 1.5 if req.stop_loss_enabled else 1.0,
            stop_loss_circuit=req.stop_loss_threshold * 2.5 if req.stop_loss_enabled else 1.0,
        )
        engine = BacktestEngine(config)
        result = engine.run()

        # 检查是否因数据不足返回了空结果
        if result.nav_series is None or len(result.nav_series) == 0:
            _results[task_id] = {
                "status": "failed",
                "error": {"code": "NO_DATA", "message": "数据不足，请先点击"更新行情数据"获取数据"},
            }
            return

        # 转换 nav_history / benchmark_history 字段格式
        nav_history = [
            {"date": str(idx), "value": float(v)}
            for idx, v in result.nav_series.items()
        ]
        benchmark_history = [
            {"date": str(idx), "value": float(v)}
            for idx, v in result.benchmark_series.items()
        ] if result.benchmark_series is not None else []

        # 转换 trades 字段格式
        trades = [
            {
                "date": t["date"],
                "etf_code": t["code"],
                "etf_name": t.get("code", ""),   # 引擎暂未传name，用code占位
                "direction": t["direction"],
                "price": t.get("price", 0),
                "volume": t.get("shares", 0),
                "amount": t.get("amount", 0),
                "reason": t.get("skip_reason", "") if t.get("skipped") else "",
            }
            for t in result.trades[:500]
        ]

        # 基准收益率
        bm_return = float(result.benchmark_series.iloc[-1] - 1) if (
            result.benchmark_series is not None and len(result.benchmark_series) > 0
        ) else 0.0

        _results[task_id] = {
            "status": "completed",
            "id": task_id,
            "metrics": {
                "total_return": result.total_return,
                "annual_return": result.annual_return,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "calmar_ratio": result.annual_return / result.max_drawdown if result.max_drawdown > 0 else 0,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "profit_factor": result.profit_loss_ratio,
                "volatility": result.annual_volatility,
                "benchmark_return": bm_return,
                "alpha": result.annual_return - bm_return,
                "beta": 1.0,  # 暂不计算
            },
            "nav_history": nav_history,
            "benchmark_history": benchmark_history,
            "trades": trades,
        }
    except Exception as e:
        _results[task_id] = {
            "status": "failed",
            "error": {"code": "ENGINE_ERROR", "message": str(e)},
        }


@router.post("/run")
def run_backtest(req: BacktestRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    _results[task_id] = {"status": "running"}
    background_tasks.add_task(_run_backtest, task_id, req)
    # 包装为前端期望的 {success, data} 格式
    return {"success": True, "data": {"id": task_id}, "error": None}


@router.get("/result/{task_id}")
def get_result(task_id: str):
    if task_id not in _results:
        return {"success": False, "data": None, "error": {"code": "NOT_FOUND", "message": "任务不存在"}}
    return {"success": True, "data": _results[task_id], "error": None}

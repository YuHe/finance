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
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 1_000_000
    top_n: int = 3
    weight_method: str = "equal"
    rebalance_freq: str = "weekly"
    momentum_window: int = 20
    ma_short: int = 5
    ma_mid: int = 20
    ma_long: int = 60
    stop_loss_single: float = 0.08
    stop_loss_portfolio: float = 0.12
    stop_loss_circuit: float = 0.20


def _run_backtest(task_id: str, req: BacktestRequest):
    config = BacktestConfig(**req.model_dump())
    engine = BacktestEngine(config)
    result = engine.run()

    _results[task_id] = {
        "status": "done",
        "metrics": {
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "annual_volatility": result.annual_volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
        },
        "nav_series": result.nav_series.reset_index().to_dict(orient="records") if result.nav_series is not None else [],
        "benchmark_series": result.benchmark_series.reset_index().to_dict(orient="records") if result.benchmark_series is not None else [],
        "trades": result.trades[:500],  # 限制返回量
        "positions_history": result.positions_history[:200],
    }


@router.post("/run")
def run_backtest(req: BacktestRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    _results[task_id] = {"status": "running"}
    background_tasks.add_task(_run_backtest, task_id, req)
    return {"task_id": task_id, "status": "running"}


@router.get("/result/{task_id}")
def get_result(task_id: str):
    if task_id not in _results:
        return {"error": "not_found"}
    return _results[task_id]

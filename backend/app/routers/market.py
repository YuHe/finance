"""市场数据 API - ETF池 / K线 / 环境 / 数据更新"""

from fastapi import APIRouter, BackgroundTasks
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_layer import DataManager, ETF_POOL, BENCHMARK_CODE
from data_layer.etf_pool import BENCHMARK_NAME
from regime_layer import RegimeEngine

router = APIRouter()

# 更新任务状态（内存，单任务）
_update_state: dict = {"status": "idle", "message": "", "progress": 0, "total": 0}


@router.get("/etfs")
def get_etf_pool():
    """ETF池列表"""
    return {"etfs": ETF_POOL, "benchmark": {"code": BENCHMARK_CODE, "name": BENCHMARK_NAME}}


@router.get("/kline/{code}")
def get_kline(code: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
    """K线数据"""
    dm = DataManager()
    df = dm.get_daily(code, start_date, end_date)
    if df.empty:
        return {"data": []}
    df["date"] = df["date"].astype(str)
    return {"data": df[["date", "open", "high", "low", "close", "volume", "amount"]].to_dict(orient="records")}


@router.get("/regime")
def get_regime():
    """当前市场环境"""
    dm = DataManager()
    benchmark_df = dm.get_daily(BENCHMARK_CODE)
    if benchmark_df.empty:
        return {"regime": "unknown", "cash_ratio": 0.3}

    benchmark_close = benchmark_df.set_index("date")["close"]
    engine = RegimeEngine()
    regime = engine.detect(benchmark_close)
    cash_ratio = engine.get_cash_ratio(regime)

    return {"regime": regime.value, "cash_ratio": cash_ratio}


@router.post("/update")
def trigger_update(background_tasks: BackgroundTasks):
    """触发数据更新（后台执行，立即返回）"""
    if _update_state["status"] == "running":
        return {"status": "running", "message": "更新已在进行中"}
    background_tasks.add_task(_do_update)
    return {"status": "started"}


@router.get("/update/status")
def get_update_status():
    """获取更新进度"""
    return _update_state


def _do_update():
    codes = [etf["code"] for etf in ETF_POOL] + [BENCHMARK_CODE]
    total = len(codes)
    _update_state.update({"status": "running", "progress": 0, "total": total, "message": "开始更新..."})
    dm = DataManager()
    for i, code in enumerate(codes, 1):
        name = next((e["name"] for e in ETF_POOL if e["code"] == code), code)
        _update_state["message"] = f"更新 {name}({code})..."
        _update_state["progress"] = i - 1
        try:
            dm.update_etf(code)
        except Exception as e:
            _update_state["message"] = f"{code} 更新失败: {e}"
    _update_state.update({"status": "done", "progress": total, "message": "全部更新完成"})

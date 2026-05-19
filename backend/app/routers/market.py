"""市场数据 API - ETF池 / K线 / 环境 / 数据更新"""

from fastapi import APIRouter, BackgroundTasks
from typing import Optional
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_layer import DataManager, ETF_POOL, BENCHMARK_CODE
from data_layer.etf_pool import BENCHMARK_NAME
from regime_layer import RegimeEngine

router = APIRouter()

# 更新任务状态（内存，单任务）+ 锁保护
_update_state: dict = {"status": "idle", "message": "", "progress": 0, "total": 0, "items": []}
_update_lock = threading.Lock()


@router.get("/etfs")
def get_etf_pool():
    """ETF池列表"""
    return {"etfs": ETF_POOL, "benchmark": {"code": BENCHMARK_CODE, "name": BENCHMARK_NAME}}


@router.get("/data-status")
def get_data_status():
    """获取所有 ETF 的数据覆盖状态（日期范围、记录数）"""
    dm = DataManager()
    status_list = dm.get_data_status()
    # 合并 ETF 名称和行业信息
    pool_map = {etf["code"]: etf for etf in ETF_POOL}
    result = []
    for item in status_list:
        code = item["code"]
        etf_info = pool_map.get(code, {})
        result.append({
            "code": code,
            "name": etf_info.get("name", BENCHMARK_NAME if code == BENCHMARK_CODE else code),
            "industry": etf_info.get("industry", "基准" if code == BENCHMARK_CODE else ""),
            "start_date": item["start_date"],
            "end_date": item["end_date"],
            "count": item["count"],
            "has_data": item["count"] > 0,
        })
    return {"data": result}


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
    with _update_lock:
        if _update_state["status"] == "running":
            return {"status": "running", "message": "更新已在进行中"}
        _update_state["status"] = "running"
    background_tasks.add_task(_do_update)
    return {"status": "started"}


@router.post("/reset")
def trigger_reset(background_tasks: BackgroundTasks):
    """清空所有数据后全量重拉（修复数据断裂）"""
    with _update_lock:
        if _update_state["status"] == "running":
            return {"status": "running", "message": "更新已在进行中"}
        _update_state["status"] = "running"
    background_tasks.add_task(_do_reset_and_update)
    return {"status": "started", "message": "已清空数据，开始全量重拉"}


@router.get("/update/status")
def get_update_status():
    """获取更新进度"""
    return _update_state


def _update_single_etf(code: str, db_path: str):
    """
    单个ETF更新（需要已登录BaoStock）
    """
    dm = DataManager(db_path=db_path)
    dm.update_etf(code, _logged_in=True)


def _do_update():
    """串行更新所有ETF（避免多进程并发写SQLite导致数据丢失）"""
    import baostock as bs

    codes = [etf["code"] for etf in ETF_POOL] + [BENCHMARK_CODE]
    total = len(codes)

    # 初始化每个标的的状态
    items = []
    for code in codes:
        name = next((e["name"] for e in ETF_POOL if e["code"] == code), BENCHMARK_NAME if code == BENCHMARK_CODE else code)
        items.append({"code": code, "name": name, "status": "pending", "error": ""})
    _update_state.update({"status": "running", "progress": 0, "total": total, "message": "开始更新...", "items": items})

    dm = DataManager()
    db_path = dm.db_path

    # 单次登录，串行更新（BaoStock 全局 socket + SQLite 单写者，串行最可靠）
    lg = bs.login()
    if lg.error_code != "0":
        _update_state.update({"status": "error", "message": f"BaoStock 登录失败: {lg.error_msg}"})
        return

    try:
        for i, code in enumerate(codes):
            items[i]["status"] = "running"
            _update_state["message"] = f"更新 {items[i]['name']}... ({i + 1}/{total})"
            try:
                _update_single_etf(code, db_path)
                items[i]["status"] = "ok"
            except Exception as e:
                items[i]["status"] = "error"
                items[i]["error"] = str(e)
            _update_state["progress"] = i + 1
    finally:
        bs.logout()

    errors = [it for it in items if it["status"] == "error"]
    if errors:
        _update_state.update({"status": "done", "progress": total,
                               "message": f"完成，{len(errors)} 个标的失败"})
    else:
        _update_state.update({"status": "done", "progress": total, "message": "全部更新完成"})


def _do_reset_and_update():
    """清空数据后全量重拉"""
    _update_state.update({"status": "running", "progress": 0, "total": 0, "message": "清空旧数据...", "items": []})
    dm = DataManager()
    dm.clear_all_data()
    _update_state["message"] = "旧数据已清空，开始全量拉取..."
    _do_update()

"""市场数据 API - ETF池 / K线 / 环境 / 数据更新"""

from fastapi import APIRouter, BackgroundTasks
from typing import Optional
import multiprocessing
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_layer import DataManager, ETF_POOL, BENCHMARK_CODE
from data_layer.etf_pool import BENCHMARK_NAME
from regime_layer import RegimeEngine

# 确保子进程使用 fork（Linux 默认，但显式设置更安全）
try:
    multiprocessing.set_start_method("fork", force=False)
except RuntimeError:
    pass  # 已设置过

router = APIRouter()

# 更新任务状态（内存，单任务）+ 锁保护
_update_state: dict = {"status": "idle", "message": "", "progress": 0, "total": 0, "items": []}
_update_lock = threading.Lock()


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
    with _update_lock:
        if _update_state["status"] == "running":
            return {"status": "running", "message": "更新已在进行中"}
        _update_state["status"] = "running"
    background_tasks.add_task(_do_update)
    return {"status": "started"}


@router.get("/update/status")
def get_update_status():
    """获取更新进度"""
    return _update_state


def _update_single_etf(code: str, db_path: str, start_date: str = "2015-01-01", end_date: str = None):
    """
    单个ETF更新（独立进程内执行）
    BaoStock 使用全局 socket，不支持多线程，必须用多进程隔离。
    fork 模式下子进程继承 sys.path 和已导入的模块。
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")
    try:
        dm = DataManager(db_path=db_path)
        dm.update_etf(code, start_date, end_date, _logged_in=True)
    finally:
        bs.logout()


def _process_worker(args):
    """进程池 worker（必须是顶层可 pickle 的函数）"""
    code, db_path = args
    try:
        _update_single_etf(code, db_path)
        return code, "ok", ""
    except Exception as e:
        return code, "error", str(e)


def _do_update():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    codes = [etf["code"] for etf in ETF_POOL] + [BENCHMARK_CODE]
    total = len(codes)

    # 初始化每个标的的状态
    items = []
    for code in codes:
        name = next((e["name"] for e in ETF_POOL if e["code"] == code), BENCHMARK_NAME if code == BENCHMARK_CODE else code)
        items.append({"code": code, "name": name, "status": "pending", "error": ""})
    _update_state.update({"status": "running", "progress": 0, "total": total, "message": "开始并行更新...", "items": items})

    # 获取 DB 路径传给子进程
    dm = DataManager()
    db_path = dm.db_path

    # 多进程并行（BaoStock 全局 socket 不支持多线程，必须用进程隔离）
    max_workers = min(4, total)
    code_to_idx = {item["code"]: i for i, item in enumerate(items)}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        work_args = [(code, db_path) for code in codes]
        futures = {pool.submit(_process_worker, args): args[0] for args in work_args}

        completed_count = 0
        for future in as_completed(futures):
            code, status, error = future.result()
            idx = code_to_idx[code]
            items[idx]["status"] = status
            items[idx]["error"] = error
            completed_count += 1
            _update_state["progress"] = completed_count
            _update_state["message"] = f"更新中... ({completed_count}/{total})"

    errors = [it for it in items if it["status"] == "error"]
    if errors:
        _update_state.update({"status": "done", "progress": total,
                               "message": f"完成，{len(errors)} 个标的失败"})
    else:
        _update_state.update({"status": "done", "progress": total, "message": "全部更新完成"})

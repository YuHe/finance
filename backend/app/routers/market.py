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


def _update_single_etf(code: str, start_date: str = "2015-01-01", end_date: str = None):
    """单个ETF更新（独立 BaoStock session + 独立 DB 连接）"""
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")
    try:
        dm = DataManager()
        dm.update_etf(code, start_date, end_date, _logged_in=True)
    finally:
        bs.logout()


def _do_update():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    codes = [etf["code"] for etf in ETF_POOL] + [BENCHMARK_CODE]
    total = len(codes)

    # 初始化每个标的的状态
    items = []
    for code in codes:
        name = next((e["name"] for e in ETF_POOL if e["code"] == code), BENCHMARK_NAME if code == BENCHMARK_CODE else code)
        items.append({"code": code, "name": name, "status": "pending", "error": ""})
    _update_state.update({"status": "running", "progress": 0, "total": total, "message": "开始并行更新...", "items": items})

    # 并行更新（BaoStock 每个线程独立 login/logout）
    max_workers = min(4, total)  # BaoStock 限制并发数，4线程较安全
    completed_count = 0

    def _worker(idx: int, code: str):
        nonlocal completed_count
        items[idx]["status"] = "running"
        try:
            _update_single_etf(code)
            items[idx]["status"] = "ok"
        except Exception as e:
            items[idx]["status"] = "error"
            items[idx]["error"] = str(e)
        finally:
            completed_count += 1
            _update_state["progress"] = completed_count
            _update_state["message"] = f"更新中... ({completed_count}/{total})"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, i, item["code"]): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            # 异常已在 _worker 内捕获
            pass

    errors = [it for it in items if it["status"] == "error"]
    if errors:
        _update_state.update({"status": "done", "progress": total,
                               "message": f"完成，{len(errors)} 个标的失败"})
    else:
        _update_state.update({"status": "done", "progress": total, "message": "全部更新完成"})

"""回测 API"""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Union
import uuid
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtest import BacktestConfig, BacktestEngine
from strategy_layer import get_strategy, STRATEGY_REGISTRY
from data_layer import DataManager

router = APIRouter()

# 回测结果存储（带 TTL 清理）
_results: dict = {}
_RESULT_TTL = 3600  # 结果保留1小时
_MAX_RESULTS = 50   # 最多保留50个结果


def _calc_beta(nav_series, benchmark_series) -> float:
    """计算策略相对基准的 Beta (CAPM)"""
    if nav_series is None or benchmark_series is None:
        return 0.0
    if len(nav_series) < 5 or len(benchmark_series) < 5:
        return 0.0
    strat_returns = np.diff(np.array(nav_series, dtype=float)) / np.array(nav_series[:-1], dtype=float)
    bench_returns = np.diff(np.array(benchmark_series, dtype=float)) / np.array(benchmark_series[:-1], dtype=float)
    min_len = min(len(strat_returns), len(bench_returns))
    strat_returns = strat_returns[:min_len]
    bench_returns = bench_returns[:min_len]
    var_bench = np.var(bench_returns)
    if var_bench < 1e-12:
        return 0.0
    cov = np.cov(strat_returns, bench_returns)[0, 1]
    return float(cov / var_bench)


def _cleanup_results():
    """清理过期结果"""
    if len(_results) <= _MAX_RESULTS:
        return
    now = time.time()
    expired = [k for k, v in _results.items() if now - v.get("_ts", 0) > _RESULT_TTL]
    for k in expired:
        del _results[k]
    # 如果还是超限，删除最旧的
    if len(_results) > _MAX_RESULTS:
        sorted_keys = sorted(_results.keys(), key=lambda k: _results[k].get("_ts", 0))
        for k in sorted_keys[:len(_results) - _MAX_RESULTS]:
            del _results[k]


class BacktestRequest(BaseModel):
    strategy_type: str = "adaptive_premium"
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 1_000_000
    top_n: int = 2
    # 前端枚举: equal / momentum_weighted / inverse_volatility
    weight_method: str = "equal"
    rebalance_freq: str = "weekly"
    momentum_window: int = 20
    # 止损参数（前端传来，映射到 BacktestConfig）
    stop_loss_enabled: bool = True
    stop_loss_threshold: float = 0.08
    trailing_stop: bool = False
    trailing_stop_threshold: float = 0.05
    # 可选：选择参与回测的 ETF 代码列表（空则全池）
    selected_codes: Optional[list[str]] = Field(default=None)
    # 新策略参数
    hold_days: int = 7
    fee: float = 0.001
    w_sharpe: float = 0.4
    w_rs: float = 0.3
    w_premium: float = 0.3
    w_momq: float = 0.15
    w_sharpe5: float = 0.20


def _weight_method_map(wm: str) -> str:
    """前端枚举 → 引擎枚举"""
    return {
        "equal": "equal",
        "momentum_weighted": "momentum_weighted",
        "inverse_volatility": "inverse_vol",
    }.get(wm, "equal")


def _run_backtest(task_id: str, req: BacktestRequest):
    try:
        # === 新策略 (hunter / steady) ===
        if req.strategy_type in STRATEGY_REGISTRY:
            _run_strategy_backtest(task_id, req)
            return

        # === 经典策略 (classic) ===
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
            selected_codes=req.selected_codes,
        )
        engine = BacktestEngine(config)

        # 诊断：检查可用数据
        import os as _os
        _db = _os.environ.get("ETF_DB_PATH", "NOT_SET")
        codes = req.selected_codes if req.selected_codes else None
        _cm = engine.data_mgr.get_close_matrix(config.start_date, config.end_date, codes=codes)
        print(f"[backtest] DB={_db} close_matrix shape={_cm.shape} start={config.start_date} end={config.end_date} selected={len(req.selected_codes) if req.selected_codes else 'all'}")

        if _cm.empty:
            _results[task_id] = {
                "status": "failed", "_ts": time.time(),
                "error": {"code": "NO_DATA", "message": "数据库中没有ETF数据，请先点击侧边栏更新行情数据"},
            }
            return
        if _cm.shape[1] < config.top_n:
            _results[task_id] = {
                "status": "failed", "_ts": time.time(),
                "error": {"code": "INSUFFICIENT_DATA",
                          "message": f"可用ETF数({_cm.shape[1]})不足，需要至少{config.top_n}个，请重新更新行情数据"},
            }
            return

        result = engine.run()

        # 检查是否因数据不足返回了空结果
        if result.nav_series is None or len(result.nav_series) == 0:
            _results[task_id] = {
                "status": "failed", "_ts": time.time(),
                "error": {"code": "NO_DATA", "message": f"回测区间内数据不足（可用ETF:{_cm.shape[1]}个），请尝试缩短回测时间范围"},
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
                "etf_name": t.get("code", ""),
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
            "status": "completed", "_ts": time.time(),
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
                "beta": _calc_beta(
                    result.nav_series.values if result.nav_series is not None else None,
                    result.benchmark_series.values if result.benchmark_series is not None else None,
                ),
            },
            "nav_history": nav_history,
            "benchmark_history": benchmark_history,
            "trades": trades,
        }
    except Exception as e:
        _results[task_id] = {
            "status": "failed", "_ts": time.time(),
            "error": {"code": "ENGINE_ERROR", "message": str(e)},
        }


def _run_strategy_backtest(task_id: str, req: BacktestRequest):
    """运行新策略 (hunter / steady)"""
    try:
        import pandas as pd

        dm = DataManager()
        codes = req.selected_codes if req.selected_codes else None
        close_matrix = dm.get_close_matrix(req.start_date, req.end_date, codes=codes)

        if close_matrix.empty:
            _results[task_id] = {
                "status": "failed", "_ts": time.time(),
                "error": {"code": "NO_DATA", "message": "数据库中没有ETF数据，请先点击侧边栏更新行情数据"},
            }
            return

        # 获取 high/low/volume 矩阵
        high_matrix = dm.get_high_matrix(req.start_date, req.end_date, codes=codes)
        low_matrix = dm.get_low_matrix(req.start_date, req.end_date, codes=codes)
        volume_matrix = dm.get_amount_matrix(req.start_date, req.end_date, codes=codes)

        # 基准
        from data_layer import BENCHMARK_CODE
        bm_df = dm.get_daily(BENCHMARK_CODE, req.start_date, req.end_date)
        if not bm_df.empty:
            benchmark = bm_df.set_index("date")["close"]
        else:
            benchmark = pd.Series(dtype=float)

        # 对齐矩阵
        high_matrix = high_matrix.reindex(close_matrix.index).ffill()
        low_matrix = low_matrix.reindex(close_matrix.index).ffill()
        volume_matrix = volume_matrix.reindex(close_matrix.index).fillna(0)
        # 用close填充缺失的high/low
        high_matrix = high_matrix.fillna(close_matrix)
        low_matrix = low_matrix.fillna(close_matrix)
        close_matrix = close_matrix.ffill()

        # 实例化策略
        strategy_kwargs = {"top_n": req.top_n, "hold_days": req.hold_days, "fee": req.fee}
        if req.strategy_type == "adaptive_premium":
            strategy_kwargs.update(w_sharpe=req.w_sharpe, w_rs=req.w_rs, w_premium=req.w_premium)
        elif req.strategy_type == "momentum_quality":
            strategy_kwargs.update(w_sharpe=req.w_sharpe, w_rs=req.w_rs,
                                   w_premium=req.w_premium, w_momq=req.w_momq, w_sharpe5=req.w_sharpe5)
        strategy = get_strategy(req.strategy_type, **strategy_kwargs)
        result = strategy.run(
            close_matrix, high_matrix, low_matrix, volume_matrix, benchmark,
            start_date=req.start_date, end_date=req.end_date,
        )

        if result.nav_series is None or len(result.nav_series) == 0:
            _results[task_id] = {
                "status": "failed", "_ts": time.time(),
                "error": {"code": "NO_DATA", "message": "回测区间内数据不足，请尝试缩短时间范围"},
            }
            return

        # 构造输出
        nav_history = [
            {"date": str(idx), "value": float(v)}
            for idx, v in result.nav_series.items()
        ]

        # 基准对齐
        bm_return = 0.0
        benchmark_history = []
        if len(benchmark) > 0:
            bm_aligned = benchmark.reindex(result.nav_series.index).ffill().dropna()
            if len(bm_aligned) > 1:
                bm_nav = bm_aligned / bm_aligned.iloc[0]
                benchmark_history = [
                    {"date": str(idx), "value": float(v)}
                    for idx, v in bm_nav.items()
                ]
                bm_return = float(bm_nav.iloc[-1] - 1)

        # 交易记录格式化 — 从 close_matrix 反查价格, 用 nav + 真实权重算金额
        from data_layer.etf_pool import ETF_POOL
        _etf_names = {e["code"]: e["name"] for e in ETF_POOL}

        # Build nav lookup: date_str -> nav_value
        nav_lookup = {str(idx): float(v) for idx, v in result.nav_series.items()}

        trades = []
        for t in result.trades[:500]:
            date_str = t.get("date", "")
            action = t.get("action", "")
            is_sell = action in ("stop_loss", "regime_liquidate", "dd_brake_3d", "dd_brake_5d")
            trade_weights = t.get("weights", {})  # {code: weight} from strategy

            # Get codes involved
            if "codes" in t:
                codes_list = t["codes"]
            elif "code" in t:
                codes_list = [t["code"]]
            else:
                codes_list = []

            # Portfolio value at trade date
            nav_at_trade = nav_lookup.get(date_str, 1.0)
            portfolio_value = req.initial_capital * nav_at_trade

            for code in codes_list:
                # Use real weight if available, else equal split
                if trade_weights and code in trade_weights:
                    weight = trade_weights[code]
                elif "weight" in t:
                    weight = t["weight"]  # stop_loss single ETF weight
                else:
                    weight = 1.0 / max(len(codes_list), 1)

                etf_amount = portfolio_value * weight

                # Lookup price from close_matrix
                price = 0.0
                try:
                    date_idx = close_matrix.index.get_loc(pd.Timestamp(date_str))
                    if code in close_matrix.columns:
                        price = float(close_matrix[code].iloc[date_idx])
                except (KeyError, IndexError):
                    pass

                volume = int(etf_amount / price / 100) * 100 if price > 0 else 0
                amount = volume * price if volume > 0 else round(etf_amount, 2)

                trades.append({
                    "date": date_str,
                    "etf_code": code,
                    "etf_name": _etf_names.get(code, code),
                    "direction": "sell" if is_sell else "buy",
                    "price": round(price, 4),
                    "volume": volume,
                    "amount": round(amount, 2),
                    "reason": action,
                })

        _results[task_id] = {
            "status": "completed", "_ts": time.time(),
            "id": task_id,
            "metrics": {
                "total_return": result.total_return,
                "annual_return": result.annual_return,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "calmar_ratio": result.calmar_ratio,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "profit_factor": 0,
                "volatility": result.annual_volatility,
                "benchmark_return": bm_return,
                "alpha": result.annual_return - bm_return,
                "beta": _calc_beta(
                    result.nav_series.values if result.nav_series is not None else None,
                    bm_nav.values if len(benchmark_history) > 0 else None,
                ),
            },
            "nav_history": nav_history,
            "benchmark_history": benchmark_history,
            "trades": trades,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        _results[task_id] = {
            "status": "failed", "_ts": time.time(),
            "error": {"code": "ENGINE_ERROR", "message": str(e)},
        }


@router.post("/run")
def run_backtest(req: BacktestRequest, background_tasks: BackgroundTasks):
    _cleanup_results()
    task_id = str(uuid.uuid4())
    _results[task_id] = {"status": "running", "_ts": time.time()}
    background_tasks.add_task(_run_backtest, task_id, req)
    return {"success": True, "data": {"id": task_id}, "error": None}


@router.get("/strategies")
def list_strategies():
    """返回所有可用策略"""
    strategies = [
        {
            "id": "adaptive_premium",
            "name": "自适应折溢价",
            "description": "信用脉冲择时 + Sharpe/RS/折溢价三信号选股，适合大ETF池(71只)，年化32.8%/Sharpe 1.67",
            "configurable": True,
        },
        {
            "id": "momentum_quality",
            "name": "动量质量",
            "description": "5信号选股(Sharpe/RS/折溢价/动量质量/短期Sharpe) + 信用脉冲择时，年化32.1%/Sharpe 1.83",
            "configurable": True,
        },
        {
            "id": "classic",
            "name": "经典动量轮动",
            "description": "基于动量+趋势+量价确认的传统轮动策略，可自定义参数",
            "configurable": True,
        },
    ]
    return {"success": True, "data": strategies, "error": None}


@router.get("/result/{task_id}")
def get_result(task_id: str):
    if task_id not in _results:
        return {"success": False, "data": None, "error": {"code": "NOT_FOUND", "message": "任务不存在"}}
    result = _results[task_id]
    # 返回时去掉内部时间戳
    return {"success": True, "data": {k: v for k, v in result.items() if k != "_ts"}, "error": None}

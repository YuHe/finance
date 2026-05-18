"""回测引擎 - 周频调仓 + T+1执行 + 真实约束"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from data_layer import DataManager, BENCHMARK_CODE
from factor_layer import FactorEngine
from regime_layer import RegimeEngine, MarketRegime
from selection_layer import Selector, WeightMethod
from execution_layer import ExecutionEngine


@dataclass
class BacktestConfig:
    """回测参数配置"""
    start_date: str = "2018-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 1_000_000.0
    # 策略参数
    top_n: int = 3
    weight_method: str = "equal"  # "equal" / "inverse_vol" / "momentum_weighted"
    rebalance_freq: str = "weekly"  # "weekly" / "biweekly" / "monthly"
    # 因子参数
    momentum_window: int = 20
    ma_short: int = 5
    ma_mid: int = 20
    ma_long: int = 60
    # 风控
    stop_loss_single: float = 0.08   # 个股止损8%
    stop_loss_portfolio: float = 0.12  # 组合止损12%
    stop_loss_circuit: float = 0.20   # 熔断20%
    circuit_cooldown: int = 10  # 熔断后冷却天数
    # 可选：指定参与回测的ETF代码列表（为空则用全池）
    selected_codes: list = None


@dataclass
class BacktestResult:
    """回测结果"""
    # 净值序列
    nav_series: pd.Series = None  # date -> nav
    benchmark_series: pd.Series = None  # date -> benchmark nav
    # 持仓历史
    positions_history: list = field(default_factory=list)  # [{date, weights}]
    # 交易记录
    trades: list = field(default_factory=list)
    # 指标
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    total_trades: int = 0
    turnover: float = 0.0


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.data_mgr = DataManager()
        self.factor_engine = FactorEngine(
            momentum_window=self.config.momentum_window,
            ma_short=self.config.ma_short,
            ma_mid=self.config.ma_mid,
            ma_long=self.config.ma_long,
        )
        self.regime_engine = RegimeEngine(
            ma_mid=self.config.ma_mid,
            ma_long=self.config.ma_long,
        )
        self.selector = Selector(
            top_n=self.config.top_n,
            weight_method=WeightMethod(self.config.weight_method),
        )
        self.executor = ExecutionEngine()

    def run(self) -> BacktestResult:
        """执行回测"""
        cfg = self.config

        # 加载数据（支持自定义 ETF 列表）
        codes = cfg.selected_codes if cfg.selected_codes else None
        close_matrix = self.data_mgr.get_close_matrix(cfg.start_date, cfg.end_date, codes=codes)
        amount_matrix = self.data_mgr.get_amount_matrix(cfg.start_date, cfg.end_date, codes=codes)
        benchmark_df = self.data_mgr.get_daily(BENCHMARK_CODE, cfg.start_date, cfg.end_date)

        if close_matrix.empty:
            print(f"[backtest] close_matrix 为空，无法回测 (start={cfg.start_date} end={cfg.end_date})")
            return BacktestResult()

        n_etfs = close_matrix.shape[1]
        if n_etfs < cfg.top_n:
            print(f"[backtest] 可用ETF数({n_etfs})少于 top_n({cfg.top_n})，无法选股")
            return BacktestResult()

        print(f"[backtest] 可用ETF数={n_etfs}, 日期={close_matrix.index[0]}~{close_matrix.index[-1]}")

        # 基准数据可选：为空时跳过基准对比
        if not benchmark_df.empty:
            benchmark_close = benchmark_df.set_index("date")["close"]
        else:
            print(f"[backtest] 基准({BENCHMARK_CODE})数据为空，跳过基准对比")
            benchmark_close = pd.Series(dtype=float)

        # 所有交易日（必须先于 rebalance_dates 获取）
        all_dates = self.data_mgr.get_trading_dates(cfg.start_date, cfg.end_date)
        if not all_dates:
            # 终极兜底：从 close_matrix index 推导交易日
            all_dates = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d)
                         for d in close_matrix.index]
            if not all_dates:
                print(f"[backtest] 交易日列表为空，无法回测")
                return BacktestResult()
            print(f"[backtest] 从 close_matrix 推导交易日: {len(all_dates)}天")

        # 获取调仓日
        rebalance_dates = self.data_mgr.get_weekly_rebalance_dates(cfg.start_date, cfg.end_date)
        if not rebalance_dates:
            # fallback: 从交易日列表自行计算周末调仓日
            _df = pd.DataFrame({"date": pd.to_datetime(all_dates)})
            iso = _df["date"].dt.isocalendar()
            _df["iso_year"] = iso["year"]
            _df["iso_week"] = iso["week"]
            weekly = _df.groupby(["iso_year", "iso_week"])["date"].max().reset_index(drop=True)
            rebalance_dates = [d.strftime("%Y-%m-%d") for d in sorted(weekly)]
        if cfg.rebalance_freq == "biweekly":
            rebalance_dates = rebalance_dates[::2]
        elif cfg.rebalance_freq == "monthly":
            rebalance_dates = rebalance_dates[::4]

        # 预构建执行日数据缓存（避免N+1查询）
        # 从 close_matrix 中提取 open 价格矩阵
        open_matrix = self.data_mgr.get_open_matrix(cfg.start_date, cfg.end_date, codes=codes)
        limit_cache = {}  # date_str -> {code: {is_limit_up, is_limit_down}}

        # 预计算5日均成交额（滚动窗口）
        amount_5d_rolling = amount_matrix.rolling(5).mean()

        # 初始化
        nav = cfg.initial_capital
        nav_list = []
        benchmark_nav_list = []
        benchmark_start = benchmark_close.iloc[0] if len(benchmark_close) > 0 else 1.0
        current_weights: dict[str, float] = {"CASH": 1.0}
        cost_basis: dict[str, float] = {}  # 各持仓成本价
        positions_history = []
        trades = []
        peak_nav = nav
        circuit_cooldown_until = None  # 熔断冷却截止日期

        # 将 rebalance_dates 转为 set 加速查找
        rebalance_set = set(rebalance_dates)

        # 逐日模拟
        for i, date in enumerate(all_dates):
            date_str = date if isinstance(date, str) else str(date)

            # ── 每日盈亏 ──
            if i > 0:
                daily_returns = {}
                prev_date_str = all_dates[i - 1] if isinstance(all_dates[i - 1], str) else str(all_dates[i - 1])
                for code in close_matrix.columns:
                    col = close_matrix[code]
                    try:
                        prev = col.loc[prev_date_str]
                        curr = col.loc[date_str]
                        if pd.notna(prev) and pd.notna(curr) and prev > 0:
                            daily_returns[code] = curr / prev - 1
                    except KeyError:
                        continue
                pnl = self.executor.compute_daily_pnl(current_weights, daily_returns)
                nav *= (1 + pnl)

            # 记录净值
            nav_list.append({"date": date_str, "nav": nav / cfg.initial_capital})

            # 基准净值
            if len(benchmark_close) > 0:
                try:
                    bm_val = benchmark_close.loc[date_str]
                    if pd.notna(bm_val):
                        benchmark_nav_list.append({
                            "date": date_str,
                            "nav": bm_val / benchmark_start
                        })
                except KeyError:
                    pass

            # ── 风控检查 ──
            peak_nav = max(peak_nav, nav)
            drawdown = (peak_nav - nav) / peak_nav

            if drawdown >= cfg.stop_loss_circuit:
                # 熔断：全部清仓，记录交易
                for code, w in list(current_weights.items()):
                    if code != "CASH" and w > 0.001:
                        sell_amount = w * nav
                        fee = sell_amount * self.executor.fee_rate
                        nav -= fee
                        trades.append({
                            "date": date_str, "code": code,
                            "direction": "sell", "price": 0, "shares": 0,
                            "amount": sell_amount, "fee": fee,
                            "skipped": False, "skip_reason": "熔断清仓",
                        })
                current_weights = {"CASH": 1.0}
                cost_basis = {}
                circuit_cooldown_until = i + cfg.circuit_cooldown
                continue

            if drawdown >= cfg.stop_loss_portfolio:
                # 组合止损：减半，记录交易
                for code in list(current_weights.keys()):
                    if code != "CASH" and current_weights[code] > 0.001:
                        sell_w = current_weights[code] * 0.5
                        sell_amount = sell_w * nav
                        fee = sell_amount * self.executor.fee_rate
                        nav -= fee
                        current_weights[code] *= 0.5
                        trades.append({
                            "date": date_str, "code": code,
                            "direction": "sell", "price": 0, "shares": 0,
                            "amount": sell_amount, "fee": fee,
                            "skipped": False, "skip_reason": "组合止损减仓",
                        })
                cash = 1.0 - sum(w for c, w in current_weights.items() if c != "CASH")
                current_weights["CASH"] = max(0, cash)

            # ── 调仓日 T+1 执行逻辑 ──
            # 如果前一日是调仓计算日，今天执行
            if i > 0 and all_dates[i - 1] in rebalance_set:
                # 熔断冷却期内不调仓
                if circuit_cooldown_until is not None and i < circuit_cooldown_until:
                    continue

                signal_date = all_dates[i - 1] if isinstance(all_dates[i - 1], str) else str(all_dates[i - 1])
                exec_date = date_str

                # 在信号日计算因子
                factors = self.factor_engine.compute_factors_at_date(
                    close_matrix, amount_matrix, signal_date
                )

                # 判断环境
                regime = self.regime_engine.detect_at_date(benchmark_close, signal_date)
                cash_ratio = self.regime_engine.get_cash_ratio(regime)

                # 选股
                target_weights = self.selector.select(
                    factors, cash_ratio, close_matrix.loc[:signal_date]
                )

                # 获取执行日数据（从预加载的矩阵中获取，避免N+1查询）
                exec_data = {}
                for code in close_matrix.columns:
                    try:
                        open_price = open_matrix[code].loc[exec_date] if code in open_matrix.columns else 0
                        if pd.notna(open_price) and open_price > 0:
                            # 获取涨跌停状态
                            if exec_date not in limit_cache:
                                limit_cache[exec_date] = self.data_mgr.get_limit_status(exec_date)
                            limits = limit_cache[exec_date].get(code, {})
                            exec_data[code] = {
                                "open": float(open_price),
                                "is_limit_up": limits.get("is_limit_up", False),
                                "is_limit_down": limits.get("is_limit_down", False),
                            }
                    except KeyError:
                        continue

                # 成交额约束（从预计算的滚动均值获取）
                amount_5d = {}
                for code in close_matrix.columns:
                    try:
                        # 用信号日的5日均额（exec_date前一天）
                        val = amount_5d_rolling[code].loc[signal_date]
                        if pd.notna(val) and val > 0:
                            amount_5d[code] = float(val)
                    except (KeyError, TypeError):
                        continue

                # 执行
                exec_result = self.executor.execute(
                    target_weights=target_weights,
                    current_weights=current_weights,
                    execution_date_data=exec_data,
                    total_value=nav,
                    amount_5d_avg=amount_5d,
                )

                current_weights = exec_result.actual_weights
                # 手续费从现金中扣除，调整 CASH 权重
                if exec_result.total_fee > 0 and nav > 0:
                    fee_ratio = exec_result.total_fee / nav
                    current_weights["CASH"] = max(0, current_weights.get("CASH", 0) - fee_ratio)
                    nav -= exec_result.total_fee

                # 记录
                positions_history.append({"date": exec_date, "weights": dict(current_weights)})
                trades.extend([{
                    "date": exec_date,
                    "code": o.code,
                    "direction": o.direction,
                    "price": o.price,
                    "shares": o.shares,
                    "amount": o.amount,
                    "fee": o.fee,
                    "skipped": o.skipped,
                    "skip_reason": o.skip_reason,
                } for o in exec_result.orders])

        # ── 计算指标 ──
        result = BacktestResult()
        result.positions_history = positions_history
        result.trades = trades
        result.total_trades = len([t for t in trades if not t.get("skipped")])

        if nav_list:
            nav_df = pd.DataFrame(nav_list)
            nav_df["date"] = pd.to_datetime(nav_df["date"])
            nav_df = nav_df.set_index("date")
            result.nav_series = nav_df["nav"]

            returns = nav_df["nav"].pct_change().dropna()
            n_years = len(returns) / 252

            result.total_return = nav_df["nav"].iloc[-1] - 1
            result.annual_return = (1 + result.total_return) ** (1 / n_years) - 1 if n_years > 0.01 else 0
            result.annual_volatility = returns.std() * np.sqrt(252)
            result.sharpe_ratio = result.annual_return / result.annual_volatility if result.annual_volatility > 0 else 0

            # Sortino
            downside = returns[returns < 0]
            downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1
            result.sortino_ratio = result.annual_return / downside_std if downside_std > 0 else 0

            # 最大回撤
            cummax = nav_df["nav"].cummax()
            drawdowns = (cummax - nav_df["nav"]) / cummax
            result.max_drawdown = drawdowns.max()

            # 胜率（周频）
            weekly_returns = returns.resample("W").apply(lambda x: (1 + x).prod() - 1)
            weekly_returns = weekly_returns.dropna()
            if len(weekly_returns) > 0:
                wins = (weekly_returns > 0).sum()
                result.win_rate = wins / len(weekly_returns)

        if benchmark_nav_list:
            bm_df = pd.DataFrame(benchmark_nav_list)
            bm_df["date"] = pd.to_datetime(bm_df["date"])
            bm_df = bm_df.set_index("date")
            result.benchmark_series = bm_df["nav"]

        return result


def main():
    """快速回测入口"""
    config = BacktestConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        top_n=3,
        weight_method="equal",
    )
    engine = BacktestEngine(config)
    result = engine.run()

    print("=" * 50)
    print("回测结果")
    print("=" * 50)
    print(f"总收益:       {result.total_return:.2%}")
    print(f"年化收益:     {result.annual_return:.2%}")
    print(f"年化波动:     {result.annual_volatility:.2%}")
    print(f"Sharpe:       {result.sharpe_ratio:.2f}")
    print(f"Sortino:      {result.sortino_ratio:.2f}")
    print(f"最大回撤:     {result.max_drawdown:.2%}")
    print(f"胜率:         {result.win_rate:.2%}")
    print(f"总交易次数:   {result.total_trades}")


if __name__ == "__main__":
    main()

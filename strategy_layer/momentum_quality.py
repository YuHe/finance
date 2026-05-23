"""
动量质量策略 (Momentum Quality)
================================
Phase 7 最优策略，适用于25只核心ETF池。
- 择时：自适应信用脉冲（同adaptive_premium）
- 选股：Sharpe_20d + RS_10d + Premium + MomQ + Sharpe_5d
- 年化32.1%, Sharpe 1.83, MaxDD -9.5%, Calmar 3.38
"""

import struct
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

from .base import BaseStrategy, StrategyResult

DATA_DIR = Path(__file__).parent.parent / "data_layer"
SIG_DB = DATA_DIR / "signals.db"


class MomentumQualityStrategy(BaseStrategy):
    name = "momentum_quality"
    display_name = "动量质量"
    description = "5信号选股(Sharpe/RS/折溢价/动量质量/短期Sharpe) + 信用脉冲择时"

    def __init__(
        self,
        top_n: int = 2,
        hold_days: int = 7,
        fee: float = 0.001,
        w_sharpe: float = 0.30,
        w_rs: float = 0.15,
        w_premium: float = 0.20,
        w_momq: float = 0.15,
        w_sharpe5: float = 0.20,
    ):
        self.top_n = top_n
        self.hold_days = hold_days
        self.fee = fee
        self.w_sharpe = w_sharpe
        self.w_rs = w_rs
        self.w_premium = w_premium
        self.w_momq = w_momq
        self.w_sharpe5 = w_sharpe5

    def _load_signals_data(self):
        if not SIG_DB.exists():
            return None, None
        conn = sqlite3.connect(SIG_DB)
        try:
            macro = pd.read_sql("SELECT month, social_financing FROM macro_monthly", conn)
            nav = pd.read_sql("SELECT date, code, nav FROM etf_nav", conn)
        except Exception:
            conn.close()
            return None, None
        conn.close()

        def decode_sf(v):
            if isinstance(v, bytes):
                return struct.unpack('<q', v)[0]
            try:
                return float(v)
            except Exception:
                return np.nan

        macro['social_financing'] = macro['social_financing'].apply(decode_sf)
        macro['month'] = pd.to_datetime(macro['month'] + '-01')
        nav['date'] = pd.to_datetime(nav['date'])
        return macro, nav

    def run(
        self,
        close_matrix: pd.DataFrame,
        high_matrix: pd.DataFrame,
        low_matrix: pd.DataFrame,
        volume_matrix: pd.DataFrame,
        benchmark: pd.Series,
        start_date: str = None,
        end_date: str = None,
    ) -> StrategyResult:
        if start_date:
            close_matrix = close_matrix.loc[start_date:]
        if end_date:
            close_matrix = close_matrix.loc[:end_date]

        macro, nav_df = self._load_signals_data()
        codes = [c for c in close_matrix.columns if c != '510300']
        dates = close_matrix.index
        bench = benchmark.reindex(dates).ffill() if len(benchmark) > 0 else close_matrix[codes].mean(axis=1)
        ret = close_matrix[codes].pct_change()

        # === Timing: Adaptive Credit ===
        ma20 = close_matrix[codes].rolling(20).mean()
        breadth = (close_matrix[codes] > ma20).mean(axis=1)
        breadth_delta = breadth - breadth.shift(5)
        sig_3b = breadth > 0.5
        sig_bt = (breadth_delta > 0.15) & (breadth > 0.4)

        credit_expanding = pd.Series(True, index=dates)
        if macro is not None and len(macro) > 0:
            sf = macro.set_index('month')['social_financing'].sort_index()
            sf_12m = sf.pct_change(12)
            ci = sf_12m.diff(1)
            ci_daily = ci.reindex(dates, method='ffill').shift(20)
            credit_expanding = ci_daily > 0
            credit_expanding = credit_expanding.fillna(True)

        timing_on = pd.Series(False, index=dates)
        for i in range(len(dates)):
            if credit_expanding.iloc[i]:
                timing_on.iloc[i] = sig_3b.iloc[i]
            else:
                timing_on.iloc[i] = sig_bt.iloc[i]

        # === Selection Signals ===
        sharpe_20d = ret.rolling(20).mean() / (ret.rolling(20).std() + 1e-8)
        rs_10d = close_matrix[codes].pct_change(10).sub(bench.pct_change(10), axis=0)

        # MomQ: momentum quality = -clip(ret_5d / (avg_abs_daily_ret_20d * 5), -5, 5)
        ret_5d = close_matrix[codes].pct_change(5)
        avg_abs_ret_20d = ret.abs().rolling(20).mean()
        raw_momq = ret_5d / (avg_abs_ret_20d * 5 + 1e-8)
        momq = -raw_momq.clip(-5, 5)

        # Sharpe_5d
        sharpe_5d = ret.rolling(5).mean() / (ret.rolling(5).std() + 1e-8)

        # Premium
        premium = pd.DataFrame(0.0, index=dates, columns=codes)
        if nav_df is not None and len(nav_df) > 0:
            nav_pivot = nav_df.pivot(index='date', columns='code', values='nav').sort_index()
            for c in codes:
                if c in nav_pivot.columns:
                    nav_aligned = nav_pivot[c].reindex(dates)
                    prem = (close_matrix[c] - nav_aligned) / (nav_aligned + 1e-8)
                    premium[c] = prem.rolling(5).mean()

        def row_zscore(df):
            return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-8, axis=0)

        z_sharpe = row_zscore(sharpe_20d)
        z_rs = row_zscore(rs_10d)
        z_premium = -row_zscore(premium)
        z_momq = row_zscore(momq)
        z_sharpe5 = row_zscore(sharpe_5d)

        composite = (self.w_sharpe * z_sharpe + self.w_rs * z_rs +
                     self.w_premium * z_premium + self.w_momq * z_momq +
                     self.w_sharpe5 * z_sharpe5)

        # === Backtest ===
        warmup = 80
        nav_val = 1.0
        portfolio = {}
        hold_timer = 0
        equity_series = []
        trades = []

        for i in range(warmup, len(dates)):
            today = dates[i]
            if portfolio:
                daily_ret = np.mean([(close_matrix.loc[today, c] / portfolio[c] - 1)
                                     for c in portfolio if not np.isnan(close_matrix.loc[today, c])])
                nav_val *= (1 + daily_ret)
                portfolio = {c: close_matrix.loc[today, c] for c in portfolio
                             if not np.isnan(close_matrix.loc[today, c])}

            equity_series.append((today, nav_val))
            hold_timer += 1
            risk_on = bool(timing_on.iloc[i])

            if portfolio and not risk_on:
                nav_val *= (1 - self.fee)
                trades.append({"date": str(today), "action": "timing_exit", "codes": list(portfolio.keys())})
                portfolio = {}
                hold_timer = 0
                continue

            need_rebal = (not portfolio and risk_on) or (hold_timer >= self.hold_days)
            if not need_rebal or not risk_on:
                continue

            sig = composite.iloc[i]
            valid = sig.dropna()
            valid = valid[[c for c in valid.index if not np.isnan(close_matrix.loc[today, c])]]
            if len(valid) < self.top_n:
                continue
            top = valid.nlargest(self.top_n).index.tolist()

            if set(top) == set(portfolio.keys()):
                continue

            if portfolio:
                nav_val *= (1 - self.fee)
            portfolio = {c: close_matrix.loc[today, c] for c in top}
            nav_val *= (1 - self.fee)
            hold_timer = 0
            trades.append({"date": str(today), "action": "rebalance", "codes": top})

        if not equity_series:
            return StrategyResult()

        eq = pd.Series(
            [x[1] for x in equity_series],
            index=pd.DatetimeIndex([x[0] for x in equity_series])
        )
        return self.compute_metrics(eq, trades)

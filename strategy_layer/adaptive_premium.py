"""
自适应信用择时 + 折溢价策略 (Adaptive Credit + Premium)
=====================================================
Phase 5/8 最优策略，适用于扩展ETF池(71只)。
- 择时：信用扩张期用breadth>50%，收缩期用breadth thrust
- 选股：Sharpe_20d + RS_10d + ETF折溢价(反向)
- 年化32.8%, Sharpe 1.67, MaxDD -10.0%, Calmar 3.27
"""

import struct
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

from .base import BaseStrategy, StrategyResult

DATA_DIR = Path(__file__).parent.parent / "data_layer"
SIG_DB = DATA_DIR / "signals.db"


class AdaptivePremiumStrategy(BaseStrategy):
    name = "adaptive_premium"
    display_name = "自适应折溢价"
    description = "信用脉冲择时 + Sharpe/RS/折溢价三信号选股，适合大ETF池"

    def __init__(
        self,
        top_n: int = 2,
        hold_days: int = 7,
        fee: float = 0.001,
        w_sharpe: float = 0.4,
        w_rs: float = 0.3,
        w_premium: float = 0.3,
    ):
        self.top_n = top_n
        self.hold_days = hold_days
        self.fee = fee
        self.w_sharpe = w_sharpe
        self.w_rs = w_rs
        self.w_premium = w_premium

    def _load_signals_data(self):
        """Load macro and NAV data from signals.db"""
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

        composite = self.w_sharpe * z_sharpe + self.w_rs * z_rs + self.w_premium * z_premium

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

"""
Backtest Strategy L: Breadth Regime-Switch Ensemble ETF Rotation

Market breadth determines regime (Bull/Neutral/Bear), each with distinct
momentum signals, position sizing, rebalance frequency, and stop-loss rules.
"""

import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "/Users/heyu11/Code/finance/data_layer/backtest_fixed.db"
BENCHMARK_CODE = "510300"
FEE_RATE = 0.0005  # 0.05% per side

# Breadth
BREADTH_MA = 20
BREADTH_BULL_THRESHOLD = 0.55
BREADTH_NEUTRAL_LOW = 0.30
BREADTH_BEAR_RECOVERY = 0.35
REGIME_CONFIRM_DAYS = 2

# Bull regime
BULL_MOM_PERIOD = 5
BULL_TOP_N = 3
BULL_WEIGHT = 1.0 / 3  # equal weight, 100% total
BULL_REBAL_FREQ = 3
BULL_ATR_PERIOD = 10
BULL_ATR_MULT = 2.5

# Neutral regime
NEUTRAL_MOM_PERIOD = 20
NEUTRAL_MA_PERIOD = 20
NEUTRAL_RET_PERIOD = 10
NEUTRAL_TOP_N = 2
NEUTRAL_WEIGHT = 0.30  # equal weight, 60% total (0.30 * 2)
NEUTRAL_REBAL_FREQ = 5
NEUTRAL_ATR_PERIOD = 14
NEUTRAL_ATR_MULT = 2.0
NEUTRAL_HARD_STOP = -0.04

# Portfolio risk
MAX_DD_TRIGGER = 0.06
MAX_DD_CASH_DAYS = 5
EQUITY_MA_PERIOD = 15
EQUITY_FILTER_REDUCE = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Indicators
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM etf_daily ORDER BY date, code", conn)
    conn.close()
    all_codes = sorted(df["code"].unique().tolist())
    etf_codes = [c for c in all_codes if c != BENCHMARK_CODE]
    dates = sorted(df["date"].unique().tolist())
    return df, etf_codes, dates


def build_matrices(df, etf_codes, dates):
    all_cols = etf_codes + [BENCHMARK_CODE]
    close = df.pivot(index="date", columns="code", values="close").reindex(index=dates, columns=all_cols)
    open_px = df.pivot(index="date", columns="code", values="open").reindex(index=dates, columns=all_cols)
    high = df.pivot(index="date", columns="code", values="high").reindex(index=dates, columns=all_cols)
    low = df.pivot(index="date", columns="code", values="low").reindex(index=dates, columns=all_cols)
    return close, open_px, high, low


def compute_indicators(close, high, low, etf_codes):
    ind = {}

    # MA20
    ind["ma20"] = close[etf_codes].rolling(BREADTH_MA).mean()

    # 5-day momentum
    ind["mom5"] = close[etf_codes].pct_change(BULL_MOM_PERIOD)

    # 20-day risk-adjusted momentum
    ret20 = close[etf_codes].pct_change(NEUTRAL_MOM_PERIOD)
    daily_ret = close[etf_codes].pct_change()
    vol20 = daily_ret.rolling(NEUTRAL_MOM_PERIOD).std()
    ind["risk_adj_mom20"] = ret20 / vol20.replace(0, np.nan)

    # 10-day return
    ind["ret10"] = close[etf_codes].pct_change(NEUTRAL_RET_PERIOD)

    # True Range and ATRs
    prev_close = close[etf_codes].shift(1)
    tr1 = high[etf_codes] - low[etf_codes]
    tr2 = (high[etf_codes] - prev_close).abs()
    tr3 = (low[etf_codes] - prev_close).abs()
    true_range = pd.DataFrame(
        np.maximum(np.maximum(tr1.values, tr2.values), tr3.values),
        index=tr1.index, columns=tr1.columns
    )
    ind["atr10"] = true_range.rolling(BULL_ATR_PERIOD).mean()
    ind["atr14"] = true_range.rolling(NEUTRAL_ATR_PERIOD).mean()

    # Market breadth
    above_ma = (close[etf_codes] > ind["ma20"]).astype(float)
    breadth = above_ma.mean(axis=1)
    ind["breadth"] = breadth
    ind["confirmed_breadth"] = (breadth + breadth.shift(1)) / 2

    return ind


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Position:
    code: str
    entry_date: str
    entry_price: float
    shares: float
    trailing_high: float
    regime_at_entry: str


@dataclass
class Trade:
    code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str
    regime: str


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────────────────────
class BreadthRegimeBacktest:

    def __init__(self):
        self.cash = 1_000_000.0
        self.initial_equity = self.cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.date_list: List[str] = []
        self.regime_history: List[str] = []

        # Regime state
        self.current_regime = "Bear"
        self.regime_pending = None
        self.regime_pending_days = 0
        self.bear_recovery_days = 0

        # Counters
        self.days_since_rebal = 0
        self.cash_override_remaining = 0
        self.peak_equity = self.cash

    def _total_equity(self, close_row) -> float:
        """Cash + mark-to-market of positions using a close row (Series)."""
        val = self.cash
        for code, pos in self.positions.items():
            px = close_row.get(code, np.nan)
            if pd.isna(px):
                px = pos.entry_price
            val += pos.shares * px
        return val

    def run(self):
        df, etf_codes, dates = load_data()
        close, open_px, high, low = build_matrices(df, etf_codes, dates)
        indicators = compute_indicators(close, high, low, etf_codes)

        warmup = max(BREADTH_MA, NEUTRAL_MOM_PERIOD, NEUTRAL_ATR_PERIOD) + 5  # 25

        for i in range(len(dates)):
            date = dates[i]
            self.date_list.append(date)

            if i < warmup:
                self.equity_curve.append(self._total_equity(close.loc[date]))
                self.regime_history.append("Warmup")
                continue

            prev_date = dates[i - 1]

            # ─── Update trailing highs (using today's high for precision) ───
            for code, pos in self.positions.items():
                h = high.loc[date, code]
                if not pd.isna(h) and h > pos.trailing_high:
                    pos.trailing_high = h

            # ─── Determine regime (signal from prev_date close) ───
            cb = indicators["confirmed_breadth"].loc[prev_date]
            if pd.isna(cb):
                self.equity_curve.append(self._total_equity(close.loc[date]))
                self.regime_history.append(self.current_regime)
                continue

            raw_regime = self._classify_regime(cb)
            self._update_regime(raw_regime, cb)
            self.regime_history.append(self.current_regime)

            # ─── Compute equity ───
            equity = self._total_equity(close.loc[date])

            # ─── DD Override: if active, stay in cash ───
            if self.cash_override_remaining > 0:
                # On first day of override, liquidate (already done when triggered)
                self.cash_override_remaining -= 1
                if self.cash_override_remaining == 0:
                    # Reset peak to current equity so we don't immediately retrigger
                    self.peak_equity = self._total_equity(close.loc[date])
                self.equity_curve.append(self._total_equity(close.loc[date]))
                continue

            # ─── Check DD trigger ───
            if equity < self.peak_equity * (1 - MAX_DD_TRIGGER):
                # Trigger: go to cash for N days
                self.cash_override_remaining = MAX_DD_CASH_DAYS
                if self.positions:
                    self._liquidate_all(open_px.loc[date], date, "dd_override")
                self.equity_curve.append(self._total_equity(close.loc[date]))
                continue

            # Update peak
            if equity > self.peak_equity:
                self.peak_equity = equity

            # ─── Equity curve filter ───
            equity_reduce = False
            if len(self.equity_curve) >= EQUITY_MA_PERIOD:
                eq_ma = np.mean(self.equity_curve[-EQUITY_MA_PERIOD:])
                if equity < eq_ma:
                    equity_reduce = True

            # ─── Stop-loss checks (intraday using today's close as proxy) ───
            self._check_stops(close.loc[date], prev_date, date, indicators)

            # ─── Bear regime: immediate liquidation ───
            if self.current_regime == "Bear":
                if self.positions:
                    self._liquidate_all(open_px.loc[date], date, "bear_regime")
                self.days_since_rebal = 0
                equity = self._total_equity(close.loc[date])
                if equity > self.peak_equity:
                    self.peak_equity = equity
                self.equity_curve.append(equity)
                continue

            # ─── Rebalance ───
            self.days_since_rebal += 1
            rebal_freq = BULL_REBAL_FREQ if self.current_regime == "Bull" else NEUTRAL_REBAL_FREQ

            if self.days_since_rebal >= rebal_freq:
                self.days_since_rebal = 0
                self._rebalance(indicators, close, open_px, prev_date, date,
                                etf_codes, equity_reduce)

            equity = self._total_equity(close.loc[date])
            if equity > self.peak_equity:
                self.peak_equity = equity
            self.equity_curve.append(equity)

        # ─── Final liquidation ───
        last_date = dates[-1]
        if self.positions:
            for code in list(self.positions.keys()):
                pos = self.positions[code]
                px = close.loc[last_date, code]
                if pd.isna(px):
                    px = pos.entry_price
                proceeds = pos.shares * px * (1 - FEE_RATE)
                self.cash += proceeds
                pnl_pct = (px / pos.entry_price) - 1 - 2 * FEE_RATE
                self.trades.append(Trade(
                    code=code, entry_date=pos.entry_date, exit_date=last_date,
                    entry_price=pos.entry_price, exit_price=px,
                    pnl_pct=pnl_pct, exit_reason="end_of_backtest",
                    regime=pos.regime_at_entry
                ))
            self.positions.clear()

        self._print_results(dates, close)

    # ─── Regime Logic ─────────────────────────────────────────────────────────

    def _classify_regime(self, breadth: float) -> str:
        if breadth > BREADTH_BULL_THRESHOLD:
            return "Bull"
        elif breadth < BREADTH_NEUTRAL_LOW:
            return "Bear"
        return "Neutral"

    def _update_regime(self, raw_regime: str, breadth: float):
        # Bear recovery: need breadth >= 0.35 for 2 consecutive days
        if self.current_regime == "Bear":
            if breadth >= BREADTH_BEAR_RECOVERY:
                self.bear_recovery_days += 1
            else:
                self.bear_recovery_days = 0
            if self.bear_recovery_days >= 2:
                self.current_regime = "Neutral"
                self.bear_recovery_days = 0
                self.regime_pending = None
                self.regime_pending_days = 0
            return

        # Anti-whipsaw: require 2 consecutive days in new regime
        if raw_regime != self.current_regime:
            if self.regime_pending == raw_regime:
                self.regime_pending_days += 1
                if self.regime_pending_days >= REGIME_CONFIRM_DAYS:
                    self.current_regime = raw_regime
                    self.regime_pending = None
                    self.regime_pending_days = 0
            else:
                self.regime_pending = raw_regime
                self.regime_pending_days = 1
        else:
            self.regime_pending = None
            self.regime_pending_days = 0

    # ─── Stop-Loss ────────────────────────────────────────────────────────────

    def _check_stops(self, close_row, prev_date, date, indicators):
        to_remove = []
        for code, pos in self.positions.items():
            px = close_row.get(code, np.nan)
            if pd.isna(px):
                continue

            exit_reason = None

            # ATR trailing stop
            if pos.regime_at_entry == "Bull":
                atr_val = indicators["atr10"].loc[prev_date, code] if prev_date in indicators["atr10"].index else np.nan
                if not pd.isna(atr_val) and atr_val > 0:
                    stop = pos.trailing_high - BULL_ATR_MULT * atr_val
                    if px <= stop:
                        exit_reason = "trailing_stop"
            else:
                atr_val = indicators["atr14"].loc[prev_date, code] if prev_date in indicators["atr14"].index else np.nan
                if not pd.isna(atr_val) and atr_val > 0:
                    stop = pos.trailing_high - NEUTRAL_ATR_MULT * atr_val
                    if px <= stop:
                        exit_reason = "trailing_stop"
                # Hard stop
                pnl = (px / pos.entry_price) - 1
                if pnl <= NEUTRAL_HARD_STOP:
                    exit_reason = "hard_stop"

            if exit_reason:
                proceeds = pos.shares * px * (1 - FEE_RATE)
                self.cash += proceeds
                pnl_pct = (px / pos.entry_price) - 1 - 2 * FEE_RATE
                self.trades.append(Trade(
                    code=code, entry_date=pos.entry_date, exit_date=date,
                    entry_price=pos.entry_price, exit_price=px,
                    pnl_pct=pnl_pct, exit_reason=exit_reason,
                    regime=pos.regime_at_entry
                ))
                to_remove.append(code)

        for code in to_remove:
            del self.positions[code]

    # ─── Liquidation ──────────────────────────────────────────────────────────

    def _liquidate_all(self, open_row, date, reason):
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            px = open_row.get(code, np.nan)
            if pd.isna(px):
                px = pos.entry_price  # fallback
            proceeds = pos.shares * px * (1 - FEE_RATE)
            self.cash += proceeds
            pnl_pct = (px / pos.entry_price) - 1 - 2 * FEE_RATE
            self.trades.append(Trade(
                code=code, entry_date=pos.entry_date, exit_date=date,
                entry_price=pos.entry_price, exit_price=px,
                pnl_pct=pnl_pct, exit_reason=reason,
                regime=pos.regime_at_entry
            ))
        self.positions.clear()

    # ─── Rebalance ────────────────────────────────────────────────────────────

    def _rebalance(self, indicators, close, open_px, signal_date, exec_date,
                   etf_codes, equity_reduce):
        if self.current_regime == "Bull":
            self._rebalance_bull(indicators, close, open_px, signal_date,
                                exec_date, etf_codes, equity_reduce)
        elif self.current_regime == "Neutral":
            self._rebalance_neutral(indicators, close, open_px, signal_date,
                                    exec_date, etf_codes, equity_reduce)

    def _rebalance_bull(self, indicators, close, open_px, signal_date,
                        exec_date, etf_codes, equity_reduce):
        mom = indicators["mom5"].loc[signal_date, etf_codes]
        valid = mom.dropna().sort_values(ascending=False)
        target_codes = valid.head(BULL_TOP_N).index.tolist()

        target_weight = BULL_WEIGHT
        if equity_reduce:
            target_weight *= EQUITY_FILTER_REDUCE

        # Use total equity at signal date for sizing
        equity_est = self._total_equity(close.loc[signal_date])
        self._execute_rebalance(target_codes, target_weight, equity_est,
                                open_px.loc[exec_date], exec_date, "Bull")

    def _rebalance_neutral(self, indicators, close, open_px, signal_date,
                           exec_date, etf_codes, equity_reduce):
        close_row = close.loc[signal_date, etf_codes]
        ma20_row = indicators["ma20"].loc[signal_date]
        ret10_row = indicators["ret10"].loc[signal_date]

        mask = (close_row > ma20_row) & (ret10_row > 0)
        eligible = mask[mask].index.tolist()

        if not eligible:
            if self.positions:
                self._liquidate_all(open_px.loc[exec_date], exec_date, "no_eligible")
            return

        scores = indicators["risk_adj_mom20"].loc[signal_date, eligible].dropna()
        scores = scores.sort_values(ascending=False)
        target_codes = scores.head(NEUTRAL_TOP_N).index.tolist()

        target_weight = NEUTRAL_WEIGHT
        if equity_reduce:
            target_weight *= EQUITY_FILTER_REDUCE

        equity_est = self._total_equity(close.loc[signal_date])
        self._execute_rebalance(target_codes, target_weight, equity_est,
                                open_px.loc[exec_date], exec_date, "Neutral")

    def _execute_rebalance(self, target_codes, target_weight, equity_est,
                           open_row, exec_date, regime):
        # Sell positions not in target
        to_sell = [c for c in self.positions if c not in target_codes]
        for code in to_sell:
            pos = self.positions[code]
            px = open_row.get(code, np.nan)
            if pd.isna(px):
                px = pos.entry_price
            proceeds = pos.shares * px * (1 - FEE_RATE)
            self.cash += proceeds
            pnl_pct = (px / pos.entry_price) - 1 - 2 * FEE_RATE
            self.trades.append(Trade(
                code=code, entry_date=pos.entry_date, exit_date=exec_date,
                entry_price=pos.entry_price, exit_price=px,
                pnl_pct=pnl_pct, exit_reason="rebalance",
                regime=pos.regime_at_entry
            ))
            del self.positions[code]

        # Buy new targets not already held
        for code in target_codes:
            if code not in self.positions:
                entry_price = open_row.get(code, np.nan)
                if pd.isna(entry_price) or entry_price <= 0:
                    continue
                alloc = equity_est * target_weight
                # Don't exceed available cash
                max_alloc = self.cash / (1 + FEE_RATE)
                if alloc > max_alloc:
                    alloc = max_alloc
                if alloc <= 0:
                    continue
                shares = alloc / entry_price
                cost = shares * entry_price * (1 + FEE_RATE)
                self.cash -= cost

                self.positions[code] = Position(
                    code=code,
                    entry_date=exec_date,
                    entry_price=entry_price,
                    shares=shares,
                    trailing_high=entry_price,
                    regime_at_entry=regime
                )

    # ─── Results Printing ─────────────────────────────────────────────────────

    def _print_results(self, dates, close):
        eq = np.array(self.equity_curve)
        bench_prices = close[BENCHMARK_CODE].reindex(dates).ffill().bfill().values

        total_days = len(eq)
        years = total_days / 252

        final_ret = (eq[-1] / eq[0]) - 1
        annual_ret = (1 + final_ret) ** (1 / years) - 1 if years > 0 else 0

        # Max drawdown
        running_max = np.maximum.accumulate(eq)
        drawdowns = (eq - running_max) / running_max
        max_dd = drawdowns.min()

        # Sharpe
        daily_returns = np.diff(eq) / eq[:-1]
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                  if np.std(daily_returns) > 0 else 0)

        # Calmar
        calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

        # Trade stats
        n_trades = len(self.trades)
        if n_trades > 0:
            wins = sum(1 for t in self.trades if t.pnl_pct > 0)
            losses = n_trades - wins
            win_rate = wins / n_trades
            avg_win = np.mean([t.pnl_pct for t in self.trades if t.pnl_pct > 0]) if wins > 0 else 0
            avg_loss = np.mean([t.pnl_pct for t in self.trades if t.pnl_pct <= 0]) if losses > 0 else 0
            total_win_pnl = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
            total_loss_pnl = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct <= 0))
            profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float('inf')
        else:
            wins = losses = 0
            win_rate = avg_win = avg_loss = profit_factor = 0

        trades_per_week = n_trades / (total_days / 5) if total_days > 0 else 0

        # Benchmark
        bench_final = (bench_prices[-1] / bench_prices[0]) - 1
        bench_annual = (1 + bench_final) ** (1 / years) - 1 if years > 0 else 0
        bench_rm = np.maximum.accumulate(bench_prices)
        bench_max_dd = ((bench_prices - bench_rm) / bench_rm).min()

        print("=" * 72)
        print("  STRATEGY L: Breadth Regime-Switch Ensemble ETF Rotation")
        print("=" * 72)
        print(f"\n  {'Period:':<28}{dates[0]} to {dates[-1]}")
        print(f"  {'Trading Days:':<28}{total_days}")
        print(f"  {'Initial Equity:':<28}{eq[0]:,.0f}")
        print(f"  {'Final Equity:':<28}{eq[-1]:,.0f}")

        print(f"\n  " + "-" * 68)
        print(f"  {'Metric':<30}{'Strategy':>15}{'Benchmark(510300)':>20}")
        print(f"  " + "-" * 68)
        print(f"  {'Total Return':<30}{final_ret:>14.2%}{bench_final:>19.2%}")
        print(f"  {'Annual Return':<30}{annual_ret:>14.2%}{bench_annual:>19.2%}")
        print(f"  {'Max Drawdown':<30}{max_dd:>14.2%}{bench_max_dd:>19.2%}")
        print(f"  {'Sharpe Ratio':<30}{sharpe:>14.3f}{'-':>20}")
        print(f"  {'Calmar Ratio':<30}{calmar:>14.3f}{'-':>20}")
        print(f"  {'Profit Factor':<30}{profit_factor:>14.3f}{'-':>20}")
        print(f"  {'Win Rate':<30}{win_rate:>14.2%}{'-':>20}")
        print(f"  {'Avg Win':<30}{avg_win:>14.2%}{'-':>20}")
        print(f"  {'Avg Loss':<30}{avg_loss:>14.2%}{'-':>20}")
        print(f"  {'Total Trades':<30}{n_trades:>14}{'-':>20}")
        print(f"  {'Trades/Week':<30}{trades_per_week:>14.2f}{'-':>20}")
        print(f"  " + "-" * 68)

        # ─── Yearly Breakdown ───
        print(f"\n{'=' * 72}")
        print("  YEARLY BREAKDOWN")
        print(f"{'=' * 72}")
        print(f"  {'Year':<7}{'Return':>9}{'MaxDD':>9}{'Sharpe':>9}{'Trades':>8}"
              f"{'Bull%':>8}{'Neut%':>8}{'Bear%':>8}")
        print(f"  " + "-" * 66)

        for year in sorted(set(d[:4] for d in dates)):
            year_idx = [j for j in range(len(dates)) if dates[j].startswith(year)]
            if len(year_idx) < 2:
                continue
            year_eq = eq[year_idx]
            yr_ret = (year_eq[-1] / year_eq[0]) - 1
            yr_max = np.maximum.accumulate(year_eq)
            yr_dd = ((year_eq - yr_max) / yr_max).min()
            yr_dr = np.diff(year_eq) / year_eq[:-1]
            yr_sharpe = (np.mean(yr_dr) / np.std(yr_dr) * np.sqrt(252)
                         if np.std(yr_dr) > 0 else 0)
            yr_trades = sum(1 for t in self.trades if t.exit_date.startswith(year))
            yr_regimes = [self.regime_history[j] for j in year_idx]
            yr_active = [r for r in yr_regimes if r != "Warmup"]
            n_a = max(len(yr_active), 1)
            bull_pct = yr_active.count("Bull") / n_a
            neut_pct = yr_active.count("Neutral") / n_a
            bear_pct = yr_active.count("Bear") / n_a
            print(f"  {year:<7}{yr_ret:>8.2%}{yr_dd:>9.2%}{yr_sharpe:>9.2f}"
                  f"{yr_trades:>8}{bull_pct:>7.0%}{neut_pct:>8.0%}{bear_pct:>8.0%}")

        # ─── Regime Breakdown ───
        print(f"\n{'=' * 72}")
        print("  REGIME BREAKDOWN")
        print(f"{'=' * 72}")

        active_regimes = [r for r in self.regime_history if r != "Warmup"]
        total_active = max(len(active_regimes), 1)

        for regime in ["Bull", "Neutral", "Bear"]:
            count = active_regimes.count(regime)
            pct = count / total_active

            # Return attribution: sum of daily equity changes during regime days
            regime_contrib = 0.0
            for j in range(1, len(eq)):
                if j < len(self.regime_history) and self.regime_history[j] == regime:
                    regime_contrib += (eq[j] - eq[j-1]) / eq[0]

            regime_trades = [t for t in self.trades if t.regime == regime]
            r_wins = sum(1 for t in regime_trades if t.pnl_pct > 0)
            r_wr = r_wins / len(regime_trades) if regime_trades else 0
            r_avg = np.mean([t.pnl_pct for t in regime_trades]) if regime_trades else 0

            print(f"\n  {regime} Regime:")
            print(f"    {'Time in Regime:':<28}{pct:>7.1%}  ({count} days)")
            print(f"    {'Return Attribution:':<28}{regime_contrib:>7.2%}")
            print(f"    {'Trades:':<28}{len(regime_trades):>7}")
            print(f"    {'Win Rate:':<28}{r_wr:>7.1%}")
            print(f"    {'Avg Trade PnL:':<28}{r_avg:>7.2%}")

        # ─── Exit Reason Breakdown ───
        print(f"\n{'=' * 72}")
        print("  EXIT REASON BREAKDOWN")
        print(f"{'=' * 72}")
        print(f"  {'Reason':<20}{'Count':>7}{'Avg PnL':>10}{'Win Rate':>10}{'Total PnL':>12}")
        print(f"  " + "-" * 57)

        reasons: Dict[str, List[float]] = {}
        for t in self.trades:
            reasons.setdefault(t.exit_reason, []).append(t.pnl_pct)

        for reason in sorted(reasons.keys(), key=lambda x: -len(reasons[x])):
            pnls = reasons[reason]
            cnt = len(pnls)
            avg = np.mean(pnls)
            wr = sum(1 for p in pnls if p > 0) / cnt
            total = sum(pnls)
            print(f"  {reason:<20}{cnt:>7}{avg:>9.2%}{wr:>10.1%}{total:>11.2%}")

        print(f"\n{'=' * 72}")
        print("  Backtest complete.")
        print(f"{'=' * 72}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bt = BreadthRegimeBacktest()
    bt.run()

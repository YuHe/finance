"""
Strategy Q: K Dual-Timeframe + Multi-Signal Confirmation
=========================================================
Combines Strategy K's dip-buying logic with multi-signal ensemble voting
for higher-conviction entries. The key insight: K gets good returns but
-13.47% MaxDD, while entries confirmed by MULTIPLE signals should have
higher win rate and lower drawdown.

Signals:
  1. K's Dual Timeframe (required): Score_K = 0.6*Rank(R20) + 0.4*Rank(-R3)
     + trend filter (close > MA10 AND close > MA20)
  2. Volume Confirmation (optional bonus): vol_5d > 1.3 * vol_20d -> +0.15
  3. Relative Strength vs Benchmark (required): ETF R10 > Benchmark R10
  4. Breadth Expansion (context): breadth_today > breadth_5d_ago

Entry requires Signal 1 (top 3) AND Signal 3 AND breadth > 0.45
  AND (Signal 4 OR breadth > 0.55)
Pick top-1 by K_score + volume_bonus.

Position sizing: breadth > 0.60 -> 100%, 0.45-0.60 -> 70%

Exit: 2.0x ATR(14) trailing, -5% hard stop, 2 consecutive close < MA20,
      portfolio DD > 7% -> clear + 5d cooldown, 12 trading day time stop.

Rebalance check every 5 trading days.
"""

from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "/Users/heyu11/Code/finance/data_layer/backtest_fixed.db"
BENCHMARK = "510300"
FEE_RATE = 0.0005  # 0.05% per side
INITIAL_CAPITAL = 1_000_000.0

# Strategy parameters
REBAL_PERIOD = 5           # rebalance check every N trading days
MAX_POSITIONS = 1          # top-1 pick
R20_WEIGHT = 0.6           # weight for 20-day return rank
R3_WEIGHT = 0.4            # weight for negative 3-day return rank (dip)
MA_SHORT = 10              # short-term MA for trend filter
MA_MEDIUM = 20             # medium-term MA for trend filter
ATR_PERIOD = 14            # ATR lookback
ATR_MULTIPLIER = 2.0       # trailing stop distance
HARD_STOP_PCT = -0.05      # -5% hard stop from entry
TREND_BREAK_DAYS = 2       # consecutive days below MA20 to trigger exit
TIME_STOP_DAYS = 12        # max holding days
PORTFOLIO_STOP_PCT = 0.07  # 7% drawdown from equity peak -> liquidate
PORTFOLIO_COOLDOWN = 5     # days to wait after portfolio stop

# Volume confirmation
VOLUME_SURGE_THRESHOLD = 1.3  # vol_5d / vol_20d ratio for volume surge
VOLUME_BONUS = 0.15           # bonus to score when volume confirms

# Breadth thresholds
BREADTH_ENTRY_MIN = 0.45      # minimum breadth for entry
BREADTH_FULL = 0.60           # above this: 100% position
BREADTH_STRONG = 0.55         # above this: bypass breadth expansion requirement

# Signal 3: Relative strength lookback
RS_PERIOD = 10  # 10-day return comparison vs benchmark

# K-score top-N filter before multi-signal confirmation
K_SCORE_TOP_N = 5  # top 5 by K score (top ~20% of 25 ETF universe)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> dict[str, pd.DataFrame]:
    """Load all ETF data from SQLite, return dict of code -> DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM etf_daily ORDER BY code, date", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    data = {}
    for code, group in df.groupby("code"):
        g = group.set_index("date").sort_index()
        data[code] = g[["open", "high", "low", "close", "volume"]].copy()
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Indicator Computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_indicators(data: dict[str, pd.DataFrame]):
    """Compute all needed indicators for each ETF."""
    for code, df in data.items():
        # Returns
        df["r20"] = df["close"].pct_change(20)
        df["r3"] = df["close"].pct_change(3)
        df["r10"] = df["close"].pct_change(RS_PERIOD)
        # Moving averages
        df["ma10"] = df["close"].rolling(MA_SHORT).mean()
        df["ma20"] = df["close"].rolling(MA_MEDIUM).mean()
        # ATR
        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hc": (df["high"] - df["close"].shift(1)).abs(),
            "lc": (df["low"] - df["close"].shift(1)).abs(),
        })
        df["tr"] = tr.max(axis=1)
        df["atr14"] = df["tr"].rolling(ATR_PERIOD).mean()
        # Volume averages
        df["vol_5d"] = df["volume"].rolling(5).mean()
        df["vol_20d"] = df["volume"].rolling(20).mean()
        # Above MA20 flag (for breadth)
        df["above_ma20"] = (df["close"] > df["ma20"]).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Trade Record
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: str = ""
    highest_close: float = 0.0       # for trailing stop
    days_below_ma20: int = 0         # for trend break
    days_held: int = 0               # for time stop
    had_volume_bonus: bool = False   # signal tracking
    had_breadth_expansion: bool = False  # signal tracking


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────────────────────────────────────
class Backtest:
    def __init__(self, data: dict[str, pd.DataFrame]):
        self.data = data
        self.etf_codes = [c for c in data.keys() if c != BENCHMARK]
        self.all_dates = sorted(data[BENCHMARK].index)
        self.equity = INITIAL_CAPITAL
        self.cash = INITIAL_CAPITAL
        self.equity_curve = []
        self.positions: list[Trade] = []  # open positions
        self.closed_trades: list[Trade] = []
        self.peak_equity = INITIAL_CAPITAL
        self.portfolio_stop_active = False
        self.portfolio_cooldown_remaining = 0
        self.days_since_rebal = REBAL_PERIOD  # trigger on first eligible day

    def compute_breadth(self, date: pd.Timestamp) -> float:
        """Fraction of ETFs with close > 20d MA."""
        above = 0
        total = 0
        for code in self.etf_codes:
            df = self.data[code]
            if date in df.index:
                val = df.loc[date, "above_ma20"]
                if not np.isnan(val):
                    above += val
                    total += 1
        return above / total if total > 0 else 0.0

    def get_position_size_mult(self, breadth: float) -> float:
        """Position size multiplier based on breadth."""
        if breadth > BREADTH_FULL:
            return 1.0
        elif breadth >= BREADTH_ENTRY_MIN:
            return 0.70
        else:
            return 0.0

    def compute_k_scores(self, date: pd.Timestamp) -> dict[str, float]:
        """Compute K dual-timeframe scores: 0.6*Rank(R20) + 0.4*Rank(-R3)."""
        r20_vals = {}
        r3_vals = {}

        for code in self.etf_codes:
            df = self.data[code]
            if date not in df.index:
                continue
            r20 = df.loc[date, "r20"]
            r3 = df.loc[date, "r3"]
            if np.isnan(r20) or np.isnan(r3):
                continue
            r20_vals[code] = r20
            r3_vals[code] = r3

        if len(r20_vals) < 2:
            return {}

        # Cross-sectional ranking
        codes = list(r20_vals.keys())
        n = len(codes)

        # Rank R20 (higher is better)
        r20_sorted = sorted(codes, key=lambda c: r20_vals[c])
        r20_rank = {c: i / (n - 1) for i, c in enumerate(r20_sorted)}

        # Rank -R3 (more negative R3 = bigger pullback = higher rank for dip-buying)
        r3_sorted = sorted(codes, key=lambda c: -r3_vals[c])  # sort by -R3 ascending
        r3_rank = {c: i / (n - 1) for i, c in enumerate(r3_sorted)}

        # Composite K score
        scores = {}
        for c in codes:
            scores[c] = R20_WEIGHT * r20_rank[c] + R3_WEIGHT * r3_rank[c]

        return scores

    def passes_trend_filter(self, code: str, date: pd.Timestamp) -> bool:
        """Signal 1 requirement: close > MA10 AND close > MA20."""
        df = self.data[code]
        if date not in df.index:
            return False
        close = df.loc[date, "close"]
        ma10 = df.loc[date, "ma10"]
        ma20 = df.loc[date, "ma20"]
        if np.isnan(ma10) or np.isnan(ma20):
            return False
        return close > ma10 and close > ma20

    def has_volume_surge(self, code: str, date: pd.Timestamp) -> bool:
        """Signal 2: volume_5d_avg > 1.3 * volume_20d_avg."""
        df = self.data[code]
        if date not in df.index:
            return False
        vol5 = df.loc[date, "vol_5d"]
        vol20 = df.loc[date, "vol_20d"]
        if np.isnan(vol5) or np.isnan(vol20) or vol20 <= 0:
            return False
        return (vol5 / vol20) > VOLUME_SURGE_THRESHOLD

    def passes_relative_strength(self, code: str, date: pd.Timestamp) -> bool:
        """Signal 3: ETF's R10 > Benchmark's R10."""
        df = self.data[code]
        bench_df = self.data[BENCHMARK]
        if date not in df.index or date not in bench_df.index:
            return False
        etf_r10 = df.loc[date, "r10"]
        bench_r10 = bench_df.loc[date, "r10"]
        if np.isnan(etf_r10) or np.isnan(bench_r10):
            return False
        return etf_r10 > bench_r10

    def has_breadth_expansion(self, date: pd.Timestamp, date_idx: int) -> bool:
        """Signal 4: breadth_today > breadth_5d_ago."""
        if date_idx < 5:
            return False
        date_5d_ago = self.all_dates[date_idx - 5]
        breadth_today = self.compute_breadth(date)
        breadth_5d_ago = self.compute_breadth(date_5d_ago)
        return breadth_today > breadth_5d_ago

    def check_exit_conditions(self, trade: Trade, date: pd.Timestamp) -> tuple[Optional[str], float]:
        """Check all exit conditions for a position.
        Returns (exit_reason, exit_price) or (None, 0).
        Uses open price for hard stop gap-downs, close for other exits."""
        df = self.data[trade.code]
        if date not in df.index:
            return None, 0.0

        open_price = df.loc[date, "open"]
        close = df.loc[date, "close"]
        ma20 = df.loc[date, "ma20"]
        atr = df.loc[date, "atr14"]

        # Increment days held
        trade.days_held += 1

        # 1. Hard stop at OPEN: if open gaps below -5%, exit at open
        if not np.isnan(open_price) and open_price > 0:
            if (open_price - trade.entry_price) / trade.entry_price <= HARD_STOP_PCT:
                return "hard_stop", open_price

        # Update highest close for trailing stop (use max of open and close)
        if close > trade.highest_close:
            trade.highest_close = close

        # 2. Hard stop at CLOSE: check close as well
        if (close - trade.entry_price) / trade.entry_price <= HARD_STOP_PCT:
            return "hard_stop", close

        # 3. ATR trailing stop
        if not np.isnan(atr) and trade.highest_close > 0:
            trail_level = trade.highest_close - ATR_MULTIPLIER * atr
            if close < trail_level:
                return "atr_trailing_stop", close

        # 4. Trend break: close below MA20 for 2 consecutive days
        if not np.isnan(ma20):
            if close < ma20:
                trade.days_below_ma20 += 1
            else:
                trade.days_below_ma20 = 0
            if trade.days_below_ma20 >= TREND_BREAK_DAYS:
                return "trend_break", close

        # 5. Time stop: 12 trading days max
        if trade.days_held >= TIME_STOP_DAYS:
            return "time_stop", close

        return None, 0.0

    def execute_sell(self, trade: Trade, date: pd.Timestamp, price: float, reason: str):
        """Close a position."""
        trade.exit_date = date
        trade.exit_price = price
        trade.exit_reason = reason
        proceeds = trade.shares * price * (1 - FEE_RATE)
        trade.pnl = proceeds - trade.shares * trade.entry_price * (1 + FEE_RATE)
        trade.pnl_pct = (price * (1 - FEE_RATE)) / (trade.entry_price * (1 + FEE_RATE)) - 1
        self.cash += proceeds
        self.closed_trades.append(trade)

    def run(self):
        """Run the backtest."""
        warmup = 25  # Need at least 20 days for indicators + buffer
        if len(self.all_dates) <= warmup:
            print("Not enough data for warmup.")
            return

        for i in range(warmup, len(self.all_dates)):
            date = self.all_dates[i]
            prev_date = self.all_dates[i - 1]

            # ── Portfolio Stop Cooldown ──
            if self.portfolio_cooldown_remaining > 0:
                self.portfolio_cooldown_remaining -= 1

            if self.portfolio_cooldown_remaining == 0 and self.portfolio_stop_active:
                self.portfolio_stop_active = False
                # Reset peak to current equity to prevent perpetual drawdown trap
                current_eq_reset = self.cash
                for t in self.positions:
                    df_t = self.data[t.code]
                    if date in df_t.index:
                        current_eq_reset += t.shares * df_t.loc[date, "close"]
                self.peak_equity = current_eq_reset
                # Also reset rebalance counter so we wait before re-entering
                self.days_since_rebal = 0

            # ── Check Exit Conditions (using today's close for signal) ──
            positions_to_remove = []
            for trade in self.positions:
                reason, exit_price = self.check_exit_conditions(trade, date)
                if reason:
                    self.execute_sell(trade, date, exit_price, reason)
                    positions_to_remove.append(trade)

            for t in positions_to_remove:
                self.positions.remove(t)

            # ── Portfolio Stop Check ──
            current_equity = self.cash
            for trade in self.positions:
                df = self.data[trade.code]
                if date in df.index:
                    current_equity += trade.shares * df.loc[date, "close"]
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            drawdown_from_peak = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown_from_peak >= PORTFOLIO_STOP_PCT and not self.portfolio_stop_active:
                # Liquidate all positions
                self.portfolio_stop_active = True
                self.portfolio_cooldown_remaining = PORTFOLIO_COOLDOWN
                for trade in self.positions:
                    close = self.data[trade.code].loc[date, "close"]
                    self.execute_sell(trade, date, close, "portfolio_stop")
                self.positions = []

            # ── Rebalance Logic (signal on prev_date close, execute today's open) ──
            self.days_since_rebal += 1
            # Check for new entry: must be flat, past cooldown, and either
            # rebalance period elapsed OR we just exited (check every day when flat)
            can_check = (not self.portfolio_stop_active and
                         self.portfolio_cooldown_remaining == 0 and
                         len(self.positions) == 0 and
                         self.days_since_rebal >= REBAL_PERIOD)
            if can_check:

                # Use previous day's data for signals (no look-ahead)
                breadth = self.compute_breadth(prev_date)

                # Breadth must be above minimum
                if breadth >= BREADTH_ENTRY_MIN:
                    # Signal 4: breadth expansion
                    breadth_expanding = self.has_breadth_expansion(prev_date, i - 1)

                    # Entry requires: breadth > 0.45 AND (Signal 4 OR breadth > 0.55)
                    if breadth_expanding or breadth > BREADTH_STRONG:
                        # Compute K scores (Signal 1)
                        k_scores = self.compute_k_scores(prev_date)

                        # Filter to top-N by K score
                        if k_scores:
                            ranked_by_k = sorted(k_scores.items(), key=lambda x: -x[1])
                            top_k_codes = set(c for c, _ in ranked_by_k[:K_SCORE_TOP_N])

                            # Apply trend filter (part of Signal 1)
                            candidates = {c: s for c, s in k_scores.items()
                                          if c in top_k_codes and self.passes_trend_filter(c, prev_date)}

                            # Apply Signal 3: relative strength vs benchmark
                            candidates = {c: s for c, s in candidates.items()
                                          if self.passes_relative_strength(c, prev_date)}

                            # Exclude currently held (shouldn't be any since we only enter when flat)
                            held_codes = {t.code for t in self.positions}
                            candidates = {c: s for c, s in candidates.items() if c not in held_codes}

                            if candidates:
                                # Signal 2: volume bonus
                                final_scores = {}
                                volume_flags = {}
                                for c, score in candidates.items():
                                    vol_surge = self.has_volume_surge(c, prev_date)
                                    volume_flags[c] = vol_surge
                                    final_scores[c] = score + (VOLUME_BONUS if vol_surge else 0.0)

                                # Pick top-1
                                best_code = max(final_scores, key=final_scores.get)

                                # Position sizing based on breadth
                                size_mult = self.get_position_size_mult(breadth)

                                if size_mult > 0:
                                    df = self.data[best_code]
                                    if date in df.index:
                                        buy_price = df.loc[date, "open"]
                                        if not np.isnan(buy_price) and buy_price > 0:
                                            # Allocate position
                                            per_position = size_mult * current_equity
                                            shares = per_position / (buy_price * (1 + FEE_RATE))
                                            cost = shares * buy_price * (1 + FEE_RATE)
                                            if cost > self.cash:
                                                shares = self.cash / (buy_price * (1 + FEE_RATE))
                                                cost = shares * buy_price * (1 + FEE_RATE)
                                            if shares > 0:
                                                self.cash -= cost
                                                trade = Trade(
                                                    code=best_code,
                                                    entry_date=date,
                                                    entry_price=buy_price,
                                                    shares=shares,
                                                    highest_close=buy_price,
                                                    had_volume_bonus=volume_flags[best_code],
                                                    had_breadth_expansion=breadth_expanding,
                                                )
                                                self.positions.append(trade)

                self.days_since_rebal = 0

            # ── Record Daily Equity ──
            equity = self.cash
            for trade in self.positions:
                df = self.data[trade.code]
                if date in df.index:
                    equity += trade.shares * df.loc[date, "close"]
            self.equity_curve.append((date, equity))
            self.equity = equity
            if equity > self.peak_equity:
                self.peak_equity = equity

        # Close any remaining positions at last day's close
        last_date = self.all_dates[-1]
        for trade in self.positions:
            df = self.data[trade.code]
            if last_date in df.index:
                self.execute_sell(trade, last_date, df.loc[last_date, "close"], "end_of_backtest")
        self.positions = []


# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(bt: Backtest, data: dict[str, pd.DataFrame]):
    """Compute and print comprehensive backtest results."""
    eq = pd.Series(
        [e for _, e in bt.equity_curve],
        index=pd.DatetimeIndex([d for d, _ in bt.equity_curve])
    )

    # Benchmark
    bench = data[BENCHMARK]["close"].reindex(eq.index).dropna()
    bench_ret = bench.iloc[-1] / bench.iloc[0] - 1

    # Basic metrics
    total_days = len(eq)
    years = total_days / 252
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    ann_return = (1 + total_return) ** (1 / years) - 1

    # Daily returns
    daily_ret = eq.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    # Max drawdown
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Trade statistics
    trades = bt.closed_trades
    n_trades = len(trades)
    wins = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    trades_per_week = n_trades / (total_days / 5) if total_days > 0 else 0

    # Profit factor
    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Signal contribution analysis
    vol_bonus_trades = [t for t in trades if t.had_volume_bonus]
    breadth_exp_trades = [t for t in trades if t.had_breadth_expansion]
    vol_bonus_wins = [t for t in vol_bonus_trades if t.pnl and t.pnl > 0]
    breadth_exp_wins = [t for t in breadth_exp_trades if t.pnl and t.pnl > 0]

    # Yearly breakdown
    eq_df = eq.to_frame("equity")
    eq_df["year"] = eq_df.index.year
    yearly_data = []
    for year, grp in eq_df.groupby("year"):
        yr_ret = grp["equity"].iloc[-1] / grp["equity"].iloc[0] - 1
        yr_daily = grp["equity"].pct_change().dropna()
        yr_sharpe = yr_daily.mean() / yr_daily.std() * np.sqrt(252) if yr_daily.std() > 0 else 0
        yr_max = grp["equity"].cummax()
        yr_dd = ((grp["equity"] - yr_max) / yr_max).min()
        yr_trades = [t for t in trades if t.entry_date.year == year]
        yearly_data.append((year, yr_ret, yr_sharpe, yr_dd, len(yr_trades)))

    # ── Print Results ──
    print("=" * 70)
    print("  STRATEGY Q: K Dual-Timeframe + Multi-Signal Confirmation")
    print("=" * 70)
    print(f"\n{'Period:':<25} {eq.index[0].strftime('%Y-%m-%d')} to {eq.index[-1].strftime('%Y-%m-%d')}")
    print(f"{'Trading Days:':<25} {total_days}")
    print(f"{'Initial Capital:':<25} {INITIAL_CAPITAL:,.0f}")
    print(f"{'Final Equity:':<25} {eq.iloc[-1]:,.0f}")

    print(f"\n{'─' * 70}")
    print(f"  PERFORMANCE METRICS")
    print(f"{'─' * 70}")
    print(f"{'Total Return:':<25} {total_return*100:>8.2f}%")
    print(f"{'Annual Return:':<25} {ann_return*100:>8.2f}%")
    print(f"{'Benchmark Return:':<25} {bench_ret*100:>8.2f}%  (510300 buy & hold)")
    print(f"{'Excess Return:':<25} {(total_return - bench_ret)*100:>8.2f}%")
    print(f"{'Sharpe Ratio:':<25} {sharpe:>8.3f}")
    print(f"{'Max Drawdown:':<25} {max_dd*100:>8.2f}%")
    print(f"{'Calmar Ratio:':<25} {calmar:>8.3f}")
    print(f"{'Profit Factor:':<25} {profit_factor:>8.3f}")

    print(f"\n{'─' * 70}")
    print(f"  TRADE STATISTICS")
    print(f"{'─' * 70}")
    print(f"{'Total Trades:':<25} {n_trades}")
    print(f"{'Trades/Week:':<25} {trades_per_week:.2f}")
    print(f"{'Win Rate:':<25} {win_rate*100:.1f}%")
    print(f"{'Avg Win:':<25} {avg_win*100:.2f}%")
    print(f"{'Avg Loss:':<25} {avg_loss*100:.2f}%")
    print(f"{'Avg Win / Avg Loss:':<25} {win_loss_ratio:.2f}")
    if trades:
        avg_hold = np.mean([t.days_held for t in trades])
        print(f"{'Avg Holding Days:':<25} {avg_hold:.1f}")

    print(f"\n{'─' * 70}")
    print(f"  EXIT REASON DISTRIBUTION")
    print(f"{'─' * 70}")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        reason_trades = [t for t in trades if t.exit_reason == reason]
        reason_avg = np.mean([t.pnl_pct for t in reason_trades if t.pnl_pct is not None])
        reason_wins = len([t for t in reason_trades if t.pnl and t.pnl > 0])
        reason_wr = reason_wins / count * 100 if count > 0 else 0
        print(f"  {reason:<22} {count:>4} trades ({count/n_trades*100:5.1f}%)  "
              f"WR: {reason_wr:5.1f}%  avg PnL: {reason_avg*100:>6.2f}%")

    print(f"\n{'─' * 70}")
    print(f"  SIGNAL CONTRIBUTION ANALYSIS")
    print(f"{'─' * 70}")
    print(f"{'Entries with Volume Bonus:':<35} {len(vol_bonus_trades):>4} / {n_trades} "
          f"({len(vol_bonus_trades)/n_trades*100:.1f}%)" if n_trades > 0 else "")
    if vol_bonus_trades:
        vol_wr = len(vol_bonus_wins) / len(vol_bonus_trades) * 100
        vol_avg_pnl = np.mean([t.pnl_pct for t in vol_bonus_trades if t.pnl_pct is not None])
        print(f"{'  -> Win Rate:':<35} {vol_wr:.1f}%")
        print(f"{'  -> Avg PnL:':<35} {vol_avg_pnl*100:.2f}%")

    no_vol_trades = [t for t in trades if not t.had_volume_bonus]
    if no_vol_trades:
        no_vol_wins = [t for t in no_vol_trades if t.pnl and t.pnl > 0]
        no_vol_wr = len(no_vol_wins) / len(no_vol_trades) * 100
        no_vol_avg = np.mean([t.pnl_pct for t in no_vol_trades if t.pnl_pct is not None])
        print(f"{'Entries WITHOUT Volume Bonus:':<35} {len(no_vol_trades):>4} / {n_trades}")
        print(f"{'  -> Win Rate:':<35} {no_vol_wr:.1f}%")
        print(f"{'  -> Avg PnL:':<35} {no_vol_avg*100:.2f}%")

    print()
    print(f"{'Entries with Breadth Expansion:':<35} {len(breadth_exp_trades):>4} / {n_trades} "
          f"({len(breadth_exp_trades)/n_trades*100:.1f}%)" if n_trades > 0 else "")
    if breadth_exp_trades:
        be_wr = len(breadth_exp_wins) / len(breadth_exp_trades) * 100
        be_avg_pnl = np.mean([t.pnl_pct for t in breadth_exp_trades if t.pnl_pct is not None])
        print(f"{'  -> Win Rate:':<35} {be_wr:.1f}%")
        print(f"{'  -> Avg PnL:':<35} {be_avg_pnl*100:.2f}%")

    no_be_trades = [t for t in trades if not t.had_breadth_expansion]
    if no_be_trades:
        no_be_wins = [t for t in no_be_trades if t.pnl and t.pnl > 0]
        no_be_wr = len(no_be_wins) / len(no_be_trades) * 100
        no_be_avg = np.mean([t.pnl_pct for t in no_be_trades if t.pnl_pct is not None])
        print(f"{'Entries WITHOUT Breadth Expansion:':<35} {len(no_be_trades):>4} / {n_trades}")
        print(f"{'  -> Win Rate:':<35} {no_be_wr:.1f}%")
        print(f"{'  -> Avg PnL:':<35} {no_be_avg*100:.2f}%")

    # Both signals present
    both_signals = [t for t in trades if t.had_volume_bonus and t.had_breadth_expansion]
    if both_signals:
        both_wins = [t for t in both_signals if t.pnl and t.pnl > 0]
        both_wr = len(both_wins) / len(both_signals) * 100
        both_avg = np.mean([t.pnl_pct for t in both_signals if t.pnl_pct is not None])
        print(f"\n{'Both Vol Bonus + Breadth Exp:':<35} {len(both_signals):>4} / {n_trades}")
        print(f"{'  -> Win Rate:':<35} {both_wr:.1f}%")
        print(f"{'  -> Avg PnL:':<35} {both_avg*100:.2f}%")

    print(f"\n{'─' * 70}")
    print(f"  YEARLY BREAKDOWN (2021-2025)")
    print(f"{'─' * 70}")
    print(f"  {'Year':<6} {'Return':>9} {'Sharpe':>8} {'MaxDD':>9} {'Trades':>7}")
    print(f"  {'─'*6} {'─'*9} {'─'*8} {'─'*9} {'─'*7}")
    for year, yr_ret, yr_sharpe, yr_dd, yr_trades_count in yearly_data:
        print(f"  {year:<6} {yr_ret*100:>8.2f}% {yr_sharpe:>8.3f} {yr_dd*100:>8.2f}% {yr_trades_count:>7}")

    print(f"\n{'─' * 70}")
    print(f"  TOP TRADED ETFs")
    print(f"{'─' * 70}")
    etf_counts = {}
    etf_pnls = {}
    for t in trades:
        etf_counts[t.code] = etf_counts.get(t.code, 0) + 1
        if t.code not in etf_pnls:
            etf_pnls[t.code] = []
        if t.pnl_pct is not None:
            etf_pnls[t.code].append(t.pnl_pct)
    sorted_etfs = sorted(etf_counts.items(), key=lambda x: -x[1])[:10]
    print(f"  {'Code':<10} {'Trades':>7} {'Win%':>7} {'AvgPnL':>9} {'TotalPnL':>12}")
    print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*9} {'─'*12}")
    for code, count in sorted_etfs:
        pnls = etf_pnls[code]
        wr = len([p for p in pnls if p > 0]) / len(pnls) * 100 if pnls else 0
        avg_p = np.mean(pnls) * 100 if pnls else 0
        total_p = sum(pnls) * 100
        print(f"  {code:<10} {count:>7} {wr:>6.1f}% {avg_p:>8.2f}% {total_p:>11.2f}%")

    print(f"\n{'=' * 70}")
    print(f"  Strategy Q Backtest Complete")
    print(f"{'=' * 70}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} ETFs, computing indicators...")
    compute_indicators(data)
    print("Running backtest...")
    bt = Backtest(data)
    bt.run()
    print("Computing metrics...\n")
    compute_metrics(bt, data)

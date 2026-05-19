"""
Strategy K: Dual Timeframe Momentum + Dip Buying ETF Rotation
=============================================================
Buy ETFs with strong 20-day momentum that experienced a short-term 3-day pullback.
Core idea: buy the dip in an uptrend with regime-aware position sizing.
"""

from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "/Users/heyu11/Code/finance/data_layer/backtest_fixed.db"
BENCHMARK = "510300"
FEE_RATE = 0.0005  # 0.05% per side
INITIAL_CAPITAL = 1_000_000.0

# Strategy parameters
REBAL_PERIOD = 5          # rebalance every N trading days
MAX_POSITIONS = 2         # top N ETFs to hold
R20_WEIGHT = 0.6          # weight for 20-day return rank
R3_WEIGHT = 0.4           # weight for negative 3-day return rank
MA_SHORT = 10             # short-term MA for trend filter
MA_MEDIUM = 20            # medium-term MA for trend filter
ATR_PERIOD = 14           # ATR lookback
ATR_MULTIPLIER = 2.0      # trailing stop distance
HARD_STOP_PCT = -0.05     # -5% hard stop from entry
TREND_BREAK_DAYS = 2      # consecutive days below MA20 to trigger exit
PORTFOLIO_STOP_PCT = 0.07 # 7% drawdown from equity peak -> liquidate
PORTFOLIO_COOLDOWN = 5    # days to wait after portfolio stop
VOLUME_BOOST_THRESHOLD = 1.5  # 5d avg vol / 20d avg vol ratio
VOLUME_BONUS = 0.1        # bonus added to composite score

# Breadth thresholds
BREADTH_FULL = 0.40       # above this: full trading
BREADTH_REDUCED = 0.25    # 0.25-0.40: reduced, below 0.25: cash only


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


def build_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a panel with all dates as index, columns = ETF codes, values = close."""
    closes = pd.DataFrame({code: df["close"] for code, df in data.items()})
    return closes.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# Indicator Computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_indicators(data: dict[str, pd.DataFrame]):
    """Compute all needed indicators for each ETF."""
    for code, df in data.items():
        # Returns
        df["r20"] = df["close"].pct_change(20)
        df["r3"] = df["close"].pct_change(3)
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
    highest_close: float = 0.0  # for trailing stop
    days_below_ma20: int = 0    # for trend break
    is_dip_buy: bool = False    # was this a "dip buy" signal?


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
        self.dip_buy_count = 0
        self.pure_momentum_count = 0

    def get_indicator(self, code: str, date: pd.Timestamp, col: str):
        """Get indicator value for code on date, returns NaN if not available."""
        df = self.data[code]
        if date in df.index:
            return df.loc[date, col]
        return np.nan

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

    def get_breadth_multiplier(self, breadth: float) -> float:
        """Position size multiplier based on breadth."""
        if breadth > 0.60:
            return 1.0
        elif breadth > 0.40:
            return 0.8
        elif breadth >= 0.25:
            return 0.4
        else:
            return 0.0

    def get_max_positions_for_breadth(self, breadth: float) -> int:
        """Max positions allowed based on breadth."""
        if breadth > 0.40:
            return MAX_POSITIONS
        elif breadth >= 0.25:
            return 1
        else:
            return 0

    def compute_composite_scores(self, date: pd.Timestamp) -> dict[str, float]:
        """Compute composite scores for all ETFs on a given date."""
        r20_vals = {}
        r3_vals = {}
        vol_boost = {}

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

            # Volume confirmation
            vol5 = df.loc[date, "vol_5d"]
            vol20 = df.loc[date, "vol_20d"]
            if not np.isnan(vol5) and not np.isnan(vol20) and vol20 > 0:
                vol_boost[code] = (vol5 / vol20) > VOLUME_BOOST_THRESHOLD
            else:
                vol_boost[code] = False

        if len(r20_vals) < 2:
            return {}

        # Cross-sectional ranking
        codes = list(r20_vals.keys())
        n = len(codes)

        # Rank R20 (higher is better)
        r20_sorted = sorted(codes, key=lambda c: r20_vals[c])
        r20_rank = {c: i / (n - 1) for i, c in enumerate(r20_sorted)}

        # Rank -R3 (more negative R3 = bigger pullback = higher rank)
        r3_sorted = sorted(codes, key=lambda c: -r3_vals[c])  # sort by -R3 ascending
        r3_rank = {c: i / (n - 1) for i, c in enumerate(r3_sorted)}

        # Composite score
        scores = {}
        for c in codes:
            score = R20_WEIGHT * r20_rank[c] + R3_WEIGHT * r3_rank[c]
            if vol_boost.get(c, False):
                score += VOLUME_BONUS
            scores[c] = score

        return scores

    def passes_trend_filter(self, code: str, date: pd.Timestamp) -> bool:
        """Check if ETF passes trend filter: close > MA10 and close > MA20."""
        df = self.data[code]
        if date not in df.index:
            return False
        close = df.loc[date, "close"]
        ma10 = df.loc[date, "ma10"]
        ma20 = df.loc[date, "ma20"]
        if np.isnan(ma10) or np.isnan(ma20):
            return False
        return close > ma10 and close > ma20

    def is_dip_signal(self, code: str, date: pd.Timestamp) -> bool:
        """Check if this is a genuine dip-buy signal (negative 3-day return)."""
        df = self.data[code]
        if date not in df.index:
            return False
        r3 = df.loc[date, "r3"]
        return not np.isnan(r3) and r3 < 0

    def check_exit_conditions(self, trade: Trade, date: pd.Timestamp) -> Optional[str]:
        """Check all exit conditions for a position. Returns exit reason or None."""
        df = self.data[trade.code]
        if date not in df.index:
            return None

        close = df.loc[date, "close"]
        ma20 = df.loc[date, "ma20"]
        atr = df.loc[date, "atr14"]

        # Update highest close for trailing stop
        if close > trade.highest_close:
            trade.highest_close = close

        # 1. Hard stop: -5% from entry
        if (close - trade.entry_price) / trade.entry_price <= HARD_STOP_PCT:
            return "hard_stop"

        # 2. ATR trailing stop
        if not np.isnan(atr) and trade.highest_close > 0:
            trail_level = trade.highest_close - ATR_MULTIPLIER * atr
            if close < trail_level:
                return "atr_trailing_stop"

        # 3. Trend break: close below MA20 for 2 consecutive days
        if not np.isnan(ma20):
            if close < ma20:
                trade.days_below_ma20 += 1
            else:
                trade.days_below_ma20 = 0
            if trade.days_below_ma20 >= TREND_BREAK_DAYS:
                return "trend_break"

        return None

    def execute_sell(self, trade: Trade, date: pd.Timestamp, price: float, reason: str):
        """Close a position."""
        trade.exit_date = date
        trade.exit_price = price
        trade.exit_reason = reason
        proceeds = trade.shares * price * (1 - FEE_RATE)
        cost = trade.shares * trade.entry_price * (1 + FEE_RATE)
        trade.pnl = proceeds - cost
        trade.pnl_pct = (price * (1 - FEE_RATE)) / (trade.entry_price * (1 + FEE_RATE)) - 1
        self.cash += proceeds
        self.closed_trades.append(trade)

    def run(self):
        """Run the backtest."""
        # Need at least 20 days of warmup for indicators
        warmup = 25
        if len(self.all_dates) <= warmup:
            print("Not enough data for warmup.")
            return

        for i in range(warmup, len(self.all_dates)):
            date = self.all_dates[i]
            prev_date = self.all_dates[i - 1]

            # ── Portfolio Stop Cooldown ──
            if self.portfolio_cooldown_remaining > 0:
                self.portfolio_cooldown_remaining -= 1

            # ── Check Exit Conditions (using today's close for signal) ──
            # But exits execute at today's close (conservative)
            positions_to_remove = []
            for trade in self.positions:
                reason = self.check_exit_conditions(trade, date)
                if reason:
                    # Exit at today's close
                    close = self.data[trade.code].loc[date, "close"]
                    self.execute_sell(trade, date, close, reason)
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

            if self.portfolio_cooldown_remaining == 0 and self.portfolio_stop_active:
                self.portfolio_stop_active = False

            # ── Rebalance Logic (signal on prev_date close, execute today's open) ──
            self.days_since_rebal += 1
            if (self.days_since_rebal >= REBAL_PERIOD and
                    not self.portfolio_stop_active and
                    self.portfolio_cooldown_remaining == 0):

                # Use previous day's data for signals (no look-ahead)
                breadth = self.compute_breadth(prev_date)
                max_pos = self.get_max_positions_for_breadth(breadth)
                size_mult = self.get_breadth_multiplier(breadth)

                if max_pos > 0 and size_mult > 0:
                    scores = self.compute_composite_scores(prev_date)
                    # Filter by trend
                    candidates = {c: s for c, s in scores.items()
                                  if self.passes_trend_filter(c, prev_date)}

                    # Exclude currently held
                    held_codes = {t.code for t in self.positions}
                    candidates = {c: s for c, s in candidates.items() if c not in held_codes}

                    # How many slots available
                    slots_available = max_pos - len(self.positions)

                    if slots_available > 0 and candidates:
                        # Sort by score descending
                        ranked = sorted(candidates.items(), key=lambda x: -x[1])
                        to_buy = ranked[:slots_available]

                        # Position sizing: equal weight, scaled by breadth
                        # Total equity allocated = size_mult * current_equity
                        current_equity = self.cash
                        for trade in self.positions:
                            df = self.data[trade.code]
                            if date in df.index:
                                current_equity += trade.shares * df.loc[date, "close"]

                        per_position = (size_mult * current_equity) / max_pos

                        for code, score in to_buy:
                            df = self.data[code]
                            if date not in df.index:
                                continue
                            buy_price = df.loc[date, "open"]
                            if np.isnan(buy_price) or buy_price <= 0:
                                continue
                            # Cost including fee
                            shares = per_position / (buy_price * (1 + FEE_RATE))
                            cost = shares * buy_price * (1 + FEE_RATE)
                            if cost > self.cash:
                                shares = self.cash / (buy_price * (1 + FEE_RATE))
                                cost = shares * buy_price * (1 + FEE_RATE)
                            if shares <= 0:
                                continue
                            self.cash -= cost

                            is_dip = self.is_dip_signal(code, prev_date)
                            trade = Trade(
                                code=code,
                                entry_date=date,
                                entry_price=buy_price,
                                shares=shares,
                                highest_close=buy_price,
                                is_dip_buy=is_dip,
                            )
                            self.positions.append(trade)
                            if is_dip:
                                self.dip_buy_count += 1
                            else:
                                self.pure_momentum_count += 1

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
    trades_per_week = n_trades / (total_days / 5)

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Dip buy vs pure momentum
    dip_trades = [t for t in trades if t.is_dip_buy]
    momentum_trades = [t for t in trades if not t.is_dip_buy]
    dip_win_rate = len([t for t in dip_trades if t.pnl and t.pnl > 0]) / len(dip_trades) if dip_trades else 0
    mom_win_rate = len([t for t in momentum_trades if t.pnl and t.pnl > 0]) / len(momentum_trades) if momentum_trades else 0
    dip_avg_pnl = np.mean([t.pnl_pct for t in dip_trades if t.pnl_pct is not None]) if dip_trades else 0
    mom_avg_pnl = np.mean([t.pnl_pct for t in momentum_trades if t.pnl_pct is not None]) if momentum_trades else 0

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
    print("  STRATEGY K: Dual Timeframe Momentum + Dip Buying ETF Rotation")
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

    print(f"\n{'─' * 70}")
    print(f"  TRADE STATISTICS")
    print(f"{'─' * 70}")
    print(f"{'Total Trades:':<25} {n_trades}")
    print(f"{'Trades/Week:':<25} {trades_per_week:.2f}")
    print(f"{'Win Rate:':<25} {win_rate*100:.1f}%")
    print(f"{'Avg Win:':<25} {avg_win*100:.2f}%")
    print(f"{'Avg Loss:':<25} {avg_loss*100:.2f}%")
    print(f"{'Win/Loss Ratio:':<25} {win_loss_ratio:.2f}")
    print(f"{'Avg Holding Days:':<25} {np.mean([(t.exit_date - t.entry_date).days for t in trades if t.exit_date]):.1f}")

    print(f"\n{'─' * 70}")
    print(f"  DIP BUYING vs PURE MOMENTUM")
    print(f"{'─' * 70}")
    print(f"{'Dip Buy Entries:':<25} {len(dip_trades)} ({len(dip_trades)/n_trades*100:.1f}% of trades)" if n_trades > 0 else "")
    print(f"{'Pure Momentum Entries:':<25} {len(momentum_trades)} ({len(momentum_trades)/n_trades*100:.1f}% of trades)" if n_trades > 0 else "")
    print(f"{'Dip Buy Win Rate:':<25} {dip_win_rate*100:.1f}%")
    print(f"{'Momentum Win Rate:':<25} {mom_win_rate*100:.1f}%")
    print(f"{'Dip Buy Avg PnL:':<25} {dip_avg_pnl*100:.2f}%")
    print(f"{'Momentum Avg PnL:':<25} {mom_avg_pnl*100:.2f}%")

    print(f"\n{'─' * 70}")
    print(f"  EXIT REASON BREAKDOWN")
    print(f"{'─' * 70}")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        reason_trades = [t for t in trades if t.exit_reason == reason]
        reason_avg = np.mean([t.pnl_pct for t in reason_trades if t.pnl_pct is not None])
        print(f"  {reason:<22} {count:>4} trades ({count/n_trades*100:5.1f}%)  avg PnL: {reason_avg*100:>6.2f}%")

    print(f"\n{'─' * 70}")
    print(f"  YEARLY BREAKDOWN")
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
    print(f"  {'Code':<10} {'Trades':>7} {'Win%':>7} {'AvgPnL':>9}")
    print(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*9}")
    for code, count in sorted_etfs:
        pnls = etf_pnls[code]
        wr = len([p for p in pnls if p > 0]) / len(pnls) * 100 if pnls else 0
        avg_p = np.mean(pnls) * 100 if pnls else 0
        print(f"  {code:<10} {count:>7} {wr:>6.1f}% {avg_p:>8.2f}%")

    print(f"\n{'=' * 70}")
    print(f"  Strategy K Backtest Complete")
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

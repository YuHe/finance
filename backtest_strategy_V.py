"""
Strategy V: Volume-Price Confirmation Momentum
===============================================
Core Idea: Use volume-weighted momentum for Chinese A-share ETFs.
  vol_ratio = mean(volume on up_days) / mean(volume on down_days) over 20 days
  score = R20 * vol_ratio (amplifies momentum when volume confirms)

Entry:
  - Rank ETFs by composite score
  - TOP_N adaptive: 1 if breadth > 0.55, 2 if 0.40-0.55
  - REQUIRE: vol_ratio > 1.2 (volume confirms uptrend)
  - REQUIRE: close > MA20 AND breadth > 0.40
  - REQUIRE: benchmark close > benchmark MA20

Position sizing:
  vol_ratio > 2.0: 100%
  vol_ratio 1.5-2.0: 80%
  vol_ratio 1.2-1.5: 60%

Exit:
  - 2.0x ATR(14) trailing stop
  - vol_ratio drops below 0.8 (distribution): exit
  - close < MA20 for 2 consecutive days
  - 3 consecutive down days
  - NO hard stop

Rebalance: every 5 days
"""

import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "/Users/heyu11/Code/finance/data_layer/backtest_fixed.db"
BENCHMARK = "510300"
FEE = 0.0005

# Parameters
ATR_MULT = 2.0
ATR_PERIOD = 14
VOL_WINDOW = 20
MA_PERIOD = 20
REBAL_FREQ = 5
BREADTH_MIN = 0.40
VOL_RATIO_ENTRY = 1.2
VOL_RATIO_EXIT = 0.8


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT code, date, open, high, low, close, volume FROM etf_daily ORDER BY code, date",
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_vol_ratio(closes, volumes, window=20):
    """Compute vol_ratio: mean volume on up days / mean volume on down days over rolling window."""
    n = len(closes)
    vol_ratio = np.full(n, np.nan)
    # prev close for up/down classification
    prev_close = np.roll(closes, 1)
    prev_close[0] = np.nan
    is_up = closes > prev_close  # True = up day

    for i in range(window, n):
        w_up = is_up[i - window + 1 : i + 1]
        w_vol = volumes[i - window + 1 : i + 1]
        up_vols = w_vol[w_up]
        down_vols = w_vol[~w_up]
        if len(up_vols) > 0 and len(down_vols) > 0 and down_vols.mean() > 0:
            vol_ratio[i] = up_vols.mean() / down_vols.mean()
    return vol_ratio


def run_backtest():
    df_all = load_data()
    codes = sorted([c for c in df_all["code"].unique() if c != BENCHMARK])

    data = {}
    for code in codes + [BENCHMARK]:
        sub = df_all[df_all["code"] == code].copy().sort_values("date").reset_index(drop=True)
        sub["ma20"] = sub["close"].rolling(MA_PERIOD).mean()
        sub["r20"] = sub["close"] / sub["close"].shift(20) - 1
        sub["prev_close"] = sub["close"].shift(1)
        sub["tr"] = np.maximum(
            sub["high"] - sub["low"],
            np.maximum(
                (sub["high"] - sub["prev_close"]).abs(),
                (sub["low"] - sub["prev_close"]).abs(),
            ),
        )
        sub["atr"] = sub["tr"].rolling(ATR_PERIOD).mean()
        # Compute vol_ratio
        sub["vol_ratio"] = compute_vol_ratio(
            sub["close"].values, sub["volume"].values.astype(float), VOL_WINDOW
        )
        sub = sub.set_index("date")
        data[code] = sub

    bench = data[BENCHMARK]
    trading_dates = bench.index.tolist()

    equity = 1.0
    peak_equity = 1.0
    position = None  # dict or None
    daily_returns = []
    trades = []
    equity_history = [1.0]
    rebal_counter = 0
    below_ma_count = 0
    consec_down_count = 0

    start_idx = 30  # allow indicators to warm up

    for i in range(start_idx, len(trading_dates)):
        date = trading_dates[i]
        prev_date = trading_dates[i - 1]
        daily_ret = 0.0

        # --- Position management ---
        if position is not None:
            code = position["code"]
            if date in data[code].index and prev_date in data[code].index:
                row = data[code].loc[date]
                prev_row = data[code].loc[prev_date]

                day_ret = (row["close"] / prev_row["close"]) - 1
                daily_ret = day_ret * position["size"]

                position["max_price"] = max(position["max_price"], row["high"])
                if not np.isnan(row["atr"]):
                    new_trail = position["max_price"] - ATR_MULT * row["atr"]
                    if position["trailing_stop"] is None or new_trail > position["trailing_stop"]:
                        position["trailing_stop"] = new_trail

                position["days_held"] += 1

                exit_triggered = False
                exit_reason = ""
                exit_price = row["close"]

                # 1. ATR trailing stop
                if position["trailing_stop"] is not None and row["close"] <= position["trailing_stop"]:
                    exit_triggered = True
                    exit_reason = "atr_trail"
                    exit_price = max(position["trailing_stop"], row["low"])

                # 2. Vol_ratio exit (distribution signal)
                if not exit_triggered:
                    # Use prev_date vol_ratio (T-1 signal)
                    vr = data[code].loc[prev_date, "vol_ratio"] if prev_date in data[code].index else np.nan
                    if not np.isnan(vr) and vr < VOL_RATIO_EXIT:
                        exit_triggered = True
                        exit_reason = "vol_exit"

                # 3. Close < MA20 for 2 consecutive days
                if not exit_triggered and not np.isnan(row["ma20"]):
                    if row["close"] < row["ma20"]:
                        below_ma_count += 1
                        if below_ma_count >= 2:
                            exit_triggered = True
                            exit_reason = "ma_break"
                    else:
                        below_ma_count = 0

                # 4. 3 consecutive down days
                if not exit_triggered:
                    if row["close"] < prev_row["close"]:
                        consec_down_count += 1
                        if consec_down_count >= 3:
                            exit_triggered = True
                            exit_reason = "consec_down"
                    else:
                        consec_down_count = 0

                if exit_triggered:
                    trade_ret = (exit_price / position["entry_price"]) - 1
                    # Adjust for position size and fees
                    net_trade_ret = trade_ret * position["size"] - FEE * 2 * position["size"]
                    daily_ret = (exit_price / prev_row["close"] - 1) * position["size"]
                    trades.append(
                        {
                            "code": code,
                            "entry_date": position["entry_date"],
                            "exit_date": date,
                            "entry_price": position["entry_price"],
                            "exit_price": exit_price,
                            "return": net_trade_ret,
                            "days": position["days_held"],
                            "reason": exit_reason,
                            "entry_vol_ratio": position["entry_vol_ratio"],
                            "size": position["size"],
                        }
                    )
                    position = None
                    below_ma_count = 0
                    consec_down_count = 0

        # --- Entry logic ---
        if position is None:
            rebal_counter += 1
            if rebal_counter >= REBAL_FREQ:
                rebal_counter = 0

                # Benchmark condition: bench close > bench MA20 on T-1
                if prev_date in bench.index:
                    b = bench.loc[prev_date]
                    bench_ok = not np.isnan(b["ma20"]) and b["close"] > b["ma20"]
                else:
                    bench_ok = False

                if bench_ok:
                    # Compute breadth on T-1
                    above = 0
                    total = 0
                    for c in codes:
                        if prev_date in data[c].index and not np.isnan(data[c].loc[prev_date, "ma20"]):
                            total += 1
                            if data[c].loc[prev_date, "close"] > data[c].loc[prev_date, "ma20"]:
                                above += 1
                    breadth = above / total if total > 0 else 0

                    if breadth >= BREADTH_MIN:
                        # Determine TOP_N
                        top_n = 1 if breadth > 0.55 else 2

                        # Score all ETFs
                        candidates = []
                        for c in codes:
                            if prev_date not in data[c].index:
                                continue
                            r = data[c].loc[prev_date]
                            if (
                                np.isnan(r["r20"])
                                or np.isnan(r["ma20"])
                                or np.isnan(r["vol_ratio"])
                            ):
                                continue
                            # Requirements
                            if r["close"] <= r["ma20"]:
                                continue
                            if r["vol_ratio"] <= VOL_RATIO_ENTRY:
                                continue
                            # Composite score
                            score = r["r20"] * r["vol_ratio"]
                            candidates.append((c, score, r["vol_ratio"]))

                        if candidates:
                            candidates.sort(key=lambda x: x[1], reverse=True)
                            # Pick top_n (but we only hold 1 position at a time)
                            best = candidates[0]
                            best_code, best_score, best_vr = best

                            # Position sizing based on vol_ratio
                            if best_vr > 2.0:
                                size = 1.0
                            elif best_vr > 1.5:
                                size = 0.8
                            else:
                                size = 0.6

                            if date in data[best_code].index:
                                entry_price = data[best_code].loc[date, "open"]
                                position = {
                                    "code": best_code,
                                    "entry_date": date,
                                    "entry_price": entry_price,
                                    "trailing_stop": None,
                                    "days_held": 0,
                                    "max_price": data[best_code].loc[date, "high"],
                                    "entry_vol_ratio": best_vr,
                                    "size": size,
                                }
                                today_close = data[best_code].loc[date, "close"]
                                daily_ret = (today_close / entry_price - 1) * size
                                # Entry fee already embedded in first day
                                daily_ret -= FEE * size
                                below_ma_count = 0
                                consec_down_count = 0

        equity *= 1 + daily_ret
        if equity > peak_equity:
            peak_equity = equity
        equity_history.append(equity)
        daily_returns.append(daily_ret)

    return {
        "equity_history": np.array(equity_history),
        "daily_returns": np.array(daily_returns),
        "trades": trades,
        "trading_dates": trading_dates[start_idx:],
    }


def print_results(result):
    returns = result["daily_returns"]
    equity = result["equity_history"]
    trades = result["trades"]
    dates = result["trading_dates"]

    total_ret = equity[-1] - 1
    n_days = len(returns)
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1
    ann_years = n_days / 252

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Sortino
    downside = returns[returns < 0]
    down_std = downside.std() if len(downside) > 0 else 1e-9
    sortino = returns.mean() / down_std * np.sqrt(252) if down_std > 0 else 0

    if trades:
        wins = [t for t in trades if t["return"] > 0]
        losses = [t for t in trades if t["return"] <= 0]
        win_rate = len(wins) / len(trades)
        avg_win = np.mean([t["return"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["return"] for t in losses]) if losses else 0
        profit_factor = (
            sum(t["return"] for t in wins) / abs(sum(t["return"] for t in losses))
            if losses and sum(t["return"] for t in losses) != 0
            else float("inf")
        )
        avg_days = np.mean([t["days"] for t in trades])
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_days = 0

    trades_per_week = len(trades) / (n_days / 5)

    # Benchmark return
    conn = sqlite3.connect(DB_PATH)
    bench_df = pd.read_sql(
        f"SELECT date, close FROM etf_daily WHERE code = '{BENCHMARK}' ORDER BY date", conn
    )
    conn.close()
    bench_df["date"] = pd.to_datetime(bench_df["date"])
    bench_df = bench_df.set_index("date")
    # align to same period
    first_date = dates[0]
    last_date = dates[-1]
    bench_period = bench_df.loc[first_date:last_date]
    if len(bench_period) > 1:
        bench_ret = bench_period["close"].iloc[-1] / bench_period["close"].iloc[0] - 1
    else:
        bench_ret = 0

    print("=" * 70)
    print("  Strategy V: Volume-Price Confirmation Momentum")
    print("=" * 70)
    print(f"  Period:             {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')} ({ann_years:.1f} yrs)")
    print(f"  Total Return:       {total_ret*100:.2f}%")
    print(f"  Annualized Return:  {ann_ret*100:.2f}%")
    print(f"  Benchmark Return:   {bench_ret*100:.2f}% (510300 buy-hold)")
    print(f"  Max Drawdown:       {max_dd*100:.2f}%")
    print(f"  Sharpe Ratio:       {sharpe:.3f}")
    print(f"  Sortino Ratio:      {sortino:.3f}")
    print(f"  Calmar Ratio:       {calmar:.3f}")
    print(f"  Win Rate:           {win_rate*100:.1f}% ({len(wins)}/{len(trades)} trades)")
    print(f"  Avg Win:            {avg_win*100:.2f}%")
    print(f"  Avg Loss:           {avg_loss*100:.2f}%")
    print(f"  Profit Factor:      {profit_factor:.2f}")
    print(f"  Trades/Week:        {trades_per_week:.2f}")
    print(f"  Avg Holding Days:   {avg_days:.1f}")
    print("=" * 70)

    # Yearly breakdown
    print(f"\n  {'Year':<6} {'Return':>10} {'MaxDD':>10} {'Trades':>8} {'WinRate':>10} {'Sharpe':>8}")
    print(f"  {'-'*54}")
    for year in range(2021, 2027):
        mask = np.array([d.year == year for d in dates])
        if not mask.any():
            continue
        yr = returns[mask]
        yr_eq = np.cumprod(1 + yr)
        yr_ret = yr_eq[-1] - 1
        yr_peak = np.maximum.accumulate(yr_eq)
        yr_dd = ((yr_eq - yr_peak) / yr_peak).min()
        yr_sharpe = (yr.mean() / yr.std() * np.sqrt(252)) if yr.std() > 0 else 0
        yr_trades = [t for t in trades if t["entry_date"].year == year]
        yr_wins = [t for t in yr_trades if t["return"] > 0]
        yr_wr = len(yr_wins) / len(yr_trades) * 100 if yr_trades else 0
        print(
            f"  {year:<6} {yr_ret*100:>9.2f}% {yr_dd*100:>9.2f}% {len(yr_trades):>8} {yr_wr:>9.1f}% {yr_sharpe:>7.2f}"
        )

    # Exit reason breakdown
    print(f"\n  Exit Reason Breakdown:")
    print(f"  {'Reason':<15} {'Count':>6} {'Pct':>7} {'AvgRet':>10} {'WinRate':>10}")
    print(f"  {'-'*50}")
    reasons = {}
    for t in trades:
        r = t["reason"]
        if r not in reasons:
            reasons[r] = []
        reasons[r].append(t["return"])
    for r, rets in sorted(reasons.items(), key=lambda x: -len(x[1])):
        cnt = len(rets)
        avg_r = np.mean(rets)
        wr = sum(1 for x in rets if x > 0) / cnt * 100
        print(f"  {r:<15} {cnt:>6} {cnt/len(trades)*100:>6.1f}% {avg_r*100:>9.2f}% {wr:>9.1f}%")

    # Volume ratio statistics
    print(f"\n  Volume Ratio Statistics:")
    if trades:
        entry_vrs = [t["entry_vol_ratio"] for t in trades]
        print(f"    Avg vol_ratio at entry:   {np.mean(entry_vrs):.3f}")
        print(f"    Median vol_ratio at entry: {np.median(entry_vrs):.3f}")
        print(f"    Min/Max at entry:          {np.min(entry_vrs):.3f} / {np.max(entry_vrs):.3f}")

        # Correlation between entry vol_ratio and trade return
        vrs = np.array(entry_vrs)
        rets_arr = np.array([t["return"] for t in trades])
        if len(vrs) > 2:
            corr = np.corrcoef(vrs, rets_arr)[0, 1]
            print(f"    Correlation (vol_ratio vs return): {corr:.4f}")

        # Position size distribution
        sizes = [t["size"] for t in trades]
        size_counts = {}
        for s in sizes:
            key = f"{s*100:.0f}%"
            size_counts[key] = size_counts.get(key, 0) + 1
        print(f"\n    Position Size Distribution:")
        for k, v in sorted(size_counts.items()):
            print(f"      {k}: {v} trades ({v/len(trades)*100:.1f}%)")

    # Top/Bottom 5 trades
    if trades:
        sorted_trades = sorted(trades, key=lambda x: x["return"], reverse=True)
        print(f"\n  Top 5 Trades:")
        for t in sorted_trades[:5]:
            print(
                f"    {t['code']} {t['entry_date'].strftime('%Y-%m-%d')} -> "
                f"{t['exit_date'].strftime('%Y-%m-%d')}: {t['return']*100:+.2f}% "
                f"({t['days']}d, {t['reason']}, vr={t['entry_vol_ratio']:.2f}, sz={t['size']*100:.0f}%)"
            )
        print(f"\n  Bottom 5 Trades:")
        for t in sorted_trades[-5:]:
            print(
                f"    {t['code']} {t['entry_date'].strftime('%Y-%m-%d')} -> "
                f"{t['exit_date'].strftime('%Y-%m-%d')}: {t['return']*100:+.2f}% "
                f"({t['days']}d, {t['reason']}, vr={t['entry_vol_ratio']:.2f}, sz={t['size']*100:.0f}%)"
            )


if __name__ == "__main__":
    print("Running Strategy V: Volume-Price Confirmation Momentum...")
    result = run_backtest()
    print_results(result)

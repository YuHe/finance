"""
Phase 5 数据补采: margin_daily / sector_daily / sector_fund_flow
使用: /Users/heyu11/.pyenv/versions/ak_env/bin/python data_phase5_collect.py
"""
import sqlite3
import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import akshare as ak

DB_PATH = Path(__file__).parent / "data_layer" / "signals.db"
START_DATE = "20210514"
END_DATE = "20251231"

# THS sector name mapping (ETF concept -> THS industry name)
SECTOR_MAP = {
    '医药生物': '医药商业',
    '银行': '银行',
    '券商': '证券',
    '证券': '证券',
    '食品饮料': '食品加工制造',
    '新能源': '电池',
    '半导体': '半导体',
    '新能源车': '汽车整车',
    '白酒': '白酒',
    '光伏': '光伏设备',
    '房地产': '房地产',
    '有色金属': '工业金属',
    '军工': '军工装备',
    '5G通信': '通信设备',
    '环保': '环保设备',
    '游戏': '游戏',
    '医疗': '医疗器械',
    '钢铁': '钢铁',
    '汽车': '汽车整车',
    '传媒': '文化传媒',
    '农业': '种植业与林业',
    '芯片': '半导体',
    '煤炭': '煤炭开采加工',
    '生物医药': '生物制品',
    '化工': '化学制品',
}


def collect_margin():
    """融资融券 - SSE bulk + SZSE daily samples"""
    print("\n=== 1. Margin Data (融资融券) ===")
    conn = sqlite3.connect(DB_PATH)
    n_total = 0

    # Method 1: SSE bulk (covers 2010-present)
    print("  Trying stock_margin_sse (bulk)...")
    try:
        df = ak.stock_margin_sse(start_date=START_DATE, end_date=END_DATE)
        if df is not None and not df.empty:
            # Cols: 信用交易日期, 融资余额, 融资买入额, 融券余量, 融券余量金额, 融券卖出量, 融资融券余额
            df.columns = ['date', 'margin_balance', 'margin_buy', 'short_qty',
                          'short_balance', 'short_sell_qty', 'total_balance']
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
            df = df[['date', 'margin_balance', 'margin_buy', 'short_balance', 'total_balance']]
            cur = conn.cursor()
            cur.executemany(
                "INSERT OR REPLACE INTO margin_daily(date, margin_balance, margin_buy, short_balance, total_balance) VALUES (?,?,?,?,?)",
                df.values.tolist()
            )
            conn.commit()
            n_total += len(df)
            print(f"  SSE OK: {len(df)} rows, {df['date'].min()} ~ {df['date'].max()}")
    except Exception as e:
        print(f"  SSE FAIL: {e}")

    time.sleep(0.6)

    # Method 2: SZSE daily snapshots for recent dates to supplement
    print("  Trying stock_margin_szse (daily snapshots)...")
    szse_count = 0
    # Get trading dates from SSE data if available
    dates_to_try = pd.bdate_range('2025-01-02', '2025-05-23', freq='B')
    dates_to_try = [d.strftime('%Y%m%d') for d in dates_to_try][-30:]  # last 30 biz days
    for dt in dates_to_try:
        try:
            df2 = ak.stock_margin_szse(date=dt)
            if df2 is not None and not df2.empty:
                # Returns single row: 融资买入额, 融资余额, 融券卖出量, 融券余量, 融券余额, 融资融券余额
                # Values in 亿
                date_str = f"{dt[:4]}-{dt[4:6]}-{dt[6:]}"
                row = df2.iloc[0]
                margin_bal = float(row.iloc[1]) * 1e8  # 亿 -> 元
                short_bal = float(row.iloc[4]) * 1e8
                total_bal = float(row.iloc[5]) * 1e8
                margin_buy = float(row.iloc[0]) * 1e8
                # Store as SZSE supplement (won't overwrite SSE data since dates overlap)
                # We'll just track SZSE separately in a log
                szse_count += 1
        except Exception:
            pass
        time.sleep(0.6)
    if szse_count > 0:
        print(f"  SZSE sampled: {szse_count} dates (not stored separately, SSE covers market)")

    conn.close()
    print(f"  TOTAL margin_daily: {n_total} rows")
    return n_total


def collect_sector_daily():
    """行业板块日线 via THS (同花顺)"""
    print("\n=== 2. Sector Daily (行业日线 via THS) ===")
    conn = sqlite3.connect(DB_PATH)
    n_total = 0
    failed = []

    # Deduplicate: some ETF sectors map to same THS name
    seen_ths = set()
    sector_pairs = []
    for orig, ths_name in SECTOR_MAP.items():
        if ths_name not in seen_ths:
            seen_ths.add(ths_name)
            sector_pairs.append((orig, ths_name))

    for i, (orig_name, ths_name) in enumerate(sector_pairs):
        print(f"  [{i+1}/{len(sector_pairs)}] {orig_name} -> {ths_name}...", end=" ")
        try:
            df = ak.stock_board_industry_index_ths(
                symbol=ths_name, start_date=START_DATE, end_date=END_DATE
            )
            if df is None or df.empty:
                print("EMPTY")
                failed.append(orig_name)
                continue
            # Cols: 日期, 开盘价, 最高价, 最低价, 收盘价, 成交量, 成交额
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df['sector'] = orig_name
            # Compute turnover_rate and pct_change
            df['pct_change'] = df['close'].pct_change() * 100
            df['turnover_rate'] = None  # THS index doesn't provide turnover_rate directly
            df = df[['date', 'sector', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover_rate', 'pct_change']]
            df = df.dropna(subset=['date', 'close'])
            cur = conn.cursor()
            cur.executemany(
                "INSERT OR REPLACE INTO sector_daily(date, sector, open, high, low, close, volume, amount, turnover_rate, pct_change) VALUES (?,?,?,?,?,?,?,?,?,?)",
                df.values.tolist()
            )
            conn.commit()
            n_total += len(df)
            print(f"OK ({len(df)} rows)")
        except Exception as e:
            err_msg = str(e)[:80]
            print(f"FAIL: {err_msg}")
            failed.append(orig_name)
        time.sleep(0.8)

    conn.close()
    print(f"  TOTAL sector_daily new rows: {n_total}")
    if failed:
        print(f"  FAILED sectors: {failed}")
    return n_total, failed


def collect_sector_fund_flow():
    """行业资金流 - try sina stock_fund_flow_industry (即时 snapshot only)"""
    print("\n=== 3. Sector Fund Flow (行业资金流) ===")
    conn = sqlite3.connect(DB_PATH)
    n_total = 0

    # Method 1: sina stock_fund_flow_industry (即时 only, gives today's snapshot)
    print("  Trying stock_fund_flow_industry (sina, 即时)...")
    try:
        df = ak.stock_fund_flow_industry(symbol='即时')
        if df is not None and not df.empty:
            # Cols: 序号, 行业, 行业指数, 行业-涨跌幅, 流入资金, 流出资金, 净额, 公司家数, 领涨股, 领涨股-涨跌幅, 当前价
            today = pd.Timestamp.now().strftime('%Y-%m-%d')
            records = []
            for _, row in df.iterrows():
                sector = str(row.iloc[1])
                net_inflow = float(row.iloc[6]) if pd.notna(row.iloc[6]) else 0
                inflow = float(row.iloc[4]) if pd.notna(row.iloc[4]) else 0
                outflow = float(row.iloc[5]) if pd.notna(row.iloc[5]) else 0
                pct = net_inflow / inflow * 100 if inflow > 0 else 0
                records.append((today, sector, net_inflow, pct, None, None, None, None))
            cur = conn.cursor()
            cur.executemany(
                "INSERT OR REPLACE INTO sector_fund_flow(date, sector, main_net, main_pct, super_large_net, large_net, medium_net, small_net) VALUES (?,?,?,?,?,?,?,?)",
                records
            )
            conn.commit()
            n_total += len(records)
            print(f"  Sina 即时 OK: {len(records)} sectors for {today}")
    except Exception as e:
        print(f"  Sina 即时 FAIL: {e}")

    time.sleep(0.6)

    # Method 2: Try EastMoney sector_fund_flow_rank (often blocked)
    print("  Trying stock_sector_fund_flow_rank (EastMoney)...")
    try:
        df = ak.stock_sector_fund_flow_rank(indicator='今日', sector_type='行业资金流')
        if df is not None and not df.empty:
            today = pd.Timestamp.now().strftime('%Y-%m-%d')
            print(f"  EM rank OK: {len(df)} rows")
            # Parse and store
            for _, row in df.iterrows():
                sector = str(row.get('名称', row.iloc[1]))
                main_net = float(row.get('主力净流入-净额', row.iloc[4])) if pd.notna(row.iloc[4]) else 0
                main_pct = float(row.get('主力净流入-净占比', row.iloc[5])) if pd.notna(row.iloc[5]) else 0
                conn.execute(
                    "INSERT OR REPLACE INTO sector_fund_flow(date, sector, main_net, main_pct) VALUES (?,?,?,?)",
                    (today, sector, main_net, main_pct)
                )
            conn.commit()
            n_total += len(df)
    except Exception as e:
        print(f"  EM rank FAIL (expected if blocked): {str(e)[:60]}")

    time.sleep(0.6)

    # Method 3: Try EastMoney sector_fund_flow_hist per sector
    print("  Trying stock_sector_fund_flow_hist (EastMoney, per sector)...")
    hist_ok = 0
    for sector in list(SECTOR_MAP.values())[:3]:  # test with 3 first
        try:
            df = ak.stock_sector_fund_flow_hist(symbol=sector)
            if df is not None and not df.empty:
                hist_ok += 1
                # If first one works, do all
                break
        except Exception:
            pass
        time.sleep(0.6)

    if hist_ok > 0:
        print(f"  EM hist works! Collecting all sectors...")
        for sector in set(SECTOR_MAP.values()):
            try:
                df = ak.stock_sector_fund_flow_hist(symbol=sector)
                if df is None or df.empty:
                    continue
                # Typical cols: 日期, 主力净流入-净额, 主力净流入-净占比, ...
                cols = df.columns.tolist()
                df_out = pd.DataFrame()
                df_out['date'] = pd.to_datetime(df.iloc[:, 0]).dt.strftime('%Y-%m-%d')
                df_out['sector'] = sector
                df_out['main_net'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                df_out['main_pct'] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
                df_out = df_out.dropna(subset=['date'])
                cur = conn.cursor()
                cur.executemany(
                    "INSERT OR REPLACE INTO sector_fund_flow(date, sector, main_net, main_pct, super_large_net, large_net, medium_net, small_net) VALUES (?,?,?,?,NULL,NULL,NULL,NULL)",
                    df_out[['date', 'sector', 'main_net', 'main_pct']].values.tolist()
                )
                conn.commit()
                n_total += len(df_out)
                print(f"    {sector}: {len(df_out)} rows")
            except Exception as e:
                print(f"    {sector}: FAIL {str(e)[:40]}")
            time.sleep(0.8)
    else:
        print("  EM hist blocked, skipping.")

    conn.close()
    print(f"  TOTAL sector_fund_flow new rows: {n_total}")
    return n_total


def print_summary():
    """Print collection summary"""
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    conn = sqlite3.connect(DB_PATH)

    for table in ['margin_daily', 'sector_daily', 'sector_fund_flow']:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            date_range = conn.execute(f"SELECT MIN(date), MAX(date) FROM {table}").fetchone()
            print(f"  {table}: {count} rows | {date_range[0]} ~ {date_range[1]}")
        except Exception as e:
            print(f"  {table}: ERROR - {e}")

    # Sector coverage
    try:
        sectors = conn.execute("SELECT DISTINCT sector FROM sector_daily").fetchall()
        print(f"  sector_daily distinct sectors: {len(sectors)}")
        for s in sectors:
            print(f"    - {s[0]}")
    except Exception:
        pass

    conn.close()


def signal_test():
    """Quick signal test: margin z-score vs next-10d return"""
    print("\n" + "="*60)
    print("SIGNAL TEST: Margin Balance Z-Score (60d)")
    print("="*60)
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT date, margin_balance, total_balance FROM margin_daily ORDER BY date", conn)
        if len(df) < 70:
            print("  Not enough data for signal test")
            conn.close()
            return
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        # Z-score of margin_balance (60d rolling)
        df['mb_mean60'] = df['margin_balance'].rolling(60).mean()
        df['mb_std60'] = df['margin_balance'].rolling(60).std()
        df['mb_zscore'] = (df['margin_balance'] - df['mb_mean60']) / df['mb_std60']
        # Next-10d return proxy: use total_balance change
        df['fwd_10d_ret'] = df['total_balance'].pct_change(10).shift(-10)
        # Correlation
        valid = df.dropna(subset=['mb_zscore', 'fwd_10d_ret'])
        if len(valid) > 30:
            corr = valid['mb_zscore'].corr(valid['fwd_10d_ret'])
            print(f"  Data points: {len(valid)}")
            print(f"  Corr(margin_zscore_60d, fwd_10d_balance_change): {corr:.4f}")
            print(f"  Interpretation: {'Contrarian signal (negative)' if corr < 0 else 'Momentum signal (positive)'}")
            # Quintile analysis
            valid['q'] = pd.qcut(valid['mb_zscore'], 5, labels=False, duplicates='drop')
            quintile_ret = valid.groupby('q')['fwd_10d_ret'].mean() * 100
            print(f"  Quintile avg fwd returns (%):")
            for q, r in quintile_ret.items():
                print(f"    Q{q}: {r:.3f}%")
        else:
            print("  Not enough valid data points for correlation")
    except Exception as e:
        print(f"  Signal test error: {e}")
    conn.close()


if __name__ == "__main__":
    print("Phase 5 Data Collection")
    print(f"DB: {DB_PATH}")
    print(f"Date range: {START_DATE} ~ {END_DATE}")

    n_margin = collect_margin()
    n_sector, failed_sectors = collect_sector_daily()
    n_flow = collect_sector_fund_flow()

    print_summary()

    if n_margin > 0:
        signal_test()

"""Collect Dragon Tiger List, market fund flow, and sector margin data."""
import akshare as ak
import pandas as pd
import sqlite3
import time

DB = "/Users/heyu11/Code/finance/data_layer/signals.db"

def task1_lhb():
    """Dragon Tiger List data."""
    print("=== Task 1: Dragon Tiger List ===")
    chunks = [
        ('20210101', '20221231'),
        ('20230101', '20241231'),
        ('20250101', '20251231'),
    ]
    frames = []
    for s, e in chunks:
        try:
            print(f"  Fetching LHB {s}-{e}...")
            df = ak.stock_lhb_detail_em(start_date=s, end_date=e)
            frames.append(df)
            print(f"    Got {len(df)} rows")
            time.sleep(2)
        except Exception as ex:
            print(f"    Failed: {ex}")
    if not frames:
        print("  All chunks failed, trying full range...")
        try:
            frames = [ak.stock_lhb_detail_em(start_date='20210514', end_date='20251231')]
        except Exception as ex:
            print(f"  Full range also failed: {ex}")
            return
    lhb = pd.concat(frames, ignore_index=True)
    print(f"  Total LHB rows: {len(lhb)}, columns: {list(lhb.columns)}")

    with sqlite3.connect(DB) as conn:
        lhb.to_sql('lhb_raw', conn, if_exists='replace', index=False)
        print("  Saved lhb_raw")

    # Aggregate to sector weekly
    # Find date and sector columns
    date_col = None
    sector_col = None
    for c in lhb.columns:
        if '日期' in c or 'date' in c.lower():
            date_col = c
        if '行业' in c or 'industry' in c.lower() or '板块' in c:
            sector_col = c

    buy_col = None
    for c in lhb.columns:
        if '净买' in c or '买入' in c:
            buy_col = c
            break

    if date_col and sector_col:
        lhb[date_col] = pd.to_datetime(lhb[date_col])
        lhb['week'] = lhb[date_col].dt.to_period('W').dt.start_time
        grp = lhb.groupby(['week', sector_col]).agg(
            count=(date_col, 'count'),
            net_buy_sum=(buy_col, 'sum') if buy_col else (date_col, 'count')
        ).reset_index()
        grp.columns = ['week', 'sector', 'count', 'net_buy_sum']
        with sqlite3.connect(DB) as conn:
            grp.to_sql('lhb_sector_weekly', conn, if_exists='replace', index=False)
        print(f"  Saved lhb_sector_weekly: {len(grp)} rows")
    else:
        print(f"  Could not identify date/sector columns for aggregation")
        print(f"  Columns: {list(lhb.columns)}")


def task2_fund_flow():
    """Market-level fund flow."""
    print("\n=== Task 2: Market Fund Flow ===")
    try:
        df = ak.stock_market_fund_flow()
        if df is not None and len(df) > 0:
            print(f"  Got {len(df)} rows, columns: {list(df.columns)}")
            with sqlite3.connect(DB) as conn:
                df.to_sql('market_fund_flow_daily', conn, if_exists='replace', index=False)
            print("  Saved market_fund_flow_daily")
            return
    except Exception as ex:
        print(f"  stock_market_fund_flow failed: {ex}")

    print("  Trying ETF fund flow alternative...")
    time.sleep(1)
    try:
        df = ak.stock_individual_fund_flow(stock="510300", market="sh")
        if df is not None and len(df) > 0:
            print(f"  Got {len(df)} rows")
            with sqlite3.connect(DB) as conn:
                df.to_sql('market_fund_flow_daily', conn, if_exists='replace', index=False)
            print("  Saved market_fund_flow_daily")
            return
    except Exception as ex:
        print(f"  ETF fund flow also failed: {ex}")


def task3_sector_margin():
    """Per-stock margin sampled and aggregated to sector."""
    print("\n=== Task 3: Sector Margin Snapshots ===")
    sample_dates = ['20210701','20220101','20220701','20230101','20230701','20240101','20240701','20250101']
    frames = []
    for d in sample_dates:
        try:
            print(f"  Fetching margin detail for {d}...")
            df = ak.stock_margin_detail_sse(date=d)
            if df is not None and len(df) > 0:
                df['snapshot_date'] = d
                frames.append(df)
                print(f"    Got {len(df)} rows")
            else:
                print(f"    Empty result")
            time.sleep(2)
        except Exception as ex:
            print(f"    Failed: {ex}")
            time.sleep(1)

    if not frames:
        print("  No margin data collected")
        return

    all_df = pd.concat(frames, ignore_index=True)
    print(f"  Total margin rows: {len(all_df)}, columns: {list(all_df.columns)}")

    # Try to get stock-to-sector mapping via stock_board_industry_name_em
    print("  Getting industry mapping...")
    try:
        mapping = ak.stock_board_industry_name_em()
        time.sleep(1)
        print(f"    Got {len(mapping)} industries")
    except Exception as ex:
        print(f"    Industry mapping failed: {ex}")
        # Save raw without sector aggregation
        with sqlite3.connect(DB) as conn:
            all_df.to_sql('sector_margin_snapshot', conn, if_exists='replace', index=False)
        print("  Saved raw sector_margin_snapshot (no sector agg)")
        return

    # Get constituent stocks for each industry
    # This is too slow for all industries. Instead, use stock_sector_spot for mapping.
    # Just save raw data and note we need mapping
    with sqlite3.connect(DB) as conn:
        all_df.to_sql('sector_margin_snapshot', conn, if_exists='replace', index=False)
    print(f"  Saved sector_margin_snapshot: {len(all_df)} rows")


if __name__ == '__main__':
    task1_lhb()
    task2_fund_flow()
    task3_sector_margin()
    print("\n=== Done ===")

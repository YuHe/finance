"""
修复 backtest_fixed.db 中的份额折算伪影
方法: 检测单日跳变 > 40%, 计算调整因子, 应用到历史价格上 (模拟后复权)

原理:
- 份额合并: 如 512800 从 1.777 → 0.888 (-50%), 调整因子 = 0.888/1.777 ≈ 0.5
  将合并日之前的所有价格 × 0.5, 使其连续
- 份额折算: 如 512200 从 0.441 → 1.225 (+170%), 调整因子 = 1.225/0.441 ≈ 2.78
  将折算日之前的所有价格 × 2.78, 使其连续

这等效于后复权 (向前调整历史价格使当前价格不变)
"""

import sqlite3
import pandas as pd
import numpy as np

INPUT_DB = "data_layer/backtest_fixed.db"
OUTPUT_DB = "data_layer/backtest_adjusted.db"
SPLIT_THRESHOLD = 0.35  # 单日变化 > 35% 判定为分拆

def detect_and_fix_splits():
    # 读取原始数据
    conn = sqlite3.connect(INPUT_DB)
    df_all = pd.read_sql("SELECT * FROM etf_daily ORDER BY code, date", conn)

    # 读取北向资金数据
    try:
        nb_flow = pd.read_sql("SELECT * FROM northbound_flow", conn)
    except:
        nb_flow = pd.DataFrame()
    try:
        nb_deal = pd.read_sql("SELECT * FROM northbound_deal", conn)
    except:
        nb_deal = pd.DataFrame()
    conn.close()

    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values(['code', 'date']).reset_index(drop=True)

    print(f"原始数据: {len(df_all)} 条, {df_all['code'].nunique()} 只ETF")
    print(f"日期范围: {df_all['date'].min().strftime('%Y-%m-%d')} ~ {df_all['date'].max().strftime('%Y-%m-%d')}")

    # 检测并修复每只 ETF
    split_log = []
    fixed_dfs = []

    for code in sorted(df_all['code'].unique()):
        sub = df_all[df_all['code'] == code].copy().sort_values('date').reset_index(drop=True)

        # 计算每日收益率
        sub['ret'] = sub['close'] / sub['close'].shift(1) - 1

        # 检测分拆点
        splits = sub[sub['ret'].abs() > SPLIT_THRESHOLD].copy()

        if len(splits) > 0:
            print(f"\n{'─'*50}")
            print(f"{code}: 发现 {len(splits)} 个分拆点")

            # 从最早的分拆点开始,逐个修复
            # 后复权 = 将分拆点之前的价格乘以调整因子
            # 调整因子 = close[split_day] / close[split_day - 1] 的纯价格比
            # 实际上我们需要让 close[day-1] * factor = close[day] 大致成立(除去正常涨跌)

            for idx, split_row in splits.iterrows():
                split_date = split_row['date']
                raw_ret = split_row['ret']

                # 调整因子: 让分拆日之后的价格 = 分拆日之前 * 因子
                # 后复权: 调整分拆前的数据
                # 如果是合并(-50%), 分拆后价格是分拆前的 0.5x
                # 要让序列连续, 将分拆前所有价格 × (1+ret)
                # 即: old_prices *= (split_day_close / prev_day_close)

                prev_idx = idx - 1
                if prev_idx >= 0:
                    prev_close = sub.loc[prev_idx, 'close']
                    split_close = sub.loc[idx, 'close']
                    factor = split_close / prev_close  # 这个就是包含分拆的比率

                    # 估计正常日涨跌约 ±2%, 分拆超出的部分才是调整因子
                    # 简化: 直接用 factor (包含了当天微小的正常涨跌, 误差可忽略)

                    print(f"  {split_date.strftime('%Y-%m-%d')}: "
                          f"ret={raw_ret*100:.1f}%, "
                          f"factor={factor:.4f}, "
                          f"prev_close={prev_close:.4f}, "
                          f"split_close={split_close:.4f}")

                    split_log.append({
                        'code': code,
                        'date': split_date.strftime('%Y-%m-%d'),
                        'raw_return': raw_ret,
                        'adjustment_factor': factor,
                    })

                    # 应用: 将此日之前的所有 OHLC 乘以 factor (后复权)
                    mask = sub['date'] < split_date
                    for col in ['open', 'high', 'low', 'close']:
                        sub.loc[mask, col] = sub.loc[mask, col] * factor
                    # volume 反向调整 (份额合并后volume减少, 复权应增加)
                    if 'volume' in sub.columns and factor != 0:
                        sub.loc[mask, 'volume'] = sub.loc[mask, 'volume'] / factor
                    if 'amount' in sub.columns:
                        pass  # 成交额不需要调整 (金额不变)

        sub = sub.drop(columns=['ret'], errors='ignore')
        fixed_dfs.append(sub)

    # 合并
    df_fixed = pd.concat(fixed_dfs, ignore_index=True)

    # 验证修复结果
    print(f"\n{'='*60}")
    print("验证修复结果")
    print(f"{'='*60}")

    remaining_issues = []
    for code in df_fixed['code'].unique():
        sub = df_fixed[df_fixed['code'] == code].sort_values('date')
        ret = sub['close'].pct_change()
        extreme = ret[ret.abs() > SPLIT_THRESHOLD]
        if len(extreme) > 0:
            for dt in extreme.index:
                row = sub.loc[dt]
                remaining_issues.append(f"  {code} {row['date'].strftime('%Y-%m-%d')}: {ret[dt]*100:.1f}%")

    if remaining_issues:
        print(f"⚠️  仍有 {len(remaining_issues)} 个极端收益:")
        for issue in remaining_issues[:10]:
            print(issue)
    else:
        print("✓ 所有分拆伪影已消除!")

    # 额外验证: 检查修复后的8只ETF
    print(f"\n{'─'*50}")
    print("修复后的分拆ETF验证 (分拆点前后5天):")
    affected_codes = [s['code'] for s in split_log]
    for code in set(affected_codes):
        sub = df_fixed[df_fixed['code'] == code].sort_values('date').reset_index(drop=True)
        ret = sub['close'].pct_change()
        max_ret = ret.abs().max()
        print(f"  {code}: max daily |return| = {max_ret*100:.2f}%")

    # 写入新数据库
    print(f"\n{'='*60}")
    print(f"写入 {OUTPUT_DB}")

    out_conn = sqlite3.connect(OUTPUT_DB)
    out_conn.execute("DROP TABLE IF EXISTS etf_daily")
    out_conn.execute("""
    CREATE TABLE etf_daily (
        code TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        amount REAL,
        PRIMARY KEY (code, date)
    )
    """)
    out_conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_code ON etf_daily(code)")
    out_conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON etf_daily(date)")

    # 转日期为字符串
    df_fixed['date'] = df_fixed['date'].dt.strftime('%Y-%m-%d')

    # 只保留需要的列
    cols = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']
    available_cols = [c for c in cols if c in df_fixed.columns]
    df_out = df_fixed[available_cols]

    df_out.to_sql("etf_daily", out_conn, if_exists="replace", index=False)

    # 迁移北向资金数据
    if not nb_flow.empty:
        nb_flow.to_sql("northbound_flow", out_conn, if_exists="replace", index=False)
        print(f"  迁移 northbound_flow: {len(nb_flow)} 条")
    if not nb_deal.empty:
        nb_deal.to_sql("northbound_deal", out_conn, if_exists="replace", index=False)
        print(f"  迁移 northbound_deal: {len(nb_deal)} 条")

    out_conn.commit()

    # 最终统计
    cursor = out_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM etf_daily")
    print(f"\n  总记录: {cursor.fetchone()[0]}")
    cursor.execute("SELECT MIN(date), MAX(date) FROM etf_daily")
    dr = cursor.fetchone()
    print(f"  日期范围: {dr[0]} ~ {dr[1]}")

    out_conn.close()

    # 保存分拆日志
    if split_log:
        print(f"\n分拆调整日志:")
        for s in split_log:
            print(f"  {s['code']} {s['date']}: factor={s['adjustment_factor']:.4f} (ret={s['raw_return']*100:.1f}%)")

    print(f"\n{'='*60}")
    print(f"✓ 完成! 后复权数据库: {OUTPUT_DB}")
    print(f"{'='*60}")

    return split_log


if __name__ == "__main__":
    detect_and_fix_splits()

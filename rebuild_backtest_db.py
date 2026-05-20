"""
重建回测数据库 - 使用 baostock 后复权数据
消除份额折算/拆分伪影

输出: data_layer/backtest_adjusted.db
"""

import sqlite3
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime
import time

# ─── 配置 ───
OUTPUT_DB = "data_layer/backtest_adjusted.db"
START_DATE = "2021-01-01"  # 多拉几个月用于warmup
END_DATE = "2025-12-31"

ETF_CODES = [
    "159819", "159825", "159870", "159928", "159996",
    "510300",  # 基准
    "512000", "512010", "512170", "512200", "512330",
    "512400", "512480", "512660", "512800", "512890",
    "512980", "513050", "515170", "515210", "515220",
    "515790", "515880", "516110", "516160", "516950",
]


def bs_code(code: str) -> str:
    """转为 baostock 格式"""
    return f"sh.{code}" if code[0] in "56" else f"sz.{code}"


def fetch_etf_data(code: str, start: str, end: str) -> pd.DataFrame:
    """获取单只ETF后复权日线数据"""
    rs = bs.query_history_k_data_plus(
        bs_code(code),
        "date,open,high,low,close,volume,amount",
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="1",  # 后复权 — 关键!
    )

    if rs.error_code != "0":
        print(f"  [ERROR] {code}: {rs.error_msg}")
        return pd.DataFrame()

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        print(f"  [WARN] {code}: 无数据")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=rs.fields)
    # 转数值
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")
    df = df.dropna(subset=["close"])
    df["code"] = code
    return df[["code", "date", "open", "high", "low", "close", "volume", "amount"]]


def validate_no_splits(df: pd.DataFrame) -> list:
    """验证无单日>40%跳变"""
    issues = []
    for code in df["code"].unique():
        sub = df[df["code"] == code].sort_values("date").copy()
        sub["ret"] = sub["close"].pct_change()
        extreme = sub[sub["ret"].abs() > 0.40]
        if len(extreme) > 0:
            for _, row in extreme.iterrows():
                issues.append({
                    "code": code,
                    "date": row["date"],
                    "return": row["ret"],
                })
    return issues


def main():
    print("=" * 60)
    print("重建回测数据库 (后复权)")
    print(f"输出: {OUTPUT_DB}")
    print(f"区间: {START_DATE} ~ {END_DATE}")
    print(f"标的: {len(ETF_CODES)} 只")
    print("=" * 60)

    # 登录 baostock
    print("\n登录 baostock...")
    lg = bs.login()
    if lg.error_code != "0":
        print(f"登录失败: {lg.error_msg}")
        return
    print("登录成功")

    # 获取数据
    all_dfs = []
    for i, code in enumerate(ETF_CODES):
        print(f"\n[{i+1}/{len(ETF_CODES)}] 获取 {code}...")
        df = fetch_etf_data(code, START_DATE, END_DATE)
        if not df.empty:
            print(f"  → {len(df)} 条记录, {df['date'].min()} ~ {df['date'].max()}")
            all_dfs.append(df)
        else:
            print(f"  → 无数据!")
        time.sleep(0.3)  # 避免请求过快

    bs.logout()
    print("\n已登出 baostock")

    if not all_dfs:
        print("ERROR: 未获取到任何数据!")
        return

    # 合并
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\n总计: {len(df_all)} 条记录, {df_all['code'].nunique()} 只ETF")

    # 验证
    print("\n验证数据质量 (检查分拆伪影)...")
    issues = validate_no_splits(df_all)
    if issues:
        print(f"⚠️  发现 {len(issues)} 个潜在分拆点:")
        for issue in issues:
            print(f"  {issue['code']} on {issue['date']}: {issue['return']*100:.1f}%")
        print("\n注意: 后复权数据理论上不应有分拆跳变。如仍有,可能是ETF合并/清盘。")
    else:
        print("✓ 无分拆伪影,数据干净!")

    # 写入数据库
    print(f"\n写入 {OUTPUT_DB}...")
    conn = sqlite3.connect(OUTPUT_DB)
    conn.execute("DROP TABLE IF EXISTS etf_daily")
    conn.execute("""
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
    conn.execute("CREATE INDEX idx_daily_code ON etf_daily(code)")
    conn.execute("CREATE INDEX idx_daily_date ON etf_daily(date)")

    df_all.to_sql("etf_daily", conn, if_exists="replace", index=False)
    conn.commit()

    # 统计
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM etf_daily")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(date), MAX(date) FROM etf_daily")
    date_range = cursor.fetchone()
    cursor.execute("SELECT code, COUNT(*), MIN(date), MAX(date) FROM etf_daily GROUP BY code ORDER BY code")
    code_stats = cursor.fetchall()

    print(f"\n数据库统计:")
    print(f"  总记录: {total}")
    print(f"  日期范围: {date_range[0]} ~ {date_range[1]}")
    print(f"\n  {'Code':<10} {'Records':>8} {'Start':>12} {'End':>12}")
    print(f"  {'─'*44}")
    for code, count, start, end in code_stats:
        print(f"  {code:<10} {count:>8} {start:>12} {end:>12}")

    # 也迁移北向资金数据(如果有)
    try:
        old_conn = sqlite3.connect("data_layer/backtest_fixed.db")
        # 复制 northbound 表
        for table in ["northbound_flow", "northbound_deal"]:
            try:
                nb_df = pd.read_sql(f"SELECT * FROM {table}", old_conn)
                if not nb_df.empty:
                    nb_df.to_sql(table, conn, if_exists="replace", index=False)
                    print(f"  迁移 {table}: {len(nb_df)} 条")
            except Exception:
                pass
        old_conn.close()
    except Exception as e:
        print(f"  北向数据迁移跳过: {e}")

    conn.close()
    print(f"\n✓ 数据库重建完成: {OUTPUT_DB}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""数据层 - 数据获取、存储、涨跌停/成交量标记"""

import os
import sqlite3
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .etf_pool import ETF_POOL, BENCHMARK_CODE

DB_PATH = os.environ.get(
    "ETF_DB_PATH",
    os.path.join(os.path.dirname(__file__), "etf_data.db")
)


def _bs_code(code: str) -> str:
    """将6位代码转为 baostock 格式: 6开头→sh., 其余→sz."""
    return f"sh.{code}" if code.startswith("6") else f"sz.{code}"


class DataManager:
    """统一数据管理：获取行情、计算涨跌停、提供回测/信号所需数据"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS etf_daily (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            turnover REAL,
            is_limit_up INTEGER DEFAULT 0,
            is_limit_down INTEGER DEFAULT 0,
            PRIMARY KEY (code, date)
        )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_code ON etf_daily(code)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_daily_date ON etf_daily(date)"
        )
        self.conn.commit()

    # ─── 数据获取 ───────────────────────────────────────────

    def update_etf(self, code: str, start_date: str = "2015-01-01", end_date: str = None):
        """增量更新单只ETF历史数据（使用 BaoStock，支持境外服务器）"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # 查询已有最新日期，做增量更新
        row = self.conn.execute(
            "SELECT MAX(date) FROM etf_daily WHERE code = ?", (code,)
        ).fetchone()
        if row[0]:
            last = pd.to_datetime(row[0]) + timedelta(days=1)
            start_date = last.strftime("%Y-%m-%d")
            if start_date > end_date:
                return  # 已最新

        try:
            lg = bs.login()
            if lg.error_code != "0":
                print(f"[DataManager] BaoStock 登录失败: {lg.error_msg}")
                return

            rs = bs.query_history_k_data_plus(
                _bs_code(code),
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",   # 后复权
            )
            bs.logout()

            if rs.error_code != "0":
                print(f"[DataManager] 获取 {code} 失败: {rs.error_msg}")
                return

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return

            df = pd.DataFrame(rows, columns=rs.fields)
            df = df.replace("", np.nan)
            for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close"])

        except Exception as e:
            print(f"[DataManager] 获取 {code} 失败: {e}")
            return

        if df.empty:
            return

        df["code"] = code
        df["turnover"] = df["turn"]
        df["pct_change"] = df["pctChg"] / 100
        df["is_limit_up"] = (df["pct_change"] >= 0.098).astype(int)
        df["is_limit_down"] = (df["pct_change"] <= -0.098).astype(int)
        df = df[["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover", "is_limit_up", "is_limit_down"]]

        df.to_sql("etf_daily", self.conn, if_exists="append", index=False)
        self.conn.commit()

    def update_all(self, start_date: str = "2015-01-01", end_date: str = None):
        """更新ETF池所有标的 + 基准"""
        codes = [etf["code"] for etf in ETF_POOL] + [BENCHMARK_CODE]
        for code in codes:
            print(f"  更新 {code}...")
            self.update_etf(code, start_date, end_date)
        print("全部更新完成")

    # ─── 数据查询 ───────────────────────────────────────────

    def get_daily(self, code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取单只ETF日频数据"""
        sql = "SELECT * FROM etf_daily WHERE code = ?"
        params = [code]
        if start_date:
            sql += " AND date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date <= ?"
            params.append(end_date)
        sql += " ORDER BY date"
        df = pd.read_sql(sql, self.conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def get_all_daily(self, start_date: str = None, end_date: str = None) -> dict[str, pd.DataFrame]:
        """获取所有ETF日频数据，返回 {code: DataFrame}"""
        codes = [etf["code"] for etf in ETF_POOL]
        return {code: self.get_daily(code, start_date, end_date) for code in codes}

    def get_close_matrix(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取收盘价矩阵 (date x code)"""
        codes = [etf["code"] for etf in ETF_POOL]
        frames = []
        for code in codes:
            df = self.get_daily(code, start_date, end_date)
            if not df.empty:
                frames.append(df.set_index("date")[["close"]].rename(columns={"close": code}))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()

    def get_amount_matrix(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取成交额矩阵 (date x code)"""
        codes = [etf["code"] for etf in ETF_POOL]
        frames = []
        for code in codes:
            df = self.get_daily(code, start_date, end_date)
            if not df.empty:
                frames.append(df.set_index("date")[["amount"]].rename(columns={"amount": code}))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()

    def get_limit_status(self, date: str) -> dict[str, dict]:
        """获取指定日期各ETF涨跌停状态"""
        rows = self.conn.execute(
            "SELECT code, is_limit_up, is_limit_down FROM etf_daily WHERE date = ?",
            (date,)
        ).fetchall()
        return {
            row[0]: {"is_limit_up": bool(row[1]), "is_limit_down": bool(row[2])}
            for row in rows
        }

    def get_trading_dates(self, start_date: str = None, end_date: str = None) -> list[str]:
        """获取交易日列表（基于基准ETF有数据的日期）"""
        sql = "SELECT DISTINCT date FROM etf_daily WHERE code = ? ORDER BY date"
        params = [BENCHMARK_CODE]
        rows = self.conn.execute(sql, params).fetchall()
        dates = [row[0] for row in rows]
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        return dates

    def get_weekly_rebalance_dates(self, start_date: str = None, end_date: str = None) -> list[str]:
        """获取每周五（每周最后一个交易日）作为调仓计算日"""
        dates = self.get_trading_dates(start_date, end_date)
        if not dates:
            return []
        df = pd.DataFrame({"date": pd.to_datetime(dates)})
        df["week"] = df["date"].dt.isocalendar().week
        df["year"] = df["date"].dt.year
        # 每周最后一个交易日
        weekly = df.groupby(["year", "week"])["date"].max().reset_index(drop=True)
        return [d.strftime("%Y-%m-%d") for d in sorted(weekly)]

    def close(self):
        self.conn.close()

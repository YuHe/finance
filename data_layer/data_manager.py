"""数据层 - 数据获取、存储、涨跌停/成交量标记"""

import os
import sqlite3
import threading
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
    """将6位代码转为 baostock 格式: 5/6开头→sh.(上交所), 其余→sz.(深交所)"""
    return f"sh.{code}" if code[0] in "56" else f"sz.{code}"


class DataManager:
    """统一数据管理：获取行情、计算涨跌停、提供回测/信号所需数据"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_tables()

    @property
    def conn(self) -> sqlite3.Connection:
        """每个线程独立的 SQLite 连接"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _ensure_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
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
        conn.commit()
        conn.close()

    # ─── 数据获取 ───────────────────────────────────────────

    def update_etf(self, code: str, start_date: str = "2015-01-01", end_date: str = None, _logged_in: bool = False):
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
            if not _logged_in:
                lg = bs.login()
                if lg.error_code != "0":
                    print(f"[DataManager] BaoStock 登录失败: {lg.error_msg}")
                    return

            # 获取后复权价格（用于技术指标计算）
            rs_adj = bs.query_history_k_data_plus(
                _bs_code(code),
                "date,open,high,low,close,volume,amount,turn",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",   # 后复权
            )
            # 获取不复权涨跌幅（用于涨跌停判断，避免复权调整干扰）
            rs_raw = bs.query_history_k_data_plus(
                _bs_code(code),
                "date,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",   # 不复权
            )

            if not _logged_in:
                bs.logout()

            if rs_adj.error_code != "0":
                print(f"[DataManager] 获取 {code} 失败: {rs_adj.error_msg}")
                return

            # 解析后复权数据
            rows_adj = []
            while rs_adj.next():
                rows_adj.append(rs_adj.get_row_data())
            if not rows_adj:
                return
            df = pd.DataFrame(rows_adj, columns=rs_adj.fields)

            # 解析不复权涨跌幅
            rows_raw = []
            while rs_raw.next():
                rows_raw.append(rs_raw.get_row_data())
            df_raw = pd.DataFrame(rows_raw, columns=["date", "pctChg"]) if rows_raw else pd.DataFrame(columns=["date", "pctChg"])

        except Exception as e:
            print(f"[DataManager] 获取 {code} 失败: {e}")
            return

        if df.empty:
            return

        # 数值转换
        df = df.replace("", np.nan)
        for col in ["open", "high", "low", "close", "volume", "amount", "turn"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])

        # 合并不复权涨跌幅
        if not df_raw.empty:
            df_raw["pctChg"] = pd.to_numeric(df_raw["pctChg"].replace("", np.nan), errors="coerce")
            df = df.merge(df_raw, on="date", how="left")
        else:
            df["pctChg"] = np.nan

        df["code"] = code
        df["turnover"] = df["turn"]
        # 涨跌停：用不复权涨跌幅（%转小数），避免复权调整产生虚假涨跌停
        pct = df["pctChg"] / 100
        df["is_limit_up"] = (pct >= 0.098).fillna(0).astype(int)
        df["is_limit_down"] = (pct <= -0.098).fillna(0).astype(int)

        df = df[["code", "date", "open", "high", "low", "close",
                 "volume", "amount", "turnover", "is_limit_up", "is_limit_down"]]

        # 使用 INSERT OR REPLACE 避免重复主键崩溃
        placeholders = ",".join(["?"] * len(df.columns))
        cols = ",".join(df.columns)
        sql = f"INSERT OR REPLACE INTO etf_daily ({cols}) VALUES ({placeholders})"
        self.conn.executemany(sql, df.values.tolist())
        self.conn.commit()

    def update_all(self, start_date: str = "2015-01-01", end_date: str = None):
        """更新ETF池所有标的 + 基准（单次登录，批量更新）"""
        lg = bs.login()
        if lg.error_code != "0":
            print(f"[DataManager] BaoStock 登录失败: {lg.error_msg}")
            return

        codes = [etf["code"] for etf in ETF_POOL] + [BENCHMARK_CODE]
        for code in codes:
            print(f"  更新 {code}...")
            self.update_etf(code, start_date, end_date, _logged_in=True)

        bs.logout()
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

    def get_close_matrix(self, start_date: str = None, end_date: str = None, codes: list = None) -> pd.DataFrame:
        """获取收盘价矩阵 (date x code)"""
        if codes is None:
            codes = [etf["code"] for etf in ETF_POOL]
        frames = []
        for code in codes:
            df = self.get_daily(code, start_date, end_date)
            if not df.empty:
                frames.append(df.set_index("date")[["close"]].rename(columns={"close": code}))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()

    def get_amount_matrix(self, start_date: str = None, end_date: str = None, codes: list = None) -> pd.DataFrame:
        """获取成交额矩阵 (date x code)"""
        if codes is None:
            codes = [etf["code"] for etf in ETF_POOL]
        frames = []
        for code in codes:
            df = self.get_daily(code, start_date, end_date)
            if not df.empty:
                frames.append(df.set_index("date")[["amount"]].rename(columns={"amount": code}))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()

    def get_open_matrix(self, start_date: str = None, end_date: str = None, codes: list = None) -> pd.DataFrame:
        """获取开盘价矩阵 (date x code)"""
        if codes is None:
            codes = [etf["code"] for etf in ETF_POOL]
        frames = []
        for code in codes:
            df = self.get_daily(code, start_date, end_date)
            if not df.empty:
                frames.append(df.set_index("date")[["open"]].rename(columns={"open": code}))
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
        """获取交易日列表（优先基于基准ETF，fallback到任意有数据的ETF）"""
        sql = "SELECT DISTINCT date FROM etf_daily WHERE code = ? ORDER BY date"
        params = [BENCHMARK_CODE]
        rows = self.conn.execute(sql, params).fetchall()
        # 如果基准没数据，fallback到整个数据库中任意有数据的日期
        if not rows:
            sql = "SELECT DISTINCT date FROM etf_daily ORDER BY date"
            rows = self.conn.execute(sql).fetchall()
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
        # 使用 isocalendar 的 year 和 week 保持一致，避免跨年边界问题
        iso = df["date"].dt.isocalendar()
        df["iso_year"] = iso["year"]
        df["iso_week"] = iso["week"]
        # 每周最后一个交易日
        weekly = df.groupby(["iso_year", "iso_week"])["date"].max().reset_index(drop=True)
        return [d.strftime("%Y-%m-%d") for d in sorted(weekly)]

    def get_data_status(self) -> list[dict]:
        """获取所有 ETF + 基准的数据状态（日期范围、记录数）"""
        all_codes = [etf["code"] for etf in ETF_POOL] + [BENCHMARK_CODE]
        results = []
        for code in all_codes:
            row = self.conn.execute(
                "SELECT MIN(date), MAX(date), COUNT(*) FROM etf_daily WHERE code = ?",
                (code,)
            ).fetchone()
            results.append({
                "code": code,
                "start_date": row[0] or None,
                "end_date": row[1] or None,
                "count": row[2] or 0,
            })
        return results

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

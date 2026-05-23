"""
Collect intraday data for A-share sector ETFs using AKShare.
Uses stock_zh_a_minute (Sina source) which provides ~1970 bars per period:
  - 15min: ~6 months history
  - 60min: ~2 years history
EastMoney functions (fund_etf_hist_min_em, stock_zh_a_hist_min_em) are currently
returning connection errors and are not usable.
"""
import sqlite3
import time
import warnings
import pandas as pd
import numpy as np
import akshare as ak

warnings.filterwarnings('ignore')

DB_PATH = '/Users/heyu11/Code/finance/data_layer/signals.db'

ETF_CODES = [
    '512010', '512800', '512000', '512880', '515180', '516160',
    '512480', '515030', '512690', '515790', '512200', '512400',
    '512660', '515050', '512580', '159869', '512170', '515210',
    '516110', '512980', '159825', '512760', '515220', '512290', '159870'
]

# SZ-listed ETFs use 'sz' prefix, SH-listed use 'sh'
def get_symbol(code):
    return f'sz{code}' if code.startswith('15') else f'sh{code}'


def init_db(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS etf_intraday (
        code TEXT NOT NULL,
        datetime TEXT NOT NULL,
        period TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL, amount REAL,
        PRIMARY KEY (code, datetime, period)
    )''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_intraday_code_dt ON etf_intraday(code, datetime)')
    conn.execute('''CREATE TABLE IF NOT EXISTS intraday_signals (
        code TEXT NOT NULL,
        date TEXT NOT NULL,
        opening_gap REAL,
        first_30min_ret REAL,
        last_30min_ret REAL,
        volume_ratio_am_pm REAL,
        vwap_deviation REAL,
        PRIMARY KEY (code, date)
    )''')
    conn.commit()


def collect_intraday(conn):
    total_rows = 0
    results = {}
    for i, code in enumerate(ETF_CODES):
        symbol = get_symbol(code)
        print(f'[{i+1}/{len(ETF_CODES)}] Collecting {code} ({symbol})...')
        code_rows = 0
        for period in ['15', '60']:
            try:
                df = ak.stock_zh_a_minute(symbol=symbol, period=period)
                if df.empty:
                    print(f'  period={period}: empty')
                    continue
                df = df.rename(columns={'day': 'datetime'})
                df['code'] = code
                df['period'] = period
                df[['code','datetime','period','open','high','low','close','volume','amount']].to_sql(
                    'etf_intraday', conn, if_exists='append', index=False,
                    method='multi'
                )
                code_rows += len(df)
                date_range = f"{df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}"
                print(f'  period={period}: {len(df)} rows, {date_range}')
            except Exception as e:
                print(f'  period={period}: FAILED - {e}')
            time.sleep(0.8)
        total_rows += code_rows
        results[code] = code_rows
        time.sleep(0.8)
    return total_rows, results


def compute_signals(conn):
    """Compute intraday signals from 15-min data."""
    print('\nComputing intraday signals...')
    df = pd.read_sql('SELECT * FROM etf_intraday WHERE period="15" ORDER BY code, datetime', conn)
    if df.empty:
        print('No 15-min data available for signal computation.')
        return 0
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date.astype(str)
    df['time'] = df['datetime'].dt.time

    signals = []
    for code, gdf in df.groupby('code'):
        dates = gdf['date'].unique()
        prev_close = None
        for date in sorted(dates):
            day = gdf[gdf['date'] == date].sort_values('datetime')
            if len(day) < 2:
                prev_close = day['close'].iloc[-1]
                continue

            day_open = day['open'].iloc[0]
            day_close = day['close'].iloc[-1]

            # opening_gap
            opening_gap = (day_open - prev_close) / prev_close if prev_close else None

            # first_30min_ret: first 2 bars of 15-min = 30 min
            first_bars = day.head(2)
            first_30min_ret = (first_bars['close'].iloc[-1] - day_open) / day_open

            # last_30min_ret: last 2 bars
            last_bars = day.tail(2)
            last_30min_ret = (day_close - last_bars['open'].iloc[0]) / last_bars['open'].iloc[0]

            # volume_ratio_am_pm: morning (before 12:00) vs afternoon
            am = day[day['datetime'].dt.hour < 12]
            pm = day[day['datetime'].dt.hour >= 13]
            am_vol = am['volume'].sum()
            pm_vol = pm['volume'].sum()
            volume_ratio = am_vol / pm_vol if pm_vol > 0 else None

            # vwap_deviation
            vwap = (day['amount'].sum() / day['volume'].sum()) if day['volume'].sum() > 0 else day_close
            vwap_dev = (day_close - vwap) / vwap

            signals.append({
                'code': code, 'date': date,
                'opening_gap': opening_gap,
                'first_30min_ret': first_30min_ret,
                'last_30min_ret': last_30min_ret,
                'volume_ratio_am_pm': volume_ratio,
                'vwap_deviation': vwap_dev
            })
            prev_close = day_close

    if signals:
        sig_df = pd.DataFrame(signals)
        sig_df.to_sql('intraday_signals', conn, if_exists='replace', index=False)
        print(f'Computed {len(sig_df)} signal rows for {sig_df["code"].nunique()} ETFs')
        print(f'Date range: {sig_df["date"].min()} to {sig_df["date"].max()}')
        return len(sig_df)
    return 0


def main():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # Clear old data to avoid duplicates on re-run
    conn.execute('DELETE FROM etf_intraday')
    conn.commit()

    total_rows, results = collect_intraday(conn)
    sig_rows = compute_signals(conn)
    conn.close()

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'Total intraday rows collected: {total_rows}')
    print(f'Signal rows computed: {sig_rows}')
    print(f'ETFs collected: {sum(1 for v in results.values() if v > 0)}/{len(ETF_CODES)}')
    print(f'\nFunction used: ak.stock_zh_a_minute (Sina source)')
    print(f'Periods: 15-min (~6 months), 60-min (~2 years)')
    print(f'\nNOTE: Sina provides max ~1970 bars per period.')
    print(f'  15-min: ~6 months history')
    print(f'  60-min: ~2 years history')
    print(f'Full 2021-2025 history NOT available via free AKShare intraday APIs.')
    print(f'EastMoney APIs (fund_etf_hist_min_em, stock_zh_a_hist_min_em) returning connection errors.')


if __name__ == '__main__':
    main()

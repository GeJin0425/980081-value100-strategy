import time

import requests
import akshare as ak

import numpy as np
import pandas as pd

from .Ashare import get_price

# 511260十年国债ETF历次(除息日, 每份分红金额)记录。
# 该ETF 2017年成立, 2025年9月起才开始现金分红。
DIVIDENDS_511260 = [
    ('2025-09-23', 1.3600),
    ('2025-12-26', 0.8330),
    ('2026-03-25', 0.6711),
    ('2026-06-25', 1.2686),
]


def apply_qfq(df, dividends):
    """对不复权日线做前复权调整，返回新增 close_raw/high_raw/low_raw/adjust_factor 列的DataFrame"""
    df = df.sort_index()
    factor = np.ones(len(df))
    for ex_str, div in dividends:
        ex = pd.Timestamp(ex_str)
        mask = df.index < ex
        if mask.any():
            prev_close = df.loc[mask, 'close'].iloc[-1]
            ex_idx = df.index.get_indexer([ex], method='nearest')[0]
            factor[:ex_idx] *= (prev_close - div) / prev_close

    return pd.DataFrame({
        'open': (df['open'].values * factor).round(3),
        'close': (df['close'].values * factor).round(3),
        'high': (df['high'].values * factor).round(3),
        'low': (df['low'].values * factor).round(3),
        'volume': df['volume'].values,
        'close_raw': df['close'].values,
        'high_raw': df['high'].values,
        'low_raw': df['low'].values,
        'adjust_factor': factor.round(6),
    }, index=df.index)


CNINDEX_API = 'https://hq.cnindex.com.cn/market/market/getIndexDailyData'
CNINDEX_INDEX = '980081'  # 国证价值100 (价值ETF易方达159263跟踪的指数)
CNINDEX_TOTAL_RETURN_INDEX = '480081'  # 国证价值100全收益(价值100R)
CSINDEX_PERF_API = 'https://www.csindex.com.cn/csindex-home/perf/index-perf'
PE_PROXY_INDEX = '000922'  # 中证红利: 980081无官方免费历史PE, 用高股息价值指数PE做估值代理


def fetch_cnindex_daily(index_code, start='2013-01-01', end=None):
    """拉取国证指数官方日线(980081价格 / 480081全收益)。

    返回与策略引擎兼容的 DataFrame:
    open/close/high/low/volume + close_raw/high_raw/low_raw/adjust_factor。
    指数没有现金分红, 因此前复权列与原始列相同, adjust_factor 恒为 1。
    """
    if end is None:
        end = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d')
    url = CNINDEX_API
    params = {
        'indexCode': index_code,
        'startDate': start,
        'endDate': end,
    }
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
        ),
        'Referer': 'https://www.cnindex.com.cn/',
    }
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=25)
            payload = resp.json()
            if payload.get('code') != 200:
                raise RuntimeError(f'CNIndex返回错误: {payload.get("code")} {payload.get("message")}')
            rows = payload['data']['data']
            if not rows:
                raise RuntimeError('CNIndex返回空数据')
            df = pd.DataFrame(rows)
            dates = pd.DatetimeIndex(
                pd.to_datetime(df.iloc[:, 0], unit='ms', utc=True)
            ).tz_convert('Asia/Shanghai')
            out = pd.DataFrame({
                'open': df.iloc[:, 1].astype(float).to_numpy(),
                'close': df.iloc[:, 5].astype(float).to_numpy(),
                'high': df.iloc[:, 2].astype(float).to_numpy(),
                'low': df.iloc[:, 4].astype(float).to_numpy(),
                'volume': pd.to_numeric(df.iloc[:, 9], errors='coerce').fillna(0.0).to_numpy(),
            }, index=pd.DatetimeIndex(dates).tz_localize(None))
            out = out.sort_index()
            out = out.drop_duplicates(keep='last')
            out['close_raw'] = out['close']
            out['high_raw'] = out['high']
            out['low_raw'] = out['low']
            out['adjust_factor'] = 1.0
            return out[['open', 'close', 'high', 'low', 'volume',
                        'close_raw', 'high_raw', 'low_raw', 'adjust_factor']]
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f'无法获取{index_code}行情: {e}') from e
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f'无法获取{index_code}行情')


def fetch_980081_daily(start='2013-01-01', end=None):
    """国证价值100价格指数日线(策略信号用)"""
    return fetch_cnindex_daily(CNINDEX_INDEX, start=start, end=end)


def fetch_480081_daily(start='2013-01-01', end=None):
    """国证价值100全收益指数日线(持仓收益/买入持有基准用)"""
    return fetch_cnindex_daily(CNINDEX_TOTAL_RETURN_INDEX, start=start, end=end)


def fetch_511260_close(count=2500):
    """拉取511260十年国债ETF前复权收盘价序列（空仓期配置资产, 含现金分红）"""
    raw = get_price('sh511260', frequency='1d', count=count)
    return apply_qfq(raw, DIVIDENDS_511260)['close']


def fetch_csindex_daily_pe(index_code=PE_PROXY_INDEX, start='2013-01-01', end=None):
    """中证指数官网: 指数日线+滚动市盈率(peg列), 免费覆盖2013至今。

    980081/480081是国证指数, 官网不提供历史估值; 这里用中证红利(000922)
    的历史PE作为价值因子的代理(高股息价值指数, 与国证价值100高度同质)。
    """
    if end is None:
        end = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y%m%d')
    params = {
        'indexCode': index_code,
        'startDate': pd.Timestamp(start).strftime('%Y%m%d'),
        'endDate': pd.Timestamp(end).strftime('%Y%m%d'),
    }
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
        ),
        'Referer': 'https://www.csindex.com.cn/',
    }
    for attempt in range(5):
        try:
            resp = requests.get(CSINDEX_PERF_API, params=params, headers=headers, timeout=30)
            payload = resp.json()
            rows = payload.get('data')
            if str(payload.get('code')) != '200' or not rows:
                raise RuntimeError(
                    f'中证接口返回空: {payload.get("code")} {payload.get("message", "")} '
                    f'resp={resp.text[:120]}'
                )
            s = pd.Series(
                [float(r['peg']) if r.get('peg') is not None else np.nan for r in rows],
                index=pd.DatetimeIndex([pd.Timestamp(r['tradeDate']) for r in rows]),
                name='pe',
            )
            s = s[~s.index.duplicated(keep='last')].sort_index()
            return s
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f'无法获取{index_code}估值: {e}') from e
            time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f'无法获取{index_code}估值')


def fetch_cn10y(start='2013-01-01', end=None):
    """中国人民银行口径的中国10年期国债到期收益率(日度, 来源于AkShare/新浪财经)。"""
    df = ak.bond_zh_us_rate(start_date=pd.Timestamp(start).strftime('%Y%m%d'))
    s = df.set_index(pd.to_datetime(df['日期']))['中国国债收益率10年'].dropna().astype(float)
    s = s[~s.index.duplicated(keep='last')].sort_index()
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    s.index.name = 'date'
    s.name = 'cn10y'
    return s


def derive_trailing_div_yield(price, total_return, window=252):
    """由价格指数与全收益指数推导近N日(默认一年)实际股息率。

    全收益/价格之比的变化率就是指数成分股分红带来的累计收益,
    因此 (TR_t/P_t)/(TR_{t-N}/P_{t-N}) - 1 即近N个交易日股息率。
    980081自身官方可获得的真实股息率序列。
    """
    p = price.reindex(total_return.index).ffill().bfill()
    ratio = total_return / p
    return (ratio / ratio.shift(window) - 1) * 100

"""生成线上 dashboard 的 data.json。

策略: 估值分位 + 波动率目标 + 股息率-央行利率差。
收益口径: 持仓按480081全收益(159263含股息再投资代理), 空仓按511260十年国债ETF。
费率: 佣金万0.5, 单笔最低0.5元, 调仓容忍度避免微小调仓。
"""

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .factor_strategy import (
    PARAMS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    build_factors,
    metrics,
    run_factor_strategy,
    weighted_backtest,
)
from .fetch import (
    fetch_480081_daily,
    fetch_511260_close,
    fetch_cn10y,
    fetch_csindex_daily_pe,
    fetch_980081_daily,
)

DISPLAY_START = '2018-01-01'
FULL_START = '2013-01-01'

# 真实交易费率: 佣金万0.5(0.005%), 单笔最低0.5元, ETF免印花税
FEE_RATE = 0.00005
FEE_MIN = 0.5

# 兼容旧MA250调优工具
def compute_stats(df2, eq2, sells, hold_series=None):
    """旧MA250策略统计(保留给 tune.py/full_tune.py 使用)。"""
    first_equity = eq2['equity'].iloc[0]
    final = eq2['equity'].iloc[-1]
    years = (eq2.index[-1] - eq2.index[0]).days / 365.25
    ann = ((final / first_equity) ** (1 / years) - 1) * 100
    dd_series = (eq2['equity'] - eq2['equity'].cummax()) / eq2['equity'].cummax() * 100
    max_dd = dd_series.min()
    dr = eq2['equity'].pct_change().dropna()
    sharpe = (dr.mean() * 252 - 0.02) / (dr.std() * np.sqrt(252))
    n_trades = len(sells)
    win_rate = (sells['pnl_pct'] > 0).mean() * 100 if n_trades > 0 else 0.0
    avg_pnl = sells['pnl_pct'].mean() if n_trades > 0 else 0.0
    bh_close = hold_series if hold_series is not None else df2['close']
    bh_ret = (bh_close.iloc[-1] / bh_close.iloc[0] - 1) * 100
    bh_ann = ((1 + bh_ret / 100) ** (1 / years) - 1) * 100
    stats = {
        'annualized_pct': round(float(ann), 1),
        'max_drawdown_pct': round(float(max_dd), 1),
        'sharpe': round(float(sharpe), 2),
        'win_rate_pct': round(float(win_rate), 0),
        'trade_count': int(n_trades),
        'avg_win_pct': round(float(avg_pnl), 1),
        'excess_annualized_pct': round(float(ann - bh_ann), 1),
        'buy_hold_annualized_pct': round(float(bh_ann), 1),
    }
    return stats, dd_series


def compute_holding_pct(buys, sells, df2):
    """旧MA250策略持仓占比(保留给 tune.py/full_tune.py 使用)。"""
    hold_days = (df2.index[-1] - df2.index[0]).days
    if hold_days <= 0:
        return 0.0
    in_pos_days = 0
    for j in range(min(len(buys), len(sells))):
        in_pos_days += (sells.iloc[j]['date'] - buys.iloc[j]['date']).days
    return round(in_pos_days / hold_days * 100, 0)


def _safe_list(series, ndigits=None):
    s = series.round(ndigits) if ndigits is not None else series
    return [None if pd.isna(v) else float(v) for v in s]


def build_current_status(sig):
    latest = sig.iloc[-1]
    holding = bool(latest['regime'] == 1)
    weight = float(latest['weight'])
    value_score = float(latest['value_score'])
    spread = float(latest['spread'])

    if holding:
        if weight >= 0.9:
            signal_text, signal_level = '持有价值100 | 估值仍便宜, 目标仓位高', 'buy'
        elif weight >= 0.5:
            signal_text, signal_level = '持有价值100 | 波动率目标降仓中', 'neutral'
        else:
            signal_text, signal_level = '持有价值100 | 波动率高, 仓位已压缩', 'watch'
    else:
        if value_score >= PARAMS['value_enter'] and spread >= PARAMS['spread_enter']:
            signal_text, signal_level = '空仓国债 | 即将满足买入条件', 'watch'
        else:
            signal_text, signal_level = '空仓国债 | 估值偏贵, 等待便宜', 'neutral'

    return {
        'holding': holding,
        'position_asset': '980081(480081全收益)' if holding else '511260',
        'date': sig.index[-1].strftime('%Y-%m-%d'),
        'price_raw': round(float(latest['close']), 3),
        'weight': round(weight, 3),
        'regime': int(latest['regime']),
        'value_score': round(value_score, 3),
        'pe_pct': round(float(latest['pe_pct']), 3),
        'dy_pct': round(float(latest['dy_pct']), 3),
        'div_yield': round(float(latest['div_yield']), 2),
        'cn10y': round(float(latest['cn10y']), 2),
        'spread': round(spread, 2),
        'realized_vol': round(float(latest['realized_vol']), 1),
        'target_weight': round(float(latest['target_weight']), 3),
        'signal_text': signal_text,
        'signal_level': signal_level,
    }


def build_regime_events(sig):
    events = []
    prev = 0
    for d, row in sig.iterrows():
        g = int(row['regime'])
        if g != prev:
            events.append({
                'date': d.strftime('%Y-%m-%d'),
                'action': '进入持仓' if g == 1 else '退出持仓',
                'value_score': round(float(row['value_score']), 3),
                'spread': round(float(row['spread']), 2),
            })
            prev = g
    return events


def build_rebalances(reb):
    if len(reb) == 0:
        return []
    return [
        {
            'date': d.strftime('%Y-%m-%d'),
            'weight_from': float(wf),
            'weight_to': float(wt),
            'fee': round(float(f), 2),
        }
        for d, wf, wt, f in zip(reb['date'], reb['weight_from'], reb['weight_to'], reb['fee'])
    ]


def build_series(sig, eq, dd_series, hold_price):
    bh = 100000 * hold_price / hold_price.iloc[0]
    eq_aligned = eq.reindex(sig.index)
    dd_aligned = dd_series.reindex(sig.index)
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in sig.index],
        'close': _safe_list(sig['close'], 3),
        'weight': _safe_list(sig['weight'], 3),
        'target_weight': _safe_list(sig['target_weight'], 3),
        'regime': [int(v) for v in sig['regime']],
        'value_score': _safe_list(sig['value_score'], 3),
        'pe_pct': _safe_list(sig['pe_pct'], 3),
        'dy_pct': _safe_list(sig['dy_pct'], 3),
        'div_yield': _safe_list(sig['div_yield'], 2),
        'cn10y': _safe_list(sig['cn10y'], 2),
        'spread': _safe_list(sig['spread'], 2),
        'realized_vol': _safe_list(sig['realized_vol'], 1),
        'equity_strategy': _safe_list(eq_aligned, 0),
        'equity_buyhold': _safe_list(bh, 0),
        'drawdown_pct': _safe_list(dd_aligned, 2),
    }


def export(output_path):
    raw = fetch_980081_daily()
    if len(raw) < 400:
        raise ValueError(f'980081数据只拉到{len(raw)}条, 远少于预期')
    hold_raw = fetch_480081_daily()
    if len(hold_raw) < 400:
        raise ValueError(f'480081全收益数据只拉到{len(hold_raw)}条, 远少于预期')
    pe = fetch_csindex_daily_pe()
    cn10y = fetch_cn10y()
    try:
        idle_price = fetch_511260_close(count=2500)
    except Exception as e:
        print(f'511260获取失败, 继续但不计空仓收益: {e}')
        idle_price = None

    hold_price = hold_raw['close'].reindex(raw.index).ffill().bfill()
    factors = build_factors(raw['close'], hold_price, pe, cn10y)
    sig = run_factor_strategy(factors)
    eq, reb = weighted_backtest(sig, idle_price=idle_price, fee_rate=FEE_RATE, min_fee=FEE_MIN)

    sig2 = sig[sig.index >= DISPLAY_START].copy()
    eq2 = eq[eq.index >= DISPLAY_START].copy()
    hold2 = hold_price[sig2.index]
    dd_series = (eq2 - eq2.cummax()) / eq2.cummax() * 100

    full = metrics(sig, eq, FULL_START, None, buy_hold=hold_price.reindex(sig.index))
    train = metrics(sig, eq, TRAIN_START, TRAIN_END, buy_hold=hold_price.reindex(sig.index))
    test = metrics(sig, eq, TEST_START, TEST_END, buy_hold=hold_price.reindex(sig.index))
    if train is None or test is None:
        raise ValueError('训练/测试区间无足够数据, 检查数据范围')

    beijing_now = datetime.now(timezone(timedelta(hours=8)))
    payload = {
        'meta': {
            'strategy': '估值分位+波动率目标+股息率-央行利率差',
            'fee_rate': FEE_RATE,
            'min_fee': FEE_MIN,
            'updated_at': beijing_now.isoformat(),
            'as_of_date': sig2.index[-1].strftime('%Y-%m-%d'),
            'return_basis': '480081全收益口径(159263含股息再投资代理), 空仓511260',
            'valuation_basis': '980081自身股息率 + 中证红利PE代理(000922)',
            'train': train,
            'test': test,
            'full': full,
        },
        'current_status': build_current_status(sig),
        'series': build_series(sig2, eq2, dd_series, hold2),
        'regime_events': build_regime_events(sig2),
        'rebalances': build_rebalances(
            reb[reb['date'] >= pd.Timestamp(DISPLAY_START)] if len(reb) else reb
        ),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    return payload


if __name__ == '__main__':
    site_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'site')
    os.makedirs(site_dir, exist_ok=True)
    export(os.path.join(site_dir, 'data.json'))

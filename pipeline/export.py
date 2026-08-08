import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .backtest import backtest
from .fetch import fetch_511260_close, fetch_980081_daily
from .indicators import add_indicators
from .strategy import PARAMS, run_strategy

SELL_TIER_ORDER = ['硬上限', 'RSI确认', '偏离回落', 'RSI下穿']
DISPLAY_START = '2013-01-01'

# 真实交易费率: 佣金万0.5(0.005%), 单笔最低0.5元, ETF免印花税
FEE_RATE = 0.00005
FEE_MIN = 0.5


def _safe_list(series, ndigits=None):
    s = series.round(ndigits) if ndigits is not None else series
    return [None if pd.isna(v) else float(v) for v in s]


def compute_stats(df2, eq2, sells):
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
    bh_ret = (df2.iloc[-1]['close'] / df2.iloc[0]['close'] - 1) * 100
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
    hold_days = (df2.index[-1] - df2.index[0]).days
    if hold_days <= 0:
        return 0.0
    in_pos_days = 0
    for j in range(min(len(buys), len(sells))):
        in_pos_days += (sells.iloc[j]['date'] - buys.iloc[j]['date']).days
    return round(in_pos_days / hold_days * 100, 0)


def build_current_status(df2, latest_position):
    latest = df2.iloc[-1]
    dev = latest['deviation']
    rsi = latest['rsi']
    ma250_raw = latest['close_raw'] / (1 + dev / 100)
    sell_soft = ma250_raw * 1.07
    sell_hard = ma250_raw * 1.14
    buy_cap = ma250_raw * 1.04

    holding = bool(latest_position == 1)
    if holding:
        if dev >= 7.0 and rsi >= 75:
            signal_text, signal_level = '卖出信号触发!', 'sell'
        elif dev >= 7.0:
            signal_text, signal_level = '持仓价值100 | 卖出监控中', 'watch'
        else:
            signal_text, signal_level = '持仓价值100 | 持有等待', 'neutral'
    else:
        if dev < -2:
            signal_text, signal_level = '空仓国债 | 极端买入触发!', 'buy'
        elif 0 <= dev <= 4:
            signal_text, signal_level = '空仓国债 | 接近买入区', 'watch'
        else:
            signal_text, signal_level = '空仓国债 | 等待回落', 'neutral'

    return {
        'holding': holding,
        'position_asset': '980081' if holding else '511260',
        'date': df2.index[-1].strftime('%Y-%m-%d'),
        'price_raw': round(float(latest['close_raw']), 3),
        'ma250': round(float(latest['ma250']), 3),
        'deviation_pct': round(float(dev), 1),
        'rsi14': round(float(rsi), 0),
        'rsi6': round(float(latest['rsi6']), 0),
        'ma250_slope_pct': round(float(latest['ma250_slope']), 2),
        'sell_trigger_price_soft': round(float(sell_soft), 3),
        'sell_trigger_price_hard': round(float(sell_hard), 3),
        'buy_trigger_price_cap': round(float(buy_cap), 3),
        'signal_text': signal_text,
        'signal_level': signal_level,
    }


def build_trades(buys, sells, df2):
    trades = []
    for j in range(min(len(buys), len(sells))):
        b, s = buys.iloc[j], sells.iloc[j]
        trades.append({
            'seq': j + 1,
            'buy_date': b['date'].strftime('%Y-%m-%d'),
            'sell_date': s['date'].strftime('%Y-%m-%d'),
            'buy_price': round(float(b['price']), 3),
            'sell_price': round(float(s['price']), 3),
            'buy_price_raw': round(float(b['price_raw']), 3),
            'sell_price_raw': round(float(s['price_raw']), 3),
            'pnl_pct': round(float(s['pnl_pct']), 1),
            'hold_days': int(s['hold_days']),
            'sell_reason': s['reason'],
            'open': False,
        })
    if len(buys) > len(sells):
        b = buys.iloc[len(sells)]
        cur_price = df2.iloc[-1]['close']
        cur_pnl = (cur_price / b['price'] - 1) * 100
        trades.append({
            'seq': len(sells) + 1,
            'buy_date': b['date'].strftime('%Y-%m-%d'),
            'sell_date': None,
            'buy_price': round(float(b['price']), 3),
            'sell_price': round(float(cur_price), 3),
            'buy_price_raw': round(float(b['price_raw']), 3),
            'sell_price_raw': round(float(df2.iloc[-1]['close_raw']), 3),
            'pnl_pct': round(float(cur_pnl), 1),
            'hold_days': int((df2.index[-1] - b['date']).days),
            'sell_reason': '未平仓（持有中）',
            'open': True,
        })
    return trades


def build_sell_reason_breakdown(sells):
    reason_map = {}
    reason_pnl = {}
    for _, s in sells.iterrows():
        for key in SELL_TIER_ORDER:
            if key in s['reason']:
                reason_map[key] = reason_map.get(key, 0) + 1
                reason_pnl.setdefault(key, []).append(s['pnl_pct'])
                break
    total = sum(reason_map.values())
    return [
        {
            'reason': k,
            'count': v,
            'avg_pnl_pct': round(float(np.mean(reason_pnl[k])), 1),
            'pct_of_total': round(v / total * 100, 0) if total else 0.0,
        }
        for k, v in reason_map.items()
    ]


def build_series(df2, eq2, dd_series):
    bh = 100000 * df2['close'] / df2.iloc[0]['close']
    eq_aligned = eq2['equity'].reindex(df2.index)
    dd_aligned = dd_series.reindex(df2.index)
    return {
        'dates': [d.strftime('%Y-%m-%d') for d in df2.index],
        'close': _safe_list(df2['close'], 3),
        'close_raw': _safe_list(df2['close_raw'], 3),
        'ma10': _safe_list(df2['ma10'], 3),
        'ma20': _safe_list(df2['ma20'], 3),
        'ma60': _safe_list(df2['ma60'], 3),
        'ma250': _safe_list(df2['ma250'], 3),
        'deviation': _safe_list(df2['deviation'], 2),
        'rsi14': _safe_list(df2['rsi'], 1),
        'rsi6': _safe_list(df2['rsi6'], 1),
        'macd': _safe_list(df2['macd'], 4),
        'macd_signal': _safe_list(df2['macd_signal'], 4),
        'macd_hist': _safe_list(df2['macd_hist'], 4),
        'equity_strategy': _safe_list(eq_aligned, 0),
        'equity_buyhold': _safe_list(bh, 0),
        'drawdown_pct': _safe_list(dd_aligned, 2),
    }


def export(output_path, count_511260=2500):
    raw = fetch_980081_daily()
    if len(raw) < 400:
        raise ValueError(
            f'980081数据只拉到{len(raw)}条,远少于预期,可能是接口返回被截断'
        )
    df = add_indicators(raw)
    df_sig = run_strategy(df, PARAMS)

    idle_price = None
    try:
        idle_price = fetch_511260_close(count=count_511260)
    except Exception as e:
        print(f'511260获取失败,继续但不计空仓收益: {e}')

    eq, tr = backtest(df_sig, idle_price=idle_price, comm=FEE_RATE, min_comm=FEE_MIN, lot_size=1)

    df2 = df_sig[df_sig.index >= DISPLAY_START].copy()
    eq2 = eq[eq.index >= DISPLAY_START].copy()
    buys = tr[(tr['action'] == 'BUY') & (tr['date'] >= DISPLAY_START)].reset_index(drop=True)
    sells = tr[(tr['action'] == 'SELL') & (tr['date'] >= DISPLAY_START)].reset_index(drop=True)

    if len(sells) == 0:
        raise ValueError('回测区间内没有任何已平仓交易，无法计算统计指标——检查策略参数或数据是否异常')

    stats, dd_series = compute_stats(df2, eq2, sells)
    stats['holding_pct'] = compute_holding_pct(buys, sells, df2)

    beijing_now = datetime.now(timezone(timedelta(hours=8)))

    payload = {
        'meta': {
            **stats,
            'fee_rate': FEE_RATE,
            'min_fee': FEE_MIN,
            'updated_at': beijing_now.isoformat(),
            'as_of_date': df2.index[-1].strftime('%Y-%m-%d'),
        },
        'current_status': build_current_status(df2, df2.iloc[-1]['position']),
        'series': build_series(df2, eq2, dd_series),
        'trades': build_trades(buys, sells, df2),
        'sell_reason_breakdown': build_sell_reason_breakdown(sells),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    return payload


if __name__ == '__main__':
    site_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'site')
    os.makedirs(site_dir, exist_ok=True)
    export(os.path.join(site_dir, 'data.json'))

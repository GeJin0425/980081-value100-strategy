"""国证价值100(980081) 估值分位 + 波动率目标 + 股息率-央行利率差策略。

策略思路
--------
1. 估值分位: 980081自身近12个月股息率分位 + 中证红利PE分位(免费代理)的
   合成便宜度得分 value_score ∈ [0,1], 越接近1越便宜。
2. 股息率-央行利率差: spread = 980081股息率 - 中国10年期国债收益率。
3. 波动率目标: 目标组合波动率 target_vol, 用60日已实现年化波动率缩放仓位。

状态机(滞后, 减少来回):
  空仓 -> 满仓候选: value_score >= value_enter 且 spread >= spread_enter
  满仓 -> 空仓:     value_score <= value_exit 或 spread <= spread_exit
目标仓位 = 状态 * min(1, target_vol / 已实现波动率), 空仓部分配置511260。
调仓容忍度 rebalance_tol 避免微小调仓产生过多佣金。
"""

import numpy as np
import pandas as pd

PARAMS = dict(
    value_enter=0.25,
    value_exit=0.15,
    spread_enter=0.30,
    spread_exit=-0.50,
    target_vol=16.0,
    vol_window=60,
    pct_window=1260,
    pct_min=252,
    rebalance_tol=0.03,
)

TRAIN_START = '2018-01-01'
TRAIN_END = '2022-12-31'
TEST_START = '2023-01-01'
TEST_END = '2026-12-31'


def _rolling_percentile(s, window, min_periods):
    """滚动窗口内, 当前值 >= 历史多少比例(0~1)。只用过去数据, 无未来函数。"""
    return s.rolling(window, min_periods=min_periods).apply(
        lambda x: float((x[-1] >= x).mean()), raw=True
    )


def build_factors(price, total_return, pe_proxy, cn10y, params=PARAMS):
    """对齐到980081交易日, 生成策略所需全部因子。"""
    df = pd.DataFrame({'close': price, 'hold': total_return})
    df['hold'] = df['hold'].ffill()
    df['pe_proxy'] = pe_proxy.reindex(df.index).ffill()
    df['cn10y'] = cn10y.reindex(df.index).ffill()

    ratio = df['hold'] / df['close']
    df['div_yield'] = (ratio / ratio.shift(252) - 1) * 100
    df['spread'] = df['div_yield'] - df['cn10y']

    df['pe_pct'] = _rolling_percentile(
        df['pe_proxy'], params['pct_window'], params['pct_min']
    )
    df['dy_pct'] = _rolling_percentile(
        df['div_yield'], params['pct_window'], params['pct_min']
    )
    df['value_score'] = 0.5 * ((1 - df['pe_pct']) + df['dy_pct'])

    ret = df['close'].pct_change()
    df['realized_vol'] = (
        ret.rolling(params['vol_window'], min_periods=max(20, params['vol_window'] // 3))
        .std(ddof=1)
        * np.sqrt(252)
        * 100
    )
    df['vol_scale'] = np.clip(params['target_vol'] / df['realized_vol'], 0, 1)
    return df


def run_factor_strategy(df, params=PARAMS):
    """生成目标仓位与状态。target_weight 是理论仓位, weight 是容忍度过滤后实际仓位。"""
    out = df.copy()
    vs = out['value_score']
    sp = out['spread']
    state = np.zeros(len(out), dtype=int)
    cur = 0
    for i in range(len(out)):
        v = vs.iloc[i]
        s = sp.iloc[i]
        if np.isnan(v) or np.isnan(s):
            state[i] = cur
            continue
        if cur == 0 and v >= params['value_enter'] and s >= params['spread_enter']:
            cur = 1
        elif cur == 1 and (v <= params['value_exit'] or s <= params['spread_exit']):
            cur = 0
        state[i] = cur
    out['regime'] = state
    out['target_weight'] = (out['regime'] * out['vol_scale']).fillna(0.0)

    w = np.zeros(len(out))
    prev = 0.0
    tol = params['rebalance_tol']
    for i, t in enumerate(out['target_weight'].to_numpy()):
        t = 0.0 if np.isnan(t) else float(t)
        if abs(t - prev) >= tol:
            prev = t
        w[i] = prev
    out['weight'] = w
    return out


def weighted_backtest(
    df,
    idle_price=None,
    initial=100000,
    fee_rate=0.00005,
    min_fee=0.5,
):
    """连续仓位回测: 按日复利, 持仓部分按480081全收益, 空仓部分按511260。

    调仓成本按成交额 * 万0.5, 单笔最低0.5元。返回权益曲线与调仓记录。
    """
    hold_ret = df['hold'].pct_change().fillna(0.0)
    if idle_price is None:
        idle_ret = pd.Series(0.0, index=df.index)
    else:
        idle = idle_price.reindex(df.index).ffill()
        idle_ret = idle.pct_change().fillna(0.0)

    wm = df['weight'].shift(1).fillna(0.0).to_numpy()
    hr = hold_ret.to_numpy()
    ir = idle_ret.to_numpy()
    w = df['weight'].to_numpy()

    equity = np.empty(len(df))
    equity[0] = initial
    rebalances = []
    prev_w = 0.0
    for i in range(len(df)):
        if i > 0:
            r = wm[i] * hr[i] + (1 - wm[i]) * ir[i]
            equity[i] = equity[i - 1] * (1 + r)
        else:
            equity[i] = initial
        dw = abs(w[i] - prev_w)
        if dw > 1e-9:
            fee = max(equity[i] * dw * fee_rate, min_fee)
            equity[i] -= fee
            rebalances.append({
                'date': df.index[i],
                'weight_from': round(float(prev_w), 4),
                'weight_to': round(float(w[i]), 4),
                'fee': round(float(fee), 2),
            })
            prev_w = w[i]
    eq = pd.Series(equity, index=df.index, name='equity')
    reb = pd.DataFrame(rebalances)
    return eq, reb


def metrics(df, eq, start=None, end=None, buy_hold=None):
    """区间统计: 年化/最大回撤/夏普/平均仓位/调仓次数/超额收益。"""
    m = pd.Series(True, index=df.index)
    if start is not None:
        m &= df.index >= pd.Timestamp(start)
    if end is not None:
        m &= df.index <= pd.Timestamp(end)
    d2 = df[m]
    e2 = eq[m]
    if len(e2) < 60:
        return None
    years = (e2.index[-1] - e2.index[0]).days / 365.25
    ann = (e2.iloc[-1] / e2.iloc[0]) ** (1 / years) * 100 - 100
    dd_series = (e2 - e2.cummax()) / e2.cummax() * 100
    max_dd = dd_series.min()
    dr = e2.pct_change().dropna()
    sharpe = (dr.mean() * 252 - 0.02) / (dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    avg_weight = d2['weight'].mean() * 100
    n_rebal = int((d2['weight'].diff().abs() > 1e-9).sum())
    bh = buy_hold[m] if buy_hold is not None else d2['hold']
    bh_ann = ((bh.iloc[-1] / bh.iloc[0]) ** (1 / years) * 100 - 100) if len(bh) > 1 else 0.0
    return {
        'start': d2.index[0].strftime('%Y-%m-%d'),
        'end': d2.index[-1].strftime('%Y-%m-%d'),
        'annualized_pct': round(float(ann), 2),
        'max_drawdown_pct': round(float(max_dd), 2),
        'sharpe': round(float(sharpe), 2),
        'avg_weight_pct': round(float(avg_weight), 1),
        'rebalance_count': n_rebal,
        'excess_annualized_pct': round(float(ann - bh_ann), 2),
        'buy_hold_annualized_pct': round(float(bh_ann), 2),
    }

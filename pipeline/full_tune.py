"""980081 MA250策略全参数随机搜索 + 局部精修 + 样本内外验证。

用法:
    python -m pipeline.full_tune --n 800 --workers 8 --seed 42
"""

import argparse
from functools import partial
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .backtest import backtest
from .export import FEE_MIN, FEE_RATE, compute_holding_pct, compute_stats
from .fetch import fetch_511260_close, fetch_980081_daily
from .indicators import add_indicators
from .strategy import PARAMS, run_strategy

DISPLAY_START = '2013-01-01'
TRAIN_START = '2013-01-01'
TRAIN_END = '2021-12-31'
TEST_START = '2022-01-01'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tuning_results')

RANGES = {
    'b1': [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5],
    'b2': [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    'b2r': [30, 35, 40, 45, 50, 55, 60, 65],
    'b3lo': [0.0],
    'b3hi': [3, 4, 5, 6, 7, 8],
    's1': [7, 8, 9, 10, 11, 12, 13, 14, 16],
    's2': [3, 4, 5, 6, 7, 8],
    's2r': [60, 65, 68, 70, 72, 75, 80],
    's3pk': [2, 3, 4, 5, 6],
    's3dp': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    's4pr': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
    's4r': [55, 60, 65, 70, 75],
    'cooldown': [3, 5, 8, 10, 15, 20],
}


def random_params(rng):
    p = dict(PARAMS)
    for k, vals in RANGES.items():
        p[k] = rng.choice(vals)
    if p['b1'] >= p['b2']:
        p['b1'] = rng.choice([v for v in RANGES['b1'] if v < p['b2']] or [-2.0])
    if p['s1'] <= p['s2']:
        p['s1'] = p['s2'] + rng.choice([1, 2, 3, 4, 5, 6])
    p['b3lo'] = 0.0
    return p


_DF = None
_IDLE = None


def _init_worker(df, idle):
    global _DF, _IDLE
    _DF = df
    _IDLE = idle


def _metrics(p, start=DISPLAY_START, end=None):
    ds = run_strategy(_DF, p)
    eq, tr = backtest(ds, idle_price=_IDLE, comm=FEE_RATE, min_comm=FEE_MIN, lot_size=1)
    end = _DF.index[-1] if end is None else pd.Timestamp(end)
    d2 = ds[(ds.index >= pd.Timestamp(start)) & (ds.index <= end)]
    e2 = eq[(eq.index >= pd.Timestamp(start)) & (eq.index <= end)]
    buys = tr[(tr['action'] == 'BUY') & (tr['date'] >= pd.Timestamp(start)) & (tr['date'] <= end)]
    sells = tr[(tr['action'] == 'SELL') & (tr['date'] >= pd.Timestamp(start)) & (tr['date'] <= end)]
    if len(sells) == 0:
        return None
    st, _ = compute_stats(d2, e2, sells.reset_index(drop=True))
    st['holding_pct'] = compute_holding_pct(buys.reset_index(drop=True), sells.reset_index(drop=True), d2)
    st['trade_count'] = len(sells)
    return st


def eval_full(p):
    return _metrics(p)


def score(st):
    if not st:
        return -1e9
    dd_penalty = max(0.0, -st['max_drawdown_pct'] - 20.0) * 0.25
    trade_penalty = max(0.0, 20.0 - st['trade_count']) * 0.05
    return st['annualized_pct'] - dd_penalty - trade_penalty


def short_params(p):
    return ','.join(f'{k}={v:g}' for k, v in p.items())


def parse_params_str(s):
    out = {}
    for kv in s.split(','):
        k, v = kv.split('=')
        out[k] = float(v)
    return out


def perturb(p, rng, step=1):
    q = dict(p)
    for k in RANGES:
        if k == 'b3lo':
            continue
        if rng.random() < 0.45:
            vals = RANGES[k]
            if k == 'cooldown':
                q[k] = max(1, q[k] + rng.choice([-3, -2, -1, 1, 2, 3]))
            else:
                idx = vals.index(q[k]) if q[k] in vals else None
                if idx is None:
                    q[k] = rng.choice(vals)
                else:
                    nxt = idx + rng.choice([-2, -1, 1, 2])
                    q[k] = vals[max(0, min(len(vals) - 1, nxt))]
    if q['b1'] >= q['b2']:
        q['b1'] = min(q['b1'], q['b2'] - 0.5)
    if q['s1'] <= q['s2']:
        q['s1'] = q['s2'] + 1
    q['b3lo'] = 0.0
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--refine', type=int, default=200, help='每个Top候选的局部扰动次数')
    ap.add_argument('--train-end', default=TRAIN_END, help='训练集截止日期,默认2021-12-31')
    ap.add_argument('--test-start', default=TEST_START, help='测试集起始日期,默认2022-01-01')
    args = ap.parse_args()

    raw = fetch_980081_daily()
    df = add_indicators(raw)
    try:
        idle = fetch_511260_close(count=2500)
    except Exception as e:
        print(f'[warn] 511260获取失败: {e}')
        idle = pd.Series(dtype=float)
    os.makedirs(OUT_DIR, exist_ok=True)
    global _DF, _IDLE
    _DF = df
    _IDLE = idle

    rng = random.Random(args.seed)
    combos = [random_params(rng) for _ in range(args.n)]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(df, idle)) as pool:
        results = list(pool.map(eval_full, combos, chunksize=8))
    print(f'[random] {args.n} evals done in {time.time() - t0:.1f}s')

    rows = []
    for p, r in zip(combos, results):
        if r is None:
            continue
        r = dict(r)
        r['score'] = round(score(r), 2)
        r['params'] = short_params(p)
        r['full_ann'] = r['annualized_pct']
        rows.append(r)

    rows.sort(key=lambda r: -r['score'])
    best = rows[:15]
    print('\n===== RANDOM TOP 15 =====')
    for i, r in enumerate(best, 1):
        print(f'{i:2d} ann={r["annualized_pct"]:5.1f} dd={r["max_drawdown_pct"]:5.1f} '
              f'sharpe={r["sharpe"]:.2f} win={r["win_rate_pct"]:3.0f} n={r["trade_count"]:3d} '
              f'hold={r["holding_pct"]:3.0f} score={r["score"]:.1f}')
        print(f'    {r["params"]}')

    # 用训练集(2013-2021)从全样本Top里重新排名, 避免只看全样本选参
    train_ranked = []
    for r in rows[:60]:
        p = parse_params_str(r['params'])
        m = _metrics(p, start=TRAIN_START, end=args.train_end)
        if not m:
            continue
        m = dict(m)
        m['score'] = round(score(m), 2)
        m['params'] = r['params']
        train_ranked.append(m)
    train_ranked.sort(key=lambda r: -r['score'])
    print('\n===== TRAIN 2013-2021 TOP 10 =====')
    for i, r in enumerate(train_ranked[:10], 1):
        print(f'{i:2d} ann={r["annualized_pct"]:5.1f} dd={r["max_drawdown_pct"]:5.1f} '
              f'sharpe={r["sharpe"]:.2f} win={r["win_rate_pct"]:3.0f} n={r["trade_count"]:3d} '
              f'hold={r["holding_pct"]:3.0f} score={r["score"]:.1f}')
        print(f'    {r["params"]}')

    # 对训练集Top候选做局部扰动精修, 精修也按训练集打分
    refined = []
    top_params = [parse_params_str(r['params']) for r in train_ranked[:10]]
    cands = []
    for p in top_params:
        cands.append(p)
        for _ in range(args.refine):
            cands.append(perturb(p, rng))
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(df, idle)) as pool:
        eval_train = partial(_metrics, start=TRAIN_START, end=args.train_end)
        refined_results = list(pool.map(eval_train, cands, chunksize=16))
    print(f'[refine] {len(cands)} evals done in {time.time() - t1:.1f}s')
    for p, r in zip(cands, refined_results):
        if r is None:
            continue
        r = dict(r)
        r['score'] = round(score(r), 2)
        r['params'] = short_params(p)
        r['train_ann'] = r['annualized_pct']
        refined.append(r)
    refined.sort(key=lambda r: -r['score'])

    final = refined[:20]
    print('\n===== REFINED TOP 20 (按训练集2013-2021打分) =====')
    for i, r in enumerate(final, 1):
        print(f'{i:2d} ann={r["annualized_pct"]:5.1f} dd={r["max_drawdown_pct"]:5.1f} '
              f'sharpe={r["sharpe"]:.2f} win={r["win_rate_pct"]:3.0f} n={r["trade_count"]:3d} '
              f'hold={r["holding_pct"]:3.0f} score={r["score"]:.1f}')
        print(f'    {r["params"]}')

    # 样本内外验证
    print(f'\n===== VALIDATION (full / train 2013-{args.train_end[:4]} / test {args.test_start[:4]}-2026) =====')
    for i, r in enumerate(final[:10], 1):
        p = parse_params_str(r['params'])
        full = _metrics(p)
        tr = _metrics(p, start=TRAIN_START, end=args.train_end)
        te = _metrics(p, start=args.test_start)
        print(f'\n{i:2d} {r["params"]}')
        for label, m in [('full', full), ('train', tr), ('test', te)]:
            if not m:
                print(f'  {label}: no closed trades')
                continue
            print(f'  {label}: ann={m["annualized_pct"]:.1f} dd={m["max_drawdown_pct"]:.1f} '
                  f'sharpe={m["sharpe"]:.2f} win={m["win_rate_pct"]:.0f} n={m["trade_count"]} '
                  f'hold={m["holding_pct"]:.0f} bh={m["buy_hold_annualized_pct"]:.1f}')

    # 汇总保存: 训练集指标 + 测试集指标
    rows_out = []
    for r in final:
        p = parse_params_str(r['params'])
        full = _metrics(p)
        tr = _metrics(p, start=TRAIN_START, end=args.train_end)
        te = _metrics(p, start=args.test_start)
        row = {'params': r['params'], 'score': r['score']}
        for prefix, m in [('full', full), ('train', tr), ('test', te)]:
            if m:
                row[f'{prefix}_ann'] = m['annualized_pct']
                row[f'{prefix}_dd'] = m['max_drawdown_pct']
                row[f'{prefix}_sharpe'] = m['sharpe']
                row[f'{prefix}_win'] = m['win_rate_pct']
                row[f'{prefix}_n'] = m['trade_count']
                row[f'{prefix}_hold'] = m['holding_pct']
        rows_out.append(row)
    out = pd.DataFrame(rows_out)
    out.to_csv(os.path.join(OUT_DIR, 'full_tune_refined.csv'), index=False)
    print(f'\n[saved] {os.path.join(OUT_DIR, "full_tune_refined.csv")}')


if __name__ == '__main__':
    main()

"""
980081 价值100 MA250策略参数网格搜索 / OFAT扫描

用法:
    python -m pipeline.tune fetch               # 抓数据缓存为 CSV
    python -m pipeline.tune sweep --params s1   # 单个参数扫描 (逗号分隔可多参数)
    python -m pipeline.tune refine              # 围绕当前最优做联合精修
    python -m pipeline.tune validate --best "..."  # 样本内/外验证一组参数

数据只抓一次缓存到 tuning_results/data_cache.csv, 后续扫描离线进行。
指标口径与 export.py 完全一致 (2013-01-01 起, 含511260空仓收益, 佣金万0.5+单笔最低0.5元)。
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from .backtest import backtest
from .export import FEE_MIN, FEE_RATE, compute_holding_pct, compute_stats
from .fetch import fetch_511260_close, fetch_980081_daily
from .indicators import add_indicators
from .strategy import PARAMS, run_strategy

DISPLAY_START = '2014-01-01'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tuning_results')
CACHE = os.path.join(OUT_DIR, 'data_cache.csv')
IDLE_CACHE = os.path.join(OUT_DIR, 'idle_cache.csv')

# 线上发布版参数(基线): tune/params-optimization 采纳候选A前的 main 参数
BASE_PARAMS = dict(
    b1=-2.5, b2=0.0, b2r=40.0, b3lo=0.0, b3hi=5.0,
    s1=12.0, s2=4.0, s2r=70.0, s3pk=4.0, s3dp=1.5,
    s4pr=3.0, s4r=65.0, cooldown=10,
)


def load_data(refresh=False):
    if not refresh and os.path.exists(CACHE) and os.path.exists(IDLE_CACHE):
        df = pd.read_csv(CACHE, parse_dates=['date']).set_index('date')
        idle = pd.read_csv(IDLE_CACHE, parse_dates=['date']).set_index('date')['close']
        print(f'[load] 从缓存读取 {len(df)} 行 ({df.index[0].date()} -> {df.index[-1].date()})')
        return df, idle

    os.makedirs(OUT_DIR, exist_ok=True)
    raw = fetch_980081_daily()
    raw.index.name = 'date'
    df = add_indicators(raw)
    try:
        idle = fetch_511260_close(count=2500)
    except Exception as e:
        print(f'[warn] 511260 抓取失败: {e}')
        idle = pd.Series(dtype=float)
    idle.index.name = 'date'
    df.reset_index().to_csv(CACHE, index=False)
    idle.reset_index().to_csv(IDLE_CACHE, index=False)
    print(f'[fetch] 已缓存 {len(df)} 行, 511260 {len(idle)} 行')
    return df, idle


def evaluate(p, df, idle, start=DISPLAY_START, comm=FEE_RATE, min_comm=FEE_MIN):
    """与线上完全一致的单参数组合评估。返回指标 dict 或 None(无可平仓交易)"""
    df_sig = run_strategy(df, p)
    try:
        eq, tr = backtest(df_sig, idle_price=idle, comm=comm, min_comm=min_comm, lot_size=1)
    except Exception:
        return None
    df2 = df_sig[df_sig.index >= start]
    eq2 = eq[eq.index >= start]
    buys = tr[(tr['action'] == 'BUY') & (tr['date'] >= start)].reset_index(drop=True)
    sells = tr[(tr['action'] == 'SELL') & (tr['date'] >= start)].reset_index(drop=True)
    if len(sells) == 0:
        return None
    try:
        stats, _ = compute_stats(df2, eq2, sells)
    except Exception:
        return None
    stats['holding_pct'] = compute_holding_pct(buys, sells, df2)
    stats['n_trades'] = stats.pop('trade_count')
    return stats


def short(p):
    return ','.join(f'{k}={v:g}' for k, v in p.items())


def sweep(params_to_sweep, fixed, df, idle, tag, eval_win=None):
    """对指定参数逐个扫描(其他固定), 输出 CSV + 控制台表格。eval_win=(start,end) 时只在窗口内评估"""
    rows = []
    for pname, values in params_to_sweep.items():
        for v in values:
            p = copy.deepcopy(fixed)
            p[pname] = v
            if eval_win:
                r = evaluate_range(p, df, idle, *eval_win)
            else:
                r = evaluate(p, df, idle)
            if r is None:
                continue
            r['param'] = pname
            r['value'] = v
            r['params'] = short(p)
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        print('[sweep] 无有效结果')
        return []
    path = os.path.join(OUT_DIR, f'sweep_{tag}.csv')
    out.to_csv(path, index=False)
    best = out.loc[out['annualized_pct'].idxmax()]

    disp = out[['param', 'value', 'annualized_pct', 'max_drawdown_pct', 'sharpe',
                'win_rate_pct', 'n_trades', 'avg_win_pct', 'holding_pct']]
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(f'\n===== SWEEP [{tag}] =====  {len(out)} evals')
        print(disp.sort_values('annualized_pct', ascending=False).to_string(index=False))
    print(f'\n[best] {best["param"]}={best["value"]:g}  ann={best["annualized_pct"]:.1f}%  '
          f'maxDD={best["max_drawdown_pct"]:.1f}%  sharpe={best["sharpe"]:.2f}  '
          f'win={best["win_rate_pct"]:.0f}%  trades={best["n_trades"]}')
    print(f'[saved] {path}')
    return rows


def cmd_fetch():
    load_data(refresh=True)
    print('done')


def cmd_sweep(args, df, idle):
    names = [s.strip() for s in args.params.split(',')]
    ALLOWED = set(PARAMS.keys())
    bad = [n for n in names if n not in ALLOWED]
    if bad:
        sys.exit(f'未知参数: {bad}. 可选: {sorted(ALLOWED)}')

    fixed = dict(PARAMS)
    if args.fixed:
        for kv in args.fixed.split(','):
            k, v = kv.split('=')
            if k.strip() not in ALLOWED:
                sys.exit(f'未知参数: {k.strip()}')
            fixed[k.strip()] = float(v)
    sweep_grid = {}
    for n in names:
        lo, hi = args.range  # (lo, hi)
        steps = args.steps
        sweep_grid[n] = [round(lo + (hi - lo) * i / (steps - 1), 2) for i in range(steps)]
    win = {'train': ('2014-01-01', '2019-12-31'), 'test': ('2020-01-01', '2026-12-31'),
           'full': None}.get(args.window)
    sweep(sweep_grid, fixed, df, idle, tag='custom_' + '_'.join(names) + f'_{args.window}', eval_win=win)


def cmd_refine(args, df, idle):
    """围绕基线 PARAMS 对全部数值参数做小步长联合扫描(笛卡尔积太大时逐步放缩)"""
    delta = dict(args.delta) if args.delta else dict(b1=0.5, b2=0.5, b2r=5, b3hi=1, s1=1, s2=1,
                                                     s2r=5, s3pk=1, s3dp=0.5, s4pr=1, s4r=5)
    grid = {}
    for k, v in PARAMS.items():
        if k in delta and isinstance(v, (int, float)):
            d = delta[k]
            grid[k] = [round(v + d, 2), round(v, 2), round(v - d, 2)]
    rows = sweep(grid, dict(PARAMS), df, idle, tag='refine')


def cmd_validate(args, df, idle):
    """样本内/外验证: 训练 2014-2019, 测试 2020-2026"""
    p = dict(PARAMS)
    if args.best:
        for kv in args.best.split(','):
            k, v = kv.split('=')
            p[k.strip()] = float(v)
    print(f'\n===== VALIDATE =====  {short(p)}\n')
    baseline = dict(PARAMS)
    labels = [('基线', baseline)] if args.best else []
    labels.append(('候选', p))
    for label, params in labels:
        print(f'--- {label}: {short(params)}')
        for name in ['全样本', '训练 2014-2019', '测试 2020-2026']:
            if name == '训练 2014-2019':
                r = evaluate_range(params, df, idle, '2014-01-01', '2019-12-31')
            else:
                r = evaluate(params, df, idle, start='2014-01-01' if name == '全样本' else '2020-01-01')
            if r is None:
                print(f'  {name}: 无可平仓交易')
                continue
            print(f'  {name}: ann={r["annualized_pct"]:.1f}%  maxDD={r["max_drawdown_pct"]:.1f}%  '
                  f'sharpe={r["sharpe"]:.2f}  win={r["win_rate_pct"]:.0f}%  '
                  f'trades={r["n_trades"]}  avg={r["avg_win_pct"]:.1f}%  hold={r["holding_pct"]:.0f}%')


def evaluate_range(params, df, idle, start, end, comm=FEE_RATE, min_comm=FEE_MIN):
    """在 [start, end] 区间内评估(交易必须在该区间内开平仓)"""
    df_sig = run_strategy(df, params)
    eq, tr = backtest(df_sig, idle_price=idle, comm=comm, min_comm=min_comm, lot_size=1)
    df2 = df_sig[(df_sig.index >= start) & (df_sig.index <= end)]
    eq2 = eq[(eq.index >= start) & (eq.index <= end)]
    buys = tr[(tr['action'] == 'BUY') & (tr['date'] >= start) & (tr['date'] <= end)]
    sells = tr[(tr['action'] == 'SELL') & (tr['date'] >= start) & (tr['date'] <= end)]
    if len(sells) == 0:
        return None
    stats, _ = compute_stats(df2, eq2, sells)
    stats['holding_pct'] = compute_holding_pct(buys, sells, df2)
    stats['n_trades'] = stats.pop('trade_count')
    return stats


def parse_params(s):
    p = dict(PARAMS)
    for kv in s.split(','):
        k, v = kv.split('=')
        if k.strip() not in p:
            raise ValueError(f'未知参数: {k.strip()}')
        p[k.strip()] = float(v)
    return p


METRIC_COLS = ['annualized_pct', 'max_drawdown_pct', 'sharpe', 'win_rate_pct',
               'n_trades', 'avg_win_pct', 'holding_pct', 'excess_annualized_pct']


def eval_matrix(params, df, idle, windows):
    """windows: [(label, start, end_or_None)] -> DataFrame 每窗口一行"""
    rows = []
    for label, start, end in windows:
        r = evaluate_range(params, df, idle, start, end) if end else evaluate(params, df, idle, start=start)
        if r is None:
            rows.append({'window': label, 'annualized_pct': None})
            continue
        row = {'window': label}
        row.update({k: r.get(k) for k in METRIC_COLS})
        rows.append(row)
    return pd.DataFrame(rows)


def cmd_compare(args, df, idle):
    """双参数集科学对比: 全指标矩阵 + 费用敏感度 + 参数扰动 + 起点敏感"""
    base = dict(BASE_PARAMS)
    cand = parse_params(args.cand)
    label_base = args.label_base or '基线'
    label_cand = args.label_cand or '候选'

    windows = [('全样本 2014-', '2014-01-01', None),
               ('训练 2014-19', '2014-01-01', '2019-12-31'),
               ('测试 2020-26', '2020-01-01', None)]

    print(f'\n===== COMPARE: {label_base} vs {label_cand} =====')
    print(f'基线: {short(base)}')
    print(f'候选: {short(cand)}')

    print('\n[1] 全指标矩阵 (年化% / maxDD% / Sharpe / 胜率% / 笔数 / 均盈% / 持仓% / 超额% / 买入持有%)')
    for label, p in [(label_base, base), (label_cand, cand)]:
        m = eval_matrix(p, df, idle, windows)
        print(f'\n--- {label} ---')
        with pd.option_context('display.width', 200, 'display.float_format', lambda v: f'{v:8.1f}'):
            print(m.to_string(index=False))

    print('\n[2] 费用敏感度 (全样本年化%)')
    rows = []
    for comm in [0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005]:
        rb = evaluate(base, df, idle, comm=comm, min_comm=FEE_MIN)
        rc = evaluate(cand, df, idle, comm=comm, min_comm=FEE_MIN)
        rows.append({'comm': f'{comm*100:.2f}%',
                     f'{label_base}_ann': rb['annualized_pct'] if rb else None,
                     f'{label_cand}_ann': rc['annualized_pct'] if rc else None,
                     'diff_pp': (rc['annualized_pct'] - rb['annualized_pct']) if rb and rc else None})
    print(pd.DataFrame(rows).to_string(index=False))

    lo, hi = 0.001, 0.02
    while hi - lo > 1e-4:
        mid = (lo + hi) / 2
        rb = evaluate(base, df, idle, comm=mid, min_comm=FEE_MIN)
        rc = evaluate(cand, df, idle, comm=mid, min_comm=FEE_MIN)
        if rb is None or rc is None:
            break
        if rc['annualized_pct'] >= rb['annualized_pct']:
            lo = mid
        else:
            hi = mid
    print(f'[breakeven] 候选年化 = 基线年化时的佣金: {lo*100:.2f}% (超过则候选不再占优)')

    print('\n[3] 参数扰动 ±10% (全样本年化% 分布, 评估过拟合脆弱性)')
    for label, p in [(label_base, base), (label_cand, cand)]:
        anns = []
        for k in p:
            if abs(p[k]) < 1e-9:
                continue
            for f in [0.9, 1.1]:
                pp = dict(p)
                pp[k] = p[k] * f
                r = evaluate(pp, df, idle)
                if r:
                    anns.append(r['annualized_pct'])
        if anns:
            print(f'{label}: n={len(anns)}  min={min(anns):.1f}  max={max(anns):.1f}  '
                  f'mean={np.mean(anns):.1f}  std={np.std(anns):.2f}  (未扰动 {evaluate(p, df, idle)["annualized_pct"]:.1f})')

    print('\n[4] 起点敏感度 (不同起始年全样本年化%)')
    rows = []
    for start in ['2014-01-01', '2016-01-01', '2018-01-01', '2020-01-01', '2022-01-01']:
        rb = evaluate(base, df, idle, start=start)
        rc = evaluate(cand, df, idle, start=start)
        rows.append({'start': start,
                     f'{label_base}_ann': rb['annualized_pct'] if rb else None,
                     f'{label_cand}_ann': rc['annualized_pct'] if rc else None,
                     'diff_pp': (rc['annualized_pct'] - rb['annualized_pct']) if rb and rc else None})
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_bootstrap(args, df, idle):
    """配对块自举: 对日度策略收益做圆形块重采样, 检验差异显著性 + Deflated Sharpe"""
    base = dict(BASE_PARAMS)
    cand = parse_params(args.cand)
    rng = np.random.default_rng(args.seed)

    def daily_rets(p):
        df_sig = run_strategy(df, p)
        eq, _ = backtest(df_sig, idle_price=idle, comm=FEE_RATE, min_comm=FEE_MIN, lot_size=1)
        eq = eq[eq.index >= DISPLAY_START]
        return eq['equity'].pct_change().dropna().values

    print(f'\n===== BOOTSTRAP: 配对块自举 (block={args.block}, n_iter={args.n_iter}, seed={args.seed}) =====')
    ra, rb = daily_rets(cand), daily_rets(base)
    n = min(len(ra), len(rb))
    ra, rb = ra[:n], rb[:n]
    print(f'[data] 对齐日度收益 {n} 天')
    print(f'[corr] 两策略日度收益相关: {np.corrcoef(ra, rb)[0, 1]:.3f}')

    def annualize(r):
        return (np.prod(1 + r)) ** (252.0 / len(r)) - 1

    def sharpe(r):
        s = r.std(ddof=1)
        return (r.mean() / s * np.sqrt(252)) if s > 0 else 0.0

    def maxdd(r):
        eq = np.cumprod(1 + r)
        return float(np.min(eq / np.maximum.accumulate(eq) - 1))

    def block_sample(r, block):
        out = np.empty(n)
        pos = 0
        while pos < n:
            start = rng.integers(0, n)
            seg = np.arange(start, min(start + block, n))
            take = min(len(seg), n - pos)
            out[pos:pos + take] = r[seg[:take]]
            pos += take
        return out

    obs_ann = annualize(ra) - annualize(rb)
    obs_sharpe = sharpe(ra) - sharpe(rb)
    obs_mdd = maxdd(ra) - maxdd(rb)
    diffs_ann, diffs_sharpe, diffs_mdd = [], [], []
    for _ in range(args.n_iter):
        sa, sb = block_sample(ra, args.block), block_sample(rb, args.block)
        diffs_ann.append(annualize(sa) - annualize(sb))
        diffs_sharpe.append(sharpe(sa) - sharpe(sb))
        diffs_mdd.append(maxdd(sa) - maxdd(sb))

    def report(name, diffs, obs):
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_gt = np.mean(np.array(diffs) > 0)
        print(f'{name}: 观测差={obs:+.2f}  95% CI=[{lo:+.2f}, {hi:+.2f}]  P(候选>基线)={p_gt:.3f}')

    report('CAGR 差(候选-基线)', diffs_ann, obs_ann)
    report('Sharpe差', diffs_sharpe, obs_sharpe)
    report('maxDD 差', diffs_mdd, obs_mdd)
    straddle = np.percentile(diffs_ann, 2.5) < 0 < np.percentile(diffs_ann, 97.5)
    print(f'\n[verdict] CAGR差 95% CI {"跨越 0 → 无统计证据候选优于基线" if straddle else "不跨 0"}')

    print('\n[deflated Sharpe] (Bailey & Lopez de Prado, N_trials≈扫描组合数)')
    n_trials = args.n_trials
    for label, r in [('候选', ra), ('基线', rb)]:
        sr = sharpe(r)
        skew = pd.Series(r).skew()
        kurt = pd.Series(r).kurtosis()
        z_max = math.sqrt(2 * math.log(n_trials))
        var_sr = (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / n
        sd_sr = math.sqrt(max(var_sr, 1e-12))
        sr_deflated = (sr - z_max) / sd_sr
        print(f'{label}: 观测Sharpe={sr:.2f}  E[max SR|{n_trials} trials]≈{z_max:.2f}  '
              f'deflated SR={sr_deflated:.2f}  {"显著(>1.96)" if sr_deflated > 1.96 else "不显著(≤1.96)"}')


def cmd_walkforward(args, df, idle):
    """滚动重调参样本外检验: 每个测试年只用此前数据选参, 验证滚动调参系统 vs 静态基线"""
    base = dict(BASE_PARAMS)
    cand = parse_params(args.cand)
    print(f'\n===== WALK-FORWARD: 滚动重调参 (lookback={args.lookback}y, 测试窗口={args.test_window}y) =====')

    def scan_best(train_start, train_end):
        """在训练窗口内 OFAT 扫敏感参数, 返回该窗口最优参数(以年化排序, 兼顾回撤)"""
        best_p, best_score = dict(base), -1e9
        grid = dict(b1=[-2.0, -1.67, -1.33], b2=[0.0, 0.5, 1.0], b2r=[30.0, 35.0, 40.0],
                    s2=[5.0, 6.0, 7.0], s2r=[70.0, 75.0, 80.0], s3dp=[1.5, 2.0, 2.5],
                    s4pr=[3.0, 4.0, 5.0], s4r=[60.0, 65.0, 70.0])
        for k, vals in grid.items():
            for v in vals:
                p = dict(best_p)
                p[k] = v
                r = evaluate_range(p, df, idle, train_start, train_end)
                if r is None:
                    continue
                score = r['annualized_pct'] - abs(r['max_drawdown_pct']) * 0.3
                if score > best_score:
                    best_score, best_p = score, p
        return best_p

    train_end = pd.Timestamp(args.start) - pd.Timedelta(days=1)
    rows = []
    for year in range(int(args.start[:4]), 2027, int(args.test_window)):
        test_start = pd.Timestamp(f'{year}-01-01')
        test_end = min(test_start + pd.Timedelta(days=365 * int(args.test_window) - 1),
                       df.index[-1])
        train_start = test_start - pd.Timedelta(days=365 * args.lookback)
        if train_start < df.index[0]:
            train_start = df.index[0]
        if test_start > df.index[-1]:
            break
        best = scan_best(train_start, test_start - pd.Timedelta(days=1))
        rb = evaluate_range(base, df, idle, str(test_start.date()), str(test_end.date()))
        rc = evaluate_range(cand, df, idle, str(test_start.date()), str(test_end.date()))
        rw = evaluate_range(best, df, idle, str(test_start.date()), str(test_end.date()))
        rows.append({'test_win': f'{test_start.date()}~{test_end.date()}',
                     'wf_ann': rw['annualized_pct'] if rw else None,
                     'base_ann': rb['annualized_pct'] if rb else None,
                     'cand_ann': rc['annualized_pct'] if rc else None})
        if rw and rb:
            diff = rw['annualized_pct'] - rb['annualized_pct']
            rows[-1]['wf_vs_base'] = diff

    out = pd.DataFrame(rows)
    with pd.option_context('display.width', 200):
        print(out.to_string(index=False))
    wf_anns = out['wf_ann'].dropna()
    base_anns = out['base_ann'].dropna()
    if len(wf_anns) and len(base_anns):
        wins = (out['wf_ann'].dropna().values > out['base_ann'].dropna().values)
        print(f'\n[verdict] 滚动调参均值={wf_anns.mean():.1f}% vs 静态基线均值={base_anns.mean():.1f}%  '
              f'调参跑赢 {wins.sum()}/{len(wins)} 段')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    f = sub.add_parser('fetch')
    f.set_defaults(fn=cmd_fetch)

    s = sub.add_parser('sweep')
    s.add_argument('--params', required=True, help='逗号分隔参数名, 如 s1,s2')
    s.add_argument('--range', nargs=2, type=float, required=True, help='扫描范围 lo hi')
    s.add_argument('--steps', type=int, default=7)
    s.add_argument('--fixed', default=None, help='非扫描参数覆盖, 形如 "s2=5,s4pr=3.5"')
    s.add_argument('--window', choices=['full', 'train', 'test'], default='full',
                   help='评估窗口: full=2014起全样本, train=2014-2019, test=2020-2026')
    s.set_defaults(fn=cmd_sweep)

    r = sub.add_parser('refine')
    r.add_argument('--delta', nargs='*', default=None,
                   help='形如 s1=1 s2=1 的步长覆盖')
    r.set_defaults(fn=cmd_refine)

    v = sub.add_parser('validate')
    v.add_argument('--best', default=None, help='候选参数, 形如 "s1=12,s2=7"')
    v.set_defaults(fn=cmd_validate)

    c = sub.add_parser('compare')
    c.add_argument('--cand', default=short(PARAMS), help='候选参数, 形如 "s2=5,s4pr=3.5"')
    c.add_argument('--label-base', default='基线')
    c.add_argument('--label-cand', default='候选')
    c.set_defaults(fn=cmd_compare)

    b = sub.add_parser('bootstrap')
    b.add_argument('--cand', default=short(PARAMS), help='候选参数, 形如 "s2=5,s4pr=3.5"')
    b.add_argument('--block', type=int, default=60, help='自举块长(交易日), 默认60≈平均持仓')
    b.add_argument('--n-iter', type=int, default=5000)
    b.add_argument('--seed', type=int, default=42)
    b.add_argument('--n-trials', type=int, default=2000, help='deflated Sharpe 假设的扫描组合数')
    b.set_defaults(fn=cmd_bootstrap)

    w = sub.add_parser('walkforward')
    w.add_argument('--cand', default=short(PARAMS), help='候选参数, 形如 "s2=5,s4pr=3.5"')
    w.add_argument('--start', default='2016-01-01', help='首个测试年起点')
    w.add_argument('--lookback', type=int, default=3, help='选参回看年数')
    w.add_argument('--test-window', type=int, default=1, help='每个测试窗口年数')
    w.set_defaults(fn=cmd_walkforward)

    args = ap.parse_args()
    if args.cmd == 'fetch':
        args.fn()
    else:
        df, idle = load_data()
        args.fn(args, df, idle)


if __name__ == '__main__':
    main()

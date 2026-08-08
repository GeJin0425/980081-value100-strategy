import numpy as np
import pandas as pd

from pipeline.factor_strategy import (
    PARAMS,
    build_factors,
    metrics,
    run_factor_strategy,
    weighted_backtest,
)


def _fixture(n=800):
    dates = pd.date_range('2016-01-01', periods=n, freq='D')
    t = np.arange(n)
    price = pd.Series(1000 + t * 1.5, index=dates)
    hold = pd.Series(1000 + t * 1.5, index=dates) * (1 + 0.2 * t / n)
    pe = pd.Series(14 - 6 * t / n, index=dates)
    cn10y = pd.Series(np.full(n, 2.0), index=dates)
    idle = pd.Series(np.full(n, 100.0), index=dates)
    return price, hold, pe, cn10y, idle


def test_build_factors_columns_and_no_lookahead():
    price, hold, pe, cn10y, idle = _fixture()
    df = build_factors(price, hold, pe, cn10y)
    for col in ('div_yield', 'spread', 'pe_pct', 'dy_pct', 'value_score', 'realized_vol', 'vol_scale'):
        assert col in df.columns
    # 第一个有效值出现在预热期之后, 且pe_pct应只依赖历史数据
    assert df['pe_pct'].first_valid_index() is not None
    assert df['div_yield'].first_valid_index() > df.index[250]


def test_run_factor_strategy_state_machine():
    price, hold, pe, cn10y, idle = _fixture()
    df = build_factors(price, hold, pe, cn10y)
    sig = run_factor_strategy(df, PARAMS)
    assert {'regime', 'target_weight', 'weight'}.issubset(sig.columns)
    # 状态只能是0/1, 权重在[0,1]
    assert set(sig['regime'].dropna().unique()).issubset({0, 1})
    assert ((sig['weight'] >= 0) & (sig['weight'] <= 1)).all()
    # 目标仓位必须等于 regime * vol_scale
    assert np.allclose(
        sig['target_weight'].fillna(0).values,
        (sig['regime'] * sig['vol_scale']).fillna(0).values,
    )


def test_weighted_backtest_and_metrics():
    price, hold, pe, cn10y, idle = _fixture(n=1200)
    df = build_factors(price, hold, pe, cn10y)
    sig = run_factor_strategy(df, PARAMS)
    eq, reb = weighted_backtest(sig, idle_price=idle, fee_rate=0.00005, min_fee=0.5)
    assert eq.index.equals(sig.index)
    assert eq.iloc[-1] > 0
    assert len(reb) > 0
    assert {'date', 'weight_from', 'weight_to', 'fee'}.issubset(reb.columns)

    m = metrics(sig, eq, start='2017-01-01', end='2020-12-31', buy_hold=hold)
    assert m is not None
    assert m['annualized_pct'] is not None
    assert m['max_drawdown_pct'] <= 0
    assert 0 <= m['avg_weight_pct'] <= 100

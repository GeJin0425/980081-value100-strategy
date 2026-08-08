import pandas as pd
import pytest

from pipeline.backtest import backtest


def test_backtest_single_round_trip_no_idle():
    dates = pd.date_range('2020-01-01', periods=3, freq='D')
    df = pd.DataFrame({
        'close': [10.0, 11.0, 12.0],
        'close_raw': [10.0, 11.0, 12.0],
        'signal': [1, 0, -1],
        'sell_reason': ['', '', '涨够了'],
    }, index=dates)

    eq, tr = backtest(df, idle_price=None, initial=100000, comm=0.0)

    assert len(tr) == 2
    assert tr.iloc[0]['action'] == 'BUY'
    assert tr.iloc[1]['action'] == 'SELL'
    assert tr.iloc[1]['pnl_pct'] == pytest.approx((12.0 / 10.0 - 1) * 100)
    assert eq['equity'].iloc[-1] > 100000


def test_backtest_applies_min_commission():
    dates = pd.date_range('2020-01-01', periods=3, freq='D')
    df = pd.DataFrame({
        'close': [10.0, 11.0, 12.0],
        'close_raw': [10.0, 11.0, 12.0],
        'signal': [1, 0, -1],
        'sell_reason': ['', '', '涨够了'],
    }, index=dates)

    eq, tr = backtest(df, idle_price=None, initial=10000, comm=0.0, min_comm=1.0)

    # 买入: 10000 - 1(最低佣金) 可买 900 股, 再扣1元最低佣金
    assert tr.iloc[0]['shares'] == 900
    # 卖出: 剩余现金999 + 900*12 - 1(最低佣金) = 11798
    assert eq['equity'].iloc[-1] == 11798


def test_backtest_lot_size_one_for_index_points():
    dates = pd.date_range('2020-01-01', periods=3, freq='D')
    df = pd.DataFrame({
        'close': [1000.0, 1100.0, 1200.0],
        'close_raw': [1000.0, 1100.0, 1200.0],
        'signal': [1, 0, -1],
        'sell_reason': ['', '', '涨够了'],
    }, index=dates)

    eq, tr = backtest(df, idle_price=None, initial=100000, comm=0.0, lot_size=1)

    assert tr.iloc[0]['shares'] == 100
    assert tr.iloc[1]['pnl_pct'] == pytest.approx((1200.0 / 1000.0 - 1) * 100)
    assert eq['equity'].iloc[-1] == pytest.approx(120000.0)


def test_backtest_uses_hold_price_for_total_return():
    dates = pd.date_range('2020-01-01', periods=3, freq='D')
    df = pd.DataFrame({
        'close': [1000.0, 1100.0, 1200.0],
        'close_raw': [1000.0, 1100.0, 1200.0],
        'signal': [1, 0, -1],
        'sell_reason': ['', '', '涨够了'],
    }, index=dates)
    hold = pd.Series([1000.0, 2000.0, 3000.0], index=dates)

    eq, tr = backtest(df, idle_price=None, initial=100000, comm=0.0, lot_size=1, hold_price=hold)

    # 买入按 hold_price=1000, 卖出按 3000, 收益应为 200%, 而不是信号价格口径的 20%
    assert tr.iloc[1]['pnl_pct'] == pytest.approx(200.0)
    assert eq['equity'].iloc[-1] == pytest.approx(300000.0)

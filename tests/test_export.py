import json

import numpy as np
import pandas as pd

import pipeline.export as export_mod


def _build_fixture():
    n = 1400
    dates = pd.date_range('2017-01-01', periods=n, freq='D')
    t = np.arange(n)
    price = 1000 + t * 2.0
    # 全收益指数相对价格指数先加速后减速(股息率先升后降), 触发一次买入和一次卖出
    peak = int(n * 0.6)
    ratio = np.ones(n)
    ratio[:peak] = 1.0 + 0.12 * (t[:peak] / peak) ** 2
    ratio[peak:] = 1.0 + 0.12 - 0.12 * ((t[peak:] - peak) / (n - peak)) ** 2
    hold = price * ratio
    df = pd.DataFrame({
        'open': price, 'close': price, 'high': price, 'low': price,
        'volume': np.full(n, 1_000_000.0),
        'close_raw': price, 'high_raw': price, 'low_raw': price,
        'adjust_factor': np.ones(n),
    }, index=dates)
    hold_df = df.copy()
    hold_df['open'] = hold_df['close'] = hold_df['high'] = hold_df['low'] = hold_df['close_raw'] = \
        hold_df['high_raw'] = hold_df['low_raw'] = hold
    pe = pd.Series(8 + 6 * t / n, index=dates, name='pe')
    cn10y = pd.Series(np.full(n, 2.0), index=dates, name='cn10y')
    idle = pd.Series(np.full(n, 100.0), index=dates)
    return df, hold_df, pe, cn10y, idle


def test_export_end_to_end(tmp_path, monkeypatch):
    price, hold, pe, cn10y, idle = _build_fixture()
    monkeypatch.setattr(export_mod, 'fetch_980081_daily', lambda: price)
    monkeypatch.setattr(export_mod, 'fetch_480081_daily', lambda: hold)
    monkeypatch.setattr(export_mod, 'fetch_csindex_daily_pe', lambda: pe)
    monkeypatch.setattr(export_mod, 'fetch_cn10y', lambda: cn10y)
    monkeypatch.setattr(export_mod, 'fetch_511260_close', lambda count=2500: idle)
    monkeypatch.setattr(export_mod, 'DISPLAY_START', '2018-06-01')
    monkeypatch.setattr(export_mod, 'TRAIN_START', '2018-06-01')
    monkeypatch.setattr(export_mod, 'TRAIN_END', '2019-06-30')
    monkeypatch.setattr(export_mod, 'TEST_START', '2019-07-01')
    monkeypatch.setattr(export_mod, 'TEST_END', '2020-06-30')

    out_path = tmp_path / 'data.json'
    payload = export_mod.export(str(out_path))

    assert out_path.exists()
    reloaded = json.loads(out_path.read_text(encoding='utf-8'))
    assert reloaded == payload

    assert payload['meta']['train']['annualized_pct'] is not None
    assert payload['meta']['test']['annualized_pct'] is not None
    assert payload['meta']['full']['annualized_pct'] is not None
    assert payload['current_status']['date'] == price.index[-1].strftime('%Y-%m-%d')
    assert 'value_score' in payload['current_status']
    for key in ('dates', 'close', 'weight', 'value_score', 'pe_pct', 'spread',
                'equity_strategy', 'equity_buyhold'):
        assert key in payload['series']
        assert len(payload['series'][key]) == len(payload['series']['dates'])
    assert 'NaN' not in out_path.read_text(encoding='utf-8')
    assert len(payload['regime_events']) >= 1
    assert len(payload['rebalances']) >= 1

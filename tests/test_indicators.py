import numpy as np
import pandas as pd

from pipeline.indicators import add_indicators


def test_add_indicators_ma_deviation_rsi_macd():
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    close = pd.Series(np.linspace(10, 20, 300), index=dates)
    df = pd.DataFrame({
        'open': close, 'close': close, 'high': close, 'low': close,
        'volume': 1000,
    }, index=dates)

    out = add_indicators(df)

    assert np.isclose(out['ma10'].iloc[9], close.iloc[0:10].mean())
    assert np.isclose(out['ma250'].iloc[249], close.iloc[0:250].mean())
    assert out['deviation'].iloc[-1] > 0  # 持续上涨，收盘价高于MA250
    assert out['rsi'].iloc[-1] > 50       # 持续上涨，RSI偏多头区间
    assert {'macd', 'macd_signal', 'macd_hist'}.issubset(out.columns)

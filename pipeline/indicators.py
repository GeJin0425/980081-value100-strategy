def add_indicators(df):
    d = df.copy()
    d['ma10'] = d['close'].rolling(10).mean()
    d['ma20'] = d['close'].rolling(20).mean()
    d['ma60'] = d['close'].rolling(60).mean()
    d['ma250'] = d['close'].rolling(250).mean()
    d['deviation'] = (d['close'] - d['ma250']) / d['ma250'] * 100
    d['ma250_slope'] = (d['ma250'] - d['ma250'].shift(10)) / d['ma250'].shift(10) * 100

    delta = d['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    ag14 = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    al14 = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    d['rsi'] = 100 - 100 / (1 + ag14 / al14)
    ag6 = gain.ewm(alpha=1 / 6, min_periods=6).mean()
    al6 = loss.ewm(alpha=1 / 6, min_periods=6).mean()
    d['rsi6'] = 100 - 100 / (1 + ag6 / al6)

    ema12 = d['close'].ewm(span=12).mean()
    ema26 = d['close'].ewm(span=26).mean()
    d['macd'] = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9).mean()
    d['macd_hist'] = d['macd'] - d['macd_signal']
    return d

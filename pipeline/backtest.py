import pandas as pd


def backtest(df, idle_price=None, initial=100000, comm=0.001, min_comm=0.0, lot_size=100):
    """回测引擎: 持仓标的 + 空仓期买入十年国债ETF

    佣金按每笔成交额计算: max(成交额 * comm, min_comm)。
    lot_size: 一手股数。ETF/股票用100; 指数点位回测传1(指数不是一手100股的股票)。
    默认保留旧的万10(0.1%)、无最低佣金口径; 线上发布请传入真实费率
    (export.py 中为 万0.5 + 单笔最低0.5元)。
    """
    capital = initial
    shares = 0
    shares_idle = 0
    trades = []
    equity = []

    def fee(notional):
        return max(notional * comm, min_comm)

    for i in range(len(df)):
        sig = df.iloc[i]['signal']
        close = df.iloc[i]['close']
        date = df.index[i]
        idle_close = idle_price.loc[date] if (idle_price is not None and date in idle_price.index) else None

        if sig == 1 and shares == 0:
            if shares_idle > 0 and idle_close:
                capital += shares_idle * idle_close - fee(shares_idle * idle_close)
                shares_idle = 0
            s = int((capital - fee(capital)) / close / lot_size) * lot_size
            capital -= s * close + fee(s * close)
            shares = s
            trades.append({
                'date': date, 'action': 'BUY', 'price': close, 'shares': s,
                'price_raw': df.iloc[i]['close_raw'],
            })
        elif sig == -1 and shares > 0:
            capital += shares * close - fee(shares * close)
            pnl = (close / trades[-1]['price'] - 1) * 100
            trades.append({
                'date': date, 'action': 'SELL', 'price': close, 'shares': shares,
                'price_raw': df.iloc[i]['close_raw'],
                'pnl_pct': pnl, 'hold_days': (date - trades[-1]['date']).days,
                'reason': df.iloc[i]['sell_reason'],
            })
            shares = 0
            if idle_close and idle_close > 0:
                si = int((capital - fee(capital)) / idle_close / lot_size) * lot_size
                if si > 0:
                    capital -= si * idle_close + fee(si * idle_close)
                    shares_idle = si

        total = capital + (shares * close if shares > 0 else 0)
        if shares_idle > 0 and idle_close:
            total += shares_idle * idle_close
        equity.append({'date': date, 'equity': total})

    return pd.DataFrame(equity).set_index('date'), pd.DataFrame(trades)

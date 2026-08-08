import pandas as pd

PARAMS = dict(
    b1=-1.5, b2=0.0, b2r=60.0, b3lo=0.0, b3hi=8.0,
    s1=13.0, s2=8.0, s2r=60.0, s3pk=3.0, s3dp=1.5,
    s4pr=3.0, s4r=60.0, cooldown=11,
)


def run_strategy(df, p=PARAMS):
    d = df.copy()
    d['signal'] = 0
    d['sell_reason'] = ''
    d['position'] = 0
    pos = 0
    last_sell = -999
    ctx = {}

    for i in range(1, len(d)):
        row = d.iloc[i]
        prev = d.iloc[i - 1]
        if pd.isna(row['ma250']) or pd.isna(row['rsi']):
            d.iloc[i, d.columns.get_loc('position')] = pos
            continue

        dev = row['deviation']
        rsi = row['rsi']

        if pos == 0 and (i - last_sell) >= p['cooldown']:
            buy = False
            if dev < p['b1']:
                buy = True
            elif dev < p['b2'] and rsi < p['b2r']:
                buy = True
            else:
                slope = row['ma250_slope'] if not pd.isna(row['ma250_slope']) else 0
                above_ma10 = row['close'] > row['ma10'] if not pd.isna(row['ma10']) else False
                if p['b3lo'] <= dev <= p['b3hi'] and slope > 0 and above_ma10:
                    buy = True
            if buy:
                d.iloc[i, d.columns.get_loc('signal')] = 1
                pos = 1
                ctx = {'entry_price': row['close'], 'max_dev': dev}

        elif pos == 1:
            ctx['max_dev'] = max(ctx.get('max_dev', 0), dev)
            max_dev = ctx['max_dev']
            profit = (row['close'] / ctx.get('entry_price', row['close']) - 1) * 100
            sell = False
            reason = ''

            if dev >= p['s1']:
                sell, reason = True, f'硬上限:{dev:.1f}%'
            elif dev >= p['s2'] and rsi >= p['s2r']:
                sell, reason = True, f'RSI确认:RSI={rsi:.0f},偏离{dev:.1f}%'
            elif max_dev >= p['s3pk'] and dev < max_dev - p['s3dp']:
                sell, reason = True, f'偏离回落:{max_dev:.1f}%→{dev:.1f}%'
            elif profit >= p['s4pr'] and rsi < p['s4r'] and prev['rsi'] >= p['s4r']:
                sell, reason = True, f'RSI下穿{p["s4r"]:.0f}:+{profit:.1f}%'

            if sell:
                d.iloc[i, d.columns.get_loc('signal')] = -1
                d.iloc[i, d.columns.get_loc('sell_reason')] = reason
                pos = 0
                last_sell = i
                ctx = {}

        d.iloc[i, d.columns.get_loc('position')] = pos
    return d

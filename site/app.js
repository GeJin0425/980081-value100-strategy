async function main() {
  let data;
  try {
    const res = await fetch('./data.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    showDataError(`数据加载失败: ${err.message}`);
    return;
  }

  checkStaleness(data.meta.as_of_date);

  renderTopbar(data);
  renderKpis(data.meta);
  renderStatusCard(data.current_status);
  renderPriceChart(data.series, data.trades);
  renderReturnsPanel(data.series, data.trades);
  renderDeviationChart(data.series);
  renderRsiChart(data.series);
  renderMacdChart(data.series);
  renderEquityChart(data.series);
  renderSellReasonChart(data.sell_reason_breakdown);
  renderTradesTable(data.trades);
}

function showDataError(message) {
  const el = document.getElementById('data-error');
  el.textContent = message;
  el.hidden = false;
}

const STALENESS_THRESHOLD_DAYS = 4;

function checkStaleness(asOfDate) {
  const asOf = new Date(`${asOfDate}T00:00:00`);
  if (isNaN(asOf.getTime())) return;
  const now = new Date();
  const diffDays = Math.floor((now.getTime() - asOf.getTime()) / (1000 * 60 * 60 * 24));
  if (diffDays > STALENESS_THRESHOLD_DAYS) {
    showDataError(`数据已 ${diffDays} 天未更新,当前显示的可能不是最新信号(数据日期: ${asOfDate})`);
  }
}

function renderTopbar(data) {
  const dot = document.getElementById('status-dot');
  dot.classList.add(data.current_status.holding ? 'holding' : 'idle');
  document.getElementById('updated-at').textContent =
    `数据日期 ${data.meta.as_of_date} · 更新于 ${data.meta.updated_at.slice(0, 16).replace('T', ' ')}`;
}

function fmtSigned(value, suffix = '%') {
  const sign = value > 0 ? '+' : '';
  return `${sign}${value}${suffix}`;
}

function renderKpis(meta) {
  const cards = [
    { value: fmtSigned(meta.annualized_pct), label: '年化(含国债·万0.5)', signed: true },
    { value: `${meta.max_drawdown_pct}%`, label: '最大回撤', signed: true },
    { value: meta.sharpe.toFixed(2), label: '夏普比率', signed: false },
    { value: `${meta.win_rate_pct}%`, label: `胜率(${meta.trade_count}笔)`, signed: false },
    { value: fmtSigned(meta.avg_win_pct), label: '平均盈利', signed: true },
    { value: `${meta.holding_pct}%`, label: '持仓占比', signed: false },
  ];
  const row = document.getElementById('kpi-row');
  row.innerHTML = cards.map(c => `
    <div class="kpi-card">
      <div class="value ${c.signed ? (parseFloat(c.value) >= 0 ? 'pos' : 'neg') : ''}">${c.value}</div>
      <div class="label">${c.label}</div>
    </div>
  `).join('');
}

function renderStatusCard(status) {
  const card = document.getElementById('status-card');
  card.innerHTML = `
    <div class="status-grid">
      <div><span class="k">日期</span><br>${status.date}</div>
      <div><span class="k">指数点位</span><br>${status.price_raw}</div>
      <div><span class="k">MA250</span><br>${status.ma250}</div>
      <div><span class="k">偏离度</span><br>${fmtSigned(status.deviation_pct)}</div>
      <div><span class="k">RSI14 / RSI6</span><br>${status.rsi14} / ${status.rsi6}</div>
      <div><span class="k">卖出监控价</span><br>${status.sell_trigger_price_soft}</div>
      <div><span class="k">硬卖价</span><br>${status.sell_trigger_price_hard}</div>
      <div><span class="k">买入上限价</span><br>${status.buy_trigger_price_cap}</div>
    </div>
    <div class="signal ${status.signal_level}">${status.signal_text}</div>
  `;
}

const DARK_AXIS = {
  axisLine: { lineStyle: { color: '#21262d' } },
  axisLabel: { color: '#8b949e' },
  splitLine: { lineStyle: { color: '#161b22' } },
};

function baseGrid() {
  return { left: 56, right: 24, top: 24, bottom: 40 };
}

function isoYearsAgo(dateStr, years) {
  const [y, m, d] = dateStr.split('-').map(Number);
  return new Date(Date.UTC(y - years, m - 1, d)).toISOString().slice(0, 10);
}

function startIndexForRange(allDates, range) {
  const lastDate = allDates[allDates.length - 1] ?? '';
  if (!lastDate) return 0;
  let startDate;
  if (range === 'ytd') startDate = `${lastDate.slice(0, 4)}-01-01`;
  else if (range === '1y') startDate = isoYearsAgo(lastDate, 1);
  else if (range === '3y') startDate = isoYearsAgo(lastDate, 3);
  else if (range === '5y') startDate = isoYearsAgo(lastDate, 5);
  else return 0;
  const idx = allDates.findIndex(d => d >= startDate);
  return idx === -1 ? 0 : idx;
}

function renderPriceChart(series, trades) {
  const chart = echarts.init(document.getElementById('chart-price'));
  const allDates = series.dates;

  function buildOption(start) {
    const dates = allDates.slice(start);
    const firstDate = dates[0] ?? '';
    const buyPoints = trades
      .filter(t => t.buy_date >= firstDate)
      .map(t => ({
        coord: [t.buy_date, t.buy_price], symbol: 'triangle',
        itemStyle: { color: '#3fb950' },
      }));
    const sellPoints = trades
      .filter(t => t.sell_date && t.sell_date >= firstDate)
      .map(t => ({
        coord: [t.sell_date, t.sell_price], symbol: 'pin',
        itemStyle: { color: '#f85149' },
      }));

    return {
      backgroundColor: 'transparent',
      grid: baseGrid(),
      tooltip: { trigger: 'axis', backgroundColor: '#161b22', borderColor: '#21262d', textStyle: { color: '#c9d1d9' } },
      legend: { data: ['收盘价', 'MA10', 'MA20', 'MA60', 'MA250'], textStyle: { color: '#8b949e' }, top: 0 },
      xAxis: { type: 'category', data: dates, ...DARK_AXIS },
      yAxis: { type: 'value', scale: true, ...DARK_AXIS },
      dataZoom: [{ type: 'slider', backgroundColor: '#161b22', fillerColor: 'rgba(88,166,255,0.15)' }],
      series: [
        { name: '收盘价', type: 'line', data: series.close.slice(start), showSymbol: false, lineStyle: { width: 1.5, color: '#c9d1d9' } },
        { name: 'MA10', type: 'line', data: series.ma10.slice(start), showSymbol: false, lineStyle: { width: 1, color: '#58a6ff', opacity: 0.5 } },
        { name: 'MA20', type: 'line', data: series.ma20.slice(start), showSymbol: false, lineStyle: { width: 1, color: '#3fb950', opacity: 0.4 } },
        { name: 'MA60', type: 'line', data: series.ma60.slice(start), showSymbol: false, lineStyle: { width: 1, color: '#d29922', opacity: 0.4 } },
        {
          name: 'MA250', type: 'line', data: series.ma250.slice(start), showSymbol: false,
          lineStyle: { width: 2, color: '#f85149', type: 'dashed' },
          markPoint: { symbolSize: 14, data: [...buyPoints, ...sellPoints] },
        },
      ],
    };
  }

  const buttons = document.querySelectorAll('#chart-price-ranges .range-btn');
  function applyRange(range) {
    chart.setOption(buildOption(startIndexForRange(allDates, range)), { notMerge: true });
    buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.range === range));
  }
  buttons.forEach(btn => btn.addEventListener('click', () => applyRange(btn.dataset.range)));
  applyRange('all');
  window.addEventListener('resize', () => chart.resize());
}

function toneOf(value) {
  return value > 0 ? 'pos' : value < 0 ? 'neg' : '';
}

function renderReturnsPanel(series, trades) {
  const chart = echarts.init(document.getElementById('chart-returns'));
  const tableBody = document.getElementById('returns-table-body');
  const allDates = series.dates;

  function buildChartOption(start) {
    const dates = allDates.slice(start);
    const eq = series.equity_strategy.slice(start);
    const bh = series.equity_buyhold.slice(start);
    const eq0 = eq[0];
    const bh0 = bh[0];
    const toReturn = (arr, base) => arr.map(v => (v === null || v === undefined ? null : (v / base - 1) * 100));

    return {
      backgroundColor: 'transparent',
      grid: baseGrid(),
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#161b22',
        borderColor: '#21262d',
        textStyle: { color: '#c9d1d9' },
        valueFormatter: (value) => (value === null || value === undefined ? '—' : `${Number(value).toFixed(1)}%`),
      },
      legend: { data: ['MA250策略', '无脑持有980081'], textStyle: { color: '#8b949e' }, top: 0 },
      xAxis: { type: 'category', data: dates, ...DARK_AXIS },
      yAxis: { type: 'value', scale: true, ...DARK_AXIS, axisLabel: { color: '#8b949e', formatter: '{value}%' } },
      series: [
        {
          name: 'MA250策略', type: 'line', data: toReturn(eq, eq0), showSymbol: false,
          lineStyle: { width: 2, color: '#3fb950' },
          areaStyle: { color: 'rgba(63,185,80,0.08)' },
        },
        {
          name: '无脑持有980081', type: 'line', data: toReturn(bh, bh0), showSymbol: false,
          lineStyle: { width: 1.5, color: '#8b949e', type: 'dashed' },
        },
      ],
    };
  }

  function periodRows(start) {
    const dates = series.dates;
    const startDate = dates[start];
    const endDate = dates[dates.length - 1];
    const eq = series.equity_strategy.slice(start);
    const bh = series.equity_buyhold.slice(start);

    const stratFirst = eq[0];
    const stratFinal = eq[eq.length - 1];
    const bhFirst = bh[0];
    const bhFinal = bh[bh.length - 1];

    const stratRet = (stratFinal / stratFirst - 1) * 100;
    const bhRet = (bhFinal / bhFirst - 1) * 100;
    const excess = stratRet - bhRet;

    const years = (Date.parse(endDate) - Date.parse(startDate)) / (365.25 * 24 * 3600 * 1000);
    const annualized = years > 0 ? (Math.pow(stratFinal / stratFirst, 1 / years) - 1) * 100 : null;

    let peak = eq[0];
    let maxDd = 0;
    for (const v of eq) {
      if (v > peak) peak = v;
      const dd = (v - peak) / peak * 100;
      if (dd < maxDd) maxDd = dd;
    }

    const inPeriod = trades.filter(t => t.buy_date >= startDate);
    const closed = inPeriod.filter(t => t.sell_date);
    const winRate = closed.length ? closed.filter(t => t.pnl_pct > 0).length / closed.length * 100 : null;

    return [
      { label: '区间起点', value: startDate },
      { label: '区间终点', value: endDate },
      { label: 'MA250策略收益', value: fmtSigned(stratRet.toFixed(1)), tone: toneOf(stratRet) },
      { label: '买入持有收益', value: fmtSigned(bhRet.toFixed(1)), tone: toneOf(bhRet) },
      { label: '超额收益', value: fmtSigned(excess.toFixed(1)), tone: toneOf(excess) },
      { label: '策略年化', value: annualized === null ? '—' : fmtSigned(annualized.toFixed(1)), tone: annualized === null ? '' : toneOf(annualized) },
      { label: '最大回撤(策略)', value: fmtSigned(maxDd.toFixed(1)), tone: toneOf(maxDd) },
      { label: '交易笔数', value: `${inPeriod.length}笔` },
      { label: '胜率', value: winRate === null ? '—' : `${winRate.toFixed(0)}%` },
    ];
  }

  function applyRange(range) {
    const start = startIndexForRange(allDates, range);
    chart.setOption(buildChartOption(start), { notMerge: true });
    tableBody.innerHTML = periodRows(start).map(row => `
      <tr>
        <td>${row.label}</td>
        <td${row.tone ? ` class="${row.tone}"` : ''}>${row.value}</td>
      </tr>
    `).join('');
    document.querySelectorAll('#returns-ranges .range-btn').forEach(btn =>
      btn.classList.toggle('active', btn.dataset.range === range)
    );
  }

  document.querySelectorAll('#returns-ranges .range-btn').forEach(btn =>
    btn.addEventListener('click', () => applyRange(btn.dataset.range))
  );
  applyRange('all');
  window.addEventListener('resize', () => chart.resize());
}

function renderDeviationChart(series) {
  const chart = echarts.init(document.getElementById('chart-deviation'));
  chart.setOption({
    backgroundColor: 'transparent',
    grid: baseGrid(),
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: series.dates, ...DARK_AXIS },
    yAxis: { type: 'value', ...DARK_AXIS },
    dataZoom: [{ type: 'inside' }],
    series: [{
      type: 'bar', data: series.deviation,
      itemStyle: { color: (p) => (p.value >= 7 ? '#f85149' : p.value >= 0 ? '#3fb950' : '#58a6ff') },
    }],
  });
  window.addEventListener('resize', () => chart.resize());
}

function renderRsiChart(series) {
  const chart = echarts.init(document.getElementById('chart-rsi'));
  chart.setOption({
    backgroundColor: 'transparent',
    grid: baseGrid(),
    tooltip: { trigger: 'axis' },
    legend: { data: ['RSI14', 'RSI6'], textStyle: { color: '#8b949e' }, top: 0 },
    xAxis: { type: 'category', data: series.dates, ...DARK_AXIS },
    yAxis: { type: 'value', min: 0, max: 100, ...DARK_AXIS },
    dataZoom: [{ type: 'inside' }],
    series: [
      { name: 'RSI14', type: 'line', data: series.rsi14, showSymbol: false, lineStyle: { color: '#a371f7' } },
      { name: 'RSI6', type: 'line', data: series.rsi6, showSymbol: false, lineStyle: { color: '#d29922', opacity: 0.5 } },
    ],
  });
  window.addEventListener('resize', () => chart.resize());
}

function renderMacdChart(series) {
  const chart = echarts.init(document.getElementById('chart-macd'));
  chart.setOption({
    backgroundColor: 'transparent',
    grid: baseGrid(),
    tooltip: { trigger: 'axis' },
    legend: { data: ['MACD', 'Signal'], textStyle: { color: '#8b949e' }, top: 0 },
    xAxis: { type: 'category', data: series.dates, ...DARK_AXIS },
    yAxis: { type: 'value', ...DARK_AXIS },
    dataZoom: [{ type: 'inside' }],
    series: [
      { name: 'MACD柱', type: 'bar', data: series.macd_hist, itemStyle: { color: (p) => (p.value >= 0 ? '#3fb950' : '#f85149') } },
      { name: 'MACD', type: 'line', data: series.macd, showSymbol: false, lineStyle: { color: '#58a6ff' } },
      { name: 'Signal', type: 'line', data: series.macd_signal, showSymbol: false, lineStyle: { color: '#d29922' } },
    ],
  });
  window.addEventListener('resize', () => chart.resize());
}

function renderEquityChart(series) {
  const chart = echarts.init(document.getElementById('chart-equity'));
  chart.setOption({
    backgroundColor: 'transparent',
    grid: baseGrid(),
    tooltip: { trigger: 'axis' },
    legend: { data: ['MA250策略', '买入持有'], textStyle: { color: '#8b949e' }, top: 0 },
    xAxis: { type: 'category', data: series.dates, ...DARK_AXIS },
    yAxis: { type: 'value', ...DARK_AXIS },
    dataZoom: [{ type: 'inside' }, { type: 'slider' }],
    series: [
      { name: 'MA250策略', type: 'line', data: series.equity_strategy, showSymbol: false, lineStyle: { width: 2, color: '#3fb950' } },
      { name: '买入持有', type: 'line', data: series.equity_buyhold, showSymbol: false, lineStyle: { width: 1, color: '#8b949e', type: 'dashed' } },
    ],
  });
  window.addEventListener('resize', () => chart.resize());
}

function renderSellReasonChart(breakdown) {
  const chart = echarts.init(document.getElementById('chart-sell-reason'));
  chart.setOption({
    backgroundColor: 'transparent',
    grid: { left: 90, right: 40, top: 20, bottom: 30 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { type: 'value', ...DARK_AXIS },
    yAxis: { type: 'category', data: breakdown.map(b => b.reason), ...DARK_AXIS },
    series: [{
      type: 'bar',
      data: breakdown.map(b => b.count),
      itemStyle: { color: '#f85149' },
      label: {
        show: true, position: 'right', color: '#c9d1d9',
        formatter: (p) => `${breakdown[p.dataIndex].count}笔 · 均${fmtSigned(breakdown[p.dataIndex].avg_pnl_pct)}`,
      },
    }],
  });
  window.addEventListener('resize', () => chart.resize());
}

let currentTrades = [];
let sortState = { key: 'seq', dir: 1 };
const TRADE_COLUMN_KEYS = ['seq', 'buy_date', 'sell_date', 'buy_price_raw', 'sell_price_raw', 'pnl_pct', 'hold_days', 'sell_reason'];

function renderTradesTable(trades) {
  currentTrades = trades;
  document.querySelectorAll('#trades-table th').forEach((th, i) => {
    th.onclick = () => {
      const key = TRADE_COLUMN_KEYS[i];
      sortState.dir = sortState.key === key ? -sortState.dir : 1;
      sortState.key = key;
      drawTradesBody();
    };
  });
  drawTradesBody();
}

function drawTradesBody() {
  const rows = [...currentTrades].sort((a, b) => {
    const av = a[sortState.key];
    const bv = b[sortState.key];
    if (av === null) return 1;
    if (bv === null) return -1;
    if (av < bv) return -1 * sortState.dir;
    if (av > bv) return 1 * sortState.dir;
    return 0;
  });
  const tbody = document.getElementById('trades-table-body');
  tbody.innerHTML = rows.map(t => `
    <tr class="${t.open ? 'open-row' : ''}">
      <td>${t.seq}</td>
      <td>${t.buy_date}</td>
      <td>${t.sell_date ?? '持仓中…'}</td>
      <td>${t.buy_price_raw}</td>
      <td>${t.sell_price_raw}</td>
      <td>${fmtSigned(t.pnl_pct)}</td>
      <td>${t.hold_days}</td>
      <td>${t.sell_reason}</td>
    </tr>
  `).join('');
}

document.addEventListener('DOMContentLoaded', main);

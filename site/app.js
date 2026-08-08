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
  renderPriceChart(data.series, data.regime_events);
  renderValuationChart(data.series);
  renderReturnsPanel(data.series);
  renderPeriodsTable(data.meta);
  renderRebalancesTable(data.rebalances);
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
  const f = meta.full;
  const cards = [
    { value: fmtSigned(f.annualized_pct), label: '年化(2013至今·全收益)', signed: true },
    { value: `${f.max_drawdown_pct}%`, label: '最大回撤', signed: true },
    { value: f.sharpe.toFixed(2), label: '夏普比率', signed: false },
    { value: `${f.avg_weight_pct}%`, label: '平均仓位', signed: false },
    { value: `${f.rebalance_count}次`, label: '调仓次数', signed: false },
    { value: fmtSigned(f.excess_annualized_pct), label: '超额年化(对480081)', signed: true },
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
  const holdingText = status.holding ? `持有(仓位 ${(status.weight * 100).toFixed(0)}%)` : '空仓(配置511260国债)';
  card.innerHTML = `
    <div class="status-grid">
      <div><span class="k">日期</span><br>${status.date}</div>
      <div><span class="k">指数点位</span><br>${status.price_raw}</div>
      <div><span class="k">当前状态</span><br>${holdingText}</div>
      <div><span class="k">便宜度得分</span><br>${status.value_score.toFixed(2)}</div>
      <div><span class="k">PE分位(代理)</span><br>${(status.pe_pct * 100).toFixed(0)}%</div>
      <div><span class="k">股息率分位</span><br>${(status.dy_pct * 100).toFixed(0)}%</div>
      <div><span class="k">股息率</span><br>${status.div_yield}%</div>
      <div><span class="k">10Y国债</span><br>${status.cn10y}%</div>
      <div><span class="k">息差</span><br>${fmtSigned(status.spread)}</div>
      <div><span class="k">波动率(60日)</span><br>${status.realized_vol}%</div>
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
  return { left: 56, right: 64, top: 32, bottom: 40 };
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

function setupRangeButtons(selector, apply) {
  const buttons = document.querySelectorAll(`${selector} .range-btn`);
  buttons.forEach(btn => btn.addEventListener('click', () => apply(btn.dataset.range)));
  return buttons;
}

function renderPriceChart(series, events) {
  const chart = echarts.init(document.getElementById('chart-price'));
  const allDates = series.dates;
  const enterPoints = events.filter(e => e.action === '进入持仓').map(e => ({
    coord: [e.date, null], value: '买入', itemStyle: { color: '#3fb950' },
  }));
  const exitPoints = events.filter(e => e.action === '退出持仓').map(e => ({
    coord: [e.date, null], value: '卖出', itemStyle: { color: '#f85149' },
  }));

  function buildOption(start) {
    const dates = allDates.slice(start);
    const firstDate = dates[0] ?? '';
    const points = [...enterPoints, ...exitPoints]
      .filter(p => p.coord[0] >= firstDate)
      .map(p => ({ ...p, coord: [p.coord[0], p.value === '买入' ? 0 : 100] }));
    return {
      backgroundColor: 'transparent',
      grid: baseGrid(),
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#161b22',
        borderColor: '#21262d',
        textStyle: { color: '#c9d1d9' },
      },
      legend: { data: ['980081价格', '目标仓位%', '进入持仓', '退出持仓'], textStyle: { color: '#8b949e' }, top: 0 },
      xAxis: { type: 'category', data: dates, ...DARK_AXIS },
      yAxis: [
        { type: 'value', scale: true, name: '点位', ...DARK_AXIS },
        { type: 'value', min: 0, max: 100, name: '仓位%', axisLabel: { color: '#8b949e', formatter: '{value}%' }, splitLine: { show: false } },
      ],
      dataZoom: [{ type: 'slider', backgroundColor: '#161b22', fillerColor: 'rgba(88,166,255,0.15)' }],
      series: [
        {
          name: '980081价格', type: 'line', data: series.close.slice(start), showSymbol: false,
          lineStyle: { width: 1.5, color: '#c9d1d9' }, yAxisIndex: 0,
        },
        {
          name: '目标仓位%', type: 'line', data: series.target_weight.slice(start).map(v => v === null ? null : v * 100),
          showSymbol: false, yAxisIndex: 1, lineStyle: { width: 2, color: '#58a6ff' },
          areaStyle: { color: 'rgba(88,166,255,0.18)' },
          markPoint: {
            symbolSize: 16, data: points,
            label: { show: true, formatter: (p) => p.value, position: 'top', color: '#c9d1d9' },
          },
        },
      ],
    };
  }

  let buttons;
  function applyRange(range) {
    chart.setOption(buildOption(startIndexForRange(allDates, range)), { notMerge: true });
    buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.range === range));
  }
  buttons = setupRangeButtons('#chart-price-ranges', applyRange);
  applyRange('all');
  window.addEventListener('resize', () => chart.resize());
}

function renderValuationChart(series) {
  const chart = echarts.init(document.getElementById('chart-valuation'));
  const allDates = series.dates;

  function buildOption(start) {
    const dates = allDates.slice(start);
    return {
      backgroundColor: 'transparent',
      grid: baseGrid(),
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#161b22',
        borderColor: '#21262d',
        textStyle: { color: '#c9d1d9' },
      },
      legend: {
        data: ['便宜度得分', 'PE分位(代理)', '股息率分位', '息差%'],
        textStyle: { color: '#8b949e' }, top: 0,
      },
      xAxis: { type: 'category', data: dates, ...DARK_AXIS },
      yAxis: [
        { type: 'value', min: 0, max: 1, name: '分位', ...DARK_AXIS },
        { type: 'value', name: '息差%', axisLabel: { color: '#8b949e', formatter: '{value}%' }, splitLine: { show: false } },
      ],
      dataZoom: [{ type: 'slider', backgroundColor: '#161b22', fillerColor: 'rgba(88,166,255,0.15)' }],
      series: [
        { name: '便宜度得分', type: 'line', data: series.value_score.slice(start), showSymbol: false, yAxisIndex: 0, lineStyle: { width: 2, color: '#3fb950' } },
        { name: 'PE分位(代理)', type: 'line', data: series.pe_pct.slice(start), showSymbol: false, yAxisIndex: 0, lineStyle: { width: 1, color: '#d29922', opacity: 0.7 } },
        { name: '股息率分位', type: 'line', data: series.dy_pct.slice(start), showSymbol: false, yAxisIndex: 0, lineStyle: { width: 1, color: '#a371f7', opacity: 0.7 } },
        { name: '息差%', type: 'line', data: series.spread.slice(start), showSymbol: false, yAxisIndex: 1, lineStyle: { width: 1.5, color: '#f85149', type: 'dashed' } },
      ],
    };
  }

  let buttons;
  function applyRange(range) {
    chart.setOption(buildOption(startIndexForRange(allDates, range)), { notMerge: true });
    buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.range === range));
  }
  buttons = setupRangeButtons('#valuation-ranges', applyRange);
  applyRange('all');
  window.addEventListener('resize', () => chart.resize());
}

function toneOf(value) {
  return value > 0 ? 'pos' : value < 0 ? 'neg' : '';
}

function renderReturnsPanel(series) {
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
      legend: { data: ['因子策略', '无脑持有480081(全收益)'], textStyle: { color: '#8b949e' }, top: 0 },
      xAxis: { type: 'category', data: dates, ...DARK_AXIS },
      yAxis: { type: 'value', scale: true, ...DARK_AXIS, axisLabel: { color: '#8b949e', formatter: '{value}%' } },
      series: [
        {
          name: '因子策略', type: 'line', data: toReturn(eq, eq0), showSymbol: false,
          lineStyle: { width: 2, color: '#3fb950' },
          areaStyle: { color: 'rgba(63,185,80,0.08)' },
        },
        {
          name: '无脑持有480081(全收益)', type: 'line', data: toReturn(bh, bh0), showSymbol: false,
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
    const avgWeight = series.target_weight.slice(start)
      .filter(v => v !== null && v !== undefined).reduce((a, b) => a + b, 0)
      / Math.max(1, series.target_weight.slice(start).filter(v => v !== null && v !== undefined).length) * 100;

    return [
      { label: '区间起点', value: startDate },
      { label: '区间终点', value: endDate },
      { label: '因子策略收益', value: fmtSigned(stratRet.toFixed(1)), tone: toneOf(stratRet) },
      { label: '买入持有收益(480081全收益)', value: fmtSigned(bhRet.toFixed(1)), tone: toneOf(bhRet) },
      { label: '超额收益', value: fmtSigned(excess.toFixed(1)), tone: toneOf(excess) },
      { label: '策略年化', value: annualized === null ? '—' : fmtSigned(annualized.toFixed(1)), tone: annualized === null ? '' : toneOf(annualized) },
      { label: '最大回撤(策略)', value: fmtSigned(maxDd.toFixed(1)), tone: toneOf(maxDd) },
      { label: '平均仓位', value: `${avgWeight.toFixed(0)}%` },
    ];
  }

  let buttons;
  function applyRange(range) {
    const start = startIndexForRange(allDates, range);
    chart.setOption(buildChartOption(start), { notMerge: true });
    tableBody.innerHTML = periodRows(start).map(row => `
      <tr>
        <td>${row.label}</td>
        <td${row.tone ? ` class="${row.tone}"` : ''}>${row.value}</td>
      </tr>
    `).join('');
    buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.range === range));
  }
  buttons = setupRangeButtons('#returns-ranges', applyRange);
  applyRange('all');
  window.addEventListener('resize', () => chart.resize());
}

function renderPeriodsTable(meta) {
  const tbody = document.getElementById('periods-table-body');
  const labels = { full: '全样本(2013至今)', train: '训练集(2018-2022)', test: '测试集(2023至今)' };
  tbody.innerHTML = ['full', 'train', 'test'].map(key => {
    const m = meta[key];
    if (!m) return '';
    return `
      <tr>
        <td>${labels[key]}</td>
        <td>${fmtSigned(m.annualized_pct)}</td>
        <td class="${toneOf(m.max_drawdown_pct)}">${m.max_drawdown_pct}%</td>
        <td>${m.sharpe.toFixed(2)}</td>
        <td>${m.avg_weight_pct}%</td>
        <td>${m.rebalance_count}</td>
        <td class="${toneOf(m.excess_annualized_pct)}">${fmtSigned(m.excess_annualized_pct)}</td>
      </tr>`;
  }).join('');
}

function renderRebalancesTable(rebalances) {
  const tbody = document.getElementById('rebalances-table-body');
  const rows = rebalances.slice(-30).reverse();
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.date}</td>
      <td>${(r.weight_from * 100).toFixed(0)}%</td>
      <td>${(r.weight_to * 100).toFixed(0)}%</td>
      <td>¥${r.fee.toFixed(2)}</td>
    </tr>
  `).join('');
}

document.addEventListener('DOMContentLoaded', main);

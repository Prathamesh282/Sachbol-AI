/* ── SachBol AI — Dashboard Script ─────────────────────────── */

/* ── Theme — must run before paint to avoid flash ───────────── */
(function initTheme() {
  const saved = localStorage.getItem('sachbol-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = saved || (prefersDark ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', theme);
})();

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('sachbol-theme', theme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('themeToggle');
  if (btn) btn.addEventListener('click', toggleTheme);
});

/* Subtle light grid */
(function initGrid() {
  const canvas = document.getElementById('gridCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
  resize();
  window.addEventListener('resize', resize);
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'rgba(37,99,235,0.06)';
    ctx.lineWidth = 1;
    const s = 60;
    for (let x = 0; x <= canvas.width; x += s)  { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,canvas.height); ctx.stroke(); }
    for (let y = 0; y <= canvas.height; y += s) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(canvas.width,y); ctx.stroke(); }
  }
  draw(); window.addEventListener('resize', draw);
})();

/* Chart instances */
let catChart, sentChart;

document.addEventListener('DOMContentLoaded', loadData);

async function loadData() {
  const loader = document.getElementById('dashLoader');
  if (loader) loader.style.display = 'flex';

  try {
    const res  = await fetch('/api/dashboard-data');
    const data = await res.json();

    if (data.success) {
      updateKPIs(data);
      renderCharts(data);
      renderTrending(data.trends || []);
      document.getElementById('lastUpdated').textContent =
        'Updated ' + new Date().toLocaleTimeString('en-IN', {hour:'2-digit', minute:'2-digit'});
    } else {
      document.getElementById('lastUpdated').textContent = 'Error: ' + (data.error || 'unknown');
      renderTrending([]);
    }
  } catch (err) {
    document.getElementById('lastUpdated').textContent = 'Network error';
    console.error('Dashboard error:', err);
    renderTrending([]);
  } finally {
    if (loader) loader.style.display = 'none';
  }
}

function updateKPIs(data) {
  const total    = data.total_analyzed || 0;
  const sentData = data.sentiment?.data || [0, 0, 0];

  // Top trending topic — from Google Trends array
  const topTrend = Array.isArray(data.trends) && data.trends.length
    ? data.trends[0]?.topic || data.trends[0] || '—'
    : (data.trends?.labels || [])[0] || '—';

  animNum('totalCount', 0, total, 800);
  animNum('kpi-total',    0, total,       800);
  animNum('kpi-positive', 0, sentData[0], 900);
  animNum('kpi-negative', 0, sentData[2], 1000);

  const wordEl = document.getElementById('kpi-top-word');
  if (wordEl) {
    const topStr = typeof topTrend === 'string' ? topTrend : (topTrend.topic || '—');
    wordEl.textContent = '#' + topStr.replace(/\s+/g, '');
  }
}

function animNum(id, from, to, dur) {
  const el = document.getElementById(id);
  if (!el) return;
  const start = performance.now();
  const step = now => {
    const t = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ── Trending Topics renderer ───────────────────────────────── */
function renderTrending(trends) {
  const container = document.getElementById('trendingTopics');
  if (!container) return;

  // Normalise both formats: array of {topic,rank,region} OR legacy {labels,data}
  let items = [];
  if (Array.isArray(trends)) {
    items = trends;
  } else if (trends && trends.labels) {
    items = (trends.labels || []).map((label, i) => ({
      topic:  label,
      rank:   i + 1,
      region: 'India',
      count:  (trends.data || [])[i] || 0,
    }));
  }

  if (!items.length) {
    container.innerHTML = `
      <div class="trending-error">
        <i class="fa-solid fa-triangle-exclamation" style="color:#d97706;margin-bottom:.4rem;font-size:1.1rem"></i><br>
        Trends unavailable — RSS feeds unreachable
      </div>`;
    return;
  }

  const REGION_COLORS = {
    India:    { bg: 'rgba(37,99,235,.1)',   fg: '#2563eb' },
    World:    { bg: 'rgba(5,150,105,.1)',   fg: '#059669' },
    Business: { bg: 'rgba(217,119,6,.1)',   fg: '#d97706' },
    Tech:     { bg: 'rgba(124,58,237,.1)',  fg: '#7c3aed' },
    Science:  { bg: 'rgba(8,145,178,.1)',   fg: '#0891b2' },
    Health:   { bg: 'rgba(220,38,38,.1)',   fg: '#dc2626' },
    Sports:   { bg: 'rgba(22,163,74,.1)',   fg: '#16a34a' },
    Global:   { bg: 'rgba(107,114,128,.1)', fg: '#6b7280' },
  };

  container.innerHTML = items.slice(0, 20).map((t, i) => {
    const topic   = typeof t === 'string' ? t : (t.topic || t);
    const region  = (typeof t === 'object' && t.region) ? t.region : 'Global';
    const rank    = i + 1;
    const hashtag = '#' + topic.replace(/\s+/g, '');
    const isHot   = rank <= 3;
    const col     = REGION_COLORS[region] || REGION_COLORS.Global;

    return `
      <div class="trend-item">
        <span class="${rank <= 3 ? 'trend-rank top3' : 'trend-rank'}">${rank}</span>
        <div class="trend-body">
          <span class="trend-hashtag">${hashtag}</span>
          <span class="trend-label">${topic}</span>
        </div>
        <span class="trend-region-badge" style="background:${col.bg};color:${col.fg};border-color:${col.fg}33">${region}</span>
        ${isHot ? '<span class="trend-badge hot">🔥</span>' : ''}
      </div>`;
  }).join('');
}

/* ── Chart palette — works on white bg ─────────────────────── */
const PALETTE = [
  '#2563eb','#7c3aed','#059669','#d97706',
  '#dc2626','#0891b2','#db2777','#65a30d',
  '#9333ea','#0d9488',
];

function baseOpts(extra = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 900, easing: 'easeOutQuart' },
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#4b5563',
          padding: 14,
          font: { family: 'Inter', size: 11 },
        },
      },
      tooltip: {
        backgroundColor: '#ffffff',
        borderColor: '#e2e5ef',
        borderWidth: 1,
        titleColor: '#111827',
        bodyColor: '#4b5563',
        padding: 10,
        cornerRadius: 8,
        boxShadow: '0 4px 12px rgba(0,0,0,.1)',
      },
    },
    ...extra,
  };
}

function renderCharts(data) {
  /* 1. Polar area — Topic distribution */
  const ctxCat = document.getElementById('categoryChart')?.getContext('2d');
  if (ctxCat) {
    if (catChart) catChart.destroy();
    catChart = new Chart(ctxCat, {
      type: 'polarArea',
      data: {
        labels: data.categories.labels,
        datasets: [{
          data: data.categories.data,
          backgroundColor: PALETTE.map(c => c + '33'),
          borderColor:     PALETTE,
          borderWidth: 2,
        }],
      },
      options: baseOpts({
        scales: {
          r: {
            grid:  { color: 'rgba(0,0,0,.06)' },
            ticks: { display: false },
          },
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#4b5563', padding: 12, font: { size: 11 } },
          },
        },
      }),
    });
  }

  /* 2. Doughnut — Sentiment */
  const ctxSent = document.getElementById('sentimentChart')?.getContext('2d');
  if (ctxSent) {
    if (sentChart) sentChart.destroy();
    sentChart = new Chart(ctxSent, {
      type: 'doughnut',
      data: {
        labels: data.sentiment.labels,
        datasets: [{
          data: data.sentiment.data,
          backgroundColor: ['#05966922','#6b728022','#dc262622'],
          borderColor:     ['#059669',  '#6b7280',  '#dc2626'],
          borderWidth: 2.5,
          hoverOffset: 8,
        }],
      },
      options: baseOpts({
        cutout: '65%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#4b5563', padding: 16, font: { size: 11 } },
          },
        },
      }),
    });
  }
}

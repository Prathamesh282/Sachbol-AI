/* ── SachBol AI — Main Script ──────────────────────────────── */

/* ── 1. Ambient light grid canvas ───────────────────────────── */
(function initGrid() {
  const canvas = document.getElementById('gridCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'rgba(37,99,235,0.06)';
    ctx.lineWidth = 1;
    const step = 60;
    for (let x = 0; x <= canvas.width; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = 0; y <= canvas.height; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }
  }
  draw();
  window.addEventListener('resize', draw);
})();

/* ── 2. Theme toggle — persists via localStorage ─────────────── */
(function initTheme() {
  // Respect saved preference, then OS preference
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

/* ── 3. System health check ─────────────────────────────────── */
async function checkHealth() {
  const pill = document.getElementById('statusPill');
  const txt  = document.getElementById('statusText');
  if (!pill) return;

  try {
    const res  = await fetch('/api/health');
    const data = await res.json();

    if (data.status === 'ok') {
      const ens  = data.ensemble || {};
      const qwen = ens.qwen_vl_8b?.available ?? ens.qwen_3b?.available;

      if (qwen) {
        pill.className = 'status-pill';
        txt.textContent = 'All models online';
      } else {
        pill.className = 'status-pill warn';
        txt.textContent = 'Groq online · Vision model offline';
      }
    } else {
      pill.className = 'status-pill error';
      txt.textContent = 'System error';
    }
  } catch {
    pill.className = 'status-pill warn';
    txt.textContent = 'Status unknown';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadFeed();
  document.getElementById('urlInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') runAgent();
  });
});

/* ── 3. Utilities ───────────────────────────────────────────── */
function clearInput() {
  document.getElementById('urlInput').value = '';
  document.getElementById('urlInput').focus();
}

function showError(msg) {
  const banner = document.getElementById('errorBanner');
  const msgEl  = document.getElementById('errorMsg');
  if (banner && msgEl) {
    msgEl.textContent = msg;
    banner.classList.remove('hidden');
  }
}
function hideError() {
  document.getElementById('errorBanner')?.classList.add('hidden');
}

/* ── 4. Pipeline animation ──────────────────────────────────── */
function startPipeline() {
  const steps = [0, 1, 2, 3];
  steps.forEach(i => {
    const el = document.getElementById(`ps-${i}`);
    if (el) {
      el.classList.remove('active','done');
      const fill = el.querySelector('.ps-fill');
      if (fill) fill.style.width = '0%';
    }
  });

  document.getElementById('pipeline')?.classList.remove('hidden');
  hideError();
  document.getElementById('results')?.classList.add('hidden');

  const TIMING = [0, 2500, 5000, 8000];
  const FILL_T  = [2000, 2500, 3500, 2000];

  steps.forEach(i => {
    setTimeout(() => {
      if (i > 0) {
        const prev = document.getElementById(`ps-${i-1}`);
        if (prev) { prev.classList.remove('active'); prev.classList.add('done'); }
      }
      const el = document.getElementById(`ps-${i}`);
      if (!el) return;
      el.classList.add('active');
      const fill = el.querySelector('.ps-fill');
      if (fill) {
        setTimeout(() => { fill.style.width = '100%'; fill.style.transition = `width ${FILL_T[i]}ms ease`; }, 80);
      }
    }, TIMING[i]);
  });
}

function stopPipeline() {
  document.getElementById('pipeline')?.classList.add('hidden');
  [0,1,2,3].forEach(i => {
    const el = document.getElementById(`ps-${i}`);
    if (el) { el.classList.remove('active','done'); }
  });
}

/* ── 5. Main agent call ─────────────────────────────────────── */
// Called from the manual URL input box — no title hint
async function runAgent(hintTitle = '') {
  const input = document.getElementById('urlInput');
  const url   = (input?.value || '').trim();

  if (!url) {
    input?.focus(); showError('Please paste a news article URL first.'); return;
  }
  if (!/^https?:\/\//i.test(url)) {
    showError('URL must start with http:// or https://'); return;
  }

  const btn = document.getElementById('verifyBtn');
  btn.disabled = true;

  // Show spinner in reasoning box and hide stale results
  const spinner = document.getElementById('reasoningSpinner');
  const rText   = document.getElementById('reasoningText');
  if (spinner) spinner.classList.remove('hidden');
  if (rText)   rText.innerHTML = '';
  document.getElementById('results')?.classList.add('hidden');

  startPipeline();

  try {
    const res  = await fetch('/api/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, title: hintTitle }),
    });
    const data = await res.json();

    stopPipeline();
    btn.disabled = false;
    if (spinner) spinner.classList.add('hidden');

    if (data.error) {
      showError('Verification failed: ' + data.error);
    } else {
      displayResults(data);
    }
  } catch (err) {
    stopPipeline();
    btn.disabled = false;
    if (spinner) spinner.classList.add('hidden');
    showError('Network error: ' + err.message);
  }
}

/* ── 6. Result display ──────────────────────────────────────── */
const VERDICT_META = {
  VERIFIED:     { cls: 'v-safe',    icon: 'fa-circle-check',        color: '#059669' },
  MOSTLY_TRUE:  { cls: 'v-caution', icon: 'fa-triangle-exclamation', color: '#d97706' },
  MOSTLY_FALSE: { cls: 'v-caution', icon: 'fa-triangle-exclamation', color: '#d97706' },
  FALSE:        { cls: 'v-danger',  icon: 'fa-circle-xmark',        color: '#dc2626' },
  UNVERIFIED:   { cls: 'v-unknown', icon: 'fa-circle-question',     color: '#6b7280' },
};

function displayResults(data) {
  const verdict = (data.verdict || 'UNVERIFIED').toUpperCase();
  const meta    = VERDICT_META[verdict] || VERDICT_META.UNVERIFIED;
  const conf    = Math.max(0, Math.min(100, data.confidence || 0));

  /* Badge */
  const badge = document.getElementById('verdictBadge');
  badge.className = `verdict-badge ${meta.cls}`;
  document.getElementById('verdictIcon').innerHTML = `<i class="fa-solid ${meta.icon}"></i>`;
  document.getElementById('verdictLabel').textContent = verdict.replace('_', ' ');

  /* Confidence ring */
  const ring = document.getElementById('ringFill');
  const circ = 2 * Math.PI * 34;
  ring.style.stroke = meta.color;
  animateNumber('confNum', 0, conf, 1200);
  setTimeout(() => {
    ring.style.strokeDashoffset = circ - (circ * conf / 100);
  }, 80);

  /* Text */
  document.getElementById('articleTitle').textContent = data.title || '';

  /* Social media post preview — only for Twitter/Reddit/Instagram/etc. */
  const preview = document.getElementById('socialPreview');
  if (preview) {
    const mediaItems = data.media || [];
    const hasPlatform = !!data.platform;  // only show if it's a real social media URL
    if (mediaItems.length && hasPlatform) {
      const PLATFORM_ICONS = {
        twitter:   'fa-brands fa-x-twitter',
        reddit:    'fa-brands fa-reddit-alien',
        instagram: 'fa-brands fa-instagram',
        facebook:  'fa-brands fa-facebook',
        tiktok:    'fa-brands fa-tiktok',
      };
      const icon   = PLATFORM_ICONS[data.platform] || 'fa-solid fa-photo-film';
      const label  = data.platform
        ? data.platform.charAt(0).toUpperCase() + data.platform.slice(1)
        : 'Post';
      const author = data.author ? ` · ${escHtml(data.author)}` : '';

      preview.innerHTML = `
        <div class="sp-header">
          <i class="${icon}"></i>
          <span>${label}${author} — scraped media</span>
        </div>
        <div class="sp-media-row">
          ${mediaItems.slice(0, 4).map(m => {
            // agent.py returns either a plain URL string or {url, type} object
            const url  = (typeof m === 'string') ? m : (m.url || '');
            const type = (typeof m === 'object' && m.type === 'video') ? 'video' : 'image';
            if (!url) return '';
            return type === 'video'
              ? `<video class="sp-media-item" src="${escHtml(url)}"
                   poster="${escHtml((m.thumb)||'')}"
                   controls muted playsinline></video>`
              : `<img class="sp-media-item" src="${escHtml(url)}"
                   loading="lazy"
                   onerror="this.style.display='none'" />`;
          }).join('')}
        </div>`;
      preview.classList.remove('hidden');
    } else {
      preview.classList.add('hidden');
    }
  }

  // Render Groq reasoning
  const groqRaw = data.groq_reasoning || data.reasoning || 'No reasoning provided.';
  const groqEl  = document.getElementById('reasoningGroq');
  if (groqEl) {
    groqEl.innerHTML = groqRaw
      .split(/\n\n+/)
      .map(p => p.trim())
      .filter(p => p.length > 0)
      .map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`)
      .join('');
  }

  // Render Qwen reasoning as a compact inline strip below LLaMA reasoning
  const qwenRaw      = data.qwen_reasoning || '';
  const stripEl      = document.getElementById('qwenInsightStrip');
  const stripTextEl  = document.getElementById('qwenInsightText');
  const qwenOffline  = document.getElementById('reasoningQwenOffline');

  if (qwenRaw && stripEl && stripTextEl) {
    // Collapse to single line — Qwen gives one-liners anyway
    stripTextEl.textContent = qwenRaw.replace(/\n+/g, ' ').trim();
    stripEl.classList.remove('hidden');
    qwenOffline?.classList.add('hidden');
  } else {
    stripEl?.classList.add('hidden');
    if (!qwenRaw) {
      qwenOffline?.classList.remove('hidden');
    }
  }

  // Also keep legacy reasoningText in sync (in case other code depends on it)
  const rEl = document.getElementById('reasoningText');
  if (rEl) rEl.innerHTML = groqEl ? groqEl.innerHTML : groqRaw;

  // Hide spinner (safety net)
  document.getElementById('reasoningSpinner')?.classList.add('hidden');

  /* Ensemble breakdown */
  const eb = data.ensemble_breakdown || {};
  fillEnsembleCard('groq',    eb.groq,       false);
  fillEnsembleCard('bge',     eb.deberta,    eb.deberta?.available === false);
  fillEnsembleCard('qwen',    eb.qwen_vl_8b, eb.qwen_vl_8b?.available === false);
  fillCredibility(eb.credibility || {});

  /* Sources */
  fillSources(data.sources || []);

  /* Show */
  document.getElementById('results').classList.remove('hidden');
  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function fillEnsembleCard(key, d, offline) {
  if (!d) return;
  const vEl   = document.getElementById(`em-${key}-v`);
  const cEl   = document.getElementById(`em-${key}-c`);
  const bar   = document.getElementById(`em-${key}-bar`);
  const offEl = document.getElementById(`em-${key}-offline`);

  const verdict = (d.verdict || 'UNVERIFIED').replace('_',' ');
  const conf    = d.confidence || 0;

  if (vEl) {
    vEl.textContent = verdict;
    const m = VERDICT_META[(d.verdict||'').toUpperCase()] || VERDICT_META.UNVERIFIED;
    vEl.style.color = m.color;
  }
  if (cEl) cEl.textContent = conf + '%';
  if (bar) setTimeout(() => { bar.style.width = conf + '%'; }, 150);
  if (offEl) { offline ? offEl.classList.remove('hidden') : offEl.classList.add('hidden'); }
}

function fillCredibility(cred) {
  const trusted  = cred.trusted_count    ?? '—';
  const score    = cred.credibility_score ?? '—';
  const debunked = cred.debunked;

  document.getElementById('cred-trusted').textContent = trusted;
  document.getElementById('cred-score').textContent   = typeof score === 'number' ? score + '/100' : score;

  const badge = document.getElementById('cred-debunked');
  if (badge) {
    badge.textContent = debunked ? 'YES' : 'NO';
    badge.className   = 'cred-badge ' + (debunked ? 'yes' : 'no');
  }
}

function fillSources(sources) {
  const list = document.getElementById('sourcesList');
  const cnt  = document.getElementById('srcCount');
  if (!list) return;

  if (cnt) cnt.textContent = sources.length + ' sources';

  list.innerHTML = '';
  if (!sources.length) {
    list.innerHTML = '<p style="color:var(--muted);font-size:.85rem">No corroborating evidence found for this query.</p>';
    return;
  }

  sources.forEach(src => {
    const a = document.createElement('a');
    a.className = 'source-card';
    a.href = src.link || '#';
    a.target = '_blank';
    a.rel = 'noopener noreferrer';

    let domain = '';
    try { domain = new URL(src.link).hostname.replace('www.',''); } catch {}

    a.innerHTML = `
      <div class="sc-domain"><i class="fa-solid fa-link"></i>${domain}</div>
      <div class="sc-title">${escHtml(src.title || '')}</div>
      <div class="sc-snippet">${escHtml(src.snippet || '')}</div>
    `;
    list.appendChild(a);
  });
}

/* ── 7. Animated counter ────────────────────────────────────── */
function animateNumber(id, from, to, duration) {
  const el = document.getElementById(id);
  if (!el) return;
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ── 8. News feed ───────────────────────────────────────────── */
const IMG_FALLBACK = 'https://placehold.co/400x220/f0f2f8/2563eb?text=SachBol';

async function loadFeed() {
  const grid = document.getElementById('feedGrid');
  const btn  = document.querySelector('.refresh-btn');
  if (!grid) return;

  // Show skeleton
  grid.innerHTML = Array.from({length: 6}, () => `
    <div class="skeleton">
      <div class="skel-img"></div>
      <div class="skel-body">
        <div class="skel-line" style="width:80%"></div>
        <div class="skel-line" style="width:55%"></div>
        <div class="skel-line" style="width:65%"></div>
      </div>
    </div>
  `).join('');

  if (btn) btn.classList.add('spinning');

  try {
    const res  = await fetch('/api/feed');
    const data = await res.json();

    grid.innerHTML = '';
    if (!data.length) {
      grid.innerHTML = '<p style="color:var(--muted)">Unable to load feed. Check your connection.</p>';
      return;
    }

    data.forEach((item, idx) => {
      const card = document.createElement('div');
      card.className = 'news-card';
      card.style.animationDelay = `${idx * 0.05}s`;

      const safeTitle = escHtml(item.headline || 'No title');
      const src       = escHtml(item.source || 'Unknown');
      const timeStr   = formatTime(item.time || '');
      const link      = item.link || '#';
      const img       = item.image || IMG_FALLBACK;
      // Encode for safe use in onclick attribute
      const safeLink  = link.replace(/'/g, "\\'");
      const safeHint  = (item.headline || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');

      card.innerHTML = `
        <img src="${img}"
             class="news-img"
             loading="lazy"
             onerror="this.src='${IMG_FALLBACK}'"
        />
        <div class="news-body">
          <div class="news-title">${safeTitle}</div>
          <div class="news-meta">
            <span class="news-source">${src}</span>
            <span>${timeStr}</span>
          </div>
          <div class="news-actions">
            <a class="read-btn" href="${link}" target="_blank" rel="noopener">
              <i class="fa-solid fa-arrow-up-right-from-square"></i> Read
            </a>
            <button class="quick-verify-btn"
                    onclick="quickVerify('${safeLink}', '${safeHint}')">
              <i class="fa-solid fa-bolt"></i> Verify
            </button>
          </div>
        </div>
      `;
      grid.appendChild(card);
    });

  } catch (err) {
    grid.innerHTML = `<p style="color:var(--muted)">Feed error: ${err.message}</p>`;
  } finally {
    if (btn) btn.classList.remove('spinning');
  }
}

/* quickVerify — fills URL input and triggers agent with title hint */
function quickVerify(url, hintTitle) {
  const input = document.getElementById('urlInput');
  if (input) {
    input.value = url;
    window.scrollTo({ top: 0, behavior: 'smooth' });
    setTimeout(() => runAgent(hintTitle || ''), 400);
  }
}

/* ── 9. Helpers ─────────────────────────────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function formatTime(raw) {
  if (!raw) return '';
  try {
    const d = new Date(raw);
    if (isNaN(d)) return raw;
    const now = Date.now();
    const diff = Math.floor((now - d) / 60000);
    if (diff < 1)   return 'Just now';
    if (diff < 60)  return `${diff}m ago`;
    if (diff < 1440) return `${Math.floor(diff/60)}h ago`;
    return d.toLocaleDateString('en-IN', { day:'numeric', month:'short' });
  } catch { return raw; }
}

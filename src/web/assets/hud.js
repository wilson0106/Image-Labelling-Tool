/* ILT web — client-side renderer.  Incremental DOM updates only: the server
   sends deltas (new log lines, changed state) and nothing is rebuilt. */
(function () {
  const ILT = window.ILT = window.ILT || {};
  ILT.notifSeq = ILT.notifSeq || 0;
  ILT.lastHash = '';
  const LOG_MAX = 6000;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  ILT.applyTheme = function (t) {
    if (!t) return;
    const r = document.documentElement.style;
    r.setProperty('--accent', t.accent); r.setProperty('--accent-2', t.accent2);
    r.setProperty('--bg', t.bg); r.setProperty('--ink', t.ink);
    r.setProperty('--glass', t.glass); r.setProperty('--radius', t.radius + 'px');
    r.setProperty('--f-display', `"${t.font_display}", "DejaVu Sans", system-ui, sans-serif`);
    r.setProperty('--f-body', `"${t.font_body}", "DejaVu Sans", system-ui, sans-serif`);
    r.setProperty('--fs', (14 * (t.font_scale || 1)).toFixed(1) + 'px');
    document.querySelectorAll('.color-pick').forEach((p) => {
      const k = (p.dataset.target || '').slice(3);
      if (t[k] && p.value !== t[k]) p.value = t[k];
    });
  };

  // Native colour pickers feed the matching text field of the Appearance page.
  document.addEventListener('input', (e) => {
    const el = e.target;
    if (el.classList && el.classList.contains('color-pick') && window.dash_clientside && window.dash_clientside.set_props) {
      window.dash_clientside.set_props(el.dataset.target, {value: el.value});
    }
  });

  function appendLog(lines, reset, epoch, boxId) {
    const box = $(boxId || 'exec-log');
    if (!box) return;
    if (reset) box.innerHTML = '';
    if (!lines || !lines.length) { box.dataset.epoch = epoch; return; }
    const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    const frag = document.createDocumentFragment();
    for (const [seq, stamp, tag, text] of lines) {
      const d = document.createElement('div');
      d.innerHTML = `<span class="t">[${stamp}]</span> <span class="${tag}">${esc(text)}</span>`;
      frag.appendChild(d);
      box.dataset.seq = seq;
    }
    box.appendChild(frag);
    let extra = box.childElementCount - LOG_MAX;
    while (extra-- > 0) box.removeChild(box.firstChild);
    box.dataset.epoch = epoch;
    if (stick) box.scrollTop = box.scrollHeight;
  }

  function drawCharts(charts, version) {
    if (!window.Plotly || !charts) return;
    const ink2 = '#a3adb5';
    const base = (extra) => Object.assign({
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: {l: 48, r: 12, t: 6, b: 34},
      font: {color: cssVar('--ink') || '#f4f7f8', size: 12},
      xaxis: {gridcolor: 'rgba(255,255,255,.06)', zeroline: false, tickfont: {color: ink2, size: 11}, title: {text: 'época', font: {color: ink2, size: 11}}},
      yaxis: {gridcolor: 'rgba(255,255,255,.06)', zeroline: false, tickfont: {color: ink2, size: 11}, rangemode: 'tozero'},
      legend: {orientation: 'h', y: 1.16, x: 0, font: {color: ink2, size: 11}},
      hovermode: 'x unified', hoverlabel: {bgcolor: '#0b1216'},
    }, extra || {});
    const cfg = {displaylogo: false, responsive: true};
    const A = cssVar('--accent') || '#22d3ee', W = '#fbbf24', OK = '#34d399', P = '#a78bfa';
    const tr = (pair, name, color, fill) => ({x: pair[0], y: pair[1], name, mode: 'lines+markers', line: {color, width: 2}, marker: {size: 5},
      fill: fill ? 'tozeroy' : 'none', fillcolor: fill ? color + '14' : undefined});
    const L = window.ILT.labels || {};
    const lossEl = $('chart-loss'), valEl = $('chart-valid');
    if (lossEl && charts.loss) {
      Plotly.react(lossEl, [tr(charts.loss.train_loss, L.train_loss || 'train loss', A, true),
                            tr(charts.loss.valid_loss, L.valid_loss || 'valid loss', W, false)], base(), cfg);
      lossEl.dataset.v = version;
    }
    if (valEl && charts.valid) {
      Plotly.react(valEl, [tr(charts.valid.valid_acc, L.valid_acc || 'valid acc', OK, true),
                           tr(charts.valid.valid_macro, L.valid_macro || 'valid macro', P, false)],
                   base({yaxis: {gridcolor: 'rgba(255,255,255,.06)', ticksuffix: '%', tickfont: {color: ink2, size: 11}, range: [0, 100]}}), cfg);
      valEl.dataset.v = version;
    }
  }

  function renderState(s) {
    const app = document.querySelector('.app');
    if (app) app.classList.toggle('locked', !!s.running);
    const st = $('exec-status');
    if (st) { st.textContent = s.status; st.style.color = s.status_color; }
    const el = $('exec-elapsed'); if (el) el.textContent = s.elapsed;
    const pct = $('exec-pct'); if (pct) pct.textContent = Math.round(s.pct) + '%';
    const bar = $('exec-bar'); if (bar) bar.style.width = Math.max(0, Math.min(100, s.pct)) + '%';
    const live = $('exec-live'); if (live) live.textContent = s.live || '';
    const stages = $('exec-stages');
    if (stages) {
      const sig = JSON.stringify(s.stages);
      if (stages.dataset.sig !== sig) {
        stages.innerHTML = s.stages.map(([label, state], i) =>
          `<div class="stage ${state}"><small>${state === 'done' ? '✓' : state === 'skipped' ? '—' : i + 1}</small><b>${esc(label)}</b></div>`).join('');
        stages.dataset.sig = sig;
      }
    }
    for (const [key, [, value]] of Object.entries(s.metrics || {})) {
      const t = document.querySelector(`#m-${key} .tv`); if (t) t.textContent = value;
    }
    const startBtns = document.querySelectorAll('.start-btn');
    startBtns.forEach((b) => { b.disabled = !!s.running; });
    document.querySelectorAll('.stop-btn').forEach((b) => { b.disabled = !s.running; });
    const sys = $('sys-status');
    if (sys) { sys.textContent = s.system; sys.parentElement.parentElement.classList.toggle('busy', !!s.running); }
  }

  function renderIat(s) {
    if (!s) return;
    const st = $('iat-status');
    if (st) { st.textContent = s.status; st.style.color = s.running ? 'var(--accent)' : 'var(--ok)'; }
    const op = $('iat-op'); if (op) op.textContent = s.operation || '';
    const el = $('iat-elapsed'); if (el) el.textContent = s.elapsed;
    const pct = $('iat-pct'); if (pct) pct.textContent = s.pct_text;
    const bar = $('iat-bar'); if (bar) bar.style.width = Math.max(0, Math.min(100, s.pct)) + '%';
    document.querySelectorAll('.iat-stop-btn').forEach((b) => { b.disabled = !s.running; });
  }

  // Gallery: one delegated handler for every card (fast with 500 cards).
  const setp = (id, props) => window.dash_clientside && window.dash_clientside.set_props && window.dash_clientside.set_props(id, props);
  document.addEventListener('click', (e) => {
    if (!e.target.closest) return;
    const open = e.target.closest('.gwrap .open');
    if (open) { setp('gopen', {data: {gid: open.dataset.gid, t: Date.now()}}); return; }
    const card = e.target.closest('.gcard[data-gid]');
    if (!card) return;
    card.classList.toggle('sel');  // instant feedback; the server confirms below
    const page = Array.from(document.querySelectorAll('.gcard[data-gid]')).map((c) => c.dataset.gid);
    setp('gclick', {data: {gid: card.dataset.gid, shift: !!e.shiftKey, page, t: Date.now()}});
  });

  // Temporal › groups matrix: one delegated handler for every cell and suggestion.
  document.addEventListener('click', (e) => {
    if (!e.target.closest) return;
    const cell = e.target.closest('[data-gmx]');
    if (cell) setp('gmx-click', {data: {k: cell.dataset.gmx, t: Date.now()}});
  });

  // Gallery › full image: zoom and a scale bar computed from the pixel size
  // (µm/pixel) and the zoom in use; the length is always 1, 2 or 5 × 10^n
  // (0.1 µm … 50 mm) and takes about a quarter of the visible width.
  const NICE = [];
  for (let ex = -1; ex <= 4; ex++) for (const m of [1, 2, 5]) NICE.push(+(m * Math.pow(10, ex)).toPrecision(3));
  const num = (v) => Number(v).toLocaleString('pt-BR', {maximumFractionDigits: 2});
  const lengthLabel = (um) => (um >= 1000 ? `${num(um / 1000)} mm` : `${num(um)} µm`);
  function fiBar(box, z) {
    const bar = box.querySelector('.fi-bar');
    const um = parseFloat(box.dataset.um);
    if (!bar) return;
    if (!(um > 0)) { bar.hidden = true; return; }
    const scroll = box.querySelector('.fi-scroll');
    const img = box.querySelector('img.fi-img');
    const pxPerUm = z / um;
    const shown = Math.min(scroll.clientWidth, img.naturalWidth * z);
    const target = Math.max(60, Math.min(240, shown * 0.25));
    let best = NICE[0];
    for (const v of NICE) if (v * pxPerUm <= target) best = v;
    bar.style.setProperty('--w', Math.max(2, best * pxPerUm).toFixed(1) + 'px');
    bar.querySelector('span').textContent = lengthLabel(best);
    bar.querySelector('small').textContent = `${num(um)} µm/pixel · zoom ${num(Math.round(z * 100) / 100)}×`;
  }
  function fiApply(box) {
    const img = box.querySelector('img.fi-img');
    const scroll = box.querySelector('.fi-scroll');
    if (!img || !scroll || !img.naturalWidth) return;
    const mode = box.dataset.z || 'fit';
    let z = parseFloat(mode) || 1;
    if (mode === 'fit') {
      z = Math.min((scroll.clientWidth - 24) / img.naturalWidth, (scroll.clientHeight - 24) / img.naturalHeight, 8);
      z = Math.max(0.05, z);
    }
    img.style.width = Math.round(img.naturalWidth * z) + 'px';
    img.style.height = Math.round(img.naturalHeight * z) + 'px';
    img.classList.toggle('px', z >= 3);
    box.dataset.zv = String(z);
    const label = box.querySelector('.fi-zoom');
    if (label) label.textContent = `${num(Math.round(z * 100) / 100)}×`;
    box.querySelectorAll('[data-fz]').forEach((b) => b.classList.toggle('on', b.dataset.fz === mode));
    fiBar(box, z);
  }
  const fiInit = () => {
    document.querySelectorAll('.fi-box:not([data-init])').forEach((box) => {
      box.dataset.init = '1';
      const img = box.querySelector('img.fi-img');
      if (!img) return;
      if (img.complete && img.naturalWidth) fiApply(box);
      img.addEventListener('load', () => fiApply(box));
    });
  };
  new MutationObserver(fiInit).observe(document.documentElement, {subtree: true, childList: true});
  window.addEventListener('resize', () => document.querySelectorAll('.fi-box[data-init]').forEach(fiApply));
  document.addEventListener('click', (e) => {
    if (!e.target.closest) return;
    const box = e.target.closest('.fi-box');
    if (!box) return;
    const zb = e.target.closest('[data-fz]'), sb = e.target.closest('[data-fzs]'), bb = e.target.closest('[data-fbar]');
    if (zb) { box.dataset.z = zb.dataset.fz; fiApply(box); }
    else if (sb) {
      const z = (parseFloat(box.dataset.zv) || 1) * parseFloat(sb.dataset.fzs);
      box.dataset.z = String(Math.max(0.1, Math.min(32, z)));
      fiApply(box);
    } else if (bb) {
      const bar = box.querySelector('.fi-bar');
      if (bar) { bar.hidden = !bar.hidden; bb.classList.toggle('on', !bar.hidden); }
    }
  });
  document.addEventListener('wheel', (e) => {
    if (!e.ctrlKey || !e.target.closest) return;
    const box = e.target.closest('.fi-box');
    if (!box) return;
    e.preventDefault();
    const z = (parseFloat(box.dataset.zv) || 1) * (e.deltaY < 0 ? 1.15 : 1 / 1.15);
    box.dataset.z = String(Math.max(0.1, Math.min(32, z)));
    fiApply(box);
  }, {passive: false});
  // Lazy thumbnails: the real URL is in data-src; it is assigned when the card
  // comes within 600 px of the viewport (500-card pages load only what shows).
  let lazyIO = null;
  const lazyThumbs = () => {
    const imgs = document.querySelectorAll('img[data-src]');
    if (!imgs.length) return;
    if (!('IntersectionObserver' in window)) {
      imgs.forEach((i) => { i.src = i.dataset.src; i.removeAttribute('data-src'); });
      return;
    }
    if (!lazyIO) {
      lazyIO = new IntersectionObserver((entries) => {
        for (const en of entries) {
          if (!en.isIntersecting) continue;
          const i = en.target;
          if (i.dataset.src) { i.src = i.dataset.src; i.removeAttribute('data-src'); }
          lazyIO.unobserve(i);
        }
      }, {rootMargin: '600px 0px'});
    }
    imgs.forEach((i) => lazyIO.observe(i));
  };

  // After every Gallery redraw, make the card frames match the server state
  // (data-sel).  React does not know about frames toggled locally on click.
  let galToken = null;
  const syncGallery = () => {
    lazyThumbs();
    const gal = document.querySelector('.gal[data-render]');
    if (!gal || gal.dataset.render === galToken) return;
    galToken = gal.dataset.render;
    gal.querySelectorAll('.gcard[data-gid]').forEach((c) => c.classList.toggle('sel', c.dataset.sel === '1'));
  };
  new MutationObserver(syncGallery).observe(document.documentElement,
    {subtree: true, childList: true, attributes: true, attributeFilter: ['data-render']});

  // Modal close (✕ or backdrop) also works without a server round-trip.
  document.addEventListener('click', (e) => {
    if (!e.target.closest) return;
    if (e.target.closest('.modal-wrap .modal-backdrop') || e.target.closest('.modal-wrap .modal-x')) {
      setp('modal-host', {children: []});
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.querySelector('#modal-host .modal-wrap')) setp('modal-host', {children: []});
  });

  function renderGauges(g) {
    if (!g) return;
    const set = (id, p, label, sub) => {
      const el = $(id); if (!el) return;
      el.style.setProperty('--p', Math.max(0, Math.min(100, p || 0)));
      el.querySelector('b').textContent = label; el.querySelector('small').textContent = sub;
    };
    set('gauge-a', g.a_pct, g.a_label, g.a_sub);
    set('gauge-b', g.b_pct, g.b_label, g.b_sub);
  }

  function toast(kind, title, msg) {
    let host = document.querySelector('.toasts');
    if (!host) { host = document.createElement('div'); host.className = 'toasts'; document.body.appendChild(host); }
    const t = document.createElement('div');
    t.className = `toast ${kind}`;
    t.innerHTML = `<b>${esc(title)}</b>${esc(msg)}`;
    t.addEventListener('click', () => t.remove());
    host.appendChild(t);
    setTimeout(() => t.remove(), kind === 'error' ? 14000 : 7000);
  }
  ILT.toast = toast;

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    ilt: {
      clientState: function (_n) {
        const box = $('exec-log');
        const chart = $('chart-loss') || $('chart-valid');
        return {
          has_log: !!box,
          seq: box ? Number(box.dataset.seq || 0) : 0,
          epoch: box ? Number(box.dataset.epoch || -1) : -1,
          has_charts: !!chart,
          charts_v: chart ? Number(chart.dataset.v || -1) : -1,
          notif: ILT.notifSeq,
          hash: ILT.lastHash,
          has_iat_log: !!$('iat-log'),
          iat_seq: $('iat-log') ? Number($('iat-log').dataset.seq || 0) : 0,
          iat_epoch: $('iat-log') ? Number($('iat-log').dataset.epoch || -1) : -1,
        };
      },
      gallerySel: function (d) {
        if (!d) return window.dash_clientside.no_update;
        const on = new Set(d.checked || []);
        // Only the frame changes here; data-sel stays as rendered by the server.
        document.querySelectorAll('.gcard[data-gid]').forEach((c) => c.classList.toggle('sel', on.has(c.dataset.gid)));
        const meta = document.querySelector('[id*="gal-meta"]');
        if (meta && d.meta) meta.textContent = d.meta;
        return window.dash_clientside.no_update;
      },
      galleryClick: function () {
        const cc = window.dash_clientside.callback_context;
        if (!cc || !cc.triggered || !cc.triggered.length) return window.dash_clientside.no_update;
        const t = cc.triggered[0];
        if (!t.value) return window.dash_clientside.no_update;
        let id;
        try { id = JSON.parse(t.prop_id.split('.n_clicks')[0]); } catch (e) { return window.dash_clientside.no_update; }
        return {gid: id.gid, shift: !!ILT.shift, t: Date.now()};
      },
      render: function (payload) {
        if (!payload) return window.dash_clientside.no_update;
        if (payload.hash) ILT.lastHash = payload.hash;
        if (payload.state) renderState(payload.state);
        if (payload.log) appendLog(payload.log.lines, payload.log.reset, payload.log.epoch);
        if (payload.iat) renderIat(payload.iat);
        if (payload.iat_log) appendLog(payload.iat_log.lines, payload.iat_log.reset, payload.iat_log.epoch, 'iat-log');
        if (payload.charts) drawCharts(payload.charts, payload.charts_v);
        if (payload.gauges) renderGauges(payload.gauges);
        for (const [seq, kind, title, msg] of payload.notifs || []) {
          ILT.notifSeq = Math.max(ILT.notifSeq, seq); toast(kind, title, msg);
        }
        return window.dash_clientside.no_update;
      },
      theme: function (t) { ILT.applyTheme(t); return window.dash_clientside.no_update; },
      labels: function (l) { ILT.labels = l || {}; if (l && l.lang) document.documentElement.lang = l.lang; return window.dash_clientside.no_update; },
    },
  });
})();

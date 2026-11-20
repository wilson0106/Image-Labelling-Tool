/* ILT mascot games (easter egg): memory game and "find the mascot".
   Runs entirely in the browser.  Difficulty "Difícil" reproduces the classic
   games exactly (memory: grids 3→6, advance below 30 % errors, 620 ms reveal;
   hunt: 10×6 dense crowd with transformed decoys).  Every finished level/round
   is stored locally and plotted in the evolution chart. */
(function () {
  const ILT = window.ILT = window.ILT || {};
  const pt = () => (document.documentElement.lang || 'pt') !== 'en';
  const T = (a, b) => (pt() ? a : b);
  const rnd = (n) => Math.floor(Math.random() * n);
  const pick = (arr) => arr[rnd(arr.length)];
  const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = rnd(i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

  const MEMORY = {
    'Fácil':  {grids: [2, 3, 4], threshold: 0.50, reveal: 1100, peek: 1800, hues: 4},
    'Médio':  {grids: [3, 4, 5], threshold: 0.40, reveal: 850, peek: 900, hues: 6},
    'Difícil': {grids: [3, 4, 5, 6], threshold: 0.30, reveal: 620, peek: 0, hues: 10},
  };
  const HUNT = {
    'Fácil':  {cols: 5, rows: 4, size: 86, decoyTransform: 0.25, sameSource: false, rounds: 5},
    'Médio':  {cols: 8, rows: 5, size: 70, decoyTransform: 0.5, sameSource: true, rounds: 5},
    'Difícil': {cols: 10, rows: 6, size: 62, decoyTransform: 0.7, sameSource: true, rounds: 5},
  };
  const HUES = [0, 22, 43, 67, 91, 116, 145, 173, 201, 229];

  function files() {
    const img = document.getElementById('mascot-img');
    try { return JSON.parse(img && img.dataset.files || '[]'); } catch (e) { return []; }
  }
  const url = (f) => '/mascot/' + encodeURIComponent(f);

  // ---------- history (per viewer, local) ----------
  const KEY = 'ilt.games.history.v1';
  function history() { try { return JSON.parse(localStorage.getItem(KEY) || '[]'); } catch (e) { return []; } }
  function record(entry) {
    try { const h = history(); h.push(entry); localStorage.setItem(KEY, JSON.stringify(h.slice(-400))); } catch (e) { /* storage off */ }
  }
  function clearHistory(game) {
    try { localStorage.setItem(KEY, JSON.stringify(history().filter((h) => h.game !== game))); } catch (e) { /* ignore */ }
  }

  // ---------- modal shell ----------
  let shell = null;
  function close() { if (shell) { clearTimeout(shell._timer); shell.remove(); shell = null; } }
  function open(title) {
    close();
    shell = el('div', 'game-wrap');
    shell.innerHTML = `<div class="modal-backdrop"></div>
      <div class="hud modal-card wide game-card">
        <div class="hh"><h3>${title}</h3>
          <span class="meta"><span class="seg game-diff"></span><button class="btn sm icon game-close" title="Fechar">✕</button></span></div>
        <div class="game-top row"></div>
        <div class="game-board"></div>
        <div class="game-msg hint"></div>
        <div class="game-chart-head row"><b class="kicker">${T('Evolução', 'Progress')}</b>
          <span class="seg game-view"></span><button class="btn sm push game-reset">${T('Limpar histórico', 'Clear history')}</button></div>
        <div class="game-chart"></div>
      </div>`;
    document.body.appendChild(shell);
    shell.querySelector('.game-close').onclick = close;
    shell.querySelector('.modal-backdrop').onclick = close;
    return shell;
  }
  function segButtons(host, options, current, onPick) {
    host.innerHTML = '';
    options.forEach((o) => {
      const b = el('button', 'seg-btn' + (o === current ? ' on' : ''), o);
      b.onclick = () => onPick(o);
      host.appendChild(b);
    });
  }

  // ---------- evolution chart ----------
  function drawChart(game, view, live) {
    const host = shell && shell.querySelector('.game-chart');
    if (!host || !window.Plotly) return;
    const ink2 = '#a3adb5';
    const base = {paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: {l: 46, r: 46, t: 8, b: 40},
      font: {color: '#f4f7f8', size: 11}, legend: {orientation: 'h', y: 1.18, font: {color: ink2}},
      xaxis: {gridcolor: 'rgba(255,255,255,.06)', tickfont: {color: ink2}}, yaxis: {gridcolor: 'rgba(255,255,255,.06)', tickfont: {color: ink2}},
      hovermode: 'x unified'};
    let data, layout;
    if (view === T('Dentro do nível', 'Within level')) {
      const ev = (live && live.events) || [];
      const x = ev.map((_, i) => i + 1);
      let h = 0, e = 0; const hits = [], errs = [], secs = [];
      ev.forEach((v) => { if (v.hit) h++; else e++; hits.push(h); errs.push(e); secs.push(+(v.t / 1000).toFixed(1)); });
      data = [
        {x, y: hits, name: T('Acertos (acum.)', 'Hits (cum.)'), mode: 'lines+markers', line: {color: '#34d399', width: 2}},
        {x, y: errs, name: T('Erros (acum.)', 'Errors (cum.)'), mode: 'lines+markers', line: {color: '#f87171', width: 2}},
        {x, y: secs, name: T('Tempo (s)', 'Time (s)'), yaxis: 'y2', mode: 'lines', line: {color: '#22d3ee', width: 1.5, dash: 'dot'}},
      ];
      layout = Object.assign({}, base, {xaxis: Object.assign({}, base.xaxis, {title: {text: T('tentativa', 'attempt')}}),
        yaxis2: {overlaying: 'y', side: 'right', tickfont: {color: '#22d3ee'}, showgrid: false, title: {text: 's', font: {color: '#22d3ee'}}}});
    } else {
      const rows = history().filter((r) => r.game === game);
      const x = rows.map((r, i) => `${i + 1}. ${r.difficulty[0]}·${r.level}`);
      data = [
        {type: 'bar', x, y: rows.map((r) => r.hits), name: T('Acertos', 'Hits'), marker: {color: '#34d399'}},
        {type: 'bar', x, y: rows.map((r) => r.errors), name: T('Erros', 'Errors'), marker: {color: '#f87171'}},
        {x, y: rows.map((r) => r.attempts), name: T('Tentativas', 'Attempts'), mode: 'lines+markers', line: {color: '#fbbf24', width: 2}},
        {x, y: rows.map((r) => r.time), name: T('Tempo (s)', 'Time (s)'), yaxis: 'y2', mode: 'lines+markers', line: {color: '#22d3ee', width: 2}},
      ];
      layout = Object.assign({}, base, {barmode: 'group', bargap: .3,
        xaxis: Object.assign({}, base.xaxis, {title: {text: T('nível / rodada concluída', 'finished level / round')}, type: 'category'}),
        yaxis2: {overlaying: 'y', side: 'right', tickfont: {color: '#22d3ee'}, showgrid: false, title: {text: 's', font: {color: '#22d3ee'}}}});
      if (!rows.length) layout.annotations = [{text: T('Conclua um nível para ver a evolução.', 'Finish a level to see your progress.'),
        x: .5, y: .5, xref: 'paper', yref: 'paper', showarrow: false, font: {color: ink2}}];
    }
    Plotly.react(host, data, layout, {displaylogo: false, responsive: true, displayModeBar: false});
  }

  // ---------- memory game ----------
  function memory(difficulty) {
    const list = files();
    if (!list.length) return;
    const cfg = MEMORY[difficulty];
    const s = open(T('Jogo da memória', 'Memory game'));
    const st = {level: 0, view: T('Entre níveis', 'Across levels')};
    const views = [T('Dentro do nível', 'Within level'), T('Entre níveis', 'Across levels')];
    segButtons(s.querySelector('.game-diff'), Object.keys(MEMORY), difficulty, (d) => memory(d));
    const setView = (v) => { st.view = v; segButtons(s.querySelector('.game-view'), views, v, setView); drawChart('memory', v, st.live); };
    s.querySelector('.game-reset').onclick = () => { clearHistory('memory'); drawChart('memory', st.view, st.live); };
    setView(st.view);

    function startLevel(idx) {
      clearTimeout(s._timer);
      st.level = Math.max(0, Math.min(cfg.grids.length - 1, idx));
      const n = cfg.grids[st.level], total = n * n, pairs = Math.floor(total / 2), bonus = total % 2 === 1;
      const live = st.live = {attempts: 0, errors: 0, hits: 0, t0: performance.now(), events: []};
      let first = null, locked = false;
      const matched = new Set();
      const pool = shuffle(list.slice());
      const hues = HUES.slice(0, cfg.hues);
      const cards = [];
      for (let p = 0; p < pairs; p++) {
        const f = pool[p % pool.length];
        const hue = (p >= pool.length) ? pick(hues.slice(1)) : pick(hues);
        cards.push({p, f, hue}, {p, f, hue});
      }
      shuffle(cards);
      if (bonus) cards.splice(Math.floor(total / 2), 0, {p: null, f: pick(pool), hue: pick(hues)});
      const size = {2: 150, 3: 132, 4: 112, 5: 92, 6: 76}[n] || 82;
      const board = s.querySelector('.game-board');
      board.innerHTML = '';
      board.classList.remove('hunt');
      board.style.gridTemplateColumns = `repeat(${n}, ${size}px)`;
      const top = s.querySelector('.game-top');
      const msg = s.querySelector('.game-msg');
      msg.textContent = T('Encontre os pares.', 'Find the pairs.');
      const status = () => {
        const rate = live.attempts ? 100 * live.errors / live.attempts : 0;
        const secs = ((performance.now() - live.t0) / 1000).toFixed(0);
        top.innerHTML = `<span class="chip">${difficulty}</span><span class="chip">${T('Nível', 'Level')}: ${n}×${n} (${st.level + 1}/${cfg.grids.length})</span>
          <span class="chip">${T('Tentativas', 'Attempts')}: ${live.attempts}</span><span class="chip">${T('Acertos', 'Hits')}: ${live.hits}</span>
          <span class="chip">${T('Erros', 'Errors')}: ${live.errors}</span><span class="chip">${T('Taxa de erro', 'Error rate')}: ${rate.toFixed(1)}%</span>
          <span class="chip">${secs} s</span><button class="btn sm push mem-restart">${T('Recomeçar', 'Restart')}</button>`;
        top.querySelector('.mem-restart').onclick = () => startLevel(0);
      };
      const face = (b, c) => { b.innerHTML = `<img src="${url(c.f)}" style="filter:hue-rotate(${c.hue}deg)">`; b.classList.add('up'); };
      const back = (b) => { b.innerHTML = ''; b.classList.remove('up'); };
      cards.forEach((c, i) => {
        const b = el('button', 'mem-card');
        b.style.width = b.style.height = size + 'px';
        if (c.p === null) { face(b, c); b.classList.add('bonus'); b.disabled = true; matched.add(i); }
        b.onclick = () => {
          if (locked || matched.has(i) || first === i || c.p === null) return;
          face(b, c);
          if (first === null) { first = i; return; }
          const a = first; first = null;
          live.attempts++;
          const hit = cards[a].p === c.p;
          live.events.push({hit, t: performance.now() - live.t0});
          if (hit) {
            live.hits++; matched.add(a); matched.add(i);
            board.children[a].classList.add('done'); b.classList.add('done');
            status(); complete();
          } else {
            live.errors++; locked = true; status();
            setTimeout(() => { [a, i].forEach((k) => { if (!matched.has(k)) back(board.children[k]); }); locked = false; }, cfg.reveal);
          }
          if (st.view === views[0]) drawChart('memory', st.view, live);
        };
        board.appendChild(b);
      });
      if (cfg.peek) {  // easier levels show the board briefly before playing
        locked = true;
        cards.forEach((c, i) => face(board.children[i], c));
        setTimeout(() => { cards.forEach((c, i) => { if (c.p !== null) back(board.children[i]); }); locked = false; live.t0 = performance.now(); }, cfg.peek);
      }
      status();
      const tick = setInterval(() => { if (!shell) clearInterval(tick); else if (st.live === live) status(); else clearInterval(tick); }, 1000);

      function complete() {
        const all = cards.every((c, i) => c.p === null || matched.has(i));
        if (!all) return;
        const time = +((performance.now() - live.t0) / 1000).toFixed(1);
        const rate = live.attempts ? live.errors / live.attempts : 0;
        record({game: 'memory', difficulty, level: `${n}×${n}`, time, attempts: live.attempts, errors: live.errors,
                hits: live.hits, at: new Date().toISOString()});
        drawChart('memory', st.view, live);
        if (rate < cfg.threshold) {
          if (st.level < cfg.grids.length - 1) {
            msg.textContent = T(`Boa! Taxa de erro < ${cfg.threshold * 100}% — próximo nível...`, `Nice! Error rate < ${cfg.threshold * 100}% — next level...`);
            s._timer = setTimeout(() => startLevel(st.level + 1), 1100);
          } else {
            msg.textContent = T('Jogo concluído!', 'Game completed!');
          }
        } else {
          msg.textContent = T(`Taxa de erro ≥ ${cfg.threshold * 100}% — repetindo o nível...`, `Error rate ≥ ${cfg.threshold * 100}% — repeating the level...`);
          s._timer = setTimeout(() => startLevel(st.level), 1200);
        }
      }
    }
    startLevel(0);
  }

  // ---------- find the mascot ----------
  function hunt(difficulty) {
    const list = files();
    if (!list.length) return;
    const cfg = HUNT[difficulty];
    const s = open(T('Onde está o mascote?', 'Find the mascot'));
    const st = {round: 0, view: T('Entre níveis', 'Across levels')};
    const views = [T('Dentro do nível', 'Within level'), T('Entre níveis', 'Across levels')];
    segButtons(s.querySelector('.game-diff'), Object.keys(HUNT), difficulty, (d) => hunt(d));
    const setView = (v) => { st.view = v; segButtons(s.querySelector('.game-view'), views, v, setView); drawChart('hunt', v, st.live); };
    s.querySelector('.game-reset').onclick = () => { clearHistory('hunt'); drawChart('hunt', st.view, st.live); };
    setView(st.view);

    function newRound() {
      clearTimeout(s._timer);
      st.round++;
      const live = st.live = {attempts: 0, errors: 0, hits: 0, t0: performance.now(), events: []};
      const target = pick(list);
      const board = s.querySelector('.game-board');
      board.innerHTML = '';
      board.style.gridTemplateColumns = '';
      board.classList.add('hunt');
      const top = s.querySelector('.game-top');
      const msg = s.querySelector('.game-msg');
      msg.textContent = T('Encontre no cenário o mascote mostrado como alvo.', 'Find the mascot shown as the target somewhere in the crowd.');
      const status = () => {
        const secs = ((performance.now() - live.t0) / 1000).toFixed(0);
        top.innerHTML = `<span class="chip">${difficulty}</span><span class="chip">${T('Alvo', 'Target')}:</span>
          <img class="hunt-target" src="${url(target)}">
          <span class="chip">${T('Rodada', 'Round')}: ${st.round}</span><span class="chip">${T('Tentativas', 'Attempts')}: ${live.attempts}</span>
          <span class="chip">${T('Erros', 'Errors')}: ${live.errors}</span><span class="chip">${secs} s</span>
          <button class="btn sm push hunt-new">${T('Nova rodada', 'New round')}</button>`;
        top.querySelector('.hunt-new').onclick = newRound;
      };
      const field = el('div', 'hunt-field');
      board.appendChild(field);
      const total = cfg.cols * cfg.rows, targetIdx = rnd(total);
      for (let k = 0; k < Math.max(24, total / 2); k++) {
        const c = el('i', 'hunt-clutter');
        c.style.left = rnd(100) + '%'; c.style.top = rnd(100) + '%';
        const r = 10 + rnd(26); c.style.width = c.style.height = r + 'px';
        if (Math.random() < .5) c.style.borderRadius = '50%';
        field.appendChild(c);
      }
      const decoys = cfg.sameSource ? list : list.filter((f) => f !== target);
      for (let idx = 0; idx < total; idx++) {
        const row = Math.floor(idx / cfg.cols), col = idx % cfg.cols;
        const isT = idx === targetIdx;
        const f = isT ? target : pick(decoys.length ? decoys : list);
        const same = f === target;
        const tf = cfg.decoyTransform;
        const hue = isT ? 0 : ((same || Math.random() < tf) ? pick(HUES.slice(1)) : 0);
        const ang = isT ? 0 : ((same || Math.random() < tf) ? pick([-24, -18, -12, -7, 7, 12, 18, 24]) : 0);
        const mir = isT ? false : ((same || Math.random() < tf * .65) ? Math.random() < .5 : false);
        const im = el('img', 'hunt-item');
        im.src = url(f);
        im.style.width = cfg.size + 'px';
        im.style.left = `calc(${(col + .5) / cfg.cols * 100}% + ${(Math.random() - .5) * 0.36 * 100 / cfg.cols}%)`;
        im.style.top = `calc(${(row + .5) / cfg.rows * 100}% + ${(Math.random() - .5) * 0.32 * 100 / cfg.rows}%)`;
        im.style.filter = `hue-rotate(${hue}deg)`;
        im.style.transform = `translate(-50%,-50%) rotate(${ang}deg) scaleX(${mir ? -1 : 1})`;
        im.onclick = () => {
          if (live.done) return;
          live.attempts++;
          live.events.push({hit: isT, t: performance.now() - live.t0});
          if (isT) {
            live.hits++; live.done = true; im.classList.add('found');
            msg.textContent = T('Encontrou! Preparando uma nova rodada...', 'Found it! Preparing a new round...');
            record({game: 'hunt', difficulty, level: `R${st.round}`, time: +((performance.now() - live.t0) / 1000).toFixed(1),
                    attempts: live.attempts, errors: live.errors, hits: live.hits, at: new Date().toISOString()});
            drawChart('hunt', st.view, live);
            s._timer = setTimeout(newRound, 1400);
          } else {
            live.errors++; im.classList.add('miss'); setTimeout(() => im.classList.remove('miss'), 450);
            msg.textContent = T('Ainda não — continue procurando.', 'Not yet — keep looking.');
          }
          status();
          if (st.view === views[0]) drawChart('hunt', st.view, live);
        };
        field.appendChild(im);
      }
      status();
      const tick = setInterval(() => { if (!shell || st.live !== live || live.done) clearInterval(tick); else status(); }, 1000);
    }
    newRound();
  }

  ILT.openGame = function (which) {
    let game = which;
    if (!game) {
      game = Math.random() < .5 ? 'memory' : 'hunt';
      if (game === ILT._lastGame && Math.random() < .55) game = game === 'memory' ? 'hunt' : 'memory';
    }
    ILT._lastGame = game;
    if (game === 'hunt') hunt('Difícil'); else memory('Difícil');
  };

  // Sidebar mascot: click = another mascot, double-click = random game (classic behaviour).
  let clickTimer = null;
  document.addEventListener('click', (e) => {
    const img = e.target.closest && e.target.closest('#mascot-img');
    if (!img) return;
    clearTimeout(clickTimer);
    clickTimer = setTimeout(() => {
      const list = files();
      if (!list.length) return;
      const cur = decodeURIComponent((img.getAttribute('src') || '').split('/').pop());
      const others = list.filter((f) => f !== cur);
      img.src = url(pick(others.length ? others : list));
    }, 220);
  });
  document.addEventListener('dblclick', (e) => {
    if (!(e.target.closest && e.target.closest('#mascot-img'))) return;
    clearTimeout(clickTimer);
    e.preventDefault();
    ILT.openGame();
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && shell) close(); });
  // Optional direct link: ?jogo=memoria | ?jogo=mascote
  window.addEventListener('load', () => {
    const q = new URLSearchParams(location.search).get('jogo');
    if (!q) return;
    const wait = (n) => { if (window.Plotly && document.getElementById('mascot-img')) ILT.openGame(q.startsWith('mem') ? 'memory' : 'hunt');
                          else if (n > 0) setTimeout(() => wait(n - 1), 300); };
    wait(40);
  });
})();

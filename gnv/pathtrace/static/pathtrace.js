/* Path traced -- an optional, self-contained add-on (see pathtrace/__init__.py).
 *
 * Adds a "Path traced" button to the view bar, a settings panel to the sidebar,
 * and an image laid over the view. The viewer's own script is not changed and
 * does not know about it.
 *
 * How it follows the view: you move around in the ordinary, fast view. Each
 * time the view changes the path-traced picture is dropped, and once the view
 * has settled a new one is started; its first, grainy pass appears in about a
 * second and is replaced as more samples arrive. While it is on, Export PNG
 * writes a path-traced picture at the Scene export size.
 *
 * Reads from the page, read-only: SPEC (camera, export size), HIRES, and the
 * #viewbar, #stage, #shot, #status, #b-export and #lightpanel elements.
 * Settings are kept in this browser (localStorage), not in the figure file.
 */
(() => {
  'use strict';
  const STORE = 'gnv.pathtrace';
  const DEFAULTS = {samples: 256, light_size: 0.6, ambient: 0.3, backdrop: false, exposure: 1.0};
  const SETTLE_MS = 350, POLL_MS = 500;
  const q = sel => document.querySelector(sel);
  const make = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  if (!q('#viewbar') || !q('#stage') || !q('#shot')) return;   // not the viewer page

  let opts = Object.assign({}, DEFAULTS);
  try { Object.assign(opts, JSON.parse(localStorage.getItem(STORE) || '{}')); } catch (e) {}
  const save = () => { try { localStorage.setItem(STORE, JSON.stringify(opts)); } catch (e) {} };

  let on = false, job = 0, settle = null, poll = null;
  const figure = () => (typeof SPEC !== 'undefined' && SPEC) ? SPEC : {};
  const hires = () => (typeof HIRES !== 'undefined' && HIRES) ? 1.6 : 1;
  const say = (text, bad) => {
    const s = q('#status'); if (!s) return;
    s.textContent = text; s.style.color = bad ? '#e5534b' : '';
  };
  const post = async (url, body) => {
    const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  };
  const mode = info => info.variant === 'scalar_rgb' ? ' · plain mode (install LLVM for about 2× speed)' : '';

  /* ---- the toggle ---- */
  const button = make('button', null, 'Path traced');
  button.title = 'render the view with a path tracer — soft shadows, light between surfaces, real tubes for fibres. ' +
                 'Slow: move around in the normal view, and the picture follows once the view settles.';
  q('#viewbar').insertBefore(button, q('#b-reset'));

  /* ---- the image over the view ---- */
  const overlay = make('img');
  overlay.alt = '';
  overlay.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;' +
    'object-fit:contain;pointer-events:none;user-select:none;display:none';
  q('#shot').after(overlay);

  function stopPolling(){ if (poll){ clearTimeout(poll); poll = null; } }

  async function start(){
    if (!on) return;
    stopPolling();
    try {
      const stage = q('#stage').getBoundingClientRect();
      const scale = hires();
      const j = await post('/pathtrace/start', {w: Math.round(stage.width * scale), h: Math.round(stage.height * scale), ls: scale,
                                                camera: figure().camera || {}, options: opts});
      job = j.job;
      say('path traced: starting …');
      poll = setTimeout(follow, POLL_MS);
    } catch (e) { say(e.message, true); }
  }

  async function follow(){
    poll = null;
    if (!on) return;
    try {
      const r = await fetch('/pathtrace/frame?t=' + Date.now());
      const info = JSON.parse(r.headers.get('X-Pathtrace') || '{}');
      if (info.job !== job) return;                 // a newer render has started
      if (info.error) throw new Error(info.error);
      if (r.status === 200){
        const url = URL.createObjectURL(await r.blob());
        const old = overlay.src;
        overlay.src = url;
        if (old) URL.revokeObjectURL(old);
        overlay.style.display = '';
      }
      const notes = (info.notes || []).length ? ' · ' + info.notes.join(' · ') : '';
      say(`path traced · ${info.samples} / ${info.target} samples` + (info.done ? '' : ' …') + mode(info) + notes);
      if (!info.done) poll = setTimeout(follow, POLL_MS);
    } catch (e) { say(e.message, true); }
  }

  /* Follow the viewer: a new fast frame means the view changed. The old
   * picture is dropped at once -- it no longer shows the view -- and a new one
   * starts when the view has been still for a moment. */
  new MutationObserver(() => {
    if (!on) return;
    overlay.style.display = 'none';
    stopPolling();
    clearTimeout(settle);
    settle = setTimeout(start, SETTLE_MS);
  }).observe(q('#shot'), {attributes: true, attributeFilter: ['src']});

  function setOn(value){
    on = value;
    button.classList.toggle('on', on);
    if (on){
      // One redrawn view at a time: line art and the path tracer each cover the view.
      const lineArt = [...q('#viewbar').querySelectorAll('button')].find(b => b.textContent === 'Line art');
      if (lineArt && lineArt.classList.contains('on')) lineArt.click();
      panel.open = true; start();
    } else {
      stopPolling(); clearTimeout(settle);
      overlay.style.display = 'none';
      post('/pathtrace/stop').catch(() => {});
      say('normal view');
    }
  }
  button.onclick = () => setOn(!on);
  // Turning line art on turns this off, for the same reason.
  q('#viewbar').addEventListener('click', e => {
    const b = e.target.closest('button');
    if (on && b && b !== button && b.textContent === 'Line art' && !b.classList.contains('on')) setOn(false);
  }, true);

  /* ---- settings panel ---- */
  const panel = make('details');
  panel.style.cssText = 'border-top:1px solid var(--line);padding:8px 14px 12px';
  const summary = make('summary', null, 'Path traced');
  summary.style.cssText = 'cursor:pointer;color:var(--ink-dim);font-size:11px;' +
    'text-transform:uppercase;letter-spacing:.6px;padding:3px 0';
  const body = make('div');
  panel.append(summary, body);
  const after = q('#lightpanel') || q('#scenepanel');
  if (after) after.after(panel); else (q('#panel') || document.body).appendChild(panel);

  const set = (k, v) => { opts[k] = v; save(); if (on) start(); };
  const row = (label, node) => {
    const r = make('div', 'row'); r.appendChild(make('label', null, label));
    const g = make('div', 'grow'); g.appendChild(node); r.appendChild(g);
    body.appendChild(r); return node;
  };
  const slider = (k, min, max, step, title) => {
    const i = make('input'); i.type = 'range'; i.min = min; i.max = max; i.step = step; i.value = opts[k]; i.title = title;
    i.onchange = () => set(k, parseFloat(i.value)); return i;
  };

  const samples = make('select');
  [[64, '64 — quick'], [256, '256 — good'], [1024, '1024 — clean'], [4096, '4096 — print']].forEach(([v, l]) => {
    const o = make('option', null, l); o.value = v; samples.appendChild(o); });
  samples.value = opts.samples;
  samples.title = 'samples per pixel: more is less grainy, and takes proportionally longer';
  samples.onchange = () => set('samples', parseInt(samples.value, 10));
  row('Samples', samples);
  row('Soft light', slider('light_size', 0.1, 2, 0.05, 'size of the lights: larger lights give softer shadows'));
  row('Ambient', slider('ambient', 0, 1, 0.05, 'even light from all round, so nothing in shadow is black'));
  row('Exposure', slider('exposure', 0.5, 2, 0.05, 'overall brightness'));
  const backdrop = make('input'); backdrop.type = 'checkbox'; backdrop.checked = !!opts.backdrop;
  backdrop.title = 'cast the scene\'s shadow onto the background';
  backdrop.onchange = () => set('backdrop', backdrop.checked);
  row('Shadow on background', backdrop);

  /* Writes the path-traced picture to ``path`` at the Scene export size. */
  async function exportTo(path){
    if (!path) return;
    if (!/\.png$/i.test(path)) path += '.png';
    const stage = q('#stage').getBoundingClientRect();
    const long = figure().export_px || 4000;
    const scale = long / Math.max(stage.width, stage.height);
    try {
      stopPolling(); clearTimeout(settle);
      const j = await post('/pathtrace/export', {path, width: Math.round(stage.width * scale), height: Math.round(stage.height * scale),
                                                 line_scale: scale, camera: figure().camera || {}, options: opts});
      job = j.job;
      for (;;){
        await new Promise(r => setTimeout(r, 1000));
        const s = await (await fetch('/pathtrace/status')).json();
        if (s.job !== job) { say('path-traced export stopped: the view changed', true); return; }
        if (s.error) throw new Error(s.error);
        if (s.done){
          say(s.path ? `exported ${s.path} (${j.width} × ${j.height}, ${s.samples} samples)` : 'path-traced export stopped', !s.path);
          break;
        }
        say(`path traced: exporting ${j.width} × ${j.height} · ${s.samples} / ${s.target} samples …` + mode(s));
      }
    } catch (e) { say(e.message, true); }
  }
  const askPath = then => {
    const folder = (q('#m-path') || {}).value || '';
    if (typeof chooseFile === 'function') chooseFile('save', folder, then);
    else then(prompt('Save the path-traced picture as (.png):', start));
  };

  /* Export PNG writes what the view shows: while this is on, a path-traced
   * picture. Caught on the document before the button's own handler runs, so
   * the viewer's script still needs to know nothing about this add-on. */
  document.addEventListener('click', e => {
    if (!on || !e.target.closest('#b-export')) return;
    e.stopImmediatePropagation();
    e.preventDefault();
    askPath(exportTo);
  }, true);

  body.appendChild(make('div', 'hint',
    'Turn on Path traced in the view bar to render the view physically: soft shadows, light bouncing between ' +
    'surfaces, and fibres as real tubes. Move around in the normal view; the picture follows once the view ' +
    'settles, grainy at first and cleaner as samples arrive. Colours are the colour bar\'s, lit by the same ' +
    'lights as the normal view. Volume bodies are not path traced yet. While it is on, Export PNG writes a ' +
    'path-traced picture at the Scene export size — minutes at 4000 pixels. These settings are kept in this ' +
    'browser, not in the figure file.'));
})();

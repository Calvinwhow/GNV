/* Line art -- an optional, self-contained add-on (see lineart/__init__.py).
 *
 * This script adds its own pieces to the page: a "Line art" button in the view
 * bar, a "Line art" settings panel in the sidebar, and an image laid over the
 * view. The viewer's own script is not changed and does not know about it.
 *
 * How it follows the view: the viewer re-renders whenever the figure or camera
 * changes and puts the result in #shot. While line art is on, every new #shot
 * is followed by a request to /lineart/render for the same camera, and the
 * answer is shown on top. Turning it off just hides that image.
 *
 * What it reads from the page, read-only: SPEC (the figure: camera, export
 * size), HIRES (the Hi-res toggle), and the #viewbar, #stage, #shot, #status
 * and #lightpanel elements. Settings are kept in this browser (localStorage),
 * not in the figure file.
 */
(() => {
  'use strict';
  const STORE = 'gnv.lineart';
  const DEFAULTS = {contours: true, axis: 'z', spacing_mm: 1.5, hatch: true, hatch_mm: 0.8,
                    region_outline: true, silhouette: true, ink: null, width: 1.0,
                    slice_levels: 6};
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

  let on = false, busy = false, pending = false;
  const figure = () => (typeof SPEC !== 'undefined' && SPEC) ? SPEC : {};
  const hires = () => (typeof HIRES !== 'undefined' && HIRES) ? 1.6 : 1;
  const say = (text, bad) => {
    const s = q('#status'); if (!s) return;
    s.textContent = text; s.style.color = bad ? '#e5534b' : '';
  };

  /* ---- the toggle ---- */
  const button = make('button', null, 'Line art');
  button.title = 'draw the view as line art — contours, hatching and outlines ' +
                 'instead of shading. Settings and SVG export are in the Line art panel.';
  q('#viewbar').insertBefore(button, q('#b-reset'));

  /* ---- the image over the view ---- */
  const overlay = make('img');
  overlay.alt = '';
  overlay.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;' +
    'object-fit:contain;pointer-events:none;user-select:none;display:none';
  q('#shot').after(overlay);

  async function draw(){
    if (!on) return;
    if (busy){ pending = true; return; }
    busy = true;
    const t0 = performance.now();
    try {
      const stage = q('#stage').getBoundingClientRect();
      const scale = hires();
      const cam = figure().camera || {};
      const p = new URLSearchParams({
        w: Math.round(stage.width * scale), h: Math.round(stage.height * scale),
        ls: scale, o: JSON.stringify(opts), t: Date.now()});
      ['azimuth', 'elevation', 'zoom'].forEach(k => { if (cam[k] != null) p.set(k, cam[k]); });
      const r = await fetch('/lineart/render?' + p);
      if (!r.ok) throw new Error((await r.json()).error || r.statusText);
      const url = URL.createObjectURL(await r.blob());
      const old = overlay.src;
      overlay.src = url;
      if (old) URL.revokeObjectURL(old);
      if (on) overlay.style.display = '';
      let notes = [];
      try { notes = JSON.parse(r.headers.get('X-Lineart-Notes') || '[]'); } catch (e) {}
      say(`line art · ${Math.round(performance.now() - t0)} ms` +
          (notes.length ? ' · ' + notes.join(' · ') : ''));
    } catch (e) {
      say(e.message, true);
    } finally {
      busy = false;
      if (pending){ pending = false; draw(); }
    }
  }

  // Follow the viewer: each new shaded frame is redrawn as line art.
  new MutationObserver(() => { if (on) draw(); })
    .observe(q('#shot'), {attributes: true, attributeFilter: ['src']});

  button.onclick = () => {
    on = !on;
    button.classList.toggle('on', on);
    if (on){ panel.open = true; say('line art: drawing …'); draw(); }
    else { overlay.style.display = 'none'; say('shaded view'); }
  };

  /* ---- settings panel ---- */
  const panel = make('details');
  panel.style.cssText = 'border-top:1px solid var(--line);padding:8px 14px 12px';
  const summary = make('summary', null, 'Line art');
  summary.style.cssText = 'cursor:pointer;color:var(--ink-dim);font-size:11px;' +
    'text-transform:uppercase;letter-spacing:.6px;padding:3px 0';
  const body = make('div');
  panel.append(summary, body);
  const after = q('#lightpanel') || q('#scenepanel');
  if (after) after.after(panel); else (q('#panel') || document.body).appendChild(panel);

  const set = (k, v) => { opts[k] = v; save(); draw(); };
  const row = (label, node) => {
    const r = make('div', 'row'); r.appendChild(make('label', null, label));
    const g = make('div', 'grow'); g.appendChild(node); r.appendChild(g);
    body.appendChild(r); return node;
  };
  const num = (k, step) => {
    const i = make('input'); i.type = 'number'; i.step = step; i.min = 0.05;
    i.value = opts[k];
    i.onchange = () => { const v = parseFloat(i.value); set(k, isFinite(v) && v > 0 ? v : DEFAULTS[k]); };
    return i;
  };
  const tick = (k, title) => {
    const i = make('input'); i.type = 'checkbox'; i.checked = !!opts[k]; i.title = title;
    i.onchange = () => set(k, i.checked); return i;
  };

  const axis = make('select');
  [['z', 'horizontal (axial)'], ['y', 'coronal'], ['x', 'sagittal']].forEach(([v, l]) => {
    const o = make('option', null, l); o.value = v; axis.appendChild(o); });
  axis.value = opts.axis; axis.onchange = () => set('axis', axis.value);
  row('Contour lines', tick('contours', 'the contour lines across each surface; off leaves the outlines, region edges and hatching'));
  row('Contours', axis);
  row('Spacing (mm)', num('spacing_mm', 0.1));
  row('Hatch maps', tick('hatch', 'cross-hatch where a colouring map paints, in its colours'));
  row('Hatch spacing (mm)', num('hatch_mm', 0.1));
  row('Region edges', tick('region_outline', 'trace the edge of each painted region'));
  row('Outlines', tick('silhouette', 'outline and fold lines, from this camera'));

  const levels = make('input'); levels.type = 'number'; levels.step = 1; levels.min = 0; levels.max = 30;
  levels.value = opts.slice_levels;
  levels.title = 'contour lines of the anatomy on each slice, at evenly spaced brightness levels; 0 for none';
  levels.onchange = () => { const v = parseInt(levels.value, 10); set('slice_levels', isFinite(v) ? Math.max(0, Math.min(30, v)) : DEFAULTS.slice_levels); };
  row('Slice isolines', levels);

  const inkBox = make('div', 'inline');
  inkBox.style.cssText = 'display:flex;gap:8px;align-items:center';
  const pick = make('input'); pick.type = 'color'; pick.value = opts.ink || '#1d1d1f';
  const auto = make('input'); auto.type = 'checkbox'; auto.checked = !opts.ink;
  auto.title = 'automatic: dark ink on a light background, light on dark';
  pick.oninput = () => { auto.checked = false; set('ink', pick.value); };
  auto.onchange = () => set('ink', auto.checked ? null : pick.value);
  const autoTag = make('label', null, 'auto');
  autoTag.style.cssText = 'color:var(--ink-dim);font-size:12px';
  inkBox.append(pick, auto, autoTag);
  row('Ink', inkBox);

  const width = make('input'); width.type = 'range'; width.min = 0.2; width.max = 3; width.step = 0.1;
  width.value = opts.width; width.title = 'line width, in pixels as seen here';
  width.oninput = () => set('width', parseFloat(width.value));
  row('Line width', width);

  /* Writes the line art to ``path`` (.svg vectors or .png) at the Scene export
   * size, the same size and line scaling the viewer's own Export PNG uses. */
  async function exportTo(path, extension){
    if (!path) return;
    if (!/\.(svg|png)$/i.test(path)) path += extension;
    const stage = q('#stage').getBoundingClientRect();
    const long = figure().export_px || 4000;
    const scale = long / Math.max(stage.width, stage.height);
    say('line art: exporting ' + path + ' …');
    try {
      const r = await fetch('/lineart/export', {method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({path, width: Math.round(stage.width * scale),
          height: Math.round(stage.height * scale), line_scale: scale,
          camera: figure().camera || {}, options: opts})});
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.statusText);
      say('exported ' + j.path);
    } catch (e) { say(e.message, true); }
  }
  // The viewer's own Save-as dialog when there is one; a prompt otherwise.
  const askPath = (extension, then) => {
    const start = (q('#m-path') || {}).value || '';
    if (typeof chooseFile === 'function') chooseFile('save', start, then);
    else then(prompt('Save line art as (' + extension + '):', start));
  };

  const exportButton = make('button', null, 'Export SVG…');
  exportButton.style.cssText = 'margin-top:8px;width:100%';
  body.appendChild(exportButton);
  exportButton.onclick = () => askPath('.svg', path => exportTo(path, '.svg'));

  /* Export PNG writes what the view shows. While line art is on, the view shows
   * the line art -- it covers the shaded frame entirely -- so the viewer's
   * Export PNG button writes the line art instead of the shaded render. Caught
   * on the document before the button's own handler runs, so the viewer's
   * script still needs to know nothing about this add-on. */
  document.addEventListener('click', e => {
    if (!on || !e.target.closest('#b-export')) return;
    e.stopImmediatePropagation();
    e.preventDefault();
    askPath('.png', path => exportTo(path, '.png'));
  }, true);

  body.appendChild(make('div', 'hint',
    'Turn on Line art in the view bar to draw the view this way: meshes become ' +
    'contour lines, colouring maps are hatched in their own colours, fibres are ' +
    'drawn as lines, and anything behind a surface is hidden. Slices show contour ' +
    'lines of the anatomy (Slice isolines sets how many), the tissue outline, and ' +
    'map regions hatched in their colours. Volume bodies are not drawn. The paper ' +
    'is the Scene background. While line art is on, Export PNG writes the line ' +
    'art, as shown. Export SVG writes vectors at the Scene export size — they print ' +
    'at any size and open in Illustrator. These settings are kept in this browser, ' +
    'not in the figure file.'));
})();

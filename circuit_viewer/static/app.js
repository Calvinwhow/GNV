/* circuit viewer -- the browser half.
 *
 * The client owns the spec and nothing else. Every edit mutates the local spec,
 * POSTs it, and asks for a fresh PNG; all geometry stays in the Python process
 * so that its caches actually get hit. Renders are serialised through one
 * in-flight request with a single pending slot, which is what keeps a drag
 * smooth: intermediate frames are dropped rather than queued.
 */
const $ = (s, r = document) => r.querySelector(s);
const el = (t, cls, txt) => { const n = document.createElement(t);
  if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };

const VIEWS = { anterior:[0,0], left:[90,0], posterior:[180,0],
                right:[270,0], superior:[0,89.9], inferior:[0,-89.9] };

let SPEC = null, CAT = null, OPEN = null, HIRES = false;
let busy = false, pending = false;
let VIEW = null;        // the last frame's framing: height in mm, right / up vectors
let RECORDING = null;   // frames shown while recording: [{blob, t}]
const sleep = ms => new Promise(r => setTimeout(r, ms));
/* Render and wait for THIS frame. render() coalesces requests while one is in
 * flight, which is right for dragging and wrong for playback, where every
 * frame has to be shown. */
async function renderSettled(quality){
  while (busy) await sleep(15);
  await render(quality);
}

/* ---------- server ---------- */
async function getJSON(u){ const r = await fetch(u); const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText); return j; }
async function postJSON(u, body){
  const r = await fetch(u, {method:'POST', headers:{'Content-Type':'application/json'},
                            body: JSON.stringify(body)});
  const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j;
}
function status(msg, err){ const s = $('#status'); s.textContent = msg; s.classList.toggle('err', !!err); }

/* ---------- render loop ---------- */
async function render(quality){
  if (busy){ pending = true; return; }
  busy = true; $('#spin').classList.add('on');
  const t0 = performance.now();
  try {
    await postJSON('/api/spec', SPEC);
    const stage = $('#stage').getBoundingClientRect();
    const scale = (quality === 'export' || HIRES) ? 1.6 : 1;
    const q = new URLSearchParams({
      w: Math.round(stage.width * scale), h: Math.round(stage.height * scale),
      azimuth: SPEC.camera.azimuth, elevation: SPEC.camera.elevation,
      zoom: SPEC.camera.zoom,
      quality: quality || (HIRES ? 'export' : 'preview'),
      aa: quality === 'export' ? 'ssaa' : (HIRES ? 'fxaa' : 'off'),
      t: Date.now(),
    });
    const r = await fetch('/api/render?' + q);
    if (!r.ok) throw new Error((await r.json()).error);
    const blob = await r.blob();
    const old = $('#shot').src;
    $('#shot').src = URL.createObjectURL(blob);
    if (old.startsWith('blob:')) URL.revokeObjectURL(old);
    const view = r.headers.get('X-View');
    if (view){ try { VIEW = JSON.parse(view); } catch {} }
    if (RECORDING) RECORDING.push({blob, t: performance.now()});
    /* The render succeeded but something in it was skipped -- a values file
     * that does not fit its atlas, most often. Say which, in the place errors
     * go, instead of leaving a layer silently unpainted. */
    const warned = r.headers.get('X-Render-Warnings');
    if (warned){
      let list = [];
      try { list = JSON.parse(warned); } catch { list = [warned]; }
      status(list.join(' · '), true);
    } else {
      status(`${Math.round(performance.now() - t0)} ms · az ${SPEC.camera.azimuth.toFixed(0)}° el ${SPEC.camera.elevation.toFixed(0)}° · ${SPEC.layers.length} layer(s)`);
    }
  } catch (e) { status(e.message, true); }
  finally {
    busy = false; $('#spin').classList.remove('on');
    if (pending){ pending = false; render(); }
  }
}

/* ---------- layer panel ---------- */
function field(parent, label, node){
  const row = el('div','row'); row.appendChild(el('label',null,label));
  const g = el('div','grow'); g.appendChild(node); row.appendChild(g);
  parent.appendChild(row); return node;
}
function select(options, value, onchange){
  const s = el('select');
  options.forEach(o => { const opt = el('option', null, o.label ?? o);
    opt.value = o.value ?? o; if (o.disabled) opt.disabled = true; s.appendChild(opt); });
  s.value = value ?? '';
  s.onchange = () => onchange(s.value);
  return s;
}
function number(value, onchange, {step = 'any', placeholder = 'none'} = {}){
  const i = el('input'); i.type = 'number'; i.step = step;
  i.value = (value ?? ''); i.placeholder = placeholder;
  i.onchange = () => onchange(i.value === '' ? null : parseFloat(i.value));
  return i;
}
function text(value, onchange){
  const i = el('input'); i.type = 'text'; i.value = value ?? '';
  i.onchange = () => onchange(i.value || null); return i;
}
/* Every boolean is a square checkbox. One control for one meaning: a number
 * box invites a value that is really on/off, and a glyph button makes the
 * reader work out which state is which. */
function check(value, onchange, title){
  const i = el('input'); i.type = 'checkbox'; i.checked = !!value;
  if (title) i.title = title;
  i.onchange = () => onchange(i.checked);
  return i;
}
function colour(value, onchange){
  const i = el('input'); i.type = 'color'; i.value = value || '#b7bcc4';
  i.oninput = () => onchange(i.value); return i;
}
function pathField(parent, label, value, kind, onchange){
  const row = el('div','row'); row.appendChild(el('label', null, label));
  const box = el('div','grow pathbox');
  const input = text(value, onchange);
  const browse = el('button', null, '…');
  browse.onclick = () => chooseFile(kind, value, p => { input.value = p; onchange(p); });
  const clear = el('button', null, '×');
  clear.onclick = () => { input.value = ''; onchange(null); };
  box.append(input, browse, clear); row.appendChild(box); parent.appendChild(row);
}

/* A prepared backdrop is a volume too. Without .npz here a volume layer built
 * from one falls through to the built-in-mesh branch, which shows a Mesh
 * dropdown and no way to change the volume -- which is what made + Volume look
 * like it had silently picked a file. */
const isVolume = p => /\.(nii(\.gz)?|npz)$/i.test(String(p || ''));


let PALETTES = null;

async function refreshPalettes(){
  try { PALETTES = await getJSON('/api/palettes'); }
  catch { PALETTES = {clut: [], matplotlib: []}; }
  return PALETTES;
}

/* ---------- sectioning ----------
 * `vol(X > -3.0) = 0` in the panel. Each condition deletes one half-space in
 * world millimetres, so you can cut down to a wall and look at the colouring
 * on it. Statistics are computed on the UNCUT region, so opening a parcel up
 * does not change the number it is showing.
 */
function sectionEditor(b, L, set){
  const wrap = el('details','fold section');
  wrap.appendChild(el('summary', null, 'Section'));
  const body = el('div'); wrap.appendChild(body); b.appendChild(wrap);
  const cuts = L.section || (L.section = []);
  if (cuts.length) wrap.open = true;

  cuts.forEach((c, i) => {
    const row = el('div','row');
    const on = el('input'); on.type='checkbox'; on.checked = c.enabled !== false;
    on.onchange = () => { c.enabled = on.checked; set('section', cuts); };
    const ax = select([{value:'x',label:'X'},{value:'y',label:'Y'},{value:'z',label:'Z'}],
                      c.axis, v => { c.axis = v; set('section', cuts); });
    ax.style.cssText = 'width:50px;flex:0 0 50px';
    const op = select([{value:'>',label:'>'},{value:'<',label:'<'}], c.op,
                      v => { c.op = v; set('section', cuts); });
    op.style.cssText = 'width:48px;flex:0 0 48px';
    const at = el('input'); at.type='number'; at.step=0.1; at.value = c.at;
    at.style.cssText = 'width:70px;flex:0 0 70px;text-align:right';
    at.onchange = () => { c.at = parseFloat(at.value) || 0;
                          slide.value = c.at; set('section', cuts); };
    /* A cut you can drag is how you read a volume: the number box is for
     * landing on a known coordinate, the slider is for finding one. Both write
     * the same field and each follows the other. */
    const slide = el('input'); slide.type='range';
    slide.min=-90; slide.max=90; slide.step=1; slide.value=c.at;
    slide.title = 'scroll the cut through the volume';
    slide.oninput = () => { c.at = parseFloat(slide.value);
                            at.value = c.at; set('section', cuts); };
    const del = el('button','icon','✕');
    del.onclick = () => { cuts.splice(i,1); set('section', cuts); drawLayers(); };
    const box = el('div','grow inline'); box.append(on, ax, op, at, el('span',null,'mm'), del);
    row.appendChild(el('label', null, i ? 'and' : 'remove'));
    row.appendChild(box); body.appendChild(row);
    const srow = el('div','row'); srow.appendChild(el('label', null, ''));
    const sbox = el('div','grow'); sbox.appendChild(slide);
    srow.appendChild(sbox); body.appendChild(srow);
  });

  /* Capping turns a cut into a cross-section. Without it you see through the
   * hole to the inside of the far shell; with it the flat face is real geometry
   * that vertex sampling paints with the anatomy, which is what makes dragging
   * the slider read as scrolling through the volume. Region colouring cannot
   * use it -- a cap belongs to no region -- so it is a choice. */
  const cap = el('input'); cap.type='checkbox'; cap.checked = !!L.section_cap;
  cap.title = 'fill the cut face — needs vertex colouring';
  cap.onchange = () => set('section_cap', cap.checked);
  const crow = el('div','row'); crow.appendChild(el('label', null, 'Cap the cut'));
  const cbox = el('div','grow'); cbox.appendChild(cap);
  crow.appendChild(cbox); body.appendChild(crow);

  const add = el('button', null, '+ Add cut');
  add.onclick = () => { cuts.push({axis:'x', op:'>', at:0, enabled:true});
                        set('section', cuts); drawLayers(); };
  const row = el('div','row'); row.appendChild(el('label', null, ''));
  const box = el('div','grow'); box.appendChild(add); row.appendChild(box);
  body.appendChild(row);
  body.appendChild(el('div','hint',
    'Each line deletes everything on one side of a plane, in world millimetres — ' +
    'vol(X > -3.0) = 0. Region statistics come from the uncut region, so cutting ' +
    'one open reveals its colouring rather than changing it.'));
}

const REGION_CACHE = {};

async function regionsFor(L){
  const key = JSON.stringify([L.mesh, L.iso, L.split]);
  if (!REGION_CACHE[key]){
    const q = new URLSearchParams({mesh: Array.isArray(L.mesh) ? JSON.stringify(L.mesh) : L.mesh,
                                   split: L.split || 'auto'});
    if (L.iso != null) q.set('iso', L.iso);
    REGION_CACHE[key] = getJSON('/api/regions?' + q).then(r => r.regions).catch(() => []);
  }
  return REGION_CACHE[key];
}

/* A group of 165 parcels is a list you search, not a list you scroll. */
function regionList(b, L, set){
  const wrap = el('details','fold regions');
  const head = el('summary', null, 'Regions');
  wrap.appendChild(head);
  const body = el('div'); wrap.appendChild(body); b.appendChild(wrap);

  wrap.addEventListener('toggle', async () => {
    if (!wrap.open || body.dataset.filled) return;
    body.dataset.filled = '1';
    body.appendChild(el('div','hint','reading group…'));
    const regions = await regionsFor(L);
    body.innerHTML = '';
    if (!regions.length){ body.appendChild(el('div','hint','no regions')); return; }

    const hidden = new Set(L.hidden_regions || []);
    const search = el('input'); search.type='text'; search.placeholder='filter…';
    const tools = el('div','inline');
    const all = el('button', null, 'all');
    const none = el('button', null, 'none');
    all.onclick = () => { L.hidden_regions = []; set('hidden_regions', []);
                          delete body.dataset.filled; wrap.open = false; wrap.open = true; };
    none.onclick = () => { const names = regions.map(r => r.name);
                           L.hidden_regions = names; set('hidden_regions', names);
                           delete body.dataset.filled; wrap.open = false; wrap.open = true; };
    tools.append(search, all, none); body.appendChild(tools);

    /* Coarse groups, so "everything except the cerebellum" is one click rather
     * than thirty-two. A box is checked when every region under it is on, and
     * indeterminate when only some are -- which is the honest state, and is
     * what tells you a group has been partly hidden by hand. */
    const groups = [...new Set(regions.map(r => r.group).filter(Boolean))];
    const supra = [];
    if (groups.length > 1){
      const row = el('div','supra');
      groups.forEach(g => {
        const members = regions.filter(r => r.group === g).map(r => r.name);
        const tag = el('label','supra-item');
        const c = el('input'); c.type = 'checkbox';
        c.onchange = () => {
          members.forEach(n => c.checked ? hidden.delete(n) : hidden.add(n));
          L.hidden_regions = [...hidden];
          set('hidden_regions', L.hidden_regions);
          syncBoxes();
        };
        tag.append(c, el('span', null, g));
        row.appendChild(tag);
        supra.push({box: c, members});
      });
      body.appendChild(row);
    }
    const tally = el('div','hint', '');
    body.appendChild(tally);

    const list = el('div','region-list');
    const boxes = {};
    /* One place that redraws every box from `hidden`, so a group toggle and an
     * individual toggle can never disagree about what is showing. */
    const syncBoxes = () => {
      // The tally is part of the state, not decoration: rendered once it went
      // on reading "0 hidden" with a whole lobe switched off.
      tally.textContent = `${regions.length} regions · ${hidden.size} hidden`;
      regions.forEach(r => { if (boxes[r.name]) boxes[r.name].checked = !hidden.has(r.name); });
      supra.forEach(({box, members}) => {
        const on = members.filter(n => !hidden.has(n)).length;
        box.checked = on === members.length;
        box.indeterminate = on > 0 && on < members.length;
      });
    };
    regions.forEach(r => {
      const row = el('label','region');
      const c = el('input'); c.type='checkbox'; c.checked = !hidden.has(r.name);
      boxes[r.name] = c;
      c.onchange = () => {
        if (c.checked) hidden.delete(r.name); else hidden.add(r.name);
        L.hidden_regions = [...hidden];
        set('hidden_regions', L.hidden_regions);
        syncBoxes();
      };
      row.append(c, el('span', null, r.name));
      row.dataset.name = r.name.toLowerCase();
      list.appendChild(row);
    });
    body.appendChild(list);
    syncBoxes();
    search.oninput = () => {
      const q = search.value.toLowerCase();
      [...list.children].forEach(row =>
        row.style.display = row.dataset.name.includes(q) ? '' : 'none');
    };
  });
}

/* ---------- overlay stack ----------
 * A layer holds an ordered list of volumes rather than one, so "motor in red,
 * cognitive in blue, over anatomy" is three entries on one group instead of
 * three groups competing for the same geometry. Later entries paint over
 * earlier ones wherever both clear their own threshold.
 */
async function newOverlay(nifti){
  const o = await getJSON('/api/overlay');
  if (nifti) o.nifti = nifti;
  return o;
}

const OVERLAY_COLOURS = ['#c15656', '#5071a0', '#9a8ed1', '#4f9d69', '#c98a33'];

/* Two number boxes that report together: either both ends are set or the pair
 * is cleared, because half a range selects nothing and reads as a broken render
 * rather than as an empty selection. */
function pair(a, z, phA, phZ, onchange){
  const box = el('div','grow inline');
  const mk = (val, ph) => {
    const n = el('input'); n.type='number'; n.step='any'; n.placeholder = ph;
    n.value = (val === null || val === undefined) ? '' : val;
    n.style.cssText = 'width:70px;flex:1 1 0;text-align:right';
    n.onchange = () => {
      const lo = parseFloat(box.children[0].value), hi = parseFloat(box.children[1].value);
      onchange(Number.isFinite(lo) ? lo : null, Number.isFinite(hi) ? hi : null);
    };
    return n;
  };
  box.append(mk(a, phA), mk(z, phZ));
  return box;
}

/* ---------- 4D series ----------
 * A 4D NIfTI, or a fibres x time value array, is shown one frame at a time.
 * The frame row appears on an overlay only when its file has more than one
 * frame. Play steps through them, waiting for each frame to be drawn, so slow
 * scenes play slower rather than skipping frames. The colour scale is read
 * from the whole series, so a value keeps its colour from frame to frame. */
const FRAMES = {};                 // path -> promise of a frame count
const PLAYING = new Set();         // overlay ids being played
function frameCount(path){
  if (!path) return Promise.resolve(1);
  if (!(path in FRAMES)){
    FRAMES[path] = getJSON('/api/frames?' + new URLSearchParams({path}))
      .then(j => j.frames || 1).catch(() => 1);
  }
  return FRAMES[path];
}
function frameControls(body, o, stack, set){
  frameCount(o.nifti).then(n => {
    if (n <= 1) return;
    const row = el('div','row'); row.appendChild(el('label', null, 'Frame'));
    const box = el('div','grow inline');
    const play = el('button', null, PLAYING.has(o.id) ? '❚❚' : '▶');
    play.title = 'play the series';
    const slider = el('input'); slider.type = 'range'; slider.min = 0; slider.max = n - 1;
    slider.step = 1; slider.value = o.frame || 0;
    const shown = el('span','hint', `${(o.frame || 0) + 1} / ${n}`);
    const fps = el('input'); fps.type = 'number'; fps.min = 1; fps.max = 30; fps.step = 1;
    fps.value = o._fps || 4; fps.title = 'frames per second'; fps.style.width = '3.5em';
    fps.onchange = () => { o._fps = parseFloat(fps.value) || 4; };
    const show = k => { o.frame = k; slider.value = k; shown.textContent = `${k + 1} / ${n}`; };
    slider.oninput = () => { show(parseInt(slider.value)); set('overlays', stack); };
    play.onclick = async () => {
      if (PLAYING.has(o.id)){ PLAYING.delete(o.id); play.textContent = '▶'; return; }
      PLAYING.add(o.id); play.textContent = '❚❚';
      while (PLAYING.has(o.id)){
        const t0 = performance.now();
        show(((o.frame || 0) + 1) % n);
        await renderSettled();
        const wait = 1000 / (o._fps || 4) - (performance.now() - t0);
        if (wait > 0) await sleep(wait);
      }
    };
    box.append(play, slider, shown, fps);
    row.appendChild(box);
    body.insertBefore(row, body.children[1] || null);
  });
}

function overlayStack(b, L, set, kind){
  b.appendChild(el('div','sec', kind === 'volume' ? 'Volumes painted on the body'
                              : kind === 'slices' ? 'Volumes on the slice'
                                                  : 'Volumes colouring this group'));
  const stack = L.overlays || (L.overlays = []);

  stack.forEach((o, i) => {
    const card = el('div','ovl');
    const head = el('div','ovl-head');
    const swatch = colour(o.color, v => { o.color = v; set('overlays', stack); });
    const label = el('div','name', o.name || (o.nifti || '(no volume)').split('/').pop());
    label.title = o.nifti || '';
    const up = el('button','icon','↑');
    up.onclick = () => { if (i > 0){ stack.splice(i-1, 0, stack.splice(i,1)[0]);
                                     set('overlays', stack); drawLayers(); } };
    const del = el('button','icon','✕');
    del.onclick = () => { stack.splice(i,1); set('overlays', stack); drawLayers(); };
    head.append(swatch, label, up, del);
    card.appendChild(head);

    const body = el('div','ovl-body');
    pathField(body, kind === 'fibers' ? 'Values' : 'NIfTI', o.nifti,
              kind === 'fibers' ? 'fibers' : 'volume',
              v => { o.nifti = v; o.frame = 0; set('overlays', stack); drawLayers(); });
    frameControls(body, o, stack, set);
    /* Two stages, kept apart on purpose.
     *
     * PROCESSING acts on the volume: it decides which values are in the figure
     * at all -- sign, the value window, |v| -- and everything it rejects is
     * simply absent. COLOUR then maps whatever survived from min to max and
     * does nothing else. Mixing them is how a scale edit silently changes which
     * voxels are shown, so moving the colour ends restyles a figure and never
     * rethresholds it. */
    body.appendChild(el('div','sec','Processing — acts on the volume'));
    if (kind === 'mesh'){
      field(body, 'Sample', select(
        [{value:'region',label:'region — one value per mesh'},
         {value:'vertex',label:'vertex — gradient'}],
        o.mode, v => { o.mode = v; set('overlays', stack); drawLayers(); }));
      if (o.mode === 'region')
        field(body, 'Statistic', select(CAT.stats, o.stat,
              v => { o.stat = v; set('overlays', stack); }));
    }
    const win = pair(o.threshold, o.max_value, 'min', 'max', (a, z) => {
      o.threshold = a; o.max_value = z; set('overlays', stack); });
    win.title = 'drop |value| below min or above max';
    field(body, 'Threshold min/max', win);
    const abs = el('input'); abs.type = 'checkbox'; abs.checked = !!o.absolute;
    abs.title = 'take |value| before anything else is decided';
    abs.onchange = () => { o.absolute = abs.checked; set('overlays', stack); };
    field(body, 'Absolute value', abs);

    /* Vertex sampling asks the volume for a value at one exact point, so a
     * vertex landing in a zero voxel gets nothing and the surface speckles.
     * Each pass spreads values into unpainted neighbours ALONG the mesh, so a
     * patch fills from its own data and nothing is invented where a whole
     * connected region is empty. */
    if (kind === 'mesh'){
      /* A 2 mm map sampled at single points on a folded cortex stripes: the
       * banks land inside it, crowns and fundi just outside. Averaging over a
       * couple of millimetres of surface -- along the mesh, never across a
       * sulcus -- before the threshold closes those stripes. Region overlays
       * reduce whole regions and ignore it. */
      field(body, 'Smooth (mm)', number(o.smooth_mm ?? 2,
            v => { o.smooth_mm = v ?? 0; set('overlays', stack); },
            {step: 0.5, placeholder: '0'}));
    }
    if (kind === 'mesh' && o.mode === 'vertex'){
      /* Stored as a number of passes, so an old spec keeps whatever it was
       * saved with, but offered as on/off -- the question is whether to close
       * the speckle, not how many rounds of averaging to run. */
      field(body, 'Fill sampling gaps', check(o.fill_gaps,
            v => { o.fill_gaps = v ? FILL_PASSES : 0; set('overlays', stack); },
            'spread values into vertices the volume gave nothing for'));
    }

    body.appendChild(el('div','sec','Colour — maps min to max'));
    /* A .clut is MRIcroGL's own scale, so a figure and the volume view it came
     * from agree about what red means. "from colour" keeps the two-stop ramp. */
    const pal = [{value:'', label:'from colour (ramp)'}];
    if (PALETTES){
      PALETTES.clut.forEach(n => pal.push({value:n, label:'MRIcroGL · ' + n}));
      PALETTES.matplotlib.forEach(n => pal.push({value:n, label:'matplotlib · ' + n}));
    }
    field(body, 'Scale', select(pal, o.palette || '',
          v => { o.palette = v || null; set('overlays', stack); }));

    /* Auto reads the data: two-signed gets a diverging scale about zero. That
     * is a good guess and a bad rule, because one stray negative voxel should
     * not decide how a figure is coloured -- hence the override. */
    field(body, 'Diverging', select(
      [{value:'auto', label:'auto (from the data)'},
       {value:'yes',  label:'yes — symmetric about 0'},
       {value:'no',   label:'no — sequential'}],
      o.symmetric === null || o.symmetric === undefined ? 'auto' : (o.symmetric ? 'yes' : 'no'),
      v => { o.symmetric = v === 'auto' ? null : (v === 'yes');
             set('overlays', stack); }));

    /* Nothing is dropped here: min and max are only where the scale's two ends
     * land. Entering them backwards inverts the scale, which is the one extra
     * thing a pair of colour limits can sensibly mean. */
    const lims = pair(o.clim ? o.clim[0] : null, o.clim ? o.clim[1] : null,
                      'min', 'max',
                      (a, z) => { o.clim = (a !== null && z !== null) ? [a, z] : null;
                                  set('overlays', stack); });
    lims.title = 'ends of the colour scale — reverse them to invert it';
    field(body, 'Colour min/max', lims);
    const skip = el('input'); skip.type = 'checkbox';
    skip.checked = !!o.hide_below_min;
    skip.title = 'below the scale minimum, contribute nothing instead of the bottom colour';
    skip.onchange = () => { o.hide_below_min = skip.checked; set('overlays', stack); };
    field(body, 'Skip below min', skip);
    /* The bar belongs to THIS map. A scene-wide switch could only caption one
     * scale, which on a two-map figure is a caption that is wrong about one of
     * them. Several bars stack in from the right edge. */
    const bar = el('input'); bar.type = 'checkbox';
    bar.checked = !!o.bar;
    bar.title = "show this map's colour bar";
    bar.onchange = () => { o.bar = bar.checked; set('overlays', stack); };
    field(body, 'Colour bar', bar);
    body.appendChild(el('div','hint',
      'Skip below min lets a stack intercalate: where this volume falls under ' +
      'the scale it leaves the surface to whatever is beneath it, instead of ' +
      'washing it with the bottom colour. Diverging scales ignore it — their ' +
      'low end is the negative extreme, not an absence.'));
    const op = el('input'); op.type='range'; op.min=0; op.max=1; op.step=0.02;
    op.value = o.opacity;
    op.oninput = () => { o.opacity = parseFloat(op.value); set('overlays', stack); };
    field(body, 'Opacity', op);
    /* How much the colour scale overrides what is under it. A LUT such as
     * MRIcroGL's x_rain fades its own low end out; at 1 every value is painted
     * at full strength -- the bottom colour included, and anything below the
     * minimum in that colour -- as MRIcroGL draws it. Lower lets the low end
     * of the scale fade into the anatomy. */
    const ovr = el('input'); ovr.type='range'; ovr.min=0; ovr.max=1; ovr.step=0.05;
    ovr.value = o.override ?? 1;
    ovr.title = 'how strongly the colour scale overrides the anatomy at its low end';
    ovr.oninput = () => { o.override = parseFloat(ovr.value); set('overlays', stack); };
    field(body, 'Override brain', ovr);
    card.appendChild(body);
    b.appendChild(card);
  });

  const add = el('button', null, '+ Add volume');
  add.onclick = () => chooseFile('volume', $('#m-path').value, async p => {
    const o = await newOverlay(p);
    o.color = OVERLAY_COLOURS[stack.length % OVERLAY_COLOURS.length];
    stack.push(o); set('overlays', stack); drawLayers();
  });
  const row = el('div','row'); row.appendChild(el('label', null, ''));
  const box = el('div','grow'); box.appendChild(add); row.appendChild(box);
  b.appendChild(row);
  if (!stack.length)
    b.appendChild(el('div','hint', kind === 'slices' || kind === 'volume'
      ? 'Add a volume to overlay a finding on the anatomy, or drop one on this layer.'
      : 'With no volume the whole group is drawn in the flat colour above. ' +
        'One volume gets a colour bar; several are composited, later over earlier.'));
}

function meshBody(b, L, set){
  const groups = {};
  CAT.meshes.forEach(m => (groups[m.group] ||= []).push({
    value: m.name, disabled: m.available === false,
    label: (m.available === false ? '— ' : '') + m.name + ' — ' +
           (m.available === false ? 'not available to this renderer'
                                  : m.what.replace(/^\[[^\]]*\]\s*/, '')),
  }));
  const opts = [...(groups.surface || []), ...(groups.parcels || [])];
  const vol = isVolume(L.mesh);

  /* Geometry comes either from a mesh that already exists or from a volume we
   * surface here. One field holds both, because downstream nothing else cares. */
  field(b, 'Geometry', select(
    [{value:'mesh', label:'built-in mesh'}, {value:'volume', label:'a NIfTI volume'}],
    vol ? 'volume' : 'mesh',
    v => set('mesh', v === 'volume' ? '' : 'pial_wholebrain')));

  if (!vol){
    field(b, 'Mesh', select(opts, L.mesh, v => set('mesh', v)));
  } else {
    /* Prepared backdrops first: they are already resampled and quantised, so
     * they mesh in seconds and can colour themselves. Any other NIfTI still
     * works through the path field below. */
    const lib = (BACKDROPS && BACKDROPS.backdrops) || [];
    if (lib.length){
      const chosen = lib.find(d => d.path === L.mesh);
      field(b, 'Prepared volume', select(
        [{value:'', label: chosen ? '— other —' : '— choose —'},
         ...lib.map(d => ({value: d.path,
           label: `${d.name} · ${d.mm} mm · ${(d.bytes/1e6).toFixed(0)} MB`}))],
        chosen ? chosen.path : '',
        v => { if (!v) return;
               L.mesh = v;
               /* Point the self-colouring overlay at the same file, so picking a
                * volume shows its anatomy rather than a flat grey surface. */
               const stack = L.overlays || (L.overlays = []);
               if (stack.length && stack[0].mode === 'vertex') stack[0].nifti = v;
               set('mesh', v); drawLayers(); }));
    }
    pathField(b, 'Volume', L.mesh, 'volume', v => set('mesh', v || ''));
    field(b, 'Threshold', number(L.iso, v => set('iso', v === null ? 0 : v)));
    field(b, 'Split', select(
      [{value:'auto',label:'auto'},
       {value:'single',label:'single — one surface'},
       {value:'labels',label:'labels — one per integer value'},
       {value:'components',label:'components — one per blob'}],
      L.split, v => set('split', v)));
    const sm = el('input'); sm.type='range'; sm.min=0; sm.max=3; sm.step=0.1;
    sm.value = L.presmooth_vox; sm.title = 'surface smoothing, in voxels';
    sm.onchange = () => set('presmooth_vox', parseFloat(sm.value));
    field(b, 'Surface smooth', sm);
    if (L.mesh) volumeHint(b, L.mesh);
    else b.appendChild(el('div','hint','Pick a NIfTI, or drop one onto the view.'));
  }
  field(b, 'Colour', colour(L.color, v => set('color', v)));

  const op = el('input'); op.type = 'range'; op.min = 0; op.max = 1; op.step = 0.01;
  op.value = L.opacity; op.oninput = () => set('opacity', parseFloat(op.value));
  field(b, 'Opacity', op);
  field(b, 'Shading', select(
    [{value:'auto',label:'auto'},{value:'faceted',label:'faceted'},{value:'smooth',label:'smooth'}],
    L.faceted === null ? 'auto' : (L.faceted ? 'faceted' : 'smooth'),
    v => set('faceted', v === 'auto' ? null : v === 'faceted')));

  /* Smoothness is the inverse of the decimation applied before sampling: slide
   * it up for the fine mesh and clean silhouettes, down for bigger plates. The
   * label matches what the eye sees; the spec stores the reduction. */
  const sm = el('input'); sm.type = 'range'; sm.min = 0.05; sm.max = 1; sm.step = 0.05;
  sm.value = 1 - (L.tessellate === null ? 0.7 : L.tessellate);
  sm.oninput = () => set('tessellate', +(1 - parseFloat(sm.value)).toFixed(2));
  field(b, 'Smoothness', sm);

  const edBox = el('div','grow inline');
  const edOn = el('input'); edOn.type = 'checkbox';
  edOn.checked = L.edges === null ? true : !!L.edges;
  edOn.onchange = () => set('edges', edOn.checked);
  const edCol = colour(L.edge_color, v => set('edge_color', v));
  /* Its own opacity, independent of the mesh's: the wireframe is drawn as a
   * separate actor precisely so a faint surface can carry a crisp one. */
  const edOp = el('input'); edOp.type='range'; edOp.min=0; edOp.max=1; edOp.step=0.02;
  edOp.value = L.edge_opacity; edOp.title = 'wireframe opacity, independent of the mesh';
  edOp.oninput = () => set('edge_opacity', parseFloat(edOp.value));
  edBox.append(edOn, edCol, edOp);
  const edRow = el('div','row'); edRow.appendChild(el('label', null, 'Edge highlight'));
  edRow.appendChild(edBox); b.appendChild(edRow);

  sectionEditor(b, L, set);

  /* Hemispheres are an explicit choice about the figure. The old behaviour --
   * dropping whichever half faced the camera -- is still here as an option,
   * because it is right for a translucent shell and wrong everywhere else. */
  const sides = el('div','grow inline');
  [['show_left','Left'], ['show_right','Right']].forEach(([key, label]) => {
    const c = el('input'); c.type='checkbox'; c.checked = L[key] !== false;
    c.onchange = () => set(key, c.checked);
    const tag = el('label', null, label);
    tag.style.cssText = 'flex:0 0 auto;color:var(--ink-dim);font-size:12px';
    sides.append(c, tag);
  });
  const hemiRow = el('div','row'); hemiRow.appendChild(el('label', null, 'Hemispheres'));
  hemiRow.appendChild(sides); b.appendChild(hemiRow);
  const cull = el('input'); cull.type='checkbox'; cull.checked = !!L.cull_near_hemisphere;
  cull.onchange = () => set('cull_near_hemisphere', cull.checked);
  field(b, 'Auto-hide near', cull);

  regionList(b, L, set);
  overlayStack(b, L, set, 'mesh');
}

let BACKDROPS = null;

async function refreshBackdrops(){
  try { BACKDROPS = await getJSON('/api/backdrops'); }
  catch (e) { BACKDROPS = {backdrops: [], dir: '?'}; status(e.message, true); }
  return BACKDROPS;
}

function sliceBody(b, L, set){
  const lib = (BACKDROPS && BACKDROPS.backdrops) || [];
  const chosen = lib.find(d => d.path === L.backdrop || d.name === L.backdrop);
  field(b, 'Backdrop', select(
    [{value:'', label: lib.length ? '— choose —' : '— none converted yet —'},
     ...lib.map(d => ({value: d.path,
       label: `${d.name} · ${d.mm} mm · ${(d.bytes/1e6).toFixed(0)} MB`}))],
    chosen ? chosen.path : '', v => set('backdrop', v || null)));

  const convert = el('button', null, 'Convert a NIfTI …');
  convert.onclick = () => chooseFile('volume', $('#m-path').value, p => startConvert(p));
  const cvRow = el('div','row'); cvRow.appendChild(el('label', null, ''));
  const cvBox = el('div','grow'); cvBox.appendChild(convert); cvRow.appendChild(cvBox);
  b.appendChild(cvRow);

  if (!chosen){
    b.appendChild(el('div','hint',
      'Converted volumes accumulate in ' + ((BACKDROPS && BACKDROPS.dir) || '?') +
      ' and are available to every figure afterwards.'));
    return;
  }

  /* Sliders run in WORLD millimetres over this backdrop's own bounds, so the
   * numbers mean the same thing as every coordinate elsewhere in the app. */
  const [lo, hi] = chosen.bounds;
  ['x','y','z'].forEach((key, axis) => {
    const world = chosen.world_axes[axis];
    const row = el('div','row');
    row.appendChild(el('label', null, key.toUpperCase() + ' slice'));
    const box = el('div','grow inline');
    const on = el('input'); on.type='checkbox'; on.checked = L['show_'+key] !== false;
    on.onchange = () => set('show_'+key, on.checked);
    /* Snap to the volume's own voxel grid. There is no plane between two
     * slices, so a control that offers 18.5 mm when the data can only give
     * 18.3 mm is lying about what will be drawn -- and the slider and the box
     * then disagree, which is how this was found. */
    const step = chosen.mm || 0.5;
    const base = lo[world];
    const snap = v => base + Math.round((v - base) / step) * step;
    const digits = step < 0.1 ? 2 : 1;

    const r = el('input'); r.type='range';
    r.min = base; r.max = hi[world]; r.step = step;
    r.value = snap(L[key] === null ? (lo[world]+hi[world])/2 : L[key]);

    /* The number box is the point of the pair: a slider is for finding a plane,
     * a coordinate is for going to the one you already know. Both write the
     * same world-millimetre field, and each follows the other. */
    const mm = el('input'); mm.type = 'number'; mm.step = step;
    mm.min = r.min; mm.max = r.max; mm.value = (+r.value).toFixed(digits);
    mm.title = 'world (MNI) millimetres — snapped to the nearest slice';
    mm.style.cssText = 'width:74px;flex:0 0 74px;text-align:right';
    const apply = v => { r.value = v; mm.value = (+v).toFixed(digits);
                         set(key, parseFloat((+v).toFixed(4))); };
    r.oninput = () => apply(+r.value);
    mm.onchange = () => {
      const typed = parseFloat(mm.value);
      if (!Number.isFinite(typed)) { mm.value = (+r.value).toFixed(digits); return; }
      apply(Math.min(Math.max(snap(typed), base), hi[world]));
    };
    box.append(on, r, mm); row.appendChild(box); b.appendChild(row);
  });

  const op = el('input'); op.type='range'; op.min=0; op.max=1; op.step=0.02;
  op.value = L.opacity; op.oninput = () => set('opacity', parseFloat(op.value));
  field(b, 'Opacity', op);
  const cut = el('input'); cut.type='range'; cut.min=0; cut.max=64; cut.step=1;
  cut.value = L.transparent_below === null ? 0 : L.transparent_below;
  cut.title = 'intensity below which the slice is transparent';
  cut.oninput = () => set('transparent_below', parseInt(cut.value, 10));
  field(b, 'Drop background', cut);
  /* Paint the dropped-out part in the scene background rather than leaving it
   * clear. The data is identical; the slice just stops showing what is behind
   * it, which is most of what makes one look cheap. */
  const fill = el('input'); fill.type='checkbox'; fill.checked = !!L.fill_background;
  fill.title = 'paint the dropped-out background in the scene colour';
  fill.onchange = () => set('fill_background', fill.checked);
  field(b, 'Fill with background', fill);
  b.appendChild(el('div','hint',
    'Coordinates are world millimetres — type one in to jump straight to a ' +
    'known plane. Background that stays opaque is a black slab in front of the ' +
    'mesh; dropping it out lets the slice sit inside the brain. Filling it back ' +
    'in with the scene colour keeps that, and makes the slice read as a solid plate.'));
  overlayStack(b, L, set, 'slices');
}

/* ---------- volume layer ----------
 * A body, ray-cast from its own scalars: colour and opacity transfer functions
 * over a window, linear interpolation, the scene's lighting. Overlays only
 * colour it -- painted onto the tissue the rays reach, or ray-cast as shaded
 * colour volumes inside a body made translucent. Cutting lives here rather than
 * on the slice layer, because removing part of a body is a different operation
 * from photographing it at one coordinate.
 */
let VOLUMES = null;
async function refreshVolumes(){
  try { VOLUMES = await getJSON('/api/volumes'); } catch { VOLUMES = []; }
  return VOLUMES;
}

function volumeBody(b, L, set){
  if (VOLUMES === null){ refreshVolumes().then(() => drawLayers()); }
  /* Prepared bodies only, exactly like the slice layer's backdrops: the
   * packaged ones, then everything the converter has turned into an .npz
   * (8-bit to the GPU, levels built for anything large). A new NIfTI goes
   * through the same preparation and is selected once it is ready. A body
   * set by path in an older figure still shows, under its file name. */
  const lib = VOLUMES || [];
  const entry = lib.find(v => v.name === L.source);
  const options = lib.map(v => ({value: v.name, label: v.label || v.name}));
  if (L.source && !entry){
    options.unshift({value: L.source,
                     label: 'file · ' + String(L.source).split('/').pop()});
  }
  if (!L.source) options.unshift({value: '', label: '— choose —'});
  field(b, 'Body', select(options, L.source || '', v => set('source', v || null)));
  const prepare = el('button', null, 'Prepare a NIfTI …');
  prepare.title = 'convert a NIfTI into a prepared body (the same conversion ' +
                  'as a slice backdrop); it is selected here when ready';
  prepare.onclick = () => chooseFile('volume', $('#m-path').value,
    p => startConvert(p, out => set('source', out.replace(/^.*\//, '').replace(/\.npz$/i, ''))));
  field(b, '', prepare);

  /* A body too large to ray-cast whole is drawn from block-averaged levels:
   * coarse while you rotate, cut and edit; the finest that fits for hi-res
   * and export. Each is built once and kept beside the archive. */
  const mmText = x => String(Math.round(x * 100) / 100);
  if (entry && entry.export_mm && entry.preview_mm > entry.mm * 1.0001){
    b.appendChild(el('div','hint',
      `${entry.shape.join('×')} at ${mmText(entry.mm)} mm is too large to ` +
      `ray-cast whole: it is drawn at ${mmText(entry.preview_mm)} mm while you ` +
      `work, and at ${mmText(entry.export_mm)} mm for hi-res and export.`));
    if (!entry.levels_ready){
      const prep = el('button', null, 'Prepare levels');
      prep.title = 'build both levels once, in the background — otherwise the ' +
                   'first render builds them and stalls until they exist';
      prep.onclick = async () => {
        try { await postJSON('/api/volumes/levels', {source: L.source}); pollConvert(); }
        catch (e) { status(e.message, true); }
      };
      field(b, 'Levels', prep);
    }
  }

  const pal = [{value:'gray', label:'gray'}];
  if (PALETTES){
    PALETTES.clut.forEach(n => pal.push({value:n, label:'MRIcroGL · ' + n}));
    PALETTES.matplotlib.forEach(n => { if (n !== 'gray') pal.push({value:n, label:'matplotlib · ' + n}); });
  } else { refreshPalettes().then(() => drawLayers()); }
  /* Greys by default -- a T1's own texture, as in MRIcroGL. A tint colours
   * that texture rather than replacing it: every colour keeps its brightness
   * and takes the tint's hue. */
  field(b, 'Colour', select(pal, L.cmap || 'gray', v => set('cmap', v)));
  /* The checkbox turns the tint on and off; the swatch picks it. Neither
   * rebuilds the panel, so the native colour picker stays open while you drag
   * through it -- the swatch is enabled in place, and picking a colour turns
   * the tint on if it was off. */
  const tintBox = el('div','grow inline');
  const tint = el('input'); tint.type = 'color'; tint.value = L.color || '#d9c2a6';
  const tintOn = el('input'); tintOn.type = 'checkbox'; tintOn.checked = !!L.color;
  tintOn.title = 'tint the palette, keeping its brightness';
  const tintText = el('span','hint', L.color || 'off');
  tintOn.onchange = () => {
    set('color', tintOn.checked ? tint.value : null);
    tintText.textContent = tintOn.checked ? tint.value : 'off';
  };
  tint.oninput = () => {
    tintOn.checked = true;
    tintText.textContent = tint.value;
    set('color', tint.value);
  };
  tintBox.append(tintOn, tint, tintText);
  field(b, 'Tint', tintBox);

  /* Where the body starts and saturates. Blank reads it off the data: the
   * start is the split between background/CSF and tissue, so the cortex is
   * the surface; the end is the 99th percentile. */
  const thr = pair(L.clim ? L.clim[0] : null, L.clim ? L.clim[1] : null,
                   'auto', 'auto',
                   (a, z) => set('clim', (a !== null && z !== null) ? [a, z] : null));
  thr.title = 'threshold: where the body becomes visible / where it saturates — ' +
              'blank reads both off the data';
  field(b, 'Threshold', thr);

  field(b, 'Opacity profile', select(
    [{value:'linear',     label:'Linear — density reads as density'},
     {value:'sigmoid',    label:'Sigmoid — a soft threshold'},
     {value:'geom',       label:'Top-weighted — faint tail'},
     {value:'foreground', label:'Foreground — flat, solid'}],
    L.opacity_profile || 'linear', v => set('opacity_profile', v)));
  const op = el('input'); op.type='range'; op.min=0; op.max=1; op.step=0.01;
  op.value = L.opacity; op.oninput = () => set('opacity', parseFloat(op.value));
  field(b, 'Opacity', op);
  field(b, 'Shade', check(L.shade !== false, v => set('shade', v),
                          'gradient shading from the scene lighting'));

  field(b, 'Overlays', select(
    [{value:'surface', label:'Paint the tissue the rays reach'},
     {value:'volume',  label:'X-ray through the brain'}],
    L.overlay_render || 'surface', v => set('overlay_render', v)));
  {
    /* Always shown: they shape x-rayed overlays and are simply unused while
     * overlays are painted on the surface.
     * X-ray: the brain is rendered as it always is -- opaque and lit -- and
     * the overlays are ray-cast through it on their own, each sample dimmed by
     * how deep behind the brain's surface it sits, exp(-depth / this). At this
     * depth a blob shows at 37%, at twice it 14%. Small: just under the cortex
     * (or a cut face). Large: everything the ray crosses. */
    const depth = el('input'); depth.type='range'; depth.min=0.5; depth.max=60; depth.step=0.5;
    depth.value = L.xray_depth_mm ?? 8;
    const shown = el('span','hint', (L.xray_depth_mm ?? 8) + ' mm');
    depth.oninput = () => { shown.textContent = depth.value + ' mm'; };
    depth.onchange = () => set('xray_depth_mm', parseFloat(depth.value));
    const wrap = el('div','grow inline'); wrap.append(depth, shown);
    b.appendChild(el('div','sec', 'X-ray' +
      ((L.overlay_render || 'surface') === 'volume' ? '' : ' — used when overlays are x-rayed')));
    field(b, 'X-ray depth', wrap);
    const fade = el('input'); fade.type='range'; fade.min=0; fade.max=1; fade.step=0.05;
    fade.value = L.overlay_fade ?? 0.03;
    fade.title = 'how much of each scale an overlay fades in over, from its threshold';
    fade.oninput = () => set('overlay_fade', parseFloat(fade.value));
    field(b, 'Fade in', fade);
    field(b, 'Smooth (vox)', number(L.overlay_smooth_vox ?? 0.75,
          v => set('overlay_smooth_vox', v ?? 0), {step: 0.25, placeholder: '0'}));
  }

  /* Overlays stay inside the body unless asked otherwise: a map that runs past
   * the brain would otherwise hang as colour in the air around it. */
  field(b, 'Allow outside volume', check(!!L.overlay_outside,
        v => set('overlay_outside', v),
        'draw overlay values that fall outside the body as well'));

  sectionEditor(b, L, set);
  b.appendChild(el('div','hint', (L.overlay_render || 'surface') === 'volume'
    ? 'Overlays are ray-cast through the brain, which stays opaque and lit. ' +
      'Each overlay sample is dimmed by its depth behind the brain\'s surface ' +
      '(or a cut face), so shallow colour is bright and deep colour faint but ' +
      'never hidden. Use a section to bring anything deep to the surface. ' +
      'Fade-in keeps a blob\'s edge from terracing; smoothing rounds the voxel ' +
      'staircase a masked map has at its edge — set it to 0 to turn it off.'
    : 'The body keeps its own shape and lighting; overlays change only its ' +
      'colour, on the cortex and on every cut face. Cuts are clipping planes ' +
      'with a solid face, so a cut through white matter reads as a flat section.'));
  overlayStack(b, L, set, 'volume');
}

let CONVERT_DONE = null;   // what to do with a finished conversion, if anything

async function startConvert(path, onDone){
  const answer = prompt(
    'Resolution in mm.\n\n' +
    'Leave blank for NATIVE — the scan\'s own voxel size, which is the point ' +
    'of having a high-resolution scan.\n\n' +
    'A native 100 um volume takes several minutes to convert and lands in the ' +
    'gigabytes. Enter a number to coarsen instead (0.4 is ~64x smaller).', '');
  if (answer === null) return;
  const mm = answer.trim() === '' ? null : parseFloat(answer);
  try {
    await postJSON('/api/backdrops', {path, mm});
    CONVERT_DONE = onDone || null;
    pollConvert();
  } catch (e) { status(e.message, true); }
}

async function pollConvert(){
  try {
    const j = await getJSON('/api/backdrops/job');
    if (j.error){ status(j.error, true); return; }
    if (j.running){
      status(`converting ${j.name} … ${(j.progress*100).toFixed(0)}%`);
      setTimeout(pollConvert, 1200); return;
    }
    await refreshBackdrops();
    await refreshVolumes();
    status('ready: ' + (j.name || ''));
    const done = CONVERT_DONE; CONVERT_DONE = null;
    if (done && j.path && /\.npz$/i.test(j.path)) done(j.path);
    drawLayers();
  } catch (e) { status(e.message, true); }
}

function electrodeBody(b, L, set){
  pathField(b, 'Reconstruction', L.source, 'fibers', v => set('source', v));
  pathField(b, 'Model (optional)', L.model, 'fibers', v => set('model', v));
  field(b, 'Insulation', colour(L.insulation_color, v => set('insulation_color', v)));
  field(b, 'Contacts', colour(L.contact_color, v => set('contact_color', v)));
  const op = el('input'); op.type='range'; op.min=0; op.max=1; op.step=0.02;
  op.value = L.opacity; op.oninput = () => set('opacity', parseFloat(op.value));
  field(b, 'Opacity', op);
  b.appendChild(el('div','hint',
    'A Lead-DBS reconstruction places the lead from its four markers — head, ' +
    'tail, x and y — so the segmented contacts of a directed lead land on the ' +
    'right side. Drop an ea_reconstruction.mat, or a model .mat to see a lead ' +
    'at its own template position. Leave Model empty to use the one the ' +
    'reconstruction names.'));
}

/* ---------- pruning ----------
 * Waypoint gates, the way a tractography viewer isolates a bundle: a fibre is
 * kept only if it crosses every "through" plane and none of the "avoid" ones.
 * Crossing, not proximity -- a fibre that runs alongside a plane and stops
 * short of it has not gone through there.
 */
function pruneEditor(b, L, set){
  const wrap = el('details','fold prune');
  wrap.appendChild(el('summary', null, 'Prune'));
  const body = el('div'); wrap.appendChild(body); b.appendChild(wrap);
  const gates = L.prune || (L.prune = []);
  if (gates.length) wrap.open = true;

  gates.forEach((g, i) => {
    const row = el('div','row');
    const on = el('input'); on.type='checkbox'; on.checked = g.enabled !== false;
    on.onchange = () => { g.enabled = on.checked; set('prune', gates); };
    const mode = select([{value:'through', label:'hits'},
                         {value:'avoid',   label:'avoids'}], g.mode || 'through',
                        v => { g.mode = v; set('prune', gates); });
    mode.style.cssText = 'width:86px;flex:0 0 86px';

    /* A VOLUME gate: the ROI is a NIfTI rather than a plane. A fibre is inside
     * it wherever |value| > threshold, so a binary mask at 0 means "any
     * non-zero voxel" and a t-map at 2 means either tail. */
    if (g.kind === 'volume'){
      const del2 = el('button','icon','✕');
      del2.onclick = () => { gates.splice(i,1); set('prune', gates); drawLayers(); };
      const name = el('div','name', (g.nifti || '(no volume)').split('/').pop());
      name.title = g.nifti || '';
      const thr = el('input'); thr.type='number'; thr.step=0.1;
      thr.value = g.threshold == null ? 0 : g.threshold;
      thr.title = 'a voxel counts as inside when |value| is above this';
      thr.style.cssText = 'width:64px;flex:0 0 64px;text-align:right';
      thr.onchange = () => { g.threshold = parseFloat(thr.value) || 0;
                             set('prune', gates); };
      const vbox = el('div','grow inline');
      vbox.append(on, mode, name, el('span',null,'>'), thr, del2);
      row.appendChild(el('label', null, i ? 'and' : 'keep'));
      row.appendChild(vbox); body.appendChild(row);
      return;
    }
    const ax = select([{value:'x',label:'X'},{value:'y',label:'Y'},{value:'z',label:'Z'}],
                      g.axis, v => { g.axis = v; set('prune', gates); });
    ax.style.cssText = 'width:50px;flex:0 0 50px';
    const at = el('input'); at.type='number'; at.step=0.5; at.value = g.at;
    at.style.cssText = 'width:64px;flex:0 0 64px;text-align:right';
    const slide = el('input'); slide.type='range';
    slide.min=-90; slide.max=90; slide.step=1; slide.value=g.at;
    slide.title = 'move the gate through the brain';
    at.onchange = () => { g.at = parseFloat(at.value) || 0; slide.value = g.at;
                          set('prune', gates); };
    slide.oninput = () => { g.at = parseFloat(slide.value); at.value = g.at;
                            set('prune', gates); };
    const del = el('button','icon','✕');
    del.onclick = () => { gates.splice(i,1); set('prune', gates); drawLayers(); };
    const box = el('div','grow inline');
    box.append(on, mode, ax, at, el('span',null,'mm'), del);
    row.appendChild(el('label', null, i ? 'and' : 'keep'));
    row.appendChild(box); body.appendChild(row);
    const srow = el('div','row'); srow.appendChild(el('label', null, ''));
    const sbox = el('div','grow'); sbox.appendChild(slide);
    srow.appendChild(sbox); body.appendChild(srow);
  });

  const add = el('button', null, '+ Add gate');
  add.onclick = () => { gates.push({axis:'x', at:0, mode:'through', enabled:true});
                        set('prune', gates); drawLayers(); };
  const row = el('div','row'); row.appendChild(el('label', null, ''));
  const box = el('div','grow'); box.appendChild(add); row.appendChild(box);
  body.appendChild(row);
  body.appendChild(el('div','hint',
    'Drop a NIfTI on this layer to gate on an ROI: fibres that hit it, or ' +
    'that avoid it. A plane gate is the same idea in world millimetres. ' +
    '"through" keeps only fibres ' +
    'that cross it, "avoid" drops them; gates combine, so several together ' +
    'isolate one bundle out of a whole-brain atlas. Applies to this layer ' +
    'only, and to the exported figure exactly as shown here.'));
}

function fiberBody(b, L, set){
  pathField(b, 'Source', L.source, 'fibers', v => set('source', v));
  field(b, 'Colour by', select(CAT.colorings, L.coloring, v => set('coloring', v)));
  if (L.coloring !== 'direction'){
    /* The atlas fibres are the backdrop to a finding. Off, only fibres an
     * overlay actually paints are drawn, which is the difference between a
     * bundle you can read and the whole tractogram. */
    const box = el('div','grow inline');
    const pick = colour(L.color, v => set('color', v));
    const show = check(!L.values_only,
          v => { set('values_only', !v); drawLayers(); },
          'draw the unpainted atlas fibres in this colour');
    box.append(pick, show);
    const row = el('div','row'); row.appendChild(el('label', null, 'Colour'));
    row.appendChild(box); b.appendChild(row);
  }

  /* Millimetres, not pixels: hi-res sweeps this as real geometry, so the
   * ceiling here is what stops a drag from producing massive tubes. */
  const w = el('input'); w.type = 'range'; w.min = 0.05; w.max = 0.5; w.step = 0.01;
  w.title = 'fibre radius in mm — the same thickness in preview and hi-res';
  w.value = L.tube_radius == null ? 0.2 : L.tube_radius;
  w.oninput = () => set('tube_radius', parseFloat(w.value));
  field(b, 'Thickness (mm)', w);
  const op = el('input'); op.type = 'range'; op.min = 0; op.max = 1; op.step = 0.01;
  op.value = L.opacity; op.oninput = () => set('opacity', parseFloat(op.value));
  field(b, 'Opacity', op);

  /* One value per atlas fibre, layered. The atlas is the geometry and each
   * .fib.values.npy is a finding painted onto it, so a bundle can carry a
   * background colouring that later findings overwrite only where they have
   * something to say. */
  overlayStack(b, L, set, 'fibers');

  pruneEditor(b, L, set);

  b.appendChild(el('div','sec','Which streamlines'));
  field(b, 'Min |value|', number(L.min_abs_value, v => set('min_abs_value', v)));
  field(b, 'Max |value|', number(L.max_abs_value, v => set('max_abs_value', v)));
  const pct = number(L.top_percent, v => set('top_percent', v));
  pct.title = 'keep only this percentage of fibres, the strongest by |value|';
  field(b, 'Keep strongest %', pct);
  field(b, 'Max lines', number(L.max_lines, v => set('max_lines', v), {step:1000}));
  b.appendChild(el('div','hint',
    'These pick which streamlines are drawn at all, by the values the layer was ' +
    'loaded with. A p-value map wants Max |value| = 0.05, not a minimum — the ' +
    'fibres you want are the small ones. Max lines then subsamples whatever ' +
    'survives, for speed.'));
}

/* Enough to close the speckle without smearing a map across the cortex. */
const FILL_PASSES = 3;

/* Fields that change which controls a panel shows, so editing them rebuilds
 * it: a volume's body (its size hints and levels button) and how its overlays
 * are drawn (the x-ray heading). */
const RESTRUCTURE = new Set(['coloring', 'mesh', 'backdrop', 'source',
                             'overlay_render']);

const LIGHT_CONTROLS = [
  ['key_azimuth',    'Key azimuth',   -180, 180, 1],
  ['key_elevation',  'Key elevation',  -90,  90, 1],
  ['key_intensity',  'Key strength',     0, 1.6, 0.05],
  ['specular',       'Specular',         0,   1, 0.02],
  ['specular_power', 'Highlight size',   1, 100, 1],
  ['ambient',        'Ambient',          0,   1, 0.02],
  ['diffuse',        'Diffuse',          0,   1, 0.02],
];

const BACKGROUNDS = [
  {value:'#ffffff', label:'white'},
  {value:'#000000', label:'black'},
  {value:'#0f1216', label:'near-black'},
  {value:'#f2f3f5', label:'paper'},
];

/* Rendering time grows with the pixel count, so 8000 is a screenshot you wait a
 * few seconds for, not a preview setting. SVG is deliberately not offered: VTK's
 * vector export falls back to a single embedded raster for a shaded mesh, so it
 * would be this same image in a wrapper, at no better resolution. */
const EXPORT_SIZES = [
  {value: 2000, label: '2000 px — screen'},
  {value: 4000, label: '4000 px — print'},
  {value: 6000, label: '6000 px — large print'},
  {value: 8000, label: '8000 px — poster'},
];

function drawScene(){
  const host = $('#scene'); host.innerHTML = '';
  const box = el('div','grow inline');
  const pick = colour(SPEC.background, v => { SPEC.background = v; drawScene(); render(); });
  const preset = select([{value:'', label:'—'}, ...BACKGROUNDS], '',
    v => { if (v){ SPEC.background = v; drawScene(); render(); } });
  box.append(pick, preset);
  const row = el('div','row'); row.appendChild(el('label', null, 'Background'));
  row.appendChild(box); host.appendChild(row);
  field(host, 'Export size', select(EXPORT_SIZES, SPEC.export_px || 4000,
        v => { SPEC.export_px = parseInt(v, 10); }));
  host.appendChild(el('div','hint',
    'Scale-bar text follows the background, so a dark backdrop does not hide it. ' +
    'Export size is the long edge of the saved PNG and is independent of this ' +
    'window — the preview pane is about a thousand pixels across, which is why ' +
    'a figure saved at pane size pixellates as soon as you zoom in.'));
}

function drawLighting(){
  const host = $('#lighting'); host.innerHTML = '';
  field(host, 'Rig', select(
    [{value:'studio', label:'studio — one key, real shadows'},
     {value:'kit',    label:'light kit — flat, matches yabplot'}],
    SPEC.lighting.rig || 'studio',
    v => { SPEC.lighting.rig = v; render(); }));
  LIGHT_CONTROLS.forEach(([k, label, lo, hi, step]) => {
    const r = el('input'); r.type = 'range'; r.min = lo; r.max = hi; r.step = step;
    r.value = SPEC.lighting[k];
    r.oninput = () => { SPEC.lighting[k] = parseFloat(r.value); render(); };
    field(host, label, r);
  });
  const par = el('input'); par.type = 'checkbox'; par.checked = !!SPEC.parallel;
  par.onchange = () => { SPEC.parallel = par.checked; render(); };
  field(host, 'Orthographic', par);
  const reset = el('button', null, 'Reset lighting');
  reset.onclick = async () => {
    SPEC.lighting = (await getJSON('/api/lighting'));
    drawLighting(); render();
  };
  host.appendChild(reset);
  host.appendChild(el('div','hint',
    'Lights follow the camera, so a region is lit the same way in every view. ' +
    'Azimuth is degrees right of the lens, elevation degrees above it; fill and ' +
    'rim swing with the key. Diffuse carries the form, ambient only keeps the ' +
    'shadow side readable — raising ambient is what flattens a render.'));
}

function drawLayers(){
  const host = $('#layers'); host.innerHTML = '';
  SPEC.layers.forEach((L, i) => {
    const card = el('div', 'layer' + (OPEN === L.id ? ' open sel' : '') +
                             (L.visible ? '' : ' hidden'));
    const head = el('div','head');
    const eye = el('button','icon', L.visible ? '👁' : '◌');
    eye.title = 'show / hide';
    eye.onclick = e => { e.stopPropagation(); L.visible = !L.visible; drawLayers(); render(); };
    const name = el('div','name', L.name);
    name.ondblclick = e => { e.stopPropagation();
      const v = prompt('Layer name', L.name); if (v){ L.name = v; drawLayers(); } };
    const badge = el('span','badge', L.kind);
    const up = el('button','icon','↑'); up.title = 'move up';
    up.onclick = e => { e.stopPropagation(); if (i > 0){
      SPEC.layers.splice(i - 1, 0, SPEC.layers.splice(i, 1)[0]); drawLayers(); render(); } };
    const down = el('button','icon','↓'); down.title = 'move down';
    down.onclick = e => { e.stopPropagation(); if (i < SPEC.layers.length - 1){
      SPEC.layers.splice(i + 1, 0, SPEC.layers.splice(i, 1)[0]); drawLayers(); render(); } };
    const del = el('button','icon','✕'); del.title = 'delete';
    del.onclick = e => { e.stopPropagation(); SPEC.layers.splice(i, 1);
      if (OPEN === L.id) OPEN = null; drawLayers(); render(); };
    head.append(eye, name, badge, up, down, del);
    head.onclick = () => { OPEN = (OPEN === L.id ? null : L.id); drawLayers(); };
    card.appendChild(head);
    dropTarget(card, files => dropOntoLayer(L, files));

    if (OPEN === L.id){
      const body = el('div','body');
      /* Rebuilding the panel on every edit destroys the control under the
       * cursor, which makes a slider drag impossible and loses focus in a text
       * field. Only fields that change WHICH controls exist trigger a rebuild. */
      const set = (k, v) => {
        L[k] = v;
        if (RESTRUCTURE.has(k)) drawLayers();
        render();
      };
      ({mesh: meshBody, fibers: fiberBody, slices: sliceBody,
        electrode: electrodeBody, volume: volumeBody}[L.kind])(body, L, set);
      card.appendChild(body);
    }
    host.appendChild(card);
  });
}

async function volumeHint(parent, path){
  const note = el('div','hint','reading volume…');
  parent.appendChild(note);
  try {
    const v = await getJSON('/api/volume?' + new URLSearchParams({path}));
    note.textContent = `${v.shape.join('×')} · range ${v.min.toFixed(2)}…` +
      `${v.max.toFixed(2)} · ${v.distinct.toLocaleString()} distinct value(s)` +
      ` · auto → ${v.suggested_split}`;
  } catch (e) { note.textContent = e.message; }
}

/* ---------- drag and drop ----------
 * Where a file lands decides what it means. On the New group box it is
 * GEOMETRY: the volume is surfaced and becomes a group of its own. On an
 * existing layer it is COLOUR: the volume is sampled inside the meshes that
 * layer already has. The same file does two different jobs, and the drop target
 * is the only thing that says which -- so every target highlights on hover.
 */
const VOLUME_RE = /\.nii(\.gz)?$/i;
const TRACT_RE  = /\.(mat|tck|trk|json|npz)$/i;
/* Values, not geometry: one number per atlas fibre. Checked before the
 * tract pattern, because .fib.values.npy matches neither cleanly. */
const VALUES_RE = /\.fib\.values\.npy$|\.values\.npy$|\.fib\.npy$/i;
/* The descriptor stands in for the values it describes, so dropping one on a
 * layer that already has fibres adds a finding rather than replacing them. */
const DESC_RE   = /\.fib\.desc\.json$/i;

async function upload(files){
  const out = [];
  for (const f of files){
    status('uploading ' + f.name + ' …');
    const body = new FormData(); body.append('file', f);
    const r = await fetch('/api/upload', {method:'POST', body});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error);
    out.push(j);
  }
  return out;
}

async function dropAsNewGroup(files){
  const volumes = files.filter(f => VOLUME_RE.test(f.name));
  const tracts  = files.filter(f => TRACT_RE.test(f.name));
  if (!volumes.length && !tracts.length)
    throw new Error('drop a .nii/.nii.gz volume, a .mat/.tck tractogram, or a lead .mat/.json');

  if (volumes.length){
    const up = await upload(volumes);
    const layer = await getJSON('/api/layer/mesh');
    Object.assign(layer, {
      name: up.length === 1 ? up[0].name.split('.nii')[0] : `${up.length} volumes`,
      mesh: up.length === 1 ? up[0].path : up.map(u => u.path),
      color: '#c15656', tessellate: 0, split: 'auto',
    });
    SPEC.layers.push(layer); OPEN = layer.id;
    status(`surfacing ${up.length} volume(s) — this takes a moment …`);
  }
  if (tracts.length){
    /* .mat is either a tractogram or a lead reconstruction. The server looked
     * inside and said which; trusting the extension would try to render a
     * 300 MB tractogram as an electrode, or vice versa. */
    const up = await upload(tracts);
    for (const u of up){
      if (u.kind === 'fiber_desc'){
        /* A descriptor is a whole layer in one file: it names the atlas the
         * values were written against, so dropping one gives the fibres AND
         * the first finding on them. More .fib.values.npy can then be dropped
         * onto the layer. */
        const d = await getJSON('/api/descriptor?path=' + encodeURIComponent(u.path));
        const layer = await getJSON('/api/layer/fibers');
        const o = await newOverlay(u.path);
        o.mode = 'vertex'; o.name = d.name; o.hide_below_min = true;
        Object.assign(layer, {name: d.name, source: d.atlas, overlays: [o]});
        SPEC.layers.push(layer); OPEN = layer.id;
        status(`${d.name}: ${d.fiber_count} fibers from ${d.atlas.split('/').pop()} …`);
        continue;
      }
      const electrode = u.kind === 'electrode';
      const layer = await getJSON('/api/layer/' + (electrode ? 'electrode' : 'fibers'));
      Object.assign(layer, {name: u.name.split('.')[0], source: u.path});
      SPEC.layers.push(layer); OPEN = layer.id;
      if (electrode) status('placing ' + u.name + ' …');
    }
  }
  drawLayers(); await render();
}

async function dropOntoLayer(layer, files){
  const volumes = files.filter(f => VOLUME_RE.test(f.name));
  const tracts  = files.filter(f => TRACT_RE.test(f.name));

  if (layer.kind === 'fibers'){
    const vals = files.filter(f => VALUES_RE.test(f.name) || DESC_RE.test(f.name));
    if (vals.length){
      /* Values join the fibres already here rather than replacing them: the
       * atlas is the geometry and these are findings on it. */
      const stack = layer.overlays || (layer.overlays = []);
      for (const up of await upload(vals)){
        const o = await newOverlay(up.path);
        o.name = up.name;
        o.color = OVERLAY_COLOURS[stack.length % OVERLAY_COLOURS.length];
        o.mode = 'vertex';
        /* On a MESH the foot of a sequential ramp is a pale neutral, and that
         * reads correctly as unpainted anatomy. A fibre has no anatomy under
         * it, so the same pale foot just looks like the colour stopped
         * applying -- set the scale to [0, 2] on data that runs to -4 and
         * two thirds of the bundle washes out to grey. A fibre below the
         * scale minimum should fall back to the atlas colour instead. Still
         * a checkbox on the card; this only changes what it starts as. */
        o.hide_below_min = true;
        stack.push(o);
      }
      status(`adding ${vals.length} finding(s) to ${layer.name} …`);
    } else if (tracts.length){
      layer.source = (await upload(tracts))[0].path;
    } else if (volumes.length){
      /* A NIfTI cannot be per-fibre values, so on a fibre layer it can only
       * mean an ROI. It used to be rejected outright; now it lands as a prune
       * gate, which is the whole "drop a volume in and keep what hits it". */
      const gates = layer.prune || (layer.prune = []);
      for (const up of await upload(volumes))
        gates.push({kind:'volume', nifti: up.path, threshold: 0,
                    mode:'through', enabled:true});
      status(`gating ${layer.name} on ${volumes.length} ROI(s) …`);
    } else {
      throw new Error('a fiber layer takes an atlas, a tractogram, a ' +
                      '.fib.values.npy, or a NIfTI to gate on');
    }
  } else {
    if (!volumes.length) throw new Error('drop a NIfTI here to colour this group');
    const stack = layer.overlays || (layer.overlays = []);
    for (const up of await upload(volumes)){
      const o = await newOverlay(up.path);
      /* Two maps that happen to share a filename now land at DIFFERENT paths,
       * so the card would read "contrast_tval_FWE_0.a1b2c3d4.nii.gz". Keep the
       * name the file was dropped under for the label; the full path is still
       * in its tooltip. */
      o.name = up.name;
      o.color = OVERLAY_COLOURS[stack.length % OVERLAY_COLOURS.length];
      stack.push(o);
    }
    status(`adding ${volumes.length} volume(s) to ${layer.name} …`);
  }
  OPEN = layer.id; drawLayers(); await render();
}

function dropTarget(node, handler){
  let depth = 0;
  node.addEventListener('dragenter', e => {
    e.preventDefault(); e.stopPropagation();
    if (depth++ === 0) node.classList.add('over');
  });
  node.addEventListener('dragover', e => { e.preventDefault(); e.stopPropagation(); });
  node.addEventListener('dragleave', e => {
    e.stopPropagation(); if (--depth <= 0){ depth = 0; node.classList.remove('over'); }
  });
  node.addEventListener('drop', async e => {
    e.preventDefault(); e.stopPropagation();
    depth = 0; node.classList.remove('over');
    const files = [...(e.dataTransfer?.files || [])];
    if (!files.length) return;
    try { await handler(files); } catch (err) { status(err.message, true); }
  });
}

dropTarget($('#newgroup'), dropAsNewGroup);
dropTarget($('#stage'), dropAsNewGroup);
$('#newgroup').addEventListener('click', () => $('#b-mesh').click());

/* ---------- file chooser ---------- */
let chooserResolve = null;
async function browse(path, kind){
  const j = await getJSON('/api/browse?' + new URLSearchParams({path: path || '', kind}));
  $('#m-path').value = j.path;
  const list = $('#list'); list.innerHTML = '';
  j.entries.forEach(e => {
    const row = el('div', e.dir ? 'd' : '', (e.dir ? '📁 ' : '📄 ') + e.name);
    row.onclick = () => { if (e.dir) browse(e.path, kind);
                          else { $('#m-name').value = e.path; } };
    row.ondblclick = () => { if (!e.dir) finishChoose(e.path); };
    list.appendChild(row);
  });
  $('#m-up').onclick = () => browse(j.parent, kind);
}
function chooseFile(kind, start, cb){
  chooserResolve = cb;
  $('#m-title').textContent = {volume:'Choose a NIfTI', fibers:'Choose a fiber file',
                               save:'Save as', mesh:'Choose a mesh'}[kind] || 'Choose a file';
  $('#m-name').value = '';
  $('#modal').classList.add('on');
  browse(start, kind).catch(e => status(e.message, true));
}
function finishChoose(path){
  $('#modal').classList.remove('on');
  const cb = chooserResolve; chooserResolve = null;
  if (cb) cb(path);
}
$('#m-cancel').onclick = () => { $('#modal').classList.remove('on'); chooserResolve = null; };
$('#m-ok').onclick = () => { const v = $('#m-name').value.trim();
  if (v) finishChoose(v.includes('/') ? v : ($('#m-path').value + '/' + v)); };
$('#m-path').onchange = () => browse($('#m-path').value, 'any').catch(e => status(e.message, true));

/* ---------- orbit, pan, zoom ----------
 * Left-drag orbits. Right-drag, middle-drag or shift-drag pans: the camera and
 * its focal point slide together, by exactly the distance the cursor moved, so
 * a close zoom can be steered onto any region. The wheel zooms towards the
 * cursor, with no practical limit, for ultra-high-resolution bodies. */
const stage = $('#stage');
let drag = null;
stage.addEventListener('contextmenu', e => e.preventDefault());
stage.addEventListener('mousedown', e => {
  const pan = e.button === 1 || e.button === 2 || e.shiftKey;
  drag = {x: e.clientX, y: e.clientY, az: SPEC.camera.azimuth, el: SPEC.camera.elevation,
          pan, start: (SPEC.pan || [0, 0, 0]).slice()};
  stage.classList.add('drag');
  e.preventDefault();
});
function mmPerPixel(){
  const h = stage.getBoundingClientRect().height || 1;
  return VIEW ? VIEW.height_mm / h : 0;
}
window.addEventListener('mousemove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (drag.pan){
    if (!VIEW) return;
    const mm = mmPerPixel();
    SPEC.pan = [0, 1, 2].map(k => drag.start[k] - VIEW.right[k] * dx * mm
                                                + VIEW.up[k] * dy * mm);
  } else {
    SPEC.camera.azimuth = (drag.az + dx * 0.45) % 360;
    SPEC.camera.elevation = Math.max(-89.9, Math.min(89.9, drag.el - dy * 0.45));
  }
  render();
});
window.addEventListener('mouseup', () => { drag = null; stage.classList.remove('drag'); });
stage.addEventListener('wheel', e => {
  e.preventDefault();
  const f = e.deltaY > 0 ? 0.94 : 1.064;
  const zoom = Math.max(0.05, Math.min(400, SPEC.camera.zoom * f));
  const applied = zoom / SPEC.camera.zoom;
  if (VIEW && applied !== 1){
    /* Keep the point under the cursor where it is. */
    const box = stage.getBoundingClientRect();
    const cx = e.clientX - box.left - box.width / 2;
    const cy = e.clientY - box.top - box.height / 2;
    const mm = mmPerPixel();
    const pan = SPEC.pan || [0, 0, 0];
    SPEC.pan = [0, 1, 2].map(k => pan[k] + (VIEW.right[k] * cx * mm
                                          - VIEW.up[k] * cy * mm) * (1 - 1 / applied));
  }
  SPEC.camera.zoom = zoom;
  render();
}, {passive: false});

/* ---------- toolbar ---------- */
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => {
  const [az, elv] = VIEWS[b.dataset.view];
  SPEC.camera.azimuth = az; SPEC.camera.elevation = elv; SPEC.pan = [0, 0, 0]; render();
});
$('#b-reset').onclick = () => {
  SPEC.camera = {azimuth:90, elevation:0, zoom:1.0}; SPEC.pan = [0, 0, 0]; render(); };
/* Hi-res swaps streamlines from line primitives to real swept tubes. It is a
 * second or two of geometry for 30k fibres, which is the right trade for a
 * screenshot and the wrong one while orbiting -- so it is a button, not a mode
 * the app chooses for you. Export always renders this way regardless. */
$('#b-hires').onclick = () => {
  HIRES = !HIRES; $('#b-hires').classList.toggle('on', HIRES);
  status(HIRES ? 'hi-res: building tube geometry …' : 'preview quality');
  render();
};
/* A prepared backdrop, as a surface you cut into rather than three planes.
 * Same layer type as + Mesh; what it saves is the setup -- surface from the
 * volume, capped cut, per-vertex colouring, one cut already in place -- which
 * is a lot of clicking to discover and easy to get subtly wrong. */
$('#b-mesh').onclick = async () => {
  SPEC.layers.push(await getJSON('/api/layer/mesh')); OPEN = SPEC.layers.at(-1).id;
  drawLayers(); render(); };
$('#b-volume').onclick = async () => {
  /* The prepared bodies are fetched first, so the new card opens with its
   * dropdown already filled rather than empty until it is reopened. */
  await refreshVolumes();
  SPEC.layers.unshift(await getJSON('/api/layer/volume'));
  OPEN = SPEC.layers[0].id; drawLayers(); render();
};
$('#b-slices').onclick = async () => {
  await refreshBackdrops();
  SPEC.layers.unshift(await getJSON('/api/layer/slices'));
  OPEN = SPEC.layers[0].id; drawLayers(); render(); };
$('#b-fibers').onclick = async () => {
  SPEC.layers.push(await getJSON('/api/layer/fibers')); OPEN = SPEC.layers.at(-1).id;
  drawLayers(); render(); };
$('#b-new').onclick = async () => {
  if (!confirm('Discard the current figure?')) return;
  SPEC.layers = [await getJSON('/api/layer/mesh')]; OPEN = null; drawLayers(); render(); };
$('#b-open').onclick = () => chooseFile('any', $('#m-path').value, async p => {
  try { SPEC = await getJSON('/api/project?' + new URLSearchParams({path: p}));
        OPEN = null; drawLayers(); render(); status('opened ' + p); }
  catch (e) { status(e.message, true); } });
$('#b-save').onclick = () => chooseFile('save', $('#m-path').value, async p => {
  if (!p.endsWith('.json')) p += '.json';
  try { const j = await postJSON('/api/project', {path: p, spec: SPEC});
        status('saved ' + j.path); } catch (e) { status(e.message, true); } });
/* Export is rendered at its own size, not the viewport's. The preview pane is
 * about 1000 px wide, so exporting at the pane's size -- which is what used to
 * happen -- gives a figure that pixellates the moment it is zoomed. */
$('#b-export').onclick = () => chooseFile('save', $('#m-path').value, async p => {
  if (!/\.(png|jpg|jpeg)$/i.test(p)) p += '.png';
  status('rendering ' + p + ' …');
  const stage = $('#stage').getBoundingClientRect();
  const long = SPEC.export_px || 4000;
  const scale = long / Math.max(stage.width, stage.height);
  try { const j = await postJSON('/api/export', {path: p,
          width: Math.round(stage.width * scale),
          height: Math.round(stage.height * scale),
                                                 scalar_bar: true});
        status('exported ' + j.path); } catch (e) { status(e.message, true); } });

/* ---------- animation: recording and camera paths ----------
 * A small floating panel, opened from the view bar so it costs the side panel
 * nothing. RECORD keeps every frame the view shows while you orbit, pan, zoom
 * or play a series, with how long each stayed up, and writes them as a GIF.
 * A PATH moves the camera by itself -- orbit, rock, tilt, zoom, or through
 * keyframes you set -- optionally stepping 4D series along with it. Preview
 * plays it here; Render GIF draws it frame by frame on the server at a chosen
 * size and quality, independent of how fast the screen keeps up. */
let KEYS = [], PATH_STOP = false;
function cameraNow(){
  return {azimuth: SPEC.camera.azimuth, elevation: SPEC.camera.elevation,
          zoom: SPEC.camera.zoom, pan: (SPEC.pan || [0, 0, 0]).slice()};
}
const ease = t => t * t * (3 - 2 * t);
function lerpCamera(a, b, t){
  let dAz = ((b.azimuth - a.azimuth) % 360 + 540) % 360 - 180;   // shortest way round
  return {azimuth: a.azimuth + dAz * t,
          elevation: a.elevation + (b.elevation - a.elevation) * t,
          zoom: a.zoom * Math.pow(b.zoom / a.zoom, t),
          pan: [0, 1, 2].map(k => a.pan[k] + (b.pan[k] - a.pan[k]) * t)};
}
async function pathSteps(preset, n, series){
  const base = cameraNow(), steps = [];
  const clampEl = v => Math.max(-89.9, Math.min(89.9, v));
  for (let i = 0; i < n; i++){
    const t = n > 1 ? i / n : 0;            // a loop: the last frame leads back to the first
    let c = {...base, pan: base.pan.slice()};
    if (preset === 'orbit') c.azimuth = base.azimuth + 360 * t;
    else if (preset === 'rock') c.azimuth = base.azimuth + 30 * Math.sin(2 * Math.PI * t);
    else if (preset === 'tilt') c.elevation = clampEl(base.elevation + 30 * Math.sin(2 * Math.PI * t));
    else if (preset === 'zoom') c.zoom = base.zoom * (1 + 1.5 * Math.sin(Math.PI * t));
    else if (preset === 'keys' && KEYS.length > 1){
      const u = (n > 1 ? i / (n - 1) : 0) * (KEYS.length - 1);
      const k = Math.min(Math.floor(u), KEYS.length - 2);
      c = lerpCamera(KEYS[k], KEYS[k + 1], ease(u - k));
    }
    steps.push(c);
  }
  if (series){
    /* Each 4D overlay runs through its frames once over the path. */
    for (const L of SPEC.layers){
      if (!L.visible) continue;
      for (const [index, o] of (L.overlays || []).entries()){
        const count = await frameCount(o.nifti);
        if (count <= 1) continue;
        steps.forEach((st, i) => {
          st.frames = st.frames || {};
          (st.frames[L.id] = st.frames[L.id] || {})[index] = Math.floor(i / steps.length * count);
        });
      }
    }
  }
  return steps;
}
function applyStep(st){
  SPEC.camera = {azimuth: st.azimuth, elevation: st.elevation, zoom: st.zoom};
  SPEC.pan = st.pan;
  for (const [id, entries] of Object.entries(st.frames || {})){
    const L = SPEC.layers.find(l => l.id === id);
    if (L) for (const [index, frame] of Object.entries(entries)) L.overlays[index].frame = frame;
  }
}
function animPanel(){
  let box = $('#anim');
  if (box){ box.classList.toggle('on'); return; }
  box = el('div'); box.id = 'anim'; box.className = 'on';
  box.innerHTML = `
    <header><strong>Animate</strong><button class="icon" id="a-close">✕</button></header>
    <div class="sec">Record what the view shows</div>
    <div class="arow"><button id="a-rec">● Record</button><span class="hint" id="a-rec-n"></span></div>
    <div class="sec">Camera path</div>
    <div class="arow"><select id="a-preset">
      <option value="orbit">Orbit 360°</option><option value="rock">Rock ±30°</option>
      <option value="tilt">Tilt ±30°</option><option value="zoom">Zoom in and out</option>
      <option value="keys">Through keyframes</option></select></div>
    <div class="arow"><button id="a-key">+ Keyframe</button><button id="a-keys-clear">Clear</button>
      <span class="hint" id="a-keys-n">0 keyframes</span></div>
    <div class="arow"><label>Seconds</label><input id="a-sec" type="number" value="6" min="1" step="1">
      <label>FPS</label><input id="a-fps" type="number" value="15" min="2" max="30" step="1"></div>
    <div class="arow"><label><input id="a-series" type="checkbox"> step 4D series along the path</label></div>
    <div class="arow"><label>GIF size</label><select id="a-size">
      <option value="1">view size</option><option value="1.5">1.5×</option><option value="2">2×</option></select>
      <label><input id="a-hq" type="checkbox"> hi-res</label></div>
    <div class="arow"><button id="a-preview">Preview</button><button id="a-gif" class="primary">Render GIF…</button></div>`;
  $('#view').appendChild(box);
  $('#a-close').onclick = () => box.classList.remove('on');
  const recN = $('#a-rec-n');
  $('#a-rec').onclick = () => {
    if (!RECORDING){
      RECORDING = []; $('#a-rec').textContent = '■ Stop and save GIF';
      recN.textContent = 'recording — move the view';
      const tick = setInterval(() => {
        if (!RECORDING){ clearInterval(tick); return; }
        recN.textContent = `${RECORDING.length} frame(s)`;
      }, 300);
      return;
    }
    const frames = RECORDING; RECORDING = null;
    $('#a-rec').textContent = '● Record'; recN.textContent = `${frames.length} frame(s)`;
    if (!frames.length){ status('nothing was recorded', true); return; }
    chooseFile('save', $('#m-path').value, async p => {
      if (!/\.gif$/i.test(p)) p += '.gif';
      const form = new FormData();
      form.append('path', p);
      const durations = frames.map((f, i) => i + 1 < frames.length
                                   ? frames[i + 1].t - f.t : 600);
      form.append('durations', JSON.stringify(durations));
      frames.forEach((f, i) => form.append('frames', f.blob, `f${i}.png`));
      status(`writing ${frames.length} frames …`);
      try {
        const r = await fetch('/api/gif', {method: 'POST', body: form});
        const j = await r.json(); if (!r.ok) throw new Error(j.error);
        status(`saved ${j.path} (${j.frames} frames)`);
      } catch (e) { status(e.message, true); }
    });
  };
  const keysN = () => $('#a-keys-n').textContent = `${KEYS.length} keyframe(s)`;
  $('#a-key').onclick = () => { KEYS.push(cameraNow()); keysN(); $('#a-preset').value = 'keys'; };
  $('#a-keys-clear').onclick = () => { KEYS = []; keysN(); };
  const settings = () => {
    const fps = Math.max(2, Math.min(30, parseFloat($('#a-fps').value) || 15));
    const n = Math.max(2, Math.round((parseFloat($('#a-sec').value) || 6) * fps));
    return {fps, n, preset: $('#a-preset').value, series: $('#a-series').checked};
  };
  $('#a-preview').onclick = async () => {
    const btn = $('#a-preview');
    if (btn.textContent === 'Stop'){ PATH_STOP = true; return; }
    const {fps, n, preset, series} = settings();
    if (preset === 'keys' && KEYS.length < 2){ status('set at least two keyframes', true); return; }
    const before = cameraNow(), steps = await pathSteps(preset, n, series);
    PATH_STOP = false; btn.textContent = 'Stop';
    for (const st of steps){
      if (PATH_STOP) break;
      const t0 = performance.now();
      applyStep(st); await renderSettled();
      const wait = 1000 / fps - (performance.now() - t0);
      if (wait > 0) await sleep(wait);
    }
    btn.textContent = 'Preview';
    applyStep(before); render();
  };
  $('#a-gif').onclick = () => {
    const {fps, n, preset, series} = settings();
    if (preset === 'keys' && KEYS.length < 2){ status('set at least two keyframes', true); return; }
    chooseFile('save', $('#m-path').value, async p => {
      if (!/\.gif$/i.test(p)) p += '.gif';
      const scale = parseFloat($('#a-size').value) || 1;
      const box = stage.getBoundingClientRect();
      const steps = await pathSteps(preset, n, series);
      try {
        await postJSON('/api/animate', {path: p, steps, fps, spec: SPEC,
          width: Math.round(box.width * scale), height: Math.round(box.height * scale),
          quality: $('#a-hq').checked ? 'export' : 'preview'});
        const poll = async () => {
          const j = await getJSON('/api/animate');
          if (j.error){ status(j.error, true); return; }
          if (j.running){ status(`rendering GIF … ${j.done} / ${j.total}`); setTimeout(poll, 700); return; }
          status(`saved ${j.path}`); render();
        };
        poll();
      } catch (e) { status(e.message, true); }
    });
  };
}
$('#b-anim').onclick = animPanel;

/* ---------- boot ---------- */
(async () => {
  try {
    CAT = await getJSON('/api/catalogue');
    SPEC = await getJSON('/api/spec');
    await Promise.all([refreshBackdrops(), refreshPalettes()]);
    drawLayers(); drawLighting(); drawScene();
    await render();
  } catch (e) { status(e.message, true); }
})();

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

function overlayStack(b, L, set, kind){
  b.appendChild(el('div','sec', kind === 'slices' ? 'Volumes on the slice'
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
              v => { o.nifti = v; set('overlays', stack); });
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
    b.appendChild(el('div','hint', kind === 'slices'
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

async function startConvert(path){
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
    status('backdrop ready: ' + (j.name || ''));
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
    const mode = select([{value:'through', label:'through'},
                         {value:'avoid',   label:'avoid'}], g.mode || 'through',
                        v => { g.mode = v; set('prune', gates); });
    mode.style.cssText = 'width:86px;flex:0 0 86px';
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
    'Each gate is a plane in world millimetres. "through" keeps only fibres ' +
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

const RESTRUCTURE = new Set(['coloring', 'mesh', 'backdrop']);

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
        electrode: electrodeBody}[L.kind])(body, L, set);
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
const TRACT_RE  = /\.(mat|tck|trk|json|npz)$|\.fib\.npy$/i;
/* Values, not geometry: one number per atlas fibre. Checked before the
 * tract pattern, because .fib.values.npy matches neither cleanly. */
const VALUES_RE = /\.fib\.values\.npy$|\.values\.npy$/i;
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
        o.mode = 'vertex'; o.name = d.name;
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
        o.color = OVERLAY_COLOURS[stack.length % OVERLAY_COLOURS.length];
        o.mode = 'vertex';
        stack.push(o);
      }
      status(`adding ${vals.length} finding(s) to ${layer.name} …`);
    } else if (tracts.length){
      layer.source = (await upload(tracts))[0].path;
    } else {
      throw new Error('a fiber layer takes an atlas, a tractogram, or .fib.values.npy');
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

/* ---------- orbit ---------- */
const stage = $('#stage');
let drag = null;
stage.addEventListener('mousedown', e => {
  drag = {x: e.clientX, y: e.clientY, az: SPEC.camera.azimuth, el: SPEC.camera.elevation};
  stage.classList.add('drag');
});
window.addEventListener('mousemove', e => {
  if (!drag) return;
  SPEC.camera.azimuth = (drag.az + (e.clientX - drag.x) * 0.45) % 360;
  SPEC.camera.elevation = Math.max(-89.9, Math.min(89.9, drag.el - (e.clientY - drag.y) * 0.45));
  render();
});
window.addEventListener('mouseup', () => { drag = null; stage.classList.remove('drag'); });
stage.addEventListener('wheel', e => {
  e.preventDefault();
  SPEC.camera.zoom = Math.max(0.4, Math.min(4, SPEC.camera.zoom * (e.deltaY > 0 ? 0.94 : 1.064)));
  render();
}, {passive: false});

/* ---------- toolbar ---------- */
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => {
  const [az, elv] = VIEWS[b.dataset.view];
  SPEC.camera.azimuth = az; SPEC.camera.elevation = elv; render();
});
$('#b-reset').onclick = () => { SPEC.camera = {azimuth:90, elevation:0, zoom:1.0}; render(); };
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

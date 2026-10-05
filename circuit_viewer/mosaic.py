"""Six-view figure mosaics, rendered by the viewer itself, from one call.

    from circuit_viewer.mosaic import mosaic

    img = mosaic("tmap.nii.gz")                          # x-rayed onto spm152
    img = mosaic("tmap.nii.gz", "fig.png", threshold=2.3)
    img = mosaic("result.fib.desc.json")                 # fibres in a glass brain
    img = mosaic("tmap.nii.gz", mode="mesh", mesh="pial_wholebrain")
    img = mosaic("tmap.nii.gz", mode="volume", render="surface", body="spm152")
    img = mosaic("tmap.nii.gz", mode="slice", axis="z")

The default layout is

    left lateral | right lateral | inferior
    left medial  | right medial  | superior

with one shared colour bar. ``mosaic`` returns a ``PIL.Image`` and also writes
it when given a path.

It is the SAME renderer as the viewer: every view goes through
``RenderSession`` and a normal spec, so a mosaic and the app agree pixel for
pixel and nothing here re-implements any drawing. That also means every viewer
setting can be reached from here:

* The common ones are named keyword arguments with the defaults listed in
  ``DEFAULTS`` below, per mode.
* Anything else is accepted by its viewer name and routed to the overlay, the
  layer or the scene, e.g. ``hide_below_min=True``, ``fade=0.3``,
  ``opacity_profile="sigmoid"``, ``lighting={"key_azimuth": 60}``. Names that
  mean different things on the overlay and the layer (``opacity``, ``color``,
  ``clim``, ``sign``, ``step``) always mean the OVERLAY; reach the layer's with
  ``layer={...}``. ``overlay={...}`` and ``scene={...}`` work the same way.
* ``spec_hook(spec, view)`` is called with each view's spec before it renders,
  as the escape hatch for anything else.

Views are named (``VIEWS``) and laid out by a nested list, one list per row,
so ``views=[["left_lateral", "right_lateral"]]`` is a two-panel figure. Medial
views cut the brain at the midline: a body or a mesh is sectioned, and fibres
are kept to the shown hemisphere.
"""

from __future__ import annotations

import copy
import io
import os
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["mosaic", "VIEWS", "DEFAULT_LAYOUT", "DEFAULTS", "MODES",
           "VOLUME_RENDERS"]

MODES = ("volume", "mesh", "slice", "fibers")

# Named cameras: (azimuth, elevation, which hemisphere is kept). Azimuth turns
# from anterior towards the LEFT; a medial view keeps one hemisphere and looks
# at its cut face from the side that was removed.
VIEWS = {
    "left_lateral":  (90.0, 0.0, None),
    "right_lateral": (270.0, 0.0, None),
    "left_medial":   (270.0, 0.0, "left"),
    "right_medial":  (90.0, 0.0, "right"),
    "superior":      (0.0, 89.9, None),
    "inferior":      (0.0, -89.9, None),
    "anterior":      (0.0, 0.0, None),
    "posterior":     (180.0, 0.0, None),
}
DEFAULT_LAYOUT = [["left_lateral", "right_lateral", "inferior"],
                  ["left_medial", "right_medial", "superior"]]

# How a body shows its overlay: the map's x-ray depth (its ``xray_mm``).
# "depth" and "xray" see the map through the brain, tinting the surface by
# exp(-depth / mm); "surface" paints only the tissue the rays reach.
VOLUME_RENDERS = {"surface": 0.0, "depth": 5.0, "xray": 25.0}

# Every named argument and its default. Shared ones apply to every mode; the
# per-mode blocks only to their mode. Anything not here is still reachable by
# its viewer name (see the module docstring).
DEFAULTS = {
    "shared": {
        # the overlay -- what the map looks like, identical in every mode
        "palette": "bwr", "symmetric": True, "sign": "both", "absolute": False,
        "threshold": None,          # |value|: both tails of a two-signed map
        "max_value": None, "opacity": 1.0,
        # The colour scale. "robust" spans the 99th percentile of |value| (two
        # signed) or the 1st..99th (one signed): a handful of extreme voxels
        # otherwise stretch the scale until everything real is pastel. A pair
        # is used as given; None is the viewer's own choice (the full range).
        "clim": "robust",
        # With no threshold, opacity rises from 0 at zero to full at this
        # fraction of the scale, so the weak bulk of an unthresholded map lets
        # the brain show instead of washing it in pale colour. 0 turns it off.
        "soft_threshold": 0.25,
        "label": None,              # colour-bar caption; None = the file name
        # the figure
        "views": DEFAULT_LAYOUT, "background": "#ffffff", "tile_px": 900,
        "scale_mm": 95.0,           # half the height every view frames, in mm
        "quality": "export",        # export: finest level + supersampling
        "colorbar": True, "title": None, "gap_px": 24,
    },
    "volume": {
        "body": "spm152",           # packaged name, backdrop name, or a path
        "render": "depth",          # surface | depth | xray (VOLUME_RENDERS)
        "depth_mm": None,           # overrides the preset's x-ray depth
        "tint": None,               # hex colour for the body; None = greyscale
        "body_threshold": None,     # [start, saturate] of the body; None = auto
        "allow_outside": False,     # draw the map outside the body too
    },
    "mesh": {
        "mesh": "pial_wholebrain",  # a registered mesh, a directory, a file
        "coloring": "vertex",       # vertex: gradient | region: one per mesh
        "mesh_color": "#b7bcc4", "mesh_opacity": 1.0,
        "edges": False, "edge_color": "#404040", "edge_opacity": 0.4,
        "smooth_mm": 2.0,           # vertex colouring: surface smoothing
        "stat": "max",              # region colouring: how a region reduces
    },
    "slice": {
        "backdrop": "mni_icbm152_t1_tal_nlin_sym_09c_brain",
        "axis": "z",                # z: axial, y: coronal, x: sagittal
        "positions": None,          # world mm; None = six spread through
    },
    "fibers": {
        "fiber_mesh": "leaddbs_surf", "mesh_color": "#a8a8a8",
        "mesh_opacity": 0.25, "edges": True, "edge_color": "#404040",
        "edge_opacity": 0.4,
        "max_lines": 20000, "tube_radius": 0.2, "top_percent": None,
        "values_only": True,        # draw only fibres the result paints
    },
}

_SLICE_SPREAD = {"z": (-30.0, 66.0), "y": (-90.0, 50.0), "x": (-60.0, 60.0)}
_SLICE_CAMERA = {"z": (0.0, 89.9), "y": (0.0, 0.0), "x": (90.0, 0.0)}


# -- the call ---------------------------------------------------------------
def mosaic(source, out=None, *, mode=None, overlay=None, layer=None,
           scene=None, spec_hook=None, **options):
    """Render ``source`` as a multi-view figure and return it as a PIL image.

    ``source`` is a NIfTI, or a fibre-result descriptor (``.fib.desc.json``),
    which selects fibre mode. ``out``, when given, is where the PNG is written.
    ``mode`` is one of ``MODES``; None picks fibres for a descriptor and the
    volume mode otherwise. Named options and their defaults are in
    ``DEFAULTS``; any other viewer setting can be passed by name.
    """
    from PIL import Image

    from circuit_viewer import spec as S
    from circuit_viewer.session import RenderSession

    source = str(Path(str(source)).expanduser())
    mode = mode or ("fibers" if _is_descriptor(source) else "volume")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    opts, extra = _options(mode, options)

    overlay_fields, layer_fields, scene_fields = _route(
        mode, extra, overlay, layer, scene)

    # One scale for the whole figure, read once from the map's own values. A
    # view only sees part of the data -- a cut hemisphere, the fibres on one
    # side -- and a scale read per view would recolour the same value from
    # panel to panel. The bar is drawn from the very same (cmap, clim).
    pinned, bar = _scale(mode, source, opts, overlay_fields)
    session = RenderSession()
    tiles = []
    rows = _rows(opts, mode)
    for row in rows:
        rendered = []
        for view in row:
            spec, camera = _view_spec(mode, source, view, opts, pinned,
                                      layer_fields, scene_fields)
            if spec_hook is not None:
                spec = spec_hook(spec, view) or spec
            session.set_spec(spec)
            png = session.render(size=(opts["tile_px"], opts["tile_px"]),
                                 quality=opts["quality"], scalar_bar=False,
                                 camera=camera)
            for message in session.warnings:
                print(f"[mosaic] {view}: {message}")
            image = Image.open(io.BytesIO(png)).convert("RGB")
            rendered.append(_crop(image, opts["background"]))
        tiles.append(rendered)

    figure = _compose(tiles, opts, [bar] if opts["colorbar"] and bar else None)
    if out:
        out = Path(str(out)).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.save(out)
    return figure


# -- options ------------------------------------------------------------------
def _options(mode, options):
    """Split named options from viewer names, filling every default."""
    named = dict(DEFAULTS["shared"])
    named.update(DEFAULTS[mode])
    known_elsewhere = {k for m in MODES for k in DEFAULTS[m]} - set(named)
    extra = {}
    for key, value in options.items():
        if key in named:
            named[key] = value
        elif key in known_elsewhere:
            raise TypeError(f"{key!r} belongs to another mode, not {mode!r}")
        else:
            extra[key] = value
    if mode == "volume" and named["render"] not in VOLUME_RENDERS:
        raise ValueError(f"render must be one of {tuple(VOLUME_RENDERS)}")
    if mode == "mesh" and named["coloring"] not in ("vertex", "region"):
        raise ValueError("coloring must be 'vertex' or 'region'")
    return named, extra


def _route(mode, extra, overlay, layer, scene):
    """Send each remaining keyword to the overlay, the layer or the scene by
    its viewer name. Overlay names win a tie; the dicts are explicit."""
    from circuit_viewer import spec as S

    layer_base = {"volume": S.VOLUME_LAYER, "mesh": S.MESH_LAYER,
                  "slice": S.SLICE_LAYER, "fibers": S.FIBER_LAYER}[mode]
    scene_keys = set(S.new_spec([]).keys()) - {"layers", "version"}
    overlay_fields = dict(overlay or {})
    layer_fields = dict(layer or {})
    scene_fields = dict(scene or {})
    for key, value in extra.items():
        if key in S.OVERLAY:
            overlay_fields[key] = value
        elif key in layer_base:
            layer_fields[key] = value
        elif key in scene_keys:
            scene_fields[key] = value
        else:
            raise TypeError(
                f"unknown option {key!r}: not a mosaic option, an overlay "
                f"field, a {mode} layer field or a scene field")
    return overlay_fields, layer_fields, scene_fields


def _rows(opts, mode):
    if mode != "slice":
        rows = [list(r) for r in opts["views"]]
        for row in rows:
            for view in row:
                if view not in VIEWS:
                    raise ValueError(f"unknown view {view!r}; one of {tuple(VIEWS)}")
        return rows
    axis = str(opts["axis"]).lower()
    if axis not in _SLICE_CAMERA:
        raise ValueError("axis must be x, y or z")
    positions = opts["positions"]
    if positions is None:
        positions = np.linspace(*_SLICE_SPREAD[axis], 6).round(1).tolist()
    views = [(axis, float(p)) for p in positions]
    per_row = 3 if len(views) > 4 else len(views)
    return [views[i:i + per_row] for i in range(0, len(views), per_row)]


# -- the scale ----------------------------------------------------------------
def _overlays(source, opts, overlay_fields):
    """The overlay stack for ``source``.

    ``threshold`` here is a MAGNITUDE for a two-tailed map: with
    ``sign="both"``, ``threshold=3`` keeps t <= -3 and t >= 3. The suite's own
    threshold is a band on the value (it would drop every negative), so a
    two-tailed cut is drawn as two overlays, one per tail, sharing one scale.
    ``max_value`` is likewise a magnitude ceiling then.
    """
    from circuit_viewer import spec as S

    base = dict(nifti=source, palette=opts["palette"],
                symmetric=opts["symmetric"], sign=opts["sign"],
                absolute=opts["absolute"], threshold=opts["threshold"],
                max_value=opts["max_value"], clim=opts["clim"],
                opacity=opts["opacity"], bar=False,
                name=opts["label"] or _label(source))
    if "render" in opts:        # a body: how it shows the map
        depth = VOLUME_RENDERS[opts["render"]]
        if depth and opts["depth_mm"] is not None:
            depth = float(opts["depth_mm"])
        base["xray_mm"] = depth
    base.update(overlay_fields)
    two_tailed = (base["sign"] == "both" and not base["absolute"]
                  and (base["threshold"] is not None
                       or base["max_value"] is not None))
    if not two_tailed:
        return [S.new_overlay(**base)]
    low = None if base["threshold"] is None else abs(float(base["threshold"]))
    high = None if base["max_value"] is None else abs(float(base["max_value"]))
    positive = dict(base, sign="positive", threshold=low, max_value=high)
    negative = dict(base, sign="negative",
                    threshold=None if high is None else -high,
                    max_value=None if low is None else -low)
    return [S.new_overlay(**positive), S.new_overlay(**negative)]


def _scale(mode, source, opts, overlay_fields):
    """``(overlays with one pinned scale, (caption, cmap, clim))``."""
    from calvin_utils.plotting_utils.mesh_paint import _select
    from calvin_utils.plotting_utils.overlay_paint import scale_for

    overlays = _overlays(source, opts, overlay_fields)
    parts = []
    for ov in overlays:
        if mode == "fibers":
            from calvin_utils.plotting_utils.fiber_render import streamline_values

            values = _select(streamline_values(source), sign=ov["sign"],
                             absolute=ov["absolute"], threshold=ov["threshold"],
                             max_value=ov.get("max_value"))
            parts.append(values[np.isfinite(values)])
        else:
            from circuit_viewer import volume_render as vr

            parts.append(vr.native_values(ov))
    finite = np.concatenate(parts) if parts else np.zeros(0)
    if not finite.size:
        print(f"[mosaic] nothing in {_label(source)} survives sign / threshold")
        return overlays, None
    first = overlays[0]
    symmetric = first.get("symmetric")
    if symmetric is None:
        symmetric = bool((finite > 0).any() and (finite < 0).any())
    asked = opts["clim"]
    if isinstance(asked, str):
        if asked != "robust":
            raise ValueError('clim must be "robust", None or [low, high]')
        if symmetric:
            top = float(np.percentile(np.abs(finite), 99.0)) or 1.0
            asked = [-top, top]
        else:
            low, high = np.percentile(finite, [1.0, 99.0])
            asked = [float(low), float(high)] if high > low else None
    cmap, clim = scale_for(finite, dict(first, symmetric=symmetric, clim=asked))
    clim = [float(clim[0]), float(clim[1])]
    palette = _soft_palette(first.get("palette"), opts, symmetric)
    for ov in overlays:
        ov["symmetric"], ov["clim"] = symmetric, list(clim)
        if palette:
            # The soft ramp IS the palette's alpha, so it must not be
            # overridden to full strength.
            ov["palette"], ov["override"] = palette, 0.0
    return overlays, (first["name"] or _label(source), cmap, clim)


def _soft_palette(name, opts, symmetric):
    """A copy of ``name`` whose opacity rises from nothing at the scale's
    zero, registered with matplotlib so a spec can name it; None when not
    wanted. Colours -- and so the colour bar -- are the palette's own."""
    soft = float(opts.get("soft_threshold") or 0.0)
    # Only for an unthresholded two-signed map on an automatic scale. A scale
    # someone set means what the viewer means by it: everything from its
    # minimum up -- and below, in the bottom colour -- is painted.
    if (soft <= 0 or opts.get("threshold") is not None or not name
            or not symmetric or not isinstance(opts.get("clim"), (str, type(None)))):
        return None
    import matplotlib
    from matplotlib.colors import ListedColormap

    from calvin_utils.plotting_utils.palettes import resolve_palette

    base = resolve_palette(name)
    u = np.linspace(0.0, 1.0, 512)
    table = np.asarray(base(u), float).copy()
    table[:, 3] = 1.0           # the ramp replaces a LUT's own alpha
    distance = np.abs(u - 0.5) * 2.0 if symmetric else u
    ramp = np.clip(distance / soft, 0.0, 1.0)
    table[:, 3] = table[:, 3] * ramp * ramp * (3.0 - 2.0 * ramp)
    label = f"{name}__soft{int(round(soft * 100))}{'s' if symmetric else 'r'}"
    if label not in matplotlib.colormaps:
        matplotlib.colormaps.register(ListedColormap(table, name=label))
    return label


# -- one view ---------------------------------------------------------------
def _view_spec(mode, source, view, opts, pinned, layer_fields, scene_fields):
    from circuit_viewer import spec as S

    ovs = copy.deepcopy(pinned)

    if mode == "slice":
        axis, at = view
        layers = [S.new_slice_layer(
            backdrop=_backdrop_path(opts["backdrop"]), overlays=ovs,
            **{f"show_{a}": a == axis for a in "xyz"}, **{axis: at})]
        azimuth, elevation = _SLICE_CAMERA[axis]
    else:
        azimuth, elevation, keep = VIEWS[view]
        layers = _layers(mode, ovs, opts, keep)
    layers[-1].update(layer_fields)

    spec = S.new_spec(layers)
    spec["background"] = opts["background"]
    for key, value in scene_fields.items():
        if isinstance(value, dict) and isinstance(spec.get(key), dict):
            spec[key] = dict(spec[key], **value)
        else:
            spec[key] = value
    camera = {"azimuth": azimuth, "elevation": elevation,
              "parallel_scale": float(opts["scale_mm"])}
    return S.normalize_spec(spec), camera


def _layers(mode, ovs, opts, keep):
    from circuit_viewer import spec as S

    cut = _midline_cut(keep)
    if mode == "volume":
        body = S.new_volume_layer(source=opts["body"], color=opts["tint"],
                                  clim=opts["body_threshold"], section=cut,
                                  overlay_outside=bool(opts["allow_outside"]),
                                  overlays=ovs)
        return [body]
    if mode == "mesh":
        for ov in ovs:
            ov.update(mode=opts["coloring"], smooth_mm=opts["smooth_mm"],
                      stat=opts["stat"])
        return [S.new_mesh_layer(
            mesh=opts["mesh"], color=opts["mesh_color"],
            opacity=opts["mesh_opacity"], edges=opts["edges"],
            edge_color=opts["edge_color"], edge_opacity=opts["edge_opacity"],
            section=cut, overlays=ovs)]
    # fibres: a glass brain, then the streamlines coloured by the result
    from calvin_utils.plotting_utils.fiber_render import read_descriptor

    found = read_descriptor(ovs[0]["nifti"])
    for ov in ovs:
        ov.update(mode="vertex", hide_below_min=True)
    shell = S.new_mesh_layer(
        name="Glass", mesh=opts["fiber_mesh"], color=opts["mesh_color"],
        opacity=opts["mesh_opacity"], edges=opts["edges"],
        edge_color=opts["edge_color"], edge_opacity=opts["edge_opacity"],
        section=cut, overlays=[])
    fibers = S.new_fiber_layer(
        name=found["name"], source=str(found["atlas"]), coloring="value",
        max_lines=opts["max_lines"], tube_radius=opts["tube_radius"],
        top_percent=opts["top_percent"], values_only=opts["values_only"],
        prune=_hemisphere_gates(keep), overlays=ovs)
    return [shell, fibers]


def _midline_cut(keep):
    """A medial view keeps one hemisphere: remove the other at x = 0."""
    if keep == "left":
        return [{"axis": "x", "op": ">", "at": 0.0}]
    if keep == "right":
        return [{"axis": "x", "op": "<", "at": 0.0}]
    return []


def _hemisphere_gates(keep):
    """Fibres wholly in the kept hemisphere: never crossing the midline, and
    never entering the other half."""
    if keep is None:
        return []
    other = _half_mask("right" if keep == "left" else "left")
    return [{"axis": "x", "at": 0.0, "mode": "avoid"},
            {"kind": "volume", "nifti": other, "threshold": 0.0,
             "mode": "avoid"}]


def _half_mask(side):
    """A 2 mm mask of one half of MNI space, written once to the temp dir."""
    import nibabel as nib

    folder = Path(tempfile.gettempdir()) / "circuit_viewer_mosaic"
    folder.mkdir(exist_ok=True)
    path = folder / f"half_{side}.nii.gz"
    if not path.is_file():
        shape = (101, 131, 111)                     # x -100..100, 2 mm
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = [-100.0, -150.0, -90.0]
        x = affine[0, 3] + 2.0 * np.arange(shape[0])
        inside = (x < 0) if side == "left" else (x > 0)
        data = np.zeros(shape, np.uint8)
        data[inside] = 1
        temp = folder / f".{side}.{os.getpid()}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(temp))
        os.replace(temp, path)
    return str(path)


# -- composing --------------------------------------------------------------
def _crop(image, background, margin=12):
    """Trim a tile to what was drawn. Every view is framed at the same scale,
    so cropping keeps millimetres the same size in every panel."""
    from matplotlib.colors import to_rgb

    pixels = np.asarray(image).astype(int)
    bg = np.array([round(c * 255) for c in to_rgb(background)])
    drawn = np.abs(pixels - bg).sum(axis=2) > 6
    if not drawn.any():
        return image
    ys, xs = np.nonzero(drawn)
    h, w = drawn.shape
    box = (max(xs.min() - margin, 0), max(ys.min() - margin, 0),
           min(xs.max() + margin + 1, w), min(ys.max() + margin + 1, h))
    return image.crop(box)


def _compose(tiles, opts, bars):
    """Tiles on a grid -- each column as wide as its widest panel, each row as
    tall as its tallest -- with the colour bar at the right."""
    from PIL import Image

    gap = int(opts["gap_px"])
    columns = max(len(r) for r in tiles)
    widths = [max((r[c].width for r in tiles if c < len(r)), default=0)
              for c in range(columns)]
    heights = [max(t.height for t in r) for r in tiles]
    grid_w = sum(widths) + gap * (columns + 1)
    grid_h = sum(heights) + gap * (len(tiles) + 1)

    title_h = 0
    if opts["title"]:
        title_h = int(opts["tile_px"] * 0.08)
    bar = _colorbar(bars[0], int(grid_h * 0.55), opts) if bars else None
    bar_w = bar.width + gap if bar is not None else 0

    figure = Image.new("RGB", (grid_w + bar_w, grid_h + title_h),
                       opts["background"])
    y = gap + title_h
    for row, height in zip(tiles, heights):
        x = gap
        for c, tile in enumerate(row):
            figure.paste(tile, (x + (widths[c] - tile.width) // 2,
                                y + (height - tile.height) // 2))
            x += widths[c] + gap
        y += height + gap
    if bar is not None:
        figure.paste(bar, (grid_w, title_h + (grid_h - bar.height) // 2))
    if opts["title"]:
        _title(figure, opts["title"], title_h, opts)
    return figure


def _colorbar(entry, height, opts):
    """The shared colour bar, drawn by matplotlib from the overlay's own
    resolved scale -- the same (cmap, clim) the renderer painted with."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, to_rgb
    from matplotlib.colorbar import ColorbarBase
    from PIL import Image

    from calvin_utils.plotting_utils.palettes import opaque

    caption, cmap, clim = entry
    cmap = opaque(cmap)
    dpi = 100
    width_px = max(int(height * 0.28), 120)
    ink = "#111111" if np.mean(to_rgb(opts["background"])) > 0.5 else "#eeeeee"
    fig = plt.figure(figsize=(width_px / dpi, height / dpi), dpi=dpi,
                     facecolor=opts["background"])
    ax = fig.add_axes([0.08, 0.06, 0.22, 0.88])
    bar = ColorbarBase(ax, cmap=cmap, norm=Normalize(*clim),
                       orientation="vertical")
    font = max(height / 32.0, 8.0)
    bar.ax.tick_params(labelsize=font, colors=ink, length=font * 0.4)
    bar.outline.set_edgecolor(ink)
    bar.set_label(str(caption), color=ink, fontsize=font * 1.1)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=opts["background"])
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _title(figure, text, height, opts):
    from matplotlib.colors import to_rgb
    from PIL import ImageDraw, ImageFont

    ink = (17, 17, 17) if np.mean(to_rgb(opts["background"])) > 0.5 else (238, 238, 238)
    draw = ImageDraw.Draw(figure)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(height * 0.55))
    except OSError:
        font = ImageFont.load_default()
    w = draw.textlength(str(text), font=font)
    draw.text(((figure.width - w) / 2, height * 0.2), str(text), fill=ink,
              font=font)


# -- small things -------------------------------------------------------------
def _is_descriptor(path):
    if not str(path).lower().endswith(".json"):
        return False
    try:
        from calvin_utils.plotting_utils.fiber_render import is_descriptor
    except ImportError:
        return str(path).lower().endswith(".fib.desc.json")
    return is_descriptor(path)


def _backdrop_path(name):
    """A backdrop name from the viewer's library, or a path, to a path."""
    text = str(name)
    if text.lower().endswith(".npz") or "/" in text:
        return str(Path(text).expanduser())
    from circuit_viewer.paths import BACKDROP_DIR

    found = Path(BACKDROP_DIR) / f"{text}.npz"
    return str(found) if found.is_file() else text


def _label(source):
    name = Path(source).name
    for suffix in (".fib.desc.json", ".nii.gz", ".nii", ".json"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _main(argv=None):
    """``python -m circuit_viewer.mosaic map.nii.gz fig.png key=value ...``

    Values are read as JSON when they parse (numbers, true/false, null, lists)
    and as text otherwise: ``threshold=2.3 render=surface views='[["superior"]]'``.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("options", nargs="*", help="key=value overrides")
    args = parser.parse_args(argv)
    options = {}
    for item in args.options:
        key, _, value = item.partition("=")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    mosaic(args.source, args.out, **options)
    print(args.out)


if __name__ == "__main__":
    _main()

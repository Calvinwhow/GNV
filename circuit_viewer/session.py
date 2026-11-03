"""Turns a spec into pixels, and keeps enough around to do it again quickly.

Orbiting a brain means re-rendering it several times a second, and the only way
that is affordable is to do the expensive work once.  Three caches, in order of
how expensive they are to miss:

``_fibers``    loading a 300 MB Lead-DBS MAT takes ten seconds or so.  Keyed on
               the source and its filters, so dragging the opacity slider on a
               fibre layer never re-reads the file.
``_geometry``  voxelising 165 parcel meshes and reducing a volume inside each
               takes a few seconds.  Keyed on everything that changes the
               NUMBERS -- mesh, volume, mode, statistic, sign, thresholds --
               and deliberately not on anything that only changes the COLOURS.
``_plotter``   one live off-screen render window, actors already added.  A
               camera move touches none of the above and costs a screenshot.

Every call that touches that render window goes through a single worker thread.
A GL context belongs to the thread that created it, and a threaded web server
hands consecutive requests to different threads: without this the second render
to arrive on a new thread fails with ``eglMakeCurrent: EGL_BAD_ACCESS`` and
then takes the process down. A lock is not enough -- the requests have to run on
the *same* thread, not merely one at a time.

The split between geometry keys and style keys is the whole design.  Changing a
palette or an opacity rebuilds actors from cached geometry, which is fast;
changing a threshold recomputes, which is not, and should not be triggered by a
colour picker.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .spec import camera_direction, normalize_spec

__all__ = ["RenderSession", "catalogue", "region_names", "side_of", "ink_for"]


def region_names(mesh, iso=None, split="auto", presmooth_vox=None):
    """The regions a group contains, each with its side and its coarse group.

    ``group`` is None for any atlas whose naming this does not recognise, and
    the panel then offers no supra-region controls rather than one catch-all
    bucket that would hide half the atlas behind a misleading label.
    """
    from calvin_utils.plotting_utils.brains import supra_region

    group = resolve_meshes(mesh, iso=iso, split=split,
                           presmooth_vox=presmooth_vox)
    return [{"name": name, "side": side_of(name, part),
             "group": supra_region(name)}
            for name, part in group.items()]


def region_dir(spec):
    """The directory of region meshes for a registered set, or None.

    Locally built sets ship beside the package. The yabplot-packaged ones live
    in yabplot's download cache, and asking yabplot where it put them is what
    makes aseg, Tian and the Melbourne atlases usable here too rather than being
    listed and then failing to load.
    """
    from calvin_utils.plotting_utils.brains import RESOURCE_DIR

    if "parcels" in spec:
        path = RESOURCE_DIR / spec["parcels"]
        return path if path.is_dir() else None

    atlas = spec.get("yabplot_atlas")
    if not atlas:
        return None
    try:
        from yabplot.data import _find_subcortical_files
    except Exception:
        return None
    try:
        files = _find_subcortical_files(atlas)
    except Exception:
        return None
    paths = [Path(f) for f in (files or [])]
    return paths[0].parent if paths else None

# Fields that change the numbers. Anything not listed here is style, and style
# never invalidates the geometry cache.
# Geometry is now only geometry: what the meshes ARE. Everything about how they
# are coloured lives in the overlay stack and is resolved at draw time, so
# adding a second finding never rebuilds the surfaces.
_GEOMETRY_KEYS = ("mesh", "tessellate", "iso", "split", "presmooth_vox",
                  "show_left", "show_right", "hidden_regions", "section",
                  "section_cap")
_LOAD_KEYS = ("mesh", "iso", "split", "presmooth_vox")
_FIBER_KEYS = ("source", "fiber_atlas_path", "sign", "min_abs_value",
               "max_abs_value", "top_percent", "max_lines", "step", "smooth")
# How many cached builds no open layer uses are kept anyway (most recent first):
# enough to step a slider back without rebuilding, few enough that a long
# session does not hold every value it passed through. A loaded fibre set is the
# largest (streamlines and tubes), so only one spare.
_SPARE_GEOMETRY = 3
_SPARE_GROUPS = 2
_SPARE_FIBERS = 1
_SPARE_INTERIORS = 4

BASE_COLOR = "#b7bcc4"

# Lead-DBS' electrode models, shipped with the viewer so a reconstruction can be
# opened without a MATLAB install to read them out of.
ELECTRODE_MODEL_DIR = Path(__file__).resolve().parent / "resources" / "electrodes"


def ink_for(background):
    """Text colour that reads against a given background."""
    from matplotlib.colors import to_rgb

    try:
        r, g, b = to_rgb(background)
    except ValueError:
        return "#33373d"
    return "#33373d" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "#e6e9ef"


def side_of(name, mesh, tolerance=4.0):
    """Which hemisphere a region belongs to: 'L', 'R' or 'M' for midline.

    The name is tried first because an atlas that bothers to say ``_L`` is more
    reliable than geometry, and a centroid only falls back for meshes that came
    from a volume and have no naming convention at all. Structures straddling
    the midline -- vermis, brainstem -- are their own answer rather than being
    forced to a side, so hiding a hemisphere does not silently take them too.
    """
    if name in ("L", "R"):
        return name
    stem = str(name).rsplit(".", 1)[0]
    for suffix, side in (("_L", "L"), ("_R", "R"), ("-L", "L"), ("-R", "R")):
        if stem.upper().endswith(suffix):
            return side
    lowered = stem.lower()
    if lowered.startswith("left_") or lowered.startswith("lh_"):
        return "L"
    if lowered.startswith("right_") or lowered.startswith("rh_"):
        return "R"
    try:
        centre = float(np.asarray(mesh.center)[0])
    except Exception:
        return "M"
    if centre < -tolerance:
        return "L"
    if centre > tolerance:
        return "R"
    return "M"


def section_mesh(mesh, conditions, cap=False):
    """Cut a mesh in world millimetres. Each condition removes a half-space.

    ``{"axis": "x", "op": ">", "at": -3.0}`` is ``vol(X > -3) = 0``: everything
    right of x = -3 goes. The cut is left OPEN rather than capped, because a cap
    carries no region identity and would paint itself with whatever scalar
    happened to be first -- an open cut shows the real interior and keeps every
    remaining cell's value honest.
    """
    for condition in conditions or ():
        if not condition.get("enabled", True):
            continue
        axis = "xyz".index(condition["axis"])
        normal = [0.0, 0.0, 0.0]
        # `clip` keeps the side the normal points AWAY from, so to delete
        # "greater than" the normal points to increasing coordinate.
        normal[axis] = 1.0 if condition["op"] == ">" else -1.0
        origin = [0.0, 0.0, 0.0]
        origin[axis] = float(condition["at"])
        try:
            if cap:
                # A capped cut is a cross-section: the flat face is real
                # geometry, and in vertex mode it samples the volume like any
                # other point, so scrolling a plane through a whole-brain mesh
                # shows the anatomy on the cut rather than the inside of the
                # far shell. It is wrong for region colouring -- a cap belongs
                # to no region and would take whichever scalar came first --
                # which is why this is a choice and not the default.
                mesh = mesh.clip_closed_surface(
                    normal=[-n for n in normal], origin=origin,
                    inplace=False).triangulate()
                continue
            cut = mesh.clip(normal=normal, origin=origin, invert=True)
            # A clip returns an unstructured grid of mixed cell types; every
            # filter downstream (decimation especially) wants triangles.
            mesh = cut.extract_surface().triangulate()
        except Exception:
            return None
        if mesh is None or mesh.n_cells == 0:
            return None
    return mesh


def _visible_regions(group, layer):
    """Drop hidden regions and hidden hemispheres before anything is merged."""
    hidden = set(layer.get("hidden_regions") or ())
    show = {"L": layer.get("show_left", True), "R": layer.get("show_right", True),
            "M": True}
    return {name: mesh for name, mesh in group.items()
            if name not in hidden and show[side_of(name, mesh)]}


def _section_group(group, layer):
    """Cut every member, dropping any that the cuts remove entirely."""
    if not any(c.get("enabled", True) for c in (layer.get("section") or ())):
        return group
    kept = {}
    cap = bool(layer.get("section_cap"))
    for name, mesh in group.items():
        cut = section_mesh(mesh, layer.get("section"), cap=cap)
        if cut is not None and cut.n_cells:
            kept[name] = cut
    return kept


def _nifti_backdrop(path):
    """A NIfTI as a slice backdrop, in memory, without converting it.

    The converter's ``.npz`` stores each slice as 8-bit for fast reads; this
    does the same in memory: the volume (first frame of a 4D) is scaled to
    0..255 between the 0.5th and 99.5th percentiles of its nonzero voxels, so
    "transparent below" and the grey ramp mean what they mean on a prepared
    backdrop. Its own affine is kept -- slices are cut and placed by it, oblique
    or not -- and nothing touches the disk.
    """
    from calvin_utils.plotting_utils.backdrops import Backdrop
    from calvin_utils.plotting_utils.mesh_paint import _volume

    data, affine = _volume(str(path))
    data = np.asarray(data, np.float32)
    nonzero = data[np.isfinite(data) & (data != 0)]
    lo, hi = (np.percentile(nonzero, [0.5, 99.5]) if nonzero.size else (0.0, 1.0))
    if not hi > lo:
        lo, hi = float(np.min(data)), float(np.max(data)) or 1.0
    scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0) * 254.0 + 1.0
    scaled[data == 0] = 0.0              # empty stays empty (and transparent)
    volume = np.rint(scaled).astype(np.uint8)

    class _InMemory(Backdrop):
        def __init__(self):             # noqa: D401 -- no store to open
            self.path = Path(path)
            self._store = None
            self.shape = tuple(int(n) for n in volume.shape)
            self.affine = np.asarray(affine, float)
            zooms = np.linalg.norm(self.affine[:3, :3], axis=0)
            self.mm = float(zooms.min())
            self.meta = {"shape": list(self.shape), "affine": self.affine.tolist(),
                         "mm": self.mm, "zooms": zooms.tolist(), "temporary": True}

        def slice(self, axis, index):
            index = int(np.clip(index, 0, self.shape[axis] - 1))
            return np.take(volume, index, axis=axis)

    return _InMemory()


def _spec_key(overlay):
    """An overlay's identity for its colour bar: its content, minus its id."""
    if overlay.get("_spec_key"):
        return overlay["_spec_key"]
    return json.dumps({k: v for k, v in overlay.items() if k != "id"},
                      sort_keys=True, default=str)


def _framed(overlay):
    """A 4D overlay pointed at its current frame (see ``frames``), and a map
    with ``extend_mm`` pointed at its grown copy (see ``extend``); anything
    else unchanged. Done here, where every painter collects its overlays, so
    meshes, slices, bodies and fibres all play a series the same way."""
    from circuit_viewer import extend, frames

    key = _spec_key(overlay)
    try:
        overlay = frames.resolve(overlay, key)
    except Exception:
        return overlay          # an unreadable file is reported by its painter
    # Then grow the (frame of the) map, if asked: see ``extend``.
    try:
        return extend.resolve(overlay, key)
    except Exception as exc:  # draw the map unextended, but say so
        import sys

        print(f"circuit_viewer: could not extend {overlay.get('nifti')}: {exc!r}",
              file=sys.stderr)
        return overlay


def _fills(overlays):
    """The overlays that paint a surface; outline overlays only trace one."""
    return [_framed(o) for o in (overlays or [])
            if o.get("draw", "fill") != "outline"]


def _outlines(overlays):
    """The overlays traced as an outline: ``outline`` on, or the older
    outline-only ``draw`` mode (which also turns the fill off)."""
    return [_framed(o) for o in (overlays or [])
            if (o.get("outline") or o.get("draw") == "outline") and o.get("nifti")]


def _outline_style(overlay):
    """``(colour, opacity, width px)`` of an overlay's outline."""
    return (overlay.get("outline_color") or overlay.get("color") or "#000000",
            float(overlay.get("outline_opacity", 1.0)),
            float(overlay.get("outline_width", 2.0) or 2.0))


def _coloured(values, overlay):
    """Where an overlay's values put colour, for its outline, as 0..1 with the
    edge at 0.5: what survived its threshold and, on a sequential scale, what
    is at or above the colour bar's minimum -- so the outline sits at the
    bar's lower limit. Near that limit the field follows the value itself, so
    the edge falls between vertices where the value actually crosses it
    rather than on whichever vertex came first. A diverging scale has no such
    limit (its low end is the strongest negative), so there it is everything
    that survived."""
    from calvin_utils.plotting_utils.overlay_paint import scale_for

    values = np.asarray(values, float)
    present = np.isfinite(values)
    field = np.zeros(values.shape, np.float32)
    if not present.any():
        return field
    _cmap, clim = scale_for(values[present], overlay)
    low, high = min(clim), max(clim)
    if low < 0.0 < high:
        field[present] = 1.0
        return field
    band = max((high - low) * 0.05, 1e-9)
    field[present] = np.clip(0.5 + (values[present] - low) / (2.0 * band), 0.0, 1.0)
    return field


# How far along the surface a mesh outline's coloured/uncoloured field is
# averaged before it is traced: enough to round off the triangles (~1 mm
# edges on a cortical mesh), far less than a sulcus.
_OUTLINE_SMOOTH_MM = 1.5


def _surface_average(mesh, values, radius_mm):
    """Plain averaging of a vertex field over its 1-ring, repeated until about
    ``radius_mm`` of surface is covered -- along the mesh, never across a
    sulcus. (``_vertex_smooth`` treats zero as missing data; this does not.)"""
    values = np.asarray(values, float)
    if radius_mm <= 0 or len(values) != mesh.n_points or mesh.n_cells == 0:
        return values
    pairs, length = _surface_edges(mesh)
    if not len(pairs):
        return values
    passes = max(1, int(np.ceil(float(radius_mm) / max(length, 1e-6))))
    a, b = pairs[:, 0], pairs[:, 1]
    n = len(values)
    degree = np.bincount(a, minlength=n) + np.bincount(b, minlength=n) + 1.0
    out = values.copy()
    for _ in range(min(passes, 20)):
        out = (out + np.bincount(a, weights=out[b], minlength=n)
               + np.bincount(b, weights=out[a], minlength=n)) / degree
    return out


def _line_actor(poly):
    """A bare line actor for ``poly`` (which may be empty, unlike add_mesh)."""
    import vtk

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLighting(False)
    return actor


def _auto(value, default):
    """A layer field of None means 'decide for me'."""
    return default if value is None else value


def _edges_on(layer, geom, fills):
    """Whether a mesh draws its plain edge highlight: asked for, or by default
    on a region set that is not painted as a gradient."""
    gradient = any(o.get("mode") == "vertex" for o in fills)
    return bool(_auto(layer["edges"], geom["region_set"] and not gradient))


def _key(layer, fields):
    return hashlib.sha1(
        json.dumps([layer.get(f) for f in fields], default=str).encode()
    ).hexdigest()


def catalogue():
    """What the UI offers in its dropdowns, straight from the mesh registry."""
    from calvin_utils.plotting_utils.brains import MESHES

    # Some registered region sets are packaged inside yabplot rather than shipped
    # as meshes on disk. yabplot can draw them; this renderer cannot, because it
    # needs actual geometry. They stay in the list, marked unavailable, so the
    # reason is visible in the UI instead of arriving as a load error later.
    meshes = []
    for name, spec in MESHES.items():
        # The registry's own family is what decides the group -- a yabplot
        # subcortical set is a region set even though it ships no directory, and
        # classifying it by whether it has one would file it under surfaces and
        # let it be picked as a cortex.
        region_set = spec.get("family") == "subcortex"
        available = (not region_set) or region_dir(spec) is not None
        meshes.append({
            "name": name,
            "what": spec.get("what", ""),
            "group": "parcels" if region_set else "surface",
            "available": bool(available),
        })
    return {"meshes": meshes,
            "stats": ["max", "mean_nonzero", "mean", "min", "sum"],
            "signs": ["both", "positive", "negative"],
            "modes": ["region", "vertex"],
            "colorings": ["solid", "direction", "value"]}


# A prepared backdrop counts as a volume everywhere a volume is accepted, so
# one can be meshed and can colour itself without a separate code path.
VOLUME_SUFFIXES = (".nii", ".nii.gz", ".npz")


def is_volume(name):
    return str(name).lower().endswith(VOLUME_SUFFIXES)


def resolve_meshes(name, iso=None, split="auto", presmooth_vox=None):
    """Any mesh reference -> an ordered ``{label: PolyData}`` group.

    A single surface, a whole-brain composite and a directory of 165 parcels all
    come back in the same shape, which is what lets one layer type cover all
    three. Hemisphere labels are preserved when they exist, because culling the
    near hemisphere in a lateral view depends on knowing which half is which.
    """
    import pyvista as pv

    from calvin_utils.plotting_utils.brains import MESHES
    from calvin_utils.plotting_utils.mesh_paint import load_parcels

    # Several volumes dropped together are ONE group, not several layers: a set
    # of ROI files is the same thing as a directory of parcel meshes, and the
    # whole point of a group is that its members share a palette and a scale.
    if isinstance(name, (list, tuple)):
        group, seen = {}, set()
        for entry in name:
            part = resolve_meshes(entry, iso=iso, split=split,
                                  presmooth_vox=presmooth_vox)
            stem = Path(str(entry)).name.split(".nii")[0]
            for key, mesh in part.items():
                # One region from a file is named by the file; a file that
                # split into several (labels, 4D frames) keeps its own names.
                label = key if len(name) == 1 or len(part) > 1 else stem
                while label in seen:
                    label = f"{label}_"
                seen.add(label)
                group[label] = mesh
        if not group:
            raise ValueError("none of those volumes produced a surface")
        return group

    text = str(name)
    if is_volume(text):
        # A volume becomes geometry through the same pipeline that built the
        # SUIT and AAL meshes, so a lesion dropped in today sits beside them and
        # looks like it belongs rather than like an import.
        from calvin_utils.plotting_utils.volume_meshes import meshes_from_volume

        meshes, _info = meshes_from_volume(
            Path(text).expanduser(), iso=iso, split=split,
            presmooth_vox=presmooth_vox,
        )
        return meshes

    spec = MESHES.get(text)
    if spec is not None and spec.get("family") == "subcortex":
        directory = region_dir(spec)
        if directory is None:
            raise FileNotFoundError(
                f"{text!r} has no region meshes on this machine. Locally built "
                "sets live in neuro_plotter_resources; yabplot's own sets are "
                "downloaded into its cache the first time yabplot draws them."
            )
        return load_parcels(directory)
    if spec is not None:
        from calvin_utils.plotting_utils.brains import mesh_pieces
        from calvin_utils.plotting_utils.glass_mesh import build_bmesh

        return build_bmesh(mesh_pieces(text))

    path = Path(text).expanduser()
    if path.is_dir():
        # A folder of meshes is a parcel set; a folder of volumes is the same
        # thing before surfacing, one region per file (4D files expand to one
        # per frame). Meshes win if the folder has both.
        if any(path.glob("*.vtk")):
            return load_parcels(path)
        volumes = sorted(p for p in path.iterdir()
                         if not p.name.startswith(".") and is_volume(p.name))
        if volumes:
            return resolve_meshes([str(p) for p in volumes], iso=iso, split=split,
                                  presmooth_vox=presmooth_vox)
        return load_parcels(path)
    if path.is_file():
        return {path.stem: pv.read(path)}
    raise FileNotFoundError(f"{text!r} is not a registered mesh, a directory of "
                            f"meshes, or a file")


def _thickness_mm(layer):
    """Fibre thickness in millimetres, for either quality.

    ``tube_radius`` is the control. ``width`` is the old pixel line width and is
    only consulted for specs saved before thickness became a physical size, so
    those keep the look they were saved with.
    """
    from calvin_utils.plotting_utils.fiber_render import (
        TUBE_RADIUS_RANGE, radius_for_width)

    if layer.get("tube_radius"):
        radius = float(layer["tube_radius"])
    elif layer.get("width"):
        radius = radius_for_width(layer["width"])
    else:
        radius = 0.2
    lo, hi = TUBE_RADIUS_RANGE
    return min(max(radius, lo), hi)


def _signed_samples(raw, overlay):
    """The suite's sign rule and |v| applied to samples; excluded become 0."""
    values = np.asarray(raw, float).copy()
    values[~np.isfinite(values)] = 0.0
    if overlay["sign"] == "positive":
        values[values < 0] = 0.0
    elif overlay["sign"] == "negative":
        values[values > 0] = 0.0
    if overlay["absolute"]:
        values = np.abs(values)
    return values


_SURFACE_EDGES = {}     # id(mesh) -> (weakref, edges, mean edge length)


def _surface_edges(mesh):
    """Every undirected edge of a surface once, and its mean length; cached
    per mesh object, which geometry caching keeps alive while it is shown."""
    import weakref

    key = (id(mesh), mesh.n_points, mesh.n_cells)
    hit = _SURFACE_EDGES.get(key)
    if hit is not None and hit[0]() is mesh:
        return hit[1], hit[2]
    surface = mesh if mesh.is_all_triangles else mesh.triangulate()
    faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    pairs = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    pairs.sort(axis=1)
    pairs = np.unique(pairs, axis=0)
    points = np.asarray(mesh.points)
    length = float(np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]],
                                  axis=1).mean()) if len(pairs) else 1.0
    try:
        _SURFACE_EDGES[key] = (weakref.ref(mesh), pairs, length)
    except TypeError:
        pass
    while len(_SURFACE_EDGES) > 64:
        _SURFACE_EDGES.pop(next(iter(_SURFACE_EDGES)))
    return pairs, length


def _vertex_smooth(mesh, values, radius_mm, support=0.5):
    """Average vertex samples over the surface around each vertex.

    A vertex sample is one point in a 2 mm voxel grid. On a folded cortex the
    crowns and fundi land just outside a masked map while the banks between
    them land inside, so a figure stripes: ridges of colour with ridges of
    nothing between them. Each pass replaces every vertex with the mean of the
    values in its 1-ring -- itself and its ~6 neighbours -- that HAVE data,
    and gives an empty vertex a value only when at least ``support`` of its
    ring has data, which closes a stripe a vertex or two wide without growing a
    blob outward past its edge. Passes repeat until about ``radius_mm`` of
    surface is covered, so the reach is the same on a dense mesh and a
    decimated one. Averaging runs ALONG the mesh: the far bank of a sulcus is
    never mixed in. Zero is "no data", as everywhere in the suite.
    """
    values = np.asarray(values, float)
    if radius_mm <= 0 or len(values) != mesh.n_points or mesh.n_cells == 0:
        return values
    pairs, length = _surface_edges(mesh)
    if not len(pairs):
        return values
    passes = max(1, int(np.ceil(float(radius_mm) / max(length, 1e-6))))
    a, b = pairs[:, 0], pairs[:, 1]
    n = len(values)
    degree = np.bincount(a, minlength=n) + np.bincount(b, minlength=n) + 1.0
    out = np.where(np.isfinite(values), values, 0.0)
    for _ in range(min(passes, 50)):
        has = (out != 0).astype(float)
        total = out + np.bincount(a, weights=out[b], minlength=n) \
            + np.bincount(b, weights=out[a], minlength=n)
        count = has + np.bincount(a, weights=has[b], minlength=n) \
            + np.bincount(b, weights=has[a], minlength=n)
        keep = (has > 0) | (count >= support * degree)
        out = np.where(keep & (count > 0), total / np.maximum(count, 1.0), 0.0)
    return out


class RenderSession:
    def __init__(self, spec=None):
        self.spec = normalize_spec(spec or {})
        # Built geometry, by everything that shapes it. OrderedDicts in the
        # order last used: _prune_caches keeps whatever an open layer needs
        # plus a few recent others, so dragging a slider back is instant but a
        # long session does not keep every value it ever passed through.
        self._geometry = OrderedDict()
        self._groups = OrderedDict()   # loaded mesh groups, before any cutting
        self._fibers = OrderedDict()
        self._electrodes = OrderedDict()   # placed lead geometry, by source, model and poses
        self._layer_warnings = {}   # layer id -> complaints, while it lives
        self._drawing = None        # the layer being built, for warn()
        self._interiors = OrderedDict()   # mesh group + step -> interior points
        self._backdrops = {}      # path -> open Backdrop (slices read on demand)
        self._volume_grids = {}   # (source, level) -> body grid
        self._framing = {}        # volume actor address -> its tissue bounds
        self._lines = {}          # outline actor address -> (actor, width)
        self._projections = {}    # volume layer id -> [(surface, colour, opacity, width)]
        self._projector = None    # off-screen renderer for those, made on first use
        self._quality = "preview" # the render being built, for body levels
        self._font_scale = 1
        # How many times larger than it will be SEEN this frame is drawn: hi-res
        # renders 1.6x the pane and an export renders the pane at ~4000 px, and
        # each is then shown or placed at the pane's size. Line widths are
        # multiplied by it, so an edge or outline keeps the thickness it has on
        # screen instead of thinning to a fraction of a pixel -- which blends it
        # into the background and drains its colour.
        self._line_scale = 1.0
        self.view = None          # the last frame's framing, for panning
        self._plotter = None
        self._plotter_size = None
        self._actors = {}         # layer id -> the actors drawn for it
        self._layer_keys = {}     # layer id -> what it looked like when drawn
        self._cull_map = {}       # layer id -> [(side, actor)] for hemisphere culling
        # Resolved (label, cmap, clim) per overlay that asked for a bar, keyed
        # on the overlay's own content so a stale scale can never be captioned
        # onto an edited map. Filled while painting, read while rendering --
        # actors are cached, so the bars cannot be registered from the draw.
        self._scales = {}
        self._light_key = None
        self._slice_state = {}    # layer id -> {axis: [poly, actor, world_mm]}
        self._gl = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")

    # -- spec ---------------------------------------------------------------
    def set_spec(self, spec):
        self.spec = normalize_spec(spec)
        return self.spec

    # -- geometry -----------------------------------------------------------
    def _mesh_geometry(self, layer):
        cache_key = _key(layer, _GEOMETRY_KEYS)
        if cache_key in self._geometry:
            self._geometry.move_to_end(cache_key)
            return self._geometry[cache_key]

        # Loading is cached apart from cutting. Meshing a whole-brain volume is
        # about five seconds and sectioning it is sixty milliseconds, so keying
        # them together made every drag of the cut slider re-mesh the volume --
        # a scroll that should be interactive took five seconds a step. These
        # are the fields that decide what gets LOADED; everything else below
        # only decides what is shown of it.
        load_key = _key(layer, _LOAD_KEYS)
        group = self._groups.get(load_key)
        if group is not None:
            self._groups.move_to_end(load_key)
        if group is None:
            group = resolve_meshes(layer["mesh"], iso=layer.get("iso"),
                                   split=layer.get("split", "auto"),
                                   presmooth_vox=layer.get("presmooth_vox"))
            self._groups[load_key] = group
        # Shallow copies, so painting writes to a mesh this render owns. Handing
        # out the cached originals let a "paint" array from one render survive
        # into the next: switch an overlay from region to vertex and the stale
        # per-cell array is still attached where a per-point one is expected,
        # which surfaces as "Array 'paint' has length (28441) but a length of
        # (14120) was expected" -- cells against points on the same mesh.
        group = {name: mesh.copy(deep=False) for name, mesh in group.items()}
        # The uncut group is kept as well: a region's statistic is a property
        # of the region, not of how much of it is left after a cut. Sectioning a
        # parcel open to look inside must not change the number it is showing.
        full = dict(group)
        group = _visible_regions(group, layer)
        if not group:
            return {"parts": {}, "region_set": False, "group": {},
                    "full_group": full, "spans": None}
        hemis = set(group) == {"L", "R"}
        # A region set is many small meshes; a surface is one or a hemisphere
        # pair. The distinction decides the whole look, so it is drawn from the
        # geometry rather than from the layer's name.
        region_set = not hemis and len(group) > 1
        reduction = _auto(layer["tessellate"], 0.0)
        if reduction:
            from calvin_utils.plotting_utils.mesh_paint import coarsen

            group = coarsen(group, reduction)
        # Cut last: decimating a cut face rounds off the very edge the section
        # was made to show.
        group = _section_group(group, layer)
        if not group:
            return {"parts": {}, "region_set": False, "group": {},
                    "full_group": full, "spans": None}

        if region_set:
            # One mesh, with each region's cells tagged, so an overlay can flood
            # a region by writing to its cells and several overlays can be
            # composited without merging anything again.
            import pyvista as pv

            names = list(group)
            meshes = [group[n] for n in names]
            merged = pv.merge(meshes) if len(meshes) > 1 else meshes[0].copy()
            spans, at = {}, 0
            for name, mesh in zip(names, meshes):
                spans[name] = (at, at + mesh.n_cells)
                at += mesh.n_cells
            # Atlas colours ride on the loaded meshes; read them off the uncut,
            # undecimated group, since decimation does not keep field data.
            rgb = {}
            for name in names:
                source = full.get(name)
                held = source.field_data.get("atlas_rgb") if source is not None else None
                if held is not None and len(held) >= 3:
                    rgb[name] = [float(v) for v in np.asarray(held).ravel()[:3]]
            result = {"parts": {"_": merged}, "region_set": True,
                      "group": group, "full_group": full, "spans": spans,
                      "order": names, "region_rgb": rgb}
        else:
            parts = group if hemis else {"_": _merge(list(group.values()))}
            result = {"parts": parts, "region_set": False, "group": group,
                      "full_group": full, "spans": None}

        self._geometry[cache_key] = result
        return result

    def _region_base(self, layer, geom):
        """Per-cell base colours for a region set, or None for one flat colour."""
        mode = layer.get("region_colors")
        if not mode or not geom.get("region_set") or not geom.get("spans"):
            return None
        from matplotlib import colormaps

        palette = (list(colormaps["tab20"].colors) + list(colormaps["tab20b"].colors)
                   + list(colormaps["tab20c"].colors))
        atlas = geom.get("region_rgb") or {}
        mesh = geom["parts"]["_"]
        base = np.empty((mesh.n_cells, 3), float)
        for i, name in enumerate(geom["order"]):
            begin, end = geom["spans"][name]
            colour = atlas.get(name) if mode == "atlas" else None
            base[begin:end] = colour if colour is not None else palette[i % len(palette)][:3]
        return base

    def _mesh_colours(self, layer, geom):
        """Resolve the overlay stack to explicit RGB.

        ONE path, whatever the stack's length. Each overlay is resolved from its
        own values alone -- its own scale, its own threshold, its own opacity --
        and composited over the layer's base coat in order, source-over. No
        overlay can reach another one's scale, and removing one leaves the rest
        pixel-for-pixel as they were.

        A single overlay used to short-circuit to a scalar field plus a colormap
        because that bought a colour bar for free. It also meant a stack changed
        MACHINERY when it changed length: delete the second overlay and the
        first was suddenly drawn with a clim taken from a different array, no
        base coat under it, and opacity meaning actor transparency rather than
        its own contribution. The bar is now built from ``scale_for`` instead.
        """
        from calvin_utils.plotting_utils.overlay_paint import (
            composite, overlay_colours, scale_for,
        )
        from matplotlib.colors import to_rgb

        overlays = _fills(layer.get("overlays"))
        region_base = self._region_base(layer, geom)
        if not overlays:
            if region_base is None:
                return None
            return {"mode": "rgb", "values": (region_base * 255).astype(np.uint8),
                    "target": self._target(geom)}
        stacks = [self._overlay_values(layer, geom, o) for o in overlays]

        base = (region_base if region_base is not None and len(region_base) == len(stacks[0])
                else np.tile(np.asarray(to_rgb(layer["color"]), float),
                             (len(stacks[0]), 1)))
        painted = composite(base, [overlay_colours(v, o)
                                   for v, o in zip(stacks, overlays)])
        for values, overlay in zip(stacks, overlays):
            self._record_scale(overlay, values)
        out = {"mode": "rgb",
               "values": (painted * 255).astype(np.uint8),
               "target": self._target(geom)}
        if layer.get("edge_paint"):
            out["edges"] = self._edge_paint(layer, stacks, overlays,
                                            _edges_on(layer, geom, overlays))
        return out

    @staticmethod
    def _edge_paint(layer, stacks, overlays, unmapped=True):
        """RGBA for a map-painted wireframe.

        The overlays are composited over the EDGE colour exactly as they are
        over the surface, then the whole wireframe is scaled by the edge
        opacity -- the slider means the same with "by map" on as off. Edges no
        map colours are drawn in the edge colour when ``unmapped`` (the edge
        highlight's own checkbox) and left out otherwise, so turning that box
        off keeps only the map's edges."""
        from calvin_utils.plotting_utils.overlay_paint import overlay_colours
        from matplotlib.colors import to_rgb

        size = len(stacks[0])
        # Straight-alpha "over", carried premultiplied: a map's colour is only
        # diluted by the edge colour as far as that edge is itself visible, so
        # with the edge opacity at 0 the map's edges keep their true colour.
        alpha = np.full(size, 1.0 if unmapped else 0.0)
        premul = np.tile(np.asarray(to_rgb(layer["edge_color"]), float),
                         (size, 1)) * alpha[:, None]
        for values, overlay in zip(stacks, overlays):
            rgb, a = overlay_colours(values, overlay)
            a = np.asarray(a, float)
            premul = np.asarray(rgb, float) * a[:, None] + premul * (1.0 - a)[:, None]
            alpha = a + alpha * (1.0 - a)
        rgb = np.divide(premul, alpha[:, None], out=np.zeros_like(premul),
                        where=alpha[:, None] > 1e-6)
        alpha = alpha * float(layer["edge_opacity"])
        rgba = np.concatenate([rgb, alpha[:, None]], axis=1)
        return (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)

    @staticmethod
    def _overlay_key(overlay):
        return _spec_key(overlay)

    def _record_scale(self, overlay, values):
        """Remember the scale an overlay resolved to, for its colour bar.

        The bar has to be the SAME (cmap, clim) that produced the pixels, or a
        figure gets captioned with numbers it does not show. It is keyed on the
        overlay's own content, so editing a threshold or a palette invalidates
        the caption instead of leaving the old one under a new picture.
        """
        if not overlay.get("bar"):
            return
        from calvin_utils.plotting_utils.overlay_paint import scale_for

        values = np.asarray(values, float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            return
        # A bar's title sits ABOVE it, so a long one collides with its
        # neighbour: keep it short, and drop the extension, which never tells
        # you which map this is.
        label = overlay.get("name") or Path(str(overlay.get("nifti") or "")).name
        for suffix in (".nii.gz", ".nii", ".npz", ".npy"):
            if label.lower().endswith(suffix):
                label = label[: -len(suffix)]
                break
        self._scales[self._overlay_key(overlay)] = (
            label, *scale_for(finite, overlay))

    def _bars(self):
        """(label, cmap, clim) for every overlay asking for a bar, in order.

        Walked off the SPEC rather than collected while drawing, because actors
        are cached: a layer that did not need redrawing would otherwise lose its
        bar. Anything whose scale has not been resolved yet is skipped rather
        than guessed at.
        """
        out = []
        for layer in self.spec["layers"]:
            if not layer.get("visible", True):
                continue
            if layer["kind"] == "fibers" and layer.get("coloring", "solid") != "value":
                continue            # solid / direction paint no value ramp
            for overlay in (layer.get("overlays") or []):
                if not (overlay.get("bar") and overlay.get("nifti")) or overlay.get("draw") == "outline":
                    continue
                found = self._scales.get(self._overlay_key(overlay))
                if found is not None:
                    out.append(found)
        return out

    @staticmethod
    def _target(geom):
        """Region sets are coloured per CELL; a surface is coloured per point."""
        return "cells" if geom["region_set"] else "points"

    def _overlay_values(self, layer, geom, overlay):
        from calvin_utils.plotting_utils.mesh_paint import (
            parcel_interiors, parcel_stats, sample_nifti, select_regions, _select,
        )

        mesh = geom["parts"]["_"] if geom["region_set"] else None
        if not geom["region_set"]:
            # Sampled and filled PER PART, then concatenated. Filling the joined
            # array would let a value cross from one mesh to another wherever
            # their vertices happen to be adjacent in the array, which is not a
            # relationship on any surface.
            from calvin_utils.plotting_utils.mesh_paint import fill_gaps

            passes = int(overlay.get("fill_gaps") or 0)
            smooth_mm = float(overlay.get("smooth_mm") or 0.0)
            out = []
            for part in geom["parts"].values():
                raw = sample_nifti(part.points, overlay["nifti"])
                if smooth_mm > 0:
                    # Sign and |v| first, so the average never mixes in values
                    # the overlay excludes; the threshold after, so it judges
                    # the smoothed surface rather than single point samples.
                    signed = _signed_samples(raw, overlay)
                    values = _select(_vertex_smooth(part, signed, smooth_mm),
                                     sign="both", absolute=False,
                                     threshold=overlay["threshold"],
                                     max_value=overlay.get("max_value"))
                else:
                    values = _select(raw, sign=overlay["sign"],
                                     absolute=overlay["absolute"],
                                     threshold=overlay["threshold"],
                                     max_value=overlay.get("max_value"))
                if passes:
                    values = fill_gaps(part, values, passes=passes)
                out.append(values)
            return np.concatenate(out) if len(out) > 1 else out[0]

        if overlay["mode"] == "vertex":
            # On a region set there are no vertices to speak of -- the regions
            # are the unit -- so a "vertex" overlay samples at cell centres,
            # which is the finest thing this geometry can actually show.
            return _select(sample_nifti(mesh.cell_centers().points,
                                        overlay["nifti"]),
                           sign=overlay["sign"], absolute=overlay["absolute"],
                           threshold=overlay["threshold"],
                           max_value=overlay.get("max_value"))

        step = float(overlay.get("step") or 1.0)
        # Keyed on the UNCUT group, so sectioning never re-voxelises anything --
        # cutting is then instant, which is the point of a slider.
        interior_key = (str(layer["mesh"]), step, _key(layer, ("tessellate",)),
                        _key(layer, ("show_left", "show_right", "hidden_regions")))
        full = geom["full_group"]
        if interior_key not in self._interiors:
            self._interiors[interior_key] = parcel_interiors(full, step=step)
            while len(self._interiors) > _SPARE_INTERIORS:
                self._interiors.popitem(last=False)
        self._interiors.move_to_end(interior_key)

        values = parcel_stats(
            full, overlay["nifti"], stat=overlay["stat"], sign=overlay["sign"],
            absolute=overlay["absolute"], threshold=overlay["threshold"],
            max_value=overlay["max_value"],
            step=step, interiors=self._interiors[interior_key],
        )
        values = select_regions(values, min_value=overlay["min_value"],
                                keep_top=overlay["keep_top"])

        # Broadcast each region's value onto the cells that actually survived.
        out = np.full(mesh.n_cells, np.nan)
        for name, (begin, end) in geom["spans"].items():
            out[begin:end] = values.get(name, np.nan)
        return out

    # -- drawing ------------------------------------------------------------
    def _ensure_plotter(self, size):
        import pyvista as pv

        if self._plotter is not None and self._plotter_size == size:
            return self._plotter
        if self._plotter is not None:
            self._plotter.close()
        self._plotter = pv.Plotter(off_screen=True, window_size=list(size),
                                   border=False)
        self._plotter_size = size
        self._actors.clear()
        self._layer_keys.clear()
        self._cull_map.clear()
        self._slice_state.clear()
        self._light_key = None
        return self._plotter

    def _sync_actors(self, pl):
        """Bring the scene in line with the spec, touching only what changed.

        Rebuilding every actor on every edit is what made a slider feel slow: a
        slice layer's own work is about 3 ms, while tearing down and re-adding a
        165-region mesh alongside it costs a hundred. Each layer is keyed on how
        it was drawn, so moving a slice plane leaves the meshes alone.
        """
        from calvin_utils.plotting_utils.render_scene import (
            add_lights, surface_material,
        )

        light = self.spec["lighting"]
        light_key = json.dumps([light, self.spec["center"]], sort_keys=True,
                               default=str)
        if light_key != self._light_key:
            add_lights(pl, key_azimuth=light["key_azimuth"],
                       key_elevation=light["key_elevation"],
                       key_intensity=light["key_intensity"],
                       rig=light.get("rig", "studio"),
                       center=tuple(self.spec["center"]))
            self._light_key = light_key

        pl.set_background(self.spec["background"])
        material = surface_material(
            specular=light["specular"], specular_power=light["specular_power"],
            ambient=light["ambient"], diffuse=light["diffuse"],
        )

        wanted = {}
        for layer in self.spec["layers"]:
            if not layer.get("visible", True):
                continue
            # A slice PLANE POSITION deliberately stays out of the key: moving
            # one is an in-place update to an existing actor, and rebuilding the
            # textured quads instead costs ~60 ms a frame for no difference.
            structural = {k: v for k, v in layer.items()
                          if not (layer["kind"] == "slices" and k in "xyz")}
            # The material is in the key because it comes from the lighting.
            # Quality deliberately is NOT: it changes pixels, never geometry --
            # except for a body too large to draw whole, whose LEVEL follows
            # it. That level is keyed, not the quality, so a body small enough
            # to draw whole is never rebuilt by toggling hi-res.
            level = self._volume_level(layer)
            wanted[layer["id"]] = (
                json.dumps([structural, material, level], sort_keys=True,
                           default=str),
                layer)

        for layer_id in list(self._layer_warnings):
            if layer_id not in wanted:
                self._layer_warnings.pop(layer_id, None)

        for layer_id in list(self._actors):
            if (layer_id in wanted
                    and self._layer_keys.get(layer_id) != wanted[layer_id][0]
                    and self._restyle_volume(pl, wanted[layer_id][1], material)):
                self._layer_keys[layer_id] = wanted[layer_id][0]
                continue
            if (layer_id not in wanted
                    or self._layer_keys.get(layer_id) != wanted[layer_id][0]):
                for actor in self._actors.pop(layer_id):
                    self._remove(pl, actor)
                self._layer_keys.pop(layer_id, None)
                self._cull_map.pop(layer_id, None)
                self._slice_state.pop(layer_id, None)
                self._projections.pop(layer_id, None)

        for layer_id, (key, layer) in wanted.items():
            if layer_id in self._actors:
                continue
            self._actors[layer_id] = self._draw_layer(pl, layer, material,
                                                      ) or []
            self._layer_keys[layer_id] = key

        self._culls = [pair for layer_id in self._actors
                       for pair in self._cull_map.get(layer_id, ())]
        self._move_slices()
        self._prune_caches()

    def _prune_caches(self):
        """Forget built geometry no open layer uses, past a few recent spares.

        Every Smoothness value, section position and fibre threshold used to
        stay cached for the life of the session -- a mesh copy or a set of
        streamlines and tubes each -- so memory climbed as you worked even with
        every layer closed (measured: about 140 MB a round of edits). Entries an
        open layer needs are always kept; of the rest, only the most
        recently used few, so stepping a slider back stays instant. A layer
        hidden with its eye still counts as open."""
        # Hidden layers count as in use: they are still in the figure, and
        # showing one again must not re-mesh a volume.
        live = {"geometry": set(), "groups": set(), "fibers": set()}
        for layer in self.spec["layers"]:
            if layer["kind"] == "mesh" and layer.get("mesh"):
                live["geometry"].add(_key(layer, _GEOMETRY_KEYS))
                live["groups"].add(_key(layer, _LOAD_KEYS))
            elif layer["kind"] == "fibers" and layer.get("source"):
                live["fibers"].add(_key(layer, _FIBER_KEYS))
        for cache, keep, spare in ((self._geometry, live["geometry"], _SPARE_GEOMETRY),
                                   (self._groups, live["groups"], _SPARE_GROUPS),
                                   (self._fibers, live["fibers"], _SPARE_FIBERS)):
            unused = [k for k in cache if k not in keep]      # oldest first
            for k in unused[:max(len(unused) - spare, 0)]:
                del cache[k]

    def _move_slices(self):
        """Re-aim slice planes without touching the actors that draw them.

        A plane move is four point coordinates and one texture. Removing the
        actor and adding a new one costs about 100 ms; assigning the points and
        swapping the texture on the actor already there costs about 38 ms.
        """
        for layer in self.spec["layers"]:
            if layer["kind"] != "slices" or not layer.get("visible", True):
                continue
            state = self._slice_state.get(layer["id"])
            if not state or not layer.get("backdrop"):
                continue
            back = self.backdrop(layer["backdrop"])
            lo, hi = back.bounds()
            middle = (lo + hi) / 2.0
            for axis, key in enumerate("xyz"):
                entry = state.get(axis)
                if entry is None:
                    continue
                world_axis, _ = back.world_axis(axis)
                wanted = layer.get(key)
                if wanted is None:
                    wanted = float(middle[world_axis])
                if abs(float(wanted) - entry[2]) < 1e-9:
                    continue
                quad, texture, _landed = back.plane(
                    axis, float(wanted), cmap=layer.get("cmap"),
                    transparent_below=layer.get("transparent_below"),
                    fill=self._slice_fill(layer),
                    overlays=self._slice_overlays(layer))
                entry[0].points = quad.points
                entry[1].SetTexture(texture)
                entry[2] = float(wanted)
                for overlay, line in entry[3] if len(entry) > 3 else ():
                    self._slice_outline(None, back, axis, float(wanted), overlay, line)

    def _slice_overlays(self, layer, record=False):
        """A slice layer's overlays with their scales pinned to the MAP.

        A plane colours only the values it happens to cut through, and the
        suite reads an unpinned scale off whatever it is handed -- so every
        slider step re-scaled the colours, and the colour bar (never recorded
        for a slice) showed whichever scale another layer had left behind.
        Pinned from the map's own voxels, a value is the same colour on every
        plane, in the volume body, and in the bar.
        """
        from circuit_viewer import volume_render as vr

        out = []
        for overlay in _fills(layer.get("overlays")):
            try:
                native = vr.native_values(overlay)
            except (OSError, ValueError) as exc:
                if record:
                    self.warn(f"overlay {overlay.get('name') or overlay['nifti']}: {exc}")
                continue
            if record:
                self._record_scale(overlay, native)
            out.append(vr.pinned(overlay, native))
        return out

    def _slice_fill(self, layer):
        """The colour a slice paints its dropped-out background in, or None.

        Reading it off the scene rather than storing a colour on the layer means
        switching the background to black does not leave every slice carrying a
        white plate behind it.
        """
        if not layer.get("fill_background"):
            return None
        colour = self.spec.get("background") or "#ffffff"
        from matplotlib.colors import to_rgb

        return tuple(int(round(c * 255)) for c in to_rgb(colour))

    def _draw_layer(self, pl, layer, material):
        self._drawing = layer["id"]
        self._layer_warnings.pop(layer["id"], None)
        if layer["kind"] == "slices":
            return self._add_slices(pl, layer)
        if layer["kind"] == "volume":
            # A body with no file yet is a normal intermediate state, exactly
            # as it is for fibres: the panel is still on its picker.
            if not layer.get("source"):
                return []
            return self._add_volume(pl, layer, material)
        if layer["kind"] == "fibers":
            # A freshly added fibre layer has no file yet. That is a normal
            # intermediate state, not an error: failing the whole render because
            # one layer is half-configured would blank the viewport the moment
            # you click "+ Fiber layer".
            if not layer.get("source"):
                return []
            return self._add_fibers(pl, layer, material)
        if layer["kind"] == "electrode":
            if not layer.get("source"):
                return []
            return self._add_electrode(pl, layer, material)
        if not layer.get("mesh"):
            return []           # a fresh mesh layer, still on its picker
        return self._add_mesh(pl, layer, material)

    def _add_volume(self, pl, layer, material, reuse=None):
        """Ray-cast a body. ``volume_render`` builds the VTK volume; this only
        adds it, so the scene's lights, background and camera apply to a body
        exactly as they do to a mesh. A missing or unreadable body is a
        warning on its layer, not a failed render."""
        from circuit_viewer import volume_render as vr

        try:
            source = vr.resolve_source(layer["source"])
            grid = self._volume_grid(source, self._volume_level(layer) or 1,
                                     float(layer.get("body_smooth_vox") or 0.0),
                                     layer.get("body_labels") or [])
            traced = _outlines(layer.get("overlays"))
            actor, scales, warnings = vr.build_actor(
                grid, layer, _fills(layer.get("overlays")), material,
                reuse=reuse, outlines=traced)
        except (OSError, ValueError, MemoryError, KeyError) as exc:
            self.warn(f"volume {layer.get('source')}: {exc}")
            return []
        for message in warnings:
            self.warn(message)
        for values, overlay in scales:
            self._record_scale(overlay, values)
        if actor is not reuse:
            pl.add_actor(actor, reset_camera=False)
        # Framed on its tissue, not its field of view (see _frame_bounds).
        self._framing[actor.GetAddressAsString("vtkObject")] = vr.tissue_bounds(grid)
        # Outlines are traced over the finished frame (_draw_projections): on
        # an x-rayed body, each region as seen through the brain; on a painted
        # one, exactly the patches of the surface the region colours.
        fills = _fills(layer.get("overlays"))
        if not traced:
            self._projections.pop(layer["id"], None)
        elif vr._mode(layer, fills, traced) == "volume":
            self._projections[layer["id"]] = self._projection_jobs(layer, grid, traced)
        else:
            self._projections[layer["id"]] = self._painted_jobs(
                layer, grid, fills, traced, material)
        return [actor]

    def _painted_jobs(self, layer, grid, fills, traced, material):
        """``[(region volume, colour, opacity, width)]`` for a painted body:
        the body drawn unlit, white where each outline overlay is, black
        elsewhere -- a mask of what that overlay colours on screen."""
        from circuit_viewer import volume_render as vr

        jobs = []
        for overlay in traced:
            try:
                actor = vr.region_actor(grid, layer, fills, overlay, material)
            except (OSError, ValueError, MemoryError) as exc:
                self.warn(f"outline {overlay.get('name') or overlay['nifti']}: {exc}")
                continue
            if actor is not None:
                jobs.append((actor, *_outline_style(overlay)))
        return jobs

    def _projection_jobs(self, layer, grid, traced):
        """``[(surface, colour, opacity, width)]``: each outline overlay's
        region as a surface, cut where the body is cut, inside the body unless
        overlays may go outside it."""
        from circuit_viewer import outlines
        from circuit_viewer import volume_render as vr

        jobs = []
        outside = bool(layer.get("overlay_outside"))
        for overlay in traced:
            try:
                surface = outlines.region_surface(
                    overlay, grid, outside,
                    None if outside else vr.body_mask(grid))
                surface = outlines.clip_to_section(surface, layer.get("section"))
            except (OSError, ValueError, MemoryError) as exc:
                self.warn(f"outline {overlay.get('name') or overlay['nifti']}: {exc}")
                continue
            if not surface.n_points:
                continue
            jobs.append((surface, *_outline_style(overlay)))
        return jobs

    def _restyle_volume(self, pl, layer, material):
        """Update a volume layer in place when only its styling changed.

        Opacity, profile, shading, cuts, x-ray depth, fade -- and in x-ray mode
        thresholds and palettes -- live in transfer functions, clipping planes
        and a shader constant. Rebuilding the actor for them re-uploads the
        whole level to the GPU, which on a hi-res level is what made a slider
        lag. ``build_actor`` hands back the same actor restyled when its voxels
        are unchanged, or a new one when they are not. False means "not a
        volume layer": draw it the ordinary way.
        """
        layer_id = layer["id"]
        old = self._actors.get(layer_id) or []
        if layer.get("kind") != "volume" or not layer.get("source") \
                or not old:
            return False
        self._drawing = layer_id
        self._layer_warnings.pop(layer_id, None)
        for extra in old[1:]:
            self._remove(pl, extra)
        actors = self._add_volume(pl, layer, material, reuse=old[0])
        if not actors or actors[0] is not old[0]:
            self._remove(pl, old[0])
        self._actors[layer_id] = actors
        self._cull_map.pop(layer_id, None)
        return True

    def _volume_level(self, layer):
        """The block factor a volume layer is drawn at for this render, or
        None for anything else. A body under the preview budget is 1 at both
        qualities; only a large one is drawn coarse while working."""
        if layer.get("kind") != "volume" or not layer.get("source"):
            return None
        from circuit_viewer import volume_render as vr

        try:
            return vr.level_for(vr.resolve_source(layer["source"]),
                                self._quality)
        except Exception:
            return None         # unreadable: drawing it says why

    def _frame_bounds(self, pl):
        """What the camera fits to: every visible 3D prop's bounds, except that
        a volume counts only its tissue. None when there is no volume, so a
        scene without one frames exactly as VTK always has."""
        import vtk

        live, boxes = {}, []
        for prop in list(pl.renderer.actors.values()):
            if not isinstance(prop, vtk.vtkProp3D) or not prop.GetVisibility():
                continue
            address = prop.GetAddressAsString("vtkObject")
            if address in self._framing:
                live[address] = self._framing[address]
                boxes.append(live[address])
            else:
                box = prop.GetBounds()
                if box is not None and box[0] <= box[1]:
                    boxes.append(tuple(box))
        self._framing = live            # forget volumes no longer on screen
        if not live:
            return None
        boxes = np.asarray(boxes, float)
        return (boxes[:, 0].min(), boxes[:, 1].max(), boxes[:, 2].min(),
                boxes[:, 3].max(), boxes[:, 4].min(), boxes[:, 5].max())

    def _volume_grid(self, source, level=1, smooth_vox=0.0, labels=()):
        """Grids are cached on the path and level: re-reading one costs more
        than the rest of a frame, and a slider must not pay for it. Both levels
        of a large body stay cached, so toggling hi-res back is free."""
        from circuit_viewer import volume_render as vr

        try:
            stamp = os.path.getmtime(source)
        except OSError:
            stamp = 0.0
        labels = tuple(sorted(int(v) for v in labels or ()))
        key = (str(source), int(level), stamp, round(float(smooth_vox), 3), labels)
        if key in self._volume_grids:
            self._volume_grids[key] = self._volume_grids.pop(key)  # most recent
            return self._volume_grids[key]
        for old in [k for k in self._volume_grids if k[0] == key[0]
                    and k[2] != stamp]:
            del self._volume_grids[old]     # the file changed under us
        self._volume_grids[key] = vr.load_grid(source, level, smooth_vox, labels)
        # Both levels of the body in use, and a couple of recent others.
        while len(self._volume_grids) > 4:
            del self._volume_grids[next(iter(self._volume_grids))]
        return self._volume_grids[key]

    def _add_mesh(self, pl, layer, material):
        geom = self._mesh_geometry(layer)
        colours = self._mesh_colours(layer, geom)

        # Geometry is cached by what it IS, not by who is drawing it, so two
        # layers showing the same mesh are handed the SAME PolyData. Painting
        # into it directly then means the second layer's colours land on the
        # first layer's actor -- and, because an actor is only rebuilt when its
        # own key changes, they stay there after the second layer is deleted.
        # A shallow copy shares the points and cells (no re-meshing, no extra
        # memory) while giving each layer its own point_data/cell_data to paint
        # into, which is the isolation the stack was always assumed to have.
        parts = {name: part.copy(deep=False)
                 for name, part in geom["parts"].items()}

        scalars = cmap = clim = None
        rgb = False
        if colours is not None:
            target = colours["target"]
            values = colours["values"]
            # Vertex values are sampled over the parts CONCATENATED, in this
            # same order, so each part gets its own slice. Handing the whole
            # array to every part is what produced "Array 'paint' has length
            # (28441) but a length of (14120) was expected" on any two-part
            # surface -- both hemispheres of a whole brain, painted per vertex.
            wire_values = colours.get("edges")
            at = 0
            for name, part in parts.items():
                store = part.cell_data if target == "cells" else part.point_data
                other = part.point_data if target == "cells" else part.cell_data
                # A mesh must never carry both, or VTK picks whichever it finds
                # first and the lengths disagree.
                other.pop("paint", None)
                other.pop("edge_paint", None)
                size = part.n_cells if target == "cells" else part.n_points
                store["paint"] = (values if len(values) == size
                                  else values[at:at + size])
                if wire_values is not None:
                    store["edge_paint"] = (wire_values if len(wire_values) == size
                                           else wire_values[at:at + size])
                at += size
            scalars = "paint"
            rgb = colours["mode"] == "rgb"
            cmap, clim = colours.get("cmap"), colours.get("clim")

        fills = _fills(layer.get("overlays"))
        gradient = any(o.get("mode") == "vertex" for o in fills)
        faceted = _auto(layer["faceted"], not gradient)
        edges = _edges_on(layer, geom, fills)
        wire_paint = colours is not None and "edges" in colours
        # Map-coloured edges are drawn even with the plain highlight off: the
        # checkbox then only decides whether the UNMAPPED edges show.
        edges = edges or wire_paint
        translucent = layer["opacity"] < 1.0
        # Only when asked. Inferring it from translucency meant a scene that
        # happened to be see-through started deleting the near hemisphere on its
        # own, which is startling and, with slices in the scene, looks like the
        # slices are being eaten rather than the mesh.
        cullable = (set(parts) == {"L", "R"}
                    and bool(layer.get("cull_near_hemisphere")))

        actors, culls, fill_actors = [], [], []
        for side, mesh in parts.items():
            actor = pl.add_mesh(
                mesh, scalars=scalars, cmap=cmap, clim=clim, rgb=rgb,
                color=None if scalars else layer["color"],
                nan_color=BASE_COLOR, nan_opacity=1.0,
                opacity=layer["opacity"],
                smooth_shading=not faceted,
                culling="back" if translucent else None,
                show_scalar_bar=False, **material,
            )
            actors.append(actor)
            fill_actors.append(actor)
            if edges:
                # A separate wireframe actor, not add_mesh(show_edges=True):
                # VTK multiplies the actor's opacity into its own edges, so a
                # translucent mesh washes its wireframe out and there is no way
                # to ask for a faint mesh under a crisp one. Two actors give the
                # two opacities the independence the eye expects.
                if wire_paint:
                    # RGBA per vertex (or cell): the alpha carries both the
                    # edge opacity and the maps' own, so the actor stays at 1.
                    wire = pl.add_mesh(
                        mesh, style="wireframe", scalars="edge_paint", rgb=True,
                        line_width=float(layer["edge_width"] or 0.6),
                        lighting=False, show_scalar_bar=False,
                    )
                else:
                    wire = pl.add_mesh(
                        mesh, style="wireframe", color=layer["edge_color"],
                        opacity=float(layer["edge_opacity"]),
                        line_width=float(layer["edge_width"] or 0.6),
                        lighting=False, show_scalar_bar=False,
                    )
                self._track_line(wire, float(layer["edge_width"] or 0.6))
                actors.append(wire)
            if cullable:
                culls.append((side, actor))
        self._cull_map[layer["id"]] = culls
        jobs = self._mesh_outline_jobs(layer, geom, parts, fill_actors)
        if jobs:
            self._projections[layer["id"]] = jobs
        else:
            self._projections.pop(layer["id"], None)
        return actors

    def _mesh_outline_jobs(self, layer, geom, parts, fills):
        """Outline jobs for a mesh layer: one per drawn piece per outlined
        overlay, ``(piece, colour, opacity, width, actor)``.

        The outline is the edge of what the overlay COLOURS on this mesh: the
        very per-vertex (or per-region) values the fill paints -- sampled,
        smoothed along the surface, thresholded exactly as the fill is -- and,
        on a sequential scale, only from the colour bar's minimum up. Each
        piece carries that as a 0/1 field; it is drawn unlit off screen with
        the frame's camera and the edge of the white is traced over the frame
        (see _draw_projections), so the line sits exactly where the colour
        starts, is hidden wherever the mesh hides it, and is ``width`` pixels.
        ``actor`` is the piece's own fill actor: a piece culled from view
        draws no outline.
        """
        traced = _outlines(layer.get("overlays"))
        if not traced:
            return []
        target = self._target(geom)
        jobs = []
        for overlay in traced:
            try:
                values = self._overlay_values(layer, geom, overlay)
            except (ValueError, FileNotFoundError, OSError) as exc:
                self.warn(f"outline {overlay.get('name') or overlay['nifti']}: {exc}")
                continue
            inside = _coloured(values, overlay)
            if not (inside > 0.5).any():
                continue
            colour, opacity, width = _outline_style(overlay)
            at = 0
            for (name, part), actor in zip(parts.items(), fills):
                size = part.n_cells if target == "cells" else part.n_points
                share = inside if len(inside) == size else inside[at:at + size]
                at += size
                if not (share > 0.5).any():
                    continue
                if target == "points":
                    # A light average along the surface, so the traced edge is
                    # round rather than stepping from triangle to triangle.
                    share = _surface_average(part, share, _OUTLINE_SMOOTH_MM)
                piece = part.copy(deep=False)
                piece.point_data.pop("paint", None)
                piece.cell_data.pop("paint", None)
                store = piece.cell_data if target == "cells" else piece.point_data
                store["_inside"] = share
                jobs.append((piece, colour, opacity, width, actor))
        return jobs

    def _remove(self, pl, actor):
        pl.remove_actor(actor, reset_camera=False, render=False)

    def _track_line(self, actor, width):
        """Remember an outline actor's width in screen pixels; it is scaled
        with the render's supersampling at draw time."""
        self._lines[actor.GetAddressAsString("vtkObject")] = (actor, width)

    def _style_lines(self, pl):
        """Before a frame: outline widths for this render's supersampling.
        Forgets actors no longer drawn."""
        live = {a.GetAddressAsString("vtkObject")
                for a in pl.renderer.actors.values()}
        self._lines = {k: v for k, v in self._lines.items() if k in live}
        # VTK's SSAA pass draws at twice the size and scales down, which
        # halves every line's width in an export (measured: 4 px -> 2 px), so
        # the width it is given is doubled to come out as asked.
        scale = (max(self._font_scale, 1) * self._line_scale
                 * (2 if self._supersampled() else 1))
        for actor, width in self._lines.values():
            actor.GetProperty().SetLineWidth(width * scale)

    def _supersampled(self):
        """Whether VTK draws this frame at twice the size and shrinks it: SSAA,
        and FXAA on an EGL/OSMesa build, where pyvista quietly swaps FXAA for
        SSAA -- without this, hi-res lines came out at half width there."""
        aa = getattr(self, "_antialias", None)
        if aa == "ssaa":
            return True
        if aa == "fxaa":
            try:
                from pyvista.plotting.utilities.gl_checks import uses_egl

                return bool(uses_egl())
            except Exception:
                return False
        return False

    def _draw_projections(self, pl, image):
        """Outlines of mesh and volume overlays, laid over a finished frame:
        each region drawn alone with this frame's camera, and a line centred
        on its edges drawn on the image (see outlines.py)."""
        from circuit_viewer import outlines

        visible = {layer["id"] for layer in self.spec["layers"]
                   if layer.get("visible", True)}
        jobs = [job for layer_id, items in self._projections.items()
                if layer_id in visible and layer_id in self._actors
                for job in items]
        if not jobs:
            # Nothing outlined any more: let the off-screen renderer drop the
            # region volumes and surfaces it was holding (otherwise kept until
            # the next outline appeared -- a whole body's voxels for a volume).
            if self._projector is not None and self._projector._pl is not None:
                self._projector.keep_only([])
            return image
        if self._projector is None:
            self._projector = outlines.Projector()
        if self._projector._pl is not None:
            self._projector.keep_only([job[0] for job in jobs])
        size = (image.shape[1], image.shape[0])
        camera = pl.renderer.GetActiveCamera()
        import pyvista as pv

        for job in jobs:
            thing, colour, opacity, width = job[:4]
            if len(job) > 4 and not job[4].GetVisibility():
                continue                    # its piece is culled from view
            # A region volume is a full body render: half size while working.
            scale = (0.5 if self._quality != "export"
                     and not isinstance(thing, pv.DataSet) else 1.0)
            try:
                mask = self._projector.mask(camera, size, thing, scale)
            except Exception as exc:        # an outline must never cost the frame
                self.warn(f"outline: {exc}")
                continue
            image = outlines.draw_outline(image, mask, colour, opacity,
                                          width * max(self._font_scale, 1)
                                          * self._line_scale)
        return image

    def backdrop(self, reference):
        """An open Backdrop, kept for the session so slices stay ~1 ms reads.

        A prepared ``.npz`` is read slice by slice from disk. A NIfTI -- one
        dropped on + Slices or opened as a temporary backdrop -- is read once
        and held in memory, 8-bit, by ``_nifti_backdrop``: nothing is written.
        """
        from calvin_utils.plotting_utils.backdrops import BACKDROP_DIR, Backdrop

        text = str(reference)
        path = Path(text).expanduser()
        if not path.suffix:
            path = Path(BACKDROP_DIR).expanduser() / f"{text}.npz"
        key = str(path)
        if key not in self._backdrops:
            if text.lower().endswith((".nii", ".nii.gz")):
                self._backdrops[key] = _nifti_backdrop(path)
            else:
                self._backdrops[key] = Backdrop(path)
            # Temporary backdrops are whole volumes in memory; keep a few.
            while len(self._backdrops) > 6:
                self._backdrops.pop(next(iter(self._backdrops)))
        return self._backdrops[key]

    def _add_slices(self, pl, layer):
        if not layer.get("backdrop"):
            return []                   # a freshly added layer has no volume yet
        back = self.backdrop(layer["backdrop"])
        lo, hi = back.bounds()
        middle = (lo + hi) / 2.0
        actors = []
        state = {}
        traced = _outlines(layer.get("overlays"))

        for axis, key in enumerate(("x", "y", "z")):
            if not layer.get(f"show_{key}", True):
                continue
            world_axis, _ = back.world_axis(axis)
            wanted = layer.get(key)
            if wanted is None:
                wanted = float(middle[world_axis])
            quad, texture, _at = back.plane(
                axis, float(wanted), cmap=layer.get("cmap"),
                transparent_below=layer.get("transparent_below"),
                fill=self._slice_fill(layer),
                overlays=self._slice_overlays(layer, record=True),
            )
            # Unlit on purpose: these are photographs of tissue, and shading
            # them would add a highlight that means nothing anatomically.
            actor = pl.add_mesh(quad, texture=texture, lighting=False,
                                opacity=float(layer["opacity"]),
                                show_scalar_bar=False)
            actors.append(actor)
            lines = []
            for overlay in traced:
                line = self._slice_outline(pl, back, axis, float(wanted), overlay)
                if line is not None:
                    actors.append(line)
                    lines.append((overlay, line))
            state[axis] = [quad, actor, float(wanted), lines]
        self._slice_state[layer["id"]] = state
        return actors

    def _slice_outline(self, pl, back, axis, world_mm, overlay, actor=None):
        """An overlay's outline on one slice plane: a new line actor, or
        ``actor`` re-pointed at the plane's new position."""
        from circuit_viewer import outlines

        try:
            poly = outlines.plane_outline(back, axis, back.index_for(axis, world_mm),
                                          overlay)
        except (OSError, ValueError, MemoryError) as exc:
            self.warn(f"outline {overlay.get('name') or overlay['nifti']}: {exc}")
            return actor
        if actor is not None:
            actor.GetMapper().SetInputData(poly)
            return actor
        colour, opacity, width = _outline_style(overlay)
        from matplotlib.colors import to_rgb

        actor = _line_actor(poly)
        actor.GetProperty().SetColor(*to_rgb(colour))
        actor.GetProperty().SetOpacity(opacity)
        pl.add_actor(actor, reset_camera=False)
        self._track_line(actor, width)
        return actor

    def _electrode_geometry(self, layer):
        """``{part name: PolyData}`` for a lead, already placed on the patient.

        Cached on the source, the model override and the per-lead poses. Reading a
        model file and solving the placement is a fraction of a second, but it is
        pure geometry: hiding a contact or recolouring the shaft must not redo it.
        """
        from calvin_utils.plotting_utils.electrode_render import electrode_meshes

        key = _key(layer, ("source", "model", "poses", "mirror"))
        group = self._electrodes.get(key)
        if group is None:
            group = electrode_meshes(layer["source"], ELECTRODE_MODEL_DIR, model=layer.get("model"), poses=layer.get("poses"), mirror=layer.get("mirror"))
            self._electrodes[key] = group
            # Every slider step of a trajectory is a new key: keep a few recent.
            while len(self._electrodes) > 4:
                self._electrodes.popitem(last=False)
        self._electrodes.move_to_end(key)
        hidden = set(layer.get("hidden_regions") or ())
        return {name: mesh for name, mesh in group.items() if name not in hidden}

    def _add_electrode(self, pl, layer, material):
        """Two actors: the shaft and the contacts, each in its own colour."""
        group = self._electrode_geometry(layer)
        actors = []
        for kind, colour in (("insulation", layer["insulation_color"]),
                             ("contacts", layer["contact_color"])):
            parts = [m for name, m in group.items() if kind in name]
            if not parts:
                continue
            merged = parts[0]
            for extra in parts[1:]:
                merged = merged.merge(extra)
            actors.append(pl.add_mesh(
                merged, color=colour, opacity=float(layer["opacity"]),
                smooth_shading=True, show_scalar_bar=False, **material))
        return actors

    def _fiber_geometry(self, layer):
        """The streamlines for a fibre layer, as (mesh, is_tubed).

        Two caches, because they cost very different amounts. Reading and
        smoothing a Lead-DBS MAT takes ten seconds or so and is keyed on the
        selection; sweeping those polylines into real tubes multiplies the point
        count by ~8 and is only worth it for a screenshot. Preview draws the
        lines with ``render_lines_as_tubes``, which is a shading trick on line
        primitives -- fast, and flat under a moving light, because a line has no
        surface to catch it. Hi-res draws actual swept geometry with normals,
        which is where the sheen comes from.
        """
        from calvin_utils.plotting_utils.fiber_render import (
            line_width_for_radius, load_fibers, radius_for_width, tube_fibers)

        key = _key(layer, _FIBER_KEYS)
        entry = self._fibers.get(key)
        if entry is not None:
            self._fibers.move_to_end(key)
        if entry is None:
            poly = load_fibers(
                layer["source"],
                fiber_atlas_path=layer.get("fiber_atlas_path"),
                sign=layer.get("sign", "both"),
                min_abs_value=layer.get("min_abs_value"),
                max_abs_value=layer.get("max_abs_value"),
                top_percent=layer.get("top_percent"),
                max_lines=layer.get("max_lines"),
                step=int(layer.get("step") or 1),
                smooth=int(layer.get("smooth") or 0),
            )
            entry = {"lines": poly, "tubes": None, "tube_shape": None}
            self._fibers[key] = entry

        # Hiding the unpainted atlas fibres happens HERE, on the streamlines,
        # before anything is swept. Doing it after tubing did nothing at all:
        # a tube is polygons, and the filter walks poly.lines, which a tubed
        # mesh does not have -- so preview filtered and export did not, and the
        # two diverged silently. Filtering first also means the tubes are only
        # built for fibres that will be drawn.
        lines = entry["lines"]
        # Gates first: they are geometry, and pruning before the value filter
        # means the "no fibres carry a value" warning is about what is actually
        # left rather than about the whole atlas.
        gates = [g for g in (layer.get("prune") or []) if g.get("enabled", True)]
        if gates:
            from calvin_utils.plotting_utils.fiber_render import prune_streamlines

            pruned = prune_streamlines(lines, gates)
            if pruned is None:
                self.warn(f"{layer['name']} — no fibres satisfy the gates")
            else:
                lines = pruned
        if layer.get("values_only"):
            lines = self._painted_fibers(layer, lines) or lines

        # ALWAYS real swept geometry, never render_lines_as_tubes. That shader
        # takes a width in PIXELS and the driver quantises it to an integer, so
        # every thickness under about 1.4 px drew the identical flat one-pixel
        # hairline: the bottom third of the slider changed nothing and caught no
        # light at all (measured: pixel counts identical and luminance spread
        # exactly 0.0 at 0%, 10% and 20%). Real tubes are smooth and lit across
        # the whole range, so thickness behaves like a slider.
        #
        # Hi-res is now what it was always described as -- more pixels, better
        # antialiasing, and a rounder tube -- rather than a switch between two
        # different kinds of geometry that the preview and the export could
        # disagree about.
        radius = _thickness_mm(layer)
        sides = int(layer.get("tube_sides") or 8) if layer.get("tubes") else 5
        shape = (round(radius, 4), sides, int(lines.n_points),
                 _key(layer, ("prune",)),
                 _key(layer, ("values_only",)),
                 json.dumps([o.get("nifti") for o in (layer.get("overlays") or [])],
                            sort_keys=True, default=str))
        if entry["tubes"] is None or entry["tube_shape"] != shape:
            entry["tubes"] = tube_fibers(lines, radius=radius, sides=sides)
            entry["tube_shape"] = shape
        return entry["tubes"], True

    def _add_fibers(self, pl, layer, material=None):
        from calvin_utils.plotting_utils.fiber_render import (
            fiber_material, line_width_for_radius)
        from calvin_utils.plotting_utils.render_scene import resolve_scale

        poly, tubed = self._fiber_geometry(layer)
        # The panel MODULATES the fibre material rather than replacing it, so
        # the sliders reach fibres without flattening the gloss that makes a
        # bundle read as a bundle. One material for the whole layer, whatever
        # colours its points end up with.
        kw = dict(opacity=float(layer["opacity"]), lighting=True,
                  show_scalar_bar=False, **fiber_material(material))

        painted = self._fiber_overlays(layer, poly)
        if painted is not None:
            poly = poly.copy(deep=False)
            poly["paint"] = painted
            if tubed:
                kw["smooth_shading"] = True
            else:
                kw.update(line_width=line_width_for_radius(_thickness_mm(layer)),
                          render_lines_as_tubes=True)
            return [pl.add_mesh(poly, scalars="paint", rgb=True, **kw)]
        if tubed:
            kw["smooth_shading"] = True
        else:
            from calvin_utils.plotting_utils.fiber_render import line_width_for_radius

            kw.update(line_width=line_width_for_radius(_thickness_mm(layer)),
                      render_lines_as_tubes=True)
        if layer["coloring"] == "direction":
            return [pl.add_mesh(poly, scalars="dir", rgb=True, **kw)]
        if layer["coloring"] == "value" and self._has_values(poly):
            cmap, clim = resolve_scale(np.asarray(poly["vals"]),
                                       cmap=layer["palette"], clim=layer["clim"],
                                       color=layer["color"])
            return [pl.add_mesh(poly, scalars="vals", cmap=cmap, clim=clim, **kw)]
        return [pl.add_mesh(poly, color=layer["color"], **kw)]

    @property
    def warnings(self):
        """Every live layer's complaints, in layer order."""
        out = []
        for layer in self.spec["layers"]:
            out.extend(self._layer_warnings.get(layer["id"], ()))
        return out

    def warn(self, message):
        """Record something the figure should say without failing the render.

        A bad file in one layer is normal while a figure is being built. The
        render still has to produce the other layers, so these are collected and
        handed back beside the image rather than raised.

        Kept against the LAYER, not the render. Actors are cached, so a layer is
        only rebuilt when something about it changes; a warning tied to one
        render would appear once and then be wiped by the next orbit, leaving a
        layer silently unpainted. This way it stands until the layer is rebuilt
        without it.
        """
        bucket = self._layer_warnings.setdefault(self._drawing, [])
        if message not in bucket:
            bucket.append(str(message))

    def _painted_fibers(self, layer, poly):
        """Keep only the streamlines some overlay has a value for.

        Dropping them rather than making them transparent, because on a
        whole-atlas tractogram the fibres a result says nothing about are the
        overwhelming majority: not drawing them is both the clearer picture and
        the far cheaper one.

        A fibre survives if ANY overlay paints it, so a stack still composites
        normally -- this removes what none of them reach, not what the topmost
        one misses.
        """
        from calvin_utils.plotting_utils.fiber_render import streamline_values
        from calvin_utils.plotting_utils.mesh_paint import _select
        from calvin_utils.plotting_utils.overlay_paint import overlay_colours

        overlays = [o for o in _fills(layer.get("overlays")) if o.get("nifti")]
        if not overlays:
            return None

        held = poly.field_data.get("atlas_fibers")
        count = int(np.asarray(held).ravel()[0]) if held is not None else None
        keep = None
        for overlay in overlays:
            try:
                values = streamline_values(overlay["nifti"], count=count)
            except (ValueError, FileNotFoundError, OSError) as exc:
                self.warn(f"{layer['name']} — {exc}")
                continue
            values = _select(values, sign=overlay["sign"],
                             absolute=overlay["absolute"],
                             threshold=overlay["threshold"],
                             max_value=overlay.get("max_value"))
            # The ALPHA an overlay would paint, not merely whether a value
            # survived processing. Checking finiteness ignored the colour scale
            # entirely, so raising Colour min washed fibres to the bottom colour
            # instead of dropping them -- they had values, they just had no
            # colour left. Alpha is what "this overlay paints nothing here"
            # actually means, and it accounts for clim and Skip below min.
            _rgb, alpha = overlay_colours(values, overlay)
            alive = np.asarray(alpha) > 0.0
            keep = alive if keep is None else (keep | alive)
        if keep is None or keep.all():
            return None

        owner = np.asarray(poly["fiber"])
        wanted = keep[owner]
        # Walk the line array rather than filtering points: a streamline is one
        # cell, and keeping a subset of its points would stitch the survivors
        # into a line that runs through places no fibre goes.
        lines, out, points = poly.lines, [], []
        cursor = i = 0
        remap = {}
        while i < len(lines):
            n = int(lines[i])
            ids = lines[i + 1:i + 1 + n]
            if wanted[ids[0]]:
                for pid in ids:
                    if pid not in remap:
                        remap[pid] = cursor + len(points)
                        points.append(pid)
                out.append(np.r_[n, [remap[pid] for pid in ids]])
            i += 1 + n
        if not out:
            self.warn(f"{layer['name']} — no fibres carry a value; showing all")
            return None

        import pyvista as pv

        index = np.asarray(points, np.int64)
        kept = pv.PolyData()
        kept.points = np.asarray(poly.points)[index]
        kept.lines = np.concatenate(out)
        for name in ("vals", "fiber", "dir"):
            if name in poly.point_data:
                kept[name] = np.asarray(poly[name])[index]
        kept.field_data["atlas_fibers"] = np.array([count or 0], dtype=np.int64)
        return kept

    def _fiber_overlays(self, layer, poly):
        """Per-point RGB for a fibre layer's overlay stack, or None.

        Each overlay is one value per ATLAS fibre. ``poly["fiber"]`` says which
        atlas fibre every point came from, which is what survives subsampling
        and thresholding -- joining on position in the array would silently
        colour the wrong bundles once any fibre had been dropped.

        The layer's own colouring is the base, and the stack paints over it, so
        a bundle can carry a background that later findings overwrite exactly
        where they have something to say and nowhere else.
        """
        from calvin_utils.plotting_utils.fiber_render import streamline_values
        from calvin_utils.plotting_utils.mesh_paint import _select
        from calvin_utils.plotting_utils.overlay_paint import (
            composite, overlay_colours,
        )
        from matplotlib.colors import to_rgb

        overlays = [o for o in _fills(layer.get("overlays")) if o.get("nifti")]
        if not overlays:
            return None

        owner = np.asarray(poly["fiber"])
        held = poly.field_data.get("atlas_fibers")
        count = int(np.asarray(held).ravel()[0]) if held is not None else None
        stack = []
        for overlay in overlays:
            try:
                values = streamline_values(overlay["nifti"], count=count)
            except (ValueError, FileNotFoundError, OSError) as exc:
                # A values file that does not fit this atlas is skipped, not
                # fatal: blanking the whole figure would hide the layers that
                # are fine and say nothing about which file was wrong.
                # The message already names the file, so do not prefix it again.
                self.warn(f"{layer['name']} — {exc}")
                continue
            values = _select(values, sign=overlay["sign"],
                             absolute=overlay["absolute"],
                             threshold=overlay["threshold"],
                             max_value=overlay.get("max_value"))
            rgb, alpha = overlay_colours(values[owner], overlay)
            rgb, alpha = self._fiber_paint(layer, poly, overlay, rgb, alpha)
            if layer.get("coloring", "solid") == "value":
                self._record_scale(overlay, values)
            stack.append((rgb, alpha))
        if not stack:
            return None

        return (composite(self._fiber_base(layer, poly), stack)
                * 255).astype(np.uint8)

    @staticmethod
    def _fiber_paint(layer, poly, overlay, rgb, alpha):
        """Apply the layer's "Colour by" to the fibres one overlay paints.

        The values placed in a fibre layer are what the layer is FOR, so the
        layer's colouring has to govern them, not only the unpainted atlas
        under them. Before this, an overlay always painted its own value ramp
        and "Colour by" only reached the backdrop -- which on an atlas has no
        values at all -- so switching between value and solid changed nothing
        that could be seen.

        The overlay still decides WHICH fibres it paints (sign, threshold,
        value window, Skip below min: all folded into ``alpha``); the colouring
        only decides what colour they get.
        """
        from matplotlib.colors import to_rgb

        coloring = layer.get("coloring", "solid")
        if coloring == "value":
            return rgb, alpha
        # Flat alpha: the value-shaped fade-in of a LUT ramp belongs to a value
        # colouring. In solid a painted fibre is simply painted.
        painted = np.asarray(alpha) > 0.0
        flat = np.where(painted, float(overlay.get("opacity", 1.0)), 0.0)
        if coloring == "direction" and "dir" in poly.point_data:
            return np.asarray(poly["dir"], float) / 255.0, flat
        solid = to_rgb(overlay.get("color") or layer["color"])
        return np.broadcast_to(np.asarray(solid, float), np.shape(rgb)).copy(), flat

    @staticmethod
    def _has_values(poly):
        """Whether a layer's OWN streamlines carry values worth a colour scale.

        A bare atlas loads with every value 0.0 -- its findings arrive as
        overlays -- and a ramp over a constant draws one arbitrary colour.
        """
        if "vals" not in poly.point_data:
            return False
        vals = np.asarray(poly["vals"], float)
        vals = vals[np.isfinite(vals)]
        return vals.size > 0 and float(np.ptp(vals)) > 0.0

    @staticmethod
    def _fiber_base(layer, poly):
        """The layer's OWN colouring, as float RGB per point.

        This is the backdrop a finding is painted over, and it has to honour
        all three colourings. "direction" was handled here and "solid" fell out
        of the else, but "value" landed in that else too -- so adding a single
        values overlay silently turned a value-coloured atlas into a flat slab
        of the layer colour, and Colour by value looked broken the moment a
        finding was dropped on it.
        """
        from calvin_utils.plotting_utils.render_scene import resolve_scale
        from matplotlib.colors import to_rgb

        coloring = layer.get("coloring", "solid")
        if coloring == "direction" and "dir" in poly.point_data:
            return np.asarray(poly["dir"], float) / 255.0
        if coloring == "value" and RenderSession._has_values(poly):
            values = np.asarray(poly["vals"], float)
            cmap, clim = resolve_scale(values, cmap=layer["palette"],
                                       clim=layer["clim"],
                                       color=layer["color"])
            span = (clim[1] - clim[0]) or 1.0
            scaled = np.clip((values - clim[0]) / span, 0.0, 1.0)
            return np.asarray(cmap(scaled))[..., :3]
        return np.tile(np.asarray(to_rgb(layer["color"]), float),
                       (poly.n_points, 1))

    # -- rendering ----------------------------------------------------------
    def render(self, *args, **kwargs):
        """Render the current spec and return PNG bytes.

        The work is handed to the render thread and waited on, so callers can be
        on whatever thread the web server gave them.
        """
        return self._gl.submit(self._render, *args, **kwargs).result()

    def _render(self, size=(760, 640), quality="preview", scalar_bar=True,
                camera=None, antialias=None, line_scale=1.0):
        size = (int(size[0]), int(size[1]))
        self._line_scale = max(float(line_scale or 1.0), 0.1)
        if antialias is None:
            antialias = "ssaa" if quality == "export" else "off"
        # VTK's SSAA pass drops ray-cast volumes -- measured: a body that
        # renders 14,600 pixels without it renders none with it, and FXAA falls
        # back to the same pass on EGL/OSMesa builds. A scene with a body is
        # supersampled here instead: rendered at twice the size with the pass
        # off, then downsampled with a Lanczos filter, which is the same
        # antialiasing done by hand and works on every build.
        factor = 1
        if antialias in ("ssaa", "fxaa") and self._has_volume():
            factor, antialias = 2, "off"
        render_size = (size[0] * factor, size[1] * factor)
        self._font_scale = factor       # bar text keeps its size after downsampling
        pl = self._ensure_plotter(render_size)
        # One render per frame. pyvista renders after nearly every scene edit
        # -- adding a mesh, removing a bar, resetting the camera -- and a frame
        # used to draw the whole scene five times over before the one render
        # that is kept. With a ray-cast body each of those is a full volume
        # render. Everything below edits the scene silently; the render at the
        # end is the only one.
        pl.suppress_rendering = True
        try:
            png = self._compose(pl, render_size, quality, scalar_bar, camera,
                                antialias)
        finally:
            pl.suppress_rendering = False
        if factor == 1:
            return png
        from PIL import Image

        image = Image.open(io.BytesIO(png)).convert("RGB")
        image = image.resize(size, Image.LANCZOS)
        return _png(np.asarray(image), fast=quality != "export")

    def _has_volume(self):
        return any(layer.get("kind") == "volume" and layer.get("visible", True)
                   and layer.get("source") for layer in self.spec["layers"])

    def _compose(self, pl, size, quality, scalar_bar, camera, antialias):
        self._quality = quality
        self._antialias = antialias
        self._sync_actors(pl)

        cam = dict(self.spec["camera"], **(camera or {}))
        d, up = camera_direction(cam["azimuth"], cam["elevation"])
        centre = np.asarray(self.spec["center"], float)

        # Hemisphere culling follows the camera, not a named view: whichever
        # half is between the lens and the brain is the one that stacks.
        for side, actor in getattr(self, "_culls", []):
            near = "L" if d[0] < 0 else "R"
            actor.visibility = not (abs(d[0]) > 0.5 and side == near)

        pl.camera_position = [tuple(centre + d * self.spec["distance"]),
                              tuple(centre), tuple(up)]
        # Fit to the actual viewport before zooming. The window is whatever
        # shape the browser pane is, and a fixed distance that frames a brain
        # nicely in a square crops it badly in a wide one; reset_camera keeps
        # the direction just set and solves the framing for this aspect ratio.
        # Parallel projection is what the earlier figures used, and it is the
        # honest choice for a figure someone will measure by eye: two regions
        # the same size render the same size whatever their depth.
        pl.camera.parallel_projection = bool(self.spec.get("parallel", True))
        pl.renderer.reset_camera(render=False, bounds=self._frame_bounds(pl))
        self._style_lines(pl)
        pl.camera.zoom(float(cam["zoom"]))
        # A fixed framing in millimetres (half the view's height), for figures
        # that put several views side by side: fitting each view to its own
        # bounds would draw a cut hemisphere bigger than a whole brain.
        if cam.get("parallel_scale"):
            pl.camera.parallel_scale = float(cam["parallel_scale"])
        # Pan: the framing above centres the scene; a pan slides the camera
        # and its focal point together, in world millimetres, so a close zoom
        # can be steered onto any region. Clipping follows the new position.
        pan = np.asarray(cam.get("pan") or self.spec.get("pan") or (0, 0, 0),
                         float)[:3]
        if np.any(pan):
            pl.camera.focal_point = tuple(np.asarray(pl.camera.focal_point) + pan)
            pl.camera.position = tuple(np.asarray(pl.camera.position) + pan)
        pl.renderer.ResetCameraClippingRange()
        # What one screen unit is in the world, for the browser to pan by.
        focal = np.asarray(pl.camera.focal_point, float)
        position = np.asarray(pl.camera.position, float)
        forward = focal - position
        forward /= max(np.linalg.norm(forward), 1e-12)
        view_up = np.asarray(pl.camera.up, float)
        right = np.cross(forward, view_up)
        right /= max(np.linalg.norm(right), 1e-12)
        if pl.camera.parallel_projection:
            height_mm = 2.0 * float(pl.camera.parallel_scale)
        else:
            height_mm = 2.0 * float(np.linalg.norm(focal - position)) * float(
                np.tan(np.radians(pl.camera.view_angle) / 2.0))
        self.view = {"height_mm": height_mm, "right": right.tolist(),
                     "up": view_up.tolist()}

        # Removal has to be BY TITLE: remove_scalar_bar() with no argument
        # raises as soon as more than one bar is present, which is exactly the
        # two-map case this exists for.
        # .keys() explicitly: pyvista's ScalarBars is a mapping with no
        # __iter__, so iterating it falls back to the sequence protocol and
        # asks for item 0 -- "KeyError: 0" the moment a bar already exists.
        existing = getattr(pl, "scalar_bars", None)
        for title in list(existing.keys()) if existing is not None else []:
            try:
                pl.remove_scalar_bar(title, render=False)
            except Exception:
                pass
        if scalar_bar:
            # Bar text has to read against whatever the background is; a dark
            # label on a black backdrop is an invisible colour bar. Several
            # bars stack in from the right edge, first overlay outermost, so a
            # two-map figure carries both scales instead of one caption that
            # only happens to be true of one of them.
            ink = ink_for(self.spec["background"])
            bars = self._bars()
            self._bar_mappers = []      # held, or VTK collects them mid-render
            # One bar stands vertically at the right edge, which suits a tall
            # brain. Two or more STACK HORIZONTALLY instead: a vertical bar's
            # title sits above it, so side-by-side bars overwrite each other's
            # labels -- and with two maps the label is the whole point.
            stacked = len(bars) > 1
            for i, (label, cmap, clim) in enumerate(bars):
                mapper = _bar_mapper_for(cmap, clim)
                self._bar_mappers.append(mapper)
                # pyvista keys its bars by title, and two overlays very often
                # carry the SAME filename, so a bare label would collide and
                # the second bar would replace the first. Trailing spaces make
                # the key unique and render as nothing.
                title = str(label)[:22] + " " * i
                if stacked:
                    place = dict(vertical=False, width=0.26, height=0.045,
                                 position_x=0.70,
                                 position_y=0.06 + i * 0.13)
                else:
                    place = dict(vertical=True, width=0.03, height=0.55,
                                 position_x=0.90, position_y=0.22)
                pl.add_scalar_bar(mapper=mapper, title=title, color=ink,
                                  label_font_size=int(12 * self._font_scale),
                                  title_font_size=int(11 * self._font_scale),
                                  n_labels=4, fmt="%.3g", **place)

        # Geometry quality and anti-aliasing are separate choices. Supersampling
        # 3 M points of tube geometry costs seconds, which is right for the file
        # you are going to publish and wrong for looking at it on screen, so the
        # hi-res TOGGLE gets real tubes with cheap AA and only the export pays
        # for both.
        if antialias == "ssaa":
            pl.enable_anti_aliasing("ssaa")
        elif antialias == "fxaa":
            pl.enable_anti_aliasing("fxaa")
        else:
            pl.disable_anti_aliasing()

        # A reused off-screen Plotter hands back the LAST RENDERED buffer:
        # screenshot() does not re-render for a camera move, so without this the
        # viewport silently freezes on the first frame while every orbit appears
        # to succeed. Found by diffing consecutive frames, not by watching it.
        pl.suppress_rendering = False
        pl.render()
        image = pl.screenshot(return_img=True)
        image = self._draw_projections(pl, image)
        return _png(image, fast=quality != "export")

    def export(self, path, size=(1700, 1400), scalar_bar=True, camera=None,
               line_scale=1.0):
        Path(path).expanduser().write_bytes(
            self.render(size=size, quality="export", scalar_bar=scalar_bar,
                        camera=camera, antialias="ssaa", line_scale=line_scale)
        )
        return str(Path(path).expanduser())


def _bar_mapper_for(cmap, clim):
    """A mapper that exists only to carry a lookup table to add_scalar_bar."""
    import pyvista as pv
    import vtk

    from calvin_utils.plotting_utils.palettes import opaque

    # The bar shows the scale's COLOURS. A LUT's alpha ramp (MRIcroGL's fade
    # their low end out) would draw the bottom of the bar as background.
    lut = pv.LookupTable(cmap=opaque(cmap) if hasattr(cmap, "N") else cmap)
    lut.scalar_range = (float(clim[0]), float(clim[1]))
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(float(clim[0]), float(clim[1]))
    return mapper


def _merge(meshes):
    meshes = [m for m in meshes if m.n_points]
    out = meshes[0].copy()
    for extra in meshes[1:]:
        out = out.merge(extra)
    return out


def _png(image, fast=False):
    """PNG bytes. Previews go over localhost, so spending 27 ms squeezing them
    smaller is 27 ms taken straight out of the frame rate."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
        buffer, format="PNG", compress_level=1 if fast else 6)
    return buffer.getvalue()

"""A NIfTI drawn as a BODY, with VTK's ray-cast volume mapper.

Its own module, and its own layer kind, because it answers a different question
from the two that already exist. A ``mesh`` layer turns a volume into a surface
and throws the interior away; a ``slices`` layer keeps the interior but shows it
as three flat plates. This draws the tissue itself. Loading a body and cutting
through one are behaviours of this layer, not of the slice layer: a slice plane
photographs the data at one coordinate, a cut here removes part of the body so
you can look into what is left.

THE BODY is always rendered from its own scalars, classified AFTER
interpolation: the mapper samples the raw values trilinearly along each ray and
only then looks them up in a colour and an opacity transfer function, with
gradient shading from the scene's lighting rig. That order is what makes it look
like tissue rather than voxels. Pre-baking colours per voxel and letting VTK
sample them with its default nearest-neighbour lookup -- what this module used
to do -- is what made a template render as a flat, blocky blob.

OVERLAYS only ever change the body's COLOUR. Its opacity, shape and lighting
are its own in every mode, so a map can never make the brain look see-through
or soft; the body's own opacity is what does that. How a map colours it is set
per overlay, by ``xray_mm``:

``0``        paint the map onto whatever tissue the rays reach. The body's
             scalars still decide opacity and shading; only its colour
             changes, on the cortex and on every cut face alike. This is the
             MRIcroGL default, and the one a figure usually wants. With every
             overlay at 0 the layer bakes colours per voxel (``surface``).

``> 0``      x-ray the map through the brain. The ray caster composites the
             brain exactly as it does on its own, and gathers the maps along
             the rest of the ray in a second accumulator, each sample weighted
             by how far behind the brain's surface it lies,
             ``exp(-depth / xray_mm)``. What that accumulator gathers TINTS the
             brain's surface: the lit brain's diffuse light is recoloured by
             it, so gyri, shading and cut faces stay exactly as crisp as they
             are with no overlay, and a deep blob shows as colour on the cortex
             above it, fainter the deeper it is. Rays keep marching past the
             opaque surface for as long as a map could still show. Any x-rayed
             overlay puts the layer in this mode (``volume``); overlays at 0 in
             it are gathered over a sliver under the surface, i.e. painted.

Both colour through the shared overlay suite -- ``overlay_colours``,
``scale_for``, ``composite`` and ``_select`` -- so sign, threshold, max value,
scale, palette (matplotlib or MRIcroGL .clut), opacity and stacking order mean
exactly what they mean on a mesh, and a colour bar captions the same numbers.
In ``volume`` mode the suite is evaluated on a sweep of values to build each
overlay's transfer functions, rather than per voxel, which is what lets the
mapper interpolate the overlay's raw values before colouring them too.

CUTS are clipping planes with a clipped-voxel intensity, not mapper cropping.
Cropping only stops the rays, so the face of a cut through white matter is shaded
by the noise in the data and sparkles; a clipped intensity gives the cut a real
normal, so it reads as a solid, flat face.

BODIES come from a NIfTI or from a prepared backdrop (``.npz``, the per-slice
uint8 archive the backdrop converter writes). A backdrop stays 8-bit all the way
to the GPU, so anything the converter can read becomes a body at a quarter of
the memory of a float volume. A body too large to ray-cast whole -- a 100 um
template is billions of voxels -- is drawn from LEVELS: block-averaged copies,
built once from the slices and kept beside the archive as ``.lodN.npy``. The
coarse level is what rotating, cutting and editing use; hi-res and export use
the finest level that fits ``EXPORT_VOXELS``. A body that fits is always drawn
at its own resolution.

Nothing here touches a plotter. ``build_actor`` returns a ready VTK volume and
the values behind each overlay's scale; ``session`` adds the actor, so lights,
background and camera stay scene-wide.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

__all__ = [
    "OPACITY_PROFILES", "LIBRARY", "library",
    "resolve_source", "source_shape", "levels", "level_for", "load_grid",
    "body_window", "build_actor", "PREVIEW_VOXELS", "EXPORT_VOXELS",
    "native_values", "pinned", "body_mask", "body_weight", "tissue_bounds",
    "atlas_labels",
]

# Closed set of transfer-function shapes, so a spec round-trips.
#   linear      opacity rises with value: density reads as density.
#   sigmoid     a soft threshold -- the low end vanishes, the high end solidifies.
#   geom        emphasises the top of the window; the long tail stays faint.
#   foreground  flat opacity above the window's floor: a solid-looking body.
OPACITY_PROFILES = ("linear", "sigmoid", "geom", "foreground")
_PROFILE_CURVES = {
    "linear": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0],
    "sigmoid": [0.0, 0.01, 0.04, 0.12, 0.35, 0.68, 0.9, 1.0],
    "geom": [0.0, 0.0, 0.01, 0.03, 0.08, 0.2, 0.5, 1.0],
    "foreground": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

# Pre-packaged bodies, by name. spm152 is the 0.74 mm MNI152 template MRIcroGL
# ships as its standard, so a figure made here and one made there start from
# the same anatomy.
_HERE = Path(__file__).resolve().parent / "volumes"
LIBRARY = OrderedDict([
    ("spm152", _HERE / "spm152.nii.gz"),
])

# Ray step as a fraction of the finest voxel. A quarter is where shading on a
# template stops improving; coarser steps band visibly on curved surfaces.
_STEP = 0.25
# Independent components the GPU mapper will shade: the body plus three.
_MAX_RAYCAST_OVERLAYS = 3
# An overlay's opacity accrues over this many millimetres of ray. A third of a
# millimetre makes a blob's edge an opaque, sharply lit surface rather than a
# glow that builds up over a centimetre of tissue, which read as fuzz.
_OVERLAY_UNIT_MM = 0.3
# A body is lit by the scene's rig, scaled here: less ambient fill and a little
# more diffuse. Gradient shading on a smooth intensity field washes out under
# the ambient level that suits a mesh, and the brain read as flat pale grey.
_BODY_SHADE = (0.55, 1.1, 1.0)

_resampled = OrderedDict()      # (source, grid) -> overlay on that grid
_RESAMPLE_CACHE = 6
# Resampled overlays on a hi-res level are half a gigabyte each, so the cache
# is held to a byte budget as well as a count.
_RESAMPLE_BYTES = 2 << 30

# Voxel budgets for the two levels of a body. Rotating, cutting and editing
# draw the coarse level, so each frame is cheap; hi-res and export draw the fine
# one. A body under the preview budget is drawn whole at both. Overridable from
# the environment, because what a GPU can hold is the machine's business.
PREVIEW_VOXELS = int(os.environ.get("GNV_PREVIEW_VOXELS", 24_000_000))
EXPORT_VOXELS = int(os.environ.get("GNV_EXPORT_VOXELS", 160_000_000))

_shapes = {}                    # (path, mtime) -> (shape, mm)
import threading as _threading
_LEVEL_LOCK = _threading.RLock()


def _backdrops():
    """``{name: path}`` for every prepared backdrop, shipped and this user's."""
    from gnv.paths import prepared_backdrops

    return OrderedDict(prepared_backdrops())


def library():
    """Every body the picker can offer: the packaged NIfTIs, then every
    prepared backdrop. Each entry says how it will be drawn, so the panel can
    say so: ``mm`` is its own voxel size, ``preview_mm`` and ``export_mm`` the
    levels it is drawn at while working and for hi-res / export."""
    from gnv.paths import index

    out = []
    entries = [(name, path, "packaged", "Packaged volumes", f"{name} (packaged)", "") for name, path in LIBRARY.items()]
    entries += [(e["name"], Path(e["file"]), "backdrop", s["label"], e.get("label", e["name"]), e.get("hint", "")) for s in index("backdrops") for e in s["entries"] if e["name"] not in LIBRARY]
    for name, path, kind, section, label, hint in entries:
        if not Path(path).is_file():
            continue
        item = {"name": name, "path": str(path), "kind": kind, "section": section, "label": label, "hint": hint}
        try:
            shape, mm = source_shape(path)
            coarse, fine = levels(shape)
            item.update(shape=list(shape), mm=mm, preview_mm=mm * coarse,
                        export_mm=mm * fine,
                        levels_ready=levels_ready(path))
        except Exception as exc:          # an unreadable archive is listed, not fatal
            item["label"] = f"{name} (unreadable: {exc})"
        out.append(item)
    return out


def resolve_source(source):
    """A packaged name, a backdrop name, or a path, to a path."""
    text = str(source or "").strip()
    if text in LIBRARY:
        return str(LIBRARY[text])
    if text and "/" not in text and not text.endswith((".nii", ".gz", ".npz")):
        found = _backdrops().get(text)
        if found is not None:
            return str(found)
    return str(Path(text).expanduser())


def _is_backdrop(path):
    return str(path).lower().endswith(".npz")


def source_shape(source):
    """``(shape, mm)`` of a body without reading its voxels."""
    path = str(Path(str(source)).expanduser())
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        stamp = 0.0
    key = (path, stamp)
    if key not in _shapes:
        if _is_backdrop(path):
            import json

            with np.load(path) as store:
                meta = json.loads(bytes(store["meta"]).decode())
            shape = tuple(int(n) for n in meta["shape"])
            mm = float(min(meta.get("zooms") or [meta["mm"]]))
        else:
            import nibabel as nib

            header = nib.load(path).header
            shape = tuple(int(n) for n in header.get_data_shape()[:3])
            mm = float(min(header.get_zooms()[:3]))
        _shapes[key] = (shape, mm)
    return _shapes[key]


def _factor(voxels, budget):
    factor = 1
    while voxels / float(factor) ** 3 > budget:
        factor += 1
    return factor


def levels(shape):
    """``(coarse, fine)`` block factors for a body of ``shape``.

    Fine is the smallest factor under ``EXPORT_VOXELS``; coarse the smallest
    MULTIPLE of fine under ``PREVIEW_VOXELS``, so the coarse level is built from
    the fine one rather than from the source a second time.
    """
    voxels = float(np.prod([int(n) for n in shape]))
    fine = _factor(voxels, EXPORT_VOXELS)
    coarse = _factor(voxels, PREVIEW_VOXELS)
    coarse = fine * -(-coarse // fine)
    return coarse, fine


def levels_ready(source):
    """Whether every level a body needs already exists and is current.
    A body drawn whole needs none; a NIfTI is downsampled in memory."""
    path = str(source)
    if not _is_backdrop(path):
        return True
    shape = source_shape(path)[0]
    return all(_read_sidecar_ok(path, f, shape) for f in set(levels(shape))
               if f > 1)


def _read_sidecar_ok(path, factor, shape):
    """``_read_sidecar`` without reading the voxels: header and mtime only."""
    side = _sidecar(path, factor)
    try:
        if side.stat().st_mtime < Path(path).stat().st_mtime:
            return False
        with open(side, "rb") as handle:
            version = np.lib.format.read_magic(handle)
            read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                    else np.lib.format.read_array_header_2_0)
            got, _, dtype = read(handle)
    except (OSError, ValueError):
        return False
    expected = tuple(-(-int(n) // factor) for n in shape)
    return dtype == np.uint8 and tuple(got) == expected


def level_for(source, quality):
    """The block factor a body is drawn at for a render ``quality``."""
    coarse, fine = levels(source_shape(source)[0])
    return fine if quality == "export" else coarse


def opacity_curve(profile):
    return list(_PROFILE_CURVES.get(str(profile or "linear"),
                                    _PROFILE_CURVES["linear"]))


# -- geometry ---------------------------------------------------------------
def load_grid(source, factor=1, smooth_vox=0.0, labels=None):
    """A body as a world-millimetre ``pv.ImageData`` with ``values`` scalars.

    ``factor`` > 1 draws a block-averaged level (see ``levels``). A backdrop's
    values stay uint8; a NIfTI's are float32.

    The ray-cast mapper needs an axis-aligned grid, so a volume whose affine
    permutes or flips axes is relabelled to the matching axis order -- lossless,
    no resampling -- which keeps the body in the same world millimetres as every
    mesh and slice in the scene, and lets ``x = -3`` mean one plane everywhere.

    ``smooth_vox`` is a Gaussian on the body, in the SOURCE's voxels. A
    template needs none; a label or segmentation map -- a CAT12 p0, a FreeSurfer
    aseg -- is a staircase of classes whose surface renders as crinkled
    partial-volume noise, and a voxel of smoothing turns it back into cortex.

    ``labels`` draws only those regions of a label atlas -- the left dentate
    of SUIT, say -- as a 0/1 body, chosen before anything is averaged so a
    level's edge is a true partial volume. Needs the atlas NIfTI itself: a
    prepared backdrop has already blended neighbouring labels together.
    """
    factor = max(int(factor or 1), 1)
    labels = [int(v) for v in (labels or [])]
    if labels and _is_backdrop(source):
        raise ValueError("atlas regions need the label NIfTI itself, not a "
                         "prepared body (its labels are averaged together)")
    if _is_backdrop(source):
        data, affine = _backdrop_level(source, factor)
    else:
        from calvin_utils.plotting_utils.mesh_paint import _volume

        data, affine = _volume(source)
        data = np.asarray(data, np.float32)
        if labels:
            data = np.isin(np.rint(data), labels).astype(np.float32)
            if not data.any():
                raise ValueError(f"no voxels carry label(s) {labels}")
        if factor > 1:
            data, affine = _block_mean(data, affine, factor, np.float32)
    sigma = float(smooth_vox or 0.0) / factor
    if sigma >= 0.2:
        from scipy.ndimage import gaussian_filter

        smoothed = gaussian_filter(np.asarray(data, np.float32), sigma)
        data = (np.clip(np.rint(smoothed), 0, 255).astype(np.uint8)
                if data.dtype == np.uint8 else smoothed)
    return _grid_from(data, affine)


# Rotation below this (about 0.06 degrees) is treated as none: template
# affines carry float noise, and an exactly axis-aligned grid keeps the fast,
# separable overlay resampling.
_OBLIQUE_TOL = 1e-3


def _grid_from(data, affine):
    """``data`` on ``affine`` as an ``ImageData`` in world millimetres, lossless.

    The voxel axes are first permuted and flipped to the nearest RAS order --
    a relabelling, no resampling -- so ``x = -3`` means one plane in every
    layer. Whatever rotation is left is an OBLIQUE acquisition: a native-space
    scan whose voxel grid is tilted in the scanner's world. It is kept as the
    grid's direction matrix, which the ray caster honours, so a patient's own
    T1 is drawn upright in world space at its own resolution, and anything
    registered to it -- same grid or not -- lands where it belongs. (Dropping
    the rotation, as this used to, drew the brain tilted and squashed while
    its overlays, resampled through the true affine, sat somewhere else.)
    """
    import pyvista as pv
    from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation

    affine = np.asarray(affine, float)
    ornt = io_orientation(affine)
    affine = affine @ inv_ornt_aff(ornt, data.shape[:3])
    data = apply_orientation(data, ornt)
    axes = affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    direction = axes / spacing
    grid = pv.ImageData(dimensions=tuple(int(n) for n in data.shape),
                        spacing=tuple(float(s) for s in spacing),
                        origin=tuple(float(c) for c in affine[:3, 3]))
    if np.abs(direction - np.eye(3)).max() > _OBLIQUE_TOL:
        grid.direction_matrix = direction
    dtype = np.uint8 if data.dtype == np.uint8 else np.float32
    grid.point_data["values"] = np.ascontiguousarray(
        data, dtype=dtype).ravel(order="F")
    return grid


def _direction(grid):
    """The grid's voxel-to-world rotation (identity when axis-aligned)."""
    return np.asarray(grid.direction_matrix, float)


# -- levels -----------------------------------------------------------------
def _level_affine(affine, factor):
    """The affine of a level: each voxel is the centre of an f-cube of source
    voxels, so the step grows by f and the origin moves half a block in."""
    affine = np.asarray(affine, float).copy()
    linear = affine[:3, :3].copy()
    affine[:3, 3] = affine[:3, 3] + linear @ np.full(3, (factor - 1) / 2.0)
    affine[:3, :3] = linear * float(factor)
    return affine


def _reduce(array, factor, axis):
    """Block mean along one axis; a short last block is averaged over what it has."""
    starts = np.arange(0, array.shape[axis], factor)
    sums = np.add.reduceat(array, starts, axis=axis, dtype=np.float32)
    counts = np.diff(np.append(starts, array.shape[axis])).astype(np.float32)
    shape = [1] * array.ndim
    shape[axis] = -1
    return sums / counts.reshape(shape)


def _block_mean(data, affine, factor, dtype):
    out = data
    for axis in range(3):
        out = _reduce(out, factor, axis)
    if dtype == np.uint8:
        out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    else:
        out = out.astype(dtype)
    return out, _level_affine(affine, factor)


def _sidecar(path, factor):
    path = Path(path)
    return path.with_name(f"{path.stem}.lod{int(factor)}.npy")


def _read_sidecar(path, factor, shape):
    side = _sidecar(path, factor)
    try:
        if side.stat().st_mtime < Path(path).stat().st_mtime:
            return None                     # the archive was rebuilt since
        data = np.load(side)
    except (OSError, ValueError):
        return None
    expected = tuple(-(-int(n) // factor) for n in shape)
    if data.dtype != np.uint8 or tuple(data.shape) != expected:
        return None
    return data


def _write_sidecar(path, factor, data):
    """Keep a level beside its archive; a read-only folder just means it is
    rebuilt next session."""
    import threading

    side = _sidecar(path, factor)
    temp = side.with_name(
        f"{side.name}.{os.getpid()}-{threading.get_ident()}.partial")
    try:
        with open(temp, "wb") as handle:
            np.save(handle, data, allow_pickle=False)
        os.replace(temp, side)
    except OSError:
        try:
            temp.unlink()
        except OSError:
            pass


def _backdrop_level(path, factor, progress=None):
    """Serialised: the conversion job and a render can ask for the same level
    at once, and building it twice wastes a minute of a 100 um read."""
    with _LEVEL_LOCK:
        return _backdrop_level_unlocked(path, factor, progress)


def _backdrop_level_unlocked(path, factor, progress=None):
    """``(uint8 data, affine)`` of a backdrop at block ``factor``.

    Factor 1 is the archive itself, read slice by slice. A coarser level is
    read from its ``.lodN.npy`` if one is current; otherwise it is built from
    the finest level it can be derived from -- itself built from the slices
    if need be -- and kept. Building streams the archive one block of slices at
    a time, so nothing larger than the level being built is ever resident: a
    100 um template is read once, never held.
    """
    from calvin_utils.plotting_utils.backdrops import Backdrop

    path = str(Path(path).expanduser())
    with Backdrop(path) as store:
        shape = tuple(int(n) for n in store.shape)
        affine = np.asarray(store.affine, float)
        if factor == 1:
            if float(np.prod(shape)) > EXPORT_VOXELS * 1.5:
                raise MemoryError(
                    f"{Path(path).name} is {np.prod(shape) / 1e6:,.0f} M voxels, "
                    f"over the {EXPORT_VOXELS / 1e6:,.0f} M a body is drawn "
                    "whole at; draw it from a level")
            data = np.empty(shape, dtype=np.uint8)
            for i in range(shape[0]):
                data[i] = store.slice(0, i)
            return data, affine

        cached = _read_sidecar(path, factor, shape)
        if cached is not None:
            return cached, _level_affine(affine, factor)

        # Derive from the coarsest existing (or required) finer level that
        # divides this one, so a preview level costs nothing once the fine one
        # exists, and the slices are only ever streamed once.
        _, fine = levels(shape)
        base = None
        for divisor in sorted((d for d in range(2, factor) if factor % d == 0),
                              reverse=True):
            found = _read_sidecar(path, divisor, shape)
            if found is not None:
                base = divisor, found
                break
        if base is None and fine > 1 and fine < factor and factor % fine == 0:
            base = fine, _backdrop_level_unlocked(path, fine, progress)[0]
        if base is not None:
            divisor, finer = base
            data = finer
            step = factor // divisor
            for axis in range(3):
                data = _reduce(data, step, axis)
            data = np.clip(np.rint(data), 0, 255).astype(np.uint8)
        else:
            data = np.empty(tuple(-(-n // factor) for n in shape), np.uint8)
            for block, begin in enumerate(range(0, shape[0], factor)):
                stop = min(begin + factor, shape[0])
                total = None
                for i in range(begin, stop):
                    plane = _reduce(_reduce(store.slice(0, i), factor, 0),
                                    factor, 1)
                    total = plane if total is None else total + plane
                data[block] = np.clip(np.rint(total / (stop - begin)),
                                      0, 255).astype(np.uint8)
                if progress:
                    progress(stop / shape[0])
    _write_sidecar(path, factor, data)
    return data, _level_affine(affine, factor)


def build_levels(source, progress=None):
    """Build and keep both levels of a backdrop now, so its first use is fast.
    Returns ``{factor: path}`` for the levels that exist afterwards."""
    path = resolve_source(source)
    coarse, fine = levels(source_shape(path)[0])
    out = {}
    for factor in sorted({fine, coarse}):
        if factor > 1:
            _backdrop_level(path, factor, progress)
            out[factor] = str(_sidecar(path, factor))
    return out


_windows = OrderedDict()        # (buffer, size, dtype, clim) -> window


def _buffer_key(values):
    values = np.asarray(values)
    return (values.__array_interface__["data"][0], values.size, values.dtype.str)


def body_window(values, clim=None):
    """The body's ``(start, saturate)`` window.

    Given, it is used as is. Otherwise it is read off the nonzero voxels (see
    ``_auto_window``): the lower three-class Otsu split to the 99th
    percentile. Remembered per array, because every restyle of a body asks
    again.
    """
    if clim and len(clim) == 2 and clim[0] is not None and clim[1] is not None \
            and float(clim[0]) != float(clim[1]):
        lo, hi = float(clim[0]), float(clim[1])
        return (lo, hi) if lo < hi else (hi, lo)
    values = np.asarray(values)
    key = _buffer_key(values)
    if key in _windows:
        return _windows[key]
    window = _auto_window(values)
    _windows[key] = window
    while len(_windows) > 16:
        _windows.popitem(last=False)
    return window


def _auto_window(values):
    """``(start, saturate)`` read off the data.

    The start is the Otsu split of the nonzero voxels -- for a T1, the valley
    between CSF / background and tissue -- so grey matter is above it and the
    cortex is the surface the rays stop at, with sulci and grey/white contrast
    as its texture. (A percentile start, as before, landed inside the grey
    matter peak and eroded the cortex down to white matter.) The end is the
    99th percentile, so a few bright voxels cannot compress the ramp.
    """
    if values.dtype == np.uint8:
        # Exact, from the histogram: a hi-res level is a hundred million
        # voxels, and sorting them would cost seconds and gigabytes.
        counts = np.bincount(values.ravel(), minlength=256)[1:].astype(float)
        levels = np.arange(1, 256, dtype=float)
    else:
        if values.size > 16_000_000:
            values = values.ravel()[::int(np.ceil(values.size / 8_000_000))]
        values = np.asarray(values, float)
        nonzero = values[np.isfinite(values) & (values != 0)]
        if not nonzero.size:
            return 0.0, 1.0
        counts, edges = np.histogram(nonzero, bins=256)
        counts = counts.astype(float)
        levels = (edges[:-1] + edges[1:]) / 2.0
    total = counts.sum()
    if total <= 0:
        top = float(np.max(values)) if np.size(values) else 1.0
        return 0.0, (top if top > 0 else 1.0)
    cdf = np.cumsum(counts) / total
    hi = float(levels[min(np.searchsorted(cdf, 0.99), len(levels) - 1)])
    lo = _lower_otsu(counts, levels)
    if not hi > lo:
        lo = float(levels[min(np.searchsorted(cdf, 0.05), len(levels) - 1)])
    if not hi > lo:
        return 0.0, (hi if hi > 0 else 1.0)
    return lo, hi


def _lower_otsu(counts, levels):
    """The lower of the two three-class Otsu thresholds of a histogram.

    Three classes because a brain is three: dark (background and CSF), grey
    matter, white matter. The lower split is the CSF / grey boundary whether or
    not the scan was skull-stripped; a two-class split of a stripped brain
    falls between grey and white instead, and erodes the cortex.
    """
    counts = np.asarray(counts, float)
    levels = np.asarray(levels, float)
    if len(counts) > 128:                   # coarsen: the search is quadratic
        step = int(np.ceil(len(counts) / 128))
        n = len(counts) // step * step
        counts = counts[:n].reshape(-1, step).sum(1)
        levels = levels[:n].reshape(-1, step).mean(1)
    p = counts / counts.sum()
    w = np.concatenate([[0.0], np.cumsum(p)])
    m = np.concatenate([[0.0], np.cumsum(p * levels)])
    total = m[-1]
    n = len(p)
    i, j = np.triu_indices(n + 1, k=1)          # class edges 0 < i < j < n
    keep = (i > 0) & (j < n)
    i, j = i[keep], j[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        w0, w1, w2 = w[i], w[j] - w[i], 1.0 - w[j]
        m0, m1, m2 = m[i], m[j] - m[i], total - m[j]
        score = m0 ** 2 / w0 + m1 ** 2 / w1 + m2 ** 2 / w2
    score[~np.isfinite(score)] = -np.inf
    if not np.isfinite(score).any():
        return float(levels[0])
    best = int(np.argmax(score))
    return float((levels[i[best] - 1] + levels[i[best]]) / 2.0)


def _stamp(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _grid_key(grid):
    return (tuple(grid.dimensions), tuple(np.round(grid.spacing, 9)),
            tuple(np.round(grid.origin, 9)),
            tuple(np.round(_direction(grid), 9).ravel()),
            _buffer_key(grid.point_data["values"]))


def _remember(cache, key, value, limit, budget=None):
    cache[key] = value
    while len(cache) > 1 and (
            len(cache) > limit
            or (budget and sum(_nbytes(v) for v in cache.values()) > budget)):
        cache.popitem(last=False)
    return value


def _nbytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (tuple, list)):
        return sum(_nbytes(v) for v in value)
    return int(getattr(value, "nbytes", 0) or 0)


# -- the body's extent -------------------------------------------------------
_masks = OrderedDict()          # body buffer -> inside-the-body mask


def body_mask(grid):
    """Where the body IS, as a flat bool array in the grid's point order: the
    half-way level of ``body_weight``."""
    return body_weight(grid) >= 0.5


def body_weight(grid):
    """How far inside the body each point is, 0..1, as flat float32.

    An overlay is a finding ABOUT the body, so by default it is drawn only
    inside it -- a map that runs past the brain (a full-field t-map, a lesion
    map with skull in it) would otherwise x-ray as colour floating in the air.
    Tissue is everything at or above the start of the body's own automatic
    window (where its opacity begins), closed over sulci, with enclosed
    spaces (ventricles) filled and one voxel of margin so partial-volume
    cortex keeps its colour. It is read from the AUTOMATIC window, not a
    user-set one, so the extent is a property of the body and restyling the
    body's contrast never moves it.

    The envelope is blurred over about a millimetre, and ``body_mask`` is its
    half-way level, so the mask's edge is smooth rather than a coarse voxel
    staircase. Overlays are cut at that edge with their values untouched:
    scaling them down across it left a shell of low values -- the scale's
    bottom colour -- around the brain. Built on a coarse copy of a large level
    and interpolated back, so a hi-res level costs about as little as a
    preview.
    """
    from scipy import ndimage

    values = np.asarray(grid.point_data["values"])
    key = _buffer_key(values)
    if key in _masks:
        _masks.move_to_end(key)
        return _masks[key]

    shape = tuple(int(n) for n in grid.dimensions)
    volume = values.reshape(shape, order="F")
    lo, _ = _auto_window(values)
    factor = 1
    while np.prod(shape) / factor ** 3 > 16_000_000:
        factor += 1
    coarse = volume[::factor, ::factor, ::factor] >= lo
    # Close the gaps between gyri a few millimetres wide, then fill what the
    # envelope encloses. Padding keeps the closing from welding to the edge.
    voxel_mm = float(min(grid.spacing)) * factor
    reach = max(1, int(round(3.0 / voxel_mm)))
    ball = ndimage.generate_binary_structure(3, 1)
    padded = np.pad(coarse, reach + 1)
    padded = ndimage.binary_closing(padded, ball, iterations=reach)
    padded = ndimage.binary_fill_holes(padded)
    padded = ndimage.binary_dilation(padded, ball, iterations=1)
    inside = padded[(slice(reach + 1, -(reach + 1)),) * 3].astype(np.float32)
    weight = ndimage.gaussian_filter(inside, max(1.0 / voxel_mm, 0.5))
    if factor > 1:
        for axis in range(3):
            coords = np.minimum(np.arange(shape[axis]) / factor,
                                weight.shape[axis] - 1)
            weight = _interp_axis(weight, coords, axis)
    weight = np.ascontiguousarray(weight, dtype=np.float32).ravel(order="F")
    return _remember(_masks, key, weight, 2)


_label_lists = {}               # (atlas, lut, mtimes) -> atlas_labels() result
_LUT_SUFFIXES = (".lut", ".txt", ".tsv", ".csv")


def atlas_labels(source, lut=None):
    """The regions of a label atlas: ``{"labels": [{"value", "name"}], "table"}``.

    A volume counts as an atlas when every voxel is a whole number and there
    are between 2 and 2000 distinct nonzero ones; otherwise ``labels`` is
    empty. Names come from a lookup table (``_lut_names``): ``lut`` when one is
    given, else one found beside the atlas. ``table`` is the file the names
    came from, or None, so the panel can say which -- or how to add one.
    """
    empty = {"labels": [], "table": None}
    path = str(Path(resolve_source(source)).expanduser())
    if _is_backdrop(path) or not os.path.isfile(path):
        return empty
    lut = str(Path(lut).expanduser()) if lut else None
    # The folder's mtime too: a table added after the atlas renames its regions.
    key = (path, _stamp(path), _stamp(str(Path(path).parent)), lut,
           _stamp(lut) if lut else 0.0)
    if key in _label_lists:
        return _label_lists[key]
    import nibabel as nib

    out = empty
    try:
        data = np.asarray(nib.load(path).dataobj)
        if data.ndim > 3:
            data = data[..., 0]
        sample = data.ravel()[::max(1, data.size // 2_000_000)]
        if np.all(np.isfinite(sample)) and np.all(sample == np.rint(sample)):
            values = np.unique(np.rint(data[np.isfinite(data)]).astype(np.int64))
            values = values[values != 0]
            if 2 <= values.size <= 2000:
                names, table = _lut_names(path, lut)
                out = {"labels": [{"value": int(v), "name": names.get(int(v), "")}
                                  for v in values],
                       "table": table}
    except Exception:
        out = empty
    _label_lists[key] = out
    return out


def _bare(name):
    """A file name without its NIfTI or table extension, lower case."""
    low = name.lower()
    for suffix in (".nii.gz", ".nii") + _LUT_SUFFIXES:
        if low.endswith(suffix):
            return low[: -len(suffix)]
    return low


def _table_matches(table, atlas):
    """Whether lookup table ``table`` names atlas ``atlas`` (both file names).

    Same base name, or one is the other plus a separator and more: SUIT's
    ``atl-Anatom.lut`` for ``atl-Anatom_space-MNI_dseg.nii``, FreeSurfer-style
    ``aseg_LUT.txt`` for ``aseg.nii.gz``. A short or partial prefix -- ``b.txt``
    for ``b_atlas.nii``, ``atl.lut`` for ``atlas.nii`` -- does not count.
    """
    t, a = _bare(table), _bare(atlas)
    if len(t) < 3 or len(a) < 3:
        return t == a
    if t == a:
        return True
    long_, short = (a, t) if len(a) > len(t) else (t, a)
    return long_.startswith(short) and long_[len(short)] in "_-. "


def _parse_table(text):
    """``{id: name}`` from a lookup table's text.

    One region per line: an integer id first, then the name -- the first run of
    words that are not numbers -- wherever it sits. That reads SUIT's
    ``id r g b name``, FreeSurfer's ``id name r g b a``, a plain ``id name``,
    names with spaces (``1 0.1 0.2 0.3 Left Crus I``), and comma or tab
    separated files; a header line (no integer first) is skipped.
    """
    def number(token):
        try:
            float(token)
            return True
        except ValueError:
            return False

    names = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].replace(",", " ").replace("\t", " ").replace('"', " ")
        parts = line.split()
        if len(parts) < 2 or not parts[0].lstrip("-").isdigit():
            continue
        words, started = [], False
        for token in parts[1:]:
            if number(token):
                if started:
                    break
                continue
            words.append(token)
            started = True
        if words:
            names.setdefault(int(parts[0]), " ".join(words))
    return names


def _lut_names(path, lut=None):
    """``({id: name}, table path or None)`` for an atlas.

    ``lut`` given: that file, wherever it is. Otherwise the tables beside the
    atlas (``.lut``, ``.txt``, ``.tsv``, ``.csv``) that ``_table_matches`` it,
    the closest name first; the first that yields any names wins.
    """
    candidates = []
    if lut:
        candidates = [Path(lut)]
    else:
        folder = Path(path).parent
        atlas = Path(path).name
        try:
            found = [t for t in folder.iterdir()
                     if t.is_file() and t.suffix.lower() in _LUT_SUFFIXES
                     and not t.name.startswith(".")
                     and _table_matches(t.name, atlas)]
        except OSError:
            found = []
        # Exact base name first, then the longest shared prefix.
        candidates = sorted(found, key=lambda t: (_bare(t.name) != _bare(atlas),
                                                  -len(_bare(t.name)), t.name))
    for table in candidates:
        try:
            names = _parse_table(table.read_text(errors="ignore"))
        except OSError:
            continue
        if names:
            return names, str(table)
    return {}, None


_tissue = OrderedDict()         # grid key -> world bounds of its tissue


def tissue_bounds(grid, margin_mm=3.0):
    """World ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the body's TISSUE.

    The grid's own bounds are the scan's field of view. For a template that is
    the brain plus a little margin, but a native scan's FOV carries air, neck
    and shoulders, and framing on it leaves the brain small and off centre.
    This is the box around every voxel at or above the body's automatic start
    (``body_window``), padded by ``margin_mm``, carried through an oblique
    grid's rotation. Framing only: the voxels and the window are untouched.
    """
    key = _grid_key(grid)
    if key in _tissue:
        return _tissue[key]
    values = np.asarray(grid.point_data["values"])
    lo, _ = body_window(values)
    shape = tuple(int(n) for n in grid.dimensions)
    # Strided so a hi-res level costs a few million points, not a hundred.
    step = max(1, int(np.ceil((np.prod(shape) / 4e6) ** (1.0 / 3.0))))
    solid = values.reshape(shape, order="F")[::step, ::step, ::step] >= lo
    ijk = np.argwhere(solid).astype(float) * step
    if not ijk.size:
        return tuple(grid.bounds)
    # The tissue's own voxels in world space: for an oblique grid, the box of
    # an index-space box's corners would be inflated by the rotation.
    world = (np.asarray(grid.origin, float)
             + (ijk * np.asarray(grid.spacing, float)) @ _direction(grid).T)
    pad = margin_mm + step * float(max(grid.spacing))
    lo3, hi3 = world.min(axis=0) - pad, world.max(axis=0) + pad
    out = (float(lo3[0]), float(hi3[0]), float(lo3[1]), float(hi3[1]),
           float(lo3[2]), float(hi3[2]))
    return _remember(_tissue, key, out, 8)


# -- resampling -------------------------------------------------------------
def _on_grid(grid, nifti, smooth_vox=0.0):
    """An overlay resampled onto the body's grid, trilinearly, as the suite does.

    Trilinear with zero outside the overlay, as ``sample_nifti`` samples. When
    the overlay's voxel axes line up with the body's -- every template and
    every map registered to one -- trilinear is separable, so it is done as
    three 1-D passes, about six times faster than a general affine resample
    on a hi-res level; anything oblique falls back to ``affine_transform``.
    Cached on the file, its mtime and the target grid, so editing a threshold
    or a palette resamples nothing.
    """
    from calvin_utils.plotting_utils.mesh_paint import _volume

    path = str(nifti)
    key = (path, _stamp(path), tuple(grid.dimensions),
           tuple(np.round(grid.spacing, 9)), tuple(np.round(grid.origin, 9)),
           tuple(np.round(_direction(grid), 9).ravel()),
           float(smooth_vox or 0.0))
    if key in _resampled:
        _resampled.move_to_end(key)
        return _resampled[key]

    data, affine = _volume(nifti)
    data = np.asarray(data, np.float32)
    if smooth_vox:
        from scipy.ndimage import gaussian_filter

        data = gaussian_filter(data, float(smooth_vox))
    inverse = np.linalg.inv(np.asarray(affine, float))
    # Body voxel -> world (through the grid's rotation, for an oblique native
    # scan) -> overlay voxel. Overlay and body need only share world space.
    matrix = inverse[:3, :3] @ _direction(grid) @ np.diag(np.asarray(grid.spacing, float))
    offset = inverse[:3, :3] @ np.asarray(grid.origin, float) + inverse[:3, 3]
    out = _separable(data, matrix, offset, tuple(int(n) for n in grid.dimensions))
    if out is None:
        from scipy.ndimage import affine_transform

        out = affine_transform(data, matrix, offset=offset,
                               output_shape=tuple(grid.dimensions), order=1,
                               mode="constant", cval=0.0)
    out = np.asarray(out, np.float32).ravel(order="F")
    return _remember(_resampled, key, out, _RESAMPLE_CACHE, _RESAMPLE_BYTES)


def _separable(data, matrix, offset, shape):
    """``affine_transform(order=1, mode='constant')`` for an axis-aligned
    ``matrix`` (a scaled permutation), as three 1-D passes; None otherwise."""
    matrix = np.asarray(matrix, float)
    scale = np.abs(matrix).max()
    if scale == 0:
        return None
    source_axis = np.argmax(np.abs(matrix), axis=0)     # target j <- source i
    if sorted(source_axis.tolist()) != [0, 1, 2]:
        return None
    off_diagonal = matrix.copy()
    off_diagonal[source_axis, range(3)] = 0.0
    if np.abs(off_diagonal).max() > 1e-6 * scale:
        return None
    out = np.transpose(data, source_axis)          # axis j is source axis p[j]
    for j in np.argsort([shape[j] / max(out.shape[j], 1) for j in range(3)]):
        i = source_axis[j]
        coords = matrix[i, j] * np.arange(shape[j]) + offset[i]
        out = _interp_axis(out, coords, int(j))
    return out


def _interp_axis(array, coords, axis):
    """Linear interpolation of ``array`` at ``coords`` along ``axis``; zero
    outside ``[0, n-1]``, as scipy's 'constant' mode, written in slabs so a
    hi-res pass never holds more than one slab of temporaries."""
    n = array.shape[axis]
    tol = 1e-6
    valid = (coords >= -tol) & (coords <= n - 1 + tol)
    c = np.clip(coords, 0.0, max(n - 1, 0))
    i0 = np.minimum(np.floor(c).astype(np.intp), max(n - 2, 0))
    i1 = np.minimum(i0 + 1, n - 1)
    w = (c - i0).astype(np.float32) * valid
    w0 = (1.0 - (c - i0)).astype(np.float32) * valid
    shape = list(array.shape)
    shape[axis] = len(coords)
    out = np.empty(shape, np.float32)
    # Slab along the largest other axis.
    others = [a for a in range(3) if a != axis]
    along = max(others, key=lambda a: shape[a])
    per = int(np.prod(shape)) // max(shape[along], 1)
    width = max(1, (8 << 20) // max(per, 1))
    view = [1, 1, 1]
    view[axis] = -1
    w, w0 = w.reshape(view), w0.reshape(view)
    for begin in range(0, shape[along], width):
        stop = min(begin + width, shape[along])
        index = [slice(None)] * 3
        index[along] = slice(begin, stop)
        block = array[tuple(index)]
        out[tuple(index)] = (np.take(block, i0, axis=axis) * w0
                             + np.take(block, i1, axis=axis) * w)
    return out


# -- overlay values -----------------------------------------------------------
_natives = OrderedDict()        # (file, mtime, band) -> selected native values


def _native(overlay):
    """The values an overlay's scale is read from: ITS OWN voxels, selected.

    Read off the map in its own space, not off the body's grid, so the colour
    scale -- and the colour bar -- is the same at the preview level and the
    export level, in both overlay modes, whatever smoothing is applied. A
    resampled copy's extremes move with the grid; the map's own do not.
    """
    from calvin_utils.plotting_utils.mesh_paint import _select, _volume

    path = str(overlay["nifti"])
    band = (overlay["sign"], bool(overlay["absolute"]), overlay["threshold"],
            overlay.get("max_value"))
    key = (path, _stamp(path)) + band
    if key in _natives:
        _natives.move_to_end(key)
        return _natives[key]
    data, _ = _volume(overlay["nifti"])
    selected = _select(data, sign=band[0], absolute=band[1], threshold=band[2],
                       max_value=band[3])
    finite = selected[np.isfinite(selected)].astype(np.float32)
    return _remember(_natives, key, finite, 12)


def _signed_range(raw, overlay):
    """``(lo, hi)`` of ``_signed(raw)`` without building it."""
    lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
    if overlay["sign"] == "positive":
        lo, hi = 0.0, max(hi, 0.0)
    elif overlay["sign"] == "negative":
        lo, hi = min(lo, 0.0), 0.0
    if overlay["absolute"]:
        lo, hi = 0.0, max(abs(lo), abs(hi))
    return min(lo, 0.0), max(hi, 0.0)


def _signed(raw, overlay):
    """Apply the suite's sign rule and |v| to the DATA, before any lookup.

    A transfer function sees one number per sample and cannot tell a negative
    from its magnitude, so these two have to happen to the values. Zero is the
    suite's "nothing here", which is what an excluded sign becomes.
    """
    data = np.asarray(raw, np.float32).copy()
    if overlay["sign"] == "positive":
        np.maximum(data, 0.0, out=data)
    elif overlay["sign"] == "negative":
        np.minimum(data, 0.0, out=data)
    if overlay["absolute"]:
        np.abs(data, out=data)
    return data


def _level_mapping(lo, hi):
    """``(step, zero)`` putting ``[lo, hi]`` on uint8 levels with zero landing
    exactly on a level (``value = (level - zero) * step``). Zero is the suite's
    "nothing here", so it must come back as exactly zero after the mapper
    interpolates, not a sliver either side of it."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if hi <= lo:
        hi = lo + 1.0
    step = (hi - lo) / 255.0
    zero = float(np.rint(-lo / step))
    # Rounding zero onto a level can push an end of the range past 0..255;
    # widen the step just enough that both ends still fit.
    candidates = [step]
    if zero < 255.0:
        candidates.append(hi / (255.0 - zero))
    if zero > 0.0:
        candidates.append(-lo / zero)
    return max(candidates), zero


def _to_levels(data):
    """Overlay values as uint8 levels: ``(levels, (step, zero))``."""
    mapping = _level_mapping(float(np.min(data)), float(np.max(data)))
    step, zero = mapping
    levels = np.clip(np.rint(np.asarray(data, np.float32) / step + zero),
                     0, 255).astype(np.uint8)
    return levels, mapping


def _signed_levels(raw, overlay):
    """``_to_levels(_signed(raw))`` in slabs, never holding a float copy of a
    hi-res level: ``(levels, (step, zero), (lo, hi))``."""
    lo, hi = _signed_range(raw, overlay)
    step, zero = _level_mapping(lo, hi)
    out = np.empty(raw.shape, np.uint8)
    width = 8 << 20
    for begin in range(0, raw.size, width):
        chunk = _signed(raw[begin:begin + width], overlay)
        chunk /= step
        chunk += zero
        np.rint(chunk, out=chunk)
        np.clip(chunk, 0, 255, out=chunk)
        out[begin:begin + width] = chunk
    return out, (step, zero), (lo, hi)


def _mask(raw, overlay):
    """Where ``_select`` would keep a value, without building its NaN copy."""
    raw = np.asarray(raw)
    sign = overlay["sign"]
    if sign == "positive":
        keep = raw > 0
    elif sign == "negative":
        keep = raw < 0
    else:
        keep = raw != 0
    keep &= np.isfinite(raw)
    if overlay["threshold"] is not None or overlay.get("max_value") is not None:
        value = np.abs(raw) if overlay["absolute"] else raw
        if overlay["threshold"] is not None:
            keep &= value >= float(overlay["threshold"])
        if overlay.get("max_value") is not None:
            keep &= value <= float(overlay["max_value"])
    return keep


def _pinned(overlay, finite):
    """The overlay with its data-driven choices frozen from ALL its values.

    ``overlay_colours`` resolves the scale from whatever values it is handed.
    A transfer-function sweep or a chunk of voxels is not the whole map, so the
    scale and the diverging/ramp decision are fixed here from the real data
    first. A user-entered scale is left untouched, reversed or not, because a
    reversed pair is how a scale is inverted.
    """
    from calvin_utils.plotting_utils.overlay_paint import scale_for

    out = dict(overlay)
    if finite.size:
        if out.get("symmetric") is None:
            out["symmetric"] = bool((finite > 0).any() and (finite < 0).any())
        if out.get("clim") is None:
            out["clim"] = [float(v) for v in scale_for(finite, overlay)[1]]
    return out


# -- transfer functions -----------------------------------------------------
def body_cmap(layer):
    """The body's colormap: its palette (greys by default, a T1's own
    texture), optionally tinted.

    A tint does not replace the palette, it colours it: each colour keeps its
    brightness and takes the tint's hue, so sulci, grey and white matter still
    read as they do in greys -- in sepia, or blue, or whatever was picked.
    """
    from calvin_utils.plotting_utils.palettes import resolve_palette

    base = resolve_palette(layer.get("cmap") or "gray", fallback_color="#b7bcc4")
    tint = layer.get("color")
    if not tint:
        return base
    from matplotlib.colors import ListedColormap, to_rgb

    hue = np.asarray(to_rgb(tint), float)
    hue = hue / max(float(hue.max()), 1e-6)
    table = np.asarray(base(np.linspace(0.0, 1.0, 256)), float).copy()
    luminance = table[:, :3] @ np.array([0.2126, 0.7152, 0.0722])
    table[:, :3] = np.clip(luminance[:, None] * hue[None, :], 0.0, 1.0)
    return ListedColormap(table, name="cv_body_tinted")


# An atlas region is a 0/1 mask, and its surface is where the mask crosses
# one half. Opacity steps up across a narrow band around it, so the nucleus
# has a crisp edge at any zoom: a wider window ramps opacity across the whole
# smoothed edge and leaves a soft, see-through rim. Its COLOUR comes from a
# fixed, wider window instead, so the samples either side of that band are one
# light tone rather than flickering black and white.
_REGION_WINDOW = (0.47, 0.53)
_REGION_COLOURS = (-0.5, 0.8)     # the surface (0.5) lands at 77% of the palette


def _layer_clim(layer):
    """The body's window as set, or the region window for an atlas region."""
    if layer.get("clim"):
        return layer.get("clim")
    return list(_REGION_WINDOW) if layer.get("body_labels") else None


def _colour_window(layer, lo, hi):
    """Where the body's palette is spread: its window, but for an atlas
    region left on auto, the fixed region colours."""
    if layer.get("body_labels") and not layer.get("clim"):
        return _REGION_COLOURS
    return lo, hi


def _body_tfs(values, layer, scale=1.0):
    """Colour and opacity transfer functions for the body's own scalars."""
    import vtk

    lo, hi = body_window(values, _layer_clim(layer))
    cmap = body_cmap(layer)
    c0, c1 = _colour_window(layer, lo, hi)
    ctf = vtk.vtkColorTransferFunction()
    for t in np.linspace(0.0, 1.0, 64):
        ctf.AddRGBPoint(c0 + t * (c1 - c0), *map(float, cmap(t)[:3]))
    otf = vtk.vtkPiecewiseFunction()
    # Nothing below the window. The step is a few lookup texels wide: a node
    # pair closer than the GPU's lookup texture resolves makes VTK warn and
    # fall back.
    otf.AddPoint(lo - (hi - lo) / 1024.0, 0.0)
    strength = float(layer.get("opacity", 1.0)) * float(scale)
    curve = opacity_curve(layer.get("opacity_profile"))
    for t, a in zip(np.linspace(0.0, 1.0, len(curve)), curve):
        otf.AddPoint(lo + t * (hi - lo), float(a) * strength)
    return ctf, otf, (lo, hi)


def _overlay_tfs(span, finite, overlay, fade, levels8=None):
    """An overlay's transfer functions, built by asking the suite.

    The suite maps a value to ``(rgb, alpha)`` with no reference to where the
    value sits, so evaluating it on a dense sweep of values -- with the
    thresholds and scale ends included exactly -- reproduces it as a pair of
    transfer functions. The mapper then interpolates the overlay's raw values
    and colours them, so a blob's edge is smooth rather than a voxel staircase.

    ``span`` is the ``(lo, hi)`` the data on the grid covers; ``finite`` the
    map's own selected values, which fix the scale (see ``_native``).

    ``fade`` is the fraction of the surviving range over which opacity rises
    from the threshold. A cliff from 0 to 1 is an infinitely sharp isosurface,
    and rays sampling it at fixed steps terrace it into contour lines. Nothing
    below the threshold appears either way; this only shapes how what survives
    comes in.

    ``levels8`` is ``(step, zero)`` when the overlay is handed to the GPU as
    uint8 levels (``value = (level - zero) * step``); the nodes are then placed
    in level units, so the same colours land on the same values.
    """
    import vtk
    from calvin_utils.plotting_utils.mesh_paint import _select
    from calvin_utils.plotting_utils.overlay_paint import overlay_colours

    band = dict(threshold=overlay["threshold"], max_value=overlay.get("max_value"))
    pinned = _pinned(overlay, finite)

    lo, hi = float(span[0]), float(span[1])
    if hi <= lo:
        hi = lo + 1.0
    # The GPU holds a transfer function as a lookup texture sized to its
    # closest pair of nodes, and most drivers cap it at 16384 texels. Nodes
    # closer than that cap can resolve make VTK ask for millions and fall back
    # with a warning, so the step either side of a threshold is a few texels
    # wide and sweep nodes that crowd a key node are dropped.
    eps = (hi - lo) / 4096.0
    keys = [0.0, -(hi - lo) / 400.0, (hi - lo) / 400.0] + [
        k for k in (overlay["threshold"], overlay.get("max_value"))
        if k is not None] + list(pinned.get("clim") or [])
    exact = np.unique(np.concatenate(
        [[lo, hi]] + [[float(k) - eps, float(k), float(k) + eps] for k in keys]))
    sweep = np.linspace(lo, hi, 1024)
    near = np.abs(sweep[:, None] - exact[None, :]).min(axis=1) < eps
    xs = np.unique(np.concatenate([exact, sweep[~near]]))
    xs = xs[(xs >= lo) & (xs <= hi)]
    # A key that is itself a data extreme (a clim read off the data) lands a
    # rounding error from lo or hi; merge anything that close.
    xs = xs[np.concatenate([[True], np.diff(xs) > 0.5 * eps])]
    rgb, alpha = overlay_colours(_select(xs, **band), pinned)

    # Zero is "nothing here". The GPU reads a transfer function through a
    # lookup texture, so a node pair that drops to zero opacity only AT zero
    # still hands a little opacity to samples of exactly zero -- every empty
    # voxel in the grid -- and an overlay drew its whole bounding box as a dark
    # slab. Nothing within a few texels of zero is drawn.
    # VTK may hold the table in as few as 1024 texels, so the band is a few of
    # those wide: values within 1/400 of the overlay's range of zero are not
    # drawn -- on a t-map, under 0.05.
    band_width = (hi - lo) / 400.0
    alpha = np.where(np.abs(xs) <= band_width, 0.0, alpha)
    # Where nothing is drawn the colour still matters: the mapper interpolates
    # colour between neighbouring nodes, and black there darkened every edge
    # of every blob. Carry the nearest drawn colour instead.
    drawn = alpha > 0
    if drawn.any() and not drawn.all():
        nearest = np.searchsorted(xs[drawn], xs).clip(0, drawn.sum() - 1)
        left = np.clip(nearest - 1, 0, drawn.sum() - 1)
        pick = np.where(np.abs(xs[drawn][left] - xs) < np.abs(xs[drawn][nearest] - xs),
                        left, nearest)
        rgb = np.where(drawn[:, None], rgb, rgb[drawn][pick])

    # The fade-in, as a fraction of the COLOUR SCALE, from where the overlay
    # starts (its threshold, or zero). It used to be a fraction of the data's
    # whole range, so one bright voxel stretched it and the bottom of the scale
    # stayed transparent: a bar from -0.5 to 5 coloured nothing below ~2.
    if finite.size and fade:
        edge = float(np.min(np.abs(finite)))
        clim = pinned.get("clim") or [float(np.min(finite)), float(np.max(finite))]
        width = max(abs(float(clim[1]) - float(clim[0])) * float(fade), 1e-12)
        t = np.clip((np.abs(xs) - edge) / width, 0.0, 1.0)
        alpha = alpha * (t * t * (3.0 - 2.0 * t))

    ctf = vtk.vtkColorTransferFunction()
    otf = vtk.vtkPiecewiseFunction()
    at = xs if levels8 is None else xs / levels8[0] + levels8[1]
    for x, c, a in zip(at, rgb, alpha):
        ctf.AddRGBPoint(float(x), *map(float, c))
        otf.AddPoint(float(x), float(a))
    return ctf, otf


# -- actors -----------------------------------------------------------------
def _property(material, shade):
    import vtk

    prop = vtk.vtkVolumeProperty()
    prop.SetInterpolationTypeToLinear()
    prop.SetShade(bool(shade))
    ambient, diffuse, specular = _body_shading(material)
    prop.SetAmbient(ambient)
    prop.SetDiffuse(diffuse)
    prop.SetSpecular(specular)
    prop.SetSpecularPower(float(material.get("specular_power", 14.0)))
    return prop


def _body_shading(material):
    ambient = float(material.get("ambient", 0.30)) * _BODY_SHADE[0]
    diffuse = min(float(material.get("diffuse", 0.78)) * _BODY_SHADE[1], 1.0)
    specular = float(material.get("specular", 0.22)) * _BODY_SHADE[2]
    return ambient, diffuse, specular


def _mapper(grid, array, gpu=False):
    import vtk

    mapper = (vtk.vtkOpenGLGPUVolumeRayCastMapper() if gpu
              else vtk.vtkSmartVolumeMapper())
    mapper.SetInputData(grid)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectScalarArray(array)
    mapper.SetAutoAdjustSampleDistances(False)
    mapper.SetSampleDistance(float(min(grid.spacing)) * _STEP)
    if hasattr(mapper, "SetUseJittering"):
        # Breaks the wood-grain banding a fixed ray step leaves on curved
        # surfaces into noise the eye does not read as structure.
        mapper.SetUseJittering(True)
    return mapper


def _clip(mapper, prop, section, outside):
    """``section`` half-spaces as clipping planes with a solid cut face.
    Replaces whatever planes the mapper had, so a restyle can move a cut."""
    import vtk

    mapper.RemoveAllClippingPlanes()
    planes = 0
    for entry in section or []:
        if entry.get("enabled") is False:
            continue
        axis = "xyz".index(str(entry.get("axis", "x")).lower())
        normal = [0.0, 0.0, 0.0]
        # ">" removes everything above the plane, so the kept side is below it.
        normal[axis] = -1.0 if str(entry.get("op", ">")) == ">" else 1.0
        origin = [0.0, 0.0, 0.0]
        origin[axis] = float(entry.get("at", 0.0))
        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        mapper.AddClippingPlane(plane)
        planes += 1
    if planes:
        prop.SetUseClippedVoxelIntensity(True)
        prop.SetClippedVoxelIntensity(float(outside))


def _new_grid(grid, name, array):
    import pyvista as pv

    out = pv.ImageData(dimensions=grid.dimensions, spacing=grid.spacing,
                       origin=grid.origin)
    out.direction_matrix = _direction(grid)
    out.point_data[name] = array
    return out


def _volume_actor(mapper, prop):
    import vtk

    actor = vtk.vtkVolume()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)
    return actor


# What each overlay mode bakes into the voxels it hands the GPU. Everything
# else on the layer -- opacity, its profile, shading, cuts, x-ray depth, fade,
# and in x-ray mode thresholds and palettes too -- is a transfer function, a
# clipping plane or a shader constant, and changing it restyles the actor that
# is already on screen instead of rebuilding and re-uploading its volume.
_SURFACE_BAKED = ("clim", "cmap", "color")
# Overlay fields that never reach a painted voxel: captions, ids, and settings
# that belong to mesh or outline drawing. Toggling a colour bar must not
# repaint a hundred million voxels.
_NOT_PAINTED = ("id", "name", "bar", "draw", "outline_smooth", "fill_gaps", "smooth_mm", "mode", "stat", "min_value",
                "keep_top", "step", "xray_mm", "fade")
_data = OrderedDict()           # data key -> _Data
_DATA_CACHE = 3
_DATA_BYTES = 3 << 29           # 1.5 GB of GPU-ready voxels


class _Data:
    """The voxels a volume layer hands the GPU, and what styling them needs."""

    def __init__(self, mode, grid, array, outside, **extra):
        self.mode, self.grid, self.array, self.outside = mode, grid, array, outside
        self.__dict__.update(extra)

    @property
    def nbytes(self):
        return int(np.asarray(self.grid.point_data[self.array]).nbytes)


# An overlay that is painted (``xray_mm`` 0) inside an x-rayed layer is gathered
# over this much tissue under the surface: the sliver the surface paint means.
_PAINT_DEPTH_MM = 0.5


def _mode(layer, overlays, outlines=()):
    """``plain``, ``surface`` (every map painted) or ``volume`` (any x-rayed).
    Outlines alone are painted, so they make a layer ``surface`` too."""
    if not overlays:
        return "surface" if outlines else "plain"
    return "volume" if any(_xray_mm(o) > 0 for o in overlays) else "surface"





def _xray_mm(overlay):
    return float(overlay.get("xray_mm") or 0.0)


def _data_key(grid, layer, overlays, mode):
    import json

    if mode == "plain":
        parts = []
    elif mode == "volume":
        parts = [[str(o["nifti"]), _stamp(str(o["nifti"])), o["sign"],
                  bool(o["absolute"])] for o in overlays]
        parts.append(float(layer.get("overlay_smooth_vox") or 0.0))
        parts.append(bool(layer.get("overlay_outside")))
    else:
        parts = [dict({k: v for k, v in o.items() if k not in _NOT_PAINTED
                       and not k.startswith("outline")},
                      _stamp=_stamp(str(o["nifti"]))) for o in overlays]
        parts.append({k: layer.get(k) for k in _SURFACE_BAKED})
        parts.append(bool(layer.get("overlay_outside")))
    return (mode, _grid_key(grid), json.dumps(parts, sort_keys=True, default=str))


def _volume_data(grid, layer, overlays, mode):
    key = _data_key(grid, layer, overlays, mode)
    if key in _data:
        _data.move_to_end(key)
        return _data[key]
    values = np.asarray(grid.point_data["values"])
    outside = float(np.min(values))
    if mode == "plain":
        data = _Data(mode, grid, "values", outside)
    elif mode == "volume":
        data = _raycast_data(grid, values, layer, overlays, outside)
    else:
        data = _painted_data(grid, values, layer, overlays)
    return _remember(_data, key, data, _DATA_CACHE, _DATA_BYTES)


def build_actor(grid, layer, overlays, material, reuse=None, outlines=()):
    """``(vtkVolume, [(values, overlay)], [warning])`` for one volume layer.

    ``reuse`` is the actor this layer is on screen with. When the voxels it
    shows are still the right ones it is restyled in place -- new transfer
    functions, cuts and shader -- and returned, so editing a threshold or an
    opacity never re-uploads a hundred-megabyte level to the GPU.
    """
    warnings = []
    overlays = list(overlays or [])
    outlines = list(outlines or [])
    # Outlines are drawn by the session over the frame; they only decide that
    # a layer with nothing but outlines is still drawn painted.
    mode = _mode(layer, overlays, outlines)
    if mode == "volume" and len(overlays) > _MAX_RAYCAST_OVERLAYS:
        warnings.append(
            f"ray-cast overlays are limited to {_MAX_RAYCAST_OVERLAYS}; "
            f"the last {len(overlays) - _MAX_RAYCAST_OVERLAYS} were skipped")
        overlays = overlays[:_MAX_RAYCAST_OVERLAYS]

    data = _volume_data(grid, layer, overlays, mode)
    actor = reuse if reuse is not None and _shows(reuse, data) else None
    if actor is None:
        actor = _volume_actor(_mapper(data.grid, data.array,
                                      gpu=(mode == "volume")),
                              _property(material, True))
    prop = _style(data, layer, overlays, material)
    actor.SetProperty(prop)
    _clip(actor.GetMapper(), prop, layer.get("section"),
          0.0 if mode == "surface" else data.outside)
    if mode == "volume":
        _install_xray(actor, [_xray_mm(o) or _PAINT_DEPTH_MM for o in overlays],
                      outside=bool(layer.get("overlay_outside")))
    else:
        actor.GetShaderProperty().ClearAllFragmentShaderReplacements()
    scales = [(_native(o), o) for o in overlays]
    return actor, scales, warnings


def _shows(actor, data):
    """Whether ``actor`` already draws exactly ``data``'s voxels."""
    try:
        shown = actor.GetMapper().GetInput()
    except AttributeError:
        return False
    return shown is not None and (
        shown.GetAddressAsString("vtkObject")
        == data.grid.GetAddressAsString("vtkObject"))


def _style(data, layer, overlays, material):
    """A fresh volume property for ``data``: everything that is not voxels."""
    prop = _property(material, layer.get("shade", True))
    unit = float(min(data.grid.spacing))
    if data.mode == "plain":
        ctf, otf, _ = _body_tfs(data.grid.point_data["values"], layer)
        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)
        prop.SetScalarOpacityUnitDistance(unit)
        return prop

    if data.mode == "surface":
        import vtk

        prop.IndependentComponentsOff()
        otf = vtk.vtkPiecewiseFunction()
        start8 = data.start8
        end8 = getattr(data, "end8", 255.0)
        otf.AddPoint(0.0, 0.0)
        if start8 > 0.5:   # a step a few lookup texels wide, not a sliver
            otf.AddPoint(start8 - 0.25, 0.0)
        strength = float(layer.get("opacity", 1.0))
        curve = opacity_curve(layer.get("opacity_profile"))
        for t, a in zip(np.linspace(0.0, 1.0, len(curve)), curve):
            otf.AddPoint(start8 + t * (end8 - start8), float(a) * strength)
        if end8 < 254.5:
            otf.AddPoint(255.0, float(curve[-1]) * strength)
        prop.SetScalarOpacity(otf)
        prop.SetScalarOpacityUnitDistance(unit)
        return prop

    # x-ray: the brain at full strength -- the same transfer functions as the
    # plain render, because the compositing never makes it translucent -- and
    # each overlay's built from its own values on the grid.
    body_ctf, body_otf, _ = _body_tfs(data.body, layer)
    tfs = [(body_ctf, body_otf)]
    for overlay, span, mapping in zip(overlays, data.spans, data.mappings):
        fade = float(overlay.get("fade", 0.03) or 0.0)
        tfs.append(_overlay_tfs(span, _native(overlay), overlay, fade, mapping))
    prop.IndependentComponentsOn()
    for index, (ctf, otf) in enumerate(tfs):
        prop.SetColor(index, ctf)
        prop.SetScalarOpacity(index, otf)
        # The brain per voxel, exactly as when drawn alone; an overlay's per
        # _OVERLAY_UNIT_MM of ray, so it means the same depth of colour on a
        # 0.7 mm template as on a 2 mm one.
        prop.SetScalarOpacityUnitDistance(index, unit if index == 0 else _OVERLAY_UNIT_MM)
        prop.SetComponentWeight(index, 1.0)
    return prop


def _painted_data(grid, values, layer, overlays):
    """``surface``: the body's own opacity and shading, the suite's colours.

    Four dependent components: RGB is the suite's composite of the body's
    colormap and every overlay, voxel by voxel; the fourth channel is the
    body's scalar, passed through the opacity transfer function and used for
    the shading gradient. So the brain keeps exactly the shape and lighting it
    has with no overlay, and the overlay changes nothing but colour -- on the
    cortex and on every cut face. The fourth channel is 8-bit, so it is spread
    over the window and a margin below it, not over the full range, or shading
    on a template quantises into visible grain.

    Only voxels some overlay selects go through the suite -- compositing is
    per voxel -- so the cost follows the size of the blobs, not of the level.
    """
    from calvin_utils.plotting_utils.mesh_paint import _select
    from calvin_utils.plotting_utils.overlay_paint import composite, overlay_colours

    lo, hi = body_window(values, _layer_clim(layer))
    cmap = body_cmap(layer)
    outside = float(np.min(values))
    n = len(values)
    rgba = np.empty((n, 4), dtype=np.uint8)
    # The fourth channel must span 0..255 in the data itself. With dependent
    # components VTK places the opacity function against the channel's own
    # range, so a channel that never reaches 0 -- a head scan whose noisy
    # background sits above the margin's floor -- slides the function down
    # and the whole bounding box turns opaque. Never floor below the data.
    floor = max(lo - (hi - lo), outside)
    # The top of the channel. The window's end, normally: 8 bits spent where
    # the template's shading lives. An atlas region is a 0/1 mask with a
    # narrow window, and a channel that saturates a voxel inside its surface
    # steps the shading gradient into stripes -- so it spans the whole mask.
    ceiling = hi
    if layer.get("body_labels"):
        floor, ceiling = outside, max(hi, float(np.max(values)))
    start8 = (lo - floor) / (ceiling - floor) * 255.0
    end8 = (hi - floor) / (ceiling - floor) * 255.0

    c0, c1 = _colour_window(layer, lo, hi)

    def base_of(v):
        return cmap(np.clip((np.asarray(v, float) - c0) / (c1 - c0), 0.0, 1.0))[:, :3]

    if values.dtype == np.uint8:
        # 256 possible values: colour and opacity are each a table lookup.
        levels8 = np.arange(256, dtype=float)
        base_lut = np.clip(base_of(levels8) * 255.0, 0, 255).astype(np.uint8)
        alpha_lut = (np.clip((levels8 - floor) / (ceiling - floor), 0.0, 1.0)
                     * 255.0).astype(np.uint8)
        np.take(base_lut, values, axis=0, out=rgba[:, :3])
        np.take(alpha_lut, values, out=rgba[:, 3])
    else:
        step = 1 << 20      # the suite works in float64; keep chunks small
        for start in range(0, n, step):
            v = values[start:start + step].astype(float)
            rgba[start:start + step, :3] = np.clip(base_of(v) * 255.0, 0, 255)
            rgba[start:start + step, 3] = (np.clip((v - floor) / (ceiling - floor),
                                                   0.0, 1.0) * 255.0)

    # The same light smoothing the x-ray path applies (``overlay_smooth_vox``,
    # in the overlay's own voxels): a 2 mm map painted onto a finer body is
    # otherwise a lattice of its voxel corners -- trilinear kinks and the hard
    # steps of a mask edge -- printed straight onto the cortex as small dark
    # triangles (measured on a 2 mm map over the 1 mm sym 09c body).
    smooth = float(layer.get("overlay_smooth_vox") or 0.0)
    raws = [_on_grid(grid, o["nifti"], smooth_vox=smooth) for o in overlays]
    active = np.zeros(n, dtype=bool)
    for raw, overlay in zip(raws, overlays):
        active |= _mask(raw, overlay)
    if not layer.get("overlay_outside"):
        active &= body_mask(grid)
    where = np.flatnonzero(active)
    pinned = [_pinned(o, _native(o)) for o in overlays]
    step = 1 << 20
    for start in range(0, where.size, step):
        idx = where[start:start + step]
        if values.dtype == np.uint8:
            base = base_lut[values[idx]].astype(float) / 255.0
        else:
            base = base_of(values[idx])
        stack = []
        for raw, overlay, pin in zip(raws, overlays, pinned):
            sel = _select(raw[idx], sign=overlay["sign"],
                          absolute=overlay["absolute"],
                          threshold=overlay["threshold"],
                          max_value=overlay.get("max_value"))
            stack.append(overlay_colours(sel, pin))
        rgba[idx, :3] = np.clip(composite(base, stack) * 255.0, 0, 255
                                ).astype(np.uint8)
    rgba[0, 3] = 0      # a corner of the box: pins the channel's floor at 0
    return _Data("surface", _new_grid(grid, "rgba", rgba), "rgba", 0.0,
                 start8=start8, end8=end8)


def region_actor(grid, layer, overlays, overlay, material):
    """A body drawn to show only WHERE ``overlay`` is on what the rays reach.

    Same voxels, opacity, cuts and camera as the painted body, but unlit, its
    colour white where the overlay selects and black elsewhere. Rendered off
    screen, it is a mask of exactly the patches the painted body shows in that
    overlay -- on the cortex, in sulci, on cut faces -- and the session traces
    its edge as the overlay's outline (see outlines.py). ``overlays`` are the
    layer's fills, so the painted body's voxels are reused from the cache.
    """
    data = _volume_data(grid, layer, list(overlays), "surface")
    rgba = np.asarray(data.grid.point_data["rgba"])
    from gnv.outlines import colour_floor

    raw = _on_grid(grid, overlay["nifti"],
                   smooth_vox=float(layer.get("overlay_smooth_vox") or 0.0))
    region = _mask(raw, overlay)
    floor = colour_floor(overlay)
    if floor is not None:
        # Only from the colour bar's minimum up: the outline sits there.
        region &= (np.abs(raw) if overlay["absolute"] else raw) >= floor
    if not layer.get("overlay_outside"):
        region &= body_mask(grid)
    if not region.any():
        return None
    marks = np.zeros_like(rgba)
    marks[:, 3] = rgba[:, 3]
    marks[region, :3] = 255
    shown = _Data("surface", _new_grid(data.grid, "rgba", marks), "rgba", 0.0,
                  start8=data.start8, end8=getattr(data, "end8", 255.0))
    mapper = _mapper(shown.grid, "rgba")
    prop = _style(shown, layer, [], material)
    prop.ShadeOff()
    _clip(mapper, prop, layer.get("section"), 0.0)
    return _volume_actor(mapper, prop)


def _raycast_data(grid, values, layer, overlays, outside):
    """``volume``: the brain plus up to three overlays as independent
    components on one grid, drawn with the x-ray compositing installed below.

    Each overlay is lightly smoothed on its own grid first. A statistical map is
    usually masked, and the mask's edge is a voxel staircase that no amount of
    interpolation can round; ``overlay_smooth_vox`` (in the overlay's voxels)
    rounds it. It moves values a little, so 0 turns it off.

    An 8-bit body keeps every component 8-bit: the GPU holds one texel of four
    bytes per voxel instead of sixteen, which is what lets a hi-res level carry
    three overlays at all.
    """
    eight = values.dtype == np.uint8
    smooth = float(layer.get("overlay_smooth_vox") or 0.0)
    stacked = np.empty((values.size, 1 + len(overlays)),
                       np.uint8 if eight else np.float32)
    stacked[:, 0] = values
    spans, mappings = [], []
    inside = None if layer.get("overlay_outside") else body_mask(grid)
    for index, overlay in enumerate(overlays, start=1):
        raw = _on_grid(grid, overlay["nifti"], smooth_vox=smooth)
        if inside is not None:
            # Cut at the body's envelope, values untouched inside it. The
            # envelope is dilated past the tissue and the shader gathers maps
            # only once a ray is in tissue, so this edge is never drawn. Fading
            # values to zero across the edge instead left a shell of low values
            # -- the bottom colour -- around the brain, which the ray met first.
            raw = np.where(inside, raw, 0.0).astype(raw.dtype, copy=False)
        else:
            raw = raw.copy()
        # The transfer functions leave a narrow band around zero undrawn (see
        # _overlay_tfs). A real but tiny value -- 0.0001 in a smoothed map --
        # would fall in it and vanish, though anything nonzero is meant to be
        # painted (below the scale's minimum, in its bottom colour). Lift such
        # values just past the band; exact zeros stay "nothing".
        lo_s, hi_s = _signed_range(raw, overlay)
        band = max(hi_s - lo_s, 1e-12) / 400.0 * 1.5
        tiny = (raw != 0) & (np.abs(raw) < band)
        raw[tiny] = np.copysign(band, raw[tiny])
        if eight:
            stacked[:, index], mapping, span = _signed_levels(raw, overlay)
        else:
            stacked[:, index] = _signed(raw, overlay)
            mapping, span = None, _signed_range(raw, overlay)
        spans.append(span)
        mappings.append(mapping)
    return _Data("volume", _new_grid(grid, "components", stacked), "components",
                 outside, body=values, spans=spans, mappings=mappings)


# -- x-ray compositing ------------------------------------------------------
# Replacements for the GPU ray caster's fragment shader (VTK 9.7 hook names).
# Component 0 is the brain, 1.. are overlays. Sampling, transfer-function
# lookup, lighting and clipping are VTK's own functions, called as VTK calls
# them; only the compositing and the termination test change.
#
# The brain composites front to back exactly as VTK would. Its lighting is
# linear in its colour -- ambient and specular added, diffuse multiplied -- so
# alongside it the two parts are accumulated with the same weights: what the
# brain's light would be in black (ambient + specular) and in white (adding the
# diffuse). At the end the overlays' gathered colour replaces the brain's own
# colour in that sum, by as much as the overlays gathered, so a recoloured
# surface is lit exactly as the brain under it. Opacity is the brain's alone.
_XR_DECL = """//VTK::Base::Dec
float xr_surfaceT;
vec4 xr_overlay;
vec3 xr_dark;
vec3 xr_diffuse;
#define XR_DEPTH_MAX %(depth_max)s
// Maps are gathered only once the ray is in tissue, unless they may show
// outside the body.
#define XR_GATHER (%(gather)s)
"""
_XR_INIT = """//VTK::DepthPeeling::Ray::Init
  xr_surfaceT = -1.0;
  xr_overlay = vec4(0.0);
  xr_dark = vec3(0.0);
  xr_diffuse = vec3(0.0);
"""
_XR_SHADE = """
    if (!g_skip)
      {
      vec4 scalar = texture3D(in_volume[0], g_dataPos);
      scalar = scalar * in_volume_scale[0] + in_volume_bias[0];
      if (g_fragColor.a <= g_opacityThreshold)
        {
        float xr_a = computeOpacity(scalar, 0);
        if (xr_a > 0.0)
          {
          vec3 xr_albedo = texture2D(in_colorTransferFunc_0[0],
                                     vec2(scalar[0], 0.0)).xyz;
          vec3 xr_k = computeLighting(vec4(vec3(0.0), xr_a), 0, 0.0).rgb;
          vec3 xr_d = computeLighting(vec4(vec3(1.0), xr_a), 0, 0.0).rgb - xr_k;
          float xr_w = (1.0 - g_fragColor.a) * xr_a;
          g_fragColor.rgb += xr_w * clamp(xr_k + xr_d * xr_albedo, 0.0, 1.0);
          g_fragColor.a += xr_w;
          xr_dark += xr_w * xr_k;
          xr_diffuse += xr_w * xr_d;
          }
        }
      // Depth into tissue, counted from where the brain turns half opaque.
      if (xr_surfaceT < 0.0 && g_fragColor.a > 0.5) xr_surfaceT = g_currentT;
      float xr_behind = xr_surfaceT < 0.0 ? 0.0
                        : (g_currentT - xr_surfaceT) * in_sampleDistance;
%(overlays)s
      }
"""
# One block per overlay, with its index and depth written in: GLSL 1.50 only
# indexes a sampler array with a constant.
_XR_OVERLAY = """      {
      float xr_oa = computeOpacity(scalar, %(i)d);
      if (xr_oa > 0.0 && XR_GATHER)
        {
        xr_oa *= exp(-xr_behind / %(depth)s);
        vec3 xr_oc = texture2D(in_colorTransferFunc_0[%(i)d],
                               vec2(scalar[%(i)d], 0.0)).xyz;
        xr_overlay += (1.0 - xr_overlay.a) * vec4(xr_oc * xr_oa, xr_oa);
        }
      }
"""
_XR_TERMINATE = """
    if(any(greaterThan(max(g_dirStep, vec3(0.0))*(g_dataPos - in_texMax[0]),vec3(0.0))) ||
      any(greaterThan(min(g_dirStep, vec3(0.0))*(g_dataPos - in_texMin[0]),vec3(0.0))))
      {
      break;
      }
    // Keep marching past an opaque brain while a map could still tint it:
    // stop once both are saturated, or once the ray is so deep that
    // exp(-depth / depth) has fallen below 1% for the deepest-reaching map.
    if ((xr_overlay.a > g_opacityThreshold && g_fragColor.a > g_opacityThreshold) ||
        (g_fragColor.a > g_opacityThreshold && xr_surfaceT >= 0.0 &&
         (g_currentT - xr_surfaceT) * in_sampleDistance > 4.6 * XR_DEPTH_MAX) ||
        g_currentT >= g_terminatePointMax)
      {
      break;
      }
    ++g_currentT;
"""
_XR_EXIT = """//VTK::Shading::Exit
  {
  float xr_oa = xr_overlay.a;
  vec3 xr_oc = xr_oa > 1e-6 ? xr_overlay.rgb / xr_oa : vec3(0.0);
  float xr_a = g_fragColor.a;
  // The brain, its colour replaced by the maps' as far as they gathered...
  vec3 xr_rgb = g_fragColor.rgb
                + xr_oa * (xr_dark + xr_diffuse * xr_oc - g_fragColor.rgb);
  // ...and, where the brain lets light through, the maps seen directly.
  xr_rgb += (1.0 - xr_a) * xr_overlay.rgb;
  g_fragColor = vec4(clamp(xr_rgb, 0.0, 1.0), xr_a + (1.0 - xr_a) * xr_oa);
  }
"""


def _install_xray(actor, depths_mm, outside=False):
    """Swap the ray caster's compositing for the x-ray tint.

    ``depths_mm`` is one depth per overlay component, in order; ``outside``
    lets maps show where the body has no tissue.
    """
    depths = [max(float(d), 0.05) for d in depths_mm] or [_PAINT_DEPTH_MM]
    blocks = "".join(_XR_OVERLAY % {"i": i, "depth": "%.6f" % d}
                     for i, d in enumerate(depths, start=1))
    shader = actor.GetShaderProperty()
    shader.ClearAllFragmentShaderReplacements()
    for tag, code in (
            ("//VTK::Base::Dec", _XR_DECL % {
                "depth_max": "%.6f" % max(depths),
                "gather": "true" if outside else "g_fragColor.a > 0.0"}),
            ("//VTK::DepthPeeling::Ray::Init", _XR_INIT),
            ("//VTK::Shading::Impl", _XR_SHADE % {"overlays": blocks}),
            ("//VTK::Terminate::Impl", _XR_TERMINATE),
            ("//VTK::Shading::Exit", _XR_EXIT)):
        shader.AddFragmentShaderReplacement(tag, True, code, False)


# Public names for the session: an overlay's own selected values, and the
# overlay with its scale frozen from them (see ``_native`` and ``_pinned``).
native_values = _native
pinned = _pinned

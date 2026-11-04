"""Turn a NIfTI into colour on a brain mesh.

There are two honest ways to put a volume onto a surface and they answer
different questions, so both are here and neither is the "real" one:

``paint_vertices`` samples the volume at every mesh vertex and gives a
continuous gradient.  It answers "what does the map look like where it touches
this surface" and it is the right choice for a cortical surface, because the
value it shows at a point genuinely came from that point.

``paint_parcels`` reduces the volume inside each regional mesh to one number
and floods the whole region with it.  It answers "how much does this map
implicate this structure", which is the question a parcellated figure is
usually asking.  The reduction is chosen, not assumed -- see ``parcel_stats``.

Both leave values that do not survive thresholding as NaN, which the renderer
draws in the opaque base colour.  An unpainted region therefore still reads as
anatomy rather than vanishing or turning into the bottom of the colour bar --
the distinction between "no signal here" and "the most negative value in the
figure" matters, and a colormap alone cannot make it.

Sampling geometry
-----------------
Meshes are in MNI world millimetres and volumes are in voxel indices, so every
sample goes through ``inv(affine)``.  Interior points for a parcel are found by
voxelising the mesh itself on a 1 mm world grid rather than by sampling the
volume's own voxels: the mesh is the thing that was drawn, so the mesh is the
thing whose inside should be measured, and a 1 mm grid weights every cubic
millimetre of a region equally whatever the volume's resolution happens to be.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyvista as pv

__all__ = [
    "sample_nifti", "paint_vertices", "parcel_stats", "parcel_interiors",
    "paint_parcels", "select_regions", "load_parcels", "coarsen", "STATS",
    "outline_lines",
]

STATS = ("max", "mean_nonzero", "mean", "min", "sum")


# -- volume sampling --------------------------------------------------------
@lru_cache(maxsize=8)
def _load_volume(path, mtime):
    """Read and clean a volume once. ``mtime`` is in the key so an edited file
    on disk invalidates the entry rather than silently serving a stale map."""
    import nibabel as nib

    return _clean(nib.load(path))


def _clean(img):
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]
    # Statistical maps routinely carry NaN outside the analysis mask. Left in
    # place, a single NaN voxel poisons every trilinear sample that touches it,
    # so the whole neighbourhood silently drops out of the figure.
    return (np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0),
            np.asarray(img.affine, float))


def _volume(nifti):
    """Data and affine for a path or an in-memory image.

    Paths go through a small cache: reducing a volume inside 165 parcel meshes
    asks for the same volume 165 times, and re-reading a gzipped NIfTI each time
    costs more than every other step of the figure put together.
    """
    if hasattr(nifti, "affine"):
        return _clean(nifti)
    path = str(Path(nifti).expanduser())
    if path.lower().endswith(".npz"):
        from gnv.convert.volume_meshes import read_backdrop_npz

        return read_backdrop_npz(path)
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        stamp = 0.0
    return _load_volume(path, stamp)


def sample_nifti(points, nifti, order=1):
    """Sample a volume at world-millimetre ``points`` (N, 3).

    ``order=1`` is trilinear, which is what a gradient wants; ``order=0`` is
    nearest neighbour, which is what you want for a label volume or any map
    whose values are categorical.  Points outside the volume return 0.
    """
    from scipy.ndimage import map_coordinates

    data, affine = _volume(nifti)
    pts = np.asarray(points, float)
    vox = np.linalg.inv(affine) @ np.c_[pts, np.ones(len(pts))].T
    return map_coordinates(data, vox[:3], order=order, mode="constant", cval=0.0)


def _select(values, sign="both", absolute=False, threshold=None,
            max_value=None):
    """Sign selection, optional |v|, and thresholding -- shared by both painters.

    ``sign`` is applied to the ORIGINAL values and ``absolute`` afterwards, so
    ``sign="negative", absolute=True`` means "the negative tail, plotted as
    magnitudes", which is how a one-tailed figure in a single colour is made.
    """
    out = np.asarray(values, float).copy()
    if sign not in ("both", "positive", "negative"):
        raise ValueError("sign must be 'both', 'positive' or 'negative'")
    if sign == "positive":
        out[out < 0] = np.nan
    elif sign == "negative":
        out[out > 0] = np.nan
    if absolute:
        out = np.abs(out)
    # Threshold min/max is a BAND on the value itself, not on its magnitude:
    # anything outside [min, max] is NaN'd. With absolute=True the values are
    # already magnitudes by this point, so the same comparison gates |v|.
    # (A magnitude gate made min=0 a no-op, which left -7 painted.)
    if threshold is not None:
        out[out < float(threshold)] = np.nan
    if max_value is not None:
        out[out > float(max_value)] = np.nan
    out[out == 0] = np.nan
    return out


# -- continuous painting ----------------------------------------------------
def paint_vertices(mesh, nifti, sign="both", absolute=False, threshold=None,
                   order=1, scalar="value"):
    """Copy of ``mesh`` carrying the volume's value at each vertex.

    Works on a single ``PolyData`` or on a ``{'L': ..., 'R': ...}`` hemisphere
    dict, returning the same shape it was given.
    """
    if isinstance(mesh, dict):
        return {k: paint_vertices(v, nifti, sign, absolute, threshold, order, scalar)
                for k, v in mesh.items()}

    out = mesh.copy()
    out[scalar] = _select(
        sample_nifti(out.points, nifti, order=order), sign, absolute, threshold
    )
    return out


# -- parcels ----------------------------------------------------------------
def load_parcels(source):
    """``{region_name: PolyData}`` from a registered mesh name or a directory.

    Region order follows ``atlas_LUT.txt`` when the directory ships one, so a
    figure's regions stay in atlas order rather than in whatever order the
    filesystem returns.
    """
    if isinstance(source, dict):
        return source

    name = str(source)
    directory = Path(name).expanduser()
    if not directory.is_dir():
        from gnv.render.brains import mesh_subcortex

        spec = mesh_subcortex(name)
        if not spec["custom_atlas_path"]:
            raise ValueError(
                f"{name!r} is a yabplot-packaged region set, not a directory of "
                "meshes this renderer can read. Point load_parcels at a directory, "
                "or use one of the locally built parcel meshes."
            )
        directory = Path(spec["custom_atlas_path"])

    files = {p.stem: p for p in sorted(directory.glob("*.vtk"))}
    lut = directory / "atlas_LUT.txt"
    if lut.exists():
        ordered = []
        for line in lut.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] in files:
                ordered.append(parts[1])
        files = {k: files[k] for k in ordered} | files
    return {stem: pv.read(path) for stem, path in files.items()}


def coarsen(meshes, reduction=0.7):
    """Decimate a mesh or a group of them, for the tessellated look.

    The parcel meshes are built at ~1.1 mm edges, which gives clean silhouettes
    and, at figure scale, a per-triangle wireframe so dense it reads as grey
    haze rather than as structure. Decimating to ~2-3 mm edges puts the facets
    and the wireframe back on a scale the eye can resolve: the surface shades as
    visible plates instead of as a smooth blob.

    This is a DISPLAY choice, applied per mesh before anything is merged or
    sampled, so region statistics are computed on whatever geometry is actually
    drawn and the numbers never disagree with the picture.
    """
    if not reduction:
        return meshes
    reduction = float(reduction)
    if isinstance(meshes, dict):
        return {k: coarsen(v, reduction) for k, v in meshes.items()}
    if meshes.n_cells < 80:            # already coarse; decimating destroys it
        return meshes
    return meshes.decimate_pro(reduction, preserve_topology=True).triangulate()


def _interior_points(mesh, step=1.0):
    """World coordinates of the points inside a closed mesh, on a `step` grid."""
    lo = np.floor(np.asarray(mesh.bounds).reshape(3, 2)[:, 0] / step) * step - step
    hi = np.ceil(np.asarray(mesh.bounds).reshape(3, 2)[:, 1] / step) * step + step
    dims = np.maximum(((hi - lo) / step).astype(int) + 1, 2)
    ref = pv.ImageData(dimensions=dims, spacing=(step,) * 3, origin=tuple(lo))
    mask = mesh.voxelize_binary_mask(reference_volume=ref)
    inside = np.asarray(mask.point_data["mask"]).astype(bool)
    return np.asarray(ref.points)[inside]


def parcel_interiors(parcels, step=1.0):
    """``{region: (N, 3) world points}`` inside each regional mesh.

    Voxelising 165 meshes is the slow part of a parcel figure, and it depends
    only on the MESHES -- not on the volume, the statistic or the threshold. A
    caller that is going to re-reduce the same regions (a GUI slider, a sweep
    over several contrasts) computes this once and passes it back in.
    """
    return {name: _interior_points(mesh, step=step)
            for name, mesh in load_parcels(parcels).items()}


def parcel_stats(parcels, nifti, stat="max", sign="both", absolute=False,
                 threshold=None, step=1.0, order=1, interiors=None,
                 max_value=None):
    """``{region_name: value}`` reducing the volume inside each regional mesh.

    ``stat``
        ``"max"``            the most extreme voxel, sign kept -- sensitive, and
                             the right choice for a sparse thresholded map where
                             a real peak occupies a small part of a big region.
        ``"mean_nonzero"``   the average over voxels that carry a value, which
                             stops a thresholded map from being diluted by the
                             zeros around it.
        ``"mean"``           the average over the whole region, zeros included.
        ``"min"``, ``"sum"`` the obvious.

    ``sign`` keeps only one tail before reducing -- ``sign="positive",
    threshold=1.96`` is the one-tailed selection that makes a figure one-signed,
    and therefore drawn in a single-hue ramp rather than a diverging map.
    ``absolute`` takes ``|v|`` after that, so a region that is strongly negative
    ranks with the strongly positive ones instead of at the bottom.
    ``threshold`` drops voxels with ``|v| < threshold`` *before* the reduction,
    which is what makes ``"max"`` and ``"mean_nonzero"`` mean what you want on
    an unthresholded t-map.
    """
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}, got {stat!r}")

    parcels = load_parcels(parcels)
    out = {}
    for name, mesh in parcels.items():
        pts = (interiors or {}).get(name)
        if pts is None:
            pts = _interior_points(mesh, step=step)
        if not len(pts):
            out[name] = np.nan
            continue
        v = _select(sample_nifti(pts, nifti, order=order), sign, absolute,
                    threshold, max_value)
        out[name] = _reduce(v[np.isfinite(v)], stat)
    return out


def _reduce(v, stat):
    if not len(v):
        return np.nan
    # "max" is the maximum, literally. It used to be the peak MAGNITUDE with
    # its sign kept, so a region holding only -7 reduced to -7 under "max" --
    # which reads as a bug however it is documented. Tick Absolute value if
    # peak magnitude is what you want.
    if stat == "max":
        return float(np.max(v))
    if stat == "min":
        return float(np.min(v))
    if stat == "mean":
        return float(np.mean(v))
    if stat == "sum":
        return float(np.sum(v))
    nz = v[v != 0]
    return float(np.mean(nz)) if len(nz) else np.nan


def select_regions(values, min_value=None, keep_top=None):
    """NaN out regions that do not clear a floor, or are outside the top N.

    ``threshold`` in ``parcel_stats`` drops *voxels* before the reduction; this
    drops *regions* after it. They do different jobs and a parcel figure usually
    wants both: voxel thresholding decides what counts as signal, region
    selection decides what is worth colouring. Without the second one, a "max"
    figure floods every region that contains a single suprathreshold voxel,
    which on a whole-brain map is all of them.
    """
    out = dict(values)
    if min_value is not None:
        out = {k: (v if np.isfinite(v) and abs(v) >= float(min_value) else np.nan)
               for k, v in out.items()}
    if keep_top is not None:
        ranked = sorted((k for k in out if np.isfinite(out[k])),
                        key=lambda k: -abs(out[k]))[:int(keep_top)]
        keep = set(ranked)
        out = {k: (v if k in keep else np.nan) for k, v in out.items()}
    return out


def paint_parcels(parcels, nifti=None, values=None, stat="max", sign="both",
                  absolute=False, threshold=None, min_value=None, keep_top=None,
                  step=1.0, scalar="value", interiors=None):
    """One merged ``PolyData`` of all regions, flooded with their statistic.

    Pass ``nifti`` to compute the statistic, or ``values`` to supply a
    ``{region: number}`` mapping you already have.  Regions with no value come
    back as NaN and are drawn in the base colour.

    The scalar is attached per cell, not per point: regions touch, and a point
    scalar would smear one region's value across the seam into its neighbour.
    """
    parcels = load_parcels(parcels)
    if values is None:
        if nifti is None:
            raise ValueError("paint_parcels needs either a nifti or a values mapping")
        values = parcel_stats(parcels, nifti, stat=stat, sign=sign,
                              absolute=absolute, threshold=threshold, step=step,
                              interiors=interiors)
    values = select_regions(values, min_value=min_value, keep_top=keep_top)

    meshes, scalars = [], []
    for name, mesh in parcels.items():
        v = float(values.get(name, np.nan))
        meshes.append(mesh)
        scalars.append(np.full(mesh.n_cells, v, dtype=float))

    merged = pv.merge(meshes) if len(meshes) > 1 else meshes[0].copy()
    merged[scalar] = np.concatenate(scalars)
    return merged


def fill_gaps(mesh, values, passes=3):
    """Spread sampled values into unsampled vertices across the mesh.

    Vertex sampling asks the volume for a value at one exact point. A surface
    vertex that lands in a zero voxel -- just outside a mask, or in the gap
    between a gyral crown and the map's edge -- gets nothing, and the figure
    speckles even when every part of the structure has data nearby.

    Each pass replaces an unpainted vertex with the mean of its painted
    neighbours ALONG THE MESH, so a value travels across the surface rather
    than through it: the far bank of a sulcus stays unpainted unless it has its
    own data. Nothing is invented where a whole connected patch is empty, and
    already-painted vertices are never overwritten.
    """
    import numpy as np

    values = np.asarray(values, float).copy()
    if not passes or mesh.n_points != len(values):
        return values

    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    # Every undirected edge, both ways, so a vertex can be reached from either end
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    src, dst = edges[:, 0], edges[:, 1]

    for _ in range(int(passes)):
        missing = ~np.isfinite(values)
        if not missing.any():
            break
        known = np.isfinite(values)
        live = known[src]
        total = np.bincount(dst[live], weights=values[src[live]], minlength=len(values))
        count = np.bincount(dst[live], minlength=len(values))
        reachable = missing & (count > 0)
        if not reachable.any():
            break
        values[reachable] = total[reachable] / count[reachable]
    return values


# -- outlines ---------------------------------------------------------------
_OUTLINE_MASKS = {}


def _outline_mask(nifti, sign, absolute, threshold, max_value, smooth_vox):
    """The selected voxels as a 0/1 volume, blurred by ``smooth_vox`` voxels.

    Selection is exactly ``_select``: whatever a fill overlay with the same
    settings would paint is inside, everything else is outside. The blur only
    rounds the voxel staircase so a traced edge reads as a boundary rather than
    as the grid; it moves the edge by well under a voxel for sigma <= 1.
    """
    data, affine = _volume(nifti)
    if hasattr(nifti, "affine"):
        stamp = id(nifti)
    else:
        try:
            stamp = os.path.getmtime(str(Path(nifti).expanduser()))
        except OSError:
            stamp = 0.0
    key = (str(nifti), stamp, sign, bool(absolute), threshold,
           max_value, float(smooth_vox or 0.0))
    if key not in _OUTLINE_MASKS:
        chosen = _select(data.ravel(), sign=sign, absolute=absolute,
                         threshold=threshold, max_value=max_value)
        mask = np.isfinite(chosen).reshape(data.shape).astype(np.float32)
        if smooth_vox:
            from scipy.ndimage import gaussian_filter

            mask = gaussian_filter(mask, float(smooth_vox))
        if len(_OUTLINE_MASKS) > 8:
            _OUTLINE_MASKS.clear()
        _OUTLINE_MASKS[key] = mask
    return _OUTLINE_MASKS[key], affine


def outline_lines(mesh, nifti, sign="both", absolute=False, threshold=None,
                  max_value=None, smooth_vox=1.0, lift_mm=0.3):
    """Lines on ``mesh`` tracing the edge of a volume's selected region.

    The 0/1 selection (see ``_outline_mask``) is sampled trilinearly at every
    vertex and contoured at 0.5, so the line runs across the surface where the
    region starts. Lines are lifted ``lift_mm`` along the vertex normals so they
    sit on the surface instead of flickering into it. Returns an empty PolyData
    when nothing on this mesh is selected.
    """
    from scipy.ndimage import map_coordinates

    mask, affine = _outline_mask(nifti, sign, absolute, threshold, max_value,
                                 smooth_vox)
    surf = mesh.extract_surface() if not isinstance(mesh, pv.PolyData) else mesh
    surf = surf.triangulate()
    pts = np.asarray(surf.points, float)
    vox = np.linalg.inv(affine) @ np.c_[pts, np.ones(len(pts))].T
    inside = map_coordinates(mask, vox[:3], order=1, mode="constant", cval=0.0)
    if not (inside.max() > 0.5 > inside.min()):
        return pv.PolyData()
    surf = surf.copy(deep=False)
    surf.point_data["_inside"] = inside
    if lift_mm:
        surf = surf.compute_normals(point_normals=True, cell_normals=False,
                                    split_vertices=False, auto_orient_normals=False)
    lines = surf.contour([0.5], scalars="_inside")
    if lift_mm and lines.n_points and "Normals" in lines.point_data:
        lines.points = lines.points + lift_mm * np.asarray(lines.point_data["Normals"])
    return lines

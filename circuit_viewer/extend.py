"""Grow a map's coloured regions outward, so it covers what it should.

A map often stops just short of the structure it sits on: a 2 mm statistical
map ends a voxel inside the cortex, so the crowns of gyri on a mesh, the rim of
a slice, or the edge of a nucleus in a volume stay uncoloured even when the
figure is meant to show the region as a whole.

``extend_mm`` on an overlay fixes that at the source, in voxel space: every
voxel the overlay does NOT colour (after its sign, threshold and max) that lies
within ``extend_mm`` of one it does takes the value of the NEAREST coloured
voxel. It is a nearest-value extrapolation, not a blur:

* values are copied, never averaged or invented, so the colour scale's range
  is unchanged and a region's edge keeps the colour it already had;
* coloured voxels are never altered;
* distance is in millimetres (the voxel size is read from the affine), so the
  same number means the same thing on a 1 mm and a 2 mm map.

Because the grown map is handed on as a file, exactly as a 4D frame is (see
``frames``), every painter -- mesh vertices, slice planes, volume bodies, fibre
sampling, outlines -- sees the same grown map with no code of its own. The file
is a cache in this run's scratch folder: reused while the settings hold, swept
once idle, removed with the run (see ``scratch``).
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

__all__ = ["resolve", "extended_path"]

# A grown map is cheap to write but the distance transform behind it is the
# cost, so it is kept for the most recent maps: dragging the mm slider only
# re-does the copy, never the transform.
_transforms = OrderedDict()     # (path, mtime, band) -> (data, affine, dist, nearest)
_KEEP = 1        # each entry holds the map plus distance and index arrays
# Above this the transform's index arrays get large (about 20 bytes a voxel);
# the work is cropped to the region the growth can reach first.
_CROP_PAD_VOX = 2


def _stamp(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _band(overlay):
    return (str(overlay.get("sign", "both")), bool(overlay.get("absolute")),
            overlay.get("threshold"), overlay.get("max_value"))


def _transform(path, overlay, reach_mm):
    """The map, its affine, and for every voxel the distance (mm) to the
    nearest coloured voxel and that voxel's index -- inside a box no larger
    than the growth can reach."""
    from scipy import ndimage

    from calvin_utils.plotting_utils.mesh_paint import _select

    key = (path, _stamp(path)) + tuple(map(str, _band(overlay)))
    hit = _transforms.get(key)
    if hit is not None and hit[4] >= reach_mm:
        _transforms.move_to_end(key)
        return hit

    import nibabel as nib

    image = nib.load(path)
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim > 3:
        data = data.reshape(data.shape[:3] + (-1,))[..., 0]
    affine = np.asarray(image.affine, float)
    sign, absolute, threshold, max_value = _band(overlay)
    present = np.isfinite(_select(data.ravel(), sign=sign, absolute=absolute,
                                  threshold=threshold, max_value=max_value)
                          ).reshape(data.shape)
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    zooms[zooms <= 0] = 1.0

    dist = np.full(data.shape, np.inf, np.float32)
    nearest = None
    box = None
    if present.any():
        # Only the box the growth can reach: the coloured voxels' bounds,
        # padded by the reach. A small map on a fine grid then costs what its
        # neighbourhood costs, not what the whole field of view does.
        coords = np.nonzero(present)
        pad = np.ceil(reach_mm / zooms).astype(int) + _CROP_PAD_VOX
        lo = np.maximum(np.array([c.min() for c in coords]) - pad, 0)
        hi = np.minimum(np.array([c.max() for c in coords]) + pad + 1, data.shape)
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        d, idx = ndimage.distance_transform_edt(
            ~present[box], sampling=zooms, return_indices=True)
        dist[box] = d
        nearest = (box, idx.astype(np.int32))
    entry = (data, affine, dist, nearest, float(reach_mm), present)
    _transforms[key] = entry
    while len(_transforms) > _KEEP:
        _transforms.popitem(last=False)
    return entry


def extended_path(path, overlay, mm):
    """A NIfTI holding ``path`` grown by ``mm`` under ``overlay``'s selection;
    ``path`` itself when there is nothing to grow."""
    from . import scratch

    path = str(Path(str(path)).expanduser())
    mm = float(mm)
    tag = hashlib.sha1("|".join(map(str, (path, _stamp(path), *_band(overlay),
                                          round(mm, 3)))).encode()).hexdigest()[:16]
    out = scratch.area("extend") / f"{tag}.nii"
    if out.is_file():
        scratch.touch(out)
        return str(out)
    scratch.sweep()

    data, affine, dist, nearest, _reach, present = _transform(path, overlay, mm)
    if nearest is None:
        return path                         # nothing coloured to grow from
    box, idx = nearest
    grown = data.copy()
    sub = grown[box]
    fill = (~present[box]) & (dist[box] <= mm)
    if not fill.any():
        return path
    sub[fill] = data[box][tuple(i[fill] for i in idx)]
    grown[box] = sub

    import nibabel as nib

    temp = out.with_name(f".{out.name}.{os.getpid()}.partial.nii")
    nib.save(nib.Nifti1Image(grown, affine), str(temp))
    os.replace(temp, out)
    return str(out)


def resolve(overlay, spec_key):
    """``overlay`` pointed at its grown map, or ``overlay`` unchanged.

    Runs after ``frames.resolve``, so a 4D map grows the frame on show. The
    colour bar keeps the overlay's identity and name, not the cache file's.
    """
    mm = float(overlay.get("extend_mm") or 0.0)
    path = overlay.get("nifti")
    if mm <= 0 or not path or not str(path).lower().endswith((".nii", ".nii.gz")):
        return overlay
    grown = extended_path(path, overlay, mm)
    if grown == str(Path(str(path)).expanduser()):
        return overlay
    out = dict(overlay)
    out["nifti"] = grown
    out["_spec_key"] = spec_key
    if not overlay.get("name"):
        stem = Path(str(path)).name
        for suffix in (".nii.gz", ".nii"):
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        out["name"] = stem
    return out

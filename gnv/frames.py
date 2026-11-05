"""4D overlays: one frame at a time, on one colour scale.

A 4D NIfTI (x, y, z, t) or a 2D fibre-value array (fibres x t) is a series of
maps. Every painter in the app -- meshes, slices, bodies, fibres -- reads a 3D
map from a path, so a frame is handed to them as a path too: the frame is
written once to a small cache file and the overlay is pointed at it. Nothing
downstream has to know that a series exists.

The colour scale is read from the WHOLE series, not the frame on screen, so a
value is the same colour in every frame and playback shows the data changing
rather than the scale re-fitting itself each step. The colour bar keeps the
overlay's own identity (``_spec_key``), so it stays attached while the frame
changes underneath it.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

__all__ = ["frame_count", "frame_path", "resolve"]

_counts = {}                    # (path, mtime) -> frames
_series = OrderedDict()         # (path, mtime) -> (array, affine or None)
_scales = {}                    # (path, mtime, band) -> pinned scale fields


def _stamp(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _kind(path):
    lower = str(path).lower()
    if lower.endswith((".nii", ".nii.gz")):
        return "nifti"
    if lower.endswith((".npy", ".fib.json", ".fib.desc.json")):
        return "fibres"
    return None


def _fibre_values_path(path):
    if str(path).lower().endswith(".json"):
        from calvin_utils.neuroimaging_utils.tract_utils.fiber_descriptor import read_descriptor

        return str(read_descriptor(path)["values"])
    return str(path)


def frame_count(path):
    """How many frames ``path`` holds: 1 for an ordinary 3D map or 1-D values."""
    path = str(Path(str(path)).expanduser())
    key = (path, _stamp(path))
    if key in _counts:
        return _counts[key]
    count = 1
    try:
        kind = _kind(path)
        if kind == "nifti":
            import nibabel as nib

            shape = nib.load(path).header.get_data_shape()
            count = int(np.prod(shape[3:])) if len(shape) > 3 else 1
        elif kind == "fibres":
            values = np.load(_fibre_values_path(path), mmap_mode="r", allow_pickle=False)
            count = int(values.shape[1]) if values.ndim == 2 else 1
    except Exception:
        count = 1
    _counts[key] = max(count, 1)
    return _counts[key]


def _load(path):
    key = (path, _stamp(path))
    if key in _series:
        _series.move_to_end(key)
        return _series[key]
    if _kind(path) == "nifti":
        import nibabel as nib

        image = nib.load(path)
        data = np.asarray(image.dataobj, dtype=np.float32)
        data = data.reshape(data.shape[:3] + (-1,))
        entry = (data, np.asarray(image.affine, float))
    else:
        # Fibres x frames: one row per atlas fibre, one column per time point.
        values = np.asarray(np.load(_fibre_values_path(path), allow_pickle=False), dtype=np.float32)
        entry = (values, None)
    _series[key] = entry
    while len(_series) > 2:
        _series.popitem(last=False)
    return entry


_SUFFIXES = (".nii.gz", ".nii", ".fib.values.npy", ".fib.desc.json", ".fib.json", ".fib.npy", ".npy")


def _stem(path):
    """The series' name without its extension: ``bold.nii.gz`` -> ``bold``."""
    stem = Path(path).name
    for suffix in _SUFFIXES:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _copy_label_table(path, folder):
    """Put the series' label table (its .txt / .lut sidecar) beside a frame,
    so a 4D label series names its regions in every frame as it would in 3D."""
    import shutil

    try:
        from gnv.convert.volume_meshes import _sidecars

        sides = _sidecars(path)
        if sides:
            shutil.copyfile(sides[0], folder / (_stem(path) + sides[0].suffix))
    except Exception:
        pass                    # unnamed regions, not a failed frame


def frame_path(path, frame):
    """A 3D file (or 1-D values file) holding frame ``frame`` of ``path``."""
    path = str(Path(str(path)).expanduser())
    total = frame_count(path)
    frame = int(frame) % total
    from . import scratch

    tag = hashlib.sha1(f"{path}|{_stamp(path)}".encode()).hexdigest()[:16]
    # Frame copies are a cache in this run's scratch folder: touched when
    # reused, swept once idle, and gone with the run (see scratch.py). Each
    # frame keeps the series' own file name, in a folder of its own, so a
    # surface made from it is named after the series -- the same region name
    # in every frame, which is what a hidden region is remembered by.
    nifti = _kind(path) == "nifti"
    folder = scratch.area("frames") / f"{tag}_f{frame:04d}"
    out = folder / (_stem(path) + (".nii" if nifti else ".fib.values.npy"))
    if out.is_file():
        scratch.touch(out)
        return str(out)
    scratch.sweep()
    folder.mkdir(parents=True, exist_ok=True)
    if nifti:
        _copy_label_table(path, folder)
    data, affine = _load(path)
    temp = out.with_name(f".{out.name}.{os.getpid()}.partial")
    if nifti:
        import nibabel as nib

        nib.save(nib.Nifti1Image(np.ascontiguousarray(data[..., frame]), affine),
                 str(temp) + ".nii")
        os.replace(str(temp) + ".nii", out)
    else:
        with open(temp, "wb") as handle:
            np.save(handle, np.ascontiguousarray(data[:, frame]))
        os.replace(temp, out)
    return str(out)


def _series_scale(path, overlay):
    """The overlay's scale read from every frame: ``{"clim", "symmetric"}``."""
    from gnv.render.mesh_paint import _select
    from gnv.render.overlay_paint import scale_for

    band = (overlay.get("sign"), bool(overlay.get("absolute")),
            overlay.get("threshold"), overlay.get("max_value"),
            overlay.get("palette"), overlay.get("color"), overlay.get("symmetric"))
    key = (path, _stamp(path)) + tuple(map(str, band))
    if key in _scales:
        return _scales[key]
    data, _ = _load(path)
    selected = _select(data.ravel(), sign=overlay.get("sign", "both"),
                       absolute=overlay.get("absolute", False),
                       threshold=overlay.get("threshold"),
                       max_value=overlay.get("max_value"))
    finite = selected[np.isfinite(selected)]
    out = {}
    if finite.size:
        symmetric = overlay.get("symmetric")
        if symmetric is None:
            symmetric = bool((finite > 0).any() and (finite < 0).any())
        _, clim = scale_for(finite, dict(overlay, symmetric=symmetric))
        out = {"clim": [float(clim[0]), float(clim[1])], "symmetric": symmetric}
    _scales[key] = out
    return out


def resolve(overlay, spec_key):
    """``overlay`` pointing at its current frame, or ``overlay`` unchanged.

    ``spec_key`` is the overlay's identity for its colour bar; the frame copy
    carries it so the bar stays attached through playback.
    """
    path = overlay.get("nifti")
    if not path or frame_count(path) <= 1:
        return overlay
    path = str(Path(str(path)).expanduser())
    out = dict(overlay)
    frame = int(overlay.get("frame") or 0) % frame_count(path)
    out["nifti"] = frame_path(path, frame)
    out["_spec_key"] = spec_key
    # The colour bar is captioned by name; without one it would read the
    # cache file's. Name it after the series, with the frame on show.
    stem = _stem(path)
    out["name"] = f"{overlay.get('name') or stem} [{frame + 1}/{frame_count(path)}]"
    if overlay.get("clim") is None:
        out.update(_series_scale(path, overlay))
    return out

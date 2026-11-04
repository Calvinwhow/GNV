"""Anatomical backdrops: a NIfTI turned into slices you can stream one at a time.

Three orthogonal planes fanning through a high-resolution scan is the Lead-DBS
idiom, and the thing standing in its way is size.  The 100 um Edlow volume is
1971 x 2331 x 1891 = 8.7 G voxels; nothing interactive can hold that, and asking
a gzipped NIfTI for an arbitrary slice means decompressing everything before it.

Two decisions make it light, and the second matters more than the first.

**Per-slice storage.**  A backdrop is an ``.npz`` holding one compressed array
per slice per axis.  ``np.load`` on it is lazy, so reading slice 412 decompresses
exactly that member -- a few milliseconds -- rather than touching the volume.
All three axes are stored because a single array can only be sliced cheaply
along its first axis; the other two would have to touch every page.

**Native resolution by default.**  A backdrop is built at the source's own
voxel size: hand it a 100 um scan and you get 100 um slices, because the point
of having that scan is being able to show it.  ``mm`` is there to coarsen
deliberately, not as a default that quietly throws detail away.

That costs what it costs.  For the 1971 x 2331 x 1891 Edlow volume all three
axes at native are ~26 GB uncompressed; the empty background and the smoothness
of tissue bring the file well below that, but it is still gigabytes.  Coarsening
to 0.4 mm would be 64x smaller -- an option, never the default.

**Building never holds the volume.**  The source is streamed in blocks into a
temporary uint8 memmap, then each axis is written out in slabs sized to a memory
budget, so nothing larger than one slab is ever resident and the ceiling is disk
rather than RAM.  Two consequences worth knowing: a native 100 um build moves
tens of GB through the disk and takes minutes, and it needs roughly the
uncompressed size free as scratch while it runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

__all__ = ["BACKDROP_DIR", "build_backdrop", "list_backdrops", "Backdrop"]

# Outside the package: a library of these runs to hundreds of megabytes and has
# no business inside a source tree. Overridable, and created on first use.
BACKDROP_DIR = Path.home() / ".gnv" / "backdrops"
AXES = (0, 1, 2)


def _slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _percentiles(img, low=1.0, high=99.5, samples=40):
    """Window from a scatter of slices, so a huge volume is not read twice."""
    n = img.shape[0]
    picks = np.unique(np.linspace(0, n - 1, min(samples, n)).astype(int))
    values = np.concatenate([
        np.asarray(img.dataobj[i], dtype=np.float32).ravel()[::7] for i in picks
    ])
    values = values[np.isfinite(values)]
    values = values[values != 0]
    if not values.size:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [low, high])
    return float(lo), float(hi if hi > lo else lo + 1.0)


def _block_mean(array, factors):
    """Mean over integer blocks, padding the tail so nothing is dropped."""
    pads = [(0, (-s) % f) for s, f in zip(array.shape, factors)]
    if any(p[1] for p in pads):
        array = np.pad(array, pads, mode="edge")
    shape = []
    for size, factor in zip(array.shape, factors):
        shape += [size // factor, factor]
    return array.reshape(shape).mean(axis=(1, 3, 5))


def build_backdrop(nifti, out_dir=None, mm=None, name=None, window=None,
                   block=32, slab_bytes=1 << 30, compresslevel=4,
                   progress=None):
    """Convert a NIfTI into a streamable backdrop and return the ``.npz`` path.

    ``mm`` defaults to the source's own voxel size -- native. Pass a coarser
    number to trade detail for size; 0.4 is about 64x smaller than a 100 um scan
    and still finer than a 1 mm template.

    ``window`` is the intensity range mapped onto 0-255; it defaults to the 1st
    and 99.5th percentiles of the nonzero data, which keeps a bright skull or a
    single hot voxel from crushing the brain into the bottom of the range.
    ``slab_bytes`` caps how much is resident while the transposed axes are
    written, and is the only real memory knob. Raising it mostly helps the LAST
    axis, whose slices are the most scattered in the scratch file and which
    therefore re-reads it the most times.
    """
    import nibabel as nib

    path = Path(nifti).expanduser()
    img = nib.load(str(path))
    if len(img.shape) > 3:
        raise ValueError(f"{path.name} has {len(img.shape)} dimensions; need 3")

    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    if mm is None:
        factors = np.ones(3, dtype=int)
        mm = float(np.min(zooms))
    else:
        factors = np.maximum(np.round(float(mm) / zooms).astype(int), 1)
    shape = tuple(int(np.ceil(s / f)) for s, f in zip(img.shape, factors))

    lo, hi = window if window else _percentiles(img)
    scale = 255.0 / (hi - lo)

    out_dir = Path(out_dir or BACKDROP_DIR).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{_slug(name or path.name.split('.nii')[0])}.npz"
    scratch = target.with_suffix(".building.raw")

    try:
        bbox = _fill_scratch(img, scratch, shape, factors, lo, scale, block,
                             progress)
        _write_npz(target, scratch, shape, bbox, img.affine, zooms, factors, mm,
                   (lo, hi), path, slab_bytes, compresslevel, progress)
    finally:
        scratch.unlink(missing_ok=True)
    return target


def _fill_scratch(img, scratch, shape, factors, lo, scale, block, progress):
    """Stream the source into a uint8 memmap, returning the data bounding box."""
    volume = np.memmap(scratch, dtype=np.uint8, mode="w+", shape=shape)
    fy, fx = int(factors[1]), int(factors[2])
    step = int(factors[0]) * max(1, block)
    written = 0
    bounds = [[None, None], [None, None], [None, None]]

    # An integer source at native resolution -- which is what a 100 um scan
    # usually is -- can be windowed with a 256/65536-entry lookup instead of a
    # float32 round trip. On 8.7 G voxels that is the difference between a build
    # you can wait for and one you cannot.
    native = all(int(f) == 1 for f in factors)
    dtype = np.dtype(img.get_data_dtype())
    table = None
    if native and dtype in (np.uint8, np.uint16):
        levels = np.arange(np.iinfo(dtype).max + 1, dtype=np.float32)
        table = np.clip((levels - lo) * scale, 0, 255).astype(np.uint8)

    for start in range(0, img.shape[0], step):
        stop = min(start + step, img.shape[0])
        if table is not None:
            chunk = table[np.asarray(img.dataobj[start:stop], dtype=dtype)]
        else:
            chunk = np.asarray(img.dataobj[start:stop], dtype=np.float32)
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            chunk = _block_mean(chunk, (int(factors[0]), fy, fx))
            chunk = np.clip((chunk - lo) * scale, 0, 255).astype(np.uint8)
        rows = chunk.shape[0]
        volume[written:written + rows] = chunk
        _grow_bounds(bounds, chunk, written)
        written += rows
        if progress:
            progress(0.5 * min(stop / img.shape[0], 1.0))

    volume.flush()
    del volume
    return _finish_bounds(bounds, shape)


def _grow_bounds(bounds, chunk, offset, floor=1, margin=2):
    """Widen the running data bounding box with one streamed chunk."""
    mask = chunk > floor
    if not mask.any():
        return
    for axis in AXES:
        hits = np.where(mask.any(axis=tuple(a for a in AXES if a != axis)))[0]
        shift = offset if axis == 0 else 0
        low = int(hits[0]) + shift - margin
        high = int(hits[-1]) + shift + margin + 1
        current = bounds[axis]
        current[0] = low if current[0] is None else min(current[0], low)
        current[1] = high if current[1] is None else max(current[1], high)


def _finish_bounds(bounds, shape):
    out = []
    for axis in AXES:
        low, high = bounds[axis]
        if low is None:
            out.append((0, shape[axis]))
        else:
            out.append((max(low, 0), min(high, shape[axis])))
    return out


def _write_npz(target, scratch, shape, bbox, source_affine, zooms, factors, mm,
               window, source, slab_bytes, compresslevel, progress):
    """Write every slice of every axis, reading the scratch volume in slabs."""
    import io
    import zipfile

    volume = np.memmap(scratch, dtype=np.uint8, mode="r", shape=shape)
    cropped = tuple(high - low for low, high in bbox)
    offset = [low for low, _ in bbox]

    # Cropping to the data shifts the origin, and the affine has to follow or
    # every slider reads the wrong millimetre.
    affine = np.asarray(source_affine, float).copy()
    affine[:3, :3] = affine[:3, :3] @ np.diag(factors.astype(float))
    affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ np.asarray(offset, float)
    meta = {"shape": list(cropped), "affine": affine.tolist(), "mm": float(mm),
            "window": list(window), "source": str(source),
            "zooms": (zooms * factors).tolist()}

    def put(zf, member, array):
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, np.ascontiguousarray(array),
                                  allow_pickle=False)
        zf.writestr(f"{member}.npy", buffer.getvalue())

    total = sum(cropped) or 1
    done = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=compresslevel, allowZip64=True) as zf:
        put(zf, "meta", np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
        for axis in AXES:
            # Slabs are taken along the axis being written, so each read is a
            # run of contiguous bytes rather than a stride across the whole file.
            per_slice = int(np.prod([c for a, c in enumerate(cropped) if a != axis]))
            width = max(1, min(cropped[axis],
                               int(slab_bytes) // max(per_slice, 1)))
            for begin in range(0, cropped[axis], width):
                stop = min(begin + width, cropped[axis])
                index = [slice(low, high) for low, high in bbox]
                index[axis] = slice(bbox[axis][0] + begin, bbox[axis][0] + stop)
                slab = np.ascontiguousarray(
                    np.moveaxis(volume[tuple(index)], axis, 0))
                for k in range(slab.shape[0]):
                    put(zf, f"a{axis}_{begin + k:05d}", slab[k])
                done += slab.shape[0]
                if progress:
                    progress(0.5 + 0.5 * done / total)
    del volume


def list_backdrops(directory=None):
    directory = Path(directory or BACKDROP_DIR).expanduser()
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.npz")):
        try:
            with np.load(path) as store:
                meta = json.loads(bytes(store["meta"]).decode())
        except Exception:
            continue
        out.append({"name": path.stem, "path": str(path),
                    "shape": meta["shape"], "mm": meta["mm"],
                    "bytes": path.stat().st_size})
    return out


class Backdrop:
    """An open backdrop. Slices are read on demand and never all at once."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self._store = np.load(self.path)
        self.meta = json.loads(bytes(self._store["meta"]).decode())
        self.shape = tuple(self.meta["shape"])
        self.affine = np.asarray(self.meta["affine"], float)
        self.mm = float(self.meta["mm"])

    # -- geometry ----------------------------------------------------------
    @property
    def extent_mm(self):
        lo, hi = self.bounds()
        return tuple(np.round(hi - lo, 1))

    def bounds(self):
        """World-space (min, max) along each axis, corner to corner."""
        corners = np.array([[i, j, k, 1] for i in (0, self.shape[0] - 1)
                            for j in (0, self.shape[1] - 1)
                            for k in (0, self.shape[2] - 1)], float)
        world = (self.affine @ corners.T)[:3].T
        return world.min(axis=0), world.max(axis=0)

    def world_axis(self, axis):
        """Which WORLD axis this array axis runs along, and its direction.

        A backdrop's affine need not be diagonal, and assuming array axis 0 is
        world x is how a slider ends up moving the wrong plane.
        """
        column = self.affine[:3, axis]
        return int(np.argmax(np.abs(column))), float(np.sign(column[np.argmax(np.abs(column))]) or 1.0)

    def index_for(self, axis, world_mm):
        """Nearest slice index along ``axis`` to a world coordinate."""
        world_axis, _ = self.world_axis(axis)
        origin = self.affine[world_axis, 3]
        step = self.affine[world_axis, axis]
        if not step:
            return 0
        return int(np.clip(round((world_mm - origin) / step), 0, self.shape[axis] - 1))

    def world_for(self, axis, index):
        world_axis, _ = self.world_axis(axis)
        return float(self.affine[world_axis, 3] + self.affine[world_axis, axis] * index)

    # -- data --------------------------------------------------------------
    def slice(self, axis, index):
        index = int(np.clip(index, 0, self.shape[axis] - 1))
        return self._store[f"a{axis}_{index:05d}"]

    def plane_grid(self, axis, index):
        """World coordinates of every texel of a slice, shaped like the slice.

        Overlays are volumes in their own grids, so the only way to put one on
        this slice is to ask where each of its texels actually is and sample
        there.
        """
        others = [a for a in AXES if a != axis]
        u, v = np.meshgrid(np.arange(self.shape[others[0]]),
                           np.arange(self.shape[others[1]]), indexing="ij")
        ijk = np.empty(u.shape + (4,), dtype=float)
        ijk[..., axis] = index
        ijk[..., others[0]] = u
        ijk[..., others[1]] = v
        ijk[..., 3] = 1.0
        return (ijk @ self.affine.T)[..., :3]

    # Slices are stored per axis so that fanning through one reads a few
    # kilobytes. Rebuilding the whole array is the opposite access pattern, so
    # it is a separate call with its own guard rather than something the slice
    # path does by accident: the 100 um backdrop is 4.3 GB on disk and would
    # not fit in memory as a dense array.
    MAX_VOLUME_VOXELS = 300_000_000

    def volume(self):
        """The whole array, assembled from the axis-0 slices.

        Backdrops are quantised to uint8 when they are built, so this is the
        anatomy at display precision, not the source scan's values. That is the
        right precision for meshing an isosurface or shading a surface by its
        own anatomy; a statistical map still wants its original NIfTI.
        """
        n = int(np.prod(self.shape))
        if n > self.MAX_VOLUME_VOXELS:
            raise MemoryError(
                f"{Path(self.path).name} is {n / 1e9:.1f} G voxels; build a "
                f"coarser backdrop (mm=) before meshing it")
        out = np.empty(self.shape, dtype=np.uint8)
        for i in range(self.shape[0]):
            out[i] = self.slice(0, i)
        return out

    def close(self):
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

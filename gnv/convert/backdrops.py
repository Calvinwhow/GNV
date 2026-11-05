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

**Building never holds the volume, and only ever reads it in file order.**  A
NIfTI stores x fastest and z slowest, so whole z-planes are contiguous and an
x-slice is scattered through the entire file. Every pass reads runs of z-planes
(see ``_Source``); the x- and y-slices come from two re-ordered scratch copies
of the data's box, written in the same pass. Nothing larger than a slab is
resident, the ceiling is disk rather than RAM, and on a spinning disk the reads
are sequential rather than millions of seeks. It needs twice the box free as
scratch while it runs (see ``build_backdrop``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

__all__ = ["BACKDROP_DIR", "build_backdrop", "compress_backdrop", "expand_backdrop", "compressed_path", "list_backdrops", "Backdrop"]

# Outside the package: a library of these runs to hundreds of megabytes and has
# no business inside a source tree. Overridable, and created on first use.
BACKDROP_DIR = Path.home() / ".gnv" / "backdrops"
AXES = (0, 1, 2)


def _slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _block_mean(array, factors):
    """Mean over integer blocks, padding the tail so nothing is dropped."""
    pads = [(0, (-s) % f) for s, f in zip(array.shape, factors)]
    if any(p[1] for p in pads):
        array = np.pad(array, pads, mode="edge")
    shape = []
    for size, factor in zip(array.shape, factors):
        shape += [size // factor, factor]
    return array.reshape(shape).mean(axis=(1, 3, 5))


class _Source:
    """The voxels of a NIfTI, read in the order the file stores them.

    NIfTI keeps x fastest and z slowest, so a run of whole z-planes is one
    contiguous stretch of the file -- of a plain .nii, or of a .nii.gz
    decompressed front to back -- and every pass of a build reads that way. An
    x-slice is the opposite: one byte from every few thousand across the whole
    file, so reading the volume along x touches every byte of it for every
    slice taken.
    """

    def __init__(self, path, img, factors):
        self.path = path
        self.shape = tuple(int(n) for n in img.shape)
        self.dtype = np.dtype(img.get_data_dtype())
        self.offset = int(img.dataobj.offset)
        self.slope = float(img.dataobj.slope)
        self.inter = float(img.dataobj.inter)
        self.fz = int(factors[2])
        self.plane = self.shape[0] * self.shape[1] * self.dtype.itemsize

    def _open(self):
        import gzip

        return gzip.open(self.path, "rb") if self.path.name.lower().endswith(".gz") else open(self.path, "rb")

    def width(self, slab_bytes):
        """How many z-planes make one read of about ``slab_bytes``, as a whole number of z blocks."""
        return max(self.fz, int(slab_bytes) // max(self.plane, 1) // self.fz * self.fz)

    def slabs(self, width, start=0, stop=None):
        """``(z, planes)`` for z-planes ``start``..``stop`` in runs of ``width``, each an (X, Y, n) array of the stored values."""
        stop = self.shape[2] if stop is None else stop
        with self._open() as handle:
            handle.seek(self.offset + start * self.plane)
            for z in range(start, stop, width):
                n = min(width, stop - z)
                data = handle.read(n * self.plane)
                if len(data) != n * self.plane:
                    raise ValueError(f"{self.path.name} ends early, at z-plane {z}: the file is truncated")
                yield z, np.frombuffer(data, self.dtype).reshape((self.shape[0], self.shape[1], n), order="F")

    def window(self, low=1.0, high=99.5, samples=40):
        """The 1st..99.5th percentile of the nonzero values, from a scatter of z-planes."""
        picks = np.unique(np.linspace(0, self.shape[2] - 1, min(samples, self.shape[2])).astype(int))
        values = []
        with self._open() as handle:
            for z in picks:
                handle.seek(self.offset + int(z) * self.plane)
                plane = np.frombuffer(handle.read(self.plane), self.dtype)[::7]
                values.append(plane.astype(np.float32) * self.slope + self.inter)
        values = np.concatenate(values)
        values = values[np.isfinite(values) & (values != 0)]
        if not values.size:
            return 0.0, 1.0
        lo, hi = np.percentile(values, [low, high])
        return float(lo), float(hi if hi > lo else lo + 1.0)


def _converter(source, factors, lo, scale):
    """A function from stored planes to the archive's uint8, at its voxel size.

    An unsigned 8- or 16-bit source at native resolution -- what a high-res scan
    usually is -- is windowed with a lookup table instead of a float32 round
    trip, which on tens of billions of voxels is the difference between a build
    you can wait for and one you cannot."""
    native = all(int(f) == 1 for f in factors)
    if native and source.dtype.kind == "u" and source.dtype.itemsize <= 2:
        levels = np.arange(np.iinfo(source.dtype).max + 1, dtype=np.float32) * np.float32(source.slope) + np.float32(source.inter)
        table = np.clip((levels - lo) * scale, 0, 255).astype(np.uint8)
        return lambda planes: table[planes]

    def convert(planes):
        chunk = planes.astype(np.float32) * source.slope + source.inter
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        if not native:
            chunk = _block_mean(chunk, tuple(int(f) for f in factors))
        return np.clip((chunk - lo) * scale, 0, 255).astype(np.uint8)

    return convert


def build_backdrop(nifti, out_dir=None, mm=None, name=None, window=None, slab_bytes=1 << 30, compresslevel=4, workers=None, progress=None):
    """Convert a NIfTI into a streamable backdrop and return the ``.npz`` path.

    ``mm`` defaults to the source's own voxel size -- native. Pass a coarser
    number to trade detail for size; 0.4 is about 64x smaller than a 100 um scan
    and still finer than a 1 mm template.

    ``window`` is the intensity range mapped onto 0-255; it defaults to the 1st
    and 99.5th percentiles of the nonzero data, which keeps a bright skull or a
    single hot voxel from crushing the brain into the bottom of the range.
    ``slab_bytes`` is about how much of the volume is resident at once (a few
    times it, counting copies); ``workers`` how many threads compress slices.

    Four passes, all in the order the file is stored (see ``_Source``):

    1. read the file once to find the box the data fills, so the archive and
       the scratch hold only that;
    2. read it again: write the z-slices straight into the archive, and copy
       the box twice to scratch, re-ordered so that a run of x-slices and a run
       of y-slices are each one contiguous read;
    3. write the y-slices from their copy, and delete it;
    4. write the x-slices from theirs.

    The scratch is twice the box, beside the archive if that disk also has room
    for the archive, otherwise beside the source; there is a clear error up
    front if neither does, rather than a full disk hours in.
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

    report = progress or (lambda fraction: None)
    source = _Source(path, img, factors)
    lo, hi = window if window else source.window()
    convert = _converter(source, factors, lo, 255.0 / (hi - lo))
    width = source.width(slab_bytes)

    out_dir = Path(out_dir or BACKDROP_DIR).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{_slug(name or path.name.split('.nii')[0])}.npz"

    bbox = _data_box(source, convert, shape, width, lambda f: report(0.3 * f))
    meta = _meta(bbox, img.affine, zooms, factors, mm, (lo, hi), path)
    _assemble(target, meta, _box_chunks(source, convert, bbox, width), path, slab_bytes, compresslevel, workers, lambda f: report(0.3 + 0.7 * f))
    return target


def _data_box(source, convert, shape, width, report):
    """Pass 1: the box, in archive voxels, that holds everything above background."""
    bounds = [[None, None], [None, None], [None, None]]
    for z, planes in source.slabs(width):
        _grow_bounds(bounds, convert(planes), z // source.fz)
        report(min(z + width, source.shape[2]) / source.shape[2])
    return _finish_bounds(bounds, shape)


def _grow_bounds(bounds, chunk, offset, floor=1, margin=2):
    """Widen the running data bounding box with one chunk of z-planes, the first at ``offset``."""
    mask = chunk > floor
    if not mask.any():
        return
    for axis in AXES:
        hits = np.where(mask.any(axis=tuple(a for a in AXES if a != axis)))[0]
        shift = offset if axis == 2 else 0
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


def _scratch_dir(target, near, scratch_bytes, archive_bytes):
    """Where the re-ordered copies go: beside the archive if its disk holds them
    and the archive (about 1.2x the box: three axes of a noisy high-res scan
    compress about 2.8x), otherwise beside ``near`` -- the source being read."""
    import os
    import shutil

    home = os.stat(target.parent).st_dev
    for folder in (target.parent, near.parent):
        need = scratch_bytes + (archive_bytes if os.stat(folder).st_dev == home else 0)
        if shutil.disk_usage(folder).free >= need:
            return folder
    gb = 1e9
    raise OSError(f"building {target.name} needs {scratch_bytes / gb:.0f} GB of scratch space while it runs, plus about "
                  f"{archive_bytes / gb:.0f} GB for the archive in {target.parent}; free "
                  f"{shutil.disk_usage(target.parent).free / gb:.0f} GB there, or "
                  f"{shutil.disk_usage(near.parent).free / gb:.0f} GB beside {near.name}, and try again")


def _meta(bbox, source_affine, zooms, factors, mm, window, origin):
    """The archive's header. Cropping to the data shifts the origin, and the
    affine has to follow or every slider reads the wrong millimetre."""
    (x0, x1), (y0, y1), (z0, z1) = bbox
    affine = np.asarray(source_affine, float).copy()
    affine[:3, :3] = affine[:3, :3] @ np.diag(factors.astype(float))
    affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ np.asarray([x0, y0, z0], float)
    return {"shape": [x1 - x0, y1 - y0, z1 - z0], "affine": affine.tolist(), "mm": float(mm),
            "window": list(window), "source": str(origin), "zooms": (zooms * factors).tolist()}


def _box_chunks(source, convert, bbox, width):
    """``(z, chunk)``: the box's z-planes read from the source, as uint8 (x, y, n) chunks."""
    (x0, x1), (y0, y1), (z0, z1) = bbox
    fz = source.fz
    for z, planes in source.slabs(width, z0 * fz, min(z1 * fz, source.shape[2])):
        yield z // fz - z0, convert(planes)[x0:x1, y0:y1]


def _assemble(target, meta, chunks, near, slab_bytes, compresslevel, workers, report):
    """Write the archive ``target`` from its z-planes (see ``_write_archive``),
    beside it as a partial file renamed into place when complete, with its
    scratch folder removed whatever happens."""
    import shutil

    box = int(np.prod(meta["shape"]))
    scratch = _scratch_dir(target, Path(near), 2 * box, int(1.2 * box)) / f".{target.stem}.building"
    partial = target.with_name(f".{target.name}.partial")
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        _write_archive(partial, meta, chunks, scratch, slab_bytes, compresslevel, workers, report)
        partial.replace(target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        partial.unlink(missing_ok=True)


def _write_archive(target, meta, chunks, scratch, slab_bytes, compresslevel, workers, report):
    """Every slice of every axis into the archive, from ``chunks`` of z-planes
    in order (see ``build_backdrop``): the z-slices as they arrive, the box
    copied twice to scratch re-ordered for x and y, then the y-slices and the
    x-slices from those copies. ``report`` gets 0..1."""
    nx, ny, nz = meta["shape"]

    # The re-ordered copies: block b of ``by_x`` holds x-slices b*bx.. as one
    # contiguous (z, y, bx) run, and ``by_y`` likewise for y. Each z-chunk
    # lands in every block as one contiguous write.
    bx = max(1, min(nx, int(slab_bytes) // (ny * nz)))
    by = max(1, min(ny, int(slab_bytes) // (nx * nz)))
    by_x = np.memmap(scratch / "by_x.raw", dtype=np.uint8, mode="w+", shape=(-(-nx // bx), nz, ny, bx))
    by_y = np.memmap(scratch / "by_y.raw", dtype=np.uint8, mode="w+", shape=(-(-ny // by), nz, by, nx))

    with _Archive(target, compresslevel, workers) as archive:
        archive.put("meta", np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
        for k, chunk in chunks:
            n = chunk.shape[2]
            for i in range(n):
                archive.put(f"a2_{k + i:05d}", chunk[:, :, i])
            for b in range(by_x.shape[0]):
                part = chunk[b * bx:(b + 1) * bx]
                by_x[b, k:k + n, :, :part.shape[0]] = part.transpose(2, 1, 0)
            for b in range(by_y.shape[0]):
                part = chunk[:, b * by:(b + 1) * by]
                by_y[b, k:k + n, :part.shape[1], :] = part.transpose(2, 1, 0)
            report(0.5 * (k + n) / nz)
        by_x.flush()
        by_y.flush()

        for b in range(by_y.shape[0]):
            block = np.array(by_y[b])                 # (z, by, x): one contiguous read
            for i in range(min(by, ny - b * by)):
                archive.put(f"a1_{b * by + i:05d}", block[:, i, :].T)
            report(0.5 + 0.25 * (b + 1) / by_y.shape[0])
        del by_y
        (scratch / "by_y.raw").unlink()

        for b in range(by_x.shape[0]):
            block = np.array(by_x[b])                 # (z, y, bx)
            for i in range(min(bx, nx - b * bx)):
                archive.put(f"a0_{b * bx + i:05d}", block[:, :, i].T)
            report(0.75 + 0.25 * (b + 1) / by_x.shape[0])
        del by_x


# -- compressed copies ------------------------------------------------------------
# An archive holds its volume three times, once per axis, so any slice is one
# quick read. One kept for later rather than for now can be shrunk to a single
# copy -- its z-slices, bzip2 at its strongest, about a quarter of the archive --
# and rebuilt from it when it is next chosen. Both are standard .npz files.

COMPRESSED_DIR = "compressed"


def compressed_path(path):
    """Where the compressed copy of the archive ``path`` is kept: ``compressed/<name>.npz`` beside it."""
    path = Path(path)
    return path.parent / COMPRESSED_DIR / path.name


def _read_planes(path, count, workers=None):
    """``(k, slice)`` for the first ``count`` z-slices of an archive, in order,
    read and decompressed by several threads, each with its own handle."""
    import os
    import threading
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    local, opened = threading.local(), []

    def read(k):
        if not hasattr(local, "store"):
            local.store = np.load(path)
            opened.append(local.store)
        return local.store[f"a2_{k:05d}"]

    workers = workers or os.cpu_count() or 4
    try:
        with ThreadPoolExecutor(workers) as pool:
            pending = deque()
            for k in range(count):
                pending.append((k, pool.submit(read, k)))
                if len(pending) > 2 * workers:
                    j, future = pending.popleft()
                    yield j, future.result()
            while pending:
                j, future = pending.popleft()
                yield j, future.result()
    finally:
        for store in opened:
            store.close()


def compress_backdrop(path, workers=None, progress=None):
    """Replace the archive ``path`` with its compressed copy (see above) and
    return ``(bytes before, bytes after)``. The copy is complete and in place
    before the archive is deleted."""
    import os
    import zipfile

    path = Path(path)
    out = compressed_path(path)
    out.parent.mkdir(exist_ok=True)
    partial = out.with_name(f".{out.name}.partial")
    report = progress or (lambda fraction: None)
    with np.load(path) as store:
        meta = json.loads(bytes(store["meta"]).decode())
    nz = meta["shape"][2]
    # The archive's date goes with it, so the volume levels built from it are
    # still current when it comes back.
    header = dict(meta, archive_mtime=os.path.getmtime(path))
    try:
        with _Archive(partial, 9, workers, zipfile.ZIP_BZIP2) as archive:
            archive.put("meta", np.frombuffer(json.dumps(header).encode(), dtype=np.uint8))
            for k, plane in _read_planes(path, nz, workers):
                archive.put(f"a2_{k:05d}", plane)
                report((k + 1) / nz)
        partial.replace(out)
    finally:
        partial.unlink(missing_ok=True)
    before = path.stat().st_size
    path.unlink()
    return before, out.stat().st_size


def expand_backdrop(path, slab_bytes=1 << 30, compresslevel=4, workers=None, progress=None):
    """Rebuild the archive ``path`` from its compressed copy and delete the copy.
    It comes back as it was: the same slices, header and date."""
    import os

    path = Path(path)
    compact = compressed_path(path)
    with np.load(compact) as store:
        meta = json.loads(bytes(store["meta"]).decode())
    mtime = meta.pop("archive_mtime", None)
    nx, ny, nz = meta["shape"]
    width = max(1, int(slab_bytes) // max(nx * ny, 1))

    def chunks():
        planes = _read_planes(compact, nz, workers)
        for k in range(0, nz, width):
            chunk = np.empty((nx, ny, min(width, nz - k)), np.uint8, order="F")
            for i in range(chunk.shape[2]):
                chunk[:, :, i] = next(planes)[1]
            yield k, chunk

    _assemble(path, meta, chunks(), compact, slab_bytes, compresslevel, workers, progress or (lambda fraction: None))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    compact.unlink()
    try:
        compact.parent.rmdir()          # only when it is empty
    except OSError:
        pass
    return path


def _pack(name, array, level, method):
    """One slice as a compressed ``.npy`` member: ``(name, method, size, crc, data)``."""
    import bz2
    import io
    import zipfile
    import zlib

    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    raw = buffer.getvalue()
    if method == zipfile.ZIP_BZIP2:
        data = bz2.compress(raw, level)
    else:
        packer = zlib.compressobj(level, zlib.DEFLATED, -15)
        data = packer.compress(raw) + packer.flush()
    return f"{name}.npy", method, len(raw), zlib.crc32(raw), data


class _Archive:
    """A zip of ``.npy`` members, compressed by several threads and written in order.

    Compressing the slices is most of a build's time, and zlib and bz2 let go of
    the GIL, so slices are compressed in parallel and their bytes written as
    finished members: a standard zip, read by ``np.load`` like any ``.npz``.
    ``method`` is deflate (zlib) or bzip2.
    """

    def __init__(self, path, level, workers=None, method=None):
        import os
        import zipfile
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        self.zip = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True)
        self.level = level
        self.method = zipfile.ZIP_DEFLATED if method is None else method
        workers = workers or os.cpu_count() or 4
        self.pool = ThreadPoolExecutor(workers)
        self.pending = deque()
        self.limit = 2 * workers

    def put(self, name, array):
        self.pending.append(self.pool.submit(_pack, name, array, self.level, self.method))
        while len(self.pending) > self.limit:
            self._write(*self.pending.popleft().result())

    def _write(self, name, method, size, crc, data):
        import time
        import zipfile

        info = zipfile.ZipInfo(name, date_time=time.localtime(time.time())[:6])
        info.compress_type = method
        info.file_size, info.compress_size, info.CRC = size, len(data), crc
        info.external_attr = 0o644 << 16
        zf = self.zip
        info.header_offset = zf.fp.tell()
        zf.fp.write(info.FileHeader(zip64=size > zipfile.ZIP64_LIMIT))
        zf.fp.write(data)
        zf.filelist.append(info)
        zf.NameToInfo[name] = info
        zf.start_dir = zf.fp.tell()
        zf._didModify = True

    def __enter__(self):
        return self

    def __exit__(self, kind, *rest):
        try:
            if kind is None:
                while self.pending:
                    self._write(*self.pending.popleft().result())
        finally:
            for future in self.pending:
                future.cancel()
            self.pool.shutdown(wait=True)
            self.zip.close()


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

    def plane_grid(self, axis, index, shape=None):
        """World coordinates of every texel of a slice, shaped like the slice.

        Overlays are volumes in their own grids, so the only way to put one on
        this slice is to ask where each of its texels actually is and sample
        there. ``shape`` asks for a finer grid -- that many points spanning the
        same first to last voxel centres, as a slice resampled for drawing has.
        """
        others = [a for a in AXES if a != axis]
        n0, n1 = self.shape[others[0]], self.shape[others[1]]
        m0, m1 = shape or (n0, n1)
        u, v = np.meshgrid(np.linspace(0, n0 - 1, m0), np.linspace(0, n1 - 1, m1), indexing="ij")
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

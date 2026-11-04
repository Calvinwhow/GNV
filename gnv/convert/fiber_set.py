"""A fibre set: a folder of tracts, each described by its own descriptor.

    <tract>.fib.json   the descriptor (schema calvin_utils.fiber_values)
    <tract>.fib.npy    one value per fibre of its atlas
    <tract>.npz        the fibres, in the fibre-atlas format FiberAtlasConverter
                       writes -- or, for results copied whole, one <atlas>.npz
                       in the set that several descriptors share

``build_fiber_set`` converts one source, or a folder of them, into a set. Every
source is one tract, named after its file:

- a tract file (.tck .trk .trx .mat .npz) becomes its own trio; a Lead-DBS
  result keeps its ``vals``, anything else gets 1 on every fibre;
- one of your results (a ``.fib.json`` and its ``.fib.npy``) is copied whole:
  its values, its descriptor re-pointed into the set, and its atlas -- copied
  once per set however many results colour it.

``keep_percent`` instead keeps only the strongest fibres of each tract by
|value| and writes each as its own small trio, so a result drawn from a
500 MB atlas no longer needs that atlas. Tracts whose values are all equal
(a .tck's 1s) are kept whole either way.

A set folder may also hold a ``regions.json`` (``resource_index.read_regions``):
one line per tract giving the section it is grouped under, a hint saying what
it is, and optionally a ``color`` -- exactly as for an atlas's regions.
"""

from __future__ import annotations

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.fiber_descriptor import is_descriptor

GEOMETRY_SUFFIXES = (".tck", ".trk", ".trx", ".mat", ".npz")
DESCRIPTOR_SUFFIX = ".fib.json"


def tract_name(path):
    """A tract's name: its file name up to the first dot ("Tremor.mat" -> "Tremor")."""
    return Path(path).name.split(".")[0]


def is_fibre_npz(path):
    """Whether an ``.npz`` holds fibres (a ``fibers`` key) rather than anything else
    a results folder keeps in one (an optimisation history, a backdrop)."""
    try:
        with np.load(path, allow_pickle=False) as found:
            return "fibers" in found.files
    except Exception:
        return False


def tract_files(source):
    """``{tract name: file}`` for ``source``: itself, or every tract file and
    descriptor under its folder, at any depth. A file in a subfolder is named by
    its path ("fiber_motor/optimized_map.fib.json" -> "fiber_motor_optimized_map"),
    so the same file name in several result folders stays several tracts. An
    atlas that a descriptor in the folder colours is that result's geometry, not
    a tract of its own. One per name, in natural order."""
    from gnv.convert.volume_meshes import _natural

    source = Path(source).expanduser()
    if source.is_file():
        return {tract_name(source): source}
    found = {}
    for path in sorted((p for p in source.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(source).parts)), key=lambda p: [_natural(Path(part)) for part in p.relative_to(source).parts]):
        lower = path.name.lower()
        tract = is_descriptor(path) or (lower.endswith(GEOMETRY_SUFFIXES) and not lower.endswith(".npz")) or (lower.endswith(".npz") and is_fibre_npz(path))
        if tract:
            found.setdefault("_".join([*path.relative_to(source).parent.parts, tract_name(path)]), path)
    coloured = set()
    for path in found.values():
        if is_descriptor(path):
            try:
                coloured.add(descriptor_files(path)[1].resolve())
            except (OSError, ValueError):
                pass                          # reported when the result is converted
    return {name: path for name, path in found.items() if path.resolve() not in coloured}


@lru_cache(maxsize=1)
def _atlas_fibres(path, mtime_ns):
    from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import FiberFormatConverter

    return [FiberFormatConverter._as_streamline(f) for f in FiberFormatConverter._load_fibers(Path(path))]


def atlas_fibres(path):
    """The fibres of a fibre atlas, held between calls: the tracts of one set
    usually share an atlas, and reading 500 MB once per tract is what made a set
    slow to toggle."""
    path = Path(path).expanduser().resolve()
    return _atlas_fibres(str(path), path.stat().st_mtime_ns)


def descriptor_files(descriptor):
    """``(values .fib.npy, atlas)`` that a descriptor names, verified as
    ``FiberFormatConverter`` verifies them (checksum, atlas size)."""
    from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import FiberFormatConverter

    values = FiberFormatConverter.values_from_description(descriptor)
    return values, FiberFormatConverter.atlas_from_values_description(values)


def tract_fibres(values_path, sign="both", min_abs_value=None, top_percent=None):
    """``(fibres, values, ids, total)`` for one tract of a set: its atlas's
    fibres that pass the value filters (``FiberFormatConverter._fiber_value_mask``),
    their values, their numbers in the atlas, and how many fibres the atlas holds."""
    from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import FiberFormatConverter

    values_path = Path(values_path)
    atlas = FiberFormatConverter.atlas_from_values_description(values_path)
    values = np.asarray(np.load(values_path), dtype=np.float32)
    fibres = atlas_fibres(atlas)
    if len(fibres) != len(values):
        raise ValueError(f"{atlas.name} holds {len(fibres)} fibres but {values_path.name} has {len(values)} values")
    keep = FiberFormatConverter._fiber_value_mask(values, sign, min_abs_value, top_percent)
    ids = np.flatnonzero(keep)
    return [fibres[i] for i in ids], values[ids], ids, len(values)


def read_tract(path):
    """``(fibres, values)`` from one tract file or descriptor.

    ``fibres`` is a list of N x 3 float32 arrays in MNI mm; ``values`` one
    float32 per fibre -- a descriptor's values, a Lead-DBS result's ``vals``,
    else all 1. Fibres with fewer than two finite points are dropped together
    with their values.
    """
    from calvin_utils.neuroimaging_utils.tract_utils.fiber_atlas_converter import FiberAtlasConverter
    from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import FiberFormatConverter

    path = Path(path).expanduser()
    if is_descriptor(path):
        values_path, atlas = descriptor_files(path)
        fibres, values = atlas_fibres(atlas), np.load(values_path)
    elif path.suffix.lower() == ".mat":
        fibres, _lengths, values = FiberFormatConverter.load_leaddbs_mat(path)
    else:
        reader = FiberAtlasConverter(input_path=path, out_path=path.with_name(f"{tract_name(path)}.npz"))
        reader.load_fibers()
        fibres, values = reader.fibers, None
    values = np.ones(len(fibres), np.float32) if values is None else np.asarray(values, np.float32)
    fibres = [np.asarray(f, np.float32)[:, :3] for f in fibres]
    fibres = [f[np.isfinite(f).all(axis=1)] for f in fibres]
    keep = np.asarray([len(f) > 1 for f in fibres], bool)
    return [f for f, k in zip(fibres, keep) if k], values[keep]


def strongest(fibres, values, keep_percent):
    """The ``keep_percent`` % of fibres with the largest |value|; all of them
    when every value is the same (there is nothing to rank)."""
    from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import FiberFormatConverter

    if keep_percent is None or not len(values) or np.ptp(values) == 0:
        return fibres, values
    keep = FiberFormatConverter._fiber_value_mask(values, "both", None, keep_percent)
    return [f for f, k in zip(fibres, keep) if k], values[keep]


def write_tract(fibres, values, out_dir, name, source=None):
    """Write one tract's trio into ``out_dir``; returns the descriptor's path."""
    from calvin_utils.neuroimaging_utils.tract_utils.fiber_atlas_converter import FiberAtlasConverter
    from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO

    out_dir = Path(out_dir)
    atlas = out_dir / f"{name}.npz"
    writer = FiberAtlasConverter(input_path=Path(source) if source else atlas, out_path=atlas)
    writer.fibers = fibres
    writer.build_metadata()
    writer.save_npz()
    values_path = out_dir / f"{name}.fib.npy"
    np.save(values_path, np.asarray(values, np.float32))
    return FiberIO.write_values_description(values_path, atlas, source_geometry_path=source)


def copy_result(descriptor, out_dir, name, atlases):
    """Copy one of your results into the set whole: its values as
    ``<name>.fib.npy``, its atlas (once per set -- ``atlases`` maps each source
    atlas to its copy), and its descriptor as ``<name>.fib.json`` with every
    path re-pointed into the set. Everything else it records is kept."""
    values_path, atlas = descriptor_files(descriptor)
    out_dir = Path(out_dir)
    if atlas not in atlases:
        copy = out_dir / atlas.name
        while copy.exists():
            copy = copy.with_name(f"{copy.stem}_{copy.suffix}")
        shutil.copy2(atlas, copy)
        atlases[atlas] = copy
    values_copy = out_dir / f"{name}.fib.npy"
    shutil.copy2(values_path, values_copy)
    desc = json.loads(Path(descriptor).read_text())
    desc.update(values_file=values_copy.name, values_path=values_copy.name, values_relative_path=values_copy.name, converted_from=str(Path(descriptor).expanduser().resolve()))
    desc["values"] = dict(desc.get("values") or {}, path=values_copy.name, relative_path=values_copy.name)
    desc["fiber_atlas"] = dict(desc.get("fiber_atlas") or {}, path=atlases[atlas].name, relative_path=atlases[atlas].name)
    out = out_dir / f"{name}{DESCRIPTOR_SUFFIX}"
    out.write_text(json.dumps(desc, indent=2) + "\n")
    return out


def build_fiber_set(source, out_dir, keep_percent=None, progress=None):
    """Convert a tract file or result, or a folder of them, into the fibre set
    ``out_dir`` (see the module docstring for what each becomes).

    Written beside ``out_dir`` and renamed into place, so a failed build never
    leaves a half set. Returns ``{"tracts": n, "skipped": [names with no fibres]}``.
    """
    files = tract_files(source)
    if not files:
        raise ValueError(f"{source} holds no tract files ({', '.join(GEOMETRY_SUFFIXES)}) or {DESCRIPTOR_SUFFIX} results")
    out_dir = Path(out_dir).expanduser()
    report = progress or (lambda fraction: None)
    partial = out_dir.with_name(f".{out_dir.name}.{os.getpid()}.partial")
    partial.mkdir(parents=True)
    written, skipped, atlases = 0, [], {}
    try:
        for done, (name, path) in enumerate(files.items(), start=1):
            if keep_percent is None and is_descriptor(path):
                copy_result(path, partial, name, atlases)
                written += 1
            else:
                fibres, values = strongest(*read_tract(path), keep_percent)
                if fibres:
                    write_tract(fibres, values, partial, name, source=path)
                    written += 1
                else:
                    skipped.append(name)
            report(done / len(files))
        if not written:
            raise ValueError(f"no fibres in {source}")
        partial.rename(out_dir)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return {"tracts": written, "skipped": skipped}


def set_tracts(folder):
    """The tracts in the set ``folder``: ``[{"name", "values", "count", "section",
    "hint", "color"}]``, one per ``<name>.fib.json``.

    ``values`` is the tract's ``.fib.npy`` (its descriptor finds the geometry)
    and ``count`` how many fibres its atlas holds, read from the descriptor so
    listing a set never loads one. Listed in its ``regions.json`` order, then
    by name.
    """
    from gnv.convert.volume_meshes import _natural
    from gnv.resource_index import read_regions

    folder = Path(folder).expanduser()
    table = read_regions(folder)
    out = []
    for desc in sorted(folder.glob(f"*{DESCRIPTOR_SUFFIX}"), key=_natural):
        name = desc.name[:-len(DESCRIPTOR_SUFFIX)]
        info = json.loads(desc.read_text())
        row = table.get(name, {})
        out.append({"name": name, "values": str(folder / info.get("values_file", f"{name}.fib.npy")), "count": int((info.get("fiber_atlas") or {}).get("fiber_count") or 0), "section": row.get("section"), "hint": row.get("hint", ""), "color": row.get("color")})
    rank = {name: i for i, name in enumerate(table)}
    return sorted(out, key=lambda t: rank.get(t["name"], len(rank)))

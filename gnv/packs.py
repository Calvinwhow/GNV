"""Content packs: library items in one file, to hand on.

A pack is a zip named ``*.gnvpack``. Inside, files sit exactly where they sit in a
library (see ``paths``), with a ``manifest.json`` naming each item and carrying
its index entry:

    manifest.json   {"format": "gnv pack", "version": 2,
                     "title": ..., "notes": ...,
                     "items": [{"kind": "meshes", "name": ...,
                                "entry": {...its index entry, paths relative...}}]}
    meshes/atlases/DISTAL_Nano/...           a folder of region meshes
    meshes/brain_surfaces/lead_dbs/surf.vtp  a surface's files
    backdrops/full16_100um.npz               a backdrop, with its volume levels if built
    fibers/rajamani2025/...                  a fibre set: three files per tract

``notes`` is free text from whoever made the pack: where the data came from,
how to cite it, its licence. Any mesh, backdrop or fibre set the indexes list
can be packed: atlases, surfaces, backdrops, fibre sets.

Installing puts each item's files in this user's library and its entry in the
library's index, under the section it came from. An item whose name the viewer
already lists is left alone and reported -- never overwritten. Each item is
unpacked beside its destination and moved into place, so a failed install leaves
nothing half-made.

Members are stored, not compressed: the meshes are small and a backdrop is
already compressed, and a stored member unpacks at copy speed even when it is
gigabytes.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from .paths import LIBRARY_DIR, add_entry, entries, index, relative_entry

FORMAT = "gnv pack"
VERSION = 2
SUFFIX = ".gnvpack"
KINDS = ("meshes", "backdrops", "fibers")


def _entry_paths(entry):
    """The relative paths an index entry (as stored) names: files and folders."""
    paths = [entry[k] for k in ("file", "parcels", "tracts") if k in entry]
    return paths + [p for p in entry.get("pieces", []) if "/" in str(p)]


def _item_files(kind, entry):
    """``[(file on disk, path inside the pack)]`` for one indexed item."""
    root = Path(entry["root"])
    out = []
    for rel in _entry_paths(relative_entry(entry)):
        path = root / rel
        found = sorted(p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")) if path.is_dir() else [path]
        if kind == "backdrops":
            found += sorted(path.parent.glob(f"{path.stem}.lod*.npy"))
        missing = [p for p in found if not p.is_file()]
        if missing or not found:
            raise FileNotFoundError(f"{entry['name']}: {missing[0] if missing else path} is not there")
        out += [(p, f"{kind}/{p.relative_to(root).as_posix()}") for p in found]
    return out


def make_pack(items, out_path, title="", notes="", progress=None):
    """Write indexed ``items`` (``[{"kind", "name"}]``) to the pack ``out_path``.
    Returns ``{"path", "items", "bytes"}``."""
    out_path = Path(out_path).expanduser()
    if not out_path.name.lower().endswith(SUFFIX):
        out_path = out_path.with_name(out_path.name + SUFFIX)
    if not items:
        raise ValueError("a pack needs at least one item")
    listed, files = [], []
    for item in items:
        kind, name = str(item["kind"]), str(item["name"])
        if kind not in KINDS:
            raise ValueError(f"an item's kind is one of {KINDS}, not {kind!r}")
        entry = entries(kind).get(name)
        if entry is None:
            raise FileNotFoundError(f"there is no {kind[:-1]} called {name!r}")
        files += _item_files(kind, entry)
        listed.append({"kind": kind, "name": name, "entry": relative_entry(entry)})
    total = sum(path.stat().st_size for path, _ in files) or 1
    manifest = {"format": FORMAT, "version": VERSION, "title": title or out_path.stem, "notes": notes, "items": listed}
    report = progress or (lambda fraction: None)
    partial = out_path.with_name(f".{out_path.name}.{os.getpid()}.partial")
    done = 0
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as pack:
            pack.writestr("manifest.json", json.dumps(manifest, indent=1))
            for path, arcname in dict.fromkeys(files):
                pack.write(path, arcname)
                done += path.stat().st_size
                report(done / total)
        os.replace(partial, out_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {"path": str(out_path), "items": len(listed), "bytes": out_path.stat().st_size}


def read_manifest(path):
    """A pack's manifest, checked to be one this viewer can install."""
    try:
        with zipfile.ZipFile(Path(path).expanduser()) as pack:
            manifest = json.loads(pack.read("manifest.json"))
    except zipfile.BadZipFile:
        raise ValueError(f"{Path(path).name} is not a General Neuroimage Viewer pack (not a zip file)")
    except KeyError:
        raise ValueError(f"{Path(path).name} is not a General Neuroimage Viewer pack: it has no manifest.json")
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{Path(path).name} is not a General Neuroimage Viewer pack")
    if int(manifest.get("version", 0)) != VERSION:
        raise ValueError(f"{Path(path).name} is a version {manifest.get('version')} pack; this viewer reads version {VERSION}")
    return manifest


def _safe(rel):
    """Whether a relative path stays inside the folder it is relative to."""
    parts = PurePosixPath(str(rel)).parts
    return bool(parts) and not str(rel).startswith("/") and ".." not in parts and "\\" not in str(rel)


def install_pack(path, progress=None):
    """Put every item of the pack at ``path`` into this user's library and its
    index. Returns ``{"installed": [names], "skipped": [{"name", "reason"}], "notes"}``."""
    manifest = read_manifest(path)
    report = progress or (lambda fraction: None)
    installed, skipped = [], []
    with zipfile.ZipFile(Path(path).expanduser()) as pack:
        members = [m for m in pack.infolist() if not m.is_dir()]
        for m in members:
            if m.filename != "manifest.json" and not _safe(m.filename):
                raise ValueError(f"{Path(path).name} has an unsafe entry, {m.filename!r}; not installing it")
        total = sum(m.file_size for m in members) or 1
        done = 0
        for item in manifest.get("items", []):
            kind, name, entry = item.get("kind"), str(item.get("name")), dict(item.get("entry") or {})
            if kind not in KINDS or entry.get("name") != name or not all(_safe(r) for r in _entry_paths(entry)):
                skipped.append({"name": name, "reason": "its manifest entry is not one this viewer can read"})
                continue
            if name in entries(kind):
                skipped.append({"name": name, "reason": "the viewer already has one by this name"})
                continue
            rels = _entry_paths(entry)
            mine = [m for m in members if any(m.filename == f"{kind}/{r}" or m.filename.startswith(f"{kind}/{r}/") or (kind == "backdrops" and m.filename.startswith(f"{kind}/{PurePosixPath(r).stem}.lod")) for r in rels)]
            targets = [LIBRARY_DIR / m.filename for m in mine]
            if not mine:
                skipped.append({"name": name, "reason": "its files are missing from the pack"})
                continue
            if any(t.exists() for t in targets):
                skipped.append({"name": name, "reason": "files of that name are already in the library"})
                continue
            staging = LIBRARY_DIR / f".{name}.{os.getpid()}.installing"
            shutil.rmtree(staging, ignore_errors=True)
            try:
                for m in mine:
                    dest = staging / m.filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with pack.open(m) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out, 1 << 22)
                    done += m.file_size
                    report(done / total)
                for m, target in zip(mine, targets):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging / m.filename, target)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            add_entry(kind, {"name": name, "section": entry.get("section", "Installed"), **entry})
            installed.append(name)
    return {"installed": installed, "skipped": skipped, "notes": manifest.get("notes", "")}


def library():
    """Everything a pack can be made from, in dropdown order:
    ``{"dir", "meshes": [...], "backdrops": [...]}``, each ``{"name", "label", "section", "shipped", "bytes"}``."""
    def size(kind, entry):
        try:
            return sum(p.stat().st_size for p, _ in _item_files(kind, entry))
        except FileNotFoundError:
            return 0
    listing = {kind: [{"name": e["name"], "label": e.get("label", e["name"]), "section": s["label"], "shipped": e["shipped"], "bytes": size(kind, e)} for s in index(kind) for e in s["entries"] if _entry_paths(relative_entry(e))] for kind in KINDS}
    return {"dir": str(LIBRARY_DIR), **listing}

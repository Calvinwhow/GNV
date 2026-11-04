"""Conversions as background jobs: backdrops, volume levels, atlases, fibre sets.

Converting a 100 um volume is minutes of work -- streaming 8.7 GB through a
block-average -- so it cannot happen inside a request. One job runs at a time,
its progress is polled, and the library it writes into accumulates: every volume
converted once is available to every figure afterwards.

A converted volume is also a volume-layer BODY. One too large to ray-cast whole
is drawn from block-averaged levels, so the job builds those right after the
archive, and ``start_levels`` builds them for an archive that predates this.

``start_atlas`` meshes an atlas once into the prepared-atlas library, where it
joins the mesh list as a region set; ``start_fibers`` converts tract files into a
fibre set for the Fiber set list. ``start_install`` and ``start_pack`` unpack
and write content packs (see ``packs``). ``kind`` in the status says which job
ran, and ``result`` holds what a pack job reports.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread

from .paths import ATLAS_DIR, BACKDROP_DIR, FIBER_DIR, add_entry, entries, index

_state = {"running": False, "progress": 0.0, "name": None,
          "error": None, "path": None, "kind": None, "skipped": [], "result": None}
_lock = Lock()


def catalogue():
    """The backdrop dropdowns: both backdrop indexes merged (paths.index), each
    entry with what its archive says about itself."""
    from gnv.convert.backdrops import Backdrop, list_backdrops

    items, read = [], {}
    for section in index("backdrops"):
        for spec in section["entries"]:
            folder = str(Path(spec["file"]).parent)
            if folder not in read:
                read[folder] = {e["path"]: e for e in list_backdrops(folder)}
            entry = dict(read[folder].get(spec["file"], {"path": spec["file"], "error": "file not found"}), name=spec["name"], label=spec.get("label", spec["name"]), hint=spec.get("hint", ""), section=section["label"], shipped=spec["shipped"])
            if "error" not in entry:
                try:
                    with Backdrop(entry["path"]) as back:
                        lo, hi = back.bounds()
                        entry["bounds"] = [list(map(float, lo)), list(map(float, hi))]
                        entry["world_axes"] = [back.world_axis(a)[0] for a in (0, 1, 2)]
                except Exception as exc:
                    entry["error"] = str(exc)
            items.append(entry)
    return {"dir": str(BACKDROP_DIR), "backdrops": items, "job": status()}


def status():
    with _lock:
        return dict(_state)


def start(path, mm=None, name=None):
    """Convert a NIfTI into a backdrop in the library, entered in its index."""
    from gnv.convert.backdrops import _slug

    source = Path(path).expanduser()
    if _slug(name or source.name.split(".nii")[0]) in entries("backdrops"):
        raise FileExistsError(f"there is already a backdrop called {name or source.name.split('.nii')[0]!r}; choose another name")
    with _lock:
        if _state["running"]:
            raise RuntimeError(f"already converting {_state['name']}")
        if not source.is_file():
            raise FileNotFoundError(source)
        _state.update(running=True, progress=0.0, error=None, path=None, kind="backdrop",
                      name=name or source.name.split(".nii")[0])

    Thread(target=_run, args=(source, mm, _state["name"]), daemon=True).start()
    return status()


def start_levels(source):
    """Build a backdrop's volume levels in the background."""
    from .volume_render import resolve_source

    path = Path(resolve_source(source))
    with _lock:
        if _state["running"]:
            raise RuntimeError(f"already working on {_state['name']}")
        if not path.is_file():
            raise FileNotFoundError(path)
        _state.update(running=True, progress=0.0, error=None, path=None, kind="levels",
                      name=f"{path.stem} (volume levels)")
    Thread(target=_run_levels, args=(path,), daemon=True).start()
    return status()


def start_atlas(source, name, table=None):
    """Mesh an atlas into ``ATLAS_DIR/<name>`` in the background (see
    ``volume_meshes.build_atlas``) and enter it in the library's mesh index.
    ``name`` as typed is its label in the Mesh list; made file-safe it is its
    name, which must not be taken by any mesh already listed."""
    import re

    source = Path(source).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    label = str(name or "").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    if not name:
        raise ValueError("the atlas needs a name")
    if name in entries("meshes"):
        raise FileExistsError(f"there is already a mesh called {name!r}; choose another name")
    with _lock:
        if _state["running"]:
            raise RuntimeError(f"already working on {_state['name']}")
        _state.update(running=True, progress=0.0, error=None, path=None, kind="atlas", name=name, skipped=[])
    Thread(target=_run_atlas, args=(source, ATLAS_DIR / name, label, table), daemon=True).start()
    return status()


def _run_atlas(source, out, label, table):
    from gnv.convert.volume_meshes import build_atlas

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        result = build_atlas(source, out, table=table, progress=_report)
        hint = f"{result['regions']} region meshes, from {source.name}" + (f"; {len(result['skipped'])} too small to mesh" if result["skipped"] else "")
        add_entry("meshes", {"name": out.name, "section": "Converted atlases", "label": label, "hint": hint, "family": "subcortex", "parcels": f"atlases/{out.name}"})
        with _lock:
            _state.update(running=False, progress=1.0, path=str(out), skipped=result["skipped"])
    except Exception as exc:
        with _lock:
            _state.update(running=False, error=f"{type(exc).__name__}: {exc}")


def start_fibers(source, name, keep_percent=None):
    """Convert a tract file or result, or a folder of them, into the fibre set
    ``FIBER_DIR/<name>`` in the background (see ``fiber_set.build_fiber_set``,
    which ``keep_percent`` is passed to) and enter it in the library's fibre
    index. ``name`` as typed is its label in the Fiber set list; made file-safe
    it is its name, which must not be taken."""
    import re

    source = Path(source).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    label = str(name or "").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    if not name:
        raise ValueError("the fibre set needs a name")
    if name in entries("fibers"):
        raise FileExistsError(f"there is already a fibre set called {name!r}; choose another name")
    keep_percent = None if keep_percent in (None, "") else float(keep_percent)
    if keep_percent is not None and not 0 < keep_percent <= 100:
        raise ValueError("keep strongest % is a number above 0 and at most 100")
    _begin("fibers", name)
    Thread(target=_run_fibers, args=(source, FIBER_DIR / name, label, keep_percent), daemon=True).start()
    return status()


def _run_fibers(source, out, label, keep_percent):
    from gnv.convert.fiber_set import build_fiber_set

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        result = build_fiber_set(source, out, keep_percent=keep_percent, progress=_report)
        hint = f"{result['tracts']} tract{'s' if result['tracts'] != 1 else ''}, from {source.name}" + (f", strongest {keep_percent:g}% of fibres" if keep_percent else "") + (f"; {len(result['skipped'])} without fibres" if result["skipped"] else "")
        add_entry("fibers", {"name": out.name, "section": "Converted fiber sets", "label": label, "hint": hint, "tracts": out.name})
        with _lock:
            _state.update(running=False, progress=1.0, path=str(out), skipped=result["skipped"])
    except Exception as exc:
        with _lock:
            _state.update(running=False, error=f"{type(exc).__name__}: {exc}")


def _begin(kind, name):
    with _lock:
        if _state["running"]:
            raise RuntimeError(f"already working on {_state['name']}")
        _state.update(running=True, progress=0.0, error=None, path=None, kind=kind, name=name, skipped=[], result=None)


def _finish(work, path=None):
    """Run ``work()`` and record its result or its error as the job's outcome."""
    try:
        result = work()
        with _lock:
            _state.update(running=False, progress=1.0, path=path, result=result)
    except Exception as exc:
        with _lock:
            _state.update(running=False, error=f"{type(exc).__name__}: {exc}")


def start_install(path):
    """Install the content pack at ``path`` into this user's library."""
    from .packs import install_pack, read_manifest

    path = Path(path).expanduser()
    manifest = read_manifest(path)          # refuse a non-pack at once, not in the job
    _begin("pack-install", manifest.get("title") or path.stem)
    Thread(target=_finish, args=(lambda: install_pack(path, progress=_report),), daemon=True).start()
    return status()


def start_pack(items, out_path, title="", notes=""):
    """Write library ``items`` to a content pack at ``out_path``."""
    from .packs import make_pack

    _begin("pack-make", title or Path(out_path).stem)
    Thread(target=_finish, args=(lambda: make_pack(items, out_path, title=title, notes=notes, progress=_report),), daemon=True).start()
    return status()


def _report(fraction):
    with _lock:
        _state["progress"] = float(fraction)


def _levels(path):
    from .volume_render import build_levels

    with _lock:
        _state.update(name=f"{Path(path).stem} (volume levels)", progress=0.0)
    return build_levels(str(path), progress=_report)


def _run_levels(path):
    try:
        _levels(path)
        with _lock:
            _state.update(running=False, progress=1.0, path=str(path))
    except Exception as exc:
        with _lock:
            _state.update(running=False, error=f"{type(exc).__name__}: {exc}")


def _run(source, mm, name):
    from gnv.convert.backdrops import build_backdrop, list_backdrops

    try:
        out = Path(build_backdrop(source, out_dir=BACKDROP_DIR, mm=mm, name=name, progress=_report))
        made = next((e for e in list_backdrops(out.parent) if e["path"] == str(out)), {})
        hint = f"{made['mm']:g} mm, from {source.name}" if "mm" in made else f"from {source.name}"
        add_entry("backdrops", {"name": out.stem, "section": "Converted", "label": name or out.stem, "hint": hint, "file": out.name})
        # A no-op for anything small enough to draw whole.
        _levels(out)
        with _lock:
            _state.update(running=False, progress=1.0, path=str(out))
    except Exception as exc:
        with _lock:
            _state.update(running=False, error=f"{type(exc).__name__}: {exc}")

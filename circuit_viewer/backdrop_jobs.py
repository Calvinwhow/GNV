"""Conversions as background jobs: backdrops, volume levels and atlases.

Converting a 100 um volume is minutes of work -- streaming 8.7 GB through a
block-average -- so it cannot happen inside a request. One job runs at a time,
its progress is polled, and the library it writes into accumulates: every volume
converted once is available to every figure afterwards.

A converted volume is also a volume-layer BODY. One too large to ray-cast whole
is drawn from block-averaged levels, so the job builds those right after the
archive, and ``start_levels`` builds them for an archive that predates this.

``start_atlas`` meshes an atlas once into the prepared-atlas library, where it
joins the mesh list as a region set. ``kind`` in the status says which job ran.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread

from .paths import ATLAS_DIR, BACKDROP_DIR

_state = {"running": False, "progress": 0.0, "name": None,
          "error": None, "path": None, "kind": None, "skipped": []}
_lock = Lock()


def catalogue():
    from calvin_utils.plotting_utils.backdrops import list_backdrops

    items = []
    for entry in list_backdrops(BACKDROP_DIR):
        entry = dict(entry)
        try:
            from calvin_utils.plotting_utils.backdrops import Backdrop

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
    with _lock:
        if _state["running"]:
            raise RuntimeError(f"already converting {_state['name']}")
        source = Path(path).expanduser()
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
    ``volume_meshes.build_atlas``). ``name`` becomes its name in the mesh list,
    so it must not be taken -- by a built-in mesh or an earlier atlas."""
    import re

    from calvin_utils.plotting_utils.brains import MESHES

    source = Path(source).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "")).strip("._")
    if not name:
        raise ValueError("the atlas needs a name")
    if name in MESHES or (ATLAS_DIR / name).exists():
        raise FileExistsError(f"there is already a mesh called {name!r}; choose another name")
    with _lock:
        if _state["running"]:
            raise RuntimeError(f"already working on {_state['name']}")
        _state.update(running=True, progress=0.0, error=None, path=None, kind="atlas", name=name, skipped=[])
    Thread(target=_run_atlas, args=(source, ATLAS_DIR / name, table), daemon=True).start()
    return status()


def _run_atlas(source, out, table):
    from calvin_utils.plotting_utils.volume_meshes import build_atlas

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        result = build_atlas(source, out, table=table, progress=_report)
        with _lock:
            _state.update(running=False, progress=1.0, path=str(out), skipped=result["skipped"])
    except Exception as exc:
        with _lock:
            _state.update(running=False, error=f"{type(exc).__name__}: {exc}")


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
    from calvin_utils.plotting_utils.backdrops import build_backdrop

    try:
        out = build_backdrop(source, out_dir=BACKDROP_DIR, mm=mm, name=name,
                             progress=_report)
        # A no-op for anything small enough to draw whole.
        _levels(out)
        with _lock:
            _state.update(running=False, progress=1.0, path=str(out))
    except Exception as exc:
        with _lock:
            _state.update(running=False, error=f"{type(exc).__name__}: {exc}")

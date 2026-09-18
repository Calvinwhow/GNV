"""Backdrop conversion as a background job.

Converting a 100 um volume is minutes of work -- streaming 8.7 GB through a
block-average -- so it cannot happen inside a request. One job runs at a time,
its progress is polled, and the library it writes into accumulates: every volume
converted once is available to every figure afterwards.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread

from .paths import BACKDROP_DIR

_state = {"running": False, "progress": 0.0, "name": None,
          "error": None, "path": None}
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
        _state.update(running=True, progress=0.0, error=None, path=None,
                      name=name or source.name.split(".nii")[0])

    Thread(target=_run, args=(source, mm, _state["name"]), daemon=True).start()
    return status()


def _run(source, mm, name):
    from calvin_utils.plotting_utils.backdrops import build_backdrop

    def report(fraction):
        with _lock:
            _state["progress"] = float(fraction)

    try:
        out = build_backdrop(source, out_dir=BACKDROP_DIR, mm=mm, name=name,
                             progress=report)
        with _lock:
            _state.update(running=False, progress=1.0, path=str(out))
    except Exception as exc:
        with _lock:
            _state.update(running=False, error=f"{type(exc).__name__}: {exc}")

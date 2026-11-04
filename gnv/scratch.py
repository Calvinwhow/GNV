"""Every temporary file gnv writes, in one place it owns.

Only PREPARED backdrops and atlases (the library, see ``paths``: the ``.npz``, their
level sidecars, region meshes) are kept between runs. Everything else the viewer makes
-- uploaded drops, the per-frame copies a 4D map plays from, mosaic helper
masks -- lives under ``SCRATCH_ROOT`` (``scratch`` in ``paths.USER_DIR`` by default,
``$GNV_SCRATCH`` to move it), in a folder for this process alone:

    scratch/
      run-<host>-<pid>-<token>/
        drops/     files dropped on the page
        frames/    one 3D copy per shown frame of a 4D map
        extend/    maps grown by an overlay's extend_mm
        mosaic/    hemisphere masks for medial mosaic views

and it is cleaned up four ways, so a long-running or hosted server cannot fill
its disk:

1. WHILE RUNNING, ``sweep()`` (called as files are made, at most once a
   minute) deletes frame copies and grown maps idle for ``FRAME_IDLE_S`` and
   drops idle for
   ``DROP_IDLE_S`` that no live figure points at.
2. A BUDGET: when the run folder passes ``BUDGET_BYTES`` the least recently
   used files go first, never one a live figure points at.
3. AT EXIT the whole run folder is removed (``atexit``, and SIGTERM when no
   server framework has claimed that signal).
4. AT START, run folders left by processes on this host that no longer exist
   -- a crash, a SIGKILL -- are removed, as are the system-temp folders older
   versions wrote (``gnv_drops`` and friends).

"Who points at what" comes from protectors: callables that return paths in
use. The app registers one that walks the open figure, so removing a layer
lets its drop be collected and keeping it keeps the file.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import socket
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

from .paths import USER_DIR

__all__ = ["SCRATCH_ROOT", "area", "run_dir", "sweep", "touch", "protect",
           "cleanup", "usage"]

SCRATCH_ROOT = Path(os.environ.get("GNV_SCRATCH")
                    or USER_DIR / "scratch").expanduser()
BUDGET_BYTES = int(float(os.environ.get("GNV_SCRATCH_MB", 4096)) * 2 ** 20)
FRAME_IDLE_S = 10 * 60
DROP_IDLE_S = 30 * 60
_SWEEP_EVERY_S = 60
# Folders older versions left in the system temp directory.
_LEGACY = ("gnv_drops", "gnv_frames", "gnv_mosaic")

_HOST = "".join(c if c.isalnum() else "-" for c in socket.gethostname())[:40] or "host"
_lock = threading.RLock()
_run = None
_last_sweep = 0.0
_protectors = []


def run_dir():
    """This process's scratch folder, made (and its cleanup armed) on first use."""
    global _run
    with _lock:
        if _run is None:
            SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
            ignore = SCRATCH_ROOT / ".gitignore"
            if not ignore.exists():
                ignore.write_text("# gnv scratch: never committed\n*\n")
            _remove_stale()
            _run = SCRATCH_ROOT / f"run-{_HOST}-{os.getpid()}-{uuid4().hex[:8]}"
            _run.mkdir()
            atexit.register(cleanup)
            _arm_sigterm()
        return _run


def area(name):
    """A named subfolder of the run folder (``drops``, ``frames``, ...)."""
    path = run_dir() / name
    path.mkdir(exist_ok=True)
    return path


def touch(path):
    """Mark a scratch file as just used, so idle sweeps and the budget keep it."""
    try:
        os.utime(path, None)
    except OSError:
        pass


def protect(provider):
    """Register ``provider() -> iterable of paths`` that must not be collected."""
    with _lock:
        _protectors.append(provider)


def _protected():
    keep = set()
    for provider in list(_protectors):
        try:
            for item in provider() or ():
                keep.add(os.path.realpath(str(item)))
        except Exception:
            continue
    return keep


def _files(folder):
    out = []
    for root, _dirs, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            out.append((path, st.st_mtime, st.st_size))
    return out


def usage():
    """``(files, bytes)`` held in this run's scratch folder."""
    if _run is None or not _run.exists():
        return 0, 0
    files = _files(_run)
    return len(files), sum(size for _, _, size in files)


def sweep(force=False):
    """Collect idle files, then enforce the budget. Cheap to call often."""
    global _last_sweep
    if _run is None:
        return
    now = time.time()
    with _lock:
        if not force and now - _last_sweep < _SWEEP_EVERY_S:
            return
        _last_sweep = now
        keep = _protected()
        files = _files(_run)
        live = []
        for path, mtime, size in files:
            real = os.path.realpath(path)
            sub = Path(path).relative_to(_run).parts[0]
            idle = now - mtime
            name = os.path.basename(path)
            stale_partial = name.startswith(".") and idle > 300
            expired = ((sub in ("frames", "extend") and idle > FRAME_IDLE_S)
                       or (sub == "drops" and idle > DROP_IDLE_S and real not in keep))
            if stale_partial or (expired and real not in keep):
                _unlink(path)
            else:
                live.append((path, mtime, size, real))
        total = sum(size for _, _, size, _ in live)
        if total > BUDGET_BYTES:
            for path, _mtime, size, real in sorted(live, key=lambda f: f[1]):
                if total <= BUDGET_BYTES:
                    break
                if real in keep:
                    continue
                if _unlink(path):
                    total -= size


def _unlink(path):
    try:
        os.unlink(path)
        return True
    except OSError:
        return False


def cleanup():
    """Remove this process's run folder (at exit; safe to call twice)."""
    global _run
    with _lock:
        if _run is not None:
            shutil.rmtree(_run, ignore_errors=True)
            _run = None


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True     # exists, owned by someone else
    return True


def _remove_stale():
    """Run folders of dead processes on this host, and legacy temp folders."""
    prefix = f"run-{_HOST}-"
    for entry in SCRATCH_ROOT.iterdir() if SCRATCH_ROOT.exists() else ():
        if not entry.is_dir() or not entry.name.startswith(prefix):
            continue
        try:
            pid = int(entry.name[len(prefix):].split("-")[0])
        except ValueError:
            continue
        if pid != os.getpid() and not _alive(pid):
            shutil.rmtree(entry, ignore_errors=True)
    for name in _LEGACY:
        shutil.rmtree(Path(tempfile.gettempdir()) / name, ignore_errors=True)


def _arm_sigterm():
    """Clean up on SIGTERM too -- but only when nothing else handles it. A
    server framework (gunicorn, a supervisor) owns its own signals; the next
    start's stale sweep covers a worker it kills."""
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        if signal.getsignal(signal.SIGTERM) not in (signal.SIG_DFL, None):
            return

        def _on_term(signum, frame):
            cleanup()
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _on_term)
    except (ValueError, OSError):
        pass

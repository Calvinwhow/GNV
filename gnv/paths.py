"""Filesystem locations owned by gnv.

Resources live in two roots with the same layout:

    resources/                  what ships with the viewer (read-only once packaged)
    <library>/                  this user's: conversions, installed packs, anything big
        meshes/index.json       the Mesh dropdown (see calvin_utils.resource_index)
        meshes/brain_surfaces/  surface meshes
        meshes/atlases/         folders of region meshes
        meshes/<any folder>/    any other meshes, listed in the index
        backdrops/index.json    the backdrop dropdowns
        backdrops/*.npz
        electrodes/index.json   the Lead model dropdown
        electrodes/*.mat
        fibers/index.json       the Fiber set dropdown
        fibers/<set>/           one fibre set: <tract>.npz + .fib.npy + .fib.json per tract

Each dropdown shows both roots' index for it, the viewer's own entries first.
Adding something by hand is putting its files in a folder and an entry in that
folder's index. A converted atlas or backdrop, or an installed pack, is written
to the library and entered in the library's index -- never into resources/.

The library is ``$GNV_LIBRARY`` when that is set; ``library/`` beside
``resources/`` when running from a source checkout (git-ignored, so everything of
yours is in the repository folder and none of it is committed); otherwise in this
user's folder, ``USER_DIR`` -- where a packaged app keeps it.
"""

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_DIR / "resources"
KINDS = ("meshes", "backdrops", "electrodes", "fibers")


def _user_dir():
    """This user's own folder for the viewer, outside the package: the library of a
    packaged app, the launcher's files, scratch space. ``$GNV_HOME`` moves it."""
    if os.environ.get("GNV_HOME"):
        return Path(os.environ["GNV_HOME"]).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "gnv"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "gnv"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "gnv"


def _library_dir():
    if os.environ.get("GNV_LIBRARY"):
        return Path(os.environ["GNV_LIBRARY"]).expanduser()
    if (PACKAGE_DIR.parent / ".git").exists():
        return PACKAGE_DIR / "library"
    return USER_DIR / "library"


USER_DIR = _user_dir()
LIBRARY_DIR = _library_dir()
BACKDROP_DIR = LIBRARY_DIR / "backdrops"            # where converted backdrops are written
ATLAS_DIR = LIBRARY_DIR / "meshes" / "atlases"      # where converted atlases are written
FIBER_DIR = LIBRARY_DIR / "fibers"                  # where converted fibre sets are written
ELECTRODE_DIR = RESOURCE_DIR / "electrodes"


def index(kind):
    """The dropdown for ``kind`` ("meshes", "backdrops", "electrodes"):
    ``[{"label": section, "entries": [...]}, ...]`` from both roots' indexes, the
    viewer's entries first, so a section both use is listed once. Each entry is
    its index entry with absolute paths, plus ``root`` (the folder its paths are
    under) and ``shipped``. A library entry with the name of a shipped one takes
    its place."""
    from calvin_utils.resource_index import read_index, sections

    found = {}
    for root in dict.fromkeys((RESOURCE_DIR, LIBRARY_DIR)):
        for section in read_index(root / kind):
            for entry in section["entries"]:
                found.pop(entry["name"], None)
                found[entry["name"]] = dict(entry, root=str(root / kind), shipped=root == RESOURCE_DIR)
    return sections(found.values())


def entries(kind):
    """``{name: entry}`` for ``kind``, in dropdown order (see ``index``)."""
    return {e["name"]: e for section in index(kind) for e in section["entries"]}


def add_entry(kind, entry):
    """Enter ``entry`` (with its ``section``; paths relative to the library's
    ``kind`` folder) in the library's index for ``kind``."""
    from calvin_utils.resource_index import add_entry as add

    add(LIBRARY_DIR / kind, entry)


def relative_entry(entry):
    """An entry as its index stores it: paths relative to its folder, without the
    fields ``index`` adds."""
    root = Path(entry["root"])
    out = {k: v for k, v in entry.items() if k not in ("root", "shipped")}
    for key in ("file", "parcels", "tracts"):
        if key in out:
            out[key] = Path(out[key]).relative_to(root).as_posix()
    if "pieces" in out:
        out["pieces"] = [Path(p).relative_to(root).as_posix() if Path(p).is_absolute() else p for p in out["pieces"]]
    return out


def prepared_backdrops():
    """``{name: .npz path}`` for every indexed backdrop whose file is there."""
    return {n: Path(e["file"]) for n, e in entries("backdrops").items() if Path(e["file"]).is_file()}


def prepared_atlases():
    """``{name: folder}`` for every indexed folder of region meshes that is there."""
    return {n: Path(e["parcels"]) for n, e in entries("meshes").items() if "parcels" in e and Path(e["parcels"]).is_dir()}


def is_shipped(path):
    """Whether ``path`` is part of the viewer rather than this user's library."""
    return Path(path).resolve().is_relative_to(RESOURCE_DIR)

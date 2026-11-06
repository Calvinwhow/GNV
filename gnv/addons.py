"""Add-ons: optional features kept in the library, not in the package.

An add-on is a folder in ``<library>/addons/`` (gnv/library/addons in a git
checkout, so never committed; the user's library in the packaged app, where a
.gnvpack installs it). The viewer runs the same with none of them. Each is
found at startup and attached; a change takes effect when the viewer restarts.

An add-on folder holds:
    addon.json     {"name", "label", "version", "requires_gnv", "requires", "script"}
                   name          the folder's name, and the Python package it is imported as
                   requires_gnv  the oldest viewer it runs on
                   requires      top-level modules it imports (all ship with the viewer; it is
                                 skipped, with a message, if one is missing from an install)
                   script        the page script it serves, added to the page when it loads
    __init__.py    defines attach(app, session, lock): registers its routes on the Flask app

What an add-on may read of the viewer: the session (its spec, actors, plotter,
render thread, fibre cache and render()) and the shared helpers in
gnv.onscreen and gnv.render.render_scene.light_rig.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

from . import __version__, paths

__all__ = ["ADDON_DIR", "manifests", "version_key", "load", "report"]

ADDON_DIR = paths.LIBRARY_DIR / "addons"
_report = []


def manifests(root=ADDON_DIR):
    """Each add-on folder's addon.json, with its "folder"; a bad one carries "error"."""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        manifest = folder / "addon.json"
        if not manifest.is_file():
            continue
        try:
            info = json.loads(manifest.read_text())
        except ValueError as exc:
            info = {"error": f"addon.json is not valid JSON ({exc})"}
        info["name"], info["folder"] = folder.name, folder
        out.append(info)
    return out


def version_key(text):
    """A version string as a tuple to compare: "0.10.1" -> (0, 10, 1)."""
    return tuple(int(part) for part in str(text).split(".") if part.isdigit())


def _problem(info):
    """Why this add-on cannot be loaded, or None."""
    if info.get("error"):
        return info["error"]
    if not info["name"].isidentifier():
        return f"its folder name {info['name']!r} is not a Python name"
    if not (info["folder"] / "__init__.py").is_file():
        return "it has no __init__.py"
    if version_key(info.get("requires_gnv", "0")) > version_key(__version__):
        return f"it needs gnv {info['requires_gnv']} or newer (this is {__version__})"
    missing = [m for m in info.get("requires", []) if importlib.util.find_spec(m) is None]
    if missing:
        return f"it needs {', '.join(missing)}, which is not installed"
    return None


def _import(info):
    """The add-on's folder imported as the package ``gnv_addon_<name>``, so its own relative imports work."""
    name = f"gnv_addon_{info['name']}"
    spec = importlib.util.spec_from_file_location(name, info["folder"] / "__init__.py", submodule_search_locations=[str(info["folder"])])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]
        raise
    return module


def load(app, session, lock, root=ADDON_DIR):
    """Find, check and attach every add-on in ``root``. Returns the report /api/addons serves."""
    _report.clear()
    for info in manifests(root):
        entry = {"name": info["name"], "label": info.get("label", info["name"]), "version": info.get("version", ""), "script": info.get("script"), "loaded": False}
        entry["reason"] = _problem(info)
        if entry["reason"] is None:
            try:
                _import(info).attach(app, session, lock)
                entry["loaded"] = True
            except Exception as exc:
                traceback.print_exc()
                entry["reason"] = f"it failed to start: {exc!r}"
        print(f"gnv: add-on {entry['label']} " + ("loaded" if entry["loaded"] else f"not loaded: {entry['reason']}"))
        _report.append(entry)
    return report()


def report():
    """What load() found: one entry per add-on, loaded or not, and why not."""
    return [dict(entry) for entry in _report]

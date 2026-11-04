"""The cortical surfaces yabplot downloads, shipped so the viewer works offline.

yabplot fetches its surfaces from OSF into its own cache the first time one is
drawn, and the built-in cortex meshes are made of them. A packaged viewer
carries the six it uses in ``resources/yabplot`` -- put there at build time by
``packaging/fetch_yabplot_surfaces.py`` -- and ``seed()`` copies any that the
cache lacks into it, stamped as yabplot stamps its own downloads, so yabplot
finds them there and never goes online for them. A shipped surface is only
used when it is the version yabplot's registry expects; otherwise yabplot
downloads as usual. Without ``resources/yabplot`` (running from source) this
does nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .paths import RESOURCE_DIR

SEED_DIR = RESOURCE_DIR / "yabplot"


def seed():
    """Copy shipped surfaces into yabplot's cache where it lacks them. Returns the names copied."""
    if not SEED_DIR.is_dir():
        return []
    try:
        from yabplot.data import GOODBOY
    except Exception:
        return []
    cache = Path(GOODBOY.path)
    copied = []
    for folder in sorted(p for p in SEED_DIR.iterdir() if p.is_dir()):
        expected = GOODBOY.registry.get(f"{folder.name}.zip")
        shipped = folder / ".registry_hash"
        if not expected or not shipped.is_file() or shipped.read_text().strip() != expected:
            continue                    # not the version this yabplot wants
        stamp = cache / folder.name / ".registry_hash"
        if stamp.is_file() and stamp.read_text().strip() == expected:
            continue                    # already there
        staging = cache / f".{folder.name}.seeding"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(folder, staging)
        shutil.rmtree(cache / folder.name, ignore_errors=True)
        staging.rename(cache / folder.name)
        copied.append(folder.name)
    return copied

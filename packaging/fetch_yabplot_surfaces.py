"""Download the cortical surfaces the built-in meshes are made of into
gnv/resources/yabplot, for the build to ship (see
gnv/yabplot_cache.py). Run from the repository root before PyInstaller."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # the repository, for gnv

import gnv  # noqa: F401 -- first, so the renderer looks for meshes in the viewer's resources
from calvin_utils.plotting_utils.glass_mesh import BMESH_NAMES
from yabplot.data import _resolve_resource_path

DEST = Path(__file__).resolve().parent.parent / "gnv" / "resources" / "yabplot"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name in BMESH_NAMES:
        fetched = Path(_resolve_resource_path(name, "bmesh"))
        shutil.copytree(fetched, DEST / fetched.name, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__MACOSX"))
        print(f"shipping {fetched.name}")


if __name__ == "__main__":
    main()

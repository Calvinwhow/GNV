# The packaged viewer: PyInstaller, one folder, windowed; a .app on macOS.
# Build from the repository root (the renderer and packaging/requirements.txt
# installed first):
#     python -m PyInstaller --noconfirm --clean packaging/GeneralNeuroimageViewer.spec
#
# What ships: the viewer's code and page, and everything in
# gnv/resources -- which holds only what ships: the meshes,
# backdrops and electrode models its index.json files list. Anything else -- the 100 um backdrops,
# converted atlases, the Lead-DBS surfaces -- is a library (gnv/paths.py),
# filled by converting or by installing content packs.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent
PACKAGE = ROOT / "gnv"
RESOURCES = PACKAGE / "resources"


def folder(source, dest):
    """Every file under ``source``, placed under ``dest`` in the bundle."""
    if not source.is_dir():
        raise SystemExit(f"missing {source}: the build ships it")
    # Hidden files are left out.
    keep = lambda p: p.is_file() and not p.name.startswith(".")
    return [(str(p), str(Path(dest) / p.parent.relative_to(source))) for p in sorted(source.rglob("*")) if keep(p)]


datas = (
    folder(PACKAGE / "static", "gnv/static")
    + folder(PACKAGE / "lineart" / "static", "gnv/lineart/static")
    + folder(PACKAGE / "volumes", "gnv/volumes")
    + folder(RESOURCES, "gnv/resources")
)

a = Analysis(
    [str(ROOT / "scripts" / "launch.py")],
    pathex=[str(ROOT)],
    datas=datas,
    # Imported by name at run time, out of the analyser's sight. (All of VTK
    # comes along regardless: the viewer imports it as ``import vtk``.)
    hiddenimports=collect_submodules("gnv") + ["calvin_utils.neuroimaging_utils.nifti_utils.parcel_meshes"],
    # Optional imports of the libraries above that the viewer never uses.
    excludes=["torch", "jax", "tensorflow", "keras", "IPython", "notebook", "jupyterlab", "cv2", "imageio_ffmpeg", "sklearn", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="GeneralNeuroimageViewer", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="GeneralNeuroimageViewer")

if sys.platform == "darwin":
    app = BUNDLE(coll, name="General Neuroimage Viewer.app", bundle_identifier="edu.bwh.gnv", info_plist={"NSHighResolutionCapable": True})

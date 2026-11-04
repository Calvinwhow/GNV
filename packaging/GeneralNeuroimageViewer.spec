# The packaged viewer: PyInstaller, one folder, windowed; a .app on macOS.
# Build from the repository root (the renderer and packaging/requirements.txt
# installed, and packaging/fetch_yabplot_surfaces.py run first):
#     python -m PyInstaller --noconfirm --clean packaging/CircuitViewer.spec
#
# What ships: the viewer's code and page, and everything in
# circuit_viewer/resources -- which holds only what ships: the meshes,
# backdrops and electrode models its index.json files list, plus the cortical
# surfaces yabplot would download. Anything else -- the 100 um backdrops,
# converted atlases, the Lead-DBS surfaces -- is a library (circuit_viewer/paths.py),
# filled by converting or by installing content packs.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
PACKAGE = ROOT / "circuit_viewer"
RESOURCES = PACKAGE / "resources"


def folder(source, dest):
    """Every file under ``source``, placed under ``dest`` in the bundle."""
    if not source.is_dir():
        raise SystemExit(f"missing {source}: the build ships it")
    # Hidden files are left out, except yabplot's version stamp on each surface.
    keep = lambda p: p.is_file() and (not p.name.startswith(".") or p.name == ".registry_hash")
    return [(str(p), str(Path(dest) / p.parent.relative_to(source))) for p in sorted(source.rglob("*")) if keep(p)]


datas = (
    folder(PACKAGE / "static", "circuit_viewer/static")
    + folder(PACKAGE / "lineart" / "static", "circuit_viewer/lineart/static")
    + folder(PACKAGE / "volumes", "circuit_viewer/volumes")
    + folder(RESOURCES, "circuit_viewer/resources")
    + collect_data_files("yabplot")
)

a = Analysis(
    [str(ROOT / "scripts" / "launch.py")],
    pathex=[str(ROOT)],
    datas=datas,
    # Imported by name at run time, out of the analyser's sight. (All of VTK
    # comes along regardless: the viewer imports it as ``import vtk``.)
    hiddenimports=collect_submodules("circuit_viewer") + ["calvin_utils.neuroimaging_utils.nifti_utils.parcel_meshes"],
    # Optional imports of the libraries above that the viewer never uses.
    excludes=["torch", "jax", "tensorflow", "keras", "IPython", "notebook", "jupyterlab", "cv2", "imageio_ffmpeg", "sklearn", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="CircuitViewer", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="CircuitViewer")

if sys.platform == "darwin":
    app = BUNDLE(coll, name="Circuit Viewer.app", bundle_identifier="edu.bwh.circuitviewer", info_plist={"NSHighResolutionCapable": True})

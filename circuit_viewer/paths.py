"""Filesystem locations owned by circuit_viewer."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_DIR / "resources"
BACKDROP_DIR = RESOURCE_DIR / "backdrops"
MESH_DIR = RESOURCE_DIR / "meshes"
ATLAS_DIR = MESH_DIR / "atlases"       # prepared atlases: one folder of region meshes each

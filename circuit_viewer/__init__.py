"""circuit_viewer -- a layered brain-figure viewer built on the circuit_pyper renderer."""

__all__ = ["__version__"]
__version__ = "0.1.0"

# The viewer ships the geometry it depends on. Point the renderer at our copy
# before anything imports it, so installing circuit_pyper somewhere else -- or
# not shipping its resources at all -- does not leave the viewer without meshes.
# An explicit environment setting still wins, for anyone pointing it at their own.
import os as _os
from pathlib import Path as _Path

MESH_RESOURCES = _Path(__file__).resolve().parent / "resources" / "meshes" / "neuro"
_os.environ.setdefault("NEURO_PLOTTER_RESOURCES", str(MESH_RESOURCES))

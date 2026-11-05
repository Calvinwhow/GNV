"""The renderer: everything that turns loaded data into a picture.

Scene, lights and colour scales (``render_scene``, ``palettes``), surfaces and
the mesh registry (``brains``, ``glass_mesh``), painting values onto meshes and
slices (``mesh_paint``, ``overlay_paint``, ``backdrop_plane``), fibres
(``fiber_render``), electrodes (``electrode_render``), and the scripted figure
API built on them (``brain_viewer``). Reading and converting files is
``gnv.convert`` (and, for the tract formats, circuit_pyper); this package only
draws what they read.
"""


def _fix_vtk_to_numpy():
    """VTK 9.7's ``vtk_to_numpy`` refers to ``vtkConstants`` without importing it,
    on the path that copies an array numpy cannot read directly (an implicit
    array, such as fixed-size cell offsets) -- so that path raises NameError
    instead of returning the values. Give the module the name it uses; a VTK
    that imports it already is left as it is."""
    from vtkmodules.util import numpy_support, vtkConstants

    numpy_support.__dict__.setdefault("vtkConstants", vtkConstants)


_fix_vtk_to_numpy()

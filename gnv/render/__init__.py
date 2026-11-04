"""The renderer: everything that turns loaded data into a picture.

Scene, lights and colour scales (``render_scene``, ``palettes``), surfaces and
the mesh registry (``brains``, ``glass_mesh``), painting values onto meshes and
slices (``mesh_paint``, ``overlay_paint``, ``backdrop_plane``), fibres
(``fiber_render``), electrodes (``electrode_render``), and the scripted figure
API built on them (``brain_viewer``). Reading and converting files is
``gnv.convert`` (and, for the tract formats, circuit_pyper); this package only
draws what they read.
"""

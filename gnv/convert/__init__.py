"""Reading and converting what the viewer shows into what it draws from.

``backdrops``      a NIfTI into a slice archive, and reading slices back
``volume_meshes``  a NIfTI into surface meshes; an atlas into a folder of region meshes
``parcel_meshes``  a label volume into one smoothed mesh per region
``fiber_set``      tract files or results into a fibre set (one trio per tract)

Tract file formats themselves -- .tck/.trk/.mat, and the .npz + .fib.npy +
.fib.json results -- are read by circuit_pyper (calvin_utils.neuroimaging_utils.tract_utils).
"""

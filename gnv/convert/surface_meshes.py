"""Surface meshes from other tools, converted once into the mesh library.

A surface someone already has -- a Lead-DBS cortex (.mat), a Surf-Ice mesh
(.mz3), an STL from a segmentation, a GIfTI from FreeSurfer or Connectome
Workbench, an OBJ/PLY/VTK from anywhere -- is read, checked, and written as a
binary VTK file the viewer loads quickly, then entered in the library's mesh
index so it is in the Mesh list for every figure.

What each becomes:
    one surface          <name>/mesh.vtp
    a left/right pair    <name>/L.vtk and <name>/R.vtk -- a .mat holding
                         ``Vertices_lh``/``Faces_lh`` and ``_rh``, as Lead-DBS
                         writes a cortex -- so each hemisphere is its own piece
                         and a lateral view can drop the far one
    a folder             each surface file in it, as its own entry

STL stores every triangle's corners separately; points are merged on reading,
so the surface is one connected sheet that shades smoothly.
"""

from __future__ import annotations

import gzip
import re
import struct
from pathlib import Path

import numpy as np

__all__ = ["SURFACE_SUFFIXES", "polydata", "read_mz3", "read_surface", "surface_files", "convert_mesh", "entry_name", "entry_labels"]

SURFACE_SUFFIXES = (".vtk", ".vtp", ".stl", ".ply", ".obj", ".gii", ".mz3", ".mat")


def polydata(points, triangles):
    """A triangle mesh from (N, 3) points and (M, 3) zero-based triangles, checked."""
    import pyvista as pv

    points = np.asarray(points, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {points.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"faces must have shape (N, 3), got {triangles.shape}")
    if not np.isfinite(points).all():
        raise ValueError("vertices contain non-finite coordinates")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(points)):
        raise ValueError("a face index is outside the vertex array")
    cells = np.empty((len(triangles), 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = triangles
    return pv.PolyData(points, cells.ravel())


def read_mz3(path):
    """The face-and-vertex part of a Surf-Ice MZ3 file (plain or gzipped)."""
    path = Path(path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"{path.name}: truncated MZ3 header")
        magic, attributes, face_count, vertex_count, skip = struct.unpack("<HHIII", header)
        if magic != 0x5A4D:
            raise ValueError(f"{path.name}: bad MZ3 magic 0x{magic:04x}")
        if attributes & 0x03 != 0x03:
            raise ValueError(f"{path.name}: MZ3 has no face-and-vertex geometry")
        stream.read(skip)
        faces = np.frombuffer(stream.read(face_count * 3 * 4), dtype="<u4")
        coordinate_type = "<f8" if attributes & 0x10 else "<f4"
        points = np.frombuffer(stream.read(vertex_count * 3 * np.dtype(coordinate_type).itemsize), dtype=coordinate_type)
    if faces.size != face_count * 3 or points.size != vertex_count * 3:
        raise ValueError(f"{path.name}: truncated MZ3 geometry")
    return polydata(points.reshape(vertex_count, 3), faces.reshape(face_count, 3))


def _gifti(path):
    import nibabel as nib

    image = nib.load(str(path))
    points = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
    faces = image.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
    if not points or not faces:
        raise ValueError(f"{Path(path).name} holds no surface (it may be a map of values, not a mesh)")
    return polydata(points[0].data, faces[0].data)


def _mat_variables(path):
    """The variables of a .mat file, MATLAB v5 or v7.3 (HDF5), as {name: value};
    a struct's fields as {name: {field: value}}."""
    import h5py

    if not h5py.is_hdf5(str(path)):
        from scipy.io import loadmat

        data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        out = {}
        for key, value in data.items():
            if key.startswith("__"):
                continue
            if hasattr(value, "_fieldnames"):
                out[key] = {f: getattr(value, f) for f in value._fieldnames}
            else:
                out[key] = value
        return out
    # v7.3 is an HDF5 file, whose arrays come out the other way round.
    out = {}
    with h5py.File(path, "r") as handle:
        for key, item in handle.items():
            if isinstance(item, h5py.Group):
                out[key] = {f: np.asarray(v).T for f, v in item.items() if isinstance(v, h5py.Dataset)}
            elif isinstance(item, h5py.Dataset):
                out[key] = np.asarray(item).T
    return out


def _rows(array, width=3):
    """``array`` as rows of ``width``: MATLAB may store either way round."""
    array = np.asarray(array)
    if array.ndim == 2 and array.shape[1] != width and array.shape[0] == width:
        array = array.T
    return array


def _mat_surface(vertices, faces):
    vertices, faces = _rows(vertices).astype(float), _rows(faces).astype(np.int64)
    # MATLAB counts from 1.
    if faces.size and faces.min() >= 1 and faces.max() == len(vertices):
        faces = faces - 1
    return polydata(vertices, faces)


def _mat(path):
    """A Lead-DBS-style .mat: ``Vertices_lh``/``Faces_lh`` and ``_rh`` (a
    pair), ``vertices``/``faces`` at the top level, or a struct holding them
    (an ``fv`` patch). Names are matched without regard to case."""
    variables = _mat_variables(path)
    lower = {k.lower(): v for k, v in variables.items()}
    if all(f"{n}_{s}" in lower for n in ("vertices", "faces") for s in ("lh", "rh")):
        return {"L": _mat_surface(lower["vertices_lh"], lower["faces_lh"]), "R": _mat_surface(lower["vertices_rh"], lower["faces_rh"])}
    if "vertices" in lower and "faces" in lower:
        return _mat_surface(lower["vertices"], lower["faces"])
    for value in variables.values():
        if isinstance(value, dict):
            fields = {k.lower(): v for k, v in value.items()}
            if "vertices" in fields and "faces" in fields:
                return _mat_surface(fields["vertices"], fields["faces"])
    raise ValueError(f"{Path(path).name} holds no surface: expected vertices and faces (or Vertices_lh/_rh and Faces_lh/_rh), found {', '.join(variables) or 'nothing'}")


def read_surface(path):
    """A surface file as a triangle mesh, or ``{"L": mesh, "R": mesh}`` for a pair."""
    import pyvista as pv

    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".mz3":
        mesh = read_mz3(path)
    elif suffix == ".gii":
        mesh = _gifti(path)
    elif suffix == ".mat":
        mesh = _mat(path)
    elif suffix in (".vtk", ".vtp", ".stl", ".ply", ".obj"):
        mesh = pv.read(str(path))
    else:
        raise ValueError(f"{path.name}: not a surface file ({', '.join(SURFACE_SUFFIXES)})")
    if isinstance(mesh, dict):
        return {side: _tidy(m, path) for side, m in mesh.items()}
    return _tidy(mesh, path)


def _tidy(mesh, path):
    """Triangles only, shared corners merged (an STL repeats them per triangle)."""
    surface = mesh.extract_surface(algorithm=None).triangulate().clean()
    if surface.n_cells == 0:
        raise ValueError(f"{Path(path).name} has no surface triangles")
    return surface


def entry_labels(files):
    """A label for each file of a folder: its name without the extension, and
    the extension too where two files share a name (brain.stl, brain.obj)."""
    stems = [f.stem for f in files]
    return [f"{f.stem} ({f.suffix.lower()})" if stems.count(f.stem) > 1 else f.stem for f in files]


def surface_files(source):
    """The surface files to convert from ``source``: itself, or every surface
    file directly in the folder (hidden files and Finder metadata left out)."""
    source = Path(source).expanduser()
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SURFACE_SUFFIXES)


def convert_mesh(path, out_dir):
    """Write one surface file into ``out_dir``: ``mesh.vtp``, or ``L.vtk`` and
    ``R.vtk`` for a pair. Returns ``{"piece", "points", "triangles"}`` where
    ``piece`` is what the mesh index names: the file, or the folder of a pair."""
    surface = read_surface(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(surface, dict):
        for side, mesh in surface.items():
            mesh.save(out_dir / f"{side}.vtk", binary=True)
        meshes = list(surface.values())
        piece = out_dir
    else:
        piece = out_dir / "mesh.vtp"
        surface.save(piece, binary=True)
        meshes = [surface]
    return {"piece": piece, "points": sum(m.n_points for m in meshes), "triangles": sum(m.n_cells for m in meshes)}


def entry_name(text):
    """A mesh name made file-safe."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")

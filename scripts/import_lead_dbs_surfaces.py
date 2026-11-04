"""Normalize a Lead-DBS surface bundle into a meshes folder: the surfaces go in
brain_surfaces/lead_dbs/, and each gets an entry in the folder's index.json.
By default that is the library's (gnv/paths.py), since these
surfaces are not shipped with the viewer."""

from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.io import loadmat

from calvin_utils.resource_index import add_entry
from gnv.paths import LIBRARY_DIR


def polydata(points, triangles):
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
    """Read the face-and-vertex subset of the Surf-Ice MZ3 format."""
    path = Path(path)
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"{path.name}: truncated MZ3 header")
        magic, attributes, face_count, vertex_count, skip = struct.unpack(
            "<HHIII", header)
        if magic != 0x5A4D:
            raise ValueError(f"{path.name}: bad MZ3 magic 0x{magic:04x}")
        if attributes & 0x03 != 0x03:
            raise ValueError(f"{path.name}: MZ3 has no face-and-vertex geometry")
        stream.read(skip)
        faces = np.frombuffer(stream.read(face_count * 3 * 4), dtype="<u4")
        coordinate_type = "<f8" if attributes & 0x10 else "<f4"
        itemsize = np.dtype(coordinate_type).itemsize
        points = np.frombuffer(
            stream.read(vertex_count * 3 * itemsize), dtype=coordinate_type)
    if faces.size != face_count * 3 or points.size != vertex_count * 3:
        raise ValueError(f"{path.name}: truncated MZ3 geometry")
    return polydata(points.reshape(vertex_count, 3),
                    faces.reshape(face_count, 3))


def save(mesh, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(path, binary=True)
    print(f"{path}: {mesh.n_points:,} points, {mesh.n_cells:,} faces")


def annotation(path):
    entries = loadmat(path, squeeze_me=True, struct_as_record=False)["annot"]
    result = {}
    for entry in np.atleast_1d(entries):
        side = "R" if Path(entry.filename).name.startswith("rh.") else "L"
        table = np.asarray(entry.colortable.table, dtype=np.int32)
        names = [str(x) for x in np.atleast_1d(entry.colortable.struct_names)]
        packed_to_index = {int(row[-1]): i for i, row in enumerate(table)}
        labels = np.asarray(entry.label, dtype=np.int32)
        indices = np.fromiter(
            (packed_to_index.get(int(value), 0) for value in labels),
            dtype=np.int16, count=len(labels))
        result[side] = {"labels": labels, "indices": indices,
                        "table": table, "names": names}
    return result


def write_lut(info, path):
    rows = ["index\tname\tred\tgreen\tblue\talpha\tpacked_label"]
    for i, (name, values) in enumerate(zip(info["names"], info["table"])):
        rows.append("\t".join(map(str, [i, name, *map(int, values)])))
    Path(path).write_text("\n".join(rows) + "\n")


def matlab_group(source, output, annotations=None):
    data = loadmat(source, variable_names=[
        "Vertices_lh", "Vertices_rh", "Faces_lh", "Faces_rh"])
    for suffix, side in (("lh", "L"), ("rh", "R")):
        mesh = polydata(data[f"Vertices_{suffix}"],
                        np.asarray(data[f"Faces_{suffix}"], dtype=np.int64) - 1)
        if annotations is not None:
            info = annotations[side]
            if len(info["labels"]) != mesh.n_points:
                raise ValueError(f"DKT labels do not match {side} vertices")
            mesh.point_data["DKT_label"] = info["labels"]
            mesh.point_data["DKT_index"] = info["indices"]
        save(mesh, Path(output) / f"{side}.vtk")


def import_bundle(source, meshes):
    source, meshes = Path(source).expanduser(), Path(meshes).expanduser()
    output = meshes / "brain_surfaces" / "lead_dbs"
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    # Standalone surfaces already readable by VTK are normalized to binary VTP.
    for original, target in (
        ("BrainMesh_Ch2.obj", "BrainMesh_Ch2.vtp"),
        ("BrainMesh_ICBM152_tal.obj", "BrainMesh_ICBM152_tal.vtp"),
        ("surf.rh.ply", "surf.rh.ply.vtp"),
    ):
        save(pv.read(source / original).extract_surface().triangulate(),
             output / target)

    # MZ3 keeps shared vertices, unlike STL. The STL and right OBJ duplicates
    # are intentionally not imported.
    save(read_mz3(source / "brainMeshRight.mz3"),
         output / "brainMeshRight.vtp")
    save(read_mz3(source / "surf.mz3"), output / "surf.vtp")
    for original, side in (("surf.lh.mz3", "L"), ("surf.rh.mz3", "R")):
        save(read_mz3(source / original), output / "surf.lh-rh" / f"{side}.vtk")

    dkt = annotation(source / "annot_DKT.mat")
    matlab_group(source / "CortexHiRes.mat", output / "CortexHiRes", dkt)
    write_lut(dkt["L"], output / "CortexHiRes" / "dkt_lut.tsv")
    matlab_group(source / "CortexLowRes_15000V.mat",
                 output / "CortexLowRes_15000V")

    # Each surface becomes an entry in the meshes folder's index: the Mesh dropdown.
    folder = output.relative_to(meshes).as_posix()
    for name, label, hint, piece in (
        ("leaddbs_ch2", "Ch2 whole brain", "Lead-DBS Ch2 whole-brain surface", "BrainMesh_Ch2.vtp"),
        ("leaddbs_icbm152", "ICBM152 whole brain", "Lead-DBS ICBM152 Talairach whole-brain surface", "BrainMesh_ICBM152_tal.vtp"),
        ("leaddbs_surf", "Combined cortex", "Lead-DBS combined cortical surface", "surf.vtp"),
        ("leaddbs_cortex_hires", "Cortex, high resolution (DKT)", "Lead-DBS bilateral high-resolution cortex, DKT labels", "CortexHiRes"),
        ("leaddbs_cortex_lowres", "Cortex, 15,000 vertices", "Lead-DBS bilateral cortex, 15,000 vertices", "CortexLowRes_15000V"),
        ("leaddbs_surf_lh_rh", "Cortex, left/right pair", "Lead-DBS bilateral cortical surface pair", "surf.lh-rh"),
        ("leaddbs_right", "Right hemisphere", "Lead-DBS right-hemisphere brain surface", "brainMeshRight.vtp"),
        ("leaddbs_surf_rh_ply", "Right hemisphere (PLY variant)", "Lead-DBS right-hemisphere PLY surface variant", "surf.rh.ply.vtp"),
    ):
        add_entry(meshes, {"name": name, "section": "Lead-DBS surfaces", "label": label, "hint": hint, "family": "surface", "pieces": [f"{folder}/{piece}"]})

    manifest = """# Imported Lead-DBS surfaces

Source: {source}

Normalized surfaces are binary VTK/VTP files. The high-resolution cortex keeps
the Desikan-Killiany annotations as `DKT_label` and `DKT_index` point arrays.

Not imported:

- `brainMeshRight.stl`: geometry duplicate of `brainMeshRight.mz3`.
- `surf.rh.obj`: geometry duplicate of `surf.rh.mz3`.
- `BigBrain_100um.mat`: anatomical image volume, not a surface mesh.
- `Untitled.blend1`: Blender backup/project data, not an interchange mesh.
- `Icon`: Finder metadata.
""".format(source=source)
    (output / "README.md").write_text(manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="directory containing the Lead-DBS bundle")
    parser.add_argument("--meshes", type=Path, default=LIBRARY_DIR / "meshes", help="the meshes folder to import into")
    args = parser.parse_args()
    import_bundle(args.source, args.meshes)


if __name__ == "__main__":
    main()

"""Glass-brain backdrop meshes for yabplot's voxelwise renderer.

`yabplot.plot_voxelwise` draws a statistical map as nested isosurfaces inside a
translucent context mesh, and its ``bmesh`` argument accepts a
``pyvista.PolyData`` or a ``{name: mesh}`` dict.  This module supplies the two
meshes that go there.

Both are prebuilt and ship in ``brain_surfaces/`` of the meshes folder (``brains.RESOURCE_DIR``).
Plotting just reads them -- no marching cubes, no atlas lookup, no NIfTI, at
plot time::

    from gnv.render.glass_mesh import glass_bmesh

    yabplot.plot_voxelwise(map_path, bmesh=glass_bmesh(), ignore_bmesh=False,
                           bmesh_alpha=0.16, bmesh_color="#8d99ae")

The shipped meshes
------------------
``glass_cerebrum.vtk`` (85,700 triangles, 6 connected components)
    A smooth cerebral hull marching-cubed from the 1 mm ICBM152 mask, with the
    cerebellum and the deep grey structures **subtracted from the volume**
    before meshing.  Subtracting rather than drawing those structures on top is
    what gives them their hollow, shaded look: marching cubes emits the cavity
    walls as part of the same surface, so the whole backdrop stays one mesh
    carrying one colour and one alpha -- which is all ``plot_voxelwise``'s
    internal ``add_brain()`` can apply.  Carved: hippocampus, caudate, putamen,
    pallidum, accumbens, thalamus (aseg).  Caudate/putamen/pallidum are
    contiguous at 1 mm and fuse into one striatopallidal void per side.

``glass_cerebellum.vtk`` (69,078 triangles)
    The SUIT cerebellar surface, left and right merged into a single mesh so
    the pair shares one actor and one alpha.  Its anterior face is deeply
    invaginated around the peduncles; that inward fold is real geometry and is
    what reads as a hollow when the mesh is drawn translucent.

Neither file stores normals -- ``add_mesh(smooth_shading=True)`` recomputes
them -- which roughly halves the files on disk.

Regenerating
------------
`rebuild_glass_meshes` rebuilds the hull from source volumes.  It is the cold
path and is never called while plotting; it additionally needs nibabel, scipy
and scikit-image, plus the source mask, a cerebellar label volume and the aseg
meshes.  Use it to change what gets carved, not to plot.
"""

from __future__ import annotations

from pathlib import Path

import pyvista as pv

__all__ = [
    "GLASS_MESH_DIR",
    "CEREBRUM_MESH",
    "CEREBELLUM_MESH",
    "PLAIN_CEREBRUM_MESH",
    "PIAL_CEREBELLUM_MESH",
    "BRAINSTEM_MESH",
    "DEFAULT_CARVE",
    "glass_bmesh",
    "build_bmesh",
    "BMESH_NAMES",
    "rebuild_glass_meshes",
]


# One source of truth: brain_surfaces/ in the meshes folder brains.RESOURCE_DIR
# resolved, so an application that redirects the resources gets both the
# prebuilt meshes and the atlases from its own copy rather than half from each.
from gnv.render.brains import RESOURCE_DIR as _MESH_FOLDER
GLASS_MESH_DIR = _MESH_FOLDER / "brain_surfaces"
CEREBRUM_MESH = GLASS_MESH_DIR / "glass_cerebrum.vtk"
CEREBELLUM_MESH = GLASS_MESH_DIR / "glass_cerebellum.vtk"
# Uncarved counterpart of CEREBRUM_MESH: the same smooth hull with nothing
# subtracted, so the backdrop is purely the outer cerebral surface. Paired with
# the same cerebellum mesh, which never had anything carved out of it.
PLAIN_CEREBRUM_MESH = GLASS_MESH_DIR / "plain_cerebrum.vtk"
# The SUIT cerebellum with its mask eroded 1.5 mm before meshing, so it sits at
# a depth comparable to a midthickness cortical surface. Eroding the mask and
# re-meshing is the only sound way to do this: offsetting the finished mesh
# along its normals drives opposite walls of the cerebellar fissures through
# each other and collapses the volume (164,653 -> 42,189 mm3 at 1.5 mm).
# To rebuild at another depth:
#   dist = scipy.ndimage.distance_transform_edt(mask, sampling=[1,1,1])
#   mesh_from_mask(dist > depth_mm, affine, presmooth_vox=1.2, target_mm=1.0)
PIAL_CEREBELLUM_MESH = GLASS_MESH_DIR / "pial_cerebellum.vtk"
# Brain-Stem (label 8) of the 1 mm HarvardOxford subcortical atlas, meshed the
# same way. yabplot's fsLR32k surfaces are cortex only, so without this the
# pial view stops at the cerebrum and the brain looks decapitated from below.
# Spans z -67..-1, i.e. medulla to midbrain top, which tucks under the cortex
# and abuts the cerebellar peduncles rather than floating.
BRAINSTEM_MESH = GLASS_MESH_DIR / "brainstem.vtk"

# Structures carved out of the shipped hull. Kept here so callers can see what
# is in the mesh without opening it; changing it does nothing unless you also
# call rebuild_glass_meshes().
DEFAULT_CARVE = (
    "Hippocampus",
    "Caudate",
    "Putamen",
    "Pallidum",
    "Accumbens-area",
    "Thalamus",
)


# =============================================================================
# plotting path -- read the shipped meshes, nothing else
# =============================================================================


def _read(path, what):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"glass {what} mesh not found at {path}. It ships in brain_surfaces/ of "
            f"the meshes folder; if it is missing, regenerate it with "
            f"gnv.render.glass_mesh.rebuild_glass_meshes()."
        )
    return pv.read(str(path))


def glass_bmesh(cerebrum=None, cerebellum=None, carved=True):
    """Return the ``bmesh`` dict for ``yabplot.plot_voxelwise``.

    Parameters
    ----------
    cerebrum, cerebellum : str or Path, optional
        Override either shipped mesh. Pass ``cerebellum=False`` to draw the
        cerebral hull alone.
    carved : bool
        ``True`` uses the hull with the deep grey structures subtracted out of
        it; ``False`` uses ``plain_cerebrum.vtk``, the same hull with nothing
        subtracted, so the backdrop is only the outer cerebral surface and the
        cerebellum. Ignored when ``cerebrum`` is given explicitly.

    Notes
    -----
    yabplot only special-cases the keys 'L' and 'R' (for hemisphere filtering
    in lateral views). These are whole-brain pieces that belong in every view,
    so they are keyed by name and always drawn.
    """
    default = CEREBRUM_MESH if carved else PLAIN_CEREBRUM_MESH
    bmesh = {"cerebrum": _read(cerebrum or default, "cerebrum")}

    if cerebellum is not False:
        bmesh["cerebellum"] = _read(cerebellum or CEREBELLUM_MESH, "cerebellum")

    return bmesh


BMESH_NAMES = ("midthickness", "pial", "white", "swm", "inflated", "very_inflated")


def _resolve_piece(name):
    """Work out what one entry of a mesh spec refers to.

    Returns ``("bmesh", name)`` for one of yabplot's cortical surfaces, or
    ``("file", Path)`` for a mesh on disk -- either a prebuilt asset in
    ``GLASS_MESH_DIR`` named by its stem, or any path -- or ``("dir", Path)``
    for a folder holding a hemisphere pair.
    """
    if name in BMESH_NAMES:
        return "bmesh", name

    candidate = Path(name).expanduser()
    if candidate.is_dir() and candidate.is_absolute():
        return "dir", candidate
    if candidate.suffix and candidate.exists():
        return "file", candidate

    for folder in MESH_SEARCH_PATH:
        for suffix in MESH_SUFFIXES:
            shipped = folder / f"{name}{suffix}"
            if shipped.is_file():
                return "file", shipped
        # A directory is a hemisphere pair (L.vtk / R.vtk) rather than a single
        # surface; build_bmesh routes its members by their own names.
        grouped = folder / name
        if grouped.is_dir():
            return "dir", grouped

    available = sorted(
        {q.stem for folder in MESH_SEARCH_PATH for suffix in MESH_SUFFIXES
         for q in folder.glob(f"*{suffix}")}
        | {q.name for folder in MESH_SEARCH_PATH for q in folder.iterdir()
           if q.is_dir()})
    raise ValueError(
        f"{name!r} is neither a cortical surface {BMESH_NAMES}, a prebuilt mesh "
        f"{available}, nor an existing file path."
    )


MESH_SUFFIXES = (".vtk", ".vtp", ".ply", ".stl", ".obj")
# Where a bare stem is looked for. Meshes anywhere else are named by path, in
# the mesh index (brains.meshes), rather than found by searching.
MESH_SEARCH_PATH = (GLASS_MESH_DIR,)


def _side_from_stem(stem):
    """Which hemisphere a file inside a group belongs to, or None.

    A pair is written either as ``thing_L`` / ``thing_R`` or, in the Lead-DBS
    sets, as bare ``L`` / ``R``. Without the bare spelling those get cut at the
    midline as if they were whole brains, which quietly halves each hemisphere.
    """
    lowered = stem.lower()
    if lowered in ("l", "lh", "left"):
        return "left"
    if lowered in ("r", "rh", "right"):
        return "right"
    # Names that say a side in words or as a dotted segment -- brainMeshRight,
    # surf.rh.ply. Without these a genuinely one-sided mesh is cut at x = 0 and
    # a sliver of it is filed under the other hemisphere.
    parts = set(lowered.replace("-", ".").replace("_", ".").split("."))
    if lowered.endswith("left") or {"lh", "left"} & parts:
        return "left"
    if lowered.endswith("right") or {"rh", "right"} & parts:
        return "right"
    if stem.upper().endswith("_L"):
        return "left"
    if stem.upper().endswith("_R"):
        return "right"
    return None


MIDLINE_OFFSET = 0.08


def _split_at_midline(mesh):
    """Cut one mesh into left and right halves at x=0.

    yabplot culls a context mesh from a lateral view only when its key is
    exactly 'L' or 'R' (``scene.add_context_to_view``). A cerebellum keyed
    "cerebellum" is therefore drawn in every view, and in a left lateral view
    you look through the far hemisphere as well as the near one -- which is
    what makes a translucent whole-brain mesh stack up.

    Each half is capped so it reads as a solid, and the cut faces are nudged
    ``MIDLINE_OFFSET`` mm past the midline into the opposite half so they are
    not coplanar. Two coincident faces z-fight, which would speckle the very
    plane the cut creates.
    """
    import numpy as np

    halves = []
    for normal in ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)):
        try:
            half = mesh.clip_closed_surface(
                normal=normal, origin=(0.0, 0.0, 0.0), inplace=False
            ).triangulate()
        except ValueError:
            # clip_closed_surface needs a manifold solid, which the carved
            # hull is not -- it has interior cavity walls as separate
            # components. An uncapped cut is fine there: the hull is drawn
            # translucent, so the open face reads as the cut edge either way.
            half = (
                mesh.clip(normal=normal, origin=(0.0, 0.0, 0.0), invert=False)
                .extract_surface()
                .triangulate()
            )
        if half.n_points and MIDLINE_OFFSET:
            pts = np.asarray(half.points).copy()
            on_plane = np.abs(pts[:, 0]) < 1e-6
            pts[on_plane, 0] -= normal[0] * MIDLINE_OFFSET
            half.points = pts
        halves.append(half)
    return halves          # [left, right]


def build_bmesh(pieces):
    """Assemble a ``bmesh`` dict from a list of mesh names or paths.

    One list decides the whole geometry, and each entry says what it is rather
    than toggling something:

    * a cortical surface name -- "pial", "midthickness", "inflated", ... --
      loads yabplot's fsLR32k pair;
    * a prebuilt mesh's stem -- "brainstem", "pial_cerebellum",
      "glass_cerebrum", ... -- loads it from ``GLASS_MESH_DIR``;
    * any path to a ``.vtk`` / ``.gii`` mesh loads that file, and a path to a
      folder of ``L`` / ``R`` meshes loads the pair. (A mesh index entry's
      pieces are paths: see brains.meshes.)

    Returns exactly ``{"L": ..., "R": ...}``: every piece is split at the
    midline and the halves merged, so each hemisphere is one mesh. That is what
    lets yabplot drop the far hemisphere in a lateral view -- keys other than
    'L' and 'R' are drawn in every view, so a cerebellum or brainstem keyed by
    name would stack up behind the cortex you are trying to look at.

    Merging also means one actor and one alpha per side, so a translucent brain
    has uniform density instead of darkening wherever two pieces overlap.
    """
    import numpy as np

    if isinstance(pieces, (str, Path)):
        pieces = [pieces]

    left, right = [], []
    for entry in pieces:
        kind, value = _resolve_piece(str(entry))

        if kind == "bmesh":
            from yabplot.data import get_surface_paths
            from yabplot.utils import load_gii

            lh_path, rh_path = get_surface_paths(value, "bmesh")
            for side, path in ((left, lh_path), (right, rh_path)):
                verts, faces = load_gii(path)
                side.append(pv.PolyData(
                    np.asarray(verts, dtype=np.float32),
                    np.hstack(
                        [np.full((len(faces), 1), 3, np.int64),
                         np.asarray(faces, np.int64)]
                    ).ravel(),
                ))
        else:
            # Already-lateral files (…_L.vtk / …_R.vtk) go straight to a side;
            # anything spanning the midline is cut.
            if kind == "dir":
                for item in sorted(value.iterdir()):
                    if item.suffix.lower() not in MESH_SUFFIXES:
                        continue
                    side = _side_from_stem(item.stem)
                    mesh = _read(item, item.stem)
                    if side == "left":
                        left.append(mesh)
                    elif side == "right":
                        right.append(mesh)
                    else:
                        lh, rh = _split_at_midline(mesh.triangulate())
                        left.append(lh)
                        right.append(rh)
                continue
            mesh = _read(value, value.stem)
            side = _side_from_stem(value.stem)
            if side == "left":
                left.append(mesh)
            elif side == "right":
                right.append(mesh)
            else:
                lh, rh = _split_at_midline(mesh.triangulate())
                left.append(lh)
                right.append(rh)

    if not left and not right:
        raise ValueError("mesh spec is empty; give at least one mesh")

    def _merge(parts):
        parts = [m for m in parts if m.n_points]
        if not parts:
            return None
        out = parts[0]
        for extra in parts[1:]:
            out = out.merge(extra)
        return out

    bmesh = {}
    for key, parts in (("L", left), ("R", right)):
        merged = _merge(parts)
        if merged is not None:
            bmesh[key] = merged
    return bmesh


# =============================================================================
# rebuild path -- not used when plotting
# =============================================================================


def _rebuild_deps():
    try:
        import nibabel as nib
        from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates
        from skimage import measure
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rebuilding the glass meshes needs nibabel, scipy and scikit-image "
            "in this environment. Plotting does not -- it reads the prebuilt "
            "meshes from brain_surfaces/ in the meshes folder."
        ) from exc
    return nib, gaussian_filter, binary_dilation, map_coordinates, measure


def _resample_to(src_img, ref_img, order, nib, map_coordinates):
    import numpy as np

    matrix = np.linalg.inv(src_img.affine) @ ref_img.affine
    grid = np.indices(ref_img.shape, dtype=np.float32).reshape(3, -1)
    grid = np.vstack([grid, np.ones((1, grid.shape[1]), np.float32)])

    source = np.asanyarray(src_img.dataobj)
    if source.ndim > 3:
        source = source[..., 0]

    out = map_coordinates(
        np.asarray(source, dtype=np.float32),
        (matrix @ grid)[:3],
        order=order,
        mode="constant",
        cval=0.0,
    )
    return out.reshape(ref_img.shape)


def _voxelize(mesh, ref_img, nib, pad=3):
    """Rasterise a closed surface onto ref_img's grid, bounding-box limited."""
    import numpy as np

    inverse = np.linalg.inv(ref_img.affine)
    corners = nib.affines.apply_affine(inverse, np.array(mesh.bounds).reshape(3, 2).T)

    lo = np.maximum(np.floor(corners.min(0)).astype(int) - pad, 0)
    hi = np.minimum(np.ceil(corners.max(0)).astype(int) + pad, np.array(ref_img.shape))
    if np.any(hi <= lo):
        return None, None

    axes = [np.arange(lo[d], hi[d]) for d in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)

    selection = pv.PolyData(nib.affines.apply_affine(ref_img.affine, points)).select_enclosed_points(
        mesh.triangulate(), tolerance=0.0, check_surface=False
    )
    inside = selection["SelectedPoints"].astype(bool)
    return inside.reshape(len(axes[0]), len(axes[1]), len(axes[2])), (lo, hi)


def _polys_only(mesh):
    """Drop stray vertex/line cells that cleaning leaves behind.

    `clean()` collapses degenerate triangles into lines, and a PolyData holding
    even a handful of those is rejected by `decimate_pro`.
    """
    import numpy as np

    mesh = mesh.triangulate()
    return pv.PolyData(np.asarray(mesh.points), np.asarray(mesh.faces))


def _aseg_files(structures, atlas="aseg", aseg_dir=None):
    """Resolve aseg region names to mesh paths."""
    if aseg_dir is None:
        # yabplot exposes no public "give me the mesh paths for this atlas"
        # call -- plot_subcortical reaches for these two internals itself.
        # Pinned to private API, so fail loudly if it moves.
        try:
            from yabplot.data import _find_subcortical_files, _resolve_resource_path
        except ImportError as exc:
            raise ImportError(
                "could not import yabplot's atlas helpers. Install yabplot, or pass "
                "aseg_dir=<folder of .vtk meshes> to bypass the lookup entirely."
            ) from exc

        aseg_dir = _resolve_resource_path(atlas, "subcortical")
        file_map = _find_subcortical_files(aseg_dir)
    else:
        file_map = {p.stem: str(p) for p in sorted(Path(aseg_dir).glob("*.vtk"))}

    wanted = {s.lower() for s in structures}
    hits, seen = [], set()
    for name, path in file_map.items():
        bare = name.split("-", 1)[1].lower() if "-" in name else name.lower()
        if bare in wanted:
            hits.append((name, path))
            seen.add(bare)

    missing = wanted - seen
    if missing:
        raise ValueError(f"structures not found in '{atlas}': {sorted(missing)}")
    return sorted(hits)


def rebuild_glass_meshes(
    mask_path,
    cerebellum_labels=None,
    cerebellum_surface=None,
    carve=DEFAULT_CARVE,
    carve_paths=(),
    atlas="aseg",
    aseg_dir=None,
    hull_sigma=4.0,
    cavity_sigma=1.0,
    decimate=0.70,
    smooth_i=25,
    smooth_f=0.10,
    cerebellum_dilation=2,
    out_dir=None,
    verbose=True,
):
    """Regenerate the shipped meshes. Not called when plotting.

    Two blur stages matter. ``hull_sigma`` is applied to the brain mask and the
    result re-binarised, fixing the smooth glass outer shape; ``cavity_sigma``
    is the blur used for the marching cubes itself and is kept small so thin
    cavity walls survive. A single large blur erases small structures outright
    -- pallidum is ~1.4k voxels at 1 mm.

    Parameters
    ----------
    mask_path : str or Path
        Binary brain mask defining the hull (1 mm ICBM152 works well).
    cerebellum_labels : str or Path, optional
        Cerebellar label/mask volume, any affine -- resampled onto the mask
        grid and subtracted so the cerebellum can carry its own mesh.
    cerebellum_surface : str or Path, optional
        Directory holding ``surface/Cerebellum_L.vtk`` and ``_R.vtk`` (the SUIT
        export layout), or a single mesh file. Merged and written out as the
        cerebellum piece.
    carve : sequence of str
        aseg region names to subtract as interior voids.
    carve_paths : sequence of str or Path
        Extra NIfTI ROIs to subtract, thresholded at 0.35 occupancy.
    """
    import numpy as np

    nib, gaussian_filter, binary_dilation, map_coordinates, measure = _rebuild_deps()

    out_dir = Path(out_dir) if out_dir else GLASS_MESH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = nib.load(str(mask_path))
    brain = np.asarray(np.asanyarray(ref.dataobj), dtype=np.float32) > 0.5

    if cerebellum_labels is not None:
        cerebellum = _resample_to(nib.load(str(cerebellum_labels)), ref, 0, nib, map_coordinates) > 0
        if cerebellum_dilation:
            cerebellum = binary_dilation(cerebellum, iterations=cerebellum_dilation)
        brain = brain & ~cerebellum

    hull = gaussian_filter(brain.astype(np.float32), sigma=hull_sigma) > 0.5

    voids = np.zeros_like(hull)
    for name, path in (_aseg_files(carve, atlas, aseg_dir) if carve else []):
        inside, box = _voxelize(pv.read(path), ref, nib)
        if inside is None:
            continue
        lo, hi = box
        voids[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= inside
        if verbose:
            print(f"  carve {name}: {int(inside.sum())} voxels")

    for path in carve_paths:
        roi = _resample_to(nib.load(str(path)), ref, 1, nib, map_coordinates) > 0.35
        voids |= roi
        if verbose:
            print(f"  carve {Path(path).name}: {int(roi.sum())} voxels")

    volume = gaussian_filter((hull & ~voids).astype(np.float32), sigma=cavity_sigma)
    verts, faces, _, _ = measure.marching_cubes(volume, level=0.5)
    verts = nib.affines.apply_affine(ref.affine, verts)

    mesh = pv.PolyData(
        verts.astype(np.float32),
        np.hstack([np.full((faces.shape[0], 1), 3, np.int64), faces.astype(np.int64)]).ravel(),
    )
    # never extract_largest here: the cavities are separate connected
    # components and keeping only the biggest would discard every one of them.
    mesh = _polys_only(mesh.clean())
    if decimate:
        mesh = mesh.decimate_pro(decimate, preserve_topology=True)
    mesh = _polys_only(mesh.smooth_taubin(n_iter=smooth_i, pass_band=smooth_f).clean())
    mesh.clear_data()  # normals are recomputed by smooth_shading at draw time
    mesh.save(str(out_dir / CEREBRUM_MESH.name), binary=True)

    written = [out_dir / CEREBRUM_MESH.name]
    if verbose:
        components = len(np.unique(mesh.connectivity("all")["RegionId"]))
        print(f"cerebrum: {mesh.n_cells} cells, {components} components -> {written[0]}")

    if cerebellum_surface is not None:
        path = Path(cerebellum_surface).expanduser()
        if path.is_file():
            cb = pv.read(str(path))
        else:
            parts = []
            for stem in ("Cerebellum_L", "Cerebellum_R"):
                for sub in (path / "surface", path):
                    hit = sub / f"{stem}.vtk"
                    if hit.exists():
                        parts.append(pv.read(str(hit)))
                        break
            if not parts:
                raise FileNotFoundError(f"no Cerebellum_L/R.vtk found under {path}")
            cb = parts[0]
            for extra in parts[1:]:
                cb = cb.merge(extra)

        cb = cb.clean()
        cb.clear_data()
        cb.save(str(out_dir / CEREBELLUM_MESH.name), binary=True)
        written.append(out_dir / CEREBELLUM_MESH.name)
        if verbose:
            print(f"cerebellum: {cb.n_cells} cells -> {written[-1]}")

    return written

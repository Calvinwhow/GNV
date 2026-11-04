"""
parcel_meshes.py
================
Build yabplot-compatible parcel meshes from a directory of ROI masks.

This is the generalisation of :mod:`suit_cerebellum`, which does the same job
for one specific atlas. Point it at a folder of binary NIfTI ROIs and it emits
one smoothed ``.vtk`` per region plus an ``atlas_LUT.txt``, in the layout
``yabplot.data._find_subcortical_files`` expects -- so the output drops into
``ParcelwisePlot`` as a ``cerebellum_atlas_path`` / ``custom_atlas_path``, or
sits beside another atlas's meshes to parcellate a whole brain.

The mesh pipeline is shared with :mod:`suit_cerebellum` (that module imports
:func:`mesh_from_mask` from here), so cerebral and cerebellar parcels built by
the two look like each other and can be mixed in one directory.

Why the pipeline looks the way it does
--------------------------------------
Three things decide whether a mask-derived mesh looks hand-made or blocky, and
all three are easy to get wrong:

1. **Blur the mask, not the mesh.** Marching cubes on a binary mask can only
   produce stair-steps at the voxel scale, and mesh relaxation then fights
   geometry that should never have existed. ``presmooth_vox`` blurs the mask
   first and contours a sub-voxel iso-surface.
2. **Keep the volume.** A blur spreads mass outward, so contouring at a fixed
   0.5 erodes convex structures. The iso-level is chosen so the enclosed voxel
   count matches the mask, which decouples fidelity from how hard you blur.
3. **Contour on a fine enough grid.** Marching cubes puts vertices one voxel
   apart, so a 2 mm atlas yields 2 mm facets that no amount of smoothing can
   remove -- the triangles themselves are the chunkiness. ``target_mm``
   upsamples the blurred field before contouring so vertex spacing is set by
   the output you want, not by the atlas's sampling. A 1 mm atlas is already
   there and is left untouched.
4. **Scale to the structure.** A blur wider than a small ROI's own radius
   destroys it. Both the blur and the smoothing iterations taper below a
   characteristic radius of ~8 voxels, so a 30-voxel nucleus survives while a
   10,000-voxel gyrus gets the full treatment.

``presmooth_vox`` is in **voxels**, not millimetres. That is deliberate and
it is the one unit that transfers between atlases: stair-steps are an artefact
of the sampling grid, so their amplitude is one voxel whatever the voxel
measures in millimetres. A blur fixed in millimetres under-smooths coarse data
-- 1.2 mm on a 2 mm atlas is 0.6 voxels, barely half the blur the same setting
gives a 1 mm atlas, and the steps survive.

Lateralisation
--------------
Many ROI sets ship bilaterally merged -- one ``Insula.nii.gz`` spanning both
hemispheres. A single mesh crossing the midline cannot be culled from lateral
views, so :class:`ParcelMeshAtlas` splits such masks at x=0 into ``_L`` / ``_R``
pairs. Structures that genuinely sit on the midline are detected (most of their
voxels within ``midline_mm`` of x=0) and left whole.

Region names
------------
``_plot_cortical_with_cerebellum`` decides which side a mesh belongs to with a
substring test -- it drops names containing ``"_r"`` from left views and
``"_l"`` from right views. That silently breaks names like ``Thal_LGN_R`` or
``Paracentral_Lobule_L``, which contain *both* tokens and so vanish from every
lateral view. :func:`sanitize_region_name` rewrites the offending separator
before the side suffix is added, and the QC report lists anything it changed.

Usage
-----
::

    from gnv.convert.parcel_meshes import (
        ParcelMeshAtlas,
    )

    ParcelMeshAtlas(
        parcel_dir="/Volumes/HowExp/.../AAL_MNI_V7_fine_rois",
        out_dir="~/hires_backdrops/aal/yabplot_aal",
        exclude=("Cerebellum", "Vermis"),   # SUIT supplies those
    ).run()

CLI
---
::

    python -m gnv.convert.parcel_meshes \
        --parcels /path/to/AAL_MNI_V7_fine_rois \
        --out ~/hires_backdrops/aal/yabplot_aal \
        --exclude Cerebellum Vermis
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Iterable, Sequence

import numpy as np


__all__ = [
    "mesh_from_mask",
    "sanitize_region_name",
    "assert_name_is_plottable",
    "ParcelMeshAtlas",
]


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #

def sanitize_region_name(name: str) -> str:
    """Remove ``_l`` / ``_r`` tokens from a region's base name.

    The plotter's per-view hemisphere test is a substring check, so an
    underscore followed by l or r anywhere in the name reads as a side marker.
    ``Thal_LGN`` + ``_R`` would contain both ``_l`` and ``_r`` and be dropped
    from every lateral view. Swapping that separator for a hyphen keeps the
    name readable and unambiguous: ``Thal-LGN_R``.
    """
    return re.sub(r"_(?=[lLrR])", "-", name)


def assert_name_is_plottable(name: str, midline: bool = False) -> None:
    """Fail loudly on a name the per-view hemisphere test would mishandle."""
    low = name.lower()
    has_l, has_r = "_l" in low, "_r" in low
    if has_l and has_r:
        raise ValueError(
            f"{name!r} contains both '_l' and '_r', so lateral views would drop "
            f"it entirely. Run it through sanitize_region_name() first."
        )
    if midline and (has_l or has_r):
        raise ValueError(
            f"midline region {name!r} carries a side token and would vanish "
            f"from one lateral view."
        )


# --------------------------------------------------------------------------- #
# Mesh pipeline (shared with suit_cerebellum)
# --------------------------------------------------------------------------- #

def _adaptive(n_voxels: int, sigma_vox: float, smooth_i: int) -> tuple[float, int]:
    """Taper blur and smoothing for structures smaller than the blur."""
    radius = (3.0 * max(n_voxels, 1) / (4.0 * np.pi)) ** (1.0 / 3.0)
    sigma = min(sigma_vox, radius / 4.0)
    if sigma < 0.25:
        sigma = 0.0
    n_iter = max(5, int(round(smooth_i * min(1.0, radius / 8.0))))
    return sigma, n_iter


def _volume_preserving_level(field: np.ndarray, n_target: int) -> float:
    """Iso-level enclosing ``n_target`` voxels of ``field``."""
    flat = field.ravel()
    if n_target <= 0 or n_target >= flat.size:
        return 0.5
    level = float(np.partition(flat, -n_target)[-n_target])
    return level if 1e-3 < level < 1.0 - 1e-3 else 0.5


def _drop_small_components(mask: np.ndarray, frac: float, verbose: bool) -> np.ndarray:
    """Remove speckle islands well below the region's main body."""
    if frac <= 0:
        return mask
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.flatnonzero(sizes >= sizes.max() * frac) + 1
    if len(keep) < n and verbose:
        print(f"       dropped {n - len(keep)} speckle component(s)")
    return np.isin(lab, keep)


def mesh_from_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    presmooth_vox: float = 1.2,
    target_mm: float = 1.0,
    preserve_volume: bool = True,
    smooth: str = "taubin",
    smooth_i: int = 40,
    smooth_f: float = 0.05,
    decimate: float | None = 0.4,
    min_component_frac: float = 0.05,
    verbose: bool = False,
):
    """Turn a binary mask into a smoothed surface in world coordinates.

    Returns a ``pyvista.PolyData``, or ``None`` when the mask is too small or
    too thin to form one. ``presmooth_vox`` is in voxels, so the same value
    removes the same stair-steps at any resolution; ``target_mm`` is the
    vertex spacing to contour at, reached by upsampling coarser data (set it
    to ``None`` to contour on the atlas's own grid).
    """
    import nibabel as nib
    import pyvista as pv
    from skimage import measure

    mask = _drop_small_components(mask, min_component_frac, verbose)
    n_target = int(mask.sum())
    if n_target < 4:
        return None

    sigma, n_iter = _adaptive(n_target, presmooth_vox, smooth_i)

    # Pad so structures touching the array edge close instead of leaving open
    # faces, then shift vertices back into the original index frame.
    pad = max(1, int(np.ceil(3 * sigma))) if sigma else 1
    field = np.pad(mask.astype(np.float32), pad, mode="constant")

    zoom_mm = float(np.mean(np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))))
    up = 1 if not target_mm else max(1, int(round(zoom_mm / target_mm)))
    grid_affine = np.asarray(affine, dtype=float).copy()

    if sigma or up > 1:
        from scipy import ndimage

        if sigma:
            field = ndimage.gaussian_filter(field, sigma=sigma)
        if up > 1:
            # Resample onto the finer grid the surface will be contoured on.
            # Blurring first and interpolating second keeps the iso-surface
            # smooth; the reverse just interpolates the stair-steps.
            field = ndimage.zoom(field, up, order=3)
            grid_affine[:3, :3] = grid_affine[:3, :3] / up
            # zoom() maps fine index j to coarse (j + 0.5)/up - 0.5
            grid_affine[:3, 3] = grid_affine[:3, 3] + np.asarray(affine)[:3, :3] @ (
                np.full(3, 0.5 / up - 0.5)
            )
        if preserve_volume:
            level = _volume_preserving_level(field, n_target * up**3)
        else:
            level = 0.5
    else:
        level = 0.5

    try:
        verts, faces, _, _ = measure.marching_cubes(field, level=level)
    except (ValueError, RuntimeError):
        return None

    verts -= float(pad * up)
    verts_world = nib.affines.apply_affine(grid_affine, verts)
    faces_pv = (
        np.column_stack((np.full(len(faces), 3), faces)).astype(np.int64).ravel()
    )

    # No clean() here: marching cubes already emits shared vertices, and
    # PolyData.clean() collapses some into degenerate cells that decimate_pro
    # then rejects as non-triangular.
    mesh = pv.PolyData(verts_world, faces_pv)
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    if decimate and mesh.is_all_triangles:
        mesh = mesh.decimate_pro(decimate, preserve_topology=True)

    if smooth == "taubin":
        mesh = mesh.smooth_taubin(
            n_iter=n_iter, pass_band=smooth_f, normalize_coordinates=True
        )
    elif smooth == "laplacian":
        mesh = mesh.smooth(n_iter=n_iter, relaxation_factor=smooth_f)
    elif smooth not in (None, "none"):
        raise ValueError(f"Unknown smoother: {smooth!r}")

    mesh.compute_normals(inplace=True, auto_orient_normals=True)
    if mesh.n_points < 4 or abs(mesh.volume) < 0.01:
        return None
    return mesh


# --------------------------------------------------------------------------- #
# Atlas builder
# --------------------------------------------------------------------------- #

class ParcelMeshAtlas:
    """Directory of binary ROI NIfTIs -> directory of yabplot ``.vtk`` meshes.

    Parameters
    ----------
    parcel_dir
        Folder of one binary NIfTI per region. Region names come from the
        filenames. macOS ``._`` resource forks are ignored.
    out_dir
        Meshes are written to ``<out_dir>/subcortical`` (the name yabplot's
        resource lookup expects), alongside ``atlas_LUT.txt`` and a QC report.
    exclude, include
        Case-insensitive substrings filtering which regions are built. Use
        ``exclude=("Cerebellum", "Vermis")`` when another atlas supplies those.
    split_midline
        Split bilaterally-merged masks into ``_L`` / ``_R`` at x=0 so lateral
        views can cull the far hemisphere.
    midline_mm, midline_frac
        A region with more than ``midline_frac`` of its voxels within
        ``midline_mm`` of x=0 is treated as a midline structure and kept whole.
    min_voxels
        Regions smaller than this are skipped; below a few voxels there is no
        surface worth drawing.
    """

    def __init__(
        self,
        parcel_dir: str | pathlib.Path,
        out_dir: str | pathlib.Path,
        exclude: Sequence[str] = (),
        include: Sequence[str] = (),
        split_midline: bool = True,
        midline_mm: float = 1.0,
        midline_frac: float = 0.15,
        min_voxels: int = 12,
        presmooth_vox: float = 1.2,
        target_mm: float = 1.0,
        smooth: str = "taubin",
        smooth_i: int = 40,
        smooth_f: float = 0.05,
        decimate: float | None = 0.4,
        min_component_frac: float = 0.05,
    ) -> None:
        self.parcel_dir = pathlib.Path(parcel_dir).expanduser()
        self.out_dir = pathlib.Path(out_dir).expanduser()
        self.mesh_dir = self.out_dir / "subcortical"
        self.exclude = tuple(e.lower() for e in exclude)
        self.include = tuple(i.lower() for i in include)
        self.split_midline = split_midline
        self.midline_mm = midline_mm
        self.midline_frac = midline_frac
        self.min_voxels = min_voxels
        self.mesh_kwargs = dict(
            presmooth_vox=presmooth_vox,
            target_mm=target_mm,
            smooth=smooth,
            smooth_i=smooth_i,
            smooth_f=smooth_f,
            decimate=decimate,
            min_component_frac=min_component_frac,
        )
        self.written: list[tuple[int, str]] = []
        self.report: list[dict] = []
        self.renamed: dict[str, str] = {}
        self.skipped: list[tuple[str, str]] = []

    # -- public ------------------------------------------------------------ #

    def parcel_files(self) -> list[pathlib.Path]:
        files = sorted(
            p
            for p in self.parcel_dir.iterdir()
            if p.name.endswith((".nii", ".nii.gz")) and not p.name.startswith("._")
        )
        if not files:
            raise FileNotFoundError(f"No NIfTI parcels found in {self.parcel_dir}")
        return [p for p in files if self._wanted(self._stem(p))]

    def run(self) -> dict:
        import nibabel as nib

        files = self.parcel_files()
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

        rid = 1
        for path in files:
            raw = self._stem(path)
            base = sanitize_region_name(raw)
            if base != raw:
                self.renamed[raw] = base

            img = nib.load(str(path))
            mask = np.asarray(img.dataobj) > 0
            n_vox = int(mask.sum())
            if n_vox < self.min_voxels:
                self.skipped.append((raw, f"{n_vox} voxels"))
                print(f"[skip] {raw}: {n_vox} voxels")
                continue

            for name, piece, midline in self._sides(base, mask, img.affine):
                assert_name_is_plottable(name, midline=midline)
                mesh = mesh_from_mask(piece, img.affine, verbose=True, **self.mesh_kwargs)
                if mesh is None:
                    self.skipped.append((name, "no mesh"))
                    print(f"[skip] {name}: no mesh")
                    continue

                mesh.save(self.mesh_dir / f"{name}.vtk")
                self.written.append((rid, name))
                self.report.append(
                    {
                        "name": name,
                        "source": raw,
                        "voxels": int(piece.sum()),
                        "points": mesh.n_points,
                        "volume": float(abs(mesh.volume)),
                        "centre_x": float(mesh.center[0]),
                    }
                )
                print(
                    f"[ok]   {name:28s} {int(piece.sum()):6d} vox -> "
                    f"{mesh.n_points:6d} pts  {abs(mesh.volume):9.1f} mm3"
                )
                rid += 1

        self._write_lut()
        self._write_report()
        return {
            "out_dir": self.out_dir,
            "mesh_dir": self.mesh_dir,
            "n_meshes": len(self.written),
            "names": [n for _, n in self.written],
            "renamed": self.renamed,
            "skipped": self.skipped,
        }

    # -- internals --------------------------------------------------------- #

    @staticmethod
    def _stem(path: pathlib.Path) -> str:
        name = path.name
        for suffix in (".nii.gz", ".nii"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name.replace(" ", "_").replace("/", "-")

    def _wanted(self, name: str) -> bool:
        low = name.lower()
        if self.include and not any(i in low for i in self.include):
            return False
        return not any(e in low for e in self.exclude)

    def _sides(self, base: str, mask: np.ndarray, affine):
        """Yield ``(name, mask, is_midline)`` for one source ROI."""
        import nibabel as nib

        if not self.split_midline:
            yield base, mask, False
            return

        idx = np.argwhere(mask)
        x = nib.affines.apply_affine(affine, idx)[:, 0]
        near = np.abs(x) <= self.midline_mm

        # A midline structure sits *on* x=0; a paired one straddles it with a
        # gap. Splitting the former would halve a single anatomical body.
        if near.mean() > self.midline_frac:
            yield base, mask, True
            return

        left = np.zeros_like(mask)
        right = np.zeros_like(mask)
        left[tuple(idx[x < 0].T)] = True
        right[tuple(idx[x > 0].T)] = True

        for name, piece in ((f"{base}_L", left), (f"{base}_R", right)):
            if piece.sum() >= self.min_voxels:
                yield name, piece, False

    def _write_lut(self) -> None:
        with open(self.mesh_dir / "atlas_LUT.txt", "w") as fh:
            for rid, name in self.written:
                fh.write(f"{rid} {name}\n")

    def _write_report(self) -> None:
        with open(self.mesh_dir / "qc_mesh_properties.txt", "w") as fh:
            fh.write("region\tsource\tvoxels\tpoints\tvolume_mm3\tcentre_x\n")
            for row in self.report:
                fh.write(
                    f"{row['name']}\t{row['source']}\t{row['voxels']}\t"
                    f"{row['points']}\t{row['volume']:.1f}\t{row['centre_x']:.1f}\n"
                )
            if self.renamed:
                fh.write("\n# renamed to keep the _L/_R view test unambiguous\n")
                for old, new in sorted(self.renamed.items()):
                    fh.write(f"# {old} -> {new}\n")
            if self.skipped:
                fh.write("\n# skipped\n")
                for name, why in self.skipped:
                    fh.write(f"# {name}: {why}\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cli(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build yabplot parcel meshes from ROI masks.")
    ap.add_argument("--parcels", required=True, help="directory of binary ROI NIfTIs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--include", nargs="*", default=[])
    ap.add_argument("--no-split", action="store_true", help="keep bilateral ROIs whole")
    ap.add_argument("--target-mm", type=float, default=1.0,
                    help="vertex spacing to contour at; coarser atlases are upsampled")
    ap.add_argument("--presmooth-vox", type=float, default=1.2,
                help="mask blur in voxels (not mm); 1.2 suits most atlases")
    ap.add_argument("--smooth-i", type=int, default=40)
    ap.add_argument("--smooth-f", type=float, default=0.05)
    ap.add_argument("--decimate", type=float, default=0.4)
    ap.add_argument("--min-voxels", type=int, default=12)
    args = ap.parse_args(argv)

    info = ParcelMeshAtlas(
        parcel_dir=args.parcels,
        out_dir=args.out,
        exclude=args.exclude,
        include=args.include,
        split_midline=not args.no_split,
        presmooth_vox=args.presmooth_vox,
        target_mm=args.target_mm,
        smooth_i=args.smooth_i,
        smooth_f=args.smooth_f,
        decimate=args.decimate,
        min_voxels=args.min_voxels,
    ).run()

    print(f"\n{info['n_meshes']} meshes -> {info['mesh_dir']}")
    if info["renamed"]:
        print(f"renamed {len(info['renamed'])}: " +
              ", ".join(f"{k}->{v}" for k, v in sorted(info["renamed"].items())))
    if info["skipped"]:
        print(f"skipped {len(info['skipped'])}: " +
              ", ".join(f"{n} ({w})" for n, w in info["skipped"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

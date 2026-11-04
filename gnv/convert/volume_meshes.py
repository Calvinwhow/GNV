"""Turn a NIfTI into drawable geometry, so any volume can become a layer.

A lesion, a stimulation volume, a thresholded cluster, a whole atlas -- these
arrive as volumes, not meshes, and the only thing standing between them and a
figure is a surface.  This builds that surface with the SAME pipeline that made
the SUIT and AAL parcel meshes (``nifti_utils.parcel_meshes.mesh_from_mask``),
so a lesion dropped in today sits beside those meshes and looks like it belongs:
identical presmoothing, volume-preserving iso level, Taubin smoothing and
decimation.

What a volume becomes depends on what is in it, and ``split`` says which:

``single``      everything above the threshold, as one surface.  A lesion mask,
                a VTA, a thresholded cluster map.
``labels``      one surface per distinct integer value -- any atlas or
                segmentation becomes a region GROUP, stylable exactly like the
                built-in parcel sets.
``components``  one surface per connected blob above the threshold, which is how
                a multi-cluster statistical map becomes separately colourable
                pieces without anyone having to label them first.
``volumes``     one surface per volume along the 4th dimension -- the usual
                layout of a probabilistic or multi-map atlas, one parcel per
                frame. Each frame is "inside" where its value > iso. Names
                come from ``names`` or a sidecar labels file (see
                ``frame_names``), otherwise ``<file>_<index>``.
``auto``        ``volumes`` for a 4D file with more than one frame; otherwise
                ``labels`` when the volume looks like a segmentation (integer
                valued, a handful of distinct values), otherwise ``single``.

Auto exists because getting this wrong is expensive in both directions: running
marching cubes 400 times over what was actually a probability map, or flooding
an entire atlas into one blob.  It guesses from the data, and the answer it
picked is returned so the caller can show it rather than leave the user
wondering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["meshes_from_volume", "volume_summary", "frame_names", "n_frames",
           "label_table", "read_label_table", "build_atlas",
           "SPLITS", "MAX_LABELS",
           "MIN_VOXELS", "MAX_MESHES"]

SPLITS = ("auto", "single", "labels", "components", "volumes")

# Above this many distinct values a volume is not a segmentation, whatever its
# dtype says; it is a parametric map that happens to be integer-valued, and
# meshing every level of it would take minutes and produce nothing legible.
MAX_LABELS = 256

# Marching cubes plus smoothing costs roughly a second per structure, so a
# threshold that leaves 400 speckles is minutes of work for a figure nobody can
# read. Small pieces are dropped by VOXEL COUNT before any of them is meshed,
# and only the largest survivors are built; what got dropped is reported rather
# than silently discarded, because "my cluster vanished" must be answerable.
MIN_VOXELS = 27          # 3x3x3 -- below this nothing survives smoothing anyway
MAX_MESHES = 64

# Matching the parcel meshes: voxel-space presmoothing, 1 mm contouring.
DEFAULT_SMOOTHING = dict(presmooth_vox=1.2, target_mm=1.0, smooth_i=40,
                         smooth_f=0.05, decimate=0.4, min_component_frac=0.05)


def read_backdrop_npz(path):
    """``(data, affine)`` for a prepared backdrop, as if it were a NIfTI.

    A backdrop is already the anatomy resampled, quantised to uint8 and stored
    slice by slice, so reading one costs nothing next to meshing it. Accepting
    it wherever a volume is accepted is what lets a prepared backdrop be the
    surface AND the thing that colours it, instead of only a set of planes.
    """
    from gnv.convert.backdrops import Backdrop

    with Backdrop(path) as store:
        return store.volume().astype(np.float32), np.asarray(store.affine, float)


def is_backdrop(path):
    return str(path).lower().endswith(".npz")


def _load(path, frames=False):
    """``(data, affine)``. A 4D file gives its first volume unless ``frames``,
    in which case the whole (x, y, z, t) array comes back."""
    import nibabel as nib

    path = Path(path).expanduser()
    if is_backdrop(path):
        return read_backdrop_npz(path)
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim > 4:
        data = data.reshape(data.shape[:3] + (-1,))
    if data.ndim == 4 and not frames:
        data = data[..., 0]
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), np.asarray(img.affine, float)


def n_frames(path):
    """Volumes along the 4th dimension (1 for a 3D file or a backdrop)."""
    import nibabel as nib

    if is_backdrop(path):
        return 1
    shape = nib.load(str(Path(path).expanduser())).shape
    return int(np.prod(shape[3:])) if len(shape) > 3 else 1


LABEL_SIDECARS = (".txt", ".csv", ".tsv", ".lut", ".json")


def _is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def read_label_table(side):
    """``[(index or None, name, rgb or None), ...]`` in file order.

    Reads a JSON list / {index: name} object, or a text table with one region
    per line in any of the usual layouts: ``name``; ``index name``; SUIT /
    MRIcroGL style ``index r g b name``; FreeSurfer style ``index name r g b a``.
    Three numbers straight after the index, or three or more at the end of the
    line, are the colour (0-1 or 0-255); everything else is the name, numbers
    included, so HCP-MMP's area "1" in ``27 0.74 0.09 0.46 1 L`` is "1 L". A
    last number equal to the index (``27 L 1 27``) is a repeat of it. Comment
    (#) and header lines are skipped.
    """
    import json

    side = Path(side)
    rows = []
    if side.suffix == ".json":
        raw = json.loads(side.read_text())
        if isinstance(raw, dict):
            for key in sorted(raw, key=lambda k: int(k)):
                rows.append((int(key), str(raw[key]), None))
        else:
            rows = [(None, str(v), None) for v in raw]
        return rows
    for line in side.read_text().splitlines():
        parts = [p for p in line.replace(",", " ").replace("\t", " ").split() if p]
        if not parts or parts[0].startswith("#"):
            continue
        index = int(parts[0]) if parts[0].isdigit() else None
        rest = parts[1:] if index is not None else parts
        trailing = len(rest) - next((i for i in range(len(rest) - 1, -1, -1) if not _is_number(rest[i])), -1) - 1
        numbers = []
        if len(rest) > 3 and all(_is_number(p) for p in rest[:3]):
            numbers, rest = [float(p) for p in rest[:3]], rest[3:]
        elif trailing >= 3 and trailing < len(rest):
            numbers, rest = [float(p) for p in rest[-trailing:]], rest[:-trailing]
        if len(rest) > 1 and index is not None and rest[-1] == str(index):
            rest = rest[:-1]
        if not rest or all(_is_number(p) for p in rest) and index is None:
            continue
        name = " ".join(rest)
        if index is None and name.lower() in ("name", "label", "region", "index name"):
            continue                      # a header row
        rgb = None
        if len(numbers) >= 3:
            rgb = numbers[:3]
            if max(rgb) > 1.0:
                rgb = [v / 255.0 for v in rgb]
            rgb = [min(max(v, 0.0), 1.0) for v in rgb]
        rows.append((index, name, rgb))
    return rows


def _sidecars(path):
    """Label tables that can describe ``path``, most specific first.

    ``<stem>.<ext>``, ``<stem>_labels.<ext>``, then any table in the same folder
    whose name the volume's name starts with (BIDS: ``atl-Anatom.lut`` describes
    ``atl-Anatom_space-MNI_dseg.nii``), then ``labels.<ext>``.
    """
    path = Path(path).expanduser()
    stem = path.name.split(".nii")[0]
    exact = [path.with_name(f"{stem}{ext}") for ext in LABEL_SIDECARS]
    exact += [path.with_name(f"{stem}_labels{ext}") for ext in LABEL_SIDECARS]
    prefix = sorted((p for p in path.parent.iterdir()
                     if p.suffix.lower() in LABEL_SIDECARS and p.is_file()
                     and stem.startswith(p.stem) and p not in exact),
                    key=lambda p: -len(p.stem)) if path.parent.is_dir() else []
    generic = [path.with_name(f"labels{ext}") for ext in LABEL_SIDECARS]
    return [p for p in exact + prefix + generic if p.is_file()]


def label_table(path):
    """The first readable sidecar table for ``path``, or ``[]``."""
    for side in _sidecars(path):
        try:
            rows = read_label_table(side)
        except (OSError, ValueError, KeyError):
            continue
        if rows:
            return rows
    return []


def frame_names(path, count):
    """``(names, colours)`` for the frames of a 4D atlas.

    Frame i takes the i-th row of the sidecar table (see ``label_table``);
    frames it does not cover are named ``<stem>_<index>``. ``colours`` is a list
    of rgb triples, or None when the table carries no colours.
    """
    stem = Path(path).name.split(".nii")[0]
    rows = label_table(path)
    names = [f"{stem}_{i:03d}" for i in range(count)]
    colours = [None] * count
    for i, (_index, name, rgb) in enumerate(rows[:count]):
        names[i], colours[i] = name, rgb
    return names, (colours if any(c is not None for c in colours) else None)


def _crop(mask, affine, margin=6):
    """``(mask, affine)`` cut to the mask's bounding box plus ``margin`` voxels.

    Meshing upsamples and blurs the WHOLE array it is given, so a small parcel
    handed over on a full-brain grid costs as much as the whole brain: a 34-frame
    atlas took ~70 s, cropped ~5 s. The margin covers the presmoothing kernel.
    Surfaces agree with uncropped ones to within ~0.6 mm / ~4% area on the SUIT
    atlas: mesh_from_mask's upsampling places the fine grid up to a quarter
    voxel off its affine, and where that error falls depends on the array's
    extent, so cropping moves it slightly rather than adding a new one.
    """
    hit = np.argwhere(mask)
    if not hit.size:
        return mask, affine
    lo = np.maximum(hit.min(axis=0) - margin, 0)
    hi = np.minimum(hit.max(axis=0) + margin + 1, mask.shape)
    shift = np.eye(4)
    shift[:3, 3] = lo
    return (mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]],
            np.asarray(affine, float) @ shift)


def _looks_like_labels(data):
    nonzero = data[data != 0]
    if not nonzero.size:
        return False
    if not np.allclose(nonzero, np.round(nonzero)):
        return False
    return 1 < len(np.unique(nonzero)) <= MAX_LABELS


def volume_summary(path):
    """What is in a volume, for the UI to show before anything is meshed."""
    data, _ = _load(path)
    frames = n_frames(path)
    nonzero = data[data != 0]
    labels = _looks_like_labels(data)
    return {
        "shape": list(data.shape) + ([frames] if frames > 1 else []),
        "frames": frames,
        "min": float(data.min()), "max": float(data.max()),
        "nonzero": int(nonzero.size),
        "distinct": int(len(np.unique(nonzero))) if nonzero.size else 0,
        "looks_like_labels": bool(labels),
        "suggested_split": ("volumes" if frames > 1 else
                            "labels" if labels else "single"),
    }


def meshes_from_volume(path, iso=None, split="auto", names=None,
                       min_voxels=MIN_VOXELS, max_meshes=MAX_MESHES,
                       verbose=False, **smoothing):
    """``({name: PolyData}, info)`` for one volume.

    ``iso`` is the threshold that defines "inside": voxels with ``value > iso``.
    It defaults to 0, which is what a mask or an already-thresholded map wants.
    For a raw statistical map, set it to the same number you would threshold at.

    Pieces under ``min_voxels`` are dropped. ``components`` builds at most
    ``max_meshes`` of the largest blobs; ``labels`` and ``volumes`` are named
    regions someone asked for, so they are capped only at MAX_LABELS.
    ``info["skipped"]`` and ``info["over_cap"]`` name what did not make it.
    """
    from gnv.convert.parcel_meshes import (
        mesh_from_mask, sanitize_region_name,
    )

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    frames = n_frames(path)
    if split == "auto" and frames > 1:
        split = "volumes"
    data, affine = _load(path, frames=(split == "volumes"))
    if split == "volumes" and data.ndim == 3:
        data = data[..., None]           # a 3D file is a one-frame stack
    iso = 0.0 if iso is None else float(iso)
    options = dict(DEFAULT_SMOOTHING, **{k: v for k, v in smoothing.items()
                                         if v is not None})

    if split == "auto":
        split = "labels" if _looks_like_labels(data) else "single"

    masks, colours = {}, {}
    if split == "volumes":
        if isinstance(names, (list, tuple)):
            labels_for, rgb_for = list(names), None
        else:
            labels_for, rgb_for = frame_names(path, data.shape[3])
        for index in range(data.shape[3]):
            name = sanitize_region_name(str(labels_for[index])) or f"frame_{index:03d}"
            while name in masks:
                name += "_"
            masks[name] = data[..., index] > iso
            if rgb_for and rgb_for[index] is not None:
                colours[name] = rgb_for[index]
    elif split == "labels":
        table = {} if names else {i: (n, c) for i, n, c in label_table(path) if i is not None}
        for value in np.unique(data[data != 0]):
            label = int(round(float(value)))
            if names:
                name = names.get(label, f"label_{label}")
            else:
                name = table.get(label, (f"label_{label}", None))[0]
            name = sanitize_region_name(str(name))
            masks[name] = data == value
            if label in table and table[label][1] is not None:
                colours[name] = table[label][1]
    elif split == "components":
        from scipy import ndimage

        tagged, count = ndimage.label(data > iso)
        # Largest first, so "component_1" is the one someone means by "the
        # cluster" and colour choices stay stable if the threshold nudges.
        sizes = np.bincount(tagged.ravel())[1:] if count else np.zeros(0)
        for rank, index in enumerate(np.argsort(sizes)[::-1], start=1):
            masks[f"component_{rank}"] = tagged == (index + 1)
    else:
        masks[Path(path).name.split(".nii")[0]] = data > iso

    # Drop the specks before meshing anything, then keep the largest.
    sized = [(name, mask, int(mask.sum())) for name, mask in masks.items()]
    too_small = [name for name, _, n in sized if n < int(min_voxels)]
    sized = [item for item in sized if item[2] >= int(min_voxels)]
    cap = int(max_meshes) if split == "components" else MAX_LABELS
    if split == "components":
        sized.sort(key=lambda item: -item[2])     # largest first, then capped
    over_cap = [name for name, _, _ in sized[cap:]]
    masks = {name: mask for name, mask, _ in sized[:cap]}

    meshes, skipped = {}, list(too_small)
    for name, mask in masks.items():
        small, small_affine = _crop(mask, affine)
        surface = mesh_from_mask(small, small_affine, verbose=verbose, **options)
        if surface is None or not surface.n_points:
            skipped.append(name)      # too small or too thin to close a surface
            continue
        if name in colours:
            # Carried on the mesh so the atlas's own colour survives into the
            # figure; renderers read it back from field_data["atlas_rgb"].
            surface.field_data["atlas_rgb"] = np.asarray(colours[name], float)
        meshes[name] = surface

    if not meshes:
        raise ValueError(
            f"nothing above {iso} in {Path(path).name} formed a surface"
            + (f"; {len(skipped)} piece(s) were under {min_voxels} voxels"
               if skipped else "")
        )
    return meshes, {"split": split, "iso": iso, "n_meshes": len(meshes),
                    "skipped": skipped, "over_cap": over_cap}


def _natural(path):
    """Sort key that reads the numbers in a file name as numbers: parcel_2 before parcel_10."""
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


# Lead-DBS lays an atlas out as side folders of region NIfTIs, and the folder
# says the side. The suffix is the plotter's own: "_L" / "_R" is what its
# hemisphere test reads, so lateral views and the Hemispheres control work.
SIDE_FOLDERS = {"lh": "_L", "left": "_L", "rh": "_R", "right": "_R", "midline": "", "mixed": ""}


def _nifti_files(folder):
    """The NIfTIs in ``folder``, one per name, in natural file-name order: an
    ``fx.nii`` beside its own ``fx.nii.gz`` is one region, not "fx" and "fx_"."""
    found = {}
    for path in sorted((p for p in folder.iterdir() if not p.name.startswith(".") and p.name.lower().endswith((".nii", ".nii.gz"))), key=_natural):
        found.setdefault(path.name.split(".nii")[0], path)
    return list(found.values())


def _file_name(name):
    """A region name safe as a file name and as one word of ``atlas_LUT.txt``:
    spaces and path separators become hyphens (not underscores, which the
    plotter would read as an _L / _R side marker)."""
    import re

    return re.sub(r"[\s/\\:]+", "-", str(name)).strip("-")


def build_atlas(source, out_dir, table=None, progress=None):
    """Mesh an atlas once into ``out_dir``: one ``.vtk`` per region plus an
    ``atlas_LUT.txt`` in region order -- the layout ``gnv.render.mesh_paint.load_parcels``
    reads, so the result is a region set like the shipped SUIT and AAL ones.
    (Listing it in a mesh index is the caller's business: see
    gnv.resource_index.)

    ``source`` is a 4D NIfTI (one region per volume), a 3D label NIfTI (one
    region per integer value), or a folder of NIfTIs (one region per file, in
    natural file-name order). A folder laid out the Lead-DBS way -- ``lh/``,
    ``rh/``, ``midline/``, ``mixed/`` -- takes its regions from those folders,
    suffixed by side (``SIDE_FOLDERS``: STN in lh/ is "STN_L"), and files
    beside them (a ``gm_mask.nii.gz``) are not regions. Every region is
    surfaced exactly as a NIfTI dropped on a mesh layer is:
    ``meshes_from_volume`` with its defaults.

    ``table`` is a label table (.txt / .lut / .csv / .tsv / .json, see
    ``read_label_table``). For a 4D file or a folder, row i names region i --
    in a side folder, the i-th file of that folder, before its side suffix;
    for a label image, rows name values by their index. Without one, a 4D or
    label file uses its own sidecar table when it has one (``label_table``),
    and a folder names each region after its file.

    ``progress(fraction)`` is called as regions finish. Returns
    ``{"regions": n, "skipped": [names too small to mesh]}``.
    """
    import os
    import shutil

    from gnv.convert.parcel_meshes import sanitize_region_name

    source, out_dir = Path(source).expanduser(), Path(out_dir).expanduser()
    rows = read_label_table(Path(table).expanduser()) if table else []
    report = progress or (lambda fraction: None)
    meshes, skipped = {}, []
    if source.is_dir():
        sides = [(d, SIDE_FOLDERS[d.name.lower()]) for d in sorted(source.iterdir()) if d.is_dir() and d.name.lower() in SIDE_FOLDERS]
        groups = [(_nifti_files(d), suffix) for d, suffix in sides] if sides else [(_nifti_files(source), "")]
        # (file, its position in its folder, side suffix)
        work = [(path, i, suffix) for files, suffix in groups for i, path in enumerate(files)]
        if not work:
            raise ValueError(f"{source} has no NIfTI files in it" + (" or in its lh/ rh/ folders" if sides else ""))
        for done, (path, i, suffix) in enumerate(work, start=1):
            stem = path.name.split(".nii")[0]
            try:
                part, info = meshes_from_volume(path)
            except ValueError:
                skipped.append(stem + suffix)          # nothing in it formed a surface
                continue
            skipped += [name + suffix for name in info["skipped"]]
            if len(part) == 1:
                # One region from a file is named by the table, else by the file.
                part = {sanitize_region_name(rows[i][1] if i < len(rows) else stem): next(iter(part.values()))}
            for name, mesh in part.items():
                # The side goes on after sanitising, which would turn "_L" into "-L".
                name += suffix
                while name in meshes:
                    name += "_"
                meshes[name] = mesh
            report(done / len(work))
    else:
        names = None
        if rows and n_frames(source) > 1:
            stem = source.name.split(".nii")[0]
            names = [name for _index, name, _rgb in rows][:n_frames(source)]
            names += [f"{stem}_{i:03d}" for i in range(len(names), n_frames(source))]
        elif rows:
            names = {index: name for index, name, _rgb in rows if index is not None}
        meshes, info = meshes_from_volume(source, names=names)
        skipped = info["skipped"] + info["over_cap"]
        report(1.0)
    if not meshes:
        raise ValueError(f"nothing in {source.name} formed a surface")
    # A table's colours ride on the meshes, as a sidecar's do (field_data["atlas_rgb"]).
    colours = {sanitize_region_name(name): rgb for _index, name, rgb in rows if rgb is not None}
    for name, mesh in meshes.items():
        rgb = colours.get(name, colours.get(name[:-2] if name.endswith(("_L", "_R")) else None))
        if rgb is not None:
            mesh.field_data["atlas_rgb"] = np.asarray(rgb, float)

    # Written beside the target and renamed into place, so a failed build never
    # leaves a half-made atlas in the list.
    partial = out_dir.with_name(f".{out_dir.name}.{os.getpid()}.partial")
    partial.mkdir(parents=True)
    try:
        with open(partial / "atlas_LUT.txt", "w") as fh:
            taken = set()
            for index, (name, mesh) in enumerate(meshes.items(), start=1):
                name = _file_name(name)
                # macOS and Windows file names ignore case: HCP-MMP's cortical
                # "PF" and HCPex's thalamic "Pf" would be one file, the later
                # silently replacing the earlier.
                while name.lower() in taken:
                    name += "_"
                taken.add(name.lower())
                mesh.save(partial / f"{name}.vtk")
                fh.write(f"{index} {name}\n")
        partial.rename(out_dir)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return {"regions": len(meshes), "skipped": skipped}

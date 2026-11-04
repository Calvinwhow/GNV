"""Render Lead-DBS / discriminative-fiber results inside the composite glass brain.

This is the standalone renderer that produced the SCA fibre figures.  It does
not go through yabplot: it draws the streamlines itself with PyVista so that
line width, tube shading, hemisphere culling and camera placement are all under
direct control, and it is one layer type in
:class:`gnv.render.brain_viewer.BrainScene`, so streamlines can
be drawn inside the same backdrop -- and alongside the same painted meshes -- as
everything else.

Typical use::

    from gnv.render.fiber_render import load_fibers, panel, hero

    fibers = load_fibers(
        ".../contrast_pval_FWE_0_positive_ftr.mat",
        max_abs_value=0.05,        # p-value map: keep the significant fibres
        max_lines=30000,
    )
    panel(fibers, "fibers_panel.png")
    hero(fibers, "fibers_left.png", view="left")

or, mixed with anything else::

    BrainScene().surface("pial_wholebrain", glass=True).fibers(fibers).render(out)

Value filtering
---------------
``sign`` / ``min_abs_value`` / ``top_percent`` mirror
``FiberFormatConverter._fiber_value_mask``.  ``max_abs_value`` is the extra one
this module adds, because a ``contrast_pval_*`` MAT stores p-values, where the
fibres you want are the *small* ones -- ``max_abs_value=0.05`` is the p<0.05 set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from calvin_utils.neuroimaging_utils.tract_utils.fiber_descriptor import is_atlas
from gnv.render.render_scene import GLASS_ALPHA, RED

__all__ = ["load_fibers", "smooth_streamlines", "tube_fibers", "panel", "hero",
           "FIBER_COLOR", "LINE_WIDTH", "TUBE_RADIUS", "TUBE_SIDES",
           "FIBER_MATERIAL", "fiber_material"]

FIBER_COLOR = RED
LINE_WIDTH = 1.5

# Real geometry, not a line primitive. `render_lines_as_tubes` shades a flat
# line as though it were round, which is cheap and looks it: the silhouette
# stays a constant-width ribbon and there is no surface for a highlight to sit
# on, so the bundle reads as hair rather than as silk. A tube filter builds an
# actual swept surface with real normals, which is what catches the light.
TUBE_RADIUS = 0.2        # mm; a streamline is a line, so this is a drawing choice
TUBE_SIDES = 8

# Thickness is a physical size in millimetres, because that is what a figure
# caption can state and what stays true when the camera moves. Hi-res sweeps it
# as real geometry; preview has only line primitives, whose width is in pixels,
# so it approximates. At the default framing a ~180 mm brain spans the ~640 px
# viewport, making a pixel ~0.28 mm and a radius of R mm about 2R/0.28 px wide.
MM_PER_LINE_WIDTH = 0.14
TUBE_RADIUS_RANGE = (0.05, 0.5)   # mm; the ceiling is what keeps tubes from
                                  # going massive when the slider is dragged


def line_width_for_radius(radius):
    """Preview line width in pixels that approximates a tube of ``radius`` mm."""
    return max(float(radius) / MM_PER_LINE_WIDTH, 0.5)


def radius_for_width(width):
    """Legacy: the mm radius a saved spec's pixel ``width`` used to imply."""
    return max(float(width) * MM_PER_LINE_WIDTH, 0.01)

# Tight, bright highlight against a dim ambient -- the anisotropic sheen you get
# off a bundle of parallel filaments.
FIBER_MATERIAL = dict(specular=0.85, specular_power=60, ambient=0.18, diffuse=0.62)


def fiber_material(panel=None):
    """``FIBER_MATERIAL`` MODULATED by the lighting panel, not replaced by it.

    A fibre is a thin glossy cylinder and a cortical surface is not: fibres want
    a hot, tight specular (0.85 at power 60) where a surface wants a soft one
    (0.22 at power 14). Handing fibres the panel's numbers directly dropped them
    to the surface material, and a bundle that no longer carries a highlight
    reads as having stopped interacting with the light altogether.

    So the panel is read as a RATIO against its own defaults. At the default
    settings a fibre renders exactly as it always did; turn the specular slider
    up and the fibre's own gloss rises in proportion. Every fibre is shaded the
    same way whether it is painted by an overlay, carries the atlas colour, or
    is the backdrop under a finding -- the material never depends on how the
    colour was arrived at.
    """
    from gnv.render.render_scene import SURFACE_MATERIAL

    out = dict(FIBER_MATERIAL)
    for key, value in (panel or {}).items():
        if value is None:
            continue
        if key not in out:
            out[key] = value
            continue
        reference = float(SURFACE_MATERIAL.get(key) or 0.0)
        out[key] = (float(out[key]) * float(value) / reference if reference
                    else float(value))
    for key in ("ambient", "diffuse", "specular"):
        out[key] = min(max(float(out[key]), 0.0), 1.0)
    out["specular_power"] = min(max(float(out["specular_power"]), 1.0), 128.0)
    return out


# Loading ------------------------------------------------------------------
# Fibre geometry: the converter's own .npz / .npy, and standard tractograms
# (MRtrix .tck, TrackVis .trk, .trx), which it reads with nibabel. A Lead-DBS
def smooth_streamlines(poly, passes=3):
    """Round off the polyline kinks, keeping every endpoint where it was.

    Streamlines arrive as ~0.5 mm polylines and are subsampled further, so at
    figure scale the corners read as faceting. A binomial pass over the interior
    points removes them without letting a bundle drift off its own geometry:
    the ends are pinned, so a tract still starts and stops where the tractogram
    said it did.
    """
    if not passes:
        return poly

    points = np.array(poly.points, dtype=float)   # a copy: never edit the caller's mesh
    lines = poly.lines
    i = 0
    while i < len(lines):
        n = lines[i]
        ids = lines[i + 1:i + 1 + n]
        if n >= 3:
            seg = points[ids]
            for _ in range(int(passes)):
                seg[1:-1] = (seg[:-2] + 2.0 * seg[1:-1] + seg[2:]) / 4.0
            points[ids] = seg
        i += 1 + n
    out = poly.copy()
    out.points = points
    return out


def tube_fibers(poly, radius=TUBE_RADIUS, sides=TUBE_SIDES):
    """Sweep the streamlines into real tubes, carrying their scalars along."""
    return poly.tube(radius=float(radius), n_sides=int(sides))


def load_fibers(
    path,
    fiber_atlas_path=None,
    sign="both",
    min_abs_value=None,
    max_abs_value=None,
    top_percent=None,
    max_lines=None,
    step=2,
    smooth=3,
    seed=0,
):
    """Read a ``.mat`` / ``.fib.npy`` fibre result into a line ``pv.PolyData``.

    ``step`` keeps every Nth vertex of each streamline (Lead-DBS geometry is
    ~0.5 mm-spaced, so ``step=2`` is ~1 mm and invisible at figure scale while
    halving the point count).  ``max_lines`` randomly subsamples streamlines;
    past ~30k the render stops looking denser and only gets slower.

    The returned mesh carries ``"vals"`` (per point, the streamline's value) and
    ``"dir"`` (per point uint8 RGB, the |direction| DTI convention).
    """
    from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import (
        FiberFormatConverter,
    )

    path = Path(path).expanduser()
    if is_atlas(path):
        # A bare atlas is geometry and nothing else. Its fibres get no values,
        # because the values live in separate .fib.values.npy files that are
        # layered on afterwards -- one bundle of streamlines, many findings
        # painted onto it, rather than a new copy of the geometry per finding.
        from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import (
            FiberFormatConverter,
        )

        streamlines = [FiberFormatConverter._as_streamline(f)
                       for f in FiberFormatConverter._load_fibers(path)]
        values = np.zeros(len(streamlines), dtype=np.float32)
    elif path.name.lower().endswith(".fib.npy"):
        streamlines, values, _ = FiberFormatConverter.load_fib_npy(
            path, fiber_atlas_path=fiber_atlas_path, sign=sign,
            min_abs_value=min_abs_value, top_percent=top_percent,
        )
    else:
        streamlines, _, values = FiberFormatConverter.load_leaddbs_mat(path)
        if values is None:
            values = np.zeros(len(streamlines), dtype=np.float32)
        keep = FiberFormatConverter._fiber_value_mask(
            values, sign, min_abs_value, top_percent
        )
        streamlines = [s for s, k in zip(streamlines, keep) if k]
        values = np.asarray(values)[keep]

    return fibres_to_lines(streamlines, values, max_abs_value=max_abs_value, max_lines=max_lines, step=step, smooth=smooth, seed=seed)


def fibres_to_lines(streamlines, values, ids=None, total=None, max_abs_value=None, max_lines=None, step=2, smooth=3, seed=0):
    """Streamlines and one value each -> the line ``pv.PolyData`` ``load_fibers`` returns.

    ``ids`` numbers each streamline in its atlas (default: its position here) and
    becomes ``"fiber"``; ``total`` is how many fibres that atlas holds (default:
    how many are given) and becomes ``atlas_fibers``. The rest is as ``load_fibers``.
    """
    values = np.asarray(values, dtype=np.float32)
    if max_abs_value is not None:
        keep = np.abs(values) <= float(max_abs_value)
        if not keep.any():
            raise ValueError(f"no fibers survive max_abs_value={max_abs_value}")
        streamlines = [s for s, k in zip(streamlines, keep) if k]
        values = values[keep]
        ids = None if ids is None else np.asarray(ids)[keep]

    total = len(streamlines) if total is None else int(total)
    order = np.arange(len(streamlines))
    if max_lines is not None and len(order) > max_lines:
        order = np.sort(np.random.default_rng(seed).choice(len(order), max_lines, False))

    pts, cells, vals, owner, cursor = [], [], [], [], 0
    for i in order:
        f = np.asarray(streamlines[i], dtype=np.float32)
        d = f[::max(1, int(step))]
        if len(d) < 2:
            continue
        if not np.allclose(d[-1], f[-1]):
            d = np.vstack([d, f[-1]])
        n = len(d)
        pts.append(d)
        cells.append(np.r_[n, np.arange(cursor, cursor + n)])
        vals.append(np.full(n, values[i], dtype=np.float32))
        # Which streamline this run of points came from, in the ATLAS's own
        # numbering. A values file has one number per atlas fibre, so without
        # this there is nothing to join it to once subsampling and thresholding
        # have dropped some and reordered the rest.
        owner.append(np.full(n, int(i) if ids is None else int(ids[i]), dtype=np.int64))
        cursor += n

    poly = pv.PolyData()
    poly.points = np.concatenate(pts)
    poly.lines = np.concatenate(cells)
    poly["vals"] = np.concatenate(vals)
    poly["fiber"] = np.concatenate(owner)
    # How many fibres the ATLAS holds, which is not how many are drawn:
    # max_lines subsamples and the thresholds drop more. A values file is one
    # number per atlas fibre, so this is the only honest thing to check its
    # length against -- the drawn count would call a correct file mismatched
    # the moment any subsampling happened.
    poly.field_data["atlas_fibers"] = np.array([total], dtype=np.int64)
    if smooth:
        poly = smooth_streamlines(poly, smooth)
    # Directions come from the SMOOTHED geometry, so the orientation colouring
    # matches the tube the eye actually sees rather than the polyline under it.
    poly["dir"] = _direction_rgb(poly)
    return poly


def _direction_rgb(poly):
    pts, lines = poly.points, poly.lines
    rgb = np.zeros_like(pts)
    i = 0
    while i < len(lines):
        n = lines[i]
        ids = lines[i + 1:i + 1 + n]
        seg = np.gradient(pts[ids], axis=0)
        rgb[ids] = np.abs(seg) / np.clip(np.linalg.norm(seg, axis=1, keepdims=True), 1e-6, None)
        i += 1 + n
    return (rgb * 255).astype(np.uint8)



# Convenience wrappers -----------------------------------------------------
# These exist because "draw these fibres in a glass brain" is by far the most
# common thing to want; anything more layered goes through BrainScene directly.
def _scene(poly, coloring, color, width, mesh, glass_alpha):
    from gnv.render.brain_viewer import BrainScene

    return (BrainScene()
            .surface(mesh, glass=True, opacity=glass_alpha)
            .fibers(poly, coloring=coloring, color=color, width=width))


def panel(poly, out, views=("left", "right", "anterior", "superior"),
          coloring="solid", color=FIBER_COLOR, width=LINE_WIDTH,
          mesh="pial_wholebrain", glass_alpha=GLASS_ALPHA, **render_kw):
    return _scene(poly, coloring, color, width, mesh, glass_alpha).render(
        out, views=views, **render_kw)


def hero(poly, out, view="left", coloring="solid", color=FIBER_COLOR,
         width=1.6, mesh="pial_wholebrain", glass_alpha=GLASS_ALPHA, zoom=1.12,
         **render_kw):
    return _scene(poly, coloring, color, width, mesh, glass_alpha).render(
        out, views=view, zoom=zoom, **render_kw)


# -- pruning ----------------------------------------------------------------
def crosses_plane(points, axis, at):
    """Whether a streamline passes through a plane, not merely near it.

    Ported from FiberSelector._toggle_by_plane: a fibre counts when its points
    fall on BOTH sides, which is what "goes through here" means. Testing
    proximity instead would catch fibres that run alongside a plane and stop
    short of it, which is the opposite of a waypoint.
    """
    d = np.asarray(points, float)[:, axis] - float(at)
    return bool((d > 0).any() and (d < 0).any())


def prune_streamlines(poly, gates):
    """Keep only the streamlines that satisfy every enabled gate.

    Gates are ANDed, the way waypoint ROIs work in a tractography viewer:
    "through the internal capsule AND through the brainstem, but not crossing
    the midline" is three gates rather than three separate selections. Each is a
    plane in world millimetres, addressed the same way a mesh section is, so a
    gate can be typed from a coordinate someone read off a slice.

    A gate is either a PLANE -- ``{axis, at}`` -- or a VOLUME:
    ``{kind: "volume", nifti, threshold}``, which is an ROI read straight off a
    NIfTI. A streamline is inside the volume where ``|value| > threshold``, so a
    binary mask at threshold 0 means "any non-zero voxel" and a t-map at 2 means
    either tail. Both kinds live in the same list and AND together, so "through
    this cluster, avoiding the midline" is one volume gate and one plane gate.

    ``mode`` is "through" to require the hit or "avoid" to forbid it.
    """
    import pyvista as pv

    gates = [g for g in (gates or []) if g.get("enabled", True)]
    if not gates or poly.n_points == 0:
        return poly

    points = np.asarray(poly.points, float)

    # Each volume gate is resolved ONCE over every point in the layer -- a
    # single vectorised sample, not one per streamline -- and the per-streamline
    # test below is then just an index into a boolean array. Nearest neighbour,
    # because an ROI is categorical: interpolating a mask invents values on its
    # boundary that are in neither the inside nor the outside.
    inside = {}
    for gi, gate in enumerate(gates):
        if gate.get("kind") == "volume" and gate.get("nifti"):
            from gnv.render.mesh_paint import sample_nifti

            sampled = np.asarray(
                sample_nifti(points, gate["nifti"], order=0), float)
            cut = float(gate.get("threshold") or 0.0)
            inside[gi] = np.abs(sampled) > cut

    lines = poly.lines
    kept, mapped, order = [], {}, []
    cursor = i = 0
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1:i + 1 + n]
        run = points[ids]
        ok = True
        for gi, gate in enumerate(gates):
            if gi in inside:
                hit = bool(inside[gi][ids].any())
            else:
                axis = "xyz".index(gate["axis"])
                hit = crosses_plane(run, axis, gate["at"])
            if gate.get("mode", "through") == "through":
                ok = ok and hit
            else:
                ok = ok and not hit
            if not ok:
                break
        if ok:
            for pid in ids:
                if pid not in mapped:
                    mapped[pid] = cursor
                    order.append(pid)
                    cursor += 1
            kept.append(np.r_[n, [mapped[pid] for pid in ids]])
        i += 1 + n

    if not kept:
        return None
    index = np.asarray(order, np.int64)
    out = pv.PolyData()
    out.points = points[index]
    out.lines = np.concatenate(kept)
    for name in ("vals", "fiber", "dir"):
        if name in poly.point_data:
            out[name] = np.asarray(poly[name])[index]
    for name in poly.field_data:
        out.field_data[name] = poly.field_data[name]
    return out

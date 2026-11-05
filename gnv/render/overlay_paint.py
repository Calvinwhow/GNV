"""Compositing a stack of volumes onto one piece of geometry.

A finding is rarely one map.  "The motor network in red and the cognitive one in
blue, over anatomy" is three volumes on the same surface, and the honest way to
draw that is a stack: each volume is coloured independently and painted where it
survives its own threshold, later entries over earlier ones.

Every overlay is resolved here, on its own values alone, and composited over the
base coat in order.  ONE overlay takes exactly the same path as five.  It used to
short-circuit to a scalar array plus a colormap -- cheaper, and it bought a
colour bar for free -- but that made a stack's behaviour depend on its length:
deleting the second overlay restyled the first, because it landed on different
machinery with a different scale and a different meaning for opacity.  Overlays
have to be independent of each other to be trustworthy, so the bar is built
separately from ``scale_for`` instead of being bought with that divergence.

Transparency is what makes it readable.  Every overlay contributes alpha only
where it clears threshold, so the layers below show through everywhere else
instead of being hidden under a full-coverage wash of the top map's colormap.
"""

from __future__ import annotations

import numpy as np

__all__ = ["overlay_colours", "composite", "sample_for_overlay",
           "scale_for"]


def sample_for_overlay(points, overlay, order=1):
    """Values of an overlay's volume at world points, with its own selection."""
    from gnv.render.mesh_paint import _select, sample_nifti

    raw = sample_nifti(points, overlay["nifti"], order=order)
    return _select(raw, sign=overlay.get("sign", "both"),
                   absolute=bool(overlay.get("absolute", False)),
                   threshold=overlay.get("threshold"),
                   max_value=overlay.get("max_value"))


def scale_for(values, overlay):
    """The ``(cmap, clim)`` one overlay resolves to, from ITS OWN values only.

    Split out so the colour bar and the pixels cannot disagree, and so that
    nothing an overlay does can reach another one's scale.
    """
    from gnv.render.render_scene import resolve_scale

    return resolve_scale(values, cmap=overlay.get("palette"),
                         clim=overlay.get("clim"),
                         color=overlay.get("color", "#c15656"),
                         symmetric=overlay.get("symmetric"))


def overlay_colours(values, overlay):
    """``(rgb, alpha)`` for one overlay's values. NaN means "not here".

    ``alpha`` is the overlay's opacity where the value survived and zero where
    it did not, which is the whole mechanism: an overlay is a stencil as much as
    a colour.
    """
    values = np.asarray(values, float)
    present = np.isfinite(values)
    rgb = np.zeros(values.shape + (3,), dtype=float)
    alpha = np.zeros(values.shape, dtype=float)
    if not present.any():
        return rgb, alpha

    cmap, clim = scale_for(values[present], overlay)
    span = (clim[1] - clim[0]) or 1.0
    scaled = np.clip((values[present] - clim[0]) / span, 0.0, 1.0)
    sampled = cmap(scaled)
    rgb[present] = sampled[..., :3]

    # Every value the overlay keeps is painted at its opacity -- the bottom of
    # the scale (and anything below it) in the bottom colour, as MRIcroGL draws
    # a LUT -- shaped only by an alpha the palette was built to have: a
    # mosaic's soft threshold fades its low end in that way. MRIcroGL's .clut
    # files carry an alpha ramp of their own, which is not used. (Every other
    # palette is opaque.)
    from gnv.render.palettes import is_clut

    shape = 1.0 if is_clut(overlay.get("palette")) else sampled[..., 3]
    alpha[present] = np.clip(shape, 0.0, 1.0) * float(overlay.get("opacity", 1.0))

    if overlay.get("hide_below_min"):
        # Opt in only. The scale's low end normally just means "bottom colour";
        # this makes it mean "absent" outright. A diverging scale is left
        # alone: its low end is the negative extreme, not an absence, and
        # dropping what is past it would erase the strongest negatives.
        if not (clim[0] < 0.0 < clim[1]):
            # Binary, not a fade. A ramp leaves a trace of this overlay on every
            # mesh whose value sits just under the foot of the scale, so with a
            # low minimum the whole brain picks up a tint -- the opposite of the
            # point. Below the minimum this overlay paints NOTHING, and the mesh
            # shows whatever is under it: the next overlay down, or the layer's
            # own colour if none of them reached it.
            below = np.zeros(values.shape, bool)
            below[present] = values[present] < clim[0]
            alpha[below] = 0.0

    # Otherwise no gating, deliberately. A colormap maps min..max to colours and does
    # nothing else; which values exist at all is processing -- the value window
    # and the absolute toggle, applied upstream in _select. Keeping the two
    # apart is what makes the scale safe to move: renarrowing it restyles the
    # figure without silently changing which voxels are in it.
    return rgb, alpha


def composite(base_rgb, stack):
    """Paint ``(rgb, alpha)`` pairs over a base, in order. Returns float RGB.

    Straight source-over: the last overlay to cover a point wins there, and
    anything it does not cover still shows what is underneath.
    """
    out = np.array(base_rgb, dtype=float, copy=True)
    for rgb, alpha in stack:
        a = np.asarray(alpha, float)[..., None]
        out = out * (1.0 - a) + np.asarray(rgb, float) * a
    return out

"""One slice of a backdrop (``gnv.convert.backdrops.Backdrop``) as a
textured quad in the scene, with any overlays painted onto it."""

from __future__ import annotations

import numpy as np

from gnv.convert.backdrops import AXES


def _colourise(grey, cmap=None):
    """uint8 slice -> RGB. Anatomy wants grey; a cmap is for overlays."""
    if not cmap:
        return np.repeat(grey[:, :, None], 3, axis=2)
    from matplotlib import colormaps

    table = colormaps[cmap](np.linspace(0, 1, 256))[:, :3]
    return (table[grey] * 255).astype(np.uint8)


def backdrop_plane(backdrop, axis, world_mm, cmap=None, transparent_below=None, overlays=None, fill=None):
    """A textured quad for one slice: ``(pv.PolyData, pv.Texture, world_mm)``.

    The quad is built from the volume's own corner coordinates rather than
    from a plane normal, so it lands correctly whatever the affine does, and
    texture coordinates are attached explicitly -- letting VTK infer them is
    how a slice ends up mirrored, which on a brain is both easy to miss and
    serious.
    """
    import pyvista as pv

    index = backdrop.index_for(axis, world_mm)
    image = backdrop.slice(axis, index)
    others = [a for a in AXES if a != axis]

    points = []
    for u, v in ((0, 0), (1, 0), (0, 1), (1, 1)):
        ijk = [0.0, 0.0, 0.0]
        ijk[axis] = index
        ijk[others[0]] = (backdrop.shape[others[0]] - 1) * u
        ijk[others[1]] = (backdrop.shape[others[1]] - 1) * v
        points.append((backdrop.affine @ np.array([*ijk, 1.0]))[:3])

    quad = pv.PolyData(np.asarray(points, float),
                       faces=np.array([4, 0, 1, 3, 2]))
    quad.active_texture_coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    # VTK samples textures with t=0 at the BOTTOM row, while the array's
    # row 0 is the low index of `others[0]`; transposing and flipping puts
    # the two in agreement.
    picture = np.ascontiguousarray(np.asarray(image).T[::-1])
    rgb = _colourise(picture, cmap)
    if overlays:
        from gnv.render.overlay_paint import (
            composite, overlay_colours, sample_for_overlay,
        )

        grid = backdrop.plane_grid(axis, index)
        flat = grid.reshape(-1, 3)
        stack = []
        for overlay in overlays:
            values = sample_for_overlay(flat, overlay)
            over_rgb, over_a = overlay_colours(values, overlay)
            # Back into the picture's orientation, the same transpose and
            # flip the base went through.
            shape = grid.shape[:2]
            stack.append((
                np.ascontiguousarray(
                    over_rgb.reshape(*shape, 3).transpose(1, 0, 2)[::-1]),
                np.ascontiguousarray(over_a.reshape(shape).T[::-1]),
            ))
        rgb = np.clip(composite(rgb / 255.0, stack) * 255.0, 0, 255).astype(np.uint8)
    if transparent_below is not None:
        # Background that stays opaque is a black slab in front of the
        # mesh. Dropping it out is what lets a slice sit INSIDE a brain
        # rather than in front of one.
        keep = picture > int(transparent_below)
        if fill is None:
            alpha = np.where(keep, 255, 0).astype(np.uint8)
        else:
            # Paint the dropped-out part in the scene's own background
            # instead of clearing it. Nothing about the data changes; the
            # slice just stops showing whatever is behind it, so it reads as
            # a solid plate rather than a translucent film. Still an alpha
            # channel, all-opaque, because the texture format cannot change
            # between renders without rebuilding the actor.
            rgb = np.where(keep[..., None], rgb, np.asarray(fill, np.uint8))
            alpha = np.full(picture.shape, 255, np.uint8)
        rgb = np.dstack([rgb, alpha])
    return quad, pv.Texture(np.ascontiguousarray(rgb)), backdrop.world_for(axis, index)

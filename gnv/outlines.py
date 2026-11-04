"""Outlines of a colouring NIfTI: the edge of what it selects, as lines.

An overlay with ``outline`` on is traced wherever it selects anything -- the
same sign / absolute / threshold / max a fill would use, so on an ordinary map
that is its nonzero voxels. The selection is a 0/1 volume, blurred by
``outline_smooth`` voxels so the edge reads as a boundary rather than the voxel
grid (``mesh_paint._outline_mask``), and the line is its 0.5 contour:

* on a MESH, around exactly what the overlay colours there: the fill's own
  per-vertex (or per-region) values, from the colour bar's minimum up on a
  sequential scale, marked 0/1 on each piece and traced like a volume's
  (below) -- so it sits where the colour starts and the mesh hides it where
  it hides the colour;
* on a SLICE, in the plane (``plane_outline``), moved with the plane;
* on a VOLUME body, traced over the finished frame. Something is drawn alone
  off screen with the frame's camera (``Projector``) to give a mask of where
  the region is on screen, and a line centred on the mask's edges is laid
  over the image (``draw_outline``), so width and opacity are exact pixels:
  - a PAINTED body: the body again, unlit, white where the region is
    (``volume_render.region_actor``) -- the mask is exactly the patches the
    fill colours, on the cortex and on cut faces, hidden where the brain
    hides them;
  - an X-RAYED body: the region's 0.5 surface (``region_surface``), cut like
    the body -- the region as seen through the brain.

Colour, opacity and width (pixels) are the overlay's own ``outline_color``,
``outline_opacity`` and ``outline_width``. Nothing here is written to disk.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

__all__ = ["plane_outline", "region_surface", "clip_to_section",
           "Projector", "draw_outline", "outline_mask"]

_surfaces = OrderedDict()       # key -> region surface (PolyData)


_masks = OrderedDict()          # key -> (mask, affine)


def colour_floor(overlay):
    """Where an overlay's colour starts: the low end of its colour bar, on a
    sequential scale; None on a diverging one, whose low end is the strongest
    negative rather than a limit. The scale is the map's own (pinned from all
    its selected voxels), the same one slices and bodies paint with."""
    from gnv import volume_render as vr

    native = vr.native_values(overlay)
    if not native.size:
        return None
    clim = vr.pinned(overlay, native).get("clim")
    if not clim:
        return None
    low, high = min(clim), max(clim)
    return None if low < 0.0 < high else float(low)


def outline_mask(overlay):
    """``(mask, affine)``: where the overlay puts colour, 0..1, lightly blurred.

    What survives its sign / threshold / max (``_select``, exactly as a fill)
    and, on a sequential scale, only from the colour bar's minimum up -- so an
    outline sits at the bar's lower limit. Blurred by ``outline_smooth``
    voxels so a traced edge reads as a boundary rather than as the grid.
    """
    from calvin_utils.plotting_utils.mesh_paint import _select, _volume
    from gnv import volume_render as vr

    floor = colour_floor(overlay)
    smooth = float(overlay.get("outline_smooth", 1.0) or 0.0)
    path = str(overlay["nifti"])
    key = (path, vr._stamp(path), overlay.get("sign"), bool(overlay.get("absolute")),
           overlay.get("threshold"), overlay.get("max_value"), smooth, floor)
    if key in _masks:
        _masks.move_to_end(key)
        return _masks[key]
    data, affine = _volume(overlay["nifti"])
    chosen = _select(np.asarray(data).ravel(), sign=overlay.get("sign", "both"),
                     absolute=bool(overlay.get("absolute")),
                     threshold=overlay.get("threshold"),
                     max_value=overlay.get("max_value"))
    inside = np.isfinite(chosen)
    if floor is not None:
        inside[inside] = chosen[inside] >= floor
    mask = inside.reshape(np.asarray(data).shape).astype(np.float32)
    if smooth:
        from scipy.ndimage import gaussian_filter

        mask = gaussian_filter(mask, smooth)
    _masks[key] = (mask, affine)
    while len(_masks) > 8:
        _masks.popitem(last=False)
    return _masks[key]


def _sample(mask, affine, world):
    """The mask at world points (N, 3), trilinear, 0 outside."""
    from scipy.ndimage import map_coordinates

    inverse = np.linalg.inv(np.asarray(affine, float))
    vox = inverse[:3, :3] @ np.asarray(world, float).T + inverse[:3, 3:4]
    return map_coordinates(mask, vox, order=1, mode="constant", cval=0.0)


def plane_outline(back, axis, index, overlay, step_mm=0.5, lift_mm=0.12):
    """Lines on slice ``index`` of ``back`` along ``axis``, world mm.

    The plane is sampled at about ``step_mm`` (the mask is smooth at the map's
    own resolution, so a 0.1 mm backdrop gains nothing from every texel) and
    contoured at 0.5. Each line is drawn a hair either side of the plane, so it
    sits on the slice from both directions instead of flickering into it.
    """
    import pyvista as pv

    mask, affine = outline_mask(overlay)
    grid = back.plane_grid(axis, index)                 # (nu, nv, 3) world mm
    stride = max(1, int(round(step_mm / max(float(back.mm), 1e-6))))
    grid = grid[::stride, ::stride]
    nu, nv = grid.shape[:2]
    if nu < 2 or nv < 2:
        return pv.PolyData()
    values = _sample(mask, affine, grid.reshape(-1, 3))
    if not (values.max() > 0.5 > values.min()):
        return pv.PolyData()
    # A structured grid's points run with the first index fastest.
    plane = pv.StructuredGrid()
    plane.points = grid.transpose(1, 0, 2).reshape(-1, 3)
    plane.dimensions = (nu, nv, 1)
    plane.point_data["inside"] = values.reshape(nu, nv).T.ravel()
    lines = plane.contour([0.5], scalars="inside")
    if not lines.n_points:
        return pv.PolyData()
    normal = np.cross(grid[1, 0] - grid[0, 0], grid[0, 1] - grid[0, 0])
    normal = normal / (np.linalg.norm(normal) or 1.0)
    front = lines.copy()
    back_side = lines.copy()
    front.points = np.asarray(lines.points) + lift_mm * normal
    back_side.points = np.asarray(lines.points) - lift_mm * normal
    return front.merge(back_side)


def region_surface(overlay, body_grid=None, outside=False, body_inside=None):
    """The overlay's region as a closed surface (its 0.5 contour), world mm.

    Built on the map's own grid -- oblique or not, through ``_grid_from`` --
    and cut to the body (``body_inside``: the body's own inside mask, on
    ``body_grid``) unless ``outside``. Cached on everything it depends on.
    """
    from gnv import volume_render as vr

    mask, affine = outline_mask(overlay)
    key = (str(overlay["nifti"]), vr._stamp(str(overlay["nifti"])),
           overlay.get("sign"), bool(overlay.get("absolute")),
           overlay.get("threshold"), overlay.get("max_value"),
           float(overlay.get("outline_smooth", 1.0) or 0.0), colour_floor(overlay),
           bool(outside),
           None if (outside or body_grid is None) else vr._grid_key(body_grid))
    if key in _surfaces:
        _surfaces.move_to_end(key)
        return _surfaces[key]
    data = np.asarray(mask, np.float32)
    if not outside and body_grid is not None and body_inside is not None:
        # The body's inside, looked up at each map voxel (nearest): a region
        # that runs past the brain stops at its surface, like the fill does.
        shape = data.shape
        ijk = np.indices(shape).reshape(3, -1).T
        world = (np.asarray(affine, float)[:3, :3] @ ijk.T).T + np.asarray(affine, float)[:3, 3]
        rot = vr._direction(body_grid) @ np.diag(np.asarray(body_grid.spacing, float))
        idx = np.linalg.solve(rot, (world - np.asarray(body_grid.origin, float)).T).T
        idx = np.rint(idx).astype(np.int64)
        dims = np.asarray(body_grid.dimensions)
        ok = np.all((idx >= 0) & (idx < dims), axis=1)
        inside = np.zeros(len(idx), bool)
        solid = np.asarray(body_inside).reshape(tuple(dims), order="F")
        inside[ok] = solid[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        data = data * inside.reshape(shape)
    grid = vr._grid_from(data, affine)
    surface = grid.contour([0.5], scalars="values")
    if surface.n_points:
        surface = surface.clean()
    _surfaces[key] = surface
    while len(_surfaces) > 8:
        _surfaces.popitem(last=False)
    return surface


def clip_to_section(mesh, section):
    """``mesh`` with the layer's cut half-spaces removed, as the body is."""
    out = mesh
    for entry in section or []:
        if entry.get("enabled") is False or not out.n_points:
            continue
        axis = "xyz".index(str(entry.get("axis", "x")).lower())
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0
        origin = [0.0, 0.0, 0.0]
        origin[axis] = float(entry.get("at", 0.0))
        # ">" removes everything above the plane: keep the side below it.
        out = out.clip(normal=normal, origin=origin,
                       invert=str(entry.get("op", ">")) == ">")
    return out


class Projector:
    """Masks of regions as the camera sees them, from a small off-screen
    renderer kept for the session: one region drawn alone, flat white on
    black, with a copy of the scene's camera.

    Things stay in its renderer between frames and are shown one at a time
    by visibility. Removing a volume from a renderer frees its texture, so
    clearing every frame re-uploaded each region volume to the GPU per frame.
    """

    def __init__(self):
        self._pl = None
        self._props = {}            # id(thing) -> (thing, prop in the renderer)

    def _prop(self, thing):
        import pyvista as pv

        key = id(thing)
        if key not in self._props:
            if isinstance(thing, pv.DataSet) and (
                    "_inside" in thing.point_data or "_inside" in thing.cell_data):
                # A mesh piece with its coloured part marked 0/1: the whole
                # piece is drawn, so it hides its own far side, and the
                # marking is interpolated before mapping, so the white ends
                # where the fill's colour does.
                prop = self._pl.add_mesh(
                    thing, scalars="_inside", cmap="gray", clim=[0.0, 1.0],
                    interpolate_before_map=True, lighting=False,
                    reset_camera=False, show_scalar_bar=False,
                    preference="cell" if "_inside" in thing.cell_data else "point")
            elif isinstance(thing, pv.DataSet):
                prop = self._pl.add_mesh(thing, color="white", lighting=False,
                                         reset_camera=False, show_scalar_bar=False)
            else:
                self._pl.add_actor(thing, reset_camera=False)
                prop = thing
            self._props[key] = (thing, prop)
        return self._props[key][1]

    def keep_only(self, things):
        """Drop what is no longer outlined (its GPU memory with it)."""
        wanted = {id(t) for t in things}
        for key in [k for k in self._props if k not in wanted]:
            _thing, prop = self._props.pop(key)
            self._pl.remove_actor(prop, reset_camera=False, render=False)

    def mask(self, camera, size, thing, scale=1.0):
        """Where ``thing`` -- a region surface, or a region volume from
        ``volume_render.region_actor`` -- covers the frame, as booleans.

        ``scale`` < 1 renders the mask smaller and scales it back up: a region
        volume costs a full body render, and while working half the
        resolution is a quarter of the rays for an edge that moves well under
        a pixel."""
        import pyvista as pv

        full = (int(size[0]), int(size[1]))
        size = (max(int(full[0] * scale), 8), max(int(full[1] * scale), 8))
        if self._pl is None:
            self._pl = pv.Plotter(off_screen=True, window_size=size)
            self._pl.set_background("black")
            self._pl.disable_anti_aliasing()
        pl = self._pl
        if tuple(pl.window_size) != size:
            pl.window_size = size
        shown = self._prop(thing)
        for _thing, prop in self._props.values():
            prop.SetVisibility(prop is shown)
        pl.camera.DeepCopy(camera)
        pl.render()
        image = np.asarray(pl.screenshot(return_img=True))[..., 0]
        if size != full:
            from PIL import Image

            image = np.asarray(Image.fromarray(image).resize(full, Image.BILINEAR))
        return image > 127


def draw_outline(image, mask, colour, opacity, width):
    """``image`` (H, W, 3) with a line ``width`` pixels wide CENTRED on every
    edge of ``mask``, its outer pixel softened. Centred, so it also covers the
    pixel or so over which a fill fades out into what is under it. A mask is
    flat, so the folds of a region's surface draw nothing; only where it
    starts and stops on screen does."""
    from matplotlib.colors import to_rgb
    from scipy import ndimage

    if not mask.any() or mask.all():
        return image
    # Signed distance to the edge, in pixels (half a pixel either side of it).
    inside = ndimage.distance_transform_edt(mask) - 0.5
    outside = ndimage.distance_transform_edt(~mask) - 0.5
    distance = np.where(mask, inside, outside)
    alpha = np.clip(float(width) / 2.0 + 0.5 - distance, 0.0, 1.0) * float(opacity)
    if not alpha.any():
        return image
    out = np.asarray(image, np.float32).copy()
    ink = np.asarray(to_rgb(colour), np.float32) * 255.0
    out[..., :3] = out[..., :3] * (1.0 - alpha[..., None]) + ink * alpha[..., None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)

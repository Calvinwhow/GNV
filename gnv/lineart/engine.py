"""Line-art drawing: contour lines, hatching and outlines instead of shaded
surfaces, in the manner of Michael Fogleman's ``ln``.

The idea is ln's: a picture made only of lines drawn ON the 3D geometry, with
the parts hidden behind something removed. Visibility comes from a depth buffer
rather than ray casting -- the surfaces are rendered once, opaque, and a point
of a line is kept only where nothing in that buffer is nearer than it.

This module knows nothing about the viewer. It is handed a ``view`` (see
``viewer.py``, the only file that reads the viewer) listing what is on screen,
and the plotter to take the depth pass with.

    surfaces  contour lines (planes every ``spacing_mm`` along ``axis``). On a
              mesh: in the ink, and where a colouring map paints it, denser
              contours in two directions (hatching) in the map's colours with
              the region's edge traced. A surface coloured per cell (a
              parcellation) is drawn region by region in each region's colour.
              Anything else (an electrode) in its own colour.
    lines     fibres, as given, in their colours
    slices    isolines of the anatomy on the slice (contour lines at several
              brightness levels of the image as shown), the tissue's outline,
              and map regions hatched in the map's colours with their edges
              traced. A slice hides what is behind its tissue, not behind its
              dropped-out background.
    always    silhouette and fold lines in the ink, from the camera
"""

from __future__ import annotations

import json

import numpy as np

__all__ = ["DEFAULTS", "settings", "draw", "svg"]

DEFAULTS = {
    "contours": True,        # contour lines across each surface (off: outlines, edges and hatching only)
    "spacing_mm": 1.5,       # contour spacing on unmapped surface
    "axis": "z",             # contour planes are perpendicular to this axis
    "hatch": True,           # cross-hatch what a colouring map paints
    "hatch_mm": 0.8,         # spacing of the hatching
    "region_outline": True,  # trace the edge of each painted region
    "silhouette": True,      # outline and fold lines, from the camera
    "ink": None,             # None -> dark on a light background, light on dark
    "width": 1.0,            # line width in pixels as seen in the pane
    "slice_levels": 6,       # isolines of the anatomy per slice; 0 = none
}

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
# The second hatching direction for each contour axis.
_CROSS = {"x": "y", "y": "z", "z": "y"}
# Relative weights of the stroke kinds.
_WEIGHT = {"contour": 1.0, "hatch": 0.8, "edge": 1.7, "silhouette": 1.5,
           "fibre": 1.0, "iso": 0.9}
# A line point is visible when nothing in the depth buffer is nearer than it
# by more than this, in millimetres. Lines lie ON the surfaces that wrote the
# buffer, so their own surface never hides them.
_EPS_MM = 0.6
# Lines are cut into pieces no longer than this (pixels) for the visibility
# test, so a line is clipped where it goes behind something, not at a vertex.
_PIECE_PX = 1.5
# Colours are quantised to this many levels per channel to join pieces of one
# line into a single stroke (an SVG with one element per piece is enormous).
_LEVELS = 32


def settings(raw):
    """The drawing options: DEFAULTS, overridden by whatever the page sent
    that is known and sane."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (raw or {}).items() if k in DEFAULTS})
    if out["axis"] not in _AXES:
        out["axis"] = "z"
    for key in ("spacing_mm", "hatch_mm", "width"):
        try:
            out[key] = max(float(out[key]), 0.05)
        except (TypeError, ValueError):
            out[key] = DEFAULTS[key]
    for key in ("contours", "hatch", "region_outline", "silhouette"):
        out[key] = bool(out[key])
    try:
        out["slice_levels"] = int(min(max(int(out["slice_levels"]), 0), 30))
    except (TypeError, ValueError):
        out["slice_levels"] = DEFAULTS["slice_levels"]
    return out


def _ink(opts, background):
    if opts.get("ink"):
        return opts["ink"]
    from matplotlib.colors import to_rgb

    try:
        r, g, b = to_rgb(background)
    except ValueError:
        return "#1d1d1f"
    return "#1d1d1f" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "#e8e4dc"


def _rgb01(colour):
    from matplotlib.colors import to_rgb

    return np.asarray(to_rgb(colour), float)


# -- strokes in the world -------------------------------------------------
class Strokes:
    """Polylines in world space: points, per-point RGB (0..1), and a weight per
    polyline. Kept flat (one point array, offsets) so a frame is numpy work."""

    def __init__(self):
        self.points, self.rgb, self.sizes, self.weight = [], [], [], []

    def add(self, points, rgb, weight):
        points = np.asarray(points, np.float64)
        if len(points) < 2:
            return
        rgb = np.asarray(rgb, float)
        if rgb.ndim == 1:
            rgb = np.broadcast_to(rgb, (len(points), 3))
        self.points.append(points)
        self.rgb.append(np.asarray(rgb, np.float32))
        self.sizes.append(len(points))
        self.weight.append(float(weight))

    def extend(self, other):
        self.points += other.points
        self.rgb += other.rgb
        self.sizes += other.sizes
        self.weight += other.weight

    def flat(self):
        if not self.points:
            return None
        return (np.concatenate(self.points), np.concatenate(self.rgb),
                np.asarray(self.sizes, int), np.asarray(self.weight, float))


def _polylines(poly):
    """Each polyline of a PolyData's lines as an index array."""
    lines = np.asarray(poly.lines)
    out, i = [], 0
    while i < len(lines):
        n = int(lines[i])
        out.append(lines[i + 1:i + 1 + n])
        i += n + 1
    return out


def _stripped(vtk_output):
    import pyvista as pv
    import vtk

    strip = vtk.vtkStripper()
    strip.SetInputData(vtk_output)
    strip.JoinContiguousSegmentsOn()
    strip.Update()
    return pv.wrap(strip.GetOutput())


def _contours(mesh, axis, spacing):
    """The mesh cut by planes perpendicular to ``axis`` at every multiple of
    ``spacing`` (multiples, so the lines of two objects line up), joined into
    polylines. Point data -- the colours -- is interpolated along."""
    import vtk

    normal = np.asarray(_AXES[axis])
    pts = np.asarray(mesh.points)
    if not len(pts) or spacing <= 0:
        return None
    d = pts @ normal
    levels = np.arange(np.ceil(d.min() / spacing) * spacing, d.max(), spacing)
    if not len(levels):
        return None
    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, 0.0)
    plane.SetNormal(*normal)
    cut = vtk.vtkCutter()
    cut.SetCutFunction(plane)
    cut.SetInputData(mesh)
    for i, level in enumerate(levels):
        cut.SetValue(i, float(level))
    cut.Update()
    return _stripped(cut.GetOutput())


def _boundary(mesh):
    import vtk

    geometry = vtk.vtkGeometryFilter()     # an extracted piece is a grid
    geometry.SetInputData(mesh)
    geometry.Update()
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(geometry.GetOutput())
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.ColoringOff()
    edges.Update()
    return _stripped(edges.GetOutput())


def _add_lines(strokes, poly, rgb_name, colour, weight):
    if poly is None or poly.n_points == 0:
        return
    pts = np.asarray(poly.points)
    rgb = None
    if rgb_name and rgb_name in poly.point_data:
        rgb = np.asarray(poly.point_data[rgb_name], float)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = rgb[:, :3]
    for ids in _polylines(poly):
        strokes.add(pts[ids], rgb[ids] if rgb is not None else colour, weight)


def _actor_colours(actor, poly):
    """``(rgb 0..1, on_cells)`` an actor actually draws ``poly`` in, or
    ``(constant rgb, None)`` for a single colour."""
    from vtk.util.numpy_support import vtk_to_numpy

    mapper = actor.GetMapper()
    if mapper is not None and mapper.GetScalarVisibility():
        mapped = mapper.MapScalars(1.0)
        if mapped is not None and mapped.GetNumberOfTuples() > 0:
            rgba = vtk_to_numpy(mapped).astype(float)[:, :3] / 255.0
            if len(rgba) == poly.n_points:
                return rgba, False
            if len(rgba) == poly.n_cells:
                return rgba, True
    return np.asarray(actor.GetProperty().GetColor(), float), None


def surface_strokes(actor, poly, opts, ink, role="mesh", base=None):
    """Camera-independent strokes for one surface actor.

    ``role`` "mesh": a single-coloured surface is drawn in the ink (a mesh's
    grey would be a faint line), and where a map paints it -- where its colour
    differs from the layer's own ``base`` colour -- is hatched in the map's
    colours. Any other role draws in the surface's own colours. With
    ``contours`` off a surface draws no contour lines of its own -- only its
    hatching and region edges, and the outlines drawn for every surface."""
    strokes = Strokes()
    lines = opts["contours"]
    colours, on_cells = _actor_colours(actor, poly)
    ink_rgb = _rgb01(ink)
    axis, cross = opts["axis"], _CROSS[opts["axis"]]
    spacing, hatch_mm = float(opts["spacing_mm"]), float(opts["hatch_mm"])

    if on_cells is None:
        # One colour for the whole thing: a plain mesh, an electrode contact.
        colour = ink_rgb if role == "mesh" else colours
        if lines:
            _add_lines(strokes, _contours(poly, axis, spacing), None, colour, _WEIGHT["contour"])
        return strokes

    if on_cells:
        # Coloured per cell (a parcellation): every region its own colour, so
        # cut region by region rather than blending colours across borders.
        keys = np.round(colours * 255).astype(np.int32)
        code = keys[:, 0] * 65536 + keys[:, 1] * 256 + keys[:, 2]
        for value in (np.unique(code) if lines else ()):
            cells = np.flatnonzero(code == value)
            piece = poly.extract_cells(cells)
            _add_lines(strokes, _contours(piece, axis, spacing), None,
                       colours[cells[0]], _WEIGHT["contour"])
        return strokes

    # Coloured per point: a map sampled at the vertices. What the map paints
    # is wherever the colour is not the layer's own.
    mesh = poly.copy(deep=False)
    mesh.point_data["_rgb"] = (colours * 255).astype(np.uint8)
    if role != "mesh" or base is None:
        if lines:
            _add_lines(strokes, _contours(mesh, axis, spacing), "_rgb", None, _WEIGHT["contour"])
        return strokes
    mapped = np.abs(colours - _rgb01(base)[None, :]).max(1) > 2.5 / 255
    faces = mesh.faces.reshape(-1, 4)[:, 1:] if mesh.n_cells else np.zeros((0, 3), int)
    inside = mapped[faces].all(1)
    rest = np.flatnonzero(~inside)
    painted = np.flatnonzero(inside)
    if len(rest) and lines:
        _add_lines(strokes, _contours(mesh.extract_cells(rest), axis, spacing),
                   None, ink_rgb, _WEIGHT["contour"])
    if len(painted):
        part = mesh.extract_cells(painted)
        if opts["hatch"]:
            _add_lines(strokes, _contours(part, axis, hatch_mm), "_rgb", None, _WEIGHT["hatch"])
            _add_lines(strokes, _contours(part, cross, hatch_mm), "_rgb", None, _WEIGHT["hatch"])
        elif lines:
            _add_lines(strokes, _contours(part, axis, spacing), "_rgb", None, _WEIGHT["contour"])
        if opts["region_outline"]:
            _add_lines(strokes, _boundary(part), "_rgb", None, _WEIGHT["edge"])
    return strokes


def line_strokes(poly, rgb):
    """Strokes for polylines (fibres) with a colour per point (0..1), or one
    colour for all."""
    strokes = Strokes()
    pts = np.asarray(poly.points)
    rgb = np.asarray(rgb, float)
    for ids in _polylines(poly):
        strokes.add(pts[ids], rgb[ids] if rgb.ndim == 2 else rgb, _WEIGHT["fibre"])
    return strokes


def silhouette_strokes(poly, camera, ink):
    """Outline and fold lines of a surface as seen from ``camera``."""
    import vtk

    sil = vtk.vtkPolyDataSilhouette()
    sil.SetInputData(poly)
    sil.SetCamera(camera)
    if camera.GetParallelProjection():
        sil.SetDirectionToCameraVector()
    else:
        sil.SetDirectionToCameraOrigin()
    sil.BorderEdgesOn()
    sil.SetEnableFeatureAngle(False)
    sil.PieceInvariantOff()
    sil.Update()
    strokes = Strokes()
    _add_lines(strokes, _stripped(sil.GetOutput()), None, _rgb01(ink),
               _WEIGHT["silhouette"])
    return strokes


# -- slices ---------------------------------------------------------------
# A slice is a flat image on a plane: ``plane`` is (origin, u axis, v axis) in
# world mm -- texture (u, v) in [0, 1] maps to origin + u*U + v*V -- and
# ``rgba`` is the image exactly as shown (rows from v = 0), alpha 0 where the
# background is dropped out.
_SMOOTH_PX = 0.7          # light smoothing of the image before contouring
_MAP_CHROMA = 0.12        # a pixel this coloured is painted by a map


def _chaikin(p, n=2):
    """Corner-cutting smoothing of a 2D polyline (closed if it ends where it
    starts), so pixel-scale contours read as drawn curves."""
    closed = len(p) > 3 and np.allclose(p[0], p[-1])
    for _ in range(n):
        if len(p) < 3:
            break
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        p = np.vstack([mid, mid[:1]]) if closed else np.vstack([p[:1], mid, p[-1:]])
    return p


def _smooth_within(values, weight, sigma):
    """Gaussian smoothing that only averages over ``weight`` pixels, so the
    dropped-out background (or a map) does not bleed into the edge and lay
    rings of false contours along it."""
    from scipy.ndimage import gaussian_filter

    w = weight.astype(float)
    num = gaussian_filter(values * w, sigma)
    den = gaussian_filter(w, sigma)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)


def _image_lines(field, level, mask=None):
    """Contour lines of a 2D field at ``level`` as (x, y) pixel arrays."""
    import contourpy

    z = np.ma.array(field, mask=mask) if mask is not None else field
    gen = contourpy.contour_generator(z=z, line_type="Separate")
    return [np.asarray(line, float) for line in gen.lines(float(level)) if len(line) > 1]


def slice_strokes(plane, rgba, opts, ink):
    """Camera-independent strokes for one slice."""
    origin, U, V = (np.asarray(a, float) for a in plane)
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3].astype(float) / 255.0
    tissue = rgba[..., 3] > 127
    chroma = rgb.max(2) - rgb.min(2)
    mapped = tissue & (chroma > _MAP_CHROMA)
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    ink_rgb = _rgb01(ink)
    strokes = Strokes()

    def world(xy):
        u = (xy[:, 0] + 0.5) / w
        v = (xy[:, 1] + 0.5) / h
        return origin + u[:, None] * U + v[:, None] * V

    # Isolines of the anatomy, where no map paints over it.
    anatomy = tissue & ~mapped
    n = int(opts["slice_levels"])
    if n > 0 and anatomy.sum() > 16:
        field = _smooth_within(lum, anatomy, _SMOOTH_PX)
        levels = np.unique(np.quantile(lum[anatomy], np.linspace(0.12, 0.92, n)))
        for level in levels:
            for line in _image_lines(field, level, mask=~anatomy):
                strokes.add(world(_chaikin(line)), ink_rgb, _WEIGHT["iso"])

    # The tissue's outline, where the slice drops out.
    if opts["silhouette"] and tissue.any() and not tissue.all():
        from scipy.ndimage import gaussian_filter

        edge = gaussian_filter(np.pad(tissue.astype(float), 1), _SMOOTH_PX)
        for line in _image_lines(edge, 0.5):
            strokes.add(world(_chaikin(line - 1.0)), ink_rgb, _WEIGHT["silhouette"])

    # Map regions: hatched in their colours, edges traced.
    if mapped.any():
        from scipy.ndimage import distance_transform_edt, gaussian_filter

        near = distance_transform_edt(~mapped, return_distances=False, return_indices=True)
        colour = rgb[near[0], near[1]]           # every pixel: its nearest map colour

        def colour_at(xy):
            ix = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
            iy = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
            return colour[iy, ix]

        if opts["hatch"]:
            mm_u = np.linalg.norm(U) / w
            mm_v = np.linalg.norm(V) / h
            step = float(opts["hatch_mm"])
            for along, count, gap, other in ((0, h, step / mm_v, w), (1, w, step / mm_u, h)):
                ts = np.arange(0.0, other - 0.5, 0.5)
                for c in np.arange(gap / 2.0, count - 0.5, gap):
                    xy = (np.c_[ts, np.full(len(ts), c)] if along == 0
                          else np.c_[np.full(len(ts), c), ts])
                    ix = np.clip(xy[:, 0].astype(int), 0, w - 1)
                    iy = np.clip(xy[:, 1].astype(int), 0, h - 1)
                    on = mapped[iy, ix]
                    if not on.any():
                        continue
                    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
                    for a, b in zip(edges[0::2], edges[1::2]):
                        seg = xy[a:b][::4] if b - a > 8 else xy[a:b]
                        seg = np.vstack([seg, xy[b - 1]]) if len(seg) and not np.allclose(seg[-1], xy[b - 1]) else seg
                        if len(seg) > 1:
                            strokes.add(world(seg), colour_at(seg), _WEIGHT["hatch"])
        if opts["region_outline"]:
            ring = gaussian_filter(np.pad(mapped.astype(float), 1), _SMOOTH_PX)
            for line in _image_lines(ring, 0.5):
                line = _chaikin(line - 1.0)
                strokes.add(world(line), colour_at(line), _WEIGHT["edge"])
    return strokes


def _homography(src, dst):
    """3x3 H with dst ~ H @ src for four 2D point pairs."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows, float))
    return vt[-1].reshape(3, 3)


def _slice_into_depth(zmm, camera, w, h, plane, tissue):
    """Lay a slice's TISSUE into the depth map (mm), in software: for each
    pixel the slice covers, the depth of the plane there, where the image
    there is not dropped out. (Rendered as a textured quad, the background
    would hide whatever is behind it too.)"""
    origin, U, V = (np.asarray(a, float) for a in plane)
    corners_uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    corners = origin + corners_uv[:, :1] * U + corners_uv[:, 1:] * V
    x, y, _ = _project(camera, corners, w, h)
    x0, x1 = int(max(np.floor(x.min()), 0)), int(min(np.ceil(x.max()), w))
    y0, y1 = int(max(np.floor(y.min()), 0)), int(min(np.ceil(y.max()), h))
    if x1 <= x0 or y1 <= y0:
        return
    H = _homography(np.c_[x, y], corners_uv)
    gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    homo = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)]) 
    uvw = H @ homo
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
    th, tw = tissue.shape
    inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    hit = np.zeros(gx.size, bool)
    ii = np.flatnonzero(inside)
    hit[ii] = tissue[(v[ii] * th).astype(int), (u[ii] * tw).astype(int)]
    if not hit.any():
        return
    pts = origin + u[hit, None] * U + v[hit, None] * V
    depth = (pts - np.asarray(camera.GetPosition())) @ np.asarray(camera.GetDirectionOfProjection())
    rows = (gy.ravel()[hit] - 0.5).astype(int)
    cols = (gx.ravel()[hit] - 0.5).astype(int)
    np.minimum.at(zmm, (rows, cols), depth)


# -- one frame ------------------------------------------------------------
def _depth(pl, occluders, bounds):
    """The depth buffer of the occluding surfaces alone, opaque and unlit-fast,
    with everything else hidden; the scene is put back as it was."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    ren = pl.renderer
    props = ren.GetViewProps()
    props.InitTraversal()
    saved = []
    keep = {a.GetAddressAsString("vtkObject") for a in occluders}
    for _ in range(props.GetNumberOfItems()):
        prop = props.GetNextProp()
        saved.append((prop, prop.GetVisibility()))
        prop.SetVisibility(prop.GetAddressAsString("vtkObject") in keep)
    opacity = [(a, a.GetProperty().GetOpacity()) for a in occluders]
    for a, _o in opacity:
        a.GetProperty().SetOpacity(1.0)
    try:
        if bounds is not None:
            ren.ResetCameraClippingRange(*bounds)
        # The render window itself, not pl.render(): pyvista skips rendering a
        # plotter that has not been shown yet, which left a new plotter's
        # depth buffer undrawn (garbage values; found by checking its range).
        pl.ren_win.Render()
        # Through vtkWindowToImageFilter, not GetZbufferData: on a multisampled
        # off-screen buffer the direct read returned garbage (+-3e38, or all
        # zeros at large sizes) -- measured. The filter resolves the buffer.
        grab = vtk.vtkWindowToImageFilter()
        grab.SetInput(pl.ren_win)
        grab.SetScale(1)
        grab.ReadFrontBufferOff()
        grab.SetInputBufferTypeToZBuffer()
        grab.Update()
        image = grab.GetOutput()
        w, h = image.GetDimensions()[:2]
        z = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(h, w).copy()
    finally:
        for a, o in opacity:
            a.GetProperty().SetOpacity(o)
        for prop, vis in saved:
            prop.SetVisibility(vis)
    return z


def _project(camera, points, w, h):
    """World points -> pixel x, y (origin bottom-left) and depth in mm along
    the view direction; and the near/far planes."""
    m = camera.GetCompositeProjectionTransformMatrix(w / h, -1, 1)
    M = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
    homo = np.c_[points, np.ones(len(points))] @ M.T
    ndc = homo[:, :3] / homo[:, 3:4]
    x = (ndc[:, 0] + 1) * 0.5 * w
    y = (ndc[:, 1] + 1) * 0.5 * h
    pos = np.asarray(camera.GetPosition())
    fwd = np.asarray(camera.GetDirectionOfProjection())
    depth = (points - pos) @ fwd
    return x, y, depth


def _buffer_depth(z01, camera):
    """The depth buffer in mm along the view direction."""
    near, far = camera.GetClippingRange()
    if camera.GetParallelProjection():
        return near + z01 * (far - near)
    zn = 2.0 * z01 - 1.0
    return 2.0 * near * far / ((far + near) - zn * (far - near))


def _visible_runs(flat, camera, zmm, w, h, scale):
    """Cut every polyline into short pieces, keep those whose both ends are in
    front of the depth buffer, and join kept pieces back into runs of one
    colour. Returns a list of (xy[n,2], rgb, width)."""
    from scipy.ndimage import maximum_filter

    points, rgb, sizes, weight = flat
    line_of = np.repeat(np.arange(len(sizes)), sizes)
    # segments: consecutive points of the same polyline
    last = np.zeros(len(points), bool)
    last[np.cumsum(sizes) - 1] = True
    a_idx = np.flatnonzero(~last)
    b_idx = a_idx + 1
    if not len(a_idx):
        return []
    xa, ya, _ = _project(camera, points[a_idx], w, h)
    xb, yb, _ = _project(camera, points[b_idx], w, h)
    length = np.hypot(xb - xa, yb - ya)
    pieces = np.clip(np.ceil(length / _PIECE_PX).astype(int), 1, 400)
    seg = np.repeat(np.arange(len(a_idx)), pieces)
    first = np.r_[0, np.cumsum(pieces)[:-1]]
    k = np.arange(len(seg)) - np.repeat(first, pieces)
    t0 = k / np.repeat(pieces, pieces)
    t1 = (k + 1) / np.repeat(pieces, pieces)
    pa, pb = points[a_idx][seg], points[b_idx][seg]
    p0 = pa + (pb - pa) * t0[:, None]
    p1 = pa + (pb - pa) * t1[:, None]
    ca, cb = rgb[a_idx][seg], rgb[b_idx][seg]
    colour = ca + (cb - ca) * ((t0 + t1) * 0.5)[:, None]

    # visibility of both ends of every piece
    nearest = maximum_filter(zmm, size=3, mode="nearest")
    mm_per_px = _mm_per_px(camera, h)
    eps = max(_EPS_MM, 2.0 * mm_per_px)

    def seen(p):
        x, y, d = _project(camera, p, w, h)
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        ok = np.zeros(len(p), bool)
        ok[inside] = d[inside] <= nearest[iy[inside], ix[inside]] + eps
        return ok, x, y

    v0, x0, y0 = seen(p0)
    v1, x1, y1 = seen(p1)
    keep = v0 & v1
    # Join kept pieces into runs: same polyline, adjacent, same colour level.
    line = line_of[a_idx][seg]
    level = np.round(colour * (_LEVELS - 1)).astype(np.int16)
    code = level[:, 0].astype(np.int32) * 4096 + level[:, 1] * 64 + level[:, 2]
    joined = np.zeros(len(seg), bool)
    joined[1:] = (keep[1:] & keep[:-1] & (line[1:] == line[:-1])
                  & (code[1:] == code[:-1])
                  & (np.abs(x0[1:] - x1[:-1]) < 1e-3) & (np.abs(y0[1:] - y1[:-1]) < 1e-3))
    runs = []
    idx = np.flatnonzero(keep)
    if not len(idx):
        return runs
    widths = weight[line] * scale
    # A run starts at every kept piece not joined to the one before it.
    bounds = np.flatnonzero(~joined[idx])
    bounds = np.r_[bounds, len(idx)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        pcs = idx[s:e]
        # The test pieces are collinear cuts of one segment: keep only where a
        # segment starts (a real vertex) and the run's two ends, so the stroke
        # has the geometry's own points and an SVG stays small.
        vertex = pcs[(k[pcs] == 0) | (np.arange(len(pcs)) == 0)]
        xy = np.empty((len(vertex) + 1, 2))
        xy[:-1, 0], xy[:-1, 1] = x0[vertex], y0[vertex]
        xy[-1] = (x1[pcs[-1]], y1[pcs[-1]])
        runs.append((xy, level[pcs[0]] / (_LEVELS - 1), float(widths[pcs[0]])))
    return runs


def _mm_per_px(camera, h):
    if camera.GetParallelProjection():
        return 2.0 * camera.GetParallelScale() / h
    dist = camera.GetDistance()
    return 2.0 * dist * np.tan(np.radians(camera.GetViewAngle()) / 2.0) / h


def _raster(runs, size, background):
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    w, h = size
    fig = Figure(figsize=(w / 72.0, h / 72.0), dpi=72)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    fig.patch.set_facecolor(background)
    if runs:
        ax.add_collection(LineCollection(
            [r[0] for r in runs], colors=[tuple(r[1]) for r in runs],
            linewidths=[r[2] for r in runs], capstyle="round", joinstyle="round"))
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    return image


def svg(runs, size, background):
    """An SVG of the runs, y flipped to SVG's top-left origin."""
    w, h = size
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}">',
           f'<rect width="100%" height="100%" fill="{background}"/>',
           '<g fill="none" stroke-linecap="round" stroke-linejoin="round">']
    for xy, rgb, width in runs:
        r, g, b = (np.clip(rgb, 0, 1) * 255).round().astype(int)
        pts = " ".join(f"{x:.1f},{h - y:.1f}" for x, y in xy)
        out.append(f'<polyline points="{pts}" stroke="#{r:02x}{g:02x}{b:02x}" '
                   f'stroke-width="{width:.2f}"/>')
    out.append("</g></svg>")
    return "\n".join(out)


# Camera-independent strokes, per surface or fibre set, reused frame to frame
# while the thing they were drawn from is unchanged. Pruned to what is live.
_CACHE = {}


def draw(view, size, line_scale=1.0, options=None):
    """The current view as line art: ``(image, runs, notes)``.

    ``view`` (viewer.View) supplies the plotter, the background and the items
    on screen; ``runs`` are the visible strokes in pixels, for ``svg``."""
    opts = settings(options)
    background = view.background
    ink = _ink(opts, background)
    w, h = int(size[0]), int(size[1])
    pl = view.plotter
    camera = pl.renderer.GetActiveCamera()
    stroke_key = json.dumps([opts, ink], sort_keys=True, default=str)

    notes, occluders, strokes, sil_polys, live = [], [], Strokes(), [], set()
    slices = []
    for item in view.items(notes):
        if item["kind"] == "slice":
            slices.append(item)
            key = ("slice", item["key"], stroke_key)
            live.add(key)
            if key not in _CACHE:
                try:
                    _CACHE[key] = slice_strokes(item["plane"], item["rgba"], opts, ink)
                except ImportError as exc:      # contourpy comes with matplotlib >= 3.6
                    _CACHE[key] = Strokes()
                    notes.append(f"line art: slices need contourpy ({exc})")
            strokes.extend(_CACHE[key])
            continue
        if item["kind"] == "lines":
            key = ("lines", item["key"])
            live.add(key)
            if key not in _CACHE:
                _CACHE[key] = line_strokes(*item["make"]())
            strokes.extend(_CACHE[key])
            continue
        actor, poly = item["actor"], item["poly"]
        occluders.append(actor)
        key = ("surface", item["key"], stroke_key)
        live.add(key)
        if key not in _CACHE:
            _CACHE[key] = surface_strokes(actor, poly, opts, ink,
                                          role=item["role"], base=item.get("base"))
        strokes.extend(_CACHE[key])
        if opts["silhouette"]:
            sil_polys.append(poly)
    for key in [k for k in _CACHE if k not in live]:
        del _CACHE[key]

    for poly in sil_polys:
        strokes.extend(silhouette_strokes(poly, camera, ink))

    runs = []
    flat = strokes.flat()
    if flat is not None:
        lo, hi = flat[0].min(0), flat[0].max(0)
        bounds = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
        boxes = [a.GetBounds() for a in occluders]
        for item in slices:
            o, U, V = (np.asarray(a, float) for a in item["plane"])
            c = np.array([o, o + U, o + V, o + U + V])
            boxes.append((c[:, 0].min(), c[:, 0].max(), c[:, 1].min(),
                          c[:, 1].max(), c[:, 2].min(), c[:, 2].max()))
        for b in boxes:
            bounds = [min(bounds[0], b[0]), max(bounds[1], b[1]),
                      min(bounds[2], b[2]), max(bounds[3], b[3]),
                      min(bounds[4], b[4]), max(bounds[5], b[5])]
        z = _depth(pl, occluders, bounds)
        zmm = _buffer_depth(z, camera)
        for item in slices:
            _slice_into_depth(zmm, camera, w, h, item["plane"], item["rgba"][..., 3] > 127)
        runs = _visible_runs(flat, camera, zmm, w, h,
                             float(opts["width"]) * float(line_scale))
    return _raster(runs, (w, h), background), runs, notes

"""Everything the path tracer reads from the viewer -- and the ONLY place it does.

Like line art, this add-on lets the viewer draw a frame the ordinary way and
then looks at what it drew: the actors on screen, the colours they were given,
the camera, the lights. ``snapshot`` copies all of that into plain arrays, so
the slow part -- the path tracing -- runs on its own thread without touching
VTK or the viewer again.

Read-only, always. What it relies on (the whole contract):
    session.spec                 the figure: layers (id, kind, visible,
                                 tube_radius), background, lighting
    session._actors[layer_id]    the VTK actors drawn for each layer
    session._plotter             the off-screen pyvista Plotter
    session._gl                  the render thread (an executor)
    session.render(...)          to bring the scene up to date for a camera
"""

from __future__ import annotations

import numpy as np

# A line that is not a fibre (an outline, a region edge) is drawn this thick, in mm.
LINE_RADIUS_MM = 0.15


def snapshot(session, size, camera=None, line_scale=1.0):
    """The current view as plain data: ``{"camera", "lights", "background",
    "surfaces", "curves", "slices", "bars", "notes"}``. Call with the viewer's
    lock held. The viewer draws the frame the ordinary way first, so its actors
    and camera are exactly what the pane shows; they are then read on its
    render thread (VTK is single-threaded)."""
    session.render(size=size, quality="preview", scalar_bar=True, camera=camera or None, antialias="off", line_scale=line_scale)
    return session._gl.submit(_read, session, size).result()


def _read(session, size):
    pl = session._plotter
    spec = session.spec or {}
    out = {"camera": _camera(pl, size), "lights": dict(spec.get("lighting") or {}), "background": spec.get("background", "#ffffff"),
           "surfaces": [], "curves": [], "slices": [], "notes": []}
    for layer in spec.get("layers", []):
        if not layer.get("visible", True):
            continue
        actors = (getattr(session, "_actors", {}) or {}).get(layer.get("id")) or []
        if layer.get("kind") == "volume":
            if actors and "volume layers are not path traced yet" not in out["notes"]:
                out["notes"].append("volume layers are not path traced yet")
            continue
        for actor in actors:
            _read_actor(session, actor, layer, out)
    out["bars"] = _bars_only(pl)
    return out


def _read_actor(session, actor, layer, out):
    """Sort one actor into a surface, a set of curves or a slice. A fibre layer's
    tubes become curves along its streamlines, which the path tracer draws as
    true tubes."""
    import pyvista as pv
    import vtk

    from gnv.onscreen import fibre_streamlines

    if not isinstance(actor, vtk.vtkActor) or not actor.GetVisibility():
        return
    prop = actor.GetProperty()
    if prop.GetRepresentation() != 2:
        return                      # a wireframe (mesh edges): not part of the picture
    mapper = actor.GetMapper()
    poly = pv.wrap(mapper.GetInput()) if mapper is not None else None
    if poly is None or poly.n_points == 0:
        return
    matrix = _matrix(actor)
    if actor.GetTexture() is not None and poly.n_points == 4:
        item = _slice(actor, poly, matrix)
        if item is not None:
            out["slices"].append(item)
        return
    key = (actor.GetAddressAsString("vtkObject"), int(poly.GetMTime()), int(mapper.GetMTime()))
    radius = float(layer.get("tube_radius") or 0.2) if layer.get("kind") == "fibers" else LINE_RADIUS_MM
    if layer.get("kind") == "fibers":
        found = fibre_streamlines(session, actor, poly)
        if found is not None:
            lines, colours = found
            out["curves"].append({"polylines": _polylines(lines, matrix), "colours": colours, "radius": radius, "key": key})
            return
    colours = _colours(actor, poly)
    faces = poly.GetNumberOfPolys() + poly.GetNumberOfStrips()
    if poly.GetNumberOfLines() and not faces:
        out["curves"].append({"polylines": _polylines(poly, matrix), "colours": colours, "radius": radius, "key": key})
        return
    if not faces:
        return
    surface = poly.extract_surface(algorithm=None).triangulate()
    if colours.ndim == 2 and len(colours) == poly.n_cells:
        surface = poly.copy(deep=False)
        surface.cell_data["_rgb"] = colours
        surface = surface.extract_surface(algorithm=None).triangulate()
        colours = np.asarray(surface.cell_data["_rgb"], float)
    elif colours.ndim == 2 and len(colours) == poly.n_points:
        surface = poly.copy(deep=False)
        surface.point_data["_rgb"] = colours
        surface = surface.extract_surface(algorithm=None).triangulate()
        colours = np.asarray(surface.point_data["_rgb"], float)
    smooth = prop.GetInterpolation() != 0      # VTK_FLAT is 0
    points = _apply(matrix, np.asarray(surface.points, float))
    normals = None
    if smooth:
        try:
            normals = np.asarray(surface.compute_normals(cell_normals=False, split_vertices=False, consistent_normals=False).point_data["Normals"], float)
            normals = _apply(matrix, normals, direction=True)
        except RuntimeError:
            normals = None          # degenerate geometry: drawn faceted rather than not at all
    out["surfaces"].append({"points": points, "faces": np.asarray(surface.regular_faces, np.int64), "normals": normals, "colours": colours,
                            "opacity": float(prop.GetOpacity()), "specular": float(prop.GetSpecular()), "kind": layer.get("kind"),
                            "key": key})


def _colours(actor, poly):
    """RGB (0..1) as the actor draws it: one per point, one per cell, or one colour."""
    from vtk.util.numpy_support import vtk_to_numpy

    mapper = actor.GetMapper()
    if mapper is not None and mapper.GetScalarVisibility():
        mapped = mapper.MapScalars(1.0)
        if mapped is not None and mapped.GetNumberOfTuples() in (poly.n_points, poly.n_cells):
            return vtk_to_numpy(mapped).astype(float)[:, :3] / 255.0
    return np.asarray(actor.GetProperty().GetColor(), float)


def _polylines(poly, matrix):
    """Each polyline of ``poly`` with at least two points, as (point ids, points)."""
    from gnv.onscreen import polyline_ids

    points = _apply(matrix, np.asarray(poly.points, float))
    return [(ids, points[ids]) for ids in polyline_ids(poly) if len(ids) > 1]


def _slice(actor, poly, matrix):
    """A slice as the viewer draws it: a textured quad, as {corners, rgba}.
    ``corners`` are the world points at texture (0,0), (1,0) and (0,1);
    ``rgba`` has row 0 at texture t=0, as VTK stores it."""
    from vtk.util.numpy_support import vtk_to_numpy

    image = actor.GetTexture().GetInput()
    tcoords = poly.GetPointData().GetTCoords()
    if image is None or tcoords is None:
        return None
    pts = _apply(matrix, np.asarray(poly.points, float))
    tc = vtk_to_numpy(tcoords).astype(float)
    corner = lambda uv: pts[int(np.argmin(np.abs(tc - np.asarray(uv)).sum(1)))]
    w, h = image.GetDimensions()[:2]
    scalars = vtk_to_numpy(image.GetPointData().GetScalars())
    scalars = scalars[:, None] if scalars.ndim == 1 else scalars
    rgba = np.full((h * w, 4), 255, np.uint8)
    rgba[:, :3] = scalars[:, :3] if scalars.shape[1] >= 3 else scalars[:, :1]
    if scalars.shape[1] in (2, 4):
        rgba[:, 3] = scalars[:, -1]
    return {"corners": (corner((0, 0)), corner((1, 0)), corner((0, 1))), "rgba": rgba.reshape(h, w, 4),
            "opacity": float(actor.GetProperty().GetOpacity()),
            "key": (actor.GetAddressAsString("vtkObject"), int(image.GetMTime()), int(poly.GetMTime()))}


def _camera(pl, size):
    """The camera exactly as the view uses it."""
    cam = pl.camera
    return {"position": np.asarray(cam.position, float), "focal_point": np.asarray(cam.focal_point, float), "up": np.asarray(cam.up, float),
            "parallel": bool(cam.parallel_projection), "parallel_scale": float(cam.parallel_scale), "view_angle": float(cam.view_angle),
            "size": (int(size[0]), int(size[1]))}


def _bars_only(pl):
    """The colour bars (and any other 2D annotation) on a clear background, to
    lay over the path-traced picture: the 3D props are hidden for one frame."""
    renderer = pl.renderer
    props = [renderer.GetViewProps().GetItemAsObject(i) for i in range(renderer.GetViewProps().GetNumberOfItems())]
    hidden = [p for p in props if p.IsA("vtkProp3D") and p.GetVisibility()]
    for p in hidden:
        p.SetVisibility(False)
    try:
        pl.render()                 # a fresh frame: screenshot alone may hand back the last one
        image = pl.screenshot(transparent_background=True, return_img=True)
    finally:
        for p in hidden:
            p.SetVisibility(True)
    return np.asarray(image)


def _matrix(actor):
    """The actor's own placement, or None when it has none."""
    m = actor.GetMatrix()
    out = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)], float)
    return None if np.allclose(out, np.eye(4)) else out


def _apply(matrix, points, direction=False):
    if matrix is None or not len(points):
        return points
    if direction:
        moved = points @ np.linalg.inv(matrix[:3, :3]).T
        return moved / np.maximum(np.linalg.norm(moved, axis=1, keepdims=True), 1e-12)
    return points @ matrix[:3, :3].T + matrix[:3, 3]

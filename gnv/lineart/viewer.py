"""Everything line art reads from the viewer -- and the ONLY place it does.

Line art is an add-on. The viewer does not know it exists: nothing in
session.py, spec.py or the page script was changed for it. Instead, after the
viewer has drawn a frame the ordinary way, this module looks at what it drew
-- the actors on screen, the colours they were given, the camera -- and hands
that to ``engine.py``.

So if the viewer's internals change, this is the file to update, and a break
here can only ever cost the line-art view: every read below is defensive, and
the routes report a failure instead of raising into the viewer.

Read-only, always: nothing here sets, edits or caches anything on the session.

What it relies on (the whole contract):
    session.spec                 the figure: layers (id, kind, visible, color),
                                 background
    session._actors[layer_id]    the VTK actors drawn for each layer
    session._plotter             the off-screen pyvista Plotter
    session._gl                  the render thread (an executor)
    session._fibers              the fibre cache: entries with "lines" (the
                                 loaded streamlines) and "tubes" (what is drawn)
    session.render(...)          to bring the scene up to date for a camera
"""

from __future__ import annotations

import numpy as np

from gnv.onscreen import colours_on_points


class View:
    """What is on screen right now, in the terms ``engine.draw`` wants."""

    def __init__(self, session):
        self._session = session

    @property
    def plotter(self):
        return self._session._plotter

    @property
    def background(self):
        return (self._session.spec or {}).get("background", "#ffffff")

    def items(self, notes):
        """Surfaces and line sets on screen, as dicts:

        surface: kind="surface", actor, poly, role ("mesh" | "other"),
                 base (a mesh layer's own colour), key (changes when the
                 geometry or its colours do)
        lines:   kind="lines", key, make() -> (polylines, rgb per point)

        ``notes`` collects what could not be drawn, for the status line.
        """
        import pyvista as pv
        import vtk

        session = self._session
        actors_by_layer = getattr(session, "_actors", {}) or {}
        for layer in (session.spec or {}).get("layers", []):
            if not layer.get("visible", True):
                continue
            actors = actors_by_layer.get(layer.get("id")) or []
            kind = layer.get("kind")
            if kind == "volume":
                if actors:
                    note = "line art: volume layers are not drawn"
                    if note not in notes:
                        notes.append(note)
                continue
            if kind == "slices":
                for actor in actors:
                    item = _slice_item(actor)
                    if item is not None:
                        yield item
                continue
            for actor in actors:
                if not isinstance(actor, vtk.vtkActor) or not actor.GetVisibility():
                    continue
                if actor.GetProperty().GetRepresentation() != 2:
                    continue        # a wireframe (mesh edges): not a surface
                mapper = actor.GetMapper()
                poly = pv.wrap(mapper.GetInput()) if mapper is not None else None
                if poly is None or poly.n_cells == 0:
                    continue
                key = (actor.GetAddressAsString("vtkObject"),
                       int(poly.GetMTime()), int(mapper.GetMTime()))
                if kind == "fibers":
                    lines = self._fibre_lines(actor, poly)
                    if lines is not None:
                        yield {"kind": "lines", "key": key, "make": lines}
                    continue
                if poly.GetNumberOfPolys() == 0 and poly.GetNumberOfStrips() == 0:
                    yield {"kind": "lines", "key": key,
                           "make": lambda a=actor, p=poly: (p, colours_on_points(a, p))}
                    continue
                yield {"kind": "surface", "key": key, "actor": actor, "poly": poly,
                       "role": "mesh" if kind == "mesh" else "other",
                       "base": layer.get("color")}

    # -- fibres ---------------------------------------------------------------
    def _fibre_lines(self, actor, tubes):
        """A fibre layer is drawn as swept tubes; line art wants the
        streamlines inside them, in the colours on screen (see
        ``gnv.onscreen.fibre_streamlines``). Returned as a function, so they
        are only gathered when line art has not already drawn them."""
        from gnv.onscreen import fibre_source, fibre_streamlines

        if fibre_source(self._session, tubes) is None:
            return None
        return lambda actor=actor, tubes=tubes: fibre_streamlines(self._session, actor, tubes)


def _slice_item(actor):
    """A slice as the viewer draws it -- a four-cornered quad carrying the
    image as a texture -- as {kind="slice", plane=(origin, U, V), rgba}.
    Anything else (an outline line, a frame) is skipped."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    if not isinstance(actor, vtk.vtkActor) or not actor.GetVisibility():
        return None
    texture = actor.GetTexture()
    image = texture.GetInput() if texture is not None else None
    poly = actor.GetMapper().GetInput() if actor.GetMapper() is not None else None
    if image is None or poly is None or poly.GetNumberOfPoints() != 4:
        return None
    tcoords = poly.GetPointData().GetTCoords()
    if tcoords is None:
        return None
    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    tc = vtk_to_numpy(tcoords).astype(float)
    corner = lambda uv: pts[int(np.argmin(np.abs(tc - np.asarray(uv)).sum(1)))]
    origin = corner((0, 0))
    plane = (origin, corner((1, 0)) - origin, corner((0, 1)) - origin)
    w, h = image.GetDimensions()[:2]
    scalars = vtk_to_numpy(image.GetPointData().GetScalars())
    if scalars.ndim == 1:
        scalars = scalars[:, None]
    rgba = np.zeros((h * w, 4), np.uint8)
    if scalars.shape[1] >= 3:
        rgba[:, :3] = scalars[:, :3]
    else:
        rgba[:, :3] = scalars[:, :1]
    rgba[:, 3] = scalars[:, 3] if scalars.shape[1] == 4 else (
        scalars[:, 1] if scalars.shape[1] == 2 else 255)
    key = (actor.GetAddressAsString("vtkObject"), texture.GetAddressAsString("vtkObject"),
           int(image.GetMTime()), int(poly.GetMTime()))
    return {"kind": "slice", "key": key, "plane": plane, "rgba": rgba.reshape(h, w, 4)}


def bring_up_to_date(session, size, camera):
    """Have the viewer draw this frame the ordinary way, so its actors and
    camera are exactly what the pane shows. Plain, unsupersampled, no bars:
    line art takes its depth pass at this exact size."""
    session.render(size=size, quality="preview", scalar_bar=False,
                   camera=camera or None, antialias="off")


def on_render_thread(session, fn, *args):
    """Run ``fn`` on the viewer's render thread (VTK is single-threaded)."""
    return session._gl.submit(fn, *args).result()

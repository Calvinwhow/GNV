"""What the viewer has drawn, read back from its actors -- shared by the add-ons
that redraw the view another way (line art, path traced).

Read-only: nothing here changes the session or its actors. What it relies on:
``session._fibers``, the fibre cache, whose entries hold "lines" (the loaded
streamlines) and "tubes" (what is drawn).
"""

from __future__ import annotations

import numpy as np

__all__ = ["colours_on_points", "fibre_source", "fibre_streamlines", "polyline_ids"]


def colours_on_points(actor, poly):
    """RGB (0..1) per point of ``poly`` as ``actor`` draws it, or one RGB."""
    from vtk.util.numpy_support import vtk_to_numpy

    mapper = actor.GetMapper()
    if mapper is not None and mapper.GetScalarVisibility():
        mapped = mapper.MapScalars(1.0)
        if mapped is not None and mapped.GetNumberOfTuples() == poly.n_points:
            return vtk_to_numpy(mapped).astype(float)[:, :3] / 255.0
    return np.asarray(actor.GetProperty().GetColor(), float)


def fibre_streamlines(session, actor, tubes):
    """The streamlines inside a fibre layer's drawn ``tubes``, as
    ``(lines PolyData, RGB per line point)``, or None.

    A fibre layer is drawn as swept tubes; the streamlines are found in the
    viewer's fibre cache -- the entry whose tubes share this actor's points --
    and kept only for the fibres actually drawn (gates and "values only" drop
    whole streamlines, and the tubes carry each point's fibre id). Each line
    point takes the colour of the nearest tube point, i.e. exactly the colour
    on screen, however the layer is coloured."""
    from scipy.spatial import cKDTree

    lines = fibre_source(session, tubes)
    if lines is None:
        return None
    if "fiber" in tubes.point_data and "fiber" in lines.point_data:
        shown = np.unique(np.asarray(tubes.point_data["fiber"]))
        owner = np.asarray(lines.point_data["fiber"])
        keep = np.isin(owner, shown)
        if not keep.all():
            lines = _rebuild(lines, [ids for ids in polyline_ids(lines) if len(ids) and keep[ids[0]]])
    colours = colours_on_points(actor, tubes)
    if colours.ndim == 2:
        colours = colours[cKDTree(np.asarray(tubes.points)).query(np.asarray(lines.points), k=1)[1]]
    return lines, colours


def fibre_source(session, tubes):
    """The loaded streamlines the drawn ``tubes`` were swept from, or None: a
    cheap lookup, for deciding whether ``fibre_streamlines`` is worth calling."""
    cache = getattr(session, "_fibers", {}) or {}
    points_id = tubes.GetPoints().GetAddressAsString("vtkPoints") if tubes.GetPoints() else None
    for entry in list(cache.values()):
        drawn = entry.get("tubes") if isinstance(entry, dict) else None
        if drawn is not None and drawn.GetPoints() is not None and drawn.GetPoints().GetAddressAsString("vtkPoints") == points_id:
            lines = entry.get("lines")
            return lines if lines is not None and lines.n_points else None
    return None


def polyline_ids(poly):
    """The point ids of each polyline of ``poly``."""
    lines = np.asarray(poly.lines)
    out, i = [], 0
    while i < len(lines):
        n = int(lines[i])
        out.append(lines[i + 1:i + 1 + n])
        i += n + 1
    return out


def _rebuild(poly, kept):
    """A line PolyData of only the ``kept`` polylines (index arrays)."""
    import pyvista as pv

    if not kept:
        return pv.PolyData()
    used = np.concatenate(kept)
    remap = np.full(poly.n_points, -1, np.int64)
    remap[used] = np.arange(len(used))
    cells = np.concatenate([np.r_[len(ids), remap[ids]] for ids in kept])
    return pv.PolyData(np.asarray(poly.points)[used], lines=cells)

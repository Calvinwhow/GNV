"""Layered brain figures: stack anatomy, painted data, parcels and fibres.

A figure here is a list of layers drawn into one or more camera views.  Each
layer knows only what it is and how it should look; the scene owns the camera,
the lights and the background.  That is the whole model::

    from gnv.render.brain_viewer import BrainScene

    (BrainScene()
        .surface("pial_wholebrain", nifti=tmap, threshold=3.0)
        .render("cortex.png"))

Layer types
-----------
``surface(...)``  a whole-brain or cortical mesh.  Plain by default, or painted
                  by passing ``nifti=`` (continuous, sampled per vertex), or
                  turned into a see-through shell with ``glass=True``.
``parcels(...)``  a directory of regional meshes, each flooded with one number
                  -- from a NIfTI via ``stat=``, or from a ``values=`` mapping
                  you computed elsewhere.
``fibers(...)``   streamlines from a Lead-DBS ``.mat`` / ``.fib.npy``, solid,
                  direction-coloured, or scaled by their own values.

Any layer takes ``color`` and ``opacity``, so "just show me this region in
#9a8ed1 at 60%" needs no NIfTI and no colormap at all.

Hemispheres
-----------
A surface built from ``mesh_pieces`` comes back as ``{'L': ..., 'R': ...}``, and
lateral views drop the near hemisphere so you look through one shell rather
than two.  Pass ``cull_lateral=False`` to ``render`` to keep both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from gnv.render.render_scene import (
    BACKGROUND, BASE_COLOR, CENTER, DISTANCE, EDGE_COLOR, EDGE_OPACITY,
    GLASS_ALPHA, GLASS_COLOR, RED, VIEWS, add_lights, resolve_scale,
    set_camera, surface_material,
)

__all__ = ["BrainScene"]


def _key_lights(scene_lighting, override):
    """Only the rig's own arguments; the material terms are applied elsewhere."""
    merged = dict(scene_lighting, **(override or {}))
    return {k: v for k, v in merged.items()
            if k in ("key_azimuth", "key_elevation", "key_intensity", "rig")
            and v is not None}


def _as_sides(mesh):
    """Any mesh spec -> ``{'L': ..., 'R': ...}`` or ``{'_': ...}``."""
    if isinstance(mesh, dict):
        return mesh
    if isinstance(mesh, pv.DataSet):
        return {"_": mesh}

    name = str(mesh)
    path = Path(name).expanduser()
    if path.suffix and path.exists():
        return {"_": pv.read(path)}

    from gnv.render.brains import mesh_pieces
    from gnv.render.glass_mesh import build_bmesh

    return build_bmesh(mesh_pieces(name))


class BrainScene:
    def __init__(self, center=CENTER, distance=DISTANCE, background=BACKGROUND,
                 lighting=None, parallel=True):
        self.center, self.distance, self.background = center, distance, background
        # Scene-wide, not per layer: one brain lit two ways in one figure reads
        # as two brains.
        self.lighting = dict(lighting or {})
        self.parallel = parallel
        self.bar_vertical = False
        # PyVista keys scalar-bar actors by TITLE. Two scenes in one figure that
        # both pass title="" therefore share one bar, and every row silently
        # inherits the last row's colours and range. A unique title per scene is
        # what keeps them apart, so every scene gets one whether or not it is
        # ever displayed.
        self.bar_title = f" {id(self) % 100000} "
        self.layers = []

    # -- layers -------------------------------------------------------------
    def surface(self, mesh="pial_wholebrain", nifti=None, color=None, opacity=None,
                faceted=None, edges=None, edge_color=EDGE_COLOR, edge_width=0.6,
                edge_opacity=EDGE_OPACITY, glass=False, sign="both", absolute=False, threshold=None,
                cmap=None, clim=None, ramp_color=RED, order=1, label=None):
        """Anatomy, optionally painted with a volume sampled at each vertex."""
        sides = _as_sides(mesh)
        scalars = None
        if nifti is not None:
            from gnv.render.mesh_paint import paint_vertices

            sides = {k: paint_vertices(v, nifti, sign=sign, absolute=absolute,
                                       threshold=threshold, order=order)
                     for k, v in sides.items()}
            scalars = "value"
            cmap, clim = resolve_scale(
                np.concatenate([np.asarray(v["value"]) for v in sides.values()]),
                cmap=cmap, clim=clim, color=ramp_color,
            )

        self.layers.append(dict(
            kind="surface", sides=sides, scalars=scalars, cmap=cmap, clim=clim,
            color=color or (GLASS_COLOR if glass else BASE_COLOR),
            opacity=GLASS_ALPHA if (glass and opacity is None) else (
                1.0 if opacity is None else opacity),
            # A continuous per-vertex gradient is the one thing that bands
            # badly under flat shading and is cluttered by a wireframe; every
            # other surface gets the tessellated treatment.
            faceted=(nifti is None) if faceted is None else faceted,
            edges=(nifti is None and not glass) if edges is None else bool(edges),
            edge_color=edge_color, edge_width=edge_width,
            edge_opacity=edge_opacity,
            # Hiding the near hemisphere is what stops two translucent shells
            # stacking in a lateral view. Doing it to an OPAQUE surface instead
            # turns a lateral view into a medial one, which is not what anyone
            # asked for, so an opaque layer keeps both halves.
            cullable=set(sides) == {"L", "R"} and (
                glass or (opacity is not None and opacity < 1.0)),
            label=label,
        ))
        return self

    def parcels(self, mesh="aal3_suit_parcels", nifti=None, values=None, stat="max",
                sign="both", absolute=False, threshold=None, min_value=None,
                keep_top=None, step=1.0, color=None, opacity=1.0, faceted=True,
                edges=True, edge_color=EDGE_COLOR, edge_width=0.6,
                edge_opacity=EDGE_OPACITY, tessellate=0.0,
                cmap=None, clim=None, ramp_color=RED, label=None):
        """Regional meshes, each flooded with one number (or one flat colour)."""
        from gnv.render.mesh_paint import (
            coarsen, load_parcels, paint_parcels,
        )

        # Decimating before anything is merged or sampled keeps the statistic on
        # the geometry that is actually drawn.
        mesh = coarsen(load_parcels(mesh), tessellate) if tessellate else mesh

        if nifti is None and values is None:
            merged = None
            for m in load_parcels(mesh).values():
                merged = m.copy() if merged is None else merged.merge(m)
            self.layers.append(dict(
                kind="surface", sides={"_": merged}, scalars=None, cmap=None,
                clim=None, color=color or BASE_COLOR, opacity=opacity,
                faceted=faceted, edges=edges, edge_color=edge_color,
                edge_width=edge_width, edge_opacity=edge_opacity,
                cullable=False, label=label))
            return self

        merged = paint_parcels(mesh, nifti=nifti, values=values, stat=stat,
                               sign=sign, absolute=absolute, threshold=threshold,
                               min_value=min_value, keep_top=keep_top, step=step)
        cmap, clim = resolve_scale(np.asarray(merged["value"]), cmap=cmap,
                                   clim=clim, color=ramp_color)
        self.layers.append(dict(
            kind="surface", sides={"_": merged}, scalars="value", cmap=cmap,
            clim=clim, color=color or BASE_COLOR, opacity=opacity,
            faceted=faceted, edges=edges, edge_color=edge_color,
            edge_width=edge_width, edge_opacity=edge_opacity,
            cullable=False, label=label))
        return self

    def fibers(self, source, color=RED, coloring="solid", width=1.5, opacity=1.0,
               cmap=None, clim=None, label=None, **load_kw):
        """Streamlines, from a path or an already-loaded PolyData."""
        from gnv.render.fiber_render import load_fibers

        poly = source if isinstance(source, pv.DataSet) else load_fibers(source, **load_kw)
        if coloring == "value":
            cmap, clim = resolve_scale(np.asarray(poly["vals"]), cmap=cmap,
                                       clim=clim, color=color)
        self.layers.append(dict(kind="fibers", poly=poly, color=color,
                                coloring=coloring, width=width, opacity=opacity,
                                cmap=cmap, clim=clim, label=label))
        return self

    # -- drawing ------------------------------------------------------------
    def _draw(self, pl, view, cull_lateral, scalar_bar):
        cull = VIEWS[view]["cull"] if cull_lateral else None
        painted_mapper = None
        for layer in self.layers:
            if layer["kind"] == "fibers":
                self._draw_fibers(pl, layer)
                continue
            for side, mesh in layer["sides"].items():
                if layer["cullable"] and side == cull:
                    continue
                actor = pl.add_mesh(
                    mesh,
                    scalars=layer["scalars"], cmap=layer["cmap"], clim=layer["clim"],
                    color=None if layer["scalars"] else layer["color"],
                    nan_color=BASE_COLOR, nan_opacity=1.0,
                    opacity=layer["opacity"],
                    smooth_shading=not layer["faceted"],
                    show_edges=layer["edges"], edge_color=layer["edge_color"],
                    line_width=layer["edge_width"],
                    edge_opacity=layer.get("edge_opacity", EDGE_OPACITY),
                    culling="back" if layer["opacity"] < 1.0 else None,
                    show_scalar_bar=False, **surface_material(
                        **{k: self.lighting.get(k) for k in
                           ("specular", "specular_power", "ambient", "diffuse")}),
                )
                if layer["scalars"] and painted_mapper is None:
                    painted_mapper = actor.mapper
        # The bar must be bound to the painted SURFACE's mapper. Left to pick
        # for itself, add_scalar_bar takes the most recently added mapper, which
        # in a layered scene is usually the fibres -- so the figure ends up
        # captioned with the streamline values instead of the map's.
        if scalar_bar and painted_mapper is not None:
            bar = dict(mapper=painted_mapper, title=self.bar_title,
                       color="#33373d", label_font_size=13, title_font_size=13,
                       n_labels=4, fmt="%.3g")
            if self.bar_vertical:
                pl.add_scalar_bar(vertical=True, width=0.035, height=0.62,
                                  position_x=0.90, position_y=0.19, **bar)
            else:
                pl.add_scalar_bar(vertical=False, width=0.44, height=0.05,
                                  position_x=0.28, position_y=0.04, **bar)

    @staticmethod
    def _draw_fibers(pl, layer):
        kw = dict(line_width=layer["width"], render_lines_as_tubes=True,
                  opacity=layer["opacity"], lighting=True, specular=0.4,
                  specular_power=20, ambient=0.25, show_scalar_bar=False)
        if layer["coloring"] == "direction":
            pl.add_mesh(layer["poly"], scalars="dir", rgb=True, **kw)
        elif layer["coloring"] == "value":
            pl.add_mesh(layer["poly"], scalars="vals", cmap=layer["cmap"],
                        clim=layer["clim"], **kw)
        else:
            pl.add_mesh(layer["poly"], color=layer["color"], **kw)

    @staticmethod
    def grid(scenes, out, views=("left", "posterior", "inferior"), row_labels=None,
             size=None, zoom=1.25, cull_lateral=True, labels=True,
             scalar_bar=True, lighting=None, title=None):
        """One figure, one scene per row, the same views down every column.

        This is the contrast-comparison layout: three maps of the same brain
        read far better stacked than as three separate files, because the eye
        compares rows directly and each row can keep its own colour and its own
        scale bar.
        """
        scenes = list(scenes)
        views = (views,) if isinstance(views, str) else tuple(views)
        shape = (len(scenes), len(views))
        size = size or (760 * shape[1], 640 * shape[0])

        pl = pv.Plotter(off_screen=True, shape=shape, window_size=size, border=False)
        for r, scene in enumerate(scenes):
            for c, view in enumerate(views):
                pl.subplot(r, c)
                pl.set_background(scene.background)
                scene.bar_vertical = True
                if row_labels and scene.bar_title.strip().isdigit():
                    scene.bar_title = f"{row_labels[r]} "
                scene._draw(pl, view, cull_lateral,
                            scalar_bar and c == len(views) - 1)
                add_lights(pl, center=scene.center,
                           **_key_lights(scene.lighting, lighting))
                pl.camera.parallel_projection = scene.parallel
                set_camera(pl, view, zoom=zoom, center=scene.center,
                           distance=scene.distance)
                if labels and r == 0:
                    pl.add_text(VIEWS[view]["label"], font_size=12,
                                color="#33373d", position="upper_edge")
                if row_labels and c == 0:
                    pl.add_text(str(row_labels[r]), font_size=12, color="#33373d",
                                position="upper_left")
        pl.enable_anti_aliasing("ssaa")
        pl.screenshot(str(out))
        pl.close()
        return out

    def render(self, out, views=("left", "right", "anterior", "superior"),
               shape=None, size=None, zoom=1.25, cull_lateral=True, labels=True,
               scalar_bar=False, lighting=None):
        """Write a figure. One view gives a single large panel."""
        views = (views,) if isinstance(views, str) else tuple(views)
        if shape is None:
            shape = (1, 1) if len(views) == 1 else ((1, len(views)) if len(views) < 4
                                                    else (2, (len(views) + 1) // 2))
        if size is None:
            size = (1700, 1400) if len(views) == 1 else (900 * shape[1], 750 * shape[0])

        pl = pv.Plotter(off_screen=True, shape=shape, window_size=size, border=False)
        for k, view in enumerate(views):
            pl.subplot(k // shape[1], k % shape[1])
            pl.set_background(self.background)
            self._draw(pl, view, cull_lateral, scalar_bar and k == len(views) - 1)
            add_lights(pl, center=self.center, **_key_lights(self.lighting, lighting))
            pl.camera.parallel_projection = self.parallel
            set_camera(pl, view, zoom=zoom, center=self.center, distance=self.distance)
            if labels and len(views) > 1:
                pl.add_text(VIEWS[view]["label"], font_size=11, color="#33373d",
                            position="upper_left")
        pl.enable_anti_aliasing("ssaa")
        pl.screenshot(str(out))
        pl.close()
        return out

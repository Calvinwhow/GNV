"""The declarative description of a figure: layers, camera, background.

A figure is a list of LAYERS and a CAMERA, and nothing else.  Everything the
app can do is a change to this structure, and the structure round-trips to JSON,
so a figure you built by clicking can be committed beside a manuscript, diffed
when a reviewer asks what changed, hand-edited, and loaded back.

Two kinds of layer, because there are two kinds of thing to draw:

``mesh``    one mesh or a group of them.  A group is the unit of styling: every
            mesh in it shares a palette and an opacity.  With no volume it is
            flat anatomy in one colour.  With a volume it is painted, and
            ``mode`` picks which question the painting answers --

              ``vertex``  sample the volume where the surface actually is, giving
                          a continuous gradient across each mesh.
              ``region``  reduce the volume inside each mesh to one number and
                          flood that mesh with it.

            On a single whole-brain mesh ``vertex`` is the useful one; on a group
            of 165 parcels ``region`` is.  Both work on both, deliberately: a
            group of parcels painted per vertex is a legitimate figure, and so is
            a single structure flooded with its own peak.

``fibers``  streamlines.  They are their own group because they do not hang off a
            mesh -- geometry comes from the tractogram, so they need no surface to
            live on and are styled entirely independently.

Nothing here imports a renderer.  A spec is data; ``session`` turns it into
pixels.  That separation is what lets the app validate and save a figure whose
files are missing, or on a machine with no display.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

__all__ = [
    "SPEC_VERSION", "Spec", "DEFAULT_LIGHTING", "new_spec", "new_mesh_layer",
    "new_fiber_layer", "new_slice_layer", "new_electrode_layer",
    "new_overlay", "OVERLAY",
    "normalize_spec", "load_spec", "save_spec", "camera_direction", "VIEW_PRESETS",
    "new_volume_layer", "VOLUME_LAYER",
    "STATS", "SIGNS", "MODES", "DRAWS", "SPLITS", "FIBER_COLORINGS",
]

SPEC_VERSION = 1

STATS = ("max", "mean_nonzero", "mean", "min", "sum")
SIGNS = ("both", "positive", "negative")
MODES = ("region", "vertex")
DRAWS = ("fill", "outline")
REGION_COLORS = (None, "distinct", "atlas")
SPLITS = ("auto", "single", "labels", "components", "volumes")
FIBER_COLORINGS = ("solid", "direction", "value", "tract")
OPACITY_PROFILES = ("linear", "sigmoid", "geom", "foreground")

# Azimuth is measured in the axial plane from anterior, turning towards the LEFT
# side of the brain; elevation lifts the camera towards the vertex. Storing the
# camera this way rather than as a position means an orbit is two numbers that
# survive a save/load, and that a named view is just a preset of those numbers.
VIEW_PRESETS = {
    "anterior":  (0.0, 0.0),
    "left":      (90.0, 0.0),
    "posterior": (180.0, 0.0),
    "right":     (270.0, 0.0),
    "superior":  (0.0, 89.9),
    "inferior":  (0.0, -89.9),
}

DEFAULT_CAMERA = {"azimuth": 90.0, "elevation": 0.0, "zoom": 1.0}
# A three-point rig with the key above and right of frame, which in a lateral
# view lights the parietal and shadows the frontal pole. Scene-wide, because one
# brain lit two ways in one figure reads as two brains.
DEFAULT_LIGHTING = {
    "rig": "studio", "key_azimuth": 35.0, "key_elevation": 35.0,
    "key_intensity": 0.85,
    "specular": 0.22, "specular_power": 14.0, "ambient": 0.30, "diffuse": 0.78,
    "occlusion": False,      # ambient occlusion: shade creases where surfaces meet (the "studio" look)
}

DEFAULT_CENTER = [0.0, -17.0, 5.0]
DEFAULT_DISTANCE = 460.0

MESH_LAYER = {
    "kind": "mesh",
    "name": "Mesh",
    "visible": True,
    "mesh": "pial_wholebrain",   # a registered name, a directory, or a NIfTI
    "frame": 0,                  # which frame of a 4D NIfTI is shown (see split)
    "iso": 0.0,                  # NIfTI source: "inside" is value > iso
    "split": "auto",             # auto | single | labels | components | volumes (4D)
    "presmooth_vox": 1.2,        # NIfTI source: surface smoothing, in voxels
    "color": "#b7bcc4",
    "opacity": 1.0,
    "faceted": None,          # None -> flat, except a vertex-sampled gradient
    "edges": None,            # None -> on for a region set, off for a surface
    "tessellate": 0.7,        # decimation before sampling; the Smoothness slider
    "edge_color": "#ffffff",
    "edge_width": 0.6,
    "edge_opacity": 0.10,     # independent of the mesh's own opacity
    # Paint the wireframe with the colouring maps too: each edge takes the
    # map's colour where the map survives and the edge colour elsewhere (or
    # nothing, with "edges" off). edge_opacity scales the whole wireframe. With
    # the mesh's own opacity at 0 this leaves a see-through cage whose map
    # region still reads -- so a slice inside the brain stays visible.
    "edge_paint": False,
    # Which halves of the group to draw at all. Explicit, because hiding a
    # hemisphere is a decision about the figure, not about the camera.
    "show_left": True,
    "show_right": True,
    "hidden_regions": [],     # names inside the group that are not drawn
    # Cut the group in world millimetres. Each entry removes everything on one
    # side of a plane -- {"axis":"x","op":">","at":-3.0} is `vol(X > -3) = 0`.
    "section": [],
    # Cap the cut so it is a cross-section rather than an open shell.
    # Only meaningful with vertex colouring, which samples the cap.
    "section_cap": False,
    # Drop the NEAR hemisphere automatically in a lateral view. Useful for a
    # translucent shell, surprising everywhere else, so it is off unless asked.
    "cull_near_hemisphere": False,
    "overlays": [],           # volumes colouring this group, in order
    # Region sets only. None paints every region `color`; "distinct" gives each
    # region its own colour from a qualitative palette; "atlas" uses the colours
    # the atlas's label table carries (falls back to distinct where it has none).
    # Fill overlays paint over whichever base this gives.
    "region_colors": None,
}

FIBER_LAYER = {
    "kind": "fibers",
    "name": "Fibers",
    "visible": True,
    # A fibre file, or the name of a fibre set in the Fiber set list.
    "source": None,
    "hidden_regions": [],     # tracts of a fibre set that are not drawn
    "frame": 0,               # which frame of a fibre set's fibres x time values is shown
    "fiber_atlas_path": None,
    "sign": "both",
    "min_abs_value": None,
    "max_abs_value": None,    # p-value maps want an upper bound, not a lower one
    "top_percent": None,
    "max_lines": 20000,
    "step": 2,
    "smooth": 3,             # binomial passes over the polyline, ends pinned
    # Real swept geometry instead of the line-width shading trick. A property
    # of the layer, so the view and the export are the same scene -- the Hi-res
    # button sets this, and an export then differs only in pixels.
    "tubes": False,
    "tube_sides": 8,         # sides of the swept tube
    "tube_radius": 0.2,      # thickness in mm, for preview and hi-res alike
    "coloring": "solid",
    "color": "#c15656",
    "width": 1.5,            # legacy pixel line width; read only for old specs
    "opacity": 1.0,
    "palette": None,
    "clim": None,
    # Drop atlas fibres that no overlay paints. A whole-atlas tractogram is
    # mostly fibres a given result says nothing about, and drawing them buries
    # the ones it does. On by default: an atlas has no values of its own, so its
    # fibres get no colour unless an overlay's threshold window shows them.
    "values_only": True,
    # Waypoint gates, ANDed. Each is a plane in world millimetres that a fibre
    # must cross ("through") or must not ("avoid"), which is how a bundle is
    # isolated from a whole-brain tractogram to show its course.
    "prune": [],
    # Findings painted onto these streamlines, in order. Each is a file of one
    # value per atlas fibre, so a bundle can carry a background colouring that
    # later entries overwrite where they have something to say.
    "overlays": [],
}

# One volume painted onto whatever a layer is made of. A layer holds an ordered
# stack of these, so "the motor network in red and the cognitive one in blue,
# over anatomy" is three entries rather than three layers fighting over the same
# geometry. Later entries paint over earlier ones where both survive threshold.
OVERLAY = {
    "nifti": None,
    "name": "",
    "color": "#c15656",      # a ramp is built from this when palette is None
    "palette": None,
    "sign": "both",
    "absolute": False,
    "threshold": None,       # |v| below this stays transparent
    "max_value": None,       # |v| above this is DROPPED, not saturated
    # None lets the data decide (two-signed -> diverging about zero); True and
    # False force it, because "this map happens to have one negative voxel" is a
    # property of the data, not a decision about the figure.
    "symmetric": None,
    # The colour scale is a separate pair from the value cutoffs above.
    # clim[0] is where colour starts, clim[1] is saturation: everything
    # over it takes the top colour rather than being dropped.
    "clim": None,            # [start, saturate]; None reads them off the data
    # Off: everything below the scale's low end takes the bottom colour, which
    # is the normal saturating behaviour. On: it contributes nothing at all, so
    # a second volume shows through wherever this one has nothing to say. That
    # is what lets a stack of findings intercalate rather than the topmost one
    # laying a flat wash over everything beneath it.
    "hide_below_min": False,
    # A colour bar belongs to the MAP, not to the scene: it captions one
    # overlay's numbers, so it is toggled on that overlay and several can be
    # shown side by side. A scene-wide switch could only ever caption one of
    # them, which is how a figure ends up labelled with the wrong scale.
    "bar": False,
    # Vertex sampling asks the volume for a value at one exact point, so a
    # vertex landing in a zero voxel gets nothing and the surface speckles.
    # This spreads values from painted vertices into their unpainted neighbours
    # across the mesh, so a connected patch is coloured throughout.
    "fill_gaps": 0,
    # Grow the map's coloured regions outward by this many millimetres: each
    # voxel it leaves uncoloured within reach takes the value of the NEAREST
    # coloured voxel (copied, never averaged), so a map that stops just short of
    # the cortex, a slice rim or a nucleus edge covers it. Applies to every
    # geometry the map is drawn on, because the map itself is grown. 0 = off.
    "extend_mm": 0.0,
    # Which frame of a 4D map (or of a fibres x time value array) is shown.
    "frame": 0,
    # Volume layers only. X-RAY is a property of the map, not of the brain:
    # 0 paints this map on the tissue the rays reach (the cortex, a cut face);
    # above 0 it is seen THROUGH the brain as well, tinting the surface above
    # it by exp(-depth / xray_mm) -- at this depth a blob tints at 37%. The
    # brain itself is never made see-through by it; its own opacity does that.
    "xray_mm": 0.0,
    # X-rayed maps: the fraction of the colour scale over which a value's
    # strength rises from the threshold, so a blob's edge does not terrace.
    "fade": 0.03,
    # Vertex-sampled surfaces only: average each vertex's samples over the
    # surface around it, about this many millimetres along the mesh, before
    # sign and threshold decide anything. A 2 mm map sampled at single points
    # on a folded cortex stripes -- banks inside the map, crowns and fundi just
    # outside it -- and this closes those stripes. 0 turns it off.
    "smooth_mm": 2.0,
    "opacity": 1.0,
    # meshes only -- a slice samples where it is, a region has to reduce
    "mode": "region",
    "stat": "max",
    "min_value": None,
    "keep_top": None,
    "step": 1.0,
    # "fill" paints the surface. "outline" paints nothing: it draws the edge of
    # whatever survives sign / threshold / max as lines in `color`, so a full
    # t map can colour the surface while its significant part is only traced.
    # Mesh layers only.
    "draw": "fill",
    # OUTLINE: trace the edge of whatever this overlay selects (its nonzero
    # voxels, for a plain map) in its own colour and opacity, on meshes, slices
    # and volume bodies alike -- with the fill, or instead of it (draw:
    # "outline"). Like a mesh's edge highlight: a line, not a colour change.
    "outline": False,
    "outline_color": "#ffffff",
    "outline_opacity": 1.0,
    "outline_width": 2.0,     # line width, pixels
    "outline_smooth": 1.0,    # Gaussian sigma in voxels on the mask before tracing
}


# A DBS lead. Its own kind rather than a mesh layer, because it is the one
# thing here made of two materials that must not share a colour: the shaft is
# pale plastic and the contacts are dark metal, and that contrast is what makes
# a lead legible against a brain.
ELECTRODE_LAYER = {
    "kind": "electrode",
    "name": "Electrode",
    "visible": True,
    "source": None,           # ea_reconstruction.mat, a model .mat, or JSON; None for a model alone
    # A model's name in resources/electrodes ("boston_vercise_directed") or a
    # model .mat path: with a source, it replaces the model the reconstruction
    # names; without one, it is the lead drawn.
    "model": None,
    "insulation_color": "#d4d6da",
    "contact_color": "#43484f",
    "opacity": 1.0,
    "hidden_regions": [],     # part names, so one contact can be dropped
    # Per-lead trajectory, {side: {tip: [x, y, z], sagittal, coronal, marker}} in
    # world mm and degrees, each angle on the unit circle of its plane (see
    # electrode_render.pose_markers). A lead not listed sits where its file puts
    # it; "R"/"L" for a reconstruction, "" for a bare model.
    "poses": {},
    # Mirror across the midline (x = 0): the side the other lead follows -- "R"
    # or "L" for a reconstruction, "" to give a bare model a mirrored copy.
    # None for off. See electrode_render.mirror_pose.
    "mirror": None,
}


# A NIfTI drawn as a BODY, by ray casting, rather than as a surface or as three
# plates. Its own kind, because loading a body and cutting through one belong to
# it, not to the slice layer: a slice plane photographs the data at one
# coordinate, while a cut here removes part of the body so you can see inside.
# The body is always rendered from its own scalars; overlays only colour it.
VOLUME_LAYER = {
    "kind": "volume",
    "name": "Volume",
    "visible": True,
    "source": "spm152",        # a packaged name, or a NIfTI path
    "frame": 0,                # which frame of a 4D NIfTI is shown
    # -- the body ----------------------------------------------------------
    "cmap": "gray",            # matplotlib name or MRIcroGL .clut
    "color": None,             # a hex tint; overrides cmap when set
    "clim": None,              # [start, saturate]; None reads it off the data
    "opacity_profile": "linear",
    "opacity": 1.0,            # scales the body's opacity transfer function
    "shade": True,             # gradient shading from the scene's lighting rig
    # Gaussian on the body, in its own voxels. Unsmoothed, the lighting on a
    # body shows its voxel grid as a lattice of small dimples ("pocking") --
    # on an 8-bit 1 mm template as much as a label map -- so the default is a
    # light 0.75; about 1 for a label / segmentation map (CAT12 p0, aseg),
    # whose class staircase otherwise renders as crinkled partial-volume noise.
    "body_smooth_vox": 0.75,
    # A label atlas as the body: only these label values are drawn (SUIT's
    # 29 is the left dentate). Empty draws the whole volume.
    "body_labels": [],
    # The lookup table that names those labels, when it is not a file beside
    # the atlas named like it (see volume_render._lut_names). Names only.
    "body_lut": None,
    # -- how overlays colour it ------------------------------------------
    # Overlays only ever change the body's colour; its shape, opacity and
    # lighting stay its own. Whether a map is painted on the surface or x-rayed
    # through the brain is set on each overlay (``xray_mm``).
    "overlay_smooth_vox": 0.75,  # every map, painted or x-rayed: Gaussian, its voxels
    # Overlays are drawn only inside the body unless this is on: a map that
    # runs past the brain would otherwise show as colour in the air around it.
    "overlay_outside": False,
    # Cut the body in world millimetres, same grammar as a mesh layer. Rendered
    # as clipping planes with a solid cut face.
    "section": [],
    "overlays": [],
}


SLICE_LAYER = {
    "kind": "slices",
    "name": "Slices",
    "visible": True,
    "backdrop": None,          # a library name, a .npz path, or a NIfTI (temporary, in memory)
    "x": None, "y": None, "z": None,      # world mm; None -> the middle
    "show_x": True, "show_y": True, "show_z": True,
    "opacity": 1.0,
    "cmap": None,              # None -> grey, which is what anatomy wants
    "transparent_below": 8,    # background drops out so a slice sits INSIDE a brain
    # Painting the dropped-out part in the scene background instead of
    # leaving it clear makes the slice read as a solid plate. It is an
    # illusion -- the data is identical -- but it hides whatever sits
    # behind the slice, which is usually what makes one look cheap.
    "fill_background": False,
    "smooth": True,            # interpolate between voxels, as MRIcroGL does; off shows each voxel square
    "overlays": [],            # volumes composited onto the slice, in order
}

Spec = dict


def _new_id():
    return uuid.uuid4().hex[:8]


def new_overlay(**overrides):
    layer = dict(OVERLAY, id=_new_id())
    layer.update(overrides)
    return layer


def new_mesh_layer(**overrides):
    layer = dict(MESH_LAYER, id=_new_id())
    layer.update(overrides)
    return layer


def new_electrode_layer(**overrides):
    layer = dict(ELECTRODE_LAYER, id=_new_id())
    layer.update(overrides)
    return layer


def new_fiber_layer(**overrides):
    layer = dict(FIBER_LAYER, id=_new_id())
    layer.update(overrides)
    return layer


def new_volume_layer(**overrides):
    layer = dict(VOLUME_LAYER, id=_new_id())
    layer.update(overrides)
    return layer


def new_slice_layer(**overrides):
    layer = dict(SLICE_LAYER, id=_new_id())
    layer.update(overrides)
    return layer


def new_spec(layers=None):
    return {
        "version": SPEC_VERSION,
        "background": "#ffffff",
        "center": list(DEFAULT_CENTER),
        "distance": DEFAULT_DISTANCE,
        "camera": dict(DEFAULT_CAMERA),
        "lighting": dict(DEFAULT_LIGHTING),
        "parallel": True,
        "pan": [0.0, 0.0, 0.0],
        "layers": list(layers) if layers else [new_mesh_layer(name="Anatomy")],
    }


# -- validation -------------------------------------------------------------
class SpecError(ValueError):
    pass


def _check(condition, message):
    if not condition:
        raise SpecError(message)


def _pose(pose):
    """A lead's trajectory with numbers where numbers belong, or None when it
    lacks a tip or one of its three angles: such a lead stays where its file
    puts it rather than be aimed by angles nobody chose."""
    if not all(k in pose for k in ("tip", "sagittal", "coronal", "marker")):
        return None
    tip = [float(v) for v in (list(pose["tip"]) + [0, 0, 0])[:3]]
    return {"tip": tip, "sagittal": float(pose["sagittal"]), "coronal": float(pose["coronal"]), "marker": float(pose["marker"]) % 360.0}


def normalize_spec(spec):
    """Fill in defaults and reject anything the renderer could not draw.

    Validation happens here rather than at render time so the app can refuse a
    bad edit with a message about the field, instead of surfacing a VTK error
    from six frames down.
    """
    _check(isinstance(spec, dict), "a spec must be a JSON object")
    out = {
        "version": SPEC_VERSION,
        "background": spec.get("background", "#ffffff"),
        "center": list(spec.get("center") or DEFAULT_CENTER),
        "distance": float(spec.get("distance") or DEFAULT_DISTANCE),
        "camera": dict(DEFAULT_CAMERA, **(spec.get("camera") or {})),
        "lighting": dict(DEFAULT_LIGHTING, **(spec.get("lighting") or {})),
        "parallel": bool(spec.get("parallel", True)),
        # Camera pan in world millimetres, on top of the automatic framing.
        "pan": [float(v) for v in (list(spec.get("pan") or [0, 0, 0]) + [0, 0, 0])[:3]],
        # Long edge of an exported PNG, in pixels. Export used to be pinned
        # at 1700 wide whatever the Hi-res button said, which is where the
        # pixellation on a zoomed-in figure came from.
        "export_px": int(spec.get("export_px") or 4000),
        "layers": [],
    }
    _check(len(out["center"]) == 3, "center must be three numbers")

    seen = set()
    for raw in spec.get("layers") or []:
        _check(isinstance(raw, dict), "every layer must be an object")
        kind = raw.get("kind", "mesh")
        _check(kind in ("mesh", "fibers", "slices", "electrode", "volume"),
               f"unknown layer kind {kind!r}")
        base = {"mesh": MESH_LAYER, "fibers": FIBER_LAYER,
                "slices": SLICE_LAYER, "electrode": ELECTRODE_LAYER,
                "volume": VOLUME_LAYER}[kind]
        layer = dict(base, id=raw.get("id") or _new_id())
        layer.update({k: v for k, v in raw.items() if k in base or k == "id"})

        while layer["id"] in seen:
            layer["id"] = _new_id()
        seen.add(layer["id"])

        if kind in ("mesh", "slices", "fibers", "volume"):
            layer["overlays"] = _overlays(raw, layer, _legacy_xray(raw, kind))

        if kind == "electrode":
            layer["hidden_regions"] = [str(r) for r in
                                       (layer.get("hidden_regions") or [])]
            layer["opacity"] = min(max(float(layer["opacity"]), 0.0), 1.0)
            layer["poses"] = {str(side): _pose(pose) for side, pose in (layer.get("poses") or {}).items() if pose and _pose(pose)}
            layer["mirror"] = None if layer.get("mirror") is None else str(layer["mirror"])

        if kind == "volume":
            _check(layer["opacity_profile"] in OPACITY_PROFILES,
                   f"opacity_profile must be one of {OPACITY_PROFILES}")
            layer["section"] = _section(layer.get("section"))
            layer["source"] = str(layer.get("source") or "") or None
            layer["color"] = str(layer["color"]) if layer.get("color") else None
            layer["opacity"] = min(max(float(layer["opacity"]), 0.0), 1.0)
            layer["overlay_smooth_vox"] = max(float(layer.get("overlay_smooth_vox") or 0.0), 0.0)
            layer["body_smooth_vox"] = min(max(float(layer.get("body_smooth_vox") or 0.0), 0.0), 5.0)
            layer["body_lut"] = str(layer["body_lut"]) if layer.get("body_lut") else None
            try:
                layer["body_labels"] = sorted({int(v) for v in (layer.get("body_labels") or [])})
            except (TypeError, ValueError):
                raise SpecError("body_labels must be whole numbers")
            layer["overlay_outside"] = bool(layer.get("overlay_outside"))

        if kind == "mesh":
            _check(layer["split"] in SPLITS, f"split must be one of {SPLITS}")
            layer["hidden_regions"] = [str(r) for r in
                                       (layer.get("hidden_regions") or [])]
            layer["section"] = _section(layer.get("section"))
            _check(layer.get("region_colors") in REGION_COLORS,
                   f"region_colors must be one of {REGION_COLORS}")
            mesh = layer["mesh"]
            if isinstance(mesh, (list, tuple)):
                layer["mesh"] = [str(m) for m in mesh if str(m).strip()]
                _check(layer["mesh"], "a mesh layer needs at least one volume")
            else:
                # "" is a layer that has not been given geometry yet: it draws
                # nothing and its panel opens on the library picker.
                layer["mesh"] = str(mesh or "")

            layer["opacity"] = min(max(float(layer["opacity"]), 0.0), 1.0)
        elif kind == "fibers":
            layer["hidden_regions"] = [str(r) for r in (layer.get("hidden_regions") or [])]
            _check(layer["coloring"] in FIBER_COLORINGS,
                   f"coloring must be one of {FIBER_COLORINGS}")
            _check(layer["sign"] in SIGNS, f"sign must be one of {SIGNS}")
            layer["opacity"] = min(max(float(layer["opacity"]), 0.0), 1.0)
        else:
            layer["opacity"] = min(max(float(layer["opacity"]), 0.0), 1.0)

        if "frame" in base:
            layer["frame"] = max(int(layer.get("frame") or 0), 0)

        for key in ("threshold", "min_value", "clim", "min_abs_value",
                    "max_abs_value", "top_percent"):
            if layer.get(key) in ("", []):
                layer[key] = None
        out["layers"].append(layer)

    return out


def _section(raw):
    """Validate the cutting conditions. Each is one half-space to remove."""
    out = []
    for entry in raw or []:
        _check(isinstance(entry, dict), "every section condition must be an object")
        axis = str(entry.get("axis", "x")).lower()
        op = str(entry.get("op", ">"))
        _check(axis in ("x", "y", "z"), "section axis must be x, y or z")
        _check(op in (">", "<"), "section op must be > or <")
        try:
            at = float(entry.get("at", 0.0))
        except (TypeError, ValueError):
            raise SpecError("section 'at' must be a number in millimetres")
        out.append({"axis": axis, "op": op, "at": at,
                    "enabled": bool(entry.get("enabled", True))})
    return out


def _legacy_xray(raw, kind):
    """Overlay defaults for a volume layer saved when x-ray was layer-wide.

    Such a layer carried ``overlay_render: "volume"`` with one depth and one
    fade for every map on it. Each of its overlays inherits them unless it
    sets its own, so an old figure renders as it was saved.
    """
    if kind != "volume" or raw.get("overlay_render") != "volume":
        return {}
    out = {"xray_mm": float(raw.get("xray_depth_mm") or 8.0)}
    if raw.get("overlay_fade") is not None:
        out["fade"] = float(raw["overlay_fade"])
    return out


def _overlays(raw, layer, inherited=None):
    """The layer's overlay stack, migrating the older single-volume form.

    Figures saved before a layer could hold more than one volume carry `nifti`
    and its colouring at the top level. Reading those as a one-entry stack keeps
    them loadable, which matters because the whole point of the project file is
    that a figure stays reproducible. ``inherited`` are defaults the layer
    hands down (see ``_legacy_xray``); an overlay's own value wins.
    """
    stack = raw.get("overlays")
    if stack is None and raw.get("nifti"):
        stack = [{k: raw[k] for k in OVERLAY if k in raw}]
        stack[0]["nifti"] = raw["nifti"]
    out = []
    for entry in stack or []:
        _check(isinstance(entry, dict), "every overlay must be an object")
        item = dict(OVERLAY, **(inherited or {}), id=entry.get("id") or _new_id())
        item.update({k: v for k, v in entry.items() if k in OVERLAY or k == "id"})
        if not item["nifti"]:
            continue
        _check(item["mode"] in MODES, f"overlay mode must be one of {MODES}")
        _check(item["stat"] in STATS, f"overlay stat must be one of {STATS}")
        _check(item["sign"] in SIGNS, f"overlay sign must be one of {SIGNS}")
        _check(item["draw"] in DRAWS, f"overlay draw must be one of {DRAWS}")
        # The older outline-only mode traced in the overlay's own colour; keep
        # that look for figures saved with it.
        if item["draw"] == "outline":
            item["outline"] = True
            if "outline_color" not in entry:
                item["outline_color"] = item.get("color") or "#000000"
        item["outline"] = bool(item.get("outline"))
        item["outline_color"] = str(item.get("outline_color") or "#ffffff")
        item["outline_opacity"] = min(max(float(1.0 if item.get("outline_opacity") is None
                                                else item["outline_opacity"]), 0.0), 1.0)
        item["outline_width"] = max(float(item["outline_width"] or 2.0), 0.5)
        item["outline_smooth"] = max(float(item["outline_smooth"] or 0.0), 0.0)
        item["opacity"] = min(max(float(item["opacity"]), 0.0), 1.0)
        item["smooth_mm"] = min(max(float(item.get("smooth_mm") or 0.0), 0.0), 20.0)
        item["frame"] = max(int(item.get("frame") or 0), 0)
        item["xray_mm"] = min(max(float(item.get("xray_mm") or 0.0), 0.0), 300.0)
        item["fade"] = min(max(float(item.get("fade") or 0.0), 0.0), 1.0)
        for key in ("threshold", "max_value", "min_value", "clim"):
            if item.get(key) in ("", []):
                item[key] = None
        # A window with its ends the wrong way round selects nothing, which
        # looks like a broken render rather than an empty selection.
        lo, hi = item.get("threshold"), item.get("max_value")
        if lo is not None and hi is not None and float(hi) < float(lo):
            item["threshold"], item["max_value"] = hi, lo
        if isinstance(item.get("clim"), (list, tuple)):
            pair = [v for v in item["clim"] if v is not None and v != ""]
            item["clim"] = ([float(pair[0]), float(pair[1])]
                            if len(pair) == 2 else None)
        if item.get("symmetric") not in (None, True, False):
            item["symmetric"] = None
        out.append(item)
    return out


def load_spec(path):
    return normalize_spec(json.loads(Path(path).expanduser().read_text()))


def save_spec(spec, path):
    path = Path(path).expanduser()
    path.write_text(json.dumps(normalize_spec(spec), indent=2) + "\n")
    return path


# -- camera -----------------------------------------------------------------
def camera_direction(azimuth, elevation):
    """Unit vector from the brain towards the camera, plus a stable up vector.

    The up vector is world +Z projected perpendicular to the view direction,
    which keeps the brain upright through any orbit. Straight up or straight
    down that projection collapses, so those fall back to +Y -- which is also
    exactly the convention a superior view is normally drawn with (anterior at
    the top of the page), so the preset and the fallback agree.
    """
    import numpy as np

    az, el = np.radians(float(azimuth)), np.radians(float(elevation))
    d = np.array([-np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    d /= np.linalg.norm(d)

    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(d @ reference)) > 0.999:
        reference = np.array([0.0, 1.0, 0.0])
    up = reference - float(d @ reference) * d
    return d, up / np.linalg.norm(up)

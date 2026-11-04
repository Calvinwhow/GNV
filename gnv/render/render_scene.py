"""Camera, lighting and colour for the PyVista brain renderer.

Everything that decides *how a figure looks* -- where the camera sits, how the
brain is lit, how a range of numbers becomes colours -- lives here, so
``mesh_paint`` and ``fiber_render`` only have to decide *what* to draw.

Three things are worth knowing before changing anything:

Cameras are explicit.  ``pl.view_xy()`` and friends flip handedness depending
on their ``negative`` flag, and a mirrored brain is very hard to notice in a
figure.  Every view here is a literal (position, focal point, up) triple.

Lighting is a three-point rig with a movable key, not VTK's light kit.  The kit
fills a surface from five directions against high ambient, so nothing is ever in
shadow and nothing has form -- legible, and cartoonish.  Here one key light does
the modelling, and the fill and rim exist only to keep the dark side readable.
``rig="kit"`` is still available for reproducing a figure that was lit that way.
Every light follows the camera, so the same region is lit the same way in a
lateral view and an inferior one.

Faceted shading is a deliberate option, not an oversight.  ``faceted=True``
turns off normal interpolation, so every triangle shades as a flat plate, and
together with the light wireframe over the top it is what makes a parcellated
brain read as a tessellated model rather than as smooth jelly.  On a continuous
per-vertex gradient the same setting reads as banding, so that one case -- and
only that one -- defaults to smooth, without edges.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap, to_rgb

__all__ = [
    "VIEWS", "CENTER", "DISTANCE", "BACKGROUND", "BASE_COLOR", "EDGE_COLOR",
    "EDGE_OPACITY", "GLASS_COLOR", "GLASS_ALPHA", "SURFACE_MATERIAL",
    "KEY_AZIMUTH", "KEY_ELEVATION", "KEY_INTENSITY", "RIGS", "surface_material",
    "BLUE", "RED", "PURPLE",
    "set_camera", "add_lights", "diverging_cmap", "ramp_cmap", "resolve_scale",
]

# -- palette ----------------------------------------------------------------
# The three hexes are the ones the MRIcroGL LUTs were built from, so a volume
# opened in MRIcroGL and a mesh rendered here carry the same colour.
BLUE = "#5071a0"
RED = "#c15656"
PURPLE = "#9a8ed1"

BACKGROUND = "#ffffff"
BASE_COLOR = "#b7bcc4"      # unpainted, opaque anatomy
EDGE_COLOR = "#ffffff"      # the wireframe drawn over a region mesh
# The wireframe is a TEXTURE, not a line drawing. At full strength it is a white
# grid that flattens every colour under it; at ~0.18 the same lines read as the
# faceting of a solid object and the fill keeps its saturation. This one number
# is the difference between the tessellated look and a highlighter.
EDGE_OPACITY = 0.10

# Diffuse carries the form and ambient only keeps the shadow side readable. The
# kit's ratio is the other way round (0.65 ambient against 0.4 diffuse), which
# is precisely what flattens it.
SURFACE_MATERIAL = dict(ambient=0.30, diffuse=0.78, specular=0.22, specular_power=14)


def surface_material(specular=None, specular_power=None, ambient=None, diffuse=None):
    """``SURFACE_MATERIAL`` with any of its terms overridden."""
    out = dict(SURFACE_MATERIAL)
    for key, value in (("specular", specular), ("specular_power", specular_power),
                       ("ambient", ambient), ("diffuse", diffuse)):
        if value is not None:
            out[key] = float(value)
    return out
GLASS_COLOR = "#8d99ae"     # unpainted, translucent anatomy
GLASS_ALPHA = 0.13

# -- cameras ----------------------------------------------------------------
# `cull` names the hemisphere to drop in that view: in a left lateral view the
# left hemisphere is between the camera and everything interesting, so dropping
# it means looking through one shell instead of two stacked ones.
VIEWS = {
    "left":      dict(dir=(-1, 0, 0), up=(0, 0, 1), cull="L", label="Left lateral"),
    "right":     dict(dir=(1, 0, 0),  up=(0, 0, 1), cull="R", label="Right lateral"),
    "anterior":  dict(dir=(0, 1, 0),  up=(0, 0, 1), cull=None, label="Anterior"),
    "posterior": dict(dir=(0, -1, 0), up=(0, 0, 1), cull=None, label="Posterior"),
    "superior":  dict(dir=(0, 0, 1),  up=(0, 1, 0), cull=None, label="Superior"),
    "inferior":  dict(dir=(0, 0, -1), up=(0, 1, 0), cull=None, label="Inferior"),
}
CENTER = (0.0, -17.0, 5.0)     # MNI mm, ~centroid of the whole-brain backdrop
DISTANCE = 460.0


def set_camera(pl, view, zoom=1.25, center=CENTER, distance=DISTANCE):
    v = VIEWS[view]
    c = np.asarray(center, float)
    pl.camera_position = [
        tuple(c + np.asarray(v["dir"], float) * distance), tuple(c), v["up"]
    ]
    pl.camera.zoom(zoom)


# A three-point rig, not VTK's light kit. The kit fills every surface from five
# directions with high ambient, which is why anything lit with it reads as flat
# and cartoonish: nothing is in shadow, so nothing has form. Here one key light
# does the modelling and the fill and rim only keep the dark side readable.
#
# The key sits above and to the RIGHT of frame, which in a left lateral view
# puts the highlight on the parietal and the shadow on the frontal pole. Fill
# and rim are placed RELATIVE to the key, so moving the key swings the whole rig
# and the lighting stays coherent instead of coming apart at some angles.
KEY_AZIMUTH = 35.0          # degrees right of the lens
KEY_ELEVATION = 35.0        # degrees above it
KEY_INTENSITY = 0.85
FILL_OFFSET = (-150.0, -18.0, 0.31)     # (d_azimuth, elevation, share of key)
RIM_OFFSET = (170.0, 16.0, 0.24)
KEY_COLOR = "#fffaf2"       # a touch warm, so the fill and rim read as cool
FILL_COLOR = "#eef1f6"
RIM_COLOR = "#dfe6f0"

# VTK's kit, kept because it is what earlier yabplot figures were lit with and
# reproducing one exactly is occasionally the point.
KIT_LIGHTS = ((50.0, 10.0, 1.00, "#fff6ea"), (-75.0, -10.0, 0.333, "#eef3fb"),
              (110.0, 0.0, 0.286, "#eef3fb"), (-110.0, 0.0, 0.286, "#eef3fb"),
              (0.0, 0.0, 0.333, "#ffffff"))
RIGS = ("studio", "kit")


def _camera_light(azimuth, elevation, intensity, color):
    """A light placed by angle in CAMERA space: +x right, +y up, +z at the lens."""
    az, el = np.radians(azimuth), np.radians(elevation)
    light = pv.Light(light_type="camera light", intensity=max(intensity, 0.0),
                     color=color)
    light.position = (np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el))
    return light


def add_lights(pl, key_azimuth=KEY_AZIMUTH, key_elevation=KEY_ELEVATION,
               key_intensity=KEY_INTENSITY, rig="studio", center=CENTER):
    """Place the lights. ``rig="kit"`` swaps in VTK's flat five-light kit.

    Every light follows the camera, so a region is lit the same way in a lateral
    view and an inferior one. A world-fixed key looks better in exactly one view
    and leaves the underside of the brain black.
    """
    pl.remove_all_lights()
    key = float(key_intensity)

    if rig == "kit":
        for azimuth, elevation, share, color in KIT_LIGHTS:
            pl.add_light(_camera_light(azimuth, elevation, key * share, color))
        return

    fill_daz, fill_el, fill_share = FILL_OFFSET
    rim_daz, rim_el, rim_share = RIM_OFFSET
    pl.add_light(_camera_light(key_azimuth, key_elevation, key, KEY_COLOR))
    pl.add_light(_camera_light(key_azimuth + fill_daz, fill_el, key * fill_share,
                               FILL_COLOR))
    pl.add_light(_camera_light(key_azimuth + rim_daz, rim_el, key * rim_share,
                               RIM_COLOR))


# -- colour -----------------------------------------------------------------
def diverging_cmap(low=BLUE, high=RED, mid="#f4f5f7", name="cp_div"):
    """Symmetric two-tailed map: deepened `low`, `mid` at zero, deepened `high`."""
    return LinearSegmentedColormap.from_list(
        name, [_shade(low, 0.55), low, mid, high, _shade(high, 0.55)]
    )


def ramp_cmap(color=RED, base="#eef0f3", name="cp_ramp"):
    """One-tailed map: near-white through `color` into a deepened `color`."""
    return LinearSegmentedColormap.from_list(
        name, [base, _tint(color, 0.45), color, _shade(color, 0.55)]
    )


def _shade(color, factor):
    return tuple(np.asarray(to_rgb(color)) * factor)


def _tint(color, factor):
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * (1.0 - factor))


def resolve_scale(values, cmap=None, clim=None, color=RED, symmetric=None):
    """Pick a colormap and colour limits for a set of values.

    With no ``cmap``, data that carries both signs gets a symmetric diverging
    map centred on zero, and one-signed data gets a ramp built from ``color``.
    A named matplotlib colormap or a ``#rrggbb`` string overrides the choice;
    a hex string always means "build a ramp from this colour".
    """
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return ramp_cmap(color), (0.0, 1.0)

    chosen = symmetric is None        # auto, rather than asked for outright
    if symmetric is None:
        symmetric = bool((finite > 0).any() and (finite < 0).any())

    if clim is None:
        if symmetric:
            m = float(np.max(np.abs(finite)))
            clim = (-m, m)
        else:
            lo, hi = float(finite.min()), float(finite.max())
            clim = (lo, hi) if hi > lo else (lo, lo + 1.0)

    # Entering the limits backwards means "invert this scale" -- the natural way
    # to ask for it, and the only thing a reversed pair could sensibly mean.
    # Swapping the numbers silently would throw the request away.
    invert = float(clim[0]) > float(clim[1])
    if invert:
        clim = (clim[1], clim[0])

    # A diverging map means "distance from ZERO": its pale midpoint IS zero and
    # its two tails are the two signs. That only holds when the SCALE straddles
    # zero. Choosing one off the DATA alone put a two-signed map under a scale
    # of [0, 2] -- which parks the pale midpoint at 1.0, so mid-range values
    # wash out to near-white and everything at or below 0 saturates to the far
    # tail. A perfectly good map then looks like the colours stopped applying.
    # Only the AUTOMATIC choice is overridden here; asking for diverging
    # outright still gets it, at whatever limits were asked for.
    if chosen and symmetric and not (float(clim[0]) < 0.0 < float(clim[1])):
        symmetric = False

    if cmap is None:
        # The overlay's colour decides the POSITIVE tail either way. It used to
        # be thrown away the moment the data carried both signs, so every
        # two-signed map -- every t-map -- came out in the same blue/white/red
        # whatever swatch was picked. Two overlays then looked identical, which
        # reads as the top one having replaced the bottom one.
        cmap = diverging_cmap(high=color) if symmetric else ramp_cmap(color)
    elif isinstance(cmap, str) and cmap.startswith("#"):
        cmap = diverging_cmap(high=cmap) if symmetric else ramp_cmap(cmap)
    elif isinstance(cmap, str):
        # A name can be matplotlib's or one of MRIcroGL's .clut files, so that
        # a figure and the volume view it came from agree about what red means.
        from gnv.render.palettes import resolve_palette

        cmap = resolve_palette(cmap, fallback_color=color)
    if invert:
        cmap = cmap.reversed()
    return cmap, clim

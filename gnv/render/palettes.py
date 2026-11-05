"""Colour scales, including MRIcroGL's own ``.clut`` files.

A figure and the volume view it came from should agree about what red means, so
the scales here are read from MRIcroGL's LUT folder directly rather than being
eyeballed into matplotlib equivalents.  A ``.clut`` is an INI file of colour
nodes with intensities and RGBA, and it carries something a matplotlib colormap
does not: **alpha**.  MRIcroGL's stock LUTs ramp alpha from 0 at the bottom of
the scale, which is how a hot-iron overlay fades out instead of laying a slab of
dark red over the anatomy underneath -- and that ramp is preserved here.

Anything matplotlib knows is also available by name, and a bare ``#rrggbb``
still builds a two-stop ramp, so nothing that worked before stops working.
"""

from __future__ import annotations

import configparser
import os
import re
from pathlib import Path

import numpy as np

__all__ = ["CLUT_DIRS", "is_clut", "palette_catalogue", "resolve_palette", "read_clut",
           "opaque", "MATPLOTLIB_PICKS"]

# Where MRIcroGL keeps them on macOS, plus a place of one's own.
CLUT_DIRS = (
    Path(os.environ.get(
        "GNV_LUT_DIR",
        Path.home() / ".gnv" / "luts",
    )),
    Path("/Applications/MRIcroGL.app/Contents/Resources/lut"),
    Path.home() / ".gnv" / "luts",
)

# Matplotlib has hundreds; these are the ones worth offering for brain maps.
MATPLOTLIB_PICKS = (
    "viridis", "magma", "inferno", "plasma", "cividis", "turbo",
    "coolwarm", "RdBu_r", "bwr", "seismic", "Spectral_r",
    "hot", "afmhot", "gist_heat", "copper", "bone", "gray",
    "YlOrRd", "YlGnBu", "PuBuGn", "BuPu",
)

_CACHE = {}


def read_clut(path):
    """One MRIcroGL ``.clut`` as ``(positions, rgba)`` with alpha in 0-1.

    Node intensities are 0-255 positions along the scale, not data values, so
    they map straight onto a colormap's 0-1 domain.
    """
    parser = configparser.ConfigParser(strict=False)
    parser.optionxform = str
    parser.read_string(Path(path).read_text(errors="replace"))

    count = int(parser["INT"]["numnodes"])
    positions, colours = [], []
    for i in range(count):
        positions.append(int(parser["BYT"][f"nodeintensity{i}"]) / 255.0)
        r, g, b, a = (int(v) for v in parser["RGBA255"][f"nodergba{i}"].split("|"))
        colours.append((r / 255.0, g / 255.0, b / 255.0, a / 255.0))

    order = np.argsort(positions)
    return np.asarray(positions)[order], np.asarray(colours)[order]


def _clut_colormap(path, name):
    from matplotlib.colors import LinearSegmentedColormap

    positions, rgba = read_clut(path)
    if len(positions) < 2:
        raise ValueError(f"{Path(path).name} has fewer than two nodes")
    # Duplicated stops make from_list raise; nudging keeps a legal, increasing
    # domain without moving any colour anywhere the file did not put it.
    positions = np.maximum.accumulate(positions)
    for i in range(1, len(positions)):
        if positions[i] <= positions[i - 1]:
            positions[i] = min(positions[i - 1] + 1e-6, 1.0)
    positions = (positions - positions[0]) / max(positions[-1] - positions[0], 1e-9)
    return LinearSegmentedColormap.from_list(
        name, list(zip(positions, [tuple(c) for c in rgba]))
    )


def opaque(cmap, name=None):
    """The same colours with alpha forced to 1.

    A LUT's alpha ramp is what makes an overlay fade into the anatomy under it,
    and exactly the wrong thing on a solid surface: there, "low value" would
    render as "see through the brain", which is not what a low value means. The
    surface has its own opacity; the colormap should not quietly override it.
    """
    from matplotlib.colors import ListedColormap

    table = np.asarray(cmap(np.linspace(0, 1, 256)))
    table[:, 3] = 1.0
    return ListedColormap(table, name=name or f"{getattr(cmap, 'name', 'cmap')}_opaque")


def _clut_files():
    found = {}
    for directory in CLUT_DIRS:
        try:
            entries = sorted(Path(directory).expanduser().glob("*.clut"))
        except OSError:
            continue
        for path in entries:
            found.setdefault(path.stem, path)
    return found


def is_clut(name):
    """Whether the palette reference ``name`` is one of MRIcroGL's .clut files
    (looked up as ``resolve_palette`` does: a .clut path, or a .clut name before
    a matplotlib one)."""
    if not isinstance(name, str) or not name or name.startswith("#"):
        return False
    path = Path(name).expanduser()
    return (path.suffix == ".clut" and path.is_file()) or re.sub(r"\.clut$", "", name) in _clut_files()


def palette_catalogue():
    """Everything selectable, grouped for a dropdown."""
    cluts = sorted(_clut_files())
    return {
        "clut": cluts,
        "matplotlib": [n for n in MATPLOTLIB_PICKS if _has_matplotlib(n)],
    }


def _has_matplotlib(name):
    from matplotlib import colormaps

    return name in colormaps


def resolve_palette(name, fallback_color="#c15656"):
    """A matplotlib colormap for any palette reference.

    Accepts a ``.clut`` name or path, a matplotlib colormap name, a ``#rrggbb``
    (built into a ramp) or None (ditto, from ``fallback_color``).
    """
    from matplotlib import colormaps

    from gnv.render.render_scene import ramp_cmap

    if name is None or name == "":
        return ramp_cmap(fallback_color)
    if not isinstance(name, str):
        return name                      # already a colormap
    if name.startswith("#"):
        return ramp_cmap(name)

    if name in _CACHE:
        return _CACHE[name]

    path = Path(name).expanduser()
    if path.suffix == ".clut" and path.is_file():
        _CACHE[name] = _clut_colormap(path, path.stem)
        return _CACHE[name]

    stem = re.sub(r"\.clut$", "", name)
    files = _clut_files()
    if stem in files:
        _CACHE[name] = _clut_colormap(files[stem], stem)
        return _CACHE[name]

    if name in colormaps:
        return colormaps[name]
    return ramp_cmap(fallback_color)

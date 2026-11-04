"""A resource folder's ``index.json``: what its dropdown offers, and where each
entry's files are.

Every resource folder (meshes, backdrops, electrodes) has one: a flat list of
entries, one per line, each naming the section of the dropdown it goes under:

    {"entries": [
      {"name": "pial_wholebrain", "section": "Whole brain", "label": "Pial whole brain", "hint": "pial cortex + eroded SUIT cerebellum + brainstem", "family": "surface", "pieces": ["pial", "brain_surfaces/pial_cerebellum.vtk", "brain_surfaces/brainstem.vtk"]},
      {"name": "suit_parcels", "section": "Atlases", "label": "SUIT cerebellum, 32 regions", "family": "subcortex", "parcels": "atlases/suit"}
    ]}

``name`` is what a figure saves and code looks up. ``section`` is the heading
it is listed under; sections appear in the order they first appear in the list,
entries in list order. ``label`` is the name the list shows, and ``hint``
(optional) the line shown under it when it is picked.

The keys that hold paths -- ``file``, ``parcels`` (a folder of region meshes),
``tracts`` (a fibre set's folder), and each item of ``pieces``
that contains a "/" -- are relative to the folder holding the index;
``read_index`` makes them absolute. A ``pieces`` item without a "/" is one of
yabplot's cortical surfaces ("pial") and is left as it is. Any other keys are
the resource's own (``family``, ``yabplot_atlas``) and pass through.

Adding something by hand is: put its files in the folder, add a line. Code that
adds things (an atlas conversion, a content pack) uses ``add_entry``.

An atlas folder may hold a ``regions.json`` of the same shape, one line per
region, which is how the viewer's Regions list is grouped and annotated:

    {"entries": [
      {"name": "al_L", "section": "Prosencephalon white matter", "hint": "Ansa lenticularis"}
    ]}

``name`` is the region's mesh file name without ``.vtk``; ``section`` the
heading it is grouped under, whose box shows or hides the whole section;
``hint`` what the region is, shown beside it. Regions are listed in file
order. ``read_regions`` reads it; it is optional.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

INDEX = "index.json"
REGIONS = "regions.json"


def _absolute(entry, folder):
    out = dict(entry)
    for key in ("file", "parcels", "tracts"):
        if key in out:
            out[key] = str(folder / out[key])
    if "pieces" in out:
        out["pieces"] = [str(folder / p) if "/" in str(p) else str(p) for p in out["pieces"]]
    return out


def _load(folder, filename=INDEX):
    path = Path(folder).expanduser() / filename
    return json.loads(path.read_text()) if path.is_file() else {"entries": []}


def sections(entries):
    """``[{"label": section, "entries": [...]}, ...]``: entries under their
    ``section``, sections in the order they first appear."""
    out = {}
    for entry in entries:
        out.setdefault(str(entry.get("section", "")), []).append(entry)
    return [{"label": label, "entries": listed} for label, listed in out.items()]


def read_index(folder):
    """The dropdown ``folder/index.json`` describes: ``sections()`` of its entries,
    with every path absolute. ``[]`` when there is no index."""
    folder = Path(folder).expanduser()
    return sections([_absolute(e, folder) for e in _load(folder).get("entries", [])])


def index_entries(folder):
    """``{name: entry}`` from ``folder/index.json``, in dropdown order."""
    return {e["name"]: e for section in read_index(folder) for e in section["entries"]}


def read_regions(folder):
    """``{region name: entry}`` from ``folder/regions.json``, in file order;
    ``{}`` when the atlas has none."""
    return {e["name"]: e for e in _load(folder, REGIONS).get("entries", [])}


def add_entry(folder, entry):
    """Add ``entry`` (with its ``section``; paths relative to ``folder``) to
    ``folder/index.json``, creating it if needed. An entry of the same name is
    replaced where it stands; a new one goes at the end."""
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    raw = _load(folder)
    listed = raw.get("entries", [])
    names = [e.get("name") for e in listed]
    if entry["name"] in names:
        listed[names.index(entry["name"])] = entry
    else:
        listed.append(entry)
    raw["entries"] = listed
    write_index(folder, raw)


def write_index(folder, raw, filename=INDEX):
    """Write ``raw`` as ``folder/index.json`` (or ``filename``), one entry per
    line so it reads and edits like the list it is."""
    lines = ["{"]
    if "about" in raw:
        lines.append(f'  "about": {json.dumps(raw["about"])},')
    entries = raw.get("entries", [])
    lines.append('  "entries": [')
    lines += [f"    {json.dumps(e)}" + ("," if i < len(entries) - 1 else "") for i, e in enumerate(entries)]
    lines += ["  ]", "}"]
    path = Path(folder).expanduser() / filename
    partial = path.with_name(f".{filename}.{os.getpid()}.partial")
    partial.write_text("\n".join(lines) + "\n")
    os.replace(partial, path)

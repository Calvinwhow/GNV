"""
brains.py
=========
What ``neuro_plotter`` can draw, and what can colour it.

Two registries, two families, one shape. A figure names a ``mesh`` (the
geometry) and an ``atlas`` (the parcellation that colours it), and the plot
family decides which registry entries are legal:

``surface``    a continuous cortical surface, optionally with whole-brain
               pieces attached. Coloured by a cortical ``atlas``, or with
               ``atlas=None`` drawn translucent with the map as isosurfaces
               inside it -- the glass-brain rendering.
``subcortex``  a set of per-region meshes, one value each. The meshes *are*
               the parcellation, so a subcortex figure needs no ``atlas``.

Compatibility is stated in every description and enforced by
:func:`require_family`, so a mesh used by the wrong family fails with a
sentence naming the family it belongs to rather than rendering something
arbitrary.
"""

from __future__ import annotations

from gnv import paths

# The meshes folder the viewer ships: its index.json (what there is),
# brain_surfaces/ (surface meshes), atlases/ (folders of region meshes), and any
# other folder of meshes someone adds.
RESOURCE_DIR = paths.RESOURCE_DIR / "meshes"

FAMILIES = ("surface", "subcortex")


# --------------------------------------------------------------------------- #
# Meshes -- the geometry a figure draws
# --------------------------------------------------------------------------- #

# The meshes are listed in ``index.json`` in RESOURCE_DIR, not here: it is the
# Mesh dropdown written down (see gnv.resource_index), so adding a mesh
# is adding its files and an entry, with no code to change. Each entry has a
# ``family``, a ``what``, and either ``pieces`` (a surface: yabplot cortical
# surface names and mesh files) or ``parcels`` / ``yabplot_atlas`` (a set of
# region meshes: a folder of them, or one of yabplot's).
_MESHES = None


def meshes() -> dict:
    """``{name: spec}`` for every mesh in ``RESOURCE_DIR/index.json``, read once."""
    global _MESHES
    if _MESHES is None:
        from gnv.resource_index import index_entries

        _MESHES = index_entries(RESOURCE_DIR)
    return _MESHES


def __getattr__(name):
    if name == "MESHES":
        return meshes()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- #
# Atlases -- what colours a surface
# --------------------------------------------------------------------------- #

ATLASES = {
    "aal3": {
        "hint": "[surface only] AAL3 cortical parcellation",
        "family": "surface", "yabplot_atlas": "aal3",
    },
    "aparc": {
        "hint": "[surface only] Desikan-Killiany (aparc)",
        "family": "surface", "yabplot_atlas": "aparc",
    },
    "brainnetome": {
        "hint": "[surface only] Brainnetome cortical parcellation",
        "family": "surface", "yabplot_atlas": "brainnetome",
    },
    "schaefer100": {
        "hint": "[surface only] Schaefer 100 parcels",
        "family": "surface", "yabplot_atlas": "schaefer100",
    },
    "schaefer200": {
        "hint": "[surface only] Schaefer 200 parcels",
        "family": "surface", "yabplot_atlas": "schaefer200",
    },
    "schaefer300": {
        "hint": "[surface only] Schaefer 300 parcels",
        "family": "surface", "yabplot_atlas": "schaefer300",
    },
    "schaefer400": {
        "hint": "[surface only] Schaefer 400 parcels",
        "family": "surface", "yabplot_atlas": "schaefer400",
    },
    "schaefer1000": {
        "hint": "[surface only] Schaefer 1000 parcels",
        "family": "surface", "yabplot_atlas": "schaefer1000",
    },
}


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #

def describe(registry) -> dict:
    """``{name: what it is}`` -- the catalogue for the script's config block."""
    return {name: spec["hint"] for name, spec in registry.items()}


def _lookup(registry, name, label):
    if name not in registry:
        raise ValueError(
            f"{label} {name!r} is not known. Available:\n"
            + "\n".join(f"  {k:26s} {v['hint']}" for k, v in registry.items())
        )
    return dict(registry[name])


def require_family(name, family, registry=None, label="mesh"):
    """Look a name up and insist it belongs to ``family``.

    A subcortex mesh in a surface figure is not a near-miss to be coerced -- it
    is a different kind of object -- so this names the family the entry does
    belong to, and the entries that would have worked.
    """
    registry = meshes() if registry is None else registry
    spec = _lookup(registry, name, label)
    if spec["family"] != family:
        ok = [k for k, v in registry.items() if v["family"] == family]
        raise ValueError(
            f"{label} {name!r} is a {spec['family']} entry ({spec['hint']}), "
            f"but this is a {family} plot. {label}es that work here: {ok}"
        )
    return spec


def mesh_pieces(name) -> list:
    """The pieces of a surface mesh, for ``glass_mesh.build_bmesh``."""
    return list(require_family(name, "surface")["pieces"])


def mesh_bmesh(name, fallback="midthickness") -> str:
    """The cortical surface a surface mesh is built on.

    Parcel-on-surface plots draw on one fsLR32k surface, so a whole-brain mesh
    yields whichever surface it contains; its other pieces have no vertices the
    cortical atlas knows about.
    """
    from gnv.render.glass_mesh import BMESH_NAMES

    for piece in require_family(name, "surface")["pieces"]:
        if piece in BMESH_NAMES:
            return piece
    return fallback


def mesh_subcortex(name) -> dict:
    """Where a subcortex mesh's regions come from.

    Returns ``{"custom_atlas_path": ...}`` for meshes built locally, or
    ``{"atlas": ...}`` for a set packaged with yabplot -- the two ways yabplot
    accepts a region set.
    """
    spec = require_family(name, "subcortex")
    if "parcels" in spec:
        path = RESOURCE_DIR / spec["parcels"]
        if not path.is_dir():
            raise FileNotFoundError(
                f"mesh {name!r} needs {path}, which does not exist. Rebuild it with "
                f"gnv.convert.parcel_meshes."
            )
        return {"custom_atlas_path": str(path), "atlas": None}
    return {"custom_atlas_path": None, "atlas": spec["yabplot_atlas"]}


def atlas_name(name):
    """The yabplot cortical atlas a surface figure is parcellated by.

    ``None`` means no parcellation: there is nothing to reduce the map into, so
    the surface is drawn translucent with the map as isosurfaces inside it.
    """
    if name is None:
        return None
    return require_family(name, "surface", ATLASES, "atlas")["yabplot_atlas"]


__all__ = [
    "MESHES", "ATLASES", "FAMILIES", "RESOURCE_DIR",
    "meshes", "describe", "require_family",
    "mesh_pieces", "mesh_bmesh", "mesh_subcortex", "atlas_name",
]


# --- supra-regions ---------------------------------------------------------
# A 170-region atlas is too many checkboxes to work through when what you want
# is "everything except the cerebellum". These are the coarse groupings, matched
# on the naming AAL3 and SUIT already use rather than stored as a 170-line
# table, so a region added later lands in the right place without an edit here.
#
# Every rule is a prefix or an exact stem, tried in order, so the specific ones
# come first: "Frontal_Med_Orb" is frontal, and "OFCmed" is too, but "Olfactory"
# has to be named because nothing in it says frontal.
SUPRA_RULES = (
    ("Cerebellum", ("Cerebellum", "Vermis")),
    ("Brainstem",  ("Brainstem", "Red_N", "SN_", "Raphe", "VTA", "LC_", "PAG")),
    ("Thalamus",   ("Thal_", "Thal-")),
    ("Basal ganglia", ("Caudate", "Putamen", "Pallidum", "N_Acc")),
    ("Cingulate",  ("ACC_", "Cingulate_")),
    ("Insula",     ("Insula",)),
    # Medial temporal sits with the temporal lobe: hippocampus and amygdala are
    # what someone means by "turn off the temporal lobe" in a lateral view.
    ("Temporal",   ("Temporal", "Heschl", "Fusiform", "Hippocampus",
                    "ParaHippocampal", "Amygdala")),
    ("Occipital",  ("Occipital", "Calcarine", "Cuneus_", "Cuneus", "Lingual")),
    ("Parietal",   ("Parietal", "Postcentral", "Precuneus", "SupraMarginal",
                    "Angular")),
    ("Frontal",    ("Frontal", "Precentral", "Supp_Motor_Area", "OFC", "Rectus",
                    "Olfactory", "Paracentral", "Rolandic_Oper")),
    ("Midline",    ("Midline",)),
)

SUPRA_NAMES = tuple(name for name, _ in SUPRA_RULES)


def supra_region(name):
    """The coarse group a region belongs to, or ``None`` if nothing matches.

    ``None`` rather than an "Other" bucket, so an atlas this does not understand
    shows no supra-region controls instead of one misleading catch-all.
    """
    stem = str(name).rsplit(".", 1)[0]
    for suffix in ("_L", "_R", "-L", "-R"):
        if stem.upper().endswith(suffix):
            stem = stem[:-2]
            break
    # Precuneus before Cuneus, Paracentral before Postcentral: longest first
    # inside each group is not enough, because the groups themselves overlap.
    for group, prefixes in SUPRA_RULES:
        for prefix in prefixes:
            if stem.lower().startswith(prefix.lower()):
                return group
    return None

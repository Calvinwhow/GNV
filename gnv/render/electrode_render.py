"""Lead-DBS electrode models, rendered from a reconstruction.

Ported from Lead-DBS: ``ea_showelectrode.m`` and ``ea_mapelmodel2reco.m``.

A model file holds the electrode built once at a canonical position, as two
groups of surfaces -- ``insulation`` and ``contacts`` -- plus four fiducials
(``head``, ``tail``, ``x``, ``y``). A reconstruction holds the same four
fiducials where the lead actually sits in a patient. Fitting one to the other is
a single affine solve, which is the whole of the geometry:

    A = [head 1; tail 1; x 1; y 1]   in the model
    B = [head 1; tail 1; x 1; y 1]   in the patient
    X = (A \\ B)'                     and every vertex goes through X

Four non-coplanar points determine the affine exactly, so this is a solve rather
than a fit: head and tail give the trajectory, x and y fix the roll, which is
what makes a directed lead's segmented contacts land on the right side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_model", "read_markers", "electrode_meshes", "electrode_poses", "pose_markers", "marker_pose", "mirror_pose", "shaft_direction", "marker_direction", "shaft_stretch", "model_file", "available_models", "model_for",
           "looks_like_electrode", "INSULATION_COLOR", "CONTACT_COLOR"]

# Lead-DBS' own defaults: a pale grey shaft with dark metal bands.
INSULATION_COLOR = "#d4d6da"
CONTACT_COLOR = "#43484f"


# -- MATLAB files -----------------------------------------------------------
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def _is_hdf5(path):
    """Whether this is a v7.3 (HDF5) MAT-file rather than an older one.

    A v7.3 file keeps MATLAB's 128-byte text header and puts the HDF5 signature
    at offset 512, so testing byte 0 says "not HDF5" for every v7.3 file there
    is -- and every Lead-DBS electrode model is v7.3.
    """
    with open(path, "rb") as fh:
        if fh.read(8) == _HDF5_MAGIC:
            return True
        fh.seek(512)
        return fh.read(8) == _HDF5_MAGIC


def _h5_meshes(handle, group):
    """``[(vertices, faces)]`` from one struct-of-arrays group in a v7.3 file.

    A part with several meshes holds a reference per mesh; a part with just
    one (a 1x1 struct -- the SmartFlow cannula's contacts, the rat electrodes)
    holds that mesh's arrays directly."""
    import h5py

    faces, verts = group["faces"], group["vertices"]
    if faces.dtype == h5py.ref_dtype:
        pairs = [(np.array(handle[verts[i, 0]]), np.array(handle[faces[i, 0]])) for i in range(faces.shape[0])]
    else:
        pairs = [(np.array(verts), np.array(faces))]
    # MATLAB writes column-major, so HDF5 hands both back transposed.
    return [(np.ascontiguousarray(v.T, float), np.ascontiguousarray(f.T).astype(np.int64) - 1) for v, f in pairs]


def _mat_meshes(struct):
    """The same, from the scipy path for pre-7.3 files."""
    out = []
    entries = np.atleast_1d(struct).ravel()
    for item in entries:
        v = np.asarray(item.vertices, float)
        f = np.asarray(item.faces).astype(np.int64) - 1
        out.append((v, f))
    return out


def _polydata(vertices, faces):
    import pyvista as pv

    quads = faces.shape[1]
    sizes = np.full((len(faces), 1), quads, np.int64)
    return pv.PolyData(vertices, np.hstack([sizes, faces]).ravel())


def load_model(path):
    """One electrode model: its surfaces and the four fiducials that place it."""
    path = Path(path).expanduser()
    if _is_hdf5(path):
        import h5py

        with h5py.File(path, "r") as handle:
            e = handle["electrode"]
            parts = {name: _h5_meshes(handle, e[name])
                     for name in ("insulation", "contacts")}
            fids = {k: np.array(e[k]).ravel()[:3]
                    for k in ("head_position", "tail_position",
                              "x_position", "y_position")}
    else:
        import scipy.io as sio

        e = sio.loadmat(str(path), struct_as_record=False,
                        squeeze_me=True)["electrode"]
        parts = {name: _mat_meshes(getattr(e, name))
                 for name in ("insulation", "contacts")}
        fids = {k: np.asarray(getattr(e, k), float).ravel()[:3]
                for k in ("head_position", "tail_position",
                          "x_position", "y_position")}

    A = np.array([[*fids["head_position"], 1.0],
                  [*fids["tail_position"], 1.0],
                  [*fids["x_position"], 1.0],
                  [*fids["y_position"], 1.0]])
    return {"name": path.stem, "A": A,
            "insulation": [_polydata(v, f) for v, f in parts["insulation"]],
            "contacts": [_polydata(v, f) for v, f in parts["contacts"]]}


def model_for(elmodel, models_dir):
    """Find a model file from Lead-DBS' human-readable electrode name.

    Lead-DBS keeps the mapping in ea_resolve_elspec.m as a long switch. Rather
    than port a table that changes with every new lead, the name is slugged the
    way the models are named in ``models_dir/index.json`` -- "Medtronic 3389" ->
    medtronic_3389 -- and matched against them.
    """
    from gnv.resource_index import index_entries

    models_dir = Path(models_dir).expanduser()
    slug = "".join(c if c.isalnum() else "_" for c in str(elmodel).lower())
    slug = "_".join(p for p in slug.split("_") if p)
    have = {name: Path(entry["file"]) for name, entry in index_entries(models_dir).items()}
    if slug in have:
        return have[slug]
    # Fall back to the longest stem the name starts with, so
    # "boston_scientific_vercise_directed" still finds "boston_vercise_directed".
    tokens = set(slug.split("_"))
    scored = sorted(((len(tokens & set(stem.split("_"))), stem)
                     for stem in have), reverse=True)
    if scored and scored[0][0] >= 2:
        return have[scored[0][1]]
    raise FileNotFoundError(
        f"no electrode model matching {elmodel!r} in {models_dir}")


# -- reconstructions --------------------------------------------------------
def _walk(node, handle=None):
    """MATLAB struct -> nested dict, for either file generation."""
    import h5py

    if handle is not None and isinstance(node, h5py.Group):
        return {k: _walk(node[k], handle) for k in node.keys()}
    if handle is not None and isinstance(node, h5py.Dataset):
        if node.dtype == h5py.ref_dtype:
            refs = np.asarray(node).ravel()
            return [_walk(handle[r], handle) for r in refs]
        return np.array(node)
    if hasattr(node, "_fieldnames"):
        return {k: _walk(getattr(node, k)) for k in node._fieldnames}
    # loadmat hands back a PLAIN DICT at the top level, whose values are the
    # mat_structs. Without this branch the walk stopped at that dict and every
    # struct under it came through unconverted, so "reco" was a mat_struct and
    # the markers lookup found nothing -- "has no markers; not a
    # reconstruction?" on a perfectly good reconstruction.
    if isinstance(node, dict):
        return {k: _walk(v) for k, v in node.items()
                if not str(k).startswith("__")}
    if isinstance(node, np.ndarray) and node.dtype == object:
        return [_walk(x) for x in node.ravel()]
    return node


def _text(value):
    """A MATLAB char array, bytes, or nested 1-element array, as a str."""
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "SU":
            flat = value.ravel()
            return str(flat[0]) if flat.size == 1 else "".join(
                str(x) for x in flat)
        if value.size == 1:
            return _text(value.ravel()[0])
        # A v7.3 char array arrives as uint16 code points.
        try:
            return "".join(chr(int(c)) for c in value.ravel())
        except (TypeError, ValueError):
            return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return "" if value is None else str(value)


def _records(struct):
    """A MATLAB struct array as a list of dicts, one per element.

    A v5 file hands a struct array back as a list already, and a 1x1 struct as
    one dict. A v7.3 file hands back ONE dict for the whole array, each field a
    list with one entry per element -- so a two-lead reconstruction read as if
    it were one dict looked like a single lead whose numbers and name were both
    leads' run together.
    """
    if isinstance(struct, list):
        return struct
    if not isinstance(struct, dict):
        return []
    lengths = {len(v) if isinstance(v, list) else None for v in struct.values()}
    if len(lengths) == 1 and None not in lengths:
        n = lengths.pop()
        return [{k: v[i] for k, v in struct.items()} for i in range(n)]
    return [struct]


def read_markers(path):
    """``[{side, head, tail, x, y, elmodel}]`` from a reconstruction.

    Accepts an ``ea_reconstruction.mat`` of either generation, or a JSON file
    with the same shape -- which is the escape hatch for anything that is not
    MATLAB, and the reason this takes a structure rather than a file format.
    """
    path = Path(path).expanduser()
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(path.read_text())
    elif _is_hdf5(path):
        import h5py

        with h5py.File(path, "r") as handle:
            data = {k: _walk(handle[k], handle) for k in handle.keys()
                    if not k.startswith("#")}
    else:
        import scipy.io as sio

        data = _walk(sio.loadmat(str(path), struct_as_record=False,
                                 squeeze_me=True))
        data.pop("__header__", None)
        data.pop("__version__", None)
        data.pop("__globals__", None)

    reco = data.get("reco", data)
    # MNI first: a figure in template space is the usual case, and a native
    # reconstruction plotted on an MNI brain would be silently in the wrong
    # place rather than obviously missing.
    space = None
    for key in ("mni", "scrf", "native"):
        if isinstance(reco, dict) and key in reco:
            space = reco[key]
            break
    if space is None:
        space = reco

    markers = space.get("markers") if isinstance(space, dict) else None
    if markers is None:
        raise ValueError(f"{path.name} has no markers; not a reconstruction?")
    markers = _records(markers)

    # The model name lives PER LEAD, in reco.props(i).elmodel -- not at the top
    # level. Two leads in one file can be different models, and reading one
    # name for both would place the wrong contacts on one side.
    props = _records(reco.get("props")) if isinstance(reco, dict) else []
    models = [_text(p.get("elmodel")) for p in (props or [])
              if isinstance(p, dict)]
    models = [m for m in models if m]
    fallback = ""
    for holder in (reco if isinstance(reco, dict) else {}, data):
        if isinstance(holder, dict) and holder.get("elmodel") is not None:
            fallback = _text(holder["elmodel"])
            break
    if not models and fallback:
        models = [fallback]

    out = []
    for i, entry in enumerate(markers):
        if not isinstance(entry, dict):
            continue
        try:
            fids = {k: np.asarray(entry[k], float).ravel()[:3]
                    for k in ("head", "tail", "x", "y")}
        except (KeyError, TypeError, ValueError):
            continue
        if any(len(v) != 3 or not np.isfinite(v).all() for v in fids.values()):
            continue
        model = models[i] if i < len(models) else (models[0] if models
                                                   else fallback)
        out.append({"side": ["R", "L"][i] if i < 2 else str(i), **fids,
                    "elmodel": model or None})
    if not out:
        raise ValueError(f"{path.name} has no usable markers")
    return out


# -- the transform ----------------------------------------------------------
def fit(A, markers):
    """The affine that carries the model's fiducials onto the patient's."""
    B = np.array([[*markers["head"], 1.0],
                  [*markers["tail"], 1.0],
                  [*markers["x"], 1.0],
                  [*markers["y"], 1.0]])
    return np.linalg.solve(A, B).T


def _apply(mesh, X):
    out = mesh.copy()
    pts = np.asarray(mesh.points, float)
    out.points = (X @ np.c_[pts, np.ones(len(pts))].T)[:3].T
    return out


# -- trajectory poses ----------------------------------------------------------
# A POSE places one lead the way a stereotactic frame does: put the tip on a
# point, then aim the shaft and turn the marker by angles read off the three
# anatomical planes. World mm and degrees; x right, y anterior, z superior:
#
#   tip       T, shape (3,)  the lead's deepest point
#   sagittal  s              the shaft seen from the side (y-z plane):
#                            0 = toward the nose, 90 = straight up, 180 = toward the back
#   coronal   c              the shaft seen from the front (x-z plane):
#                            0 = toward the right ear, 90 = straight up, 180 = toward the left ear
#   marker    mu             the y fiducial seen from above (x-y plane):
#                            0 = right ear, 90 = nose, 180 = left ear, 270 = back of the head
#
# Every angle is the direction of a shadow on its plane, on the unit circle.
# Directions built from them, each a unit vector of shape (3,):
#   d = (cot c, cot s, 1), normalised    the shaft, tip -> tail (it always rises)
#   h = (cos mu, sin mu, 0)              the marker's shadow on the axial plane
#   m = d_z h - (h . d) z, normalised    the one direction around the shaft whose
#                                        shadow is h: where the y fiducial points
#
# The marker angle is Lead-DBS's orientation angle plus 90 (ea_calcxy,
# ea_calc_rotation; DiODe reports it): theirs puts 0 at the nose and 90 at the
# left ear, and finds the fiducial from its shadow the same way. A model is built
# along +z with its y fiducial on +y; on Boston and Abbott directed leads that is
# the orientation band and the first segment of each level. A pose becomes the
# same four markers a reconstruction holds -- head, tail, x, y -- so it is placed
# by the one solve every lead goes through (``fit``). A lead with no pose is never
# converted: it is placed straight from its file's markers.

def _model_frame(spec):
    """The model's own geometry: head h (3,), shaft unit u (3,), head-tail length L,
    tip depth s below the head along u, and the x / y fiducial radii rx, ry."""
    head, tail, x, y = spec["A"][:, :3]
    L = float(np.linalg.norm(tail - head))
    u = (tail - head) / L
    points = np.vstack([np.asarray(m.points, float) for m in spec["insulation"] + spec["contacts"]])
    s = float(-np.min((points - head) @ u))
    return head, u, L, s, float(np.linalg.norm(x - head)), float(np.linalg.norm(y - head))


def shaft_direction(sagittal, coronal, flattest=1.0):
    """Unit shaft d (3,), tip -> tail, from the sagittal and coronal angles in
    degrees. Each is held ``flattest`` degrees off the horizontal, where the
    shaft would lie flat and the angles stop pinning it down."""
    s, c = (np.radians(min(max(float(v), flattest), 180.0 - flattest)) for v in (sagittal, coronal))
    d = np.array([1.0 / np.tan(c), 1.0 / np.tan(s), 1.0])
    return d / np.linalg.norm(d)


def marker_direction(d, marker):
    """Unit m (3,) perpendicular to the shaft d whose shadow on the axial plane
    points at ``marker`` degrees (0 right ear, 90 nose)."""
    h = np.array([np.cos(np.radians(float(marker))), np.sin(np.radians(float(marker))), 0.0])
    m = d[2] * h - (h @ d) * np.array([0.0, 0.0, 1.0])
    return m / np.linalg.norm(m)


def pose_markers(spec, pose, stretch=1.0):
    """The head / tail / x / y markers that put model ``spec`` at ``pose``.

    ``stretch`` lengthens the lead along its shaft by that factor (see
    ``shaft_stretch``), so a posed lead keeps the length its file gave it."""
    _head, _u, L, depth, rx, ry = _model_frame(spec)
    d = shaft_direction(pose["sagittal"], pose["coronal"])
    m = marker_direction(d, pose["marker"])
    head = np.asarray(pose["tip"], float) + stretch * depth * d
    return {"head": head, "tail": head + stretch * L * d, "x": head + rx * np.cross(m, d), "y": head + ry * m}


def shaft_stretch(spec, X):
    """How much the 4x4 affine ``X`` lengthens model ``spec`` along its shaft.

    A reconstruction's MNI markers come out of the nonlinear warp with the
    head-tail distance changed -- 1.0 to 1.5 times the model's in the files this
    was checked on -- while x and y stay one lead radius out. Lead-DBS draws the
    lead stretched that way, and so does the file placement here; a pose keeps
    the same stretch so that aiming a lead never changes its length. 1 for a
    lead placed rigidly."""
    _head, u, _L, _depth, _rx, _ry = _model_frame(spec)
    return float(np.linalg.norm(np.asarray(X, float)[:3, :3] @ u))


def marker_pose(spec, X):
    """The pose of model ``spec`` once placed by the 4x4 affine ``X`` -- what a
    reconstruction says, in the terms a pose is edited in."""
    head, u, L, depth, _rx, _ry = _model_frame(spec)
    _, _, _, y = spec["A"][:, :3]
    place = lambda p: (X @ np.r_[p, 1.0])[:3]
    d = place(head + L * u) - place(head)
    d = d / np.linalg.norm(d)
    m = place(y) - place(head)
    m = m - (m @ d) * d
    tip = place(head - depth * u)
    sagittal = float(np.degrees(np.arctan2(d[2], d[1])))
    coronal = float(np.degrees(np.arctan2(d[2], d[0])))
    marker = float(np.degrees(np.arctan2(m[1], m[0])) % 360.0)
    return {"tip": [round(float(v), 3) for v in tip], "sagittal": round(sagittal, 3), "coronal": round(coronal, 3), "marker": round(marker, 3)}


def mirror_pose(pose):
    """``pose`` reflected across the midsagittal plane x = 0.

    The tip goes to (-x, y, z). Seen from the side nothing changes, so the
    sagittal angle stays; seen from the front or from above, right and left
    swap, so the coronal and marker angles each go to 180 minus themselves.
    The twin is still placed by a rotation, not a reflection: it is a real
    lead, so its segments keep their physical order and only where it sits and
    which way its marker faces mirror.
    """
    x, y, z = (float(v) for v in pose["tip"])
    return {"tip": [-x, y, z], "sagittal": float(pose["sagittal"]), "coronal": 180.0 - float(pose["coronal"]), "marker": (180.0 - float(pose["marker"])) % 360.0}

_MODELS = {}     # (path, mtime) -> load_model(path); a slider drag must not re-read the .mat


def _model(path):
    path = Path(path).expanduser()
    key = (str(path), path.stat().st_mtime)
    if key not in _MODELS:
        if len(_MODELS) >= 8:
            _MODELS.pop(next(iter(_MODELS)))
        _MODELS[key] = load_model(path)
    return _MODELS[key]


def model_file(model, models_dir):
    """A model's file: ``model`` is either a path to one, or a model's name in
    ``models_dir/index.json`` ("boston_vercise_directed")."""
    from gnv.resource_index import index_entries

    path = Path(str(model)).expanduser()
    if path.is_file():
        return path
    entry = index_entries(models_dir).get(str(model))
    if entry is None:
        raise FileNotFoundError(f"no electrode model called {model!r} in {Path(models_dir) / 'index.json'}")
    return Path(entry["file"])


def available_models(models_dir):
    """``[{value, label, hint, section}]``: the models in ``models_dir/index.json``,
    in its order, for a picker."""
    from gnv.resource_index import read_index

    return [{"value": e["name"], "label": e["label"], "hint": e.get("hint", ""), "section": section["label"]} for section in read_index(models_dir) for e in section["entries"]]


def _sites(source, models_dir, model=None):
    """``[(site, spec, X)]``: each lead in ``source``, its model, and the affine its
    file places it with. With no source, ``model`` alone is one lead, side "",
    drawn where it was built (the identity); so is a model file passed as the
    source."""
    models_dir = Path(models_dir).expanduser()
    if not source:
        spec = _model(model_file(model, models_dir))
        return [({"side": "", "elmodel": spec["name"]}, spec, np.eye(4))]
    source = Path(source).expanduser()
    try:
        sites = read_markers(source)
    except (ValueError, KeyError, OSError):
        spec = _model(model_file(model, models_dir) if model else source)
        return [({"side": "", "elmodel": spec["name"]}, spec, np.eye(4))]
    out = []
    for site in sites:
        spec = _model(model_file(model, models_dir) if model else model_for(site["elmodel"], models_dir))
        out.append((site, spec, fit(spec["A"], site)))
    return out


def electrode_poses(source, models_dir, model=None):
    """``{side: pose}`` for every lead in ``source``, as its file places it."""
    return {site["side"]: marker_pose(spec, X) for site, spec, X in _sites(source, models_dir, model)}


def electrode_meshes(source, models_dir, model=None, poses=None, mirror=None):
    """``{name: PolyData}`` for an electrode file, ready to be a group.

    ``source`` is a reconstruction; with ``model`` given (a model file, or a
    model's name in ``models_dir``), that model is used instead of the one the
    reconstruction names. With no ``source``, ``model`` alone is drawn as one
    lead, the way a hypothetical figure starts. A model file passed as the
    source renders at its own canonical position, which is what makes it
    possible to look at a lead without a patient. ``poses`` (``{side: pose}``,
    see above) moves any lead to a pose of its own; leads without one stay
    where their file puts them.

    ``mirror`` names the lead the other one follows, reflected across x = 0
    (``mirror_pose``): "R" puts the left lead at the right one's mirror image,
    "L" the reverse. A source with a single lead ("") gains a mirrored copy,
    named "mirror", which is how one model becomes a bilateral mock-up.
    """
    poses = poses or {}
    leads = [(site, spec, fit(spec["A"], pose_markers(spec, poses[site["side"]], shaft_stretch(spec, X))) if poses.get(site["side"]) else X) for site, spec, X in _sites(source, models_dir, model)]
    by_side = {site["side"]: (site, spec, X) for site, spec, X in leads}
    if mirror is not None and mirror in by_side:
        _site, spec, X = by_side[mirror]
        twin = mirror_pose(marker_pose(spec, X))
        follower = {"R": "L", "L": "R"}.get(mirror, "mirror")
        twin_spec = by_side[follower][1] if follower in by_side else spec
        twin_X = fit(twin_spec["A"], pose_markers(twin_spec, twin, shaft_stretch(spec, X)))
        leads = [lead for lead in leads if lead[0]["side"] != follower] + [({"side": follower}, twin_spec, twin_X)]
    group = {}
    for site, spec, X in leads:
        tag = f"{site['side']}_" if site["side"] else ""
        for kind in ("insulation", "contacts"):
            for i, mesh in enumerate(spec[kind], start=1):
                group[f"{tag}{kind}_{i:02d}"] = _apply(mesh, X)
    return group


def looks_like_electrode(path):
    """Whether this file is a lead rather than a tractogram.

    Both arrive as ``.mat``, so the extension decides nothing and the caller has
    to look inside. Only the top-level names are read -- a Lead-DBS tractogram
    is hundreds of megabytes and must not be loaded to answer this.
    """
    path = Path(path).expanduser()
    wanted = {"electrode", "reco", "markers", "elmodel"}
    try:
        if path.suffix.lower() == ".json":
            import json

            data = json.loads(path.read_text())
            return bool(wanted & set(data)) or "reco" in data
        if _is_hdf5(path):
            import h5py

            with h5py.File(path, "r") as handle:
                return bool(wanted & set(handle.keys()))
        import scipy.io as sio

        names = {n for n, _, _ in sio.whosmat(str(path))}
        return bool(wanted & names)
    except Exception:
        return False

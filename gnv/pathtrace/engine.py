"""The path tracer: a snapshot of the view (``viewer.snapshot``) as a Mitsuba 3
scene, rendered a few samples at a time so a picture appears at once and
sharpens as more arrive.

Mitsuba runs on the CPU. Its fast mode compiles the renderer with LLVM, which
it finds on the computer (Homebrew's ``llvm`` on a Mac); without it, Mitsuba's
plain mode renders the same picture about half as fast. ``variant`` picks.

What the picture keeps from the view, so the two agree:
    camera       position, direction, up, and the parallel or perspective lens
    colours      every surface, curve and slice in the colour it is drawn in --
                 a map's colours are the colour bar's, lit
    lights       the viewer's rig (``render_scene.light_rig``), as soft area
                 lights that follow the camera, plus an even ambient light
    background   the Scene background, behind everything
What it adds: soft shadows, light bouncing between surfaces, real tubes for
fibres and lines, and optionally the scene's shadow on the background.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

DEFAULTS = {
    "samples": 256,          # samples per pixel when it is finished
    "light_size": 0.6,       # size of the lights, as a share of the scene: larger is softer
    "ambient": 0.3,          # even light from all round, so nothing in shadow is black
    "backdrop": False,       # catch the scene's shadow on the background
    "exposure": 1.0,
    "bounces": 6,
}

# Each pass renders at most this many samples in all, so a full-size export
# never asks for more memory than a preview does.
MAX_PASS_SAMPLES = 1 << 26


def settings(raw):
    """The options: DEFAULTS, overridden by whatever the page sent that is known and sane."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in (raw or {}).items() if k in DEFAULTS})
    out["samples"] = int(min(max(int(out["samples"]), 1), 8192))
    out["bounces"] = int(min(max(int(out["bounces"]), 1), 32))
    for key, low, high in (("light_size", 0.05, 3.0), ("ambient", 0.0, 2.0), ("exposure", 0.1, 8.0)):
        out[key] = float(min(max(float(out[key]), low), high))
    out["backdrop"] = bool(out["backdrop"])
    return out


_VARIANT = None


def variant():
    """Mitsuba's fast LLVM mode when LLVM is on this computer, its plain mode otherwise."""
    global _VARIANT
    if _VARIANT is None:
        import drjit as dr
        import mitsuba as mi

        _VARIANT = "llvm_ad_rgb" if dr.has_backend(dr.JitBackend.LLVM) else "scalar_rgb"
        mi.set_variant(_VARIANT)
    return _VARIANT


def linear(srgb):
    """sRGB (0..1, as the view draws colours) to the linear light Mitsuba works in."""
    c = np.asarray(srgb, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb(lin):
    """Linear light back to sRGB (0..1)."""
    c = np.clip(np.asarray(lin, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def build_scene(snap, opts, scratch):
    """The Mitsuba scene for one snapshot. ``scratch`` is a folder for the
    curve files Mitsuba reads fibres from."""
    import mitsuba as mi

    variant()
    cam = snap["camera"]
    bounds = _bounds(snap)
    radius = float(np.linalg.norm(bounds[1] - bounds[0])) / 2 or 100.0
    scene = {"type": "scene", "integrator": {"type": "path", "max_depth": opts["bounces"], "hide_emitters": True}, "sensor": _sensor(cam)}
    for i, surface in enumerate(snap["surfaces"]):
        scene[f"surface{i}"] = _mesh(f"surface{i}", surface)
    for i, curves in enumerate(snap["curves"]):
        for j, path in enumerate(_curve_files(curves, scratch)):
            colour, filename = path
            scene[f"curves{i}_{j}"] = {"type": "linearcurve", "filename": filename,
                                       "bsdf": {"type": "principled", "base_color": {"type": "rgb", "value": colour.tolist()}, "roughness": 0.4, "specular": 0.5}}
    for i, item in enumerate(snap["slices"]):
        scene[f"slice{i}"] = _slice(item)
    lights = _lights(snap, cam, bounds, radius, opts)
    scene.update(lights)
    if opts["ambient"] > 0:
        scene["ambient"] = {"type": "constant", "radiance": {"type": "rgb", "value": opts["ambient"]}}
    if opts["backdrop"]:
        scene["backdrop"] = _backdrop(snap, cam, bounds, radius, opts, lights)
    return mi.load_dict(scene)


def render(scene, size, opts, publish, cancelled):
    """Render ``scene`` progressively: ``publish(rgba, samples)`` after every
    pass with the image so far (linear, alpha premultiplied); stops early when
    ``cancelled()``. Returns the final image."""
    import mitsuba as mi

    w, h = size
    cap = max(1, MAX_PASS_SAMPLES // max(w * h, 1))
    plan, done = [], 0
    for want in (2, 6, 8, 16):
        if done >= opts["samples"]:
            break
        step = min(want, opts["samples"] - done)
        plan.append(step)
        done += step
    # Later passes stay small enough that moving the view stops a render quickly.
    while done < opts["samples"]:
        step = min(32, opts["samples"] - done)
        plan.append(step)
        done += step
    total, acc = 0, None
    for seed, spp in enumerate(plan):
        for part in range(0, spp, cap):
            if cancelled():
                return None if acc is None else acc / total
            n = min(cap, spp - part)
            image = np.asarray(mi.render(scene, spp=n, seed=seed * 7919 + part), dtype=np.float64)
            acc = image * n if acc is None else acc + image * n
            total += n
        publish(acc / total, total)
    return acc / total


def compose(rgba, background, bars, exposure=1.0):
    """The rendered picture as the view shows it: over the Scene background, in
    sRGB, with the colour bars laid on top. ``uint8`` (h, w, 3)."""
    from matplotlib.colors import to_rgb

    rgb, alpha = rgba[..., :3] * exposure, np.clip(rgba[..., 3:4], 0.0, 1.0)
    out = srgb(rgb + linear(to_rgb(background)) * (1.0 - alpha))
    if bars is not None and bars.shape[:2] == out.shape[:2] and bars.shape[2] == 4:
        a = bars[..., 3:4] / 255.0
        out = out * (1.0 - a) + bars[..., :3] / 255.0 * a
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


# -- the pieces ------------------------------------------------------------------
def _bounds(snap):
    points = [s["points"] for s in snap["surfaces"]]
    points += [p for c in snap["curves"] for _ids, p in c["polylines"]]
    points += [np.asarray(item["corners"]) for item in snap["slices"]]
    points = [p for p in points if len(p)]
    if not points:
        return np.zeros(3), np.ones(3)
    allp = np.concatenate([np.asarray(p, float).reshape(-1, 3) for p in points])
    return allp.min(0), allp.max(0)


def _frame(cam):
    """The camera's own axes: forward, right and up, as the view has them."""
    forward = cam["focal_point"] - cam["position"]
    forward /= max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, cam["up"])
    right /= max(np.linalg.norm(right), 1e-12)
    return forward, right, np.cross(right, forward)


def _sensor(cam):
    import mitsuba as mi

    w, h = cam["size"]
    look = mi.ScalarTransform4f().look_at(origin=cam["position"].tolist(), target=cam["focal_point"].tolist(), up=cam["up"].tolist())
    film = {"type": "hdrfilm", "width": w, "height": h, "pixel_format": "rgba", "rfilter": {"type": "gaussian"}, "sample_border": True}
    sampler = {"type": "independent"}
    if cam["parallel"]:
        # Mitsuba's orthographic lens spans -1..1 across the film's width;
        # VTK's parallel scale is half the HEIGHT.
        half_width = cam["parallel_scale"] * w / h
        return {"type": "orthographic", "to_world": look @ mi.ScalarTransform4f().scale([half_width, half_width, 1.0]), "film": film, "sampler": sampler, "near_clip": 1e-3, "far_clip": 1e5}
    return {"type": "perspective", "fov": cam["view_angle"], "fov_axis": "y", "to_world": look, "film": film, "sampler": sampler, "near_clip": 1e-2, "far_clip": 1e5}


def _mesh(name, surface):
    """One surface as a Mitsuba mesh, coloured as the view colours it."""
    import mitsuba as mi

    colours = surface["colours"]
    per_point = colours.ndim == 2 and len(colours) == len(surface["points"])
    per_face = colours.ndim == 2 and not per_point and len(colours) == len(surface["faces"])
    if per_point or per_face:
        base = {"type": "mesh_attribute", "name": "vertex_color" if per_point else "face_color"}
    else:
        base = {"type": "rgb", "value": linear(colours.reshape(-1, 3)[0]).tolist()}
    bsdf = {"type": "principled", "base_color": base, "roughness": 0.45, "specular": min(0.2 + 0.6 * surface["specular"], 0.8)}
    if surface["opacity"] < 0.999:
        bsdf = {"type": "mask", "opacity": {"type": "rgb", "value": surface["opacity"]}, "material": bsdf}
    props = mi.Properties()
    props["bsdf"] = mi.load_dict(bsdf)
    normals = surface["normals"] is not None
    mesh = mi.Mesh(name, len(surface["points"]), len(surface["faces"]), props, has_vertex_normals=normals)
    params = mi.traverse(mesh)
    # The buffers' own types: arrays in the LLVM mode, but in the plain mode
    # mi.Float is ONE number, and handing it an array is a TypeError.
    floats, ints = type(params["vertex_positions"]), type(params["faces"])
    params["vertex_positions"] = floats(np.asarray(surface["points"], np.float32).ravel())
    params["faces"] = ints(np.asarray(surface["faces"], np.uint32).ravel())
    if normals:
        params["vertex_normals"] = floats(np.asarray(surface["normals"], np.float32).ravel())
    params.update()
    if per_point or per_face:
        mesh.add_attribute("vertex_color" if per_point else "face_color", 3, floats(linear(colours).astype(np.float32).ravel()))
    return mesh


def _curve_files(curves, scratch, chunk=8, levels=32):
    """Fibres and lines as Mitsuba curve files, one per colour: each polyline is
    cut into short runs, each run takes its own colour (so a direction colouring
    still changes along a fibre), and runs of one colour share a file.
    Returns ``[(linear colour, file)]``; files are kept per geometry, so moving
    the camera does not write them again."""
    key = hashlib.sha1(json.dumps([str(curves["key"]), curves["radius"], chunk, levels]).encode()).hexdigest()[:16]
    folder = scratch / key
    index = folder / "index.json"
    if index.is_file():
        return [(np.asarray(c), str(folder / f)) for c, f in json.loads(index.read_text())]
    folder.mkdir(parents=True, exist_ok=True)
    colours = curves["colours"]
    groups = {}
    for ids, points in curves["polylines"]:
        for start in range(0, len(points) - 1, chunk):
            run = slice(start, min(start + chunk + 1, len(points)))
            rgb = colours[ids[run]].mean(0) if colours.ndim == 2 else colours
            groups.setdefault(tuple(np.round(np.asarray(rgb) * (levels - 1)).astype(int)), []).append(points[run])
    radius = float(curves["radius"])
    listing = []
    for n, (code, runs) in enumerate(groups.items()):
        name = f"g{n}.txt"
        with open(folder / name, "w") as handle:
            for run in runs:
                np.savetxt(handle, np.column_stack([run, np.full(len(run), radius)]), fmt="%.4f")
                handle.write("\n")
        listing.append((linear(np.asarray(code) / (levels - 1)).tolist(), name))
    index.write_text(json.dumps(listing))
    return [(np.asarray(c), str(folder / f)) for c, f in listing]


def _slice(item):
    """A slice as a two-sided textured rectangle; its dropped-out background is
    cut away, so the slice still sits inside the brain."""
    import mitsuba as mi

    o, a, b = (np.asarray(c, float) for c in item["corners"])
    u, v = a - o, b - o
    normal = np.cross(u, v)
    normal /= max(np.linalg.norm(normal), 1e-12)
    centre = o + u / 2 + v / 2
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = u / 2, v / 2, normal, centre
    # Row 0 of the slice's texture is at the quad's v = 0, as Mitsuba reads a bitmap too.
    rgba = np.asarray(item["rgba"], float) / 255.0
    colour = mi.Bitmap(np.ascontiguousarray(linear(rgba[..., :3]).astype(np.float32)))
    alpha = mi.Bitmap(np.ascontiguousarray((rgba[..., 3:4] * item["opacity"]).astype(np.float32)))
    surface = {"type": "twosided", "material": {"type": "diffuse", "reflectance": {"type": "bitmap", "bitmap": colour, "raw": True, "filter_type": "bilinear"}}}
    return {"type": "rectangle", "to_world": mi.ScalarTransform4f(matrix.tolist()),
            "bsdf": {"type": "mask", "opacity": {"type": "bitmap", "bitmap": alpha, "raw": True, "filter_type": "bilinear"}, "material": surface}}


def _light_directions(snap, cam):
    """The viewer's rig as ``[(direction from the scene to the light, intensity, linear colour)]``."""
    from matplotlib.colors import to_rgb

    from gnv.render.render_scene import light_rig

    lights = snap["lights"]
    rig = light_rig(lights.get("key_azimuth", 35.0), lights.get("key_elevation", 35.0), lights.get("key_intensity", 0.85), lights.get("rig", "studio"))
    forward, right, up = _frame(cam)
    out = []
    for azimuth, elevation, intensity, colour in rig:
        az, el = np.radians(azimuth), np.radians(elevation)
        # Camera space as the viewer places its lights: +x right, +y up, +z towards the lens.
        d = np.sin(az) * np.cos(el) * right + np.sin(el) * up - np.cos(az) * np.cos(el) * forward
        out.append((d / np.linalg.norm(d), float(intensity), linear(to_rgb(colour))))
    return out


def _lights(snap, cam, bounds, radius, opts):
    """Square area lights where the viewer's lights are, far enough to light
    the scene evenly, sized by ``light_size``. A light of intensity I gives a
    surface facing it the same brightness the viewer's directional light does."""
    import mitsuba as mi

    centre = (bounds[0] + bounds[1]) / 2
    distance = 4.0 * radius
    half = max(opts["light_size"] * radius / 2, 1e-3)
    out = {}
    for n, (d, intensity, colour) in enumerate(_light_directions(snap, cam)):
        if intensity <= 0:
            continue
        # Irradiance at the centre is L * area / distance^2 for a small light;
        # a diffuse surface shows E / pi of it, and the viewer shows I.
        radiance = np.pi * intensity * distance ** 2 / (4 * half * half)
        place = mi.ScalarTransform4f().look_at(origin=(centre + d * distance).tolist(), target=centre.tolist(), up=_any_up(d).tolist())
        out[f"light{n}"] = {"type": "rectangle", "to_world": place @ mi.ScalarTransform4f().scale([half, half, 1.0]), "flip_normals": False,
                            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": (colour * radiance).tolist()}}}
    return out


def _backdrop(snap, cam, bounds, radius, opts, lights):
    """A wall behind the scene, facing the camera, that catches its shadow. Its
    colour is set so that where the lights reach it fully it shows the Scene
    background: how brightly they reach it is measured by a tiny render of the
    wall alone, and its colour scaled to match."""
    import mitsuba as mi
    from matplotlib.colors import to_rgb

    forward, _right, up = _frame(cam)
    lo, hi = bounds
    box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    depth = float(np.max((box - cam["focal_point"]) @ forward))
    centre = cam["focal_point"] + forward * (depth + 0.25 * radius)
    place = mi.ScalarTransform4f().look_at(origin=centre.tolist(), target=(centre - forward).tolist(), up=up.tolist())
    # A flat raw texture (2 x 2, Mitsuba's smallest) rather than an "rgb" colour, which Mitsuba holds to 0..1.
    wall = lambda colour: {"type": "rectangle", "to_world": place @ mi.ScalarTransform4f().scale([radius * 20, radius * 20, 1.0]),
                           "bsdf": {"type": "diffuse", "reflectance": {"type": "bitmap", "raw": True, "bitmap": mi.Bitmap(np.tile(np.asarray(colour, np.float32), (2, 2, 1)))}}}
    probe = {"type": "scene", "integrator": {"type": "path", "max_depth": 2, "hide_emitters": True}, "wall": wall([1.0, 1.0, 1.0]), **lights,
             "sensor": dict(_sensor(dict(cam, size=(48, 36))), sampler={"type": "independent", "sample_count": 16})}
    if opts["ambient"] > 0:
        probe["ambient"] = {"type": "constant", "radiance": {"type": "rgb", "value": opts["ambient"]}}
    # The lights are near enough that the wall falls off away from the key, as
    # a studio wall does; its LIT part is matched to the background (the
    # brightest tenth), so the shadow and the fall-off read as light, not as a
    # different colour. A colour above 1 is allowed: the wall only has to look
    # right, not conserve energy.
    seen = np.percentile(np.asarray(mi.render(mi.load_dict(probe)), float)[..., :3].reshape(-1, 3), 90, axis=0)
    return wall(np.clip(linear(to_rgb(snap["background"])) / np.maximum(seen, 1e-3), 0.0, 4.0))


def _any_up(direction):
    """An up vector not parallel to ``direction``, for placing a light."""
    return np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([0.0, 1.0, 0.0])

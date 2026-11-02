"""HTTP surface for the viewer. Local only, one session, no state on the client.

The browser is a control panel, not a renderer: it holds the spec, posts it back
on every edit, and asks for a PNG. All the geometry stays in this process, which
is what lets a layer edit reuse the caches in ``RenderSession`` instead of
shipping meshes over the wire.

Bound to 127.0.0.1 by design. It exposes a file browser and writes files where
you tell it to, which is fine for a tool running as you on your own machine and
would not be fine on an interface anyone else can reach.
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from threading import Lock, Thread, Timer

from flask import Flask, jsonify, request, send_from_directory

from . import scratch
from .session import RenderSession, catalogue
from .spec import (
    new_electrode_layer, new_volume_layer,
    DEFAULT_LIGHTING, SpecError, load_spec, new_fiber_layer, new_mesh_layer,
    new_overlay, new_slice_layer, new_spec, normalize_spec, save_spec,
)

STATIC = Path(__file__).resolve().parent / "static"
MESH_SUFFIXES = {".vtk", ".vtp", ".stl", ".ply", ".obj", ".gii"}
VOLUME_SUFFIXES = {".nii", ".gz", ".npz"}
FIBER_SUFFIXES = {".mat", ".npy", ".tck", ".trk", ".json"}


def _sha256(path, chunk=1 << 20):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _drop_target(drops, name, item):
    """Save an upload into the drop folder WITHOUT clobbering a different file.

    The folder was keyed by filename alone, and two unrelated maps are very
    often called the same thing -- contrast_tval_FWE_0.nii.gz out of two
    different analyses. The second drop overwrote the first, so both overlays
    ended up pointing at one path and the older overlay silently started
    drawing the newer file's data. Deleting the new overlay then left the old
    one showing the new one's distribution, which looks exactly like one
    overlay mutating another and is not that at all.

    Same name and same bytes reuses the existing file. Same name, different
    bytes gets its own name, so every overlay keeps pointing at what it was
    given.
    """
    import os
    from uuid import uuid4

    drops.mkdir(parents=True, exist_ok=True)
    staged = drops / f".incoming-{uuid4().hex}"
    try:
        item.save(staged)
    except BaseException:
        staged.unlink(missing_ok=True)      # a broken upload leaves nothing
        raise

    target = drops / name
    if not target.exists():
        os.replace(staged, target)
        return target
    if _sha256(staged) == _sha256(target):
        staged.unlink()
        scratch.touch(target)               # used again: not idle
        return target

    # ".nii.gz" has to survive whole, so the split is at the FIRST dot.
    stem, dot, ext = name.partition(".")
    tag = _sha256(staged)[:8]
    target = drops / (f"{stem}.{tag}{dot}{ext}" if dot else f"{stem}.{tag}")
    os.replace(staged, target)
    return target


def _write_gif(frames, path, fps=None, durations=None):
    """Frames to an animated GIF that loops. Each frame gets its own 256-colour
    palette, which keeps a coloured overlay from banding against grey anatomy
    the way one shared palette would."""
    if not frames:
        raise ValueError("no frames to write")
    path = Path(path).expanduser()
    if path.suffix.lower() != ".gif":
        path = path.with_suffix(".gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    size = frames[0].size
    palettised = [f.resize(size) if f.size != size else f for f in frames]
    palettised = [f.quantize(colors=256, method=2, dither=1) for f in palettised]
    if durations:
        duration = [max(int(round(d)), 20) for d in durations][:len(palettised)]
        duration += [duration[-1] if duration else 66] * (len(palettised) - len(duration))
    else:
        duration = max(int(round(1000.0 / float(fps or 15))), 20)
    palettised[0].save(path, save_all=True, append_images=palettised[1:],
                       duration=duration, loop=0, disposal=2, optimize=False)
    return str(path)


def _spec_paths(value):
    """Every string in a spec, for the scratch protector: a dropped file stays
    while any layer or overlay of the open figure still names it."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _spec_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _spec_paths(item)


def create_app(spec=None, drop_dir=None):
    app = Flask(__name__, static_folder=None)
    # Drops are temporary: they live in this run's scratch folder (see
    # scratch.py), are collected once nothing uses them, and go with the run.
    app.config["DROP_DIR"] = str(drop_dir or scratch.area("drops"))
    Path(app.config["DROP_DIR"]).mkdir(parents=True, exist_ok=True)
    session = RenderSession(spec or new_spec())
    lock = Lock()
    scratch.protect(lambda: [p for p in _spec_paths(session.spec)
                             if p.startswith(str(scratch.SCRATCH_ROOT))])

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC, name)

    @app.get("/api/catalogue")
    def api_catalogue():
        return jsonify(catalogue())

    @app.get("/api/spec")
    def api_get_spec():
        return jsonify(session.spec)

    @app.post("/api/spec")
    def api_set_spec():
        try:
            with lock:
                return jsonify(session.set_spec(request.get_json(force=True)))
        except SpecError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/layer/<kind>")
    def api_new_layer(kind):
        makers = {"mesh": (new_mesh_layer, "Mesh"),
                  "fibers": (new_fiber_layer, "Fibers"),
                  "slices": (new_slice_layer, "Slices"),
                  "electrode": (new_electrode_layer, "Electrode"),
                  "volume": (new_volume_layer, "Volume")}
        if kind not in makers:
            return jsonify({"error": f"unknown layer kind {kind!r}"}), 400
        maker, label = makers[kind]
        return jsonify(maker(name=label))

    @app.get("/api/volumes")
    def api_volumes():
        """Packaged bodies and prepared backdrops for the volume layer's picker."""
        from circuit_viewer.volume_render import library

        return jsonify(library())

    @app.get("/api/slices/info")
    def api_slice_info():
        """Bounds, axes and voxel size of any slice backdrop -- a temporary
        NIfTI included -- in the shape the backdrop library lists them."""
        reference = request.args.get("backdrop", "")
        try:
            with lock:
                back = session.backdrop(reference)
                lo, hi = back.bounds()
                info = {"path": reference, "name": Path(reference).name,
                        "bounds": [[float(v) for v in lo], [float(v) for v in hi]],
                        "world_axes": [back.world_axis(a)[0] for a in range(3)],
                        "mm": float(back.mm)}
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify(info)

    @app.get("/api/volumes/labels")
    def api_volume_labels():
        """The regions of a label atlas used as a body, named from its LUT:
        ``{"labels": [...], "table": path or None}``."""
        from circuit_viewer.volume_render import atlas_labels

        return jsonify(atlas_labels(request.args.get("source", ""),
                                    request.args.get("lut") or None))

    @app.post("/api/volumes/levels")
    def api_volume_levels():
        """Build a large body's levels as a background job, polled on the same
        job endpoint as a conversion, so its first render does not stall."""
        from .backdrop_jobs import start_levels

        body = request.get_json(force=True)
        try:
            return jsonify(start_levels(body["source"]))
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400

    @app.post("/api/upload")
    def api_upload():
        """Accept a dropped file and hand back a path the renderer can open.

        A browser will not tell a page where a dropped file lives on disk, so
        the bytes come across and land in a scratch folder for this session. For
        a lesion mask that is a moment; the file picker stays the better route
        for anything large, and for anything you want the saved project to keep
        pointing at after this scratch folder is gone.
        """
        item = request.files.get("file")
        if item is None or not item.filename:
            return jsonify({"error": "no file in the request"}), 400
        name = Path(item.filename).name
        allowed = (".nii", ".nii.gz", ".mat", ".npy", ".npz", ".tck", ".trk",
                   ".trx", ".json", ".lut", ".txt", ".tsv", ".csv")
        if not name.lower().endswith(allowed):
            return jsonify(
                {"error": f"{name} is not a volume, tractogram, lead or "
                          f"fibre descriptor"}), 400
        scratch.sweep()
        target = _drop_target(Path(app.config["DROP_DIR"]), name, item)
        # A tractogram and an electrode reconstruction are both ".mat", so what
        # this is has to be read out of the file rather than off its name. Done
        # here, once, so the browser can route the drop without a second call.
        kind = "volume"
        lower = name.lower()
        if lower.endswith(".fib.npy"):
            kind = "fiber_values"
        elif lower.endswith((".mat", ".json")):
            from calvin_utils.plotting_utils.electrode_render import (
                looks_like_electrode,
            )
            from calvin_utils.plotting_utils.fiber_render import is_descriptor

            # A .json here is either a fibre-value descriptor or an electrode
            # reconstruction, and a .mat is either a lead or a tractogram. All
            # three are told apart by what is inside, never by the extension.
            if is_descriptor(target):
                kind = "fiber_desc"
            else:
                kind = "electrode" if looks_like_electrode(target) else "tract"
        elif lower.endswith((".tck", ".trk", ".trx", ".npz", ".npy")):
            kind = "tract"
        elif lower.endswith((".lut", ".txt", ".tsv", ".csv")):
            # An atlas's lookup table, dropped with it so the two sit side by
            # side in the drop folder and its regions can be listed by name.
            kind = "lut"
        return jsonify({"path": str(target), "name": name, "kind": kind})

    @app.get("/api/descriptor")
    def api_descriptor():
        """Resolve a .fib.json to the atlas and values file it names.

        Done server-side because it is the descriptor's own promises that are
        being checked -- the values' SHA-256 and the atlas' size -- and those
        need the files, which the browser never has.
        """
        from calvin_utils.plotting_utils.fiber_render import read_descriptor

        path = request.args.get("path", "")
        try:
            found = read_descriptor(path)
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify({"atlas": str(found["atlas"]),
                        "values": str(found["values"]),
                        "name": found["name"],
                        "fiber_count": found["fiber_count"]})

    @app.get("/api/volume")
    def api_volume():
        """What is in a volume, so the UI can suggest a split before meshing."""
        from .volume_info import summarise

        try:
            return jsonify(summarise(request.args["path"]))
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400

    @app.get("/api/backdrops")
    def api_backdrops():
        from .backdrop_jobs import catalogue

        return jsonify(catalogue())

    @app.post("/api/backdrops")
    def api_build_backdrop():
        """Start a conversion. It returns at once: a 100 um volume takes minutes,
        and a request that long would simply time out."""
        from .backdrop_jobs import start

        body = request.get_json(force=True)
        try:
            mm = body.get("mm")
            return jsonify(start(body["path"],
                                 mm=None if mm in (None, "", 0) else float(mm),
                                 name=body.get("name")))
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400

    @app.get("/api/backdrops/job")
    def api_backdrop_job():
        from .backdrop_jobs import status as job_status

        return jsonify(job_status())

    @app.get("/api/lighting")
    def api_default_lighting():
        return jsonify(dict(DEFAULT_LIGHTING))

    @app.get("/api/regions")
    def api_regions():
        """The regions inside a mesh group, so they can be listed and toggled."""
        from .session import region_names

        q = request.args
        try:
            mesh = json.loads(q["mesh"]) if q.get("mesh", "").startswith("[") \
                else q["mesh"]
            return jsonify({"regions": region_names(
                mesh, iso=float(q["iso"]) if q.get("iso") else None,
                split=q.get("split", "auto"))})
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400

    @app.get("/api/palettes")
    def api_palettes():
        from calvin_utils.plotting_utils.palettes import (
            CLUT_DIRS, palette_catalogue,
        )

        out = palette_catalogue()
        out["dirs"] = [str(d) for d in CLUT_DIRS]
        return jsonify(out)

    @app.get("/api/overlay")
    def api_new_overlay():
        return jsonify(new_overlay())

    @app.get("/api/render")
    def api_render():
        q = request.args
        camera = {k: float(q[k]) for k in ("azimuth", "elevation", "zoom") if k in q}
        try:
            with lock:
                png = session.render(
                    size=(int(q.get("w", 760)), int(q.get("h", 640))),
                    quality=q.get("quality", "preview"),
                    # Which bars are drawn is a per-overlay toggle in the
                    # spec now, not a scene-wide flag, so the render always
                    # honours them. "bar=0" is still accepted as a way to
                    # suppress every bar at once.
                    scalar_bar=q.get("bar", "1") != "0",
                    camera=camera or None,
                    antialias=q.get("aa"),
                    line_scale=float(q.get("ls", 1) or 1),
                )
        except Exception as exc:                     # surfaced in the UI status line
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        headers = {"Cache-Control": "no-store"}
        # Non-fatal complaints ride back on a header rather than in the body,
        # because the body is the PNG and the render succeeded. The browser puts
        # them in the status line, where an error would have gone.
        warnings = session.warnings
        if warnings:
            headers["X-Render-Warnings"] = json.dumps(warnings)
        # How big the view is in the world and which way is right / up, so a
        # drag in the browser pans by exactly the distance the cursor moved.
        if session.view:
            headers["X-View"] = json.dumps(session.view)
        return app.response_class(png, mimetype="image/png", headers=headers)

    @app.get("/api/frames")
    def api_frames():
        """How many frames a map holds (4D NIfTI, or fibres x time values)."""
        from .frames import frame_count

        try:
            return jsonify({"frames": frame_count(request.args["path"])})
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400

    # -- animation -----------------------------------------------------------
    # A camera path (or a series played through) rendered frame by frame into a
    # GIF, in the background: a few hundred volume renders is minutes, not a
    # request. One job at a time; progress is polled like a conversion.
    anim = {"running": False, "done": 0, "total": 0, "path": None,
            "error": None}

    @app.post("/api/animate")
    def api_animate():
        body = request.get_json(force=True)
        if anim["running"]:
            return jsonify({"error": "already rendering an animation"}), 400
        try:
            steps = list(body["steps"])
            out = str(Path(body["path"]).expanduser())
            size = (int(body.get("width", 800)), int(body.get("height", 640)))
            fps = max(float(body.get("fps", 15)), 1.0)
            quality = body.get("quality", "preview")
            line_scale = float(body.get("line_scale", 1) or 1)
            base = normalize_spec(body.get("spec") or session.spec)
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        anim.update(running=True, done=0, total=len(steps), path=out,
                    error=None)

        def run():
            import copy
            import io as _io

            from PIL import Image

            frames = []
            try:
                with lock:
                    saved = session.spec
                    try:
                        for step in steps:
                            spec = copy.deepcopy(base)
                            if step.get("pan") is not None:
                                spec["pan"] = step["pan"]
                            for layer_id, entries in (step.get("frames") or {}).items():
                                for layer in spec["layers"]:
                                    if layer["id"] != layer_id:
                                        continue
                                    for index, frame in entries.items():
                                        layer["overlays"][int(index)]["frame"] = int(frame)
                            session.set_spec(spec)
                            camera = {k: float(step[k]) for k in
                                      ("azimuth", "elevation", "zoom") if k in step}
                            png = session.render(size=size, quality=quality,
                                                 camera=camera,
                                                 line_scale=line_scale)
                            frames.append(Image.open(_io.BytesIO(png)).convert("RGB"))
                            anim["done"] += 1
                    finally:
                        session.set_spec(saved)
                _write_gif(frames, out, fps)
            except Exception as exc:
                anim["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                anim["running"] = False

        Thread(target=run, daemon=True).start()
        return jsonify(dict(anim))

    @app.get("/api/animate")
    def api_animate_status():
        return jsonify(dict(anim))

    @app.post("/api/gif")
    def api_gif():
        """Assemble a screen recording -- the frames the browser displayed,
        with how long each stayed on screen -- into a GIF."""
        from PIL import Image

        try:
            out = str(Path(request.form["path"]).expanduser())
            durations = json.loads(request.form.get("durations") or "[]")
            files = request.files.getlist("frames")
            frames = [Image.open(f.stream).convert("RGB") for f in files]
            if not frames:
                return jsonify({"error": "nothing was recorded"}), 400
            _write_gif(frames, out, None, durations)
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify({"path": out, "frames": len(frames)})

    @app.post("/api/export")
    def api_export():
        body = request.get_json(force=True)
        try:
            with lock:
                path = session.export(
                    body["path"],
                    size=(int(body.get("width", 4000)), int(body.get("height", 3200))),
                    scalar_bar=bool(body.get("scalar_bar", True)),
                    line_scale=float(body.get("line_scale", 1) or 1),
                )
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify({"path": path,
                        "width": int(body.get("width", 4000)),
                        "height": int(body.get("height", 3200))})

    @app.post("/api/project")
    def api_save_project():
        body = request.get_json(force=True)
        try:
            path = save_spec(body.get("spec") or session.spec, body["path"])
        except (OSError, SpecError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"path": str(path)})

    @app.get("/api/project")
    def api_load_project():
        try:
            with lock:
                return jsonify(session.set_spec(load_spec(request.args["path"])))
        except (OSError, SpecError, json.JSONDecodeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/browse")
    def api_browse():
        """Directory listing, so paths can be picked instead of typed."""
        root = Path(request.args.get("path") or Path.home()).expanduser()
        if root.is_file():
            root = root.parent
        if not root.is_dir():
            return jsonify({"error": f"{root} is not a directory"}), 400
        want = request.args.get("kind", "any")
        suffixes = {"mesh": MESH_SUFFIXES, "volume": VOLUME_SUFFIXES,
                    "fibers": FIBER_SUFFIXES}.get(want)

        entries = []
        try:
            for item in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    entries.append({"name": item.name, "path": str(item), "dir": True})
                elif suffixes is None or item.suffix.lower() in suffixes:
                    entries.append({"name": item.name, "path": str(item),
                                    "dir": False, "size": item.stat().st_size})
        except PermissionError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"path": str(root), "parent": str(root.parent),
                        "entries": entries[:2000]})

    # Optional add-on, self-contained in lineart/ (its own routes). Delete
    # these lines to remove it; the viewer runs the same with or without it.
    try:
        from .lineart import attach as _attach_lineart
        _attach_lineart(app, session, lock)
    except Exception as exc:
        print(f"circuit_viewer: line art add-on not loaded: {exc!r}")

    return app


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="circuit-viewer",
                                     description="Layered brain figure viewer.")
    parser.add_argument("project", nargs="?", help="a .json layer spec to open")
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    try:
        import calvin_utils.plotting_utils.brains  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "circuit_viewer draws with circuit_pyper's renderer, which is not "
            f"importable here ({exc}).\n"
            "Install it, or run this from an environment that has it:\n"
            "    pip install -e /path/to/calvin_utils_project/circuit_pyper"
        )

    spec = load_spec(args.project) if args.project else None
    app = create_app(spec)
    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"circuit viewer -> {url}")
    app.run(host=args.host, port=args.port, threaded=True)

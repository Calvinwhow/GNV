"""The add-on's own endpoints. Nothing here is called by the viewer; the page's
pathtrace.js calls them.

    GET  /pathtrace/pathtrace.js   the page script
    POST /pathtrace/start          path trace the current view; answers at once
    GET  /pathtrace/frame          the picture so far (PNG), with how far it has got
    POST /pathtrace/stop           stop rendering
    POST /pathtrace/export         path trace the view at export size into a file
    GET  /pathtrace/status         how far the current render (or export) has got

One render runs at a time: starting another stops the one before. The view is
copied out of the viewer at once (``viewer.snapshot``, under the viewer's lock)
and the path tracing itself runs on a thread of its own, so the viewer stays
free while it works.
"""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent

_state = {"job": 0, "samples": 0, "target": 0, "done": True, "error": None, "path": None, "notes": [], "variant": None, "png": None, "cancel": None}
_lock = threading.Lock()          # guards _state
_mitsuba = threading.Lock()       # one Mitsuba render at a time


def attach(app, session, lock):
    """Register the path-tracing routes on ``app``. ``lock`` is the viewer's
    render lock, held while the view is copied out of it."""
    from flask import Blueprint, jsonify, request, send_from_directory

    from . import engine, viewer

    bp = Blueprint("pathtrace", __name__)

    def start(size, camera, options, line_scale, path=None):
        opts = engine.settings(options)
        with lock:
            snap = viewer.snapshot(session, size, camera, line_scale)
        cancel = threading.Event()
        with _lock:
            if _state["cancel"] is not None:
                _state["cancel"].set()
            _state.update(job=_state["job"] + 1, samples=0, target=opts["samples"], done=False, error=None, path=None, notes=snap["notes"], png=None, cancel=cancel)
            job = _state["job"]
        threading.Thread(target=_run, args=(job, snap, opts, path, cancel), daemon=True).start()
        return job

    def _camera(raw):
        return {k: float(v) for k, v in (raw or {}).items() if k in ("azimuth", "elevation", "zoom")}

    @bp.get("/pathtrace/pathtrace.js")
    def script():
        return send_from_directory(HERE / "static", "pathtrace.js", mimetype="text/javascript", max_age=0)

    @bp.post("/pathtrace/start")
    def start_render():
        body = request.get_json(force=True) or {}
        try:
            job = start((int(body.get("w", 760)), int(body.get("h", 640))), _camera(body.get("camera")), body.get("options") or {}, float(body.get("ls", 1) or 1))
        except Exception as exc:
            return jsonify({"error": _failure(exc)}), 400
        return jsonify({"job": job})

    @bp.get("/pathtrace/frame")
    def frame():
        with _lock:
            info = {k: _state[k] for k in ("job", "samples", "target", "done", "error", "notes", "variant")}
            png = _state["png"]
        headers = {"Cache-Control": "no-store", "X-Pathtrace": json.dumps(info)}
        if png is None:
            return app.response_class(b"", status=204, headers=headers)
        return app.response_class(png, mimetype="image/png", headers=headers)

    @bp.post("/pathtrace/stop")
    def stop():
        with _lock:
            if _state["cancel"] is not None:
                _state["cancel"].set()
            _state["done"] = True
        return jsonify({"stopped": True})

    @bp.post("/pathtrace/export")
    def export():
        body = request.get_json(force=True) or {}
        try:
            path = Path(str(body["path"])).expanduser()
            if path.suffix.lower() != ".png":
                path = path.with_suffix(".png")
            size = (int(body.get("width", 4000)), int(body.get("height", 3200)))
            job = start(size, _camera(body.get("camera")), body.get("options") or {}, float(body.get("line_scale", 1) or 1), path)
        except Exception as exc:
            return jsonify({"error": _failure(exc)}), 400
        return jsonify({"job": job, "path": str(path), "width": size[0], "height": size[1]})

    @bp.get("/pathtrace/status")
    def status():
        with _lock:
            return jsonify({k: _state[k] for k in ("job", "samples", "target", "done", "error", "path", "notes", "variant")})

    app.register_blueprint(bp)


def _run(job, snap, opts, path, cancel):
    """One render, on its own thread: build the scene, render it pass by pass,
    publish each pass (or, for an export, write the file at the end)."""
    from gnv import scratch

    from . import engine

    def current():
        with _lock:
            return _state["job"] == job and not cancel.is_set()

    def publish(rgba, samples):
        if not current():
            return
        png = None if path else _png(engine.compose(rgba, snap["background"], snap["bars"], opts["exposure"]))
        with _lock:
            if _state["job"] == job:
                _state["samples"] = samples
                if png is not None:
                    _state["png"] = png

    try:
        with _mitsuba:
            if not current():
                return
            with _lock:
                _state["variant"] = engine.variant()
            scene = engine.build_scene(snap, opts, scratch.area("pathtrace"))
            final = engine.render(scene, snap["camera"]["size"], opts, publish, lambda: not current())
        finished = final is not None and current()
        if path and finished:
            Path(path).write_bytes(_png(engine.compose(final, snap["background"], snap["bars"], opts["exposure"])))
        with _lock:
            if _state["job"] == job:
                _state.update(done=True, path=str(path) if path and finished else None)
    except Exception as exc:
        failure = _failure(exc)
        with _lock:
            if _state["job"] == job:
                _state.update(done=True, error=failure)


def _failure(exc):
    """The error for the status line, naming the line of gnv it came from, with
    the whole traceback written to stderr -- the terminal, or launcher.log in
    the packaged app -- so a report says where it happened, not only what."""
    import sys
    import traceback

    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    frames = traceback.extract_tb(exc.__traceback__)
    ours = [f for f in frames if "gnv" in f.filename.replace("\\", "/").split("/")]
    where = ours[-1] if ours else (frames[-1] if frames else None)
    at = f" (at {where.filename.replace(chr(92), '/').split('/gnv/')[-1]}:{where.lineno}, {where.name})" if where else ""
    return f"path traced: {type(exc).__name__}: {exc}{at}"


def _png(image):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG", compress_level=3)
    return buf.getvalue()

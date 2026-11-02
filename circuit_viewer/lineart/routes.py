"""The add-on's own endpoints. Nothing here is called by the viewer; the page's
lineart.js calls them.

    GET  /lineart/lineart.js   the page script
    GET  /lineart/render       the current view as line art (PNG)
    POST /lineart/export       write it to a file: .svg (vectors) or .png
"""

from __future__ import annotations

import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def attach(app, session, lock):
    """Register the line-art routes on ``app``. ``lock`` is the viewer's
    render lock, held while a frame is drawn so nothing edits the scene
    under it."""
    from flask import Blueprint, jsonify, request, send_from_directory

    from . import engine, viewer

    bp = Blueprint("lineart", __name__)

    def frame(size, camera, options, line_scale):
        viewer.bring_up_to_date(session, size, camera)
        return viewer.on_render_thread(
            session, engine.draw, viewer.View(session), size, line_scale, options)

    @bp.get("/lineart/lineart.js")
    def script():
        return send_from_directory(HERE / "static", "lineart.js",
                                   mimetype="text/javascript", max_age=0)

    @bp.get("/lineart/render")
    def render():
        q = request.args
        try:
            size = (int(q.get("w", 760)), int(q.get("h", 640)))
            camera = {k: float(q[k]) for k in ("azimuth", "elevation", "zoom") if k in q}
            options = json.loads(q.get("o") or "{}")
            with lock:
                image, _runs, notes = frame(size, camera, options,
                                            float(q.get("ls", 1) or 1))
        except Exception as exc:
            return jsonify({"error": f"line art: {type(exc).__name__}: {exc}"}), 400
        return app.response_class(_png(image), mimetype="image/png", headers={
            "Cache-Control": "no-store",
            "X-Lineart-Notes": json.dumps(notes)})

    @bp.post("/lineart/export")
    def export():
        body = request.get_json(force=True) or {}
        try:
            path = Path(str(body["path"])).expanduser()
            size = (int(body.get("width", 4000)), int(body.get("height", 3200)))
            camera = {k: float(v) for k, v in (body.get("camera") or {}).items()
                      if k in ("azimuth", "elevation", "zoom")}
            with lock:
                image, runs, _notes = frame(size, camera, body.get("options") or {},
                                            float(body.get("line_scale", 1) or 1))
                background = viewer.View(session).background
            if path.suffix.lower() == ".svg":
                path.write_text(engine.svg(runs, size, background))
            else:
                path.write_bytes(_png(image))
        except Exception as exc:
            return jsonify({"error": f"line art: {type(exc).__name__}: {exc}"}), 400
        return jsonify({"path": str(path), "width": size[0], "height": size[1]})

    app.register_blueprint(bp)


def _png(image):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG", compress_level=3)
    return buf.getvalue()

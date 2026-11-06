"""Path traced -- an optional, self-contained add-on to the General Neuroimage Viewer.

Renders the current view with a physically based path tracer (Mitsuba 3)
instead of the interactive renderer: soft shadows, light bouncing between
surfaces, real tubes for fibres. It is slow -- seconds per view, minutes for a
full-size figure -- so it is switched on for the figure, not for exploring:
the interactive view is what you move around in, and the path-traced picture
is laid over it once the view settles, refining as more samples arrive.

Kept apart like line art. The viewer is unchanged by it except for two lines,
both of which may be deleted to remove it entirely:

    app.py             attach(app, session, lock)       -- its own routes
    static/index.html  <script src="/pathtrace/pathtrace.js">  -- its own button

Everything else lives in this folder:

    viewer.py            the only code that reads the viewer (read-only)
    engine.py            the Mitsuba scene and the progressive render
    routes.py            /pathtrace/... endpoints, and the background job
    static/pathtrace.js  the toggle, the settings panel and the overlay image
"""

from .routes import attach

__all__ = ["attach"]

"""Line art -- an optional, self-contained add-on to the circuit viewer.

Draws the current view as contour lines, hatching and outlines (after Michael
Fogleman's ``ln``) instead of shaded surfaces, and exports it as an SVG.

It is kept apart on purpose. The viewer is unchanged by it except for two
lines, both of which may be deleted to remove it entirely:

    app.py           attach(app, session, lock)    -- its own routes
    static/index.html  <script src="/lineart/lineart.js">  -- its own button

Everything else lives in this folder:

    engine.py          the drawing; knows nothing about the viewer
    viewer.py          the only code that reads the viewer (read-only)
    routes.py          /lineart/... endpoints
    static/lineart.js  the toggle, the settings panel and the overlay image;
                       it adds them to the page itself
"""

from .routes import attach

__all__ = ["attach"]

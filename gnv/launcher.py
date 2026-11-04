"""Open the viewer in a window of its own: the double-click way in.

``gnv`` (``app.main``) serves the viewer and opens a browser tab.
This does the same job the way a desktop app should:

1. If a viewer this launcher started is still running, it opens another window
   onto it and leaves -- one server however many times it is launched.
2. Otherwise it serves the viewer on 127.0.0.1, port 8731 when free (a fixed
   address keeps the page's saved settings, which a browser keeps per address),
   any free port when not.
3. It opens the page in Chrome or Edge as an app window (``--app``: no tabs, no
   address bar) with a browser profile of its own, so the window belongs to a
   browser instance nothing else uses.
4. When the viewer window is closed, that browser instance is quit, the server
   stops and this process exits: nothing is left running.

Step 4 cannot wait for the browser process to end: on macOS closing a browser's
last window leaves the browser running. So the private instance is started with
a debugging port on 127.0.0.1, and its list of open pages is polled; when no
page shows the viewer any more, the window was closed.

With neither Chrome nor Edge installed, the page opens as a tab in the default
browser; a tab's closing cannot be seen from here, so the server then runs until
this process is stopped (Ctrl-C).

The browser profile, the record of the running server and the log (when there
is no console to print to) live in ``paths.USER_DIR``.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from threading import Thread

from .paths import USER_DIR

HOST = "127.0.0.1"
PREFERRED_PORT = 8731
PROFILE_DIR = USER_DIR / "browser-profile"
SERVER_RECORD = USER_DIR / "launcher.json"
LOG_FILE = USER_DIR / "launcher.log"


# -- the browser ----------------------------------------------------------------

def browser_candidates():
    """Chrome and Edge executables where each platform installs them, most preferred first."""
    if sys.platform == "darwin":
        apps = [Path("/Applications"), Path.home() / "Applications"]
        names = [("Google Chrome.app", "Google Chrome"), ("Microsoft Edge.app", "Microsoft Edge"), ("Chromium.app", "Chromium")]
        return [root / app / "Contents" / "MacOS" / exe for app, exe in names for root in apps]
    if sys.platform == "win32":
        roots = [os.environ.get(k) for k in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)")]
        names = [Path("Google") / "Chrome" / "Application" / "chrome.exe", Path("Microsoft") / "Edge" / "Application" / "msedge.exe"]
        return [Path(root) / name for name in names for root in roots if root]
    found = [shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge")]
    return [Path(p) for p in found if p]


def find_app_browser():
    """The first Chrome or Edge on this machine, or None."""
    return next((p for p in browser_candidates() if p.is_file()), None)


def debugging_port():
    """The debugging port of the browser instance running on ``PROFILE_DIR``, or
    None. The browser writes it to DevToolsActivePort when it starts."""
    try:
        port = int((PROFILE_DIR / "DevToolsActivePort").read_text().split()[0])
        urllib.request.urlopen(f"http://{HOST}:{port}/json/version", timeout=1).read()
        return port
    except (OSError, ValueError, IndexError):
        return None


def open_app_window(browser, url):
    """Open ``url`` as an app window of the private browser instance, starting it
    if it is not running. ``(process or None, debugging port or None)``: the
    process is None when an instance already running took the window."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    running = debugging_port()
    if running is None:
        (PROFILE_DIR / "DevToolsActivePort").unlink(missing_ok=True)     # a dead instance's
    command = [str(browser), f"--app={url}", f"--user-data-dir={PROFILE_DIR}", "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check", "--window-size=1500,950"]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if running is not None:
        return None, running
    for _ in range(100):                       # up to 20 s for the browser to start
        port = debugging_port()
        if port is not None:
            return process, port
        if process.poll() is not None:
            break
        time.sleep(0.2)
    return process, None


def viewer_windows(port, url):
    """How many pages of the browser on ``port`` show the viewer at ``url``; None
    once the browser no longer answers."""
    try:
        pages = json.loads(urllib.request.urlopen(f"http://{HOST}:{port}/json/list", timeout=2).read())
    except (OSError, ValueError):
        return None
    return sum(1 for page in pages if page.get("type") == "page" and str(page.get("url", "")).startswith(url))


def wait_for_windows_closed(port, url):
    """Return once no window shows the viewer -- two quiet polls in a row, so a
    reload is not mistaken for a close."""
    quiet = 0
    while quiet < 2:
        time.sleep(1.0)
        count = viewer_windows(port, url)
        quiet = quiet + 1 if not count else 0


# -- the server -----------------------------------------------------------------

def port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((HOST, port)) != 0


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def running_viewer():
    """The address of a viewer this launcher started that still answers, or None."""
    try:
        url = json.loads(SERVER_RECORD.read_text())["url"]
        urllib.request.urlopen(url + "api/spec", timeout=1).read()
        return url
    except (OSError, ValueError, KeyError):
        return None


def start_server(project):
    """Serve the viewer from a background thread. ``(server, url)``."""
    from werkzeug.serving import make_server

    from .app import create_app, require_renderer
    from .spec import load_spec

    require_renderer()
    app = create_app(load_spec(project) if project else None)
    port = PREFERRED_PORT if port_is_free(PREFERRED_PORT) else free_port()
    server = make_server(HOST, port, app, threaded=True)
    Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://{HOST}:{port}/"
    SERVER_RECORD.write_text(json.dumps({"url": url, "pid": os.getpid()}))
    return server, url


def stop_server(server):
    server.shutdown()
    try:
        if json.loads(SERVER_RECORD.read_text()).get("pid") == os.getpid():
            SERVER_RECORD.unlink()
    except (OSError, ValueError):
        pass


# -- main -----------------------------------------------------------------------

def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(prog="gnv-launch", description="Open the General Neuroimage Viewer in its own window.")
    parser.add_argument("project", nargs="?", help="a .json figure to open")
    parser.add_argument("--check", action="store_true", help="start the viewer without a window, check that it serves its data, and exit 0 if it does (for build pipelines)")
    return parser.parse_args(argv)


def log_to_file_without_console():
    """A packaged app has no console: send what would be printed to the log."""
    if sys.stdout is None or getattr(sys, "frozen", False):
        log = open(LOG_FILE, "a", buffering=1)
        sys.stdout = sys.stderr = log


def check():
    """Serve the viewer, ask it for what a page asks for first, and say whether
    every answer came back and has the shipped defaults in it. 0 when it did."""
    server, url = start_server(None)
    failures = []
    try:
        answers = {}
        for route in ("api/spec", "api/catalogue", "api/library", "api/electrode/models", "api/backdrops", "api/palettes", "api/lighting"):
            try:
                answers[route] = json.loads(urllib.request.urlopen(url + route, timeout=60).read())
                print(f"ok    {route}")
            except Exception as exc:
                failures.append(route)
                print(f"FAIL  {route}: {exc}")
        shipped = {
            "electrode models": len(answers.get("api/electrode/models") or []),
            "available meshes": sum(1 for m in (answers.get("api/catalogue") or {}).get("meshes", []) if m.get("available")),
            "built-in backdrops": sum(1 for b in (answers.get("api/library") or {}).get("backdrops", []) if b.get("shipped")),
        }
        for what, count in shipped.items():
            print(f"{'ok  ' if count else 'FAIL'}  {count} {what}")
            if not count:
                failures.append(what)
    finally:
        stop_server(server)
    print("check passed" if not failures else f"check FAILED: {', '.join(failures)}")
    return 0 if not failures else 1


def main(argv=None):
    args = parse_args(argv)
    USER_DIR.mkdir(parents=True, exist_ok=True)
    if args.check:
        if sys.stdout is None:          # a windowed Windows build: its report goes to the log
            sys.stdout = sys.stderr = open(LOG_FILE, "a", buffering=1)
        raise SystemExit(check())
    log_to_file_without_console()
    browser = find_app_browser()

    # Already running: one more window onto it, and done.
    url = running_viewer()
    if url is not None:
        if browser is not None:
            open_app_window(browser, url)
        else:
            webbrowser.open(url)
        return

    server, url = start_server(args.project)
    print(f"General Neuroimage Viewer -> {url}")
    try:
        if browser is None:
            print("no Chrome or Edge found: opening a browser tab; stop with Ctrl-C")
            webbrowser.open(url)
            while True:
                time.sleep(3600)
        process, port = open_app_window(browser, url)
        if port is None:
            print("the browser did not report a debugging port: serving until its process ends")
            if process is not None:
                process.wait()
        else:
            wait_for_windows_closed(port, url)
        if process is not None and process.poll() is None:
            process.terminate()                # the private instance, now windowless
    except KeyboardInterrupt:
        pass
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()

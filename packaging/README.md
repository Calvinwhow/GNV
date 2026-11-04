# Packaging the circuit viewer

`.github/workflows/build-app.yml` builds the viewer on every push to `main`:

| Build | What a user gets |
|---|---|
| macOS (Apple Silicon), macOS (Intel) | `CircuitViewer-macOS-*.dmg`: open it, drag **Circuit Viewer** into Applications |
| Windows | `CircuitViewer-Windows-x64.zip`: unzip, run `CircuitViewer.exe` |

The builds are kept as workflow artifacts (Actions tab, 90 days). Pushing a tag
`v*` (`git tag v0.2.0 && git push --tags`) also attaches them to a GitHub Release,
which is the link to hand out.

Each build is checked before it is uploaded: the packaged app is started without a
window (`--check`) and must serve its data, with the shipped electrodes, meshes and
backdrops in it.

## What ships, and what does not

Everything in `circuit_viewer/resources/`, and nothing else (`CircuitViewer.spec`
ships the folder whole). That folder holds only what ships, each kind with an
`index.json` that is its dropdown (see `circuit_viewer/paths.py`):

* `meshes/` -- `brain_surfaces/`, `atlases/` (SUIT, AAL3 + SUIT), `index.json`
* `backdrops/` -- three small MNI templates, `index.json`
* `electrodes/` -- the 73 Lead-DBS models, `index.json`
* `yabplot/` -- the six cortical surfaces yabplot would otherwise download on first
  use, fetched at build time by `fetch_yabplot_surfaces.py` so the app works offline

Everything else -- the 100 um backdrops, converted atlases, the Lead-DBS surfaces --
lives in a **library** with the same layout: `circuit_viewer/library/` in your
checkout (git-ignored), or this user's folder in a packaged app:

* macOS: `~/Library/Application Support/circuit_viewer/library`
* Windows: `%APPDATA%\circuit_viewer\library`

(`$CIRCUIT_VIEWER_LIBRARY` puts it anywhere, e.g. a shared drive for a lab.) Users
fill theirs by converting, or by installing a **content pack** (`.cvpack`) someone
made for them: drop it on the view, or Library -> Install pack. Library -> Make pack
writes one from ticked meshes and backdrops.

## One-time setup

1. Put this repository on GitHub and push `main`.
2. Commit `circuit_viewer/resources/` whole: it is exactly what ships. (The root
   `.gitignore` keeps `circuit_viewer/library/` and build products out.)
3. The renderer is installed from `github.com/Calvinwhow/CircuitPyPer`, branch
   `development` (`CIRCUITPYPER_REF` in the workflow). Push the viewer-related
   changes there. While CircuitPyPer is private, add a repository secret
   `CIRCUITPYPER_TOKEN`: a fine-grained token with read access to it.

## For people you give it to

* **macOS:** the builds are not signed. The first time, macOS refuses to open the app.
  Open System Settings -> Privacy & Security and click *Open Anyway* (or run
  `xattr -dr com.apple.quarantine "/Applications/Circuit Viewer.app"`).
* **Windows:** SmartScreen warns about an unrecognised app: *More info* -> *Run anyway*.
* The viewer opens in a Chrome or Edge window of its own. Without either it opens a
  tab in the default browser.

Signing (an Apple Developer ID with notarization, a Windows code-signing certificate)
removes those warnings; it can be added to the workflow later.

## Building locally

From the repository root, in a fresh Python 3.11 environment:

    python -m pip install -r packaging/requirements.txt
    python -m pip install --no-deps git+https://github.com/Calvinwhow/CircuitPyPer.git@development
    python packaging/fetch_yabplot_surfaces.py
    python -m PyInstaller --noconfirm --clean packaging/CircuitViewer.spec

The app is in `dist/` (`Circuit Viewer.app` on macOS, `CircuitViewer/` elsewhere).

# Packaging the General Neuroimage Viewer

`.github/workflows/build-app.yml` builds the viewer on every push to `main`:

| Build | What a user gets |
|---|---|
| macOS (Apple Silicon), macOS (Intel) | `GeneralNeuroimageViewer-macOS-*.dmg`: open it, drag **General Neuroimage Viewer** into Applications |
| Windows | `GeneralNeuroimageViewer-Windows-x64.zip`: unzip, run `GeneralNeuroimageViewer.exe` |

The builds are kept as workflow artifacts (Actions tab, 90 days). Pushing a tag
`v*` (`git tag v0.2.0 && git push --tags`) also attaches them to a GitHub Release,
which is the link to hand out.

Each build is checked before it is uploaded: the packaged app is started without a
window (`--check`) and must serve its data, with the shipped electrodes, meshes and
backdrops in it.

## What ships, and what does not

Everything in `gnv/resources/`, and nothing else (`GeneralNeuroimageViewer.spec`
ships the folder whole). That folder holds only what ships, each kind with an
`index.json` that is its dropdown (see `gnv/paths.py`):

* `meshes/` -- `brain_surfaces/`, `atlases/` (SUIT, AAL3 + SUIT), `index.json`
* `backdrops/` -- three small MNI templates, `index.json`
* `electrodes/` -- the 73 Lead-DBS models, `index.json`

Everything else -- the 100 um backdrops, converted atlases, the Lead-DBS surfaces --
lives in a **library** with the same layout: `gnv/library/` in your
checkout (git-ignored), or this user's folder in a packaged app:

* macOS: `~/Library/Application Support/gnv/library`
* Windows: `%APPDATA%\gnv\library`

(`$GNV_LIBRARY` puts it anywhere, e.g. a shared drive for a lab.) Users
fill theirs by converting, or by installing a **content pack** (`.gnvpack`) someone
made for them: drop it on the view, or Library -> Install pack. Library -> Make pack
writes one from ticked meshes and backdrops.

## One-time setup

1. Put this repository on GitHub and push `main`.
2. Commit `gnv/resources/` whole: it is exactly what ships. (The root
   `.gitignore` keeps `gnv/library/` and build products out.)
3. circuit_pyper, which reads the tract file formats, is installed from
   `github.com/Calvinwhow/CircuitPyPer`, branch `development`
   (`CIRCUITPYPER_REF` in the workflow), in its light form -- its default
   install, which is only numpy, scipy, nibabel, h5py and tqdm. While CircuitPyPer is private, add a repository secret
   `CIRCUITPYPER_TOKEN`: a fine-grained token with read access to it.

## For people you give it to

* **macOS:** the builds are not signed. The first time, macOS refuses to open the app.
  Open System Settings -> Privacy & Security and click *Open Anyway* (or run
  `xattr -dr com.apple.quarantine "/Applications/General Neuroimage Viewer.app"`).
* **Windows:** SmartScreen warns about an unrecognised app: *More info* -> *Run anyway*.
* The viewer opens in a Chrome or Edge window of its own. Without either it opens a
  tab in the default browser.

Signing (an Apple Developer ID with notarization, a Windows code-signing certificate)
removes those warnings; it can be added to the workflow later.

## Building locally

From the repository root, in a fresh Python 3.11 environment:

    python -m pip install -r packaging/requirements.txt
    python -m pip install git+https://github.com/Calvinwhow/CircuitPyPer.git@development
    python -m PyInstaller --noconfirm --clean packaging/GeneralNeuroimageViewer.spec

The app is in `dist/` (`General Neuroimage Viewer.app` on macOS, `GeneralNeuroimageViewer/` elsewhere).

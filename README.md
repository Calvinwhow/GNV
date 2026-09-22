# circuit viewer

A layered viewer for brain figures.  It draws with the same off-screen PyVista
renderer as `circuit_pyper`, so what you arrange here is pixel-for-pixel what
the export writes -- the app is a control panel, not a second renderer.

STARTUP
```bash
pip install -e .
circuit-viewer                     # opens http://127.0.0.1:8731
circuit-viewer figure.json         # ... with a saved figure loaded
```

SHUTOFF 
```bash
lsof -ti :8731 | xargs kill
```

It needs `circuit_pyper` importable in the same environment; that is where the
meshes, the volume sampling and the colour rules live.

## The model

A figure is a list of **layers** and a **camera**.  There are two kinds of layer.

Every mesh and slice layer holds an ordered **stack of volumes** rather than one.
"Motor in red, cognitive in blue, over anatomy" is three entries on one group,
not three groups competing for the same geometry.  Each entry is coloured on its
own terms and painted only where it clears its own threshold, later entries over
earlier ones — the transparency is what makes it readable, since an overlay is a
stencil as much as a colour.  Dropping a NIfTI on a layer appends to its stack.

One volume keeps the scalar-plus-colormap path, which is what gives the figure a
colour bar.  Two cannot share one scale, so the stack is composited to explicit
RGB and the bar is dropped rather than captioning the drawing with one map's
numbers.

**Mesh layers** hold one mesh or a group of them, and the group is the unit of
styling: everything in it shares a base colour, an opacity and a wireframe.  With
no volume attached the group is flat anatomy.  On a volume, *Sample* decides what
the colouring means:

| Sample   | What it does                                            | Use it for |
|----------|---------------------------------------------------------|------------|
| `region` | reduces the volume inside each mesh to one number and floods that mesh with it | 165 parcels, "how much does this map implicate this structure" |
| `vertex` | samples the volume at every vertex, giving a gradient across the surface | a whole-brain surface, "what does the map look like where it touches cortex" |

`region` takes a statistic (`max` keeps the signed peak; `mean_nonzero` averages
only voxels that carry a value, so a thresholded map is not diluted by the zeros
around it).  *Voxel min |v|* drops voxels before the reduction; *Region min |v|*
and *Keep top N* drop whole regions after it.  Those are different jobs: the
first decides what counts as signal, the second decides what is worth colouring.

A mesh layer's geometry does not have to be a mesh.  Set *Geometry* to "a NIfTI
volume" -- or drag one in -- and the volume is surfaced
with the same pipeline that built the SUIT and AAL parcel meshes, so a lesion or
a stimulation volume dropped in today sits beside them and looks like it belongs
rather than like an import.  *Split* decides what the volume becomes:

| Split        | Result |
|--------------|--------|
| `single`     | everything above the threshold as one surface — a lesion, a VTA, a thresholded cluster |
| `labels`     | one surface per distinct integer value — any atlas or segmentation becomes a region group, stylable like the built-in ones |
| `components` | one surface per connected blob — a multi-cluster map becomes separately colourable pieces with nobody labelling anything |
| `auto`       | `labels` when the volume looks like a segmentation, otherwise `single` |

Pieces under 27 voxels are dropped and at most 64 of the largest are built,
because a threshold that leaves 400 speckles is minutes of marching cubes for a
figure nobody can read.  What was dropped is reported, not silently discarded.

Locally converted surfaces live in `circuit_viewer/resources/meshes/`. Standalone
`.vtp` surfaces and folders containing a bilateral `L.vtk` / `R.vtk` pair are
added to the mesh-library menu automatically. An adjacent `catalogue.json` can
give each resource the basename and explanation shown in that menu.

## Dropping files

Where a file lands decides what it means, and that is the whole interface:

- **onto the New group box** (or the 3D view) it is GEOMETRY.  The volume is
  surfaced and becomes a group of its own.  Drop several volumes together and
  they become ONE group, not several layers -- a set of ROI files is the same
  thing as a directory of parcel meshes, and the point of a group is that its
  members share a palette and a scale.
- **onto an existing layer** it is COLOUR.  The volume is sampled inside the
  meshes that layer already has, exactly as if you had set its NIfTI field.

The same file does two different jobs, so every drop target highlights while you
are over it. A `.mat` / `.tck` tractogram dropped on the New group box becomes
a fiber layer; dropped on an existing fiber layer it replaces its source. A
`.fib.npy` is one value per atlas fiber and can be dropped onto an existing
fiber layer as an overlay. A `.fib.json` result can be dropped as a new fiber
layer because it identifies both the atlas geometry and its `.fib.npy` values.

Dropped files are copied into a scratch folder, since a browser will not tell a
page where a dropped file lives.  For anything large, or anything a saved
project should keep pointing at, use the file picker.

On a region set a "vertex" overlay samples at cell centres — the regions are the
unit, so that is the finest thing the geometry can actually show.

**Slice layers** fan three orthogonal planes through an anatomical volume, the
Lead-DBS idiom, as a backdrop for the meshes.  Three sliders move them in world
millimetres and each can be switched off on its own.

A volume has to be converted once before it can be used this way.  *Convert a
NIfTI…* writes a `.npz` into a library that accumulates — every volume converted
once is available to every figure afterwards — holding one compressed array per
slice per axis.  Reading slice 412 decompresses exactly that member, a
millisecond or so, instead of touching the volume; all three axes are stored
because a single array can only be sliced cheaply along its first.

The library lives in `circuit_viewer/resources/backdrops/` inside this local
installation. Generated `.npz` files and in-progress `.building.raw` scratch
files are ignored by Git, keeping the data beside the viewer without committing
multi-gigabyte artifacts.

Conversion is at the source's **native** resolution by default: hand it a 100 µm
scan and you get 100 µm slices, because the point of having that scan is being
able to show it.  Pass an `mm` to coarsen deliberately.  That is not free — the
1971 × 2331 × 1891 Edlow volume is ~26 GB of slices before compression, takes
minutes to convert, and wants roughly its uncompressed size free as scratch
while it runs.  Point the converter at the uncompressed `.nii` rather than the
`.nii.gz` if you have both; the source is then memory-mapped instead of
decompressed.

Building never holds the volume in RAM.  The source is streamed into a temporary
`uint8` memmap and each axis is written out in slabs sized to a memory budget,
so the ceiling is disk rather than memory.  *Drop background* makes near-zero
intensity transparent, which is what lets a slice sit inside a brain rather than
as a black slab in front of one.  Slice layers take the same overlay stack, so a
network can be composited straight onto the anatomy it is being explained by.

Moving a plane updates the existing actor in place rather than rebuilding it:
that is 161 ms down to 71 ms a frame, and the slice read inside it is under a
millisecond.  The volumes an overlay samples are cached after first use, so the
first drag after adding one pays a read the rest do not.

**Fiber layers** hold streamlines from a Lead-DBS `.mat` or a tractogram atlas.
They are their own kind because they do not hang off a mesh -- the geometry is
in the tractogram. A `.fib.npy` supplies one value per atlas fiber, while its
`.fib.json` descriptor identifies the atlas and values together. Note
*Max |value|*: a `contrast_pval_*` file stores p-values, where the fibres you
want are the small ones, so `0.05` is the significant set.

Anything a layer does not colour stays in the opaque base grey, so "no signal
here" still reads as anatomy instead of turning into the bottom of a colour bar.

## Why the regions look tessellated

The parcel meshes are built at ~1.1 mm edges.  That is right for a clean
silhouette and wrong for a figure: at print scale a per-triangle wireframe on a
mesh that fine is grey haze, and smooth shading over it turns 165 regions into
smooth jelly.  Regions are drawn flat-shaded with a faint wireframe over them.  The wireframe
is a texture, not a line drawing: at full strength it is a white grid that
flattens every colour under it, so its opacity is fixed at 0.1 and *Edge
highlight* is simply on or off.

*Smoothness* is the other lever.  The parcel meshes are built at ~1.1 mm edges,
which is right for a clean silhouette and wrong for a figure -- at print scale a
wireframe that fine is grey haze.  Sliding Smoothness down decimates each mesh
before anything is merged or sampled; the default lands around 2 mm edges, where
the facets and the wireframe are things the eye can resolve.  Because the
decimation happens before sampling, the region statistic is computed on the
geometry that is actually drawn.

Because the decimation happens before sampling, the region statistic is computed
on the geometry that is actually drawn; the numbers never disagree with the
picture.  Region sets default to this treatment and cortical surfaces do not --
a wireframe over 32k fsLR vertices is only noise.

A translucent layer shows the wireframes of the regions behind it as well as its
own, so a low opacity and a dense wireframe fight each other.

## Scene

*Background* defaults to white and offers black for showing a dark scan off.
Scale-bar text is derived from it rather than fixed, because a dark label on a
black backdrop is an invisible colour bar.

## Hemispheres and regions

A mesh group draws both hemispheres unless told otherwise; *Left* and *Right*
switch each off, and every region inside the group can be switched off on its
own from the *Regions* list (with a filter, because 165 parcels is a list you
search rather than scroll).  Both happen before anything is merged, so a hidden
region costs nothing to draw and is genuinely absent rather than transparent.

Sides are worked out from the region's name when it has one -- an atlas that
says `_L` is more reliable than geometry -- and from its centroid otherwise, so
volume-derived groups with no naming convention still split.  Structures
straddling the midline (vermis, brainstem) are classed as midline and are not
taken by either hemisphere toggle.

*Auto-hide near* is the older behaviour: drop whichever hemisphere faces the
camera in a lateral view.  It is right for a translucent shell and surprising
everywhere else, so it is off by default and is now a setting rather than
something that happens to you.

## Lighting

A three-point rig: one key light doing the modelling, with a fill and a rim that
exist only to keep the dark side readable.  All three follow the camera, so a
region is lit the same way in a lateral view and an inferior one.

The key sits above and to the RIGHT of frame, which in a left lateral view puts
the highlight on the parietal and drops the frontal pole into shadow.  Fill and
rim are placed relative to the key, so moving the key swings the whole rig rather
than pulling it apart.

*Rig* also offers `kit` -- VTK's five-light kit, which is what PyVista gives a
Plotter by default and therefore what every yabplot figure was lit with.  It
fills a surface from five directions against high ambient, so nothing is ever in
shadow and nothing has form: legible, and cartoonish.  It is kept for
reproducing a figure that was lit that way, not as a default.

The material matters as much as the rig.  Diffuse (0.78) carries the form and
ambient (0.30) only keeps the shadow side readable; the kit's ratio is the other
way round, and that inversion is precisely what flattens it.  Raising ambient is
the fastest way to make a render look like a cartoon.

*Orthographic* projection is on by default, and is the honest choice for a figure
someone will measure by eye -- two regions of the same size render the same size
whatever their depth.

## Palettes

Leave the palette on **auto** and the data picks: a map carrying both signs gets
a symmetric diverging scale centred on zero, and a one-signed map gets a ramp
built from the layer's colour.  Setting `Sign` to `positive` is therefore what
turns a two-tailed t-map into a single-hue figure.

## Project files

Save writes the whole figure as JSON -- every layer, every threshold, the
camera.  It is meant to be read: commit it beside a manuscript figure, diff it
when a reviewer asks what changed, hand-edit it, or build it here and load it
from a script:

```python
from circuit_viewer.session import RenderSession
from circuit_viewer.spec import load_spec

RenderSession(load_spec("figure.json")).export("figure.png", size=(1700, 1400))
```

## Performance

Geometry is cached separately from style, which is what makes the panel usable:
changing a colour or an opacity re-renders in ~150 ms, changing a threshold or a
statistic in ~0.5 s, and orbiting costs only a camera move (~40 ms/frame).  The
expensive steps -- voxelising a parcel group, reading a 300 MB tractogram -- are
done once and survive every change that does not alter the numbers.

## Scope

It binds to `127.0.0.1` and it will happily browse your filesystem and write
files where you tell it to.  That is fine for a tool running as you on your own
machine and would not be fine on an interface anyone else can reach.

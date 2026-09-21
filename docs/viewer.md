# Result Viewer

`fmri_utils.viewer` builds a static web page for looking at brain maps: a
template with thresholded overlays, a montage of subjects, region outlines, and
a surface view that inflates and flattens. It is one HTML file, one
`manifest.json`, and the NIfTI files the manifest points at. Nothing runs on
the server, so it goes on any static host — a lab web directory, GitHub Pages,
a shared drive — and a link shows everyone the same thing.

The page knows nothing about any particular study. What it offers comes from
the manifest: no atlas, no region panel; no surfaces, no surface mode. Your job
is to say what maps you have.

## See It First

```bash
pip install "git+https://github.com/efirdc/fmri_utils.git"
fmri-viewer demo --output-root viewer
cd viewer && python -m http.server 8000
```

Open <http://127.0.0.1:8000>. That is the whole viewer, built from synthetic
volumes. [`demo.py`](../src/fmri_utils/viewer/demo.py) is about eighty lines and
is the shortest thing to copy from.

## The Shortest Real Build

A template and one map per subject is enough:

```python
from pathlib import Path
from fmri_utils.viewer import (
    About, Endpoint, MapEntry, Report, ViewerSpec, build_viewer,
)

spec = ViewerSpec(
    about=About(
        title="Motion localiser",
        kicker="study 42",
        lede="Held-out t maps for the motion contrast, eight subjects.",
    ),
    template=Path("MNI152_T1_1mm_brain.nii.gz"),
    reports=[Report(id="loc", label="Localiser", endpoints=[
        Endpoint(
            id="t_motion", label="motion > static", statistic="t",
            degrees_of_freedom={"sub-01": 240, "sub-02": 240},
            blurb="First-level t after prewhitening.",
            maps=[
                MapEntry(subject="sub-01", path=Path("sub-01_t.nii.gz")),
                MapEntry(subject="sub-02", path=Path("sub-02_t.nii.gz")),
            ],
        ),
    ])],
)
build_viewer(spec, Path("viewer"))
```

`build_viewer` writes `viewer/index.html`, `viewer/manifest.json` and
`viewer/data/...`, and copies the page from the installed package. Copy the
directory to the host, or `rsync` it.

## What The Pieces Mean

**Report** — a group of endpoints that belong to one analysis. It is the top
level of the left rail. A project with one analysis has one report.

**Endpoint** — a quantity the reader can look at: a t map, a correlation, a
contrast, a difference. Endpoints inside a report share a menu.

**MapEntry** — one map file, for one subject, optionally for one *feature*.
Features are a second axis within an endpoint: three feature spaces fitted the
same way, four models compared. Leave `feature=""` and the control disappears.

**Subject** — whatever labels your maps. `"group"` is a subject as far as the
page is concerned, which is how a group map sits beside the individuals.

The intensity window comes from the data: each map gets a starting range at
`ViewerSpec.percentile` (99.5 by default) of its own absolute values, and the
endpoint's range is the median across its maps, so switching subject does not
move the scale. `Endpoint.display_range` overrides it with an explicit
`(threshold, top)`, which is what a comparison needs: give several endpoints
the same one and a colour means the same number in all of them. Switching
endpoint then keeps whatever window the reader has dialled in, and only
re-reads the default when the new endpoint declares a different range.

## Thresholds

Set `statistic="t"` (or `"r"`) and give `degrees_of_freedom` per subject, and
the page offers three threshold modes: raw value, uncorrected *p*, and FDR *q*.

The reader types any *p* or *q*, not one of a few fixed levels. That works
because the builder stores curves rather than levels. The *p* curve is
analytic. The *q* curve is not — a Benjamini–Hochberg threshold depends on the
map's own distribution of *p* — so it is evaluated here at 64 values of *q* and
interpolated in the browser. Maps with no `degrees_of_freedom` entry keep the
value threshold only.

For a correlation map, `degrees_of_freedom` is the number of samples; the
builder converts through the equivalent *t* and back.

## Optional Parts

Each one switches itself on when you supply it.

### Subject anatomy

```python
from fmri_utils.viewer import CoordinateMaps, SubjectSpace

subject_space = SubjectSpace(
    templates={"sub-01": Path("sub-01/T1_brain.nii.gz")},
    templates_coarse={"sub-01": Path("sub-01/T1_2mm.nii.gz")},
    coordinates={"sub-01": CoordinateMaps(
        to_template=Path("sub-01/anat_to_mni.nii.gz"),
        from_template=Path("sub-01/mni_to_anat.nii.gz"),
    )},
)
```

Add `subject_space_path=` to a `MapEntry` and that map becomes available in the
subject's own anatomy, under a **space** toggle. Coordinate maps are
three-component volumes giving, per voxel, the matching millimetres in the
other space; they are what lets a montage of different brains hold one
crosshair. They are thinned to every fourth voxel on the way out — the field is
smooth and the browser interpolates.

`templates_coarse` is the same anatomy at 2 mm, used only for montage panels,
where a cell is a quarter of the window wide and 1 mm costs eight times the
texture for nothing. The same applies to the template itself via
`ViewerSpec.template_coarse`.

### Regions

```python
Atlas(
    id="cort", label="cortical",
    labels=[(1, "Left Frontal Pole"), (2, "Right Frontal Pole")],
    template_path=Path("HarvardOxford-cort-maxprob.nii.gz"),
    subject_paths={"sub-01": Path("sub-01/HarvardOxford.nii.gz")},
)
```

Pass a list of them as `ViewerSpec.atlases`. Several atlases coexist: the panel
shows one at a time and selections in the others are kept. `label` is what the
chooser shows. Names beginning "Left "/"Right " are sorted together.

### Surfaces

Surfaces are exported separately, because reading a pycortex or FreeSurfer
database is slow and you do not want it inside every rebuild:

```bash
fmri-viewer export-surfaces --pycortex-db /path/to/db --output-root surf_export \
    --subjects sub-01,sub-02
fmri-viewer package-surfaces --input-root surf_export --output-root surf_pack
```

Then point the spec at the packaged directory with `surfaces=Path("surf_pack")`
and the builder folds `surfaces.json` into the manifest and copies the binaries
across.

`--fsaverage` exports the shared fsaverage surfaces instead (via nilearn, no
pycortex needed) together with the MNI305→MNI152 affine, so template-space maps
can be projected without any subject-specific geometry.

Which map the surface samples depends on the space:

| surface space | manifest key | samples |
|---|---|---|
| fsnative | `surfaces[<subject>]` | that subject's `subject_space_path` |
| fsaverage | `surfaces["fsaverage"]` | the template-space map, through `volume_transform` |

A subject with no native surfaces falls back to fsaverage.

## Turning Things Off

Data decides by default. `Features` overrides it downwards — for a project that
has the data but does not want the control:

```python
ViewerSpec(..., features=Features(montage=False, subject_space=False))
```

| field | manifest key | hides |
|---|---|---|
| `surface` | `features.surface` | the volume/surface mode toggle |
| `regions` | `features.regions` | the region panel and its opacity slider |
| `montage` | `features.montage` | the montage view button |
| `subject_space` | `features.subjectSpace` | the MNI/subject space toggle |

There is no switch that turns something *on* without the data for it.

`Features` carries one switch that is not about a control. `fine_underlay`
defaults to true, meaning maps are composited onto the full-resolution
anatomy. NiiVue blends an overlay onto the underlay's grid, so that decides how
much texture each panel holds: about 29 MB at 1 mm against 3.6 MB at 2 mm. Set
it to false and any map coarser than 1.8 mm gets the coarse underlay instead,
which for a 2 or 3 mm statistical map shows exactly the same voxels and is
several times faster to switch. Leave it on when the anatomy itself is the
point.

## The Manifest Contract

The page reads `manifest.json` and nothing else. A project with its own
pipeline can write that file directly and skip this module; this is the shape.

```jsonc
{
  "about":    { "title": "", "kicker": "", "lede": "", "footnote": "" },
  "features": { "surface": true, "regions": true, "montage": true,
                "subjectSpace": true },

  "template": "data/template.nii.gz",       // uint8 anatomy, any resolution
  "template_label": "MNI152_T1_1mm",
  "template_montage": "data/template_coarse.nii.gz",   // optional

  "reports": [{
    "id": "loc", "label": "Localiser", "blurb": "",
    "endpoints": [{
      "id": "t_motion", "label": "motion > static", "blurb": "",
      "warp": "nonlinear",          // shown as a badge; omit for none
      "statistic": "t",             // "t", "r", or "" for value-only
      "features": [{ "id": "", "label": "—" }],
      "subjects": ["sub-01", "sub-02"],
      "range": [0.0, 6.2],          // the window the controls open at
      "maps": [{
        "subject": "sub-01", "feature": "",
        "path": "data/loc/t_motion/map_sub-01.nii.gz",   // float32
        "range": [0.0, 6.4],
        "thresholds": {
          "degrees_of_freedom": 240,
          "p":   [1e-12, ...], "p_t": [7.13, ...],       // both ascending in p
          "q":   [1e-06, ...], "q_t": [5.42, ...]        // null where none survive
        }
      }]
    }]
  }],

  "subject_space": {
    "templates":    { "sub-01": "data/subject_space/sub-01/template.nii.gz" },
    "templates_2mm": { "sub-01": "..." },
    "coords": { "sub-01": {
      "anat_to_mni": { "path": "...bin", "dims": [64, 64, 40],
                       "affine": [row-major 4x4; the page reads the first 12] },
      "mni_to_anat": { "path": "...bin", "dims": [...], "affine": [...] }
    }},
    "maps": [{ "report": "loc", "endpoint": "t_motion", "feature": "",
               "subject": "sub-01", "path": "..." }]
  },

  "atlas": {
    "atlases": [{ "id": "cort", "label": "cortical", "mni": "data/atlas/cort.nii.gz",
                  "regions": [{ "value": 1, "name": "Left Frontal Pole" }] }],
    "subject": { "sub-01": { "cort": "data/atlas/sub-01/cort.nii.gz" } }
  },

  "surfaces": { "<subject or fsaverage>": {
    "hemispheres": { "lh": {
      "n_vertices": 134483, "n_faces": 268962,
      "mesh": "data/surfaces/sub-01/lh_wm.mz3",   // base geometry, carries the faces
      "base_geometry": "wm",
      "geometries": {
        "wm":       { "vertices": "...bin", "anatomical": true },
        "inflated": { "vertices": "...bin", "anatomical": true },
        "flat":     { "vertices": "...bin", "anatomical": false,
                      "face_mask": "...bin", "n_faces": 255052 }
      },
      "curvature": "data/surfaces/sub-01/lh_curv.bin"
    }},
    "rois": { "pSTS": { "path": "data/surfaces/sub-01/rois/pSTS.bin",
                        "n_vertices": 2975 } },
    "volume_transform": [16 floats],   // fsaverage only: template mm -> surface mm
    "volume_space": "MNI152"
  }}
}
```

Every path is relative to the directory holding `index.html`. The binaries are
raw little-endian: vertices `float32[n_vertices][3]`, curvature
`float32[n_vertices]`, face masks `uint8[n_faces]`, ROIs `int32` vertex
indices, coordinate maps `float32[3][x][y][z]`.

## File Sizes

Anatomy is written as uint8 with the scale in the header — nobody reads a
number off an underlay, and it keeps a 1 mm brain near 3 MB instead of 12.
Statistical maps stay float32, because the reader thresholds them and a
quantised map would move the threshold under their hands.

A viewer with eight subjects, three feature spaces and native surfaces runs
around 2 GB, most of it surface geometry. The page loads what the reader is
looking at and caches a bounded number of volumes, so the size of the whole
does not decide what a phone can open.

## Using The Page

- click or drag in any panel to move the crosshair; panels stay linked
- `A` toggles the overlay off and back on, `S` does the same for the regions
- the window has two knobs: below the lower one the map fades out or is clipped,
  depending on the **blend** setting
- in surface mode the **unfold** slider runs white matter → pial → inflated →
  flat, the **separation** slider pulls the hemispheres apart, and when the
  surface is fully flat the camera locks to face it
- pinch to zoom and two-finger drag to pan, on both volume and surface views

## Extending It

The page is [`resources/index.html`](../src/fmri_utils/viewer/resources/index.html),
one self-contained file with no build step: plain ES5, NiiVue from a CDN for
volumes, and a small WebGL2 renderer of its own for surfaces. Edit it, point
`build_viewer` at your copy by overwriting `viewer/index.html` afterwards, or
fork the file and keep the manifest contract.

The two rules worth keeping if you change it: the page reads capability from
the manifest rather than hard-coding a study, and it degrades to whatever is
present rather than failing on what is missing.

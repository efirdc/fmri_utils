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

### The group beside its participants

A group t map and the first-level contrasts it was computed from are different
quantities, so they cannot share one colour window or one set of thresholds.
Give the group its own window and put everything in one endpoint:

```python
Endpoint(
    id="t_3d_minus_2d", label="3D - 2D", statistic="t",
    degrees_of_freedom={"group": 30},            # only the group map gets p and q
    subject_display_ranges={"group": (0, 5)},    # the group's own window
    maps=[*participant_maps, MapEntry("group", group_t)],
)
```

The reader then switches between the group and any participant without
leaving the endpoint, and the page keeps each honest:

- a subject with its own window leads the subject list, so the endpoint opens
  on it; the endpoint's own range comes from the other subjects only;
- moving between subjects resets the window only when the scale changes;
- p and q are offered only when the map on screen has curves, and a p or q
  setting comes back when the reader returns to a map that has them;
- a montage shows the subjects on the shared scale and leaves the group out;
- the group has no anatomy of its own, so subject T1 space and the fsnative
  surface fall back to MNI and fsaverage for it, and return to what the
  reader chose on the next participant;
- features or variants that only some subjects have are greyed or hidden for
  the others: a group test of 3D - 2D has no 2D map, and a permutation-
  corrected variant exists only for the group (see *Thresholds*).

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
interpolated in the browser. Both curves also carry the conventional levels
(*p* .05, .01, .005, .001, …; *q* .1, .05, .01, …), where the threshold is
exact; between them it is interpolated and can differ by a borderline voxel. Maps with no `degrees_of_freedom` entry keep the
value threshold only.

For a correlation map, `degrees_of_freedom` is the number of samples; the
builder converts through the equivalent *t* and back.

A map that is already a corrected result -- a permutation FDR or FWER mask --
must not be corrected again. Put such maps in a `Variant` with
`analytic=False` and the builder writes no curves for them, so the page offers
value thresholding only:

```python
Endpoint(..., variant_label="Group inference", variants=[
    Variant("observed", "observed t"),
    Variant("fdr", "perm FDR q<.05", "Voxels surviving FDR over permutation p.", analytic=False),
], maps=[MapEntry("group", observed_t, variant="observed"),
         MapEntry("group", fdr_mask, variant="fdr"), *participant_maps])
```

A map with no `variant` is shown whichever variant is chosen.

**Significance companions.** A test the page cannot redo -- permutations,
TFCE with sign flipping, a region test -- can still drive the *p* and *q*
modes. Give the map a companion volume of −log10 *p* (and of −log10 *q*) on
its own grid: `MapEntry(significance_p=..., significance_q=...,
significance_label="TFCE FWE")`. A third, `significance_fwe`, is −log10 of a
family-wise corrected *p* (a permutation max-*t*, say). It adds an **FWE <**
mode, shown only for maps that have it. A permutation analysis can then be one
map, the observed statistic, with *p*, *q* and FWE thresholds and **Cluster ≥**
for extent, instead of a fixed map for each correction.
`significance_defaults={"p": 0.005}` sets the level a mode opens at. In *p*< or *q*< mode the map keeps its own colours (the effect, not
the statistic) and is shown only where the companion clears the level; the
readout says `p<0.05 · TFCE FWE`. Until the companion has loaded, the map is
held back rather than shown uncorrected. On a surface the companion is sampled
at the same vertices as the map; on a region-level map (below) *p* and *q* come
from the rows. `Variant(threshold_default=("p", 0.05))` switches the bar to that
mode the first time the reader picks the variant, if they were on the value
threshold. With a companion the *p* mode opens at 0.05 rather than the
analytic default of 0.001, since the *p* is already corrected.

**Minimum cluster size.** The **k ≥** box beside the threshold hides voxels
that pass the threshold but sit in a cluster smaller than *k*. Clusters are
face-connected (6 neighbours), with positive and negative values clustered
apart. They are found after the brain mask and any *p*/*q* companion, which is
the rule of the usual "p < .005, k ≥ 20" maps. Voxels under the threshold are
not affected, so the soft blend still fades them in. *k* counts voxels of the
map's own grid, so the same *k* is a smaller volume in a 1 mm participant map
than in a 2.5 mm group map.

On a surface, clusters are still found on the volume and carried to the
surface through the depth points its values are averaged over. Each vertex
remembers which voxel each of its five points landed on, in the subject's own
surface and on fsaverage while morphing. A vertex is dropped when one of its
points is in a voxel the rule removed and none is in a voxel that survived.
A region-level map or a stack on a surface is not clustered, and the box is
hidden for it.

It is the page's display rule, not a cluster-level test: a permutation
cluster-extent threshold still has to come from the analysis. The link keeps
it as `k=`.

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
crosshair. They are thinned to every `CoordinateMaps.stride`-th voxel (4 by
default; 2 suits fields on a 2 mm grid) on the way out — the field is smooth and
the browser interpolates. With fMRIPrep outputs, warp an image whose voxel
values are its own world millimetres through the `from-MNI…_to-T1w` transform
onto the T1 grid (anat→MNI), and the reverse for MNI→anat; see the 3DfMRI
project's `prepare_3dfmri_web_viewer_assets.py`.

`templates_coarse` is the same anatomy at 2 mm, used for montage panels on a
phone (more than three panels in a narrow window) and when `fine_underlay` is
off; everywhere else the 1 mm anatomy is shown. The same applies to the
template itself via `ViewerSpec.template_coarse`.

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

In volume mode, selected regions are drawn in one of two **region styles**.
**outline** (the default) draws each one as a contour with its abbreviation on
the slice, the way pycortex draws ROIs on a flat map. **fill** paints the
voxels solid. The **display** panel sets how outlines look:

- **colour**: each region in its own colour, or all white.
- **labels**: on or off (this switch also covers surface labels).
- **smoothing**, in four steps:
  - *voxel edges*: the outline follows the voxel boundaries exactly.
  - *straight*: marching squares' straight cuts across corners.
  - *light*: one relaxation pass and one round of Chaikin corner cutting.
  - *smooth* (the default): two of each.
- **specks**: hides pieces of one or two voxels (stray voxels, and one-voxel
  holes), which otherwise show up as tiny loops.

An atlas is kept as a compact label grid: its labels in RAS voxel order,
cropped to the labelled voxels, in the smallest integer type that holds them
(a subject's 1 mm atlas goes from 19 MB as loaded to a few). The selected
regions become one mask through a lookup table per atlas, one read per voxel
however many regions are picked, cropped to their box and uploaded as bytes.
Selecting all 63 Harvard-Oxford regions takes about 0.1 s per participant;
it used to take about 9 s, with a 78 MB float copy per montage panel. The caches
hold a full montage page.

Outlines are traced per slice on the atlas's own voxel grid and cached by
slice and settings. Each label sits at the region's widest point on that
slice. When labels would overlap, the largest region's label wins. Clicking a
label opens the region's info box. An atlas whose voxel axes are not aligned
with the world axes falls back to fill.

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

#### FreeSurfer subjects without pycortex

A plain recon-all subject has no flat map, and a recon from a high-resolution
T1 has far more vertices than a browser wants (about 570,000 per hemisphere
from a 0.5 mm T1). Do not decimate it: on cortex that folded, a decimator lays
triangles across sulci and flips them, and white and pial render as broken
glass. Resample it onto the fsaverage mesh instead:

```python
from fmri_utils.viewer import export_freesurfer_subject, export_surfaces
export_surfaces(Path("."), export_root, [], fsaverage=True)      # once
export_freesurfer_subject(subjects_dir / "sub-01", export_root, "sub-01",
                          fsaverage_export=export_root / "fsaverage")
```

Each fsaverage vertex is located on the subject's `?h.sphere.reg`, and the
subject's own white, pial and inflated positions are read off there
(barycentric within the containing native triangle). That gives the subject's
anatomy on fsaverage's triangulation -- the idea behind HCP's fs_LR meshes --
with about 1 mm triangles, and fsaverage's flat patch applies unchanged: the
subject's flat map is cut exactly where registration puts fsaverage's cut.
Positions are written in scanner millimetres (the T1w space of subject-level
maps). Inflated is centred on the origin, as fsaverage's is. A missing
`?h.pial` link falls back to `?h.pial.T1`.

A sub-millimetre recon needs two more things to look like fsaverage does,
both on by default and both measured against fsaverage on the same mesh:

- **Anti-aliasing** (`antialias_iterations=5`): white and pial are
  Taubin-smoothed on the native mesh *before* sampling, the low-pass that
  precedes any downsampling; detail finer than the ~0.9 mm mesh otherwise
  samples into jagged shading. Vertices move about 0.1 mm. Two more Taubin
  passes on the fsaverage mesh (`post_smooth_iterations=2`) soften facets
  where registration stretches triangles along sulcal walls.
- **Inflation** (`inflate=True`): `mris_inflate` runs a fixed number of
  neighbourhood steps, so on a mesh 3-4 times denser than usual it stops
  visibly folded (about 2.5 times the bending of fsaverage's inflated surface).
  Implicit smoothing steps continue it, area preserved, until it bends no
  more than fsaverage's (`inflate_to`; bending is the normal part of the
  Laplacian, since the tangential part only measures uneven vertex spacing).

The shading curvature is FreeSurfer's `?h.curv`, resampled and left raw. A
high-resolution `?h.curv` changes sign about 2.7 times as often per edge as
fsaverage's, which reads as speckle, but it is real: its sign agrees with the
displayed white surface's folding at about 80% of vertices (fsaverage's own
curvature against its own white: 82%), and every smoothing tried lowered that.
Diffusing it on this mesh is the worst, at 69%, because the mesh's triangles
are stretched along sulcal walls and the diffusion carries curvature across
the crown. `curvature_scale` > 0 still enables it (`smooth_curvature`), but it
is not recommended.

The export also writes `?h.sulc`, sulcal depth, resampled the same way, and
the page shades by it by default (**display -> surface -> folding**, depth or
curvature). Shading uses only the sign, and a sub-millimetre recon's
curvature, faithful as it is, changes sign at a scale finer than an
fsaverage-resolution mesh shows. On a 0.5 mm recon, 13% of mesh edges cross
a sign change, against 4% for sulc. Depth is not a smoothing of curvature: it
is how far each point lies below the surface's envelope, so it follows the
sulci the mesh does show and none it does not. `export_curvature` rewrites an
export's curvature and sulcal depth, leaving its geometry alone.

**Sampling depth.** A volume is projected onto a surface by averaging five
points per vertex, from the white surface to the pial, over the values the
points find. This applies to maps, significance companions and stack images,
on subject surfaces and fsaverage alike. The white surface alone sits on the
grey-white boundary and reads white matter as much as cortex.

**fsaverage and MNI152: registration fusion.** fsaverage's vertices are in
MNI305, and a linear affine to MNI152 (`MNI305_TO_MNI152`) is a median 2 mm off
(90th percentile 3.2-3.6 mm), about a cortical thickness. The fsaverage export
therefore also carries, per hemisphere, every vertex's coordinate in FSL's
MNI152 (MNI152NLin6Asym) from Wu et al.'s registration fusion (2018, RF-ANTs;
`resources/regfusion/README.md`):

- **Export:** `surfaces.json` → `hemispheres.<lh|rh>.volume_points =
  {"MNI152NLin6Asym": "<lh|rh>_volume_points_MNI152NLin6Asym.bin"}`. The file
  is float32 x, y, z interleaved per vertex, in the mesh's vertex order, in
  millimetres. `package_surfaces` carries it into the catalogue as
  `data/surfaces/fsaverage/...`.
- **Sampling:** a map in that template is sampled once per vertex at its point,
  as Wu et al. project; no depth average and no affine. The point is the
  subject-averaged mid-thickness position (`mri_vol2surf --projfrac 0.5`).
- **Declaring the template:** `ViewerSpec(template_space="MNI152NLin6Asym")`,
  or `Endpoint(template_space=...)` for an endpoint whose maps are in another
  one. The page uses a hemisphere's points only for a template-space map on
  fsaverage whose declared template has points. That covers its values, its
  *p*/*q*/FWE companions, **Cluster ≥** (one voxel per vertex), and a
  participant's surface morphing to fsaverage. With nothing declared, or a
  template without points (fMRIPrep's MNI152NLin2009cAsym, say), the map keeps
  the affine route. A map in a subject's frame never uses the points.
- **Atlases:** `atlas_surface.add_atlas_parcellation(..., space=...)` labels
  fsaverage the same way. The default is MNI152NLin6Asym, the space of FSL's
  atlases; pass `space=None` or a `transform` for the old depth sampling.
- **Subject surfaces** have real white and pial geometry and keep the
  five-depth average.

**fsnative to fsaverage, one slider.** A subject resampled this way shares
fsaverage's mesh vertex for vertex, so the **fsnative** and **fsaverage**
buttons become the two ends of one slider, as the geometry buttons are for
the unfold slider. The page stays on the subject's record along the whole
slider and blends three things towards fsaverage's:

- **shape:** fsaverage's geometry is placed in the subject's frame (its
  `volume_transform` into the template, then the subject's
  `linear_from_template`; without a registration the centroids are aligned),
  so the morph shows the change of shape;
- **folding:** sulcal depth (or curvature);
- **the map:** the subject-space map on the subject's cortex crossfades into
  the template map on fsaverage, sampled as the fsaverage view samples it.
  Past halfway, the bar's window and thresholds follow the template map.
  The two ends are not the same numbers. Vertex i of the subject and vertex
  i of fsaverage are matched by the surface registration, while the template
  map got to the template by the volume registration, and the two disagree
  by a few millimetres on the cortex. For one 3DfMRI participant's
  first-level map, the vertex-wise correlation between the ends was 0.19
  sampling at the white surface and 0.29 averaged over depth.

Parcellations stay the subject's throughout. A subject whose mesh is not
fsaverage's (a pycortex native mesh), and the group, keep the two buttons as
separate views. URL key `tf` (0-100).

The record's `qc` reports, per hemisphere, the fraction of resampled faces
whose normal disagrees with the native surface's at the same place (under
0.1% on white and about 1% on pial, where it self-contacts at sulcal fundi),
the median anti-aliasing displacement, and the inflation figures before and
after.

`sample_volume_labels(atlas, white, pial)` labels each vertex from a volume
atlas (the commonest label over five cortical depths); `atlas_surface` below
wraps it into a whole parcellation.

Which map the surface samples depends on the space:

| surface space | manifest key | samples |
|---|---|---|
| fsnative | `surfaces[<subject>]` | that subject's `subject_space_path` |
| fsaverage | `surfaces["fsaverage"]` | the template-space map, through `volume_transform` |

A subject with no native surfaces falls back to fsaverage.

### Surface parcellations

A volume atlas can be carried onto the surfaces as a *parcellation*: a label
for every cortical vertex, and one boundary network shared by all regions.
Where pycortex ROIs are outlined one by one, a parcellation's boundaries are
traced once through the mesh's mixed triangles, so neighbouring regions share
a single line and lines meet exactly where three regions do -- no gaps at the
corners, no doubled edges.

**Any atlas, one call.** `fmri_utils.viewer.atlas_surface` takes a label
volume and a surface export and does the whole job: it samples the atlas at
five depths between the white and pial surfaces (commonest label wins, with
unlabelled vertices filled from the nearest labelled one within `fill_mm`),
cleans and traces it with `write_parcellation`, and adds it to the export's
`surfaces.json` so `package-surfaces` carries it into the viewer. It reads the
export alone, so it needs neither pycortex nor FreeSurfer.

```python
from fmri_utils.viewer import add_atlas_parcellations, fsl_atlas_labels
from fmri_utils.viewer.region_info import harvard_oxford_cortical

regions = fsl_atlas_labels("HarvardOxford-Cortical.xml", harvard_oxford_cortical())
# fsnative: each subject's atlas already in that subject's T1 space
add_atlas_parcellations("surfaces_export", ["sub-01", "sub-02"], "ho", "Harvard-Oxford",
                        "atlases/{subject}/HarvardOxford-cort_space-T1.nii.gz", regions)
# fsaverage: an MNI152 atlas as it is
add_atlas_parcellations("surfaces_export", ["fsaverage"], "ho", "Harvard-Oxford",
                        "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz", regions)
```

or from the shell:

```
fmri-viewer atlas-parcellation --export-root surfaces_export --subjects sub-01,sub-02 \
    --atlas 'atlases/{subject}/HarvardOxford-cort_space-T1.nii.gz' \
    --labels HarvardOxford-Cortical.xml --id ho --label Harvard-Oxford
```

**Atlases that cover part of the cortex.** A handful of functional parcels
(the Saxe-lab ToM parcels, say) would be grown over the whole cortex by the
cleaning, which fills every unlabelled vertex. Pass `sparse=True` (and a small
`fill_mm`) and uncovered cortex is held as a background region while the
labels are cleaned and traced, so each parcel's edge against it is smoothed
and outlined like any boundary; the background is written back as 0 and gets
no label anchor.

Surface coordinates are the export's world millimetres. On fsnative that is
the subject's own anatomy, so the atlas must already be in that subject's T1
space; any grid will do, since only its affine is used. On fsaverage it is
MNI305, and the export's `volume_transform` (MNI305 to MNI152) is applied, so
an MNI152 atlas works as it is. `transform=` overrides either.

- **Region lists.** `fsl_atlas_labels` reads an FSL atlas XML (value = index
  + 1). `table_labels` reads a `value<TAB>name[<TAB>abbrev]` table or a
  FreeSurfer colour table. Abbreviations come from a `{name: abbrev}` dict or
  a `region_info` dict.
- **Tissue classes.** Leave out region names such as "Left Cerebral White
  Matter" with `exclude=`.
- **Absent regions.** Regions no vertex reached are dropped from the list, so
  the viewer only offers what is on that surface.

Rebuilt this way, the Harvard-Oxford parcellation agrees with the one
DATT_analysis made through pycortex on 99.4% of fsaverage vertices, and on
95.5% of UTS01's, whose earlier version sampled the atlas on the 2.5 mm
functional grid.

Underneath is `write_parcellation`, for labels you have sampled some other way:

```python
from fmri_utils.viewer.parcellation import write_parcellation

record["parcellations"] = {"ho": write_parcellation(
    export_dir / "sub-01", "ho", "Harvard-Oxford",
    regions=[{"value": 1, "name": "Frontal Pole", "abbrev": "FP"}, ...],
    hemispheres={"lh": {"labels": raw_lh, "faces": faces_lh, "flat_faces": flat_lh},
                 "rh": {...}},
)}
```

Write it into the subject's export `surfaces.json` and `package-surfaces` copies
it along. `raw_*` is whatever sampling of the atlas you have (0 where
unlabelled); `write_parcellation` cleans it first: unlabelled cortex is filled
from its neighbours, boundaries are smoothed by diffusing each region's
indicator over the mesh (`smooth_rounds`), and stray fragments are folded into
their surroundings, so every cortical vertex has a label and each region is
one piece per hemisphere. `flat_faces` should be the flat map's kept faces, so
nothing is traced across the medial-wall cut. `clean_parcellation`,
`boundary_network`, `region_anchors` and `relax_polylines` are the pieces, for
drawing the same outlines in static figures.

In surface mode the region panel then offers the parcellation (the default)
next to the pycortex ROIs. Every region starts shown. The boxes switch
regions on and off, and **select all** and **clear** act on the whole
parcellation. A boundary line is drawn when a region on either side of it is
shown, so a shown region is always fully outlined, and a hidden region keeps
only the edges it shares with shown ones. A line's two regions are the two
commonest labels among the mesh vertices of all its points. The lines are
relaxed after tracing, so any single point can sit among one region's
vertices only. Labels are culled largest region
first so they never overlap; zoom in to see the small ones.

### Region-level endpoints

An endpoint whose values are per region -- a group test run on region means,
say -- carries `region_stats`, and each map its own rows:

```jsonc
"region_stats": { "atlas": "ho", "volume_atlas": ["cort", "sub"], "statistic": "q",
                  "effect_label": "Δr" },
"maps": [{ "feature": "...", "subject": "group", "path": "…region-painted.nii.gz",
           "region_rows": [{ "name": "Angular Gyrus", "delta": 0.0068, "t": 8.0,
                             "p": 9e-5, "q": 4e-4, "significant": true,
                             "n_positive": 8, "n": 8, "value": 3.4 }] }]
```

An endpoint can also carry **variants**: the same maps computed another way,
chosen under the endpoint instead of listed as separate endpoints. Each map
(and each subject-space map) names its variant. A variant can bring its own
`range` (the window resets when it differs), its own `blurb` (appended to the
endpoint's), and a `statistic` that overrides `region_stats.statistic` for
labels and readouts. The first variant is the default.

```jsonc
"variants": [{ "id": "q", "label": "FDR q", "statistic": "q" },
             { "id": "p", "label": "uncorrected p", "statistic": "p" }],
"maps": [{ "feature": "...", "subject": "group", "variant": "q", "path": "…" }, …]
```

By default the variants are a segmented choice headed "Statistic" (or
`variant_control.label`). Two variants that are one idea switched on or off
can be shown as a single checkbox instead, with an explanation on hover. A
toggle keeps its setting when you move to another endpoint with a toggle of
the same name:

```jsonc
"variant_control": { "kind": "toggle", "label": "net of control",
                     "tip": "On: … Off: …", "on": "net", "off": "raw" }
```

From Python, these are `Variant(statistic=...)`, `Endpoint(variant_toggle=
VariantToggle(label, on, off, tip))`, `Endpoint(region_stats=RegionStats(atlas,
volume_atlas, statistic, effect_label, rows))` and `MapEntry(region_rows=
[RegionRow(name, delta, value, t, p, q, significant, n_positive, n)])`.
`ViewerSpec.check` rejects a toggle that does not name exactly the endpoint's
two variants, and `region_stats` naming an atlas that is not in the spec:

```python
Endpoint(id="region", label="Region test", maps=[
             MapEntry("group", q_map, variant="q", region_rows=rows),
             MapEntry("group", p_map, variant="p", region_rows=rows)],
         variants=[Variant("q", "FDR q", statistic="q"),
                   Variant("p", "uncorrected p", statistic="p")],
         region_stats=RegionStats(atlas="ho", volume_atlas=["cort", "sub"],
                                  effect_label="Δr"))
Endpoint(id="delta", label="Δr, ToM removed", maps=[...],
         variants=[Variant("net", "net"), Variant("raw", "raw")],
         variant_toggle=VariantToggle("net of control", on="net", off="raw",
                                      tip="Subtract the time-shifted control."))
```

`path` is an ordinary volume with each region painted with `value` (for
example signed −log10 q), so volume mode works as for any map. On a surface
the page paints from the rows through the parcellation named by `atlas`, so a
region's colour ends exactly at its drawn boundary. Labels then read
`AG* / +0.0068 / q=4e-4` (* = significant), and the status bar reports the
region under the crosshair with its effect, p and q -- in volume mode by
sampling `volume_atlas` directly, whether or not the region is selected.
`volume_atlas` is one atlas id or a list in priority order. Where two atlases
label the same voxel, the first one wins, which should match how `path` was
painted. Only regions with rows are reported, so tissue classes carried by an
atlas (Harvard-Oxford subcortical labels cortex, white matter and the
ventricles) never name a voxel.

A region test can also be one choice among an endpoint's others (see
*Several controls*): give that variant, not the endpoint, its
`Variant(region_stats=...)`. The variant's statistics win over the
endpoint's, only its maps' rows count, and choosing it behaves like entering
a region-level endpoint (below). Two region variants over different atlases
(Harvard-Oxford, then a set of parcels) swap the tested selection, and on a
surface the page shows the parcellation the chosen test reads through and
gives back the reader's own when they leave it.

Entering a region-level endpoint selects exactly the regions it tested (from
the `volume_atlas` atlases, first atlas first) and switches outlines to white,
so they carry the same labels as on the surface. From then on they are
ordinary selections: toggle them, clear them, or switch colour back on.
Leaving the endpoint restores the selection and outline colour you had
before.

### Several controls

An endpoint can offer more than one choice about its maps at once -- whether
to show a correlation or the drop when a direction is removed, which control
condition to subtract, which rating, which group test. Declare the choices as
`Endpoint(variant_controls=[VariantControl(...)])` and say, on each variant,
which option of every control it is:

```python
controls = [
    VariantControl("delta", "Δr, ToM removed", on="1", off="0", tip="…"),
    VariantControl("control", "Net of", options=[VariantOption("sh", "shifts"),
                                                  VariantOption("raw", "none (raw)")]),
    VariantControl("test", "Group test", options=[
        VariantOption("m", "none"),
        VariantOption("tf", "TFCE voxels", unavailable="run on the mean of the features")]),
]
variants = [
    Variant("r", "r", values={"delta": "0", "test": "m"}),
    Variant("d.sh.m", "Δr", values={"delta": "1", "control": "sh", "test": "m"}),
    Variant("d.sh.tf", "Δr, TFCE", values={"delta": "1", "control": "sh", "test": "tf"},
            threshold_default=("p", 0.05), subject_display_ranges={"group": (0, 0.004)}),
    ...
]
```

A control is a checkbox when it has `on`/`off`, a segmented choice over
`options` otherwise. The page keeps the reader's picks and shows the variant
that agrees with the most important of them among those the subject and
feature on screen actually have (an earlier control outranks every later one).
So a pick that has nothing to apply to -- a group test on a single subject --
is kept and comes back with the group. A control shows only while the variant
on screen has a value for it (a variant leaves out the controls that do not
apply to it: no control condition for a plain correlation) and while at least
two of its options are open. An option is open when some variant available
here has it and agrees with the controls above it; one that is not is greyed,
with `unavailable` as its tooltip. Picks carry over to another endpoint that
has a control with the same id, so moving between configurations compares like
with like. The URL keeps the variant id, as for a single choice.
`Variant.subject_display_ranges` gives the group its own window per variant,
as `Endpoint.subject_display_ranges` does per endpoint. `ViewerSpec.check`
rejects a variant whose `values` name an option no control offers.

### Cohorts

A viewer can show its results for more than one set of participants: for
example, the group maps without some excluded participants, beside the earlier
maps over everyone.

- **Cohorts:** `ViewerSpec.cohorts` lists them as `Cohort(id, label, blurb)`.
  The first is the default.
- **Tagging maps:** `MapEntry.cohort` names the cohort a map belongs to. A
  map with a cohort shows only while that cohort is chosen. One with none is
  shared.
- **Resolution:** for each subject, feature and variant, the page shows the
  chosen cohort's own map, else the shared one, else nothing. So a map every
  cohort shares (a participant's own first-level map) is listed once, with no
  cohort. A map that differs gets the shared entry plus an entry for each
  cohort where it differs (the group map; a participant map built from other
  participants, like a leave-one-out template). A result that exists only for
  one cohort names that cohort, so the others do not show it.
- **Hidden when empty:** a subject with no map in the chosen cohort leaves the
  subject list, and so does a variant control option. An endpoint (or report)
  with none leaves the rail.
- **Degrees of freedom:** `MapEntry.degrees_of_freedom` overrides the
  endpoint's for that map's p and q curves. The same group t over another
  cohort has other degrees of freedom.
- **Excluded participants:** `ViewerSpec.excluded_subjects` maps a participant
  to the reason they are left out of the group. The rail shows them in italics
  with the reason as a tooltip. The button's text stays the ID.

```python
spec = ViewerSpec(..., cohorts=[Cohort("n28", "28 participants"), Cohort("all", "All 31 (previous)")],
                  excluded_subjects={"sub-03": "reason shown as a tooltip"})
MapEntry("group", new_t, degrees_of_freedom=None)               # shared; the endpoint's df
MapEntry("group", old_t, cohort="all", degrees_of_freedom=30)   # the earlier group map, in "all"
MapEntry("group", extra_t, variant="r1p5", cohort="n28")        # run only for n28: not in "all"
```

With two or more cohorts, **Participants** appears at the top of the rail. The
choice is resolved before anything reads a map list, so which features and
variants are offered, montages and the surface all follow it. URL key `co`,
written only when it is not the default.

### Registration inspection

**Surface outlines in any volume view.** **Surfaces** in the toolbar draws the
subject's white (blue) and pial (red) surfaces where each slice cuts them.
This works in every volume view, for any endpoint, as long as the manifest has
surfaces. They come from the surface package's `wm` and `pia` vertices, which
are in the subject's anatomical world, and are carried into the frame of the
map on screen:

- **subject space:** as they are;
- **template space:** through the subject's `anat_to_mni` coordinate field
  (vertices off the field fall back to the registration's linear part);
- **a registration stack:** by the same fraction of the registration as its
  images;
- **any other frame:** through the map's `surface_affine`, a 4 x 4 from the
  subject's anatomical world to the map's;
- **no surfaces of its own** (the group): fsaverage's surfaces through their
  `volume_transform`. fsaverage is an average brain, so these outlines only
  approximate the template's anatomy.

`Endpoint.outlines` ("white", "pial" or "both") switches the outlines on when
the reader enters that endpoint.

The outlines are drawn on the GPU, in the overlay layer's shared context:

- **Data:** a surface's vertices are a half-float texture, and its faces an
  integer texture shared by every surface with the same mesh.
- **Drawing:** each face is an instance whose vertex shader finds the segment
  where the slice plane crosses it and widens it into an anti-aliased line.
- **A surface that stays put:** its faces are counting-sorted along an axis
  the first time a slice cuts that axis (about 40 ms), so a slice draws only
  the faces within a few millimetres of it.
- **A surface that moves with a slider** (a warp, a registration stack):
  moved in the shader from its end positions (or its T1 position and the
  slider's affine), so dragging costs no CPU. Its faces are sorted once by
  the range each covers over the whole motion, so a slice still draws only
  the faces that can reach it. The first version drew every face on every
  tile, which a software renderer (SwiftShader) took 25-30 s per redraw to do.
- **Long faces:** the window a slice draws is set by the 99th percentile of
  face spans (at least 3 mm); the few longer faces are checked by every
  slice.

On an RTX 2060, a slice step in a 12-panel T1 montage takes about 30 ms with
or without both surfaces outlined. Slicing on the CPU cost about 1 ms per
surface per panel, and the old bucketing of every face over every warp
position cost 1-2 s per subject.

**Registration stacks.** A map with `images` -- a tuple of `AnatomyImage` -- is
a registration stack. Each image says its frame:

- "subject": in the subject's anatomical world, like their T1;
- "template": in the template's, like the MNI template;
- "affine": `affine` maps the subject's world to the image's, like a boldref
  through its run's coregistration.

The page resamples every image itself, in the overlay shader, and the bar
gains two sliders with button stops:

- **Space** runs from the subject's T1 rigidly aligned to the template (0),
  through the registration's linear part (0.5, "affine"), to the template (1).
  The rigid start is `coords.rigid_from_template`, which the builder derives
  from the linear part: the rotation of its polar decomposition, anchored so
  the two agree at the brain's centre. So the slider moves only what is not
  rigid, first scaling and shear, then the nonlinear warp, and the view does
  not swing round from the scanner's head position. A screen point y stands
  for the subject point x = R y + a (A y - R y) + b (phi(y) - A y), with
  a = min(1, 2 s) and b = max(0, 2 s - 1). R is the rigid part, A the linear
  part and phi the subject's template-to-anatomy field. Each image is read at
  x in its own frame: a template image through the subject's `anat_to_mni`
  field, so the template is inverse-warped into the subject's space as the
  subject is warped into the template's.
- **Image** crossfades between the images in their listed order; **flicker**
  (key F) alternates the two either side of the slider.

Selected regions follow the stack too. The mask is built from the subject's
own atlas (`Atlas.subject_paths`). For each slice on screen, it is sampled
(nearest voxel) at x for every point of a regular grid on the slice plane,
at the atlas's spacing. That slice is then outlined, smoothed and labelled
like any atlas, and filled as squares. So the regions sit on the anatomy at
every Space position. Slices are cached by mask, Space position and slice,
and one costs a few milliseconds. The crosshair's region and value readouts
are taken at x too.

**Resampling.** Each image is genuinely resampled, onto an output grid: the
template image's own voxel grid (1 mm for MNI152), or 1 mm without one.

- **Per output voxel:** every screen point stands for the centre of the
  output voxel it falls in. That centre goes through the registration to x.
- **Reading the source:** the source is read at x by trilinear
  interpolation, so each output voxel shows as one square, as a resampled
  image does.
- **Why not per pixel:** reading every screen pixel through the warp, nearest
  or linear, instead stretches the source voxels into sheared blocks.
- **Regions:** the region slices above are sampled on the same grid.

Checked against ANTs for one participant at Space = MNI:

| Comparison | Result |
| --- | --- |
| Registered points, viewer vs ANTs with fMRIPrep's transform | within 0.12 mm (median), 0.39 mm (95th percentile), 1.2 mm (max) |
| Viewer's resampled T1 vs ANTs resampling the same 1 mm T1 | r = 0.986 |
| Viewer vs fMRIPrep's own T1 in MNI | r = 0.943, against 0.956 for ANTs on the 1 mm T1 |

The small residual is the coordinate field's 4 mm spacing.

Opacity is the map opacity under display -> map. The builder writes a blank
2 mm canvas as the stack's underlay, spanning the subject's and the
template's brain with a margin, which gives the view its extent. The map's
`path` is what the crosshair snaps to and reads values from; the subject's
T1 is the natural choice. A stack needs `display="anatomy"` and the subject's
coordinate maps in `subject_space`.

In surface mode a stack is projected like any map (below), averaged over
the cortex's depth. Each image is sampled trilinearly, scaled by its
1st-99th percentile on the subject's cortex, and crossfaded by the same Image
slider. On the fsaverage end of the
surface slider the images are reached through the subject's `mni_to_anat`
field.

```python
MapEntry(subject="sub-01", feature="run 1", path=t1_brain, images=(
    AnatomyImage("boldref", "boldref", boldref, frame="affine", affine=t1_to_bold),
    AnatomyImage("t1", "T1", t1_brain, frame="subject"),
    AnatomyImage("template", "MNI", template, frame="template"),
))
```

`CoordinateMaps.linear` supplies A (template mm -> subject mm, RAS). Without
it, the builder fits A to the field by least squares, which suits a
registration with no separate affine stage (FNIRT coefficients, for
instance). With fMRIPrep, A is `TransformGroup/1` of the T1w-to-template h5
composite.

**Simpler anatomy endpoints.** Without `images`, `Endpoint(display="anatomy")`
still draws each map in grey over its own `MapEntry.underlay`, with a
**Blend** slider and flicker in the volume views, and grey on surfaces under
the map opacity. `surface_frame` and `surface_affine` place a map's
surfaces, and `MapEntry(warp=True)` offers a single warp slider for the
subject's anatomy warped into the template. These are the building blocks the
stacks replaced in 3DfMRI; they remain for viewers that use them.

### Region information

`region_info` in the manifest, keyed by region name, feeds an info box opened
by clicking a region's label (on the surface or on a slice), the `i` at the end
of its row in the region list, or the region readout in the status bar. It
shows the full name, a description, related work, and on a region-level
endpoint the region's statistics. While it is open it follows the crosshair:
moving onto another region switches the box to that region, whether or not
that region is selected. `fmri_utils.viewer.region_info.harvard_oxford_cortical()`
provides entries for all 48 Harvard-Oxford cortical regions, and
`saxe_tom_parcels()` for the seven Saxe-lab theory-of-mind parcels (Dufour et
al., 2013; named e.g. "Left TPJ (ToM parcel)"). The descriptions
are general neuroanatomy (location, landmarks, the functions each region is
usually associated with), written so they hold for any dataset. Each entry
carries an `attribution`, shown under the description: the descriptions were
written by Claude Opus 5.5, an AI model, and say so. Related-work links are
Google Scholar searches for each paper's exact title.

```jsonc
"region_info": { "Angular Gyrus": { "abbrev": "AG", "description": "…",
  "related": [{ "citation": "Seghier (2013) …", "url": "https://scholar.google.com/…" }] } }
```

## Sharing A View

The page keeps the current view in the URL's hash as you work -- report,
endpoint, feature, subject, mode, window and threshold, blend and opacities,
the crosshair (volume) or camera, unfold and marker (surface), the regions
chosen, and an open info box -- so the address bar is always a link to what is
on screen. **copy link** in the toolbar copies it. Opening the link reproduces
the view; a hash the manifest cannot satisfy (an endpoint that no longer
exists) falls back to the default view.

Links are kept short:

- Settings at their defaults are left out, including the endpoint's own
  window, its first feature and subject, and multi view.
- Lists are joined with `_` and `~`, which need no percent-escaping.
- Region selections are packed per atlas with runs: `rg=cort.1-48~sub.4-11.15-21`
  is 63 regions.
- The info box is named by atlas and value (`info=cort.33`).
- Numbers carry only the digits they need.

A region-test view with a crosshair is about 80 characters after the host.
Links in the older, longer format are no longer read.

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
defaults to true: every panel, montage included, shows the 1 mm anatomy. The
statistical map is not composited onto that grid -- the page's overlay layer
draws it at its own resolution (see *How Maps Are Drawn*) -- so the fine
anatomy costs only its own texture. Set it to false to ask for the coarse
anatomy anyway under maps of 1.8 mm or coarser. Phones showing more than three
panels always get the coarse one.

`smooth_shading` (manifest `features.smoothShading`, default false) sets how
surfaces open: faceted, each triangle lit by its own face normal, or smooth,
lit by interpolated vertex normals. The reader can switch it under display ->
surface -> shading, and a link records the choice (`sh`) when it differs from
the manifest's.

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
      "subject_ranges": { "group": [0.0, 5.0] },   // optional: a subject's own window
      "variants": [{ "id": "observed", "label": "observed t", "blurb": "",  // optional
                     "range": [0.0, 6.0],   // optional: this variant's own window
                     "statistic": "q" }],   // optional: region endpoints, "q" or "p"
      // several controls: each variant also carries "values": {control: option},
      // and optionally "subject_ranges", "threshold_default": {"mode", "value"}
      // and its own "region_stats"
      "variant_controls": [{ "id": "delta", "label": "Δr", "kind": "toggle", "on": "1", "off": "0" },
                           { "id": "test", "label": "Group test",
                             "options": [{ "id": "m", "label": "none" }, …] }],   // optional
      "variant_control": { "label": "Group inference" },                     // optional
      // or, for two variants as one checkbox:
      // "variant_control": { "kind": "toggle", "label": "net of control",
      //                      "tip": "…", "on": "net", "off": "raw" },
      "region_stats": {             // optional: values per region, not per voxel
        "atlas": "ho",              // surface parcellation the rows paint through
        "volume_atlas": ["cort", "sub"],   // atlas ids tested, in priority order
        "statistic": "q", "effect_label": "Δr",
        "rows": [{ "name": "Angular Gyrus", "delta": 0.0068 }]   // optional, shared
      },
      "display": "anatomy",         // optional: images, drawn grey with blend and flicker
      "outlines": "both",           // optional: surface outlines on entering the endpoint
      "maps": [{
        "subject": "sub-01", "feature": "",
        "variant": "",              // optional; empty = shown under every variant
        "path": "data/loc/t_motion/map_sub-01.nii.gz",   // float32
        "significance": { "p": "…_neglog10p.nii.gz", "q": "…", "label": "TFCE FWE" },  // optional
        "region_rows": [{ "name": "Angular Gyrus", "delta": 0.0068, "value": 3.4,
                          "t": 8.0, "p": 9e-5, "q": 4e-4, "significant": true,
                          "n_positive": 8, "n": 8 }],   // optional; region endpoints
        "range": [0.0, 6.4],
        "underlay": "data/underlays/sub-01/boldref_….nii.gz",   // optional: its own underlay
        "surface_frame": "subject",       // optional: "subject" or "template"
        "surface_affine": [[…4 x 4…]],    // optional: subject anatomy mm -> this map's mm
        "warp": true,                     // optional: the subject's anatomy, warped live
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
      "mni_to_anat": { "path": "...bin", "dims": [...], "affine": [...] },
      "linear_from_template": [16 floats]   // the registration's linear part, row-major
    }},
    "maps": [{ "report": "loc", "endpoint": "t_motion", "feature": "",
               "subject": "sub-01", "variant": "", "path": "..." }]
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
      "curvature": "data/surfaces/sub-01/lh_curv.bin",
      "sulc": "data/surfaces/sub-01/lh_sulc.bin"   // optional: sulcal depth, the default shading
    }},
    "rois": { "pSTS": { "path": "data/surfaces/sub-01/rois/pSTS.bin",
                        "n_vertices": 2975 } },
    "parcellations": { "ho": { "label": "Harvard-Oxford",
      "regions": [{ "value": 21, "name": "Angular Gyrus", "abbrev": "AG" }],
      "hemispheres": { "lh": {
        "labels": "…lh_labels.bin",            // int16 per vertex, 0 = medial wall
        "lines_index": "…lh_lines_index.bin",  // uint32 [points][3]
        "lines_weight": "…lh_lines_weight.bin",// float32 [points][3]
        "lines_offsets": "…lh_lines_offsets.bin", // uint32 [lines + 1]
        "anchors": { "21": 10423 }              // label vertex per region
      }}}},
    "volume_transform": [16 floats],   // fsaverage only: surface mm -> template mm (MNI305 -> MNI152)
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

- click or drag in any panel to move the crosshair; panels stay linked, and
  the crosshair snaps to overlay voxels (the wheel steps one voxel)
- right drag zooms, about the point where the drag began, which stays under
  the pointer; middle drag pans
- in montage, zooming or panning one panel zooms and pans them all, and each
  panel has its own status line (value, voxel, selected regions)
- the status bar lists every selected region at the crosshair (cortical and
  subcortical labels can overlap)
- click a region's label on the surface (or its `i` in the list, or the region
  readout) for its info box; `Esc` closes it
- the rail chooses what is shown (report, endpoint, features, statistic,
  subject, regions); the bar chooses how it is viewed (mode, space, view,
  threshold, and in surface mode the geometry jump buttons with the unfold
  and open sliders beside them); everything cosmetic (blend, opacities, the
  exact window, hiding voxels outside the brain, region outline style,
  curvature shading, hemispheres) is in the **display** panel. In surface
  mode the sliders' labels, and the unfold position between geometries, are
  in their tooltips, so the bar stays one row at 1440 px Controls that do not apply to an endpoint are greyed out
  rather than removed, so the bar keeps its layout as you move between
  endpoints
- map voxels outside the brain are hidden by default (display → outside
  brain). The mask is the underlay itself, since the MNI template and each
  subject's T1 are skull-stripped. A map voxel is kept when the anatomy at its
  centre is brain. The mask is built on the map's own grid, so it is small and
  quick to build
- switching subject, endpoint or option keeps the current picture up until
  the next one is ready. A small "loading" chip appears in the corner only if
  the wait passes 0.3 s; the full-screen overlay is for when nothing is
  showing yet
- **copy link** copies a link to exactly what is on screen
- `A` toggles the overlay off and back on, `S` does the same for the regions
- the window has two knobs: below the lower one the map fades out or is clipped,
  depending on the **blend** setting
- in surface mode the **unfold** slider runs white matter → pial → inflated →
  flat, the **separation** slider pulls the hemispheres apart, and when the
  surface is fully flat the camera locks to face it
- on a 3D surface a left drag turns the brain; on the flat map it moves the
  crosshair instead. Middle drag pans, and the wheel or a right drag zooms
  about the pointer. The 3D view is framed by the surface's bounding sphere,
  so turning it never changes its scale; the flat map is framed to fill the
  window
- the status bar is one line on a desktop; a long region readout ends in an
  ellipsis, and its full text is the readout's tooltip
- pinch to zoom and two-finger drag to pan, on both volume and surface views

## How Maps Are Drawn

NiiVue draws the anatomy and nothing else. The statistical map and the
selected regions are drawn by a small WebGL layer over each NiiVue canvas,
which samples them in their own voxel grids at screen resolution. A 2.5 mm
voxel over 1 mm anatomy is a 2.5 mm square, straddling underlay voxels
wherever its affine puts it -- not resampled onto the underlay grid, which is
what NiiVue's own overlays do and what made voxels alternate between 2 and 3
mm wide. The layer uses NiiVue's own per-tile pixel-to-millimetre geometry, so
it stays registered through zoom and pan.

All layers share **one** WebGL context. Chrome keeps only about sixteen live
WebGL contexts per page and silently kills the oldest past that, which breaks
the page. A context per layer took an eight-panel montage over the limit. Each
layer is rendered in the shared context and copied onto a plain 2D canvas
over its panel. Surface outlines are drawn in the shared context before the
copy, and region outlines onto the 2D canvas after it. A
montage of eight uses about ten contexts in all. Surface faces are read
directly from the mz3 files (`NVMeshLoaders.readMZ3`), not loaded into the
NiiVue instance, where every load re-uploaded its volume. Anything that adds a
canvas per panel should keep this budget in mind.

The layer also draws the crosshair: thin, translucent, and snapped to the
centre of an overlay voxel. Clicking lands on a voxel centre, one wheel notch
moves one overlay voxel, and the status bar reports the overlay's voxel and
value. The 3D volume render is gone; it could not show the layer.

## Driving It From Another Page

A page in the same origin (a slide deck with the viewer in an iframe) can
drive it through `window.__viewer`:

- `markVertex(hemisphere, vertex)`: moves the surface marker and readout to a
  vertex, as a click would. False when not in surface mode or the vertex is
  not loaded.
- `setCamera(azimuth, elevation, zoom?)`: the orbit drag's own steps (azimuth
  wraps, elevation within ±89°), cheap enough to call every frame. A
  flattened surface's camera follows the morph instead, so animate only
  before flattening.
- `vertexDirection(hemisphere, vertex, from?)`: the `[azimuth, elevation]`
  that faces a vertex of the geometry on screen. `from` is `"hemisphere"`
  (default; the direction from the hemisphere's centroid) or `"scene"` (from
  the centre the view is framed on, which also puts the vertex mid-frame).

Such pages also click the page's own buttons and hide its chrome. Keep these
names stable, since the LeBel lab-meeting deck relies on them:
`#modes [data-mode-kind]`, `#views [data-view]`, `#geometries [data-geometry]`,
`#hemispheres [data-hemi]`, `#threshold-modes [data-mode]`,
`#features [data-feature="<feature id>"]`,
`#variants [data-control="<control id>"] [data-option="<option id>"]` (a
variant control's buttons; the ids are the manifest's), `#variants
[data-variant]` (an endpoint's plain variant buttons), and the `.rail`, `.bar`,
`.shell` and `.status` classes. The colour bar (`#colorbar`) is drawn for its
CSS height, so a host can enlarge it with CSS alone.

## Extending It

The page is [`resources/index.html`](../src/fmri_utils/viewer/resources/index.html),
one self-contained file with no build step: plain ES5, NiiVue from a CDN for
volumes, and a small WebGL2 renderer of its own for surfaces. Edit it, point
`build_viewer` at your copy by overwriting `viewer/index.html` afterwards, or
fork the file and keep the manifest contract.

The two rules worth keeping if you change it: the page reads capability from
the manifest rather than hard-coding a study, and it degrades to whatever is
present rather than failing on what is missing.

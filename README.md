# fmri_utils

Reusable fMRI utilities for:
- voxelwise stimulus-to-BOLD encoding analysis
- explicit first-level and group-level Nilearn GLMs
- second-level group analysis
- fMRIPrep-based transforms
- montage rendering
- decoding/searchlight helpers
- static web viewers for brain maps: volumes, surfaces, regions and registration checks
- LLM ratings of stories, and text features (language models) on a scanner clock
- encoding on column data (run tables), annotation ablations, cross-participant encoding, group inference
- web slide decks that embed the viewers, with an in-browser editor

## Quick Start

Install from GitHub:

```bash
pip install "git+https://github.com/efirdc/fmri_utils.git"
```

Editable install for development:

```bash
git clone https://github.com/efirdc/fmri_utils.git
cd fmri_utils
pip install -e .
```

## Documentation

See [docs/README.md](docs/README.md) for the full docs index.

New GLM analyses should use the focused [first-level](docs/first_level_glm.md) and
[group-level](docs/group_level_glm.md) interfaces. The older combined second-level helper
is retained for compatibility.

Direct links:
- [Installation](docs/installation.md)
- [Second-level CLI](docs/second_level_cli.md)
- [Second-level Python API](docs/second_level_python.md)
- [Montage API](docs/montage_api.md)
- [Registration QC](docs/registration_qc.md)
- [Transformations](docs/transformations.md)
- [Voxelwise Encoding](docs/encoding.md)
- [Chunked Encoding And Stitching](docs/encoding_chunked.md)
- [Encoding Output Reference](docs/encoding_outputs.md)
- [Temporal Continuous Decoding](docs/temporal_decoding.md)
- [Result Viewer](docs/viewer.md)
- [Story Ratings](docs/story_ratings.md)
- [Text Features](docs/features.md)
- [Encoding On Column Data](docs/encoding_columns.md)
- [Ablating An Annotation](docs/ablation.md)
- [Cross-Participant Encoding](docs/cross_participant.md)
- [Group Inference On Encoding Maps](docs/group_inference.md)
- [Slide Decks](docs/slides.md)

## Example Outputs

The registration QC docs include example heatmap/APNG outputs so users can check that their render geometry, resolution, and contour overlays look sensible:

- [Registration QC](docs/registration_qc.md)

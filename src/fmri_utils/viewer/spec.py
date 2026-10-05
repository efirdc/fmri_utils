"""What a viewer is made of.

A viewer is one static page plus a manifest plus the files the manifest points
at. These dataclasses describe it in the terms an analysis already thinks in --
reports, endpoints, maps, subjects -- and the builder turns that into the
files. Nothing here knows about any particular study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class AnatomyImage:
    """One image of a registration stack (see ``MapEntry.images``).

    ``frame`` says where the image lives relative to the subject:
    "subject" (the subject's anatomical world, like their T1), "template"
    (the template's, like the MNI template itself) or "affine", for which
    ``affine`` (4 x 4, RAS mm) maps the subject's anatomical world to the
    image's (a boldref, through the run's coregistration).
    """

    id: str
    label: str
    path: Path
    frame: str = "subject"
    affine: Sequence[Sequence[float]] | None = None


@dataclass(frozen=True)
class MapEntry:
    """One statistical map, in one space, for one subject.

    ``path`` is the map in the template space every subject shares.
    ``subject_space_path`` is the same map in that subject's own anatomy, and
    is what the surface view samples; leave it out and the subject keeps only
    its template-space map.
    """

    subject: str
    path: Path
    feature: str = ""
    subject_space_path: Path | None = None
    # Which of the endpoint's variants this map belongs to. Leave it empty and
    # the map is shown whichever variant is chosen.
    variant: str = ""
    # On a region-level endpoint (``Endpoint.region_stats``), this map's own
    # per-region results; they win over the endpoint's shared rows.
    region_rows: Sequence["RegionRow"] = ()
    # Significance companions: -log10 p (and -log10 q, and -log10 of a
    # family-wise p) on this map's grid, from a test the page cannot redo
    # (permutations, TFCE with sign flipping, a region test). In the page's p<,
    # q< and FWE< modes the map keeps its own colours and is shown only where
    # the companion clears the level; the FWE< mode appears only for maps that
    # have that companion. ``significance_label`` names the test in the
    # threshold readout ("TFCE FWE"); ``significance_defaults`` sets the level
    # a mode opens at, e.g. {"p": 0.005} for an uncorrected permutation p.
    significance_p: Path | None = None
    significance_q: Path | None = None
    significance_fwe: Path | None = None
    significance_label: str = ""
    significance_defaults: Mapping[str, float] = field(default_factory=dict)
    # The image drawn under this map instead of the template (or the
    # subject's anatomy in subject space): a registration check shows a
    # boldref under a resampled T1, in a space no other map is in.
    underlay: Path | None = None
    # Where the subject's white and pial surfaces sit relative to this map,
    # for the volume view's surface outlines. "" works it out: a template-space
    # map gets them through the subject's coordinate field, a subject-space map
    # as they are. "subject" says this map is in the subject's anatomical world
    # although it is listed as ``path``. ``surface_affine`` (4 x 4, RAS mm)
    # maps the subject's anatomical world to this map's, for any other frame.
    surface_frame: str = ""
    surface_affine: Sequence[Sequence[float]] | None = None
    # This map is the subject's anatomy (``SubjectSpace.templates``) warped
    # into the template. The page can then redo the warp itself at any
    # fraction between the registration's linear part and the whole of it,
    # from the subject's coordinate field (see ``CoordinateMaps.linear``).
    warp: bool = False
    # A registration stack: images the page resamples itself into any space
    # between the subject's anatomy and the template, crossfading between
    # them in this order. ``path`` is still the map values are read from (the
    # subject's T1, say). Needs an anatomy endpoint and the subject's
    # coordinate maps; the builder writes a blank underlay big enough for
    # every image in both spaces.
    images: Sequence[AnatomyImage] = ()
    # Which of the viewer's cohorts (``ViewerSpec.cohorts``) this map belongs
    # to. A map with a cohort shows only while that cohort is chosen; one with
    # none is shared, shown in every cohort that has no map of its own under
    # the same subject, feature and variant. So a result that exists only for
    # one cohort names it, and the others do not show it.
    cohort: str = ""
    # The degrees of freedom this map's p and q curves use, when they differ
    # from the endpoint's for its subject (the same group t map over another
    # cohort of participants).
    degrees_of_freedom: int | None = None


@dataclass(frozen=True)
class Cohort:
    """One set of participants a viewer can show its results for.

    The first is the default. A reader switches between them in the rail; each
    map is shown from the chosen cohort if it has one there, else the shared
    map if there is one, else not at all (``MapEntry.cohort``).
    """

    id: str
    label: str
    blurb: str = ""


@dataclass(frozen=True)
class Variant:
    """The same maps computed another way, chosen under the endpoint.

    ``analytic`` says whether the page may threshold these maps by p and q
    from the endpoint's degrees of freedom. A map that is already a corrected
    result -- a permutation FDR mask, say -- is not a t map to be corrected
    again, so its variant sets it False and the page offers value only.
    """

    id: str
    label: str
    blurb: str = ""
    display_range: tuple[float, float] | None = None
    analytic: bool = True
    # On a region-level endpoint, which statistic this variant's labels and
    # readouts report ("q" or "p"); empty keeps the endpoint's.
    statistic: str = ""
    # With several controls (``Endpoint.variant_controls``): which option of
    # each control this variant is, as {control id: option id}. A control the
    # variant leaves out does not apply to it and is hidden while it is shown.
    values: Mapping[str, str] = field(default_factory=dict)
    # A window per subject for this variant (the group mean beside the
    # individual maps), as in ``Endpoint.subject_display_ranges``.
    subject_display_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    # The threshold mode and level to switch to when the reader picks this
    # variant, e.g. ("p", 0.05) for a corrected test; None leaves it alone.
    threshold_default: tuple[str, float] | None = None
    # Makes this variant, and only it, region-level (see RegionStats): a
    # region test as one of an endpoint's choices. Wins over the endpoint's.
    region_stats: "RegionStats | None" = None


@dataclass(frozen=True)
class VariantToggle:
    """Two variants that are one idea switched on or off, shown as a checkbox.

    ``on`` and ``off`` are variant ids; ``tip`` is the explanation shown on
    hover. A toggle keeps its setting when the reader moves to another
    endpoint whose toggle has the same ``label`` -- "net of control" on two
    ablations, say -- so flipping endpoints compares like with like.
    """

    label: str
    on: str
    off: str
    tip: str = ""


@dataclass(frozen=True)
class VariantOption:
    id: str
    label: str
    tip: str = ""
    # Shown on the greyed-out button when the subject or feature on screen has
    # no map for this option ("TFCE was run on the mean of the features").
    unavailable: str = ""


@dataclass(frozen=True)
class VariantControl:
    """One of several choices about an endpoint's maps.

    An endpoint with ``variant_controls`` shows one control per entry: a
    checkbox when ``on`` and ``off`` are given, a segmented choice over
    ``options`` otherwise. Each variant names its option of every control in
    ``Variant.values``. The page keeps the reader's picks and shows the
    variant, among those the subject and feature on screen have, that agrees
    with the most important of them (earlier controls outrank later ones). A
    control shows only while the variant on screen has a value for it and the
    subject has at least two of its options. Picks carry over to another
    endpoint with a control of the same ``id``.
    """

    id: str
    label: str
    options: Sequence[VariantOption] = ()
    on: str = ""
    off: str = ""
    tip: str = ""

    @property
    def is_toggle(self) -> bool:
        return bool(self.on or self.off)


@dataclass(frozen=True)
class RegionRow:
    """One region's result on a region-level endpoint.

    ``name`` must match the region's name in the atlas (volume) and the
    parcellation (surface). ``value`` is what the region is painted with in
    the endpoint's maps -- signed -log10 q, say -- and ``delta`` the effect the
    labels show; ``significant`` puts a star on the label.
    """

    name: str
    delta: float
    value: float | None = None
    t: float | None = None
    p: float | None = None
    q: float | None = None
    significant: bool = False
    n_positive: int | None = None
    n: int | None = None
    atlas: str = ""


@dataclass(frozen=True)
class RegionStats:
    """Makes an endpoint region-level: its values are per region, not voxel.

    The maps are ordinary volumes with each region painted with its value, so
    volume mode works as for any map. ``atlas`` is the id of the surface
    parcellation the page paints the rows through (so a region's colour ends
    exactly at its drawn boundary). ``volume_atlas`` lists the ``Atlas`` ids
    whose regions were tested, in priority order where two label the same
    voxel -- the order the maps were painted in. Entering the endpoint selects
    exactly those regions, outlined in white, with labels like ``AG* /
    +0.0068 / q=4e-4``. ``rows`` are shared by every map unless a map brings
    its own ``region_rows``.
    """

    atlas: str
    volume_atlas: Sequence[str] = ()
    statistic: str = "q"
    effect_label: str = "effect"
    rows: Sequence[RegionRow] = ()


@dataclass(frozen=True)
class Endpoint:
    """A quantity the reader can look at: a t map, a correlation, a contrast.

    ``statistic`` drives which thresholds the page offers. "t" or "r" earn an
    uncorrected-p and an FDR-q mode, because a p value can be worked out from
    the value and the degrees of freedom; anything else is thresholded by
    value alone.

    ``display_range`` pins the window the controls open at, as ``(threshold,
    top)``. Leave it out and each endpoint gets its own, from its maps; set the
    same one on several and a colour means the same thing across all of them,
    which is what a comparison needs. A threshold of zero means "a fifth of the
    top", which is the page's own default.

    ``subject_display_ranges`` gives a subject its own window. It is how a
    group statistic sits in the same endpoint as the individual maps it was
    made from: a group t map beside first-level contrasts is a different
    quantity, so ``{"group": (0, 5)}`` opens it on its own scale, the page
    resets the window when the reader moves between them, and the montage of
    individuals leaves it out. The endpoint's range is then worked out from the
    other subjects only. ``degrees_of_freedom`` decides separately which maps
    get p and q curves, so a group t map can be thresholded by p while the
    contrasts beside it are thresholded by value.
    """

    id: str
    label: str
    maps: Sequence[MapEntry]
    statistic: str = ""
    degrees_of_freedom: Mapping[str, int] = field(default_factory=dict)
    blurb: str = ""
    warp: str = ""
    display_range: tuple[float, float] | None = None
    subject_display_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    variants: Sequence[Variant] = ()
    # The heading over the variant choice; empty means the page's "Statistic".
    variant_label: str = ""
    # Show exactly two variants as one checkbox instead of a choice.
    variant_toggle: VariantToggle | None = None
    # Several choices at once; see VariantControl. Overrides variant_label
    # and variant_toggle.
    variant_controls: Sequence[VariantControl] = ()
    # Values per region rather than per voxel; see RegionStats.
    region_stats: RegionStats | None = None
    # "anatomy": the maps are images, not statistics. They are drawn in grey
    # over their underlay with a blend slider and a flicker instead of a
    # threshold, which is what checking one image's alignment to another asks.
    display: str = ""
    # Surface outlines to switch on when the reader enters this endpoint:
    # "white", "pial" or "both". Empty leaves the reader's setting alone.
    outlines: str = ""
    # The template its template-space maps are in ("MNI152NLin6Asym"), when it
    # differs from the viewer's (ViewerSpec.template_space).
    template_space: str = ""


@dataclass(frozen=True)
class Report:
    """A group of endpoints that belong to one analysis."""

    id: str
    label: str
    endpoints: Sequence[Endpoint]
    blurb: str = ""


@dataclass(frozen=True)
class Atlas:
    """A label volume whose regions the reader can outline.

    ``labels`` is a sequence of ``(value, name)``. ``subject`` maps a subject
    to the same atlas warped into that subject's anatomy, for the views that
    show subject space.
    """

    id: str
    label: str
    labels: Sequence[tuple[int, str]]
    template_path: Path
    subject_paths: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class CoordinateMaps:
    """Per-subject lookups between a subject's anatomy and the template.

    They are what lets a montage of different brains hold one crosshair: a
    click in one panel becomes template millimetres and then each other
    subject's own millimetres.
    """

    to_template: Path
    from_template: Path
    # Every stride-th voxel is kept. The field is smooth and the page
    # interpolates it, so a 1 mm field at 4 or a 2 mm field at 2 is plenty.
    stride: int = 4
    # The registration's linear part, template mm -> subject mm (4 x 4, RAS),
    # for the warp slider: fraction 0 shows the subject's anatomy under this
    # alone, fraction 1 under the whole field. Leave it out and the builder
    # fits it to ``from_template`` by least squares, which is what a warp
    # without a separate affine stage (FNIRT's coefficients, say) calls for.
    linear: Sequence[Sequence[float]] | None = None


@dataclass(frozen=True)
class SubjectSpace:
    """Everything the subject-anatomy views need."""

    templates: Mapping[str, Path]
    templates_coarse: Mapping[str, Path] = field(default_factory=dict)
    coordinates: Mapping[str, CoordinateMaps] = field(default_factory=dict)


@dataclass(frozen=True)
class Features:
    """Switches for things the data would otherwise turn on by itself.

    A viewer offers whatever its manifest has: surfaces if surfaces were
    exported, regions if an atlas was given, subject space if subject
    templates were given. These say "not this one" -- for a project that has
    the data but does not want the control.

    ``fine_underlay`` is not a control but a rendering choice, and it is here
    because it is the same kind of thing: leave it on and a map is composited
    onto the full-resolution anatomy, which is sharp and slow; turn it off and
    a map coarser than 1.8 mm gets the coarse underlay instead, which is eight
    times less blend texture per panel and, for a 2 or 3 mm map, shows exactly
    the same voxels.
    """

    surface: bool = True
    regions: bool = True
    montage: bool = True
    subject_space: bool = True
    fine_underlay: bool = True
    # Where the surface view opens: faceted (one normal per triangle) or smooth
    # (interpolated vertex normals). The reader can switch it under display.
    smooth_shading: bool = False


@dataclass(frozen=True)
class About:
    """The words on the page: what this viewer is and how to read it."""

    title: str = "Result Browser"
    kicker: str = ""
    lede: str = ""
    footnote: str = ""


@dataclass(frozen=True)
class ViewerSpec:
    """A whole viewer.

    ``template`` is the anatomy every template-space map is drawn on;
    ``template_coarse`` is an optional lower-resolution copy for montages,
    where a panel is a quarter of the window wide and the finer one costs
    eight times the texture for nothing.
    """

    reports: Sequence[Report]
    template: Path
    about: About = field(default_factory=About)
    template_coarse: Path | None = None
    subject_space: SubjectSpace | None = None
    atlases: Sequence[Atlas] = ()
    surfaces: Path | None = None
    features: Features = field(default_factory=Features)
    percentile: float = 99.5
    # Descriptions and related work for the region info box, keyed by region
    # name; see fmri_utils.viewer.region_info.
    region_info: dict | None = None
    # Switchable participant sets (see ``Cohort``), and participants left out
    # of group results, with the reason; the rail marks the latter.
    cohorts: Sequence[Cohort] = ()
    excluded_subjects: Mapping[str, str] = field(default_factory=dict)
    # The template the template-space maps are in, by its TemplateFlow name
    # ("MNI152NLin6Asym", "MNI152NLin2009cAsym"). On fsaverage, a map in a
    # space the surface export has registration-fusion points for is sampled
    # once per vertex at those points; any other map goes through fsaverage's
    # MNI305 affine. Empty declares nothing, so every map takes the affine.
    template_space: str = ""

    def subjects(self) -> list[str]:
        seen: list[str] = []
        for report in self.reports:
            for endpoint in report.endpoints:
                for entry in endpoint.maps:
                    if entry.subject not in seen:
                        seen.append(entry.subject)
        return seen

    def check(self) -> None:
        """Fail early, with the path that is wrong, rather than half way through."""
        if not self.reports:
            raise ValueError("a viewer needs at least one report")
        missing = [str(self.template)] if not Path(self.template).exists() else []
        cohort_ids = [cohort.id for cohort in self.cohorts]
        if len(set(cohort_ids)) != len(cohort_ids) or any(not c for c in cohort_ids):
            raise ValueError("cohort ids must be unique and non-empty")
        for report in self.reports:
            if not report.endpoints:
                raise ValueError(f"report {report.id} has no endpoints")
            for endpoint in report.endpoints:
                if not endpoint.maps:
                    raise ValueError(f"endpoint {report.id}/{endpoint.id} has no maps")
                variant_ids = {variant.id for variant in endpoint.variants}
                toggle = endpoint.variant_toggle
                if toggle is not None:
                    if len(endpoint.variants) != 2 or {toggle.on, toggle.off} != variant_ids:
                        raise ValueError(f"{report.id}/{endpoint.id}: a variant toggle needs "
                                         "exactly two variants, named by its on and off")
                if endpoint.variant_controls:
                    controls = {c.id: c for c in endpoint.variant_controls}
                    if len(controls) != len(endpoint.variant_controls):
                        raise ValueError(f"{report.id}/{endpoint.id}: variant control ids repeat")
                    for variant in endpoint.variants:
                        for key, value in variant.values.items():
                            control = controls.get(key)
                            allowed = ({control.on, control.off} if control and control.is_toggle
                                       else {o.id for o in control.options} if control else set())
                            if value not in allowed:
                                raise ValueError(f"{report.id}/{endpoint.id}: variant {variant.id!r} "
                                                 f"gives {key}={value!r}, which no control offers")
                for stats in [endpoint.region_stats] + [v.region_stats for v in endpoint.variants]:
                    if stats is None:
                        continue
                    known = {atlas.id for atlas in self.atlases}
                    unknown = [a for a in stats.volume_atlas if a not in known]
                    if unknown:
                        raise ValueError(f"{report.id}/{endpoint.id}: region_stats names "
                                         f"atlases that are not in the spec: {unknown}")
                    if stats.statistic not in ("q", "p"):
                        raise ValueError(f"{report.id}/{endpoint.id}: region_stats.statistic "
                                         "must be 'q' or 'p'")
                if endpoint.display not in ("", "anatomy"):
                    raise ValueError(f"{report.id}/{endpoint.id}: display must be '' or 'anatomy'")
                if endpoint.outlines not in ("", "white", "pial", "both"):
                    raise ValueError(f"{report.id}/{endpoint.id}: outlines must be '', 'white', "
                                     "'pial' or 'both'")
                keys = set()
                for entry in endpoint.maps:
                    if entry.cohort and entry.cohort not in cohort_ids:
                        raise ValueError(f"{report.id}/{endpoint.id}: map cohort {entry.cohort!r} "
                                         "is not one of the viewer's cohorts")
                    key = (entry.subject, entry.feature, entry.variant, entry.cohort or "*shared*")
                    if key in keys:
                        raise ValueError(f"{report.id}/{endpoint.id}: two maps for {key}")
                    keys.add(key)
                    if not Path(entry.path).exists():
                        missing.append(str(entry.path))
                    if entry.underlay and not Path(entry.underlay).exists():
                        missing.append(str(entry.underlay))
                    if entry.surface_frame not in ("", "subject", "template"):
                        raise ValueError(f"{report.id}/{endpoint.id}: surface_frame must be '', "
                                         "'subject' or 'template'")
                    if entry.surface_affine is not None and len(_matrix(entry.surface_affine)) != 4:
                        raise ValueError(f"{report.id}/{endpoint.id}: surface_affine must be 4 x 4")
                    if entry.images:
                        if endpoint.display != "anatomy":
                            raise ValueError(f"{report.id}/{endpoint.id}: an image stack needs "
                                             "display='anatomy'")
                        space = self.subject_space
                        if not space or entry.subject not in space.coordinates:
                            raise ValueError(f"{report.id}/{endpoint.id}: an image stack needs "
                                             f"{entry.subject}'s coordinate maps in subject_space")
                        for image in entry.images:
                            if not Path(image.path).exists():
                                missing.append(str(image.path))
                            if image.frame not in ("subject", "template", "affine"):
                                raise ValueError(f"{report.id}/{endpoint.id}: image frame must be "
                                                 "'subject', 'template' or 'affine'")
                            if image.frame == "affine" and len(_matrix(image.affine or [])) != 4:
                                raise ValueError(f"{report.id}/{endpoint.id}: an 'affine' image "
                                                 "needs a 4 x 4 affine")
                    if entry.warp:
                        space = self.subject_space
                        if not space or entry.subject not in space.templates                                 or entry.subject not in space.coordinates:
                            raise ValueError(f"{report.id}/{endpoint.id}: a warp map needs "
                                             f"{entry.subject}'s anatomy and coordinate maps "
                                             "in subject_space")
                    if entry.subject_space_path and not Path(entry.subject_space_path).exists():
                        missing.append(str(entry.subject_space_path))
                    for companion in (entry.significance_p, entry.significance_q, entry.significance_fwe):
                        if companion and not Path(companion).exists():
                            missing.append(str(companion))
                    unknown = set(entry.significance_defaults) - {"p", "q", "fwe"}
                    if unknown:
                        raise ValueError(f"{report.id}/{endpoint.id}: significance_defaults "
                                         f"names {sorted(unknown)}; modes are p, q and fwe")
                    if entry.variant and entry.variant not in variant_ids:
                        raise ValueError(f"{report.id}/{endpoint.id}: map variant "
                                         f"{entry.variant!r} is not one of the endpoint's variants")
        for atlas in self.atlases:
            if not Path(atlas.template_path).exists():
                missing.append(str(atlas.template_path))
            missing.extend(str(p) for p in atlas.subject_paths.values() if not Path(p).exists())
        if self.subject_space:
            for maps in self.subject_space.coordinates.values():
                missing.extend(str(p) for p in (maps.to_template, maps.from_template)
                               if not Path(p).exists())
        if missing:
            raise FileNotFoundError(
                "these inputs are missing:\n  " + "\n  ".join(sorted(set(missing))[:20])
            )


def _matrix(values) -> list[list[float]]:
    """A 4 x 4 as nested lists of floats, from any nested sequence or array."""
    rows = [[float(v) for v in row] for row in values]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        return []
    return rows

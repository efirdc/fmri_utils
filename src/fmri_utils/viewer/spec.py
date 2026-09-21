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
    """

    id: str
    label: str
    maps: Sequence[MapEntry]
    statistic: str = ""
    degrees_of_freedom: Mapping[str, int] = field(default_factory=dict)
    blurb: str = ""
    warp: str = ""
    display_range: tuple[float, float] | None = None


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
        for report in self.reports:
            if not report.endpoints:
                raise ValueError(f"report {report.id} has no endpoints")
            for endpoint in report.endpoints:
                if not endpoint.maps:
                    raise ValueError(f"endpoint {report.id}/{endpoint.id} has no maps")
                for entry in endpoint.maps:
                    if not Path(entry.path).exists():
                        missing.append(str(entry.path))
                    if entry.subject_space_path and not Path(entry.subject_space_path).exists():
                        missing.append(str(entry.subject_space_path))
        for atlas in self.atlases:
            if not Path(atlas.template_path).exists():
                missing.append(str(atlas.template_path))
        if missing:
            raise FileNotFoundError(
                "these inputs are missing:\n  " + "\n  ".join(sorted(set(missing))[:20])
            )

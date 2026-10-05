"""Group statistics on subjects' maps in a common template space (MNI152 2 mm, typically).

Each subject's map is a vector over its response columns (r, delta r, a null-standardised drop), carried
to the template with its ``fsl_transforms.ColumnMNIWarp`` (or any object with ``apply`` and
``template_index``). Over the template voxels every subject reaches:

``sign_flip``    voxelwise one-sample t over subjects, two-sided p, Benjamini-Hochberg q, and a
                 family-wise p from the maximum |t| over all 2^n sign flips (n = 8: 256 patterns, so
                 the smallest family-wise p is 1/256 and only very large effects pass it);
``region_test``  the subjects' mean within each atlas region (Harvard-Oxford, or any label image),
                 one-sample t over subjects, BH q over regions: fewer questions than voxels, so it is
                 where eight subjects have power;
``null_draws``   (with the optional null) the group mean drop against 10,000 draws of one null per
                 subject: z, one-sided p, BH q, and family-wise p from each draw's largest z. A
                 fixed-effects question (do these subjects' drops beat the null), not a population one.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .fsl_transforms import MNI_AFFINE, MNI_SHAPE, ColumnMNIWarp, save_template_map


def bh_adjusted(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (q), in the input order."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(p.size)
    out[order] = np.minimum(ranked, 1.0)
    return out


def to_template(maps: dict[str, np.ndarray], warps: dict[str, ColumnMNIWarp]) -> tuple[np.ndarray, np.ndarray]:
    """Subjects' column maps -> (subjects x voxels) on the template voxels all of them reach, and those voxels."""
    on_mni = {s: warps[s].apply(v) for s, v in maps.items()}
    common = None
    for s, values in on_mni.items():
        reach = set(warps[s].template_index[np.isfinite(values)].tolist())
        common = reach if common is None else common & reach
    voxels = np.array(sorted(common), dtype=np.int64)
    stack = np.stack([on_mni[s][np.searchsorted(warps[s].template_index, voxels)] for s in maps])
    return stack, voxels


def volume(values: np.ndarray, voxels: np.ndarray, fill=np.nan) -> np.ndarray:
    out = np.full(int(np.prod(MNI_SHAPE)), fill, dtype=np.float32)
    out[voxels] = values
    return out.reshape(MNI_SHAPE)


def sign_flip(data: np.ndarray) -> dict[str, np.ndarray]:
    """data: subjects x voxels (finite). t, two-sided p, BH q and the exact sign-flip family-wise p."""
    n = data.shape[0]
    mean, sd = data.mean(axis=0), data.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, mean / (sd / np.sqrt(n)), 0.0)
    p = 2 * stats.t.sf(np.abs(t), n - 1)
    patterns = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    sum_sq = (data ** 2).sum(axis=0)
    max_t = np.empty(len(patterns))
    for i, signs in enumerate(patterns):
        flipped = signs @ data / n
        var = (sum_sq - n * flipped ** 2) / (n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = np.where(var > 0, flipped / np.sqrt(var / n), 0.0)
        max_t[i] = np.abs(tt).max()
    ordered = np.sort(max_t)
    fwe = (len(max_t) - np.searchsorted(ordered, np.abs(t) - 1e-9, side="left")) / len(max_t)
    return {"mean": mean, "t": t, "p": p, "q": bh_adjusted(p), "fwe": fwe, "max_t_95": float(np.percentile(max_t, 95))}


def region_test(stack: np.ndarray, voxels: np.ndarray, atlases: dict[str, tuple[np.ndarray, dict[int, str]]],
                min_voxels: int = 25) -> list[dict]:
    """One-sample t over subjects of each region's mean (``atlases``: name -> (template label volume, names))."""
    rows = []
    for atlas_name, (labels, names) in atlases.items():
        at = np.asarray(labels).reshape(-1)[voxels]
        for value, name in names.items():
            region = at == value
            if region.sum() < min_voxels:
                continue
            per_subject = np.nanmean(stack[:, region], axis=1)
            if not np.isfinite(per_subject).all():
                continue
            n, spread = per_subject.size, per_subject.std(ddof=1)
            t = per_subject.mean() / (spread / np.sqrt(n)) if spread > 0 else 0.0
            rows.append({"atlas": atlas_name, "region": name, "voxels": int(region.sum()),
                         "mean": float(per_subject.mean()), "t": float(t), "p": float(2 * stats.t.sf(abs(t), n - 1)),
                         "subjects_positive": int((per_subject > 0).sum())})
    for row, q in zip(rows, bh_adjusted(np.array([r["p"] for r in rows]))):
        row["q"] = float(q)
    return sorted(rows, key=lambda r: -r["t"])


def harvard_oxford(fsldir_or_folder: Path) -> dict[str, tuple[np.ndarray, dict[int, str]]]:
    """Harvard-Oxford cortical and subcortical (max-probability, 25%, 2 mm) with their names; tissue labels dropped."""
    import xml.etree.ElementTree as ET
    import nibabel as nib
    folder = Path(fsldir_or_folder)
    if (folder / "data/atlases").is_dir():
        folder = folder / "data/atlases"
    tissue = {"Left Cerebral White Matter", "Right Cerebral White Matter", "Left Cerebral Cortex", "Right Cerebral Cortex",
              "Left Lateral Ventrical", "Right Lateral Ventricle", "Left Lateral Ventricle", "Brain-Stem"}
    out = {}
    for name, image, xml in (("ho_cortical", "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Cortical.xml"),
                             ("ho_subcortical", "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz", "HarvardOxford-Subcortical.xml")):
        image_path = next((p for p in (folder / image, folder / "HarvardOxford" / image) if p.exists()), None)
        xml_path = next((p for p in (folder / xml, folder / "HarvardOxford" / xml) if p.exists()), None)
        if image_path is None or xml_path is None:
            continue
        names = {int(node.get("index")) + 1: (node.text or "").strip() for node in ET.parse(xml_path).getroot().iter("label")}
        out[name] = (np.asarray(nib.load(image_path).dataobj).astype(int),
                     {v: n for v, n in names.items() if n not in tissue})
    return out


def null_draws(observed: dict[str, np.ndarray], nulls: dict[str, np.ndarray], warps: dict[str, ColumnMNIWarp],
               n_draws: int = 10000, seed: int = 0) -> dict[str, np.ndarray]:
    """observed[s]: a subject's drop over its columns; nulls[s]: nulls x columns. Group z, p, q, family-wise p."""
    subjects = list(observed)
    on = {s: (warps[s].apply(observed[s]), warps[s].apply(nulls[s].T)) for s in subjects}
    common = None
    for s in subjects:
        reach = set(warps[s].template_index[np.isfinite(on[s][0]) & np.isfinite(on[s][1]).all(axis=1)].tolist())
        common = reach if common is None else common & reach
    voxels = np.array(sorted(common), dtype=np.int64)
    columns = {}
    for s in subjects:
        position = np.searchsorted(warps[s].template_index, voxels)
        columns[s] = (on[s][0][position], np.ascontiguousarray(on[s][1][position].T))
    n, n_nulls = len(subjects), columns[subjects[0]][1].shape[0]
    group = np.mean([columns[s][0] for s in subjects], axis=0)
    null_mean = np.mean([columns[s][1].mean(axis=0) for s in subjects], axis=0)
    null_sd = np.sqrt(np.sum([columns[s][1].var(axis=0) for s in subjects], axis=0)) / n
    z = (group - null_mean) / null_sd
    rng = np.random.default_rng(seed)
    exceed = np.zeros(voxels.size, dtype=np.int64)
    max_z = np.empty(n_draws)
    for start in range(0, n_draws, 250):
        size = min(250, n_draws - start)
        picks = rng.integers(0, n_nulls, size=(size, n))
        draws = np.zeros((size, voxels.size), dtype=np.float32)
        for i, s in enumerate(subjects):
            draws += columns[s][1][picks[:, i]]
        draws /= n
        exceed += (draws >= group[None]).sum(axis=0)
        max_z[start:start + size] = ((draws - null_mean[None]) / null_sd[None]).max(axis=1)
    p = (1 + exceed) / (n_draws + 1)
    fwe = (1 + n_draws - np.searchsorted(np.sort(max_z), z - 1e-9, side="left")) / (n_draws + 1)
    return {"voxels": voxels, "mean": group, "net": group - null_mean, "z": z, "p": p, "q": bh_adjusted(p), "fwe": fwe}


def save_results(result: dict, voxels: np.ndarray, output: Path, prefix: str) -> list[Path]:
    """Writes ``<prefix>_<name>_space-MNI152.nii.gz`` for each map (p, q, fwe as -log10)."""
    paths = []
    for name, values in result.items():
        if not isinstance(values, np.ndarray) or values.shape != voxels.shape or name == "voxels":
            continue
        if name in ("p", "q", "fwe"):
            values, name = -np.log10(np.maximum(values, 1e-300)), f"neglog10{name}"
        paths.append(save_template_map(volume(values, voxels), Path(output) / f"{prefix}_{name}_space-MNI152.nii.gz"))
    return paths


def write_regions(rows: list[dict], path: Path) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return Path(path)


def top_voxel_summary(full: np.ndarray, removed: np.ndarray | None = None, top: float = 5.0) -> dict:
    """A subject's headline numbers over its best-predicted ``top`` % of voxels (by full r):
    mean r, and with a removed fit, mean delta r and delta r as a share of r."""
    keep = np.isfinite(full) & (np.isfinite(removed) if removed is not None else True)
    r = full[keep]
    best = r >= np.percentile(r, 100 - top)
    out = {"voxels": int(keep.sum()), "top_voxels": int(best.sum()), "mean_r": float(np.mean(r)), "top_mean_r": float(np.mean(r[best]))}
    if removed is not None:
        drop = (full - removed)[keep]
        out.update({"mean_delta_r": float(np.mean(drop)), "top_mean_delta_r": float(np.mean(drop[best])),
                    "top_delta_share_of_r": float(np.mean(drop[best]) / np.mean(r[best]))})
    return out


def _subject_maps(args, subjects, warps, with_nulls: bool):
    """Each subject's map (r, or delta r = full - removed) and, for the null, its null drops."""
    from fmri_utils.encoding.columns import read_chunks
    maps, null_maps = {}, {}
    for s in subjects:
        n_columns = warps[s].n_columns if warps else None
        full = read_chunks(Path(args.full) / s, n_columns)[getattr(args, "metric", "correlation_raw")]
        removed = None
        if args.removed is not None:
            removed = read_chunks(Path(args.removed) / s, full.size)["correlation_raw"]
        elif args.xsub_ablation is not None:
            from fmri_utils.cross_participant import read_ablation
            ids = [-1] + (list(range(args.n_nulls)) if with_nulls else [])
            series = read_ablation(Path(args.xsub_ablation) / s, full.size, ids)
            removed = series[-1]
            if with_nulls:
                null_maps[s] = full[None] - np.stack([series[i] for i in range(args.n_nulls)])
        if with_nulls and getattr(args, "nulls", None) is not None:
            from fmri_utils.ablation import read_nulls
            null_maps[s] = full[None] - read_nulls(Path(args.nulls) / s, full.size, args.n_nulls)
        maps[s] = full if removed is None else full - removed
    return maps, null_maps


def main(argv=None) -> None:
    """fmri-group-inference warp-matrix | voxel | summary | nulls"""
    import argparse
    parser = argparse.ArgumentParser(prog="fmri-group-inference", description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("warp-matrix", help="a subject's columns -> MNI warp (needs fslpy)")
    p.add_argument("--run-table", type=Path, required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--anatomical", required=True, help="the FNIRT input image; {subject} is replaced")
    p.add_argument("--warpcoef", required=True, help="anatomy -> template FNIRT coefficients; {subject} is replaced")
    p.add_argument("--func-to-anat", required=True, help="FLIRT-convention matrix; {subject} is replaced")
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="writes <output>/<subject>.npz")

    def inputs(p, nulls=False):
        p.add_argument("--full", type=Path, required=True, help="<fits>/<model> holding <subject>/chunks")
        p.add_argument("--removed", type=Path, default=None, help="<fits>/<model>: maps become full - removed")
        p.add_argument("--xsub-ablation", type=Path, default=None, help="<work>/ablation/<name>/fits (cross-participant)")
        p.add_argument("--subjects", required=True, help="comma separated")
        if nulls:
            p.add_argument("--nulls", type=Path, default=None, help="fmri-ablation nulls-fit output folder")
            p.add_argument("--n-nulls", type=int, default=1000)
    p = sub.add_parser("voxel", help="sign-flip voxel t, BH q, family-wise p, region tests")
    inputs(p)
    p.add_argument("--warps", type=Path, required=True)
    p.add_argument("--metric", default="correlation_raw")
    p.add_argument("--fsldir", type=Path, default=None, help="Harvard-Oxford images and names (FSL data/atlases, or a copy)")
    p.add_argument("--atlas", action="append", default=[], metavar="NAME=IMAGE:NAMES_JSON",
                   help="more template label images for region tests (names: {value: name})")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p = sub.add_parser("summary", help="per subject, over its best-predicted 5%% of voxels: r, delta r, delta r / r")
    inputs(p)
    p.add_argument("--top", type=float, default=5.0)
    p.add_argument("--output", type=Path, default=None, help="a JSON file")
    p = sub.add_parser("nulls", help="group null draws (the variance-matched null)")
    inputs(p, nulls=True)
    p.add_argument("--warps", type=Path, required=True)
    p.add_argument("--n-draws", type=int, default=10000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    args = parser.parse_args(argv)

    if args.command == "warp-matrix":
        from fmri_utils.encoding.run_table import RunTable
        s = args.subject
        warp = ColumnMNIWarp.build(RunTable(args.run_table), s, Path(args.anatomical.format(subject=s)),
                                   Path(args.warpcoef.format(subject=s)), Path(args.func_to_anat.format(subject=s)), args.template)
        warp.save(Path(args.output) / f"{s}.npz")
        return
    subjects = [s for s in args.subjects.split(",") if s]
    warps = {s: ColumnMNIWarp.load(Path(args.warps) / f"{s}.npz") for s in subjects} if getattr(args, "warps", None) else {}
    maps, null_maps = _subject_maps(args, subjects, warps, args.command == "nulls")
    if args.command == "summary":
        from fmri_utils.encoding.columns import read_chunks
        rows = {}
        has_removed = args.removed is not None or args.xsub_ablation is not None
        for s in subjects:
            r = read_chunks(Path(args.full) / s)["correlation_raw"]
            rows[s] = top_voxel_summary(r, r - maps[s] if has_removed else None, args.top)
            print(s, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[s].items()}), flush=True)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    elif args.command == "voxel":
        stack, voxels = to_template(maps, warps)
        result = sign_flip(stack)
        save_results(result, voxels, args.output, args.prefix)
        atlases = harvard_oxford(args.fsldir) if args.fsldir else {}
        for entry in args.atlas:
            import nibabel as nib
            name, rest = entry.split("=", 1)
            image, names = rest.rsplit(":", 1)
            atlases[name] = (np.asarray(nib.load(image).dataobj).astype(int),
                             {int(k): v for k, v in json.loads(Path(names).read_text(encoding="utf-8")).items()})
        if atlases:
            write_regions(region_test(stack, voxels, atlases), Path(args.output) / f"{args.prefix}_regions.json")
        print(f"{voxels.size} voxels; max t {np.nanmax(result['t']):.2f}; q < 0.05: {(result['q'] < 0.05).sum()}; "
              f"FWE < 0.05: {(result['fwe'] < 0.05).sum()}", flush=True)
    else:
        result = null_draws(maps, null_maps, warps, n_draws=args.n_draws)
        save_results(result, result["voxels"], args.output, args.prefix)
        print(f"max z {np.nanmax(result['z']):.2f}; q < 0.05: {(result['q'] < 0.05).sum()}; "
              f"FWE < 0.05: {(result['fwe'] < 0.05).sum()}", flush=True)


if __name__ == "__main__":
    main()

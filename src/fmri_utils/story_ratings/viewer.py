from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

PARAGRAPH_PAUSE_S = 1.5


def _story_payload(story_dir: Path) -> Dict[str, Any]:
    summary = json.loads((story_dir / "run_summary.json").read_text(encoding="utf-8"))
    ratings = pd.read_csv(story_dir / "segment_ratings.csv")
    scale = summary["scale"]["name"]
    flags: List[str] = summary.get("flags", [])
    categoricals: List[str] = list(summary.get("categoricals", {}))
    replicates = []
    for path in sorted((story_dir / "raw").glob("replicate-*.json")):
        stored = json.loads(path.read_text(encoding="utf-8"))
        replicates.append(sorted(stored["ratings"], key=lambda item: int(item["index"])))

    segments = []
    previous_offset: Optional[float] = None
    for position, row in ratings.iterrows():
        onset = None if pd.isna(row.get("onset_s")) else float(row["onset_s"])
        offset = None if pd.isna(row.get("offset_s")) else float(row["offset_s"])
        gap = None if onset is None or previous_offset is None else onset - previous_offset
        previous_offset = offset if offset is not None else previous_offset
        segments.append(
            {
                "i": int(row["index"]),
                "t0": onset,
                "t1": offset,
                "text": str(row["text"]),
                "v": round(float(row[f"{scale}_mean"]), 3),
                "sd": round(float(row[f"{scale}_sd"]), 3),
                "flags": {flag: round(float(row[f"{flag}_mean"]), 2) for flag in flags},
                "cats": {name: str(row[f"{name}_mode"]) for name in categoricals},
                "para": bool(position > 0 and gap is not None and gap >= PARAGRAPH_PAUSE_S),
                "reps": [
                    {
                        "v": int(rep[position][scale]),
                        "reason": str(rep[position].get("reason", "")),
                        "confidence": rep[position].get("confidence"),
                        "cats": {name: rep[position].get(name) for name in categoricals},
                    }
                    for rep in replicates
                ],
            }
        )

    series_path = story_dir / "timeseries.csv"
    series: Dict[str, Any] = {}
    if series_path.exists():
        frame = pd.read_csv(series_path)
        column = f"{scale}_per_word"
        if column in frame and "sample_time_s" in frame:
            series = {
                "t": [round(float(value), 2) for value in frame["sample_time_s"]],
                "v": [None if not math.isfinite(float(value)) else round(float(value), 3) for value in frame[column]],
            }
    return {
        "scale": scale,
        "levels": summary["scale"]["level_labels"] or [str(level) for level in range(summary["scale"]["minimum"], summary["scale"]["maximum"] + 1)],
        "minimum": summary["scale"]["minimum"],
        "maximum": summary["scale"]["maximum"],
        "model": summary["model"],
        "backend": summary["backend"],
        "replicates": summary["replicates"],
        "segmentation": summary["segmentation"]["mode"],
        "nWords": summary["n_words"],
        "nSegments": summary["n_segments"],
        "agreement": summary.get("agreement", {}),
        "distribution": summary.get("distribution", {}),
        "usage": summary.get("usage_total", {}),
        "series": series,
        "segments": segments,
    }


def collect_runs(roots: Sequence[Path], stories: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Gather every rated story under one or more run directories."""
    payload: Dict[str, Dict[str, Any]] = {}
    run_names: List[str] = []
    for root in roots:
        root = Path(root)
        run_name = root.name
        found = False
        for summary_path in sorted(root.glob("*/run_summary.json")):
            story = summary_path.parent.name
            if stories and story not in stories:
                continue
            payload.setdefault(story, {})[run_name] = _story_payload(summary_path.parent)
            found = True
        if found:
            run_names.append(run_name)
    if not payload:
        raise FileNotFoundError(f"no rated stories found under {[str(root) for root in roots]}")
    return {"runs": run_names, "stories": dict(sorted(payload.items()))}


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --ground: #f5f6f8; --surface: #fff; --ink: #1a1f2b; --muted: #5d6677; --rule: #dde1e8;
  --heat: #5b4fcf; --chart: #3f35a8; --focus: #e0a526;
  --serif: "Newsreader", Georgia, serif;
  --sans: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #12141b; --surface: #1a1d26; --ink: #e5e7ee; --muted: #9aa2b3;
    --rule: #2c303c; --heat: #8f83ff; --chart: #b3aaff; --focus: #f0bf4c;
  }
}
:root[data-theme="dark"] {
  --ground: #12141b; --surface: #1a1d26; --ink: #e5e7ee; --muted: #9aa2b3;
  --rule: #2c303c; --heat: #8f83ff; --chart: #b3aaff; --focus: #f0bf4c;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--ground); color: var(--ink); font-family: var(--sans); padding-inline: 16px; padding-block: 0 48px; }
.wrap { max-width: 1120px; margin: 0 auto; }
header.top { padding-block: 24px 12px; display: grid; gap: 12px; }
.eyebrow { font: 500 12px/1 var(--sans); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.pickers { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: end; }
.pickers label { display: grid; gap: 4px; font: 500 11px/1 var(--sans); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
select { font: 500 clamp(18px, 2.6vw, 26px)/1.2 var(--serif); color: var(--ink); background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 4px 8px; max-width: 100%; }
select#run { font: 500 13px/1.3 var(--mono); padding: 8px; }
.stats { display: flex; flex-wrap: wrap; gap: 6px 22px; font: 400 13px/1.4 var(--sans); color: var(--muted); }
.stats b { font: 500 13px/1.4 var(--mono); color: var(--ink); font-variant-numeric: tabular-nums; }
.sticky { position: sticky; top: 0; z-index: 5; background: var(--ground); padding-block: 8px 10px; border-bottom: 1px solid var(--rule); }
.scale { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-bottom: 10px; }
.level { display: grid; grid-template-columns: auto 1fr; gap: 2px 8px; align-items: center; font: 400 12px/1.35 var(--sans); color: var(--muted); }
.swatch { width: 28px; height: 28px; border-radius: 4px; border: 1px solid var(--rule); display: grid; place-items: center; font: 500 12px/1 var(--mono); color: var(--ink); }
.chart { width: 100%; height: 80px; display: block; cursor: crosshair; }
.caption { display: flex; justify-content: space-between; gap: 12px; font: 400 11px/1.3 var(--sans); color: var(--muted); }
main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 32px; padding-block: 24px; align-items: start; }
.transcript { font: 400 19px/1.95 var(--serif); max-width: 68ch; }
.transcript p { margin: 0 0 1.1em; }
.u { border-radius: 3px; padding: .08em .12em; cursor: pointer; box-decoration-break: clone; -webkit-box-decoration-break: clone;
  background: color-mix(in srgb, var(--heat) calc(var(--r) * 62%), transparent); outline: none; }
.u.split { text-decoration: underline dotted color-mix(in srgb, var(--ink) 55%, transparent); text-underline-offset: .28em; }
.u:hover, .u:focus-visible, .u.sel { box-shadow: 0 0 0 2px var(--focus); }
.ts { font: 400 11px/1 var(--mono); color: var(--muted); margin-right: .5em; user-select: none; }
aside { position: sticky; top: 180px; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 16px; display: grid; gap: 12px; font: 400 13px/1.45 var(--sans); }
aside h2 { margin: 0; font: 600 12px/1 var(--sans); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.quote { font: italic 400 17px/1.5 var(--serif); }
.meta { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font: 500 11px/1 var(--sans); padding: 5px 7px; border-radius: 99px; border: 1px solid var(--rule); color: var(--muted); }
.chip.on { background: color-mix(in srgb, var(--heat) 30%, transparent); border-color: transparent; color: var(--ink); }
table.reps { width: 100%; border-collapse: collapse; font-size: 12.5px; }
table.reps td { padding: 6px 4px; border-top: 1px solid var(--rule); vertical-align: top; }
table.reps td.n { font-family: var(--mono); white-space: nowrap; color: var(--muted); }
.note { font-size: 12px; color: var(--muted); }
@media (max-width: 860px) { main { grid-template-columns: minmax(0, 1fr); } aside { position: static; order: -1; } }
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="eyebrow">__EYEBROW__</div>
    <div class="pickers">
      <label for="story">Story<select id="story"></select></label>
      <label for="run">Rating run<select id="run"></select></label>
    </div>
    <div class="stats" id="stats"></div>
  </header>
  <div class="sticky">
    <div class="scale" id="scale"></div>
    <svg class="chart" id="chart" role="img" aria-label="rating over story time"></svg>
    <div class="caption"><span id="chart-note"></span><span id="cursor-time"></span></div>
  </div>
  <main>
    <div class="transcript" id="transcript"></div>
    <aside id="detail" aria-live="polite"></aside>
  </main>
</div>
<script>
const DATA = __DATA__;
const fmt = s => s == null ? "" : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const el = (tag, attrs = {}, text) => { const n = document.createElement(tag); Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v)); if (text != null) n.textContent = text; return n; };
const NS = "http://www.w3.org/2000/svg";
const node = (tag, attrs) => { const n = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v)); return n; };
const storySel = document.getElementById("story"), runSel = document.getElementById("run");
const svg = document.getElementById("chart"), transcript = document.getElementById("transcript"), detail = document.getElementById("detail");
Object.keys(DATA.stories).forEach(s => storySel.append(el("option", { value: s }, s)));
let view = null, spans = [], selected = -1, cursor = null, scaleX = null;

function fillRuns(preferred) {
  const available = Object.keys(DATA.stories[storySel.value]);
  runSel.replaceChildren(...available.map(r => el("option", { value: r }, r)));
  runSel.value = available.includes(preferred) ? preferred : available[0];
}
function writeHash() { try { history.replaceState(null, "", `#story=${encodeURIComponent(storySel.value)}&run=${encodeURIComponent(runSel.value)}`); } catch (e) {} }

function renderScale() {
  const box = document.getElementById("scale");
  box.replaceChildren();
  const span = view.maximum - view.minimum;
  view.levels.forEach((label, k) => {
    const level = view.minimum + k;
    const item = el("div", { class: "level" });
    const sw = el("div", { class: "swatch" }, String(level));
    sw.style.background = `color-mix(in srgb, var(--heat) ${(span ? (level - view.minimum) / span : 0) * 62}%, transparent)`;
    item.append(sw, el("span", {}, label));
    box.append(item);
  });
}

function renderView() {
  view = DATA.stories[storySel.value][runSel.value];
  writeHash();
  const d = view.distribution || {}, a = view.agreement || {};
  document.getElementById("stats").innerHTML = [
    ["segments", view.nSegments], ["words", view.nWords], ["segmentation", view.segmentation],
    ["raters", `${view.replicates} × ${view.model}`],
    a.all_raters_exact != null ? ["all raters agree", `${Math.round(a.all_raters_exact * 100)}%`] : null,
    a.weighted_kappa != null ? ["weighted kappa", a.weighted_kappa.toFixed(2)] : null,
    ["levels", Object.keys(d).map(k => `${Math.round(d[k] * 100)}%`).join(" / ")],
  ].filter(Boolean).map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join("");
  renderScale();
  transcript.replaceChildren();
  spans = [];
  let para = el("p");
  const span = view.maximum - view.minimum || 1;
  view.segments.forEach((u, idx) => {
    if (u.para && para.childNodes.length) { transcript.append(para); para = el("p"); }
    if (!para.childNodes.length && u.t0 != null) para.append(el("span", { class: "ts" }, fmt(u.t0)));
    const s = el("span", { class: "u" + (u.sd > 0 ? " split" : ""), tabindex: "0" }, u.text);
    s.style.setProperty("--r", ((u.v - view.minimum) / span).toFixed(3));
    s.addEventListener("mouseenter", () => select(idx, false));
    s.addEventListener("focus", () => select(idx, false));
    s.addEventListener("click", () => select(idx, false));
    para.append(s, document.createTextNode(" "));
    spans.push(s);
  });
  transcript.append(para);
  selected = -1;
  drawChart();
  select(view.segments.reduce((best, u, k, all) => u.v > all[best].v ? k : best, 0), false);
}

function drawChart() {
  const timed = view.segments.some(u => u.t0 != null);
  document.getElementById("chart-note").textContent = timed
    ? "Mean rating over story time. Click to jump." : "This story has no word timings.";
  svg.replaceChildren();
  if (!timed) { scaleX = null; return; }
  const W = svg.clientWidth || 800, H = 80, L = 26, R = 6, T = 6, B = 16;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const end = Math.max(...view.segments.map(u => u.t1 || 0));
  const x = t => L + (t / end) * (W - L - R);
  const y = v => T + (1 - (v - view.minimum) / ((view.maximum - view.minimum) || 1)) * (H - T - B);
  const css = getComputedStyle(document.documentElement);
  const muted = css.getPropertyValue("--muted").trim(), rule = css.getPropertyValue("--rule").trim();
  for (let level = view.minimum; level <= view.maximum; level++) {
    svg.append(node("line", { x1: L, x2: W - R, y1: y(level), y2: y(level), stroke: rule, "stroke-width": 1 }));
    const label = node("text", { x: L - 8, y: y(level) + 3.5, "text-anchor": "end", fill: muted, "font-size": 10, "font-family": "IBM Plex Mono, monospace" });
    label.textContent = level; svg.append(label);
  }
  const step = end > 900 ? 240 : 120;
  for (let t = 0; t <= end; t += step) {
    const label = node("text", { x: x(t), y: H - 3, "text-anchor": t === 0 ? "start" : "middle", fill: muted, "font-size": 10, "font-family": "IBM Plex Mono, monospace" });
    label.textContent = fmt(t); svg.append(label);
  }
  view.segments.forEach(u => {
    if (u.t0 == null) return;
    svg.append(node("rect", { x: x(u.t0), width: Math.max(1, x(u.t1) - x(u.t0)), y: y(u.v), height: y(view.minimum) - y(u.v), fill: "var(--heat)", "fill-opacity": 0.22 }));
  });
  if (view.series && view.series.t) {
    let d = "", pen = false;
    view.series.t.forEach((t, k) => {
      const v = view.series.v[k];
      if (v == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(t).toFixed(1)},${y(Math.max(view.minimum, Math.min(view.maximum, v))).toFixed(1)}`; pen = true;
    });
    if (d) svg.append(node("path", { d, fill: "none", stroke: "var(--chart)", "stroke-width": 1.6, "stroke-linejoin": "round" }));
  }
  cursor = node("line", { y1: T, y2: y(view.minimum), stroke: "var(--focus)", "stroke-width": 2, visibility: "hidden" });
  svg.append(cursor);
  scaleX = { x, end, W, L, R };
}

svg.addEventListener("click", ev => {
  if (!scaleX) return;
  const box = svg.getBoundingClientRect();
  const t = ((ev.clientX - box.left) / box.width * scaleX.W - scaleX.L) / (scaleX.W - scaleX.L - scaleX.R) * scaleX.end;
  let best = 0;
  view.segments.forEach((u, k) => { if (u.t0 != null && u.t0 <= t) best = k; });
  select(best, true);
});

function select(idx, scroll) {
  if (selected >= 0 && spans[selected]) spans[selected].classList.remove("sel");
  selected = idx;
  const u = view.segments[idx];
  spans[idx].classList.add("sel");
  if (scroll) spans[idx].scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  if (cursor && u.t0 != null) {
    const cx = scaleX.x(u.t0);
    cursor.setAttribute("x1", cx); cursor.setAttribute("x2", cx); cursor.setAttribute("visibility", "visible");
  }
  document.getElementById("cursor-time").textContent = u.t0 == null ? "" : `${fmt(u.t0)}–${fmt(u.t1)}`;
  detail.replaceChildren(el("h2", {}, `Segment ${u.i}${u.t0 == null ? "" : " · " + fmt(u.t0)}`), el("div", { class: "quote" }, `“${u.text}”`));
  const meta = el("div", { class: "meta" });
  meta.append(el("span", { class: "chip on" }, `${view.scale} ${u.v.toFixed(2)}`));
  Object.entries(u.cats).forEach(([k, v]) => meta.append(el("span", { class: "chip" }, `${k}: ${v}`)));
  Object.entries(u.flags).forEach(([k, v]) => meta.append(el("span", { class: "chip" + (v >= 0.5 ? " on" : "") }, k.replace(/_/g, " "))));
  detail.append(meta);
  const table = el("table", { class: "reps" });
  u.reps.forEach((r, k) => {
    const tr = el("tr");
    const bits = [r.reason, r.confidence != null ? `conf ${r.confidence}` : null].filter(Boolean).join(" · ");
    tr.append(el("td", { class: "n" }, `R${k + 1}: ${r.v}`), el("td", {}, bits));
    table.append(tr);
  });
  detail.append(table);
  if (u.sd > 0) detail.append(el("div", { class: "note" }, "Raters disagreed here (dotted underline)."));
}

storySel.addEventListener("change", () => { fillRuns(runSel.value); renderView(); window.scrollTo(0, 0); });
runSel.addEventListener("change", renderView);
window.addEventListener("resize", () => { if (view) { drawChart(); if (selected >= 0) select(selected, false); } });
const params = new URLSearchParams(location.hash.slice(1));
storySel.value = DATA.stories[params.get("story")] ? params.get("story") : Object.keys(DATA.stories)[0];
fillRuns(params.get("run"));
renderView();
</script>
</body>
</html>
"""


def render_viewer(
    roots: Sequence[Path],
    output_path: Path,
    *,
    stories: Optional[Sequence[str]] = None,
    title: str = "Story Ratings",
    eyebrow: str = "LLM segment ratings",
) -> Path:
    """Write a self-contained HTML reader for one or more rating runs.

    The page shows each story's text with every segment tinted by its
    consensus rating, a rating-over-time chart when word timings exist, and
    each replicate's rating and reason for the selected segment. Story and run
    selectors make two runs directly comparable on the same text.
    """
    payload = collect_runs(roots, stories)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", data).replace("__TITLE__", title).replace("__EYEBROW__", eyebrow)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path

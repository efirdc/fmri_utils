# Slide Decks

`fmri_utils.slides` makes web slide decks that embed live
[result viewers](viewer.md) and other pages. A deck is a folder: plain HTML
slides plus a small script. You publish it by copying the folder next to the
viewers it embeds.

On a local copy, the deck has an in-browser editor:
- text, boxes, pictures and LaTeX equations;
- animations, groups, and a slide sorter;
- undo, and saving straight back into `index.html`.

```bash
fmri-slides new talks/my-talk --title "My talk" --mount viewer=../../results/viewer
fmri-slides serve talks/my-talk          # http://127.0.0.1:8740/my-talk/ ; E edits, Ctrl+S saves
fmri-slides bundle talks/my-talk --output D:/my-talk-offline   # runs anywhere with Python 3
```

## Contents

- [A Deck Folder](#a-deck-folder)
- [Presenting](#presenting)
- [Editing](#editing)
- [Writing Slides By Hand](#writing-slides-by-hand)
- [Brain Views](#brain-views)
- [Publishing And Offline Copies](#publishing-and-offline-copies)

## A Deck Folder

```text
my-talk/
  index.html     the slides, between <!--SLIDES--> and <!--/SLIDES-->
  deck.json      the deck's URL name, where it is published, the folders it embeds
  deck.js        navigation, steps and animations, embeds, brain views, KaTeX, charts
  charts.js      the deck's own interactive charts (window.DeckCharts); not refreshed by `assets`
  editor.js      the slide sorter and the editor
  deck.css
  vendor/        KaTeX and NiiVue, so the deck works offline
  data/  figures/
```

`deck.json`:

```json
{
 "name": "my-talk",
 "title": "My talk",
 "public_base": "https://example.org/~me/",
 "mounts": {
  "viewer": "../../results/viewer",
  "reader/audio": "../../results/reader_audio",
  "reader": "../../results/reader_site"
 }
}
```

- **Mounts** are the sibling folders the slides embed as `../<mount>/`. The local
  server serves them side by side with the deck, the way they sit when published,
  so embedded viewers are same-origin and the deck can drive them. Paths are
  relative to the deck folder. A longer mount (`reader/audio`) wins over a
  shorter one (`reader`).
- **`public_base`** is where the deck and its mounts are published. Each embed's
  "open in viewer" link goes there. The same value is set in `index.html` as
  `<meta name="deck-public-base">`.

`fmri-slides assets my-talk` refreshes a deck's copy of `deck.js`, `editor.js`,
`deck.css` and `vendor/` from the installed package.

## Presenting

| key | does |
|---|---|
| → ↓ Space PageDown N | next step |
| ← ↑ PageUp Backspace P | previous step |
| Shift + any of those | next / previous slide, skipping steps |
| a digit, or G | type a slide number into the counter; Enter jumps (`7` or `7.2`) |
| ] / [ | next / previous cluster on a cluster-list slide |
| Home / End | first / last slide |
| F | full screen |
| E | edit (local copy only) |

Inside an embedded viewer, only PageDown and PageUp move the deck, so a clicker
still works and the viewer keeps every other key. The bottom-right counter shows
`slide.step` and is also an input. Links like `#/7/3` open slide 7 at step 3.

Except in full screen, a slide list sits on the left; « hides it.

## Editing

The editor runs only on a local copy served by `fmri-slides serve`. Press E or
click ✎. Everything is undoable (Ctrl+Z / Ctrl+Y), and Ctrl+S writes the slides
into `index.html`, keeping a `.bak` of the previous version.

| action | how |
|---|---|
| type | click text. Esc leaves the text with its element selected, a second Esc goes up a group, then deselects, then leaves the editor |
| select | click; Shift-click adds or removes; Ctrl+A selects the whole slide. A group is selected whole first; click again for an element inside it |
| move | drag the selected element's frame edge, or use the arrow keys (Shift: further). Pictures, viewers, charts, equations and groups can be dragged anywhere. Snaps to edges and centres; Alt turns snapping off |
| resize | drag a handle. A picture's corners keep its shape; hold Shift to free them |
| group | Ctrl+G groups, Ctrl+Shift+G ungroups |
| insert | **Text** box; **Image** (or paste or drop one; saved to `figures/uploads/`); **Equation** (inline at the caret while typing, otherwise a displayed equation) |
| equations | click one to edit its LaTeX with a live preview |
| format | style (title, heading, subheading, body, note), size, bold / italic / underline / strike, colour, alignment, bullets / numbers |
| animate | the toolbar names its target: the selection, or the list item or paragraph the caret is in. Set the step it appears at, its effect (fade, appear, rise, from left, zoom), dim after, and the step it disappears at. **one by one** steps through a list's items. Badges show each element's step |
| slides | in the slide list, drag to reorder; Ctrl+C / X / V / D and Delete; right-click for a menu; **+ New slide** offers layouts |
| right-click | on an element: what applies to it (cut/copy/paste, LaTeX, animation, group, align, front/back); on the slide: paste, new text box or equation, insert picture |

**Layout.** Slides are written in flow: a heading, then a list, a `.cols` grid.
The first time anything on a slide is moved or resized, that level is frozen
where it stands. Each element becomes a `.box` with `left/top/width/height` in %
of its container, and a container in the flow becomes a group. So moving one
column never shifts the other.

## Writing Slides By Hand

A slide is a `<section class="slide">` between the markers. The editor writes
the same markup.

```html
<section class="slide">
  <h2>Ablation</h2>
  <ul>
    <li data-step="1">first point</li>
    <li data-step="2" data-anim="rise" data-dim-after>second, rising in, dimmed after</li>
  </ul>
  <span class="math-display" data-step="3" data-tex="\Delta r = r_{\text{full}} - r_{\text{removed}}"></span>
  <p class="note">Inline math: <span class="math" data-tex="z"></span>.</p>
</section>
```

| attribute or class | meaning |
|---|---|
| `data-step="n"` | appears at step n |
| `data-anim` | how it appears: `appear`, `rise`, `left` or `zoom` (default fade) |
| `data-dim-after` | dims once the next step appears |
| `data-hide="m"` | gone from step m on |
| `.title-slide`, `.cols`, `.cols.three`, `.note`, `.figure` | layouts and styles in `deck.css` |
| `.embed data-src="../reader/#..."` | an embedded page, loaded when its slide is near (`data-label` names it in the slide list) |
| `.chart data-chart="name"` | an interactive chart: the deck's own `charts.js` registers `window.DeckCharts.name = function (element) {...}` |

## Brain Views

```html
<div class="embed brain"
     data-src="../viewer/#r=my_report&amp;e=my_endpoint&amp;f=bert&amp;s=group&amp;v=multi"
     data-pick="feature: english1000=English1000, bert=BERT"
     data-tour="data/clusters.json"></div>
```

An `.embed.brain` shows a viewer link (any view the viewer can encode in its URL)
as just the brain:
- The viewer's own controls are hidden, and the status bar and colour bar are
  sized from the slide.
- Above the brain is a deck bar: volume / surface, and the white / pial /
  inflated / flat geometry in surface mode. Volume mode stays on the
  three-panel view, zoomed in by `data-zoom` (default 1.2).
- `data-pick` adds button groups:
  - `feature:` presses the viewer's feature buttons;
  - any other name is a variant control of the endpoint, with option ids from
    the manifest;
  - groups are separated by `;`.
- The deck drives the viewer through its documented scripting API and stable
  selectors ([viewer.md](viewer.md#driving-it-from-another-page)).

**Cluster lists.** With `data-tour`, a panel beside the brain lists a map's
clusters. Picking one glides the crosshair to its centre in volume mode. In
surface mode it marks the nearest fsaverage vertex and turns the camera there; a
cluster on the medial wall shows its hemisphere alone. The file is JSON:

```json
{"n_clusters": 26, "mode": "fwe", "level": 0.05, "k": 20, "start_mm": [28, 22, 24],
 "start_link": "https://.../viewer/#...",
 "clusters": [{"size": 788, "volume_mm3": 6303, "centre_mm": [-32, -74, 38], "peak_value": 0.051,
               "region": "Lateral Occipital Cortex, superior division", "region_share": 0.94,
               "link": "https://.../viewer/#...&x=-32_-74_38",
               "fsaverage": {"hemisphere": "lh", "vertex": 61641, "outward": [-0.05, -0.95, 0.31]}}]}
```

`fsaverage.vertex` is the vertex whose registration-fusion point (the
`volume_points` of the viewer's surface export) is nearest the centre.
`outward` is that vertex's direction from its inflated hemisphere's centre; a
cluster whose medial component is at least 0.25 shows its hemisphere alone.

**Loading.** A viewer takes a few seconds to start: most of it is NiiVue
compiling shaders, faster once the browser has compiled them once. The deck
keeps two viewers loaded: this slide's and the next one ahead, however far
away. Going forward, every viewer is ready when its slide comes up. Each viewer
holds 5 WebGL contexts and the browser keeps 16 per page, dropping the oldest
beyond that, so two is the limit. Until a viewer is ready, its slide shows
"loading…".

## Publishing And Offline Copies

**Publishing:** copy the deck folder next to its mounts on the web host. Leave
out `*.bak`; `deck.json` is not needed there.

**Offline:** `fmri-slides bundle DECK --output DIR` writes:
- the deck;
- each mount (for a viewer build, only the files its manifest and surface
  catalogue reference);
- `serve.py` (a copy of the server, standard library only) and `start.bat`.

Run `python serve.py` in the bundle on any machine with Python 3. The bundle's
deck can be edited and saved too. Rebuilding copies only files that changed
size, and does not remove files that were deleted from the deck.

The server, `fmri_utils.slides.server`, also:
- rewrites a viewer's CDN NiiVue to the deck's `vendor/` copy;
- drops Google Fonts;
- answers range requests, so audio and video seek.

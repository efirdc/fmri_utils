// Lab-meeting deck. Slides are <section class="slide"> in index.html; this file adds:
// - steps: [data-step="n"] elements appear at sub-step n; a slide's last step is its largest n;
// - the slide.substep counter (bottom right; type a number there to jump) and #/slide/step links;
// - embedded pages (.embed[data-src]) loaded when their slide is near, each with a link to the full page;
// - brain views (.embed.brain): the LeBel viewer reduced to its brain, with deck buttons for
//   volume/surface, the slice view and the surface geometry, plus any data-pick choices
//   (features, or a variant control's options);
// - cluster lists (.embed.brain[data-tour]): a side panel to move through a map's clusters, which
//   glides the crosshair (volume) or the marker and camera (surface) to each one;
// - KaTeX ([data-tex]) and interactive charts ([data-chart]);
// - animations: data-step (appears at), data-anim (fade, appear, rise, left, zoom), data-dim-after,
//   data-hide (gone from that step on);
// - window.Deck, the API editor.js builds the slide sorter and the editor on.
(function () {
  "use strict";
  // Where the deck and its sibling pages are published (<meta name="deck-public-base">): the
  // "open in viewer" links go there; without it they open the local copy.
  const PUBLIC = ((document.querySelector('meta[name="deck-public-base"]') || {}).content || "").replace(/\/?$/, "/");
  const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
  const deck = document.getElementById("deck");
  let slides = Array.from(deck.querySelectorAll(":scope > .slide"));
  const counter = document.getElementById("counter");
  const status = document.getElementById("status");
  let index = 0, step = 0;
  const editing = () => !!(window.Editor && window.Editor.active);

  // ---- steps and navigation ---------------------------------------------------------
  function stepsOf(slide) {
    let n = 0;
    slide.querySelectorAll("[data-step], [data-hide]").forEach(el => {
      n = Math.max(n, Number(el.dataset.step) || 0, Number(el.dataset.hide) || 0);
    });
    return n;
  }

  let shownSlide = null;
  function show(i, s, fromHash) {
    index = Math.max(0, Math.min(slides.length - 1, i));
    const slide = slides[index];
    const arriving = slide !== shownSlide;
    shownSlide = slide;
    if (arriving) deck.classList.add("no-anim");
    step = Math.max(0, Math.min(stepsOf(slide), s));
    slides.forEach((el, k) => el.classList.toggle("current", k === index));
    slide.querySelectorAll("[data-step]").forEach(el => {
      const n = Number(el.dataset.step) || 0;
      el.classList.toggle("future", n > step);
      el.classList.toggle("past", el.hasAttribute("data-dim-after") && n < step);
    });
    slide.querySelectorAll("[data-hide]").forEach(el => el.classList.toggle("gone", step >= Number(el.dataset.hide)));
    if (arriving) {
      void deck.offsetWidth;
      requestAnimationFrame(() => requestAnimationFrame(() => deck.classList.remove("no-anim")));
    }
    if (document.activeElement !== counter) counter.value = label();
    if (!fromHash) { try { history.replaceState(null, "", `#/${index + 1}/${step + 1}`); } catch (e) { /* file:// */ } }
    listeners.forEach(fn => fn(index, step));
    // Brain viewers: this slide's and the next one ahead however far (so it has loaded by the time
    // it is reached), or with none here, the next ahead and the last behind. Two at most: each holds
    // 5 WebGL contexts and the browser keeps 16 per page, dropping the oldest beyond that (a dropped
    // panel draws white). A viewer's first start also compiles the browser's shaders for the rest,
    // so opening the deck starts the first viewer straight away. Other embeds: this slide and the next.
    const brains = brainSlides(index);
    slides.forEach((el, k) => {
      if (brains.includes(k)) loadEmbeds(el, ".embed.brain"); else unloadEmbeds(el);
      if (k === index || k === index + 1) loadEmbeds(el, ".embed:not(.brain)");
    });
  }
  function brainSlides(at) {
    const has = k => !!slides[k].querySelector(".embed.brain");
    const keep = [];
    if (has(at)) keep.push(at);
    for (let k = at + 1; k < slides.length; k += 1) if (has(k)) { keep.push(k); break; }
    if (keep.length < 2) for (let k = at - 1; k >= 0; k -= 1) if (has(k)) { keep.push(k); break; }
    return keep;
  }
  function unloadEmbeds(slide) {
    slide.querySelectorAll(".embed.brain").forEach(embed => {
      if (!embed.querySelector("iframe")) return;
      clearInterval(embed._zoomWatch);
      clearTimeout(embed._retry); clearTimeout(embed._morph); clearTimeout(embed._ready);
      embed.classList.remove("booting");
      embed.replaceChildren();
      ["_zoomWatch", "_tour", "_tourLoading", "_cluster", "_glide", "_pending"].forEach(key => { delete embed[key]; });
    });
  }
  function label() { return stepsOf(slides[index]) ? `${index + 1}.${step + 1}` : `${index + 1}`; }

  function next() {
    if (step < stepsOf(slides[index])) show(index, step + 1);
    else if (index < slides.length - 1) show(index + 1, 0);
  }
  function previous() {
    if (step > 0) show(index, step - 1);
    else if (index > 0) show(index - 1, stepsOf(slides[index - 1]));
  }
  // Whole slides, skipping sub-steps: forward to the next slide's start, back to the previous slide
  // as it ends (fully built), or to this slide's start when it is part-way through.
  function nextSlide() { if (index < slides.length - 1) show(index + 1, 0); else show(index, stepsOf(slides[index])); }
  function previousSlide() {
    if (step > 0) show(index, 0);
    else if (index > 0) show(index - 1, stepsOf(slides[index - 1]));
  }

  function onKey(event, insideFrame) {
    if (editing()) return;   // the editor takes the keys while it is on
    if (!insideFrame && event.target === counter) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    // Inside an embedded viewer only a clicker's page keys move the deck; the viewer keeps the rest.
    const advance = insideFrame ? ["PageDown"] : ["ArrowRight", "ArrowDown", "PageDown", " ", "n", "N"];
    const back = insideFrame ? ["PageUp"] : ["ArrowLeft", "ArrowUp", "PageUp", "Backspace", "p", "P"];
    if (advance.includes(event.key)) { event.preventDefault(); if (event.shiftKey) nextSlide(); else next(); }
    else if (back.includes(event.key)) { event.preventDefault(); if (event.shiftKey) previousSlide(); else previous(); }
    else if (insideFrame) return;
    else if (event.key === "Home") show(0, 0);
    else if (event.key === "End") show(slides.length - 1, 0);
    else if (event.key === "]" || event.key === "[") stepCluster(event.key === "]" ? 1 : -1);
    else if (event.key === "f") { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen(); }
    else if (event.key === "e" && LOCAL && window.Editor) window.Editor.toggle();
    else if (event.key === "g" || /^[0-9]$/.test(event.key)) {
      // A digit (or g) starts typing a slide number into the counter; Enter jumps there.
      event.preventDefault();
      counter.focus();
      counter.value = event.key === "g" ? "" : event.key;
    }
  }
  document.addEventListener("keydown", event => {
    if (editing()) { window.Editor.onKey(event); return; }
    onKey(event, false);
  });
  window.addEventListener("hashchange", readHash);
  function readHash() {
    const m = /^#\/(\d+)(?:\/(\d+))?/.exec(location.hash);
    if (m) show(Number(m[1]) - 1, (Number(m[2]) || 1) - 1, true);
  }

  // The counter is an input: "7" goes to slide 7, "7.2" to its second step.
  counter.addEventListener("focus", () => counter.select());
  counter.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      const m = /^\s*(\d+)(?:[.\/ ](\d+))?\s*$/.exec(counter.value);
      if (m) show(Number(m[1]) - 1, (Number(m[2]) || 1) - 1);
      counter.blur();
    } else if (event.key === "Escape") counter.blur();
    event.stopPropagation();
  });
  counter.addEventListener("blur", () => { counter.value = label(); });

  // ---- embedded pages ---------------------------------------------------------------------
  // data-src is relative to the deck ("../lebel/#..."); the link out goes to the same page where
  // the deck is published.
  function publicUrl(src) {
    const m = /^\.\.\/([^/#]+)\/?(.*)$/.exec(src);
    return m && PUBLIC !== "/" ? `${PUBLIC}${m[1]}/${m[2]}` : src;
  }
  function make(tag, className, parent, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    if (parent) parent.append(node);
    return node;
  }
  function loadEmbeds(slide, which = ".embed") {
    slide.querySelectorAll(`${which}[data-src]`).forEach(embed => {
      if (embed.querySelector("iframe")) return;
      const brain = embed.classList.contains("brain");
      // A brain view is a bar of view buttons over the frame, with the cluster panel beside it.
      const holder = brain ? make("div", "brain-frame", null) : embed;
      if (brain) embed.classList.add("booting");
      const frame = document.createElement("iframe");
      frame.src = embed.dataset.src;
      frame.allow = "autoplay; fullscreen";
      frame.addEventListener("load", () => {
        const loading = holder.querySelector(".loading");
        if (loading && !brain) loading.remove();
        try { frame.contentWindow.document.addEventListener("keydown", event => onKey(event, true)); } catch (e) { /* other origin */ }
        if (brain) { strip(embed); watchReady(embed); }
      });
      const link = make("a", "open-link", null, "open in viewer ↗");
      link.target = "_blank";
      link.rel = "noopener";
      link.href = publicUrl(embed.dataset.src);
      if (brain) {
        buildBar(embed, link);
        holder.append(frame);
        make("div", "loading", holder, "loading…");
        embed.append(holder);
        if (embed.dataset.tour) loadTour(embed);
      } else {
        embed.append(frame);
        make("div", "loading", embed, "loading…");
        embed.append(link);
      }
    });
  }

  // ---- brain views ------------------------------------------------------------------------
  // The viewer is same-origin, so the deck hides its controls and presses its (hidden) buttons.
  // The status bar's text and colour bar are sized from the slide unit, so they are never smaller
  // than the slides' own small print (.note, 1.15 units).
  function slideUnit() { return unit; }
  function chrome() {
    const u = slideUnit(), font = Math.round(1.25 * u), bar = Math.round(4.2 * u);
    return `
    .shell { grid-template-columns: 1fr !important; height: 100% !important; }
    .shell > .rail, .main > .bar { display: none !important; }
    /* The desktop layout at any frame size (below 960 px the viewer stacks for phones). */
    html, body { overflow: hidden !important; }
    .main { height: 100% !important; overflow: hidden !important; display: flex !important; flex-direction: column !important; }
    .stage { flex: 1 1 auto !important; height: auto !important; min-height: 0 !important; }
    .montage { grid-template-columns: repeat(3, 1fr) !important; }
    .status { flex-wrap: nowrap !important; white-space: nowrap !important; overflow: hidden !important; }
    .status > * { flex: none; }
    #roi-readout { flex: 1 1 0 !important; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
    .status { font-size: ${font}px !important; gap: ${Math.round(0.5 * u)}px ${Math.round(1.4 * u)}px !important;
              padding: ${Math.round(0.4 * u)}px ${Math.round(1 * u)}px !important; }
    #colorbar { height: ${bar}px !important; width: ${Math.round(30 * u)}px !important;
                flex: 0 1 ${Math.round(30 * u)}px !important; }
  `;
  }
  const BAR = {
    volume: [],   // volume mode stays on the multi view (axial, coronal and sagittal side by side)
    surface: [["wm", "white"], ["pia", "pial"], ["inflated", "inflated"], ["flat", "flat"]],
  };
  function frameOf(embed) { return embed.querySelector("iframe"); }
  function viewerOf(embed) {
    const frame = frameOf(embed);
    try { return frame && frame.contentWindow && frame.contentWindow.__viewer; } catch (e) { return null; }
  }
  function viewerDocument(embed) {
    const frame = frameOf(embed);
    try { return frame && frame.contentDocument; } catch (e) { return null; }
  }
  function strip(embed) {
    const doc = viewerDocument(embed);
    if (!doc) return;
    let style = doc.getElementById("deck-chrome");
    if (!style) { style = doc.createElement("style"); style.id = "deck-chrome"; doc.head.append(style); }
    style.textContent = chrome();
    try { frameOf(embed).contentWindow.dispatchEvent(new Event("resize")); } catch (e) { /* ignore */ }
    redrawColorbar(embed);
    syncBar(embed);
    if (!embed._zoomWatch) embed._zoomWatch = setInterval(() => zoomPanels(embed), 500);
  }
  // Ready: the endpoint is up and its panels (or surfaces) hold their images, zoomed.
  function watchReady(embed, started = performance.now()) {
    if (!embed.isConnected || !embed.querySelector("iframe")) return;
    const viewer = viewerOf(embed);
    const state = viewer && viewer.state;
    let ready = false;
    if (state && state.endpoint) {
      if (state.mode === "surface") ready = !!(viewer.parts && viewer.parts.length);
      else {
        zoomPanels(embed);
        const panels = viewer.montage || [];
        ready = panels.length > 0 && panels.every(p => p.nv && p.nv.volumes && p.nv.volumes.length && p.nv.__deckZoom);
      }
    }
    if (ready || performance.now() - started > 40000) {
      syncBar(embed);
      setTimeout(() => {
        embed.classList.remove("booting");
        const loading = embed.querySelector(".brain-frame .loading");
        if (loading) loading.remove();
      }, 150);
      return;
    }
    embed._ready = setTimeout(() => watchReady(embed, started), 150);
  }
  // Each slice panel is zoomed in a little (data-zoom, default 1.2) about the brain's centre, so
  // less of it is empty. The viewer rebuilds its panels on some changes (back from surface mode),
  // so new panels are found and zoomed as they appear. NiiVue draws a point p at zoom·p + pan.
  const BRAIN_CENTRE = [0, -18, 15];
  function zoomPanels(embed) {
    const viewer = viewerOf(embed);
    if (!viewer || !viewer.state || viewer.state.mode !== "volume") return;
    const zoom = Number(embed.dataset.zoom || 1.2);
    (viewer.montage || []).forEach(panel => {
      const nv = panel && panel.nv;
      if (!nv || !nv.scene || nv.__deckZoom || !nv.volumes || !nv.volumes.length) return;
      nv.scene.pan2Dxyzmm = [(1 - zoom) * BRAIN_CENTRE[0], (1 - zoom) * BRAIN_CENTRE[1], (1 - zoom) * BRAIN_CENTRE[2], zoom];
      nv.__deckZoom = true;
      try { nv.drawScene(); } catch (e) { /* not ready */ }
    });
  }
  // The colour bar is drawn for its CSS size; redraw it once the viewer is up.
  // A redraw asked for while the map is still loading draws nothing, so check and ask again.
  function redrawColorbar(embed, tries = 150) {
    const viewer = viewerOf(embed), doc = viewerDocument(embed);
    const canvas = doc && doc.getElementById("colorbar");
    if (canvas && canvas.clientHeight &&
        canvas.height === Math.round(canvas.clientHeight * (frameOf(embed).contentWindow.devicePixelRatio || 1))) return;
    if (viewer && viewer.refreshStyle && viewer.state && viewer.state.endpoint) viewer.refreshStyle();
    if (tries > 0) setTimeout(() => redrawColorbar(embed, tries - 1), 400);
  }
  function restyleViewers() {
    document.querySelectorAll(".embed.brain").forEach(embed => { if (frameOf(embed)) strip(embed); });
  }
  // data-pick="feature: english1000=English1000, bert=BERT; show: r=held-out r, d_x=minus X":
  // one button group per control. "feature" is the viewer's feature choice; any other name is a
  // variant control of the endpoint (its id in the manifest), and the values are its option ids.
  function picksOf(embed) {
    return (embed.dataset.pick || "").split(";").map(s => s.trim()).filter(Boolean).map(group => {
      const [control, list] = group.split(/:(.*)/s).map(s => s.trim());
      return { control, options: list.split(",").map(s => s.trim()).filter(Boolean).map(o => o.split("=").map(s => s.trim())) };
    });
  }
  function pickSelector(control, value) {
    return control === "feature" ? `#features [data-feature="${value}"]`
      : `#variants [data-control="${control}"] [data-option="${value}"]`;
  }
  function buildBar(embed, link) {
    const bar = make("div", "brain-bar", embed);
    const modes = make("div", "seg", bar);
    ["volume", "surface"].forEach(mode => {
      const button = make("button", null, modes, mode);
      button.type = "button";
      button.dataset.mode = mode;
      button.addEventListener("click", () => setMode(embed, mode));
    });
    Object.entries(BAR).forEach(([mode, options]) => {
      if (!options.length) return;
      const group = make("div", "seg", bar);
      group.dataset.for = mode;
      options.forEach(([value, text]) => {
        const button = make("button", null, group, text);
        button.type = "button";
        button.dataset.value = value;
        button.addEventListener("click", () => pick(embed, mode, value));
      });
    });
    picksOf(embed).forEach(({ control, options }) => {
      const group = make("div", "seg", bar);
      group.dataset.pick = control;
      options.forEach(([value, text]) => {
        const button = make("button", null, group, text || value);
        button.type = "button";
        button.dataset.value = value;
        button.addEventListener("click", () => { press(embed, pickSelector(control, value)); setTimeout(() => { syncBar(embed); redrawColorbar(embed); }, 50); });
      });
    });
    make("span", "spacer", bar);
    bar.append(link);
  }
  function press(embed, selector) {
    const doc = viewerDocument(embed);
    const button = doc && doc.querySelector(selector);
    if (!button || button.disabled) return false;
    button.click();
    return true;
  }
  function setMode(embed, mode) {
    if (!press(embed, `#modes [data-mode-kind="${mode}"]`)) return;
    syncBar(embed);
    // Surfaces load after the switch; bring the current cluster back once they are there.
    if (embed._tour) settle(embed);
  }
  function pick(embed, mode, value) {
    press(embed, mode === "volume" ? `#views [data-view="${value}"]` : `#geometries [data-geometry="${value}"]`);
    syncBar(embed);
    if (embed._tour && mode === "volume") settle(embed);
    // A new geometry morphs in over about a second; face the cluster again on the finished shape.
    if (embed._tour && mode === "surface") { clearTimeout(embed._morph); embed._morph = setTimeout(() => settle(embed), 1300); }
  }
  // The deck's buttons mirror the viewer's: its mode, its view and its geometry.
  function syncBar(embed) {
    const viewer = viewerOf(embed);
    const bar = embed.querySelector(".brain-bar");
    if (!viewer || !viewer.state || !bar) { setTimeout(() => syncBar(embed), 300); return; }
    const state = viewer.state;
    const doc = viewerDocument(embed);
    bar.querySelectorAll("[data-mode]").forEach(b => {
      const source = doc.querySelector(`#modes [data-mode-kind="${b.dataset.mode}"]`);
      b.disabled = !source || source.disabled;
      b.setAttribute("aria-pressed", String(state.mode === b.dataset.mode));
    });
    bar.querySelectorAll("[data-for]").forEach(group => {
      group.hidden = group.dataset.for !== state.mode;
      const current = group.dataset.for === "volume" ? state.view : state.geometry;
      group.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.value === current)));
    });
    bar.querySelectorAll("[data-pick]").forEach(group => {
      group.querySelectorAll("button").forEach(b => {
        const source = doc.querySelector(pickSelector(group.dataset.pick, b.dataset.value));
        b.disabled = !source || source.disabled;
        b.setAttribute("aria-pressed", String(!!source && source.getAttribute("aria-pressed") === "true"));
      });
    });
    // The viewer builds its feature and variant buttons once an endpoint is up; until then, look again.
    if (bar.querySelector("[data-pick] button") && !doc.querySelector("#features button, #variants button")) setTimeout(() => syncBar(embed), 300);
  }

  // ---- cluster lists ------------------------------------------------------------------------
  // A side panel lists every cluster of the map; choosing one glides the crosshair to its centre in
  // volume mode, or marks its nearest fsaverage vertex and turns the camera to it in surface mode.
  // Slide steps do not move it; the [ and ] keys and the panel's buttons do.
  function loadTour(embed) {
    if (embed._tourLoading) return;
    embed._tourLoading = fetch(embed.dataset.tour).then(r => r.json()).then(tour => {
      embed._tour = tour;
      embed._cluster = -1;   // -1 is the starting view
      const panel = make("aside", "cluster-panel", embed);
      const head = make("div", "cluster-head", panel);
      make("b", null, head, `${tour.n_clusters} clusters`);
      make("span", "cluster-sub", head, ` · ${tour.mode === "q" ? "FDR q" : "FWE"} < ${tour.level}, ≥ ${tour.k} voxels`);
      const nav = make("div", "cluster-nav", panel);
      const prev = make("button", null, nav, "◀");
      const position = make("span", "cluster-position", nav);
      const nextButton = make("button", null, nav, "▶");
      const overview = make("button", "ghost", nav, "start");
      prev.type = nextButton.type = overview.type = "button";
      prev.title = "previous cluster ([)"; nextButton.title = "next cluster (])";
      prev.addEventListener("click", () => clusterTo(embed, Math.max(0, embed._cluster - 1)));
      nextButton.addEventListener("click", () => clusterTo(embed, Math.min(tour.clusters.length - 1, embed._cluster + 1)));
      overview.addEventListener("click", () => clusterTo(embed, -1));
      make("div", "cluster-card", panel);
      const list = make("ol", "cluster-list", panel);
      tour.clusters.forEach((c, k) => {
        const item = make("li", null, list);
        item.innerHTML = `<span class="n">${k + 1}</span><span class="r">${short(c.region)}</span><span class="s">${c.size}</span>`;
        item.title = `${c.region} · ${c.size} voxels`;
        item.addEventListener("click", () => clusterTo(embed, k));
      });
      clusterTo(embed, -1);
    });
  }
  function short(region) {
    return region.replace("Lateral Occipital Cortex", "Lat. occipital").replace(", superior division", ", sup.")
      .replace(", inferior division", ", inf.").replace(", posterior division", ", post.").replace(", anterior division", ", ant.")
      .replace(", temporooccipital part", ", temporo-occ.").replace("Inferior Frontal Gyrus", "Inf. frontal gyrus")
      .replace("Superior Frontal Gyrus", "Sup. frontal gyrus").replace("Middle Frontal Gyrus", "Mid. frontal gyrus")
      .replace("Middle Temporal Gyrus", "Mid. temporal gyrus").replace("Superior Temporal Gyrus", "Sup. temporal gyrus");
  }
  function stepCluster(direction) {
    const embed = slides[index].querySelector(".embed[data-tour]");
    if (!embed || !embed._tour) return;
    const last = embed._tour.clusters.length - 1;
    clusterTo(embed, Math.max(direction < 0 ? -1 : 0, Math.min(last, embed._cluster + direction)));
  }
  function clusterTo(embed, k) {
    const tour = embed._tour;
    embed._cluster = k;
    const c = k >= 0 ? tour.clusters[k] : null;
    const panel = embed.querySelector(".cluster-panel");
    panel.querySelector(".cluster-position").textContent = c ? `${k + 1} / ${tour.clusters.length}` : `– / ${tour.clusters.length}`;
    panel.querySelectorAll(".cluster-list li").forEach((item, j) => item.classList.toggle("on", j === k));
    const on = panel.querySelector(".cluster-list li.on");
    if (on) on.scrollIntoView({ block: "nearest" });
    const card = panel.querySelector(".cluster-card");
    card.innerHTML = c
      ? `<div class="cluster-title">${c.region}</div>` +
        `<div>${c.size.toLocaleString()} voxels · ${(c.volume_mm3 / 1000).toFixed(1)} cm³` +
        `${c.region_share ? ` · ${Math.round(c.region_share * 100)}% in region` : ""}</div>` +
        `<div class="muted">centre ${c.centre_mm.map(Math.round).join(", ")} mm · peak Δr − null ${c.peak_value.toFixed(3)}</div>`
      : `<div class="muted">Starting view. Pick a cluster, or press ] for the largest.</div>`;
    const link = embed.querySelector(".open-link");
    if (link) link.href = c ? c.link : tour.start_link;
    settle(embed);
  }
  // Brings the viewer to the chosen cluster in whichever mode it is in, waiting for it to be ready.
  function settle(embed) {
    clearTimeout(embed._retry);
    const viewer = viewerOf(embed);
    if (!viewer || !viewer.state || !viewer.state.endpoint) { embed._retry = setTimeout(() => settle(embed), 400); return; }
    const tour = embed._tour;
    const c = embed._cluster >= 0 ? tour.clusters[embed._cluster] : null;
    if (viewer.state.mode === "surface") {
      // A cluster on the medial wall faces the other hemisphere, which would hide it: show only
      // its own hemisphere then, and both otherwise. Each toggle rebuilds the surface.
      const s = c && c.fsaverage;
      const want = s && medial(s) ? { lh: s.hemisphere === "lh", rh: s.hemisphere === "rh" } : { lh: true, rh: true };
      const have = viewer.state.hemispheres || { lh: true, rh: true };
      const toggles = ["lh", "rh"].filter(h => !have[h] && want[h]).concat(["lh", "rh"].filter(h => have[h] && !want[h]));
      if (toggles.length) {
        toggles.forEach(h => press(embed, `#hemispheres [data-hemi="${h}"]`));
        embed._retry = setTimeout(() => settle(embed), 500);
        return;
      }
      const shown = (viewer.parts || []).map(part => part.hemisphere).sort().join();
      if (shown !== ["lh", "rh"].filter(h => want[h]).join()) { embed._retry = setTimeout(() => settle(embed), 400); return; }
      if (!c) return;
      if (!viewer.parts || !viewer.parts.length || typeof viewer.markVertex !== "function" ||
          viewer.markVertex(s.hemisphere, s.vertex) === false) {
        if (typeof viewer.markVertex === "function") embed._retry = setTimeout(() => settle(embed), 400);
        return;
      }
      turnTo(embed, viewer, s);
    } else {
      glide(embed, c ? c.centre_mm : tour.start_mm);
    }
  }
  // Facing the midline: the vertex's direction from its (inflated) hemisphere's centre points medially.
  function medial(s) { return (s.hemisphere === "lh" ? s.outward[0] : -s.outward[0]) >= 0.25; }
  // Turns the surface camera to face the vertex, over ~1 s.
  function turnTo(embed, viewer, s) {
    // On the flat map the morph drives the camera, so only the marker moves there.
    if (typeof viewer.setCamera !== "function" || viewer.state.geometry === "flat") return;
    // "scene": face the vertex from the centre the view is framed on, which also centres it.
    const target = typeof viewer.vertexDirection === "function"
      ? viewer.vertexDirection(s.hemisphere, s.vertex, "scene") : null;
    if (!target) return;
    const from = [viewer.state.azimuth, viewer.state.elevation];
    const turn = ((target[0] - from[0] + 540) % 360) - 180;   // the short way round
    animate(embed, 1000, e => viewer.setCamera(from[0] + turn * e, from[1] + (target[1] - from[1]) * e));
  }
  function glide(embed, target) {
    const viewer = viewerOf(embed);
    // The instance the viewer itself takes the crosshair from (its crosshairSource): in the
    // multi and montage views the panels are their own NiiVue instances, kept in step by the viewer.
    const panels = viewer && viewer.montage;
    const nv = viewer && (panels && panels.length && panels[0].nv ? panels[0].nv : viewer.nv);
    if (!nv || !nv.volumes || nv.volumes.length < 1 || !nv.scene) { embed._retry = setTimeout(() => settle(embed), 400); return; }
    const from = Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0, 3);
    animate(embed, 1200, e => {
      nv.scene.crosshairPos = nv.mm2frac(from.map((v, k) => v + (target[k] - v) * e));
      nv.createOnLocationChange();
      nv.drawScene();
    });
  }
  function animate(embed, duration, apply) {
    const started = performance.now();
    const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
    const token = (embed._glide = (embed._glide || 0) + 1);
    function frame(now) {
      if (token !== embed._glide) return;   // a newer move took over
      const t = Math.min(1, (now - started) / duration);
      try { apply(ease(t)); } catch (err) { return; }
      if (t < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  // ---- math and charts ----------------------------------------------------------------
  function renderMath(root) {
    if (!window.katex) return;
    root.querySelectorAll("[data-tex]").forEach(el => {
      try { katex.render(el.dataset.tex, el, { displayMode: el.classList.contains("math-display"), throwOnError: false }); } catch (e) { el.textContent = el.dataset.tex; }
    });
  }

  const charts = {
    // The share of feature variance each ToM-ablation null removes, relative to ToM itself.
    "null-variance": function (el) {
      fetch("data/null_variance.json").then(r => r.json()).then(data => {
        const families = [["lanczos", "Lanczos sum features"], ["hann", "Hann mean features"]];
        const nulls = ["variance-matched", "TR shifts", "story permutations", "rating shifts"];
        const features = [["english1000", "English1000", "#6aa7f0"], ["bert_wordctx10", "BERT", "#f0a15e"], ["gpt2xl_l24_wordctx10", "GPT-2 XL", "#3ec58f"]];
        const W = 1000, H = 520, L = 190, R = 20, T = 30, B = 70, gap = 40;
        const panel = (W - L - R - gap) / 2;
        const NS = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        const tip = document.createElement("div");
        tip.className = "tip"; tip.hidden = true;
        const add = (tag, attrs, text) => { const n = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v)); if (text != null) n.textContent = text; svg.append(n); return n; };
        const rowY = k => T + (k + 0.5) * (H - T - B) / nulls.length;
        nulls.forEach((name, k) => add("text", { x: L - 14, y: rowY(k) + 6, "text-anchor": "end", "font-size": 20 }, name));
        families.forEach(([family, title], f) => {
          const x0 = L + f * (panel + gap);
          const x = v => x0 + (Math.max(-0.05, Math.min(1.15, v)) + 0.05) / 1.2 * panel;
          add("rect", { x: x(0.9), y: T, width: x(1.1) - x(0.9), height: H - T - B, fill: "rgba(106,167,240,0.15)" });
          add("line", { x1: x(1), x2: x(1), y1: T, y2: H - B, stroke: "#e8ebf0", "stroke-width": 2 });
          add("text", { x: x(1), y: T - 8, "text-anchor": "middle", "font-size": 18, fill: "#e8ebf0" }, "ToM");
          [0, 0.25, 0.5, 0.75, 1].forEach(v => {
            add("line", { x1: x(v), x2: x(v), y1: H - B, y2: H - B + 6, stroke: "#9aa4b2" });
            add("text", { x: x(v), y: H - B + 26, "text-anchor": "middle", "font-size": 17 }, v);
          });
          add("text", { x: x0 + panel / 2, y: H - 12, "text-anchor": "middle", "font-size": 18 }, `${title}: variance removed ÷ ToM's`);
          nulls.forEach((name, k) => {
            features.forEach(([feature, label, colour], j) => {
              const entry = ((data.families[family] || {})[name] || {})[feature];
              if (!entry) return;
              const y0 = rowY(k) + (j - 1) * 18;
              entry.values.forEach((v, i) => {
                const dot = add("circle", { cx: x(v), cy: y0 + ((i * 7919) % 13 - 6), r: entry.values.length > 50 ? 2.5 : 5,
                                            fill: colour, "fill-opacity": entry.values.length > 50 ? 0.35 : 0.9 });
                dot.addEventListener("mousemove", ev => {
                  tip.hidden = false;
                  tip.innerHTML = `<b>${name}</b> · ${label}<br>this draw: ${v.toFixed(3)} × ToM<br>median ${entry.median} × ToM (${entry.n} draws)<br>ToM removes ${entry.tom_percent}% of ${label}'s variance`;
                  const box = el.getBoundingClientRect();
                  tip.style.left = `${ev.clientX - box.left + 12}px`; tip.style.top = `${ev.clientY - box.top + 12}px`;
                });
                dot.addEventListener("mouseleave", () => { tip.hidden = true; });
              });
            });
          });
        });
        features.forEach(([, label, colour], j) => {
          add("circle", { cx: 20 + j * 150, cy: 12, r: 6, fill: colour });
          add("text", { x: 32 + j * 150, y: 18, "font-size": 17 }, label);
        });
        el.replaceChildren(svg, tip);
      });
    },
  };
  function renderCharts(root) {
    root.querySelectorAll("[data-chart]").forEach(el => { if (charts[el.dataset.chart]) charts[el.dataset.chart](el); });
  }

  // ---- layout ---------------------------------------------------------------------------
  // The slide is 16:9, sized to the window less whatever the editor puts beside it (the slide
  // sorter on the left, the toolbar on top). --u is 1% of the slide's width.
  let unit = 16, insets = { left: 0, top: 0 };
  function layout() {
    const width = innerWidth - insets.left, height = innerHeight - insets.top;
    const pad = insets.left || insets.top ? 16 : 0;
    unit = Math.max(1, Math.min((width - 2 * pad) / 100, (height - 2 * pad) * 1.7778 / 100));
    document.documentElement.style.setProperty("--u", `${unit}px`);
    deck.style.margin = "0";
    deck.style.left = `${insets.left + (width - 100 * unit) / 2}px`;
    deck.style.top = `${insets.top + (height - 56.25 * unit) / 2}px`;
    deck.style.right = deck.style.bottom = "auto";
    restyleViewers();
    listeners.forEach(fn => fn(index, step, "layout"));
  }
  window.addEventListener("resize", layout);
  document.addEventListener("fullscreenchange", () => {
    document.body.classList.toggle("fullscreen", !!document.fullscreenElement);
    if (window.Editor) window.Editor.layoutChanged(); else layout();
  });

  // ---- source -----------------------------------------------------------------------------
  // A slide's (or any element's) source: rendered math, loaded embeds, drawn charts and runtime
  // state stripped.
  function slideSource(section) {
    const copy = section.cloneNode(true);
    const all = [copy, ...copy.querySelectorAll("*")];
    // Embeds are empty in the source; math and charts are drawn from their attributes.
    all.filter(el => el.matches("[data-tex], .embed, [data-chart]")).forEach(el => { el.innerHTML = ""; });
    [copy, ...copy.querySelectorAll("*")].forEach(el => {
      el.removeAttribute("contenteditable");
      el.removeAttribute("spellcheck");
      el.classList.remove("current", "future", "past", "gone", "ed-selected", "booting");
      if (el.getAttribute("class") === "") el.removeAttribute("class");
      if (el.getAttribute("style") === "") el.removeAttribute("style");
    });
    return copy.outerHTML;
  }
  // Only the slides themselves: the <!--SLIDES--> markers stay in index.html, outside what is saved.
  function serialise() { return slides.map(slideSource).join("\n\n"); }
  // Builds a slide from its source, with its math and charts drawn.
  function build(html) {
    const holder = document.createElement("template");
    holder.innerHTML = html.trim();
    const section = holder.content.firstElementChild;
    renderMath(section);
    renderCharts(section);
    return section;
  }
  // The deck's slides changed (added, removed, reordered): look again and show slide i.
  function refresh(i, s) {
    slides = Array.from(deck.querySelectorAll(":scope > .slide"));
    show(i == null ? index : i, s == null ? 0 : s, false);
  }

  let flashTimer = null;
  function flash(text, ms) {
    status.textContent = text; status.hidden = false;
    clearTimeout(flashTimer);
    if (ms) flashTimer = setTimeout(() => { status.hidden = true; }, ms);
  }

  const listeners = [];
  window.Deck = {
    LOCAL,
    get slides() { return slides; },
    get index() { return index; },
    get step() { return step; },
    get unit() { return unit; },
    element: deck,
    show, refresh, next, previous, nextSlide, previousSlide, stepsOf,
    slideSource, serialise, build, renderMath, renderCharts, flash,
    setInsets(value) { insets = value; layout(); },
    onChange(fn) { listeners.push(fn); },
    onNavigationKey(event) { onKey(event, false); },
  };

  renderMath(document);
  renderCharts(document);
  layout();
  show(0, 0, true);
  readHash();
})();

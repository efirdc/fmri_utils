// Slide sorter and slide editor for the lab-meeting deck, on top of deck.js (window.Deck).
//
// Slide sorter (left, whenever the deck is not full screen): thumbnails of every slide; click to go
// there. On the local copy (served by serve_talk.py) it also edits the deck: drag to reorder,
// Ctrl+X / C / V / D and Delete on the selected slides, a right-click menu, and new slides.
//
// Editor (local copy only; E, or the pencil in the sorter): a toolbar over the slide.
// - Click text to type in it; Esc leaves the text with its box selected; Esc again deselects.
//   Shift-click adds to the selection. Right-click for a menu of what applies.
// - A selected element shows a frame: drag its edge to move it, its handles to resize it.
// - Layout: the first time anything on a slide is moved or resized, the slide's flow layout is
//   frozen in place: every element at that level becomes a box (.box, left/top/width/height in %
//   of the box or slide it sits in), so nothing else shifts. A container in the flow (a .cols
//   grid, a column) becomes a group the same way. Groups: select several, Group (Ctrl+G); a
//   click selects the whole group, a second click an element in it; Ungroup (Ctrl+Shift+G).
// - Insert text boxes, images (file, paste or drop; saved under figures/uploads/) and equations:
//   inline in text at the caret, or a displayed equation of its own (a box that is only that).
// - Text: paragraph style, size, bold/italic/underline/strike, colour, alignment, lists.
// - Animation of the selection, or of the list item / paragraph under the caret (the toolbar
//   names which): the step it appears at, its effect, dim after, the step it goes at; "one by one"
//   gives a list's items (a text box's paragraphs, a group's elements) successive steps.
// - Undo/redo (Ctrl+Z / Ctrl+Y) over every change; Ctrl+S saves the slides into index.html.
(function () {
  "use strict";
  const D = window.Deck;
  if (!D) return;
  const LOCAL = D.LOCAL;
  const deckEl = D.element;
  const SORTER_WIDTH = 214, THUMB_WIDTH = 166;

  // ---- small helpers ------------------------------------------------------------------------
  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (v == null || v === false) return;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    });
    children.flat().forEach(c => { if (c != null) node.append(c); });
    return node;
  }
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const round = (v, d = 2) => Math.round(v * 10 ** d) / 10 ** d;
  const currentSlide = () => D.slides[D.index];
  const docOrder = (a, b) => (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1);
  function fromSource(html) {
    // Elements (or slides) built from source, with their math and charts drawn.
    const holder = h("div");
    holder.innerHTML = html.trim();
    D.renderMath(holder);
    D.renderCharts(holder);
    return Array.from(holder.children);
  }
  // Without transitions for a frame (entering or leaving edit mode shows or hides every step).
  function instantly(fn) {
    deckEl.classList.add("no-anim");
    fn();
    void deckEl.offsetWidth;
    requestAnimationFrame(() => requestAnimationFrame(() => deckEl.classList.remove("no-anim")));
  }

  // ---- history ------------------------------------------------------------------------------
  // Every change commits a snapshot (each slide's source); undo/redo restore one, keeping the
  // slides that did not change (and so their loaded viewers) in place.
  const history = { stack: [], at: -1, saved: "" };
  const snapshot = () => D.slides.map(D.slideSource);
  function commit() {
    const snap = snapshot();
    const last = history.stack[history.at];
    if (last && last.join("\u0000") === snap.join("\u0000")) { updateState(); return; }
    history.stack = history.stack.slice(0, history.at + 1);
    history.stack.push(snap);
    if (history.stack.length > 200) history.stack.shift();
    history.at = history.stack.length - 1;
    afterChange();
  }
  function restore(snap) {
    const current = D.slides.slice();
    const sources = current.map(D.slideSource);
    const used = new Set();
    const nodes = snap.map(html => {
      const k = sources.findIndex((s, i) => !used.has(i) && s === html);
      if (k >= 0) { used.add(k); return current[k]; }
      return fromSource(html)[0];
    });
    current.forEach((node, i) => { if (!used.has(i)) node.remove(); });
    nodes.forEach((node, i) => {
      const at = deckEl.querySelectorAll(":scope > .slide")[i];
      if (at !== node) deckEl.insertBefore(node, at || overlay);
    });
    deselect();
    D.refresh(Math.min(D.index, nodes.length - 1), 0);
    if (editor.active) D.slides.forEach(prepare);   // after the refresh: D.slides is the new list
    afterChange();
  }
  function undo() { if (history.at > 0) { history.at -= 1; restore(history.stack[history.at]); } }
  function redo() { if (history.at < history.stack.length - 1) { history.at += 1; restore(history.stack[history.at]); } }
  const dirty = () => (history.stack[history.at] || []).join("\u0000") !== history.saved;
  function afterChange() {
    renderSorter();
    updateState();
    drawOverlay();
  }
  function save() {
    if (!LOCAL) return;
    const body = D.serialise();
    fetch("__save", { method: "POST", body })
      .then(r => { if (!r.ok) throw new Error(r.status); history.saved = (history.stack[history.at] || []).join("\u0000"); updateState(); D.flash("saved to index.html", 1500); })
      .catch(e => D.flash(`not saved (${e.message}); is serve_talk.py running?`, 4000));
  }
  window.addEventListener("beforeunload", event => { if (LOCAL && dirty()) { event.preventDefault(); event.returnValue = ""; } });

  // ---- slide sorter -------------------------------------------------------------------------
  let sorterOpen = true;
  try { sorterOpen = localStorage.getItem("deck-sorter") !== "closed"; } catch (e) { /* no storage */ }
  const sorterList = h("div", { class: "sorter-list" });
  const sorterState = h("span", { class: "sorter-state", title: "unsaved changes" });
  const sorter = h("aside", { id: "sorter", tabindex: "0" },
    h("div", { class: "sorter-head" },
      h("span", { class: "sorter-title", text: "Slides" }), sorterState,
      LOCAL ? h("button", { type: "button", class: "icon", title: "edit slides (E)", text: "✎", onclick: () => editor.toggle() }) : null,
      h("button", { type: "button", class: "icon", title: "hide the slide list", text: "«", onclick: () => setSorter(false) })),
    sorterList,
    LOCAL ? h("div", { class: "sorter-foot" },
      h("button", { type: "button", text: "+ New slide", onclick: event => newSlideMenu(event.currentTarget) }),
      h("button", { type: "button", text: "Duplicate", onclick: () => duplicateSlides() }),
      h("button", { type: "button", text: "Delete", onclick: () => deleteSlides() })) : null);
  const sorterTab = h("button", { id: "sorter-tab", type: "button", title: "show the slide list", text: "»", onclick: () => setSorter(true) });
  document.body.append(sorter, sorterTab);
  function setSorter(open) {
    sorterOpen = open;
    try { localStorage.setItem("deck-sorter", open ? "open" : "closed"); } catch (e) { /* no storage */ }
    layoutChanged();
  }
  let selectedSlides = new Set([0]), anchor = 0;
  function embedLabel(embed) {
    if (embed.classList.contains("brain")) return "brain viewer";
    return embed.dataset.label || "embedded page";
  }
  function thumbOf(section) {
    const copy = section.cloneNode(true);
    [copy, ...copy.querySelectorAll("*")].forEach(el => {
      el.classList.remove("current", "future", "past", "gone", "ed-selected");
      el.removeAttribute("contenteditable");
      el.removeAttribute("id");
    });
    copy.querySelectorAll(".embed").forEach(embed => {
      embed.classList.remove("booting");
      embed.replaceChildren(h("div", { class: "thumb-embed", text: embedLabel(embed) }));
    });
    const holder = h("div", { class: "thumb" }, copy);
    holder.style.setProperty("--u", `${THUMB_WIDTH / 100}px`);
    return holder;
  }
  function renderSorter() {
    const scroll = sorterList.scrollTop;
    sorterList.replaceChildren(...D.slides.map((section, i) => {
      const item = h("div", { class: "sorter-item", draggable: LOCAL ? "true" : null, "data-index": i },
        h("span", { class: "sorter-number", text: String(i + 1) }), thumbOf(section));
      item.addEventListener("click", event => clickSlide(event, i));
      item.addEventListener("contextmenu", event => { if (LOCAL) { event.preventDefault(); slideMenu(event, i); } });
      if (LOCAL) wireDrag(item, i);
      return item;
    }));
    sorterList.scrollTop = scroll;
    markSorter();
  }
  function markSorter() {
    selectedSlides = new Set([...selectedSlides].filter(i => i < D.slides.length));
    if (!selectedSlides.size) selectedSlides.add(D.index);
    sorterList.querySelectorAll(".sorter-item").forEach((item, i) => {
      item.classList.toggle("current", i === D.index);
      item.classList.toggle("picked", selectedSlides.has(i));
    });
  }
  function clickSlide(event, i) {
    sorter.focus();
    if (event.shiftKey) {
      selectedSlides = new Set();
      for (let k = Math.min(anchor, i); k <= Math.max(anchor, i); k += 1) selectedSlides.add(k);
    } else if (event.ctrlKey || event.metaKey) {
      if (selectedSlides.has(i) && selectedSlides.size > 1) selectedSlides.delete(i); else selectedSlides.add(i);
      anchor = i;
    } else {
      selectedSlides = new Set([i]);
      anchor = i;
    }
    D.show(i, 0);
  }
  function scrollToCurrent() {
    const item = sorterList.children[D.index];
    if (item) item.scrollIntoView({ block: "nearest" });
  }
  sorter.addEventListener("keydown", event => {
    const ctrl = event.ctrlKey || event.metaKey, key = event.key.toLowerCase();
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault(); event.stopPropagation();
      const i = clamp(D.index + (event.key === "ArrowDown" ? 1 : -1), 0, D.slides.length - 1);
      selectedSlides = new Set([i]); anchor = i; D.show(i, 0);
      return;
    }
    if (!LOCAL) return;
    let handled = true;
    if (ctrl && key === "c") copySlides();
    else if (ctrl && key === "x") { copySlides(); deleteSlides(); }
    else if (ctrl && key === "v") pasteSlides();
    else if (ctrl && key === "d") duplicateSlides();
    else if (ctrl && key === "z") { if (event.shiftKey) redo(); else undo(); }
    else if (ctrl && key === "y") redo();
    else if (ctrl && key === "s") save();
    else if (event.key === "Delete" || event.key === "Backspace") deleteSlides();
    else handled = false;
    if (handled) { event.preventDefault(); event.stopPropagation(); }
  });

  // Slide operations (local copy).
  const TEMPLATES = {
    "Title and bullets": `<section class="slide"><h2>Title</h2><ul><li>First point</li></ul></section>`,
    "Title only": `<section class="slide"><h2>Title</h2></section>`,
    "Two columns": `<section class="slide"><h2>Title</h2><div class="cols"><div><p>Left</p></div><div><p>Right</p></div></div></section>`,
    "Section title": `<section class="slide title-slide"><h1>Section title</h1><p class="byline">Subtitle</p></section>`,
    "Blank": `<section class="slide"></section>`,
  };
  let clipboard = null;   // { kind: "slides" | "elements", html: [...] }
  const picked = () => [...selectedSlides].filter(i => i < D.slides.length).sort((a, b) => a - b);
  function insertSlides(htmls, after) {
    const ref = D.slides[after + 1] || overlay;
    const nodes = htmls.flatMap(fromSource);
    nodes.forEach(node => { deckEl.insertBefore(node, ref); if (editor.active) prepare(node); });
    selectedSlides = new Set(nodes.map((_, k) => after + 1 + k));
    anchor = after + 1;
    D.refresh(after + 1, 0);
    commit();
    scrollToCurrent();
  }
  function newSlide(name) { insertSlides([TEMPLATES[name]], D.index); }
  function copySlides() {
    clipboard = { kind: "slides", html: picked().map(i => D.slideSource(D.slides[i])) };
    D.flash(`copied ${clipboard.html.length} slide${clipboard.html.length > 1 ? "s" : ""}`, 1200);
  }
  function pasteSlides() {
    if (!clipboard || clipboard.kind !== "slides") return;
    insertSlides(clipboard.html, Math.max(...picked(), D.index));
  }
  function duplicateSlides() {
    const list = picked();
    insertSlides(list.map(i => D.slideSource(D.slides[i])), list[list.length - 1]);
  }
  function deleteSlides() {
    const list = picked();
    if (!list.length) return;
    if (list.length >= D.slides.length) {
      list.slice(1).forEach(i => D.slides[i].remove());
      D.slides[list[0]].replaceWith(fromSource(TEMPLATES["Blank"])[0]);
    } else {
      list.forEach(i => D.slides[i].remove());
    }
    deselect();
    const next = Math.min(list[0], deckEl.querySelectorAll(":scope > .slide").length - 1);
    selectedSlides = new Set([next]); anchor = next;
    D.refresh(next, 0);
    commit();
  }
  function moveSlides(list, to) {
    // Moves the slides in list so the first lands before the slide now at index to.
    const nodes = list.map(i => D.slides[i]);
    const ref = D.slides.slice(to).find(node => !nodes.includes(node)) || overlay;
    nodes.forEach(node => deckEl.insertBefore(node, ref));
    const order = Array.from(deckEl.querySelectorAll(":scope > .slide"));
    selectedSlides = new Set(nodes.map(node => order.indexOf(node)));
    anchor = order.indexOf(nodes[0]);
    D.refresh(anchor, 0);
    commit();
  }
  let dragFrom = null;
  function wireDrag(item, i) {
    item.addEventListener("dragstart", event => {
      if (!selectedSlides.has(i)) { selectedSlides = new Set([i]); anchor = i; }
      dragFrom = picked();
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", "slides");
    });
    item.addEventListener("dragover", event => {
      if (!dragFrom) return;
      event.preventDefault();
      const box = item.getBoundingClientRect();
      const after = event.clientY > box.top + box.height / 2;
      sorterList.querySelectorAll(".drop-before, .drop-after").forEach(n => n.classList.remove("drop-before", "drop-after"));
      item.classList.add(after ? "drop-after" : "drop-before");
    });
    item.addEventListener("drop", event => {
      if (!dragFrom) return;
      event.preventDefault();
      const after = item.classList.contains("drop-after");
      const list = dragFrom;
      dragFrom = null;
      moveSlides(list, after ? i + 1 : i);
    });
    item.addEventListener("dragend", () => {
      dragFrom = null;
      sorterList.querySelectorAll(".drop-before, .drop-after").forEach(n => n.classList.remove("drop-before", "drop-after"));
    });
  }

  // A small menu (right click, + New slide).
  let menu = null;
  function closeMenu() { if (menu) { menu.remove(); menu = null; } }
  function openMenu(x, y, items) {
    closeMenu();
    const rows = items.filter(item => item && (item === "-" || !item.hidden));
    // No separator first, last or twice in a row.
    const clean = rows.filter((item, k) => item !== "-" || (k > 0 && k < rows.length - 1 && rows[k + 1] !== "-"));
    menu = h("div", { class: "ed-menu" }, clean.map(item => item === "-" ? h("hr")
      : h("button", { type: "button", disabled: item.disabled, onclick: () => { closeMenu(); item.run(); } }, item.label,
          item.keys ? h("span", { class: "keys", text: item.keys }) : null)));
    document.body.append(menu);
    const box = menu.getBoundingClientRect();
    menu.style.left = `${Math.min(x, innerWidth - box.width - 8)}px`;
    menu.style.top = `${Math.max(8, Math.min(y, innerHeight - box.height - 8))}px`;
  }
  document.addEventListener("pointerdown", event => { if (menu && !menu.contains(event.target)) closeMenu(); }, true);
  function newSlideMenu(anchorEl) {
    const box = anchorEl.getBoundingClientRect();
    openMenu(box.left, box.top - 8 - 36 * Object.keys(TEMPLATES).length, Object.keys(TEMPLATES).map(name => ({ label: name, run: () => newSlide(name) })));
  }
  function slideMenu(event, i) {
    if (!selectedSlides.has(i)) { selectedSlides = new Set([i]); anchor = i; D.show(i, 0); }
    const n = picked().length, s = n > 1 ? `${n} slides` : "slide";
    openMenu(event.clientX, event.clientY, [
      ...Object.keys(TEMPLATES).map(name => ({ label: `New: ${name.toLowerCase()}`, run: () => newSlide(name) })),
      "-",
      { label: `Duplicate ${s}`, keys: "Ctrl+D", run: duplicateSlides },
      { label: `Cut ${s}`, keys: "Ctrl+X", run: () => { copySlides(); deleteSlides(); } },
      { label: `Copy ${s}`, keys: "Ctrl+C", run: copySlides },
      { label: "Paste after", keys: "Ctrl+V", disabled: !clipboard || clipboard.kind !== "slides", run: pasteSlides },
      "-",
      { label: "Move up", disabled: picked()[0] === 0, run: () => moveSlides(picked(), picked()[0] - 1) },
      { label: "Move down", disabled: picked()[picked().length - 1] >= D.slides.length - 1, run: () => moveSlides(picked(), picked()[picked().length - 1] + 2) },
      "-",
      { label: `Delete ${s}`, keys: "Del", run: deleteSlides },
    ]);
  }

  // ---- layout ---------------------------------------------------------------------------------
  function layoutChanged() {
    const full = !!document.fullscreenElement;
    const showSorter = sorterOpen && !full;
    sorter.hidden = !showSorter;
    sorterTab.hidden = showSorter || full;
    toolbar.hidden = !editor.active || full;
    const top = toolbar.hidden ? 0 : toolbar.offsetHeight;
    sorterTab.style.top = `${top + 6}px`;   // below the toolbar while editing
    D.setInsets({ left: showSorter ? SORTER_WIDTH : 0, top });
    toolbar.style.left = `${showSorter ? SORTER_WIDTH : 0}px`;
    drawOverlay();
  }

  // ---- the editor -----------------------------------------------------------------------------
  const TEXT_ROOT = ".box.text, h1, h2, h3, h4, p, ul, ol, figcaption, td, th, .editable, blockquote";
  const NOT_TEXT = ".embed, [data-chart], [data-tex], .thumb-embed";
  let selection = [], selected = null;   // selected: the last one picked (the toolbar's subject)
  const editor = {
    active: false,
    toggle() { if (editor.active) exit(); else enter(); },
    onKey,
    layoutChanged,
  };
  window.Editor = editor;

  function prepare(section) {
    section.querySelectorAll(TEXT_ROOT).forEach(el => {
      if (el.closest(NOT_TEXT)) return;
      if (el.parentElement && el.parentElement.closest("[contenteditable='true']")) return;
      el.setAttribute("contenteditable", "true");
    });
    section.querySelectorAll("[data-tex]").forEach(el => el.setAttribute("contenteditable", "false"));
  }
  function unprepare(section) {
    section.querySelectorAll("[contenteditable]").forEach(el => el.removeAttribute("contenteditable"));
  }
  function enter() {
    if (!LOCAL || editor.active) return;
    editor.active = true;
    instantly(() => document.body.classList.add("editing"));
    D.slides.forEach(prepare);
    layoutChanged();
    updateToolbar();
    D.flash("editing · click text to type · drag a frame's edge to move · right-click for more · Ctrl+S saves", 3500);
  }
  function exit() {
    if (!editor.active) return;
    flushTyping();
    closeLatex();
    deselect();
    editor.active = false;
    D.slides.forEach(unprepare);
    if (document.activeElement && deckEl.contains(document.activeElement)) document.activeElement.blur();
    instantly(() => {
      document.body.classList.remove("editing");
      D.show(D.index, D.step, true);
    });
    layoutChanged();
    if (dirty()) D.flash("unsaved changes: E to edit again, then Ctrl+S", 3000);
  }

  // ---- what a click selects -------------------------------------------------------------------
  // A group (a box holding boxes) is picked whole until something in it is selected; then a click
  // picks the element in it. Otherwise: an equation, the innermost box, or the block in the flow
  // (a child of the slide, of a .cols grid or of one of its columns).
  const isGroup = el => !!(el && el.classList.contains("box") && el.querySelector(":scope > .box"));
  function isColumn(el) {
    return el && el.tagName === "DIV" && el.parentElement && el.parentElement.classList.contains("cols");
  }
  function blockOf(node) {
    if (!node) return null;
    if (node.nodeType !== 1) node = node.parentElement;
    const slide = node && node.closest(".slide");
    if (!slide || !deckEl.contains(slide)) return null;
    const chain = [];
    for (let e = node; e && e !== slide; e = e.parentElement) if (e.classList.contains("box")) chain.unshift(e);
    if (chain.length && isGroup(chain[0]) && !selection.some(s => chain[0].contains(s))) return chain[0];
    const math = node.closest("[data-tex]");
    const inner = chain[chain.length - 1];
    if (math && (!inner || inner.contains(math))) return math;
    if (inner) return inner;
    let el = node;
    while (el && el !== slide && el.parentElement !== slide && !el.parentElement.classList.contains("cols") && !isColumn(el.parentElement)) el = el.parentElement;
    return el === slide ? null : el;
  }
  const textMode = () => {
    const a = document.activeElement;
    return !!(a && a.isContentEditable && deckEl.contains(a));
  };
  function setSelection(list) {
    selection.forEach(el => el.classList.remove("ed-selected"));
    selection = list.filter((el, k) => el && list.indexOf(el) === k);
    selected = selection[selection.length - 1] || null;
    animItem = null;
    selection.forEach(el => el.classList.add("ed-selected"));
    drawOverlay();
    updateToolbar();
  }
  function select(el) { if (selection.length !== 1 || selected !== el) setSelection(el ? [el] : []); }
  function toggleSelect(el) {
    setSelection(selection.includes(el) ? selection.filter(s => s !== el) : selection.filter(s => !s.contains(el) && !el.contains(s)).concat(el));
  }
  function deselect() { if (selection.length) setSelection([]); }

  // ---- geometry ---------------------------------------------------------------------------------
  // A box is placed in % of its container: the nearest box around it, else the slide.
  const containerOf = el => el.parentElement.closest(".box, .slide");
  function pctIn(r, c) {
    return { left: (r.left - c.left) / c.width * 100, top: (r.top - c.top) / c.height * 100,
             width: r.width / c.width * 100, height: r.height / c.height * 100 };
  }
  const rel = el => pctIn(el.getBoundingClientRect(), containerOf(el).getBoundingClientRect());
  const needsHeight = el => el.matches(".embed, [data-chart], .figure, .box.image, .chart, .group") || !!el.style.height;
  function setRect(el, r, withHeight) {
    el.style.left = `${round(r.left)}%`;
    el.style.top = `${round(r.top)}%`;
    el.style.width = `${round(Math.max(0.5, r.width))}%`;
    if (withHeight) el.style.height = `${round(Math.max(0.5, r.height))}%`;
  }
  // Freezes a container's flow: each of its children becomes a box where it now is, so moving
  // one moves nothing else. The container itself is frozen into its own container first.
  const positionable = el => el.nodeType === 1 && !el.matches("script, style, br, #ed-overlay");
  function freeze(container) {
    const slide = container.closest(".slide");
    if (container !== slide && !container.classList.contains("box")) freeze(container.parentElement);
    const kids = Array.from(container.children).filter(k => positionable(k) && !k.classList.contains("box"));
    if (!kids.length) return;
    const c = container.getBoundingClientRect();
    const rects = kids.map(k => k.getBoundingClientRect());
    if (container !== slide) {
      // Its children are about to leave its flow: it keeps its size, and is now a group.
      container.style.height = `${round(rel(container).height)}%`;
      container.classList.add("group");
    }
    kids.forEach((k, i) => {
      k.classList.add("box");
      setRect(k, pctIn(rects[i], c), needsHeight(k));
    });
  }
  function makeFree(el) {
    if (el.classList.contains("box")) return true;
    if (el.matches(".math")) return false;   // inline equations stay in their text
    freeze(el.parentElement);
    return el.classList.contains("box");
  }

  // Snapping (one element): edges and centres to its container's edges and centre and to the
  // other boxes beside it.
  const SNAP = 0.7;
  function snapLines(el) {
    const xs = [0, 50, 100], ys = [0, 50, 100];
    const container = containerOf(el);
    Array.from(container.children).forEach(other => {
      if (other === el || !other.classList.contains("box")) return;
      const r = rel(other);
      xs.push(r.left, r.left + r.width / 2, r.left + r.width);
      ys.push(r.top, r.top + r.height / 2, r.top + r.height);
    });
    return { xs, ys, container };
  }
  function snap(value, edges, lines) {
    let best = null;
    edges.forEach(offset => lines.forEach(line => {
      const diff = line - (value + offset);
      if (Math.abs(diff) <= SNAP && (!best || Math.abs(diff) < Math.abs(best.diff))) best = { diff, line };
    }));
    return best;
  }

  // Dragging the selection (move) or a handle of one element (resize).
  function startDrag(event, els, handle, onClick) {
    event.preventDefault();
    event.stopPropagation();
    const x0 = event.clientX, y0 = event.clientY;
    let starts = null, moved = false, lines = null;
    const one = els.length === 1 ? els[0] : null;
    const aspect = one && (one.matches(".box.image") || !!one.querySelector(":scope > img"));
    function move(ev) {
      if (!moved) {
        if (Math.hypot(ev.clientX - x0, ev.clientY - y0) < 3) return;
        if (!els.every(makeFree)) return;
        moved = true;
        starts = els.map(el => ({ el, r: rel(el), c: containerOf(el).getBoundingClientRect() }));
        if (one) lines = snapLines(one);
      }
      const guides = [];
      starts.forEach(({ el, r: start, c }) => {
        const dx = (ev.clientX - x0) / c.width * 100, dy = (ev.clientY - y0) / c.height * 100;
        const r = { ...start };
        if (!handle) {
          r.left += dx; r.top += dy;
          if (one && !ev.altKey) {
            const sx = snap(r.left, [0, r.width / 2, r.width], lines.xs);
            const sy = snap(r.top, [0, r.height / 2, r.height], lines.ys);
            if (sx) { r.left += sx.diff; guides.push({ x: sx.line, c }); }
            if (sy) { r.top += sy.diff; guides.push({ y: sy.line, c }); }
          }
        } else {
          if (handle.includes("e")) r.width = start.width + dx;
          if (handle.includes("s")) r.height = start.height + dy;
          if (handle.includes("w")) { r.left = start.left + dx; r.width = start.width - dx; }
          if (handle.includes("n")) { r.top = start.top + dy; r.height = start.height - dy; }
          if (aspect && handle.length === 2 && !ev.shiftKey) {
            // Corners keep the picture's shape (its height/width in % carries the container's shape).
            const ratio = start.height / start.width;
            const w = Math.max(r.width, r.height / ratio);
            if (handle.includes("w")) r.left = start.left + start.width - w;
            if (handle.includes("n")) r.top = start.top + start.height - w * ratio;
            r.width = w; r.height = w * ratio;
          }
          if (r.width < 2) { if (handle.includes("w")) r.left -= 2 - r.width; r.width = 2; }
          if (r.height < 2) { if (handle.includes("n")) r.top -= 2 - r.height; r.height = 2; }
        }
        const withHeight = handle ? (/[ns]/.test(handle) || needsHeight(el)) : !!el.style.height;
        setRect(el, r, withHeight);
      });
      drawOverlay(guides);
    }
    function up() {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      drawOverlay();
      if (moved) commit();
      else if (!handle && one && one.matches("[data-tex]")) openLatex(one);
      else if (!handle && onClick) onClick();
    }
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  // ---- pointer, typing, paste and drop ----------------------------------------------------------
  deckEl.addEventListener("pointerdown", event => {
    if (!editor.active || event.button !== 0) return;
    if (event.target.closest("#ed-overlay")) return;
    closeMenu();
    const block = blockOf(event.target);
    if (!block) { closeLatex(); deselect(); if (textMode()) document.activeElement.blur(); return; }
    if (latexTarget && latexTarget !== block) closeLatex();
    if (event.shiftKey) {
      event.preventDefault();
      if (textMode()) document.activeElement.blur();
      toggleSelect(block);
      return;
    }
    if (!selection.includes(block)) select(block);
    const inText = event.target.closest("[contenteditable='true']");
    const editable = inText && block.contains(inText) && !isGroup(block) && !block.matches("[data-tex]");
    if (editable && selection.length === 1 && !event.altKey) return;   // the caret lands where clicked
    const x = event.clientX, y = event.clientY;
    startDrag(event, selection.slice(), null, () => {
      if (selection.length <= 1) return;
      select(block);
      if (editable) {
        const range = document.caretRangeFromPoint && document.caretRangeFromPoint(x, y);
        inText.focus();
        if (range) { const sel = getSelection(); sel.removeAllRanges(); sel.addRange(range); }
      }
    });
  }, true);
  // Typing: commit a while after the last key, and refresh the toolbar as the caret moves.
  let typingTimer = null;
  function flushTyping() { if (typingTimer) { clearTimeout(typingTimer); typingTimer = null; commit(); } }
  function commitSoon(ms) { clearTimeout(typingTimer); typingTimer = setTimeout(() => { typingTimer = null; commit(); }, ms); }
  deckEl.addEventListener("input", () => { if (editor.active) { commitSoon(700); drawOverlay(); } });
  deckEl.addEventListener("focusout", () => { if (editor.active) flushTyping(); });
  let lastRange = null;
  document.addEventListener("selectionchange", () => {
    if (!editor.active) return;
    const sel = getSelection();
    if (sel.rangeCount && deckEl.contains(sel.anchorNode)) {
      lastRange = sel.getRangeAt(0).cloneRange();
      const block = blockOf(sel.anchorNode);
      if (block && block !== selected && textMode()) select(block);
      if (textMode()) animItem = caretItem();
      updateToolbar();
    }
  });
  // Pasting into text: plain text only. Pictures (pasted or dropped) become image boxes.
  document.addEventListener("paste", event => {
    if (!editor.active) return;
    const files = Array.from(event.clipboardData.files || []).filter(f => f.type.startsWith("image/"));
    if (files.length) { event.preventDefault(); files.forEach(f => insertImage(f)); return; }
    if (textMode()) {
      event.preventDefault();
      document.execCommand("insertText", false, event.clipboardData.getData("text/plain"));
    }
  });
  const slidePoint = event => {
    const d = deckEl.getBoundingClientRect();
    return { left: (event.clientX - d.left) / d.width * 100, top: (event.clientY - d.top) / d.height * 100 };
  };
  deckEl.addEventListener("dragover", event => { if (editor.active && event.dataTransfer.types.includes("Files")) event.preventDefault(); });
  deckEl.addEventListener("drop", event => {
    if (!editor.active) return;
    const files = Array.from(event.dataTransfer.files || []).filter(f => f.type.startsWith("image/"));
    if (!files.length) return;
    event.preventDefault();
    const at = slidePoint(event);
    files.forEach((f, k) => insertImage(f, { left: at.left + 2 * k, top: at.top + 2 * k }));
  });

  // ---- keys while editing (deck.js hands them all over) -----------------------------------------
  function onKey(event) {
    const ctrl = event.ctrlKey || event.metaKey, key = event.key.toLowerCase();
    if (ctrl && key === "s") { event.preventDefault(); flushTyping(); save(); return; }
    if (event.target.closest && event.target.closest("#ed-toolbar, #ed-latex, #counter, .ed-menu")) {
      if (event.key === "Escape") { closeLatex(); if (event.target.blur) event.target.blur(); animItem = null; updateToolbar(); }
      return;
    }
    if (event.key === "Escape" && menu) { closeMenu(); return; }
    if (textMode()) {
      if (event.key === "Escape") { event.preventDefault(); flushTyping(); document.activeElement.blur(); animItem = null; drawOverlay(); updateToolbar(); }
      else if (ctrl && (key === "z" || key === "y")) { flushTyping(); }   // the browser's own undo inside text
      return;
    }
    if (ctrl && key === "z") { event.preventDefault(); if (event.shiftKey) redo(); else undo(); return; }
    if (ctrl && key === "y") { event.preventDefault(); redo(); return; }
    if (ctrl && key === "g") { event.preventDefault(); if (event.shiftKey) ungroupSelected(); else groupSelected(); return; }
    if (ctrl && key === "a") { event.preventDefault(); selectAll(); return; }
    if (selected) {
      if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); deleteSelected(); return; }
      if (event.key === "Escape") {
        event.preventDefault(); closeLatex();
        // Out of a group a level at a time, then nothing.
        const up = selection.length === 1 && containerOf(selected);
        if (up && up.classList.contains("box")) select(up); else deselect();
        return;
      }
      if (ctrl && key === "d") { event.preventDefault(); duplicateSelected(); return; }
      if (ctrl && key === "c") { event.preventDefault(); copySelected(); return; }
      if (ctrl && key === "x") { event.preventDefault(); copySelected(); deleteSelected(); return; }
      if (ctrl && key === "v") { event.preventDefault(); pasteElements(); return; }
      if (event.key.startsWith("Arrow")) {
        event.preventDefault();
        if (!selection.every(makeFree)) return;
        selection.forEach(el => {
          const c = containerOf(el).getBoundingClientRect(), d = deckEl.getBoundingClientRect();
          const step = (event.shiftKey ? 2 : 0.25) * d.width / c.width, r = rel(el);
          if (event.key === "ArrowLeft") r.left -= step;
          if (event.key === "ArrowRight") r.left += step;
          if (event.key === "ArrowUp") r.top -= step * c.width / c.height * d.height / d.width;
          if (event.key === "ArrowDown") r.top += step * c.width / c.height * d.height / d.width;
          setRect(el, r, !!el.style.height);
        });
        drawOverlay();
        commitSoon(500);
        return;
      }
      if (event.key === "Enter") {
        const root = selected.matches("[contenteditable='true']") ? selected : selected.querySelector("[contenteditable='true']");
        if (root && !isGroup(selected)) { event.preventDefault(); placeCaret(root); }
        else if (isGroup(selected)) { event.preventDefault(); const first = selected.querySelector(":scope > .box"); if (first) select(first); }
        return;
      }
      return;
    }
    if (ctrl && key === "v") { event.preventDefault(); if (clipboard && clipboard.kind === "slides") pasteSlides(); else pasteElements(); return; }
    if (event.key === "Escape") { event.preventDefault(); exit(); return; }
    D.onNavigationKey(event);   // nothing selected: arrows, PageUp/PageDown, F and E work as when presenting
  }
  function placeCaret(root, selectAll) {
    root.focus();
    const range = document.createRange();
    range.selectNodeContents(root);
    if (!selectAll) range.collapse(false);
    const sel = getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  }

  // ---- element operations ---------------------------------------------------------------------
  function deleteSelected() {
    if (!selection.length) return;
    const els = selection.slice();
    deselect();
    closeLatex();
    els.forEach(el => el.remove());
    commit();
  }
  function copySelected() {
    if (!selection.length) return;
    clipboard = { kind: "elements", html: selection.slice().sort(docOrder).map(el => D.slideSource(el)) };
    D.flash(`copied ${selection.length > 1 ? `${selection.length} elements` : ""}`.trim() || "copied", 900);
  }
  function placeElements(htmls, offset, into) {
    const slide = currentSlide();
    const target = into || slide;
    const added = [];
    htmls.flatMap(fromSource).forEach(node => {
      if (node.classList.contains("box") && offset) {
        node.style.left = `${round(parseFloat(node.style.left || 0) + offset)}%`;
        node.style.top = `${round(parseFloat(node.style.top || 0) + offset)}%`;
      }
      target.append(node);
      added.push(node);
    });
    prepare(slide);
    setSelection(added);
    commit();
  }
  function pasteElements() { if (clipboard && clipboard.kind === "elements") placeElements(clipboard.html, 2); }
  function duplicateSelected() {
    const els = selection.filter(el => !el.matches(".math"));
    if (!els.length) return;
    const copies = els.sort(docOrder).map(el => {
      const [node] = fromSource(D.slideSource(el));
      if (node.classList.contains("box")) {
        node.style.left = `${round(parseFloat(node.style.left || 0) + 2)}%`;
        node.style.top = `${round(parseFloat(node.style.top || 0) + 2)}%`;
      }
      el.after(node);
      return node;
    });
    prepare(currentSlide());
    setSelection(copies);
    commit();
  }
  function selectAll() {
    const slide = currentSlide();
    setSelection(Array.from(slide.children).filter(k => positionable(k) && !k.matches("#ed-overlay")));
  }
  function groupSelected() {
    const els = selection.slice();
    if (els.length < 2) { D.flash("select two or more elements (Shift-click) to group them", 2000); return; }
    if (!els.every(makeFree)) return;
    const container = containerOf(els[0]);
    if (!els.every(el => containerOf(el) === container)) { D.flash("only elements side by side (in the same group) can be grouped", 2500); return; }
    const c = container.getBoundingClientRect();
    const rects = els.map(el => el.getBoundingClientRect());
    const left = Math.min(...rects.map(r => r.left)), top = Math.min(...rects.map(r => r.top));
    const right = Math.max(...rects.map(r => r.right)), bottom = Math.max(...rects.map(r => r.bottom));
    const outer = { left, top, width: right - left, height: bottom - top };
    const group = h("div", { class: "box group" });
    setRect(group, pctIn(outer, c), true);
    const ordered = els.slice().sort(docOrder);
    ordered[0].before(group);
    ordered.forEach(el => {
      group.append(el);
      setRect(el, pctIn(rects[els.indexOf(el)], outer), !!el.style.height || needsHeight(el));
    });
    select(group);
    commit();
  }
  function ungroupSelected() {
    const group = selected;
    if (!group || selection.length !== 1 || !(isGroup(group) || group.classList.contains("group"))) return;
    freeze(group);
    const c = containerOf(group).getBoundingClientRect();
    const kids = Array.from(group.children).filter(k => k.classList.contains("box"));
    const rects = kids.map(k => k.getBoundingClientRect());
    kids.forEach((k, i) => {
      group.before(k);
      setRect(k, pctIn(rects[i], c), !!k.style.height);
    });
    // The group's own animation passes to elements that have none.
    ["step", "anim", "hide"].forEach(key => {
      if (group.dataset[key]) kids.forEach(k => { if (!k.dataset[key]) k.dataset[key] = group.dataset[key]; });
    });
    if (group.hasAttribute("data-dim-after")) kids.forEach(k => k.setAttribute("data-dim-after", ""));
    group.remove();
    setSelection(kids);
    commit();
  }
  function align(how) {
    const els = selection.slice();
    if (els.length < 2 || !els.every(makeFree)) return;
    const container = containerOf(els[0]);
    if (!els.every(el => containerOf(el) === container)) return;
    const rs = els.map(rel);
    const left = Math.min(...rs.map(r => r.left)), right = Math.max(...rs.map(r => r.left + r.width));
    const top = Math.min(...rs.map(r => r.top)), bottom = Math.max(...rs.map(r => r.top + r.height));
    els.forEach((el, i) => {
      const r = rs[i];
      if (how === "left") r.left = left;
      if (how === "right") r.left = right - r.width;
      if (how === "centre") r.left = (left + right) / 2 - r.width / 2;
      if (how === "top") r.top = top;
      if (how === "bottom") r.top = bottom - r.height;
      if (how === "middle") r.top = (top + bottom) / 2 - r.height / 2;
      setRect(el, r, !!el.style.height);
    });
    commit();
  }
  function arrange(direction) {
    const els = selection.filter(el => !el.matches(".math"));
    if (!els.length || !els.every(makeFree)) return;
    els.forEach(el => {
      const siblings = Array.from(containerOf(el).children).filter(k => k.classList.contains("box") && k !== el);
      const z = siblings.map(b => Number(b.style.zIndex) || 0);
      el.style.zIndex = String(direction > 0 ? Math.max(0, ...z) + 1 : Math.min(0, ...z) - 1);
    });
    commit();
  }
  function insertTextBox(at) {
    const left = at ? at.left : 30, top = at ? at.top : 42;
    const [node] = fromSource(`<div class="box text" style="left: ${round(left)}%; top: ${round(top)}%; width: 40%;"><p>Text</p></div>`);
    currentSlide().append(node);
    prepare(currentSlide());
    select(node);
    placeCaret(node, true);
    commit();
  }
  function upload(file) {
    return fetch("__upload", { method: "POST", headers: { "X-Filename": encodeURIComponent(file.name || "pasted.png") }, body: file })
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(json => json.path);
  }
  function insertImage(file, at) {
    const url = URL.createObjectURL(file);
    const probe = new Image();
    probe.onload = () => {
      const width = 40, height = width * 1.7778 * probe.naturalHeight / probe.naturalWidth;
      const left = at ? at.left - width / 2 : 30, top = at ? at.top - height / 2 : clamp(50 - height / 2, 2, 90);
      upload(file).then(path => {
        URL.revokeObjectURL(url);
        placeElements([`<div class="box image" style="left: ${round(left)}%; top: ${round(top)}%; width: ${width}%; height: ${round(height)}%;"><img src="${path}" alt=""></div>`], 0);
      }).catch(e => D.flash(`image not saved (${e.message}); is serve_talk.py running?`, 4000));
    };
    probe.src = url;
  }
  // Equations are either inline, inside text (where the caret is), or displayed: an element that
  // is only the equation, placed and sized like any other.
  function insertEquation(at) {
    if (textMode() && lastRange && !at) {
      const sel = getSelection();
      sel.removeAllRanges(); sel.addRange(lastRange);
      const span = h("span", { class: "math", "data-tex": "x", contenteditable: "false" });
      lastRange.deleteContents();
      lastRange.insertNode(span);
      D.renderMath(span.parentElement);
      commit();
      openLatex(span);
      return;
    }
    const left = at ? at.left : 30, top = at ? at.top : 44;
    const [node] = fromSource(`<div class="box math-display" data-tex="E = mc^2" style="left: ${round(left)}%; top: ${round(top)}%; width: 40%;"></div>`);
    currentSlide().append(node);
    node.setAttribute("contenteditable", "false");
    select(node);
    commit();
    openLatex(node);
  }

  // ---- the LaTeX popover ----------------------------------------------------------------------
  let latexTarget = null, latexBefore = null;
  const latexInput = h("textarea", { spellcheck: "false", rows: "3" });
  const latexPreview = h("div", { class: "ed-latex-preview" });
  const latexKind = h("span", { class: "ed-latex-kind" });
  const latex = h("div", { id: "ed-latex", hidden: true },
    h("div", { class: "ed-latex-head" }, h("b", { text: "LaTeX" }), latexKind,
      h("button", { type: "button", text: "Done", onclick: () => closeLatex() })),
    latexInput, latexPreview,
    h("div", { class: "ed-latex-note", text: "Ctrl+Enter or Esc closes." }));
  document.body.append(latex);
  const displayed = el => el.classList.contains("math-display");
  function renderLatex() {
    if (!latexTarget) return;
    const tex = latexInput.value;
    latexTarget.dataset.tex = tex;
    try { window.katex.render(tex, latexTarget, { displayMode: displayed(latexTarget), throwOnError: false }); } catch (e) { latexTarget.textContent = tex; }
    try { window.katex.render(tex, latexPreview, { displayMode: true, throwOnError: false }); } catch (e) { latexPreview.textContent = tex; }
    drawOverlay();
  }
  function openLatex(el) {
    latexTarget = el;
    latexBefore = el.dataset.tex;
    latexInput.value = el.dataset.tex || "";
    latexKind.textContent = displayed(el) ? "displayed equation" : "inline, in the text";
    latex.hidden = false;
    const r = el.getBoundingClientRect();
    const box = latex.getBoundingClientRect();
    latex.style.left = `${clamp(r.left, 8, innerWidth - box.width - 8)}px`;
    latex.style.top = `${r.bottom + 10 + box.height < innerHeight ? r.bottom + 10 : Math.max(8, r.top - box.height - 10)}px`;
    renderLatex();
    latexInput.focus();
    latexInput.select();
  }
  function closeLatex() {
    if (!latexTarget) return;
    const changed = latexTarget.dataset.tex !== latexBefore;
    latexTarget = null;
    latex.hidden = true;
    if (changed) commit();
  }
  latexInput.addEventListener("input", renderLatex);
  latexInput.addEventListener("keydown", event => {
    if (event.key === "Escape" || (event.key === "Enter" && (event.ctrlKey || event.metaKey))) { event.preventDefault(); closeLatex(); }
    event.stopPropagation();
  });

  // ---- text formatting ------------------------------------------------------------------------
  // The text a command applies to: the selection while typing, else all of the selected elements.
  function editableRoots(el) {
    if (!el) return [];
    if (el.matches("[contenteditable='true']")) return [el];
    return Array.from(el.querySelectorAll("[contenteditable='true']"));
  }
  function withText(fn) {
    if (textMode() && lastRange) {
      const root = document.activeElement;
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(lastRange);
      fn(root, false);
      commit();
      return;
    }
    selection.flatMap(editableRoots).forEach(root => {
      root.focus();
      const range = document.createRange();
      range.selectNodeContents(root);
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      fn(root, true);
    });
    getSelection().removeAllRanges();
    if (document.activeElement && deckEl.contains(document.activeElement)) document.activeElement.blur();
    commit();
  }
  function exec(command, value) {
    withText(() => {
      document.execCommand("styleWithCSS", false, command === "foreColor");
      document.execCommand(command, false, value);
      document.execCommand("styleWithCSS", false, false);
    });
  }
  // The block elements a change of alignment, size or style applies to.
  const BLOCKS = "h1, h2, h3, h4, p, li, figcaption, blockquote, td, th";
  function targetBlocks() {
    if (textMode() && lastRange) {
      const root = document.activeElement;
      const blocks = Array.from(root.querySelectorAll(BLOCKS)).filter(b => lastRange.intersectsNode(b));
      if (blocks.length) return blocks;
      const own = (lastRange.startContainer.nodeType === 1 ? lastRange.startContainer : lastRange.startContainer.parentElement).closest(BLOCKS);
      return [own && root.contains(own) ? own : root];
    }
    return selection.slice();
  }
  // Bullets or numbers on the paragraphs under the caret (in a text box); on items already in such
  // a list, back to paragraphs; in the other kind of list, the list changes kind.
  function toggleList(tag) {
    const root = document.activeElement;
    if (!textMode() || !lastRange) return;
    if (root.matches("ul, ol")) {
      if (root.tagName !== tag) {
        const list = document.createElement(tag);
        Array.from(root.attributes).forEach(a => list.setAttribute(a.name, a.value));
        list.append(...root.childNodes);
        root.replaceWith(list);
        if (selected === root) select(list);
        placeCaret(list);
      }
      commit();
      return;
    }
    const blocks = targetBlocks().filter(b => b !== root && b.parentElement && root.contains(b));
    if (!blocks.length) return;
    const items = blocks.filter(b => b.matches("li"));
    let caretIn = null;
    if (items.length === blocks.length && items.every(li => li.parentElement.tagName === tag)) {
      items.forEach(li => {
        const list = li.parentElement;
        const p = h("p");
        p.append(...li.childNodes);
        const after = Array.from(list.children).slice(Array.from(list.children).indexOf(li) + 1);
        if (after.length) {
          const rest = document.createElement(list.tagName);
          rest.append(...after);
          list.after(rest);
        }
        list.after(p);
        li.remove();
        if (!list.children.length) list.remove();
        caretIn = p;
      });
    } else if (items.length === blocks.length) {
      new Set(items.map(li => li.parentElement)).forEach(list => {
        const other = document.createElement(tag);
        other.append(...list.childNodes);
        list.replaceWith(other);
        caretIn = other.lastElementChild;
      });
    } else {
      const list = document.createElement(tag);
      blocks[0].before(list);
      blocks.forEach(b => {
        if (b.matches("li")) { list.append(b); return; }
        const li = h("li");
        li.append(...b.childNodes);
        if (!li.childNodes.length) li.append(h("br"));
        list.append(li);
        b.remove();
      });
      caretIn = list.lastElementChild;
    }
    if (caretIn) {
      const range = document.createRange();
      range.selectNodeContents(caretIn);
      range.collapse(false);
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
    commit();
    updateToolbar();
  }
  function setAlign(value) {
    targetBlocks().forEach(b => {
      b.style.textAlign = value;
      b.querySelectorAll("[style]").forEach(c => { c.style.textAlign = ""; });
    });
    commit();
    updateToolbar();
  }
  const sizeOf = el => round(parseFloat(getComputedStyle(el).fontSize) / D.unit * 10, 1);
  function setSize(size) {
    size = clamp(Number(size) || 10, 3, 200);
    const css = `calc(var(--u) * ${round(size / 10, 3)})`;
    if (textMode() && lastRange && !lastRange.collapsed) {
      const root = document.activeElement;
      const sel = getSelection(); sel.removeAllRanges(); sel.addRange(lastRange);
      document.execCommand("styleWithCSS", false, false);
      document.execCommand("fontSize", false, "7");
      root.querySelectorAll("font[size='7']").forEach(font => {
        const span = h("span");
        span.style.fontSize = css;
        span.append(...font.childNodes);
        font.replaceWith(span);
      });
    } else {
      targetBlocks().forEach(b => { b.style.fontSize = css; });
    }
    commit();
    updateToolbar();
  }
  function caretElement() {
    if (textMode() && lastRange) {
      const n = lastRange.startContainer;
      return n.nodeType === 1 ? n : n.parentElement;
    }
    return selected;
  }
  const STYLES = { title: ["H1", ""], heading: ["H2", ""], subheading: ["H3", ""], body: ["P", ""], note: ["P", "note"] };
  function setStyle(name) {
    const [tag, cls] = STYLES[name];
    targetBlocks().forEach(block => {
      const target = block.matches(".box.text") ? Array.from(block.children) : [block];
      target.forEach(b => {
        if (!b.matches("h1, h2, h3, h4, p")) return;
        let node = b;
        if (b.tagName !== tag) {
          node = document.createElement(tag);
          Array.from(b.attributes).forEach(a => node.setAttribute(a.name, a.value));
          node.append(...b.childNodes);
          b.replaceWith(node);
          if (selection.includes(b)) setSelection(selection.map(s => (s === b ? node : s)));
        }
        node.classList.remove("note", "byline");
        if (cls) node.classList.add(cls);
        node.style.fontSize = "";
      });
    });
    prepare(currentSlide());
    commit();
    updateToolbar();
  }
  function styleOf(el) {
    if (!el) return "";
    const b = el.closest(BLOCKS) || el;
    if (b.matches("h1")) return "title";
    if (b.matches("h2")) return "heading";
    if (b.matches("h3")) return "subheading";
    if (b.matches("p.note")) return "note";
    if (b.matches("p")) return "body";
    return "";
  }

  // ---- animation ------------------------------------------------------------------------------
  // What an animation setting applies to: while typing in a list or a text box, the item (or
  // paragraph) under the caret; else the selected elements.
  // The item is remembered as the caret enters it, so the toolbar's fields (which take the focus
  // from the text) still apply to it; Esc out of the text goes back to the whole element.
  let animItem = null;   // a list item or paragraph: under the caret, or picked from the menu or a badge
  function caretItem() {
    if (!lastRange || !selected) return null;
    const at = lastRange.startContainer.nodeType === 1 ? lastRange.startContainer : lastRange.startContainer.parentElement;
    const li = at.closest("li");
    if (li && selected.contains(li)) return li;
    const para = at.closest(BLOCKS);
    if (para && para !== selected && para.parentElement && para.parentElement.matches(".box.text")) return para;
    return null;
  }
  function animTargets() {
    if (animItem && animItem.isConnected && selected && selected.contains(animItem) && animItem !== selected) return [animItem];
    return selection.slice();
  }
  function describe(targets) {
    if (!targets.length) return "nothing selected";
    if (targets.length > 1) return `${targets.length} elements`;
    const el = targets[0];
    const nth = () => Array.from(el.parentElement.children).indexOf(el) + 1;
    if (el.matches("li")) return `list item ${nth()}`;
    if (el.parentElement && el.parentElement.matches(".box.text") && el.matches(BLOCKS)) return `paragraph ${nth()}`;
    if (isGroup(el)) return "group";
    if (el.matches("[data-tex]")) return "equation";
    if (el.matches(".box.image, img, .figure")) return "picture";
    if (el.matches(".embed")) return "viewer";
    if (el.matches("[data-chart], .chart")) return "chart";
    if (el.matches("ul, ol")) return "whole list";
    if (el.matches("h1, h2, h3")) return "heading";
    return "text";
  }
  function setAnim(field, value) {
    const targets = animTargets();
    if (!targets.length) return;
    targets.forEach(el => {
      if (field === "step") { if (Number(value) > 0) el.dataset.step = String(Math.round(Number(value))); else delete el.dataset.step; }
      if (field === "effect") { if (value) el.dataset.anim = value; else delete el.dataset.anim; }
      if (field === "dim") { if (value) el.setAttribute("data-dim-after", ""); else el.removeAttribute("data-dim-after"); }
      if (field === "hide") { if (Number(value) > 0) el.dataset.hide = String(Math.round(Number(value))); else delete el.dataset.hide; }
    });
    D.show(D.index, D.step, true);
    commit();
    updateToolbar();
  }
  // The parts "one by one" steps through: a list's items, a text box's paragraphs (and the items of
  // its lists), a group's elements top to bottom.
  function partsOf(el) {
    if (!el) return [];
    if (el.matches("ul, ol")) return Array.from(el.children).filter(k => k.matches("li"));
    if (el.matches(".box.text")) return Array.from(el.children).flatMap(k => (k.matches("ul, ol") ? Array.from(k.children) : [k]));
    if (isGroup(el)) return Array.from(el.children).filter(k => k.classList.contains("box")).sort((a, b) => rel(a).top - rel(b).top || rel(a).left - rel(b).left);
    return [];
  }
  function oneByOne(el) {
    const parts = partsOf(el);
    if (parts.length < 2) return;
    const first = Number(el.dataset.step) || 1;
    ["step", "anim", "hide"].forEach(key => { delete el.dataset[key]; });
    el.removeAttribute("data-dim-after");
    parts.forEach((part, k) => { part.dataset.step = String(first + k); });
    D.show(D.index, D.step, true);
    commit();
    updateToolbar();
  }
  function clearAnim(els) {
    els.forEach(el => [el, ...el.querySelectorAll("[data-step], [data-hide], [data-anim], [data-dim-after]")].forEach(n => {
      ["step", "anim", "hide"].forEach(key => { delete n.dataset[key]; });
      n.removeAttribute("data-dim-after");
    }));
    D.show(D.index, D.step, true);
    commit();
    updateToolbar();
  }

  // ---- right-click menu on the slide ------------------------------------------------------------
  deckEl.addEventListener("contextmenu", event => {
    if (!editor.active) return;
    event.preventDefault();
    const block = blockOf(event.target);
    if (block && !selection.includes(block)) select(block);
    if (!block) deselect();
    const item = event.target.closest("li") || (block && block.matches(".box.text") && event.target.closest(BLOCKS));
    const at = slidePoint(event);
    const one = selection.length === 1 ? selected : null;
    const many = selection.length > 1;
    const parent = one && containerOf(one);
    const animated = selection.some(el => el.matches("[data-step], [data-hide]") || el.querySelector("[data-step], [data-hide]"));
    const elements = selection.length ? [
      { label: "Cut", keys: "Ctrl+X", run: () => { copySelected(); deleteSelected(); } },
      { label: "Copy", keys: "Ctrl+C", run: copySelected },
      { label: "Paste", keys: "Ctrl+V", disabled: !clipboard || clipboard.kind !== "elements", run: pasteElements },
      { label: "Duplicate", keys: "Ctrl+D", run: duplicateSelected },
      { label: "Delete", keys: "Del", run: deleteSelected },
      "-",
      { label: "Edit LaTeX", hidden: !(one && one.matches("[data-tex]")), run: () => openLatex(one) },
      { label: "Edit text", hidden: !(one && !isGroup(one) && editableRoots(one).length), run: () => placeCaret(editableRoots(one)[0]) },
      "-",
      { label: `Animate this ${item && item.matches("li") ? "item" : "paragraph"} only`, hidden: !(item && one && one.contains(item) && item !== one),
        run: () => { animItem = item; updateToolbar(); D.flash(`the toolbar's animation now applies to ${describe([item])}`, 1800); } },
      { label: `Animate ${one && one.matches("ul, ol") ? "items" : one && isGroup(one) ? "elements" : "paragraphs"} one by one`, hidden: !(one && partsOf(one).length > 1), run: () => oneByOne(one) },
      { label: "Remove animation", hidden: !animated, run: () => clearAnim(selection.slice()) },
      "-",
      { label: "Group", keys: "Ctrl+G", hidden: !many, run: groupSelected },
      { label: "Ungroup", keys: "Ctrl+Shift+G", hidden: !(one && (isGroup(one) || one.classList.contains("group"))), run: ungroupSelected },
      { label: "Select the group", keys: "Esc", hidden: !(parent && parent.classList.contains("box")), run: () => select(parent) },
      "-",
      { label: "Align left edges", hidden: !many, run: () => align("left") },
      { label: "Align centres", hidden: !many, run: () => align("centre") },
      { label: "Align right edges", hidden: !many, run: () => align("right") },
      { label: "Align tops", hidden: !many, run: () => align("top") },
      { label: "Align middles", hidden: !many, run: () => align("middle") },
      { label: "Align bottoms", hidden: !many, run: () => align("bottom") },
      "-",
      { label: "Bring to front", hidden: !!(one && one.matches(".math")), run: () => arrange(1) },
      { label: "Send to back", hidden: !!(one && one.matches(".math")), run: () => arrange(-1) },
    ] : [];
    const slideItems = [
      { label: "Paste", keys: "Ctrl+V", hidden: selection.length > 0, disabled: !clipboard || clipboard.kind !== "elements", run: pasteElements },
      { label: "New text box here", hidden: selection.length > 0, run: () => insertTextBox(at) },
      { label: "New equation here", hidden: selection.length > 0, run: () => insertEquation(at) },
      { label: "Insert picture…", hidden: selection.length > 0, run: () => filePicker.click() },
      { label: "Select all", keys: "Ctrl+A", hidden: selection.length > 0, run: selectAll },
    ];
    openMenu(event.clientX, event.clientY, [...elements, ...slideItems]);
  });

  // ---- toolbar --------------------------------------------------------------------------------
  const ICON = {
    left: '<svg viewBox="0 0 16 16"><path d="M2 3h12M2 6.5h8M2 10h12M2 13.5h8"/></svg>',
    center: '<svg viewBox="0 0 16 16"><path d="M2 3h12M4 6.5h8M2 10h12M4 13.5h8"/></svg>',
    right: '<svg viewBox="0 0 16 16"><path d="M2 3h12M6 6.5h8M2 10h12M6 13.5h8"/></svg>',
    justify: '<svg viewBox="0 0 16 16"><path d="M2 3h12M2 6.5h12M2 10h12M2 13.5h12"/></svg>',
    bullets: '<svg viewBox="0 0 16 16"><circle cx="3" cy="4" r="1.2"/><circle cx="3" cy="8" r="1.2"/><circle cx="3" cy="12" r="1.2"/><path d="M6.5 4h7.5M6.5 8h7.5M6.5 12h7.5"/></svg>',
    numbers: '<svg viewBox="0 0 16 16"><text x="0.5" y="5.5" font-size="5">1</text><text x="0.5" y="13.5" font-size="5">2</text><path d="M6.5 4h7.5M6.5 8h7.5M6.5 12h7.5"/></svg>',
  };
  const SWATCHES = ["#e8ebf0", "#9aa4b2", "#e0a526", "#6aa7f0", "#d6452a", "#3ec58f", "#c58af0", "#000000"];
  const button = (attrs, content) => h("button", { type: "button", ...attrs, html: content });
  const sizeInput = h("input", { type: "number", class: "ed-size", min: "3", max: "200", step: "0.5", title: "text size (body text is 16.5)" });
  const styleSelect = h("select", { title: "paragraph style" },
    h("option", { value: "", text: "style" }),
    ...Object.keys(STYLES).map(name => h("option", { value: name, text: name })));
  const colourPicker = h("input", { type: "color", title: "any colour" });
  const stepInput = h("input", { type: "number", min: "0", step: "1", class: "ed-num", title: "appears at this step (0: with the slide)" });
  const hideInput = h("input", { type: "number", min: "0", step: "1", class: "ed-num", title: "gone from this step on (empty: stays)" });
  const effectSelect = h("select", { title: "how it appears" },
    ...[["", "fade"], ["appear", "appear"], ["rise", "rise"], ["left", "from left"], ["zoom", "zoom"]].map(([v, t]) => h("option", { value: v, text: t })));
  const dimInput = h("input", { type: "checkbox", title: "dims once the next step appears" });
  const animLabel = h("span", { class: "ed-anim-target", title: "what the animation settings apply to: click into a list item or paragraph to animate just that" });
  const oneByOneButton = button({ title: "give the items (paragraphs, grouped elements) successive steps", onclick: () => oneByOne(selected) }, "one by one");
  const undoButton = button({ title: "undo (Ctrl+Z)", onclick: undo }, "↶");
  const redoButton = button({ title: "redo (Ctrl+Y)", onclick: redo }, "↷");
  const stateLabel = h("span", { class: "ed-state" });
  const group = (label, ...children) => h("div", { class: "ed-group" }, label ? h("span", { class: "ed-label", text: label }) : null, ...children);
  const arrangeGroup = group("Arrange",
    button({ title: "group (Ctrl+G; select several with Shift-click)", "data-needs": "many", onclick: groupSelected }, "Group"),
    button({ title: "ungroup (Ctrl+Shift+G)", "data-needs": "group", onclick: ungroupSelected }, "Ungroup"),
    button({ title: "bring to front", "data-needs": "one", onclick: () => arrange(1) }, "Front"),
    button({ title: "send to back", "data-needs": "one", onclick: () => arrange(-1) }, "Back"),
    button({ title: "duplicate (Ctrl+D)", "data-needs": "one", onclick: duplicateSelected }, "Duplicate"),
    button({ title: "delete (Del)", "data-needs": "one", onclick: deleteSelected }, "Delete"));
  const toolbar = h("div", { id: "ed-toolbar", hidden: true },
    group("Insert",
      button({ title: "text box", onclick: () => insertTextBox() }, "Text"),
      button({ title: "picture from a file (or paste / drop one on the slide)", onclick: () => filePicker.click() }, "Image"),
      button({ title: "equation: inline at the caret while typing, else a displayed equation", onclick: () => insertEquation() }, "Equation")),
    group("Text", styleSelect,
      button({ title: "smaller", "data-size": "", onclick: () => setSize(Number(sizeInput.value || 16.5) / 1.1) }, "A−"), sizeInput,
      button({ title: "larger", "data-size": "", onclick: () => setSize(Number(sizeInput.value || 16.5) * 1.1) }, "A+"),
      button({ title: "bold (Ctrl+B)", "data-cmd": "bold", onclick: () => exec("bold") }, "<b>B</b>"),
      button({ title: "italic (Ctrl+I)", "data-cmd": "italic", onclick: () => exec("italic") }, "<i>I</i>"),
      button({ title: "underline (Ctrl+U)", "data-cmd": "underline", onclick: () => exec("underline") }, "<u>U</u>"),
      button({ title: "strike through", "data-cmd": "strikeThrough", onclick: () => exec("strikeThrough") }, "<s>S</s>"),
      h("span", { class: "ed-swatches" }, ...SWATCHES.map(c => {
        const b = button({ class: "ed-swatch", title: c, onclick: () => exec("foreColor", c) }, "");
        b.style.background = c;
        return b;
      }), colourPicker)),
    group("", ...["left", "center", "right", "justify"].map(a => button({ title: `align ${a}`, "data-align": a, onclick: () => setAlign(a) }, ICON[a])),
      button({ title: "bulleted list (while typing in a text box)", "data-list": "UL", onclick: () => toggleList("UL") }, ICON.bullets),
      button({ title: "numbered list (while typing in a text box)", "data-list": "OL", onclick: () => toggleList("OL") }, ICON.numbers)),
    group("Animate", animLabel,
      h("label", { title: "appears at this step (0: with the slide)" }, "step ", stepInput), effectSelect,
      h("label", { title: "dims once the next step appears" }, dimInput, " dim after"),
      h("label", { class: "ed-gone", title: "gone from this step on" }, "gone at ", hideInput),
      oneByOneButton),
    arrangeGroup,
    group("", undoButton, redoButton,
      button({ title: "save into index.html (Ctrl+S)", onclick: () => { flushTyping(); save(); } }, "Save"),
      stateLabel,
      button({ title: "stop editing (Esc with nothing selected)", class: "ed-done", onclick: () => exit() }, "Done")));
  const filePicker = h("input", { type: "file", accept: "image/*", multiple: true, hidden: true });
  filePicker.addEventListener("change", () => { Array.from(filePicker.files).forEach((f, k) => insertImage(f, k ? { left: 50 + 2 * k, top: 50 + 2 * k } : null)); filePicker.value = ""; });
  document.body.append(toolbar, filePicker);
  // Buttons must not take the focus (and with it the text selection) from the slide.
  toolbar.addEventListener("mousedown", event => { if (event.target.closest("button")) event.preventDefault(); });
  sizeInput.addEventListener("change", () => setSize(sizeInput.value));
  // Keys in the toolbar's fields stay there; Esc leaves the field (and goes back to the whole element).
  function fieldKey(event) {
    event.stopPropagation();
    if (event.key === "Escape") { event.preventDefault(); event.target.blur(); animItem = null; updateToolbar(); }
  }
  sizeInput.addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); setSize(sizeInput.value); } fieldKey(event); });
  styleSelect.addEventListener("change", () => { if (styleSelect.value) setStyle(styleSelect.value); });
  colourPicker.addEventListener("change", () => exec("foreColor", colourPicker.value));
  stepInput.addEventListener("change", () => setAnim("step", stepInput.value));
  hideInput.addEventListener("change", () => setAnim("hide", hideInput.value));
  effectSelect.addEventListener("change", () => setAnim("effect", effectSelect.value));
  dimInput.addEventListener("change", () => setAnim("dim", dimInput.checked));
  [stepInput, hideInput].forEach(input => input.addEventListener("keydown", fieldKey));
  function updateToolbar() {
    if (!editor.active) return;
    if (animItem && !(selected && selected.contains(animItem) && animItem.isConnected)) animItem = null;
    const typing = textMode();
    const caret = caretElement();
    const textual = typing || selection.some(el => editableRoots(el).length > 0);
    const sizeable = textual || selection.some(el => el.matches(".math-display"));
    styleSelect.disabled = !textual;
    sizeInput.disabled = !sizeable;
    toolbar.querySelectorAll("[data-size]").forEach(b => { b.disabled = !sizeable; });
    toolbar.querySelectorAll("[data-cmd], [data-align], .ed-swatch").forEach(b => { b.disabled = !textual; });
    colourPicker.disabled = !textual;
    toolbar.querySelectorAll("[data-list]").forEach(b => {
      b.disabled = !typing || !document.activeElement.matches(".box.text, ul, ol");
    });
    if (sizeable && caret) {
      sizeInput.value = sizeOf(caret);
      styleSelect.value = textual ? styleOf(caret) : "";
      ["bold", "italic", "underline", "strikeThrough"].forEach(cmd => {
        const b = toolbar.querySelector(`[data-cmd='${cmd}']`);
        let on = false;
        try { on = typing && document.queryCommandState(cmd); } catch (e) { /* not supported */ }
        b.setAttribute("aria-pressed", String(on));
      });
      const align = getComputedStyle(caret.closest(BLOCKS) || caret).textAlign;
      toolbar.querySelectorAll("[data-align]").forEach(b => b.setAttribute("aria-pressed", String(textual && (align === b.dataset.align || (align === "start" && b.dataset.align === "left")))));
    } else {
      sizeInput.value = ""; styleSelect.value = "";
      toolbar.querySelectorAll("[data-cmd], [data-align]").forEach(b => b.setAttribute("aria-pressed", "false"));
    }
    const targets = animTargets();
    const anim = targets[targets.length - 1];
    animLabel.textContent = describe(targets);
    [stepInput, hideInput, effectSelect, dimInput].forEach(c => { c.disabled = !anim; });
    stepInput.value = anim ? (anim.dataset.step || 0) : "";
    hideInput.value = anim ? (anim.dataset.hide || "") : "";
    effectSelect.value = anim ? (anim.dataset.anim || "") : "";
    dimInput.checked = !!(anim && anim.hasAttribute("data-dim-after"));
    oneByOneButton.disabled = !(selection.length === 1 && partsOf(selected).length > 1);
    arrangeGroup.querySelectorAll("[data-needs]").forEach(b => {
      const need = b.dataset.needs;
      b.disabled = need === "many" ? selection.length < 2
        : need === "group" ? !(selection.length === 1 && (isGroup(selected) || selected.classList.contains("group")))
        : !selection.length;
    });
    updateState();
  }
  function updateState() {
    const unsaved = LOCAL && dirty();
    stateLabel.textContent = unsaved ? "unsaved" : "saved";
    stateLabel.classList.toggle("unsaved", unsaved);
    sorterState.classList.toggle("unsaved", unsaved);
    undoButton.disabled = history.at <= 0;
    redoButton.disabled = history.at >= history.stack.length - 1;
  }

  // ---- overlay: selection frames, handles, step badges, snap guides ----------------------------
  const overlay = h("div", { id: "ed-overlay" });
  deckEl.append(overlay);
  const frame = h("div", { class: "ed-frame", hidden: true },
    ...["n", "e", "s", "w"].map(side => h("div", { class: `ed-edge ed-edge-${side}`, title: "drag to move" })),
    ...["nw", "n", "ne", "e", "se", "s", "sw", "w"].map(dir => h("div", { class: `ed-handle ed-${dir}`, "data-dir": dir })));
  overlay.append(frame);
  overlay.addEventListener("pointerdown", event => {
    if (!selection.length || event.button !== 0) return;
    const handle = event.target.closest(".ed-handle");
    if (textMode()) { flushTyping(); document.activeElement.blur(); }
    animItem = null;
    updateToolbar();
    if (handle && selected) startDrag(event, [selected], handle.dataset.dir);
    else if (event.target.closest(".ed-edge")) startDrag(event, selection.slice(), null);
  });
  let overlayQueued = false;
  function drawOverlay(guides) {
    if (guides) { paintOverlay(guides); return; }
    if (overlayQueued) return;
    overlayQueued = true;
    requestAnimationFrame(() => { overlayQueued = false; paintOverlay([]); });
  }
  function paintOverlay(guides) {
    overlay.querySelectorAll(".ed-badge, .ed-guide, .ed-multi").forEach(n => n.remove());
    const slide = currentSlide();
    const on = editor.active && !document.fullscreenElement;
    overlay.hidden = !on;
    if (!on) return;
    const d = deckEl.getBoundingClientRect();
    const place = (node, r) => {
      node.style.left = `${r.left - d.left}px`; node.style.top = `${r.top - d.top}px`;
      node.style.width = `${r.width}px`; node.style.height = `${r.height}px`;
    };
    const shown = selection.filter(el => slide && slide.contains(el) && el.isConnected);
    frame.hidden = shown.length !== 1;
    if (shown.length === 1) {
      place(frame, shown[0].getBoundingClientRect());
      frame.classList.toggle("inline", shown[0].matches(".math"));
      frame.classList.toggle("group", isGroup(shown[0]));
    } else {
      shown.forEach(el => {
        const outline = h("div", { class: "ed-frame ed-multi" },
          ...["n", "e", "s", "w"].map(side => h("div", { class: `ed-edge ed-edge-${side}`, title: "drag to move" })));
        place(outline, el.getBoundingClientRect());
        overlay.append(outline);
      });
    }
    if (slide) slide.querySelectorAll("[data-step], [data-hide]").forEach(el => {
      const r = el.getBoundingClientRect();
      if (!r.width && !r.height) return;
      const text = `${el.dataset.step || 0}${el.dataset.hide ? `→${el.dataset.hide}` : ""}`;
      const badge = h("div", { class: "ed-badge", text, title: "appears at step (→ gone at step); click to pick it for the toolbar's animation settings" });
      badge.style.left = `${r.left - d.left - 22}px`;
      badge.style.top = `${r.top - d.top + 1}px`;
      badge.addEventListener("pointerdown", event => {
        event.preventDefault(); event.stopPropagation();
        const block = blockOf(el);
        select(block === el || !block ? el : block);
        animItem = el !== selected ? el : null;
        updateToolbar();
      });
      overlay.append(badge);
    });
    guides.forEach(g => {
      const line = h("div", { class: "ed-guide" });
      const c = g.c;
      if (g.x != null) {
        line.style.left = `${c.left - d.left + g.x / 100 * c.width}px`; line.style.top = `${c.top - d.top}px`;
        line.style.height = `${c.height}px`; line.style.width = "1px";
      } else {
        line.style.top = `${c.top - d.top + g.y / 100 * c.height}px`; line.style.left = `${c.left - d.left}px`;
        line.style.width = `${c.width}px`; line.style.height = "1px";
      }
      overlay.append(line);
    });
  }
  window.addEventListener("resize", () => drawOverlay());

  D.onChange((i, s, why) => {
    if (why === "layout") { drawOverlay(); return; }
    if (selected && !currentSlide().contains(selected)) { closeLatex(); deselect(); }
    if (!selectedSlides.has(i)) { selectedSlides = new Set([i]); anchor = i; }
    markSorter();
    scrollToCurrent();
    drawOverlay();
  });

  // ---- start ------------------------------------------------------------------------------------
  history.stack = [snapshot()];
  history.at = 0;
  history.saved = history.stack[0].join("\u0000");
  renderSorter();
  setTimeout(renderSorter, 2000);   // again once the charts (fetched) are drawn
  layoutChanged();
  updateState();
})();

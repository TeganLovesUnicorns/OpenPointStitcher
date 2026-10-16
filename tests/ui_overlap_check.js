// Finds text and controls that land on top of each other or are cut off. Evaluated in the bench page by
// tests/test_browser_bench.py. Returns { items, overlaps, covered, clipped, cutOff }; each entry has rects, boxes in
// page pixels, so a caller can outline them in a screenshot.
() => {
  const MIN_SIDE = 3;                       // px: an overlap thinner than this in either direction is antialiasing, not a collision
  const CUT_FRACTION = 0.12;                // share of a piece of text or a control hidden by its container that counts as cut off
  const SKIP_TAGS = new Set(['SCRIPT', 'STYLE', 'CANVAS', 'LINK', 'META', 'HEAD', 'TITLE', 'NOSCRIPT', 'TEXTAREA', 'OPTION']);
  const CONTROL_TAGS = new Set(['BUTTON', 'SELECT', 'INPUT']);
  const boxOf = r => ({ x0: r.left, y0: r.top, x1: r.right, y1: r.bottom });
  const areaOf = b => Math.max(0, b.x1 - b.x0) * Math.max(0, b.y1 - b.y0);
  const overlapOf = (a, b) => ({ w: Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0), h: Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0) });
  const rectOut = b => ({ x: Math.round(b.x0), y: Math.round(b.y0), w: Math.round(b.x1 - b.x0), h: Math.round(b.y1 - b.y0) });

  const shown = el => {
    for (let p = el; p && p.nodeType === 1; p = p.parentElement) {
      const s = getComputedStyle(p);
      if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) return false;
      if (p.hasAttribute('data-overlap-ok')) return false;
    }
    return true;
  };
  const clipsOf = (el, includeSelf) => {     // every ancestor that clips its contents, nearest first (text: its own element clips it too)
    const out = [];
    for (let p = includeSelf ? el : el.parentElement; p && p !== document.documentElement; p = p.parentElement) {
      const s = getComputedStyle(p);
      if (s.overflowX !== 'visible' || s.overflowY !== 'visible') {
        const r = p.getBoundingClientRect();
        out.push({ el: p, scrolls: /auto|scroll/.test(s.overflowX + s.overflowY), ellipsis: s.textOverflow === 'ellipsis',
                   box: { x0: s.overflowX !== 'visible' ? r.left : -1e9, x1: s.overflowX !== 'visible' ? r.right : 1e9,
                          y0: s.overflowY !== 'visible' ? r.top : -1e9, y1: s.overflowY !== 'visible' ? r.bottom : 1e9 } });
      }
    }
    return out;
  };
  const cutTo = (box, clips) => {
    let c = Object.assign({}, box);
    for (const k of [{ box: { x0: 0, y0: 0, x1: innerWidth, y1: innerHeight } }, ...clips]) {
      c = { x0: Math.max(c.x0, k.box.x0), y0: Math.max(c.y0, k.box.y0), x1: Math.min(c.x1, k.box.x1), y1: Math.min(c.y1, k.box.y1) };
    }
    return c;
  };
  const describe = (el, text) => {
    const id = el.id ? '#' + el.id : '';
    const cls = (typeof el.className === 'string' ? el.className : '').split(/\s+/).filter(Boolean).slice(0, 2).map(x => '.' + x).join('');
    const role = el.getAttribute('role') ? '[' + el.getAttribute('role') + (el.getAttribute('aria-label') ? ' ' + el.getAttribute('aria-label') : '') + ']' : '';
    return el.tagName.toLowerCase() + id + cls + role + (text ? ' "' + text.replace(/\s+/g, ' ').trim().slice(0, 40) + '"' : '');
  };
  // The two thumbs of one range slider are allowed to sit on each other (a narrow band on a long track): not a collision.
  const sameSliderThumbs = (a, b) => a.getAttribute('role') === 'slider' && b.getAttribute('role') === 'slider' && a.parentElement === b.parentElement;

  // With a menu, dialog or popup open, only it is judged: what it covers is meant to be covered.
  const popups = [...document.querySelectorAll('.startOverlay, [role=dialog], .mantine-Popover-dropdown, .ctxmenu')]
    .filter(e => shown(e) && e.getBoundingClientRect().width > 0);
  popups.sort((a, b) => (parseInt(getComputedStyle(a).zIndex) || 0) - (parseInt(getComputedStyle(b).zIndex) || 0));
  const scope = popups.length ? popups[popups.length - 1] : null;

  // Every piece of text (its real drawn line boxes) and every control (its whole box).
  const items = [];
  const consider = (el, full, text, kind) => {
    if (scope && !scope.contains(el)) return;
    const clips = clipsOf(el, kind === 'text');
    const visible = cutTo(full, clips);
    const seen = areaOf(visible) / Math.max(areaOf(full), 1);
    items.push({ el, full, box: visible, seen, clips, text, kind });
  };
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    if (n.nodeType === 3) {
      const el = n.parentElement;
      if (!el || SKIP_TAGS.has(el.tagName) || !n.textContent.trim() || !shown(el)) continue;
      const range = document.createRange();
      range.selectNodeContents(n);
      for (const r of range.getClientRects()) if (r.width >= 1 && r.height >= 1) consider(el, boxOf(r), n.textContent, 'text');
    } else if (CONTROL_TAGS.has(n.tagName) && !(n.tagName === 'INPUT' && ['hidden', 'range', 'checkbox', 'radio'].includes(n.type)) && shown(n)) {
      consider(n, boxOf(n.getBoundingClientRect()), n.value || n.textContent || '', 'control');
    }
  }
  // Mantine's sliders and checkboxes are divs, not inputs: count each as the control it looks like.
  for (const el of document.querySelectorAll('.mantine-Slider-root, .mantine-RangeSlider-root, .mantine-Checkbox-inner, [role=slider]')) {
    if (shown(el)) consider(el, boxOf(el.getBoundingClientRect()), '', 'control');
  }
  const drawn = items.filter(it => it.seen > 0);

  // Is `el` actually seen at this point, or is something opaque, not part of it, painted above it? Text under a
  // sticky header, a dropdown or a dialog is hidden, not jumbled. (elementsFromPoint skips pointer-events:none
  // overlays, and those are always drawn above, so they count as seen.)
  const opaque = e => {
    const m = getComputedStyle(e).backgroundColor.match(/rgba?\(([^)]+)\)/);
    if (!m) return false;
    const p = m[1].split(',').map(parseFloat);
    return (p.length > 3 ? p[3] : 1) >= 0.9;
  };
  const seenAt = (el, stack) => {
    if (getComputedStyle(el).pointerEvents === 'none') return true;
    const at = stack.findIndex(e => e === el || el.contains(e));
    if (at < 0) return true;
    return !stack.slice(0, at).some(e => opaque(e) && !e.contains(el) && !el.contains(e));
  };

  const overlaps = [];
  for (let i = 0; i < drawn.length; i++) {
    for (let j = i + 1; j < drawn.length; j++) {
      const a = drawn[i], b = drawn[j];
      if (a.el === b.el || a.el.contains(b.el) || b.el.contains(a.el) || sameSliderThumbs(a.el, b.el)) continue;
      const o = overlapOf(a.box, b.box);
      if (o.w >= MIN_SIDE && o.h >= MIN_SIDE) {
        const x0 = Math.max(a.box.x0, b.box.x0), y0 = Math.max(a.box.y0, b.box.y0);
        const stack = document.elementsFromPoint(x0 + o.w / 2, y0 + o.h / 2);
        if (!(seenAt(a.el, stack) && seenAt(b.el, stack))) continue;
        overlaps.push({ a: describe(a.el, a.text), b: describe(b.el, b.text), w: Math.round(o.w), h: Math.round(o.h),
                        rects: [rectOut({ x0, y0, x1: x0 + o.w, y1: y0 + o.h })] });
      }
    }
  }

  // A control with something else on top of its middle, where it would be clicked.
  const covered = [];
  for (const it of drawn) {
    if (it.kind !== 'control') continue;
    const cx = (it.box.x0 + it.box.x1) / 2, cy = (it.box.y0 + it.box.y1) / 2;
    const top = document.elementsFromPoint(cx, cy).find(e => e.tagName !== 'HTML' && e.tagName !== 'BODY');
    const stuck = e => { for (let p = e; p; p = p.parentElement) if (getComputedStyle(p).position === 'sticky') return true; return false; };
    if (top && top !== it.el && !it.el.contains(top) && !top.contains(it.el) && !stuck(top) && !sameSliderThumbs(it.el, top)) {   // a sticky header covering scrolled rows is scrolling
      covered.push({ control: describe(it.el, it.text), coveredBy: describe(top, top.textContent || ''), rects: [rectOut(it.box)] });
    }
  }

  // Text or a control wider than the box it sits in, cut off because of it.
  const clipped = [];
  const seenEl = new Set();
  for (const it of drawn) {
    const el = it.el;
    if (seenEl.has(el)) continue;
    seenEl.add(el);
    if (el.scrollWidth > el.clientWidth + 1 && el.clientWidth > 0) {
      const s = getComputedStyle(el);
      if ((s.overflowX === 'hidden' || s.overflowX === 'clip') && s.textOverflow !== 'ellipsis') {
        clipped.push({ el: describe(el, it.text), cut: el.scrollWidth - el.clientWidth, why: 'text cut off with no ellipsis', rects: [rectOut(it.box)] });
      } else if (s.overflowX === 'visible' && it.kind === 'control') {
        clipped.push({ el: describe(el, it.text), cut: el.scrollWidth - el.clientWidth, why: 'text wider than its button', rects: [rectOut(it.box)] });
      }
    }
  }

  // A control or line of text partly or wholly hidden by a container that clips without scrolling
  // (a scroll list showing half a row is normal; a toolbar that hides its buttons is not).
  const cutOff = [];
  const cutSeen = new Set();
  for (const it of items) {
    const near = it.clips[0];
    const insideEllipsis = it.clips.some(c => c.ellipsis);
    if (it.seen >= 1 - CUT_FRACTION || insideEllipsis) continue;
    if (near && near.scrolls) continue;
    if (it.clips.some(c => c.scrolls)) continue;
    const key = it.el;
    if (cutSeen.has(key)) continue;
    cutSeen.add(key);
    cutOff.push({ el: describe(it.el, it.text), hidden: Math.round((1 - it.seen) * 100), by: near ? describe(near.el, '') : 'the window',
                  rects: [rectOut(it.full)] });
  }
  return { items: items.length, overlaps, covered, clipped, cutOff };
}

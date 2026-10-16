import { useRef } from "react";
import { Box } from "@mantine/core";

// Two-thumb range slider for one clip axis, vertical (higher value up) or horizontal (higher value right).
// Thumbs never come closer than minGap. Dragging the band moves the whole slice; arrow keys step it by its own
// thickness (Shift: 1 unit). Keep role="slider" and aria-value* on each thumb, lower first: the browser tests find
// thumbs by them.
const THUMB = 14;

export default function ClipRangeTrack({
  min, max, value, onChange, onChangeEnd, vertical = true, length = 160, dimmed = false, format = (v) => String(v),
}) {
  // Read at use time: Settings, Appearance, Slicing can change them while the view is open. Values are cm.
  const minGap = () => +(window.clipMinGapCm ?? 5);
  const stepBy = (sliceCm) => (window.clipStepCm === "slice" || window.clipStepCm == null ? sliceCm : +window.clipStepCm);
  const [lo, hi] = value;
  const span = max - min || 1;
  const inner = Math.max(1, length - THUMB);           // travel of a thumb's centre
  const latest = useRef(value);
  latest.current = value;
  const grab = useRef(null);                           // {mode: "lo" | "hi" | "band", offset}
  const root = useRef(null);

  const toPx = (v) => THUMB / 2 + (vertical ? 1 - (v - min) / span : (v - min) / span) * inner;
  const toValue = (px) => {
    const f = Math.min(1, Math.max(0, (px - THUMB / 2) / inner));
    return Math.round(min + (vertical ? 1 - f : f) * span);
  };
  const pointerPx = (e) => {
    const r = root.current.getBoundingClientRect();
    return vertical ? e.clientY - r.top : e.clientX - r.left;
  };
  const clampSlice = (a, b) => {
    const w = b - a;
    if (a < min) return [min, min + w];
    if (b > max) return [max - w, max];
    return [a, b];
  };
  const place = (mode, v, offset = 0) => {
    const [curLo, curHi] = latest.current;
    if (mode === "band") return clampSlice(v - offset, v - offset + (curHi - curLo));
    if (mode === "lo") return [Math.max(min, Math.min(v, curHi - minGap())), curHi];
    return [curLo, Math.min(max, Math.max(v, curLo + minGap()))];
  };

  const onPointerDown = (e) => {
    const px = pointerPx(e), v = toValue(px);
    const [curLo, curHi] = latest.current;
    const dLo = Math.abs(px - toPx(curLo)), dHi = Math.abs(px - toPx(curHi));
    let mode;
    const touching = Math.abs(toPx(curHi) - toPx(curLo)) < THUMB;
    if (touching && Math.min(dLo, dHi) <= THUMB) mode = "either";       // decided by the first move's direction
    else if (Math.min(dLo, dHi) <= THUMB / 2 + 2) mode = dLo <= dHi ? "lo" : "hi";
    else if (v > curLo && v < curHi) mode = "band";
    else mode = Math.abs(v - curLo) <= Math.abs(v - curHi) ? "lo" : "hi";
    grab.current = { mode, offset: v - curLo };
    root.current.setPointerCapture(e.pointerId);
    if (mode === "lo" || mode === "hi") onChange(place(mode, v));
    e.preventDefault();
  };
  const onPointerMove = (e) => {
    if (!grab.current) return;
    const v = toValue(pointerPx(e));
    if (grab.current.mode === "either") {
      const [curLo, curHi] = latest.current;
      if (v === Math.round((curLo + curHi) / 2)) return;
      grab.current.mode = v > (curLo + curHi) / 2 ? "hi" : "lo";
    }
    onChange(place(grab.current.mode, v, grab.current.offset));
  };
  const onPointerUp = (e) => {
    if (!grab.current) return;
    grab.current = null;
    root.current.releasePointerCapture(e.pointerId);
    if (onChangeEnd) onChangeEnd(latest.current);
  };
  // Arrows here step the slice and stop there, so they never also nudge a scan (the page listens on window).
  const onKeyDown = (e) => {
    const up = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1 }[e.key];
    if (!up) return;
    e.preventDefault();
    e.stopPropagation();
    const [curLo, curHi] = latest.current;
    const step = e.shiftKey ? 1 : Math.max(1, stepBy(curHi - curLo));
    onChange(clampSlice(curLo + up * step, curHi + up * step));
    if (onChangeEnd) onChangeEnd(latest.current);
  };

  const along = vertical ? "top" : "left";
  const across = vertical ? "left" : "top";
  const bandStart = Math.min(toPx(lo), toPx(hi)), bandLen = Math.abs(toPx(hi) - toPx(lo));
  return (
    <Box
      ref={root}
      tabIndex={0}
      title="Drag an end to change the slice, drag between the ends to move it, or use the arrow keys to step through"
      onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onKeyDown={onKeyDown}
      style={{
        position: "relative", [vertical ? "width" : "height"]: 20, [vertical ? "height" : "width"]: length,
        cursor: "pointer", opacity: dimmed ? 0.5 : 1, touchAction: "none", userSelect: "none", outline: "none",
      }}
    >
      <Box style={{
        position: "absolute", [across]: "50%", [along]: THUMB / 2, [vertical ? "bottom" : "right"]: THUMB / 2,
        [vertical ? "width" : "height"]: 4, transform: vertical ? "translateX(-50%)" : "translateY(-50%)",
        borderRadius: 2, background: "var(--mantine-color-dark-5)",
      }} />
      <Box className="clipBand" style={{
        position: "absolute", [across]: "50%", [along]: bandStart, [vertical ? "height" : "width"]: bandLen,
        [vertical ? "width" : "height"]: 8, transform: vertical ? "translateX(-50%)" : "translateY(-50%)",
        borderRadius: 3, background: "var(--mantine-color-bench-5)", cursor: "grab",
      }} />
      {[["lo", lo], ["hi", hi]].map(([id, v]) => (
        <Box
          key={id}
          role="slider"
          aria-orientation={vertical ? "vertical" : "horizontal"}
          aria-label={id === "lo" ? "lower edge" : "upper edge"}
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={v}
          aria-valuetext={format(v)}
          title={format(v)}
          style={{
            position: "absolute", [across]: "50%", [along]: toPx(v) - THUMB / 2, width: THUMB, height: THUMB,
            transform: vertical ? "translateX(-50%)" : "translateY(-50%)", borderRadius: "50%",
            background: "var(--mantine-color-bench-5)", border: "2px solid var(--mantine-color-dark-8)",
            boxSizing: "border-box",
          }}
        />
      ))}
    </Box>
  );
}

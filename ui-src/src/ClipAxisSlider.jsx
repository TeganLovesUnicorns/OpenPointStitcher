import { useEffect, useRef, useState } from "react";
import { Checkbox, Text } from "@mantine/core";
import ClipRangeTrack from "./ClipRangeTrack.jsx";

// About two thirds of the view's edge, less room for the box and the readout.
const EDGE_SHARE = 0.66, CHROME_PX = 90, MIN_LENGTH = 120;
function useEdgeLength(vertical) {
  const measure = () => {
    const st = document.getElementById("stage");
    const edge = st ? (vertical ? st.clientHeight : st.clientWidth) : 0;
    return Math.max(MIN_LENGTH, Math.round(edge * EDGE_SHARE) - CHROME_PX);
  };
  const [len, setLen] = useState(measure);
  useEffect(() => {
    const st = document.getElementById("stage");
    if (!st || typeof ResizeObserver === "undefined") return undefined;
    const ro = new ResizeObserver(() => setLen(measure()));
    ro.observe(st);
    return () => ro.disconnect();
  }, [vertical]);
  return len;
}

// One clip axis (slider pair) per instance, mounted on a viewport edge; a view onto the shared clip global.
export default function ClipAxisSlider({ axis, orientation, style = {} }) {
  const length = useEdgeLength(orientation === "vertical");
  const [on, setOn] = useState(() => (typeof clip !== "undefined" ? clip[axis].on : false));
  const [range, setRange] = useState(() => {
    if (typeof clip === "undefined") return [0, 100];
    const c = clip[axis];
    return [c.lo ?? 0, c.hi ?? 100];
  });
  const [bounds, setBounds] = useState(() => {
    if (typeof clip === "undefined") return [0, 100];
    const c = clip[axis];
    return [c.rangeMin ?? 0, c.rangeMax ?? 100];
  });
  // Don't fight an in-progress drag: the poll below must not overwrite what the drag just set, and
  // updateClipRange() can move rangeMin/rangeMax as scans stream in.
  const draggingRef = useRef(false);

  useEffect(() => {
    let raf;
    const tick = () => {
      if (!draggingRef.current && typeof clip !== "undefined") {
        const c = clip[axis];
        setOn(c.on);
        // Only a CHANGED value may replace the state: a fresh [lo, hi] array
        // every frame is a new reference, i.e. a re-render 60 times a second.
        if (c.lo != null && c.hi != null) setRange((p) => (p[0] === c.lo && p[1] === c.hi ? p : [c.lo, c.hi]));
        if (c.rangeMin != null && c.rangeMax != null) setBounds((p) => (p[0] === c.rangeMin && p[1] === c.rangeMax ? p : [c.rangeMin, c.rangeMax]));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [axis]);

  const writeOn = (checked) => {
    if (typeof clip === "undefined") return;
    clip[axis].on = checked;
    setOn(checked);
    if (typeof updateClipToggleBtnVisual === "function") updateClipToggleBtnVisual();
    if (typeof draw === "function") draw();
    // A clicked checkbox keeps focus and swallows every hotkey, so give focus back.
    if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
  };

  const writeRange = (v) => {
    if (typeof clip === "undefined") return;
    clip[axis].lo = v[0];
    clip[axis].hi = v[1];
    setRange(v);
    if (typeof draw === "function") draw();
  };

  const toMetres = (cm) => (cm / 100).toFixed(2);

  return (
    <div style={{ position: "absolute", zIndex: 6, ...style }}>
      <div
        style={{
          display: "flex",
          flexDirection: orientation === "vertical" ? "column" : "row",
          alignItems: "center",
          gap: 6,
        }}
      >
        <Checkbox
          checked={on}
          onChange={(e) => writeOn(e.currentTarget.checked)}
          size="xs"
          label={axis.toUpperCase()}
          title={`Clip along ${axis.toUpperCase()} on or off (shortcut: ${axis.toUpperCase()})`}
          styles={{ label: { color: "var(--ink, #e6e9ee)", fontSize: 11 } }}
        />
        <ClipRangeTrack
          min={bounds[0]}
          max={bounds[1]}
          value={range}
          vertical={orientation === "vertical"}
          length={length}
          onChange={(v) => {
            draggingRef.current = true;
            writeRange(v);
          }}
          onChangeEnd={() => {
            draggingRef.current = false;
          }}
          dimmed={!on}
          format={(v) => `${toMetres(v)} m`}
        />
        <Text size="xs" c="dimmed" ff="var(--sans, sans-serif)">
          {toMetres(range[0])} to {toMetres(range[1])} m
        </Text>
      </div>
    </div>
  );
}

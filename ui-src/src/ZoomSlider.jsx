import { useEffect, useRef, useState } from "react";
import { Slider } from "@mantine/core";

// On-screen zoom, docked on the stage. Polls the page's CAM_S_MIN/MAX: that script runs after this bundle, and
// reading them once would fall back to 2-160 for good.
const FALLBACK_MIN_S = 2, FALLBACK_MAX_S = 160;

export default function ZoomSlider() {
  const [value, setValue] = useState(() => (typeof cam !== "undefined" ? cam.s : 14));
  const [range, setRange] = useState([FALLBACK_MIN_S, FALLBACK_MAX_S]);
  // While a drag is in progress the poll below must not overwrite it. Reset on onChangeEnd, so it covers a drag of
  // any length.
  const draggingRef = useRef(false);

  useEffect(() => {
    let raf;
    const tick = () => {
      if (!draggingRef.current && typeof cam !== "undefined" && cam.s !== undefined) {
        setValue(cam.s);
      }
      if (typeof CAM_S_MIN !== "undefined" && typeof CAM_S_MAX !== "undefined") {
        setRange(([lo, hi]) => (lo === CAM_S_MIN && hi === CAM_S_MAX ? [lo, hi] : [CAM_S_MIN, CAM_S_MAX]));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  const [MIN_S, MAX_S] = range;

  // Zoom about the viewport middle via the shared zoomTo(); never write cam directly.
  const applyZoom = (v) => {
    if (typeof zoomTo !== "function" || typeof cv === "undefined") return;
    zoomTo(v, cv.width / 2, cv.height / 2);
  };

  return (
    <div style={{ width: 200 }}>
      <Slider
        min={MIN_S}
        max={MAX_S}
        step={0.5}
        value={value}
        label={(v) => `${v.toFixed(0)} px/m`}
        onChange={(v) => {
          draggingRef.current = true;
          setValue(v);
          applyZoom(v);
        }}
        onChangeEnd={() => {
          draggingRef.current = false;
        }}
        size="sm"
      />
    </div>
  );
}

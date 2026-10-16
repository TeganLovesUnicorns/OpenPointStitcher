import { useEffect, useState } from "react";
import { Text } from "@mantine/core";

// Scale bar from the live cam.s (px per metre, read only). Orthographic camera, so it is right in every view.
const ROUND_STEPS_M = [5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01];
const TARGET_PX_MIN = 40;
const TARGET_PX_MAX = 90;

function pickRoundLength(pxPerM) {
  for (const m of ROUND_STEPS_M) {
    const px = m * pxPerM;
    if (px >= TARGET_PX_MIN && px <= TARGET_PX_MAX) return { m, px };
  }
  // At the extremes of cam.s's clamp no round step may fit the target band; pick the closest rather than none.
  let best = ROUND_STEPS_M[0], bestDist = Infinity;
  for (const m of ROUND_STEPS_M) {
    const px = m * pxPerM;
    const dist = px < TARGET_PX_MIN ? TARGET_PX_MIN - px : px - TARGET_PX_MAX;
    if (dist < bestDist) { bestDist = dist; best = m; }
  }
  return { m: best, px: best * pxPerM };
}

function formatLength(m) {
  return m >= 1 ? `${m} m` : `${Math.round(m * 100)} cm`;
}

export default function ScaleBar() {
  const [pxPerM, setPxPerM] = useState(() => (typeof cam !== "undefined" ? cam.s : 14));

  useEffect(() => {
    let raf;
    const tick = () => {
      if (typeof cam !== "undefined" && cam.s !== undefined) setPxPerM(cam.s);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const { m, px } = pickRoundLength(pxPerM);

  return (
    <div
      style={{
        position: "absolute", left: 134, bottom: 44, zIndex: 5,
        display: "flex", flexDirection: "column", alignItems: "center",
        pointerEvents: "none",
      }}
    >
      <Text size="xs" c="dimmed" mb={2} ff="var(--sans, sans-serif)">
        {formatLength(m)}
      </Text>
      <div style={{ width: px, height: 0, position: "relative" }}>
        <div style={{ position: "absolute", left: 0, right: 0, top: 0, borderTop: "2px solid var(--accent, #3d8bfd)" }} />
        <div style={{ position: "absolute", left: 0, top: -4, width: 1, height: 8, background: "var(--accent, #3d8bfd)" }} />
        <div style={{ position: "absolute", right: 0, top: -4, width: 1, height: 8, background: "var(--accent, #3d8bfd)" }} />
      </div>
    </div>
  );
}

import { useState } from "react";
import { Button, Group, NumberInput, Slider, Stack, Text } from "@mantine/core";
import { useBenchChanges, benchReady } from "./bridge.js";

// The right sidebar's "Selected" panel. Reads S/sel/selection/AXES; every edit goes through the page's functions
// (applyAxisDelta, nudgeSelected, ...) so the multi-select and lock guards live in one place.
const fmt = (v) => Number(v.toFixed(2));

function Row({ label, children }) {
  return (
    <Group justify="space-between" wrap="nowrap" gap={8}>
      <Text size="xs" ff="var(--mono)" c="dimmed">{label}</Text>
      {children}
    </Group>
  );
}

function AxisRow({ a, s, dis }) {
  const live = a.get(s);
  // While the box has focus its own text is the truth; otherwise it mirrors
  // the scan. Committed on blur/Enter (the old input's native `change`), not
  // per keystroke, so typing "12" does not apply 1 then 12.
  const [draft, setDraft] = useState(null);
  const commit = (raw) => {
    setDraft(null);
    // An emptied or invalid box is a no-op, not a delta of 0; the box falls back to the scan's real value.
    if (raw === "" || raw == null || Number.isNaN(+raw)) { renderProps(); return; }
    applyAxisDelta(a, +raw);
    renderProps();
    draw();
  };
  return (
    <>
      <Row label={a.label}>
        <Group gap={4} wrap="nowrap">
          <NumberInput
            id={`in_${a.key}`}
            classNames={{ input: "num" }}
            size="xs"
            w={84}
            hideControls
            step={a.step}
            value={draft ?? fmt(live)}
            disabled={dis}
            onChange={(v) => setDraft(v === "" ? "" : v)}
            onBlur={(ev) => { if (draft !== null) commit(ev.currentTarget.value); }}
            onKeyDown={(ev) => { if (ev.key === "Enter") { ev.preventDefault(); commit(ev.currentTarget.value); } }}
          />
          <Text size="xs" ff="var(--mono)" c="dimmed" w={14}>{a.unit === "deg" ? "°" : a.unit}</Text>
        </Group>
      </Row>
      <Slider
        id={`sl_${a.key}`}
        min={a.min}
        max={a.max}
        step={a.step}
        value={Math.min(a.max, Math.max(a.min, live))}
        disabled={dis}
        size="xs"
        label={null}
        onChange={(v) => { applyAxisDelta(a, v); notifyUI(true); draw(); }}
      />
      {a.nudges && (
        <Group gap={4} justify="flex-end" wrap="nowrap">
          {a.nudges.map((n) => (
            <Button
              key={n}
              size="compact-xs"
              variant="default"
              data-nudge-axis={a.key}
              data-nudge={n}
              disabled={dis}
              onClick={() => nudgeSelected(a.key, n)}
            >
              {n > 0 ? "+" : ""}{n}
            </Button>
          ))}
        </Group>
      )}
    </>
  );
}

export default function PropsPanel() {
  useBenchChanges();
  if (!benchReady() || !S.length || !S[sel]) return null;
  const s = S[sel];
  // Greyed out while the as-imported view is showing, like a locked scan: setPose() refuses edits then.
  const dis = !!(s.lock || viewAsImported);
  const movedByAlign = s.alignMovedM > 0.001 && !viewAsImported;

  return (
    <Stack gap={6}>
      <Row label="name"><Text size="xs" ta="right" style={{ wordBreak: "break-word" }}>{s.name}</Text></Row>
      <Row label="colour">
        <input
          type="color"
          id="in_col"
          value={_rgbToHex(s.col)}
          title="False-colour display only - picking which two clouds are which colour is how a person compares a pair. Never touches exported data, which always keeps the real captured colour."
          onChange={(ev) => setSelectedColour(_hexToRgb(ev.target.value))}
        />
      </Row>
      {s.lock && <Text size="xs" c="dimmed" ff="var(--mono)">locked: not movable</Text>}
      {AXES.map((a) => <AxisRow key={`${a.key}:${s.i}`} a={a} s={s} dis={dis} />)}
      <Button
        id="bResetRot"
        variant="default"
        fullWidth
        size="xs"
        disabled={dis}
        title="Sets yaw, pitch and roll to exactly 0 degrees for every selected scan (Alt+R). The scan turns about its own centre, so the centre stays put and the stored x/y move with it."
        onClick={() => resetRotationSelected()}
      >
        Reset rotation (0&deg;)
      </Button>
      <Row label="levelled by"><Text size="sm" ff="var(--mono)">{s.tilt}&deg;</Text></Row>
      <Row label="walls found"><Text size="sm" ff="var(--mono)">{s.walls.length}</Text></Row>
      <Row label="footprint"><Text size="sm" ff="var(--mono)">{s.area} m&sup2;</Text></Row>
      {movedByAlign && (
        <>
          <Row label="moved by align">
            <Text size="sm" ff="var(--mono)" c="yellow">{s.alignMovedM.toFixed(2)} m, {s.alignMovedDeg.toFixed(1)}&deg;</Text>
          </Row>
          <Button id="bRevertOne" variant="default" fullWidth size="xs" onClick={() => revertSelectedToPlacement()}>
            Revert this scan to my placement
          </Button>
        </>
      )}
      {!isAtAsImported(s) && !viewAsImported && (
        <Button
          id="bRevertAsImported"
          variant="default"
          fullWidth
          size="xs"
          title="Goes all the way back to how this scan looked the moment it came in - further back than 'revert to my placement'."
          onClick={() => revertSelectedToAsImported()}
        >
          Revert this scan to as imported
        </Button>
      )}
    </Stack>
  );
}

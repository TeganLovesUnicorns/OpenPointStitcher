import { Button, Group, Stack, Table, Text, UnstyledButton } from "@mantine/core";
import { useBenchChanges, benchReady } from "./bridge.js";

// The small dynamic lists in the right sidebar.

// "Floor height": the heights where the selected scan has the most flat points (floor, seats, stage lip, ceiling).
// Its "floor" and "ceiling" buttons set the same marks as the review view, which height settle uses.
export function FloorBands() {
  useBenchChanges();
  if (!benchReady() || !S.length || !S[sel]) return null;
  const s = S[sel];
  const max = Math.max(...s.bands.map((b) => b.n), 1);
  const near = (h, v) => v !== null && v !== undefined && Math.abs(h - v) < 0.03;
  return (
    <Stack gap={2}>
      {[...s.bands].sort((a, b) => b.h - a.h).map((b, k) => {
        const isFloor = near(b.h, floorOf(s)), isCeil = near(b.h, ceilingOf(s));
        return (
          <Group key={k} gap={4} wrap="nowrap" data-b={k}>
            <div className={`band ${isFloor || isCeil ? "pick" : ""}`} style={{ flex: 1, minWidth: 0 }}>
              <div className="bfill" style={{ width: `${((b.n / max) * 100).toFixed(0)}%` }} />
              <div className="blab">{b.h.toFixed(2)} m{isFloor ? "  ← floor" : ""}{isCeil ? "  ← ceiling" : ""}</div>
            </div>
            <UnstyledButton className="bandset" data-set="floor" title="This height is the floor"
                            onClick={() => setMark(s, "floor_h_m", b.h)}>floor</UnstyledButton>
            <UnstyledButton className="bandset" data-set="ceiling" title="This height is the ceiling"
                            onClick={() => setMark(s, "ceiling_h_m", b.h)}>ceiling</UnstyledButton>
          </Group>
        );
      })}
    </Stack>
  );
}

// Named landmarks grouped by name: which scans carry each and whether it links them.
export function LandmarksList() {
  useBenchChanges();
  if (!benchReady()) return null;
  if (!landmarks.length) return <Text size="xs" c="dimmed" py={4}>none yet</Text>;
  const groups = new Map();
  for (const e of landmarks) {
    const k = landmarkKey(e.name);
    if (!groups.has(k)) groups.set(k, { name: e.name, kind: e.kind, marks: [] });
    groups.get(k).marks.push(e);
  }
  const shown = (scan) => { const s = S.find((x) => x.name === scan); return s ? displayName(s) : scan; };
  return (
    <Stack gap={4}>
      {[...groups.values()].map((g) => {
        const scans = [...new Set(g.marks.map((e) => e.scan))];
        return (
          <Group key={g.name} className="projectRow" justify="space-between" wrap="nowrap" gap={8} data-landmark={g.name}>
            <div style={{ minWidth: 0, flex: 1 }}>
              <Text size="xs" fw={600} style={{ color: "rgb(120,230,160)" }}>{g.name} ({g.kind})</Text>
              <Text size="xs" c="dimmed">
                {scans.length > 1 ? `links ${scans.length} scans: ` : "in one scan only, name it in another to link them: "}
                {scans.map(shown).join(", ")}
              </Text>
            </div>
            <Group gap={4} wrap="nowrap" style={{ flexShrink: 0 }}>
              {g.marks.map((e) => (
                <Button key={e.id} size="compact-xs" variant="default" data-droplandmark={e.id}
                  title={`Take this name off ${shown(e.scan)}. Undo puts it back`}
                  onClick={() => removeLandmark(e.id)}>drop from {shown(e.scan)}</Button>
              ))}
            </Group>
          </Group>
        );
      })}
    </Stack>
  );
}

// Labels, the answer key the wall finders are scored against.
export function LabelsList() {
  useBenchChanges();
  if (!benchReady()) return null;
  if (!labels.length) return <Text size="xs" c="dimmed" py={4}>none yet</Text>;
  const shown = (scan) => { const s = S.find((x) => x.name === scan); return s ? displayName(s) : scan; };
  const nameOf = (cls) => (LABEL_CLASSES.find((c) => c[0] === cls) || [cls, cls, [200, 200, 200]]);
  return (
    <Stack gap={4}>
      {labels.map((e) => {
        const [, name, c] = nameOf(e.cls);
        return (
          <Group key={e.id} className="projectRow" justify="space-between" wrap="nowrap" gap={8} data-label-row={e.cls}>
            <Text size="xs" style={{ flex: 1, minWidth: 0 }}>
              <span style={{ color: `rgb(${c})`, fontWeight: 600 }}>{name}</span>
              {` in ${shown(e.scan)}, ${e.n_points.toLocaleString()} preview points`}
            </Text>
            <Button size="compact-xs" variant="default" data-droplabel={e.id} title="Take this label off. Undo puts it back"
              onClick={() => removeLabel(e.id)}>drop</Button>
          </Group>
        );
      })}
    </Stack>
  );
}

// Distances measured on site against what the layout reads now.
const mm = (m) => `${m >= 0 ? "+" : ""}${(m * 1000).toFixed(0)} mm`;
const inch = (m) => `${m >= 0 ? "+" : ""}${(m / 0.0254).toFixed(2)}"`;
function typical(rows) {
  const v = rows.map((r) => Math.abs(r.diff_m)).sort((a, b) => a - b);
  return v.length ? v[Math.floor((v.length - 1) / 2)] : null;
}
export function MeasurementsList() {
  useBenchChanges();
  if (!benchReady()) return null;
  if (!measurements.length) return <Text size="xs" c="dimmed" py={4}>none yet</Text>;
  const rows = measurements.map((m) => ({ m, r: measureReading(m) }));
  const ok = rows.filter(({ r }) => !r.gone && !r.sameWall).map(({ r }) => r);
  const groups = [["across scans", ok.filter((r) => !r.oneScan)], ["inside one scan", ok.filter((r) => r.oneScan)]];
  return (
    <Stack gap={4}>
      <Text size="xs" c="dimmed" data-measure-summary>
        {groups.filter(([, g]) => g.length).map(([name, g]) =>
          `${name}: ${g.length}, typically ${(typical(g) * 1000).toFixed(0)} mm off`).join(";  ")}
      </Text>
      <Table verticalSpacing={3} horizontalSpacing="sm" fz="xs" ff="var(--mono)">
        <Table.Thead>
          <Table.Tr>
            <Table.Th>#</Table.Th><Table.Th>between</Table.Th><Table.Th>on site</Table.Th><Table.Th>layout</Table.Th>
            <Table.Th>difference</Table.Th><Table.Th>walls seen by</Table.Th><Table.Th>note</Table.Th><Table.Th />
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rows.map(({ m, r }, i) => (
            <Table.Tr key={m.id} data-measure-row={i}>
              <Table.Td>M{i + 1}</Table.Td>
              <Table.Td style={{ fontFamily: "var(--sans)" }} title="The scans that saw each wall">
                {r.gone ? `${shownScan(m.a.scan)} to ${shownScan(m.b.scan)}`
                  : `${r.scansA.map(shownScan).join(" + ")}  to  ${r.scansB.map(shownScan).join(" + ")}`}
              </Table.Td>
              <Table.Td title={`typed as ${m.entered}`}>{m.site_m.toFixed(3)} m ({feetInches(m.site_m)})</Table.Td>
              {r.gone ? (
                <Table.Td colSpan={3} style={{ color: "var(--warn)" }}>
                  a wall it was pinned to is gone (deleted in review, or its scan was removed)
                </Table.Td>
              ) : r.sameWall ? (
                <Table.Td colSpan={3} style={{ color: "var(--warn)" }}>both ends are now on one joined wall</Table.Td>
              ) : (
                <>
                  <Table.Td>{r.layout_m.toFixed(3)} m</Table.Td>
                  <Table.Td data-measure-diff title={r.parallel ? "" : `the two walls are ${r.angle.toFixed(1)} degrees apart, so the distance depends on where along the wall you clicked`}
                    style={{ color: r.parallel ? undefined : "var(--warn)" }}>
                    {mm(r.diff_m)} ({inch(r.diff_m)}, {((r.diff_m / m.site_m) * 100).toFixed(1)}%){r.parallel ? "" : ` walls ${r.angle.toFixed(0)}° apart`}
                  </Table.Td>
                  <Table.Td style={{ fontFamily: "var(--sans)" }}>{r.oneScan ? "one scan sees both" : "different scans"}</Table.Td>
                </>
              )}
              <Table.Td style={{ fontFamily: "var(--sans)" }}>{m.note}</Table.Td>
              <Table.Td>
                <Button size="compact-xs" variant="default" data-dropmeasure={m.id} title="Take this measurement off. Undo puts it back"
                  onClick={() => removeMeasurement(m.id)}>drop</Button>
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Stack>
  );
}

// One kind each: exclude regions (out of alignment, kept in the export by default) or, with removals, points removed
// in review (out of everything). Restore goes through removeExclusion() and pushUndo().
export function ExclusionsList({ removals = false }) {
  useBenchChanges();
  if (!benchReady()) return null;
  const rows = exclusions.map((e, i) => [e, i]).filter(([e]) => (e.source === "review") === removals);
  if (!rows.length) {
    return <Text size="xs" c="dimmed" py={4}>none yet</Text>;
  }
  return (
    <Stack gap={4}>
      {rows.map(([e, i]) => (
        <Group key={e.id} className="projectRow" justify="space-between" wrap="nowrap" gap={8}>
          <div style={{ minWidth: 0, flex: 1 }}>
            <Text size="xs" fw={600} c="red.4">{e.scan}</Text>
            <Text size="xs" ff="var(--mono)" c="dimmed">
              {removals
                ? `${e.n_points.toLocaleString()} preview pt(s) removed`
                : `${e.n_points.toLocaleString()} pt(s), ${e.extent.map((v) => v.toFixed(2)).join(" x ")} m`}
            </Text>
          </div>
          <Button
            size="compact-xs"
            variant="default"
            style={{ flexShrink: 0 }}
            data-restore={i}
            title={removals ? "Put these points back" : "Use this region for alignment again"}
            onClick={() => { pushUndo(); removeExclusion(e.id); }}
          >
            restore
          </Button>
        </Group>
      ))}
    </Stack>
  );
}

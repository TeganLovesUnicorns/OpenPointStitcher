import { Fragment, createContext, useContext, useEffect, useState } from "react";
import { ActionIcon, Box, Button, ColorSwatch, Group, Stack, Table, Tabs, Text, Textarea } from "@mantine/core";
import { useBenchChanges, benchReady } from "./bridge.js";
import PropsPanel from "./PropsPanel.jsx";
import SettingSlider from "./SettingSlider.jsx";
import { FloorBands, ExclusionsList, LandmarksList, LabelsList, MeasurementsList } from "./SmallLists.jsx";

// The right sidebar, the bottom links table and the scan panel's Select all / Deselect all header. Same contract as
// MenuBar.jsx: the ids the page script binds at load are kept, the bundle runs before it, and anything the page
// script writes into (labels, the status textarea) has no React-managed children.

// ---- collapsible "?" explanations ------------------------------------------
// One open/closed state per group, saved in localStorage. Every note in a group is always rendered (just hidden):
// the tests read .note[data-notegroup] nodes whether or not they show.
const NoteCtx = createContext({ group: null, open: false });

function Section({ title, group, children }) {
  const key = group ? `openpoint_note_open_${group}` : null;
  const [open, setOpen] = useState(() => {
    try { return key ? localStorage.getItem(key) === "1" : false; } catch (e) { return false; }
  });
  const toggle = () => {
    const next = !open;
    setOpen(next);
    try { localStorage.setItem(key, next ? "1" : "0"); } catch (e) { /* private window: just do not persist */ }
  };
  return (
    <NoteCtx.Provider value={{ group, open }}>
      <Group justify="space-between" px="sm" pt={10} pb={6} style={{ borderTop: "1px solid var(--line)" }}>
        <Text component="h2" size="xs" tt="uppercase" c="dimmed" style={{ letterSpacing: ".09em" }}>{title}</Text>
        {group && (
          <ActionIcon
            className={`noteToggle${open ? " on" : ""}`}
            data-notegroup={group}
            variant={open ? "filled" : "default"}
            size="xs"
            radius="xl"
            title="Show/hide the explanation"
            onClick={toggle}
          >
            ?
          </ActionIcon>
        )}
      </Group>
      <div style={{ padding: "0 12px 10px" }}>{children}</div>
    </NoteCtx.Provider>
  );
}

function Note({ children, mt, mb }) {
  const { group, open } = useContext(NoteCtx);
  return (
    <Text className="note" data-notegroup={group} hidden={!open} size="xs" c="dimmed" mt={mt} mb={mb} style={{ lineHeight: 1.5 }}>
      {children}
    </Text>
  );
}

function Labelled({ label, labId, labInitial }) {
  return (
    <Group justify="space-between" mb={2}>
      <Text size="xs" ff="var(--mono)" c="dimmed">{label}</Text>
      <Text
        component="span"
        id={labId}
        size="xs"
        ff="var(--mono)"
        ref={(el) => { if (el && !el.dataset.init) { el.textContent = labInitial; el.dataset.init = "1"; } }}
      />
    </Group>
  );
}

// ---- right sidebar ----------------------------------------------------------
export function RightSidebar() {
  return (
    <>
      <Section title="Selected">
        <Box id="props"><PropsPanel /></Box>
      </Section>

      <Section title="Move step" group="moveStep">
        <Labelled label="Keyboard step" labId="granLab" labInitial="1 cm" />
        <SettingSlider id="granularity" min={0} max={1000} defaultValue={500} />
        <Note mt={4}>
          How far arrow keys / Ctrl+Up/Down move the selected scan(s) per key press.
          Logarithmic: the low end is fine millimetre nudges, the high end is a metre at a time.
        </Note>
        <div style={{ marginTop: 10 }}>
          <Labelled label="Turn step" labId="rotGranLab" labInitial={"1°"} />
          <SettingSlider id="rotGranularity" min={0} max={6} step={1} defaultValue={2} />
        </div>
        <Note mt={4}>
          Seven fixed steps, coarse to fine: 10, 5, 1, 0.5, 0.25, 0.1, 0.05 degrees. Not continuous
          on purpose, so a turn can be repeated exactly. Drives the yaw nudge buttons below, and
          Ctrl+Left/Right (yaw) and Alt+arrows (pitch/roll) on the keyboard.
        </Note>
      </Section>

      <Section title="Floor height" group="floorHeight">
        <Note mb={7}>
          Flat surfaces found in this scan, by height. The tool picked the lowest strong one as the
          floor. Click another if it guessed wrong.
        </Note>
        <Box id="bands"><FloorBands /></Box>
      </Section>

      <Section title="Messages">
        {/* Run, Cancel and Save are on the top bar. The page script reads and writes this textarea's .value and
            .style.display. */}
        <Textarea id="out" readOnly rows={8} styles={{ input: { fontFamily: "var(--mono)", fontSize: 10 } }} />
      </Section>
    </>
  );
}

// ---- scan panel header ------------------------------------------------------
export function SelBar() {
  return (
    <Group gap={6} px="sm" pb={6} wrap="wrap">
      <Button id="sbSelectAll" variant="default" size="compact-xs" title="Select every scan in the list (Ctrl+A)">Select all</Button>
      <Button id="sbSelectNone" variant="default" size="compact-xs" title="Clear the selection. Clicking empty space in the list does the same">Deselect all</Button>
      <Button id="sbReviewSelected" variant="default" size="compact-xs"
        title="Review and clean up the selected scans one after another (Next scan moves on)">Review selected</Button>
      <Text component="span" id="selcount" size="xs" ff="var(--mono)" c="dimmed" />
    </Group>
  );
}

// ---- bottom links table -------------------------------------------------------
// 'declined' (the tool looked and kept the placement, a success), 'stale' (a scan on this row moved since the
// number was measured) and 'refined' (the placement was nudged and the result failed a quality check) each get
// their own look.
const STATE_LABEL = { none: "not run", ok: "matched", weak: "rejected", declined: "kept placement", refined: "kept, fine-tuned",
  stale: "stale - rerun" };
const STATE_COLOR = { ok: "var(--good)", weak: "var(--warn)", declined: "var(--accent)", refined: "var(--warn)", stale: "var(--dim)" };

// Four tabs. Building a link switches to Anchors, where its finish and cancel buttons are.
export function BottomPanel() {
  useBenchChanges();
  const ready = benchReady();
  const [tab, setTab] = useState("links");
  const building = ready && !!pendingAnchor;
  useEffect(() => { if (building) setTab("anchors"); }, [building]);
  const measuring = ready && tool === "measure";
  useEffect(() => { if (measuring) setTab("measurements"); }, [measuring]);
  const nOf = (removals) => (ready ? exclusions.filter((e) => (e.source === "review") === removals).length : 0);
  return (
    <Tabs id="bottomTabs" value={tab} onChange={setTab} keepMounted>
      <Tabs.List px="sm">
        <Tabs.Tab value="links" title="What each run found between pairs of scans">Links from runs</Tabs.Tab>
        <Tabs.Tab value="anchors" title="Links you drew yourself with the anchor tool">Anchors ({ready ? anchors.length : 0})</Tabs.Tab>
        <Tabs.Tab value="exclusions" title="Mostly right but flawed areas kept out of alignment">Exclusions ({nOf(false)})</Tabs.Tab>
        <Tabs.Tab value="removals" title="People, drift and other junk removed in review">Removed in review ({nOf(true)})</Tabs.Tab>
        <Tabs.Tab value="landmarks" title="Features you named in review; the same name in two scans links them">
          Landmarks ({ready ? new Set(landmarks.map((e) => landmarkKey(e.name))).size : 0})</Tabs.Tab>
        <Tabs.Tab value="labels" title="What you labelled points as in review: the answer key the tool's finders are scored against">
          Labels ({ready ? labels.length : 0})</Tabs.Tab>
        <Tabs.Tab value="measurements" title="Distances you measured on site, against what the layout reads">
          Measurements ({ready ? measurements.length : 0})</Tabs.Tab>
      </Tabs.List>
      <Tabs.Panel value="links">
      {/* Scans the last automatic run found may be bent. Text comes from the page's describeBentScans. */}
      {ready && bentScanFlags.length > 0 && (
        <Stack id="bentScans" gap={2} px="sm" pb={4}>
          {describeBentScans(bentScanFlags).map((line, i) => (
            <Text key={i} size="xs" style={{ color: i ? "var(--warn)" : "var(--dim)" }}>{line.trim()}</Text>
          ))}
        </Stack>
      )}
      <Table verticalSpacing={3} horizontalSpacing="sm" fz="xs" ff="var(--mono)">
        <Table.Thead>
          <Table.Tr>
            <Table.Th>pair</Table.Th><Table.Th>state</Table.Th><Table.Th>fit</Table.Th><Table.Th>rmse</Table.Th>
            <Table.Th>tilt</Table.Th><Table.Th>moved from where you put it</Table.Th><Table.Th />
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody id="links">
          {ready && LINKS.map((L, k) => {
            const c = STATE_COLOR[L.state] || "var(--dim)";
            return (
              <Fragment key={`l${k}`}>
              <Table.Tr style={{ opacity: L.on ? 1 : 0.42 }} title={L.why || undefined}>
                <Table.Td>
                  <Group gap={6} wrap="nowrap">
                    <ColorSwatch size={8} color={L.on ? c : "#39414d"} />
                    <span>{S[L.a].name} &rarr; {S[L.b].name}</span>
                  </Group>
                </Table.Td>
                <Table.Td style={{ color: L.on ? c : "var(--dim)" }}>{L.on ? (STATE_LABEL[L.state] || L.state) : "not linked"}</Table.Td>
                <Table.Td>{L.fit ?? "-"}</Table.Td>
                <Table.Td>{L.rmse ?? "-"}</Table.Td>
                <Table.Td>{L.tilt ?? "-"}</Table.Td>
                <Table.Td>{L.moved ?? "-"}</Table.Td>
                <Table.Td>
                  <Button size="compact-xs" variant="default" data-link={k} onClick={() => toggleLink(k)}>
                    {L.on ? "unlink" : "link"}
                  </Button>
                </Table.Td>
              </Table.Tr>
              {/* What happened to this pair, in plain words. */}
              {L.on && L.why && (
                <Table.Tr className="linkWhy" style={{ opacity: 0.85 }}>
                  <Table.Td colSpan={7} style={{ paddingTop: 0, color: "var(--dim)", whiteSpace: "normal", fontFamily: "var(--sans)" }}>
                    {L.why}
                  </Table.Td>
                </Table.Tr>
              )}
              </Fragment>
            );
          })}
        </Table.Tbody>
      </Table>
      </Tabs.Panel>
      <Tabs.Panel value="anchors">
        <Group justify="space-between" px="sm" pt={6} pb={4} wrap="nowrap">
          <Text size="xs" c="dimmed">Two boxes on the same feature in two scans, drawn with the Anchor tool.</Text>
          {/* No `disabled`: the vanilla script owns it (job running / as-imported). */}
          <Button id="bClearLinks" variant="default" size="compact-xs" style={{ flexShrink: 0 }}
            title="Remove every anchor you drew. Undo brings them back until you save">Clear all anchors</Button>
        </Group>
        <Table verticalSpacing={3} horizontalSpacing="sm" fz="xs" ff="var(--mono)">
        <Table.Tbody id="anchorsList">
          {ready && anchors.map((a, i) => (
            <Table.Tr key={`a${i}`}>
              <Table.Td>
                <Group gap={6} wrap="nowrap">
                  <ColorSwatch size={8} color="#fff" />
                  <span>anchor {i + 1}{a.label ? ` (${a.label})` : ""}: {a.members.map((m) => m.scan).join(" + ")}</span>
                </Group>
              </Table.Td>
              <Table.Td style={{ color: "#fff" }}>manual</Table.Td>
              <Table.Td>-</Table.Td><Table.Td>-</Table.Td><Table.Td>-</Table.Td><Table.Td>-</Table.Td>
              <Table.Td>
                <Group gap={4} wrap="nowrap">
                  {a.members.map((m, k) => (
                    <Button key={k} size="compact-xs" variant="default" data-redraw={`${i}:${k}`}
                      title={`Redraw ${m.scan}'s end of this anchor: drag a new box over the right place; the rest of the anchor is kept`}
                      onClick={() => redrawAnchorMember(i, k)}>redraw {k + 1}</Button>
                  ))}
                  <Button size="compact-xs" variant="default" data-delanchor={i} onClick={() => dropAnchor(i)}>drop</Button>
                </Group>
              </Table.Td>
            </Table.Tr>
          ))}
          {ready && pendingAnchor && (
            <Table.Tr key="pending" style={{ opacity: 0.65 }}>
              <Table.Td colSpan={5}>
                building anchor: {pendingAnchor.members.map((m) => m.scan).join(" + ") || "(drag a box on the selected scan)"}
              </Table.Td>
              <Table.Td colSpan={2}>
                <Group gap={4} wrap="nowrap">
                  <Button id="bFinishAnchor" size="compact-xs" variant="default" disabled={pendingAnchor.members.length < 2} onClick={() => finishPendingAnchor()}>finish</Button>
                  <Button id="bCancelAnchor" size="compact-xs" variant="default" onClick={() => cancelPendingAnchor()}>cancel</Button>
                </Group>
              </Table.Td>
            </Table.Tr>
          )}
        </Table.Tbody>
      </Table>
      </Tabs.Panel>
      <Tabs.Panel value="exclusions" px="sm" pt={6}>
        <Text size="xs" c="dimmed" mb={6}>
          Mostly right but flawed areas (say a wall that scanned as a thick band instead of a thin line), drawn with the
          Exclude region tool. Left out of alignment; kept in the export unless Settings, Exporting says otherwise.
          Restoring one never touches the scan file.
        </Text>
        <Box id="exclusionsList"><ExclusionsList /></Box>
      </Tabs.Panel>
      <Tabs.Panel value="removals" px="sm" pt={6}>
        <Text size="xs" c="dimmed" mb={6}>
          People, drift and other junk you removed in the review view. Left out of alignment and the export.
          Restoring puts the points back; the scan file is never touched.
        </Text>
        <Box id="removalsList"><ExclusionsList removals /></Box>
      </Tabs.Panel>
      <Tabs.Panel value="landmarks" px="sm" pt={6}>
        <Text size="xs" c="dimmed" mb={6}>
          Features you named in review (select the points, then "selected points are a landmark..."). The same name in
          two or more scans links those scans on it, in every run that uses anchors.
        </Text>
        <Box id="landmarksList"><LandmarksList /></Box>
      </Tabs.Panel>
      <Tabs.Panel value="labels" px="sm" pt={6}>
        <Text size="xs" c="dimmed" mb={6}>
          What you said points really are (review: select, then a "label as" button). They change nothing about
          alignment or export; they are the answer key the tool's wall, door, window and furniture finding is scored on.
        </Text>
        <Box id="labelsList"><LabelsList /></Box>
      </Tabs.Panel>
      <Tabs.Panel value="measurements" px="sm" pt={6}>
        <Text size="xs" c="dimmed" mb={6}>
          Distances you measured at the venue. With the Measure tool, in the Plan view, click a wall, then the wall at the
          other end of the run, and type what the tape or laser read. Each row is what the layout reads between the
          venue's walls right now (several scans' copies of one wall joined into one), so it updates as scans move.
        </Text>
        <Box id="measurementsList"><MeasurementsList /></Box>
      </Tabs.Panel>
    </Tabs>
  );
}

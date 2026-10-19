import { Group, Modal, NativeSelect, Tabs, Text } from "@mantine/core";
import { Tg, SliderRow, Heading, Pad, HelpLink } from "./MenuControls.jsx";

// A NativeSelect's `label` isn't wrapped by Tg/SliderRow (it draws its own),
// so its "?" is built inline the same way, as the label node itself.
const selLabel = (text, anchor, onHelp) => (
  <Group gap={4} wrap="nowrap" data-setting={anchor}>
    <span>{text}</span>
    <HelpLink anchor={anchor} onHelp={onHelp} />
  </Group>
);

// One tabbed modal. Every control keeps its old id and hidden tabs stay in the DOM (keepMounted), because the page
// script binds to those ids once at load.

export default function SettingsModal({ opened, onClose, onHelp }) {
  return (
    <Modal
      opened={opened}
      onClose={onClose}
      keepMounted
      title="Settings"
      size="lg"
      radius="xs"
      shadow="lg"
      withinPortal
      trapFocus={false}
      returnFocus={false}
      transitionProps={{ duration: 0 }}
      id="settingsModal"
      styles={{ body: { padding: 0 }, content: { zoom: "var(--ui-zoom)" } }}
    >
      {/* data-menu-modal is our own marker: tests find the modal's visibility through it, since keepMounted only toggles
          display on an ancestor. */}
      <div data-menu-modal="menuSettings">
      <Tabs defaultValue="resources" orientation="vertical" keepMounted>
        <Tabs.List>
          <Tabs.Tab value="resources">Resources</Tabs.Tab>
          <Tabs.Tab value="prepare">Preparing scans</Tabs.Tab>
          <Tabs.Tab value="align">Aligning</Tabs.Tab>
          <Tabs.Tab value="export">Exporting</Tabs.Tab>
          <Tabs.Tab value="versions">Saved versions</Tabs.Tab>
          <Tabs.Tab value="appearance">Appearance</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="resources" p={0}>
          <Heading first>Resources</Heading>
          <Pad>
            <SliderRow label="CPU threads" labId="thLab" labInitial="12" id="threads" min={1} max={16} defaultValue={12}
              helpAnchor="resources" onHelp={onHelp} />
            <SliderRow label="Pairs at once" labId="ccLab" labInitial="6" id="conc" min={1} max={16} defaultValue={6}
              helpAnchor="resources" onHelp={onHelp} />
            <div style={{ marginTop: 8 }}>
              <Tg id="cbGpu" label="Use GPU where possible" helpAnchor="resources" onHelp={onHelp} />
            </div>
            <Text
              id="gpuStatusNote"
              size="xs"
              c="dimmed"
              ref={(el) => { if (el && !el.dataset.init) { el.textContent = "checking GPU status…"; el.dataset.init = "1"; } }}
            />
          </Pad>
        </Tabs.Panel>

        <Tabs.Panel value="prepare" p={0}>
          <Heading first>Preparing scans</Heading>
          <Pad>
            <NativeSelect
              id="levelTilt"
              size="xs"
              label={selLabel("Levelling: largest floor tilt to straighten", "levelTilt", onHelp)}
              defaultValue="15"
              title={"When a scan is levelled to its floor, this is the steepest floor the tool will accept as 'the floor'. Raked seating and sloped floors lie between about 2 and 8 degrees; a lower limit (3) stops a ramp or a raked section being mistaken for the floor, at the cost of failing on a genuinely steep one. Takes effect the next time previews are built (File > Rebuild previews)."}
              data={[
                { value: "0", label: "Off: keep every scan's level from the iPhone" },
                { value: "3", label: "3 degrees (strict, for raked floors)" },
                { value: "5", label: "5 degrees" },
                { value: "8", label: "8 degrees" },
                { value: "15", label: "15 degrees (default)" },
                { value: "25", label: "25 degrees (lenient)" },
              ]}
            />
            <NativeSelect
              id="wallSearch"
              size="xs"
              mt={10}
              label={selLabel("Wall search", "wallSearch", onHelp)}
              defaultValue="standard"
              title={"Standard looks for up to 8 walls in each scan and can miss one in a busy room. Thorough looks for up to 24 and keeps only those with solid wall behind them, including half walls and walls hanging from the ceiling; it finds walls Standard misses, and can also take tall cabinets or stacked boxes for walls. Takes effect the next time previews are built (File > Rebuild previews)."}
              data={[
                { value: "standard", label: "Standard (up to 8 walls a scan)" },
                { value: "thorough", label: "Thorough (up to 24, may pick up cabinets)" },
              ]}
            />
            <NativeSelect
              id="rangeLimit"
              size="xs"
              mt={10}
              label={selLabel("Ignore far points when aligning", "rangeLimit", onHelp)}
              defaultValue=""
              title={"Alignment only, never the export: leave out points farther than this from where each scan started, measured across the floor. A phone's depth sensor is less exact far away. But a scan captured while walking has good points far from where it started, so this is off by default."}
              data={[
                { value: "", label: "Off (use every point)" },
                { value: "4", label: "4 m" },
                { value: "6", label: "6 m" },
                { value: "8", label: "8 m" },
                { value: "10", label: "10 m" },
                { value: "15", label: "15 m" },
              ]}
            />
          </Pad>
        </Tabs.Panel>

        <Tabs.Panel value="align" p={0}>
          <Heading first>Aligning</Heading>
          <Pad>
            <NativeSelect
              id="searchVoxel"
              size="xs"
              label={selLabel("Search voxel (automatic align)", "searchVoxel", onHelp)}
              defaultValue="0.15"
              title={"Coarser copy of the clouds used only to find starting positions; fine alignment still works at full detail. Large scans search far faster at 0.15 m, the default here."}
              data={[
                { value: "", label: "Off - full resolution search (slow, thorough)" },
                { value: "0.15", label: "0.15m - fast" },
                { value: "0.10", label: "0.10m - a little faster, less tested" },
                { value: "0.20", label: "0.20m - fastest, riskier" },
              ]}
            />
            <div style={{ marginTop: 10 }}>
              <Tg
                id="cbDetectorVote"
                defaultChecked
                label={"Detector voting"}
                helpAnchor="cbDetectorVote"
                onHelp={onHelp}
                title={"Only applies to \"Run from my alignment\". Instead of ranking candidates by raw point closeness alone, checks whether matched walls/floors/ceilings - and corners, where three of them meet - actually agree, and, only as a second opinion, whether their colour does too. A candidate whose surfaces or corners disagree geometrically is rejected outright, never on colour alone. Corners are the stronger check: a symmetric room can line up the wrong wall with the wrong wall, but rarely the wrong corner. On by default. Untick it to compare."}
              />
            </div>
            <div style={{ marginTop: 10 }}>
              <Tg id="cbHeightSettle" defaultChecked label="Settle heights on shared floors and ceilings"
                helpAnchor="cbHeightSettle" onHelp={onHelp}
                title="Last step of every run: nudges each scan the run moved up or down so floors and ceilings it shares with overlapping scans meet. Heights only; locked scans never move." />
              <Tg id="cbSettlePieces" indent label="... using only the floors and ceilings it found"
                helpAnchor="cbSettlePieces" onHelp={onHelp}
                title="Off by default. When on, a scan you have not marked by hand settles only on the floor and ceiling pieces the tool found in it (several floor levels and sloped ceilings included), so a table, counter or row of seats never counts as a floor. Floors and ceilings you marked by hand still come first." />
            </div>
            <div style={{ marginTop: 10 }}>
              <Tg id="cbThicknessTrust" defaultChecked label="Trust thin, crisp surfaces more than thick, fuzzy ones"
                helpAnchor="cbThicknessTrust" onHelp={onHelp}
                title="When fine-tuning each pair, points far from a crisp surface count for less, and a pair matched on fuzzy surfaces pins its scans less firmly. Weights only: no point is ever dropped. Runs take about a third longer." />
            </div>
            <div style={{ marginTop: 10 }}>
              <Tg id="cbLimitsRelative" label="Movement limits ignore the whole layout moving together"
                helpAnchor="cbLimitsRelative" onHelp={onHelp}
                title="Off by default. On: the limits on how far a run may move each scan (50 cm and 5 degrees) and the whole layout (20 cm on average) only count scans moving against each other, so the layout sliding or turning as one piece reads as no movement. Off: every move counts from where the scan started." />
            </div>
          </Pad>
        </Tabs.Panel>

        <Tabs.Panel value="export" p={0}>
          <Heading first>Exporting</Heading>
          <Pad>
            <NativeSelect
              id="strayRemoval"
              size="xs"
              label={selLabel("Remove stray points from the export", "strayRemoval", onHelp)}
              defaultValue=""
              title={"Off by default: the export keeps every point, and colour, exactly as scanned, and you reduce it in Vectorworks where you can see what goes. Turn this on to drop isolated points (people, glints, reflections) that sit far from their neighbours. It removes points, so it is your call."}
              data={[
                { value: "", label: "Off (keep every point)" },
                { value: "3.0", label: "Gentle (drops only the most isolated)" },
                { value: "2.0", label: "Normal" },
                { value: "1.0", label: "Strong (drops more)" },
              ]}
            />
            <div style={{ marginTop: 12 }}>
              <Tg id="cbExportKeepsExcluded" defaultChecked label="Keep exclude-region points in the export"
                helpAnchor="cbExportKeepsExcluded" onHelp={onHelp}
                title="On: points inside your exclude regions are left out of alignment only and still exported. Off: they are left out of the export too. Points removed in review are never exported." />
            </div>
            <div style={{ marginTop: 12 }}>
              <Tg id="cbExportRefine" label="Tighten the export with my links"
                helpAnchor="cbExportRefine" onHelp={onHelp}
                title="Off by default: the export puts every scan exactly where the layout has it. On: before merging, each scan in a link is fine-tuned against the first scan of that link, and kept only if that moves its points less than 50 cm. The export is still every point; your saved layout is not changed." />
            </div>
            <div style={{ marginTop: 12 }}>
              <Tg id="cbLandmarksInDrawings" defaultChecked label="Named landmarks in the drawings"
                helpAnchor="cbLandmarksInDrawings" onHelp={onHelp}
                title="On: the features you named in review (windows, doorways, columns...) are drawn with their names on a LANDMARKS layer in the plan view and the 3D drawing. Off: left out of both." />
            </div>
          </Pad>
        </Tabs.Panel>

        <Tabs.Panel value="versions" p={0}>
          <Heading first>Saved versions of the layout</Heading>
          <Pad>
            <Tg id="cbAutoTidy" defaultChecked label="Tidy saved versions automatically" helpAnchor="cbAutoTidy" onHelp={onHelp}
              title="Every save keeps the layout it replaced. With this on, older ones are thinned out when you save or open the project, by the two rules below." />
            <NativeSelect
              id="backupKeepAll"
              size="xs"
              mt={6}
              label={selLabel("Keep every saved version from the last", "backupKeepAll", onHelp)}
              defaultValue="1"
              data={[
                { value: "1", label: "1 hour (default)" },
                { value: "3", label: "3 hours" },
                { value: "12", label: "12 hours" },
                { value: "24", label: "24 hours" },
              ]}
            />
            <NativeSelect
              id="backupHourlyDays"
              size="xs"
              mt={10}
              label={selLabel("Then keep one per hour for", "backupHourlyDays", onHelp)}
              defaultValue="1"
              data={[
                { value: "1", label: "1 day (default), then one per day" },
                { value: "3", label: "3 days, then one per day" },
                { value: "7", label: "7 days, then one per day" },
              ]}
            />
            <div style={{ marginTop: 14 }}>
              <Tg id="cbHandLog" defaultChecked label="Keep a log of my hand moves" helpAnchor="cbHandLog" onHelp={onHelp}
                title="Each time you finish moving a scan by hand, one line goes into hand-moves.jsonl in this project's folder: which scan, where it was, where you put it, and how. It never leaves this computer and nothing reads it yet. Delete the file whenever you like." />
            </div>
          </Pad>
        </Tabs.Panel>

        <Tabs.Panel value="appearance" p={0}>
          <Heading first>Theme</Heading>
          <Pad>
            <NativeSelect
              id="themeSelect"
              size="xs"
              label={selLabel("Colours", "themeSelect", onHelp)}
              defaultValue="dark"
              title="Changes the colours of the panels and menus (the point view itself stays dark so points read the same). Remembered on this computer."
              data={[
                { value: "dark", label: "Dark (graphite and blue)" },
                { value: "light", label: "Light" },
                { value: "solarized", label: "Solarized dark" },
                { value: "synthwave", label: "Synthwave" },
              ]}
            />
          </Pad>
          <Heading>Text size</Heading>
          <Pad>
            <SliderRow
              label="Interface text"
              labId="textSizeLab"
              labInitial="Normal"
              id="textSize"
              min={0}
              max={3}
              step={1}
              defaultValue={1}
              helpAnchor="textSize"
              onHelp={onHelp}
              title="Small / Normal / Large / Extra large. Applies to sidebars, menus, the HUD and the links table - not the point cloud canvas itself. Persists across reloads."
            />
          </Pad>
          <Heading>Slicing</Heading>
          <Pad>
            <NativeSelect
              id="clipMinGap"
              size="xs"
              label={selLabel("Thinnest slice", "clipMinGap", onHelp)}
              defaultValue="5"
              title="How close the two ends of an X, Y or Z clip slider may come. 5 cm by default."
              data={[
                { value: "1", label: "1 cm" },
                { value: "2", label: "2 cm" },
                { value: "5", label: "5 cm (default)" },
                { value: "10", label: "10 cm" },
                { value: "20", label: "20 cm" },
              ]}
            />
            <NativeSelect
              id="clipStep"
              size="xs"
              mt={10}
              label={selLabel("Arrow keys move the slice by", "clipStep", onHelp)}
              defaultValue="slice"
              title="With a clip slider clicked, each arrow key press moves the whole slice by this much. Shift with an arrow always moves 1 cm."
              data={[
                { value: "slice", label: "Its own thickness (default)" },
                { value: "1", label: "1 cm" },
                { value: "5", label: "5 cm" },
                { value: "10", label: "10 cm" },
              ]}
            />
          </Pad>
        </Tabs.Panel>
      </Tabs>
      </div>
    </Modal>
  );
}

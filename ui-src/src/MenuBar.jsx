import { useEffect, useState } from "react";
import { Button, Group, Popover, Text } from "@mantine/core";
import { Item, Tg, SliderRow, Heading, Pad } from "./MenuControls.jsx";
import SettingsModal from "./SettingsModal.jsx";
import HelpModal from "./HelpModal.jsx";
import RunModal from "./RunModal.jsx";

// Top bar menus. React owns markup and open/close; the page script owns what each control does and binds to the ids
// here. Popovers stay mounted (keepMounted) because the page script clicks menu items from code. Settings, Display and
// Help are modals (SettingsModal.jsx, HelpModal.jsx).

function FileMenu() {
  return (
    <>
      <Heading first>File</Heading>
      <Pad>
        <Item id="mRescan" title="Reload the page, re-fetching every scan's preview from this project's own scans-raw folder">Rebuild previews</Item>
        <Item id="mSave">Save placement</Item>
        <Item id="mRestoreBackup" title="Every save keeps a timestamped copy of whatever it overwrote, going back to when the project was created. Pick one to bring back - restoring is itself a save, so it can be undone the same way.">Restore a backup&hellip;</Item>
        <Item id="mExport">Export merged .pts</Item>
        <Item id="mImportScans" title="Copy more .pts scans into this project, from anywhere on this computer. Your own files are never changed">Import scans&hellip;</Item>
        <Item id="mSwitchProject"title="Back to the project picker - nothing in this project is lost, it is all on disk">Switch project&hellip;</Item>
        <Item id="mWallReport" title="Measures how thick each wall reads where several scans saw it, in inches, and which scan is off by how much. Uses every point of the saved placement. A wall that reads under 2 inches is one sheet you can draw on; a thick band means the scans disagree.">Check wall thickness (inches)</Item>
        <Item id="mExportPlan" title="Cuts the placed scans at full resolution and draws the outlines as a DXF in metres, plus a picture to check it by. Uses the band of your Z clip if it is on, otherwise 1.0 to 1.6 m above the typical floor. Saved in the project's exports folder with the date and time in its name, so an earlier drawing is never overwritten.">Export plan view (DXF)</Item>
        <Item id="mExport3d" title="Every floor, ceiling piece (flat, lowered or sloped) and wall of the placed scans as a 3D plane, in one DXF in metres, plus a picture from above to check it by. Where several scans saw the same surface it is drawn once and its label says how closely they agreed. Saved in the project's exports folder with the date and time in its name.">Export floors, ceilings and walls in 3D (DXF)</Item>
      </Pad>
    </>
  );
}

function EditMenu() {
  return (
    <>
      <Heading first>Edit</Heading>
      <Pad>
        <Item id="mUndo">Undo</Item>
        <Item id="mRedo">Redo</Item>
        <Item id="mClearLinks" title="Remove every anchor you drew. Undo brings them back until you save">Clear all anchors</Item>
        <Item id="mReviewScan" title="Look at the selected scan on its own and take out what does not belong: people, pets, mirror reflections, glass, stray points. With several scans selected, review walks through them one after another">Review and clean up scan&hellip;</Item>
        <Item id="mSelectAll">Select all scans</Item>
        <Item id="mDeselect" title="Collapses the selection back to just the one scan already focused - something must always be selected">Deselect the rest</Item>
        <Item id="mBatchVis" title="Shows or hides every selected scan together - whichever state most of them are not already in">Hide/show selected</Item>
        <Item id="mBatchLock" title="Locks or unlocks every selected scan together - whichever state most of them are not already in">Lock/unlock selected</Item>
        <Item id="mBatchReset" title="Puts every selected scan back exactly where you last placed it by hand, undoing anything an automatic align moved">Reset selected to my placement</Item>
        <Item id="mBatchRemove" title="Takes the selected scans out of the sidebar and viewport for the rest of this session (undoable). Does not exclude them from a Run - every scan in the folder is still stitched regardless of this">Remove selected from view</Item>
      </Pad>
    </>
  );
}

function ViewMenu() {
  return (
    <>
      <Heading first>Views</Heading>
      <Pad>
        <Group gap={3} wrap="nowrap">
          {[["plan", "Plan"], ["iso", "Iso"], ["side", "Side"]].map(([v, l]) => (
            <Button key={v} data-menuview={v} variant="default" size="xs" style={{ flex: 1 }}>{l}</Button>
          ))}
        </Group>
      </Pad>
      <Heading>Display</Heading>
      <Pad>
        <Tg id="cbCompareOriginal" label="Compare: show my original placement" />
        <Tg id="cbAsImported" label="Show as imported (read-only)" title="Read-only: draws every scan the way it looked the moment it came in, before any placement or align. Nothing moves. Turn it off to come back exactly where you were." />
        <Tg id="cbHeatmap" label="Deviation heatmap (how good is the merge?)" title="Colours every point by distance to the nearest point in another scan: green tight, red drifting. Needs a saved placement with at least two scans built." />
        <Tg id="cbAdd" label="Glow where scans agree" defaultChecked />
        <Tg id="cbCol" label="True colour" />
        <Tg id="cbDepthCue" label="Depth cue (fade points farther from you)" title="Points farther from the viewer fade, so near and far are easier to tell apart in an angled view. Display only: nothing about the scans changes. With 'Glow where scans agree' on, heavily overlapped points still add up to white, so turn the glow off to see the fade fully." />
        <Tg id="cbGrid" label="1 m grid, floor only (Z=0)" defaultChecked />
        <SliderRow label="Grid brightness" labId="gridOpLab" labInitial="100%" id="gridOpacity" min={0} max={100} defaultValue={100}
          title="How bright the floor grid draws. Display only, does not change the grid itself." />
        <Tg id="cbOrigin" label="Origin marker" defaultChecked />
        <Tg id="cbLink" label="Links between scans" defaultChecked />
        <Tg id="cbWall" label="Detected walls" defaultChecked />
        <Tg id="cbWallAll" label={"… on every scan, not just selected"} indent />
        <Tg id="cbFloorsCeil" label="Floors and ceilings (each piece and its height)"
          title="Outlines every floor and ceiling piece the tool found in the selected scan, with its height: several floor levels, flat, lowered and sloped ceiling sections. Blue is floor, orange is ceiling, grey is a counter, table or step. Found once from every point of the scan, the first time you turn this on." />
        <Tg id="cbFloorsCeilAll" label={"… on every scan, not just selected"} indent />
        <Tg id="cbLandmarks" label="Named landmarks" defaultChecked
          title="The features you named in review (windows, doorways, columns...), with a dashed line between the same name in different scans: those scans are linked on it." />
        <Tg id="cbBoxes" label="Bounding boxes (subtle)" />
        <SliderRow label="Opacity (everything)" labId="opLab" labInitial="55%" id="opacity" min={8} max={100} defaultValue={55} />
        <SliderRow label="Point size" labId="ptSizeLab" labInitial="1.4 px" id="pointSize" min={5} max={50} defaultValue={14} />
        <SliderRow label="Brightness outside the clip slice" labId="outOfClipLab" labInitial="0%" id="outOfClipOpacity" min={0} max={100} defaultValue={0}
          title="With the X/Y/Z clip tool on, points outside the active slice are hidden by default (0%). Raise this to show them dimly instead, as a ghost of the rest of the building." />
      </Pad>
      <Heading>Mouse</Heading>
      <Pad>
        <Tg id="cbMouseAssist" label="Mouse snapping / magnetism" title="Off by default. Reserved for later: the mouse may get smarter about snapping edges and corners together, but only when this is on, and only for dragging with the mouse. The keyboard never snaps." />
      </Pad>
    </>
  );
}

const MENUS = [
  ["menuFile", "File", FileMenu],
  ["menuEdit", "Edit", EditMenu],
  ["menuView", "View", ViewMenu],
];

export default function MenuBar() {
  const [open, setOpen] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [helpAnchor, setHelpAnchor] = useState(null);

  useEffect(() => {
    const close = () => setOpen(null);
    window.addEventListener("bench:closemenus", close);
    // A setting's "?" asks for the Help modal to open at that setting's entry. Settings and Help are separate roots,
    // so this goes through a window event, like "bench:closemenus".
    const openHelp = (e) => { setHelpAnchor(e.detail?.anchor ?? null); setHelpOpen(true); };
    window.addEventListener("bench:openHelp", openHelp);
    return () => {
      window.removeEventListener("bench:closemenus", close);
      window.removeEventListener("bench:openHelp", openHelp);
    };
  }, []);

  return (
    <Group gap={0} wrap="nowrap">
      {/* Empty of React children on purpose: the vanilla script writes the
          project name into it (`$('topbarTitle').textContent = ...`). */}
      <Text component="span" id="topbarTitle" className="topbar-title" size="xs" ff="var(--mono)" c="dimmed" mr={12}
        ref={(el) => { if (el && !el.dataset.init) { el.textContent = "OpenPointStitcher"; el.dataset.init = "1"; } }} />
      {MENUS.map(([id, label, Body]) => (
        <Popover
          key={id}
          opened={open === id}
          onChange={(o) => { if (!o) setOpen((cur) => (cur === id ? null : cur)); }}
          position="bottom-start"
          offset={2}
          shadow="lg"
          radius="xs"
          withinPortal
          keepMounted
          trapFocus={false}
          returnFocus={false}
          transitionProps={{ duration: 0 }}
        >
          <Popover.Target>
            <Button
              className="menubtn"
              data-menu={id}
              variant={open === id ? "light" : "subtle"}
              color="gray"
              size="compact-sm"
              onClick={() => setOpen((cur) => (cur === id ? null : id))}
            >
              {label}
            </Button>
          </Popover.Target>
          <Popover.Dropdown id={id} p={0} w={270} style={{ maxHeight: "75vh", overflowY: "auto", zoom: "var(--ui-zoom)" }}>
            <Body />
          </Popover.Dropdown>
        </Popover>
      ))}
      <Button
        className="menubtn"
        data-menu="menuSettings"
        variant={settingsOpen ? "light" : "subtle"}
        color="gray"
        size="compact-sm"
        onClick={() => { setOpen(null); setSettingsOpen(true); }}
      >
        Settings
      </Button>
      <Button
        className="menubtn"
        data-menu="menuHelp"
        variant={helpOpen ? "light" : "subtle"}
        color="gray"
        size="compact-sm"
        onClick={() => { setOpen(null); setHelpAnchor(null); setHelpOpen(true); }}
      >
        Help
      </Button>
      <SettingsModal
        opened={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onHelp={(anchor) => { setSettingsOpen(false); setHelpAnchor(anchor); setHelpOpen(true); }}
      />
      <HelpModal opened={helpOpen} onClose={() => setHelpOpen(false)} anchor={helpAnchor} />
    </Group>
  );
}

// Undo/Redo, Run, Cancel and Save at the right of the top bar. No disabled props: the page script owns .disabled,
// and Mantine's disabled prop sets data-disabled, which the script would never clear.
export function TopBarRight() {
  const [runOpen, setRunOpen] = useState(false);
  return (
    <Group gap={6} wrap="nowrap">
      <Button id="bRunMenu" size="compact-xs" title="Choose a kind of run and start it"
        onClick={() => setRunOpen(true)}>Run&hellip;</Button>
      <Button id="bCancelJob" variant="default" size="compact-xs"
        title="Stops the running job at once, whatever it is doing. If it is already saving its results it finishes that first, about a second. Nothing from a stopped run is kept, and your placement is not changed.">
        Cancel run</Button>
      <Button id="bSave" variant="default" size="compact-xs" title="Save where every scan is, plus your anchors and exclusions">
        Save placement</Button>
      <RunModal opened={runOpen} onClose={() => setRunOpen(false)} />
      <Button id="bUndo" variant="default" size="compact-xs" title="Ctrl+Z">&#8630; Undo</Button>
      <Button id="bRedo" variant="default" size="compact-xs" title="Ctrl+Shift+Z">&#8631; Redo</Button>
    </Group>
  );
}

import { useEffect, useState } from "react";
import { Button, Group, Divider } from "@mantine/core";
import { benchReady } from "./bridge.js";

// View/tool/clip toolbar. Keep the ids, data-view/data-tool attributes and the on class: hotkeys, menus and tests use
// them. State is read from the page globals; clicks call the page's functions.
const VIEWS = [
  ["plan", "Plan", "Straight down. X/Y move here; height does not affect where a point lands on screen."],
  ["iso", "Iso", "Angled 3D view, for judging height and rough shape at a glance."],
  ["side", "Side", "Level, from the side. Best view for height (Ctrl+Up/Down, shift+drag)."],
];
const TOOLS = [
  ["navigate", "Navigate", "Pan, orbit and zoom only. NEVER moves a scan, whatever is under the cursor."],
  ["move", "Move", "Click and drag a scan to move it."],
  ["spin", "Rotate", "Drag the ring around the selected scan to spin it. The dot at the centre is the pivot it spins about."],
  ["anchor", "Anchor region", "Click 2 or more scans in the sidebar to solo them, then drag a box on each in turn - it cycles to the next one after every box. Enter commits everything drawn this way, Esc discards it and brings back whatever was visible before."],
  ["exclude", "Exclude region", "Click one or more scans in the sidebar to solo them, then drag a box on each to leave those mostly right but flawed points out of alignment (still exported unless Settings, Exporting says otherwise). Enter commits, Esc discards. Reversible once committed - see the Exclusions list."],
  ["measure", "Measure", "Plan view: click a wall, then the wall at the other end of a run you measured on site, and type the distance. The Measurements tab lists what the layout reads for each one and how far off it is."],
];
const CLIP_TITLE =
  "Clip box on/off, for every axis enabled in the Clip section of the sidebar. Works in any view, not a camera mode - see the Clip section's own '?' for detail.";

function readState() {
  if (!benchReady()) return { view: "plan", tool: "navigate", clip: false };
  return { view, tool, clip: clipMasterOn && anyClipAxisOn() };
}

function ToolButton({ on, className = "", children, ...rest }) {
  return (
    <Button
      size="compact-xs"
      radius="xs"
      variant={on ? "filled" : "default"}
      className={`${className}${on ? " on" : ""}`}
      {...rest}
    >
      {children}
    </Button>
  );
}

export default function Toolbar() {
  const [s, setS] = useState(readState);

  useEffect(() => {
    let raf;
    const tick = () => {
      const next = readState();
      setS((prev) =>
        prev.view === next.view && prev.tool === next.tool && prev.clip === next.clip ? prev : next,
      );
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <Group gap={4} wrap="wrap" justify="center">
      {VIEWS.map(([id, label, title]) => (
        <ToolButton key={id} on={s.view === id} data-view={id} title={title} onClick={() => chooseView(id)}>
          {label}
        </ToolButton>
      ))}
      <ToolButton id="clipToggleBtn" on={s.clip} data-toggle="clip" title={CLIP_TITLE} onClick={() => toggleClipMaster()}>
        Clip
      </ToolButton>
      <Divider orientation="vertical" mx={2} />
      {TOOLS.map(([id, label, title]) => (
        <ToolButton key={id} on={s.tool === id} data-tool={id} title={title} onClick={() => chooseTool(id)}>
          {label}
        </ToolButton>
      ))}
    </Group>
  );
}

import { Button, Modal, NativeSelect, Stack, Text } from "@mantine/core";

// The top bar's Run opens this to pick the kind of run. Choosing one sets the hidden #runKind and presses the hidden
// #bRun, so the page script's run code and disabled state are unchanged. keepMounted: the page script binds
// #runKind and #bRun once, at load.
export const RUN_KINDS = [
  ["align_auto_seeded", "Run from my alignment",
    "Starts from where you put each scan and fine-tunes from there. The usual choice."],
  ["align_auto_seeded_anchors", "Run from my alignment, using my anchors too",
    "The same, and the anchors you drew must line up as well."],
  ["align", "Align from my anchors only",
    "Moves scans so the anchors you drew line up. Nothing else is used."],
  ["align_auto", "Automatic align",
    "Ignores where you put the scans and solves from their own walls, floors and corners. The slowest."],
  ["tilt", "Suggest tilt and height",
    "Matches floors and ceilings between scans and suggests small tilt and height fixes for you to accept or not."],
  ["export", "Export merged points (.pts)",
    "Writes every placed scan into one point file, full resolution and full colour."],
  ["plan_dxf", "Export plan view (DXF)",
    "Cuts the placed scans at full resolution and draws the outlines as a DXF in metres."],
  ["surfaces_3d", "Export floors, ceilings and walls in 3D (DXF)",
    "Every floor, ceiling piece (flat, lowered or sloped) and wall as a 3D plane in metres, drawn once where scans overlap."],
  ["wall_report", "Check wall thickness (inches)",
    "Measures how thick each wall reads where several scans saw it."],
  ["revert", "Put everything back where I placed it",
    "Every scan returns to your own placement, undoing what runs moved."],
];

export default function RunModal({ opened, onClose }) {
  const start = (kind) => {
    onClose();
    const sel = document.getElementById("runKind"), run = document.getElementById("bRun");
    if (!sel || !run || run.disabled) return;
    sel.value = kind;
    run.click();
  };
  return (
    <Modal opened={opened} onClose={onClose} keepMounted title="What kind of run?" size="md" radius="xs" shadow="lg"
      withinPortal trapFocus={false} returnFocus={false} transitionProps={{ duration: 0 }} id="runModal"
      styles={{ content: { zoom: "var(--ui-zoom)" } }}>
      <div data-menu-modal="runModal">
        <Stack gap={6}>
          {RUN_KINDS.map(([kind, label, help]) => (
            <Button key={kind} data-run-kind={kind} variant="default" fullWidth justify="flex-start" h="auto" py={6}
              title={help} onClick={() => start(kind)}>
              <div style={{ textAlign: "left", whiteSpace: "normal" }}>
                <Text size="sm" fw={600}>{label}</Text>
                <Text size="xs" c="dimmed">{help}</Text>
              </div>
            </Button>
          ))}
        </Stack>
        {/* Hidden: the vanilla script reads #runKind and owns #bRun's click and disabled state. */}
        <div style={{ display: "none" }}>
          <NativeSelect id="runKind" data={RUN_KINDS.map(([value, label]) => ({ value, label }))} />
          <button type="button" id="bRun">Run</button>
        </div>
      </div>
    </Modal>
  );
}

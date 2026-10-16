import { useEffect, useState } from "react";
import { Modal, Tabs, Text } from "@mantine/core";
import { Heading } from "./MenuControls.jsx";
import { HELP, SETTINGS_HELP } from "./helpText.js";

// Help modal, with a "Settings, explained" tab of one entry per setting so each setting's "?" can open its entry.
export default function HelpModal({ opened, onClose, anchor }) {
  const [tab, setTab] = useState("keyboard");

  // A setting's "?" sets anchor and opens this modal: jump to the settings tab and scroll the entry into view.
  // Runs on every change, so a second "?" while the modal is open still jumps.
  useEffect(() => {
    if (!opened || !anchor) return;
    setTab("settings");
    const id = "help-" + anchor;
    // Wait a tick for the tab switch to actually mount/show that panel.
    const t = setTimeout(() => {
      document.getElementById(id)?.scrollIntoView({ block: "start" });
    }, 0);
    return () => clearTimeout(t);
  }, [opened, anchor]);

  const note = (k, extra = {}) => (
    <Text className="note" size="xs" c="dimmed" px="sm" pb={10} style={{ lineHeight: 1.5 }} {...extra}
      dangerouslySetInnerHTML={{ __html: HELP[k].html }} />
  );

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      keepMounted
      title="Help"
      size="lg"
      radius="xs"
      shadow="lg"
      withinPortal
      trapFocus={false}
      returnFocus={false}
      transitionProps={{ duration: 0 }}
      id="helpModal"
      styles={{ body: { padding: 0 }, content: { zoom: "var(--ui-zoom)" } }}
    >
      {/* Own marker, not a forwarded Mantine prop - see SettingsModal.jsx's
          matching comment. */}
      <div data-menu-modal="menuHelp">
      <Tabs value={tab} onChange={setTab} orientation="vertical" keepMounted>
        <Tabs.List>
          <Tabs.Tab value="keyboard">Keyboard</Tabs.Tab>
          <Tabs.Tab value="shortcuts">Tool and view shortcuts</Tabs.Tab>
          <Tabs.Tab value="how">How this tool works</Tabs.Tab>
          <Tabs.Tab value="settings">Settings, explained</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="keyboard" p={0}>
          <Heading first>Keyboard</Heading>
          {note("Keyboard")}
        </Tabs.Panel>

        <Tabs.Panel value="shortcuts" p={0}>
          <Heading first>Tool and view shortcuts</Heading>
          {/* Filled by the vanilla script from its HOTKEYS table, so this can
              never drift from what the keys really do - empty here on purpose. */}
          <Text id="hotkeyHelp" className="note" size="xs" c="dimmed" px="sm" pb={10} />
        </Tabs.Panel>

        <Tabs.Panel value="how" p={0}>
          <Heading first>How this tool works</Heading>
          {note("How this tool works")}
        </Tabs.Panel>

        <Tabs.Panel value="settings" p={0}>
          <Heading first>Settings, explained</Heading>
          {SETTINGS_HELP.map((s) => (
            <div key={s.anchor} id={"help-" + s.anchor}>
              <Text component="h3" size="xs" fw={700} px="sm" pt={8} pb={2}>{s.heading}</Text>
              <Text className="note" size="xs" c="dimmed" px="sm" pb={10} style={{ lineHeight: 1.5 }}
                dangerouslySetInnerHTML={{ __html: s.html }} />
            </div>
          ))}
        </Tabs.Panel>
      </Tabs>
      </div>
    </Modal>
  );
}

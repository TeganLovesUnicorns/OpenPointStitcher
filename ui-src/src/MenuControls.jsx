import { Button, Checkbox, Group, Text } from "@mantine/core";
import SettingSlider from "./SettingSlider.jsx";

// Small building blocks shared by every menu and tab body (File/Edit/View popovers, Settings and Help modals).

export const ITEM = { variant: "default", fullWidth: true, size: "xs", justify: "flex-start", mb: 5 };

export function Item({ id, title, children }) {
  return (
    <Button id={id} title={title} {...ITEM}>
      {children}
    </Button>
  );
}

// A label whose text the vanilla code rewrites (`$('opLab').textContent = ...`).
// It must have NO React-managed children or React's next render would fight that
// write; the starting text is set once from a ref instead.
export function LiveText({ id, initial }) {
  return (
    <Text
      component="span"
      id={id}
      size="xs"
      ff="var(--mono)"
      c="dimmed"
      ref={(el) => {
        if (el && !el.dataset.init) { el.textContent = initial; el.dataset.init = "1"; }
      }}
    />
  );
}

export function Tg({ id, label, title, defaultChecked = false, indent = false, helpAnchor, onHelp }) {
  return (
    <div data-setting={helpAnchor || id} title={title} style={{ marginLeft: indent ? 18 : 0, marginBottom: 6 }}>
      <Group gap={4} wrap="nowrap" align="center">
        <Checkbox id={id} size="xs" label={label} defaultChecked={defaultChecked} />
        {helpAnchor && <HelpLink anchor={helpAnchor} onHelp={onHelp} />}
      </Group>
    </div>
  );
}

export function SliderRow({ label, labId, labInitial, helpAnchor, onHelp, title, ...slider }) {
  return (
    <div data-setting={helpAnchor || slider.id}>
      <Group justify="space-between" mt={6} mb={2}>
        <Group gap={4}>
          <Text size="xs" ff="var(--mono)" c="dimmed" title={title}>{label}</Text>
          {helpAnchor && <HelpLink anchor={helpAnchor} onHelp={onHelp} />}
        </Group>
        <LiveText id={labId} initial={labInitial} />
      </Group>
      <SettingSlider title={title} {...slider} />
    </div>
  );
}

export function Heading({ children, first }) {
  return (
    <Text
      component="h2"
      size="xs"
      tt="uppercase"
      c="dimmed"
      px="sm"
      pt={first ? 8 : 10}
      pb={6}
      mt={first ? 0 : 6}
      style={{ letterSpacing: ".09em", ...(first ? {} : { borderTop: "1px solid var(--line)" }) }}
    >
      {children}
    </Text>
  );
}

export const Pad = ({ children }) => <div style={{ padding: "0 12px 10px" }}>{children}</div>;

// A "?" beside a setting, in the Settings modal only (onHelp is set there), opening its Help entry.
export function HelpLink({ anchor, onHelp }) {
  if (!onHelp) return null;
  return (
    <button
      type="button"
      className="mc-help-link"
      title="More about this setting"
      aria-label="More about this setting"
      onClick={(e) => { e.preventDefault(); e.stopPropagation(); onHelp(anchor); }}
    >
      ?
    </button>
  );
}

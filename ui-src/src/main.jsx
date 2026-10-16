import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import { MantineProvider, createTheme } from "@mantine/core";
// The @layer build, not styles.css: unlayered page rules beat anything in a cascade layer, whereas the plain
// stylesheet's global body rules would override the page's base font size and background.
import "@mantine/core/styles.layer.css";
import ScaleBar from "./ScaleBar.jsx";
import ZoomSlider from "./ZoomSlider.jsx";
import ClipAxisSlider from "./ClipAxisSlider.jsx";
import Toolbar from "./Toolbar.jsx";
import ScanList from "./ScanList.jsx";
import MenuBar, { TopBarRight } from "./MenuBar.jsx";
import { RightSidebar, SelBar, BottomPanel } from "./Panels.jsx";

// The look, mirrored from placement_bench.html's :root variables (graphite neutrals, one blue accent, square
// corners, the OS UI font). Those variables are the source of truth; change both together.
const theme = createTheme({
  primaryColor: "bench",
  primaryShade: 5,
  colors: {
    bench: [
      "#e8f1ff", "#cfe1ff", "#9fc3ff", "#6da4ff", "#4a8fff",
      "#3d8bfd", "#2f76e0", "#2461c0", "#1a4c9c", "#103878",
    ],
    dark: [
      "#e4e7eb", "#c3c8cf", "#a2a8b1", "#8a919b", "#5a616b",
      "#2c3037", "#23262b", "#1a1d21", "#121417", "#0b0e12",
    ],
  },
  defaultRadius: "xs",
  fontFamily: 'var(--sans, "Segoe UI", system-ui, sans-serif)',
  fontFamilyMonospace: 'var(--mono, Consolas, monospace)',
  fontSizes: { xs: "11px", sm: "12px", md: "13px", lg: "14px", xl: "16px" },
});

// One root per mount point; a missing container is skipped. flushSync is required: the main script binds handlers
// to these ids at load.
function mount(id, Component, props = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  const root = createRoot(el);
  flushSync(() => {
    root.render(
      <MantineProvider theme={theme} forceColorScheme="dark" withGlobalClasses={false}>
        <Component {...props} />
      </MantineProvider>,
    );
  });
}

// Static chrome first (the app script binds to these at load) ...
mount("rc-menubar", MenuBar);
mount("tools", Toolbar);
mount("rc-topbar-right", TopBarRight);
mount("selbar", SelBar);
mount("rc-right-root", RightSidebar);
mount("bottom", BottomPanel);
// ... then the dynamic pieces.
mount("scanlist", ScanList);
mount("rc-scalebar-root", ScaleBar);
mount("rc-zoomslider-root", ZoomSlider);

// X along the bottom edge (horizontal), Y along the left edge and Z along the right (vertical), 22 px in so each
// edge's panel collapse tab sits outside them.
mount("rc-clip-x-root", ClipAxisSlider, {
  axis: "x", orientation: "horizontal",
  style: { bottom: 22, left: "50%", transform: "translateX(-50%)" },
});
mount("rc-clip-y-root", ClipAxisSlider, {
  axis: "y", orientation: "vertical",
  style: { left: 22, top: "50%", transform: "translateY(-50%)" },
});
mount("rc-clip-z-root", ClipAxisSlider, {
  axis: "z", orientation: "vertical",
  style: { right: 22, top: "50%", transform: "translateY(-50%)" },
});

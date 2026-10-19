import { useRef, useState } from "react";
import { ActionIcon, TextInput } from "@mantine/core";
import { useBenchChanges, benchReady } from "./bridge.js";

// Scan rows. Re-reads the page globals (S, drawOrder, sel, selection) each render; copies no state.

// Feather icon paths (MIT), inline so nothing is fetched at runtime.
const svgProps = { viewBox: "0 0 24 24", width: 13, height: 13, fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round", strokeLinejoin: "round" };
const IconLocked = () => (<svg {...svgProps}><rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" /></svg>);
const IconUnlocked = () => (<svg {...svgProps}><rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 9.9-1" /></svg>);
const IconShown = () => (<svg {...svgProps}><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" /></svg>);
const IconHidden = () => (
  <svg {...svgProps}>
    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
    <line x1="1" y1="1" x2="23" y2="23" />
  </svg>
);

function ScanRow({ s, pos, renaming, setRenaming, dragPos }) {
  const cls = s.i === sel ? "sel" : selection.has(s.i) ? "multisel" : "";
  const commitRename = (value) => {
    s.alias = value.trim();
    setRenaming(null);
    renderAll();
  };

  return (
    <div
      className={`scan ${cls}`}
      data-i={s.i}
      data-pos={pos}
      draggable="true"
      onClick={(ev) => {
        // The whole row is the click target, name included; a double-click to rename selects first.
        selectScan(s.i, ev.shiftKey, ev.ctrlKey || ev.metaKey);
        renderAll();
      }}
      onContextMenu={(ev) => {
        ev.preventDefault();
        openScanContextMenu(s.i, ev.clientX, ev.clientY);
      }}
      onDragStart={() => { dragPos.current = pos; }}
      onDragOver={(ev) => ev.preventDefault()}
      onDrop={(ev) => {
        // Drag to reorder: changes draw order and which scan is the reference only. It splices drawOrder, never S,
        // links or exclusions, so no geometry moves because a row moved.
        ev.preventDefault();
        if (dragPos.current === null || dragPos.current === pos) return;
        const moved = drawOrder.splice(dragPos.current, 1)[0];
        drawOrder.splice(pos, 0, moved);
        if (pos === 0 || dragPos.current === 0) referenceChosen = true;   // the top changed: the user chose a reference
        dragPos.current = null;
        renderAll();
      }}
    >
      <div className="sw" style={{ background: `rgb(${s.col})` }} />
      <div className="nm">
        {pos === 0 && <span title={referenceChosen
          ? "reference: every run keeps this scan still and aligns the rest to it. Drag another scan to the top to change it"
          : "reference: runs keep this scan (the first by name) still. Drag another scan to the top to choose your own"}>&#9733; </span>}
        {renaming === s.i ? (
          <TextInput
            autoFocus
            size="xs"
            defaultValue={s.alias}
            placeholder={s.name}
            classNames={{ input: "num" }}
            onClick={(ev) => ev.stopPropagation()}
            onFocus={(ev) => ev.currentTarget.select()}
            onBlur={(ev) => commitRename(ev.currentTarget.value)}
            onKeyDown={(ev) => {
              if (ev.key === "Enter") { ev.preventDefault(); ev.currentTarget.blur(); }
              else if (ev.key === "Escape") { ev.preventDefault(); setRenaming(null); renderAll(); }
            }}
          />
        ) : (
          <span
            className="scanname"
            data-rename={s.i}
            title={s.alias ? `file: ${s.name} (double click to rename)` : "double click to rename"}
            onDoubleClick={(ev) => { ev.stopPropagation(); setRenaming(s.i); }}
          >
            {displayName(s)}
          </span>
        )}
        {s.alignMovedM > 0.001 && (
          <span
            className="scan-moved-badge"
            title={`moved by align: ${s.alignMovedM.toFixed(2)}m, ${s.alignMovedDeg.toFixed(1)}deg from where you placed it`}
          >
            &#9650; {s.alignMovedM.toFixed(2)}m
          </span>
        )}
        <div className="mini">
          {s.points.toLocaleString()} pts{" "}
          {s.buildError ? (
            <span style={{ color: "var(--bad)" }} title={s.buildError}>preview failed</span>
          ) : !s.ready ? (
            <span style={{ color: "var(--accent)" }}>building preview&hellip;</span>
          ) : s.tilt > 3 ? (
            <span style={{ color: "var(--warn)" }}>tilt {s.tilt}&deg;</span>
          ) : null}
        </div>
      </div>
      <ActionIcon
        variant={s.lock ? "light" : "subtle"}
        color={s.lock ? "yellow" : "gray"}
        size="sm"
        data-lock={s.i}
        aria-label={s.lock ? "Locked" : "Unlocked"}
        title={s.lock ? "Locked: align will not move this scan. Click to unlock" : "Unlocked: align may move this scan. Click to lock"}
        onClick={(ev) => { ev.stopPropagation(); s.lock = !s.lock; renderAll(); }}
      >
        {s.lock ? <IconLocked /> : <IconUnlocked />}
      </ActionIcon>
      <ActionIcon
        variant={s.vis ? "subtle" : "light"}
        color={s.vis ? "gray" : "yellow"}
        size="sm"
        data-vis={s.i}
        aria-label={s.vis ? "Shown" : "Hidden"}
        title={s.vis ? "Shown. Click to hide" : "Hidden. Click to show"}
        onClick={(ev) => { ev.stopPropagation(); s.vis = !s.vis; glRebuildPoints(); renderAll(); }}
      >
        {s.vis ? <IconShown /> : <IconHidden />}
      </ActionIcon>
    </div>
  );
}

export default function ScanList() {
  useBenchChanges();
  const [renaming, setRenaming] = useState(null);
  const dragPos = useRef(null);
  if (!benchReady()) return null;

  // Hidden scans are skipped, not spliced out of drawOrder, so pos stays a true index for the drag handler.
  return (
    <>
      {drawOrder.map((idx, pos) => {
        const s = S[idx];
        if (!s || s.deleted) return null;
        return <ScanRow key={s.i} s={s} pos={pos} renaming={renaming} setRenaming={setRenaming} dragPos={dragPos} />;
      })}
    </>
  );
}

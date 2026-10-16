import { useEffect, useRef, useState } from "react";
import { Slider } from "@mantine/core";

// Mantine face for a hidden native <input type=range> that stays the source of truth; drags fire a real input event.
// The page script writes the label into an empty span rendered here: do not give it children.
export default function SettingSlider({ id, min, max, step = 1, defaultValue, title }) {
  const inputRef = useRef(null);
  const [v, setV] = useState(defaultValue);
  const [disabled, setDisabled] = useState(false);

  useEffect(() => {
    let raf;
    const tick = () => {
      const el = inputRef.current;
      if (el) {
        const n = Number(el.value);
        setV((p) => (p === n ? p : n));
        setDisabled((p) => (p === el.disabled ? p : el.disabled));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <>
      <input ref={inputRef} type="range" id={id} min={min} max={max} step={step} defaultValue={defaultValue} hidden />
      <Slider
        min={min}
        max={max}
        step={step}
        value={v}
        size="xs"
        label={null}
        disabled={disabled}
        title={title}
        onChange={(n) => {
          const el = inputRef.current;
          el.value = String(n);
          el.dispatchEvent(new Event("input", { bubbles: true }));
          setV(n);
        }}
      />
    </>
  );
}

import { useSyncExternalStore } from "react";

// How React learns the page changed: notifyUI() bumps this store and components re-read globals. The listener
// attaches at bundle load, not in an effect, because the bundle runs before the main script.
// hard = synchronous re-render; soft = one bump per animation frame (canvas drags).
let version = 0;
const listeners = new Set();
let softPending = false;

function emit() {
  version += 1;
  listeners.forEach((l) => l());
}

window.addEventListener("bench:changed", (e) => {
  if (e.detail && e.detail.soft) {
    if (softPending) return;
    softPending = true;
    requestAnimationFrame(() => {
      softPending = false;
      emit();
    });
  } else {
    emit();
  }
});

function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
const getVersion = () => version;

// True once the page script has defined its globals (it sets window.benchReady just before boot()). Before that a
// component could read a let still in its temporal dead zone, which throws even for typeof, so components render
// defaults until this is true.
export const benchReady = () => window.benchReady === true;

export function useBenchChanges() {
  useSyncExternalStore(subscribe, getVersion);
}

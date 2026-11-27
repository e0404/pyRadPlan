/**
 * GUI presentation mode (Qt `widgets/_ui_mode.py`, `settings.gui.ui_mode`): "default" shows every
 * control, "educational" greys out the advanced ones (descriptors with `advanced: true`, values
 * kept). Held by the frontend; `ioActions.chooseUiMode` mirrors it to `app.settings_set`.
 */
import { create } from "zustand";

export type UiMode = "default" | "educational";

export const UI_MODES: readonly UiMode[] = ["default", "educational"];

export const UI_MODE_TIPS: Record<UiMode, string> = {
  default: "Show every control",
  educational: "Reduce the widgets to the basic operations (e.g. beam angles) for teaching",
};

export const useUiMode = create<{ uiMode: UiMode }>(() => ({ uiMode: "default" }));

export function setUiMode(mode: UiMode): void {
  useUiMode.setState({ uiMode: mode });
}

/** Whether a field with this descriptor is locked in the given mode. */
export function lockedByUiMode(field: { advanced?: boolean } | undefined, mode: UiMode): boolean {
  return mode === "educational" && Boolean(field?.advanced);
}

/** Hook form of {@link lockedByUiMode} for plan-form fields. */
export function useUiLocked(field: { advanced?: boolean } | undefined): boolean {
  return useUiMode((s) => lockedByUiMode(field, s.uiMode));
}

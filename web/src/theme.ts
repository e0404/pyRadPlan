/**
 * Light / dark colour scheme (Qt `menus/_view.py`): the page follows the platform's
 * `prefers-color-scheme` until the user picks Dark Mode in the View menu; the choice is not
 * persisted. The scheme is applied as `data-theme` on `<html>`; `theme.css` holds the colours.
 */
import { create } from "zustand";
import "./theme.css";

export interface ThemeState {
  dark: boolean;
  /** The user's choice; `null` while following the platform. */
  requested: boolean | null;
}

const QUERY = "(prefers-color-scheme: dark)";

function platformDark(): boolean {
  try {
    return globalThis.matchMedia?.(QUERY).matches ?? false;
  } catch {
    return false;
  }
}

export const useTheme = create<ThemeState>(() => ({ dark: platformDark(), requested: null }));

function apply(dark: boolean): void {
  const root = globalThis.document?.documentElement;
  if (!root) return;
  root.dataset.theme = dark ? "dark" : "light";
  root.style.colorScheme = dark ? "dark" : "light";
}

/** The user switched the scheme (stops following the platform). */
export function setDarkMode(dark: boolean): void {
  useTheme.setState({ dark, requested: dark });
}

/** Follow the platform again. */
export function followPlatformScheme(): void {
  useTheme.setState({ dark: platformDark(), requested: null });
}

let started = false;

/** Apply the scheme to the page and keep following the platform; idempotent. */
export function initTheme(): void {
  if (started) return;
  started = true;
  apply(useTheme.getState().dark);
  useTheme.subscribe((s) => apply(s.dark));
  try {
    globalThis.matchMedia?.(QUERY).addEventListener("change", (e) => {
      if (useTheme.getState().requested === null) useTheme.setState({ dark: e.matches });
    });
  } catch {
    /* no matchMedia (tests) */
  }
}

initTheme();

import { Fragment, useEffect, useMemo, useState } from "react";
import type { SettingsSectionId } from "../api/protocol/io";
import { actions, useApp } from "../store/appStore";
import {
  chooseAndAddBaseData,
  chooseAndLoadFile,
  chooseUiMode,
  installWindowHooks,
  refreshPhantoms,
  saveCst,
  saveDij,
  savePlan,
  saveResult,
  saveWorkspace,
} from "../store/ioActions";
import { useIo } from "../store/ioStore";
import { useUiMode, type UiMode } from "../store/uiMode";
import { setDarkMode, useTheme } from "../theme";
import { openBaseDataDialog } from "./basedata/BaseDataDialog";
import { openAboutDialog } from "./help/AboutDialog";
import { openLicensesDialog } from "./help/LicensesDialog";
import { DialogHost } from "./io/DialogHost";
import { buildMenus, type MenuDefinition, type MenuEntry, type MenuItem } from "./io/menuModel";
import { openPreferences } from "./settings/PreferencesDialog";
import "./io/io.css";

/** Title with the Qt mnemonic: "&File" -> <u>F</u>ile. */
function Title(props: { text: string }) {
  const at = props.text.indexOf("&");
  if (at < 0) return <>{props.text}</>;
  return (
    <>
      {props.text.slice(0, at)}
      <u>{props.text.charAt(at + 1)}</u>
      {props.text.slice(at + 2)}
    </>
  );
}

function mnemonicOf(menu: MenuDefinition): string {
  const at = menu.title.indexOf("&");
  return at >= 0 ? menu.title.charAt(at + 1).toLowerCase() : "";
}

function EntryView(props: { entry: MenuEntry; run: (item: MenuItem) => void }) {
  const { entry } = props;
  const [open, setOpen] = useState(false);
  if (entry.kind === "separator") return <div className="sep" />;
  if (entry.kind === "submenu") {
    return (
      <div className="submenu-wrap" onMouseEnter={() => entry.enabled && setOpen(true)} onMouseLeave={() => setOpen(false)}>
        <button
          className={`item has-sub ${entry.enabled ? "" : "disabled"} ${open ? "open" : ""}`}
          aria-disabled={!entry.enabled}
          aria-haspopup="menu"
          aria-expanded={open}
          title={entry.tooltip}
          data-testid={`menu-item-${entry.id}`}
          onClick={(e) => {
            e.stopPropagation();
            if (entry.enabled) setOpen(!open);
          }}
        >
          {entry.label}
          <span className="arrow">▸</span>
        </button>
        {open && (
          <div className="menu-popup sub" role="menu" data-testid={`submenu-${entry.id}`}>
            {entry.entries.map((child) => (
              <EntryView key={child.id} entry={child} run={props.run} />
            ))}
          </div>
        )}
      </div>
    );
  }
  return (
    <button
      className={`item ${entry.enabled ? "" : "disabled"}`}
      role={entry.checked === undefined ? "menuitem" : entry.radio ? "menuitemradio" : "menuitemcheckbox"}
      aria-checked={entry.checked}
      aria-disabled={!entry.enabled}
      title={entry.tooltip}
      data-testid={`menu-item-${entry.id}`}
      onClick={(e) => {
        if (!entry.enabled) {
          e.stopPropagation();
          return;
        }
        props.run(entry);
      }}
    >
      {entry.checked && <span className="check-mark">{entry.radio ? "●" : "✓"}</span>}
      {entry.label}
    </button>
  );
}

/** Menu bar mirroring `menus/` of the Qt GUI; the model lives in `io/menuModel.ts`. */
export function MenuBar() {
  const [open, setOpen] = useState<string | null>(null);
  const info = useApp((s) => s.info);
  const state = useApp((s) => s.state);
  const busy = useApp((s) => s.busy);
  const showLog = useApp((s) => s.showLogPanel);
  const phantoms = useIo((s) => s.phantoms);
  const uiMode = useUiMode((s) => s.uiMode);
  const dark = useTheme((s) => s.dark);

  const menus = useMemo(
    () => buildMenus({ info, state, busy, phantoms, uiMode, dark, showLog }),
    [info, state, busy, phantoms, uiMode, dark, showLog],
  );

  useEffect(() => installWindowHooks(), []);

  useEffect(() => {
    const close = () => setOpen(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(null);
      if (e.altKey && !e.ctrlKey && !e.metaKey && e.key.length === 1) {
        const menu = menus.find((m) => mnemonicOf(m) === e.key.toLowerCase());
        if (menu && menu.enabled) {
          e.preventDefault();
          setOpen(menu.id);
        }
      }
    };
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", onKey);
    };
  }, [menus]);

  useEffect(() => {
    if (open === "file") void refreshPhantoms();
  }, [open]);

  const handlers: Record<string, (arg?: string) => void> = {
    loadFile: () => void chooseAndLoadFile(),
    loadPhantom: (name) => name && void actions.loadPhantom(name),
    savePlan: () => void savePlan(),
    saveDij: () => void saveDij(),
    saveCst: () => void saveCst(),
    saveResult: () => void saveResult(),
    saveWorkspace: () => void saveWorkspace(),
    uiMode: (mode) => void chooseUiMode(mode as UiMode),
    toggleLog: () => useApp.setState({ showLogPanel: !showLog }),
    toggleDark: () => setDarkMode(!dark),
    showBaseData: () => void openBaseDataDialog(),
    addMachine: () => void chooseAndAddBaseData("machines"),
    addRbe: () => void chooseAndAddBaseData("rbe_tables"),
    settings: (section) => void openPreferences((section as SettingsSectionId | undefined) ?? null),
    openUrl: (url) => url && window.open(url, "_blank", "noopener,noreferrer"),
    licenses: () => void openLicensesDialog(),
    about: () => void openAboutDialog(info),
  };

  const run = (item: MenuItem) => {
    setOpen(null);
    handlers[item.action]?.(item.arg);
  };

  const focusItem = (menuEl: HTMLElement, delta: 1 | -1) => {
    const items = [...menuEl.querySelectorAll<HTMLElement>(":scope > .item:not(.disabled), :scope > .submenu-wrap > .item:not(.disabled)")];
    if (!items.length) return;
    const at = items.indexOf(document.activeElement as HTMLElement);
    items[(at + delta + items.length) % items.length].focus();
  };

  return (
    <Fragment>
      <nav className="menubar" data-testid="menubar" role="menubar">
        {menus.map((menu) => {
          const isOpen = open === menu.id;
          return (
            <div key={menu.id} className={`menu ${isOpen ? "open" : ""}`}>
              <button
                className={menu.enabled ? "" : "disabled"}
                aria-disabled={!menu.enabled}
                aria-haspopup="menu"
                aria-expanded={isOpen}
                data-testid={`menu-${menu.id}`}
                onMouseDown={(e) => {
                  e.stopPropagation();
                  if (menu.enabled) setOpen(isOpen ? null : menu.id);
                }}
                onMouseEnter={() => open && menu.enabled && setOpen(menu.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " " || e.key === "ArrowDown") {
                    e.preventDefault();
                    if (menu.enabled) setOpen(menu.id);
                  }
                }}
              >
                <Title text={menu.title} />
              </button>
              {isOpen && (
                <div
                  className="menu-popup"
                  role="menu"
                  data-testid={`popup-${menu.id}`}
                  onMouseDown={(e) => e.stopPropagation()}
                  onKeyDown={(e) => {
                    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
                      e.preventDefault();
                      focusItem(e.currentTarget, e.key === "ArrowDown" ? 1 : -1);
                    } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
                      const at = menus.findIndex((m) => m.id === menu.id);
                      const next = menus[(at + (e.key === "ArrowRight" ? 1 : menus.length - 1)) % menus.length];
                      if (next.enabled) setOpen(next.id);
                    }
                  }}
                >
                  {menu.entries.map((entry) => (
                    <EntryView key={entry.id} entry={entry} run={run} />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </nav>
      <DialogHost />
    </Fragment>
  );
}

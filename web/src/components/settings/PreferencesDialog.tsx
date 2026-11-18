import { useState } from "react";
import { BackendError, type ValidationDetail } from "../../api";
import type { SettingsSection, SettingsSectionId } from "../../api/protocol/io";
import { storeInternals } from "../../store/appStore";
import { showDialog } from "../../store/dialogStore";
import { chooseUiMode } from "../../store/ioActions";
import { useUiMode } from "../../store/uiMode";
import { ModalDialog } from "../forms/ModalDialog";
import { showCritical } from "../io/messageBox";
import { orderSections, type SettingsValues } from "./model";
import { SettingsForm } from "./SettingsForm";

function PreferencesView(props: {
  sections: SettingsSection[];
  only: SettingsSectionId | null;
  close: (applied: boolean) => void;
  isTop: () => boolean;
}) {
  const shown = props.only ? props.sections.filter((s) => s.id === props.only) : props.sections;
  const [tab, setTab] = useState<SettingsSectionId>(shown[0]?.id ?? "general");
  const [edits, setEdits] = useState<Record<string, SettingsValues>>({});
  const [invalid, setInvalid] = useState<Record<string, string>>({});
  const [serverStatus, setServerStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const status = serverStatus || Object.values(invalid).find(Boolean) || "";

  const apply = async () => {
    if (status) return;
    setBusy(true);
    try {
      for (const section of props.sections) {
        const changes = edits[section.id];
        if (!changes || !Object.keys(changes).length) continue;
        if (section.id === "gui") {
          const mode = changes.ui_mode;
          if (mode === "default" || mode === "educational") await chooseUiMode(mode);
          continue;
        }
        await storeInternals.call("app.settings_set", { section: section.id, values: changes });
      }
      props.close(true);
    } catch (err) {
      if (err instanceof BackendError && Array.isArray(err.details)) {
        const first = (err.details as ValidationDetail[])[0];
        setServerStatus(first ? `${first.field}: ${first.message}` : err.message);
      } else {
        setServerStatus(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setBusy(false);
    }
  };

  const single = props.only !== null;
  const title = single ? `${shown[0]?.title ?? ""} Settings` : "Preferences";
  const active = shown.find((s) => s.id === tab) ?? shown[0];
  return (
    <ModalDialog
      title={title}
      testId="preferences-dialog"
      width={single ? 460 : 520}
      onClose={() => props.close(false)}
      isTop={props.isTop}
      buttons={
        <>
          <button className="btn primary" disabled={busy} onClick={() => void apply()} data-testid="preferences-ok">
            OK
          </button>
          <button className="btn" onClick={() => props.close(false)} data-testid="preferences-cancel">
            Cancel
          </button>
        </>
      }
    >
      {!single && (
        <div className="tabs" role="tablist">
          {shown.map((s) => (
            <button key={s.id} role="tab" aria-selected={tab === s.id} className={`tab ${tab === s.id ? "active" : ""}`} onClick={() => setTab(s.id)} data-testid={`preferences-tab-${s.id}`}>
              {s.id === "general" ? "General" : s.title}
            </button>
          ))}
        </div>
      )}
      <div className="tab-panel scroll">
        {active && (
          <SettingsForm
            key={active.id}
            schema={active.form}
            onChange={(values, formStatus) => {
              setServerStatus("");
              setEdits((e) => ({ ...e, [active.id]: values }));
              setInvalid((e) => ({ ...e, [active.id]: formStatus.error ? `${formStatus.error.name}: ${formStatus.error.message}` : "" }));
            }}
          />
        )}
      </div>
      <div className="form-error" data-testid="preferences-status">
        {status}
      </div>
    </ModalDialog>
  );
}

/** Settings menu / Preferences: opens the tabbed editor, or one section's form. */
export async function openPreferences(section: SettingsSectionId | null = null): Promise<boolean> {
  let sections: SettingsSection[];
  try {
    sections = (await storeInternals.call("app.settings", {})).sections;
  } catch (err) {
    await showCritical("Preferences", err instanceof Error ? err.message : String(err));
    return false;
  }
  const ordered = orderSections(sections, useUiMode.getState().uiMode);
  return showDialog<boolean>(({ close, isTop }) => <PreferencesView sections={ordered} only={section} close={close} isTop={isTop} />);
}

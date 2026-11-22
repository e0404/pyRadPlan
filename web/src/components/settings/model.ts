import type { SettingsSection } from "../../api/protocol/io";
import type { SettingsSectionId } from "../../api/protocol/io";
import { UI_MODES, type UiMode } from "../../store/uiMode";

export type SettingsValues = Record<string, unknown>;

/** Qt tab order of the Preferences dialog. */
export const SECTION_ORDER: SettingsSectionId[] = ["general", "xp", "ai", "gui"];

export const GUI_SECTION: SettingsSection = {
  id: "gui",
  title: "GUI",
  form: {
    id: "settings:gui",
    title: "GUI settings",
    fields: [
      {
        name: "ui_mode",
        label: "Ui mode",
        type: "choice",
        default: "default",
        description:
          "Presentation mode of the GUI widgets: 'default' shows every control, 'educational' reduces the widgets to the basic operations for teaching.",
        choices: UI_MODES.map((m) => ({ value: m, label: m })),
      },
    ],
    values: { ui_mode: "default" },
    explicit: [],
  },
};

/** Sections in Qt's tab order; the GUI section always exists and shows the frontend's mode. */
export function orderSections(sections: SettingsSection[], uiMode: UiMode): SettingsSection[] {
  const byId = new Map(sections.map((s) => [s.id, s]));
  if (!byId.has("gui")) byId.set("gui", GUI_SECTION);
  const gui = byId.get("gui")!;
  byId.set("gui", { ...gui, form: { ...gui.form, values: { ...gui.form.values, ui_mode: uiMode } } });
  return SECTION_ORDER.filter((id) => byId.has(id)).map((id) => byId.get(id)!);
}

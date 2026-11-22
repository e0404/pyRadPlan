/**
 * Form of one settings section, driven by the `FieldDescriptor`s of `app.settings`: package A's
 * generic `ConfigForm`. `onChange` receives the values to send back (edited and explicit ones),
 * `Qt ConfigFormWidget.values()`. Fields the backend marks `editable: false` (browser runtime:
 * host paths, GPU, AI) are shown disabled with their reason as tooltip.
 */
import type { FormSchema } from "../../api";
import { ConfigForm, type FormStatus } from "../forms/ConfigForm";

export function SettingsForm(props: {
  schema: FormSchema;
  onChange: (values: Record<string, unknown>, status: FormStatus) => void;
}) {
  const section = props.schema.id.replace(/^settings:/, "");
  return <ConfigForm schema={props.schema} onChange={props.onChange} testId={`settings-${section}`} />;
}

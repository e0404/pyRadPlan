/**
 * Generic configuration form rendered from a backend `FormSchema` (Qt `ConfigFormWidget`,
 * `gui/widgets/_config_form.py`) and its modal wrapper (`ConfigFormDialog`). Editors per
 * descriptor type, collapsible nested groups, live validation (red border + tooltip), and
 * only explicit or edited values reported back. Shared with package C (Preferences).
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { FieldDescriptor, FormSchema, ValidationDetail } from "../../api/protocol";
import { Combo, EditableCombo, choiceOptions } from "./Combo";
import {
  comboChoices,
  editText,
  editValue,
  editorKind,
  fieldLabel,
  firstError,
  formValues,
  initFields,
  stepValue,
  type FormFields,
} from "./formModel";
import { ModalDialog } from "./ModalDialog";
import "./forms.css";

export interface FormStatus {
  valid: boolean;
  /** First invalid entry (Qt `validation_failed`), null while every entry is valid. */
  error: { name: string; label: string; message: string } | null;
}

export interface ConfigFormProps {
  schema: FormSchema;
  /** Called after every edit with the values to send back (`ConfigFormWidget.values()`). */
  onChange?: (values: Record<string, unknown>, status: FormStatus) => void;
  /** Educational UI mode: descriptors flagged `advanced` are locked. */
  educational?: boolean;
  disabled?: boolean;
  /** Server-side `ValidationError.details`, matched by field name (prefix stripped by the caller). */
  serverErrors?: ValidationDetail[];
  /** Unit of descriptors with `unit_from` (objective parameters). */
  unitFor?: (f: FieldDescriptor) => string | undefined;
  testId?: string;
}

function tooltip(f: FieldDescriptor, error: string | null, educational?: boolean): string | undefined {
  const lines = [error, f.description];
  if (f.editable === false && f.reason) lines.push(f.reason);
  if (educational && f.advanced) lines.push("Locked in the educational UI mode.");
  const text = lines.filter(Boolean).join("\n");
  return text || undefined;
}

/** Text input behaving like a Qt spin box: arrow keys / buttons step within the bounds. */
export function SpinInput(props: {
  field: FieldDescriptor;
  text: string;
  value: unknown;
  invalid?: boolean;
  disabled?: boolean;
  title?: string;
  placeholder?: string;
  onText: (text: string) => void;
  onStep?: (value: number) => void;
  onBlur?: () => void;
  testId?: string;
  className?: string;
}) {
  const { field } = props;
  const step = (dir: 1 | -1) => {
    const base = typeof props.value === "number" ? props.value : Number(props.text);
    if (!Number.isFinite(base)) return;
    props.onStep?.(stepValue(field, base, dir));
  };
  return (
    <span className={`spin${props.invalid ? " invalid" : ""}${props.disabled ? " disabled" : ""} ${props.className ?? ""}`} title={props.title}>
      <input
        type="text"
        inputMode={field.type === "int" ? "numeric" : "decimal"}
        value={props.text}
        placeholder={props.placeholder}
        disabled={props.disabled}
        data-testid={props.testId}
        onChange={(e) => props.onText(e.currentTarget.value)}
        onBlur={props.onBlur}
        onKeyDown={(e) => {
          if (e.key === "ArrowUp" || e.key === "ArrowDown") {
            e.preventDefault();
            step(e.key === "ArrowUp" ? 1 : -1);
          }
        }}
      />
      {field.unit && <span className="suffix">{field.unit}</span>}
      <span className="spin-btns">
        <button type="button" tabIndex={-1} disabled={props.disabled} onClick={() => step(1)} aria-label="Step up">
          ▴
        </button>
        <button type="button" tabIndex={-1} disabled={props.disabled} onClick={() => step(-1)} aria-label="Step down">
          ▾
        </button>
      </span>
    </span>
  );
}

function TriStateCheckbox(props: { value: unknown; nullable?: boolean; disabled?: boolean; title?: string; onChange: (v: boolean | null) => void; testId?: string }) {
  const ref = useRef<HTMLInputElement>(null);
  const indeterminate = props.nullable && (props.value === null || props.value === undefined);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = Boolean(indeterminate);
  }, [indeterminate]);
  return (
    <span className="cf-check" title={props.title}>
      <input
        ref={ref}
        type="checkbox"
        checked={props.value === true}
        disabled={props.disabled}
        data-testid={props.testId}
        onChange={() => {
          if (!props.nullable) props.onChange(!(props.value === true));
          else if (props.value === null || props.value === undefined) props.onChange(true);
          else if (props.value === true) props.onChange(false);
          else props.onChange(null);
        }}
      />
      {indeterminate && <span className="dim">None</span>}
    </span>
  );
}

function Section(props: { title: string; description?: string; children: ReactNode; testId?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="cf-section" data-testid={props.testId}>
      <button type="button" className="cf-toggle" title={props.description} onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="cf-arrow">{open ? "▾" : "▸"}</span>
        {props.title.replace(/:$/, "")}
      </button>
      {open && <div className="cf-body">{props.children}</div>}
    </div>
  );
}

export function ConfigForm(props: ConfigFormProps) {
  const { schema, onChange } = props;
  const [fields, setFields] = useState<FormFields>(() => initFields(schema));
  const schemaRef = useRef(schema);
  useEffect(() => {
    if (schemaRef.current !== schema) {
      schemaRef.current = schema;
      setFields(initFields(schema));
    }
  }, [schema]);

  const server = useMemo(() => new Map((props.serverErrors ?? []).map((d) => [d.field, d.message])), [props.serverErrors]);

  const update = (next: FormFields) => {
    setFields(next);
    const error = firstError(schema, next);
    onChange?.(formValues(schema, next), { valid: error === null, error });
  };

  const editor = (f: FieldDescriptor): ReactNode => {
    const state = fields[f.name] ?? { text: "", value: null, error: null, edited: false };
    const locked = props.disabled || f.editable === false || (props.educational && f.advanced);
    const error = state.error ?? server.get(f.name) ?? null;
    const title = tooltip(f, error, props.educational);
    const testId = props.testId ? `${props.testId}-${f.name}` : undefined;
    const unit = f.unit ?? props.unitFor?.(f);
    const withUnit = unit && !f.unit ? { ...f, unit } : f;
    switch (editorKind(f)) {
      case "checkbox":
        return <TriStateCheckbox value={state.value} nullable={f.nullable} disabled={locked} title={title} testId={testId} onChange={(v) => update(editValue(fields, f, v))} />;
      case "spin":
      case "optional-number":
        return (
          <SpinInput
            field={withUnit}
            text={state.text}
            value={state.value}
            invalid={Boolean(error)}
            disabled={locked}
            title={title}
            placeholder={f.nullable ? "None" : undefined}
            testId={testId}
            onText={(text) => update(editText(fields, f, text))}
            onStep={(v) => update(editValue(fields, f, v))}
          />
        );
      case "combo":
      case "image": {
        const list = comboChoices(f, state.value);
        const index = list.findIndex((c) => c.value === state.value);
        return (
          <Combo
            value={index >= 0 ? String(index) : null}
            options={choiceOptions(list)}
            disabled={locked}
            title={title}
            testId={testId}
            className={error ? "invalid" : undefined}
            onChange={(key) => update(editValue(fields, f, list[Number(key)].value))}
          />
        );
      }
      case "editable-combo":
        return (
          <EditableCombo
            text={state.text}
            options={choiceOptions(f.choices ?? [])}
            disabled={locked}
            title={title}
            testId={testId}
            className={error ? "invalid" : undefined}
            onText={(text) => update(editText(fields, f, text))}
          />
        );
      case "color": {
        const rgb = Array.isArray(state.value) ? (state.value as number[]) : null;
        const hex = rgb
          ? `#${rgb
              .slice(0, 3)
              .map((c) => Math.round(c <= 1 && rgb.every((x) => x <= 1) ? c * 255 : c).toString(16).padStart(2, "0"))
              .join("")}`
          : typeof state.value === "string"
            ? state.value
            : "#000000";
        return (
          <input
            type="color"
            value={hex}
            disabled={locked}
            title={title}
            data-testid={testId}
            onChange={(e) => {
              const h = e.currentTarget.value;
              const parts = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
              const unit01 = rgb?.every((x) => x <= 1) ?? false;
              update(editValue(fields, f, rgb ? parts.map((c) => (unit01 ? c / 255 : c)) : h));
            }}
          />
        );
      }
      case "section":
        return null;
      default:
        return (
          <span className={`cf-text${error ? " invalid" : ""}`} title={title}>
            <input
              type="text"
              value={state.text}
              disabled={locked}
              placeholder={f.nullable || f.type === "json" ? "None" : undefined}
              data-testid={testId}
              onChange={(e) => update(editText(fields, f, e.currentTarget.value))}
            />
            {unit && <span className="suffix">{unit}</span>}
          </span>
        );
    }
  };

  const rows = (list: readonly FieldDescriptor[]): ReactNode =>
    list.map((f) =>
      f.type === "group" ? (
        <Section key={f.name} title={f.label} description={f.description} testId={props.testId ? `${props.testId}-${f.name}` : undefined}>
          <div className="cf-grid">{rows(f.fields ?? [])}</div>
        </Section>
      ) : (
        <FieldRow key={f.name} field={f} educational={props.educational}>
          {editor(f)}
        </FieldRow>
      ),
    );

  if (!schema.fields.length) {
    return (
      <div className="config-form" data-testid={props.testId}>
        <div className="cf-empty">No configurable parameters.</div>
      </div>
    );
  }
  return (
    <div className="config-form" data-testid={props.testId}>
      <div className="cf-grid">{rows(schema.fields)}</div>
    </div>
  );
}

function FieldRow(props: { field: FieldDescriptor; educational?: boolean; children: ReactNode }) {
  const f = props.field;
  const dim = f.editable === false || (props.educational && f.advanced);
  return (
    <>
      <label className={`cf-label${dim ? " dim" : ""}`} title={f.description}>
        {fieldLabel(f)}
      </label>
      <div className="cf-editor">{props.children}</div>
    </>
  );
}

/** Modal "Configure …" dialog around a `ConfigForm` (Qt `ConfigFormDialog`). */
export function ConfigFormDialog(props: {
  /** The schema, or null while it is loading. */
  schema: FormSchema | null;
  title?: string;
  loadError?: string | null;
  educational?: boolean;
  onAccept: (values: Record<string, unknown>) => void;
  onCancel: () => void;
  testId?: string;
}) {
  const { schema } = props;
  const [values, setValues] = useState<Record<string, unknown> | null>(null);
  const [status, setStatus] = useState<FormStatus>({ valid: true, error: null });
  const [blocked, setBlocked] = useState(false);
  useEffect(() => {
    setValues(null);
    setStatus({ valid: true, error: null });
  }, [schema]);
  const accept = () => {
    if (!schema) return;
    if (!status.valid) {
      setBlocked(true);
      return;
    }
    props.onAccept(values ?? formValues(schema, initFields(schema)));
  };
  const message = status.error ? `${status.error.name}: ${status.error.message}` : blocked && !status.valid ? "Fix the highlighted fields first." : "";
  return (
    <ModalDialog
      title={props.title ?? schema?.title ?? "Configure"}
      className="cf-dialog"
      onClose={props.onCancel}
      testId={props.testId}
      buttons={
        <>
          <button className="btn" onClick={accept} disabled={!schema} data-testid={props.testId ? `${props.testId}-ok` : undefined}>
            OK
          </button>
          <button className="btn" onClick={props.onCancel} data-testid={props.testId ? `${props.testId}-cancel` : undefined}>
            Cancel
          </button>
        </>
      }
    >
      <div className="cf-dialog-body">
        {props.loadError ? (
          <div className="cf-status">{props.loadError}</div>
        ) : !schema ? (
          <div className="dim">Loading…</div>
        ) : (
          <>
            {schema.notes?.map((n) => (
              <div key={n} className="cf-note">
                {n}
              </div>
            ))}
            <ConfigForm
              schema={schema}
              educational={props.educational}
              testId={props.testId ? `${props.testId}-form` : undefined}
              onChange={(v, s) => {
                setValues(v);
                setStatus(s);
                if (s.valid) setBlocked(false);
              }}
            />
          </>
        )}
      </div>
      <div className="cf-status" data-testid={props.testId ? `${props.testId}-status` : undefined}>
        {message}
      </div>
    </ModalDialog>
  );
}

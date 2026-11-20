/**
 * Tissue parameters dialog (Qt `TissueParametersDialog`, `widgets/plan/_tissue_dialog.py`):
 * one row per structure with its RBE model and parameters, the tissue picker (base data
 * classes / cell line presets), the bulk "Set parameter" editor, Match base data,
 * Relabel as …, Derive from model, live derivation and comparison colouring (debounced
 * `tissue.evaluate`) and the "Apply anyway / Keep editing" gate of OK (`tissue.apply`).
 *
 * `TissueSingleRow` is the single-row mode embedded by the Edit VOI dialog (package B).
 */
import { useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState, type Ref } from "react";
import { BackendError } from "../../api/backend";
import type { FieldDescriptor, Issue, StructureDetail, TissueDraft } from "../../api/protocol";
import type { PlanDraftContext, TissueComparison, TissueEvaluateParams, TissueEvaluateResult, TissueOp, TissueSchema, TissueSource } from "../../api/protocol/plan";
import { log, useApp } from "../../store/appStore";
import { errorText, request } from "../../store/planSlice";
import { Combo, type ComboOption } from "../forms/Combo";
import { SpinInput } from "../forms/ConfigForm";
import { ModalDialog } from "../forms/ModalDialog";
import { messageBox } from "../io/messageBox";
import { identityOf, withModel, withoutIdentity } from "./tissueDraft";
import "./plan.css";

const TABLE_NOTE =
  "Tissue parameters per structure: the model parameters, then the reference parameters α_x / β_x (ref.) of the radiation the RBE refers to. MKM and mMKM derive these from their model parameters for the selected reference radiation (grey italics) unless overridden. The biological model maps the parameters to the tissue classes of its base data.";
const DERIVE_REFERENCE_TOOLTIP =
  "The reference parameters of MKM / mMKM are not model inputs: they follow from the model parameters by α_x = α_0 + β_0 · z_1D,ref and β_x = β_0, with z_1D,ref the dose-mean specific energy the reference radiation deposits in a domain of radius r_domain (saturation-corrected z*_1D,ref for mMKM, which needs r_nucleus as well). They are derived live while not overridden; typing a value overrides them, and this button discards the override and derives them again.";
const DERIVE_TOOLTIP = `Derive the reference parameters of the selected MKM / mMKM structures, or of all of them if none are selected, from their model parameters, discarding overrides.\n\n${DERIVE_REFERENCE_TOOLTIP}`;
const DERIVE_DISABLED_TOOLTIP = `Only MKM and mMKM derive their reference parameters from their model parameters, and no structure uses them. Set the model of a structure to MKM or mMKM to enable this.\n\n${DERIVE_REFERENCE_TOOLTIP}`;
const MATCH_TOOLTIP =
  "Apply the closest tissue class of the selected model's base data: its model and the parameters identifying it. The parameters describing the tissue itself (e.g. α_0 / β_0 of MKM) and explicitly set reference parameters keep their value.\n\nApplies to the selected structures, or to all structures if none are selected.";
const RELABEL_TOOLTIP =
  "The tissue matches the base data by its parameters but is labelled with another model (or LEM version), so the base data's model is evaluated. Relabelling it as that model keeps every parameter and only silences the warning; it is optional.\n\nApplies to the selected structures, or to all structures if none are selected.";
const PICKER_TOOLTIP =
  "Tissue classes of the base data and the cell line presets of its model first (✓: matches the base data exactly), then the presets of other models. A preset of another model is converted to the base data's, keeping the parameters both share.";
const PICK_APPLY_TOOLTIP =
  "Fill the selected structures, or all structures if none are selected. A preset describes a whole tissue and sets all parameters of its model; a tissue class of the base data sets its model and the parameters identifying it, keeping the others (e.g. alpha_0 / beta_0 of MKM).";
const PICK_APPLY_TOOLTIP_SINGLE =
  "Fill the tissue parameters of this structure. A preset describes a whole tissue and sets all parameters of its model; a tissue class of the base data sets its model and the parameters identifying it, keeping the others (e.g. alpha_0 / beta_0 of MKM).";
const MATCH_TOOLTIP_SINGLE =
  "Apply the closest tissue class of the selected model's base data: its model and the parameters identifying it. The parameters describing the tissue itself (e.g. α_0 / β_0 of MKM) and explicitly set reference parameters keep their value.";
const REFERENCE_FIELDS = ["alpha_x", "beta_x"];
const REFERENCE_RADIATION = "reference_radiation";
const REFERENCE_COLUMNS = [REFERENCE_RADIATION, ...REFERENCE_FIELDS];
const EVALUATE_DELAY_MS = 150;

export interface TissueRow {
  /** Position of the structure, which `tissue.apply` addresses it by. */
  index: number;
  structure: string;
  type: string;
  tissue: TissueDraft;
  issues: Issue[];
  comparison: TissueComparison | null;
  sources: Record<string, TissueSource>;
}

/** Parameter columns in Qt order: choice fields, model parameters, reference radiation, α_x / β_x. */
export function tissueColumns(schema: TissueSchema): FieldDescriptor[] {
  const all = new Map<string, FieldDescriptor>();
  for (const m of schema.models) for (const f of m.fields) if (!all.has(f.name)) all.set(f.name, f);
  const fields = [...all.values()];
  const choice = fields.filter((f) => f.type === "choice" && f.name !== REFERENCE_RADIATION);
  const model = fields.filter((f) => f.type !== "choice" && !REFERENCE_FIELDS.includes(f.name));
  const radiation = fields.filter((f) => f.name === REFERENCE_RADIATION);
  const reference = fields.filter((f) => REFERENCE_FIELDS.includes(f.name));
  return [...choice, ...model, ...radiation, ...reference];
}

export { withModel, withoutIdentity };

export function modelFieldNames(schema: TissueSchema, model: string): string[] {
  return schema.models.find((m) => m.label === model)?.fields.map((f) => f.name) ?? [];
}

export function columnHeader(f: FieldDescriptor): string {
  const label = REFERENCE_COLUMNS.includes(f.name) ? `${f.label} (ref.)` : f.label;
  return f.unit ? `${label} [${f.unit}]` : label;
}

/** The tissue as sent: only the fields of its model (hidden editors keep the others locally). */
export function tissueForModel(schema: TissueSchema | null, tissue: TissueDraft): TissueDraft {
  if (!schema) return tissue;
  const names = new Set(modelFieldNames(schema, tissue.model));
  const values = Object.fromEntries(Object.entries(tissue.values).filter(([k]) => names.has(k)));
  const sources = tissue.sources ? Object.fromEntries(Object.entries(tissue.sources).filter(([k]) => names.has(k))) : undefined;
  return { model: tissue.model, values, ...(sources ? { sources } : {}), ...identityOf(tissue) };
}

function issueClass(issues: Issue[], comparison: TissueComparison | null): string {
  if (issues.some((i) => i.severity === "error")) return "issue-error";
  if (issues.length) return "issue-warning";
  if (comparison && !comparison.matched) return "issue-error";
  if (comparison?.provenance_fix) return "issue-warning";
  return "";
}

function rowNote(row: TissueRow): string {
  const lines = row.issues.map((i) => i.message);
  const c = row.comparison;
  if (c?.closest_class) lines.push(c.matched ? `Matches the base data class ${c.closest_class}.` : `Closest base data class: ${c.closest_class}.`);
  return lines.join("\n");
}

const fmt = (v: unknown) => (typeof v === "number" ? String(Number(v.toPrecision(6))) : v === null || v === undefined ? "" : String(v));

/** Debounced `tissue.evaluate` of a set of rows. */
function useTissueEvaluate(onResult: (r: TissueEvaluateResult, params: TissueEvaluateParams) => void, draft?: PlanDraftContext) {
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const seq = useRef(0);
  const draftRef = useRef(draft);
  draftRef.current = draft;
  const run = useCallback(
    async (params: TissueEvaluateParams) => {
      const n = ++seq.current;
      try {
        const result = await request("tissue.evaluate", draftRef.current ? { ...params, draft: draftRef.current } : params);
        if (n === seq.current || params.op) onResult(result, params);
        return result;
      } catch (err) {
        log("warning", `Tissue parameters: ${errorText(err)}`);
        return null;
      }
    },
    [onResult],
  );
  const schedule = useCallback(
    (params: TissueEvaluateParams) => {
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => void run(params), EVALUATE_DELAY_MS);
    },
    [run],
  );
  useEffect(() => () => void (timer.current && clearTimeout(timer.current)), []);
  return { run, schedule };
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

function ParamCell(props: {
  field: FieldDescriptor;
  row: TissueRow;
  schema: TissueSchema;
  disabled?: boolean;
  onValue: (field: string, value: number | string | null) => void;
}) {
  const { field, row } = props;
  const value = row.tissue.values[field.name] ?? null;
  const [text, setText] = useState<string | null>(null);
  const cmp = row.comparison?.parameters.find((p) => p.name === field.name);
  const mismatch = cmp && !cmp.matches;
  const source = row.sources[field.name];
  const implicit = REFERENCE_FIELDS.includes(field.name) && source !== undefined && source !== "explicit";
  const tip = [field.description, mismatch ? `Differs from the base data class: declared ${fmt(cmp!.declared)}` : null, implicit ? `Value ${source}` : null]
    .filter(Boolean)
    .join("\n");
  if (field.name === REFERENCE_RADIATION) {
    const choices = props.schema.reference_radiations;
    const options: ComboOption[] = choices.map((c, i) => ({ key: String(i), label: c.label, title: c.description }));
    const index = value === null ? 0 : choices.findIndex((c) => c.value === value);
    return (
      <Combo
        value={String(Math.max(0, index))}
        options={options}
        disabled={props.disabled}
        className={mismatch ? "cell-mismatch" : undefined}
        title={tip || "Radiation the RBE is taken relative to."}
        testId={`tissue-${row.structure}-${field.name}`}
        onChange={(key) => props.onValue(field.name, choices[Number(key)].value as string)}
      />
    );
  }
  if (field.type === "choice") {
    const choices = field.choices ?? [];
    const index = choices.findIndex((c) => c.value === value);
    return (
      <Combo
        value={index >= 0 ? String(index) : null}
        options={choices.map((c, i) => ({ key: String(i), label: c.label }))}
        disabled={props.disabled}
        className={mismatch ? "cell-mismatch" : undefined}
        title={tip}
        testId={`tissue-${row.structure}-${field.name}`}
        onChange={(key) => props.onValue(field.name, choices[Number(key)].value as string)}
      />
    );
  }
  const shown = text ?? fmt(value);
  const invalid = text !== null && text.trim() !== "" && !Number.isFinite(Number(text));
  return (
    <SpinInput
      field={{ ...field, unit: undefined, nullable: true, step: field.step ?? 0.01 }}
      text={shown}
      value={value}
      placeholder="unset"
      invalid={invalid}
      disabled={props.disabled}
      className={`${mismatch ? "cell-mismatch" : ""}${implicit ? " implicit" : ""}`}
      title={tip}
      testId={`tissue-${row.structure}-${field.name}`}
      onBlur={() => setText(null)}
      onText={(t) => {
        setText(t);
        const raw = t.trim();
        if (raw === "") props.onValue(field.name, null);
        else if (Number.isFinite(Number(raw))) props.onValue(field.name, Number(raw));
      }}
      onStep={(v) => {
        setText(null);
        props.onValue(field.name, Number(v.toPrecision(10)));
      }}
    />
  );
}

export function TissueTable(props: {
  schema: TissueSchema;
  rows: TissueRow[];
  columns: FieldDescriptor[];
  selected?: Set<string>;
  onSelect?: (structure: string, additive: boolean) => void;
  onModel: (structure: string, model: TissueDraft["model"]) => void;
  onValue: (structure: string, field: string, value: number | string | null) => void;
  disabled?: boolean;
  showStructure?: boolean;
}) {
  const { schema, rows, columns } = props;
  const used = new Set(rows.flatMap((r) => modelFieldNames(schema, r.tissue.model)));
  const shown = columns.filter((c) => used.has(c.name));
  const models: ComboOption[] = schema.models.map((m) => ({ key: m.label, label: m.label }));
  return (
    <div className="table-wrap tissue-table">
      <table className="qt">
        <thead>
          <tr>
            {props.showStructure !== false && <th>Structure</th>}
            {props.showStructure !== false && <th>Type</th>}
            <th>Model</th>
            {shown.map((f) => (
              <th key={f.name} title={`${REFERENCE_COLUMNS.includes(f.name) ? "Reference radiation (for RBE)" : "Model parameters"}: ${f.description ?? ""}`}>
                {columnHeader(f)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const names = new Set(modelFieldNames(schema, row.tissue.model));
            const selected = props.selected?.has(row.structure);
            const provenance = row.comparison?.provenance_fix;
            return (
              <tr key={row.structure} className={selected ? "selected" : undefined} data-testid={`tissue-row-${row.structure}`}>
                {props.showStructure !== false && (
                  <td
                    className={`tissue-name ${issueClass(row.issues, row.comparison)}`}
                    title={rowNote(row)}
                    onClick={(e) => props.onSelect?.(row.structure, e.ctrlKey || e.metaKey || e.shiftKey)}
                    data-testid={`tissue-name-${row.structure}`}
                  >
                    {row.structure}
                  </td>
                )}
                {props.showStructure !== false && <td>{row.type}</td>}
                <td>
                  <Combo
                    value={row.tissue.model}
                    options={models}
                    disabled={props.disabled}
                    className={provenance ? "cell-warning" : undefined}
                    title={provenance ? "RBE model whose tissue parameters the row holds\nLabelled with another provenance than the matching base data class." : "RBE model whose tissue parameters the row holds"}
                    testId={`tissue-${row.structure}-model`}
                    onChange={(m) => props.onModel(row.structure, m as TissueDraft["model"])}
                  />
                </td>
                {shown.map((f) => (
                  <td key={f.name}>
                    {names.has(f.name) && <ParamCell field={f} row={row} schema={schema} disabled={props.disabled} onValue={(field, v) => props.onValue(row.structure, field, v)} />}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Entries of the tissue picker: the picks with their groups, a separator after the preferred ones. */
export function pickComboOptions(schema: TissueSchema): ComboOption[] {
  const out: ComboOption[] = [];
  let group: string | null = null;
  let preferred: boolean | null = null;
  for (const p of schema.picks) {
    const isPreferred = p.preferred ?? !p.group.startsWith("Other");
    if (preferred === true && !isPreferred) out.push({ key: `sep:${p.id}`, label: "", separator: true });
    preferred = isPreferred;
    if (p.group !== group) {
      group = p.group;
      if (p.group) out.push({ key: `h:${p.group}`, label: p.group, header: true });
    }
    out.push({ key: p.id, label: p.label, bold: p.exact, title: p.description });
  }
  return out;
}

/** Whether a picker selection differs from the pick the tissue corresponds to, i.e. was not applied. */
export function isUnappliedPick(selected: string | null, applied: string | null): boolean {
  return selected !== null && selected !== applied;
}

function applyEdit(row: TissueRow, field: string, value: number | string | null): TissueRow {
  const sources = { ...(row.tissue.sources ?? {}) };
  if (REFERENCE_FIELDS.includes(field)) sources[field] = "explicit";
  const edited = { ...row.tissue, values: { ...row.tissue.values, [field]: value }, sources };
  return { ...row, tissue: (row.tissue.values[field] ?? null) === value ? edited : withoutIdentity(edited) };
}

function mergeResult(rows: TissueRow[], result: TissueEvaluateResult, editing?: { structure: string; field: string }): TissueRow[] {
  const byName = new Map(result.rows.map((r) => [r.structure, r]));
  return rows.map((row) => {
    const r = byName.get(row.structure);
    if (!r) return row;
    const values = { ...row.tissue.values, ...r.tissue.values };
    if (editing && editing.structure === row.structure) values[editing.field] = row.tissue.values[editing.field] ?? null;
    const sources = (r.sources ?? r.tissue.sources ?? {}) as Record<string, TissueSource>;
    return { ...row, tissue: { ...r.tissue, values }, issues: r.issues ?? [], comparison: r.comparison ?? null, sources };
  });
}

// ---------------------------------------------------------------------------
// Dialog
// ---------------------------------------------------------------------------

/** The plan form's unapplied selection the dialog evaluates against (Qt uses the plan widget's). */
export type TissueContext = PlanDraftContext;

function rowsFrom(structures: StructureDetail[]): TissueRow[] {
  return structures.map((s) => ({
    index: s.index,
    structure: s.name,
    type: s.type,
    tissue: JSON.parse(JSON.stringify(s.tissue)) as TissueDraft,
    issues: [],
    comparison: null,
    sources: (s.tissue.sources ?? {}) as Record<string, TissueSource>,
  }));
}

export function TissueDialog(props: { context?: TissueContext; onClose: () => void }) {
  const [schema, setSchema] = useState<TissueSchema | null>(null);
  const [rows, setRows] = useState<TissueRow[]>([]);
  const initial = useRef<Record<string, string>>({});
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [statusText, setStatusText] = useState("");
  const [flags, setFlags] = useState({ can_match: false, can_relabel: [] as string[], can_derive: false });
  const [pick, setPick] = useState<string | null>(null);
  const [bulkParam, setBulkParam] = useState<string | null>(null);
  const [bulkText, setBulkText] = useState("0");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const busy = useApp((s) => s.busy);
  const editing = useRef<{ structure: string; field: string } | undefined>(undefined);

  const onResult = useCallback((result: TissueEvaluateResult, params: TissueEvaluateParams) => {
    setRows((rs) => mergeResult(rs, result, params.op ? undefined : editing.current));
    setFlags({ can_match: result.can_match, can_relabel: result.can_relabel, can_derive: result.can_derive });
    if (result.notes.length) setStatusText([...new Set(result.notes)].join("; "));
  }, []);
  const { run, schedule } = useTissueEvaluate(onResult, props.context);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [s, cst] = await Promise.all([request("tissue.schema", props.context ? { draft: props.context } : {}), request("cst.get", {})]);
        if (cancelled) return;
        const r = rowsFrom(cst.structures);
        initial.current = Object.fromEntries(r.map((x) => [x.structure, JSON.stringify(tissueForModel(s, x.tissue))]));
        setSchema(s);
        setRows(r);
        const first = s.picks[0];
        setPick(first ? first.id : null);
        void run({ rows: r.map((x) => ({ structure: x.structure, tissue: tissueForModel(s, x.tissue) })) });
      } catch (err) {
        if (!cancelled) setError(errorText(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const columns = useMemo(() => (schema ? tissueColumns(schema) : []), [schema]);
  const payload = (rs: TissueRow[]) => rs.map((r) => ({ structure: r.structure, tissue: tissueForModel(schema, r.tissue) }));
  const targets = () => (selected.size ? [...selected] : rows.map((r) => r.structure));

  const edit = (structure: string, field: string, value: number | string | null) => {
    const next = rows.map((r) => (r.structure === structure ? applyEdit(r, field, value) : r));
    setRows(next);
    editing.current = { structure, field };
    schedule({ rows: payload(next), op: "none", edited: { structure, field } });
  };
  const setModel = (structure: string, model: TissueDraft["model"]) => {
    const next = rows.map((r) => (r.structure === structure ? { ...r, tissue: withModel(schema, r.tissue, model) } : r));
    setRows(next);
    editing.current = undefined;
    schedule({ rows: payload(next), op: "none" });
  };
  const op = async (kind: TissueOp, extra: Partial<TissueEvaluateParams> = {}) => {
    editing.current = undefined;
    return run({ rows: payload(rows), op: kind, targets: targets(), ...extra });
  };

  const usedParams = useMemo(() => {
    if (!schema) return [];
    const used = new Set(rows.flatMap((r) => modelFieldNames(schema, r.tissue.model)));
    return columns.filter((c) => used.has(c.name) && (c.type === "float" || c.type === "int"));
  }, [schema, rows, columns]);
  const bulk = usedParams.find((f) => f.name === bulkParam) ?? usedParams[0];
  useEffect(() => {
    if (!bulk || bulk.name === bulkParam) return;
    setBulkParam(bulk.name);
    const first = rows.find((r) => schema && modelFieldNames(schema, r.tissue.model).includes(bulk.name) && r.tissue.values[bulk.name] !== null);
    if (first) setBulkText(fmt(first.tissue.values[bulk.name]));
  }, [bulk?.name]);

  const setForRows = () => {
    if (!schema || !bulk) return;
    const value = Number(bulkText);
    if (!Number.isFinite(value)) {
      setStatusText(`Invalid value "${bulkText}"`);
      return;
    }
    const t = targets();
    const changed = rows.filter((r) => t.includes(r.structure) && modelFieldNames(schema, r.tissue.model).includes(bulk.name)).map((r) => r.structure);
    const next = rows.map((r) => (changed.includes(r.structure) ? applyEdit(r, bulk.name, value) : r));
    setRows(next);
    let text = `Set ${bulk.label} = ${value} in ${changed.length} structure(s)`;
    if (t.length - changed.length) text += `; ${t.length - changed.length} structure(s) whose model lacks it left unchanged`;
    setStatusText(text);
    editing.current = undefined;
    schedule({ rows: payload(next), op: "none" });
  };

  const derive = async () => {
    const result = await op("derive");
    if (result && !result.notes.length) {
      await messageBox({
        title: "Nothing to derive",
        icon: "info",
        text: `No selected structure is an MKM or mMKM tissue with its model parameters set.\n\n${DERIVE_REFERENCE_TOOLTIP}`,
        testId: "tissue-nothing-to-derive",
      });
    }
  };

  const accept = async () => {
    if (!schema) return;
    const changed = payload(rows).filter((r) => JSON.stringify(r.tissue) !== initial.current[r.structure]);
    const applyRows = changed.map((r) => ({ structure: rows.find((row) => row.structure === r.structure)!.index, tissue: r.tissue }));
    if (!changed.length) {
      props.onClose();
      return;
    }
    setSaving(true);
    try {
      let applyAnyway = false;
      for (;;) {
        try {
          const { state } = await request("tissue.apply", { rows: applyRows, ...(applyAnyway ? { apply_anyway: true } : {}) });
          if (state) useApp.setState({ state });
          break;
        } catch (err) {
          if (err instanceof BackendError && err.code === "tissue_errors" && !applyAnyway) {
            const issues = Array.isArray(err.details) ? (err.details as Issue[]) : [];
            const choice = await messageBox({
              title: "Tissue parameters",
              icon: "warning",
              text: "The selected biological model cannot be evaluated for these tissues:",
              informativeText: `${issues.map((i) => i.message).join("\n") || err.message}\n\nUntil they match the base data, dose calculation offers to calculate the physical dose only.`,
              buttons: [
                { id: "apply", label: "Apply anyway" },
                { id: "keep", label: "Keep editing", primary: true, reject: true },
              ],
              testId: "tissue-apply-anyway",
            });
            if (choice !== "apply") {
              setRows((rs) => rs.map((r) => ({ ...r, issues: issues.filter((i) => i.structure === r.structure) })));
              return;
            }
            applyAnyway = true;
            continue;
          }
          throw err;
        }
      }
      try {
        const { structures } = await request("cst.get", {});
        useApp.setState({ structureDetails: Object.fromEntries(structures.map((d) => [d.index, d])) });
      } catch {
        /* the revision-driven sync refetches them */
      }
      log("info", `Tissue parameters applied to ${changed.length} structure(s)`);
      props.onClose();
    } catch (err) {
      await messageBox({ title: "Invalid value", icon: "warning", text: errorText(err) });
    } finally {
      setSaving(false);
    }
  };

  const pickOptions = useMemo(() => (schema ? pickComboOptions(schema) : []), [schema]);

  const relabelText = flags.can_relabel.length ? `Relabel as ${flags.can_relabel.join(" / ")}` : null;
  const disabled = saving || busy !== null;

  return (
    <ModalDialog
      title="Tissue parameters"
      onClose={props.onClose}
      testId="tissue-dialog"
      className="tissue-dialog"
      buttons={
        <>
          <button className="btn" onClick={() => void accept()} disabled={!schema || disabled} data-testid="tissue-ok">
            OK
          </button>
          <button className="btn" onClick={props.onClose} data-testid="tissue-cancel">
            Cancel
          </button>
        </>
      }
    >
      <div className="tissue-note">{TABLE_NOTE}</div>
      {error && <div className="cf-status">{error}</div>}
      {!schema && !error && <div className="dim">Loading…</div>}
      {schema && (
        <>
          {schema.picks.length > 0 && (
            <div className="row">
              <span>Tissue:</span>
              <Combo value={pick} options={pickOptions} className="fill" title={PICKER_TOOLTIP} testId="tissue-picker" onChange={setPick} disabled={disabled} />
              <button className="btn" title={PICK_APPLY_TOOLTIP} disabled={!pick || disabled} onClick={() => pick && void op("apply_pick", { pick })} data-testid="tissue-pick-apply">
                Apply
              </button>
            </div>
          )}
          <TissueTable
            schema={schema}
            rows={rows}
            columns={columns}
            selected={selected}
            disabled={disabled}
            onSelect={(structure, additive) =>
              setSelected((s) => {
                if (!additive) return s.size === 1 && s.has(structure) ? new Set() : new Set([structure]);
                const next = new Set(s);
                if (next.has(structure)) next.delete(structure);
                else next.add(structure);
                return next;
              })
            }
            onModel={setModel}
            onValue={edit}
          />
          <div className="row">
            <span>Set parameter:</span>
            <Combo
              value={bulk?.name ?? null}
              options={usedParams.map((f) => ({ key: f.name, label: columnHeader(f) }))}
              disabled={!usedParams.length || disabled}
              title="Tissue parameter to set in several structures at once"
              testId="tissue-bulk-param"
              onChange={(name) => {
                setBulkParam(name);
                const first = rows.find((r) => modelFieldNames(schema, r.tissue.model).includes(name) && r.tissue.values[name] !== null);
                if (first) setBulkText(fmt(first.tissue.values[name]));
              }}
            />
            <SpinInput
              field={{ name: "bulk", label: "bulk", type: "float", min: 0, step: 0.01 }}
              text={bulkText}
              value={Number(bulkText)}
              disabled={!usedParams.length || disabled}
              title={bulk?.description}
              testId="tissue-bulk-value"
              onText={setBulkText}
              onStep={(v) => setBulkText(String(Number(v.toPrecision(10))))}
            />
            <button
              className="btn"
              disabled={!usedParams.length || disabled}
              title="Set the parameter in the selected structures, or in all structures if none are selected. Structures whose model does not have the parameter are left alone."
              onClick={setForRows}
              data-testid="tissue-bulk-set"
            >
              Set
            </button>
            <span className="fill" />
          </div>
          <div className="tissue-status" data-testid="tissue-status">
            {statusText}
          </div>
          <div className="row">
            <button className="btn" disabled={!flags.can_match || disabled} title={MATCH_TOOLTIP} onClick={() => void op("match")} data-testid="tissue-match">
              Match base data
            </button>
            {relabelText && (
              <button className="btn" title={RELABEL_TOOLTIP} disabled={disabled} onClick={() => void op("relabel")} data-testid="tissue-relabel">
                {relabelText}
              </button>
            )}
            <span className="fill" />
            <button
              className="btn"
              disabled={!flags.can_derive || disabled}
              title={flags.can_derive ? DERIVE_TOOLTIP : DERIVE_DISABLED_TOOLTIP}
              onClick={() => void derive()}
              data-testid="tissue-derive"
            >
              Derive from model
            </button>
          </div>
        </>
      )}
    </ModalDialog>
  );
}

// ---------------------------------------------------------------------------
// Single-row mode (Edit VOI dialog)
// ---------------------------------------------------------------------------

export interface TissueSingleRowProps {
  /** Structure the tissue belongs to (evaluated against the plan's biological model). */
  structure: string;
  /** The draft tissue; controlled. */
  tissue: TissueDraft;
  /** Every edit, with the live issues of the row (errors block `cst.update_structure` unless `apply_anyway`). */
  onChange: (tissue: TissueDraft, issues: Issue[]) => void;
  /** A schema already loaded by the host; otherwise `tissue.schema` is requested. */
  schema?: TissueSchema;
  disabled?: boolean;
  /** Lets the host ask what to do with a pick selected but not applied when it is closed. */
  handle?: Ref<TissueSingleRowHandle>;
}

/** What the host of a single row may ask of its tissue picker. */
export interface TissueSingleRowHandle {
  /** The pick selected in the picker that differs from the one the tissue corresponds to, null if none. */
  unappliedPick(): { id: string; label: string } | null;
  /** Applies the selected pick like the Apply button; the new tissue, null if it failed. */
  applyPick(): Promise<TissueDraft | null>;
  /** Shows the pick the tissue corresponds to again. */
  discardPick(): void;
}

/**
 * One structure's tissue editor: Model combo, parameter cells (units in the headers, "unset"
 * for NaN), live derivation of α_x / β_x for MKM / mMKM, comparison colouring and the issue
 * list, as one row of the tissue dialog.
 */
export function TissueSingleRow(props: TissueSingleRowProps) {
  const [schema, setSchema] = useState<TissueSchema | null>(props.schema ?? null);
  const [row, setRow] = useState<TissueRow>({ index: -1, structure: props.structure, type: "", tissue: props.tissue, issues: [], comparison: null, sources: {} });
  const editing = useRef<{ structure: string; field: string } | undefined>(undefined);
  const onChange = useRef(props.onChange);
  onChange.current = props.onChange;
  const rowRef = useRef(row);
  rowRef.current = row;
  const [selectedPick, setSelectedPick] = useState<string | null>(null);
  const [appliedPick, setAppliedPick] = useState<string | null>(null);
  const picks = useRef({ selected: selectedPick, applied: appliedPick, started: false });
  const [flags, setFlags] = useState({ can_match: false, can_relabel: [] as string[], can_derive: false });
  const [statusText, setStatusText] = useState("");
  const onResult = useCallback((result: TissueEvaluateResult, params: TissueEvaluateParams) => {
    const [merged] = mergeResult([rowRef.current], result, editing.current);
    rowRef.current = merged;
    setRow(merged);
    setFlags({ can_match: result.can_match, can_relabel: result.can_relabel, can_derive: result.can_derive });
    if (result.notes.length) setStatusText([...new Set(result.notes)].join("; "));
    const op = params.op ?? "none";
    if (!picks.current.started || op === "apply_pick" || op === "match" || op === "relabel") {
      picks.current.started = true;
      const pick = result.rows[0]?.pick ?? null;
      picks.current.selected = pick;
      picks.current.applied = pick;
      setSelectedPick(pick);
      setAppliedPick(pick);
    }
    onChange.current(merged.tissue, merged.issues);
  }, []);
  const { run, schedule } = useTissueEvaluate(onResult);
  const pickOptions = useMemo(() => (schema ? pickComboOptions(schema) : []), [schema]);
  const structureRef = useRef(props.structure);
  structureRef.current = props.structure;
  const schemaRef = useRef(schema);
  schemaRef.current = schema;
  const evaluateOp = (kind: TissueOp, pick?: string) =>
    run({
      rows: [{ structure: structureRef.current, tissue: tissueForModel(schemaRef.current, rowRef.current.tissue) }],
      op: kind,
      targets: [structureRef.current],
      ...(pick ? { pick } : {}),
    });
  const selectPick = (id: string | null) => {
    picks.current.selected = id;
    setSelectedPick(id);
  };
  useImperativeHandle(props.handle, () => ({
    unappliedPick() {
      const { selected, applied } = picks.current;
      const pick = schemaRef.current?.picks.find((p) => p.id === selected);
      return pick && isUnappliedPick(selected, applied) ? { id: pick.id, label: pick.label } : null;
    },
    async applyPick() {
      const id = picks.current.selected;
      if (!id) return null;
      const result = await evaluateOp("apply_pick", id);
      return result ? rowRef.current.tissue : null;
    },
    discardPick() {
      selectPick(picks.current.applied);
    },
  }));

  useEffect(() => {
    if (props.schema) return;
    let cancelled = false;
    request("tissue.schema", {})
      .then((s) => !cancelled && setSchema(s))
      .catch((err: unknown) => log("warning", `Tissue parameters: ${errorText(err)}`));
    return () => {
      cancelled = true;
    };
  }, [props.schema]);

  useEffect(() => {
    if (schema) void run({ rows: [{ structure: props.structure, tissue: tissueForModel(schema, row.tissue) }] });
  }, [schema, props.structure]);

  useEffect(() => {
    if (JSON.stringify(props.tissue) !== JSON.stringify(row.tissue)) setRow((r) => ({ ...r, tissue: props.tissue }));
  }, [JSON.stringify(props.tissue)]);

  if (!schema) return <div className="dim">Loading…</div>;
  const columns = tissueColumns(schema);
  const commit = (next: TissueRow, edited?: { structure: string; field: string }) => {
    setRow(next);
    editing.current = edited;
    props.onChange(next.tissue, next.issues);
    schedule({ rows: [{ structure: next.structure, tissue: tissueForModel(schema, next.tissue) }], op: "none", ...(edited ? { edited } : {}) });
  };
  const relabelText = flags.can_relabel.length ? `Relabel as ${flags.can_relabel.join(" / ")}` : null;
  return (
    <div className="tissue-single" data-testid="tissue-single-row">
      {schema.picks.length > 0 && (
        <div className="row">
          <span>Tissue:</span>
          <Combo
            value={selectedPick}
            options={pickOptions}
            className="fill"
            placeholder="No matching tissue"
            title={PICKER_TOOLTIP}
            testId="tissue-picker"
            onChange={selectPick}
            disabled={props.disabled}
          />
          <button
            className="btn"
            title={PICK_APPLY_TOOLTIP_SINGLE}
            disabled={!selectedPick || props.disabled}
            onClick={() => selectedPick && void evaluateOp("apply_pick", selectedPick)}
            data-testid="tissue-pick-apply"
          >
            Apply
          </button>
        </div>
      )}
      <TissueTable
        schema={schema}
        rows={[row]}
        columns={columns}
        showStructure={false}
        disabled={props.disabled}
        onModel={(_s, model) => commit({ ...row, tissue: withModel(schema, row.tissue, model) })}
        onValue={(structure, field, value) => commit(applyEdit(row, field, value), { structure, field })}
      />
      <div className="tissue-status" data-testid="tissue-status">
        {statusText}
      </div>
      <div className="row">
        <button className="btn" disabled={!flags.can_match || props.disabled} title={MATCH_TOOLTIP_SINGLE} onClick={() => void evaluateOp("match")} data-testid="tissue-match">
          Match base data
        </button>
        {relabelText && (
          <button className="btn" title={RELABEL_TOOLTIP} disabled={props.disabled} onClick={() => void evaluateOp("relabel")} data-testid="tissue-relabel">
            {relabelText}
          </button>
        )}
        <span className="fill" />
        <button
          className="btn"
          disabled={!flags.can_derive || props.disabled}
          title={flags.can_derive ? DERIVE_TOOLTIP : DERIVE_DISABLED_TOOLTIP}
          onClick={() => void evaluateOp("derive")}
          data-testid="tissue-derive"
        >
          Derive from model
        </button>
      </div>
      {row.issues.length > 0 && (
        <div className={`tissue-issues ${row.issues.some((i) => i.severity === "error") ? "error" : "warning"}`} data-testid="tissue-single-issues">
          {row.issues.map((i) => (
            <div key={i.message}>{i.message}</div>
          ))}
        </div>
      )}
    </div>
  );
}

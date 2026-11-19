import { useEffect, useMemo, useState, type ComponentType } from "react";
import type { Issue, TissueDraft } from "../../api";
import { useApp } from "../../store/appStore";
import { updateStructure } from "../../store/structures";
import { Dialog } from "../common";

const VOI_TYPES = ["TARGET", "OAR", "EXTERNAL", "HELPER"];
const TISSUE_MODELS: TissueDraft["model"][] = ["Generic", "LEM", "MKM", "mMKM"];

/**
 * Package A's tissue editor in single-row mode (`components/plan/TissueDialog.tsx`,
 * `TissueSingleRow`), loaded lazily so the dialog degrades to a minimal editor without it.
 */
type SingleRowProps = { structure: string; tissue: TissueDraft; onChange: (tissue: TissueDraft, issues: Issue[]) => void };
type TissueRowEditorProps = { structure: string; value: TissueDraft; onChange: (value: TissueDraft) => void };
const tissueModules = import.meta.glob<{ TissueSingleRow?: ComponentType<SingleRowProps> }>("../plan/TissueDialog.tsx");

function useTissueSingleRow(): ComponentType<SingleRowProps> | null {
  const [component, setComponent] = useState<ComponentType<SingleRowProps> | null>(null);
  useEffect(() => {
    const loader = Object.values(tissueModules)[0];
    if (!loader) return;
    let cancelled = false;
    loader()
      .then((m) => {
        if (!cancelled && m.TissueSingleRow) setComponent(() => m.TissueSingleRow!);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);
  return component;
}

const sameTissue = (a: TissueDraft, b: TissueDraft) => a.model === b.model && JSON.stringify(a.values) === JSON.stringify(b.values);

/** Minimal tissue editor used until package A's single-row tissue editor is available. */
function FallbackTissueEditor(props: TissueRowEditorProps) {
  const { value, onChange } = props;
  const num = (key: string) => {
    const v = value.values[key];
    return v === null || v === undefined ? "" : String(v);
  };
  const setValue = (key: string, text: string) =>
    onChange({ ...value, values: { ...value.values, [key]: text.trim() === "" ? null : Number(text) } });
  return (
    <>
      <label>Tissue model:</label>
      <select value={value.model} onChange={(e) => onChange({ ...value, model: e.target.value as TissueDraft["model"] })} data-testid="edit-voi-model">
        {TISSUE_MODELS.map((m) => (
          <option key={m}>{m}</option>
        ))}
      </select>
      <label>α_x [Gy⁻¹]:</label>
      <input type="number" step={0.01} value={num("alpha_x")} onChange={(e) => setValue("alpha_x", e.target.value)} data-testid="edit-voi-alpha" />
      <label>β_x [Gy⁻²]:</label>
      <input type="number" step={0.01} value={num("beta_x")} onChange={(e) => setValue("beta_x", e.target.value)} data-testid="edit-voi-beta" />
    </>
  );
}

/** Qt `VOIMetadataDialog` ("Edit VOI: <name>"): type, tissue, overlap priority; OK recreates the VOI. */
export function EditVoiDialog(props: { index: number; onClose: () => void }) {
  const detail = useApp((s) => s.structureDetails[props.index]);
  const info = useApp((s) => s.state?.structures.find((st) => st.index === props.index));
  const name = detail?.name ?? info?.name ?? "";
  const initialTissue = useMemo<TissueDraft>(() => detail?.tissue ?? { model: "Generic", values: {} }, [detail]);
  const [type, setType] = useState(detail?.type ?? info?.type ?? "OAR");
  const [priority, setPriority] = useState(detail?.overlap_priority ?? info?.overlap_priority ?? 0);
  const [tissue, setTissue] = useState<TissueDraft>(initialTissue);
  const [issues, setIssues] = useState<Issue[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const types = VOI_TYPES.includes(type) ? VOI_TYPES : [...VOI_TYPES, type];
  const SingleRow = useTissueSingleRow();

  const submit = async (applyAnyway = false) => {
    if (!Number.isInteger(priority) || priority < 0 || priority > 9999) {
      setError("Overlap priority must be an integer from 0 to 9999");
      return;
    }
    const changes: Parameters<typeof updateStructure>[1] = {};
    if (type !== (detail?.type ?? info?.type)) changes.type = type;
    if (priority !== (detail?.overlap_priority ?? info?.overlap_priority)) changes.overlap_priority = priority;
    if (!sameTissue(tissue, initialTissue)) changes.tissue = tissue;
    if (!Object.keys(changes).length) return props.onClose();
    setSaving(true);
    const outcome = await updateStructure(props.index, changes, applyAnyway);
    setSaving(false);
    if (outcome.ok) return props.onClose();
    if (outcome.issues) {
      setIssues(outcome.issues.length ? outcome.issues : [{ structure: name, kind: "error", severity: "error", message: outcome.error ?? "" }]);
      return;
    }
    setError(outcome.error ?? "Invalid value");
  };

  return (
    <Dialog
      title={`Edit VOI: ${name}`}
      onClose={props.onClose}
      testId="edit-voi-dialog"
      buttons={
        <>
          <button className="btn" onClick={props.onClose}>
            Cancel
          </button>
          <button className="btn primary" disabled={saving} onClick={() => void submit()} data-testid="edit-voi-ok">
            OK
          </button>
        </>
      }
    >
      <div className="form-grid" style={{ gridTemplateColumns: "auto minmax(0, 1fr)" }}>
        <label>Name:</label>
        <span data-testid="edit-voi-name">{name}</span>
        <label>Type:</label>
        <select value={type} onChange={(e) => setType(e.target.value)} data-testid="edit-voi-type">
          {types.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
        {SingleRow ? (
          <div style={{ gridColumn: "1 / -1" }} data-testid="edit-voi-tissue">
            <SingleRow structure={name} tissue={tissue} onChange={(t) => setTissue(t)} />
          </div>
        ) : (
          <FallbackTissueEditor structure={name} value={tissue} onChange={setTissue} />
        )}
        <label>Overlap priority:</label>
        <input
          type="number"
          min={0}
          max={9999}
          step={1}
          value={priority}
          title="Lower priority numbers overlap higher ones"
          onChange={(e) => setPriority(Number(e.target.value))}
          data-testid="edit-voi-priority"
        />
      </div>
      {!SingleRow && (
        <div className="placeholder-note">Tissue picker, base data comparison and model parameters: see the plan's tissue dialog.</div>
      )}
      {error && <div style={{ color: "var(--danger)" }}>{error}</div>}
      {issues && (
        <Dialog
          title="Tissue parameters"
          onClose={() => setIssues(null)}
          testId="edit-voi-issues"
          buttons={
            <>
              <button className="btn" onClick={() => setIssues(null)}>
                Keep editing
              </button>
              <button
                className="btn primary"
                onClick={() => {
                  setIssues(null);
                  void submit(true);
                }}
                data-testid="edit-voi-apply-anyway"
              >
                Apply anyway
              </button>
            </>
          }
        >
          <div>The biological model cannot be evaluated for this tissue:</div>
          <ul style={{ margin: 0 }}>
            {issues.map((i, n) => (
              <li key={n}>{i.message}</li>
            ))}
          </ul>
        </Dialog>
      )}
    </Dialog>
  );
}

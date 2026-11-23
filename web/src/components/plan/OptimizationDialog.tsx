/**
 * "Configure <problem>" dialog (Qt `OptimizationConfigDialog`,
 * `widgets/plan/_optimization_dialog.py`): the planning problem's form, a solver selector
 * and the selected solver's form, values kept per solver while switching. The problem-level
 * knobs `max_iter` / `display` are shown in the solver form where the solver has them.
 */
import { useEffect, useMemo, useState } from "react";
import type { FormSchema } from "../../api/protocol";
import type { PlanOptions } from "../../api/protocol/plan";
import { errorText, planForm } from "../../store/planSlice";
import { Combo, choiceOptions, type ComboOption } from "../forms/Combo";
import { ConfigForm, type FormStatus } from "../forms/ConfigForm";
import { formValues, initFields } from "../forms/formModel";
import { ModalDialog } from "../forms/ModalDialog";
import type { ProblemProps, SolverSpec } from "./planModel";
import { usePlanForm } from "./ConfigDialogs";

/** Problem settings a problem forwards to its solver (`PlanningProblem.solver_properties`). */
const SOLVER_PROPERTIES = ["max_iter", "display"];

export function OptimizationDialog(props: {
  problem: string;
  problemLabel: string;
  radiationMode: string;
  initial: ProblemProps;
  options: PlanOptions;
  educational?: boolean;
  onAccept: (props: ProblemProps) => void;
  onCancel: () => void;
}) {
  const { options, initial } = props;
  const forwarded = useMemo(() => {
    const out: Record<string, unknown> = {};
    for (const key of SOLVER_PROPERTIES) if (initial.values[key] !== undefined && initial.values[key] !== null) out[key] = initial.values[key];
    return out;
  }, [initial]);
  const problemInitial = useMemo(() => {
    const v = { ...initial.values };
    for (const key of SOLVER_PROPERTIES) delete v[key];
    return v;
  }, [initial]);
  const { schema: problemSchema, error: problemError } = usePlanForm("problem", props.problem, props.radiationMode, problemInitial);
  const defaultSolver = options.default_solver;
  const solverGiven = initial.solver !== null;
  const [solver, setSolver] = useState<string>(initial.solver?.name ?? defaultSolver);
  const [solverValues, setSolverValues] = useState<Record<string, Record<string, unknown>>>(() =>
    initial.solver ? { [initial.solver.name]: { ...initial.solver.values } } : {},
  );
  const [shown, setShown] = useState<Set<string>>(() => new Set());
  const [solverSchema, setSolverSchema] = useState<FormSchema | null>(null);
  const [solverError, setSolverError] = useState<string | null>(null);
  const [problemValues, setProblemValues] = useState<Record<string, unknown> | null>(null);
  const [status, setStatus] = useState("");
  const [valid, setValid] = useState<{ problem: boolean; solver: boolean }>({ problem: true, solver: true });

  const listed = options.solvers.find((s) => s.value === solver);
  const unavailable = !listed || listed.enabled === false;

  useEffect(() => {
    let cancelled = false;
    setSolverSchema(null);
    setSolverError(null);
    const values = { ...(solverValues[solver] ?? {}) };
    if (!shown.has(solver)) {
      for (const [k, v] of Object.entries(forwarded)) if (values[k] === undefined) values[k] = v;
    }
    planForm("solver", solver, props.radiationMode, values)
      .then((schema) => {
        if (cancelled) return;
        setSolverSchema(schema);
        setShown((s) => new Set(s).add(solver));
        setSolverValues((all) => ({ ...all, [solver]: formValues(schema, initFields(schema)) }));
      })
      .catch((err: unknown) => !cancelled && setSolverError(errorText(err)));
    return () => {
      cancelled = true;
    };
    // Reload only when another solver is selected.
  }, [solver]);

  const solverOptions: ComboOption[] = useMemo(() => {
    const list = options.solvers.map((s) => ({
      ...s,
      description: s.description ?? (s.gpu_compatible ? "Runs on the GPU backend." : undefined),
    }));
    const opts = choiceOptions(list);
    if (!options.solvers.some((s) => s.value === solver)) {
      opts.push({
        key: `x:${solver}`,
        label: `${solver} (not available)`,
        identifier: solver,
        disabled: true,
        title: `The plan's solver, which is unavailable: it is not installed in this environment.`,
      });
    }
    return opts;
  }, [options.solvers, solver]);
  const currentKey = options.solvers.findIndex((s) => s.value === solver);

  const accept = () => {
    if (!valid.problem || !valid.solver) {
      setStatus(status || "Fix the highlighted fields first.");
      return;
    }
    const values: Record<string, unknown> = { ...(problemValues ?? (problemSchema ? formValues(problemSchema, initFields(problemSchema)) : problemInitial)) };
    const solverFields = new Set(solverSchema?.fields.map((f) => f.name) ?? []);
    for (const [k, v] of Object.entries(forwarded)) if (!solverFields.has(k)) values[k] = v;
    let spec: SolverSpec | null;
    if (unavailable) spec = initial.solver;
    else {
      const sv = solverValues[solver] ?? {};
      if (Object.keys(sv).length) spec = { name: solver, values: sv };
      else if (solverGiven || solver !== defaultSolver) spec = { name: solver, values: {} };
      else spec = null;
    }
    props.onAccept({ values, solver: spec });
  };

  const onForm = (which: "problem" | "solver") => (v: Record<string, unknown>, s: FormStatus) => {
    if (which === "problem") setProblemValues(v);
    else setSolverValues((all) => ({ ...all, [solver]: v }));
    setValid((x) => ({ ...x, [which]: s.valid }));
    setStatus(s.error ? `${s.error.name}: ${s.error.message}` : "");
  };

  const solverText = unavailable
    ? (solverSchema?.notes?.[0] ??
      `Solver '${solver}' is unavailable: ${listed?.reason ?? "it is not installed in this environment"}. Its configuration is kept; choose another solver to replace it.`)
    : (listed?.label ?? solver);

  return (
    <ModalDialog
      title={`Configure ${props.problemLabel}`}
      onClose={props.onCancel}
      testId="optimization-dialog"
      className="opt-dialog"
      buttons={
        <>
          <button className="btn" onClick={accept} data-testid="optimization-dialog-ok">
            OK
          </button>
          <button className="btn" onClick={props.onCancel} data-testid="optimization-dialog-cancel">
            Cancel
          </button>
        </>
      }
    >
      <fieldset className="qt-group">
        <legend>Planning problem: {props.problemLabel}</legend>
        {problemError ? (
          <div className="cf-status">{problemError}</div>
        ) : problemSchema ? (
          <ConfigForm schema={problemSchema} educational={props.educational} onChange={onForm("problem")} testId="problem-form" />
        ) : (
          <div className="dim">Loading…</div>
        )}
      </fieldset>
      <fieldset className="qt-group grow">
        <legend>Solver</legend>
        <Combo
          value={currentKey >= 0 ? String(currentKey) : `x:${solver}`}
          options={solverOptions}
          title="Solver used to optimize the planning problem"
          testId="solver-combo"
          onChange={(key) => {
            const next = key.startsWith("x:") ? key.slice(2) : String(options.solvers[Number(key)].value);
            setSolver(next);
            setStatus("");
          }}
        />
        <div className={unavailable ? "solver-note warn" : "solver-note"} data-testid="solver-label">
          {solverText}
        </div>
        <div className="solver-scroll">
          {solverError ? (
            <div className="cf-status">{solverError}</div>
          ) : solverSchema ? (
            unavailable ? null : (
              <ConfigForm schema={solverSchema} educational={props.educational} onChange={onForm("solver")} testId="solver-form" />
            )
          ) : (
            <div className="dim">Loading…</div>
          )}
        </div>
      </fieldset>
      <div className="cf-status" data-testid="optimization-dialog-status">
        {status}
      </div>
    </ModalDialog>
  );
}

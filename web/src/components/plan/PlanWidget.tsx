/**
 * Plan form (Qt `PlanWidget`, `widgets/plan/_plan_widget.py`): radiation mode, machine,
 * engine (+ "…"), fractions and dose convention, beam angles (beam count, presets), bixel
 * width, isocenter (Auto preview), biological model (+ "…", tissue marker, tissue
 * parameters), scenario model (+ "…"), optimized quantity (not implemented), optimization
 * problem (+ "…"), sequencing (not implemented) and the dose grid; orange frames on modified
 * fields, Discard / Apply. Options and forms come from `plan.options` / `plan.form`.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { FieldDescriptor, Vec3 } from "../../api/protocol";
import { PLAN_MACHINE, type PlanOptions, type TissueCheckResult } from "../../api/protocol/plan";
import { useApp } from "../../store/appStore";
import { registerPendingSource } from "../../store/pendingChanges";
import { setDraftBeams } from "../../store/revisions";
import { registerBeamEditor } from "../../store/viewerSlice";
import { useUiMode } from "../../store/uiMode";
import {
  applyPlanChanges,
  errorText,
  feature,
  hasMethod,
  planForm,
  planOptions,
  tissueCheck,
  usePlanStore,
  validationDetails,
} from "../../store/planSlice";
import { Combo, EditableCombo, MenuButton, choiceOptions, type ComboOption } from "../forms/Combo";
import { SpinInput } from "../forms/ConfigForm";
import { BioModelDialog, EngineDialog, ScenarioDialog } from "./ConfigDialogs";
import { OptimizationDialog } from "./OptimizationDialog";
import {
  MODIFIED_NOTE,
  captureState,
  draftFromPlan,
  equidistantAngles,
  fieldKeyOf,
  fieldOf,
  formBeamAngles,
  formatNumberList,
  modifiedKeys,
  parseNumberList,
  pickBioModel,
  pickIdentifier,
  pickMachine,
  planSetParams,
  syncedCouch,
  type FieldKey,
  type PlanDraft,
  type ProblemProps,
} from "./planModel";
import { TissueDialog } from "./TissueDialog";
import "./plan.css";

const NUM_BEAMS_TOOLTIP = "Number of beams.\nSetting it replaces the gantry angles by that many equally spaced\nangles starting at 0 deg.";
const QUANTITY_TOOLTIP =
  "Choosing a global optimized quantity not yet implemented.\nThe optimizer derives the quantity from the objectives table: with a biological model, physical-dose objectives are optimized on RBE-weighted dose, unless other objectives already use a biological quantity. \"Change all quantities…\" sets the quantity on all objectives at once.";
const SEQUENCING_TOOLTIP = "Sequencing / aperture optimization not yet implemented.";
const CONFORMAL_TOOLTIP = "3D conformal planning not yet implemented.";
const BIO_TOOLTIP =
  "Biological model used in dose calculation and optimization, listed as formalism · base data model.\nThe entries follow the radiation mode and the base data of the machine; those the machine cannot run are shown disabled, their tooltip says why.";
const ENGINE_TOOLTIP =
  "Dose engine. Which biological models can be computed depends on it: Monte Carlo engines do not use the machine's pencil-beam kernels.";
const DOSE_CONVENTION_TOOLTIP =
  "Convention for dose values in objectives and results:\nper fraction - objective doses and reported doses refer to one fraction;\ntotal - objective values (including reference images) and reported results\nrefer to the whole course; doses scale with the number of fractions, LET does not.";

/** Controls Qt registers with `_add_advanced` (locked in the educational UI mode). */
const ADVANCED = new Set(["machine", "engine", "fractions", "scenario", "quantity", "problem", "sequencing"]);

type DialogKind = "engine" | "bio" | "scenario" | "problem" | "tissue";

interface Status {
  text: string;
  error?: boolean;
  tooltip?: string;
  modifiedNote?: boolean;
}

function unitDescriptor(f: FieldDescriptor | undefined, fallback: Partial<FieldDescriptor>): FieldDescriptor {
  return { name: "x", label: "x", type: "float", ...fallback, ...(f ?? {}), unit: undefined };
}

function issuesText(result: TissueCheckResult): string {
  return result.issues.map((i) => `• ${i.message}`).join("\n");
}

export function PlanWidget() {
  const plan = useApp((s) => s.state?.plan ?? null);
  const hasCt = useApp((s) => Boolean(s.state?.has.ct));
  const hasCst = useApp((s) => Boolean(s.state?.has.cst));
  const busy = useApp((s) => s.busy);
  const info = useApp((s) => s.info);
  const session = useApp((s) => s.session);
  const cstRevision = useApp((s) => s.state?.revisions?.workspace.cst ?? s.state?.structures.length ?? 0);
  const profileGrid = useApp((s) => s.state?.options?.default_plan.dose_grid_resolution ?? null);
  const detail = usePlanStore((s) => s.detail);
  const educational = useUiMode((s) => s.uiMode === "educational");

  const appliedKey = JSON.stringify([plan, detail, profileGrid]);
  // Rebuilt only when the applied plan (not just the state object) changes.
  const fresh = useCallback(
    () => draftFromPlan(plan, detail && plan ? detail : null, (profileGrid as Vec3 | null) ?? [5, 5, 5]),
    [appliedKey],
  );
  const [draft, setDraft] = useState<PlanDraft>(fresh);
  const draftRef = useRef(draft);
  draftRef.current = draft;
  const [clean, setClean] = useState(() => captureState(draft));
  const cleanRef = useRef(clean);
  cleanRef.current = clean;
  const [options, setOptions] = useState<PlanOptions | null>(null);
  const optionsCtx = useRef<{ mode: string; machine: string; engine: string | null; options: PlanOptions | null }>({
    mode: draft.radiation_mode,
    machine: draft.machine,
    engine: draft.engine,
    options: null,
  });
  const [status, setStatus] = useState<Status>({ text: "" });
  const [errorKeys, setErrorKeys] = useState<Set<FieldKey>>(new Set());
  const [tissue, setTissue] = useState<TissueCheckResult | null>(null);
  const [dialog, setDialog] = useState<DialogKind | null>(null);
  const [bioConfigurable, setBioConfigurable] = useState<boolean>(true);
  const [beamsText, setBeamsText] = useState<string | null>(null);
  const relistSeq = useRef(0);
  const relistPending = useRef<Promise<void>>(Promise.resolve());
  const hadPlan = useRef(plan !== null);
  const applying = useRef(false);

  // Workspace -> form: a changed plan resets the form (Qt `_do_update` with "pln").
  useEffect(() => {
    const d = fresh();
    setDraft(d);
    setClean(captureState(d));
    setErrorKeys(new Set());
    const firstPlan = !hadPlan.current && plan !== null && !applying.current;
    setStatus((s) => (firstPlan ? { text: "Default plan applied." } : s.modifiedNote || s.error ? { text: "" } : s));
    hadPlan.current = plan !== null;
    void relist({ mode: d.radiation_mode, machine: d.machine, engine: d.engine }, d, "sync");
  }, [fresh, session]);

  /** Re-list options for a context and adapt machine / engine / problem / model (Qt `_refresh_*`). */
  const relist = useCallback(
    (ctx: { mode: string; machine: string; engine: string | null }, base: PlanDraft, reason: "sync" | "mode" | "machine" | "engine") => {
      // Apply awaits this so a click right after a mode change never sends the old mode's engine.
      if (reason !== "sync") draftRef.current = base;
      const run = relistBody(ctx, base, reason);
      relistPending.current = run;
      return run;
    },
    [],
  );
  const relistBody = useCallback(
    async (ctx: { mode: string; machine: string; engine: string | null }, base: PlanDraft, reason: "sync" | "mode" | "machine" | "engine") => {
      const seq = ++relistSeq.current;
      if (!useApp.getState().session) return;
      try {
        let opts = await planOptions(ctx.mode, ctx.machine, ctx.engine);
        let machine = ctx.machine;
        let engine = ctx.engine;
        if (reason === "mode") {
          machine = pickMachine(base.machine, opts);
          engine = pickIdentifier(base.engine, opts.engines);
          opts = await planOptions(ctx.mode, machine, engine);
        }
        if (seq !== relistSeq.current) return;
        const previous = optionsCtx.current;
        optionsCtx.current = { mode: ctx.mode, machine, engine, options: opts };
        setOptions(opts);
        if (reason === "sync") {
          // A plan without an engine or problem shows the mode's default (Qt `_restore_engine_from_pln`).
          const engine = base.engine ?? pickIdentifier(null, opts.engines);
          const problem = opts.problems.some((p) => p.value === base.problem) ? base.problem : String(opts.problems[0]?.value ?? base.problem);
          if ((engine !== base.engine || problem !== base.problem) && draftRef.current === base) {
            const patched = { ...base, engine, problem };
            draftRef.current = patched;
            setDraft(patched);
            setClean(captureState(patched));
          }
          return;
        }
        const engineLabel = opts.engines.find((e) => e.value === engine)?.label ?? engine;
        const pick = pickBioModel(base, { mode: previous.mode, options: previous.options }, ctx.mode, opts, machine, engineLabel);
        const problem = opts.problems.some((p) => p.value === base.problem) ? base.problem : String(opts.problems[0]?.value ?? base.problem);
        const d = draftRef.current;
        const next = {
          ...d,
          machine,
          engine,
          problem,
          bioOption: pick.option,
          bioModel: pick.model,
          bioParams: { ...d.bioParams, [pick.model]: pick.params },
        };
        draftRef.current = next;
        setDraft(next);
        if (pick.note) setStatus({ text: pick.note.text, tooltip: pick.note.reason });
      } catch (err) {
        if (seq === relistSeq.current) setStatus({ text: errorText(err), error: true });
      }
    },
    [],
  );

  const modified = useMemo(() => modifiedKeys(draft, clean), [draft, clean]);
  const isModified = modified.size > 0;
  const update = (patch: Partial<PlanDraft>) => setDraft((d) => ({ ...d, ...patch }));

  // Status label: the modified note unless an error is shown (Qt `_set_global_modified`).
  useEffect(() => {
    setStatus((s) => {
      if (isModified && !s.error && !s.modifiedNote && !s.tooltip) return { text: MODIFIED_NOTE, modifiedNote: true };
      if (!isModified && s.modifiedNote) return { text: "" };
      return s;
    });
  }, [isModified]);

  // Tissue marker for the unapplied model (Qt `_update_tissue_indicator`).
  const tissueKey = JSON.stringify([draft.radiation_mode, draft.machine, draft.engine, draft.bioOption, draft.bioModel, draft.bioParams[draft.bioModel], cstRevision, hasCst]);
  useEffect(() => {
    if (!hasCst) {
      setTissue(null);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      void tissueCheck({
        radiation_mode: draft.radiation_mode,
        machine: draft.machine,
        engine: draft.engine,
        bio_model: draft.bioOption === null ? draft.bioModel : { model: draft.bioModel, option: draft.bioOption, values: draft.bioParams[draft.bioModel] ?? {} },
      }).then((r) => !cancelled && setTissue(r));
    }, 200);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [tissueKey]);

  // Whether the selected biological model has parameters (Qt `_update_bio_config_button`):
  // from `plan.options` (`configurable`), else from its form.
  useEffect(() => {
    if (draft.bioModel === "none" || draft.bioOption === null || !hasMethod("plan.form")) {
      setBioConfigurable(false);
      return;
    }
    const choice = options?.bio_models.find((o) => o.value === draft.bioOption) ?? options?.bio_models.find((o) => o.model === draft.bioModel);
    if (choice?.configurable !== undefined) {
      setBioConfigurable(choice.configurable);
      return;
    }
    let cancelled = false;
    planForm("bio_model", draft.bioModel, draft.radiation_mode)
      .then((s) => !cancelled && setBioConfigurable(s.fields.length > 0))
      .catch(() => !cancelled && setBioConfigurable(true));
    return () => {
      cancelled = true;
    };
  }, [draft.bioModel, draft.bioOption, draft.radiation_mode, options]);

  // Edited beam angles drawn as pending beams; dragged beams enter the draft.
  const beamsDraft = useMemo(() => {
    if (!modified.has("gantry") && !modified.has("couch") && !modified.has("iso")) return null;
    try {
      const { gantry, couch } = formBeamAngles(draft);
      let iso: Vec3 | null | undefined;
      if (modified.has("iso")) {
        if (draft.isoAuto) iso = null;
        else {
          const v = parseNumberList(draft.iso);
          iso = v.length === 3 ? (v as Vec3) : undefined;
        }
      }
      return { gantry_angles: gantry, couch_angles: couch, ...(iso !== undefined ? { iso_center: iso } : {}) };
    } catch {
      return null;
    }
  }, [draft, modified]);
  useEffect(() => setDraftBeams(beamsDraft), [JSON.stringify(beamsDraft)]);
  useEffect(
    () =>
      registerBeamEditor((index, gantry) => {
        setDraft((d) => {
          try {
            const angles = parseNumberList(d.gantry);
            if (index < 0 || index >= angles.length) return d;
            angles[index] = Number(gantry.toFixed(2));
            return { ...d, gantry: formatNumberList(angles) };
          } catch {
            return d;
          }
        });
      }),
    [],
  );
  useEffect(() => () => setDraftBeams(null), []);

  const apply = useCallback(async (): Promise<boolean> => {
    await relistPending.current;
    let params;
    try {
      params = planSetParams(draftRef.current, cleanRef.current, optionsCtx.current.options ?? options);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setStatus({ text: `ValueError: ${message}`, error: true });
      const key = fieldKeyOf(message.split(":")[0]);
      setErrorKeys(new Set(key ? [key] : []));
      return false;
    }
    if (!Object.keys(params).length) {
      setStatus({ text: "Plan applied." });
      return true;
    }
    applying.current = true;
    try {
      await applyPlanChanges(params);
      setErrorKeys(new Set());
      setStatus({ text: "Plan applied." });
      return true;
    } catch (err) {
      const details = validationDetails(err);
      setErrorKeys(new Set(details.map((d) => fieldKeyOf(d.field)).filter((k): k is FieldKey => k !== null)));
      setStatus({
        text: details.length ? `ValidationError: ${details.map((d) => `${d.field}: ${d.message}`).join("; ")}` : errorText(err),
        error: true,
      });
      return false;
    } finally {
      applying.current = false;
    }
  }, [draft, clean, options]);

  // Unapplied edits are offered before Save Plan / workflow steps (Qt pending-changes source).
  const pendingRef = useRef({ modified: false, apply });
  pendingRef.current = { modified: isModified, apply };
  useEffect(
    () =>
      registerPendingSource({
        name: "plan",
        hasPending: () => pendingRef.current.modified,
        apply: () => pendingRef.current.apply(),
      }),
    [],
  );

  const discard = () => {
    if (!plan) return;
    const d = fresh();
    setDraft(d);
    setClean(captureState(d));
    setErrorKeys(new Set());
    setStatus({ text: "Changes discarded." });
    void relist({ mode: d.radiation_mode, machine: d.machine, engine: d.engine }, d, "sync");
  };

  // ------------------------------------------------------------------ view helpers
  const disabled = busy !== null || !hasCt;
  const locked = (what: string, descriptor?: FieldDescriptor) =>
    educational && (ADVANCED.has(what) || Boolean(descriptor?.advanced));
  const mod = (...keys: FieldKey[]) => (keys.some((k) => errorKeys.has(k)) ? "invalid" : keys.some((k) => modified.has(k)) ? "modified" : "");
  const advancedTitle = (what: string, title?: string) => (locked(what) ? [title, "Locked in the educational UI mode."].filter(Boolean).join("\n") : title);

  const fFractions = fieldOf(options, "num_of_fractions");
  const fConvention = fieldOf(options, "dose_convention");
  const fBixel = fieldOf(options, "bixel_width");
  const fGrid = fieldOf(options, "dose_grid_resolution");
  const gridLocked = fGrid?.editable === false;
  const gridTitle = gridLocked ? fGrid?.reason : undefined;

  let beamCount: number | null = null;
  try {
    beamCount = parseNumberList(draft.gantry).length;
  } catch {
    beamCount = null;
  }

  const modeOptions: ComboOption[] = (options?.radiation_modes ?? [{ value: draft.radiation_mode, label: draft.radiation_mode }]).map((c) => ({
    key: String(c.value),
    label: c.label,
    disabled: c.enabled === false,
    title: c.reason,
  }));
  if (!modeOptions.some((o) => o.key === draft.radiation_mode)) modeOptions.push({ key: draft.radiation_mode, label: draft.radiation_mode });

  const keyed = (choices: PlanOptions["engines"] | undefined, current: string | null, missingLabel?: (v: string) => string) => {
    const list = choiceOptions(choices ?? []).map((o, i) => (o.header ? o : { ...o, key: String((choices ?? [])[Number(o.key)]?.value ?? i) }));
    if (current && !list.some((o) => o.key === current)) list.push({ key: current, label: missingLabel ? missingLabel(current) : current, identifier: current });
    return list;
  };
  const engineOptions = keyed(options?.engines, draft.engine);
  const scenarioOptions = keyed(options?.scenario_models, draft.scenario);
  const problemOptions = keyed(options?.problems, draft.problem);
  const bioOptions: ComboOption[] = (options?.bio_models ?? []).map((o) => ({
    key: String(o.value),
    label: o.label,
    identifier: o.model,
    disabled: o.enabled === false,
    title: `${o.description ?? o.label}${o.enabled === false ? `\nNot available: ${o.reason ?? "the machine cannot run it."}` : ""}`,
  }));
  const UNSUPPORTED = "__unsupported__";
  if (draft.bioOption === null) {
    bioOptions.push({
      key: UNSUPPORTED,
      label: `${draft.bioModel} (not supported by this machine)`,
      identifier: draft.bioModel,
      disabled: true,
      title: "Model of the plan, which the machine cannot run.\nApply keeps it; select another model to replace it.",
    });
  } else if (!bioOptions.some((o) => o.key === draft.bioOption)) {
    bioOptions.push({ key: draft.bioOption, label: draft.bioOption, identifier: draft.bioModel });
  }
  const bioLabel = bioOptions.find((o) => o.key === (draft.bioOption ?? UNSUPPORTED))?.label ?? draft.bioModel;
  const machineOptions: ComboOption[] = (options?.machines ?? []).map((c) => ({
    key: String(c.value),
    label: c.label,
    title: c.description ?? (c.value === PLAN_MACHINE ? "The base data the applied plan carries itself, used instead of a machine file." : undefined),
  }));
  const machineLabel = (value: string) => options?.machines.find((c) => c.value === value)?.label ?? value;
  const machineValue = (text: string) => String(options?.machines.find((c) => c.label === text)?.value ?? text);
  const presetItems: ComboOption[] = (options?.beam_presets ?? []).map((p) => {
    const fmt = (v: number[]) => formatNumberList(v);
    const label = p.label && p.label !== p.name ? p.label : `${p.name} (${fmt(p.gantry_angles)}${p.couch_angles.some(Boolean) ? ` / couch ${fmt(p.couch_angles)}` : ""})`;
    return { key: p.name, label, title: p.description || p.name };
  });
  const quantityField = fieldOf(options, "optimized_quantity");
  const aiStatus = feature(info, "ai");
  const resolvedIso = detail?.resolved_iso_center ?? null;
  const isoText = draft.isoAuto ? (draft.perBeamIso ? "multiple iso centers" : resolvedIso ? formatNumberList(resolvedIso) : draft.iso) : draft.iso;
  const engineLabel = engineOptions.find((o) => o.key === draft.engine)?.label ?? draft.engine;
  const tissueContext = `the structures on machine '${draft.machine}'${engineLabel ? ` with the dose engine ${engineLabel}` : ""}`;
  const tissueTip =
    tissue?.marker === "error"
      ? `The biological model '${bioLabel}' cannot be evaluated for ${tissueContext}:\n${issuesText(tissue)}\nSet the tissue parameters or choose another model; otherwise dose calculation offers the physical dose only.`
      : tissue?.marker === "warning"
        ? `The biological model '${bioLabel}' is evaluated for ${tissueContext}, with warnings:\n${issuesText(tissue)}\nDose calculation uses the base data's model; relabelling the tissues in the tissue parameters is optional.`
        : "";

  const onBioPick = (key: string) => {
    const o = options?.bio_models.find((x) => String(x.value) === key);
    if (!o) return;
    setDraft((d) => {
      const kept = Object.fromEntries(Object.entries(d.bioParams[o.model] ?? {}).filter(([k]) => !["data_model", "kernel_set", "quantity_variants", "quantity_table_name"].includes(k)));
      return { ...d, bioOption: key, bioModel: o.model, bioParams: { ...d.bioParams, [o.model]: { ...kept, ...JSON.parse(JSON.stringify(o.parameters)) } } };
    });
  };

  const setGantryText = (gantry: string) => update({ gantry });
  const commitGantry = () => {
    const couch = syncedCouch(draft.gantry, draft.couch);
    if (couch !== null) update({ couch });
  };
  const setBeamCount = (n: number) => {
    const count = Math.max(1, Math.min(360, Math.round(n)));
    const gantry = formatNumberList(equidistantAngles(count));
    setDraft((d) => ({ ...d, gantry, couch: syncedCouch(gantry, d.couch) ?? d.couch }));
    setBeamsText(null);
  };

  const row = (label: ReactNode, content: ReactNode, opts: { span?: boolean; adv?: string; title?: string } = {}) => (
    <>
      <label className={opts.adv && locked(opts.adv) ? "dim" : undefined} title={opts.title}>
        {label}
      </label>
      <div className={`pw-cell${opts.span ? " span3" : ""}`}>{content}</div>
    </>
  );

  const formsServed = hasMethod("plan.form");
  // A plan without an engine runs the mode's default (Qt shows it selected in the combo).
  const effectiveEngine = draft.engine ?? pickIdentifier(null, options?.engines ?? []);
  const configButton = (kind: DialogKind, key: FieldKey, enabled: boolean, title: string, adv?: string, testId?: string) => (
    <button
      type="button"
      className={`btn dots ${mod(key)}`}
      disabled={!enabled || !formsServed || (adv ? locked(adv) : false)}
      title={formsServed ? (adv ? advancedTitle(adv, title) : title) : "The backend does not serve configuration forms (plan.form)"}
      onClick={() => setDialog(kind)}
      data-testid={testId}
    >
      …
    </button>
  );

  const problemProps: ProblemProps = draft.problemProps[draft.problem] ?? { values: {}, solver: null };

  return (
    <div className="plan-widget" data-testid="plan-widget">
      <fieldset disabled={disabled} className="pw-fieldset">
        <div className="form-grid pw-grid">
          {row(
            "Radiation mode:",
            <Combo
              value={draft.radiation_mode}
              options={modeOptions}
              className={mod("radiation")}
              testId="plan-mode"
              disabled={disabled}
              onChange={(mode) => {
                const next = { ...draft, radiation_mode: mode };
                setDraft(next);
                void relist({ mode, machine: draft.machine, engine: draft.engine }, next, "mode");
              }}
            />,
          )}
          {row(
            "Machine:",
            <EditableCombo
              text={machineLabel(draft.machine)}
              options={machineOptions}
              className={`fill ${mod("machine")}`}
              disabled={disabled || locked("machine")}
              title={advancedTitle("machine")}
              testId="plan-machine"
              onText={(text) => update({ machine: machineValue(text) })}
              onCommit={(text) => {
                const m = machineValue(text) || "Generic";
                if (m === optionsCtx.current.machine && draft.radiation_mode === optionsCtx.current.mode) return;
                const next = { ...draft, machine: m };
                setDraft(next);
                void relist({ mode: draft.radiation_mode, machine: m, engine: draft.engine }, next, "machine");
              }}
            />,
            { adv: "machine" },
          )}

          {row(
            "Dose engine:",
            <div className="row">
              <Combo
                value={effectiveEngine}
                options={engineOptions}
                className={`fill ${mod("engine")}`}
                disabled={disabled || locked("engine")}
                title={advancedTitle("engine", ENGINE_TOOLTIP)}
                testId="plan-engine"
                onChange={(engine) => {
                  const next = { ...draft, engine };
                  setDraft(next);
                  void relist({ mode: draft.radiation_mode, machine: draft.machine, engine }, next, "engine");
                }}
              />
              {configButton("engine", "engine_props", engineOptions.length > 0 && Boolean(effectiveEngine), "Configure the selected dose engine", "engine", "plan-engine-config")}
            </div>,
            { adv: "engine" },
          )}
          {row(
            "Fractions:",
            <div className="row">
              <SpinInput
                field={unitDescriptor(fFractions, { type: "int", min: 1, max: 1000 })}
                text={draft.fractions}
                value={Number(draft.fractions)}
                className={`fill ${mod("fractions")}`}
                disabled={disabled || locked("fractions", fFractions)}
                title={advancedTitle("fractions", fFractions?.description)}
                testId="plan-fractions"
                onText={(fractions) => update({ fractions })}
                onStep={(v) => update({ fractions: String(v) })}
              />
              <span className={locked("fractions") ? "locked" : undefined}>Doses:</span>
              <Combo
                value={draft.doseConvention}
                options={(fConvention?.choices ?? [
                  { value: "per_fraction", label: "per fraction" },
                  { value: "total", label: "total" },
                ]).map((c) => ({ key: String(c.value), label: c.label }))}
                className={`fill ${mod("dose_convention")}`}
                disabled={disabled || locked("fractions", fConvention)}
                title={advancedTitle("fractions", fConvention?.description ?? DOSE_CONVENTION_TOOLTIP)}
                testId="plan-dose-convention"
                onChange={(v) => update({ doseConvention: v as PlanDraft["doseConvention"] })}
              />
            </div>,
            { adv: "fractions" },
          )}

          {row(
            "Gantry angles [deg]:",
            <div className="row">
              <input
                type="text"
                className={`fill ${mod("gantry")}`}
                value={draft.gantry}
                onChange={(e) => setGantryText(e.currentTarget.value)}
                onBlur={commitGantry}
                onKeyDown={(e) => e.key === "Enter" && commitGantry()}
                title={"Gantry angles in the matRad coordinate system.\nEvery gantry angle defines a beam; separate angles by blanks."}
                data-testid="plan-gantry"
              />
              <span className="nowrap">Couch angles [deg]:</span>
              <input
                type="text"
                className={`fill ${mod("couch")}`}
                value={draft.couch}
                onChange={(e) => update({ couch: e.currentTarget.value })}
                title={
                  "Couch angles in the matRad coordinate system.\nEvery couch angle belongs to a gantry angle; separate angles by blanks.\nThe field follows the gantry angles: a single couch angle is repeated for\nadded beams, otherwise an added beam gets a couch angle of 0."
                }
                data-testid="plan-couch"
              />
              <span className={`spin beams${beamCount === null ? " disabled" : ""}`} title={beamCount === null ? "Gantry angles are not a valid list of numbers." : NUM_BEAMS_TOOLTIP}>
                <input
                  type="text"
                  inputMode="numeric"
                  value={beamsText ?? String(beamCount ?? "")}
                  disabled={beamCount === null || disabled}
                  data-testid="plan-num-beams"
                  onChange={(e) => setBeamsText(e.currentTarget.value)}
                  onBlur={() => {
                    if (beamsText !== null && Number(beamsText) > 0 && Number(beamsText) !== beamCount) setBeamCount(Number(beamsText));
                    else setBeamsText(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") e.currentTarget.blur();
                    if ((e.key === "ArrowUp" || e.key === "ArrowDown") && beamCount) {
                      e.preventDefault();
                      setBeamCount(beamCount + (e.key === "ArrowUp" ? 1 : -1));
                    }
                  }}
                />
                <span className="suffix">beams</span>
                <span className="spin-btns">
                  <button type="button" tabIndex={-1} disabled={!beamCount || disabled} onClick={() => beamCount && setBeamCount(beamCount + 1)} aria-label="More beams">
                    ▴
                  </button>
                  <button type="button" tabIndex={-1} disabled={!beamCount || disabled} onClick={() => beamCount && setBeamCount(beamCount - 1)} aria-label="Fewer beams">
                    ▾
                  </button>
                </span>
              </span>
              <MenuButton
                label="Choose preset…"
                items={presetItems}
                disabled={disabled || !presetItems.length}
                title={"Set gantry and couch angles from a preset;\npresets typical for the radiation mode are listed first."}
                testId="plan-presets"
                onPick={(name) => {
                  const p = options?.beam_presets.find((x) => x.name === name);
                  if (p) update({ gantry: formatNumberList(p.gantry_angles), couch: formatNumberList(p.couch_angles) });
                }}
              />
              <span className="tip-wrap" title={aiStatus.available ? "Suggest gantry/couch angles using an LLM" : (aiStatus.reason ?? "")}>
                <button type="button" className="btn" disabled={!aiStatus.available || disabled} data-testid="plan-ai-beams">
                  ✨ AI
                </button>
              </span>
            </div>,
            { span: true },
          )}

          {row(
            "Bixel width / spot spacing [mm]:",
            <SpinInput
              field={unitDescriptor(fBixel, { type: "float", min: 0.1, max: 100, decimals: 2, step: 0.5 })}
              text={draft.bixel}
              value={Number(draft.bixel)}
              className={mod("bixel")}
              disabled={disabled}
              title={fBixel?.description ?? "Photons: width (and height) of quadratic photon bixel\nParticles: lateral spot distance"}
              testId="plan-bixel"
              onText={(bixel) => update({ bixel })}
              onStep={(v) => update({ bixel: v.toFixed(2) })}
            />,
          )}
          {row(
            "Iso center [mm]:",
            <div className="row">
              <input
                type="text"
                className={`fill ${mod("iso")}`}
                value={isoText}
                disabled={draft.isoAuto || disabled}
                onChange={(e) => update({ iso: e.currentTarget.value })}
                data-testid="plan-iso"
              />
              <label className="check" title="If checked, the isocenter is computed automatically from the target structures">
                <input
                  type="checkbox"
                  checked={draft.isoAuto}
                  data-testid="plan-iso-auto"
                  onChange={(e) => {
                    const auto = e.currentTarget.checked;
                    update({ isoAuto: auto, perBeamIso: null, iso: auto ? draft.iso : isoText === "multiple iso centers" ? "0 0 0" : isoText });
                  }}
                />
                Auto.
              </label>
            </div>,
          )}

          {row(
            "Biological model:",
            <div className="row">
              <Combo
                value={draft.bioOption ?? UNSUPPORTED}
                options={bioOptions}
                className={`fill ${mod("bio_model")}`}
                title={BIO_TOOLTIP}
                testId="plan-bio"
                disabled={disabled}
                onChange={onBioPick}
              />
              <button
                type="button"
                className={`btn dots ${mod("bio_params")}`}
                disabled={!bioConfigurable || disabled}
                title={bioConfigurable ? "Configure the selected biological model" : `The biological model '${bioLabel}' has no parameters to configure`}
                onClick={() => setDialog("bio")}
                data-testid="plan-bio-config"
              >
                …
              </button>
            </div>,
          )}
          {row(
            "Scenario model:",
            <div className="row">
              <Combo
                value={draft.scenario}
                options={scenarioOptions}
                className={`fill ${mod("scenario")}`}
                disabled={disabled || locked("scenario")}
                title={advancedTitle("scenario", "Scenario sampling model for uncertainty handling")}
                testId="plan-scenario"
                onChange={(scenario) => update({ scenario })}
              />
              {configButton(
                "scenario",
                "scenario_props",
                draft.scenario !== "explicit",
                "Configure the selected scenario model: setup and range uncertainties, CT phases, worst-case extent and sampling",
                "scenario",
                "plan-scenario-config",
              )}
            </div>,
            { adv: "scenario" },
          )}

          {row(
            "Optimized quantity:",
            <Combo
              value="auto"
              options={(quantityField?.choices ?? [{ value: "auto", label: "auto" }]).map((c) => ({ key: String(c.value), label: c.label }))}
              disabled
              title={QUANTITY_TOOLTIP}
              testId="plan-quantity"
              className="fill"
              onChange={() => undefined}
            />,
            { adv: "quantity" },
          )}
          <span />
          <div className="row pw-cell">
            {tissue && tissue.marker && (
              <span className={`tissue-mark ${tissue.marker}`} title={tissueTip} data-testid="plan-tissue-marker" data-marker={tissue.marker}>
                {tissue.marker === "error" ? "⚠" : "ℹ"}
              </span>
            )}
            <button
              type="button"
              className="btn fill"
              disabled={!hasCst || disabled || !hasMethod("tissue.schema")}
              title="Edit the tissue properties of the loaded structures: the reference photon alpha_x / beta_x and, per RBE model, its own parameters"
              onClick={() => setDialog("tissue")}
              data-testid="plan-tissue"
            >
              Set tissue parameters
            </button>
          </div>

          {row(
            "Optimization:",
            <div className="row">
              <Combo
                value={draft.problem}
                options={problemOptions}
                className={`fill ${mod("problem")}`}
                disabled={disabled || locked("problem")}
                title={advancedTitle("problem", "Planning problem formulation used by the optimization")}
                testId="plan-problem"
                onChange={(problem) => update({ problem })}
              />
              {configButton("problem", "problem_props", problemOptions.length > 0 && Boolean(options), "Configure the selected planning problem and choose its solver", "problem", "plan-problem-config")}
            </div>,
            { adv: "problem" },
          )}
          <span />
          <span />

          <div className={`row pw-cell span2${locked("sequencing") ? " dim" : ""}`}>
            <label className="check" title={SEQUENCING_TOOLTIP}>
              <input type="checkbox" disabled />
              Run sequencing
            </label>
            <span className="dim">Levels:</span>
            <span className="spin disabled small" title={SEQUENCING_TOOLTIP}>
              <input type="text" value="7" disabled readOnly />
              <span className="spin-btns">
                <button type="button" disabled tabIndex={-1}>
                  ▴
                </button>
                <button type="button" disabled tabIndex={-1}>
                  ▾
                </button>
              </span>
            </span>
          </div>
          <div className={`row pw-cell span2${locked("sequencing") ? " dim" : ""}`}>
            <label className="check" title={SEQUENCING_TOOLTIP}>
              <input type="checkbox" disabled />
              Run DAO
            </label>
            <label className="check" title={CONFORMAL_TOOLTIP}>
              <input type="checkbox" disabled />
              3D conformal
            </label>
          </div>

          {row(
            "Dose grid res. [mm]:",
            <div className="row" title={gridTitle}>
              {(["x", "y", "z"] as const).map((axis, i) => (
                <span key={axis} className="row fill">
                  {axis}:
                  <SpinInput
                    field={unitDescriptor(fGrid, { type: "float", min: 0.1, max: 50, decimals: 2, step: 0.5 })}
                    text={draft.res[i]}
                    value={Number(draft.res[i])}
                    className={`fill ${mod((["res_x", "res_y", "res_z"] as const)[i])}`}
                    disabled={disabled || gridLocked}
                    title={gridTitle}
                    testId={`plan-res-${axis}`}
                    onText={(t) => {
                      const res = [...draft.res] as PlanDraft["res"];
                      res[i] = t;
                      update({ res });
                    }}
                    onStep={(v) => {
                      const res = [...draft.res] as PlanDraft["res"];
                      res[i] = v.toFixed(2);
                      update({ res });
                    }}
                  />
                </span>
              ))}
              <button
                className="btn"
                type="button"
                disabled={disabled || gridLocked}
                title={gridTitle ?? "Use the CT voxel spacing"}
                data-testid="plan-ct-grid"
                onClick={() => {
                  const sp = useApp.getState().state?.ct?.spacing;
                  if (!sp) {
                    setStatus({ text: "No CT in workspace.", error: true });
                    return;
                  }
                  update({ res: sp.map((v) => v.toFixed(2)) as PlanDraft["res"] });
                  setStatus({ text: "Dose grid set from CT." });
                }}
              >
                Use CT grid
              </button>
            </div>,
            { span: true },
          )}
        </div>
      </fieldset>
      <div className="plan-footer">
        <span
          className={`pw-status${status.error ? " error" : status.modifiedNote ? " note" : ""}`}
          title={status.tooltip || status.text}
          data-testid="plan-note"
        >
          {status.text}
        </span>
        <button className="btn" disabled={!isModified || !plan || busy !== null} onClick={discard} title="Reset the form to the applied plan" data-testid="plan-discard">
          Discard
        </button>
        <button className={`btn${isModified ? " bold" : ""}`} disabled={disabled} onClick={() => void apply()} data-testid="plan-apply">
          {isModified ? "Apply *" : "Apply"}
        </button>
      </div>

      {dialog === "engine" && effectiveEngine && (
        <EngineDialog
          name={effectiveEngine}
          radiationMode={draft.radiation_mode}
          values={draft.engineProps[effectiveEngine] ?? {}}
          educational={educational}
          onCancel={() => setDialog(null)}
          onAccept={(values) => {
            setDraft((d) => ({ ...d, engine: d.engine ?? effectiveEngine, engineProps: { ...d.engineProps, [d.engine ?? effectiveEngine]: values } }));
            setDialog(null);
          }}
        />
      )}
      {dialog === "bio" && (
        <BioModelDialog
          name={draft.bioModel}
          radiationMode={draft.radiation_mode}
          values={draft.bioParams[draft.bioModel] ?? {}}
          educational={educational}
          onCancel={() => setDialog(null)}
          onAccept={(values) => {
            setDraft((d) => ({ ...d, bioParams: { ...d.bioParams, [d.bioModel]: values } }));
            setDialog(null);
          }}
        />
      )}
      {dialog === "scenario" && (
        <ScenarioDialog
          name={draft.scenario}
          radiationMode={draft.radiation_mode}
          values={draft.scenarioProps[draft.scenario] ?? {}}
          educational={educational}
          onCancel={() => setDialog(null)}
          onAccept={(values) => {
            setDraft((d) => ({ ...d, scenarioProps: { ...d.scenarioProps, [d.scenario]: values } }));
            setDialog(null);
          }}
        />
      )}
      {dialog === "problem" && options && (
        <OptimizationDialog
          problem={draft.problem}
          problemLabel={problemOptions.find((o) => o.key === draft.problem)?.label ?? draft.problem}
          radiationMode={draft.radiation_mode}
          initial={problemProps}
          options={options}
          educational={educational}
          onCancel={() => setDialog(null)}
          onAccept={(props) => {
            setDraft((d) => ({ ...d, problemProps: { ...d.problemProps, [d.problem]: props } }));
            setDialog(null);
          }}
        />
      )}
      {dialog === "tissue" && (
        <TissueDialog
          context={{
            radiation_mode: draft.radiation_mode,
            machine: draft.machine,
            engine: draft.engine,
            bio_model: draft.bioOption === null ? draft.bioModel : { model: draft.bioModel, option: draft.bioOption, values: draft.bioParams[draft.bioModel] ?? {} },
          }}
          onClose={() => setDialog(null)}
        />
      )}
    </div>
  );
}

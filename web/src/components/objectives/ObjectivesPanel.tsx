/**
 * "Objectives & constraints" panel (Qt `OptimizationWidget`,
 * `widgets/optimization/_optimization_widget.py`): VOI selector, + Add objective, ✨ AI
 * (unavailable in the browser), Change all quantities…, the objective count and one table row
 * per objective (type, penalty, quantity, robustness, parameters, Remove). Every edit is a
 * `objectives.*` write; rejected changes show in the status line. Also hosts the optimization
 * status window (portal).
 */
import { useEffect, useMemo, useState } from "react";
import type { FieldDescriptor, ObjectiveData, StructureDetail } from "../../api/protocol";
import type { ObjectiveType, ObjectivesCatalog } from "../../api/protocol/plan";
import { useApp } from "../../store/appStore";
import { feature, objectiveActions, refreshCatalog, refreshStructures, usePlanStore } from "../../store/planSlice";
import { Combo, MenuButton, choiceOptions, type ComboOption } from "../forms/Combo";
import { SpinInput } from "../forms/ConfigForm";
import { checkValue, formatValue } from "../forms/formModel";
import { StatusWindow } from "../optimization/StatusWindow";
import { objectiveRows, parameterUnit, referenceChoices, robustnessOptions } from "./objectivesModel";
import "../plan/plan.css";

const ROBUSTNESS_TOOLTIP = "How the objective is combined over the scenarios of the dose influence data; 'None' uses the nominal scenario only";
const EMPTY: Record<string, StructureDetail> = {};

/** Number editor committing on Enter, focus-out or a step (Qt spin box without keyboard tracking). */
function CommitSpin(props: {
  field: FieldDescriptor;
  value: number;
  disabled?: boolean;
  title?: string;
  testId?: string;
  onCommit: (value: number) => void;
  onInvalid: (message: string) => void;
}) {
  const [text, setText] = useState<string | null>(null);
  const shown = text ?? formatValue(props.field, props.value);
  const parsed = Number(shown);
  const error = text === null ? null : !Number.isFinite(parsed) || shown.trim() === "" ? "Input should be a valid number" : checkValue(props.field, parsed);
  const commit = (v: number) => {
    setText(null);
    if (v !== props.value) props.onCommit(v);
  };
  return (
    <SpinInput
      field={props.field}
      text={shown}
      value={props.value}
      invalid={Boolean(error)}
      disabled={props.disabled}
      title={[error, props.title].filter(Boolean).join("\n") || undefined}
      testId={props.testId}
      className="obj-spin"
      onText={setText}
      onStep={(v) => commit(v)}
      onBlur={() => {
        if (text === null) return;
        if (error) {
          props.onInvalid(`Ignored invalid value ${JSON.stringify(text)}: ${error}`);
          setText(null);
          return;
        }
        commit(parsed);
      }}
    />
  );
}

function ParameterEditors(props: {
  structure: number;
  structureName: string;
  objective: ObjectiveData;
  type: ObjectiveType | undefined;
  catalog: ObjectivesCatalog;
  unitOf: (quantity: string) => string;
  disabled: boolean;
  setStatus: (text: string) => void;
}) {
  const { objective, type } = props;
  const params = type?.parameters ?? [];
  if (!params.length) return <span className="dim">—</span>;
  return (
    <div className="obj-params">
      {params.map((f) => {
        const value = objective.parameters[f.name];
        const testId = `obj-${props.structureName}-${objective.index}-${f.name}`;
        let editor;
        if (f.type === "image_ref") {
          const { options, current } = referenceChoices(props.catalog, value);
          editor = (
            <Combo
              value={current}
              options={options}
              disabled={props.disabled}
              testId={testId}
              title={
                props.catalog.reference_images.length
                  ? "Reference dose image (picked from the workspace results)"
                  : "No result doses available yet — compute or import a dose first"
              }
              onChange={(key) => {
                if (key.startsWith("$custom")) return;
                void objectiveActions.update(props.structure, objective.index, { parameters: { [f.name]: { $result: key } } }, `reference image for ${f.name}`);
              }}
            />
          );
        } else if (f.type === "choice") {
          const list = f.choices ?? [];
          const index = list.findIndex((c) => c.value === value);
          editor = (
            <Combo
              value={index >= 0 ? String(index) : null}
              options={choiceOptions(list)}
              disabled={props.disabled}
              title={f.description}
              testId={testId}
              onChange={(key) => void objectiveActions.update(props.structure, objective.index, { parameters: { [f.name]: list[Number(key)].value } }, `value for ${f.name}`)}
            />
          );
        } else if (typeof value === "number" || f.type === "float" || f.type === "int") {
          const unit = parameterUnit(f, objective.quantity, props.unitOf);
          editor = (
            <CommitSpin
              field={{ ...f, unit }}
              value={typeof value === "number" ? value : Number(value ?? f.default ?? 0)}
              disabled={props.disabled}
              title={f.description}
              testId={testId}
              onCommit={(v) => void objectiveActions.update(props.structure, objective.index, { parameters: { [f.name]: v } }, `value for ${f.name}`)}
              onInvalid={(m) => props.setStatus(`Ignored invalid value for ${f.name}: ${m}`)}
            />
          );
        } else {
          editor = <span>{String(value)}</span>;
        }
        return (
          <span key={f.name} className="obj-param">
            <span className="obj-param-name" title={f.description}>
              {f.name}:
            </span>
            {editor}
          </span>
        );
      })}
    </div>
  );
}

export function ObjectivesPanel() {
  const details = useApp((s) => s.structureDetails ?? EMPTY);
  const hasCst = useApp((s) => Boolean(s.state?.has.cst));
  const hasPln = useApp((s) => Boolean(s.state?.has.pln));
  const structureNames = useApp((s) => s.state?.structures.map((x) => x.name).join("\u0000") ?? "");
  const busy = useApp((s) => s.busy);
  const info = useApp((s) => s.info);
  const catalog = usePlanStore((s) => s.catalog);
  const status = usePlanStore((s) => s.objectivesStatus);
  const [voi, setVoi] = useState<number | null>(null);

  useEffect(() => {
    if (!hasCst) return;
    if (!catalog) void refreshCatalog();
    if (!Object.keys(details).length) void refreshStructures();
  }, [hasCst, structureNames]);

  const structures = useMemo(
    () => (hasCst ? Object.values(details).sort((a, b) => a.index - b.index) : []),
    [details, hasCst],
  );
  useEffect(() => {
    if (structures.length && (voi === null || !structures.some((s) => s.index === voi))) setVoi(structures[0].index);
    if (!structures.length) setVoi(null);
  }, [structures, voi]);

  const rows = useMemo(() => objectiveRows(structures), [structures]);
  const total = rows.length;
  const disabled = busy !== null;
  const unitOf = (quantity: string) => info?.quantities?.find((q) => q.identifier === quantity)?.unit ?? "Gy";
  const setStatus = (text: string) => usePlanStore.setState({ objectivesStatus: text });

  const ai = feature(info, "ai");
  const aiTitle = !ai.available ? (ai.reason ?? "") : !hasCst ? "Load a structure set first" : !hasPln ? "Configure a plan first" : "Suggest objectives for all VOIs using an LLM";
  const voiOptions: ComboOption[] = structures.map((s) => ({ key: String(s.index), label: `${s.name} (${s.type})` }));
  const objectiveOptions: ComboOption[] = (catalog?.objectives ?? []).map((o) => ({ key: o.name, label: o.name }));
  const quantityOptions: ComboOption[] = (catalog?.quantities ?? []).map((q) => ({ key: String(q.value), label: q.label, identifier: q.identifier ?? String(q.value) }));

  return (
    <div className="objectives-panel" data-testid="objectives">
      <div className="row obj-controls">
        <span>VOI:</span>
        <Combo value={voi === null ? null : String(voi)} options={voiOptions} onChange={(v) => setVoi(Number(v))} disabled={!structures.length || disabled} testId="obj-voi" />
        <button className="btn" disabled={voi === null || !catalog || disabled} onClick={() => voi !== null && void objectiveActions.add(voi)} data-testid="obj-add">
          + Add objective
        </button>
        <span className="tip-wrap" title={aiTitle}>
          <button className="btn" disabled={!ai.available || !hasCst || !hasPln || disabled} data-testid="obj-ai">
            ✨ AI
          </button>
        </span>
        <span className="fill" />
        <MenuButton
          label="Change all quantities…"
          items={quantityOptions.map((q) => ({ key: q.key, label: q.label, title: q.identifier }))}
          disabled={!catalog || !hasCst || disabled}
          title={"Set the quantity of the objectives of all VOIs at once.\nSingle objectives can still be changed in the Quantity column below."}
          testId="obj-all-quantities"
          onPick={(q) => void objectiveActions.setQuantityAll(q)}
        />
        <span className="nowrap" data-testid="obj-count">
          {total} objective{total === 1 ? "" : "s"}
        </span>
      </div>
      <div className="table-wrap obj-table">
        <table className="qt">
          <thead>
            <tr>
              <th>VOI</th>
              <th>Objective</th>
              <th>Penalty</th>
              <th>Quantity</th>
              <th>Robustness</th>
              <th>Parameters</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {catalog &&
              rows.map(({ structure, objective, first }) => {
                const type = catalog.objectives.find((o) => o.name === objective.name);
                const rob = robustnessOptions(type, objective.robustness);
                const key = `${structure.name}-${objective.index}`;
                const q = quantityOptions.some((o) => o.key === objective.quantity)
                  ? quantityOptions
                  : [...quantityOptions, { key: objective.quantity, label: objective.quantity }];
                return (
                  <tr key={`${structure.index}-${objective.index}`} data-testid={`obj-row-${key}`}>
                    <td className={first ? "obj-voi bold" : "obj-voi"}>{first ? `${structure.name} (${structure.type})` : ""}</td>
                    <td>
                      <Combo
                        value={objective.name}
                        options={objectiveOptions}
                        chars={9}
                        disabled={disabled}
                        title={objective.name}
                        testId={`obj-${key}-type`}
                        onChange={(name) => void objectiveActions.changeType(structure.index, objective.index, name)}
                      />
                    </td>
                    <td>
                      <CommitSpin
                        field={{ name: "priority", label: "Penalty", type: "float", min: 0, max: 1e9, step: 1 }}
                        value={objective.priority}
                        disabled={disabled}
                        testId={`obj-${key}-penalty`}
                        onCommit={(v) => void objectiveActions.update(structure.index, objective.index, { priority: v }, "penalty")}
                        onInvalid={(m) => setStatus(`Ignored invalid penalty: ${m}`)}
                      />
                    </td>
                    <td>
                      <Combo
                        value={objective.quantity}
                        options={q}
                        chars={5}
                        disabled={disabled}
                        title={q.find((o) => o.key === objective.quantity)?.label}
                        testId={`obj-${key}-quantity`}
                        onChange={(quantity) => void objectiveActions.update(structure.index, objective.index, { quantity }, `quantity ${JSON.stringify(quantity)}`)}
                      />
                    </td>
                    <td>
                      <Combo
                        value={rob.current}
                        options={rob.options}
                        chars={7}
                        disabled={disabled || rob.options.length <= 1}
                        title={ROBUSTNESS_TOOLTIP}
                        testId={`obj-${key}-robustness`}
                        onChange={(robustness) =>
                          void objectiveActions.update(structure.index, objective.index, { robustness }, `robustness ${JSON.stringify(robustness)}`)
                        }
                      />
                    </td>
                    <td>
                      <ParameterEditors
                        structure={structure.index}
                        structureName={structure.name}
                        objective={objective}
                        type={type}
                        catalog={catalog}
                        unitOf={unitOf}
                        disabled={disabled}
                        setStatus={setStatus}
                      />
                    </td>
                    <td>
                      <button className="btn" disabled={disabled} onClick={() => void objectiveActions.remove(structure.index, objective.index)} data-testid={`obj-${key}-remove`}>
                        Remove
                      </button>
                    </td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      </div>
      <div className="obj-status" data-testid="obj-status">
        {!hasCst ? "No structure set loaded" : status}
      </div>
      <StatusWindow />
    </div>
  );
}

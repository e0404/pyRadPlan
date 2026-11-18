import type { ResultGroup } from "../api/protocol";
import { componentText, locateKey } from "../results/resultGroups";

/**
 * Two-level result quantity selector (Qt `QuantitySelector`): quantity group (incl. snapshot
 * groups "dose (photons_30fx)"), then component full / beam i / scenario i: label. Keeps the
 * component label ("beam 1") when switching to a group that has it.
 */
export function QuantitySelector(props: {
  groups: readonly ResultGroup[];
  value: string | null;
  onChange: (key: string | null) => void;
  scenarioLabels?: readonly string[];
  /** Adds a first entry selecting nothing (the compare selector's "— None —"). */
  noneLabel?: string;
  disabled?: boolean;
  testId?: string;
  title?: string;
}) {
  const { groups, value, onChange, noneLabel } = props;
  const located = locateKey(groups, value);
  const groupIndex = located ? groups.indexOf(located.group) : noneLabel !== undefined ? -1 : groups.length ? 0 : -1;
  const group = groupIndex >= 0 ? groups[groupIndex] : null;
  const component = located?.component ?? group?.components[0] ?? null;

  return (
    <span className="row fill" data-testid={props.testId} title={props.title}>
      <select
        className="fill"
        value={groupIndex}
        disabled={props.disabled || (!groups.length && noneLabel === undefined)}
        data-testid={props.testId ? `${props.testId}-group` : undefined}
        onChange={(e) => {
          const index = Number(e.target.value);
          if (index < 0) return onChange(null);
          const next = groups[index];
          const same = next.components.find((c) => c.label === component?.label);
          onChange((same ?? next.components[0]).key);
        }}
      >
        {noneLabel !== undefined && <option value={-1}>{noneLabel}</option>}
        {!groups.length && noneLabel === undefined && <option value={-1}>— None —</option>}
        {groups.map((g, i) => (
          <option key={`${g.base}|${g.tag}`} value={i} title={g.base}>
            {g.label}
          </option>
        ))}
      </select>
      <select
        value={component?.key ?? ""}
        disabled={props.disabled || !group || group.components.length <= 1}
        title="Total of all beams, a single beam contribution or the result of a single uncertainty scenario"
        data-testid={props.testId ? `${props.testId}-component` : undefined}
        onChange={(e) => onChange(e.target.value)}
      >
        {(group?.components ?? []).map((c) => (
          <option key={c.key} value={c.key} title={c.key}>
            {componentText(c.label, c.kind, props.scenarioLabels ?? [])}
          </option>
        ))}
        {!group && <option value="">full</option>}
      </select>
    </span>
  );
}

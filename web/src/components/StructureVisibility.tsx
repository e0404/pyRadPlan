import { useRef, useState } from "react";
import type { StructureDetail, StructureInfo, Vec3 } from "../api";
import { setViewer, useApp } from "../store/appStore";
import { updateStructure } from "../store/structures";
import { EditVoiDialog } from "./vois/EditVoiDialog";
import "./viewer.css";

/** Stable empty selector result (a fresh [] would re-render forever). */
const EMPTY: never[] = [];

const TYPE_GROUPS: [string, string][] = [
  ["TARGET", "Targets"],
  ["OAR", "OARs"],
  ["EXTERNAL", "External"],
  ["HELPER", "Helpers"],
];

/** `TruncatedCheckBox`: names longer than 12 characters are elided (full name in the tooltip). */
export function truncateName(name: string, max = 12): string {
  return name.length > max ? `${name.slice(0, max - 1)}…` : name;
}

export function groupByType(structures: readonly StructureInfo[]): [string, StructureInfo[]][] {
  const sorted = [...structures].sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
  const groups = new Map<string, StructureInfo[]>();
  for (const s of sorted) {
    const key = (s.type || "OTHER").toUpperCase();
    groups.set(key, [...(groups.get(key) ?? []), s]);
  }
  const out: [string, StructureInfo[]][] = [];
  for (const [key, title] of TYPE_GROUPS) {
    const members = groups.get(key);
    if (members?.length) out.push([title, members]);
    groups.delete(key);
  }
  const other = groups.get("OTHER");
  groups.delete("OTHER");
  if (other?.length) out.push(["Other", other]);
  for (const [key, members] of groups) out.push([key[0] + key.slice(1).toLowerCase(), members]);
  return out;
}

/** "By Overlap": one group per overlap priority, lowest (dominant) first. */
export function groupByOverlap(structures: readonly StructureInfo[], details: Record<string, StructureDetail>): [string, StructureInfo[]][] {
  const sorted = [...structures].sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
  const groups = new Map<number, StructureInfo[]>();
  for (const s of sorted) {
    const p = details[s.name]?.overlap_priority ?? s.overlap_priority ?? 0;
    groups.set(p, [...(groups.get(p) ?? []), s]);
  }
  return [...groups.entries()].sort((a, b) => a[0] - b[0]).map(([p, members]) => [`Overlap priority ${p}`, members]);
}

/** Qt `_voi_tooltip` as plain text. */
export function voiTooltip(s: StructureInfo, d: StructureDetail | undefined): string {
  if (!d) return `${s.name} (${s.type})\nClick the name to edit`;
  const ratio = d.beta_x ? `${+(d.alpha_x / d.beta_x).toPrecision(4)} Gy` : "∞";
  const params = Object.entries(d.tissue.values)
    .filter(([k, v]) => k !== "alpha_x" && k !== "beta_x" && v !== null && v !== "" && !(typeof v === "number" && Number.isNaN(v)))
    .map(([k, v]) => `${k} = ${typeof v === "number" ? +v.toPrecision(6) : v}`)
    .join(", ");
  const lines = [
    `${d.name} (${d.type})`,
    `Tissue model: ${d.tissue.model}`,
    `α_x = ${+d.alpha_x.toPrecision(6)} Gy⁻¹, β_x = ${+d.beta_x.toPrecision(6)} Gy⁻²`,
    `α/β = ${ratio}`,
  ];
  if (params) lines.push(params);
  lines.push(`Overlap priority: ${d.overlap_priority}`, "Click the name to edit");
  return lines.join("\n");
}

const toHex = (rgb: readonly number[]) => `#${rgb.map((c) => Math.round(c).toString(16).padStart(2, "0")).join("")}`;
const fromHex = (hex: string): Vec3 => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)) as Vec3;

/** Mirror of `VOIsWidget`: groups by type or overlap, swatch (colour pick), checkbox, name (Edit VOI). */
export function StructureVisibility() {
  const structures = useApp((s) => s.state?.structures ?? EMPTY);
  const visible = useApp((s) => s.viewer.visible);
  const details = useApp((s) => s.structureDetails);
  const busy = useApp((s) => s.busy);
  const editVoi = useApp((s) => s.editVoi);
  const [mode, setMode] = useState<"type" | "overlap">("type");
  const colorInput = useRef<HTMLInputElement>(null);
  const picking = useRef<string | null>(null);
  const setAll = (value: boolean) =>
    setViewer({ visible: Object.fromEntries(structures.map((s) => [s.name, value && (s.num_voxels ?? 1) > 0])) });
  const groups = mode === "overlap" ? groupByOverlap(structures, details) : groupByType(structures);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1, minHeight: 0 }} data-testid="structures">
      <div className="voi-scroll">
        {structures.length === 0 && <div className="placeholder-note">No structures loaded</div>}
        {groups.map(([title, members]) => (
          <div className="subgroup" key={title} data-testid={`voi-group-${title}`}>
            <div className="sg-title" style={{ background: "var(--base)" }}>
              {title}
            </div>
            <div className="voi-grid">
              {members.map((s) => (
                <div className="voi-row" key={s.name} title={voiTooltip(s, details[s.name])}>
                  <button
                    className="swatch swatch-button"
                    style={{ background: `rgb(${s.color.join(",")})` }}
                    title={`Pick color for ${s.name}`}
                    disabled={busy !== null}
                    data-testid={`voi-color-${s.name}`}
                    onClick={() => {
                      picking.current = s.name;
                      const input = colorInput.current!;
                      input.value = toHex(s.color);
                      input.click();
                    }}
                  />
                  <input
                    type="checkbox"
                    checked={visible[s.name] ?? false}
                    onChange={(e) => setViewer((v) => ({ visible: { ...v.visible, [s.name]: e.target.checked } }))}
                    data-testid={`voi-${s.name}`}
                  />
                  <span
                    className="voi-name"
                    onClick={() => busy === null && useApp.setState({ editVoi: s.name })}
                    data-testid={`voi-name-${s.name}`}
                  >
                    {truncateName(s.name)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="row">
        <button className="btn small" style={{ width: 48 }} onClick={() => setAll(true)}>
          All
        </button>
        <button className="btn small" style={{ width: 48 }} onClick={() => setAll(false)}>
          None
        </button>
        <span className="fill" />
        <select
          value={mode}
          title="Group the VOI list by type or by overlap priority"
          onChange={(e) => setMode(e.target.value as "type" | "overlap")}
          data-testid="voi-group-mode"
        >
          <option value="type">By Type</option>
          <option value="overlap">By Overlap</option>
        </select>
      </div>
      <input
        ref={colorInput}
        type="color"
        hidden
        data-testid="voi-color-input"
        onChange={(e) => {
          const name = picking.current;
          picking.current = null;
          if (name) void updateStructure(name, { color: fromHex(e.target.value) });
        }}
      />
      {editVoi && <EditVoiDialog name={editVoi} onClose={() => useApp.setState({ editVoi: null })} />}
    </div>
  );
}

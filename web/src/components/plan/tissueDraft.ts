import type { TissueDraft } from "../../api/protocol";
import type { TissueSchema } from "../../api/protocol/plan";

/** The cell line identity a draft carries (absent from drafts of older backends). */
export function identityOf(tissue: TissueDraft): Pick<TissueDraft, "cell_line" | "cell_line_source"> {
  const out: Pick<TissueDraft, "cell_line" | "cell_line_source"> = {};
  if ("cell_line" in tissue) out.cell_line = tissue.cell_line;
  if ("cell_line_source" in tissue) out.cell_line_source = tissue.cell_line_source;
  return out;
}

/**
 * A draft whose parameters no longer describe its cell line: the identity is cleared. Qt keeps the
 * cell line of the tissue applied only while every parameter still matches it.
 */
export function withoutIdentity(tissue: TissueDraft): TissueDraft {
  return "cell_line" in tissue || "cell_line_source" in tissue ? { ...tissue, cell_line: null, cell_line_source: null } : tissue;
}

/** The draft under another model label; its cell line only survives a model of the same parameter class (MKM / mMKM). */
export function withModel(schema: TissueSchema | null, tissue: TissueDraft, model: TissueDraft["model"]): TissueDraft {
  const names = (label: string) => schema?.models.find((m) => m.label === label)?.fields.map((f) => f.name).join();
  const next = { ...tissue, model };
  return schema && names(tissue.model) === names(model) ? next : withoutIdentity(next);
}

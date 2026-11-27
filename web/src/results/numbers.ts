/** `parse_number_list` of the Qt GUI: blank / comma separated numbers, `null` when invalid. */
export function parseNumbers(text: string): number[] | null {
  const parts = text.split(/[\s,;]+/).filter(Boolean);
  const values = parts.map(Number);
  return values.some((v) => !Number.isFinite(v)) ? null : values;
}

/** `format_number_list`: shortest representation, blank separated. */
export function formatNumbers(values: readonly number[]): string {
  return values.map((v) => String(+v.toFixed(6))).join(" ");
}

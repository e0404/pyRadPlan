/**
 * Import Image dialog (Qt `_image_import_dialog.py`): what a bare image file represents. Only opened
 * when the backend reports image file IO; the browser runtime keeps Load File to container files.
 * The Qt dialog infers the default from the pixel data; without reading the image the default is
 * a guess from the file name.
 */
import { useState } from "react";
import { showDialog } from "../../store/dialogStore";
import type { LoadKind } from "../../api/protocol/io";
import { Dialog } from "../common";
import "./io.css";

export interface ImageImportChoice {
  kind: Exclude<LoadKind, "auto">;
  /** Result name of a dose import. */
  name?: string;
}

export function defaultDoseName(filename: string): string {
  const stem = (filename.split(/[\\/]/).pop() ?? filename).replace(/(\.nii\.gz|\.[^.]+)$/i, "");
  return stem.replace(/\W+/g, "_").replace(/^_+|_+$/g, "") || "imported_dose";
}

export function guessImageKind(filename: string, hasCt: boolean): ImageImportChoice["kind"] {
  const lower = filename.toLowerCase();
  if (hasCt && /dose|rtdose/.test(lower)) return "dose";
  if (hasCt && /mask|struct|seg|label|gtv|ptv|ctv|oar|\bpv\b/.test(lower)) return "structures";
  return "ct_new";
}

function ImportImageView(props: {
  filename: string;
  hasCt: boolean;
  close: (value: ImageImportChoice | null) => void;
  isTop: () => boolean;
}) {
  const [kind, setKind] = useState<ImageImportChoice["kind"]>(guessImageKind(props.filename, props.hasCt));
  const [name, setName] = useState(defaultDoseName(props.filename));
  const radio = (value: ImageImportChoice["kind"], label: string, enabled = true) => (
    <label className={`check radio ${enabled ? "" : "disabled"}`}>
      <input type="radio" name="kind" checked={kind === value} disabled={!enabled} data-testid={`import-${value}`} onChange={() => setKind(value)} />
      {label}
    </label>
  );
  return (
    <Dialog
      title="Import Image"
      testId="import-image-dialog"
      onClose={() => props.isTop() && props.close(null)}
      buttons={
        <>
          <button className="btn primary" data-testid="import-ok" onClick={() => props.close({ kind, ...(kind === "dose" ? { name: name.trim() || defaultDoseName(props.filename) } : {}) })}>
            OK
          </button>
          <button className="btn" onClick={() => props.close(null)}>
            Cancel
          </button>
        </>
      }
    >
      <div>
        Import <b>{props.filename}</b> as:
      </div>
      {radio("ct_new", "CT — new patient (clears all loaded data)")}
      {radio("ct_replace", "CT — replace current CT only", props.hasCt)}
      {radio("structures", "Structure(s) — add to structure set", props.hasCt)}
      {radio("dose", "Dose — add to result collection", props.hasCt)}
      {kind === "dose" && (
        <input type="text" className="indented" value={name} placeholder="Result name" data-testid="import-dose-name" onChange={(e) => setName(e.currentTarget.value)} />
      )}
    </Dialog>
  );
}

export function askImportImage(filename: string, hasCt: boolean): Promise<ImageImportChoice | null> {
  return showDialog<ImageImportChoice | null>(({ close, isTop }) => (
    <ImportImageView filename={filename} hasCt={hasCt} close={close} isTop={isTop} />
  ));
}

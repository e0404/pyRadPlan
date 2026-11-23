/** Save dialogs: the file-name dialog standing in for QFileDialog and the Save Result checklist. */
import { useState, type ReactNode } from "react";
import type { ExportFormat } from "../../api";
import { showDialog } from "../../store/dialogStore";
import { Dialog } from "../common";
import { baseNameOf } from "./models";
import "./io.css";

export interface SaveFileChoice {
  format: ExportFormat;
  basename: string;
}

function validName(name: string): string | null {
  if (!name.trim()) return "Enter a file name.";
  if (/[\\/]/.test(name)) return "A file name cannot contain slashes.";
  return null;
}

function SaveFileView(props: {
  title: string;
  formats: ExportFormat[];
  basename: string;
  note?: (format: ExportFormat) => ReactNode;
  close: (value: SaveFileChoice | null) => void;
  isTop: () => boolean;
}) {
  const firstAvailable = Math.max(
    0,
    props.formats.findIndex((f) => f.available),
  );
  const [index, setIndex] = useState(firstAvailable);
  const [name, setName] = useState(props.basename);
  const format = props.formats[index];
  const problem = validName(name) ?? (format?.available ? null : (format?.reason ?? "This format is not available."));
  const submit = () => {
    if (!problem && format) props.close({ format, basename: baseNameOf(name, format.extension) });
  };
  return (
    <Dialog
      title={props.title}
      testId="save-dialog"
      onClose={() => props.isTop() && props.close(null)}
      buttons={
        <>
          <button className="btn" onClick={() => props.close(null)} data-testid="save-cancel">
            Cancel
          </button>
          <button className="btn primary" disabled={problem !== null} onClick={submit} data-testid="save-ok">
            Save
          </button>
        </>
      }
    >
      <div className="form-rows">
        <label htmlFor="save-format">Format:</label>
        <select id="save-format" value={index} onChange={(e) => setIndex(Number(e.currentTarget.value))} data-testid="save-format" title={format?.reason ?? undefined}>
          {props.formats.map((f, i) => (
            <option key={`${f.format}-${f.label}`} value={i} disabled={!f.available} title={f.reason ?? undefined}>
              {f.available ? f.label : `${f.label} (not available)`}
            </option>
          ))}
        </select>
        <label htmlFor="save-name">File name:</label>
        <div className="row">
          <input
            id="save-name"
            type="text"
            className="fill"
            autoFocus
            value={name}
            data-testid="save-name"
            onChange={(e) => setName(e.currentTarget.value)}
            onFocus={(e) => e.currentTarget.select()}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
          />
          <span className="dim mono" data-testid="save-extension">
            {format?.extension}
          </span>
        </div>
      </div>
      {format?.reason && !format.available && <div className="form-error">{format.reason}</div>}
      {format?.note && <div className="dim">{format.note}</div>}
      {format && props.note && <div className="dim">{props.note(format)}</div>}
      <div className="dim">The file is saved through your browser's download.</div>
    </Dialog>
  );
}

/** Ask for the format and file name of a save (the browser has no save dialog of its own). */
export function askSaveFile(options: {
  title: string;
  formats: ExportFormat[];
  basename: string;
  /** Extra text for the selected format. */
  note?: (format: ExportFormat) => ReactNode;
}): Promise<SaveFileChoice | null> {
  return showDialog<SaveFileChoice | null>(({ close, isTop }) => <SaveFileView {...options} close={close} isTop={isTop} />);
}

export interface QuantityChoice {
  key: string;
  text: string;
}

function SaveResultView(props: {
  quantities: QuantityChoice[];
  close: (value: string[] | null) => void;
  isTop: () => boolean;
}) {
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const [status, setStatus] = useState("");
  const selected = props.quantities.filter((q) => checked[q.key]).map((q) => q.key);
  const setAll = (value: boolean) => setChecked(Object.fromEntries(props.quantities.map((q) => [q.key, value])));
  const accept = () => {
    if (!selected.length) {
      setStatus("Select at least one quantity to export.");
      return;
    }
    props.close(selected);
  };
  return (
    <Dialog
      title="Save Result"
      testId="save-result-dialog"
      onClose={() => props.isTop() && props.close(null)}
      buttons={
        <>
          <button className="btn primary" onClick={accept} data-testid="save-result-ok">
            OK
          </button>
          <button className="btn" onClick={() => props.close(null)} data-testid="save-result-cancel">
            Cancel
          </button>
        </>
      }
    >
      <div>Select the quantities to export:</div>
      <div className="check-list" data-testid="save-result-list">
        {props.quantities.map((q) => (
          <label key={q.key} className="check" title={q.text}>
            <input
              type="checkbox"
              checked={Boolean(checked[q.key])}
              data-testid={`save-quantity-${q.key}`}
              onChange={(e) => {
                setStatus("");
                setChecked({ ...checked, [q.key]: e.currentTarget.checked });
              }}
            />
            {q.text}
          </label>
        ))}
      </div>
      <div className="row">
        <button className="btn" onClick={() => setAll(true)} data-testid="save-result-all">
          All
        </button>
        <button className="btn" onClick={() => setAll(false)} data-testid="save-result-none">
          None
        </button>
      </div>
      <div className="form-error" data-testid="save-result-status">
        {status}
      </div>
    </Dialog>
  );
}

/** The Save Result dialog (Qt `SaveResultDialog`): resolves with the checked keys, or `null`. */
export function askSaveResult(quantities: QuantityChoice[]): Promise<string[] | null> {
  return showDialog<string[] | null>(({ close, isTop }) => <SaveResultView quantities={quantities} close={close} isTop={isTop} />);
}

import { useCallback, useEffect, useState } from "react";
import { feature } from "../io/capabilities";
import type { BaseDataKind, BaseDataList, BaseDataRow } from "../../api/protocol/io";
import { storeInternals, useApp } from "../../store/appStore";
import { addBaseDataFile, pickFile, removeBaseDataFile } from "../../store/ioActions";
import { useIo } from "../../store/ioStore";
import { showDialog } from "../../store/dialogStore";
import { ModalDialog } from "../forms/ModalDialog";

type ListedKind = Exclude<BaseDataKind, "phantoms">;

const SOURCE_LABELS: Record<string, string> = { bundled: "pyRadPlan", user: "User", shared: "Shared" };

interface KindView {
  kind: ListedKind;
  title: string;
  columns: string[];
  cells: (row: BaseDataRow) => string[];
}

const source = (row: BaseDataRow) => SOURCE_LABELS[row.source] ?? row.source;

const VIEWS: KindView[] = [
  {
    kind: "machines",
    title: "Machines",
    columns: ["Radiation mode", "Name", "Source", "File"],
    cells: (r) => [r.radiation_mode ?? "", r.name, source(r), r.file],
  },
  {
    kind: "rbe_tables",
    title: "RBE Tables",
    columns: ["File", "Model", "Tissue classes", "Source", "Reference"],
    cells: (r) =>
      r.problem !== undefined && r.problem !== null && r.problem !== ""
        ? [r.file, "unreadable", "", source(r), ""]
        : [r.file, r.model ?? "", r.tissue_classes ?? "", source(r), r.reference ?? ""],
  },
  {
    kind: "stopping_power_tables",
    title: "Stopping Power Tables",
    columns: ["File", "Source"],
    cells: (r) => [r.file, source(r)],
  },
];

const FOLDER_LABELS: Record<string, string> = { bundled: "Shipped with pyRadPlan", user: "Added by you", shared: "Shared" };

function KindTab(props: { view: KindView; listing: BaseDataList | null; onChanged: () => void }) {
  const { view, listing } = props;
  const info = useApp((s) => s.info);
  const openFolder = feature(info, "open_folder");
  const persistence = feature(info, "base_data_persistence");
  const [selected, setSelected] = useState<string | null>(null);
  const rows = listing?.rows ?? [];
  const entry = rows.find((r) => r.file === selected);
  const folderOf = (row: BaseDataRow) => listing?.folders.find((f) => f.source === row.source)?.path;

  const add = async () => {
    const file = await pickFile(".mat");
    if (file && (await addBaseDataFile(view.kind, file))) props.onChanged();
  };
  const remove = async () => {
    if (entry && (await removeBaseDataFile(view.kind, entry.file, entry.name))) {
      setSelected(null);
      props.onChanged();
    }
  };

  return (
    <div className="kind-tab" data-testid={`base-data-tab-${view.kind}`}>
      <div className="folder-lines" data-testid={`base-data-folders-${view.kind}`}>
        {(listing?.folders ?? []).map((f) => (
          <div key={f.source + f.path}>
            {FOLDER_LABELS[f.source] ?? f.label}: {f.path}
          </div>
        ))}
        {!persistence.available && <div className="dim">Files added here last until the page is closed.</div>}
      </div>
      <div className="table-wrap base-data-table">
        <table className="qt selectable">
          <thead>
            <tr>
              {view.columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const problem = row.problem ? row.problem : "";
              const tooltip = [row.path ?? (folderOf(row) ? `${folderOf(row)}/${row.file}` : row.file), problem].filter(Boolean).join("\n");
              return (
                <tr
                  key={row.file}
                  className={`${selected === row.file ? "selected" : ""} ${problem ? "problem" : ""}`}
                  title={tooltip}
                  onClick={() => setSelected(row.file)}
                  data-testid={`base-data-row-${row.file}`}
                >
                  {view.cells(row).map((text, i) => (
                    <td key={i}>{text}</td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="row">
        <button className="btn" onClick={() => void add()} data-testid={`base-data-add-${view.kind}`}>
          Add…
        </button>
        <button
          className="btn"
          disabled={!entry?.removable}
          title="Only files you added can be removed"
          onClick={() => void remove()}
          data-testid={`base-data-remove-${view.kind}`}
        >
          Remove
        </button>
        <span className="spacer" />
        <button className="btn" disabled={!openFolder.available} title={openFolder.reason ?? listing?.folders.find((f) => f.source === "user")?.path} data-testid={`base-data-open-${view.kind}`}>
          Open folder
        </button>
      </div>
    </div>
  );
}

/** Qt `BaseDataDialog`: tabs for machines, RBE tables and stopping power tables. */
export function BaseDataDialog(props: { onClose: () => void; isTop?: () => boolean }) {
  const [tab, setTab] = useState<ListedKind>("machines");
  const [listings, setListings] = useState<Partial<Record<ListedKind, BaseDataList>>>({});
  const [error, setError] = useState<string | null>(null);
  const version = useIo((s) => s.baseDataVersion);

  const load = useCallback(async () => {
    try {
      const entries = await Promise.all(VIEWS.map(async (v) => [v.kind, await storeInternals.call("base_data.list", { kind: v.kind })] as const));
      setListings(Object.fromEntries(entries));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, version]);

  const view = VIEWS.find((v) => v.kind === tab) ?? VIEWS[0];
  return (
    <ModalDialog
      title="Base Data"
      testId="base-data-dialog"
      width={760}
      height={460}
      onClose={props.onClose}
      isTop={props.isTop}
      buttons={
        <button className="btn" onClick={props.onClose} data-testid="base-data-close">
          Close
        </button>
      }
    >
      <div className="tabs" role="tablist">
        {VIEWS.map((v) => (
          <button key={v.kind} role="tab" aria-selected={tab === v.kind} className={`tab ${tab === v.kind ? "active" : ""}`} onClick={() => setTab(v.kind)} data-testid={`base-data-tab-btn-${v.kind}`}>
            {v.title}
          </button>
        ))}
      </div>
      <div className="tab-panel">
        {error && <div className="form-error">{error}</div>}
        <KindTab key={view.kind} view={view} listing={listings[view.kind] ?? null} onChanged={() => void load()} />
      </div>
    </ModalDialog>
  );
}

export function openBaseDataDialog(): Promise<void> {
  return showDialog<void>(({ close, isTop }) => <BaseDataDialog onClose={() => close()} isTop={isTop} />);
}

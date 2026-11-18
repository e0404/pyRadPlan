import { useEffect, useMemo, useState } from "react";
import type { LicenseDetail, LicenseRow } from "../../api/protocol/io";
import { showDialog } from "../../store/dialogStore";
import { storeInternals } from "../../store/appStore";
import { ModalDialog } from "../forms/ModalDialog";

export const NO_LICENSE_TEXT = "No license text found for this distribution.";

/** Qt `ThirdPartyLicensesDialog`: filter, package table, license text of the selection. */
export function LicensesDialog(props: { onClose: () => void; isTop?: () => boolean }) {
  const [rows, setRows] = useState<LicenseRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [texts, setTexts] = useState<Record<string, string>>({});

  useEffect(() => {
    let alive = true;
    storeInternals
      .call("app.licenses", {})
      .then((result) => alive && setRows(result as LicenseRow[]))
      .catch((err: unknown) => alive && setError(err instanceof Error ? err.message : String(err)));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!selected || texts[selected] !== undefined) return;
    let alive = true;
    storeInternals
      .call("app.licenses", { name: selected })
      .then((result) => {
        const text = (result as LicenseDetail).text;
        if (alive) setTexts((t) => ({ ...t, [selected]: text?.trim() ? text : NO_LICENSE_TEXT }));
      })
      .catch(() => alive && setTexts((t) => ({ ...t, [selected]: NO_LICENSE_TEXT })));
    return () => {
      alive = false;
    };
  }, [selected, texts]);

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    return (rows ?? []).filter((r) => !needle || r.name.toLowerCase().includes(needle));
  }, [rows, filter]);

  return (
    <ModalDialog
      title="Third-party Licenses"
      testId="licenses-dialog"
      width={900}
      height={560}
      onClose={props.onClose}
      isTop={props.isTop}
      buttons={
        <button className="btn" onClick={props.onClose} data-testid="licenses-close">
          Close
        </button>
      }
    >
      <div>pyRadPlan builds on the following open-source packages. Select one to read its license.</div>
      <div className="clearable">
        <input
          type="text"
          className="fill"
          placeholder="Filter by name…"
          value={filter}
          data-testid="licenses-filter"
          onChange={(e) => setFilter(e.currentTarget.value)}
        />
        {filter && (
          <button className="clear" aria-label="Clear filter" onClick={() => setFilter("")}>
            ×
          </button>
        )}
      </div>
      <div className="splitter">
        <div className="table-wrap licenses-table">
          <table className="qt selectable" data-testid="licenses-table">
            <thead>
              <tr>
                <th>Package</th>
                <th>Version</th>
                <th>License</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((r) => (
                <tr key={r.name} className={selected === r.name ? "selected" : ""} onClick={() => setSelected(r.name)} data-testid={`license-row-${r.name}`}>
                  <td>{r.name}</td>
                  <td>{r.version}</td>
                  <td>{r.license}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows === null && !error && <div className="placeholder-note">Loading…</div>}
          {error && <div className="form-error">{error}</div>}
        </div>
        <textarea
          readOnly
          className="license-text mono"
          placeholder="Select a package to show its license."
          value={selected ? (texts[selected] ?? "") : ""}
          data-testid="license-text"
        />
      </div>
    </ModalDialog>
  );
}

export function openLicensesDialog(): Promise<void> {
  return showDialog<void>(({ close, isTop }) => <LicensesDialog onClose={() => close()} isTop={isTop} />);
}

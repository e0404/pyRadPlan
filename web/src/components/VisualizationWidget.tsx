import { useState } from "react";
import { DEFAULT_VIEW, slicesThrough } from "../renderer/geometry";
import { parseNumbers } from "../results/numbers";
import { setViewer, useApp, type ViewerSettings } from "../store/appStore";
import { displayIsocenter, resultGroups, selectQuantity } from "../store/revisions";
import { AnalysisWindow } from "./analysis/AnalysisWindow";
import { QuantitySelector } from "./QuantitySelector";
import { Dialog } from "./common";

type Toggle = keyof Pick<ViewerSettings, "showCT" | "showCST" | "showQuantity" | "showIsolines" | "showIsocenter" | "showOrientation">;

const BEAMS_TOOLTIP =
  "Draw each beam as a line running from its source towards the isocenter,\n" +
  "projected onto the displayed plane.\n" +
  "A beam leaving the plane is drawn as a wedge: its source end is wider when\n" +
  "the source lies in front of the plane and narrower when it lies behind it,\n" +
  "and its label is marked ⊙ (coming out of the screen) or ⊗ (going into it).\n" +
  "A beam along the viewing direction gets that mark at the isocenter.\n" +
  "Drag a beam's source marker along its dashed orbit to change its gantry angle.";

/** Qt "Recenter": slices through the isocenter and the full view. */
export function recenterToIsocenter(): void {
  const { ct } = useApp.getState();
  const iso = displayIsocenter();
  if (!ct || !iso) {
    setViewer({ view: DEFAULT_VIEW });
    return;
  }
  setViewer({ slices: slicesThrough(ct.grid, iso), view: DEFAULT_VIEW });
}

/** Mirror of `VisualizationWidget`: overlay toggles, overlay quantity, isolines, DVH / QI. */
export function VisualizationWidget() {
  const viewer = useApp((s) => s.viewer);
  const state = useApp((s) => s.state);
  const hasResult = useApp((s) => Boolean(s.state?.has.result));
  const beams = useApp((s) => s.beams);
  const showBeams = useApp((s) => s.showBeams);
  const [isoDialog, setIsoDialog] = useState(false);
  const [isoText, setIsoText] = useState("");
  const groups = resultGroups(state);
  const beamsAvailable = Boolean(beams?.beams.length);
  const analysisOpen = useApp((s) => s.analysisOpen);

  const toggle = (key: Toggle, label: string, testId: string, title?: string) => (
    <label className="check" title={title}>
      <input
        type="checkbox"
        checked={viewer[key]}
        onChange={(e) => setViewer({ [key]: e.target.checked } as Partial<ViewerSettings>)}
        data-testid={testId}
      />
      {label}
    </label>
  );

  return (
    <div data-testid="visualization">
      <div className="vis-grid">
        {toggle("showCT", "CT", "toggle-ct")}
        {toggle("showCST", "CST", "toggle-cst")}
        {toggle("showQuantity", "Quantity", "toggle-quantity")}
        {toggle("showIsolines", "Isolines", "toggle-isolines")}
        {toggle("showIsocenter", "Isocenter", "toggle-isocenter")}
        <label className="check" title={beamsAvailable ? BEAMS_TOOLTIP : "Configure a plan with gantry angles first"}>
          <input
            type="checkbox"
            disabled={!beamsAvailable}
            checked={showBeams && beamsAvailable}
            onChange={(e) => useApp.setState({ showBeams: e.target.checked })}
            data-testid="toggle-beams"
          />
          Beams
        </label>
        {toggle(
          "showOrientation",
          "Orientation",
          "toggle-orientation",
          "Show a small figure of the patient in the corner of the viewer,\nlying as the patient does in the displayed plane.",
        )}
      </div>
      <div className="row" style={{ marginTop: 4 }}>
        <span>Overlay quantity:</span>
        <QuantitySelector
          groups={groups}
          value={viewer.activeQuantity}
          scenarioLabels={state?.results?.scenario_labels}
          onChange={(key) => void selectQuantity(key)}
          testId="quantity-select"
        />
      </div>
      <div className="row" style={{ marginTop: 4 }}>
        <button
          className="btn fill"
          title="Set custom isoline levels (e.g. 30 50 95)"
          onClick={() => {
            setIsoText("");
            setIsoDialog(true);
          }}
        >
          Set isolines
        </button>
        <button className="btn fill" title="Recenter view to isocenter" onClick={recenterToIsocenter} data-testid="btn-recenter">
          Recenter
        </button>
      </div>
      <button
        className="btn"
        style={{ width: "100%", marginTop: 4 }}
        title="Show DVH and QI analysis window"
        disabled={!hasResult}
        onClick={() => useApp.setState({ analysisOpen: true })}
        data-testid="btn-dvh"
      >
        Show DVH / QI
      </button>

      {analysisOpen && hasResult && <AnalysisWindow onClose={() => useApp.setState({ analysisOpen: false })} />}
      {isoDialog && (
        <Dialog
          title="Set isolines"
          onClose={() => setIsoDialog(false)}
          buttons={
            <>
              <button
                className="btn"
                onClick={() => {
                  setViewer({ isolineLevels: [] });
                  setIsoDialog(false);
                }}
              >
                Reset
              </button>
              <button className="btn" onClick={() => setIsoDialog(false)}>
                Cancel
              </button>
              <button
                className="btn primary"
                data-testid="isolines-ok"
                onClick={() => {
                  // As Qt: unparsable text is ignored, empty text keeps the levels.
                  const levels = parseNumbers(isoText);
                  if (isoText.trim() && levels) setViewer({ isolineLevels: levels, showIsolines: true });
                  setIsoDialog(false);
                }}
              >
                OK
              </button>
            </>
          }
        >
          <label>Enter isoline levels (Gy) separated by space:</label>
          <input type="text" autoFocus value={isoText} onChange={(e) => setIsoText(e.target.value)} data-testid="isolines-input" />
        </Dialog>
      )}
    </div>
  );
}

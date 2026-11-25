import { showDialog } from "../../store/dialogStore";
import { ModalDialog } from "../forms/ModalDialog";
import { ABOUT, PRIVACY_TITLE, acknowledgeNotice, privacyParagraphs, useNotice } from "./notice";

/** Disclaimer and privacy notice; `firstVisit` asks for the acknowledgement before the app starts. */
export function NoticeDialog(props: { firstVisit?: boolean; onClose: () => void; isTop?: () => boolean }) {
  const first = props.firstVisit ?? false;
  return (
    <ModalDialog
      title="pyRadPlan in the browser"
      testId="notice-dialog"
      className="notice-dialog"
      backdropClassName={first ? "notice-backdrop" : ""}
      onClose={props.onClose}
      isTop={props.isTop}
      dismissible={!first}
      buttons={
        first ? (
          <button className="btn primary" onClick={props.onClose} autoFocus data-testid="notice-accept">
            I understand, continue
          </button>
        ) : (
          <button className="btn primary" onClick={props.onClose} autoFocus data-testid="notice-close">
            Close
          </button>
        )
      }
    >
      <div className="disclaimer-box" data-testid="notice-disclaimer">
        <b>{ABOUT.disclaimer_title}</b>
        <br />
        {ABOUT.disclaimer}
      </div>
      <div>{ABOUT.license_note}</div>
      <div className="notice-heading">{PRIVACY_TITLE}</div>
      {privacyParagraphs().map((text) => (
        <p key={text} className="notice-paragraph">
          {text}
        </p>
      ))}
      <p className="notice-paragraph" data-testid="notice-legal">
        Details:{" "}
        <a href={ABOUT.privacy_url} target="_blank" rel="noreferrer">
          privacy policy
        </a>{" "}
        and{" "}
        <a href={ABOUT.imprint_url} target="_blank" rel="noreferrer">
          imprint
        </a>
        .
      </p>
      {first && <div className="dim">pyRadPlan starts loading after you continue.</div>}
    </ModalDialog>
  );
}

/** First visit: the notice covers the page until it is acknowledged. */
export function NoticeGate() {
  const acknowledged = useNotice((s) => s.acknowledged);
  if (acknowledged) return null;
  return <NoticeDialog firstVisit onClose={acknowledgeNotice} />;
}

export function openNoticeDialog(): Promise<void> {
  return showDialog<void>(({ close, isTop }) => <NoticeDialog onClose={() => close()} isTop={isTop} />);
}

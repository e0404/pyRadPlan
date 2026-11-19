import type { AppInfo } from "../../api";
import pyradplanLogo from "../../assets/pyradplan_logo_full_landscape.svg";
import { showDialog } from "../../store/dialogStore";
import { DOCUMENTATION_URL, SOURCE_URL } from "../io/menuModel";
import { ModalDialog } from "../forms/ModalDialog";
import { openLicensesDialog } from "./LicensesDialog";
import { ABOUT, buildLabel } from "./notice";
import { openNoticeDialog } from "./NoticeDialog";

/** Qt `AboutDialog`: logo, version, description, links, license note, disclaimer; plus the web build and privacy. */
export function AboutDialog(props: { info: AppInfo | null; onClose: () => void; isTop?: () => boolean }) {
  const { info } = props;
  const build = buildLabel();
  return (
    <ModalDialog
      title="About pyRadPlan"
      testId="about-dialog"
      width={460}
      className="about-dialog"
      onClose={props.onClose}
      isTop={props.isTop}
      leftButtons={
        <>
          <button className="btn" onClick={() => void openLicensesDialog()} data-testid="about-licenses">
            Third-party Licenses…
          </button>
          <button className="btn" onClick={() => void openNoticeDialog()} data-testid="about-notice">
            Privacy…
          </button>
        </>
      }
      buttons={
        <button className="btn primary" onClick={props.onClose} autoFocus data-testid="about-close">
          Close
        </button>
      }
    >
      <div className="about">
        <img src={pyradplanLogo} alt="pyRadPlan" className="about-logo" />
        <div data-testid="about-title">
          <b>pyRadPlan</b> {info?.version ?? ""} (Python {info?.python ?? "…"})
        </div>
        <div>{ABOUT.description}</div>
        <div>
          <a href={DOCUMENTATION_URL} target="_blank" rel="noreferrer">
            Documentation
          </a>{" "}
          &nbsp;|&nbsp;{" "}
          <a href={SOURCE_URL} target="_blank" rel="noreferrer">
            Source code
          </a>
        </div>
        <div>{ABOUT.license_note}</div>
        <div className="disclaimer-box">
          <b>{ABOUT.disclaimer_title}</b>
          <br />
          {ABOUT.disclaimer}
        </div>
        <div className="dim">All computation and data stay in this browser; no telemetry.</div>
        <div className="dim" data-testid="about-build">
          Web build: {build ?? "local build"}
        </div>
      </div>
    </ModalDialog>
  );
}

export function openAboutDialog(info: AppInfo | null): Promise<void> {
  return showDialog<void>(({ close, isTop }) => <AboutDialog info={info} onClose={() => close()} isTop={isTop} />);
}

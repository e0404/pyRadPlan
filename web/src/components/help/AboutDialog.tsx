import type { AppInfo } from "../../api";
import pyradplanLogo from "../../assets/pyradplan_logo_full_landscape.svg";
import { showDialog } from "../../store/dialogStore";
import { DOCUMENTATION_URL, SOURCE_URL } from "../io/menuModel";
import { ModalDialog } from "../forms/ModalDialog";
import { openLicensesDialog } from "./LicensesDialog";

export const DISCLAIMER =
  "pyRadPlan is intended for research and education. It is not a medical device and must not be used for clinical treatment planning.";
export const LICENSE_NOTE = "Released under the BSD 3-Clause License. Copyright (c) 2024-2026, DKFZ.";
export const DESCRIPTION = "Python radiotherapy treatment planning toolkit, interoperable with matRad.";

/** Qt `AboutDialog`: logo, version, description, links, license note, disclaimer. */
export function AboutDialog(props: { info: AppInfo | null; onClose: () => void; isTop?: () => boolean }) {
  const { info } = props;
  return (
    <ModalDialog
      title="About pyRadPlan"
      testId="about-dialog"
      width={460}
      onClose={props.onClose}
      isTop={props.isTop}
      leftButtons={
        <button className="btn" onClick={() => void openLicensesDialog()} data-testid="about-licenses">
          Third-party Licenses…
        </button>
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
        <div>{DESCRIPTION}</div>
        <div>
          <a href={DOCUMENTATION_URL} target="_blank" rel="noreferrer">
            Documentation
          </a>{" "}
          &nbsp;|&nbsp;{" "}
          <a href={SOURCE_URL} target="_blank" rel="noreferrer">
            Source code
          </a>
        </div>
        <div>{LICENSE_NOTE}</div>
        <div className="disclaimer-box">
          <b>NOT FOR CLINICAL USE</b>
          <br />
          {DISCLAIMER}
        </div>
      </div>
    </ModalDialog>
  );
}

export function openAboutDialog(info: AppInfo | null): Promise<void> {
  return showDialog<void>(({ close, isTop }) => <AboutDialog info={info} onClose={() => close()} isTop={isTop} />);
}

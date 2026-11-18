import { useEffect, type CSSProperties, type ReactNode } from "react";
import "./forms.css";
import "../io/io.css";

/**
 * Modal dialog like a Qt `QDialog.exec()`: title bar with a close button, Escape rejects, a click
 * beside it does not (unlike the message-box style `Dialog` in `common.tsx`). Used by the plan
 * dialogs and the larger dialogs (About, Licenses, Base Data, Preferences).
 */
export function ModalDialog(props: {
  title: string;
  children: ReactNode;
  /** Buttons on the right (Close, OK / Cancel). */
  buttons?: ReactNode;
  /** Buttons on the left (Qt "action role" buttons). */
  leftButtons?: ReactNode;
  onClose: () => void;
  /** Escape only acts on the topmost dialog; without it a message box on top takes Escape. */
  isTop?: () => boolean;
  testId?: string;
  className?: string;
  style?: CSSProperties;
  width?: number | string;
  height?: number | string;
}) {
  const { onClose, isTop } = props;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      const top = isTop ? isTop() : !document.querySelector(".modal-backdrop:not(.qt-modal)");
      if (top) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, isTop]);
  const sized = props.width !== undefined || props.height !== undefined;
  return (
    <div className="modal-backdrop qt-modal">
      <div
        className={`dialog qt-dialog ${sized ? "wide" : ""} ${props.className ?? ""}`}
        role="dialog"
        aria-label={props.title}
        data-testid={props.testId}
        style={{ ...props.style, ...(props.width !== undefined ? { width: props.width } : {}), ...(props.height !== undefined ? { height: props.height } : {}) }}
      >
        <div className="title">
          <span>{props.title}</span>
          <button type="button" className="title-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        <div className="body">{props.children}</div>
        {(props.buttons || props.leftButtons) && (
          <div className="buttons">
            {props.leftButtons}
            {props.leftButtons && <span className="spacer" />}
            {props.buttons}
          </div>
        )}
      </div>
    </div>
  );
}

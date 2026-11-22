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
  /** Extra class of the backdrop (stacking above the splash). */
  backdropClassName?: string;
  style?: CSSProperties;
  width?: number | string;
  height?: number | string;
  /** `false`: no title-bar close button and Escape does nothing; only the buttons close it. */
  dismissible?: boolean;
}) {
  const { onClose, isTop } = props;
  const dismissible = props.dismissible ?? true;
  useEffect(() => {
    if (!dismissible) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      const top = isTop ? isTop() : !document.querySelector(".modal-backdrop:not(.qt-modal)");
      if (top) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, isTop, dismissible]);
  const sized = props.width !== undefined || props.height !== undefined;
  return (
    <div className={`modal-backdrop qt-modal ${props.backdropClassName ?? ""}`}>
      <div
        className={`dialog qt-dialog ${sized ? "wide" : ""} ${props.className ?? ""}`}
        role="dialog"
        aria-label={props.title}
        data-testid={props.testId}
        style={{ ...props.style, ...(props.width !== undefined ? { width: props.width } : {}), ...(props.height !== undefined ? { height: props.height } : {}) }}
      >
        <div className="title">
          <span>{props.title}</span>
          {dismissible && (
            <button type="button" className="title-close" onClick={onClose} aria-label="Close">
              ✕
            </button>
          )}
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

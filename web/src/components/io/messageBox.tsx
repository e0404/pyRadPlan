/** QMessageBox / QInputDialog look-alikes on top of `showDialog`. */
import { useState, type ReactNode } from "react";
import { showDialog } from "../../store/dialogStore";
import { Dialog } from "../common";
import "./io.css";

export type MessageIcon = "info" | "warning" | "question" | "critical";

export interface MessageButton {
  id: string;
  label: string;
  /** The default button (focused, highlighted). */
  primary?: boolean;
  /** Result of Escape and of a click on the backdrop. */
  reject?: boolean;
}

export interface MessageBoxOptions {
  title: string;
  text: ReactNode;
  informativeText?: ReactNode;
  icon?: MessageIcon;
  buttons?: MessageButton[];
  details?: string;
  testId?: string;
}

const ICON_GLYPH: Record<MessageIcon, string> = { info: "i", warning: "!", question: "?", critical: "×" };

export function MessageIconBadge(props: { icon: MessageIcon }) {
  return <div className={`icon ${props.icon}`}>{ICON_GLYPH[props.icon]}</div>;
}

function MessageBoxView(props: MessageBoxOptions & { close: (id: string) => void; isTop: () => boolean }) {
  const buttons = props.buttons ?? [{ id: "ok", label: "OK", primary: true }];
  const rejectId = buttons.find((b) => b.reject)?.id ?? buttons[buttons.length - 1].id;
  const [showDetails, setShowDetails] = useState(false);
  return (
    <Dialog
      title={props.title}
      testId={props.testId ?? "message-box"}
      onClose={() => props.isTop() && props.close(rejectId)}
      buttons={
        <>
          {props.details && (
            <button className="btn" onClick={() => setShowDetails(!showDetails)}>
              {showDetails ? "Hide Details" : "Show Details…"}
            </button>
          )}
          {buttons.map((b) => (
            <button
              key={b.id}
              className={`btn ${b.primary ? "primary" : ""}`}
              autoFocus={b.primary}
              data-testid={`msg-${b.id}`}
              onClick={() => props.close(b.id)}
            >
              {b.label}
            </button>
          ))}
        </>
      }
    >
      <div className="icon-row">
        <MessageIconBadge icon={props.icon ?? "info"} />
        <div className="msg-text">
          <div>{props.text}</div>
          {props.informativeText && <div className="msg-informative">{props.informativeText}</div>}
        </div>
      </div>
      {showDetails && props.details && <pre>{props.details}</pre>}
    </Dialog>
  );
}

/** Show a message box; resolves with the id of the clicked button (Escape: the reject button). */
export function messageBox(options: MessageBoxOptions): Promise<string> {
  return showDialog<string>(({ close, isTop }) => <MessageBoxView {...options} close={close} isTop={isTop} />);
}

export function showInfo(title: string, text: ReactNode, testId?: string): Promise<string> {
  return messageBox({ title, text, icon: "info", testId });
}

export function showWarning(title: string, text: ReactNode, testId?: string): Promise<string> {
  return messageBox({ title, text, icon: "warning", testId });
}

export function showCritical(title: string, text: ReactNode, details?: string, testId?: string): Promise<string> {
  return messageBox({ title, text, icon: "critical", details, testId });
}

/** Yes / No question (default No, like Qt's quit and remove questions). */
export async function askYesNo(title: string, text: ReactNode, testId?: string): Promise<boolean> {
  const answer = await messageBox({
    title,
    text,
    icon: "question",
    testId,
    buttons: [
      { id: "yes", label: "Yes" },
      { id: "no", label: "No", primary: true, reject: true },
    ],
  });
  return answer === "yes";
}

function TextPrompt(props: {
  title: string;
  label: ReactNode;
  value: string;
  placeholder?: string;
  validate?: (value: string) => string | null;
  close: (value: string | null) => void;
  isTop: () => boolean;
}) {
  const [value, setValue] = useState(props.value);
  const problem = props.validate?.(value) ?? null;
  const submit = () => {
    if (!problem) props.close(value);
  };
  return (
    <Dialog
      title={props.title}
      testId="prompt-dialog"
      onClose={() => props.isTop() && props.close(null)}
      buttons={
        <>
          <button className="btn" onClick={() => props.close(null)} data-testid="prompt-cancel">
            Cancel
          </button>
          <button className="btn primary" disabled={problem !== null} onClick={submit} data-testid="prompt-ok">
            OK
          </button>
        </>
      }
    >
      <label className="prompt-label">{props.label}</label>
      <input
        type="text"
        autoFocus
        value={value}
        placeholder={props.placeholder}
        data-testid="prompt-input"
        onChange={(e) => setValue(e.currentTarget.value)}
        onFocus={(e) => e.currentTarget.select()}
        onKeyDown={(e) => {
          if (e.key === "Enter") submit();
        }}
      />
      {problem && <div className="form-error">{problem}</div>}
    </Dialog>
  );
}

/** QInputDialog.getText: resolves with the text, or `null` when cancelled. */
export function promptText(options: {
  title: string;
  label: ReactNode;
  value?: string;
  placeholder?: string;
  validate?: (value: string) => string | null;
}): Promise<string | null> {
  return showDialog<string | null>(({ close, isTop }) => (
    <TextPrompt {...options} value={options.value ?? ""} close={close} isTop={isTop} />
  ));
}

function ItemPrompt(props: {
  title: string;
  label: ReactNode;
  items: Array<{ label: string; enabled?: boolean; reason?: string }>;
  index: number;
  close: (value: number | null) => void;
  isTop: () => boolean;
}) {
  const [index, setIndex] = useState(props.index);
  return (
    <Dialog
      title={props.title}
      testId="item-dialog"
      onClose={() => props.isTop() && props.close(null)}
      buttons={
        <>
          <button className="btn" onClick={() => props.close(null)}>
            Cancel
          </button>
          <button className="btn primary" onClick={() => props.close(index)} data-testid="item-ok">
            OK
          </button>
        </>
      }
    >
      <label className="prompt-label">{props.label}</label>
      <select value={index} autoFocus onChange={(e) => setIndex(Number(e.currentTarget.value))} data-testid="item-select">
        {props.items.map((item, i) => (
          <option key={item.label} value={i} disabled={item.enabled === false} title={item.reason}>
            {item.label}
          </option>
        ))}
      </select>
    </Dialog>
  );
}

/** QInputDialog.getItem (not editable): resolves with the index, or `null` when cancelled. */
export function promptItem(options: {
  title: string;
  label: ReactNode;
  items: Array<{ label: string; enabled?: boolean; reason?: string }>;
  index?: number;
}): Promise<number | null> {
  return showDialog<number | null>(({ close, isTop }) => (
    <ItemPrompt {...options} index={options.index ?? 0} close={close} isTop={isTop} />
  ));
}

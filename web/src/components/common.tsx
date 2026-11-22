import { useEffect, useState, type ReactNode } from "react";

export function GroupBox(props: { title: string; className?: string; children: ReactNode; testId?: string }) {
  return (
    <section className={`groupbox ${props.className ?? ""}`} data-testid={props.testId}>
      <div className="gb-title">{props.title}</div>
      <div className="gb-body">{props.children}</div>
    </section>
  );
}

export function SubGroup(props: { title: string; children: ReactNode }) {
  return (
    <div className="subgroup">
      <div className="sg-title">{props.title}</div>
      {props.children}
    </div>
  );
}

export function ProgressBar(props: { fraction: number | null; testId?: string }) {
  const determinate = props.fraction !== null && Number.isFinite(props.fraction);
  const pct = determinate ? Math.round(Math.min(1, Math.max(0, props.fraction!)) * 100) : 0;
  return (
    <div
      className={`progress ${determinate ? "" : "indeterminate"}`}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={determinate ? pct : undefined}
      data-testid={props.testId}
    >
      <div className="bar" style={determinate ? { width: `${pct}%` } : undefined} />
    </div>
  );
}

export function Dialog(props: {
  title: string;
  children: ReactNode;
  buttons: ReactNode;
  onClose: () => void;
  testId?: string;
}) {
  const { onClose } = props;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" role="dialog" aria-label={props.title} data-testid={props.testId}>
        <div className="title">{props.title}</div>
        <div className="body">{props.children}</div>
        <div className="buttons">{props.buttons}</div>
      </div>
    </div>
  );
}

/**
 * Number input like a Qt spin box with keyboard tracking: every valid edit is committed,
 * clamped to [min, max]; the text follows outside changes while the field is not focused.
 */
export function NumberField(props: {
  value: number;
  onChange: (value: number) => void;
  step?: number;
  min?: number;
  max?: number;
  decimals?: number;
  disabled?: boolean;
  className?: string;
  title?: string;
  testId?: string;
}) {
  const { value, decimals } = props;
  const format = (v: number) => (decimals !== undefined ? v.toFixed(decimals) : String(v));
  const [text, setText] = useState(format(value));
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setText(format(value));
  }, [value, focused, decimals]);
  const commit = (raw: string) => {
    const v = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(v)) return;
    let clamped = v;
    if (props.min !== undefined) clamped = Math.max(props.min, clamped);
    if (props.max !== undefined) clamped = Math.min(props.max, clamped);
    if (clamped !== value) props.onChange(clamped);
  };
  return (
    <input
      type="number"
      value={text}
      step={props.step}
      min={props.min}
      max={props.max}
      disabled={props.disabled}
      className={props.className}
      title={props.title}
      data-testid={props.testId}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      onChange={(e) => {
        setText(e.currentTarget.value);
        commit(e.currentTarget.value);
      }}
    />
  );
}

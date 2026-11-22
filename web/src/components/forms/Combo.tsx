/**
 * Qt-like combo box and menu button: the closed box shows the display text; the open popup
 * prints the Python identifier faintly after each entry (`show_identifiers_when_open`),
 * disabled entries keep their reason as tooltip, and section headers / separators are
 * supported (tissue picker). An editable combo (`QComboBox.setEditable`) accepts free text.
 */
import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from "react";
import "./forms.css";

export interface ComboOption {
  key: string;
  label: string;
  identifier?: string;
  disabled?: boolean;
  /** Tooltip (description, or why the entry is disabled). */
  title?: string;
  /** Non-selectable italic section header. */
  header?: boolean;
  separator?: boolean;
  bold?: boolean;
}

const selectable = (o: ComboOption) => !o.header && !o.separator && !o.disabled;

function PopupList(props: {
  anchor: HTMLElement;
  options: ComboOption[];
  current: string | null;
  onPick: (key: string) => void;
  onClose: () => void;
  minWidth?: number;
  testId?: string;
}) {
  const { anchor, onClose } = props;
  const ref = useRef<HTMLDivElement>(null);
  const [style, setStyle] = useState<CSSProperties>({ visibility: "hidden" });
  const [active, setActive] = useState<string | null>(props.current);

  useLayoutEffect(() => {
    const r = anchor.getBoundingClientRect();
    const el = ref.current;
    const h = el?.offsetHeight ?? 200;
    const below = window.innerHeight - r.bottom - 4;
    const top = below >= Math.min(h, 240) || r.top < below ? r.bottom + 1 : Math.max(4, r.top - h - 1);
    const width = Math.max(r.width, props.minWidth ?? 0);
    const left = Math.max(4, Math.min(r.left, window.innerWidth - (el?.offsetWidth ?? width) - 4));
    setStyle({ top, left, minWidth: width, maxHeight: Math.max(120, top > r.top ? below : r.top - 8) });
  }, [anchor, props.minWidth]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current?.contains(e.target as Node) || anchor.contains(e.target as Node)) return;
      onClose();
    };
    const onScroll = (e: Event) => {
      if (ref.current?.contains(e.target as Node)) return;
      onClose();
    };
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("mousedown", onDown, true);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onClose);
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("mousedown", onDown, true);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onClose);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [anchor, onClose]);

  useEffect(() => {
    ref.current?.querySelector<HTMLElement>(".combo-item.current")?.scrollIntoView({ block: "nearest" });
  }, []);

  return (
    <div className="combo-popup" ref={ref} style={style} role="listbox" data-testid={props.testId}>
      {props.options.map((o, i) =>
        o.separator ? (
          <div key={`sep${i}`} className="combo-sep" />
        ) : o.header ? (
          <div key={`h${i}`} className="combo-header">
            {o.label}
          </div>
        ) : (
          <div
            key={o.key}
            role="option"
            aria-selected={o.key === props.current}
            aria-disabled={o.disabled || undefined}
            className={`combo-item${o.key === props.current ? " current" : ""}${o.key === active ? " active" : ""}${o.disabled ? " disabled" : ""}${o.bold ? " bold" : ""}`}
            title={o.title}
            onMouseEnter={() => setActive(o.key)}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => {
              if (o.disabled) return;
              props.onPick(o.key);
            }}
          >
            <span>{o.label}</span>
            {o.identifier && o.identifier !== o.label && <span className="combo-ident">({o.identifier})</span>}
          </div>
        ),
      )}
    </div>
  );
}

export function Combo(props: {
  value: string | null;
  options: ComboOption[];
  onChange: (key: string) => void;
  disabled?: boolean;
  className?: string;
  title?: string;
  testId?: string;
  /** Text shown when no entry is selected. */
  placeholder?: string;
  /** Closed-box width in characters (Qt `CompactComboBox`); longer labels are elided. */
  chars?: number;
  style?: CSSProperties;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const current = props.options.find((o) => o.key === props.value && !o.header && !o.separator);
  const pick = (key: string) => {
    setOpen(false);
    if (key !== props.value) props.onChange(key);
  };
  const onKey = (e: KeyboardEvent) => {
    if (props.disabled) return;
    if (e.key === "Enter" || e.key === " " || (e.altKey && e.key === "ArrowDown")) {
      e.preventDefault();
      setOpen(true);
      return;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const list = props.options.filter(selectable);
      const i = list.findIndex((o) => o.key === props.value);
      const next = list[Math.max(0, Math.min(list.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)))];
      if (next) pick(next.key);
    }
  };
  const style: CSSProperties = { ...props.style, ...(props.chars ? { width: `calc(${props.chars}ch + 26px)` } : {}) };
  return (
    <>
      <button
        ref={ref}
        type="button"
        className={`combo ${props.className ?? ""}`}
        disabled={props.disabled}
        title={props.title ?? current?.title}
        data-testid={props.testId}
        data-value={props.value ?? ""}
        aria-haspopup="listbox"
        aria-expanded={open}
        style={style}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={onKey}
      >
        <span className={`combo-text${current?.bold ? " bold" : ""}`}>{current?.label ?? props.placeholder ?? ""}</span>
        <span className="combo-arrow" aria-hidden>
          ▾
        </span>
      </button>
      {open && ref.current && (
        <PopupList
          anchor={ref.current}
          options={props.options}
          current={props.value}
          onPick={pick}
          onClose={() => setOpen(false)}
          testId={props.testId ? `${props.testId}-popup` : undefined}
        />
      )}
    </>
  );
}

/** Editable combo: free text plus a list of suggestions (Qt `setEditable(True)`). */
export function EditableCombo(props: {
  text: string;
  options: ComboOption[];
  onText: (text: string) => void;
  onCommit?: (text: string) => void;
  disabled?: boolean;
  className?: string;
  title?: string;
  testId?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const [open, setOpen] = useState(false);
  return (
    <span ref={ref} className={`combo editable ${props.className ?? ""}${props.disabled ? " disabled" : ""}`} title={props.title}>
      <input
        type="text"
        value={props.text}
        disabled={props.disabled}
        data-testid={props.testId}
        onChange={(e) => props.onText(e.currentTarget.value)}
        onBlur={(e) => props.onCommit?.(e.currentTarget.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") props.onCommit?.(e.currentTarget.value);
          if (e.altKey && e.key === "ArrowDown") setOpen(true);
        }}
      />
      <button type="button" className="combo-drop" tabIndex={-1} disabled={props.disabled} onClick={() => setOpen((o) => !o)} aria-label="Show list">
        ▾
      </button>
      {open && ref.current && (
        <PopupList
          anchor={ref.current}
          options={props.options}
          current={props.options.find((o) => o.label === props.text)?.key ?? null}
          onPick={(key) => {
            setOpen(false);
            const o = props.options.find((x) => x.key === key);
            if (o) {
              props.onText(o.label);
              props.onCommit?.(o.label);
            }
          }}
          onClose={() => setOpen(false)}
        />
      )}
    </span>
  );
}

/** Tool button with an instant popup menu (Qt `QToolButton.InstantPopup`). */
export function MenuButton(props: {
  label: ReactNode;
  items: ComboOption[];
  onPick: (key: string) => void;
  disabled?: boolean;
  title?: string;
  testId?: string;
  className?: string;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        ref={ref}
        type="button"
        className={`btn menu-btn ${props.className ?? ""}`}
        disabled={props.disabled}
        title={props.title}
        data-testid={props.testId}
        onClick={() => setOpen((o) => !o)}
      >
        {props.label}
        <span className="menu-arrow" aria-hidden>
          ▾
        </span>
      </button>
      {open && ref.current && (
        <PopupList
          anchor={ref.current}
          options={props.items}
          current={null}
          minWidth={180}
          onPick={(key) => {
            setOpen(false);
            props.onPick(key);
          }}
          onClose={() => setOpen(false)}
          testId={props.testId ? `${props.testId}-popup` : undefined}
        />
      )}
    </>
  );
}

/** Combo options of protocol `Choice`s, keyed by position (values may be bool / number / null). */
export function choiceOptions(
  choices: ReadonlyArray<{ label: string; identifier?: string; enabled?: boolean; reason?: string; description?: string; group?: string }>,
): ComboOption[] {
  const out: ComboOption[] = [];
  let group: string | undefined;
  choices.forEach((c, i) => {
    if (c.group && c.group !== group) {
      group = c.group;
      out.push({ key: `__h${i}`, label: c.group, header: true });
    }
    const disabled = c.enabled === false;
    const title = [c.description, disabled ? `Not available: ${c.reason ?? "unavailable"}` : undefined].filter(Boolean).join("\n") || undefined;
    out.push({ key: String(i), label: c.label, identifier: c.identifier, disabled, title });
  });
  return out;
}

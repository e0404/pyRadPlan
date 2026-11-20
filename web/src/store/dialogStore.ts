/**
 * Promise-based modal dialogs: `showDialog(render)` pushes a dialog onto a stack rendered by
 * `components/io/DialogHost` and resolves with the value its content passes to `close`.
 */
import type { ReactNode } from "react";
import { create } from "zustand";

export interface DialogContext<T> {
  /** Close the dialog and resolve the promise with `value`. */
  close: (value: T) => void;
  /** Whether this is the topmost dialog (stacked dialogs must not all react to Escape). */
  isTop: () => boolean;
}

interface DialogEntry {
  id: number;
  render: () => ReactNode;
}

export const useDialogs = create<{ stack: DialogEntry[] }>(() => ({ stack: [] }));

let nextId = 1;

export function showDialog<T>(render: (ctx: DialogContext<T>) => ReactNode): Promise<T> {
  return new Promise<T>((resolve) => {
    const id = nextId++;
    const close = (value: T) => {
      useDialogs.setState((s) => ({ stack: s.stack.filter((d) => d.id !== id) }));
      resolve(value);
    };
    const isTop = () => useDialogs.getState().stack.at(-1)?.id === id;
    useDialogs.setState((s) => ({ stack: [...s.stack, { id, render: () => render({ close, isTop }) }] }));
  });
}

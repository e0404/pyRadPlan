import { useDialogs } from "../../store/dialogStore";

/** Renders the stack of `showDialog` dialogs (mounted once, by the menu bar). */
export function DialogHost() {
  const stack = useDialogs((s) => s.stack);
  return (
    <>
      {stack.map((entry) => (
        <div key={entry.id} className="dialog-layer">
          {entry.render()}
        </div>
      ))}
    </>
  );
}

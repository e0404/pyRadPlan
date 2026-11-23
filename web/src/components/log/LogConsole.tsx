import { useEffect, useRef } from "react";
import { useApp } from "../../store/appStore";
import { setLogLevel } from "../../store/ioActions";
import { LOG_LEVELS, useIo, type LogThreshold } from "../../store/ioStore";
import { formatLogLine, visibleLogs } from "../io/models";
import "../io/io.css";

/** Mirror of `gui/widgets/_log_console.py`: level combo, Clear, coloured read-only view. */
export function LogConsole() {
  const logs = useApp((s) => s.logs);
  const level = useIo((s) => s.logLevel);
  const ref = useRef<HTMLDivElement>(null);
  const shown = visibleLogs(logs, level);
  const stick = useRef(true);

  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [shown.length]);

  return (
    <div className="log-console">
      <div className="log-toolbar">
        <label htmlFor="log-level">Level:</label>
        <select
          id="log-level"
          value={level}
          title="Minimum level shown; also sets the pyRadPlan logger verbosity"
          data-testid="log-level"
          onChange={(e) => void setLogLevel(e.currentTarget.value as LogThreshold)}
        >
          {LOG_LEVELS.map((l) => (
            <option key={l.id} value={l.id}>
              {l.label}
            </option>
          ))}
        </select>
        <span className="spacer" />
        <button className="btn" onClick={() => useApp.setState({ logs: [] })} data-testid="log-clear">
          Clear
        </button>
      </div>
      <div
        className="log"
        ref={ref}
        data-testid="log"
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
        }}
      >
        {shown.map((l) => (
          <div key={l.id} className={l.level}>
            {formatLogLine(l)}
          </div>
        ))}
      </div>
    </div>
  );
}

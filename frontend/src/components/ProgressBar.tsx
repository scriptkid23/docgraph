interface ProgressBarProps {
  percent: number;
  inverted?: boolean;
}

export function ProgressBar({ percent, inverted }: ProgressBarProps) {
  const pct = Math.min(100, Math.max(0, Math.round(percent)));
  return (
    <div
      className={`progress-track${inverted ? " progress-track--inverted" : ""}`}
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="progress-bar" style={{ width: `${pct}%` }} />
    </div>
  );
}

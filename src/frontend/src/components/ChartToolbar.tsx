interface ChartToolbarProps {
  title: string;
  markersVisible: boolean;
  equityVisible: boolean;
  isReplaying?: boolean;
  replayProgress?: number;
  replaySpeed?: number;
  onFit: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onGoToLatest: () => void;
  onToggleMarkers: () => void;
  onToggleEquity: () => void;
  onScreenshot: () => void;
  onReplayPlayPause?: () => void;
  onReplayReset?: () => void;
  onReplayStep?: () => void;
  onReplaySpeedChange?: (speed: number) => void;
}

export function ChartToolbar({
  title,
  markersVisible,
  equityVisible,
  isReplaying = false,
  replayProgress = 100,
  replaySpeed = 6,
  onFit,
  onZoomIn,
  onZoomOut,
  onGoToLatest,
  onToggleMarkers,
  onToggleEquity,
  onScreenshot,
  onReplayPlayPause,
  onReplayReset,
  onReplayStep,
  onReplaySpeedChange,
}: ChartToolbarProps) {
  return (
    <div className="chart-toolbar">
      <div>
        <div className="section-kicker">TradingView-style Replay</div>
        <h2>{title}</h2>
      </div>

      <div className="chart-toolbar-actions">
        <button type="button" onClick={onReplayPlayPause} className="toolbar-button replay-button">
          {isReplaying ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={onReplayStep} className="toolbar-button">
          Step
        </button>
        <button type="button" onClick={onReplayReset} className="toolbar-button">
          Reset
        </button>

        <label className="speed-control">
          <span>{replaySpeed}x</span>
          <input
            type="range"
            min="1"
            max="20"
            value={replaySpeed}
            onChange={(event) => onReplaySpeedChange?.(Number(event.target.value))}
          />
        </label>

        <span className="progress-pill">{Math.round(replayProgress)}%</span>

        <button type="button" onClick={onFit} className="toolbar-button">
          Fit
        </button>
        <button type="button" onClick={onZoomIn} className="toolbar-button">
          +
        </button>
        <button type="button" onClick={onZoomOut} className="toolbar-button">
          -
        </button>
        <button type="button" onClick={onGoToLatest} className="toolbar-button">
          Latest
        </button>
        <button
          type="button"
          onClick={onToggleMarkers}
          className={markersVisible ? "toolbar-button active" : "toolbar-button"}
        >
          Trades
        </button>
        <button
          type="button"
          onClick={onToggleEquity}
          className={equityVisible ? "toolbar-button active" : "toolbar-button"}
        >
          Equity
        </button>
        <button type="button" onClick={onScreenshot} className="toolbar-button">
          PNG
        </button>
      </div>
    </div>
  );
}

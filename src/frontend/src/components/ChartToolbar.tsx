export type MarkerMode = 'auto' | 'all' | 'off';

interface ChartToolbarProps {
  title: string;
  subtitle?: string;
  markerMode: MarkerMode;
  equityVisible: boolean;
  drawdownVisible?: boolean;
  isReplaying?: boolean;
  replayProgress?: number;
  replaySpeed?: number;
  onFit: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onGoToLatest: () => void;
  onMarkerModeChange: (mode: MarkerMode) => void;
  onToggleEquity: () => void;
  onToggleDrawdown?: () => void;
  onScreenshot: () => void;
  onReplayPlayPause?: () => void;
  onReplayReset?: () => void;
  onReplayStep?: () => void;
  onReplaySpeedChange?: (speed: number) => void;
}

export function ChartToolbar({
  title,
  subtitle,
  markerMode,
  equityVisible,
  drawdownVisible = false,
  isReplaying = false,
  replayProgress = 100,
  replaySpeed = 6,
  onFit,
  onZoomIn,
  onZoomOut,
  onGoToLatest,
  onMarkerModeChange,
  onToggleEquity,
  onToggleDrawdown,
  onScreenshot,
  onReplayPlayPause,
  onReplayReset,
  onReplayStep,
  onReplaySpeedChange,
}: ChartToolbarProps) {
  return (
    <div className="chart-toolbar">
      <div>
        <span className="kicker">TradingView-style workspace</span>
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </div>

      <div className="toolbar-actions" aria-label="Chart controls">
        {onReplayPlayPause && (
          <button type="button" onClick={onReplayPlayPause} className="toolbar-primary">
            {isReplaying ? 'Pause' : 'Play'}
          </button>
        )}
        {onReplayStep && <button type="button" onClick={onReplayStep}>Step</button>}
        {onReplayReset && <button type="button" onClick={onReplayReset}>Reset</button>}
        {onReplaySpeedChange && (
          <label className="speed-control">
            <span>{replaySpeed}x</span>
            <input
              aria-label="Replay speed"
              type="range"
              min="1"
              max="50"
              value={replaySpeed}
              onChange={(event) => onReplaySpeedChange(Number(event.target.value))}
            />
          </label>
        )}
        {onReplayPlayPause && <span className="progress-pill">{Math.round(replayProgress)}%</span>}

        <button type="button" onClick={onFit}>Fit</button>
        <button type="button" onClick={onZoomIn}>+</button>
        <button type="button" onClick={onZoomOut}>−</button>
        <button type="button" onClick={onGoToLatest}>Latest</button>

        <select value={markerMode} onChange={(event) => onMarkerModeChange(event.target.value as MarkerMode)} aria-label="Marker mode">
          <option value="auto">Trades Auto</option>
          <option value="all">Trades All</option>
          <option value="off">Trades Off</option>
        </select>

        <button type="button" className={equityVisible ? 'active' : ''} onClick={onToggleEquity}>Equity</button>
        {onToggleDrawdown && <button type="button" className={drawdownVisible ? 'active' : ''} onClick={onToggleDrawdown}>Drawdown</button>}
        <button type="button" onClick={onScreenshot}>PNG</button>
      </div>
    </div>
  );
}
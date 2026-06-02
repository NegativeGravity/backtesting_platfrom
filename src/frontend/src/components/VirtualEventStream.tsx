import { useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import type { LiveEvent } from '../types';
import { formatDateTime } from '../utils/formatters';

interface VirtualEventStreamProps {
  events: LiveEvent[];
}

export function VirtualEventStream({ events }: VirtualEventStreamProps) {
  const parentRef = useRef<HTMLDivElement | null>(null);
  const virtualizer = useVirtualizer({
    count: events.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 58,
    overscan: 16,
  });

  if (events.length === 0) return <div className="empty-state">No live events yet.</div>;

  return (
    <div ref={parentRef} className="event-stream">
      <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
        {virtualizer.getVirtualItems().map((item) => {
          const event = events[item.index];
          return (
            <article key={`${event.event_id ?? item.index}-${event.timestamp ?? item.index}`} className="event-row" style={{ transform: `translateY(${item.start}px)` }}>
              <span><b>{event.event_type}</b><small>{event.source}</small></span>
              <small>{formatDateTime(event.timestamp)}</small>
              <code>{JSON.stringify(event.payload).slice(0, 140)}</code>
            </article>
          );
        })}
      </div>
    </div>
  );
}

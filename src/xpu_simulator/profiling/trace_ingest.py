"""Profiler trace ingestion helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TraceEvent:
    name: str
    duration_us: float
    category: str


def load_trace_events(path: str | Path) -> list[TraceEvent]:
    raw = json.loads(Path(path).read_text())
    events = raw.get("traceEvents", raw if isinstance(raw, list) else [])
    parsed: list[TraceEvent] = []
    for event in events:
        if event.get("ph") != "X":
            continue
        parsed.append(
            TraceEvent(
                name=event.get("name", "unknown"),
                duration_us=float(event.get("dur", 0.0)),
                category=str(event.get("cat", "uncategorized")),
            )
        )
    return parsed


def summarize_trace_events(events: list[TraceEvent]) -> dict[str, object]:
    by_name: dict[str, list[TraceEvent]] = defaultdict(list)
    for event in events:
        by_name[event.name].append(event)

    total = sum(event.duration_us for event in events)
    summary = {
        "event_count": len(events),
        "total_duration_us": total,
        "ops": {},
    }
    for name, rows in by_name.items():
        duration = sum(row.duration_us for row in rows)
        summary["ops"][name] = {
            "samples": len(rows),
            "total_duration_us": duration,
            "mean_duration_us": duration / len(rows),
        }
    return summary

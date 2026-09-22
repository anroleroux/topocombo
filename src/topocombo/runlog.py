"""Structured run logging.

The optimization loop is terminal-driven and writes result artifacts to disk;
nothing in the loop knows about presentation.  This module is the same idea for
the pipeline itself: every stage appends structured records here, which are
printed to the terminal as they happen and written to ``run.json`` /
``pipeline.log`` at the end.  The HTML report is a separate, downstream consumer
of those files (see :mod:`topocombo.report`).
"""

from __future__ import annotations

import json
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Step:
    name: str
    title: str
    started_at: str
    lines: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float | None = None
    status: str = "running"


class RunLog:
    """Collects timed, structured records for one pipeline run."""

    def __init__(self, name: str, out_dir: Path, echo: bool = True) -> None:
        self.name = name
        self.out_dir = Path(out_dir)
        self.echo = echo
        self.started_at = _utcnow()
        self.steps: list[Step] = []
        self.params: dict[str, Any] = {}
        self.environment: dict[str, Any] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        self._t0 = time.perf_counter()
        self._current: Step | None = None

    # -- recording ---------------------------------------------------------
    @contextmanager
    def step(self, name: str, title: str) -> Iterator[Step]:
        step = Step(name=name, title=title, started_at=_utcnow())
        self.steps.append(step)
        self._current = step
        t0 = time.perf_counter()
        self._emit(f"\n=== {title} ===")
        try:
            yield step
        except Exception as exc:  # pragma: no cover - failure path
            step.status = "failed"
            step.duration_s = time.perf_counter() - t0
            self.log(f"FAILED: {exc!r}")
            raise
        else:
            step.status = "ok"
            step.duration_s = time.perf_counter() - t0
            self._emit(f"--- {title}: ok ({step.duration_s:.2f} s)")
        finally:
            self._current = None

    def log(self, message: str) -> None:
        if self._current is not None:
            self._current.lines.append(message)
        self._emit(message)

    def record(self, **values: Any) -> None:
        """Attach machine-readable values to the current step."""
        if self._current is None:
            raise RuntimeError("record() called outside of a step")
        self._current.data.update(values)

    def artifact(self, path: Path, description: str) -> Path:
        path = Path(path)
        size = path.stat().st_size if path.exists() else 0
        try:  # keep logs readable: paths relative to where the run was started
            shown = path.relative_to(Path.cwd())
        except ValueError:
            shown = path
        entry = {"path": str(path), "shown": str(shown), "description": description, "bytes": size}
        if self._current is not None:
            self._current.artifacts.append(entry)
        self.log(f"wrote {shown} ({size / 1024:.1f} kB) — {description}")
        return path

    def tool(self, name: str, version: str) -> None:
        self.environment.setdefault("tools", {})[name] = version

    def _emit(self, message: str) -> None:
        if self.echo:
            print(message, flush=True)

    # -- output ------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "finished_at": _utcnow(),
            "duration_s": round(time.perf_counter() - self._t0, 3),
            "environment": self.environment,
            "params": self.params,
            "steps": [
                {
                    "name": s.name,
                    "title": s.title,
                    "started_at": s.started_at,
                    "duration_s": None if s.duration_s is None else round(s.duration_s, 3),
                    "status": s.status,
                    "lines": s.lines,
                    "data": s.data,
                    "artifacts": s.artifacts,
                }
                for s in self.steps
            ],
        }

    def write(self) -> tuple[Path, Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.out_dir / "run.json"
        json_path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        text_path = self.out_dir / "pipeline.log"
        text_path.write_text(self.as_text())
        return json_path, text_path

    def as_text(self) -> str:
        out: list[str] = [f"# {self.name} — started {self.started_at}"]
        for s in self.steps:
            out.append("")
            out.append(f"=== {s.title} ===")
            out.extend(s.lines)
            dur = "?" if s.duration_s is None else f"{s.duration_s:.2f} s"
            out.append(f"--- {s.title}: {s.status} ({dur})")
        out.append("")
        return "\n".join(out)

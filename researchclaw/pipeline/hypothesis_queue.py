"""Durable work queue for per-hypothesis validation attempts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from researchclaw.pipeline.hypothesis_store import (
    TREE_DIRNAME,
    _file_lock,
    _utcnow_iso,
)


@dataclass(frozen=True)
class WorkItem:
    node_id: str
    attempt_id: str
    branch_run_dir: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", str(self.node_id or "").strip())
        object.__setattr__(self, "attempt_id", str(self.attempt_id or "").strip())
        object.__setattr__(
            self,
            "branch_run_dir",
            str(self.branch_run_dir or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "attempt_id": self.attempt_id,
            "branch_run_dir": self.branch_run_dir,
        }

    @classmethod
    def from_dict(cls, data: Any) -> WorkItem:
        data = data if isinstance(data, dict) else {}
        return cls(
            node_id=data.get("node_id") or "",
            attempt_id=data.get("attempt_id") or "",
            branch_run_dir=data.get("branch_run_dir") or "",
        )


class DurableWorkQueue:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.tree_dir = self.run_dir / TREE_DIRNAME
        self.queue_path = self.tree_dir / "work_queue.jsonl"
        self.lock_path = self.tree_dir / ".lock"

    def append(self, item: WorkItem, *, created_at: str | None = None) -> None:
        payload = {
            "event_type": "work_item_queued",
            "item": item.to_dict(),
            "timestamp": created_at or _utcnow_iso(),
        }
        with _file_lock(self.lock_path):
            self.tree_dir.mkdir(parents=True, exist_ok=True)
            with self.queue_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                fh.write("\n")

    def read_items(self) -> list[WorkItem]:
        if not self.queue_path.exists():
            return []
        items: list[WorkItem] = []
        seen_attempts: set[str] = set()
        for line in self.queue_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            item = WorkItem.from_dict(payload.get("item"))
            if not item.attempt_id or item.attempt_id in seen_attempts:
                continue
            seen_attempts.add(item.attempt_id)
            items.append(item)
        return items

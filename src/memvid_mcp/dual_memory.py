"""Dual memory manager for project and user memory scopes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .memory import MemvidMemory
from .scope_classifier import ScopeClassifier


def detect_project_root(start_path: str | None = None) -> Path | None:
    """
    Detect project root by looking for common project markers.

    Args:
        start_path: Starting directory (default: current working directory)

    Returns:
        Project root path or None if not found
    """
    current = Path(start_path or os.getcwd()).resolve()
    markers = [
        ".memvid_project",  # Custom marker (highest priority)
        ".git",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "CMakeLists.txt",
    ]

    max_depth = 10
    for _ in range(max_depth):
        for marker in markers:
            marker_path = current / marker
            if marker_path.exists():
                return current
        if current == current.parent:
            break
        current = current.parent

    return None


def _resolve_data_dir(value: str, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


class DualMemoryManager:
    """
    Manages dual memory system with project and user scopes.

    Project memory: stored in <project_root>/.memvid_data/
    User memory: stored in ~/memvid_data/
    """

    CONFIDENCE_THRESHOLD = 0.65
    PROJECT_BOOST = 1.2
    DEDUP_THRESHOLD = 0.9

    def __init__(self, project_root: str | None = None):
        """
        Initialize dual memory manager.

        Args:
            project_root: Explicit project root (auto-detected if None)
        """
        # Detect project root
        start_path = project_root or os.environ.get("MEMVID_PROJECT_ROOT")
        self.project_root = detect_project_root(start_path)

        # Initialize project memory
        project_base_dir = self.project_root or Path.cwd()
        project_data_dir_override = os.environ.get("MEMVID_PROJECT_DATA_DIR") or os.environ.get("MEMVID_DATA_DIR")
        if project_data_dir_override:
            project_data_dir = _resolve_data_dir(project_data_dir_override, base_dir=project_base_dir)
        else:
            project_data_dir = project_base_dir / ".memvid_data"

        self.project_memory = MemvidMemory(data_dir=str(project_data_dir))

        # Initialize user memory
        user_base_dir = Path.home()
        user_data_dir_override = os.environ.get("MEMVID_USER_DATA_DIR")
        if user_data_dir_override:
            user_data_dir = _resolve_data_dir(user_data_dir_override, base_dir=user_base_dir)
        else:
            user_data_dir = user_base_dir / "memvid_data"
        self.user_memory = MemvidMemory(data_dir=str(user_data_dir))

        # Initialize scope classifier
        self._classifier = ScopeClassifier()

    def store(
        self,
        content: str,
        scope: str = "auto",
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        sector: str | None = None,
    ) -> dict[str, Any]:
        """
        Store memory with automatic or manual scope classification.

        Args:
            content: Memory content
            scope: "auto", "project", or "user"
            user_id: User identifier
            tags: Optional tags
            metadata: Optional metadata
            sector: Override cognitive sector classification

        Returns:
            Dict with id, scope, classification info
        """
        # Determine target scope
        if scope == "auto":
            classification = self._classifier.classify(content, metadata)
            if classification["confidence"] < self.CONFIDENCE_THRESHOLD:
                target_scope = "user"
                classification["warning"] = "Low confidence, defaulting to user memory"
            else:
                target_scope = classification["scope"]
        else:
            target_scope = scope
            classification = {"scope": scope, "confidence": 1.0, "reasoning": "explicit"}

        # Add scope info to metadata
        meta = (metadata or {}).copy()
        meta["scope"] = target_scope
        meta["classification"] = classification

        # Store to appropriate memory
        if target_scope == "project":
            result = self.project_memory.add(content, user_id=user_id, tags=tags, metadata=meta, sector=sector)
        else:
            result = self.user_memory.add(content, user_id=user_id, tags=tags, metadata=meta, sector=sector)

        result["scope"] = target_scope
        result["classification"] = classification
        return result

    def recall(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
        tags: list[str] | None = None,
        sector: str | None = None,
        expand_waypoints: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Search both memories and merge results.

        Args:
            query: Search query
            user_id: User identifier
            limit: Maximum results
            tags: Filter by tags
            sector: Filter by sector
            expand_waypoints: Enable waypoint expansion

        Returns:
            Merged and deduplicated results
        """
        fetch_limit = int(limit * 1.5)

        # Search both memories
        project_results = self.project_memory.search(
            query=query, user_id=user_id, limit=fetch_limit, tags=tags, sector=sector, expand_waypoints=expand_waypoints
        )
        user_results = self.user_memory.search(
            query=query, user_id=user_id, limit=fetch_limit, tags=tags, sector=sector, expand_waypoints=expand_waypoints
        )

        # Add scope and boost project results
        for r in project_results:
            r["scope"] = "project"
            r["original_score"] = r.get("score", 0)
            r["score"] = r.get("score", 0) * self.PROJECT_BOOST

        for r in user_results:
            r["scope"] = "user"

        # Merge and deduplicate
        all_results = project_results + user_results
        if len(all_results) <= 1:
            return all_results[:limit]

        deduplicated = self._deduplicate(all_results)
        deduplicated.sort(key=lambda x: x.get("score", 0), reverse=True)
        return deduplicated[:limit]

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        """Remove duplicate results based on content similarity."""
        if len(results) <= 1:
            return results

        contents = [r["content"] for r in results]
        embeddings = self._classifier.encode(contents)

        kept = []
        kept_embs = []

        for i, (result, emb) in enumerate(zip(results, embeddings)):
            is_dup = False
            for kept_emb in kept_embs:
                if np.dot(emb, kept_emb) >= self.DEDUP_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(result)
                kept_embs.append(emb)

        return kept

    def get(self, memory_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        """Get memory by ID from either store."""
        result = self.project_memory.get(memory_id, user_id)
        if result:
            result["scope"] = "project"
            return result

        result = self.user_memory.get(memory_id, user_id)
        if result:
            result["scope"] = "user"
            return result

        return None

    def delete(self, memory_id: str, user_id: str | None = None) -> bool:
        """Delete memory from either store."""
        if self.project_memory.delete(memory_id, user_id):
            return True
        return self.user_memory.delete(memory_id, user_id)

    def delete_all(self, user_id: str | None = None) -> int:
        """Delete all memories for a user from both stores."""
        count = self.project_memory.delete_all(user_id)
        count += self.user_memory.delete_all(user_id)
        return count

    def list_memories(
        self,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        tags: list[str] | None = None,
        sector: str | None = None,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """List memories from one or both stores."""
        results = []

        if scope in (None, "project"):
            for m in self.project_memory.list_memories(user_id, limit, offset, tags, sector):
                m["scope"] = "project"
                results.append(m)

        if scope in (None, "user"):
            for m in self.user_memory.list_memories(user_id, limit, offset, tags, sector):
                m["scope"] = "user"
                results.append(m)

        return results[:limit]

    def stats(self) -> dict[str, Any]:
        """Get statistics for both memory stores."""
        return {
            "project": {
                **self.project_memory.stats(),
                "scope": "project",
            },
            "user": {
                **self.user_memory.stats(),
                "scope": "user",
            },
        }

    # Delegate temporal and decay operations to both stores
    def store_fact(self, subject: str, predicate: str, obj: str, scope: str = "project", **kwargs) -> str:
        """Store fact in specified scope."""
        if scope == "project":
            return self.project_memory.store_fact(subject, predicate, obj, **kwargs)
        return self.user_memory.store_fact(subject, predicate, obj, **kwargs)

    def query_facts(self, subject: str = None, predicate: str = None, obj: str = None, at: str = None) -> list[dict]:
        """Query facts from both stores."""
        project_facts = self.project_memory.query_facts(subject, predicate, obj, at)
        user_facts = self.user_memory.query_facts(subject, predicate, obj, at)
        for f in project_facts:
            f["scope"] = "project"
        for f in user_facts:
            f["scope"] = "user"
        return project_facts + user_facts

    def get_timeline(self, subject: str, predicate: str = None) -> list[dict]:
        """Get timeline from both stores."""
        project_timeline = self.project_memory.get_timeline(subject, predicate)
        user_timeline = self.user_memory.get_timeline(subject, predicate)
        for t in project_timeline:
            t["scope"] = "project"
        for t in user_timeline:
            t["scope"] = "user"
        return project_timeline + user_timeline

    def reinforce_memory(self, memory_id: str, boost: float = 0.15) -> float | None:
        """Reinforce memory in either store."""
        result = self.project_memory.reinforce_memory(memory_id, boost)
        if result is not None:
            return result
        return self.user_memory.reinforce_memory(memory_id, boost)

    def apply_decay(self) -> dict[str, int]:
        """Apply decay to both stores."""
        return {
            "project": self.project_memory.apply_decay(),
            "user": self.user_memory.apply_decay(),
        }

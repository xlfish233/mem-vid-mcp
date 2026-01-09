"""
Core memory operations using memvid for storage.

Provides OpenMemory-style cognitive memory with:
- Multi-sector classification (episodic/semantic/procedural/emotional/reflective)
- Temporal knowledge graph for time-versioned facts
- Waypoint association graph for memory linking
- Salience-based decay and reinforcement
"""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from memvid import MemvidEncoder, MemvidRetriever

from .classifier import classify_content, get_sector_penalty, SECTORS
from .decay import apply_decay_to_memories, reinforce
from .temporal import TemporalGraph
from .waypoint import WaypointGraph


class MemvidMemory:
    """
    Memory storage using memvid video encoding with cognitive features.

    Inspired by OpenMemory's architecture, this provides:
    - add: Store memories with automatic sector classification
    - search: Semantic search with sector penalties and waypoint expansion
    - get/delete: CRUD operations
    - store_fact/query_facts: Temporal knowledge graph
    - reinforce: Manual salience boost
    - apply_decay: Time-based salience decay
    """

    def __init__(self, data_dir: str = None, user_id: str = None):
        """
        Initialize memory storage.

        Args:
            data_dir: Directory for data files (default: ./memvid_data or MEMVID_DATA_DIR env)
            user_id: Default user ID for isolation
        """
        self.data_dir = Path(data_dir or os.environ.get("MEMVID_DATA_DIR", "./memvid_data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.default_user = user_id or "default"

        # File paths
        self._video_path = self.data_dir / "memory.mp4"
        self._index_path = self.data_dir / "memory_index.json"
        self._meta_path = self.data_dir / "memory_meta.json"

        # Memvid components
        self._encoder: MemvidEncoder | None = None
        self._retriever: MemvidRetriever | None = None

        # Memory storage
        self._memories: dict[str, dict] = {}
        self._dirty = False

        # Advanced features
        self._temporal = TemporalGraph(self.data_dir / "temporal_facts.json")
        self._waypoints = WaypointGraph(self.data_dir / "waypoints.json")

        self._load_metadata()

    def _load_metadata(self):
        """Load memory metadata from disk."""
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                self._memories = json.load(f)

    def _save_metadata(self):
        """Save memory metadata to disk."""
        with open(self._meta_path, "w") as f:
            json.dump(self._memories, f, indent=2, default=str)

    def _get_retriever(self) -> MemvidRetriever | None:
        """Get retriever if video exists."""
        if self._retriever is None and self._video_path.exists():
            self._retriever = MemvidRetriever(str(self._video_path), str(self._index_path))
        return self._retriever

    def _rebuild_video(self):
        """Rebuild video from all memories."""
        if not self._memories:
            return

        encoder = MemvidEncoder()
        chunks = []
        for mem_id, mem in self._memories.items():
            # Include sector and tags in chunk for better retrieval
            chunk = f"[ID:{mem_id}][SEC:{mem.get('primary_sector', 'semantic')}] {mem['content']}"
            if mem.get("tags"):
                chunk += f" [tags:{','.join(mem['tags'])}]"
            chunks.append(chunk)

        encoder.add_chunks(chunks)
        encoder.build_video(str(self._video_path), str(self._index_path))
        self._retriever = None  # Reset retriever
        self._dirty = False

    def _create_waypoints_for_memory(self, mem_id: str, content: str):
        """Create waypoints to similar existing memories."""
        retriever = self._get_retriever()
        if retriever is None:
            return

        # Find similar memories (memvid returns list of texts)
        results = retriever.search(content, top_k=5)
        for i, text in enumerate(results):
            if not text.startswith("[ID:"):
                continue
            other_id = text[4:text.index("]")]
            if other_id == mem_id:
                continue
            if other_id not in self._memories:
                continue

            # Use position-based score (earlier = more similar)
            score = 1.0 - (i * 0.1)
            # Create waypoint if similarity above threshold
            if score >= WaypointGraph.SIMILARITY_THRESHOLD:
                self._waypoints.create_waypoint(mem_id, other_id, weight=float(score))

    # ==================== Core Memory Operations ====================

    def add(
        self,
        content: str,
        user_id: str = None,
        tags: list[str] = None,
        metadata: dict = None,
        sector: str = None,
    ) -> dict[str, Any]:
        """
        Add a memory with automatic sector classification.

        Args:
            content: Text content to store
            user_id: User identifier for isolation
            tags: Optional tags for categorization
            metadata: Optional additional metadata
            sector: Override automatic sector classification

        Returns:
            Dict with memory id, sector, and creation time
        """
        mem_id = str(uuid.uuid4())
        now = datetime.now()
        now_iso = now.isoformat()
        now_ts = int(now.timestamp() * 1000)
        uid = user_id or self.default_user

        # Classify content into sector
        meta_with_sector = metadata.copy() if metadata else {}
        if sector:
            meta_with_sector["sector"] = sector
        classification = classify_content(content, meta_with_sector)

        memory = {
            "id": mem_id,
            "content": content,
            "user_id": uid,
            "tags": tags or [],
            "metadata": metadata or {},
            "created_at": now_iso,
            "updated_at": now_iso,
            # Sector classification
            "primary_sector": classification["primary"],
            "additional_sectors": classification["additional"],
            "sector_confidence": classification["confidence"],
            # Decay tracking
            "salience": 1.0,  # Start at max salience
            "decay_lambda": classification["decay_lambda"],
            "last_seen_at": now_ts,
            "coactivations": 0,
        }

        self._memories[mem_id] = memory
        self._dirty = True
        self._save_metadata()

        # Rebuild video with new content
        self._rebuild_video()

        # Create waypoints to similar memories
        self._create_waypoints_for_memory(mem_id, content)

        return {
            "id": mem_id,
            "primary_sector": classification["primary"],
            "confidence": classification["confidence"],
            "created_at": now_iso,
        }

    def search(
        self,
        query: str,
        user_id: str = None,
        limit: int = 10,
        tags: list[str] = None,
        sector: str = None,
        expand_waypoints: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Search memories with sector penalties and waypoint expansion.

        Args:
            query: Search query text
            user_id: Filter by user
            limit: Maximum results to return
            tags: Filter by tags
            sector: Filter by sector
            expand_waypoints: Whether to expand results via waypoint graph

        Returns:
            List of matching memories with scores
        """
        retriever = self._get_retriever()
        if retriever is None:
            return []

        uid = user_id or self.default_user

        # Classify query to determine sector penalties
        query_classification = classify_content(query)
        query_sector = query_classification["primary"]

        # Get more results for filtering and expansion
        results = retriever.search(query, top_k=limit * 3)

        matches = []
        matched_ids = set()

        for i, text in enumerate(results):
            if not text.startswith("[ID:"):
                continue

            mem_id = text[4:text.index("]")]
            if mem_id not in self._memories:
                continue

            mem = self._memories[mem_id]

            # Filter by user
            if mem["user_id"] != uid:
                continue

            # Filter by sector
            if sector and mem["primary_sector"] != sector:
                continue

            # Filter by tags
            if tags and not any(t in mem.get("tags", []) for t in tags):
                continue

            # Apply cross-sector penalty
            mem_sector = mem.get("primary_sector", "semantic")
            sector_penalty = get_sector_penalty(query_sector, mem_sector)

            # Apply salience boost
            salience = mem.get("salience", 0.5)

            # Use position-based score (earlier = more similar)
            base_score = 1.0 - (i * 0.05)
            # Final score = base_score * sector_penalty * salience
            final_score = base_score * sector_penalty * (0.5 + salience * 0.5)

            matches.append({
                **mem,
                "score": final_score,
                "base_score": base_score,
                "sector_penalty": sector_penalty,
            })
            matched_ids.add(mem_id)

        # Expand via waypoints if enabled and we have matches
        if expand_waypoints and matches:
            seed_ids = [m["id"] for m in matches[:5]]  # Top 5 as seeds
            expanded = self._waypoints.expand(seed_ids, max_expansion=limit)

            for exp in expanded:
                if exp["id"] in matched_ids:
                    continue
                if exp["id"] not in self._memories:
                    continue

                mem = self._memories[exp["id"]]
                if mem["user_id"] != uid:
                    continue
                if sector and mem["primary_sector"] != sector:
                    continue

                # Expanded results get lower base score
                matches.append({
                    **mem,
                    "score": exp["weight"] * 0.5,  # Waypoint-based score
                    "expanded_via": exp["path"],
                })
                matched_ids.add(exp["id"])

        # Sort by score and limit
        matches.sort(key=lambda m: m["score"], reverse=True)
        top_results = matches[:limit]

        # Reinforce retrieved memories
        for result in top_results:
            mem = self._memories.get(result["id"])
            if mem:
                reinforce(mem)

                # Reinforce waypoint paths
                if "expanded_via" in result:
                    self._waypoints.reinforce(result["expanded_via"])

        self._save_metadata()

        return top_results

    def get(self, memory_id: str, user_id: str = None) -> dict[str, Any] | None:
        """Get a specific memory by ID."""
        mem = self._memories.get(memory_id)
        if mem is None:
            return None

        uid = user_id or self.default_user
        if mem["user_id"] != uid:
            return None

        return mem

    def delete(self, memory_id: str, user_id: str = None) -> bool:
        """Delete a memory and its waypoints."""
        mem = self._memories.get(memory_id)
        if mem is None:
            return False

        uid = user_id or self.default_user
        if mem["user_id"] != uid:
            return False

        del self._memories[memory_id]
        self._waypoints.remove_memory(memory_id)
        self._dirty = True
        self._save_metadata()
        self._rebuild_video()

        return True

    def delete_all(self, user_id: str = None) -> int:
        """Delete all memories for a user."""
        uid = user_id or self.default_user
        to_delete = [mid for mid, m in self._memories.items() if m["user_id"] == uid]

        for mid in to_delete:
            del self._memories[mid]
            self._waypoints.remove_memory(mid)

        if to_delete:
            self._dirty = True
            self._save_metadata()
            self._rebuild_video()

        return len(to_delete)

    def list_memories(
        self,
        user_id: str = None,
        limit: int = 50,
        offset: int = 0,
        tags: list[str] = None,
        sector: str = None,
    ) -> list[dict[str, Any]]:
        """List memories with optional filtering."""
        uid = user_id or self.default_user

        memories = []
        for mem in self._memories.values():
            if mem["user_id"] != uid:
                continue
            if tags and not any(t in mem.get("tags", []) for t in tags):
                continue
            if sector and mem.get("primary_sector") != sector:
                continue
            memories.append(mem)

        # Sort by salience * recency
        now_ts = int(datetime.now().timestamp() * 1000)
        memories.sort(
            key=lambda m: m.get("salience", 0.5) * (1 - (now_ts - m.get("last_seen_at", now_ts)) / 86400000 / 30),
            reverse=True,
        )

        return memories[offset:offset + limit]

    # ==================== Temporal Knowledge Graph ====================

    def store_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: str = None,
        confidence: float = 1.0,
        metadata: dict = None,
    ) -> str:
        """
        Store a temporal fact.

        Args:
            subject: Entity the fact is about
            predicate: Relationship type
            obj: Value/target
            valid_from: When fact became true (ISO string or None for now)
            confidence: Certainty level 0-1
            metadata: Additional data

        Returns:
            Fact ID
        """
        return self._temporal.insert_fact(subject, predicate, obj, valid_from, confidence, metadata)

    def query_facts(
        self,
        subject: str = None,
        predicate: str = None,
        obj: str = None,
        at: str = None,
    ) -> list[dict[str, Any]]:
        """
        Query facts valid at a point in time.

        Args:
            subject: Filter by subject
            predicate: Filter by predicate
            obj: Filter by object
            at: Point in time (ISO string or None for now)

        Returns:
            List of matching facts
        """
        return self._temporal.query_at_time(subject, predicate, obj, at)

    def get_timeline(self, subject: str, predicate: str = None) -> list[dict[str, Any]]:
        """Get chronological history of facts for a subject."""
        return self._temporal.get_timeline(subject, predicate)

    # ==================== Decay & Reinforcement ====================

    def reinforce_memory(self, memory_id: str, boost: float = 0.15) -> float | None:
        """
        Manually reinforce a memory's salience.

        Args:
            memory_id: Memory to reinforce
            boost: Salience boost amount

        Returns:
            New salience or None if not found
        """
        mem = self._memories.get(memory_id)
        if mem is None:
            return None

        new_salience = reinforce(mem, boost)
        self._save_metadata()
        return new_salience

    def apply_decay(self) -> int:
        """
        Apply time-based decay to all memories.

        Returns:
            Number of memories updated
        """
        # Get sector-specific decay rates
        sector_lambdas = {name: cfg["decay_lambda"] for name, cfg in SECTORS.items()}

        updated = apply_decay_to_memories(self._memories, sector_lambdas)

        if updated:
            self._save_metadata()

        # Also decay temporal facts
        self._temporal.apply_confidence_decay()

        # Prune weak waypoints
        self._waypoints.prune_weak_edges()

        return len(updated)

    # ==================== Statistics ====================

    def stats(self) -> dict[str, Any]:
        """Get comprehensive statistics."""
        total = len(self._memories)
        by_user = {}
        by_sector = {}

        for mem in self._memories.values():
            uid = mem["user_id"]
            by_user[uid] = by_user.get(uid, 0) + 1

            sector = mem.get("primary_sector", "semantic")
            by_sector[sector] = by_sector.get(sector, 0) + 1

        video_size = self._video_path.stat().st_size if self._video_path.exists() else 0

        return {
            "total_memories": total,
            "by_user": by_user,
            "by_sector": by_sector,
            "video_size_bytes": video_size,
            "data_dir": str(self.data_dir),
            "temporal": self._temporal.stats(),
            "waypoints": self._waypoints.stats(),
        }

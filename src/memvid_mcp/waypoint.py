"""
Waypoint association graph for memory linking.

Creates and manages associative links between memories based on
semantic similarity. Enables graph-based search expansion and
reinforcement learning through usage patterns.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class WaypointGraph:
    """
    Associative graph linking related memories.

    Waypoints are directed edges between memories with weights
    representing association strength. Weights increase with
    co-activation (reinforcement learning).
    """

    # Minimum similarity to create a waypoint
    SIMILARITY_THRESHOLD = 0.75
    # Initial weight for new waypoints
    INITIAL_WEIGHT = 0.5
    # Weight boost per reinforcement
    REINFORCE_BOOST = 0.05
    # Maximum weight
    MAX_WEIGHT = 1.0
    # Minimum weight before pruning
    MIN_WEIGHT = 0.05

    def __init__(self, data_path: Path):
        """
        Initialize waypoint graph.

        Args:
            data_path: Path to JSON file for persistence
        """
        self._path = data_path
        # Structure: {src_id: {dst_id: {"weight": float, "created_at": int, "updated_at": int}}}
        self._edges: dict[str, dict[str, dict]] = {}
        self._load()

    def _load(self):
        """Load graph from disk."""
        if self._path.exists() and self._path.stat().st_size > 0:
            with open(self._path) as f:
                self._edges = json.load(f)

    def _save(self):
        """Save graph to disk."""
        with open(self._path, "w") as f:
            json.dump(self._edges, f, indent=2)

    def _now_ts(self) -> int:
        """Current timestamp in milliseconds."""
        return int(datetime.now().timestamp() * 1000)

    def create_waypoint(
        self,
        src_id: str,
        dst_id: str,
        weight: float = None,
        bidirectional: bool = True,
    ):
        """
        Create a waypoint (edge) between two memories.

        Args:
            src_id: Source memory ID
            dst_id: Destination memory ID
            weight: Edge weight (default: INITIAL_WEIGHT)
            bidirectional: Create reverse edge too
        """
        if src_id == dst_id:
            return

        weight = weight if weight is not None else self.INITIAL_WEIGHT
        now = self._now_ts()

        # Create forward edge
        if src_id not in self._edges:
            self._edges[src_id] = {}

        self._edges[src_id][dst_id] = {
            "weight": weight,
            "created_at": now,
            "updated_at": now,
        }

        # Create reverse edge
        if bidirectional:
            if dst_id not in self._edges:
                self._edges[dst_id] = {}

            self._edges[dst_id][src_id] = {
                "weight": weight,
                "created_at": now,
                "updated_at": now,
            }

        self._save()

    def get_neighbors(self, mem_id: str) -> list[dict[str, Any]]:
        """
        Get all memories linked to a given memory.

        Args:
            mem_id: Memory ID to get neighbors for

        Returns:
            List of {id, weight} sorted by weight descending
        """
        if mem_id not in self._edges:
            return []

        neighbors = []
        for dst_id, edge in self._edges[mem_id].items():
            neighbors.append({
                "id": dst_id,
                "weight": edge["weight"],
            })

        neighbors.sort(key=lambda x: x["weight"], reverse=True)
        return neighbors

    def expand(
        self,
        seed_ids: list[str],
        max_expansion: int = 10,
        min_weight: float = 0.1,
    ) -> list[dict[str, Any]]:
        """
        Expand search results via graph traversal.

        Performs BFS from seed nodes, accumulating weight along paths.
        Used to find related memories not directly matched by search.

        Args:
            seed_ids: Starting memory IDs
            max_expansion: Maximum nodes to expand
            min_weight: Minimum accumulated weight to include

        Returns:
            List of {id, weight, path} for expanded nodes
        """
        expanded = []
        visited = set(seed_ids)

        # Queue: [(id, accumulated_weight, path)]
        queue = [{"id": sid, "weight": 1.0, "path": [sid]} for sid in seed_ids]
        count = 0

        while queue and count < max_expansion:
            current = queue.pop(0)
            neighbors = self.get_neighbors(current["id"])

            for neighbor in neighbors:
                dst_id = neighbor["id"]
                if dst_id in visited:
                    continue

                # Weight decays along path: parent_weight * edge_weight * 0.8
                new_weight = current["weight"] * neighbor["weight"] * 0.8

                if new_weight < min_weight:
                    continue  # Prune low-weight paths

                item = {
                    "id": dst_id,
                    "weight": new_weight,
                    "path": current["path"] + [dst_id],
                }

                expanded.append(item)
                visited.add(dst_id)
                queue.append(item)
                count += 1

                if count >= max_expansion:
                    break

        # Sort by weight descending
        expanded.sort(key=lambda x: x["weight"], reverse=True)
        return expanded

    def reinforce(self, path: list[str]):
        """
        Reinforce waypoints along a traversal path.

        Called when a path is successfully used in retrieval,
        strengthening the associations.

        Args:
            path: List of memory IDs in traversal order
        """
        if len(path) < 2:
            return

        now = self._now_ts()

        for i in range(len(path) - 1):
            src_id = path[i]
            dst_id = path[i + 1]

            if src_id in self._edges and dst_id in self._edges[src_id]:
                edge = self._edges[src_id][dst_id]
                new_weight = min(self.MAX_WEIGHT, edge["weight"] + self.REINFORCE_BOOST)
                edge["weight"] = new_weight
                edge["updated_at"] = now

        self._save()

    def remove_memory(self, mem_id: str):
        """
        Remove all waypoints involving a memory.

        Args:
            mem_id: Memory ID to remove
        """
        # Remove outgoing edges
        if mem_id in self._edges:
            del self._edges[mem_id]

        # Remove incoming edges
        for src_id in list(self._edges.keys()):
            if mem_id in self._edges[src_id]:
                del self._edges[src_id][mem_id]

        self._save()

    def prune_weak_edges(self, min_weight: float = None) -> int:
        """
        Remove edges below minimum weight threshold.

        Args:
            min_weight: Threshold (default: MIN_WEIGHT)

        Returns:
            Number of edges pruned
        """
        min_weight = min_weight if min_weight is not None else self.MIN_WEIGHT
        pruned = 0

        for src_id in list(self._edges.keys()):
            for dst_id in list(self._edges[src_id].keys()):
                if self._edges[src_id][dst_id]["weight"] < min_weight:
                    del self._edges[src_id][dst_id]
                    pruned += 1

            # Clean up empty source nodes
            if not self._edges[src_id]:
                del self._edges[src_id]

        if pruned > 0:
            self._save()

        return pruned

    def stats(self) -> dict[str, Any]:
        """Get statistics about the waypoint graph."""
        total_edges = sum(len(dsts) for dsts in self._edges.values())

        # Count all unique nodes (both sources and destinations)
        all_nodes = set(self._edges.keys())
        for dsts in self._edges.values():
            all_nodes.update(dsts.keys())
        total_nodes = len(all_nodes)

        weights = []
        for dsts in self._edges.values():
            for edge in dsts.values():
                weights.append(edge["weight"])

        avg_weight = sum(weights) / len(weights) if weights else 0

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "average_weight": avg_weight,
        }

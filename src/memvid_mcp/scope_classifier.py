"""Semantic scope classifier for dual-memory system."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# Few-shot examples for classification
PROJECT_EXAMPLES = [
    "This codebase uses FastAPI for REST APIs",
    "Bug in auth.py line 42 causing null pointer",
    "The MemvidMemory class handles all storage operations",
    "Dependencies are managed via pyproject.toml",
    "We use Redis for caching in this project",
    "The server.py module implements MCP protocol",
    "Tests are located in the tests/ directory",
    "The project requires Python 3.10 or higher",
    "Memory leak in the video encoding module",
    "The API endpoint /users returns 500 error",
]

USER_EXAMPLES = [
    "I prefer using pytest over unittest for testing",
    "I like clean code with type hints",
    "I always write docstrings for public functions",
    "I prefer functional programming patterns",
    "I use Ruff for linting Python code",
    "I like VS Code as my primary editor",
    "I commit frequently with small atomic changes",
    "I prefer reviewing PRs in the morning",
    "Python uses duck typing for polymorphism",
    "REST APIs should be stateless",
]


class ScopeClassifier:
    """Classify memory content into project or user scope using semantic similarity."""

    CONFIDENCE_THRESHOLD = 0.65

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._project_embeddings = self.model.encode(PROJECT_EXAMPLES, normalize_embeddings=True)
        self._user_embeddings = self.model.encode(USER_EXAMPLES, normalize_embeddings=True)

    def classify(self, content: str, metadata: dict | None = None) -> dict:
        """
        Classify content into project or user scope.

        Args:
            content: The memory content to classify
            metadata: Optional metadata (can contain explicit "scope" override)

        Returns:
            dict with keys: scope, confidence, reasoning
        """
        # Allow explicit scope override
        if metadata and metadata.get("scope") in ("project", "user"):
            return {
                "scope": metadata["scope"],
                "confidence": 1.0,
                "reasoning": "explicit_override",
            }

        # Compute content embedding
        content_emb = self.model.encode([content], normalize_embeddings=True)[0]

        # Calculate similarity scores
        project_scores = np.dot(self._project_embeddings, content_emb)
        user_scores = np.dot(self._user_embeddings, content_emb)

        project_score = float(np.mean(project_scores))
        user_score = float(np.mean(user_scores))

        # Determine scope
        total = project_score + user_score
        if total == 0:
            scope = "user"
            confidence = 0.5
        elif project_score > user_score:
            scope = "project"
            confidence = project_score / total
        else:
            scope = "user"
            confidence = user_score / total

        return {
            "scope": scope,
            "confidence": round(confidence, 3),
            "reasoning": f"project_score={project_score:.3f}, user_score={user_score:.3f}",
        }

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts for external use (e.g., deduplication)."""
        return self.model.encode(texts, normalize_embeddings=True)

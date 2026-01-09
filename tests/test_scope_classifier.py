"""Tests for scope classifier."""

import pytest
from memvid_mcp.scope_classifier import ScopeClassifier


@pytest.fixture(scope="module")
def classifier():
    """Create classifier once for all tests (model loading is slow)."""
    return ScopeClassifier()


class TestScopeClassifier:
    """Test scope classification functionality."""

    def test_classify_project_memory(self, classifier):
        """Project-related content should be classified as project."""
        project_texts = [
            "This codebase uses FastAPI for REST APIs",
            "Bug in auth.py line 42 causing null pointer",
            "The MemvidMemory class handles all storage operations",
            "Dependencies are managed via pyproject.toml",
        ]
        for text in project_texts:
            result = classifier.classify(text)
            assert result["scope"] == "project", f"Failed for: {text}"
            assert result["confidence"] > 0.5

    def test_classify_user_memory(self, classifier):
        """User preference content should be classified as user."""
        user_texts = [
            "I prefer using pytest over unittest for testing",
            "I like clean code with type hints",
            "I always write docstrings for public functions",
            "I prefer functional programming patterns",
        ]
        for text in user_texts:
            result = classifier.classify(text)
            assert result["scope"] == "user", f"Failed for: {text}"
            assert result["confidence"] > 0.5

    def test_explicit_scope_override(self, classifier):
        """Explicit scope in metadata should override classification."""
        result = classifier.classify("some random content", {"scope": "project"})
        assert result["scope"] == "project"
        assert result["confidence"] == 1.0
        assert result["reasoning"] == "explicit_override"

        result = classifier.classify("some random content", {"scope": "user"})
        assert result["scope"] == "user"
        assert result["confidence"] == 1.0

    def test_confidence_range(self, classifier):
        """Confidence should be between 0 and 1."""
        result = classifier.classify("This is a test memory")
        assert 0 <= result["confidence"] <= 1

    def test_reasoning_format(self, classifier):
        """Reasoning should contain score information."""
        result = classifier.classify("This is a test memory")
        assert "project_score" in result["reasoning"]
        assert "user_score" in result["reasoning"]

    def test_encode_method(self, classifier):
        """Encode method should return normalized embeddings."""
        texts = ["hello world", "test content"]
        embeddings = classifier.encode(texts)
        assert embeddings.shape[0] == 2
        # Check normalization (L2 norm should be ~1)
        import numpy as np
        norms = np.linalg.norm(embeddings, axis=1)
        assert all(abs(n - 1.0) < 0.01 for n in norms)

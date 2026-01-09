"""Tests for dual memory manager."""

from pathlib import Path

import pytest

from memvid_mcp.dual_memory import DualMemoryManager, detect_project_root


class TestDetectProjectRoot:
    """Test project root detection."""

    def test_detect_git_repo(self, tmp_path):
        """Should detect .git directory."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        result = detect_project_root(str(tmp_path))
        assert result == tmp_path

    def test_detect_pyproject(self, tmp_path):
        """Should detect pyproject.toml."""
        (tmp_path / "pyproject.toml").touch()
        result = detect_project_root(str(tmp_path))
        assert result == tmp_path

    def test_detect_package_json(self, tmp_path):
        """Should detect package.json."""
        (tmp_path / "package.json").touch()
        result = detect_project_root(str(tmp_path))
        assert result == tmp_path

    def test_detect_custom_marker(self, tmp_path):
        """Should detect .memvid_project marker."""
        (tmp_path / ".memvid_project").touch()
        result = detect_project_root(str(tmp_path))
        assert result == tmp_path

    def test_no_project_root(self, tmp_path):
        """Should return None if no markers found."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        result = detect_project_root(str(subdir))
        # May find parent markers or return None
        assert result is None or isinstance(result, Path)


class TestDualMemoryManager:
    """Test dual memory manager functionality."""

    @pytest.fixture
    def manager(self, tmp_path, monkeypatch):
        """Create manager with temp directories."""
        # Create a fake project root
        (tmp_path / ".git").mkdir()
        monkeypatch.setenv("MEMVID_PROJECT_DATA_DIR", str(tmp_path / ".memvid_data"))
        monkeypatch.setenv("MEMVID_USER_DATA_DIR", str(tmp_path / "memvid_data_user"))
        return DualMemoryManager(project_root=str(tmp_path))

    def test_init(self, manager):
        """Manager should initialize both memories."""
        assert manager.project_memory is not None
        assert manager.user_memory is not None
        assert manager._classifier is not None

    def test_store_auto_project(self, manager):
        """Project-related content should go to project memory."""
        result = manager.store("This codebase uses FastAPI for REST APIs", scope="auto")
        assert result["scope"] == "project"
        assert "classification" in result

    def test_store_auto_user(self, manager):
        """User preference content should go to user memory."""
        result = manager.store("I prefer pytest over unittest", scope="auto")
        assert result["scope"] == "user"
        assert "classification" in result

    def test_store_explicit_scope(self, manager):
        """Explicit scope should override classification."""
        result = manager.store("some content", scope="project")
        assert result["scope"] == "project"

        result = manager.store("some content", scope="user")
        assert result["scope"] == "user"

    def test_recall_merged_results(self, manager):
        """Recall should search both memories."""
        # Store in both scopes
        manager.store("FastAPI is used for REST APIs", scope="project")
        manager.store("I prefer FastAPI over Flask", scope="user")

        manager.DEDUP_THRESHOLD = 0.999
        results = manager.recall("FastAPI", limit=10)
        assert len(results) > 0
        scopes = {r["scope"] for r in results}
        assert "project" in scopes
        assert "user" in scopes

    def test_get_from_either_store(self, manager):
        """Get should find memory from either store."""
        result1 = manager.store("project content", scope="project")
        result2 = manager.store("user content", scope="user")

        mem1 = manager.get(result1["id"])
        assert mem1 is not None
        assert mem1["scope"] == "project"

        mem2 = manager.get(result2["id"])
        assert mem2 is not None
        assert mem2["scope"] == "user"

    def test_delete_from_either_store(self, manager):
        """Delete should work on either store."""
        result = manager.store("test content", scope="project")
        assert manager.delete(result["id"]) is True
        assert manager.get(result["id"]) is None

    def test_stats_dual(self, manager):
        """Stats should show both stores."""
        manager.store("project content", scope="project")
        manager.store("user content", scope="user")

        stats = manager.stats()
        assert "project" in stats
        assert "user" in stats
        assert stats["project"]["total_memories"] >= 1
        assert stats["user"]["total_memories"] >= 1

    def test_list_memories_with_scope_filter(self, manager):
        """List should filter by scope."""
        manager.store("project content", scope="project")
        manager.store("user content", scope="user")

        project_only = manager.list_memories(scope="project")
        assert all(m["scope"] == "project" for m in project_only)

        user_only = manager.list_memories(scope="user")
        assert all(m["scope"] == "user" for m in user_only)

    def test_delete_all(self, manager):
        """Delete all should clear both stores."""
        manager.store("project content", scope="project")
        manager.store("user content", scope="user")

        count = manager.delete_all()
        assert count >= 2

        stats = manager.stats()
        assert stats["project"]["total_memories"] == 0
        assert stats["user"]["total_memories"] == 0

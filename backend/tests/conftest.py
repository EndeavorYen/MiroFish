import pytest
import tempfile
import os


@pytest.fixture
def tmp_db_path(tmp_path):
    """Provide a temporary directory for SQLite databases."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def graph_id():
    """Standard test graph ID."""
    return "test-graph-001"

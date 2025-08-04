"""Pytest configuration and shared fixtures.

This module provides:
- Common fixtures for all test modules
- Configuration for pytest markers
- Import shortcuts for agent fixtures
- Shared test utilities
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio


# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import fixtures from other modules
from tests.utils.fixtures import *  # Import all fixtures from fixtures.py


# Configure pytest-asyncio
pytest_asyncio.fixture(scope="session")


def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Agent fixtures are imported from tests.utils.agent_fixtures
# Additional fixtures for specific test scenarios


@pytest.fixture
def agent_state_with_research_context():
    """Agent state with research context."""
    return {
        "task_id": 1,
        "task_data": {
            "id": 1,
            "title": "Implement authentication",
            "description": "Create secure login system",
            "component_area": "security",
        },
        "messages": [],
        "agent_outputs": {
            "research": {
                "output": {
                    "research_type": "web_search",
                    "key_findings": ["JWT tokens provide stateless authentication"],
                    "sources_found": [
                        {"title": "JWT Guide", "url": "https://example.com/jwt"}
                    ],
                }
            }
        },
        "coordination_context": {},
        "batch_id": None,
        "error_context": None,
        "next_agent": None,
    }


@pytest.fixture
def agent_state_with_all_outputs():
    """Agent state with outputs from all agents."""
    return {
        "task_id": 1,
        "task_data": {
            "id": 1,
            "title": "Complete system implementation",
            "description": "End-to-end system development",
            "component_area": "full_stack",
        },
        "messages": [],
        "agent_outputs": {
            "research": {
                "status": "completed",
                "output": {
                    "research_type": "comprehensive",
                    "key_findings": [
                        "Use modern tech stack",
                        "Follow security best practices",
                    ],
                    "sources_found": [
                        {"title": "Tech Guide", "url": "https://example.com/tech"}
                    ],
                },
            },
            "coding": {
                "status": "completed",
                "output": {
                    "implementation_type": "full_stack",
                    "files_created": ["backend.py", "frontend.js"],
                    "design_decisions": ["RESTful API", "JWT authentication"],
                },
            },
            "testing": {
                "status": "completed",
                "output": {
                    "test_type": "comprehensive",
                    "tests_generated": ["test_backend.py", "test_frontend.js"],
                    "coverage_report": {"overall": 95},
                },
            },
            "documentation": {
                "status": "completed",
                "output": {
                    "doc_type": "complete_suite",
                    "docs_created": ["api.md", "user_guide.md"],
                    "sections": ["API", "User Guide", "Setup"],
                },
            },
        },
        "coordination_context": {
            "workflow_complete": True,
            "all_agents_executed": True,
        },
        "batch_id": None,
        "error_context": None,
        "next_agent": None,
    }


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for testing."""
    orchestrator = MagicMock()
    orchestrator.execute_task = AsyncMock()
    orchestrator.execute_batch = AsyncMock()
    orchestrator.get_agent_health_status = MagicMock()
    orchestrator.cleanup = AsyncMock()
    return orchestrator


@pytest.fixture
def mock_agent_registry():
    """Mock agent registry for testing."""
    registry = MagicMock()
    registry.register = MagicMock()
    registry.unregister = MagicMock()
    registry.get_agent = MagicMock()
    registry.list_agents = MagicMock()
    registry.get_health_status = MagicMock()
    registry.cleanup_all = AsyncMock()
    return registry


@pytest.fixture
def mock_task_manager():
    """Mock task manager for testing."""
    task_manager = MagicMock()
    task_manager.create_task = MagicMock()
    task_manager.get_task_by_id = MagicMock()
    task_manager.update_task_status = MagicMock()
    task_manager.get_task_analytics = MagicMock()
    return task_manager


@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {
        "id": 1,
        "title": "Implement user authentication",
        "description": "Create secure user login and registration system",
        "component_area": "security",
        "phase": 1,
        "priority": "high",
        "complexity": "medium",
        "status": "not_started",
        "success_criteria": "Users can securely log in and register",
        "time_estimate_hours": 8.0,
    }


@pytest.fixture
def sample_batch_config():
    """Sample batch configuration for testing."""
    return {
        "batch_size": 5,
        "max_concurrent_tasks": 3,
        "timeout_minutes": 30,
        "retry_attempts": 2,
    }


# Additional pytest markers for different test types
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow


@pytest.fixture
def clean_environment():
    """Fixture to ensure clean test environment."""
    # Setup

    yield

    # Cleanup - restore any modified environment variables
    # This is a placeholder for environment cleanup if needed


@pytest.fixture(scope="session")
def test_database_path(tmp_path_factory):
    """Create a temporary database path for testing."""
    return tmp_path_factory.mktemp("test_db") / "test.db"


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = AsyncMock()
    client.achat = AsyncMock()
    client.chat = MagicMock()
    return client

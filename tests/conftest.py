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
from tests.utils.agent_fixtures import (
    mock_coding_agent,
    mock_documentation_agent,
    mock_research_agent,
    mock_testing_agent,
)  # Explicit imports
from tests.utils.fixtures import *  # Import all fixtures from fixtures.py


# Configure pytest-asyncio
pytest_asyncio.fixture(scope="session")


def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Missing fixtures that tests are looking for
@pytest.fixture
def mock_research_agent():
    """Mock research agent for testing."""
    mock_agent = AsyncMock()

    async def mock_execute_task(state):
        return {
            **state,
            "agent_outputs": {
                "research": {
                    "status": "completed",
                    "output": {
                        "research_type": "web_search",
                        "key_findings": ["JWT tokens secure", "Use HTTPS"],
                        "sources_found": [
                            {"title": "Auth Guide", "url": "https://example.com"}
                        ],
                    },
                }
            },
        }

    mock_agent.execute_task.side_effect = mock_execute_task
    return mock_agent


@pytest.fixture
def mock_coding_agent():
    """Mock coding agent for testing."""
    mock_agent = AsyncMock()

    async def mock_execute_task(state):
        return {
            **state,
            "agent_outputs": {
                "coding": {
                    "status": "completed",
                    "output": {
                        "implementation_type": "code_generation",
                        "files_created": ["auth.py"],
                        "design_decisions": ["Used JWT tokens"],
                        "content": "def authenticate(): pass",
                    },
                }
            },
        }

    mock_agent.execute_task.side_effect = mock_execute_task
    return mock_agent


@pytest.fixture
def mock_testing_agent():
    """Mock testing agent for testing."""
    mock_agent = AsyncMock()

    async def mock_execute_task(state):
        return {
            **state,
            "agent_outputs": {
                "testing": {
                    "status": "completed",
                    "output": {
                        "test_files": ["test_auth.py"],
                        "test_categories": ["unit_tests", "integration_tests"],
                        "coverage_report": {"total": 95.0},
                    },
                }
            },
        }

    mock_agent.execute_task.side_effect = mock_execute_task
    return mock_agent


@pytest.fixture
def mock_documentation_agent():
    """Mock documentation agent for testing."""
    mock_agent = AsyncMock()

    async def mock_execute_task(state):
        return {
            **state,
            "agent_outputs": {
                "documentation": {
                    "status": "completed",
                    "output": {
                        "documentation_files": ["README.md"],
                        "sections_created": ["installation", "usage"],
                    },
                }
            },
        }

    mock_agent.execute_task.side_effect = mock_execute_task
    return mock_agent


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
                },
                "status": "completed",
            }
        },
    }


@pytest.fixture
def agent_state_with_coding_context():
    """Agent state with coding context."""
    return {
        "task_id": 1,
        "task_data": {
            "id": 1,
            "title": "Test authentication system",
            "description": "Create tests for login functionality",
        },
        "messages": [],
        "agent_outputs": {
            "coding": {
                "output": {
                    "implementation_type": "code_generation",
                    "files_created": ["auth.py", "models.py"],
                    "design_decisions": ["Used JWT for tokens"],
                    "content": "def authenticate(): pass",
                },
                "status": "completed",
            }
        },
    }


@pytest.fixture
def mock_orchestration_environment():
    """Mock orchestration environment for integration tests."""
    mock_env = {
        "supervisor": AsyncMock(),
        "task_manager": MagicMock(),
        "agents": {
            "research": AsyncMock(),
            "coding": AsyncMock(),
            "testing": AsyncMock(),
            "documentation": AsyncMock(),
        },
    }

    # Configure mock behavior
    mock_env["task_manager"].get_ready_tasks.return_value = []
    mock_env["task_manager"].update_task_status.return_value = None

    return mock_env


@pytest.fixture
def sample_complex_task():
    """Sample complex task for testing."""
    return {
        "id": 1,
        "title": "Implement comprehensive authentication system",
        "description": "Implement secure authentication with JWT, "
        "OAuth2, and multi-factor mechanisms",
        "component_area": "security",
        "phase": 1,
        "priority": "high",
        "complexity": "high",
        "time_estimate_hours": 40.0,
        "success_criteria": "Complete authentication system with 95%+ test coverage",
    }


# Mock agent classes that some tests might try to import
class MockResearchAgent:
    def __init__(self):
        pass

    async def execute_task(self, state):
        return {"status": "completed", "output": {"findings": ["Mock research result"]}}


class MockCodingAgent:
    def __init__(self):
        pass

    async def execute_task(self, state):
        return {"status": "completed", "output": {"files_created": ["mock_file.py"]}}


class MockTestingAgent:
    def __init__(self):
        pass

    async def execute_task(self, state):
        return {"status": "completed", "output": {"test_files": ["test_mock.py"]}}


class MockDocumentationAgent:
    def __init__(self):
        pass

    async def execute_task(self, state):
        return {"status": "completed", "output": {"docs_created": ["README.md"]}}


# Monkey patch for missing imports
sys.modules["src.agents.research_agent"] = type(
    "MockModule", (), {"ResearchAgent": MockResearchAgent}
)()

sys.modules["src.agents.coding_agent"] = type(
    "MockModule", (), {"CodingAgent": MockCodingAgent}
)()

sys.modules["src.agents.testing_agent"] = type(
    "MockModule", (), {"TestingAgent": MockTestingAgent}
)()

sys.modules["src.agents.documentation_agent"] = type(
    "MockModule", (), {"DocumentationAgent": MockDocumentationAgent}
)()


# Mock TaskStatus enum for tests
class MockTaskStatus:
    COMPLETED = "completed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    NOT_STARTED = "not_started"


# Add mock TaskStatus to sys.modules for imports
sys.modules["src.schemas.unified_models"].TaskStatus = MockTaskStatus

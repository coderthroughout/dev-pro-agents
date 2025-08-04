"""Pytest configuration and fixtures for dev-pro-agents orchestration tests."""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.schemas.unified_models import (
    AgentReport,
    AgentType,
    ComponentArea,
    TaskComplexity,
    TaskCore,
    TaskDelegation,
    TaskPriority,
    TaskStatus,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_task_data():
    """Sample task data for orchestration testing."""
    return {
        "id": 1,
        "title": "Implement user authentication system",
        "description": "Create a secure authentication system with JWT tokens",
        "component_area": ComponentArea.SECURITY,
        "phase": 1,
        "priority": TaskPriority.HIGH,
        "complexity": TaskComplexity.MEDIUM,
        "success_criteria": "Users can login and logout securely",
        "time_estimate_hours": 8.0,
    }


@pytest.fixture
def sample_task_core(sample_task_data):
    """Sample TaskCore instance for testing."""
    return TaskCore.model_validate(sample_task_data)


@pytest.fixture
def sample_agent_state(sample_task_data):
    """Sample agent state for testing orchestration workflows."""
    return {
        "messages": [],
        "task_id": 1,
        "task_data": sample_task_data,
        "agent_outputs": {},
        "batch_id": None,
        "coordination_context": {},
        "error_context": None,
        "next_agent": None,
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = AsyncMock()

    # Mock delegation response
    mock_response = MagicMock()
    mock_response.content = """
    {
        "assigned_agent": "coding",
        "reasoning": "This task requires implementation work with authentication systems",
        "priority": "high",
        "estimated_duration": 60,
        "dependencies": [],
        "context_requirements": [],
        "confidence_score": 0.85
    }
    """

    mock_client.ainvoke.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_task_manager():
    """Mock task manager for orchestration testing."""
    mock_tm = MagicMock()
    mock_tm.get_tasks_by_status.return_value = []
    mock_tm.update_task_status.return_value = None

    # Mock connection and cursor for database operations
    mock_connection = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_connection.cursor.return_value = mock_cursor
    mock_connection.__enter__.return_value = mock_connection
    mock_connection.__exit__.return_value = None
    mock_tm._get_connection.return_value = mock_connection

    return mock_tm


@pytest.fixture
def sample_agent_report():
    """Sample agent report for testing."""
    return AgentReport(
        agent_name=AgentType.CODING,
        task_id=1,
        status=TaskStatus.COMPLETED,
        success=True,
        execution_time_minutes=15.5,
        outputs={"implementation": "Authentication system implemented"},
        artifacts=["auth.py", "models.py"],
        recommendations=["Add rate limiting", "Implement 2FA"],
        next_actions=["test", "deploy"],
        confidence_score=0.9,
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_task_delegation():
    """Sample task delegation for testing."""
    return TaskDelegation(
        assigned_agent=AgentType.CODING,
        reasoning="This task requires implementation of authentication systems with proper security measures",
        priority=TaskPriority.HIGH,
        estimated_duration=120,
        dependencies=[],
        context_requirements=["security best practices", "JWT implementation"],
        confidence_score=0.85,
    )


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_exa_client():
    """Mock Exa client for integration testing."""
    mock_client = AsyncMock()

    # Mock search result
    mock_search_result = MagicMock()
    mock_search_result.title = "Authentication Best Practices"
    mock_search_result.url = "https://example.com/auth-guide"
    mock_search_result.summary = "Comprehensive guide to authentication security"
    mock_search_result.score = 0.95
    mock_search_result.text = "JWT tokens provide secure authentication..."

    mock_search_response = MagicMock()
    mock_search_response.results = [mock_search_result]

    mock_client.search = AsyncMock(return_value=mock_search_response)
    return mock_client


@pytest.fixture
def mock_firecrawl_client():
    """Mock Firecrawl client for integration testing."""
    mock_client = AsyncMock()

    # Mock scrape response
    mock_document = MagicMock()
    mock_document.markdown = (
        "# Authentication Guide\nImplement secure authentication..."
    )
    mock_document.html = (
        "<h1>Authentication Guide</h1><p>Implement secure authentication...</p>"
    )

    mock_scrape_response = MagicMock()
    mock_scrape_response.success = True
    mock_scrape_response.data = mock_document

    mock_client.scrape = AsyncMock(return_value=mock_scrape_response)
    return mock_client


@pytest_asyncio.fixture
async def orchestration_test_environment(
    mock_openai_client, mock_task_manager, mock_exa_client, mock_firecrawl_client
):
    """Complete orchestration test environment with all mocked dependencies."""
    return {
        "openai_client": mock_openai_client,
        "task_manager": mock_task_manager,
        "exa_client": mock_exa_client,
        "firecrawl_client": mock_firecrawl_client,
    }


# Mark configuration for integration tests
@pytest.fixture
def integration_test_marker():
    """Marker for integration tests that require external services."""
    return pytest.mark.integration


# Configuration for asyncio tests
pytest_plugins = ["pytest_asyncio"]

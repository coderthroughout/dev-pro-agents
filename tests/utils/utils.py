"""Testing utilities and helper functions for dev-pro-agents.

Provides common testing utilities, assertion helpers, database management,
and test execution helpers for comprehensive testing workflows.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from src.schemas.database import Task, TaskExecutionLog
from src.schemas.unified_models import AgentReport, TaskCore, TaskStatus


# ============================================================================
# DATABASE TESTING UTILITIES
# ============================================================================


class DatabaseTestManager:
    """Manager for database testing operations with isolation and cleanup."""

    def __init__(self, db_url: str = "sqlite:///:memory:"):
        self.db_url = db_url
        self.engine = None
        self.session = None

    def setup_database(self) -> Session:
        """Set up test database with schema."""
        self.engine = create_engine(self.db_url, echo=False)
        SQLModel.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        return self.session

    def cleanup_database(self) -> None:
        """Clean up database and close connections."""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()

    @contextmanager
    def isolated_transaction(self) -> Generator[Session, None, None]:
        """Context manager for isolated database transactions."""
        if not self.session:
            self.setup_database()

        transaction = self.session.begin()
        try:
            yield self.session
            transaction.commit()
        except Exception:
            transaction.rollback()
            raise
        finally:
            transaction.close()


def create_test_database() -> DatabaseTestManager:
    """Create a test database manager for isolated testing."""
    return DatabaseTestManager()


def clear_database_tables(session: Session) -> None:
    """Clear all database tables in proper dependency order."""
    from sqlmodel import delete

    from src.schemas.database import (
        TaskComment,
        TaskDependency,
        TaskExecutionLog,
        TaskProgress,
    )

    # Clear in reverse dependency order
    session.exec(delete(TaskComment))
    session.exec(delete(TaskExecutionLog))
    session.exec(delete(TaskProgress))
    session.exec(delete(TaskDependency))
    session.exec(delete(Task))
    session.commit()


def count_database_records(session: Session) -> dict[str, int]:
    """Count records in all database tables."""
    from src.schemas.database import (
        TaskComment,
        TaskDependency,
        TaskExecutionLog,
        TaskProgress,
    )

    return {
        "tasks": len(session.exec(select(Task)).all()),
        "dependencies": len(session.exec(select(TaskDependency)).all()),
        "progress": len(session.exec(select(TaskProgress)).all()),
        "executions": len(session.exec(select(TaskExecutionLog)).all()),
        "comments": len(session.exec(select(TaskComment)).all()),
    }


# ============================================================================
# ASYNC TESTING UTILITIES
# ============================================================================


class AsyncTestHelper:
    """Helper for async testing operations."""

    @staticmethod
    async def wait_for_condition(
        condition_func, timeout: float = 5.0, interval: float = 0.1, *args, **kwargs
    ) -> bool:
        """Wait for a condition to become true with timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if await condition_func(*args, **kwargs):
                    return True
            except Exception:
                logging.debug("Condition check failed", exc_info=True)
            await asyncio.sleep(interval)
        return False

    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0):
        """Run a coroutine with timeout."""
        return await asyncio.wait_for(coro, timeout=timeout)

    @staticmethod
    @asynccontextmanager
    async def async_timer() -> AsyncGenerator[dict[str, float], None]:
        """Context manager to measure async execution time."""
        start_time = time.time()
        metrics = {"start_time": start_time}
        try:
            yield metrics
        finally:
            end_time = time.time()
            metrics.update({"end_time": end_time, "duration": end_time - start_time})


def create_async_mock(return_value=None, side_effect=None) -> AsyncMock:
    """Create a configured AsyncMock."""
    mock = AsyncMock()
    if return_value is not None:
        mock.return_value = return_value
    if side_effect is not None:
        mock.side_effect = side_effect
    return mock


# ============================================================================
# ASSERTION HELPERS
# ============================================================================


class AssertionHelpers:
    """Collection of custom assertion helpers for testing."""

    @staticmethod
    def assert_task_status_transition(
        initial_status: TaskStatus,
        final_status: TaskStatus,
        valid_transitions: dict[TaskStatus, list[TaskStatus]] | None = None,
    ) -> None:
        """Assert that a task status transition is valid."""
        if valid_transitions is None:
            from src.schemas.unified_models import can_transition_status

            assert can_transition_status(initial_status, final_status), (
                f"Invalid transition from {initial_status} to {final_status}"
            )
        else:
            assert final_status in valid_transitions.get(initial_status, []), (
                f"Invalid transition from {initial_status} to {final_status}"
            )

    @staticmethod
    def assert_task_core_valid(task: TaskCore) -> None:
        """Assert that a TaskCore instance is valid."""
        assert task.title, "Task must have a title"
        assert task.time_estimate_hours > 0, "Time estimate must be positive"
        assert 1 <= task.phase <= 10, "Phase must be between 1 and 10"

    @staticmethod
    def assert_agent_report_consistent(report: AgentReport) -> None:
        """Assert that an AgentReport is internally consistent."""
        if report.status == TaskStatus.FAILED:
            assert not report.success, "Failed status requires success=False"
            assert report.issues_found or report.error_details, (
                "Failed status requires issues or error details"
            )

        if report.status == TaskStatus.COMPLETED:
            assert report.success, "Completed status requires success=True"

        assert 0.0 <= report.confidence_score <= 1.0, (
            "Confidence score must be between 0 and 1"
        )

    @staticmethod
    def assert_json_serializable(obj: Any) -> None:
        """Assert that an object is JSON serializable."""
        try:
            json.dumps(obj, default=str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Object is not JSON serializable: {e}")

    @staticmethod
    def assert_execution_time_reasonable(
        start_time: datetime, end_time: datetime, max_duration_minutes: float = 120.0
    ) -> None:
        """Assert that execution time is reasonable."""
        duration = (end_time - start_time).total_seconds() / 60
        assert duration >= 0, "End time must be after start time"
        assert duration <= max_duration_minutes, (
            f"Execution time {duration:.2f}min exceeds maximum "
            f"{max_duration_minutes}min"
        )


# ============================================================================
# MOCK HELPERS AND FACTORIES
# ============================================================================


class MockFactory:
    """Factory for creating various mock objects."""

    @staticmethod
    def create_mock_response(
        data: Any, status_code: int = 200, headers: dict[str, str] | None = None
    ) -> MagicMock:
        """Create a mock HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = data
        mock_response.text = json.dumps(data) if isinstance(data, dict) else str(data)
        mock_response.headers = headers or {"content-type": "application/json"}
        return mock_response

    @staticmethod
    def create_mock_openai_completion(
        content: str, model: str = "gpt-4", usage_tokens: int = 100
    ) -> MagicMock:
        """Create a mock OpenAI completion response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = model
        mock_response.usage.total_tokens = usage_tokens
        return mock_response

    @staticmethod
    def create_mock_exa_search_result(
        title: str = "Test Result",
        url: str = "https://example.com",
        summary: str = "Test summary",
        score: float = 0.9,
    ) -> MagicMock:
        """Create a mock Exa search result."""
        mock_result = MagicMock()
        mock_result.title = title
        mock_result.url = url
        mock_result.summary = summary
        mock_result.score = score
        mock_result.text = f"{title}: {summary}"
        return mock_result

    @staticmethod
    def create_mock_firecrawl_document(
        markdown: str = "# Test Document", html: str = "<h1>Test Document</h1>"
    ) -> MagicMock:
        """Create a mock Firecrawl document."""
        mock_doc = MagicMock()
        mock_doc.markdown = markdown
        mock_doc.html = html
        return mock_doc


# ============================================================================
# TEST DATA VALIDATION
# ============================================================================


class TestDataValidator:
    """Validator for test data integrity."""

    @staticmethod
    def validate_task_relationships(tasks: list[TaskCore]) -> None:
        """Validate parent-child relationships in a list of tasks."""
        task_ids = {task.id for task in tasks if task.id}

        for task in tasks:
            if task.parent_task_id:
                assert task.parent_task_id in task_ids, (
                    f"Parent task {task.parent_task_id} not found for task {task.id}"
                )

    @staticmethod
    def validate_execution_log_consistency(
        logs: list[TaskExecutionLog], task_id: int
    ) -> None:
        """Validate consistency of execution logs for a task."""
        task_logs = [log for log in logs if log.task_id == task_id]

        # Sort by start time
        task_logs.sort(key=lambda x: x.start_time)

        for i, log in enumerate(task_logs):
            assert log.task_id == task_id, f"Log {i} has wrong task_id"

            if log.end_time:
                assert log.end_time >= log.start_time, (
                    f"Log {i} end_time before start_time"
                )

            # Check for overlapping executions (might be valid in some cases)
            if i > 0:
                prev_log = task_logs[i - 1]
                if prev_log.end_time and log.start_time < prev_log.end_time:
                    # This might be intentional for concurrent executions
                    pass


# ============================================================================
# FILE AND DIRECTORY UTILITIES
# ============================================================================


class FileTestHelper:
    """Helper for file system operations in tests."""

    @staticmethod
    @contextmanager
    def temporary_directory() -> Generator[Path, None, None]:
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @staticmethod
    @contextmanager
    def temporary_file(
        suffix: str = ".tmp", content: str | None = None
    ) -> Generator[Path, None, None]:
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=suffix, delete=False
        ) as temp_file:
            if content:
                temp_file.write(content)
                temp_file.flush()
            temp_path = Path(temp_file.name)

        try:
            yield temp_path
        finally:
            from contextlib import suppress

            with suppress(OSError):
                temp_path.unlink()

    @staticmethod
    def create_test_config(
        config_data: dict[str, Any], file_format: str = "yaml"
    ) -> str:
        """Create a test configuration file content."""
        if file_format.lower() == "yaml":
            import yaml

            return yaml.dump(config_data, default_flow_style=False)
        elif file_format.lower() == "json":
            return json.dumps(config_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# PERFORMANCE TESTING UTILITIES
# ============================================================================


class PerformanceTestHelper:
    """Helper for performance testing."""

    @staticmethod
    @contextmanager
    def measure_time() -> Generator[dict[str, float], None, None]:
        """Measure execution time."""
        start_time = time.perf_counter()
        metrics = {"start": start_time}

        try:
            yield metrics
        finally:
            end_time = time.perf_counter()
            metrics.update({"end": end_time, "duration": end_time - start_time})

    @staticmethod
    def assert_performance_bounds(
        duration: float,
        min_duration: float | None = None,
        max_duration: float | None = None,
    ) -> None:
        """Assert that execution duration is within bounds."""
        if min_duration is not None:
            assert duration >= min_duration, (
                f"Execution too fast: {duration:.3f}s < {min_duration:.3f}s"
            )

        if max_duration is not None:
            assert duration <= max_duration, (
                f"Execution too slow: {duration:.3f}s > {max_duration:.3f}s"
            )


# ============================================================================
# ENVIRONMENT AND CONFIGURATION HELPERS
# ============================================================================


class TestEnvironmentHelper:
    """Helper for managing test environment variables."""

    @staticmethod
    @contextmanager
    def environment_variables(**env_vars) -> Generator[None, None, None]:
        """Temporarily set environment variables."""
        original_values = {}

        # Set new values and store originals
        for key, value in env_vars.items():
            original_values[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)

        try:
            yield
        finally:
            # Restore original values
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    @staticmethod
    def skip_if_no_api_key(api_key_env: str) -> None:
        """Skip test if API key is not available."""
        if not os.getenv(api_key_env):
            pytest.skip(f"Skipping test: {api_key_env} not available")

    @staticmethod
    def skip_if_integration_disabled() -> None:
        """Skip test if integration tests are disabled."""
        if os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true":
            pytest.skip("Integration tests disabled")


# ============================================================================
# ERROR TESTING UTILITIES
# ============================================================================


class ErrorTestHelper:
    """Helper for testing error conditions."""

    @staticmethod
    def assert_raises_with_message(
        exception_class, expected_message: str, callable_obj, *args, **kwargs
    ) -> None:
        """Assert that an exception is raised with a specific message."""
        with pytest.raises(exception_class) as exc_info:
            callable_obj(*args, **kwargs)

        assert expected_message in str(exc_info.value), (
            f"Expected message '{expected_message}' not found in '{exc_info.value}'"
        )

    @staticmethod
    async def assert_async_raises_with_message(
        exception_class, expected_message: str, async_callable, *args, **kwargs
    ) -> None:
        """Assert that an async function raises an exception with a specific message."""
        with pytest.raises(exception_class) as exc_info:
            await async_callable(*args, **kwargs)

        assert expected_message in str(exc_info.value), (
            f"Expected message '{expected_message}' not found in '{exc_info.value}'"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Assertion helpers
    "AssertionHelpers",
    # Async utilities
    "AsyncTestHelper",
    # Database utilities
    "DatabaseTestManager",
    # Error testing
    "ErrorTestHelper",
    # File utilities
    "FileTestHelper",
    # Mock helpers
    "MockFactory",
    # Performance utilities
    "PerformanceTestHelper",
    # Validation
    "TestDataValidator",
    # Environment utilities
    "TestEnvironmentHelper",
    "clear_database_tables",
    "count_database_records",
    "create_async_mock",
    "create_test_database",
]

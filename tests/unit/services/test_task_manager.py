"""Comprehensive tests for TaskManager class.

This module provides comprehensive testing for task CRUD operations,
dependency resolution, validation, analytics, and edge cases.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.schemas.unified_models import (
    ComponentArea,
    DependencyType,
    TaskComplexity,
    TaskCore,
    TaskPriority,
    TaskStatus,
)
from src.task_manager import TaskDependency, TaskManager


class TestTaskManagerFixtures:
    """Test fixtures and database setup for TaskManager tests."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temp_file:
            db_path = temp_file.name

        # Create basic schema for testing
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tasks table
        cursor.execute("""
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                component_area TEXT NOT NULL,
                phase INTEGER DEFAULT 1,
                priority TEXT NOT NULL,
                complexity TEXT NOT NULL,
                status TEXT DEFAULT 'not_started',
                source_document TEXT DEFAULT '',
                success_criteria TEXT DEFAULT '',
                time_estimate_hours REAL DEFAULT 1.0,
                parent_task_id INTEGER,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (parent_task_id) REFERENCES tasks (id)
            )
        """)

        # Create task_dependencies table
        cursor.execute("""
            CREATE TABLE task_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                depends_on_task_id INTEGER NOT NULL,
                dependency_type TEXT DEFAULT 'blocks',
                created_at TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks (id),
                FOREIGN KEY (depends_on_task_id) REFERENCES tasks (id),
                UNIQUE(task_id, depends_on_task_id)
            )
        """)

        # Create task_progress table
        cursor.execute("""
            CREATE TABLE task_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                progress_percentage INTEGER DEFAULT 0,
                notes TEXT DEFAULT '',
                created_at TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """)

        # Create task_comments table
        cursor.execute("""
            CREATE TABLE task_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                comment TEXT NOT NULL,
                comment_type TEXT DEFAULT 'note',
                created_at TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks (id)
            )
        """)

        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def task_manager(self, temp_db_path):
        """Create TaskManager instance with temporary database."""
        return TaskManager(db_path=temp_db_path)

    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing."""
        return {
            "title": "Implement user authentication",
            "description": "Create secure authentication system with JWT tokens",
            "component_area": ComponentArea.SECURITY,
            "phase": 1,
            "priority": TaskPriority.HIGH,
            "complexity": TaskComplexity.MEDIUM,
            "source_document": "auth_requirements.md",
            "success_criteria": "Users can login and logout securely",
            "time_estimate_hours": 8.0,
        }

    @pytest.fixture
    def sample_task_core(self, sample_task_data):
        """Create TaskCore instance from sample data."""
        return TaskCore(**sample_task_data)


class TestTaskManagerInitialization(TestTaskManagerFixtures):
    """Test TaskManager initialization and database setup."""

    def test_init_with_custom_db_path(self, temp_db_path):
        """Test TaskManager initialization with custom database path."""
        tm = TaskManager(db_path=temp_db_path)
        assert tm.db_path == temp_db_path
        assert tm._task_cache == {}
        assert tm._cache_ttl == 300

    def test_init_without_db_path_raises_error(self):
        """Test TaskManager initialization without database raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TaskManager(db_path="/non/existent/path.db")

    def test_get_connection_returns_sqlite_connection(self, task_manager):
        """Test _get_connection returns proper SQLite connection."""
        conn = task_manager._get_connection()
        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_ensure_database_exists_with_missing_file(self):
        """Test _ensure_database_exists raises error for missing database."""
        tm = TaskManager.__new__(TaskManager)
        tm.db_path = "/non/existent/path.db"

        with pytest.raises(FileNotFoundError, match="Database not found"):
            tm._ensure_database_exists()


class TestTaskCRUDOperations(TestTaskManagerFixtures):
    """Test basic CRUD operations for tasks."""

    def test_create_task_with_dict_data(self, task_manager, sample_task_data):
        """Test creating task with dictionary data."""
        task = task_manager.create_task(sample_task_data)

        assert isinstance(task, TaskCore)
        assert task.id is not None
        assert task.title == sample_task_data["title"]
        assert task.description == sample_task_data["description"]
        assert task.status == TaskStatus.NOT_STARTED

        # Verify task was stored in database
        retrieved_task = task_manager.get_task_by_id(task.id)
        assert retrieved_task is not None
        assert retrieved_task.title == task.title

    def test_create_task_with_task_core_instance(self, task_manager, sample_task_core):
        """Test creating task with TaskCore instance."""
        task = task_manager.create_task(sample_task_core)

        assert isinstance(task, TaskCore)
        assert task.id is not None
        assert task.title == sample_task_core.title
        assert task.description == sample_task_core.description

    def test_create_task_with_invalid_data_raises_validation_error(self, task_manager):
        """Test creating task with invalid data raises ValidationError."""
        invalid_data = {
            "title": "",  # Empty title should fail validation
            "component_area": "invalid_area",
            "priority": "invalid_priority",
        }

        with pytest.raises(ValueError):
            task_manager.create_task(invalid_data)

    def test_get_task_by_id_existing_task(self, task_manager, sample_task_data):
        """Test retrieving existing task by ID."""
        created_task = task_manager.create_task(sample_task_data)
        retrieved_task = task_manager.get_task_by_id(created_task.id)

        assert retrieved_task is not None
        assert retrieved_task.id == created_task.id
        assert retrieved_task.title == created_task.title

    def test_get_task_by_id_non_existent_returns_none(self, task_manager):
        """Test retrieving non-existent task returns None."""
        task = task_manager.get_task_by_id(999999)
        assert task is None

    def test_get_task_by_id_uses_cache(self, task_manager, sample_task_data):
        """Test that get_task_by_id uses caching mechanism."""
        created_task = task_manager.create_task(sample_task_data)

        # First call should populate cache
        first_retrieval = task_manager.get_task_by_id(created_task.id)
        assert f"task_{created_task.id}" in task_manager._task_cache

        # Second call should use cache
        second_retrieval = task_manager.get_task_by_id(created_task.id)
        assert first_retrieval is second_retrieval  # Same object from cache

    def test_add_task_legacy_method(self, task_manager):
        """Test legacy add_task method for backwards compatibility."""
        task_id = task_manager.add_task(
            title="Legacy Task",
            description="Testing legacy method",
            component_area="testing",
            phase=1,
            priority="medium",
            complexity="low",
            source_document="legacy.md",
            success_criteria="Task completed",
            time_estimate_hours=2.0,
        )

        assert isinstance(task_id, int)
        assert task_id > 0

        # Verify task exists
        task = task_manager.get_task_by_id(task_id)
        assert task is not None
        assert task.title == "Legacy Task"


class TestTaskQueryOperations(TestTaskManagerFixtures):
    """Test task querying and filtering operations."""

    def test_get_tasks_with_computed_fields_no_filters(
        self, task_manager, sample_task_data
    ):
        """Test getting all tasks with computed fields."""
        # Create multiple tasks
        task1_data = sample_task_data.copy()
        task1_data["title"] = "Task 1"
        task_manager.create_task(task1_data)

        task2_data = sample_task_data.copy()
        task2_data["title"] = "Task 2"
        task2_data["priority"] = "low"
        task_manager.create_task(task2_data)

        tasks = task_manager.get_tasks_with_computed_fields()

        assert len(tasks) == 2
        assert all(isinstance(task, TaskCore) for task in tasks)

        # Tasks should be sorted by priority DESC, created_at ASC
        assert tasks[0].title == "Task 1"  # High priority first
        assert tasks[1].title == "Task 2"  # Low priority second

    def test_get_tasks_with_computed_fields_with_filters(
        self, task_manager, sample_task_data
    ):
        """Test getting tasks with filters applied."""
        # Create tasks with different statuses
        task1_data = sample_task_data.copy()
        task1_data["status"] = "not_started"
        task_manager.create_task(task1_data)

        task2_data = sample_task_data.copy()
        task2_data["status"] = "in_progress"
        task_manager.create_task(task2_data)

        # Filter by status
        not_started_tasks = task_manager.get_tasks_with_computed_fields(
            filters={"status": "not_started"}
        )
        assert len(not_started_tasks) == 1
        assert not_started_tasks[0].status == TaskStatus.NOT_STARTED

        # Filter by priority
        high_priority_tasks = task_manager.get_tasks_with_computed_fields(
            filters={"priority": "high"}
        )
        assert len(high_priority_tasks) == 2
        assert all(task.priority == TaskPriority.HIGH for task in high_priority_tasks)

    def test_get_tasks_by_phase(self, task_manager, sample_task_data):
        """Test retrieving tasks by phase."""
        # Create tasks in different phases
        task1_data = sample_task_data.copy()
        task1_data["phase"] = 1
        task_manager.create_task(task1_data)

        task2_data = sample_task_data.copy()
        task2_data["phase"] = 2
        task_manager.create_task(task2_data)

        phase1_tasks = task_manager.get_tasks_by_phase(1)
        assert len(phase1_tasks) == 1
        assert phase1_tasks[0]["phase"] == 1

        phase2_tasks = task_manager.get_tasks_by_phase(2)
        assert len(phase2_tasks) == 1
        assert phase2_tasks[0]["phase"] == 2

    def test_get_tasks_by_component(self, task_manager, sample_task_data):
        """Test retrieving tasks by component area."""
        # Create tasks in different components
        task1_data = sample_task_data.copy()
        task1_data["component_area"] = "security"
        task_manager.create_task(task1_data)

        task2_data = sample_task_data.copy()
        task2_data["component_area"] = "ui"
        task_manager.create_task(task2_data)

        security_tasks = task_manager.get_tasks_by_component("security")
        assert len(security_tasks) == 1
        assert security_tasks[0]["component_area"] == "security"

        ui_tasks = task_manager.get_tasks_by_component("ui")
        assert len(ui_tasks) == 1
        assert ui_tasks[0]["component_area"] == "ui"

    def test_get_tasks_by_status(self, task_manager, sample_task_data):
        """Test retrieving tasks by status."""
        task1 = task_manager.create_task(sample_task_data)

        # Update one task to in_progress
        task_manager.update_task_status(task1.id, "in_progress", "Started work")

        not_started_tasks = task_manager.get_tasks_by_status("not_started")
        in_progress_tasks = task_manager.get_tasks_by_status("in_progress")

        assert len(not_started_tasks) == 0
        assert len(in_progress_tasks) == 1
        assert in_progress_tasks[0]["status"] == "in_progress"

    def test_search_tasks(self, task_manager, sample_task_data):
        """Test searching tasks by title or description."""
        # Create tasks with different titles and descriptions
        task1_data = sample_task_data.copy()
        task1_data["title"] = "Authentication system"
        task1_data["description"] = "Implement JWT tokens"
        task_manager.create_task(task1_data)

        task2_data = sample_task_data.copy()
        task2_data["title"] = "Database migration"
        task2_data["description"] = "Update authentication tables"
        task_manager.create_task(task2_data)

        # Search by title
        auth_tasks = task_manager.search_tasks("authentication")
        assert len(auth_tasks) == 2  # Both contain "authentication"

        # Search by description
        jwt_tasks = task_manager.search_tasks("JWT")
        assert len(jwt_tasks) == 1
        assert jwt_tasks[0]["title"] == "Authentication system"


class TestTaskDependencyOperations(TestTaskManagerFixtures):
    """Test task dependency creation and management."""

    def test_create_task_dependency_with_dict(self, task_manager, sample_task_data):
        """Test creating task dependency with dictionary data."""
        # Create two tasks
        task1 = task_manager.create_task(sample_task_data)

        task2_data = sample_task_data.copy()
        task2_data["title"] = "Setup database"
        task2 = task_manager.create_task(task2_data)

        # Create dependency
        dependency_data = {
            "task_id": task1.id,
            "depends_on_task_id": task2.id,
            "dependency_type": DependencyType.BLOCKS,
        }

        dependency = task_manager.create_task_dependency(dependency_data)

        assert isinstance(dependency, TaskDependency)
        assert dependency.id is not None
        assert dependency.task_id == task1.id
        assert dependency.depends_on_task_id == task2.id
        assert dependency.dependency_type == DependencyType.BLOCKS

    def test_create_task_dependency_with_instance(self, task_manager, sample_task_data):
        """Test creating task dependency with TaskDependency instance."""
        task1 = task_manager.create_task(sample_task_data)

        task2_data = sample_task_data.copy()
        task2_data["title"] = "Setup database"
        task2 = task_manager.create_task(task2_data)

        dependency = TaskDependency(
            task_id=task1.id,
            depends_on_task_id=task2.id,
            dependency_type=DependencyType.REQUIRES,
        )

        created_dependency = task_manager.create_task_dependency(dependency)

        assert created_dependency.id is not None
        assert created_dependency.dependency_type == DependencyType.REQUIRES

    def test_create_dependency_non_existent_task_raises_error(
        self, task_manager, sample_task_data
    ):
        """Test creating dependency with non-existent task raises error."""
        task1 = task_manager.create_task(sample_task_data)

        dependency_data = {
            "task_id": task1.id,
            "depends_on_task_id": 999999,  # Non-existent
            "dependency_type": DependencyType.BLOCKS,
        }

        with pytest.raises(ValueError, match="does not exist"):
            task_manager.create_task_dependency(dependency_data)

    def test_create_circular_dependency_raises_error(
        self, task_manager, sample_task_data
    ):
        """Test creating circular dependency raises error."""
        task1 = task_manager.create_task(sample_task_data)

        task2_data = sample_task_data.copy()
        task2_data["title"] = "Task 2"
        task2 = task_manager.create_task(task2_data)

        # Create first dependency: task1 depends on task2
        dependency1_data = {
            "task_id": task1.id,
            "depends_on_task_id": task2.id,
            "dependency_type": DependencyType.BLOCKS,
        }
        task_manager.create_task_dependency(dependency1_data)

        # Try to create circular dependency: task2 depends on task1
        dependency2_data = {
            "task_id": task2.id,
            "depends_on_task_id": task1.id,
            "dependency_type": DependencyType.BLOCKS,
        }

        with pytest.raises(ValueError, match="circular dependency"):
            task_manager.create_task_dependency(dependency2_data)

    def test_get_task_dependencies(self, task_manager, sample_task_data):
        """Test retrieving task dependencies."""
        task1 = task_manager.create_task(sample_task_data)

        task2_data = sample_task_data.copy()
        task2_data["title"] = "Setup database"
        task2 = task_manager.create_task(task2_data)

        # Create dependency
        task_manager.add_dependency(task1.id, task2.id, "blocks")

        dependencies = task_manager.get_task_dependencies(task1.id)
        assert len(dependencies) == 1
        assert dependencies[0]["depends_on_task_id"] == task2.id
        assert dependencies[0]["depends_on_title"] == "Setup database"

    def test_add_dependency_legacy_method(self, task_manager, sample_task_data):
        """Test legacy add_dependency method."""
        task1 = task_manager.create_task(sample_task_data)

        task2_data = sample_task_data.copy()
        task2_data["title"] = "Setup database"
        task2 = task_manager.create_task(task2_data)

        # No exception should be raised
        task_manager.add_dependency(task1.id, task2.id, "requires")

        # Verify dependency was created
        dependencies = task_manager.get_task_dependencies(task1.id)
        assert len(dependencies) == 1
        assert dependencies[0]["dependency_type"] == "requires"


class TestTaskStatusAndProgressOperations(TestTaskManagerFixtures):
    """Test task status updates and progress tracking."""

    def test_update_task_status(self, task_manager, sample_task_data):
        """Test updating task status with progress tracking."""
        task = task_manager.create_task(sample_task_data)

        # Update status
        task_manager.update_task_status(
            task.id, "in_progress", "Started implementation"
        )

        # Verify status was updated
        updated_task = task_manager.get_task_by_id(task.id)
        assert updated_task.status == TaskStatus.IN_PROGRESS

        # Verify progress was recorded
        progress_records = task_manager.get_task_progress(task.id)
        assert len(progress_records) >= 2  # Initial + update

        # Find the progress update
        progress_update = next(
            (p for p in progress_records if p["notes"] == "Started implementation"),
            None,
        )
        assert progress_update is not None
        assert progress_update["progress_percentage"] == 50  # in_progress = 50%

    def test_update_task_status_with_comments(self, task_manager, sample_task_data):
        """Test status update creates comments when notes provided."""
        task = task_manager.create_task(sample_task_data)

        notes = "Implemented authentication middleware"
        task_manager.update_task_status(task.id, "completed", notes)

        # Verify comment was created
        comments = task_manager.get_task_comments(task.id)
        assert len(comments) >= 1

        # Find the status comment
        status_comment = next((c for c in comments if c["comment"] == notes), None)
        assert status_comment is not None
        assert status_comment["comment_type"] == "note"

    def test_get_task_progress(self, task_manager, sample_task_data):
        """Test retrieving task progress history."""
        task = task_manager.create_task(sample_task_data)

        # Update status multiple times
        task_manager.update_task_status(task.id, "in_progress", "Started work")
        task_manager.update_task_status(task.id, "completed", "Finished work")

        progress_records = task_manager.get_task_progress(task.id)

        # Should have initial + 2 updates
        assert len(progress_records) == 3

        # Records should be ordered by created_at DESC
        assert progress_records[0]["notes"] == "Finished work"
        assert progress_records[1]["notes"] == "Started work"
        assert progress_records[2]["notes"] == "Task created"

    def test_add_task_comment(self, task_manager, sample_task_data):
        """Test adding comments to tasks."""
        task = task_manager.create_task(sample_task_data)

        task_manager.add_task_comment(
            task.id, "Need to review security implications", "review"
        )

        comments = task_manager.get_task_comments(task.id)
        assert len(comments) == 1
        assert comments[0]["comment"] == "Need to review security implications"
        assert comments[0]["comment_type"] == "review"

    def test_get_task_comments(self, task_manager, sample_task_data):
        """Test retrieving task comments."""
        task = task_manager.create_task(sample_task_data)

        # Add multiple comments
        task_manager.add_task_comment(task.id, "First comment", "note")
        task_manager.add_task_comment(task.id, "Second comment", "issue")

        comments = task_manager.get_task_comments(task.id)
        assert len(comments) == 2

        # Comments should be ordered by created_at DESC
        assert comments[0]["comment"] == "Second comment"
        assert comments[1]["comment"] == "First comment"


class TestTaskAnalyticsAndStatistics(TestTaskManagerFixtures):
    """Test task analytics and project statistics."""

    def test_get_task_analytics_empty_database(self, task_manager):
        """Test analytics with no tasks returns proper empty structure."""
        analytics = task_manager.get_task_analytics()

        assert analytics["total_tasks"] == 0
        assert analytics["analytics"] == {}
        assert analytics["computed_metrics"] == {}

    def test_get_task_analytics_with_tasks(self, task_manager, sample_task_data):
        """Test analytics with multiple tasks."""
        # Create tasks with different attributes
        task1_data = sample_task_data.copy()
        task1_data["priority"] = "high"
        task1_data["complexity"] = "high"
        task1_data["status"] = "completed"
        task_manager.create_task(task1_data)

        task2_data = sample_task_data.copy()
        task2_data["priority"] = "low"
        task2_data["complexity"] = "low"
        task2_data["status"] = "not_started"
        task_manager.create_task(task2_data)

        analytics = task_manager.get_task_analytics()

        assert analytics["total_tasks"] == 2

        # Check status distribution
        assert analytics["status_distribution"]["completed"] == 1
        assert analytics["status_distribution"]["not_started"] == 1

        # Check priority distribution
        assert analytics["priority_distribution"]["high"] == 1
        assert analytics["priority_distribution"]["low"] == 1

        # Check computed metrics
        assert analytics["computed_metrics"]["completion_rate"] == 50.0
        assert analytics["computed_metrics"]["total_estimated_hours"] == 16.0  # 8.0 * 2

    def test_get_project_stats(self, task_manager, sample_task_data):
        """Test comprehensive project statistics."""
        # Create tasks with different statuses
        task1_data = sample_task_data.copy()
        task1_data["status"] = "completed"
        task_manager.create_task(task1_data)

        task2_data = sample_task_data.copy()
        task2_data["status"] = "in_progress"
        task_manager.create_task(task2_data)

        task3_data = sample_task_data.copy()
        task3_data["status"] = "blocked"
        task_manager.create_task(task3_data)

        stats = task_manager.get_project_stats()

        assert stats["total_tasks"] == 3
        assert stats["completed_tasks"] == 1
        assert stats["in_progress_tasks"] == 1
        assert stats["blocked_tasks"] == 1
        assert stats["not_started_tasks"] == 0
        assert stats["completion_percentage"] == pytest.approx(33.33, rel=1e-2)
        assert stats["total_estimated_hours"] == 24.0  # 8.0 * 3
        assert stats["completed_hours"] == 8.0
        assert stats["remaining_hours"] == 16.0

    def test_get_critical_path(self, task_manager, sample_task_data):
        """Test retrieving critical path tasks."""
        # Create high priority task
        critical_task_data = sample_task_data.copy()
        critical_task_data["priority"] = "critical"
        critical_task_data["title"] = "Critical Security Fix"
        task_manager.create_task(critical_task_data)

        # Create regular task
        task_manager.create_task(sample_task_data)

        critical_path = task_manager.get_critical_path()

        # Critical priority task should be first
        assert len(critical_path) >= 1
        assert critical_path[0]["priority"] == "critical"
        assert critical_path[0]["title"] == "Critical Security Fix"

    def test_get_blocked_tasks(self, task_manager, sample_task_data):
        """Test retrieving blocked tasks."""
        # Create tasks with dependencies
        task1_data = sample_task_data.copy()
        task1_data["title"] = "Setup database"
        task1_data["status"] = "not_started"
        task1 = task_manager.create_task(task1_data)

        task2_data = sample_task_data.copy()
        task2_data["title"] = "Create user table"
        task2_data["status"] = "not_started"
        task2 = task_manager.create_task(task2_data)

        # Create dependency: task2 depends on task1
        task_manager.add_dependency(task2.id, task1.id, "blocks")

        blocked_tasks = task_manager.get_blocked_tasks()

        # task2 should be blocked by task1
        assert len(blocked_tasks) == 1
        assert blocked_tasks[0]["title"] == "Create user table"
        assert "Setup database" in blocked_tasks[0]["blocking_tasks"]

    def test_get_ready_tasks(self, task_manager, sample_task_data):
        """Test retrieving ready-to-start tasks."""
        # Create independent task (ready)
        ready_task_data = sample_task_data.copy()
        ready_task_data["title"] = "Independent task"
        ready_task_data["status"] = "not_started"
        ready_task = task_manager.create_task(ready_task_data)

        # Create dependent task (not ready)
        blocked_task_data = sample_task_data.copy()
        blocked_task_data["title"] = "Dependent task"
        blocked_task_data["status"] = "not_started"
        blocked_task = task_manager.create_task(blocked_task_data)

        # Create completed task as dependency
        completed_task_data = sample_task_data.copy()
        completed_task_data["title"] = "Completed task"
        completed_task_data["status"] = "completed"
        completed_task = task_manager.create_task(completed_task_data)

        # Create another dependent task that should be ready
        ready_dependent_data = sample_task_data.copy()
        ready_dependent_data["title"] = "Ready dependent task"
        ready_dependent_data["status"] = "not_started"
        ready_dependent_task = task_manager.create_task(ready_dependent_data)

        # Create dependencies
        task_manager.add_dependency(
            blocked_task.id, ready_task.id, "blocks"
        )  # blocked by not_started
        task_manager.add_dependency(
            ready_dependent_task.id, completed_task.id, "blocks"
        )  # blocked by completed

        ready_tasks = task_manager.get_ready_tasks()

        # Should include independent task and task depending on completed task
        ready_titles = [task["title"] for task in ready_tasks]
        assert "Independent task" in ready_titles
        assert "Ready dependent task" in ready_titles
        assert "Dependent task" not in ready_titles

    def test_get_next_tasks(self, task_manager, sample_task_data):
        """Test getting next tasks to work on."""
        # Create multiple ready tasks with different priorities
        high_priority_data = sample_task_data.copy()
        high_priority_data["title"] = "High priority task"
        high_priority_data["priority"] = "high"
        high_priority_data["status"] = "not_started"
        task_manager.create_task(high_priority_data)

        low_priority_data = sample_task_data.copy()
        low_priority_data["title"] = "Low priority task"
        low_priority_data["priority"] = "low"
        low_priority_data["status"] = "not_started"
        task_manager.create_task(low_priority_data)

        next_tasks = task_manager.get_next_tasks(limit=5)

        assert len(next_tasks) >= 2
        # High priority task should come first
        assert next_tasks[0]["title"] == "High priority task"
        assert next_tasks[0]["priority"] == "high"


class TestTaskDependencyValidation(TestTaskManagerFixtures):
    """Test TaskDependency model validation and business logic."""

    def test_task_dependency_creation_with_valid_data(self):
        """Test TaskDependency creation with valid data."""
        dependency = TaskDependency(
            task_id=1,
            depends_on_task_id=2,
            dependency_type=DependencyType.BLOCKS,
        )

        assert dependency.task_id == 1
        assert dependency.depends_on_task_id == 2
        assert dependency.dependency_type == DependencyType.BLOCKS
        assert dependency.is_blocking is True
        assert dependency.dependency_strength == 1.0

    def test_task_dependency_validation_positive_ids(self):
        """Test TaskDependency validates positive task IDs."""
        with pytest.raises(ValueError, match="Task IDs must be positive integers"):
            TaskDependency(task_id=0, depends_on_task_id=1)

        with pytest.raises(ValueError, match="Task IDs must be positive integers"):
            TaskDependency(task_id=1, depends_on_task_id=-1)

    def test_task_dependency_validation_no_self_dependency(self):
        """Test TaskDependency prevents self-dependency."""
        with pytest.raises(ValueError, match="Task cannot depend on itself"):
            TaskDependency(task_id=1, depends_on_task_id=1)

    def test_task_dependency_computed_fields(self):
        """Test TaskDependency computed fields."""
        blocking_dep = TaskDependency(
            task_id=1, depends_on_task_id=2, dependency_type=DependencyType.BLOCKS
        )
        assert blocking_dep.is_blocking is True
        assert blocking_dep.dependency_strength == 1.0

        requires_dep = TaskDependency(
            task_id=1, depends_on_task_id=2, dependency_type=DependencyType.REQUIRES
        )
        assert requires_dep.is_blocking is False
        assert requires_dep.dependency_strength == 0.8

        enables_dep = TaskDependency(
            task_id=1, depends_on_task_id=2, dependency_type=DependencyType.ENABLES
        )
        assert enables_dep.dependency_strength == 0.6

        enhances_dep = TaskDependency(
            task_id=1, depends_on_task_id=2, dependency_type=DependencyType.ENHANCES
        )
        assert enhances_dep.dependency_strength == 0.3


class TestRowConversionMethods(TestTaskManagerFixtures):
    """Test database row to model conversion methods."""

    def test_row_to_task_with_valid_data(self, task_manager, sample_task_data):
        """Test converting database row to TaskCore model."""
        # Create task first
        task = task_manager.create_task(sample_task_data)

        # Get raw row from database
        with task_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE id = ?", (task.id,))
            row = cursor.fetchone()

            converted_task = task_manager._row_to_task(row)

            assert isinstance(converted_task, TaskCore)
            assert converted_task.id == task.id
            assert converted_task.title == task.title
            assert converted_task.status == task.status

    def test_row_to_task_with_invalid_data_raises_error(self, task_manager):
        """Test converting invalid row data raises ValueError."""
        # Create mock row with invalid data
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = KeyError("Missing required field")

        with pytest.raises(ValueError, match="Failed to create Task from database row"):
            task_manager._row_to_task(mock_row)

    def test_row_to_task_dependency_with_valid_data(
        self, task_manager, sample_task_data
    ):
        """Test converting database row to TaskDependency model."""
        # Create tasks and dependency
        task1 = task_manager.create_task(sample_task_data)

        task2_data = sample_task_data.copy()
        task2_data["title"] = "Task 2"
        task2 = task_manager.create_task(task2_data)

        dependency = task_manager.create_task_dependency(
            {
                "task_id": task1.id,
                "depends_on_task_id": task2.id,
                "dependency_type": DependencyType.BLOCKS,
            }
        )

        # Get raw row from database
        with task_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM task_dependencies WHERE id = ?", (dependency.id,)
            )
            row = cursor.fetchone()

            converted_dependency = task_manager._row_to_task_dependency(row)

            assert isinstance(converted_dependency, TaskDependency)
            assert converted_dependency.id == dependency.id
            assert converted_dependency.task_id == dependency.task_id
            assert (
                converted_dependency.depends_on_task_id == dependency.depends_on_task_id
            )


class TestTaskManagerErrorHandling(TestTaskManagerFixtures):
    """Test error handling and edge cases."""

    def test_create_task_with_missing_required_fields(self, task_manager):
        """Test creating task with missing required fields raises error."""
        incomplete_data = {"description": "Missing title and other required fields"}

        with pytest.raises(ValueError):
            task_manager.create_task(incomplete_data)

    def test_create_dependency_with_same_task_ids(self, task_manager, sample_task_data):
        """Test creating dependency with same task IDs raises error."""
        task = task_manager.create_task(sample_task_data)

        dependency_data = {
            "task_id": task.id,
            "depends_on_task_id": task.id,  # Same as task_id
            "dependency_type": DependencyType.BLOCKS,
        }

        with pytest.raises(ValueError, match="Task cannot depend on itself"):
            task_manager.create_task_dependency(dependency_data)

    @patch("sqlite3.connect")
    def test_database_connection_error_handling(self, mock_connect, temp_db_path):
        """Test handling of database connection errors."""
        mock_connect.side_effect = sqlite3.Error("Database connection failed")

        tm = TaskManager(db_path=temp_db_path)

        with pytest.raises(sqlite3.Error):
            tm._get_connection()

    def test_update_non_existent_task_status(self, task_manager):
        """Test updating status of non-existent task."""
        # Should not raise error, but also shouldn't update anything
        task_manager.update_task_status(999999, "completed", "Non-existent task")

        # Verify no task exists
        task = task_manager.get_task_by_id(999999)
        assert task is None


class TestTaskManagerPerformanceAndCaching(TestTaskManagerFixtures):
    """Test performance optimizations and caching behavior."""

    def test_task_cache_population(self, task_manager, sample_task_data):
        """Test that task cache is populated correctly."""
        task = task_manager.create_task(sample_task_data)

        # Cache should be empty initially
        assert len(task_manager._task_cache) == 0

        # First retrieval should populate cache
        retrieved_task = task_manager.get_task_by_id(task.id)
        cache_key = f"task_{task.id}"
        assert cache_key in task_manager._task_cache
        assert task_manager._task_cache[cache_key] is retrieved_task

    def test_cache_ttl_setting(self, task_manager):
        """Test cache TTL setting."""
        assert task_manager._cache_ttl == 300  # 5 minutes default

    def test_bulk_task_operations_efficiency(self, task_manager, sample_task_data):
        """Test efficiency of bulk operations."""
        # Create multiple tasks
        tasks = []
        for i in range(10):
            task_data = sample_task_data.copy()
            task_data["title"] = f"Task {i}"
            task = task_manager.create_task(task_data)
            tasks.append(task)

        # Bulk retrieval should be efficient
        all_tasks = task_manager.get_tasks_with_computed_fields()
        assert len(all_tasks) == 10

        # Analytics should handle multiple tasks efficiently
        analytics = task_manager.get_task_analytics()
        assert analytics["total_tasks"] == 10


class TestTaskManagerIntegration(TestTaskManagerFixtures):
    """Integration tests combining multiple TaskManager features."""

    def test_complete_task_workflow(self, task_manager, sample_task_data):
        """Test complete task workflow from creation to completion."""
        # 1. Create task
        task = task_manager.create_task(sample_task_data)
        assert task.status == TaskStatus.NOT_STARTED

        # 2. Start work
        task_manager.update_task_status(
            task.id, "in_progress", "Started implementation"
        )
        updated_task = task_manager.get_task_by_id(task.id)
        assert updated_task.status == TaskStatus.IN_PROGRESS

        # 3. Add progress comment
        task_manager.add_task_comment(
            task.id, "Completed authentication middleware", "progress"
        )

        # 4. Complete task
        task_manager.update_task_status(task.id, "completed", "Implementation finished")
        final_task = task_manager.get_task_by_id(task.id)
        assert final_task.status == TaskStatus.COMPLETED

        # 5. Verify analytics reflect completion
        analytics = task_manager.get_task_analytics()
        assert analytics["computed_metrics"]["completion_rate"] == 100.0

    def test_dependency_workflow(self, task_manager, sample_task_data):
        """Test complete dependency workflow."""
        # Create prerequisite task
        prereq_data = sample_task_data.copy()
        prereq_data["title"] = "Setup database schema"
        prereq_task = task_manager.create_task(prereq_data)

        # Create dependent task
        dependent_data = sample_task_data.copy()
        dependent_data["title"] = "Create user authentication"
        dependent_task = task_manager.create_task(dependent_data)

        # Create dependency
        task_manager.add_dependency(dependent_task.id, prereq_task.id, "blocks")

        # Verify dependent task is blocked
        blocked_tasks = task_manager.get_blocked_tasks()
        assert len(blocked_tasks) == 1
        assert blocked_tasks[0]["title"] == "Create user authentication"

        # Complete prerequisite
        task_manager.update_task_status(prereq_task.id, "completed", "Schema created")

        # Verify dependent task is now ready
        ready_tasks = task_manager.get_ready_tasks()
        ready_titles = [task["title"] for task in ready_tasks]
        assert "Create user authentication" in ready_titles

    def test_multi_phase_project_workflow(self, task_manager, sample_task_data):
        """Test workflow across multiple project phases."""
        # Create tasks in different phases
        phases_data = []
        for phase in range(1, 4):
            for i in range(3):  # 3 tasks per phase
                task_data = sample_task_data.copy()
                task_data["title"] = f"Phase {phase} Task {i + 1}"
                task_data["phase"] = phase
                task_data["priority"] = ["high", "medium", "low"][i]
                phases_data.append(task_data)

        # Create all tasks
        created_tasks = []
        for task_data in phases_data:
            task = task_manager.create_task(task_data)
            created_tasks.append(task)

        # Verify phase distribution
        stats = task_manager.get_project_stats()
        phase_breakdown = {p["phase"]: p["count"] for p in stats["phase_breakdown"]}
        assert phase_breakdown[1] == 3
        assert phase_breakdown[2] == 3
        assert phase_breakdown[3] == 3

        # Complete all phase 1 tasks
        phase1_tasks = task_manager.get_tasks_by_phase(1)
        for task in phase1_tasks:
            task_manager.update_task_status(
                task["id"], "completed", f"Completed {task['title']}"
            )

        # Verify completion statistics
        updated_stats = task_manager.get_project_stats()
        assert updated_stats["completed_tasks"] == 3
        assert updated_stats["completion_percentage"] == pytest.approx(33.33, rel=1e-2)

"""Comprehensive tests for repository classes.

This module provides testing for BaseRepository, TaskRepository, and
TaskExecutionRepository with full CRUD operations, relationships,
and business logic validation.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlmodel import Field, Session, SQLModel

from src.repositories.base import BaseRepository
from src.repositories.task_repository import TaskExecutionRepository, TaskRepository
from src.schemas.database import (
    Task,
    TaskDependency,
    TaskExecutionLog,
    TaskProgress,
)
from src.schemas.unified_models import (
    AgentReport,
    AgentType,
    ComponentArea,
    TaskComplexity,
    TaskCore,
    TaskPriority,
    TaskStatus,
)


class TestRepositoryFixtures:
    """Test fixtures and database setup for repository tests."""

    @pytest.fixture
    def temp_engine(self):
        """Create temporary SQLite engine for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temp_file:
            db_path = temp_file.name

        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        SQLModel.metadata.create_all(engine)

        yield engine

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def test_session(self, temp_engine):
        """Create test database session."""
        with Session(temp_engine) as session:
            yield session

    @pytest.fixture
    def sample_task_core(self):
        """Sample TaskCore for testing."""
        return TaskCore(
            title="Test Authentication System",
            description="Implement JWT-based authentication with secure tokens",
            component_area=ComponentArea.SECURITY,
            phase=1,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            status=TaskStatus.NOT_STARTED,
            source_document="auth_requirements.md",
            success_criteria="Users can login and logout securely",
            time_estimate_hours=8.0,
        )

    @pytest.fixture
    def sample_task_entity(self, sample_task_core):
        """Sample Task entity for testing."""
        return Task.from_core_model(sample_task_core)

    @pytest.fixture
    def sample_agent_report(self):
        """Sample AgentReport for testing."""
        return AgentReport(
            agent_name=AgentType.CODING,
            task_id=1,
            status=TaskStatus.COMPLETED,
            success=True,
            execution_time_minutes=45.5,
            outputs={"implementation": "Authentication system completed"},
            artifacts=["auth.py", "models.py", "tests.py"],
            recommendations=["Add rate limiting", "Implement 2FA"],
            next_actions=["Deploy to staging", "Run security audit"],
            confidence_score=0.92,
            error_details=None,
        )


class TestBaseRepository(TestRepositoryFixtures):
    """Test BaseRepository generic functionality."""

    class MockEntity(SQLModel, table=True):
        """Mock entity for testing."""

        __tablename__ = "mock_entities"

        id: int | None = Field(default=None, primary_key=True)
        name: str
        value: int = 0
        updated_at: datetime | None = None

    class MockBusinessModel:
        """Mock business model for testing."""

        def __init__(self, name: str, value: int = 0):
            self.name = name
            self.value = value

        def model_dump(self, exclude_unset: bool = False) -> dict:
            """Mock model_dump method."""
            return {"name": self.name, "value": self.value}

    class TestRepository(BaseRepository):
        """Test implementation of BaseRepository."""

        def get_entity_class(self):
            return TestBaseRepository.MockEntity

        def get_business_class(self):
            return TestBaseRepository.MockBusinessModel

    @pytest.fixture
    def mock_entity_engine(self):
        """Create engine with mock entity table."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temp_file:
            db_path = temp_file.name

        engine = create_engine(f"sqlite:///{db_path}", echo=False)

        # Create only the mock entity table
        from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table

        metadata = MetaData()

        Table(
            "mock_entities",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String),
            Column("value", Integer, default=0),
            Column("updated_at", DateTime),
        )

        metadata.create_all(engine)
        yield engine

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def mock_session(self, mock_entity_engine):
        """Create session with mock entity."""
        with Session(mock_entity_engine) as session:
            yield session

    @pytest.fixture
    def test_repository(self, mock_session):
        """Create test repository instance."""
        return self.TestRepository(mock_session)

    def test_repository_initialization(self, mock_session):
        """Test repository initialization."""
        repo = self.TestRepository(mock_session)
        assert repo.session is mock_session
        assert repo.get_entity_class() == self.MockEntity
        assert repo.get_business_class() == self.MockBusinessModel

    def test_create_entity(self, test_repository):
        """Test creating entity from business model."""
        business_model = self.MockBusinessModel("Test Entity", 42)
        entity = test_repository.create(business_model)

        assert isinstance(entity, self.MockEntity)
        assert entity.name == "Test Entity"
        assert entity.value == 42
        assert entity.id is not None

    def test_get_by_id_existing(self, test_repository):
        """Test getting entity by ID."""
        # Create entity
        business_model = self.MockBusinessModel("Get Test", 100)
        created_entity = test_repository.create(business_model)
        test_repository.session.commit()

        # Retrieve by ID
        retrieved_entity = test_repository.get_by_id(created_entity.id)
        assert retrieved_entity is not None
        assert retrieved_entity.id == created_entity.id
        assert retrieved_entity.name == "Get Test"

    def test_get_by_id_non_existent(self, test_repository):
        """Test getting non-existent entity returns None."""
        entity = test_repository.get_by_id(999999)
        assert entity is None

    def test_update_entity(self, test_repository):
        """Test updating entity."""
        # Create entity
        business_model = self.MockBusinessModel("Update Test", 200)
        entity = test_repository.create(business_model)
        test_repository.session.commit()

        # Update entity
        updated_entity = test_repository.update(
            entity.id, {"name": "Updated Name", "value": 300}
        )

        assert updated_entity is not None
        assert updated_entity.name == "Updated Name"
        assert updated_entity.value == 300
        assert updated_entity.updated_at is not None

    def test_update_non_existent(self, test_repository):
        """Test updating non-existent entity returns None."""
        result = test_repository.update(999999, {"name": "Not Found"})
        assert result is None

    def test_delete_existing(self, test_repository):
        """Test deleting existing entity."""
        # Create entity
        business_model = self.MockBusinessModel("Delete Test", 400)
        entity = test_repository.create(business_model)
        test_repository.session.commit()

        # Delete entity
        result = test_repository.delete(entity.id)
        assert result is True

        # Verify deletion
        deleted_entity = test_repository.get_by_id(entity.id)
        assert deleted_entity is None

    def test_delete_non_existent(self, test_repository):
        """Test deleting non-existent entity returns False."""
        result = test_repository.delete(999999)
        assert result is False

    def test_list_all(self, test_repository):
        """Test listing all entities."""
        # Create multiple entities
        for i in range(5):
            business_model = self.MockBusinessModel(f"Entity {i}", i * 10)
            test_repository.create(business_model)
        test_repository.session.commit()

        # List all
        entities = test_repository.list_all()
        assert len(entities) == 5
        assert all(isinstance(e, self.MockEntity) for e in entities)

    def test_list_all_with_limit(self, test_repository):
        """Test listing entities with limit."""
        # Create multiple entities
        for i in range(10):
            business_model = self.MockBusinessModel(f"Limited {i}", i)
            test_repository.create(business_model)
        test_repository.session.commit()

        # List with limit
        entities = test_repository.list_all(limit=3)
        assert len(entities) == 3

    def test_count_entities(self, test_repository):
        """Test counting entities."""
        # Initially should be 0
        assert test_repository.count() == 0

        # Create entities
        for i in range(7):
            business_model = self.MockBusinessModel(f"Count {i}", i)
            test_repository.create(business_model)
        test_repository.session.commit()

        # Count should be 7
        assert test_repository.count() == 7

    def test_exists_true(self, test_repository):
        """Test exists returns True for existing entity."""
        business_model = self.MockBusinessModel("Exists Test", 500)
        entity = test_repository.create(business_model)
        test_repository.session.commit()

        assert test_repository.exists(entity.id) is True

    def test_exists_false(self, test_repository):
        """Test exists returns False for non-existent entity."""
        assert test_repository.exists(999999) is False


class TestTaskRepository(TestRepositoryFixtures):
    """Test TaskRepository specific functionality."""

    @pytest.fixture
    def task_repository(self, test_session):
        """Create TaskRepository instance."""
        return TaskRepository(test_session)

    def test_repository_classes(self, task_repository):
        """Test repository class definitions."""
        assert task_repository.get_entity_class() == Task
        assert task_repository.get_business_class() == TaskCore

    def test_create_task_from_core_model(self, task_repository, sample_task_core):
        """Test creating task from TaskCore business model."""
        task = task_repository.create(sample_task_core)

        assert isinstance(task, Task)
        assert task.id is not None
        assert task.title == sample_task_core.title
        assert task.description == sample_task_core.description
        assert task.status == sample_task_core.status

    def test_get_by_status(self, task_repository, sample_task_core):
        """Test getting tasks by status."""
        # Create tasks with different statuses
        task_repository.create(sample_task_core)

        task2_core = sample_task_core.model_copy()
        task2_core.title = "In Progress Task"
        task2 = task_repository.create(task2_core)
        task2.status = TaskStatus.IN_PROGRESS
        task_repository.session.add(task2)

        task_repository.session.commit()

        # Test getting by status
        not_started_tasks = task_repository.get_by_status(TaskStatus.NOT_STARTED)
        assert len(not_started_tasks) == 1
        assert not_started_tasks[0].status == TaskStatus.NOT_STARTED

        in_progress_tasks = task_repository.get_by_status(TaskStatus.IN_PROGRESS)
        assert len(in_progress_tasks) == 1
        assert in_progress_tasks[0].status == TaskStatus.IN_PROGRESS

    def test_get_by_component_area(self, task_repository, sample_task_core):
        """Test getting tasks by component area."""
        # Create tasks in different component areas
        task_repository.create(sample_task_core)

        ui_task_core = sample_task_core.model_copy()
        ui_task_core.title = "UI Task"
        ui_task_core.component_area = ComponentArea.UI
        task_repository.create(ui_task_core)

        task_repository.session.commit()

        # Test getting by component area
        security_tasks = task_repository.get_by_component_area(ComponentArea.SECURITY)
        assert len(security_tasks) == 1
        assert security_tasks[0].component_area == ComponentArea.SECURITY

        ui_tasks = task_repository.get_by_component_area(ComponentArea.UI)
        assert len(ui_tasks) == 1
        assert ui_tasks[0].component_area == ComponentArea.UI

    def test_get_actionable_tasks(self, task_repository, sample_task_core):
        """Test getting actionable (ready to work on) tasks."""
        # Create tasks with different statuses
        task_repository.create(sample_task_core)

        in_progress_core = sample_task_core.model_copy()
        in_progress_core.title = "In Progress Task"
        in_progress_task = task_repository.create(in_progress_core)
        in_progress_task.status = TaskStatus.IN_PROGRESS
        task_repository.session.add(in_progress_task)

        completed_core = sample_task_core.model_copy()
        completed_core.title = "Completed Task"
        completed_task = task_repository.create(completed_core)
        completed_task.status = TaskStatus.COMPLETED
        task_repository.session.add(completed_task)

        task_repository.session.commit()

        # Get actionable tasks
        actionable_tasks = task_repository.get_actionable_tasks()
        assert len(actionable_tasks) == 2  # not_started + in_progress

        statuses = [task.status for task in actionable_tasks]
        assert TaskStatus.NOT_STARTED in statuses
        assert TaskStatus.IN_PROGRESS in statuses
        assert TaskStatus.COMPLETED not in statuses

    def test_get_actionable_tasks_with_limit(self, task_repository, sample_task_core):
        """Test getting actionable tasks with limit."""
        # Create multiple actionable tasks
        for i in range(5):
            task_core = sample_task_core.model_copy()
            task_core.title = f"Actionable Task {i}"
            task_repository.create(task_core)

        task_repository.session.commit()

        # Get with limit
        limited_tasks = task_repository.get_actionable_tasks(limit=3)
        assert len(limited_tasks) == 3

    def test_get_by_phase(self, task_repository, sample_task_core):
        """Test getting tasks by phase."""
        # Create tasks in different phases
        task_repository.create(sample_task_core)

        phase2_core = sample_task_core.model_copy()
        phase2_core.title = "Phase 2 Task"
        phase2_core.phase = 2
        task_repository.create(phase2_core)

        task_repository.session.commit()

        # Test getting by phase
        phase1_tasks = task_repository.get_by_phase(1)
        assert len(phase1_tasks) == 1
        assert phase1_tasks[0].phase == 1

        phase2_tasks = task_repository.get_by_phase(2)
        assert len(phase2_tasks) == 1
        assert phase2_tasks[0].phase == 2

    def test_get_ready_tasks(self, task_repository, sample_task_core):
        """Test getting ready-to-start tasks (no blocking dependencies)."""
        # Create independent task (should be ready)
        task_repository.create(sample_task_core)

        # Create task with completed dependency (should be ready)
        completed_dependency_core = sample_task_core.model_copy()
        completed_dependency_core.title = "Completed Dependency"
        completed_dependency_core.status = TaskStatus.COMPLETED
        completed_dep_task = task_repository.create(completed_dependency_core)

        ready_dependent_core = sample_task_core.model_copy()
        ready_dependent_core.title = "Ready Dependent Task"
        ready_dependent_task = task_repository.create(ready_dependent_core)

        # Create task with incomplete dependency (should not be ready)
        incomplete_dependency_core = sample_task_core.model_copy()
        incomplete_dependency_core.title = "Incomplete Dependency"
        incomplete_dep_task = task_repository.create(incomplete_dependency_core)

        blocked_task_core = sample_task_core.model_copy()
        blocked_task_core.title = "Blocked Task"
        blocked_task = task_repository.create(blocked_task_core)

        task_repository.session.commit()

        # Create dependencies
        ready_dependency = TaskDependency(
            task_id=ready_dependent_task.id,
            depends_on_task_id=completed_dep_task.id,
            dependency_type="blocks",
        )
        task_repository.session.add(ready_dependency)

        blocking_dependency = TaskDependency(
            task_id=blocked_task.id,
            depends_on_task_id=incomplete_dep_task.id,
            dependency_type="blocks",
        )
        task_repository.session.add(blocking_dependency)

        task_repository.session.commit()

        # Get ready tasks
        ready_tasks = task_repository.get_ready_tasks()
        ready_titles = [task.title for task in ready_tasks]

        assert "Test Authentication System" in ready_titles  # Independent task
        assert "Ready Dependent Task" in ready_titles  # Depends on completed task
        assert "Blocked Task" not in ready_titles  # Depends on incomplete task

    def test_update_status_with_progress(self, task_repository, sample_task_core):
        """Test updating task status with progress tracking."""
        task = task_repository.create(sample_task_core)
        task_repository.session.commit()

        # Update status with progress
        updated_task = task_repository.update_status_with_progress(
            task.id,
            TaskStatus.IN_PROGRESS,
            progress_percentage=30,
            notes="Started implementation phase",
            updated_by="test_user",
        )

        assert updated_task is not None
        assert updated_task.status == TaskStatus.IN_PROGRESS
        assert updated_task.updated_at is not None

        # Verify progress record was created
        task_repository.session.commit()
        progress_records = (
            task_repository.session.query(TaskProgress)
            .filter(TaskProgress.task_id == task.id)
            .all()
        )

        assert len(progress_records) >= 1
        progress_record = next(
            (p for p in progress_records if p.progress_percentage == 30), None
        )
        assert progress_record is not None
        assert progress_record.notes == "Started implementation phase"
        assert progress_record.updated_by == "test_user"

    def test_create_task_with_dependencies(self, task_repository, sample_task_core):
        """Test creating task with dependencies in single transaction."""
        # Create prerequisite tasks
        prereq1 = task_repository.create(sample_task_core)

        prereq2_core = sample_task_core.model_copy()
        prereq2_core.title = "Prerequisite 2"
        prereq2 = task_repository.create(prereq2_core)

        task_repository.session.commit()

        # Create task with dependencies
        dependent_core = sample_task_core.model_copy()
        dependent_core.title = "Dependent Task"
        dependent_task = task_repository.create_task_with_dependencies(
            dependent_core, dependency_task_ids=[prereq1.id, prereq2.id]
        )

        task_repository.session.commit()

        # Verify task was created
        assert dependent_task.title == "Dependent Task"

        # Verify dependencies were created
        dependencies = (
            task_repository.session.query(TaskDependency)
            .filter(TaskDependency.task_id == dependent_task.id)
            .all()
        )

        assert len(dependencies) == 2
        dependency_ids = [dep.depends_on_task_id for dep in dependencies]
        assert prereq1.id in dependency_ids
        assert prereq2.id in dependency_ids

    def test_search_tasks(self, task_repository, sample_task_core):
        """Test searching tasks by title or description."""
        # Create tasks with different content
        task_repository.create(sample_task_core)

        database_core = sample_task_core.model_copy()
        database_core.title = "Database Migration"
        database_core.description = "Migrate authentication tables to new schema"
        task_repository.create(database_core)

        ui_core = sample_task_core.model_copy()
        ui_core.title = "User Interface Updates"
        ui_core.description = "Update login forms and user dashboard"
        task_repository.create(ui_core)

        task_repository.session.commit()

        # Search by title term
        auth_results = task_repository.search_tasks("Authentication")
        assert len(auth_results) == 1
        assert auth_results[0].title == "Test Authentication System"

        # Search by description term
        jwt_results = task_repository.search_tasks("JWT")
        assert len(jwt_results) == 1
        assert jwt_results[0].description.find("JWT") != -1

        # Search by common term in multiple tasks
        user_results = task_repository.search_tasks("user")
        assert len(user_results) >= 1  # Should find tasks with "user" in description

    def test_get_task_statistics(self, task_repository, sample_task_core):
        """Test comprehensive task statistics."""
        # Create tasks with different attributes
        completed_core = sample_task_core.model_copy()
        completed_core.title = "Completed Task"
        completed_core.status = TaskStatus.COMPLETED
        completed_core.time_estimate_hours = 5.0
        task_repository.create(completed_core)

        in_progress_core = sample_task_core.model_copy()
        in_progress_core.title = "In Progress Task"
        in_progress_core.status = TaskStatus.IN_PROGRESS
        in_progress_core.phase = 2
        in_progress_core.component_area = ComponentArea.UI
        in_progress_core.time_estimate_hours = 10.0
        task_repository.create(in_progress_core)

        task_repository.create(sample_task_core)  # 8.0 hours

        task_repository.session.commit()

        # Get statistics
        stats = task_repository.get_task_statistics()

        assert stats["total_tasks"] == 3

        # Status breakdown
        assert stats["status_breakdown"]["completed"] == 1
        assert stats["status_breakdown"]["in_progress"] == 1
        assert stats["status_breakdown"]["not_started"] == 1

        # Completion percentage
        assert stats["completion_percentage"] == pytest.approx(33.3, rel=1e-1)

        # Phase breakdown
        phase_counts = {
            item["phase"]: item["count"] for item in stats["phase_breakdown"]
        }
        assert phase_counts[1] == 2  # completed + not_started
        assert phase_counts[2] == 1  # in_progress

        # Component breakdown
        component_counts = {
            item["area"]: item["count"] for item in stats["component_breakdown"]
        }
        assert component_counts["security"] == 2
        assert component_counts["ui"] == 1

        # Time estimates
        assert stats["total_estimated_hours"] == 23.0  # 5 + 10 + 8
        assert stats["completed_hours"] == 5.0
        assert stats["progress_percentage"] == pytest.approx(21.7, rel=1e-1)

    def test_get_dependencies(self, task_repository, sample_task_core):
        """Test getting dependencies for a task."""
        # Create tasks
        main_task = task_repository.create(sample_task_core)

        dep1_core = sample_task_core.model_copy()
        dep1_core.title = "Dependency 1"
        dep1_task = task_repository.create(dep1_core)

        dep2_core = sample_task_core.model_copy()
        dep2_core.title = "Dependency 2"
        dep2_task = task_repository.create(dep2_core)

        task_repository.session.commit()

        # Create dependencies
        dependency1 = TaskDependency(
            task_id=main_task.id,
            depends_on_task_id=dep1_task.id,
            dependency_type="blocks",
        )
        dependency2 = TaskDependency(
            task_id=main_task.id,
            depends_on_task_id=dep2_task.id,
            dependency_type="requires",
        )

        task_repository.session.add(dependency1)
        task_repository.session.add(dependency2)
        task_repository.session.commit()

        # Get dependencies
        dependencies = task_repository.get_dependencies(main_task.id)
        assert len(dependencies) == 2

        dependency_types = [dep.dependency_type for dep in dependencies]
        assert "blocks" in dependency_types
        assert "requires" in dependency_types

    def test_add_dependency(self, task_repository, sample_task_core):
        """Test adding dependency between tasks."""
        # Create tasks
        task1 = task_repository.create(sample_task_core)

        task2_core = sample_task_core.model_copy()
        task2_core.title = "Task 2"
        task2 = task_repository.create(task2_core)

        task_repository.session.commit()

        # Add dependency
        dependency = task_repository.add_dependency(task1.id, task2.id, "enhances")

        assert dependency.task_id == task1.id
        assert dependency.depends_on_task_id == task2.id
        assert dependency.dependency_type == "enhances"
        assert dependency.id is not None


class TestTaskExecutionRepository(TestRepositoryFixtures):
    """Test TaskExecutionRepository functionality."""

    @pytest.fixture
    def task_execution_repository(self, test_session):
        """Create TaskExecutionRepository instance."""
        return TaskExecutionRepository(test_session)

    @pytest.fixture
    def sample_task_for_execution(self, test_session, sample_task_core):
        """Create a task for execution testing."""
        task = Task.from_core_model(sample_task_core)
        test_session.add(task)
        test_session.commit()
        return task

    def test_repository_classes(self, task_execution_repository):
        """Test repository class definitions."""
        assert task_execution_repository.get_entity_class() == TaskExecutionLog
        assert task_execution_repository.get_business_class() == AgentReport

    def test_log_execution_start(
        self, task_execution_repository, sample_task_for_execution
    ):
        """Test logging execution start."""
        execution_id = uuid4()

        log = task_execution_repository.log_execution_start(
            task_id=sample_task_for_execution.id,
            agent_type=AgentType.CODING,
            execution_id=execution_id,
        )

        assert isinstance(log, TaskExecutionLog)
        assert log.task_id == sample_task_for_execution.id
        assert log.execution_id == execution_id
        assert log.agent_type == AgentType.CODING
        assert log.status == TaskStatus.IN_PROGRESS
        assert log.start_time is not None
        assert log.end_time is None

    def test_log_execution_start_auto_uuid(
        self, task_execution_repository, sample_task_for_execution
    ):
        """Test logging execution start with auto-generated UUID."""
        log = task_execution_repository.log_execution_start(
            task_id=sample_task_for_execution.id, agent_type=AgentType.TESTING
        )

        assert isinstance(log.execution_id, UUID)
        assert log.agent_type == AgentType.TESTING

    def test_log_execution_complete_success(
        self, task_execution_repository, sample_task_for_execution
    ):
        """Test logging successful execution completion."""
        # Start execution
        execution_id = uuid4()
        task_execution_repository.log_execution_start(
            task_id=sample_task_for_execution.id,
            agent_type=AgentType.CODING,
            execution_id=execution_id,
        )
        task_execution_repository.session.commit()

        # Complete execution
        outputs = {"result": "success", "files_created": ["auth.py", "test_auth.py"]}
        completion_log = task_execution_repository.log_execution_complete(
            execution_id=execution_id,
            status=TaskStatus.COMPLETED,
            outputs=outputs,
            confidence_score=0.95,
        )

        assert completion_log is not None
        assert completion_log.status == TaskStatus.COMPLETED
        assert completion_log.end_time is not None
        assert completion_log.outputs == outputs
        assert completion_log.confidence_score == 0.95
        assert completion_log.error_details is None

    def test_log_execution_complete_failure(
        self, task_execution_repository, sample_task_for_execution
    ):
        """Test logging failed execution completion."""
        # Start execution
        execution_id = uuid4()
        task_execution_repository.log_execution_start(
            task_id=sample_task_for_execution.id,
            agent_type=AgentType.CODING,
            execution_id=execution_id,
        )
        task_execution_repository.session.commit()

        # Complete with failure
        error_details = "Import error: module 'jwt' not found"
        completion_log = task_execution_repository.log_execution_complete(
            execution_id=execution_id,
            status=TaskStatus.FAILED,
            error_details=error_details,
            confidence_score=0.3,
        )

        assert completion_log.status == TaskStatus.FAILED
        assert completion_log.error_details == error_details
        assert completion_log.confidence_score == 0.3

    def test_log_execution_complete_non_existent(self, task_execution_repository):
        """Test completing non-existent execution returns None."""
        non_existent_id = uuid4()
        result = task_execution_repository.log_execution_complete(
            execution_id=non_existent_id, status=TaskStatus.COMPLETED
        )

        assert result is None

    def test_get_execution_history(
        self, task_execution_repository, sample_task_for_execution
    ):
        """Test getting execution history for a task."""
        # Create multiple execution logs
        execution_ids = [uuid4() for _ in range(3)]
        agent_types = [AgentType.CODING, AgentType.TESTING, AgentType.DOCUMENTATION]

        for i, (exec_id, agent_type) in enumerate(
            zip(execution_ids, agent_types, strict=False)
        ):
            task_execution_repository.log_execution_start(
                task_id=sample_task_for_execution.id,
                agent_type=agent_type,
                execution_id=exec_id,
            )

            # Complete some executions
            if i < 2:
                task_execution_repository.log_execution_complete(
                    execution_id=exec_id,
                    status=TaskStatus.COMPLETED if i == 0 else TaskStatus.FAILED,
                    confidence_score=0.9 if i == 0 else 0.4,
                )

        task_execution_repository.session.commit()

        # Get execution history
        history = task_execution_repository.get_execution_history(
            sample_task_for_execution.id
        )

        assert len(history) == 3
        # Should be ordered by start_time DESC (most recent first)
        assert history[0].agent_type == AgentType.DOCUMENTATION  # Last created
        assert history[1].agent_type == AgentType.TESTING
        assert history[2].agent_type == AgentType.CODING

    def test_get_agent_performance_stats(
        self, task_execution_repository, sample_task_for_execution
    ):
        """Test getting performance statistics for specific agent type."""
        # Create execution logs for coding agent
        coding_executions = []
        for i in range(5):
            exec_id = uuid4()
            log = task_execution_repository.log_execution_start(
                task_id=sample_task_for_execution.id,
                agent_type=AgentType.CODING,
                execution_id=exec_id,
            )

            # Complete with varying success rates
            status = TaskStatus.COMPLETED if i < 3 else TaskStatus.FAILED
            confidence = 0.9 if i < 3 else 0.3

            task_execution_repository.log_execution_complete(
                execution_id=exec_id, status=status, confidence_score=confidence
            )

            coding_executions.append(log)

        # Create execution logs for testing agent
        testing_exec_id = uuid4()
        task_execution_repository.log_execution_start(
            task_id=sample_task_for_execution.id,
            agent_type=AgentType.TESTING,
            execution_id=testing_exec_id,
        )
        task_execution_repository.log_execution_complete(
            execution_id=testing_exec_id,
            status=TaskStatus.COMPLETED,
            confidence_score=0.85,
        )

        task_execution_repository.session.commit()

        # Get performance stats for coding agent
        coding_stats = task_execution_repository.get_agent_performance_stats(
            AgentType.CODING
        )

        assert coding_stats["agent_type"] == "coding"
        assert coding_stats["total_executions"] == 5
        assert coding_stats["successful_executions"] == 3
        assert coding_stats["success_rate"] == 60.0  # 3/5 * 100
        assert coding_stats["average_confidence_score"] == 0.66  # (0.9*3 + 0.3*2) / 5

        # Get performance stats for testing agent
        testing_stats = task_execution_repository.get_agent_performance_stats(
            AgentType.TESTING
        )

        assert testing_stats["total_executions"] == 1
        assert testing_stats["successful_executions"] == 1
        assert testing_stats["success_rate"] == 100.0
        assert testing_stats["average_confidence_score"] == 0.85

    def test_get_recent_executions(
        self, task_execution_repository, sample_task_for_execution
    ):
        """Test getting recent execution logs."""
        # Create multiple execution logs
        execution_logs = []
        for _i in range(15):  # More than default limit
            exec_id = uuid4()
            log = task_execution_repository.log_execution_start(
                task_id=sample_task_for_execution.id,
                agent_type=AgentType.CODING,
                execution_id=exec_id,
            )
            execution_logs.append(log)

        task_execution_repository.session.commit()

        # Get recent executions (default limit is 10)
        recent_executions = task_execution_repository.get_recent_executions()
        assert len(recent_executions) == 10

        # Should be ordered by start_time DESC (most recent first)
        for i in range(len(recent_executions) - 1):
            assert (
                recent_executions[i].start_time >= recent_executions[i + 1].start_time
            )

        # Test custom limit
        recent_five = task_execution_repository.get_recent_executions(limit=5)
        assert len(recent_five) == 5


class TestRepositoryIntegration(TestRepositoryFixtures):
    """Integration tests combining repository functionality."""

    @pytest.fixture
    def task_repository(self, test_session):
        """Create TaskRepository instance."""
        return TaskRepository(test_session)

    @pytest.fixture
    def execution_repository(self, test_session):
        """Create TaskExecutionRepository instance."""
        return TaskExecutionRepository(test_session)

    def test_complete_task_execution_workflow(
        self,
        task_repository,
        execution_repository,
        sample_task_core,
        sample_agent_report,
    ):
        """Test complete workflow from task creation to execution completion."""
        # 1. Create task
        task = task_repository.create(sample_task_core)
        task_repository.session.commit()

        # 2. Start execution
        execution_id = uuid4()
        execution_repository.log_execution_start(
            task_id=task.id, agent_type=AgentType.CODING, execution_id=execution_id
        )

        # 3. Update task status to in_progress
        task_repository.update_status_with_progress(
            task.id,
            TaskStatus.IN_PROGRESS,
            progress_percentage=25,
            notes="Started implementation",
            updated_by="coding_agent",
        )

        # 4. Complete execution successfully
        outputs = {"files_created": ["auth.py", "models.py"], "tests_passed": 15}
        execution_repository.log_execution_complete(
            execution_id=execution_id,
            status=TaskStatus.COMPLETED,
            outputs=outputs,
            confidence_score=0.92,
        )

        # 5. Update task status to completed
        task_repository.update_status_with_progress(
            task.id,
            TaskStatus.COMPLETED,
            progress_percentage=100,
            notes="Implementation completed successfully",
            updated_by="coding_agent",
        )

        task_repository.session.commit()

        # Verify final state
        final_task = task_repository.get_by_id(task.id)
        assert final_task.status == TaskStatus.COMPLETED

        execution_history = execution_repository.get_execution_history(task.id)
        assert len(execution_history) == 1
        assert execution_history[0].status == TaskStatus.COMPLETED
        assert execution_history[0].outputs == outputs

    def test_dependency_resolution_workflow(self, task_repository, sample_task_core):
        """Test workflow with task dependencies."""
        # Create prerequisite tasks
        prereq1_core = sample_task_core.model_copy()
        prereq1_core.title = "Setup Database"
        prereq1_core.status = TaskStatus.COMPLETED
        prereq1 = task_repository.create(prereq1_core)

        prereq2_core = sample_task_core.model_copy()
        prereq2_core.title = "Install Dependencies"
        prereq2_core.status = TaskStatus.NOT_STARTED
        prereq2 = task_repository.create(prereq2_core)

        # Create dependent task
        dependent_core = sample_task_core.model_copy()
        dependent_core.title = "Implement Authentication"
        task_repository.create_task_with_dependencies(
            dependent_core, dependency_task_ids=[prereq1.id, prereq2.id]
        )

        task_repository.session.commit()

        # Verify task is not ready (has incomplete dependencies)
        ready_tasks = task_repository.get_ready_tasks()
        ready_titles = [task.title for task in ready_tasks]
        assert "Implement Authentication" not in ready_titles

        # Complete second prerequisite
        task_repository.update_status_with_progress(
            prereq2.id,
            TaskStatus.COMPLETED,
            progress_percentage=100,
            notes="Dependencies installed",
        )

        task_repository.session.commit()

        # Verify task is now ready
        ready_tasks_updated = task_repository.get_ready_tasks()
        ready_titles_updated = [task.title for task in ready_tasks_updated]
        assert "Implement Authentication" in ready_titles_updated

    def test_analytics_across_repositories(
        self, task_repository, execution_repository, sample_task_core
    ):
        """Test analytics combining data from multiple repositories."""
        # Create tasks with various statuses
        tasks_data = [
            ("Completed Task 1", TaskStatus.COMPLETED, AgentType.CODING),
            ("Completed Task 2", TaskStatus.COMPLETED, AgentType.TESTING),
            ("Failed Task", TaskStatus.FAILED, AgentType.CODING),
            ("In Progress Task", TaskStatus.IN_PROGRESS, AgentType.DOCUMENTATION),
            ("Not Started Task", TaskStatus.NOT_STARTED, None),
        ]

        created_tasks = []
        for title, status, agent_type in tasks_data:
            task_core = sample_task_core.model_copy()
            task_core.title = title
            task_core.status = status
            task = task_repository.create(task_core)
            created_tasks.append((task, agent_type))

        task_repository.session.commit()

        # Create execution logs for tasks with agent types
        for task, agent_type in created_tasks:
            if agent_type:
                exec_id = uuid4()
                execution_repository.log_execution_start(
                    task_id=task.id, agent_type=agent_type, execution_id=exec_id
                )
                execution_repository.log_execution_complete(
                    execution_id=exec_id,
                    status=task.status,
                    confidence_score=0.9
                    if task.status == TaskStatus.COMPLETED
                    else 0.4,
                )

        execution_repository.session.commit()

        # Get task statistics
        task_stats = task_repository.get_task_statistics()
        assert task_stats["total_tasks"] == 5
        assert task_stats["status_breakdown"]["completed"] == 2
        assert task_stats["status_breakdown"]["failed"] == 1
        assert task_stats["status_breakdown"]["in_progress"] == 1
        assert task_stats["status_breakdown"]["not_started"] == 1
        assert task_stats["completion_percentage"] == 40.0  # 2/5 * 100

        # Get agent performance statistics
        coding_stats = execution_repository.get_agent_performance_stats(
            AgentType.CODING
        )
        assert coding_stats["total_executions"] == 2  # Completed Task 1 + Failed Task
        assert coding_stats["successful_executions"] == 1  # Only Completed Task 1
        assert coding_stats["success_rate"] == 50.0

        testing_stats = execution_repository.get_agent_performance_stats(
            AgentType.TESTING
        )
        assert testing_stats["total_executions"] == 1
        assert testing_stats["successful_executions"] == 1
        assert testing_stats["success_rate"] == 100.0


class TestRepositoryErrorHandling(TestRepositoryFixtures):
    """Test error handling and edge cases in repositories."""

    @pytest.fixture
    def task_repository(self, test_session):
        """Create TaskRepository instance."""
        return TaskRepository(test_session)

    def test_update_status_non_existent_task(self, task_repository):
        """Test updating status of non-existent task."""
        result = task_repository.update_status_with_progress(
            task_id=999999, status=TaskStatus.COMPLETED, notes="Non-existent task"
        )

        assert result is None

    def test_create_dependency_invalid_task_ids(
        self, task_repository, sample_task_core
    ):
        """Test creating dependency with invalid task IDs."""
        task = task_repository.create(sample_task_core)
        task_repository.session.commit()

        # This should not raise an error at repository level (business logic handles it)
        dependency = task_repository.add_dependency(
            task_id=task.id,
            depends_on_task_id=999999,  # Non-existent
            dependency_type="blocks",
        )

        # Dependency is created but will fail at database level due to foreign
        # key constraint
        assert dependency.task_id == task.id
        assert dependency.depends_on_task_id == 999999

    def test_get_statistics_empty_database(self, task_repository):
        """Test getting statistics from empty database."""
        stats = task_repository.get_task_statistics()

        assert stats["total_tasks"] == 0
        assert stats["status_breakdown"] == {
            "not_started": 0,
            "in_progress": 0,
            "completed": 0,
            "blocked": 0,
            "failed": 0,
            "requires_assistance": 0,
            "partial": 0,
        }
        assert stats["completion_percentage"] == 0
        assert stats["total_estimated_hours"] == 0
        assert stats["completed_hours"] == 0
        assert stats["progress_percentage"] == 0

    def test_search_empty_results(self, task_repository, sample_task_core):
        """Test searching with no matching results."""
        task_repository.create(sample_task_core)
        task_repository.session.commit()

        results = task_repository.search_tasks("nonexistent_term_xyz123")
        assert len(results) == 0


class TestRepositoryTransactionHandling(TestRepositoryFixtures):
    """Test transaction handling and rollback scenarios."""

    @pytest.fixture
    def task_repository(self, test_session):
        """Create TaskRepository instance."""
        return TaskRepository(test_session)

    def test_transaction_rollback_on_error(self, task_repository, sample_task_core):
        """Test transaction rollback when error occurs."""
        initial_count = task_repository.count()

        try:
            # Start a transaction that will fail
            task = task_repository.create(sample_task_core)

            # Force an error by trying to create invalid dependency
            # This would fail at database level in a real scenario
            task_repository.session.add(
                TaskDependency(
                    task_id=task.id,
                    depends_on_task_id=999999,  # Non-existent task
                    dependency_type="blocks",
                )
            )

            # This would cause a foreign key constraint error in production
            # For this test, we'll manually raise an error
            if True:  # Simulate constraint error
                raise Exception("Simulated database constraint error")

            task_repository.session.commit()

        except Exception:
            task_repository.session.rollback()

        # Verify no tasks were created due to rollback
        final_count = task_repository.count()
        assert final_count == initial_count

    def test_partial_transaction_success(self, task_repository, sample_task_core):
        """Test partial transaction with some operations succeeding."""
        # Create a task successfully
        task_repository.create(sample_task_core)
        task_repository.session.commit()

        initial_count = task_repository.count()
        assert initial_count == 1

        # Try to create another task with valid operation first, then invalid
        try:
            task2_core = sample_task_core.model_copy()
            task2_core.title = "Second Task"
            task_repository.create(task2_core)

            # This part succeeds
            task_repository.session.flush()

            # Now try something that would fail
            # Simulate an error condition
            if True:
                raise Exception("Simulated processing error")

            task_repository.session.commit()

        except Exception:
            task_repository.session.rollback()

        # The second task should not be committed due to rollback
        final_count = task_repository.count()
        assert final_count == 1  # Only the first task remains

"""Comprehensive tests for database.py SQLModel entities.

This module tests all SQLModel entities, relationships, constraints,
conversions, and database operations.
"""

from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, create_engine

from src.schemas.database import (
    Task,
    TaskComment,
    TaskCommentTable,
    TaskDependency,
    TaskDependencyTable,
    TaskExecutionLog,
    TaskExecutionLogTable,
    TaskProgress,
    TaskProgressTable,
    # Legacy aliases
    TaskTable,
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


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(in_memory_db):
    """Create database session for testing."""
    with Session(in_memory_db) as session:
        yield session


class TestTaskEntity:
    """Test Task SQLModel entity."""

    def test_task_creation_with_defaults(self, db_session):
        """Test creating Task with default values."""
        task = Task(title="Test Task")

        # Check defaults before saving
        assert task.id is None
        assert task.title == "Test Task"
        assert task.description == ""
        assert task.component_area == ComponentArea.TASK
        assert task.phase == 1
        assert task.priority == TaskPriority.MEDIUM
        assert task.complexity == TaskComplexity.MEDIUM
        assert task.status == TaskStatus.NOT_STARTED
        assert task.source_document == ""
        assert task.success_criteria == ""
        assert task.time_estimate_hours == 1.0
        assert task.parent_task_id is None
        assert isinstance(task.uuid, UUID)
        assert isinstance(task.created_at, datetime)
        assert isinstance(task.updated_at, datetime)

        # Save to database
        db_session.add(task)
        db_session.commit()
        db_session.refresh(task)

        # Check persisted values
        assert task.id is not None
        assert task.uuid is not None

    def test_task_creation_full_fields(self, db_session):
        """Test creating Task with all fields specified."""
        custom_uuid = uuid4()
        task = Task(
            uuid=custom_uuid,
            title="Full Task",
            description="Complete task description",
            component_area=ComponentArea.SECURITY,
            phase=3,
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.VERY_HIGH,
            status=TaskStatus.IN_PROGRESS,
            source_document="requirements.md",
            success_criteria="All security tests pass",
            time_estimate_hours=12.5,
            parent_task_id=None,
        )

        db_session.add(task)
        db_session.commit()
        db_session.refresh(task)

        assert task.uuid == custom_uuid
        assert task.title == "Full Task"
        assert task.description == "Complete task description"
        assert task.component_area == ComponentArea.SECURITY
        assert task.phase == 3
        assert task.priority == TaskPriority.CRITICAL
        assert task.complexity == TaskComplexity.VERY_HIGH
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.time_estimate_hours == 12.5

    def test_task_title_validation(self):
        """Test Task title validation constraints."""
        # Title too short
        with pytest.raises(ValidationError):
            Task(title="")

        # Title too long
        with pytest.raises(ValidationError):
            Task(title="x" * 201)

        # Valid title lengths
        task_min = Task(title="x")
        assert task_min.title == "x"

        task_max = Task(title="x" * 200)
        assert len(task_max.title) == 200

    def test_task_description_length_limit(self):
        """Test Task description length constraint."""
        # Valid description
        task = Task(title="Test", description="x" * 2000)
        assert len(task.description) == 2000

        # Description too long should be handled by database constraint
        with pytest.raises(ValidationError):
            Task(title="Test", description="x" * 2001)

    def test_task_phase_validation(self):
        """Test Task phase validation constraints."""
        # Phase too low
        with pytest.raises(ValidationError):
            Task(title="Test", phase=0)

        # Phase too high
        with pytest.raises(ValidationError):
            Task(title="Test", phase=11)

        # Valid phases
        task_min = Task(title="Test", phase=1)
        assert task_min.phase == 1

        task_max = Task(title="Test", phase=10)
        assert task_max.phase == 10

    def test_task_time_estimate_validation(self):
        """Test Task time estimate validation constraints."""
        # Time too low
        with pytest.raises(ValidationError):
            Task(title="Test", time_estimate_hours=0.05)

        # Time too high
        with pytest.raises(ValidationError):
            Task(title="Test", time_estimate_hours=101.0)

        # Valid time estimates
        task_min = Task(title="Test", time_estimate_hours=0.1)
        assert task_min.time_estimate_hours == 0.1

        task_max = Task(title="Test", time_estimate_hours=100.0)
        assert task_max.time_estimate_hours == 100.0

    def test_task_parent_relationship(self, db_session):
        """Test Task parent-child relationship."""
        # Create parent task
        parent_task = Task(title="Parent Task")
        db_session.add(parent_task)
        db_session.commit()
        db_session.refresh(parent_task)

        # Create child task
        child_task = Task(title="Child Task", parent_task_id=parent_task.id)
        db_session.add(child_task)
        db_session.commit()
        db_session.refresh(child_task)

        # Test relationships
        assert child_task.parent_task_id == parent_task.id
        assert child_task.parent_task.id == parent_task.id
        assert len(parent_task.subtasks) == 1
        assert parent_task.subtasks[0].id == child_task.id

    def test_task_to_core_model_conversion(self, db_session):
        """Test Task to TaskCore conversion."""
        task = Task(
            title="Conversion Test",
            description="Test conversion",
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            status=TaskStatus.IN_PROGRESS,
        )

        db_session.add(task)
        db_session.commit()
        db_session.refresh(task)

        # Convert to core model
        core_model = task.to_core_model()

        assert isinstance(core_model, TaskCore)
        assert core_model.id == task.id
        assert core_model.title == task.title
        assert core_model.description == task.description
        assert core_model.priority == task.priority
        assert core_model.complexity == task.complexity
        assert core_model.status == task.status

    def test_task_from_core_model_creation(self):
        """Test Task creation from TaskCore."""
        core_model = TaskCore(
            title="From Core",
            description="Core model conversion",
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.HIGH,
        )

        # Convert to entity
        task = Task.from_core_model(core_model)

        assert isinstance(task, Task)
        assert task.title == core_model.title
        assert task.description == core_model.description
        assert task.priority == core_model.priority
        assert task.complexity == core_model.complexity
        assert task.status == core_model.status

    def test_task_update_from_core_model(self, db_session):
        """Test updating Task from TaskCore."""
        # Create original task
        task = Task(title="Original", priority=TaskPriority.LOW)
        db_session.add(task)
        db_session.commit()
        db_session.refresh(task)

        original_created_at = task.created_at
        original_id = task.id

        # Create updated core model
        updated_core = TaskCore(
            id=task.id,
            title="Updated Title",
            description="Updated description",
            priority=TaskPriority.HIGH,
            status=TaskStatus.COMPLETED,
        )

        # Update task from core model
        task.update_from_core_model(updated_core)

        # Verify updates
        assert task.id == original_id  # ID should not change
        assert task.title == "Updated Title"
        assert task.description == "Updated description"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.COMPLETED
        assert task.created_at == original_created_at  # Created at unchanged
        assert task.updated_at > original_created_at  # Updated at changed

    def test_task_uuid_uniqueness_constraint(self, db_session):
        """Test Task UUID uniqueness constraint."""
        shared_uuid = uuid4()

        task1 = Task(title="Task 1", uuid=shared_uuid)
        db_session.add(task1)
        db_session.commit()

        # Attempt to create second task with same UUID
        task2 = Task(title="Task 2", uuid=shared_uuid)
        db_session.add(task2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_task_indexes_exist(self, in_memory_db):
        """Test that expected indexes exist on Task table."""
        # This test verifies indexes are defined in the table metadata
        task_table = Task.__table__
        index_names = {idx.name for idx in task_table.indexes}

        expected_indexes = {
            "ix_tasks_component_area",
            "ix_tasks_phase",
            "ix_tasks_status",
            "ix_tasks_priority",
            "ix_tasks_uuid",
        }

        assert expected_indexes.issubset(index_names)


class TestTaskDependency:
    """Test TaskDependency SQLModel entity."""

    def test_task_dependency_creation(self, db_session):
        """Test TaskDependency creation."""
        # Create tasks
        task1 = Task(title="Task 1")
        task2 = Task(title="Task 2")
        db_session.add_all([task1, task2])
        db_session.commit()

        # Create dependency
        dependency = TaskDependency(
            task_id=task1.id, depends_on_task_id=task2.id, dependency_type="blocks"
        )

        db_session.add(dependency)
        db_session.commit()
        db_session.refresh(dependency)

        assert dependency.task_id == task1.id
        assert dependency.depends_on_task_id == task2.id
        assert dependency.dependency_type == "blocks"

    def test_task_dependency_relationships(self, db_session):
        """Test TaskDependency relationships with tasks."""
        # Create tasks
        dependent_task = Task(title="Dependent Task")
        blocking_task = Task(title="Blocking Task")
        db_session.add_all([dependent_task, blocking_task])
        db_session.commit()

        # Create dependency
        dependency = TaskDependency(
            task_id=dependent_task.id, depends_on_task_id=blocking_task.id
        )
        db_session.add(dependency)
        db_session.commit()

        # Test forward relationships
        assert dependency.task.id == dependent_task.id
        assert dependency.depends_on_task.id == blocking_task.id

        # Test reverse relationships
        assert len(dependent_task.dependencies) == 1
        assert dependent_task.dependencies[0].depends_on_task_id == blocking_task.id

        assert len(blocking_task.dependents) == 1
        assert blocking_task.dependents[0].task_id == dependent_task.id

    def test_task_dependency_uniqueness_constraint(self, db_session):
        """Test TaskDependency uniqueness constraint."""
        task1 = Task(title="Task 1")
        task2 = Task(title="Task 2")
        db_session.add_all([task1, task2])
        db_session.commit()

        # Create first dependency
        dep1 = TaskDependency(task_id=task1.id, depends_on_task_id=task2.id)
        db_session.add(dep1)
        db_session.commit()

        # Try to create duplicate dependency
        dep2 = TaskDependency(task_id=task1.id, depends_on_task_id=task2.id)
        db_session.add(dep2)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_task_dependency_self_reference_constraint(self, db_session):
        """Test TaskDependency prevents self-referencing dependencies."""
        task = Task(title="Self Task")
        db_session.add(task)
        db_session.commit()

        # Try to create self-dependency
        dep = TaskDependency(task_id=task.id, depends_on_task_id=task.id)
        db_session.add(dep)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_task_dependency_foreign_key_constraints(self, db_session):
        """Test TaskDependency foreign key constraints."""
        # Try to create dependency with invalid task IDs
        dep = TaskDependency(task_id=999, depends_on_task_id=888)
        db_session.add(dep)

        with pytest.raises(IntegrityError):
            db_session.commit()


class TestTaskProgress:
    """Test TaskProgress SQLModel entity."""

    def test_task_progress_creation(self, db_session):
        """Test TaskProgress creation."""
        task = Task(title="Progress Task")
        db_session.add(task)
        db_session.commit()

        progress = TaskProgress(
            task_id=task.id,
            progress_percentage=75,
            notes="Making good progress",
            updated_by="test_user",
        )

        db_session.add(progress)
        db_session.commit()
        db_session.refresh(progress)

        assert progress.task_id == task.id
        assert progress.progress_percentage == 75
        assert progress.notes == "Making good progress"
        assert progress.updated_by == "test_user"
        assert isinstance(progress.created_at, datetime)

    def test_task_progress_defaults(self, db_session):
        """Test TaskProgress default values."""
        task = Task(title="Default Progress Task")
        db_session.add(task)
        db_session.commit()

        progress = TaskProgress(task_id=task.id)

        assert progress.progress_percentage == 0
        assert progress.notes == ""
        assert progress.updated_by == "system"

    def test_task_progress_percentage_validation(self):
        """Test TaskProgress percentage validation."""
        # Valid percentage
        progress = TaskProgress(task_id=1, progress_percentage=50)
        assert progress.progress_percentage == 50

        # Invalid percentages
        with pytest.raises(ValidationError):
            TaskProgress(task_id=1, progress_percentage=-1)

        with pytest.raises(ValidationError):
            TaskProgress(task_id=1, progress_percentage=101)

    def test_task_progress_relationship(self, db_session):
        """Test TaskProgress relationship with Task."""
        task = Task(title="Task with Progress")
        db_session.add(task)
        db_session.commit()

        # Create multiple progress records
        progress1 = TaskProgress(task_id=task.id, progress_percentage=25)
        progress2 = TaskProgress(task_id=task.id, progress_percentage=50)
        db_session.add_all([progress1, progress2])
        db_session.commit()

        # Test relationship
        assert len(task.progress_records) == 2
        assert task.progress_records[0].task_id == task.id
        assert task.progress_records[1].task_id == task.id

    def test_task_progress_notes_length(self):
        """Test TaskProgress notes length constraint."""
        # Valid notes
        progress = TaskProgress(task_id=1, notes="x" * 1000)
        assert len(progress.notes) == 1000

        # Notes too long
        with pytest.raises(ValidationError):
            TaskProgress(task_id=1, notes="x" * 1001)


class TestTaskExecutionLog:
    """Test TaskExecutionLog SQLModel entity."""

    def test_task_execution_log_creation(self, db_session):
        """Test TaskExecutionLog creation."""
        task = Task(title="Execution Task")
        db_session.add(task)
        db_session.commit()

        execution_id = uuid4()
        start_time = datetime.now()

        log = TaskExecutionLog(
            task_id=task.id,
            execution_id=execution_id,
            agent_type=AgentType.CODING,
            status=TaskStatus.IN_PROGRESS,
            start_time=start_time,
            outputs={"step": "initialization"},
            confidence_score=0.85,
        )

        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)

        assert log.task_id == task.id
        assert log.execution_id == execution_id
        assert log.agent_type == AgentType.CODING
        assert log.status == TaskStatus.IN_PROGRESS
        assert log.start_time == start_time
        assert log.outputs == {"step": "initialization"}
        assert log.confidence_score == 0.85
        assert log.end_time is None
        assert log.error_details is None

    def test_task_execution_log_completed(self, db_session):
        """Test TaskExecutionLog for completed execution."""
        task = Task(title="Completed Task")
        db_session.add(task)
        db_session.commit()

        start_time = datetime.now()
        end_time = datetime.now()

        log = TaskExecutionLog(
            task_id=task.id,
            agent_type=AgentType.TESTING,
            status=TaskStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            outputs={"tests_passed": 15, "tests_failed": 0},
            confidence_score=0.95,
        )

        db_session.add(log)
        db_session.commit()

        assert log.status == TaskStatus.COMPLETED
        assert log.end_time == end_time
        assert log.outputs["tests_passed"] == 15

    def test_task_execution_log_failed(self, db_session):
        """Test TaskExecutionLog for failed execution."""
        task = Task(title="Failed Task")
        db_session.add(task)
        db_session.commit()

        log = TaskExecutionLog(
            task_id=task.id,
            agent_type=AgentType.DOCUMENTATION,
            status=TaskStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            error_details=(
                "Failed to generate documentation due to missing source files"
            ),
            confidence_score=0.2,
        )

        db_session.add(log)
        db_session.commit()

        assert log.status == TaskStatus.FAILED
        assert "missing source files" in log.error_details
        assert log.confidence_score == 0.2

    def test_task_execution_log_relationship(self, db_session):
        """Test TaskExecutionLog relationship with Task."""
        task = Task(title="Task with Logs")
        db_session.add(task)
        db_session.commit()

        # Create multiple execution logs
        log1 = TaskExecutionLog(
            task_id=task.id,
            agent_type=AgentType.RESEARCH,
            status=TaskStatus.COMPLETED,
            start_time=datetime.now(),
        )
        log2 = TaskExecutionLog(
            task_id=task.id,
            agent_type=AgentType.CODING,
            status=TaskStatus.IN_PROGRESS,
            start_time=datetime.now(),
        )
        db_session.add_all([log1, log2])
        db_session.commit()

        # Test relationship
        assert len(task.execution_logs) == 2
        agent_types = {log.agent_type for log in task.execution_logs}
        assert agent_types == {AgentType.RESEARCH, AgentType.CODING}

    def test_task_execution_log_to_agent_report(self, db_session):
        """Test TaskExecutionLog to AgentReport conversion."""
        task = Task(title="Report Task")
        db_session.add(task)
        db_session.commit()

        execution_id = uuid4()
        log = TaskExecutionLog(
            task_id=task.id,
            execution_id=execution_id,
            agent_type=AgentType.TESTING,
            status=TaskStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            outputs={"coverage": 95},
            confidence_score=0.9,
        )

        db_session.add(log)
        db_session.commit()

        # Convert to agent report
        report = log.to_agent_report()

        assert isinstance(report, AgentReport)
        assert report.task_id == task.id
        assert report.agent_name == f"{AgentType.TESTING.value}_agent"
        assert report.status == TaskStatus.COMPLETED
        assert report.outputs == {"coverage": 95}
        assert report.confidence_score == 0.9

    def test_task_execution_log_from_agent_report(self, db_session):
        """Test TaskExecutionLog creation from AgentReport."""
        task = Task(title="From Report Task")
        db_session.add(task)
        db_session.commit()

        execution_id = uuid4()
        report = AgentReport(
            agent_name=AgentType.CODING,
            task_id=task.id,
            execution_id=execution_id,
            status=TaskStatus.PARTIAL,
            success=True,
            outputs={"lines_added": 150},
            confidence_score=0.8,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        # Convert to execution log
        log = TaskExecutionLog.from_agent_report(report)

        assert log.task_id == task.id
        assert log.execution_id == execution_id
        assert log.agent_type == AgentType.CODING
        assert log.status == TaskStatus.PARTIAL
        assert log.outputs == {"lines_added": 150}
        assert log.confidence_score == 0.8

    def test_task_execution_log_confidence_score_validation(self):
        """Test TaskExecutionLog confidence score validation."""
        # Valid confidence scores
        log = TaskExecutionLog(
            task_id=1,
            agent_type=AgentType.CODING,
            status=TaskStatus.COMPLETED,
            start_time=datetime.now(),
            confidence_score=0.5,
        )
        assert log.confidence_score == 0.5

        # Invalid confidence scores
        with pytest.raises(ValidationError):
            TaskExecutionLog(
                task_id=1,
                agent_type=AgentType.CODING,
                status=TaskStatus.COMPLETED,
                start_time=datetime.now(),
                confidence_score=1.5,
            )

        with pytest.raises(ValidationError):
            TaskExecutionLog(
                task_id=1,
                agent_type=AgentType.CODING,
                status=TaskStatus.COMPLETED,
                start_time=datetime.now(),
                confidence_score=-0.1,
            )

    def test_task_execution_log_execution_id_uniqueness(self, db_session):
        """Test TaskExecutionLog execution_id uniqueness."""
        task = Task(title="Unique Execution Task")
        db_session.add(task)
        db_session.commit()

        execution_id = uuid4()

        log1 = TaskExecutionLog(
            task_id=task.id,
            execution_id=execution_id,
            agent_type=AgentType.CODING,
            status=TaskStatus.IN_PROGRESS,
            start_time=datetime.now(),
        )
        db_session.add(log1)
        db_session.commit()

        # Try to create second log with same execution_id
        log2 = TaskExecutionLog(
            task_id=task.id,
            execution_id=execution_id,
            agent_type=AgentType.TESTING,
            status=TaskStatus.IN_PROGRESS,
            start_time=datetime.now(),
        )
        db_session.add(log2)

        with pytest.raises(IntegrityError):
            db_session.commit()


class TestTaskComment:
    """Test TaskComment SQLModel entity."""

    def test_task_comment_creation(self, db_session):
        """Test TaskComment creation."""
        task = Task(title="Comment Task")
        db_session.add(task)
        db_session.commit()

        comment = TaskComment(
            task_id=task.id,
            comment="This is a test comment",
            comment_type="note",
            created_by="test_user",
        )

        db_session.add(comment)
        db_session.commit()
        db_session.refresh(comment)

        assert comment.task_id == task.id
        assert comment.comment == "This is a test comment"
        assert comment.comment_type == "note"
        assert comment.created_by == "test_user"
        assert isinstance(comment.created_at, datetime)

    def test_task_comment_defaults(self, db_session):
        """Test TaskComment default values."""
        task = Task(title="Default Comment Task")
        db_session.add(task)
        db_session.commit()

        comment = TaskComment(task_id=task.id, comment="Default comment")

        assert comment.comment_type == "note"
        assert comment.created_by == "system"

    def test_task_comment_length_validation(self):
        """Test TaskComment length constraints."""
        # Valid comment
        comment = TaskComment(task_id=1, comment="x" * 2000)
        assert len(comment.comment) == 2000

        # Comment too long
        with pytest.raises(ValidationError):
            TaskComment(task_id=1, comment="x" * 2001)

        # Valid comment type
        comment = TaskComment(task_id=1, comment="test", comment_type="x" * 50)
        assert len(comment.comment_type) == 50

        # Comment type too long
        with pytest.raises(ValidationError):
            TaskComment(task_id=1, comment="test", comment_type="x" * 51)


class TestDatabaseIntegration:
    """Test database integration scenarios."""

    def test_complete_task_workflow(self, db_session):
        """Test complete task workflow with all related entities."""
        # Create main task
        main_task = Task(
            title="Complete Workflow Task",
            description="Test complete workflow",
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
        )
        db_session.add(main_task)
        db_session.commit()

        # Create dependency task
        dependency_task = Task(title="Dependency Task")
        db_session.add(dependency_task)
        db_session.commit()

        # Create dependency relationship
        dependency = TaskDependency(
            task_id=main_task.id, depends_on_task_id=dependency_task.id
        )
        db_session.add(dependency)

        # Create progress records
        progress1 = TaskProgress(
            task_id=main_task.id, progress_percentage=25, notes="Initial progress"
        )
        progress2 = TaskProgress(
            task_id=main_task.id, progress_percentage=75, notes="Significant progress"
        )
        db_session.add_all([progress1, progress2])

        # Create execution log
        execution_log = TaskExecutionLog(
            task_id=main_task.id,
            agent_type=AgentType.CODING,
            status=TaskStatus.IN_PROGRESS,
            start_time=datetime.now(),
            outputs={"phase": "implementation"},
        )
        db_session.add(execution_log)

        # Create comment
        comment = TaskComment(
            task_id=main_task.id,
            comment="Task is progressing well",
            comment_type="status_update",
        )
        db_session.add(comment)

        db_session.commit()

        # Verify all relationships work
        assert len(main_task.dependencies) == 1
        assert main_task.dependencies[0].depends_on_task.title == "Dependency Task"
        assert len(main_task.progress_records) == 2
        assert len(main_task.execution_logs) == 1
        assert main_task.execution_logs[0].agent_type == AgentType.CODING

    def test_task_cascade_operations(self, db_session):
        """Test cascade behavior when deleting tasks."""
        # Create parent and child tasks
        parent = Task(title="Parent Task")
        db_session.add(parent)
        db_session.commit()

        child = Task(title="Child Task", parent_task_id=parent.id)
        db_session.add(child)
        db_session.commit()

        # Create related entities
        progress = TaskProgress(task_id=child.id, progress_percentage=50)
        log = TaskExecutionLog(
            task_id=child.id,
            agent_type=AgentType.TESTING,
            status=TaskStatus.IN_PROGRESS,
            start_time=datetime.now(),
        )
        comment = TaskComment(task_id=child.id, comment="Test comment")

        db_session.add_all([progress, log, comment])
        db_session.commit()

        # Verify parent-child relationship
        assert len(parent.subtasks) == 1
        assert parent.subtasks[0].id == child.id

        # Clean up would depend on actual cascade configuration
        # This test mainly verifies the relationships exist

    def test_bulk_operations(self, db_session):
        """Test bulk database operations."""
        # Create multiple tasks
        tasks = [
            Task(title=f"Bulk Task {i}", priority=TaskPriority.LOW) for i in range(10)
        ]

        db_session.add_all(tasks)
        db_session.commit()

        # Verify all were created
        task_count = (
            db_session.query(Task).filter(Task.title.like("Bulk Task %")).count()
        )
        assert task_count == 10

        # Create progress records for all tasks
        progress_records = [
            TaskProgress(task_id=task.id, progress_percentage=0) for task in tasks
        ]

        db_session.add_all(progress_records)
        db_session.commit()

        # Verify progress records
        progress_count = (
            db_session.query(TaskProgress)
            .join(Task)
            .filter(Task.title.like("Bulk Task %"))
            .count()
        )
        assert progress_count == 10

    def test_complex_queries(self, db_session):
        """Test complex database queries."""
        # Create test data
        high_priority_task = Task(
            title="High Priority Task",
            priority=TaskPriority.HIGH,
            status=TaskStatus.IN_PROGRESS,
            component_area=ComponentArea.SECURITY,
        )
        low_priority_task = Task(
            title="Low Priority Task",
            priority=TaskPriority.LOW,
            status=TaskStatus.NOT_STARTED,
            component_area=ComponentArea.TESTING,
        )

        db_session.add_all([high_priority_task, low_priority_task])
        db_session.commit()

        # Query by priority
        high_priority_tasks = (
            db_session.query(Task).filter(Task.priority == TaskPriority.HIGH).all()
        )
        assert len(high_priority_tasks) == 1
        assert high_priority_tasks[0].title == "High Priority Task"

        # Query by status
        in_progress_tasks = (
            db_session.query(Task).filter(Task.status == TaskStatus.IN_PROGRESS).all()
        )
        assert len(in_progress_tasks) == 1

        # Query by component area
        security_tasks = (
            db_session.query(Task)
            .filter(Task.component_area == ComponentArea.SECURITY)
            .all()
        )
        assert len(security_tasks) == 1

        # Complex query with multiple conditions
        complex_query_result = (
            db_session.query(Task)
            .filter(
                Task.priority.in_([TaskPriority.HIGH, TaskPriority.CRITICAL]),
                Task.status != TaskStatus.COMPLETED,
            )
            .all()
        )
        assert len(complex_query_result) == 1


class TestLegacyAliases:
    """Test legacy compatibility aliases."""

    def test_legacy_aliases_exist(self):
        """Test that legacy aliases point to correct classes."""
        assert TaskTable is Task
        assert TaskDependencyTable is TaskDependency
        assert TaskProgressTable is TaskProgress
        assert TaskExecutionLogTable is TaskExecutionLog
        assert TaskCommentTable is TaskComment

    def test_legacy_aliases_functionality(self, db_session):
        """Test that legacy aliases work for creating instances."""
        # Use legacy alias to create task
        task = TaskTable(title="Legacy Task")
        db_session.add(task)
        db_session.commit()

        # Should work exactly like Task
        assert isinstance(task, Task)
        assert task.title == "Legacy Task"

        # Use legacy alias for progress
        progress = TaskProgressTable(task_id=task.id, progress_percentage=100)
        db_session.add(progress)
        db_session.commit()

        assert isinstance(progress, TaskProgress)
        assert progress.progress_percentage == 100


class TestDatabaseConstraints:
    """Test database-level constraints and validations."""

    def test_check_constraints(self, db_session):
        """Test check constraints defined in table args."""
        task = Task(title="Constraint Task")
        db_session.add(task)
        db_session.commit()

        # Test positive time constraint
        with pytest.raises(IntegrityError):
            task.time_estimate_hours = -1.0
            db_session.commit()

        db_session.rollback()

        # Test valid phase constraint
        with pytest.raises(IntegrityError):
            task.phase = 0  # Below minimum
            db_session.commit()

        db_session.rollback()

        with pytest.raises(IntegrityError):
            task.phase = 11  # Above maximum
            db_session.commit()

    def test_foreign_key_constraints(self, db_session):
        """Test foreign key constraint enforcement."""
        # Try to create task with invalid parent_task_id
        task = Task(title="Invalid Parent", parent_task_id=999)
        db_session.add(task)

        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_index_performance_hints(self, in_memory_db):
        """Test that indexes are properly defined for query performance."""
        # This test ensures commonly queried fields have indexes
        task_table = Task.__table__

        # Get all indexed column names
        indexed_columns = set()
        for index in task_table.indexes:
            for column in index.columns:
                indexed_columns.add(column.name)

        # Verify critical fields are indexed
        critical_fields = {"component_area", "phase", "status", "priority", "uuid"}

        assert critical_fields.issubset(indexed_columns)


class TestModelValidationEdgeCases:
    """Test edge cases and boundary conditions for model validation."""

    def test_task_boundary_values(self, db_session):
        """Test Task with boundary values."""
        # Minimum valid values
        min_task = Task(
            title="x",  # Minimum length
            phase=1,  # Minimum phase
            time_estimate_hours=0.1,  # Minimum time
        )
        db_session.add(min_task)
        db_session.commit()

        assert min_task.title == "x"
        assert min_task.phase == 1
        assert min_task.time_estimate_hours == 0.1

        # Maximum valid values
        max_task = Task(
            title="x" * 200,  # Maximum length
            phase=10,  # Maximum phase
            time_estimate_hours=100.0,  # Maximum time
        )
        db_session.add(max_task)
        db_session.commit()

        assert len(max_task.title) == 200
        assert max_task.phase == 10
        assert max_task.time_estimate_hours == 100.0

    def test_json_field_handling(self, db_session):
        """Test JSON field storage and retrieval."""
        task = Task(title="JSON Test Task")
        db_session.add(task)
        db_session.commit()

        # Create execution log with complex JSON data
        complex_outputs = {
            "nested": {"key": "value", "number": 42},
            "array": [1, 2, 3, "string"],
            "boolean": True,
            "null_value": None,
        }

        log = TaskExecutionLog(
            task_id=task.id,
            agent_type=AgentType.CODING,
            status=TaskStatus.COMPLETED,
            start_time=datetime.now(),
            outputs=complex_outputs,
        )

        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)

        # Verify JSON data integrity
        assert log.outputs == complex_outputs
        assert log.outputs["nested"]["key"] == "value"
        assert log.outputs["array"] == [1, 2, 3, "string"]
        assert log.outputs["boolean"] is True
        assert log.outputs["null_value"] is None

    def test_datetime_precision(self, db_session):
        """Test datetime field precision and timezone handling."""
        task = Task(title="DateTime Test")
        db_session.add(task)
        db_session.commit()

        # Record precise timestamps
        start_time = datetime.now()
        end_time = datetime.now()

        log = TaskExecutionLog(
            task_id=task.id,
            agent_type=AgentType.TESTING,
            status=TaskStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
        )

        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)

        # Verify datetime precision (within reasonable tolerance)
        assert abs((log.start_time - start_time).total_seconds()) < 1
        assert abs((log.end_time - end_time).total_seconds()) < 1

    def test_unicode_and_special_characters(self, db_session):
        """Test handling of unicode and special characters."""
        special_title = "Task with émojis 🚀 and ñoñó characters"
        special_description = "Description with\nnewlines\tand\ttabs"

        task = Task(title=special_title, description=special_description)

        db_session.add(task)
        db_session.commit()
        db_session.refresh(task)

        assert task.title == special_title
        assert task.description == special_description

        # Test in comments too
        comment = TaskComment(
            task_id=task.id,
            comment="Comment with special chars: àáâãäå çèéêë",
        )

        db_session.add(comment)
        db_session.commit()
        db_session.refresh(comment)

        assert "àáâãäå" in comment.comment
        assert "çèéêë" in comment.comment

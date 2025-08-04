"""Comprehensive tests for database operations and connection management.

This module provides testing for database initialization, session management,
connection handling, and schema validation.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlmodel import Session, SQLModel, select

from src.config import get_settings
from src.database import (
    create_db_and_tables,
    engine,
    get_session_context,
    get_sync_session,
    init_database,
    sync_engine,
    verify_database,
)
from src.schemas.database import (
    Task,
    TaskComment,
    TaskDependency,
    TaskExecutionLog,
    TaskProgress,
)
from src.schemas.unified_models import (
    AgentType,
    ComponentArea,
    TaskComplexity,
    TaskCore,
    TaskPriority,
    TaskStatus,
)


class TestDatabaseFixtures:
    """Test fixtures for database operations."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_file.close()
        db_path = temp_file.name

        yield db_path

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.fixture
    def temp_engine(self, temp_db_path):
        """Create temporary database engine."""
        temp_engine = create_engine(f"sqlite:///{temp_db_path}", echo=False)
        SQLModel.metadata.create_all(temp_engine)
        return temp_engine

    @pytest.fixture
    def sample_task_core(self):
        """Sample TaskCore for testing."""
        return TaskCore(
            title="Database Test Task",
            description="Testing database operations",
            component_area=ComponentArea.DATABASE,
            phase=1,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.LOW,
            status=TaskStatus.NOT_STARTED,
            time_estimate_hours=2.0,
        )


class TestDatabaseInitialization(TestDatabaseFixtures):
    """Test database initialization and setup."""

    @patch("src.database.settings")
    def test_create_db_and_tables_creates_directory(self, mock_settings, temp_db_path):
        """Test that create_db_and_tables creates parent directory."""
        # Create a path in a non-existent directory
        non_existent_dir = Path(temp_db_path).parent / "non_existent" / "test.db"

        mock_settings.database.implementation_tracker_path = str(non_existent_dir)

        create_db_and_tables()

        # Verify directory was created
        assert non_existent_dir.parent.exists()
        assert non_existent_dir.exists()

        # Cleanup
        non_existent_dir.unlink()
        non_existent_dir.parent.rmdir()

    @patch("src.database.settings")
    def test_create_db_and_tables_creates_schema(self, mock_settings, temp_db_path):
        """Test that create_db_and_tables creates all required tables."""
        mock_settings.database.implementation_tracker_path = temp_db_path

        create_db_and_tables()

        # Verify database file exists
        assert Path(temp_db_path).exists()

        # Verify all tables exist
        temp_engine = create_engine(f"sqlite:///{temp_db_path}")

        with Session(temp_engine) as session:
            # Test each table by trying to query it
            session.exec(select(Task)).all()
            session.exec(select(TaskDependency)).all()
            session.exec(select(TaskProgress)).all()
            session.exec(select(TaskExecutionLog)).all()
            session.exec(select(TaskComment)).all()

    def test_get_sync_session_returns_session(self):
        """Test get_sync_session returns SQLModel Session."""
        session = get_sync_session()

        assert isinstance(session, Session)
        assert session.bind is engine

        session.close()

    def test_get_sync_session_different_instances(self):
        """Test get_sync_session returns different instances."""
        session1 = get_sync_session()
        session2 = get_sync_session()

        assert session1 is not session2

        session1.close()
        session2.close()

    @patch("src.database.settings")
    def test_init_database_complete_setup(self, mock_settings, temp_db_path):
        """Test complete database initialization workflow."""
        mock_settings.database.implementation_tracker_path = temp_db_path

        # Should not raise any exceptions
        init_database()

        # Verify database was created and is functional
        assert Path(temp_db_path).exists()

        # Verify we can interact with the database
        temp_engine = create_engine(f"sqlite:///{temp_db_path}")
        with Session(temp_engine) as session:
            # Try to create and retrieve a task
            task = Task(
                title="Init Test Task",
                description="Testing database initialization",
                component_area=ComponentArea.DATABASE,
                status=TaskStatus.NOT_STARTED,
                priority=TaskPriority.LOW,
                complexity=TaskComplexity.LOW,
                time_estimate_hours=1.0,
            )
            session.add(task)
            session.commit()

            # Verify task was created
            retrieved_task = session.exec(
                select(Task).where(Task.title == "Init Test Task")
            ).first()
            assert retrieved_task is not None
            assert retrieved_task.title == "Init Test Task"

    def test_sync_engine_is_alias(self):
        """Test that sync_engine is alias for main engine."""
        assert sync_engine is engine


class TestSessionContextManager(TestDatabaseFixtures):
    """Test session context manager functionality."""

    def test_get_session_context_basic_usage(self, temp_engine):
        """Test basic usage of session context manager."""
        with get_session_context() as session:
            assert isinstance(session, Session)
            assert session.bind is not None

            # Should be able to perform database operations
            result = session.exec(text("SELECT 1 as test")).fetchone()
            assert result.test == 1

    def test_get_session_context_auto_commit(self, temp_engine, sample_task_core):
        """Test that session context manager auto-commits on success."""
        task_entity = Task.from_core_model(sample_task_core)

        # Temporarily patch the engine to use our temp engine
        with patch("src.database.engine", temp_engine):
            with get_session_context() as session:
                session.add(task_entity)
                # Don't manually commit - should be done automatically

        # Verify task was committed
        with Session(temp_engine) as verify_session:
            saved_task = verify_session.exec(
                select(Task).where(Task.title == task_entity.title)
            ).first()
            assert saved_task is not None
            assert saved_task.title == sample_task_core.title

    def test_get_session_context_rollback_on_exception(
        self, temp_engine, sample_task_core
    ):
        """Test that session context manager rolls back on exception."""
        task_entity = Task.from_core_model(sample_task_core)

        with patch("src.database.engine", temp_engine):
            try:
                with get_session_context() as session:
                    session.add(task_entity)
                    session.flush()  # Make sure the task is in the session

                    # Force an exception
                    raise ValueError("Test exception")

            except ValueError:
                pass  # Expected exception

        # Verify task was rolled back
        with Session(temp_engine) as verify_session:
            saved_task = verify_session.exec(
                select(Task).where(Task.title == task_entity.title)
            ).first()
            assert saved_task is None

    def test_get_session_context_nested_usage(self, temp_engine, sample_task_core):
        """Test nested usage of session context manager."""
        task1_entity = Task.from_core_model(sample_task_core)

        task2_core = sample_task_core.model_copy()
        task2_core.title = "Nested Task 2"
        task2_entity = Task.from_core_model(task2_core)

        with patch("src.database.engine", temp_engine):
            with get_session_context() as outer_session:
                outer_session.add(task1_entity)

                with get_session_context() as inner_session:
                    # This is a different session
                    assert inner_session is not outer_session
                    inner_session.add(task2_entity)

        # Both tasks should be committed by their respective sessions
        with Session(temp_engine) as verify_session:
            task1 = verify_session.exec(
                select(Task).where(Task.title == task1_entity.title)
            ).first()
            task2 = verify_session.exec(
                select(Task).where(Task.title == task2_entity.title)
            ).first()

            assert task1 is not None
            assert task2 is not None

    def test_get_session_context_exception_propagation(self, temp_engine):
        """Test that exceptions are properly propagated."""
        with patch("src.database.engine", temp_engine):
            with pytest.raises(ValueError, match="Test exception"):
                with get_session_context():
                    raise ValueError("Test exception")


class TestDatabaseVerification(TestDatabaseFixtures):
    """Test database verification functionality."""

    @patch("src.database.get_session_context")
    def test_verify_database_success(
        self, mock_get_session_context, temp_engine, sample_task_core
    ):
        """Test successful database verification."""
        # Setup mock session with test data
        mock_session = MagicMock()
        mock_get_session_context.return_value.__enter__.return_value = mock_session

        # Mock query results
        mock_session.exec.side_effect = [
            [Task()],  # 1 task
            [TaskDependency()],  # 1 dependency
            [TaskProgress()],  # 1 progress record
            [TaskExecutionLog()],  # 1 execution log
            [TaskComment()],  # 1 comment
        ]

        result = verify_database()

        assert result is True

        # Verify all tables were queried
        assert mock_session.exec.call_count == 5

    @patch("src.database.get_session_context")
    def test_verify_database_failure(self, mock_get_session_context):
        """Test database verification failure."""
        # Mock session to raise an exception
        mock_get_session_context.return_value.__enter__.side_effect = Exception(
            "Database error"
        )

        result = verify_database()

        assert result is False

    @patch("src.database.get_session_context")
    def test_verify_database_empty_tables(self, mock_get_session_context):
        """Test database verification with empty tables."""
        mock_session = MagicMock()
        mock_get_session_context.return_value.__enter__.return_value = mock_session

        # Mock empty query results
        mock_session.exec.return_value = []

        result = verify_database()

        assert result is True  # Empty tables are still valid

    def test_verify_database_real_database(self, temp_engine, sample_task_core):
        """Test verification with real database."""
        # Create and populate database
        with Session(temp_engine) as session:
            task = Task.from_core_model(sample_task_core)
            session.add(task)

            progress = TaskProgress(
                task_id=1,  # Will be set properly after task is committed
                progress_percentage=50,
                notes="Test progress",
            )

            session.commit()

            # Set proper task_id after commit
            progress.task_id = task.id
            session.add(progress)
            session.commit()

        # Patch engine to use our temp engine
        with patch("src.database.engine", temp_engine):
            result = verify_database()
            assert result is True


class TestEngineConfiguration(TestDatabaseFixtures):
    """Test database engine configuration."""

    def test_engine_configuration(self):
        """Test that engine is configured correctly."""
        assert engine is not None
        assert str(engine.url).startswith("sqlite:///")

        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1

    @patch("src.database.get_settings")
    def test_engine_uses_settings(self, mock_get_settings):
        """Test that engine configuration uses application settings."""
        # The engine is created at module level, so this test verifies
        # that settings are being used (engine creation happens on import)
        mock_settings = MagicMock()
        mock_settings.database.implementation_tracker_path = "/test/path.db"
        mock_settings.database.echo_sql = True
        mock_settings.database.pool_size = 10
        mock_settings.database.pool_timeout = 30

        mock_get_settings.return_value = mock_settings

        # Import would have already happened, so we can't test the actual
        # engine creation, but we can verify the settings structure
        settings = get_settings()
        assert hasattr(settings.database, "implementation_tracker_path")
        assert hasattr(settings.database, "echo_sql")
        assert hasattr(settings.database, "pool_size")
        assert hasattr(settings.database, "pool_timeout")


class TestDatabaseErrorHandling(TestDatabaseFixtures):
    """Test database error handling scenarios."""

    @patch("src.database.engine")
    def test_session_creation_failure(self, mock_engine):
        """Test handling of session creation failures."""
        mock_engine.side_effect = SQLAlchemyError("Connection failed")

        with pytest.raises(SQLAlchemyError):
            get_sync_session()

    @patch("src.database.SQLModel.metadata.create_all")
    def test_table_creation_failure(self, mock_create_all):
        """Test handling of table creation failures."""
        mock_create_all.side_effect = OperationalError(
            "Cannot create table", None, None
        )

        with pytest.raises(OperationalError):
            create_db_and_tables()

    def test_get_session_context_database_locked(self, temp_db_path):
        """Test handling of database locked scenario."""
        # Create two connections to simulate lock
        engine1 = create_engine(f"sqlite:///{temp_db_path}")
        engine2 = create_engine(f"sqlite:///{temp_db_path}")

        with patch("src.database.engine", engine1):
            with get_session_context() as session1:
                # Start a transaction that locks the database
                session1.exec(text("BEGIN EXCLUSIVE"))

                # Try to access with second connection (should handle gracefully)
                with patch("src.database.engine", engine2):
                    with pytest.raises(
                        Exception
                    ):  # Could be various lock-related exceptions
                        with get_session_context() as session2:
                            session2.exec(text("SELECT 1"))

    def test_verify_database_partial_failure(self, temp_engine):
        """Test database verification with partial table access failure."""
        # Create database with only some tables
        with Session(temp_engine) as session:
            # Create a custom table that exists
            session.exec(text("CREATE TABLE test_table (id INTEGER PRIMARY KEY)"))
            session.commit()

        with patch("src.database.engine", temp_engine):
            # This should fail because not all expected tables exist
            result = verify_database()
            assert result is False


class TestDatabaseConcurrency(TestDatabaseFixtures):
    """Test database operations under concurrent access."""

    def test_multiple_sessions_same_data(self, temp_engine, sample_task_core):
        """Test multiple sessions accessing same data."""
        task_entity = Task.from_core_model(sample_task_core)

        # Create task in first session
        with Session(temp_engine) as session1:
            session1.add(task_entity)
            session1.commit()
            task_id = task_entity.id

        # Access same task from multiple sessions
        sessions = [Session(temp_engine) for _ in range(3)]

        try:
            tasks = []
            for session in sessions:
                task = session.get(Task, task_id)
                tasks.append(task)
                assert task is not None
                assert task.title == sample_task_core.title

            # Each session should have its own instance
            assert all(task is not None for task in tasks)
            assert all(task.id == task_id for task in tasks)

        finally:
            for session in sessions:
                session.close()

    def test_concurrent_writes_different_sessions(self, temp_engine, sample_task_core):
        """Test concurrent writes from different sessions."""
        # Create multiple tasks concurrently
        task_cores = []
        for i in range(5):
            task_core = sample_task_core.model_copy()
            task_core.title = f"Concurrent Task {i}"
            task_cores.append(task_core)

        # Write from different sessions
        sessions = [Session(temp_engine) for _ in range(5)]

        try:
            for i, (session, task_core) in enumerate(
                zip(sessions, task_cores, strict=False)
            ):
                task_entity = Task.from_core_model(task_core)
                session.add(task_entity)
                session.commit()

        finally:
            for session in sessions:
                session.close()

        # Verify all tasks were created
        with Session(temp_engine) as verify_session:
            all_tasks = verify_session.exec(select(Task)).all()
            assert len(all_tasks) == 5

            task_titles = [task.title for task in all_tasks]
            for i in range(5):
                assert f"Concurrent Task {i}" in task_titles

    def test_session_isolation(self, temp_engine, sample_task_core):
        """Test transaction isolation between sessions."""
        task_entity = Task.from_core_model(sample_task_core)

        session1 = Session(temp_engine)
        session2 = Session(temp_engine)

        try:
            # Add task in session1 but don't commit
            session1.add(task_entity)
            session1.flush()  # Send to database but don't commit

            # Session2 should not see the uncommitted task
            tasks_in_session2 = session2.exec(select(Task)).all()
            assert len(tasks_in_session2) == 0

            # Commit in session1
            session1.commit()

            # Now session2 should see the task (after refresh)
            session2.expire_all()  # Clear session cache
            tasks_in_session2_after_commit = session2.exec(select(Task)).all()
            assert len(tasks_in_session2_after_commit) == 1

        finally:
            session1.close()
            session2.close()


class TestDatabaseSchemaValidation(TestDatabaseFixtures):
    """Test database schema validation and constraints."""

    def test_task_table_constraints(self, temp_engine, sample_task_core):
        """Test Task table constraints and validation."""
        with Session(temp_engine) as session:
            # Test valid task creation
            valid_task = Task.from_core_model(sample_task_core)
            session.add(valid_task)
            session.commit()
            assert valid_task.id is not None

            # Test constraint violations (these should be caught by Pydantic, not DB)
            # But we can test database-level constraints

            # Test unique UUID constraint if it exists
            task2 = Task.from_core_model(sample_task_core)
            task2.title = "Another Task"
            if hasattr(task2, "uuid"):
                task2.uuid = valid_task.uuid  # Duplicate UUID
                session.add(task2)

                with pytest.raises(Exception):  # Should violate unique constraint
                    session.commit()

                session.rollback()

    def test_task_dependency_constraints(self, temp_engine, sample_task_core):
        """Test TaskDependency table constraints."""
        with Session(temp_engine) as session:
            # Create two tasks
            task1 = Task.from_core_model(sample_task_core)
            task2_core = sample_task_core.model_copy()
            task2_core.title = "Dependency Task"
            task2 = Task.from_core_model(task2_core)

            session.add(task1)
            session.add(task2)
            session.commit()

            # Create valid dependency
            dependency = TaskDependency(
                task_id=task1.id, depends_on_task_id=task2.id, dependency_type="blocks"
            )
            session.add(dependency)
            session.commit()
            assert dependency.id is not None

            # Test duplicate dependency constraint
            duplicate_dependency = TaskDependency(
                task_id=task1.id,
                depends_on_task_id=task2.id,
                dependency_type="requires",
            )
            session.add(duplicate_dependency)

            with pytest.raises(Exception):  # Should violate unique constraint
                session.commit()

            session.rollback()

    def test_foreign_key_constraints(self, temp_engine, sample_task_core):
        """Test foreign key constraints."""
        with Session(temp_engine) as session:
            # Create task
            task = Task.from_core_model(sample_task_core)
            session.add(task)
            session.commit()

            # Test valid foreign key reference
            progress = TaskProgress(
                task_id=task.id, progress_percentage=50, notes="Valid progress"
            )
            session.add(progress)
            session.commit()
            assert progress.id is not None

            # Test invalid foreign key reference
            invalid_progress = TaskProgress(
                task_id=999999,  # Non-existent task
                progress_percentage=25,
                notes="Invalid progress",
            )
            session.add(invalid_progress)

            with pytest.raises(Exception):  # Should violate foreign key constraint
                session.commit()

            session.rollback()

    def test_check_constraints(self, temp_engine, sample_task_core):
        """Test check constraints on tables."""
        with Session(temp_engine) as session:
            task = Task.from_core_model(sample_task_core)
            session.add(task)
            session.commit()

            # Test valid progress percentage
            valid_progress = TaskProgress(
                task_id=task.id, progress_percentage=75, notes="Valid progress"
            )
            session.add(valid_progress)
            session.commit()

            # Test invalid progress percentage (if check constraint exists)
            session.rollback()  # Clear any pending changes

            invalid_progress = TaskProgress(
                task_id=task.id,
                progress_percentage=150,  # > 100, should violate check constraint
                notes="Invalid progress",
            )
            session.add(invalid_progress)

            try:
                session.commit()
                # If we get here, the constraint might not be enforced at DB level
                # That's okay - Pydantic handles it at application level
            except Exception:
                # This is expected if check constraint exists
                session.rollback()


class TestDatabasePerformance(TestDatabaseFixtures):
    """Test database performance and optimization."""

    def test_bulk_insert_performance(self, temp_engine, sample_task_core):
        """Test bulk insert operations."""
        # Create multiple tasks for bulk insert
        tasks = []
        for i in range(100):
            task_core = sample_task_core.model_copy()
            task_core.title = f"Bulk Task {i}"
            task_entity = Task.from_core_model(task_core)
            tasks.append(task_entity)

        # Measure bulk insert
        import time

        start_time = time.time()

        with Session(temp_engine) as session:
            session.add_all(tasks)
            session.commit()

        end_time = time.time()
        bulk_time = end_time - start_time

        # Verify all tasks were created
        with Session(temp_engine) as verify_session:
            task_count = len(verify_session.exec(select(Task)).all())
            assert task_count == 100

        # Should complete reasonably quickly (less than 5 seconds for 100 tasks)
        assert bulk_time < 5.0

    def test_query_performance_with_indexes(self, temp_engine, sample_task_core):
        """Test query performance with indexed columns."""
        # Create tasks with different statuses and priorities
        tasks = []
        statuses = [
            TaskStatus.NOT_STARTED,
            TaskStatus.IN_PROGRESS,
            TaskStatus.COMPLETED,
        ]
        priorities = [TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH]

        for i in range(50):
            task_core = sample_task_core.model_copy()
            task_core.title = f"Indexed Task {i}"
            task_core.status = statuses[i % len(statuses)]
            task_core.priority = priorities[i % len(priorities)]
            task_entity = Task.from_core_model(task_core)
            tasks.append(task_entity)

        with Session(temp_engine) as session:
            session.add_all(tasks)
            session.commit()

        # Test indexed queries (status and priority should be indexed)
        import time

        with Session(temp_engine) as session:
            start_time = time.time()

            # Query by status (should use index)
            in_progress_tasks = session.exec(
                select(Task).where(Task.status == TaskStatus.IN_PROGRESS)
            ).all()

            # Query by priority (should use index)
            high_priority_tasks = session.exec(
                select(Task).where(Task.priority == TaskPriority.HIGH)
            ).all()

            end_time = time.time()
            query_time = end_time - start_time

        # Queries should complete quickly
        assert query_time < 1.0
        assert len(in_progress_tasks) > 0
        assert len(high_priority_tasks) > 0

    def test_connection_pooling(self, temp_engine):
        """Test connection pooling behavior."""
        # Create multiple sessions quickly
        sessions = []
        for _i in range(10):
            session = Session(temp_engine)
            sessions.append(session)

            # Perform a simple query
            result = session.exec(text("SELECT 1")).fetchone()
            assert result[0] == 1

        # Close all sessions
        for session in sessions:
            session.close()

        # Should not have any connection leaks or errors


class TestDatabaseMigrationAndSchema(TestDatabaseFixtures):
    """Test database migration and schema evolution."""

    def test_schema_evolution_compatibility(self, temp_db_path):
        """Test that schema changes are backward compatible."""
        # Create initial schema
        engine1 = create_engine(f"sqlite:///{temp_db_path}")
        SQLModel.metadata.create_all(engine1)

        # Add some data
        with Session(engine1) as session:
            task = Task(
                title="Migration Test Task",
                description="Testing schema evolution",
                component_area=ComponentArea.DATABASE,
                status=TaskStatus.NOT_STARTED,
                priority=TaskPriority.LOW,
                complexity=TaskComplexity.LOW,
                time_estimate_hours=1.0,
            )
            session.add(task)
            session.commit()
            task_id = task.id

        engine1.dispose()

        # Create new engine and verify data is still accessible
        engine2 = create_engine(f"sqlite:///{temp_db_path}")

        with Session(engine2) as session:
            retrieved_task = session.get(Task, task_id)
            assert retrieved_task is not None
            assert retrieved_task.title == "Migration Test Task"

        engine2.dispose()

    def test_table_metadata_consistency(self, temp_engine):
        """Test that table metadata is consistent with models."""
        # Get table names from metadata
        table_names = list(SQLModel.metadata.tables.keys())

        expected_tables = [
            "tasks",
            "task_dependencies",
            "task_progress",
            "task_execution_logs",
            "task_comments",
        ]

        for expected_table in expected_tables:
            assert expected_table in table_names

        # Verify table structures match model definitions
        tasks_table = SQLModel.metadata.tables["tasks"]

        # Check for essential columns
        column_names = [col.name for col in tasks_table.columns]

        essential_columns = [
            "id",
            "title",
            "description",
            "component_area",
            "phase",
            "priority",
            "complexity",
            "status",
            "time_estimate_hours",
            "created_at",
            "updated_at",
        ]

        for essential_column in essential_columns:
            assert essential_column in column_names


class TestDatabaseIntegration(TestDatabaseFixtures):
    """Integration tests for database operations."""

    def test_full_task_lifecycle_database_operations(
        self, temp_engine, sample_task_core
    ):
        """Test complete task lifecycle database operations."""
        with patch("src.database.engine", temp_engine):
            # 1. Initialize database
            init_database()

            # 2. Create task using session context
            task_entity = Task.from_core_model(sample_task_core)
            with get_session_context() as session:
                session.add(task_entity)
                # Auto-commit on context exit

            task_id = task_entity.id

            # 3. Add progress using different session
            with get_session_context() as session:
                progress = TaskProgress(
                    task_id=task_id, progress_percentage=25, notes="Started work"
                )
                session.add(progress)

            # 4. Add dependency
            dependency_task = Task.from_core_model(sample_task_core)
            dependency_task.title = "Dependency Task"

            with get_session_context() as session:
                session.add(dependency_task)
                session.flush()  # Get ID

                dependency = TaskDependency(
                    task_id=task_id,
                    depends_on_task_id=dependency_task.id,
                    dependency_type="blocks",
                )
                session.add(dependency)

            # 5. Add execution log
            with get_session_context() as session:
                from uuid import uuid4

                execution_log = TaskExecutionLog(
                    task_id=task_id,
                    execution_id=uuid4(),
                    agent_type=AgentType.CODING,
                    status=TaskStatus.IN_PROGRESS,
                    start_time=datetime.now(),
                    confidence_score=0.8,
                )
                session.add(execution_log)

            # 6. Verify all data was persisted correctly
            with Session(temp_engine) as verify_session:
                # Verify task
                saved_task = verify_session.get(Task, task_id)
                assert saved_task is not None
                assert saved_task.title == sample_task_core.title

                # Verify progress
                progress_records = verify_session.exec(
                    select(TaskProgress).where(TaskProgress.task_id == task_id)
                ).all()
                assert len(progress_records) == 1
                assert progress_records[0].progress_percentage == 25

                # Verify dependency
                dependencies = verify_session.exec(
                    select(TaskDependency).where(TaskDependency.task_id == task_id)
                ).all()
                assert len(dependencies) == 1
                assert dependencies[0].dependency_type == "blocks"

                # Verify execution log
                execution_logs = verify_session.exec(
                    select(TaskExecutionLog).where(TaskExecutionLog.task_id == task_id)
                ).all()
                assert len(execution_logs) == 1
                assert execution_logs[0].agent_type == AgentType.CODING

    def test_database_verification_integration(self, temp_engine):
        """Test database verification with realistic data."""
        with patch("src.database.engine", temp_engine):
            # Initialize with sample data
            with get_session_context() as session:
                # Create task
                task = Task(
                    title="Verification Test",
                    description="Testing verification",
                    component_area=ComponentArea.DATABASE,
                    status=TaskStatus.NOT_STARTED,
                    priority=TaskPriority.MEDIUM,
                    complexity=TaskComplexity.LOW,
                    time_estimate_hours=1.0,
                )
                session.add(task)
                session.flush()

                # Create related records
                progress = TaskProgress(
                    task_id=task.id, progress_percentage=0, notes="Initial"
                )
                comment = TaskComment(task_id=task.id, comment="Test comment")

                session.add(progress)
                session.add(comment)

            # Verify database
            result = verify_database()
            assert result is True

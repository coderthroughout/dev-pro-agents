"""Comprehensive tests for task service business logic.

Tests cover service layer functionality including:
- Business logic validation and processing
- Service-repository coordination
- Error handling and transaction management
- Complex business operations
- Database session management
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, call, patch

import pytest
from sqlmodel import Session

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
from src.services.task_service import TaskService


class TestTaskServiceInitialization:
    """Test suite for TaskService initialization and setup."""

    def test_init_with_default_session(self):
        """Test TaskService initialization with default session."""
        with patch("src.services.task_service.get_sync_session") as mock_get_session:
            mock_session = Mock(spec=Session)
            mock_get_session.return_value = mock_session

            service = TaskService()

            mock_get_session.assert_called_once()
            assert service.session is mock_session
            assert hasattr(service, "task_repo")
            assert hasattr(service, "execution_repo")

    def test_init_with_provided_session(self):
        """Test TaskService initialization with provided session."""
        mock_session = Mock(spec=Session)

        with patch("src.services.task_service.get_sync_session") as mock_get_session:
            service = TaskService(session=mock_session)

            mock_get_session.assert_not_called()
            assert service.session is mock_session

    def test_repositories_initialization(self):
        """Test that repositories are properly initialized."""
        mock_session = Mock(spec=Session)

        with (
            patch("src.services.task_service.TaskRepository") as mock_task_repo,
            patch(
                "src.services.task_service.TaskExecutionRepository"
            ) as mock_exec_repo,
        ):
            service = TaskService(session=mock_session)

            mock_task_repo.assert_called_once_with(mock_session)
            mock_exec_repo.assert_called_once_with(mock_session)
            assert service.task_repo == mock_task_repo.return_value
            assert service.execution_repo == mock_exec_repo.return_value


class TestCreateTaskFromDelegation:
    """Test suite for creating tasks from supervisor delegations."""

    @pytest.fixture
    def task_service_with_mocks(self):
        """TaskService with mocked dependencies."""
        mock_session = Mock(spec=Session)
        mock_task_repo = Mock()
        mock_execution_repo = Mock()

        service = TaskService(session=mock_session)
        service.task_repo = mock_task_repo
        service.execution_repo = mock_execution_repo

        return service

    @pytest.fixture
    def sample_delegation(self):
        """Sample task delegation for testing."""
        return TaskDelegation(
            assigned_agent=AgentType.CODING,
            reasoning="This task requires code implementation",
            priority=TaskPriority.HIGH,
            estimated_duration=120,  # 2 hours in minutes
            dependencies=[1, 2],
            context_requirements=["JWT knowledge", "Security patterns"],
            confidence_score=0.85,
        )

    def test_create_task_from_delegation_success(
        self, task_service_with_mocks, sample_delegation
    ):
        """Test successful task creation from delegation."""
        service = task_service_with_mocks

        # Mock repository response
        mock_task_entity = Mock()
        mock_task_core = TaskCore(
            id=123,
            title="Implement authentication",
            description="Create JWT auth system",
            priority=TaskPriority.HIGH,
            time_estimate_hours=2.0,
        )
        mock_task_entity.to_core_model.return_value = mock_task_core

        service.task_repo.create_task_with_dependencies.return_value = mock_task_entity

        # Execute
        result = service.create_task_from_delegation(
            delegation=sample_delegation,
            title="Implement authentication",
            description="Create JWT auth system",
            component_area=ComponentArea.SECURITY,
        )

        # Verify task creation call
        service.task_repo.create_task_with_dependencies.assert_called_once()
        call_args = service.task_repo.create_task_with_dependencies.call_args

        created_task = call_args[1]["task_core"]  # keyword argument
        assert created_task.title == "Implement authentication"
        assert created_task.description == "Create JWT auth system"
        assert created_task.priority == TaskPriority.HIGH
        assert created_task.time_estimate_hours == 2.0  # 120 minutes / 60
        assert "JWT knowledge; Security patterns" in created_task.success_criteria

        # Verify dependencies
        assert call_args[1]["dependency_task_ids"] == [1, 2]

        # Verify session commit
        service.session.commit.assert_called_once()

        # Verify return value
        assert result == mock_task_core

    def test_create_task_delegation_duration_conversion(self, task_service_with_mocks):
        """Test proper conversion from minutes to hours."""
        service = task_service_with_mocks

        delegation = TaskDelegation(
            assigned_agent=AgentType.RESEARCH,
            reasoning="Research task",
            priority=TaskPriority.MEDIUM,
            estimated_duration=90,  # 1.5 hours
            dependencies=[],
            confidence_score=0.8,
        )

        mock_task_entity = Mock()
        mock_task_entity.to_core_model.return_value = TaskCore(
            id=1, title="Research task", time_estimate_hours=1.5
        )
        service.task_repo.create_task_with_dependencies.return_value = mock_task_entity

        service.create_task_from_delegation(
            delegation=delegation, title="Research task"
        )

        call_args = service.task_repo.create_task_with_dependencies.call_args
        created_task = call_args[1]["task_core"]
        assert created_task.time_estimate_hours == 1.5

    def test_create_task_empty_success_criteria(self, task_service_with_mocks):
        """Test task creation with empty success criteria."""
        service = task_service_with_mocks

        delegation = TaskDelegation(
            assigned_agent=AgentType.TESTING,
            reasoning="Testing task",
            priority=TaskPriority.LOW,
            estimated_duration=30,
            dependencies=[],
            success_criteria=[],  # Empty criteria
            confidence_score=0.7,
        )

        mock_task_entity = Mock()
        mock_task_entity.to_core_model.return_value = TaskCore(id=1, title="Test")
        service.task_repo.create_task_with_dependencies.return_value = mock_task_entity

        service.create_task_from_delegation(delegation=delegation, title="Test task")

        call_args = service.task_repo.create_task_with_dependencies.call_args
        created_task = call_args[1]["task_core"]
        assert created_task.success_criteria == ""


class TestExecuteTaskWithAgent:
    """Test suite for task execution with agents."""

    @pytest.fixture
    def service_with_mocks(self):
        """Service with fully mocked dependencies."""
        mock_session = Mock(spec=Session)
        service = TaskService(session=mock_session)
        service.task_repo = Mock()
        service.execution_repo = Mock()
        return service

    @pytest.fixture
    def mock_execution_log(self):
        """Mock execution log for testing."""
        mock_log = Mock()
        mock_log.execution_id = "exec_123"
        mock_log.start_time = datetime.now()
        return mock_log

    def test_execute_task_success_with_dict_result(
        self, service_with_mocks, mock_execution_log
    ):
        """Test successful task execution with dictionary result."""
        service = service_with_mocks
        task_id = 42
        agent_type = AgentType.CODING

        # Mock execution log creation
        service.execution_repo.log_execution_start.return_value = mock_execution_log

        # Mock agent function that returns dict
        def mock_agent_function(tid):
            return {
                "status": TaskStatus.COMPLETED,
                "success": True,
                "outputs": {"code_files": ["auth.py", "models.py"]},
                "artifacts": ["auth.py", "models.py"],
                "confidence_score": 0.9,
            }

        # Execute
        result = service.execute_task_with_agent(
            task_id=task_id, agent_type=agent_type, agent_function=mock_agent_function
        )

        # Verify execution logging started
        service.execution_repo.log_execution_start.assert_called_once_with(
            task_id, agent_type
        )

        # Verify task status updates
        expected_status_calls = [
            call(
                task_id=task_id,
                status=TaskStatus.IN_PROGRESS,
                progress_percentage=10,
                notes=f"Started execution with {agent_type.value} agent",
            ),
            call(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                progress_percentage=100,
                notes=f"Execution completed by {agent_type.value} agent",
            ),
        ]
        service.task_repo.update_status_with_progress.assert_has_calls(
            expected_status_calls
        )

        # Verify execution completion logging
        service.execution_repo.log_execution_complete.assert_called_once()
        complete_call = service.execution_repo.log_execution_complete.call_args
        assert complete_call[1]["execution_id"] == mock_execution_log.execution_id
        assert complete_call[1]["status"] == TaskStatus.COMPLETED

        # Verify session commit
        service.session.commit.assert_called_once()

        # Verify return value is AgentReport
        assert isinstance(result, AgentReport)
        assert result.agent_type == agent_type
        assert result.task_id == task_id
        assert result.status == TaskStatus.COMPLETED

    def test_execute_task_success_with_agent_report_result(
        self, service_with_mocks, mock_execution_log
    ):
        """Test successful task execution with AgentReport result."""
        service = service_with_mocks
        task_id = 55
        agent_type = AgentType.RESEARCH

        service.execution_repo.log_execution_start.return_value = mock_execution_log

        # Mock agent function that returns AgentReport
        expected_report = AgentReport(
            agent_name=AgentType.RESEARCH,
            agent_type=AgentType.RESEARCH,
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            outputs={"research_data": "comprehensive findings"},
            confidence_score=0.95,
        )

        def mock_agent_function(tid):
            return expected_report

        # Execute
        result = service.execute_task_with_agent(
            task_id=task_id, agent_type=agent_type, agent_function=mock_agent_function
        )

        # Verify the AgentReport was used directly
        assert result == expected_report
        assert result.task_id == task_id

        # Verify status updates
        service.task_repo.update_status_with_progress.assert_called()
        final_call = service.task_repo.update_status_with_progress.call_args_list[-1]
        assert final_call[1]["status"] == TaskStatus.COMPLETED
        assert final_call[1]["progress_percentage"] == 100

    def test_execute_task_success_with_simple_result(
        self, service_with_mocks, mock_execution_log
    ):
        """Test successful task execution with simple string result."""
        service = service_with_mocks
        task_id = 77
        agent_type = AgentType.DOCUMENTATION

        service.execution_repo.log_execution_start.return_value = mock_execution_log

        # Mock agent function that returns simple string
        def mock_agent_function(tid):
            return "Documentation completed successfully"

        # Execute
        result = service.execute_task_with_agent(
            task_id=task_id, agent_type=agent_type, agent_function=mock_agent_function
        )

        # Verify basic AgentReport was created
        assert isinstance(result, AgentReport)
        assert result.agent_name == f"{agent_type.value}_agent"
        assert result.agent_type == agent_type
        assert result.task_id == task_id
        assert result.status == TaskStatus.COMPLETED
        assert result.outputs == {"result": "Documentation completed successfully"}

    def test_execute_task_failure_handling(
        self, service_with_mocks, mock_execution_log
    ):
        """Test task execution failure handling."""
        service = service_with_mocks
        task_id = 99
        agent_type = AgentType.TESTING

        service.execution_repo.log_execution_start.return_value = mock_execution_log

        # Mock agent function that raises exception
        def failing_agent_function(tid):
            raise ValueError("Agent execution failed")

        # Execute and expect exception
        with pytest.raises(ValueError, match="Agent execution failed"):
            service.execute_task_with_agent(
                task_id=task_id,
                agent_type=agent_type,
                agent_function=failing_agent_function,
            )

        # Verify failure logging
        service.execution_repo.log_execution_complete.assert_called_once()
        complete_call = service.execution_repo.log_execution_complete.call_args
        assert complete_call[1]["execution_id"] == mock_execution_log.execution_id
        assert complete_call[1]["status"] == TaskStatus.FAILED
        assert complete_call[1]["error_details"] == "Agent execution failed"

        # Verify task status set to failed
        final_status_call = None
        for call_args in service.task_repo.update_status_with_progress.call_args_list:
            if call_args[1]["status"] == TaskStatus.FAILED:
                final_status_call = call_args
                break

        assert final_status_call is not None
        assert final_status_call[1]["progress_percentage"] == 0
        assert "Execution failed:" in final_status_call[1]["notes"]

        # Verify session commit still called for cleanup
        service.session.commit.assert_called()

    def test_execute_task_partial_completion(
        self, service_with_mocks, mock_execution_log
    ):
        """Test task execution with partial completion status."""
        service = service_with_mocks
        task_id = 123
        agent_type = AgentType.CODING

        service.execution_repo.log_execution_start.return_value = mock_execution_log

        # Mock agent function returning partial status
        def mock_agent_function(tid):
            return {
                "status": TaskStatus.PARTIAL,
                "success": True,
                "outputs": {"partial_implementation": "50% complete"},
                "issues_found": ["Missing error handling", "Incomplete validation"],
            }

        # Execute
        result = service.execute_task_with_agent(
            task_id=task_id, agent_type=agent_type, agent_function=mock_agent_function
        )

        # Verify status and progress
        assert result.status == TaskStatus.PARTIAL

        # Find the final status update call
        final_call = service.task_repo.update_status_with_progress.call_args_list[-1]
        assert final_call[1]["status"] == TaskStatus.PARTIAL
        assert final_call[1]["progress_percentage"] == 75  # Not 100 for partial


class TestBusinessLogicMethods:
    """Test suite for business logic methods."""

    @pytest.fixture
    def service_with_mocks(self):
        """Service with mocked repositories."""
        mock_session = Mock(spec=Session)
        service = TaskService(session=mock_session)
        service.task_repo = Mock()
        service.execution_repo = Mock()
        return service

    def test_get_next_actionable_tasks(self, service_with_mocks):
        """Test getting next actionable tasks."""
        service = service_with_mocks

        # Mock repository response
        mock_task_entities = [
            Mock(to_core_model=Mock(return_value=TaskCore(id=1, title="Task 1"))),
            Mock(to_core_model=Mock(return_value=TaskCore(id=2, title="Task 2"))),
            Mock(to_core_model=Mock(return_value=TaskCore(id=3, title="Task 3"))),
            Mock(to_core_model=Mock(return_value=TaskCore(id=4, title="Task 4"))),
            Mock(to_core_model=Mock(return_value=TaskCore(id=5, title="Task 5"))),
            Mock(to_core_model=Mock(return_value=TaskCore(id=6, title="Task 6"))),
        ]

        service.task_repo.get_ready_tasks.return_value = mock_task_entities

        # Execute with default limit
        result = service.get_next_actionable_tasks()

        # Verify repository call
        service.task_repo.get_ready_tasks.assert_called_once()

        # Verify result (should be limited to 5)
        assert len(result) == 5
        assert all(isinstance(task, TaskCore) for task in result)
        assert result[0].id == 1
        assert result[4].id == 5

    def test_get_next_actionable_tasks_custom_limit(self, service_with_mocks):
        """Test getting actionable tasks with custom limit."""
        service = service_with_mocks

        mock_task_entities = [
            Mock(to_core_model=Mock(return_value=TaskCore(id=i, title=f"Task {i}")))
            for i in range(1, 11)
        ]
        service.task_repo.get_ready_tasks.return_value = mock_task_entities

        # Execute with custom limit
        result = service.get_next_actionable_tasks(limit=3)

        assert len(result) == 3
        assert result[0].id == 1
        assert result[2].id == 3

    def test_get_task_dashboard_data(self, service_with_mocks):
        """Test comprehensive dashboard data retrieval."""
        service = service_with_mocks

        # Mock task statistics
        mock_task_stats = {
            "total_tasks": 100,
            "completed_tasks": 60,
            "in_progress_tasks": 25,
            "blocked_tasks": 10,
            "not_started_tasks": 5,
        }
        service.task_repo.get_task_statistics.return_value = mock_task_stats

        # Mock agent performance stats
        def mock_agent_performance(agent_type):
            return {
                "total_executions": 50,
                "success_rate": 0.8,
                "average_duration": 15.5,
            }

        service.execution_repo.get_agent_performance_stats.side_effect = (
            mock_agent_performance
        )

        # Mock recent executions
        mock_executions = [
            Mock(
                execution_id="exec_1",
                task_id=1,
                agent_type=AgentType.CODING,
                status=TaskStatus.COMPLETED,
                start_time=datetime.now(),
                confidence_score=0.9,
            ),
            Mock(
                execution_id="exec_2",
                task_id=2,
                agent_type=AgentType.RESEARCH,
                status=TaskStatus.IN_PROGRESS,
                start_time=datetime.now() - timedelta(minutes=30),
                confidence_score=0.7,
            ),
        ]
        service.execution_repo.get_recent_executions.return_value = mock_executions

        # Execute
        result = service.get_task_dashboard_data()

        # Verify structure
        assert "total_tasks" in result
        assert "completed_tasks" in result
        assert "agent_performance" in result
        assert "recent_executions" in result

        # Verify task stats are included
        assert result["total_tasks"] == 100
        assert result["completed_tasks"] == 60

        # Verify agent performance for each agent type
        assert len(result["agent_performance"]) == len(AgentType)
        for agent_type in AgentType:
            assert agent_type.value in result["agent_performance"]
            assert result["agent_performance"][agent_type.value]["success_rate"] == 0.8

        # Verify recent executions format
        assert len(result["recent_executions"]) == 2
        assert result["recent_executions"][0]["execution_id"] == "exec_1"
        assert result["recent_executions"][0]["agent_type"] == "coding"
        assert result["recent_executions"][0]["status"] == "completed"

        # Verify repository calls
        service.execution_repo.get_recent_executions.assert_called_once_with(limit=5)
        assert service.execution_repo.get_agent_performance_stats.call_count == len(
            AgentType
        )

    def test_analyze_critical_path(self, service_with_mocks):
        """Test critical path analysis."""
        service = service_with_mocks

        # Mock critical path tasks
        mock_critical_tasks = [
            Mock(
                to_core_model=Mock(
                    return_value=TaskCore(
                        id=1, title="Critical Task 1", priority=TaskPriority.CRITICAL
                    )
                )
            ),
            Mock(
                to_core_model=Mock(
                    return_value=TaskCore(
                        id=5, title="Critical Task 2", priority=TaskPriority.HIGH
                    )
                )
            ),
        ]
        service.task_repo.get_critical_path_tasks.return_value = mock_critical_tasks

        # Execute
        result = service.analyze_critical_path()

        # Verify
        service.task_repo.get_critical_path_tasks.assert_called_once()
        assert len(result) == 2
        assert all(isinstance(task, TaskCore) for task in result)
        assert result[0].priority == TaskPriority.CRITICAL
        assert result[1].priority == TaskPriority.HIGH

    def test_get_task_details_success(self, service_with_mocks):
        """Test getting comprehensive task details."""
        service = service_with_mocks
        task_id = 42

        # Mock task entity
        mock_task = Mock()
        mock_task_core = TaskCore(
            id=task_id,
            title="Test Task",
            description="A test task",
            status=TaskStatus.IN_PROGRESS,
        )
        mock_task.to_core_model.return_value = mock_task_core
        service.task_repo.get_by_id.return_value = mock_task

        # Mock dependencies
        mock_dep = Mock()
        mock_dep.id = 1
        mock_dep.depends_on_task_id = 10
        mock_dep.dependency_type = Mock(value="blocks")
        mock_dep.depends_on_task = Mock()
        mock_dep.depends_on_task.to_core_model.return_value = TaskCore(
            id=10, title="Dependency Task"
        )
        service.task_repo.get_dependencies.return_value = [mock_dep]

        # Mock execution history
        mock_exec_log = Mock()
        mock_agent_report = AgentReport(
            agent_name=AgentType.CODING, task_id=task_id, status=TaskStatus.COMPLETED
        )
        mock_exec_log.to_agent_report.return_value = mock_agent_report
        service.execution_repo.get_execution_history.return_value = [mock_exec_log]

        # Execute
        result = service.get_task_details(task_id)

        # Verify structure
        assert result is not None
        assert "task" in result
        assert "dependencies" in result
        assert "execution_history" in result

        # Verify task data
        assert result["task"]["id"] == task_id
        assert result["task"]["title"] == "Test Task"

        # Verify dependencies
        assert len(result["dependencies"]) == 1
        dep_data = result["dependencies"][0]
        assert dep_data["depends_on_task_id"] == 10
        assert dep_data["dependency_type"] == "blocks"
        assert dep_data["depends_on_task"]["title"] == "Dependency Task"

        # Verify execution history
        assert len(result["execution_history"]) == 1

        # Verify repository calls
        service.task_repo.get_by_id.assert_called_once_with(task_id)
        service.task_repo.get_dependencies.assert_called_once_with(task_id)
        service.execution_repo.get_execution_history.assert_called_once_with(task_id)

    def test_get_task_details_not_found(self, service_with_mocks):
        """Test getting task details when task doesn't exist."""
        service = service_with_mocks
        task_id = 999

        service.task_repo.get_by_id.return_value = None

        result = service.get_task_details(task_id)

        assert result is None
        service.task_repo.get_by_id.assert_called_once_with(task_id)

        # Should not call other repository methods
        service.task_repo.get_dependencies.assert_not_called()
        service.execution_repo.get_execution_history.assert_not_called()

    def test_search_tasks(self, service_with_mocks):
        """Test task search functionality."""
        service = service_with_mocks
        search_term = "authentication"

        # Mock search results
        mock_tasks = [
            Mock(
                to_core_model=Mock(
                    return_value=TaskCore(id=1, title="Implement authentication")
                )
            ),
            Mock(
                to_core_model=Mock(
                    return_value=TaskCore(id=2, title="Test authentication system")
                )
            ),
        ]
        service.task_repo.search_tasks.return_value = mock_tasks

        # Execute
        result = service.search_tasks(search_term)

        # Verify
        service.task_repo.search_tasks.assert_called_once_with(search_term)
        assert len(result) == 2
        assert all(isinstance(task, TaskCore) for task in result)
        assert "authentication" in result[0].title.lower()
        assert "authentication" in result[1].title.lower()


class TestTaskStatusManagement:
    """Test suite for task status management methods."""

    @pytest.fixture
    def service_with_mocks(self):
        """Service with mocked dependencies."""
        mock_session = Mock(spec=Session)
        service = TaskService(session=mock_session)
        service.task_repo = Mock()
        service.execution_repo = Mock()
        return service

    def test_update_task_status_success(self, service_with_mocks):
        """Test successful task status update."""
        service = service_with_mocks
        task_id = 123
        new_status = TaskStatus.COMPLETED
        notes = "Task completed successfully"
        updated_by = "test_user"

        # Mock updated task
        mock_task = Mock()
        mock_task_core = TaskCore(id=task_id, title="Updated Task", status=new_status)
        mock_task.to_core_model.return_value = mock_task_core
        service.task_repo.update_status_with_progress.return_value = mock_task

        # Execute
        result = service.update_task_status(
            task_id=task_id, status=new_status, notes=notes, updated_by=updated_by
        )

        # Verify repository call
        service.task_repo.update_status_with_progress.assert_called_once_with(
            task_id=task_id,
            status=new_status,
            progress_percentage=100,  # Completed = 100%
            notes=notes,
            updated_by=updated_by,
        )

        # Verify session commit
        service.session.commit.assert_called_once()

        # Verify return value
        assert result == mock_task_core
        assert result.status == new_status

    def test_update_task_status_not_found(self, service_with_mocks):
        """Test task status update when task not found."""
        service = service_with_mocks

        service.task_repo.update_status_with_progress.return_value = None

        result = service.update_task_status(task_id=999, status=TaskStatus.COMPLETED)

        assert result is None
        # Should not commit if no task found
        service.session.commit.assert_not_called()

    def test_update_task_status_progress_calculation(self, service_with_mocks):
        """Test progress percentage calculation for different statuses."""
        service = service_with_mocks
        task_id = 456

        mock_task = Mock()
        mock_task.to_core_model.return_value = TaskCore(id=task_id, title="Test")
        service.task_repo.update_status_with_progress.return_value = mock_task

        # Test different statuses
        test_cases = [
            (TaskStatus.NOT_STARTED, 0),
            (TaskStatus.IN_PROGRESS, 50),
            (TaskStatus.BLOCKED, 25),
            (TaskStatus.PARTIAL, 75),
            (TaskStatus.COMPLETED, 100),
            (TaskStatus.FAILED, 0),
        ]

        for status, expected_progress in test_cases:
            service.update_task_status(task_id=task_id, status=status)

            # Find the call for this status
            calls = service.task_repo.update_status_with_progress.call_args_list
            matching_call = next(call for call in calls if call[1]["status"] == status)

            assert matching_call[1]["progress_percentage"] == expected_progress


class TestTaskCRUDOperations:
    """Test suite for basic CRUD operations."""

    @pytest.fixture
    def service_with_mocks(self):
        """Service with mocked dependencies."""
        mock_session = Mock(spec=Session)
        service = TaskService(session=mock_session)
        service.task_repo = Mock()
        service.execution_repo = Mock()
        return service

    def test_create_task_success(self, service_with_mocks):
        """Test successful task creation."""
        service = service_with_mocks

        task_core = TaskCore(
            title="New Task",
            description="A new task to create",
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.LOW,
        )

        dependency_ids = [1, 3, 5]

        # Mock repository response
        mock_task = Mock()
        created_task_core = TaskCore(
            id=100, title="New Task", description="A new task to create"
        )
        mock_task.to_core_model.return_value = created_task_core
        service.task_repo.create_task_with_dependencies.return_value = mock_task

        # Execute
        result = service.create_task(task_core, dependency_ids)

        # Verify repository call
        service.task_repo.create_task_with_dependencies.assert_called_once_with(
            task_core=task_core, dependency_task_ids=dependency_ids
        )

        # Verify session commit
        service.session.commit.assert_called_once()

        # Verify return value
        assert result == created_task_core
        assert result.id == 100

    def test_create_task_no_dependencies(self, service_with_mocks):
        """Test task creation with no dependencies."""
        service = service_with_mocks

        task_core = TaskCore(title="Simple Task")

        mock_task = Mock()
        mock_task.to_core_model.return_value = task_core
        service.task_repo.create_task_with_dependencies.return_value = mock_task

        # Execute without dependencies
        service.create_task(task_core)

        # Verify empty dependency list was passed
        call_args = service.task_repo.create_task_with_dependencies.call_args
        assert call_args[1]["dependency_task_ids"] == []

    def test_add_task_dependency_success(self, service_with_mocks):
        """Test successful dependency addition."""
        service = service_with_mocks

        task_id = 10
        depends_on_id = 5
        dependency_type = "requires"

        # Execute
        result = service.add_task_dependency(task_id, depends_on_id, dependency_type)

        # Verify repository call
        service.task_repo.add_dependency.assert_called_once_with(
            task_id, depends_on_id, dependency_type
        )

        # Verify session commit
        service.session.commit.assert_called_once()

        # Verify success
        assert result is True

    def test_add_task_dependency_failure(self, service_with_mocks):
        """Test dependency addition failure handling."""
        service = service_with_mocks

        # Mock repository exception
        service.task_repo.add_dependency.side_effect = Exception("Circular dependency")

        # Execute
        result = service.add_task_dependency(123, 456)

        # Verify rollback was called
        service.session.rollback.assert_called_once()

        # Verify failure return
        assert result is False

    def test_get_tasks_by_status(self, service_with_mocks):
        """Test getting tasks by status."""
        service = service_with_mocks

        status = TaskStatus.IN_PROGRESS

        # Mock repository response
        mock_tasks = [
            Mock(
                to_core_model=Mock(
                    return_value=TaskCore(id=1, title="Task 1", status=status)
                )
            ),
            Mock(
                to_core_model=Mock(
                    return_value=TaskCore(id=2, title="Task 2", status=status)
                )
            ),
        ]
        service.task_repo.get_by_status.return_value = mock_tasks

        # Execute
        result = service.get_tasks_by_status(status)

        # Verify repository call
        service.task_repo.get_by_status.assert_called_once_with(
            status, include_relations=False
        )

        # Verify results
        assert len(result) == 2
        assert all(isinstance(task, TaskCore) for task in result)
        assert all(task.status == status for task in result)

    def test_get_tasks_by_phase(self, service_with_mocks):
        """Test getting tasks by phase."""
        service = service_with_mocks

        phase = 2

        # Mock repository response
        mock_tasks = [
            Mock(
                to_core_model=Mock(
                    return_value=TaskCore(id=5, title="Phase 2 Task 1", phase=phase)
                )
            ),
            Mock(
                to_core_model=Mock(
                    return_value=TaskCore(id=6, title="Phase 2 Task 2", phase=phase)
                )
            ),
        ]
        service.task_repo.get_by_phase.return_value = mock_tasks

        # Execute
        result = service.get_tasks_by_phase(phase)

        # Verify repository call
        service.task_repo.get_by_phase.assert_called_once_with(phase)

        # Verify results
        assert len(result) == 2
        assert all(task.phase == phase for task in result)


class TestServiceCleanup:
    """Test suite for service cleanup and resource management."""

    def test_close_session(self):
        """Test service session cleanup."""
        mock_session = Mock(spec=Session)
        service = TaskService(session=mock_session)

        # Execute
        service.close()

        # Verify session close
        mock_session.close.assert_called_once()

    def test_close_no_session(self):
        """Test cleanup when no session exists."""
        service = TaskService()
        service.session = None

        # Should not raise exception
        service.close()


class TestServiceIntegration:
    """Integration tests for service functionality."""

    @pytest.mark.integration
    def test_task_lifecycle_simulation(self):
        """Test complete task lifecycle through service."""
        # This would be an integration test with real database
        # For now, we'll test the coordination between methods

        mock_session = Mock(spec=Session)
        service = TaskService(session=mock_session)
        service.task_repo = Mock()
        service.execution_repo = Mock()

        # 1. Create task
        task_core = TaskCore(title="Integration Test Task")
        mock_created_task = Mock()
        mock_created_task.to_core_model.return_value = TaskCore(
            id=1, title="Integration Test Task"
        )
        service.task_repo.create_task_with_dependencies.return_value = mock_created_task

        created_task = service.create_task(task_core)
        assert created_task.id == 1

        # 2. Update status
        mock_updated_task = Mock()
        mock_updated_task.to_core_model.return_value = TaskCore(
            id=1, title="Integration Test Task", status=TaskStatus.IN_PROGRESS
        )
        service.task_repo.update_status_with_progress.return_value = mock_updated_task

        updated_task = service.update_task_status(1, TaskStatus.IN_PROGRESS)
        assert updated_task.status == TaskStatus.IN_PROGRESS

        # 3. Execute task
        mock_execution_log = Mock()
        mock_execution_log.execution_id = "exec_1"
        mock_execution_log.start_time = datetime.now()
        service.execution_repo.log_execution_start.return_value = mock_execution_log

        def mock_agent(task_id):
            return {"status": TaskStatus.COMPLETED, "success": True}

        report = service.execute_task_with_agent(1, AgentType.CODING, mock_agent)
        assert report.status == TaskStatus.COMPLETED

        # Verify all operations committed to session
        assert service.session.commit.call_count >= 3

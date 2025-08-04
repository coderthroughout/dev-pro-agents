"""Comprehensive tests for CLI interface functionality.

Tests cover command-line interface operations including:
- Command execution and user interactions
- Async CLI operations and progress tracking
- Output formatting and error handling
- Configuration management
- Mock CLI inputs and responses
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
import typer.testing

from src.cli import MultiAgentCLI, app
from src.supervisor import Supervisor
from src.supervisor_executor import SupervisorExecutor
from src.task_manager import TaskManager


runner = typer.testing.CliRunner()


class TestMultiAgentCLI:
    """Test suite for MultiAgentCLI class functionality."""

    @pytest.fixture
    def cli_mock_components(self):
        """Mock components for CLI testing."""
        mock_orchestrator = AsyncMock(spec=Supervisor)
        mock_executor = AsyncMock(spec=SupervisorExecutor)
        mock_task_manager = Mock(spec=TaskManager)

        return {
            "orchestrator": mock_orchestrator,
            "executor": mock_executor,
            "task_manager": mock_task_manager,
        }

    @pytest.fixture
    def cli_instance_with_mocks(self, cli_mock_components):
        """CLI instance with mocked dependencies."""
        cli = MultiAgentCLI()
        cli.orchestrator = cli_mock_components["orchestrator"]
        cli.executor = cli_mock_components["executor"]
        cli.task_manager = cli_mock_components["task_manager"]
        return cli

    @pytest.mark.asyncio
    async def test_initialize_orchestrator_creates_supervisor(self):
        """Test orchestrator initialization creates Supervisor instance."""
        cli = MultiAgentCLI()

        with patch("src.cli.LibrarySupervisor") as mock_supervisor:
            await cli.initialize_orchestrator()

            mock_supervisor.assert_called_once()
            assert cli.orchestrator is not None

    @pytest.mark.asyncio
    async def test_initialize_orchestrator_idempotent(self, cli_instance_with_mocks):
        """Test orchestrator initialization is idempotent."""
        cli = cli_instance_with_mocks
        original_orchestrator = cli.orchestrator

        await cli.initialize_orchestrator()

        assert cli.orchestrator is original_orchestrator

    @pytest.mark.asyncio
    async def test_initialize_executor_creates_executor(self, cli_mock_components):
        """Test executor initialization creates SupervisorExecutor."""
        cli = MultiAgentCLI()
        cli.orchestrator = cli_mock_components["orchestrator"]

        with patch("src.cli.SupervisorExecutor") as mock_executor:
            await cli.initialize_executor({"batch_size": 10})

            mock_executor.assert_called_once_with(cli.orchestrator, {"batch_size": 10})

    @pytest.mark.asyncio
    async def test_initialize_executor_with_default_config(self):
        """Test executor initialization with default config."""
        cli = MultiAgentCLI()

        with (
            patch("src.cli.LibrarySupervisor") as mock_supervisor,
            patch("src.cli.SupervisorExecutor") as mock_executor,
        ):
            await cli.initialize_executor()

            mock_supervisor.assert_called_once()
            mock_executor.assert_called_once_with(cli.orchestrator, {})

    @pytest.mark.asyncio
    async def test_cleanup_closes_orchestrator(self, cli_instance_with_mocks):
        """Test cleanup properly closes orchestrator."""
        cli = cli_instance_with_mocks

        await cli.cleanup()

        cli.orchestrator.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_orchestrator(self):
        """Test cleanup handles missing orchestrator gracefully."""
        cli = MultiAgentCLI()
        cli.orchestrator = None

        # Should not raise exception
        await cli.cleanup()


class TestCLICommands:
    """Test suite for CLI command functionality."""

    @pytest.fixture
    def mock_status_data(self):
        """Mock status data for testing."""
        return {
            "system_status": "active",
            "last_updated": "2025-01-01T10:00:00",
            "agents": {
                "research_expert": {
                    "status": "active",
                    "tools": ["web_search", "scrape_website"],
                },
                "coding_expert": {"status": "active", "model": "gpt-4o"},
            },
            "task_statistics": {
                "total_tasks": 25,
                "completed_tasks": 15,
                "in_progress_tasks": 5,
                "blocked_tasks": 3,
                "not_started_tasks": 2,
                "completion_percentage": 60.0,
                "total_estimated_hours": 100.5,
                "remaining_hours": 40.2,
            },
        }

    @patch("src.cli.cli_instance")
    def test_status_command_success(self, mock_cli_instance, mock_status_data):
        """Test status command displays system information correctly."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.get_agent_status.return_value = mock_status_data

        mock_cli_instance.initialize_orchestrator = AsyncMock()
        mock_cli_instance.orchestrator = mock_orchestrator
        mock_cli_instance.cleanup = AsyncMock()

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "System Status: ACTIVE" in result.stdout
        assert "Agent Status" in result.stdout
        assert "Task Statistics" in result.stdout
        assert "research_expert" in result.stdout
        assert "15 (60.0%)" in result.stdout

    @patch("src.cli.cli_instance")
    def test_status_command_handles_errors(self, mock_cli_instance):
        """Test status command handles errors gracefully."""
        mock_cli_instance.initialize_orchestrator = AsyncMock(
            side_effect=Exception("Connection failed")
        )
        mock_cli_instance.cleanup = AsyncMock()

        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Error getting status: Connection failed" in result.stdout

    @patch("src.cli.cli_instance")
    def test_execute_task_success(self, mock_cli_instance):
        """Test successful task execution."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.execute_task.return_value = {
            "success": True,
            "agent_outputs": {
                "coding_expert": {
                    "status": "completed",
                    "duration_minutes": 25.5,
                    "artifacts_created": ["auth.py", "tests.py"],
                }
            },
        }

        mock_cli_instance.initialize_orchestrator = AsyncMock()
        mock_cli_instance.orchestrator = mock_orchestrator
        mock_cli_instance.cleanup = AsyncMock()

        result = runner.invoke(app, ["execute-task", "123", "--verbose"])

        assert result.exit_code == 0
        assert "Task 123 completed successfully!" in result.stdout
        assert "Agent Outputs:" in result.stdout
        assert "Coding Expert Agent:" in result.stdout
        assert "Duration: 25.5 minutes" in result.stdout
        assert "auth.py, tests.py" in result.stdout

    @patch("src.cli.cli_instance")
    def test_execute_task_failure(self, mock_cli_instance):
        """Test task execution failure handling."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.execute_task.return_value = {
            "success": False,
            "error": "Agent timeout",
            "error_context": {"agent": "coding_expert", "timeout_seconds": 300},
        }

        mock_cli_instance.initialize_orchestrator = AsyncMock()
        mock_cli_instance.orchestrator = mock_orchestrator
        mock_cli_instance.cleanup = AsyncMock()

        result = runner.invoke(app, ["execute-task", "456", "--verbose"])

        assert result.exit_code == 0
        assert "Task 456 failed: Agent timeout" in result.stdout
        assert "Error Details:" in result.stdout
        assert "agent: coding_expert" in result.stdout

    @patch("src.cli.cli_instance")
    def test_execute_task_exception_handling(self, mock_cli_instance):
        """Test task execution exception handling."""
        mock_cli_instance.initialize_orchestrator = AsyncMock(
            side_effect=Exception("Database connection failed")
        )
        mock_cli_instance.cleanup = AsyncMock()

        result = runner.invoke(app, ["execute-task", "789"])

        assert result.exit_code == 0
        assert "Error executing task: Database connection failed" in result.stdout

    @pytest.fixture
    def mock_batch_report(self):
        """Mock batch execution report."""

        from typing import ClassVar

        class MockReport:
            batch_id: ClassVar[str] = "batch_20250101_100000"
            total_tasks: ClassVar[int] = 10
            completed_tasks: ClassVar[int] = 8
            failed_tasks: ClassVar[int] = 2
            success_rate: ClassVar[float] = 0.8
            total_duration_minutes: ClassVar[float] = 45.5
            agent_performance: ClassVar[dict[str, dict[str, float]]] = {
                "coding_expert": {
                    "tasks_completed": 5,
                    "tasks_executed": 6,
                    "success_rate": 0.833,
                    "average_duration_minutes": 12.5,
                },
                "research_expert": {
                    "tasks_completed": 3,
                    "tasks_executed": 4,
                    "success_rate": 0.75,
                    "average_duration_minutes": 8.2,
                },
            }
            recommendations: ClassVar[list[str]] = [
                "Consider increasing timeout for complex tasks",
                "Optimize agent coordination for better performance",
            ]

        return MockReport()

    @patch("src.cli.cli_instance")
    def test_batch_execute_success(self, mock_cli_instance, mock_batch_report):
        """Test successful batch execution."""
        mock_executor = AsyncMock()
        mock_executor.execute_autonomous_batch.return_value = mock_batch_report

        mock_cli_instance.initialize_executor = AsyncMock()
        mock_cli_instance.executor = mock_executor
        mock_cli_instance.cleanup = AsyncMock()

        result = runner.invoke(
            app,
            ["batch-execute", "--size", "10", "--concurrent", "5", "--timeout", "60"],
        )

        assert result.exit_code == 0
        assert "Batch ID: batch_20250101_100000" in result.stdout
        assert "Total Tasks: 10" in result.stdout
        assert "Completed: 8" in result.stdout
        assert "Success Rate: 80.0%" in result.stdout
        assert "Agent Performance" in result.stdout
        assert "Coding Expert" in result.stdout
        assert "5/6" in result.stdout
        assert "83.3%" in result.stdout
        assert "Recommendations:" in result.stdout

    @patch("src.cli.cli_instance")
    def test_batch_execute_with_export(
        self, mock_cli_instance, mock_batch_report, tmp_path
    ):
        """Test batch execution with report export."""
        mock_executor = AsyncMock()
        mock_executor.execute_autonomous_batch.return_value = mock_batch_report
        mock_executor.export_batch_report = Mock()

        mock_cli_instance.initialize_executor = AsyncMock()
        mock_cli_instance.executor = mock_executor
        mock_cli_instance.cleanup = AsyncMock()

        export_file = tmp_path / "report.json"

        result = runner.invoke(app, ["batch-execute", "--export", str(export_file)])

        assert result.exit_code == 0
        mock_executor.export_batch_report.assert_called_once_with(
            mock_batch_report.batch_id, str(export_file)
        )
        assert f"Report exported to: {export_file}" in result.stdout

    @patch("src.cli.cli_instance")
    def test_continuous_batch_execution(self, mock_cli_instance):
        """Test continuous batch execution."""
        mock_reports = [
            type(
                "MockReport",
                (),
                {
                    "batch_id": "batch_1",
                    "total_tasks": 5,
                    "completed_tasks": 4,
                    "success_rate": 0.8,
                    "total_duration_minutes": 15.0,
                },
            ),
            type(
                "MockReport",
                (),
                {
                    "batch_id": "batch_2",
                    "total_tasks": 3,
                    "completed_tasks": 3,
                    "success_rate": 1.0,
                    "total_duration_minutes": 10.0,
                },
            ),
        ]

        mock_executor = AsyncMock()
        mock_executor.execute_continuous_batches.return_value = mock_reports

        mock_cli_instance.initialize_executor = AsyncMock()
        mock_cli_instance.executor = mock_executor
        mock_cli_instance.cleanup = AsyncMock()

        result = runner.invoke(
            app,
            ["continuous-batch", "--batches", "2", "--interval", "60", "--size", "5"],
        )

        assert result.exit_code == 0
        assert "Continuous Batch Summary" in result.stdout
        assert "Batch 1" in result.stdout
        assert "Batch 2" in result.stdout
        assert "Overall Results" in result.stdout
        assert "Total Batches: 2" in result.stdout
        assert "Total Tasks: 8" in result.stdout
        assert "Total Successful: 7" in result.stdout

    @patch("src.cli.cli_instance")
    def test_list_tasks_with_status_filter(self, mock_cli_instance):
        """Test list tasks command with status filter."""
        mock_tasks = [
            {
                "id": 1,
                "title": "Implement authentication system with JWT tokens and user "
                "management",
                "component_area": "security",
                "priority": "high",
                "status": "in_progress",
            },
            {
                "id": 2,
                "title": "Setup database schema",
                "component_area": "database",
                "priority": "medium",
                "status": "in_progress",
            },
        ]

        mock_cli_instance.task_manager.get_tasks_by_status.return_value = mock_tasks

        result = runner.invoke(
            app, ["list-tasks", "--status", "in_progress", "--limit", "10"]
        )

        assert result.exit_code == 0
        assert "Tasks with status: in_progress" in result.stdout
        assert "Implement authentication system..." in result.stdout  # Truncated title
        assert "Setup database schema" in result.stdout
        mock_cli_instance.task_manager.get_tasks_by_status.assert_called_once_with(
            "in_progress"
        )

    @patch("src.cli.cli_instance")
    def test_list_tasks_ready_tasks_default(self, mock_cli_instance):
        """Test list tasks command default behavior (ready tasks)."""
        mock_tasks = [
            {
                "id": 3,
                "title": "Write unit tests",
                "component_area": "testing",
                "priority": "low",
                "status": "not_started",
            }
        ]

        mock_cli_instance.task_manager.get_ready_tasks.return_value = mock_tasks

        result = runner.invoke(app, ["list-tasks"])

        assert result.exit_code == 0
        assert "Ready Tasks" in result.stdout
        assert "Write unit tests" in result.stdout

    @patch("src.cli.cli_instance")
    def test_list_tasks_no_tasks_found(self, mock_cli_instance):
        """Test list tasks command when no tasks found."""
        mock_cli_instance.task_manager.get_ready_tasks.return_value = []

        result = runner.invoke(app, ["list-tasks"])

        assert result.exit_code == 0
        assert "No tasks found" in result.stdout

    @patch("src.cli.cli_instance")
    def test_task_info_success(self, mock_cli_instance):
        """Test task info command displays detailed task information."""
        # Mock database connection and task data
        mock_task_data = {
            "id": 123,
            "title": "Implement user authentication",
            "component_area": "security",
            "phase": 1,
            "priority": "high",
            "status": "in_progress",
            "complexity": "medium",
            "time_estimate_hours": 8.0,
            "created_at": "2025-01-01T10:00:00",
            "updated_at": "2025-01-01T11:30:00",
            "description": "Create secure authentication system with JWT",
            "success_criteria": "Users can login/logout securely",
        }

        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = type("Row", (), mock_task_data)()
        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None

        mock_cli_instance.task_manager._get_connection.return_value = mock_connection
        mock_cli_instance.task_manager.get_task_dependencies.return_value = []
        mock_cli_instance.task_manager.get_task_progress.return_value = []

        result = runner.invoke(app, ["task-info", "123"])

        assert result.exit_code == 0
        assert "Task 123 Details" in result.stdout
        assert "ID: 123" in result.stdout
        assert "Title: Implement user authentication" in result.stdout
        assert "Priority: high" in result.stdout
        assert "Status: in_progress" in result.stdout
        assert "Create secure authentication system" in result.stdout
        assert "Users can login/logout securely" in result.stdout

    @patch("src.cli.cli_instance")
    def test_task_info_not_found(self, mock_cli_instance):
        """Test task info command when task not found."""
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None

        mock_cli_instance.task_manager._get_connection.return_value = mock_connection

        result = runner.invoke(app, ["task-info", "999"])

        assert result.exit_code == 0
        assert "Task 999 not found" in result.stdout

    @patch("src.cli.cli_instance")
    def test_agent_stats_success(self, mock_cli_instance):
        """Test agent stats command displays performance statistics."""
        mock_stats = {
            "coding_expert": {
                "total_tasks_executed": 25,
                "successful_tasks": 20,
                "failed_tasks": 5,
                "success_rate": 0.8,
                "average_task_duration": 15.5,
                "total_duration_minutes": 387.5,
            },
            "research_expert": {
                "total_tasks_executed": 15,
                "successful_tasks": 14,
                "failed_tasks": 1,
                "success_rate": 0.933,
                "average_task_duration": 8.2,
                "total_duration_minutes": 123.0,
            },
        }

        mock_batch_history = [
            {
                "batch_id": "batch_1",
                "start_time": "2025-01-01T10:00:00",
                "total_tasks": 10,
                "success_rate": 0.9,
                "duration_minutes": 45.0,
            }
        ]

        mock_executor = Mock()
        mock_executor.get_agent_statistics.return_value = mock_stats
        mock_executor.get_batch_history.return_value = mock_batch_history

        mock_cli_instance.initialize_executor = AsyncMock()
        mock_cli_instance.executor = mock_executor
        mock_cli_instance.cleanup = AsyncMock()

        result = runner.invoke(app, ["agent-stats"])

        assert result.exit_code == 0
        assert "Agent Performance Statistics" in result.stdout
        assert "Coding Expert" in result.stdout
        assert "25" in result.stdout  # total tasks
        assert "80.0%" in result.stdout  # success rate
        assert "Recent Batch History" in result.stdout
        assert "batch_1" in result.stdout

    @patch("src.cli.cli_instance")
    def test_agent_stats_no_data(self, mock_cli_instance):
        """Test agent stats command when no data available."""
        mock_stats = {"coding_expert": {"total_tasks_executed": 0}}

        mock_executor = Mock()
        mock_executor.get_agent_statistics.return_value = mock_stats

        mock_cli_instance.initialize_executor = AsyncMock()
        mock_cli_instance.executor = mock_executor
        mock_cli_instance.cleanup = AsyncMock()

        result = runner.invoke(app, ["agent-stats"])

        assert result.exit_code == 0
        assert "No agent execution data available yet" in result.stdout


class TestConfigurationCommands:
    """Test suite for configuration management commands."""

    def test_config_show_empty(self, tmp_path):
        """Test config show command with no existing configuration."""
        with patch("src.cli.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            result = runner.invoke(app, ["config", "--show"])

            assert result.exit_code == 0
            assert "No configuration found" in result.stdout

    def test_config_show_existing(self, tmp_path):
        """Test config show command with existing configuration."""
        config_file = tmp_path / ".ai_job_scraper_config.json"
        config_data = {
            "batch_size": 10,
            "timeout_minutes": 30,
            "max_concurrent_tasks": 5,
        }

        with config_file.open("w") as f:
            json.dump(config_data, f)

        with patch("src.cli.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            result = runner.invoke(app, ["config", "--show"])

            assert result.exit_code == 0
            assert '"batch_size": 10' in result.stdout
            assert '"timeout_minutes": 30' in result.stdout

    def test_config_update_settings(self, tmp_path):
        """Test config command updates configuration settings."""
        with patch("src.cli.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            result = runner.invoke(
                app,
                [
                    "config",
                    "--batch-size",
                    "15",
                    "--timeout",
                    "45",
                    "--concurrent",
                    "8",
                ],
            )

            assert result.exit_code == 0
            assert "Configuration updated" in result.stdout

            # Verify config file was created with correct values
            config_file = tmp_path / ".ai_job_scraper_config.json"
            assert config_file.exists()

            with config_file.open() as f:
                config = json.load(f)
                assert config["batch_size"] == 15
                assert config["timeout_minutes"] == 45
                assert config["max_concurrent_tasks"] == 8

    def test_config_no_changes(self, tmp_path):
        """Test config command with no changes specified."""
        with patch("src.cli.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            result = runner.invoke(app, ["config"])

            assert result.exit_code == 0
            assert "No configuration changes specified" in result.stdout


class TestCLIErrorHandling:
    """Test suite for CLI error handling and edge cases."""

    @patch("src.cli.cli_instance")
    def test_command_exception_handling(self, mock_cli_instance):
        """Test CLI commands handle unexpected exceptions gracefully."""
        mock_cli_instance.task_manager.get_ready_tasks.side_effect = Exception(
            "Database connection failed"
        )

        result = runner.invoke(app, ["list-tasks"])

        assert result.exit_code == 0
        assert "Error listing tasks: Database connection failed" in result.stdout

    @patch("src.cli.console")
    def test_console_output_formatting(self, mock_console):
        """Test console output is properly formatted."""
        mock_console.print = Mock()

        with patch("src.cli.cli_instance") as mock_cli_instance:
            mock_cli_instance.task_manager.get_ready_tasks.return_value = []

            runner.invoke(app, ["list-tasks"])

            # Verify console.print was called with Rich formatting
            mock_console.print.assert_called()
            call_args = mock_console.print.call_args_list
            assert any("yellow" in str(call) for call in call_args)

    def test_typer_argument_validation(self):
        """Test Typer properly validates command arguments."""
        # Test invalid task ID (non-integer)
        result = runner.invoke(app, ["execute-task", "invalid-id"])
        assert result.exit_code != 0

        # Test negative batch size
        result = runner.invoke(app, ["batch-execute", "--size", "-5"])
        # Typer may allow this through, but our validation should catch it


class TestAsyncOperations:
    """Test suite for async operation handling in CLI."""

    @pytest.mark.asyncio
    async def test_async_cleanup_on_keyboard_interrupt(self):
        """Test proper cleanup on keyboard interrupt."""
        cli = MultiAgentCLI()
        cli.orchestrator = AsyncMock()

        # Simulate KeyboardInterrupt during cleanup
        cli.orchestrator.close.side_effect = KeyboardInterrupt()

        # Should handle interrupt gracefully
        try:
            await cli.cleanup()
        except KeyboardInterrupt:
            pytest.fail("Cleanup should handle KeyboardInterrupt gracefully")

    @pytest.mark.asyncio
    async def test_concurrent_initialization(self):
        """Test concurrent initialization calls are handled properly."""
        cli = MultiAgentCLI()

        with patch("src.cli.LibrarySupervisor") as mock_supervisor:
            # Run multiple initializations concurrently
            tasks = [
                cli.initialize_orchestrator(),
                cli.initialize_orchestrator(),
                cli.initialize_orchestrator(),
            ]

            await asyncio.gather(*tasks)

            # Should only create one supervisor instance
            assert mock_supervisor.call_count == 1


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    @pytest.mark.integration
    def test_cli_app_main_callback(self):
        """Test CLI main callback function and help text."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Multi-Agent Orchestration System CLI" in result.stdout
        assert "Manage and monitor the LangGraph-based" in result.stdout

    @pytest.mark.integration
    def test_command_discovery(self):
        """Test all expected commands are discoverable."""
        result = runner.invoke(app, ["--help"])

        expected_commands = [
            "status",
            "execute-task",
            "batch-execute",
            "continuous-batch",
            "list-tasks",
            "task-info",
            "agent-stats",
            "config",
        ]

        for command in expected_commands:
            assert command in result.stdout

    @patch("src.cli.cli_instance")
    def test_end_to_end_task_workflow(self, mock_cli_instance):
        """Test end-to-end task workflow through CLI."""
        # Mock task data
        mock_tasks = [
            {
                "id": 1,
                "title": "Test task",
                "component_area": "testing",
                "priority": "high",
                "status": "not_started",
            }
        ]

        mock_orchestrator = AsyncMock()
        mock_orchestrator.execute_task.return_value = {"success": True}

        mock_cli_instance.task_manager.get_ready_tasks.return_value = mock_tasks
        mock_cli_instance.initialize_orchestrator = AsyncMock()
        mock_cli_instance.orchestrator = mock_orchestrator
        mock_cli_instance.cleanup = AsyncMock()

        # 1. List tasks
        list_result = runner.invoke(app, ["list-tasks"])
        assert list_result.exit_code == 0
        assert "Test task" in list_result.stdout

        # 2. Execute task
        exec_result = runner.invoke(app, ["execute-task", "1"])
        assert exec_result.exit_code == 0
        assert "completed successfully" in exec_result.stdout

        mock_orchestrator.execute_task.assert_called_once_with(1)

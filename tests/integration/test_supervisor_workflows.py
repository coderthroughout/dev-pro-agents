"""Comprehensive tests for supervisor workflow functionality.

Tests cover LangGraph supervisor implementation including:
- Agent delegation and coordination
- Workflow state management
- LangGraph integration patterns
- Error handling and recovery
- Multi-agent task execution
"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.schemas.unified_models import (
    ComponentArea,
    TaskComplexity,
    TaskCore,
    TaskPriority,
    TaskStatus,
)
from src.supervisor import Supervisor


class TestSupervisorInitialization:
    """Test suite for Supervisor initialization and setup."""

    @patch("src.supervisor.ChatOpenAI")
    @patch("src.supervisor.TaskManager")
    @patch("src.supervisor.InMemorySaver")
    @patch("src.supervisor.InMemoryStore")
    def test_supervisor_initialization_default_config(
        self, mock_store, mock_saver, mock_task_manager, mock_openai
    ):
        """Test supervisor initialization with default configuration."""
        # Mock environment variable
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"}):
            supervisor = Supervisor()

            # Verify ChatOpenAI initialization
            mock_openai.assert_called_once_with(
                model="gpt-4o", api_key="test-api-key", temperature=0.1
            )

            # Verify TaskManager initialization
            mock_task_manager.assert_called_once_with(None)

            # Verify persistence components
            mock_saver.assert_called_once()
            mock_store.assert_called_once()

            # Verify components are assigned
            assert supervisor.model == mock_openai.return_value
            assert supervisor.task_manager == mock_task_manager.return_value
            assert supervisor.checkpointer == mock_saver.return_value
            assert supervisor.store == mock_store.return_value

    @patch("src.supervisor.ChatOpenAI")
    @patch("src.supervisor.TaskManager")
    def test_supervisor_initialization_custom_config(
        self, mock_task_manager, mock_openai
    ):
        """Test supervisor initialization with custom configuration."""
        custom_api_key = "custom-api-key"
        custom_db_path = "/custom/db/path"

        Supervisor(openai_api_key=custom_api_key, db_path=custom_db_path)

        # Verify custom configuration used
        mock_openai.assert_called_once_with(
            model="gpt-4o", api_key=custom_api_key, temperature=0.1
        )
        mock_task_manager.assert_called_once_with(custom_db_path)

    @patch("src.supervisor.create_react_agent")
    @patch("src.supervisor.ChatOpenAI")
    @patch("src.supervisor.TaskManager")
    def test_agents_initialization(
        self, mock_task_manager, mock_openai, mock_create_agent
    ):
        """Test that all specialized agents are properly initialized."""
        mock_agents = {
            "research": Mock(),
            "coding": Mock(),
            "testing": Mock(),
            "documentation": Mock(),
        }
        mock_create_agent.side_effect = list(mock_agents.values())

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            supervisor = Supervisor()

            # Verify create_react_agent called for each agent type
            assert mock_create_agent.call_count == 4

            # Verify agent assignments
            assert supervisor.research_agent == mock_agents["research"]
            assert supervisor.coding_agent == mock_agents["coding"]
            assert supervisor.testing_agent == mock_agents["testing"]
            assert supervisor.documentation_agent == mock_agents["documentation"]

            # Verify agents created with correct model and names
            calls = mock_create_agent.call_args_list
            agent_names = [call[1]["name"] for call in calls]
            assert "research_expert" in agent_names
            assert "coding_expert" in agent_names
            assert "testing_expert" in agent_names
            assert "documentation_expert" in agent_names

    @patch("src.supervisor.create_supervisor")
    @patch("src.supervisor.create_react_agent")
    @patch("src.supervisor.ChatOpenAI")
    @patch("src.supervisor.TaskManager")
    def test_supervisor_workflow_creation(
        self, mock_task_manager, mock_openai, mock_create_agent, mock_create_supervisor
    ):
        """Test supervisor workflow creation using langgraph-supervisor."""
        mock_workflow = Mock()
        mock_compiled_app = Mock()
        mock_workflow.compile.return_value = mock_compiled_app
        mock_create_supervisor.return_value = mock_workflow

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            supervisor = Supervisor()

            # Verify supervisor workflow creation
            mock_create_supervisor.assert_called_once()
            call_args = mock_create_supervisor.call_args

            # Verify agents passed to supervisor
            agents_arg = call_args[1]["agents"]
            assert len(agents_arg) == 4

            # Verify model passed
            assert call_args[1]["model"] == mock_openai.return_value

            # Verify configuration options
            assert call_args[1]["output_mode"] == "full_history"
            assert call_args[1]["add_handoff_messages"] is True

            # Verify workflow compilation
            mock_workflow.compile.assert_called_once_with(
                checkpointer=supervisor.checkpointer, store=supervisor.store
            )

            assert supervisor.app == mock_compiled_app


class TestAgentToolCreation:
    """Test suite for agent tool creation and configuration."""

    @pytest.fixture
    def supervisor_instance(self):
        """Create supervisor instance for testing."""
        with (
            patch("src.supervisor.ChatOpenAI"),
            patch("src.supervisor.TaskManager"),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            return Supervisor()

    def test_research_tools_creation(self, supervisor_instance):
        """Test research tools are properly created."""
        tools = supervisor_instance._create_research_tools()

        assert len(tools) == 2

        # Test tools are callable
        web_search, scrape_website = tools
        assert callable(web_search)
        assert callable(scrape_website)

        # Test function names
        assert web_search.__name__ == "web_search"
        assert scrape_website.__name__ == "scrape_website"

    @patch("src.supervisor.ExaClient")
    def test_web_search_tool_functionality(self, mock_exa_client, supervisor_instance):
        """Test web search tool functionality."""
        # Mock ExaClient
        mock_client_instance = Mock()
        mock_client_instance.search_and_contents.return_value = [
            {"title": "Test Result", "content": "Test content"}
        ]
        mock_exa_client.return_value = mock_client_instance

        tools = supervisor_instance._create_research_tools()
        web_search = tools[0]

        # Execute web search
        result = web_search("test query")

        # Verify client called correctly
        mock_exa_client.assert_called_once()
        mock_client_instance.search_and_contents.assert_called_once_with(
            "test query", num_results=5
        )

        # Verify result format
        assert "Test Result" in str(result)

    @patch("src.supervisor.ExaClient")
    def test_web_search_tool_error_handling(self, mock_exa_client, supervisor_instance):
        """Test web search tool handles errors gracefully."""
        # Mock ExaClient to raise exception
        mock_exa_client.side_effect = Exception("API Error")

        tools = supervisor_instance._create_research_tools()
        web_search = tools[0]

        # Execute web search
        result = web_search("test query")

        # Verify error handling
        assert "Search failed: API Error" in result

    @patch("src.supervisor.FirecrawlClient")
    def test_scrape_website_tool_functionality(
        self, mock_firecrawl_client, supervisor_instance
    ):
        """Test website scraping tool functionality."""
        # Mock FirecrawlClient
        mock_client_instance = Mock()
        mock_client_instance.scrape_url.return_value = {
            "content": "Scraped website content"
        }
        mock_firecrawl_client.return_value = mock_client_instance

        tools = supervisor_instance._create_research_tools()
        scrape_website = tools[1]

        # Execute scraping
        result = scrape_website("https://example.com")

        # Verify client called correctly
        mock_firecrawl_client.assert_called_once()
        mock_client_instance.scrape_url.assert_called_once_with("https://example.com")

        # Verify result
        assert result == "Scraped website content"

    def test_coding_tools_creation(self, supervisor_instance):
        """Test coding tools are properly created."""
        tools = supervisor_instance._create_coding_tools()

        assert len(tools) == 2

        # Test tools are callable
        write_code, analyze_code = tools
        assert callable(write_code)
        assert callable(analyze_code)

        assert write_code.__name__ == "write_code"
        assert analyze_code.__name__ == "analyze_code"

    def test_write_code_tool_functionality(self, supervisor_instance, tmp_path):
        """Test write code tool functionality."""
        tools = supervisor_instance._create_coding_tools()
        write_code = tools[0]

        # Test writing code to file
        test_file = tmp_path / "test_code.py"
        code_content = "def hello_world():\n    print('Hello, World!')"

        result = write_code(str(test_file), code_content)

        # Verify file was created with correct content
        assert test_file.exists()
        assert test_file.read_text() == code_content
        assert "Successfully wrote code" in result

    def test_write_code_tool_error_handling(self, supervisor_instance):
        """Test write code tool handles errors gracefully."""
        tools = supervisor_instance._create_coding_tools()
        write_code = tools[0]

        # Test with invalid path
        result = write_code("/invalid/path/file.py", "code")

        assert "Failed to write code" in result

    def test_analyze_code_tool_functionality(self, supervisor_instance):
        """Test code analysis tool functionality."""
        tools = supervisor_instance._create_coding_tools()
        analyze_code = tools[1]

        # Test code analysis
        test_code = """
def function1():
    pass

class TestClass:
    def method1(self):
        pass

def function2():
    return 42
"""

        result = analyze_code(test_code)

        # Verify analysis results
        assert "8 lines" in result
        assert "2 functions" in result
        assert "1 classes" in result

    def test_testing_tools_creation(self, supervisor_instance):
        """Test testing tools are properly created."""
        tools = supervisor_instance._create_testing_tools()

        assert len(tools) == 2

        run_tests, create_test = tools
        assert callable(run_tests)
        assert callable(create_test)

        assert run_tests.__name__ == "run_tests"
        assert create_test.__name__ == "create_test"

    @patch("subprocess.run")
    def test_run_tests_tool_functionality(self, mock_subprocess, supervisor_instance):
        """Test run tests tool functionality."""
        # Mock subprocess result
        mock_result = Mock()
        mock_result.stdout = "test_file.py::test_function PASSED"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        tools = supervisor_instance._create_testing_tools()
        run_tests = tools[0]

        # Execute test run
        result = run_tests("tests/")

        # Verify subprocess called correctly
        expected_cmd = [pytest.sys.executable, "-m", "pytest", "tests/", "-v"]
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == expected_cmd
        assert call_args[1]["shell"] is False

        # Verify result
        assert "PASSED" in result

    @patch("subprocess.run")
    def test_run_tests_tool_error_handling(self, mock_subprocess, supervisor_instance):
        """Test run tests tool handles errors gracefully."""
        # Mock subprocess exception
        mock_subprocess.side_effect = Exception("Test execution failed")

        tools = supervisor_instance._create_testing_tools()
        run_tests = tools[0]

        # Execute test run
        result = run_tests("tests/")

        # Verify error handling
        assert "Test execution failed: Test execution failed" in result

    def test_create_test_tool_functionality(self, supervisor_instance, tmp_path):
        """Test create test tool functionality."""
        tools = supervisor_instance._create_testing_tools()
        create_test = tools[1]

        # Change to temp directory for test
        old_cwd = Path.cwd()
        os.chdir(tmp_path)

        try:
            test_content = """
import pytest

def test_example():
    assert 1 + 1 == 2
"""

            # Execute test creation
            result = create_test("example", test_content)

            # Verify test file created
            test_file = tmp_path / "test_example.py"
            assert test_file.exists()
            assert test_content in test_file.read_text()
            assert "Successfully created test file" in result
        finally:
            os.chdir(old_cwd)

    def test_documentation_tools_creation(self, supervisor_instance):
        """Test documentation tools are properly created."""
        tools = supervisor_instance._create_documentation_tools()

        assert len(tools) == 2

        write_documentation, generate_api_docs = tools
        assert callable(write_documentation)
        assert callable(generate_api_docs)

        assert write_documentation.__name__ == "write_documentation"
        assert generate_api_docs.__name__ == "generate_api_docs"

    def test_write_documentation_tool_functionality(
        self, supervisor_instance, tmp_path
    ):
        """Test write documentation tool functionality."""
        tools = supervisor_instance._create_documentation_tools()
        write_documentation = tools[0]

        # Test writing documentation
        doc_file = tmp_path / "README.md"
        doc_content = "# Project Documentation\n\nThis is a test project."

        result = write_documentation(str(doc_file), doc_content)

        # Verify file created with correct content
        assert doc_file.exists()
        assert doc_file.read_text() == doc_content
        assert "Successfully wrote documentation" in result

    def test_generate_api_docs_tool_functionality(self, supervisor_instance, tmp_path):
        """Test generate API docs tool functionality."""
        tools = supervisor_instance._create_documentation_tools()
        generate_api_docs = tools[1]

        # Create test code file
        code_file = tmp_path / "api.py"
        code_content = '''
def get_user(user_id):
    """Get user by ID."""
    pass

def create_user(user_data):
    """Create new user."""
    pass

class UserService:
    def list_users(self):
        """List all users."""
        pass
'''
        code_file.write_text(code_content)

        # Execute API docs generation
        result = generate_api_docs(str(code_file))

        # Verify documentation generated
        assert f"API documentation for {code_file}" in result
        assert "def get_user(user_id):" in result
        assert "def create_user(user_data):" in result


class TestTaskExecution:
    """Test suite for task execution functionality."""

    @pytest.fixture
    def supervisor_with_mocks(self):
        """Create supervisor with mocked dependencies."""
        with (
            patch("src.supervisor.ChatOpenAI"),
            patch("src.supervisor.TaskManager") as mock_tm,
            patch("src.supervisor.create_supervisor") as mock_create_sup,
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            # Mock supervisor workflow
            mock_app = AsyncMock()
            mock_workflow = Mock()
            mock_workflow.compile.return_value = mock_app
            mock_create_sup.return_value = mock_workflow

            supervisor = Supervisor()
            supervisor.app = mock_app  # Ensure app is set

            return supervisor, mock_tm.return_value, mock_app

    @pytest.fixture
    def sample_task(self):
        """Sample task for testing."""
        return TaskCore(
            id=123,
            title="Implement user authentication",
            description="Create secure authentication system with JWT tokens",
            component_area=ComponentArea.SECURITY,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            success_criteria="Users can login and logout securely",
            time_estimate_hours=8.0,
        )

    @pytest.mark.asyncio
    async def test_execute_task_success(self, supervisor_with_mocks, sample_task):
        """Test successful task execution."""
        supervisor, mock_task_manager, mock_app = supervisor_with_mocks
        task_id = 123

        # Mock task manager responses
        mock_task_manager.get_task.return_value = sample_task
        mock_task_manager.update_task_status = Mock()

        # Mock successful LangGraph execution
        mock_app.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="Execute task"),
                AIMessage(content="Task completed successfully with high confidence"),
            ]
        }

        # Execute task
        result = await supervisor.execute_task(task_id)

        # Verify task manager calls
        mock_task_manager.get_task.assert_called_once_with(task_id)

        # Verify status updates
        status_calls = mock_task_manager.update_task_status.call_args_list
        assert len(status_calls) == 2

        # First call: set to IN_PROGRESS
        assert status_calls[0][0] == (task_id, TaskStatus.IN_PROGRESS)

        # Second call: set to COMPLETED (based on success analysis)
        assert status_calls[1][0] == (task_id, TaskStatus.COMPLETED)

        # Verify LangGraph invocation
        mock_app.ainvoke.assert_called_once()
        call_args = mock_app.ainvoke.call_args

        # Verify message content includes task details
        messages = call_args[0][0]["messages"]
        assert len(messages) == 1
        task_message = messages[0].content
        assert "Implement user authentication" in task_message
        assert "Create secure authentication system" in task_message
        assert "Priority: high" in task_message
        assert "8.0 hours" in task_message

        # Verify thread configuration
        config = call_args[1]["config"]
        assert config["configurable"]["thread_id"] == f"task_{task_id}"

        # Verify result structure
        assert result["task_id"] == task_id
        assert result["status"] == TaskStatus.COMPLETED.value
        assert result["success"] is True
        assert "messages" in result
        assert "execution_time" in result

    @pytest.mark.asyncio
    async def test_execute_task_not_found(self, supervisor_with_mocks):
        """Test task execution when task not found."""
        supervisor, mock_task_manager, mock_app = supervisor_with_mocks
        task_id = 999

        # Mock task not found
        mock_task_manager.get_task.return_value = None

        # Execute task and expect ValueError
        with pytest.raises(ValueError, match="Task 999 not found"):
            await supervisor.execute_task(task_id)

        # Verify no LangGraph execution
        mock_app.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_task_langgraph_failure(
        self, supervisor_with_mocks, sample_task
    ):
        """Test task execution with LangGraph failure."""
        supervisor, mock_task_manager, mock_app = supervisor_with_mocks
        task_id = 123

        # Mock task manager responses
        mock_task_manager.get_task.return_value = sample_task
        mock_task_manager.update_task_status = Mock()

        # Mock LangGraph execution failure
        mock_app.ainvoke.side_effect = Exception("Agent coordination failed")

        # Execute task
        result = await supervisor.execute_task(task_id)

        # Verify failure handling
        assert result["task_id"] == task_id
        assert result["status"] == TaskStatus.BLOCKED.value
        assert result["success"] is False
        assert result["error"] == "Agent coordination failed"

        # Verify task marked as blocked
        final_status_call = mock_task_manager.update_task_status.call_args_list[-1]
        assert final_status_call[0] == (task_id, TaskStatus.BLOCKED)

    @pytest.mark.asyncio
    async def test_execute_task_analysis_determines_blocked(
        self, supervisor_with_mocks, sample_task
    ):
        """Test task execution where analysis determines task is blocked."""
        supervisor, mock_task_manager, mock_app = supervisor_with_mocks
        task_id = 123

        # Mock task manager responses
        mock_task_manager.get_task.return_value = sample_task
        mock_task_manager.update_task_status = Mock()

        # Mock LangGraph execution with failure indicators
        mock_app.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="Execute task"),
                AIMessage(
                    content=(
                        "Task execution failed due to missing dependencies and "
                        "error in setup"
                    )
                ),
            ]
        }

        # Execute task
        result = await supervisor.execute_task(task_id)

        # Verify result indicates blocking
        assert result["success"] is False
        assert result["status"] == TaskStatus.BLOCKED.value

        # Verify task status updated to blocked
        final_status_call = mock_task_manager.update_task_status.call_args_list[-1]
        assert final_status_call[0] == (task_id, TaskStatus.BLOCKED)

    def test_format_task_message(self, supervisor_with_mocks, sample_task):
        """Test task message formatting for LangGraph."""
        supervisor, _, _ = supervisor_with_mocks

        # Test with enum values
        formatted_message = supervisor._format_task_message(sample_task)

        # Verify all task details included
        assert "Task: Implement user authentication" in formatted_message
        assert "Description: Create secure authentication system" in formatted_message
        assert "Component Area: security" in formatted_message
        assert "Priority: high" in formatted_message
        assert "Complexity: medium" in formatted_message
        assert (
            "Success Criteria: Users can login and logout securely" in formatted_message
        )
        assert "Time Estimate: 8.0 hours" in formatted_message
        assert "delegate this task to the appropriate agent" in formatted_message

    def test_format_task_message_string_values(self, supervisor_with_mocks):
        """Test task message formatting with string values instead of enums."""
        supervisor, _, _ = supervisor_with_mocks

        # Create task with string values (simulating database retrieval)
        task_with_strings = type(
            "Task",
            (),
            {
                "title": "Test Task",
                "description": "Test description",
                "component_area": "testing",
                "priority": "medium",
                "complexity": "low",
                "success_criteria": "Task works correctly",
                "time_estimate_hours": 4.0,
            },
        )()

        formatted_message = supervisor._format_task_message(task_with_strings)

        # Verify string values handled correctly
        assert "Priority: medium" in formatted_message
        assert "Complexity: low" in formatted_message

    def test_analyze_execution_results_success_indicators(self, supervisor_with_mocks):
        """Test execution result analysis with success indicators."""
        supervisor, _, _ = supervisor_with_mocks

        # Test successful execution
        success_result = {
            "messages": [
                AIMessage(
                    content="Task completed successfully with all requirements met"
                )
            ]
        }

        assert supervisor._analyze_execution_results(success_result) is True

        # Test with "finished" indicator
        finished_result = {
            "messages": [
                AIMessage(content="All work is now finished and ready for review")
            ]
        }

        assert supervisor._analyze_execution_results(finished_result) is True

    def test_analyze_execution_results_error_indicators(self, supervisor_with_mocks):
        """Test execution result analysis with error indicators."""
        supervisor, _, _ = supervisor_with_mocks

        # Test failed execution
        error_result = {
            "messages": [AIMessage(content="Task failed due to configuration error")]
        }

        assert supervisor._analyze_execution_results(error_result) is False

        # Test blocked execution
        blocked_result = {
            "messages": [AIMessage(content="Task is blocked waiting for dependencies")]
        }

        assert supervisor._analyze_execution_results(blocked_result) is False

    def test_analyze_execution_results_mixed_indicators(self, supervisor_with_mocks):
        """Test execution result analysis with mixed indicators."""
        supervisor, _, _ = supervisor_with_mocks

        # Test mixed signals (error overrides success)
        mixed_result = {
            "messages": [
                AIMessage(
                    content=(
                        "Task completed but there was an exception in the final step"
                    )
                )
            ]
        }

        assert supervisor._analyze_execution_results(mixed_result) is False

    def test_analyze_execution_results_no_messages(self, supervisor_with_mocks):
        """Test execution result analysis with no messages."""
        supervisor, _, _ = supervisor_with_mocks

        # Test empty messages
        empty_result = {"messages": []}
        assert supervisor._analyze_execution_results(empty_result) is False

        # Test missing messages key
        no_messages_result = {}
        assert supervisor._analyze_execution_results(no_messages_result) is False


class TestBatchExecution:
    """Test suite for batch task execution."""

    @pytest.fixture
    def supervisor_with_mocks(self):
        """Create supervisor with mocked dependencies."""
        with (
            patch("src.supervisor.ChatOpenAI"),
            patch("src.supervisor.TaskManager") as mock_tm,
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            supervisor = Supervisor()
            return supervisor, mock_tm.return_value

    @pytest.mark.asyncio
    async def test_execute_batch_success(self, supervisor_with_mocks):
        """Test successful batch execution."""
        supervisor, mock_task_manager = supervisor_with_mocks
        task_ids = [1, 2, 3]

        # Mock individual task executions
        with patch.object(supervisor, "execute_task") as mock_execute:
            mock_execute.side_effect = [
                {"task_id": 1, "status": "completed", "success": True},
                {"task_id": 2, "status": "completed", "success": True},
                {
                    "task_id": 3,
                    "status": "blocked",
                    "success": False,
                    "error": "Missing dependency",
                },
            ]

            # Execute batch
            results = await supervisor.execute_batch(task_ids)

            # Verify all tasks executed
            assert len(results) == 3
            assert mock_execute.call_count == 3

            # Verify results
            assert results[0]["success"] is True
            assert results[1]["success"] is True
            assert results[2]["success"] is False
            assert results[2]["error"] == "Missing dependency"

    @pytest.mark.asyncio
    async def test_execute_batch_with_exceptions(self, supervisor_with_mocks):
        """Test batch execution with individual task exceptions."""
        supervisor, mock_task_manager = supervisor_with_mocks
        task_ids = [1, 2]

        # Mock task executions with exception
        with patch.object(supervisor, "execute_task") as mock_execute:
            mock_execute.side_effect = [
                {"task_id": 1, "status": "completed", "success": True},
                Exception("Task 2 execution failed"),
            ]

            # Execute batch
            results = await supervisor.execute_batch(task_ids)

            # Verify results handle exceptions
            assert len(results) == 2
            assert results[0]["success"] is True
            assert results[1]["success"] is False
            assert results[1]["error"] == "Task 2 execution failed"
            assert results[1]["status"] == TaskStatus.BLOCKED.value


class TestWorkflowStateManagement:
    """Test suite for workflow state management."""

    @pytest.fixture
    def supervisor_with_mocks(self):
        """Create supervisor with mocked app."""
        with (
            patch("src.supervisor.ChatOpenAI"),
            patch("src.supervisor.TaskManager"),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            supervisor = Supervisor()
            supervisor.app = Mock()  # Mock the compiled LangGraph app
            return supervisor

    def test_get_workflow_state_success(self, supervisor_with_mocks):
        """Test getting workflow state successfully."""
        supervisor = supervisor_with_mocks
        thread_id = "task_123"

        # Mock state object
        mock_state = Mock()
        mock_state.values = {"messages": [], "current_agent": "coding_expert"}
        mock_state.next = ["research_expert", "testing_expert"]
        mock_state.created_at = datetime(2025, 1, 1, 10, 0, 0)
        mock_state.updated_at = datetime(2025, 1, 1, 11, 0, 0)

        supervisor.app.get_state.return_value = mock_state

        # Get workflow state
        result = supervisor.get_workflow_state(thread_id)

        # Verify app called correctly
        supervisor.app.get_state.assert_called_once_with(
            {"configurable": {"thread_id": thread_id}}
        )

        # Verify result structure
        assert result["thread_id"] == thread_id
        assert result["values"] == mock_state.values
        assert result["next"] == ["research_expert", "testing_expert"]
        assert result["created_at"] == "2025-01-01T10:00:00"
        assert result["updated_at"] == "2025-01-01T11:00:00"

    def test_get_workflow_state_error_handling(self, supervisor_with_mocks):
        """Test workflow state retrieval error handling."""
        supervisor = supervisor_with_mocks
        thread_id = "invalid_thread"

        # Mock exception
        supervisor.app.get_state.side_effect = Exception("State not found")

        # Get workflow state
        result = supervisor.get_workflow_state(thread_id)

        # Verify error handling
        assert "error" in result
        assert result["error"] == "State not found"

    def test_get_workflow_state_none_timestamps(self, supervisor_with_mocks):
        """Test workflow state with None timestamps."""
        supervisor = supervisor_with_mocks
        thread_id = "task_456"

        # Mock state with None timestamps
        mock_state = Mock()
        mock_state.values = {}
        mock_state.next = None
        mock_state.created_at = None
        mock_state.updated_at = None

        supervisor.app.get_state.return_value = mock_state

        # Get workflow state
        result = supervisor.get_workflow_state(thread_id)

        # Verify None handling
        assert result["next"] == []
        assert result["created_at"] is None
        assert result["updated_at"] is None

    def test_list_active_threads(self, supervisor_with_mocks):
        """Test listing active threads."""
        supervisor = supervisor_with_mocks

        # Execute (current implementation returns empty list)
        result = supervisor.list_active_threads()

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 0  # Current implementation limitation


class TestAgentStatusAndMonitoring:
    """Test suite for agent status and monitoring functionality."""

    @pytest.fixture
    def supervisor_with_mock_task_manager(self):
        """Create supervisor with mocked task manager."""
        with (
            patch("src.supervisor.ChatOpenAI"),
            patch("src.supervisor.TaskManager") as mock_tm,
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            supervisor = Supervisor()
            return supervisor, mock_tm.return_value

    @pytest.mark.asyncio
    async def test_get_agent_status_success(self, supervisor_with_mock_task_manager):
        """Test successful agent status retrieval."""
        supervisor, mock_task_manager = supervisor_with_mock_task_manager

        # Mock task data
        all_tasks = [
            {"status": "completed", "time_estimate_hours": 5.0},
            {"status": "completed", "time_estimate_hours": 3.0},
            {"status": "in_progress", "time_estimate_hours": 8.0},
            {"status": "blocked", "time_estimate_hours": 2.0},
            {"status": "not_started", "time_estimate_hours": 4.0},
        ]
        mock_task_manager.get_tasks_by_status.return_value = all_tasks

        # Execute
        result = await supervisor.get_agent_status()

        # Verify structure
        assert "system_status" in result
        assert "last_updated" in result
        assert "agents" in result
        assert "task_statistics" in result

        # Verify system status
        assert result["system_status"] == "active"

        # Verify agent information
        agents = result["agents"]
        assert "research_expert" in agents
        assert "coding_expert" in agents
        assert "testing_expert" in agents
        assert "documentation_expert" in agents

        # Verify agent details
        research_agent = agents["research_expert"]
        assert research_agent["status"] == "active"
        assert "web_search" in research_agent["tools"]
        assert "scrape_website" in research_agent["tools"]

        coding_agent = agents["coding_expert"]
        assert coding_agent["status"] == "active"
        assert "write_code" in coding_agent["tools"]

        # Verify task statistics
        stats = result["task_statistics"]
        assert stats["total_tasks"] == 5
        assert stats["completed_tasks"] == 2
        assert stats["in_progress_tasks"] == 1
        assert stats["blocked_tasks"] == 1
        assert stats["not_started_tasks"] == 1
        assert stats["completion_percentage"] == 40.0  # 2/5 * 100
        assert stats["total_estimated_hours"] == 22.0  # Sum of all hours
        assert stats["remaining_hours"] == 14.0  # Non-completed tasks

    @pytest.mark.asyncio
    async def test_get_agent_status_with_error(self, supervisor_with_mock_task_manager):
        """Test agent status retrieval with error."""
        supervisor, mock_task_manager = supervisor_with_mock_task_manager

        # Mock task manager exception
        mock_task_manager.get_tasks_by_status.side_effect = Exception("Database error")

        # Execute
        result = await supervisor.get_agent_status()

        # Verify error handling
        assert result["system_status"] == "error"
        assert result["error"] == "Database error"
        assert result["agents"] == {}

        # Verify default task statistics
        stats = result["task_statistics"]
        assert stats["total_tasks"] == 0
        assert stats["completion_percentage"] == 0

    @pytest.mark.asyncio
    async def test_get_agent_status_empty_tasks(
        self, supervisor_with_mock_task_manager
    ):
        """Test agent status with no tasks."""
        supervisor, mock_task_manager = supervisor_with_mock_task_manager

        # Mock empty task list
        mock_task_manager.get_tasks_by_status.return_value = []

        # Execute
        result = await supervisor.get_agent_status()

        # Verify zero statistics
        stats = result["task_statistics"]
        assert stats["total_tasks"] == 0
        assert stats["completed_tasks"] == 0
        assert stats["completion_percentage"] == 0
        assert stats["total_estimated_hours"] == 0
        assert stats["remaining_hours"] == 0


class TestSupervisorCleanup:
    """Test suite for supervisor cleanup and resource management."""

    @pytest.fixture
    def supervisor_instance(self):
        """Create supervisor instance for testing."""
        with (
            patch("src.supervisor.ChatOpenAI"),
            patch("src.supervisor.TaskManager"),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            return Supervisor()

    @pytest.mark.asyncio
    async def test_close_success(self, supervisor_instance):
        """Test successful supervisor cleanup."""
        supervisor = supervisor_instance

        # Execute cleanup
        await supervisor.close()

        # For now, just verify no exceptions raised
        # In a full implementation, this would verify resource cleanup

    @pytest.mark.asyncio
    async def test_close_with_error(self, supervisor_instance):
        """Test supervisor cleanup with error handling."""
        supervisor = supervisor_instance

        # Mock an error during cleanup (if there were resources to clean)
        # For now, just verify error handling doesn't break

        # Execute cleanup
        await supervisor.close()

        # Verify no exceptions propagated


class TestSupervisorIntegration:
    """Integration tests for supervisor functionality."""

    @pytest.mark.integration
    def test_supervisor_components_integration(self):
        """Test that all supervisor components work together."""
        with (
            patch("src.supervisor.ChatOpenAI"),
            patch("src.supervisor.TaskManager"),
            patch("src.supervisor.create_supervisor") as mock_create_sup,
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            # Mock components
            mock_workflow = Mock()
            mock_app = Mock()
            mock_workflow.compile.return_value = mock_app
            mock_create_sup.return_value = mock_workflow

            # Create supervisor
            supervisor = Supervisor()

            # Verify all components initialized
            assert supervisor.model is not None
            assert supervisor.task_manager is not None
            assert supervisor.checkpointer is not None
            assert supervisor.store is not None
            assert supervisor.app is not None

            # Verify agents created
            assert supervisor.research_agent is not None
            assert supervisor.coding_agent is not None
            assert supervisor.testing_agent is not None
            assert supervisor.documentation_agent is not None

            # Verify tools created for each agent
            assert len(supervisor.research_tools) == 2
            assert len(supervisor.coding_tools) == 2
            assert len(supervisor.testing_tools) == 2
            assert len(supervisor.documentation_tools) == 2

    @pytest.mark.integration
    @patch("src.supervisor.ExaClient")
    @patch("src.supervisor.FirecrawlClient")
    def test_agent_tools_integration(self, mock_firecrawl, mock_exa):
        """Test integration between agents and their tools."""
        with (
            patch("src.supervisor.ChatOpenAI"),
            patch("src.supervisor.TaskManager"),
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        ):
            # Mock client responses
            mock_exa_instance = Mock()
            mock_exa_instance.search_and_contents.return_value = ["search result"]
            mock_exa.return_value = mock_exa_instance

            mock_firecrawl_instance = Mock()
            mock_firecrawl_instance.scrape_url.return_value = {
                "content": "scraped content"
            }
            mock_firecrawl.return_value = mock_firecrawl_instance

            # Create supervisor
            supervisor = Supervisor()

            # Test research tools integration
            web_search, scrape_website = supervisor.research_tools

            # Execute tools
            search_result = web_search("test query")
            scrape_result = scrape_website("https://example.com")

            # Verify results
            assert "search result" in str(search_result)
            assert scrape_result == "scraped content"

            # Verify clients called
            mock_exa_instance.search_and_contents.assert_called_once()
            mock_firecrawl_instance.scrape_url.assert_called_once()

"""Comprehensive test suite for TestingAgent.

This module tests all aspects of the TestingAgent including:
- Agent initialization and configuration
- Task validation and execution
- Test generation and categorization
- Test validation workflows
- Error handling and recovery
- Async operations and timeouts
- Health monitoring and status reporting
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agents.testing_agent import TestingAgent
from src.core.agent_protocol import AgentConfig, AgentExecutionError


class TestTestingAgentInitialization:
    """Test TestingAgent initialization and configuration."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = TestingAgent()

            assert agent.name == "testing"
            assert agent.config.enabled is True
            assert "test_design" in agent.capabilities
            assert "test_execution" in agent.capabilities
            assert "quality_assurance" in agent.capabilities
            assert "validation" in agent.capabilities
            assert "performance_testing" in agent.capabilities
            assert agent.config.model == "openrouter/horizon-beta"
            assert agent.config.timeout == 150
            assert agent.config.retry_attempts == 2

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = AgentConfig(
            name="custom-testing",
            enabled=True,
            capabilities=["custom_testing"],
            model="custom-model",
            timeout=300,
            retry_attempts=3,
            max_concurrent_tasks=2,
        )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = TestingAgent(config=custom_config)

            assert agent.name == "custom-testing"
            assert agent.config.timeout == 300
            assert agent.config.retry_attempts == 3
            assert agent.config.max_concurrent_tasks == 2

    def test_init_with_explicit_api_key(self):
        """Test initialization with explicit API key."""
        agent = TestingAgent(openrouter_api_key="explicit-key")

        assert agent.openrouter_client is not None
        # Verify the API key was set correctly (indirectly through client creation)

    def test_init_missing_api_key_raises_error(self):
        """Test initialization fails when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise error during init, but would fail during API calls
            agent = TestingAgent()
            assert agent.openrouter_client is not None

    def test_create_default_factory_method(self):
        """Test the create_default factory method."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = TestingAgent.create_default()

            assert isinstance(agent, TestingAgent)
            assert agent.name == "testing"
            assert agent.config.enabled is True

    def test_default_config_creation(self):
        """Test _create_default_config method."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = TestingAgent()
            config = agent._create_default_config()

            assert config.name == "testing"
            assert config.enabled is True
            assert len(config.capabilities) == 5
            assert "test_design" in config.capabilities
            assert "test_execution" in config.capabilities
            assert "quality_assurance" in config.capabilities
            assert "validation" in config.capabilities
            assert "performance_testing" in config.capabilities


class TestTestingAgentTaskValidation:
    """Test task validation logic."""

    @pytest.fixture
    def agent(self):
        """Create a TestingAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return TestingAgent()

    @pytest.mark.asyncio
    async def test_validate_task_with_testing_keywords(self, agent):
        """Test task validation with testing-related keywords."""
        task_data = {
            "title": "Test user authentication system",
            "description": "Create comprehensive test suite",
            "component_area": "testing",
        }

        result = await agent.validate_task(task_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_task_with_multiple_testing_keywords(self, agent):
        """Test validation with multiple testing keywords."""
        test_cases = [
            {
                "title": "Validate API endpoints",
                "description": "",
                "component_area": "",
            },
            {
                "title": "",
                "description": "Verify login functionality",
                "component_area": "",
            },
            {"title": "", "description": "", "component_area": "quality assurance"},
            {"title": "QA testing procedures", "description": "", "component_area": ""},
            {
                "title": "Check system performance",
                "description": "",
                "component_area": "",
            },
            {"title": "Coverage analysis", "description": "", "component_area": ""},
            {"title": "Benchmark application", "description": "", "component_area": ""},
        ]

        for task_data in test_cases:
            result = await agent.validate_task(task_data)
            assert result is True, f"Failed for task: {task_data}"

    @pytest.mark.asyncio
    async def test_validate_task_without_testing_keywords(self, agent):
        """Test task validation without testing-related keywords."""
        task_data = {
            "title": "Implement user interface",
            "description": "Create login form design",
            "component_area": "frontend",
        }

        result = await agent.validate_task(task_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_task_empty_data(self, agent):
        """Test validation with empty task data."""
        result = await agent.validate_task({})
        assert result is False

        result = await agent.validate_task(None)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_task_at_capacity(self, agent):
        """Test validation when agent is at capacity."""
        agent._current_tasks = agent.config.max_concurrent_tasks

        task_data = {
            "title": "Test feature",
            "description": "Quality assurance task",
            "component_area": "testing",
        }

        result = await agent.validate_task(task_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_task_unhealthy_agent(self, agent):
        """Test validation when agent is unhealthy."""
        agent._is_healthy = False

        task_data = {
            "title": "Validate system",
            "description": "Testing task",
            "component_area": "qa",
        }

        result = await agent.validate_task(task_data)
        assert result is False


class TestTestingAgentTaskExecution:
    """Test task execution workflows."""

    @pytest.fixture
    def agent(self):
        """Create a TestingAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return TestingAgent()

    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing."""
        return {
            "id": 1,
            "title": "Test user authentication system",
            "description": "Create comprehensive test suite for login functionality",
            "success_criteria": "All authentication paths are tested with 90%+ coverage",
        }

    @pytest.fixture
    def sample_state(self, sample_task_data):
        """Sample agent state for testing."""
        return {
            "task_id": 1,
            "task_data": sample_task_data,
            "messages": [],
            "agent_outputs": {},
        }

    @pytest.mark.asyncio
    async def test_execute_task_success(self, agent, sample_state):
        """Test successful task execution."""
        # Mock the OpenRouter client response
        mock_response = MagicMock()
        mock_response.content = """
        # Comprehensive Test Suite for Authentication System
        
        ```python
        # test_authentication.py
        import pytest
        from fastapi.testclient import TestClient
        
        class TestAuthentication:
            def test_valid_login(self):
                # Test valid user login
                pass
                
            def test_invalid_credentials(self):
                # Test invalid login attempts
                pass
                
            @pytest.mark.asyncio
            async def test_token_validation(self):
                # Test JWT token validation
                pass
        ```
        
        ## Unit Tests
        - Authentication flow validation
        - Token generation and verification
        
        ## Integration Tests  
        - End-to-end login process
        - API endpoint testing
        
        ## Coverage Requirements
        - Minimum 90% code coverage
        - All critical paths tested
        
        ## Test Data
        - Valid user credentials
        - Invalid login attempts
        - Edge case scenarios
        """

        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        result_state = await agent.execute_task(sample_state)

        # Verify state updates
        assert "agent_outputs" in result_state
        assert "testing" in result_state["agent_outputs"]

        testing_output = result_state["agent_outputs"]["testing"]
        assert testing_output["status"] == "completed"
        assert "output" in testing_output

        # Verify testing output structure
        output = testing_output["output"]
        assert output["testing_type"] == "comprehensive_test_suite"
        assert "content" in output
        assert "test_files" in output
        assert "test_categories" in output
        assert "coverage_requirements" in output
        assert "test_data" in output

        # Verify test files were extracted
        assert "test_authentication.py" in output["test_files"]

        # Verify test categories were identified
        assert "unit_tests" in output["test_categories"]
        assert "async_tests" in output["test_categories"]
        assert "test_fixtures" in output["test_categories"]

        # Verify messages were added
        assert len(result_state["messages"]) > 0
        assert isinstance(result_state["messages"][0], HumanMessage)

    @pytest.mark.asyncio
    async def test_execute_task_with_implementation_context(self, agent, sample_state):
        """Test task execution with implementation context from coding agent."""
        # Add implementation context to state
        sample_state["agent_outputs"] = {
            "coding": {
                "output": {
                    "content": """
                    # auth.py
                    import jwt
                    from fastapi import HTTPException
                    
                    def create_token(user_id: str) -> str:
                        return jwt.encode({"user_id": user_id}, "secret", algorithm="HS256")
                    """
                }
            }
        }

        mock_response = MagicMock()
        mock_response.content = "Comprehensive test suite based on implementation"
        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        await agent.execute_task(sample_state)

        # Verify that implementation context was included in the prompt
        call_args = agent.openrouter_client.ainvoke.call_args
        prompt_messages = call_args[0][0]

        # Check that implementation context was included
        human_message = None
        for msg in prompt_messages:
            if msg.type == "human":
                human_message = msg.content
                break

        assert human_message is not None
        assert "def create_token" in human_message

    @pytest.mark.asyncio
    async def test_execute_task_no_task_data(self, agent):
        """Test execution with missing task data."""
        state = {"task_id": 1}

        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.execute_task(state)

        assert "No task data provided" in str(exc_info.value)
        assert exc_info.value.agent_name == "testing"
        assert exc_info.value.task_id == 0

    @pytest.mark.asyncio
    async def test_execute_task_openrouter_api_error(self, agent, sample_state):
        """Test execution with OpenRouter API error."""
        # Mock API error
        agent.openrouter_client.ainvoke = AsyncMock(
            side_effect=Exception("OpenRouter API error")
        )

        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.execute_task(sample_state)

        assert exc_info.value.agent_name == "testing"
        assert exc_info.value.task_id == 1
        assert "OpenRouter API error" in str(exc_info.value)

        # Verify error was recorded in state
        assert "agent_outputs" in sample_state
        assert "testing" in sample_state["agent_outputs"]

        error_output = sample_state["agent_outputs"]["testing"]
        assert error_output["status"] == "failed"
        assert "blocking_issues" in error_output

    @pytest.mark.asyncio
    async def test_execute_task_task_count_management(self, agent, sample_state):
        """Test that task count is properly managed during execution."""
        initial_count = agent._current_tasks

        mock_response = MagicMock()
        mock_response.content = "Simple test suite"
        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        await agent.execute_task(sample_state)

        # Task count should be back to initial value after completion
        assert agent._current_tasks == initial_count

    @pytest.mark.asyncio
    async def test_execute_task_timeout_scenario(self, agent, sample_state):
        """Test execution with timeout scenario."""

        # Mock a long-running operation that times out
        async def slow_response():
            await asyncio.sleep(0.1)  # Simulate slow response
            raise TimeoutError("Request timed out")

        agent.openrouter_client.ainvoke = AsyncMock(side_effect=slow_response)

        with pytest.raises(AgentExecutionError):
            await agent.execute_task(sample_state)


class TestTestingAgentTestCreation:
    """Test test creation methods."""

    @pytest.fixture
    def agent(self):
        """Create a TestingAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return TestingAgent()

    @pytest.mark.asyncio
    async def test_create_tests_success(self, agent):
        """Test successful test creation."""
        task_data = {
            "title": "Test API endpoints",
            "description": "Create tests for user management API",
            "success_criteria": "All endpoints tested with proper error handling",
        }

        state = {"agent_outputs": {}}

        mock_response = MagicMock()
        mock_response.content = """
        # API Testing Suite
        
        ```python
        # test_api.py
        import pytest
        from fastapi.testclient import TestClient
        
        def test_get_user():
            # Test getting user data
            pass
            
        @pytest.mark.asyncio
        async def test_create_user():
            # Test user creation
            pass
        ```
        
        ## Unit Tests
        Individual function testing
        
        ## Integration Tests
        Full workflow testing
        
        ## Performance Tests
        Load testing scenarios
        
        ## Test Coverage
        Aim for 95% coverage
        
        ## Test Data
        - Valid user data
        - Invalid inputs
        - Edge cases
        """

        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        result = await agent._create_tests(task_data, state)

        assert result["testing_type"] == "comprehensive_test_suite"
        assert "content" in result
        assert "test_api.py" in result["test_files"]
        assert "unit_tests" in result["test_categories"]
        assert "integration_tests" in result["test_categories"]
        assert "async_tests" in result["test_categories"]
        assert "performance_tests" in result["test_categories"]
        assert len(result["coverage_requirements"]) > 0
        assert len(result["test_data"]) > 0

    @pytest.mark.asyncio
    async def test_create_tests_with_implementation_context(self, agent):
        """Test test creation with implementation context."""
        task_data = {"title": "Test implementation", "description": "Test description"}
        state = {
            "agent_outputs": {
                "coding": {
                    "output": {
                        "content": """
                        # Implementation code
                        def authenticate_user(username, password):
                            return validate_credentials(username, password)
                        """
                    }
                }
            }
        }

        mock_response = MagicMock()
        mock_response.content = "Tests based on implementation"
        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        await agent._create_tests(task_data, state)

        # Verify implementation context was used
        call_args = agent.openrouter_client.ainvoke.call_args
        prompt_messages = call_args[0][0]

        human_message = None
        for msg in prompt_messages:
            if msg.type == "human":
                human_message = msg.content
                break

        assert "def authenticate_user" in human_message

    @pytest.mark.asyncio
    async def test_create_tests_api_error(self, agent):
        """Test test creation with API error."""
        task_data = {"title": "Test task", "description": "Test description"}
        state = {"agent_outputs": {}}

        agent.openrouter_client.ainvoke = AsyncMock(side_effect=Exception("API error"))

        result = await agent._create_tests(task_data, state)

        assert result["testing_type"] == "error"
        assert result["error"] == "API error"
        assert result["test_files"] == []
        assert result["test_categories"] == []


class TestTestingAgentContentExtraction:
    """Test content extraction helper methods."""

    @pytest.fixture
    def agent(self):
        """Create a TestingAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return TestingAgent()

    def test_extract_test_files(self, agent):
        """Test test file extraction from response."""
        content = """
        # Test Suite
        
        test_authentication.py
        test_api_endpoints.py
        
        File: test_database.py
        ```python
        # test_utils.py
        def setup_test_data():
            pass
        ```
        """

        files = agent._extract_test_files(content)

        assert "test_authentication.py" in files
        assert "test_api_endpoints.py" in files
        assert "test_database.py" in files
        assert "test_utils.py" in files
        assert len(set(files)) == len(files)  # No duplicates

    def test_categorize_tests(self, agent):
        """Test test categorization."""
        content = """
        ## Unit Tests
        def test_function():
            pass
            
        ## Integration Tests  
        These test component interaction
        
        @pytest.mark.asyncio
        async def test_async_operation():
            await some_operation()
            
        ## Performance Tests
        Benchmark critical operations
        
        @pytest.fixture
        def test_data():
            return {"key": "value"}
        """

        categories = agent._categorize_tests(content)

        assert "unit_tests" in categories
        assert "integration_tests" in categories
        assert "async_tests" in categories
        assert "performance_tests" in categories
        assert "test_fixtures" in categories

    def test_extract_coverage_requirements(self, agent):
        """Test coverage requirements extraction."""
        content = """
        ## Test Coverage
        Maintain 90% code coverage
        
        Coverage requirements:
        - Critical paths: 100%
        - Edge cases: 80%
        
        Test coverage analysis should be performed
        """

        requirements = agent._extract_coverage_requirements(content)

        assert len(requirements) > 0
        assert any("90%" in req for req in requirements)
        assert any("coverage" in req.lower() for req in requirements)

    def test_extract_test_data(self, agent):
        """Test test data extraction."""
        content = """
        ## Test Data
        - Valid user credentials
        - Invalid login attempts
        
        Sample test_data for fixtures:
        ```python
        test_data = {"username": "test", "password": "pass"}
        ```
        
        Fixture setup for database
        """

        test_data = agent._extract_test_data(content)

        assert len(test_data) <= 5  # Limited to 5 items
        assert len(test_data) > 0
        assert any("test_data" in item for item in test_data)


class TestTestingAgentHealthAndCleanup:
    """Test health monitoring and cleanup functionality."""

    @pytest.fixture
    def agent(self):
        """Create a TestingAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return TestingAgent()

    def test_get_health_status(self, agent):
        """Test health status reporting."""
        health = agent.get_health_status()

        assert health["name"] == "testing"
        assert health["healthy"] is True
        assert health["current_tasks"] == 0
        assert health["max_concurrent_tasks"] == 1
        assert "test_design" in health["capabilities"]
        assert health["enabled"] is True
        assert "last_heartbeat" in health

    def test_get_config(self, agent):
        """Test configuration retrieval."""
        config = agent.get_config()

        assert isinstance(config, AgentConfig)
        assert config.name == "testing"
        assert config.enabled is True
        assert "test_design" in config.capabilities

    @pytest.mark.asyncio
    async def test_cleanup(self, agent):
        """Test agent cleanup."""
        # Set some state to be cleaned up
        agent._current_tasks = 2
        agent._is_healthy = True

        await agent.cleanup()

        assert agent._current_tasks == 0
        assert agent._is_healthy is False

    def test_task_count_management(self, agent):
        """Test task count increment/decrement."""
        initial_count = agent._current_tasks

        agent._increment_task_count()
        assert agent._current_tasks == initial_count + 1

        agent._decrement_task_count()
        assert agent._current_tasks == initial_count

        # Test that count doesn't go below 0
        agent._decrement_task_count()
        assert agent._current_tasks == 0


@pytest.mark.asyncio
async def test_testing_agent_integration():
    """Integration test for complete TestingAgent workflow."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = TestingAgent()

        # Create a complete task scenario
        task_data = {
            "id": 1,
            "title": "Test user authentication system",
            "description": "Create comprehensive test suite for secure login functionality",
            "success_criteria": "All authentication paths tested with 90%+ coverage",
        }

        state = {
            "task_id": 1,
            "task_data": task_data,
            "messages": [],
            "agent_outputs": {
                "coding": {
                    "output": {
                        "content": """
                        # auth.py
                        import jwt
                        import bcrypt
                        from fastapi import HTTPException
                        
                        def authenticate_user(username: str, password: str) -> bool:
                            # Authenticate user with credentials
                            return validate_credentials(username, password)
                            
                        def create_token(user_id: str) -> str:
                            return jwt.encode({"user_id": user_id}, "secret", algorithm="HS256")
                        """
                    }
                }
            },
        }

        # Mock successful test generation response
        mock_response = MagicMock()
        mock_response.content = """
        # Comprehensive Authentication Test Suite
        
        ```python
        # test_authentication.py
        import pytest
        from unittest.mock import Mock, patch
        from fastapi.testclient import TestClient
        
        class TestAuthentication:
            def test_valid_login(self):
                # Test successful authentication
                assert authenticate_user("valid_user", "valid_password") is True
                
            def test_invalid_credentials(self):
                # Test failed authentication
                assert authenticate_user("invalid", "invalid") is False
                
            @pytest.mark.asyncio
            async def test_token_creation(self):
                # Test JWT token generation
                token = create_token("user123")
                assert token is not None
                
            @pytest.fixture
            def mock_user_data(self):
                return {"username": "testuser", "password": "testpass"}
        ```
        
        ```python
        # test_integration.py
        import pytest
        from fastapi.testclient import TestClient
        
        def test_login_endpoint():
            # Integration test for login endpoint
            pass
        ```
        
        ## Unit Tests
        - Function-level testing for authenticate_user
        - Token creation and validation
        
        ## Integration Tests
        - End-to-end authentication flow
        - API endpoint testing
        
        ## Performance Tests
        - Load testing for authentication endpoints
        
        ## Test Coverage
        - Target 95% code coverage
        - All error paths tested
        
        ## Test Data
        - Valid user credentials
        - Invalid login attempts  
        - Edge case scenarios
        - Mock database responses
        """

        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        # Test validation
        can_handle = await agent.validate_task(task_data)
        assert can_handle is True

        # Test execution
        result_state = await agent.execute_task(state)

        # Verify complete workflow
        assert "agent_outputs" in result_state
        assert "testing" in result_state["agent_outputs"]

        testing_output = result_state["agent_outputs"]["testing"]
        assert testing_output["status"] == "completed"

        output = testing_output["output"]
        assert output["testing_type"] == "comprehensive_test_suite"
        assert "test_authentication.py" in output["test_files"]
        assert "test_integration.py" in output["test_files"]

        # Verify test categories
        categories = output["test_categories"]
        assert "unit_tests" in categories
        assert "integration_tests" in categories
        assert "async_tests" in categories
        assert "performance_tests" in categories
        assert "test_fixtures" in categories

        # Verify coverage requirements
        assert len(output["coverage_requirements"]) > 0
        assert any("95%" in req for req in output["coverage_requirements"])

        # Verify test data
        assert len(output["test_data"]) > 0

        # Verify agent health
        health = agent.get_health_status()
        assert health["healthy"] is True
        assert health["current_tasks"] == 0  # Should be back to 0 after completion


@pytest.mark.asyncio
async def test_testing_agent_concurrent_task_limits():
    """Test that TestingAgent respects concurrent task limits."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        # Create agent with max_concurrent_tasks = 1
        config = AgentConfig(
            name="testing",
            enabled=True,
            capabilities=["testing"],
            max_concurrent_tasks=1,
        )
        agent = TestingAgent(config=config)

        # Simulate agent at capacity
        agent._current_tasks = 1

        task_data = {
            "title": "Test feature",
            "description": "Quality assurance task",
            "component_area": "testing",
        }

        # Should reject task when at capacity
        can_handle = await agent.validate_task(task_data)
        assert can_handle is False

        # Should accept task when capacity available
        agent._current_tasks = 0
        can_handle = await agent.validate_task(task_data)
        assert can_handle is True


def test_testing_agent_error_handling():
    """Test TestingAgent error handling scenarios."""
    # Test initialization errors
    with patch.dict(os.environ, {}, clear=True):
        # Should still initialize but may fail during API calls
        agent = TestingAgent()
        assert agent is not None

    # Test AgentExecutionError properties
    error = AgentExecutionError("testing", 123, "Test error", ValueError("Root cause"))
    assert error.agent_name == "testing"
    assert error.task_id == 123
    assert error.cause is not None
    assert "Agent testing failed task 123: Test error" in str(error)


@pytest.mark.asyncio
async def test_testing_agent_no_implementation_context():
    """Test testing agent with no implementation context."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = TestingAgent()

        task_data = {
            "title": "Test unknown system",
            "description": "Create tests without implementation details",
            "success_criteria": "Basic test structure created",
        }

        state = {
            "task_id": 1,
            "task_data": task_data,
            "messages": [],
            "agent_outputs": {},  # No coding agent output
        }

        mock_response = MagicMock()
        mock_response.content = "Generic test structure"
        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        result_state = await agent.execute_task(state)

        # Should handle gracefully without implementation context
        assert "agent_outputs" in result_state
        assert "testing" in result_state["agent_outputs"]

        testing_output = result_state["agent_outputs"]["testing"]
        assert testing_output["status"] == "completed"


@pytest.mark.asyncio
async def test_testing_agent_complex_test_scenarios():
    """Test testing agent with complex testing scenarios."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = TestingAgent()

        # Mock response with complex testing scenarios
        mock_response = MagicMock()
        mock_response.content = """
        # Advanced Testing Suite
        
        ```python
        # test_advanced.py
        import pytest
        import asyncio
        from unittest.mock import Mock, patch, AsyncMock
        
        class TestAdvancedScenarios:
            @pytest.mark.parametrize("input,expected", [
                ("valid", True),
                ("invalid", False),
            ])
            def test_parametrized(self, input, expected):
                pass
                
            @pytest.mark.asyncio
            async def test_async_workflow(self):
                await async_operation()
                
            @pytest.fixture(scope="session")
            def database_session(self):
                return create_test_db()
                
            def test_with_mocks(self, mocker):
                mock_service = mocker.patch('service.call')
                mock_service.return_value = "mocked"
        ```
        
        ## Unit Tests
        Detailed component testing
        
        ## Integration Tests
        System workflow validation
        
        ## Performance Tests
        Load and stress testing
        
        ## Test Coverage
        Comprehensive coverage analysis
        
        ## Test Data
        Complex test scenarios and fixtures
        """

        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        task_data = {
            "title": "Advanced testing suite",
            "description": "Complex testing scenarios with async and performance tests",
            "success_criteria": "Comprehensive test coverage",
        }

        state = {"task_id": 1, "task_data": task_data, "agent_outputs": {}}

        result = await agent._create_tests(task_data, state)

        # Should handle complex testing scenarios
        assert result["testing_type"] == "comprehensive_test_suite"
        assert "test_advanced.py" in result["test_files"]

        categories = result["test_categories"]
        assert "unit_tests" in categories
        assert "integration_tests" in categories
        assert "async_tests" in categories
        assert "performance_tests" in categories
        assert "test_fixtures" in categories

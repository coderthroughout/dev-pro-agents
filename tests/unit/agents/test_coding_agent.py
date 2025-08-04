"""Comprehensive test suite for CodingAgent.

This module tests all aspects of the CodingAgent including:
- Agent initialization and configuration
- Task validation and execution
- Code generation workflows
- Error handling and recovery
- Async operations and timeouts
- Health monitoring and status reporting
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agents.coding_agent import CodingAgent
from src.core.agent_protocol import AgentConfig, AgentExecutionError


class TestCodingAgentInitialization:
    """Test CodingAgent initialization and configuration."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = CodingAgent()

            assert agent.name == "coding"
            assert agent.config.enabled is True
            assert "implementation" in agent.capabilities
            assert "debugging" in agent.capabilities
            assert "code_generation" in agent.capabilities
            assert agent.config.model == "openrouter/horizon-beta"
            assert agent.config.timeout == 180
            assert agent.config.retry_attempts == 2

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = AgentConfig(
            name="custom-coding",
            enabled=True,
            capabilities=["custom_implementation"],
            model="custom-model",
            timeout=300,
            retry_attempts=3,
            max_concurrent_tasks=2,
        )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = CodingAgent(config=custom_config)

            assert agent.name == "custom-coding"
            assert agent.config.timeout == 300
            assert agent.config.retry_attempts == 3
            assert agent.config.max_concurrent_tasks == 2

    def test_init_with_explicit_api_key(self):
        """Test initialization with explicit API key."""
        agent = CodingAgent(openrouter_api_key="explicit-key")

        assert agent.openrouter_client is not None
        # Verify the API key was set correctly (indirectly through client creation)

    def test_init_missing_api_key_raises_error(self):
        """Test initialization fails when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise error during init, but would fail during API calls
            agent = CodingAgent()
            assert agent.openrouter_client is not None

    def test_create_default_factory_method(self):
        """Test the create_default factory method."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = CodingAgent.create_default()

            assert isinstance(agent, CodingAgent)
            assert agent.name == "coding"
            assert agent.config.enabled is True

    def test_default_config_creation(self):
        """Test _create_default_config method."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = CodingAgent()
            config = agent._create_default_config()

            assert config.name == "coding"
            assert config.enabled is True
            assert len(config.capabilities) == 5
            assert "implementation" in config.capabilities
            assert "debugging" in config.capabilities
            assert "code_generation" in config.capabilities
            assert "refactoring" in config.capabilities
            assert "code_review" in config.capabilities


class TestCodingAgentTaskValidation:
    """Test task validation logic."""

    @pytest.fixture
    def agent(self):
        """Create a CodingAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return CodingAgent()

    @pytest.mark.asyncio
    async def test_validate_task_with_coding_keywords(self, agent):
        """Test task validation with coding-related keywords."""
        task_data = {
            "title": "Implement user authentication",
            "description": "Create a secure login system",
            "component_area": "authentication",
        }

        result = await agent.validate_task(task_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_task_with_multiple_coding_keywords(self, agent):
        """Test validation with multiple coding keywords."""
        test_cases = [
            {"title": "Develop API endpoints", "description": "", "component_area": ""},
            {
                "title": "",
                "description": "Fix authentication bug",
                "component_area": "",
            },
            {"title": "", "description": "", "component_area": "refactor database"},
            {
                "title": "Debug performance issue",
                "description": "",
                "component_area": "",
            },
            {"title": "Generate test code", "description": "", "component_area": ""},
            {
                "title": "Optimize database queries",
                "description": "",
                "component_area": "",
            },
        ]

        for task_data in test_cases:
            result = await agent.validate_task(task_data)
            assert result is True, f"Failed for task: {task_data}"

    @pytest.mark.asyncio
    async def test_validate_task_without_coding_keywords(self, agent):
        """Test task validation without coding-related keywords."""
        task_data = {
            "title": "Write documentation",
            "description": "Create user manual",
            "component_area": "docs",
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
            "title": "Implement feature",
            "description": "Code implementation",
            "component_area": "backend",
        }

        result = await agent.validate_task(task_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_task_unhealthy_agent(self, agent):
        """Test validation when agent is unhealthy."""
        agent._is_healthy = False

        task_data = {
            "title": "Implement feature",
            "description": "Code implementation",
            "component_area": "backend",
        }

        result = await agent.validate_task(task_data)
        assert result is False


class TestCodingAgentTaskExecution:
    """Test task execution workflows."""

    @pytest.fixture
    def agent(self):
        """Create a CodingAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return CodingAgent()

    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing."""
        return {
            "id": 1,
            "title": "Implement user authentication",
            "description": "Create secure login system with JWT tokens",
            "component_area": "authentication",
            "success_criteria": "Users can login and logout securely",
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
        # Authentication Implementation
        
        ```python
        # auth.py
        import jwt
        from fastapi import HTTPException
        
        def create_token(user_id: str) -> str:
            return jwt.encode({"user_id": user_id}, "secret", algorithm="HS256")
        ```
        
        ## Design Decisions
        - Used JWT for stateless authentication
        - Implemented secure token validation
        
        ## Dependencies
        - jwt
        - fastapi
        
        ## Integration Notes
        - Configure secret key in environment variables
        """

        with patch.object(
            agent.openrouter_client, "ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke:
            mock_ainvoke.return_value = mock_response

        result_state = await agent.execute_task(sample_state)

        # Verify state updates
        assert "agent_outputs" in result_state
        assert "coding" in result_state["agent_outputs"]

        coding_output = result_state["agent_outputs"]["coding"]
        assert coding_output["status"] == "completed"
        assert "output" in coding_output

        # Verify implementation output structure
        output = coding_output["output"]
        assert output["implementation_type"] == "code_generation"
        assert "content" in output
        assert "files_created" in output
        assert "design_decisions" in output
        assert "dependencies" in output
        assert "integration_notes" in output

        # Verify files were extracted
        assert "auth.py" in output["files_created"]

        # Verify dependencies were extracted
        assert "jwt" in output["dependencies"]
        assert "fastapi" in output["dependencies"]

        # Verify messages were added
        assert len(result_state["messages"]) > 0
        assert isinstance(result_state["messages"][0], HumanMessage)

    @pytest.mark.asyncio
    async def test_execute_task_with_research_context(self, agent, sample_state):
        """Test task execution with research context from previous agent."""
        # Add research context to state
        sample_state["agent_outputs"] = {
            "research": {
                "output": {
                    "key_findings": [
                        "JWT tokens provide stateless authentication",
                        "Use bcrypt for password hashing",
                        "Implement rate limiting for login attempts",
                    ]
                }
            }
        }

        mock_response = MagicMock()
        mock_response.content = "Implementation with research context applied"
        with patch.object(
            agent.openrouter_client, "ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke:
            mock_ainvoke.return_value = mock_response

        await agent.execute_task(sample_state)

        # Verify that research context was included in the prompt
        call_args = agent.openrouter_client.ainvoke.call_args
        prompt_messages = call_args[0][0]

        # Check that research context was included
        human_message = None
        for msg in prompt_messages:
            if msg.type == "human":
                human_message = msg.content
                break

        assert human_message is not None
        assert "JWT tokens provide stateless authentication" in human_message

    @pytest.mark.asyncio
    async def test_execute_task_no_task_data(self, agent):
        """Test execution with missing task data."""
        state = {"task_id": 1}

        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.execute_task(state)

        assert "No task data provided" in str(exc_info.value)
        assert exc_info.value.agent_name == "coding"
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

        assert exc_info.value.agent_name == "coding"
        assert exc_info.value.task_id == 1
        assert "OpenRouter API error" in str(exc_info.value)

        # Verify error was recorded in state
        assert "agent_outputs" in sample_state
        assert "coding" in sample_state["agent_outputs"]

        error_output = sample_state["agent_outputs"]["coding"]
        assert error_output["status"] == "failed"
        assert "blocking_issues" in error_output

    @pytest.mark.asyncio
    async def test_execute_task_task_count_management(self, agent, sample_state):
        """Test that task count is properly managed during execution."""
        initial_count = agent._current_tasks

        mock_response = MagicMock()
        mock_response.content = "Simple implementation"
        with patch.object(
            agent.openrouter_client, "ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke:
            mock_ainvoke.return_value = mock_response

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


class TestCodingAgentImplementationGeneration:
    """Test implementation generation methods."""

    @pytest.fixture
    def agent(self):
        """Create a CodingAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return CodingAgent()

    @pytest.mark.asyncio
    async def test_generate_implementation_success(self, agent):
        """Test successful implementation generation."""
        task_data = {
            "title": "Create API endpoint",
            "description": "Build REST API for user management",
            "component_area": "backend",
            "success_criteria": "API responds with user data",
        }

        state = {"agent_outputs": {}}

        mock_response = MagicMock()
        mock_response.content = """
        Create the following files:
        
        ```python
        # api.py
        from fastapi import FastAPI
        app = FastAPI()
        ```
        
        Design decision: Used FastAPI for performance
        Dependencies: fastapi, pydantic
        Integration note: Add to main application
        """

        with patch.object(
            agent.openrouter_client, "ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke:
            mock_ainvoke.return_value = mock_response

        result = await agent._generate_implementation(task_data, state)

        assert result["implementation_type"] == "code_generation"
        assert "content" in result
        assert "api.py" in result["files_created"]
        assert "fastapi" in result["dependencies"]
        assert len(result["design_decisions"]) > 0
        assert len(result["integration_notes"]) > 0

    @pytest.mark.asyncio
    async def test_generate_implementation_with_research_context(self, agent):
        """Test implementation generation with research context."""
        task_data = {"title": "Test task", "description": "Test description"}
        state = {
            "agent_outputs": {
                "research": {
                    "output": {
                        "key_findings": [
                            "Use async/await for I/O operations",
                            "Implement proper error handling",
                        ]
                    }
                }
            }
        }

        mock_response = MagicMock()
        mock_response.content = "Implementation based on research"
        with patch.object(
            agent.openrouter_client, "ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke:
            mock_ainvoke.return_value = mock_response

        await agent._generate_implementation(task_data, state)

        # Verify research context was used
        call_args = agent.openrouter_client.ainvoke.call_args
        prompt_messages = call_args[0][0]

        human_message = None
        for msg in prompt_messages:
            if msg.type == "human":
                human_message = msg.content
                break

        assert "Use async/await for I/O operations" in human_message

    @pytest.mark.asyncio
    async def test_generate_implementation_api_error(self, agent):
        """Test implementation generation with API error."""
        task_data = {"title": "Test task", "description": "Test description"}
        state = {"agent_outputs": {}}

        agent.openrouter_client.ainvoke = AsyncMock(side_effect=Exception("API error"))

        result = await agent._generate_implementation(task_data, state)

        assert result["implementation_type"] == "error"
        assert result["error"] == "API error"
        assert result["files_created"] == []
        assert result["design_decisions"] == []


class TestCodingAgentContentExtraction:
    """Test content extraction helper methods."""

    @pytest.fixture
    def agent(self):
        """Create a CodingAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return CodingAgent()

    def test_extract_files_from_response(self, agent):
        """Test file extraction from implementation response."""
        content = """
        ```python
        # auth.py
        def authenticate():
            pass
        ```
        
        File: models.py
        Create file: utils.py
        """

        files = agent._extract_files_from_response(content)

        assert "auth.py" in files
        assert "models.py" in files
        assert "utils.py" in files
        assert len(set(files)) == len(files)  # No duplicates

    def test_extract_design_decisions(self, agent):
        """Test design decision extraction."""
        content = """
        ## Design Decisions
        
        Decision: Used JWT for authentication
        Design approach: Stateless authentication
        Architecture decision: Microservices pattern
        
        Some other content here.
        Additional design notes here.
        """

        decisions = agent._extract_design_decisions(content)

        assert len(decisions) <= 3  # Limited to 3 decisions
        assert any("JWT" in decision for decision in decisions)
        assert any("Stateless" in decision for decision in decisions)

    def test_extract_dependencies(self, agent):
        """Test dependency extraction from content."""
        content = """
        ```python
        import jwt
        from fastapi import FastAPI
        import asyncio
        from pydantic import BaseModel
        ```
        
        pip install requests
        pip install httpx
        """

        dependencies = agent._extract_dependencies(content)

        # Should extract external dependencies but filter out standard library
        assert "jwt" in dependencies
        assert "fastapi" in dependencies
        assert "requests" in dependencies
        assert "httpx" in dependencies
        assert "asyncio" not in dependencies  # Standard library filtered out
        assert "pydantic" in dependencies

    def test_extract_integration_notes(self, agent):
        """Test integration notes extraction."""
        content = """
        ## Integration Notes
        Configure environment variables
        
        ## Usage
        Call the authenticate function
        
        Note: Add to main application
        Additional integration info here.
        """

        notes = agent._extract_integration_notes(content)

        assert len(notes) <= 3  # Limited to 3 notes
        assert any("Configure" in note for note in notes)
        assert any("Call the authenticate" in note for note in notes)


class TestCodingAgentHealthAndCleanup:
    """Test health monitoring and cleanup functionality."""

    @pytest.fixture
    def agent(self):
        """Create a CodingAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return CodingAgent()

    def test_get_health_status(self, agent):
        """Test health status reporting."""
        health = agent.get_health_status()

        assert health["name"] == "coding"
        assert health["healthy"] is True
        assert health["current_tasks"] == 0
        assert health["max_concurrent_tasks"] == 1
        assert "implementation" in health["capabilities"]
        assert health["enabled"] is True
        assert "last_heartbeat" in health

    def test_get_config(self, agent):
        """Test configuration retrieval."""
        config = agent.get_config()

        assert isinstance(config, AgentConfig)
        assert config.name == "coding"
        assert config.enabled is True
        assert "implementation" in config.capabilities

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
async def test_coding_agent_integration():
    """Integration test for complete CodingAgent workflow."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = CodingAgent()

        # Create a complete task scenario
        task_data = {
            "id": 1,
            "title": "Implement user authentication",
            "description": "Create secure login system with JWT tokens",
            "component_area": "authentication",
            "success_criteria": "Users can login and logout securely",
        }

        state = {
            "task_id": 1,
            "task_data": task_data,
            "messages": [],
            "agent_outputs": {
                "research": {
                    "output": {
                        "key_findings": [
                            "JWT tokens provide stateless authentication",
                            "Use bcrypt for password hashing",
                        ]
                    }
                }
            },
        }

        # Mock successful implementation response
        mock_response = MagicMock()
        mock_response.content = """
        # Authentication System Implementation
        
        ```python
        # auth.py
        import jwt
        import bcrypt
        from fastapi import HTTPException
        
        def create_token(user_id: str) -> str:
            return jwt.encode({"user_id": user_id}, "secret", algorithm="HS256")
            
        def hash_password(password: str) -> str:
            return bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        ```
        
        ## Design Decisions
        - Used JWT for stateless authentication
        - Implemented bcrypt for secure password hashing
        
        ## Dependencies  
        - jwt
        - bcrypt
        - fastapi
        
        ## Integration Notes
        - Configure JWT secret in environment
        - Add rate limiting middleware
        """

        with patch.object(
            agent.openrouter_client, "ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke:
            mock_ainvoke.return_value = mock_response

        # Test validation
        can_handle = await agent.validate_task(task_data)
        assert can_handle is True

        # Test execution
        result_state = await agent.execute_task(state)

        # Verify complete workflow
        assert "agent_outputs" in result_state
        assert "coding" in result_state["agent_outputs"]

        coding_output = result_state["agent_outputs"]["coding"]
        assert coding_output["status"] == "completed"

        implementation = coding_output["output"]
        assert implementation["implementation_type"] == "code_generation"
        assert "auth.py" in implementation["files_created"]
        assert "jwt" in implementation["dependencies"]
        assert "bcrypt" in implementation["dependencies"]
        assert len(implementation["design_decisions"]) > 0
        assert len(implementation["integration_notes"]) > 0

        # Verify agent health
        health = agent.get_health_status()
        assert health["healthy"] is True
        assert health["current_tasks"] == 0  # Should be back to 0 after completion


@pytest.mark.asyncio
async def test_coding_agent_concurrent_task_limits():
    """Test that CodingAgent respects concurrent task limits."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        # Create agent with max_concurrent_tasks = 1
        config = AgentConfig(
            name="coding",
            enabled=True,
            capabilities=["implementation"],
            max_concurrent_tasks=1,
        )
        agent = CodingAgent(config=config)

        # Simulate agent at capacity
        agent._current_tasks = 1

        task_data = {
            "title": "Implement feature",
            "description": "Code implementation task",
            "component_area": "backend",
        }

        # Should reject task when at capacity
        can_handle = await agent.validate_task(task_data)
        assert can_handle is False

        # Should accept task when capacity available
        agent._current_tasks = 0
        can_handle = await agent.validate_task(task_data)
        assert can_handle is True


def test_coding_agent_error_handling():
    """Test CodingAgent error handling scenarios."""
    # Test initialization errors
    with patch.dict(os.environ, {}, clear=True):
        # Should still initialize but may fail during API calls
        agent = CodingAgent()
        assert agent is not None

    # Test AgentExecutionError properties
    error = AgentExecutionError("coding", 123, "Test error", ValueError("Root cause"))
    assert error.agent_name == "coding"
    assert error.task_id == 123
    assert error.cause is not None
    assert "Agent coding failed task 123: Test error" in str(error)

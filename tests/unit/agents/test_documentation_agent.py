"""Comprehensive test suite for DocumentationAgent.

This module tests all aspects of the DocumentationAgent including:
- Agent initialization and configuration
- Task validation and execution
- Documentation generation and context building
- File extraction and content parsing
- Error handling and recovery
- Async operations and timeouts
- Health monitoring and status reporting
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agents.documentation_agent import DocumentationAgent
from src.core.agent_protocol import AgentConfig, AgentExecutionError


class TestDocumentationAgentInitialization:
    """Test DocumentationAgent initialization and configuration."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = DocumentationAgent()

            assert agent.name == "documentation"
            assert agent.config.enabled is True
            assert "documentation_generation" in agent.capabilities
            assert "content_creation" in agent.capabilities
            assert "technical_writing" in agent.capabilities
            assert "readme_creation" in agent.capabilities
            assert "api_documentation" in agent.capabilities
            assert agent.config.model == "openrouter/horizon-beta"
            assert agent.config.timeout == 120
            assert agent.config.retry_attempts == 2
            assert agent.config.max_concurrent_tasks == 2

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = AgentConfig(
            name="custom-documentation",
            enabled=True,
            capabilities=["custom_docs"],
            model="custom-model",
            timeout=300,
            retry_attempts=3,
            max_concurrent_tasks=3,
        )

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = DocumentationAgent(config=custom_config)

            assert agent.name == "custom-documentation"
            assert agent.config.timeout == 300
            assert agent.config.retry_attempts == 3
            assert agent.config.max_concurrent_tasks == 3

    def test_init_with_explicit_api_key(self):
        """Test initialization with explicit API key."""
        agent = DocumentationAgent(openrouter_api_key="explicit-key")

        assert agent.openrouter_client is not None
        # Verify the API key was set correctly (indirectly through client creation)

    def test_init_missing_api_key_raises_error(self):
        """Test initialization fails when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise error during init, but would fail during API calls
            agent = DocumentationAgent()
            assert agent.openrouter_client is not None

    def test_create_default_factory_method(self):
        """Test the create_default factory method."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = DocumentationAgent.create_default()

            assert isinstance(agent, DocumentationAgent)
            assert agent.name == "documentation"
            assert agent.config.enabled is True

    def test_default_config_creation(self):
        """Test _create_default_config method."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = DocumentationAgent()
            config = agent._create_default_config()

            assert config.name == "documentation"
            assert config.enabled is True
            assert len(config.capabilities) == 5
            assert "documentation_generation" in config.capabilities
            assert "content_creation" in config.capabilities
            assert "technical_writing" in config.capabilities
            assert "readme_creation" in config.capabilities
            assert "api_documentation" in config.capabilities


class TestDocumentationAgentTaskValidation:
    """Test task validation logic."""

    @pytest.fixture
    def agent(self):
        """Create a DocumentationAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return DocumentationAgent()

    @pytest.mark.asyncio
    async def test_validate_task_with_documentation_keywords(self, agent):
        """Test task validation with documentation-related keywords."""
        task_data = {
            "title": "Document user authentication system",
            "description": "Create comprehensive documentation for login features",
            "component_area": "documentation",
        }

        result = await agent.validate_task(task_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_task_with_multiple_documentation_keywords(self, agent):
        """Test validation with multiple documentation keywords."""
        test_cases = [
            {"title": "Create README file", "description": "", "component_area": ""},
            {"title": "", "description": "Write user guide", "component_area": ""},
            {"title": "", "description": "", "component_area": "api specification"},
            {
                "title": "Generate API documentation",
                "description": "",
                "component_area": "",
            },
            {
                "title": "Content creation for help",
                "description": "",
                "component_area": "",
            },
            {"title": "Technical manual", "description": "", "component_area": ""},
            {"title": "Report generation", "description": "", "component_area": ""},
        ]

        for task_data in test_cases:
            result = await agent.validate_task(task_data)
            assert result is True, f"Failed for task: {task_data}"

    @pytest.mark.asyncio
    async def test_validate_task_without_documentation_keywords(self, agent):
        """Test task validation without documentation-related keywords."""
        task_data = {
            "title": "Implement user authentication",
            "description": "Create secure login system",
            "component_area": "backend",
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
            "title": "Document feature",
            "description": "Technical writing task",
            "component_area": "docs",
        }

        result = await agent.validate_task(task_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_task_unhealthy_agent(self, agent):
        """Test validation when agent is unhealthy."""
        agent._is_healthy = False

        task_data = {
            "title": "Create documentation",
            "description": "Write comprehensive guide",
            "component_area": "documentation",
        }

        result = await agent.validate_task(task_data)
        assert result is False


class TestDocumentationAgentTaskExecution:
    """Test task execution workflows."""

    @pytest.fixture
    def agent(self):
        """Create a DocumentationAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return DocumentationAgent()

    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing."""
        return {
            "id": 1,
            "title": "Document user authentication system",
            "description": "Create comprehensive documentation for secure login functionality",
            "component_area": "security",
            "success_criteria": "Complete documentation with usage examples and troubleshooting",
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
        # User Authentication System Documentation
        
        ## Overview and Purpose
        
        The user authentication system provides secure login functionality for the application.
        
        ## Implementation Details
        
        The system uses JWT tokens for stateless authentication:
        
        ```python
        def authenticate_user(username: str, password: str) -> bool:
            # Authenticate user credentials
            return validate_credentials(username, password)
        ```
        
        ## Usage Instructions
        
        1. Initialize the authentication service
        2. Call authenticate_user with credentials
        3. Handle the returned token
        
        ## API Documentation
        
        ### POST /auth/login
        Authenticate user with credentials
        
        ## Testing Information
        
        Run the test suite with:
        ```bash
        pytest tests/test_auth.py
        ```
        
        ## Troubleshooting Guide
        
        **Issue**: Login fails with valid credentials
        **Solution**: Check database connection and user table
        
        For more information, see [Authentication Guide](https://example.com/auth-guide)
        """

        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        result_state = await agent.execute_task(sample_state)

        # Verify state updates
        assert "agent_outputs" in result_state
        assert "documentation" in result_state["agent_outputs"]

        documentation_output = result_state["agent_outputs"]["documentation"]
        assert documentation_output["status"] == "completed"
        assert "output" in documentation_output

        # Verify documentation output structure
        output = documentation_output["output"]
        assert output["documentation_type"] == "comprehensive_guide"
        assert "content" in output
        assert "documentation_files" in output
        assert "sections" in output
        assert "code_examples" in output
        assert "links_and_references" in output

        # Verify sections were extracted
        sections = output["sections"]
        assert "Overview and Purpose" in sections
        assert "Implementation Details" in sections
        assert "Usage Instructions" in sections

        # Verify code examples were extracted
        assert len(output["code_examples"]) > 0

        # Verify links were extracted
        assert len(output["links_and_references"]) > 0

        # Verify messages were added
        assert len(result_state["messages"]) > 0
        assert isinstance(result_state["messages"][0], HumanMessage)

    @pytest.mark.asyncio
    async def test_execute_task_with_all_agents_context(self, agent, sample_state):
        """Test task execution with context from all other agents."""
        # Add context from all agents to state
        sample_state["agent_outputs"] = {
            "research": {
                "output": {
                    "key_findings": [
                        "JWT tokens provide stateless authentication",
                        "Use HTTPS for secure transmission",
                    ]
                },
                "status": "completed",
            },
            "coding": {
                "output": {
                    "files_created": ["auth.py", "models.py"],
                    "design_decisions": [
                        "Used JWT for tokens",
                        "Implemented bcrypt hashing",
                    ],
                },
                "status": "completed",
            },
            "testing": {
                "output": {
                    "test_files": ["test_auth.py", "test_integration.py"],
                    "test_categories": ["unit_tests", "integration_tests"],
                },
                "status": "completed",
            },
        }

        mock_response = MagicMock()
        mock_response.content = "Comprehensive documentation based on all agent outputs"
        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        await agent.execute_task(sample_state)

        # Verify that context from all agents was included in the prompt
        call_args = agent.openrouter_client.ainvoke.call_args
        prompt_messages = call_args[0][0]

        # Check that context from all agents was included
        human_message = None
        for msg in prompt_messages:
            if msg.type == "human":
                human_message = msg.content
                break

        assert human_message is not None
        assert "Research Agent (completed)" in human_message
        assert "Coding Agent (completed)" in human_message
        assert "Testing Agent (completed)" in human_message
        assert "JWT tokens provide stateless authentication" in human_message
        assert "auth.py" in human_message
        assert "test_auth.py" in human_message

    @pytest.mark.asyncio
    async def test_execute_task_no_task_data(self, agent):
        """Test execution with missing task data."""
        state = {"task_id": 1}

        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.execute_task(state)

        assert "No task data provided" in str(exc_info.value)
        assert exc_info.value.agent_name == "documentation"
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

        assert exc_info.value.agent_name == "documentation"
        assert exc_info.value.task_id == 1
        assert "OpenRouter API error" in str(exc_info.value)

        # Verify error was recorded in state
        assert "agent_outputs" in sample_state
        assert "documentation" in sample_state["agent_outputs"]

        error_output = sample_state["agent_outputs"]["documentation"]
        assert error_output["status"] == "failed"
        assert "blocking_issues" in error_output

    @pytest.mark.asyncio
    async def test_execute_task_task_count_management(self, agent, sample_state):
        """Test that task count is properly managed during execution."""
        initial_count = agent._current_tasks

        mock_response = MagicMock()
        mock_response.content = "Simple documentation"
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


class TestDocumentationAgentDocumentationCreation:
    """Test documentation creation methods."""

    @pytest.fixture
    def agent(self):
        """Create a DocumentationAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return DocumentationAgent()

    @pytest.mark.asyncio
    async def test_create_documentation_success(self, agent):
        """Test successful documentation creation."""
        task_data = {
            "title": "API Documentation",
            "description": "Create comprehensive API documentation",
            "component_area": "api",
            "success_criteria": "Complete API reference with examples",
        }

        state = {"agent_outputs": {}}

        mock_response = MagicMock()
        mock_response.content = """
        # API Documentation
        
        ## Overview and Purpose
        This API provides user management functionality
        
        ## Implementation Details  
        Built with FastAPI framework
        
        ```python
        @app.get("/users/{user_id}")
        def get_user(user_id: int):
            return {"user_id": user_id}
        ```
        
        ## Usage Instructions
        1. Install the client library
        2. Authenticate with API key
        3. Make requests to endpoints
        
        ## API Reference
        Complete endpoint documentation
        
        ## Testing Information
        Run tests with pytest
        
        ## Troubleshooting Guide
        Common issues and solutions
        
        Create file: README.md
        Additional documentation: API_REFERENCE.md
        
        See also: https://fastapi.tiangolo.com/
        """

        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        result = await agent._create_documentation(task_data, state)

        assert result["documentation_type"] == "comprehensive_guide"
        assert "content" in result
        assert "README.md" in result["documentation_files"]
        assert "API_REFERENCE.md" in result["documentation_files"]

        # Verify sections were extracted
        sections = result["sections"]
        assert "Overview and Purpose" in sections
        assert "Implementation Details" in sections
        assert "Usage Instructions" in sections

        # Verify code examples were extracted
        assert len(result["code_examples"]) > 0
        code_example = result["code_examples"][0]
        assert "@app.get" in code_example

        # Verify links were extracted
        assert len(result["links_and_references"]) > 0
        assert any(
            "fastapi.tiangolo.com" in ref for ref in result["links_and_references"]
        )

    @pytest.mark.asyncio
    async def test_create_documentation_with_context_summary(self, agent):
        """Test documentation creation with context summary."""
        task_data = {"title": "Test documentation", "description": "Test description"}
        state = {
            "agent_outputs": {
                "research": {
                    "output": {"key_findings": ["Finding 1", "Finding 2"]},
                    "status": "completed",
                },
                "coding": {
                    "output": {
                        "files_created": ["main.py"],
                        "design_decisions": ["Decision 1"],
                    },
                    "status": "completed",
                },
            }
        }

        mock_response = MagicMock()
        mock_response.content = "Documentation based on context"
        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        await agent._create_documentation(task_data, state)

        # Verify context summary was used
        call_args = agent.openrouter_client.ainvoke.call_args
        prompt_messages = call_args[0][0]

        human_message = None
        for msg in prompt_messages:
            if msg.type == "human":
                human_message = msg.content
                break

        assert "Research Agent (completed)" in human_message
        assert "Coding Agent (completed)" in human_message
        assert "Finding 1" in human_message
        assert "main.py" in human_message

    @pytest.mark.asyncio
    async def test_create_documentation_api_error(self, agent):
        """Test documentation creation with API error."""
        task_data = {"title": "Test task", "description": "Test description"}
        state = {"agent_outputs": {}}

        agent.openrouter_client.ainvoke = AsyncMock(side_effect=Exception("API error"))

        result = await agent._create_documentation(task_data, state)

        assert result["documentation_type"] == "error"
        assert result["error"] == "API error"
        assert result["documentation_files"] == []
        assert result["sections"] == []


class TestDocumentationAgentContextBuilding:
    """Test context building methods."""

    @pytest.fixture
    def agent(self):
        """Create a DocumentationAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return DocumentationAgent()

    def test_build_context_summary(self, agent):
        """Test building context summary from agent outputs."""
        state = {
            "agent_outputs": {
                "research": {
                    "output": {
                        "key_findings": [
                            "JWT provides stateless auth",
                            "HTTPS required for security",
                            "Rate limiting prevents attacks",
                        ]
                    },
                    "status": "completed",
                },
                "coding": {
                    "output": {
                        "files_created": ["auth.py", "models.py", "utils.py"],
                        "design_decisions": [
                            "Used JWT for authentication",
                            "Implemented bcrypt for passwords",
                        ],
                    },
                    "status": "completed",
                },
                "testing": {
                    "output": {
                        "test_files": ["test_auth.py", "test_models.py"],
                        "test_categories": [
                            "unit_tests",
                            "integration_tests",
                            "async_tests",
                        ],
                    },
                    "status": "completed",
                },
            }
        }

        summary = agent._build_context_summary(state)

        # Verify all agents are included
        assert "Research Agent (completed)" in summary
        assert "Coding Agent (completed)" in summary
        assert "Testing Agent (completed)" in summary

        # Verify key information is included
        assert "JWT provides stateless auth" in summary
        assert "auth.py" in summary
        assert "test_auth.py" in summary
        assert "unit_tests" in summary

    def test_build_context_summary_empty_state(self, agent):
        """Test building context summary with empty state."""
        state = {"agent_outputs": {}}

        summary = agent._build_context_summary(state)

        # Should handle empty state gracefully
        assert summary == ""

    def test_build_context_summary_partial_outputs(self, agent):
        """Test building context summary with partial agent outputs."""
        state = {
            "agent_outputs": {
                "research": {
                    "output": {"key_findings": []},  # Empty findings
                    "status": "failed",
                },
                "coding": {
                    "output": {
                        "files_created": ["main.py"],
                        "design_decisions": [],  # Empty decisions
                    },
                    "status": "completed",
                },
            }
        }

        summary = agent._build_context_summary(state)

        # Should include agents even with partial data
        assert "Research Agent (failed)" in summary
        assert "Coding Agent (completed)" in summary
        assert "main.py" in summary


class TestDocumentationAgentContentExtraction:
    """Test content extraction helper methods."""

    @pytest.fixture
    def agent(self):
        """Create a DocumentationAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return DocumentationAgent()

    def test_extract_documentation_files(self, agent):
        """Test documentation file extraction from response."""
        content = """
        # Project Documentation
        
        README.md contains project overview
        Create file: USER_GUIDE.md
        File: API_REFERENCE.md
        Documentation file: TROUBLESHOOTING.md
        """

        files = agent._extract_documentation_files(content)

        assert "README.md" in files
        assert "USER_GUIDE.md" in files
        assert "API_REFERENCE.md" in files
        assert "TROUBLESHOOTING.md" in files
        assert len(set(files)) == len(files)  # No duplicates

    def test_extract_sections(self, agent):
        """Test section extraction from documentation."""
        content = """
        # Main Title
        
        ## Overview and Purpose
        Project overview content
        
        ## Implementation Details
        Technical implementation
        
        ### Subsection
        Detailed information
        
        ## Usage Instructions
        How to use the system
        
        Regular paragraph without header
        """

        sections = agent._extract_sections(content)

        assert len(sections) <= 10  # Limited to 10 sections
        assert "Main Title" in sections
        assert "Overview and Purpose" in sections
        assert "Implementation Details" in sections
        assert "Subsection" in sections
        assert "Usage Instructions" in sections

    def test_extract_code_examples(self, agent):
        """Test code examples extraction."""
        content = """
        # Documentation
        
        Here's a Python example:
        ```python
        def authenticate(username, password):
            return validate_credentials(username, password)
        ```
        
        JavaScript example:
        ```javascript
        const token = generateToken(userId);
        ```
        
        ```bash
        npm install package
        ```
        
        Some regular text without code blocks.
        """

        examples = agent._extract_code_examples(content)

        assert len(examples) <= 5  # Limited to 5 examples
        assert len(examples) == 3
        assert any("def authenticate" in example for example in examples)
        assert any("const token" in example for example in examples)
        assert any("npm install" in example for example in examples)

    def test_extract_references(self, agent):
        """Test links and references extraction."""
        content = """
        # Documentation
        
        See the [FastAPI documentation](https://fastapi.tiangolo.com/) for details.
        
        Check out [JWT.io](https://jwt.io) for token information.
        
        Visit https://github.com/example/repo for source code.
        
        Additional resource: https://docs.python.org/3/
        """

        references = agent._extract_references(content)

        assert len(references) <= 10  # Limited to 10 references
        assert len(references) >= 2

        # Check for markdown links
        assert any(
            "FastAPI documentation: https://fastapi.tiangolo.com/" in ref
            for ref in references
        )
        assert any("JWT.io: https://jwt.io" in ref for ref in references)

        # Check for plain URLs
        assert any("https://github.com/example/repo" in ref for ref in references)
        assert any("https://docs.python.org/3/" in ref for ref in references)


class TestDocumentationAgentHealthAndCleanup:
    """Test health monitoring and cleanup functionality."""

    @pytest.fixture
    def agent(self):
        """Create a DocumentationAgent for testing."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return DocumentationAgent()

    def test_get_health_status(self, agent):
        """Test health status reporting."""
        health = agent.get_health_status()

        assert health["name"] == "documentation"
        assert health["healthy"] is True
        assert health["current_tasks"] == 0
        assert health["max_concurrent_tasks"] == 2
        assert "documentation_generation" in health["capabilities"]
        assert health["enabled"] is True
        assert "last_heartbeat" in health

    def test_get_config(self, agent):
        """Test configuration retrieval."""
        config = agent.get_config()

        assert isinstance(config, AgentConfig)
        assert config.name == "documentation"
        assert config.enabled is True
        assert "documentation_generation" in config.capabilities

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
async def test_documentation_agent_integration():
    """Integration test for complete DocumentationAgent workflow."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = DocumentationAgent()

        # Create a complete task scenario
        task_data = {
            "id": 1,
            "title": "Document user authentication system",
            "description": "Create comprehensive documentation for secure login functionality",
            "component_area": "security",
            "success_criteria": "Complete documentation with examples and troubleshooting",
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
                            "Use HTTPS for secure transmission",
                        ]
                    },
                    "status": "completed",
                },
                "coding": {
                    "output": {
                        "files_created": ["auth.py", "models.py"],
                        "design_decisions": ["Used JWT for tokens"],
                    },
                    "status": "completed",
                },
                "testing": {
                    "output": {
                        "test_files": ["test_auth.py"],
                        "test_categories": ["unit_tests", "integration_tests"],
                    },
                    "status": "completed",
                },
            },
        }

        # Mock successful documentation generation response
        mock_response = MagicMock()
        mock_response.content = """
        # User Authentication System Documentation
        
        ## Overview and Purpose
        
        The user authentication system provides secure login functionality using JWT tokens.
        
        ## Implementation Details
        
        ### Core Components
        
        The system consists of:
        - `auth.py`: Main authentication logic
        - `models.py`: User data models
        
        ```python
        def authenticate_user(username: str, password: str) -> bool:
            # Validate user credentials
            return validate_credentials(username, password)
        ```
        
        ## Usage Instructions
        
        1. Import the authentication module
        2. Call authenticate_user with credentials
        3. Handle the JWT token response
        
        ## API Documentation
        
        ### POST /auth/login
        
        Authenticate user with username and password.
        
        ## Testing Information
        
        The system includes comprehensive tests:
        - Unit tests: `test_auth.py`
        - Integration tests for full workflow
        
        Run tests with:
        ```bash
        pytest test_auth.py -v
        ```
        
        ## Troubleshooting Guide
        
        ### Common Issues
        
        **Problem**: Authentication fails with valid credentials
        **Solution**: Check database connection and verify user exists
        
        **Problem**: JWT token validation fails
        **Solution**: Ensure secret key is properly configured
        
        ## Additional Resources
        
        - [JWT Documentation](https://jwt.io/introduction/)
        - [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
        
        Create file: README.md
        Create file: AUTH_API.md
        """

        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        # Test validation
        can_handle = await agent.validate_task(task_data)
        assert can_handle is True

        # Test execution
        result_state = await agent.execute_task(state)

        # Verify complete workflow
        assert "agent_outputs" in result_state
        assert "documentation" in result_state["agent_outputs"]

        documentation_output = result_state["agent_outputs"]["documentation"]
        assert documentation_output["status"] == "completed"

        output = documentation_output["output"]
        assert output["documentation_type"] == "comprehensive_guide"

        # Verify documentation files
        assert "README.md" in output["documentation_files"]
        assert "AUTH_API.md" in output["documentation_files"]

        # Verify sections
        sections = output["sections"]
        assert "Overview and Purpose" in sections
        assert "Implementation Details" in sections
        assert "Usage Instructions" in sections
        assert "API Documentation" in sections
        assert "Testing Information" in sections
        assert "Troubleshooting Guide" in sections

        # Verify code examples
        assert len(output["code_examples"]) > 0
        code_examples = output["code_examples"]
        assert any("def authenticate_user" in example for example in code_examples)
        assert any("pytest test_auth.py" in example for example in code_examples)

        # Verify links and references
        assert len(output["links_and_references"]) > 0
        references = output["links_and_references"]
        assert any("jwt.io" in ref for ref in references)
        assert any("fastapi.tiangolo.com" in ref for ref in references)

        # Verify agent health
        health = agent.get_health_status()
        assert health["healthy"] is True
        assert health["current_tasks"] == 0  # Should be back to 0 after completion


@pytest.mark.asyncio
async def test_documentation_agent_concurrent_task_limits():
    """Test that DocumentationAgent respects concurrent task limits."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        # Create agent with max_concurrent_tasks = 2
        config = AgentConfig(
            name="documentation",
            enabled=True,
            capabilities=["documentation"],
            max_concurrent_tasks=2,
        )
        agent = DocumentationAgent(config=config)

        # Simulate agent at capacity
        agent._current_tasks = 2

        task_data = {
            "title": "Document feature",
            "description": "Technical writing task",
            "component_area": "documentation",
        }

        # Should reject task when at capacity
        can_handle = await agent.validate_task(task_data)
        assert can_handle is False

        # Should accept task when capacity available
        agent._current_tasks = 1
        can_handle = await agent.validate_task(task_data)
        assert can_handle is True


def test_documentation_agent_error_handling():
    """Test DocumentationAgent error handling scenarios."""
    # Test initialization errors
    with patch.dict(os.environ, {}, clear=True):
        # Should still initialize but may fail during API calls
        agent = DocumentationAgent()
        assert agent is not None

    # Test AgentExecutionError properties
    error = AgentExecutionError(
        "documentation", 123, "Test error", ValueError("Root cause")
    )
    assert error.agent_name == "documentation"
    assert error.task_id == 123
    assert error.cause is not None
    assert "Agent documentation failed task 123: Test error" in str(error)


@pytest.mark.asyncio
async def test_documentation_agent_no_agent_context():
    """Test documentation agent with no other agent context."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = DocumentationAgent()

        task_data = {
            "title": "Standalone documentation",
            "description": "Create documentation without other agent context",
            "success_criteria": "Basic documentation structure",
        }

        state = {
            "task_id": 1,
            "task_data": task_data,
            "messages": [],
            "agent_outputs": {},  # No other agent outputs
        }

        mock_response = MagicMock()
        mock_response.content = "Standalone documentation content"
        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        result_state = await agent.execute_task(state)

        # Should handle gracefully without other agent context
        assert "agent_outputs" in result_state
        assert "documentation" in result_state["agent_outputs"]

        documentation_output = result_state["agent_outputs"]["documentation"]
        assert documentation_output["status"] == "completed"


@pytest.mark.asyncio
async def test_documentation_agent_complex_content_extraction():
    """Test documentation agent with complex content scenarios."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = DocumentationAgent()

        # Mock response with complex documentation structure
        mock_response = MagicMock()
        mock_response.content = """
        # Advanced System Documentation
        
        ## Architecture Overview
        
        ### Core Components
        - Authentication Service
        - User Management
        - Data Layer
        
        ## Implementation Guide
        
        ```python
        # Primary authentication handler
        class AuthHandler:
            def __init__(self):
                self.secret = os.getenv("JWT_SECRET")
        ```
        
        ```javascript
        // Client-side authentication
        const auth = new AuthClient({
            baseUrl: 'https://api.example.com'
        });
        ```
        
        ```sql
        -- Database schema
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE
        );
        ```
        
        ## API Reference
        
        ### Endpoints
        
        ## Testing Strategy
        
        ## Deployment Guide
        
        ## Troubleshooting
        
        For support, contact [support team](mailto:support@example.com).
        
        Documentation files:
        - Create file: SETUP.md
        - Create file: DEPLOYMENT.md  
        - File: ARCHITECTURE.md
        
        External references:
        - https://docs.python.org/3/
        - https://nodejs.org/docs/
        - [PostgreSQL Docs](https://postgresql.org/docs/)
        """

        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)

        task_data = {
            "title": "Complex system documentation",
            "description": "Multi-component system with various technologies",
            "success_criteria": "Comprehensive technical documentation",
        }

        state = {"task_id": 1, "task_data": task_data, "agent_outputs": {}}

        result = await agent._create_documentation(task_data, state)

        # Should handle complex documentation scenarios
        assert result["documentation_type"] == "comprehensive_guide"

        # Verify multiple file types extracted
        files = result["documentation_files"]
        assert "SETUP.md" in files
        assert "DEPLOYMENT.md" in files
        assert "ARCHITECTURE.md" in files

        # Verify multiple sections extracted
        sections = result["sections"]
        assert "Architecture Overview" in sections
        assert "Implementation Guide" in sections
        assert "API Reference" in sections

        # Verify multiple code examples
        examples = result["code_examples"]
        assert len(examples) >= 3
        assert any("class AuthHandler" in example for example in examples)
        assert any("const auth" in example for example in examples)
        assert any("CREATE TABLE" in example for example in examples)

        # Verify diverse references
        references = result["links_and_references"]
        assert len(references) > 0
        assert any("support@example.com" in ref for ref in references)
        assert any("docs.python.org" in ref for ref in references)

"""Comprehensive test suite for ResearchAgent.

This module tests all aspects of the ResearchAgent including:
- Agent initialization and configuration
- Task validation and execution
- Exa and Firecrawl integration
- Research query generation and execution
- Error handling and recovery
- Async operations and timeouts
- Health monitoring and status reporting
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from src.agents.research_agent import ResearchAgent
from src.core.agent_protocol import AgentConfig, AgentExecutionError
from src.schemas.unified_models import TaskStatus


class TestResearchAgentInitialization:
    """Test ResearchAgent initialization and configuration."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        agent = ResearchAgent()

        assert agent.name == "research"
        assert agent.config.enabled is True
        assert "web_scraping" in agent.capabilities
        assert "data_collection" in agent.capabilities
        assert "market_research" in agent.capabilities
        assert "information_synthesis" in agent.capabilities
        assert "api_exploration" in agent.capabilities
        assert agent.config.model == "exa-research-pro"
        assert agent.config.timeout == 120
        assert agent.config.retry_attempts == 3
        assert agent.config.max_concurrent_tasks == 2

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = AgentConfig(
            name="custom-research",
            enabled=True,
            capabilities=["custom_research"],
            model="custom-model",
            timeout=300,
            retry_attempts=5,
            max_concurrent_tasks=3,
        )

        agent = ResearchAgent(config=custom_config)

        assert agent.name == "custom-research"
        assert agent.config.timeout == 300
        assert agent.config.retry_attempts == 5
        assert agent.config.max_concurrent_tasks == 3

    def test_init_creates_api_clients(self):
        """Test that initialization creates Exa and Firecrawl clients."""
        agent = ResearchAgent()

        assert agent.exa_client is not None
        assert agent.firecrawl_client is not None
        assert hasattr(agent, "exa_client")
        assert hasattr(agent, "firecrawl_client")

    def test_create_default_factory_method(self):
        """Test the create_default factory method."""
        agent = ResearchAgent.create_default()

        assert isinstance(agent, ResearchAgent)
        assert agent.name == "research"
        assert agent.config.enabled is True

    def test_default_config_creation(self):
        """Test _create_default_config method."""
        agent = ResearchAgent()
        config = agent._create_default_config()

        assert config.name == "research"
        assert config.enabled is True
        assert len(config.capabilities) == 5
        assert "web_scraping" in config.capabilities
        assert "data_collection" in config.capabilities
        assert "market_research" in config.capabilities
        assert "information_synthesis" in config.capabilities
        assert "api_exploration" in config.capabilities
        assert "exa_client" in config.tools
        assert "firecrawl_client" in config.tools


class TestResearchAgentTaskValidation:
    """Test task validation logic."""

    @pytest.fixture
    def agent(self):
        """Create a ResearchAgent for testing."""
        return ResearchAgent()

    @pytest.mark.asyncio
    async def test_validate_task_with_research_keywords(self, agent):
        """Test task validation with research-related keywords."""
        task_data = {
            "title": "Research authentication best practices",
            "description": "Gather information on secure login methods",
            "component_area": "security",
        }

        result = await agent.validate_task(task_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_task_with_multiple_research_keywords(self, agent):
        """Test validation with multiple research keywords."""
        test_cases = [
            {"title": "Collect market data", "description": "", "component_area": ""},
            {
                "title": "",
                "description": "Scrape competitor information",
                "component_area": "",
            },
            {"title": "", "description": "", "component_area": "analyze trends"},
            {
                "title": "Investigate performance issues",
                "description": "",
                "component_area": "",
            },
            {"title": "Explore API options", "description": "", "component_area": ""},
            {"title": "Gather user feedback", "description": "", "component_area": ""},
        ]

        for task_data in test_cases:
            result = await agent.validate_task(task_data)
            assert result is True, f"Failed for task: {task_data}"

    @pytest.mark.asyncio
    async def test_validate_task_without_research_keywords(self, agent):
        """Test task validation without research-related keywords."""
        task_data = {
            "title": "Implement user interface",
            "description": "Create login form",
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
            "title": "Research market trends",
            "description": "Data collection task",
            "component_area": "market",
        }

        result = await agent.validate_task(task_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_task_unhealthy_agent(self, agent):
        """Test validation when agent is unhealthy."""
        agent._is_healthy = False

        task_data = {
            "title": "Research competitors",
            "description": "Market research task",
            "component_area": "business",
        }

        result = await agent.validate_task(task_data)
        assert result is False


class TestResearchAgentTaskExecution:
    """Test task execution workflows."""

    @pytest.fixture
    def agent(self):
        """Create a ResearchAgent for testing."""
        return ResearchAgent()

    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing."""
        return {
            "id": 1,
            "title": "Research authentication best practices",
            "description": "Investigate secure authentication methods and JWT implementation",
            "component_area": "security",
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

    @pytest.fixture
    def mock_exa_search_result(self):
        """Mock Exa search result."""
        result = MagicMock()
        result.title = "JWT Authentication Best Practices"
        result.url = "https://example.com/jwt-guide"
        result.summary = "Comprehensive guide to JWT implementation"
        result.score = 0.95
        result.text = "JWT tokens provide secure stateless authentication. Best practices include proper secret management and token expiration."
        return result

    @pytest.fixture
    def mock_exa_search_response(self, mock_exa_search_result):
        """Mock Exa search response."""
        response = MagicMock()
        response.results = [mock_exa_search_result]
        return response

    @pytest.mark.asyncio
    async def test_execute_task_success(
        self, agent, sample_state, mock_exa_search_response
    ):
        """Test successful task execution."""
        # Mock the Exa client search
        agent.exa_client.search = AsyncMock(return_value=mock_exa_search_response)

        result_state = await agent.execute_task(sample_state)

        # Verify state updates
        assert "agent_outputs" in result_state
        assert "research" in result_state["agent_outputs"]

        research_output = result_state["agent_outputs"]["research"]
        assert research_output["status"] == TaskStatus.COMPLETED
        assert research_output["success"] is True
        assert "outputs" in research_output

        # Verify research output structure
        outputs = research_output["outputs"]
        assert outputs["research_type"] == "web_search"
        assert "queries_performed" in outputs
        assert "sources_found" in outputs
        assert "key_findings" in outputs
        assert "artifacts" in outputs
        assert "next_actions" in outputs

        # Verify sources were found
        assert len(outputs["sources_found"]) > 0
        source = outputs["sources_found"][0]
        assert source["title"] == "JWT Authentication Best Practices"
        assert source["url"] == "https://example.com/jwt-guide"
        assert source["relevance_score"] == 0.95

        # Verify key findings were extracted
        assert len(outputs["key_findings"]) > 0

        # Verify messages were added
        assert len(result_state["messages"]) > 0
        assert isinstance(result_state["messages"][0], HumanMessage)

    @pytest.mark.asyncio
    async def test_execute_task_multiple_queries(
        self, agent, sample_state, mock_exa_search_response
    ):
        """Test execution with multiple research queries."""
        agent.exa_client.search = AsyncMock(return_value=mock_exa_search_response)

        result_state = await agent.execute_task(sample_state)

        research_output = result_state["agent_outputs"]["research"]["outputs"]

        # Should generate multiple queries based on task data
        assert len(research_output["queries_performed"]) > 0
        queries = research_output["queries_performed"]

        # Verify queries are relevant to task
        assert any("authentication" in query.lower() for query in queries)

    @pytest.mark.asyncio
    async def test_execute_task_no_task_data(self, agent):
        """Test execution with missing task data."""
        state = {"task_id": 1}

        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.execute_task(state)

        assert "No task data provided" in str(exc_info.value)
        assert exc_info.value.agent_name == "research"
        assert exc_info.value.task_id == 0

    @pytest.mark.asyncio
    async def test_execute_task_exa_client_error(self, agent, sample_state):
        """Test execution with Exa client error."""
        # Mock Exa client error
        agent.exa_client.search = AsyncMock(side_effect=Exception("Exa API error"))

        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.execute_task(sample_state)

        assert exc_info.value.agent_name == "research"
        assert exc_info.value.task_id == 1

        # Verify error was recorded in state
        assert "agent_outputs" in sample_state
        assert "research" in sample_state["agent_outputs"]

        error_output = sample_state["agent_outputs"]["research"]
        assert error_output["status"] == TaskStatus.FAILED
        assert error_output["success"] is False
        assert "blocking_issues" in error_output

    @pytest.mark.asyncio
    async def test_execute_task_task_count_management(self, agent, sample_state):
        """Test that task count is properly managed during execution."""
        initial_count = agent._current_tasks

        # Mock successful search
        mock_response = MagicMock()
        mock_response.results = []
        agent.exa_client.search = AsyncMock(return_value=mock_response)

        await agent.execute_task(sample_state)

        # Task count should be back to initial value after completion
        assert agent._current_tasks == initial_count

    @pytest.mark.asyncio
    async def test_execute_task_timeout_scenario(self, agent, sample_state):
        """Test execution with timeout scenario."""

        # Mock a long-running operation that times out
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow response
            raise TimeoutError("Search timed out")

        agent.exa_client.search = AsyncMock(side_effect=slow_search)

        with pytest.raises(AgentExecutionError):
            await agent.execute_task(sample_state)


class TestResearchAgentResearchMethods:
    """Test research execution methods."""

    @pytest.fixture
    def agent(self):
        """Create a ResearchAgent for testing."""
        return ResearchAgent()

    @pytest.fixture
    def mock_search_result(self):
        """Mock search result for testing."""
        result = MagicMock()
        result.title = "Authentication Guide"
        result.url = "https://example.com/auth"
        result.summary = "Security best practices"
        result.score = 0.9
        result.text = "JWT tokens provide secure authentication. Use HTTPS for token transmission. Implement proper expiration policies."
        return result

    @pytest.mark.asyncio
    async def test_conduct_research_success(self, agent, mock_search_result):
        """Test successful research execution."""
        task_data = {
            "title": "JWT authentication",
            "description": "Research secure authentication methods",
            "component_area": "security",
        }

        # Mock search response
        mock_response = MagicMock()
        mock_response.results = [mock_search_result]
        agent.exa_client.search = AsyncMock(return_value=mock_response)

        result = await agent._conduct_research(task_data)

        assert result["research_type"] == "web_search"
        assert len(result["queries_performed"]) > 0
        assert len(result["sources_found"]) > 0
        assert len(result["key_findings"]) > 0
        assert len(result["next_actions"]) > 0

        # Verify source information
        source = result["sources_found"][0]
        assert source["title"] == "Authentication Guide"
        assert source["url"] == "https://example.com/auth"
        assert source["relevance_score"] == 0.9

    @pytest.mark.asyncio
    async def test_conduct_research_query_failure(self, agent):
        """Test research with query failures."""
        task_data = {
            "title": "Test research",
            "description": "Test description",
            "component_area": "test",
        }

        # Mock search failure
        agent.exa_client.search = AsyncMock(side_effect=Exception("Search failed"))

        result = await agent._conduct_research(task_data)

        # Should still return valid structure even with failures
        assert result["research_type"] == "web_search"
        assert "queries_performed" in result
        assert "sources_found" in result
        assert "key_findings" in result
        assert "next_actions" in result

    def test_generate_research_queries(self, agent):
        """Test research query generation."""
        queries = agent._generate_research_queries(
            title="JWT authentication",
            description="Implement secure token-based authentication with proper validation",
            component_area="security",
        )

        assert len(queries) <= 3  # Limited to 3 queries
        assert any("JWT authentication" in query for query in queries)
        assert any("security best practices" in query for query in queries)

    def test_generate_research_queries_empty_inputs(self, agent):
        """Test query generation with empty inputs."""
        queries = agent._generate_research_queries("", "", "")

        # Should handle empty inputs gracefully
        assert isinstance(queries, list)
        assert len(queries) <= 3

    def test_extract_key_findings(self, agent):
        """Test key findings extraction from text."""
        text = "JWT tokens provide stateless authentication. They are secure when properly implemented. HTTPS should always be used for transmission. Token expiration is crucial for security."
        query = "JWT authentication security"

        findings = agent._extract_key_findings(text, query)

        assert len(findings) <= 3  # Limited to 3 findings
        assert len(findings) > 0
        # Should find sentences with overlap with query terms
        assert any("JWT" in finding for finding in findings)

    def test_extract_key_findings_no_overlap(self, agent):
        """Test findings extraction with no query overlap."""
        text = "The weather is nice today. Birds are singing."
        query = "JWT authentication"

        findings = agent._extract_key_findings(text, query)

        # Should return empty list when no relevant findings
        assert findings == []

    def test_extract_key_findings_short_sentences(self, agent):
        """Test findings extraction filters out short sentences."""
        text = "JWT. Tokens provide secure authentication for web applications."
        query = "JWT authentication"

        findings = agent._extract_key_findings(text, query)

        # Should filter out sentences shorter than 50 characters
        assert len(findings) <= 1
        if findings:
            assert len(findings[0]) > 50


class TestResearchAgentHealthAndCleanup:
    """Test health monitoring and cleanup functionality."""

    @pytest.fixture
    def agent(self):
        """Create a ResearchAgent for testing."""
        return ResearchAgent()

    def test_get_health_status(self, agent):
        """Test health status reporting."""
        health = agent.get_health_status()

        assert health["name"] == "research"
        assert health["healthy"] is True
        assert health["current_tasks"] == 0
        assert health["max_concurrent_tasks"] == 2
        assert "web_scraping" in health["capabilities"]
        assert health["enabled"] is True
        assert "last_heartbeat" in health

    def test_get_config(self, agent):
        """Test configuration retrieval."""
        config = agent.get_config()

        assert isinstance(config, AgentConfig)
        assert config.name == "research"
        assert config.enabled is True
        assert "web_scraping" in config.capabilities

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
async def test_research_agent_integration():
    """Integration test for complete ResearchAgent workflow."""
    agent = ResearchAgent()

    # Create a complete task scenario
    task_data = {
        "id": 1,
        "title": "Research authentication security",
        "description": "Investigate JWT implementation and security best practices",
        "component_area": "security",
    }

    state = {
        "task_id": 1,
        "task_data": task_data,
        "messages": [],
        "agent_outputs": {},
    }

    # Mock successful search results
    mock_result1 = MagicMock()
    mock_result1.title = "JWT Security Best Practices"
    mock_result1.url = "https://example.com/jwt-security"
    mock_result1.summary = "Comprehensive JWT security guide"
    mock_result1.score = 0.95
    mock_result1.text = "JWT tokens should use strong secrets. Implement proper expiration. Use HTTPS for transmission."

    mock_result2 = MagicMock()
    mock_result2.title = "Authentication Implementation Guide"
    mock_result2.url = "https://example.com/auth-impl"
    mock_result2.summary = "Step-by-step authentication setup"
    mock_result2.score = 0.88
    mock_result2.text = (
        "Use bcrypt for password hashing. Implement rate limiting for login attempts."
    )

    mock_response = MagicMock()
    mock_response.results = [mock_result1, mock_result2]
    agent.exa_client.search = AsyncMock(return_value=mock_response)

    # Test validation
    can_handle = await agent.validate_task(task_data)
    assert can_handle is True

    # Test execution
    result_state = await agent.execute_task(state)

    # Verify complete workflow
    assert "agent_outputs" in result_state
    assert "research" in result_state["agent_outputs"]

    research_output = result_state["agent_outputs"]["research"]
    assert research_output["status"] == TaskStatus.COMPLETED
    assert research_output["success"] is True

    outputs = research_output["outputs"]
    assert outputs["research_type"] == "web_search"
    assert len(outputs["sources_found"]) == 2
    assert len(outputs["key_findings"]) > 0
    assert len(outputs["queries_performed"]) > 0

    # Verify sources contain expected information
    sources = outputs["sources_found"]
    assert any(source["title"] == "JWT Security Best Practices" for source in sources)
    assert any(
        source["title"] == "Authentication Implementation Guide" for source in sources
    )

    # Verify agent health
    health = agent.get_health_status()
    assert health["healthy"] is True
    assert health["current_tasks"] == 0  # Should be back to 0 after completion


@pytest.mark.asyncio
async def test_research_agent_concurrent_task_limits():
    """Test that ResearchAgent respects concurrent task limits."""
    # Create agent with max_concurrent_tasks = 2
    config = AgentConfig(
        name="research",
        enabled=True,
        capabilities=["research"],
        max_concurrent_tasks=2,
    )
    agent = ResearchAgent(config=config)

    # Simulate agent at capacity
    agent._current_tasks = 2

    task_data = {
        "title": "Research topic",
        "description": "Data collection task",
        "component_area": "research",
    }

    # Should reject task when at capacity
    can_handle = await agent.validate_task(task_data)
    assert can_handle is False

    # Should accept task when capacity available
    agent._current_tasks = 1
    can_handle = await agent.validate_task(task_data)
    assert can_handle is True


def test_research_agent_error_handling():
    """Test ResearchAgent error handling scenarios."""
    # Test initialization
    agent = ResearchAgent()
    assert agent is not None

    # Test AgentExecutionError properties
    error = AgentExecutionError("research", 123, "Test error", ValueError("Root cause"))
    assert error.agent_name == "research"
    assert error.task_id == 123
    assert error.cause is not None
    assert "Agent research failed task 123: Test error" in str(error)


@pytest.mark.asyncio
async def test_research_agent_with_firecrawl_integration():
    """Test ResearchAgent integration with Firecrawl client."""
    agent = ResearchAgent()

    # Verify Firecrawl client is available
    assert hasattr(agent, "firecrawl_client")
    assert agent.firecrawl_client is not None

    # In a real scenario, the research agent could use Firecrawl
    # for detailed content extraction from discovered URLs
    # This test verifies the client is properly initialized


@pytest.mark.asyncio
async def test_research_agent_empty_search_results():
    """Test research agent handling of empty search results."""
    agent = ResearchAgent()

    task_data = {
        "title": "Obscure technical topic",
        "description": "Research very specific technical details",
        "component_area": "technical",
    }

    # Mock empty search response
    mock_response = MagicMock()
    mock_response.results = []
    agent.exa_client.search = AsyncMock(return_value=mock_response)

    result = await agent._conduct_research(task_data)

    # Should handle empty results gracefully
    assert result["research_type"] == "web_search"
    assert result["sources_found"] == []
    assert result["key_findings"] == []
    assert len(result["next_actions"]) > 0  # Should still provide next actions


@pytest.mark.asyncio
async def test_research_agent_partial_search_failures():
    """Test research agent with partial search failures."""
    agent = ResearchAgent()

    task_data = {
        "title": "Multi-query research",
        "description": "Research topic with multiple aspects",
        "component_area": "complex",
    }

    # Mock mixed success/failure for different queries
    call_count = 0

    async def mixed_search_results(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First query succeeds
            mock_result = MagicMock()
            mock_result.title = "Successful Result"
            mock_result.url = "https://example.com/success"
            mock_result.summary = "Good information"
            mock_result.score = 0.9
            mock_result.text = "Useful content about the topic"

            mock_response = MagicMock()
            mock_response.results = [mock_result]
            return mock_response
        else:
            # Subsequent queries fail
            raise Exception("Search failed")

    agent.exa_client.search = AsyncMock(side_effect=mixed_search_results)

    result = await agent._conduct_research(task_data)

    # Should include successful results and handle failures gracefully
    assert len(result["sources_found"]) > 0  # At least one successful result
    assert len(result["queries_performed"]) > 1  # Multiple queries attempted

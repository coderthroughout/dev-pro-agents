"""Comprehensive test suite for core/agent_registry.py.

Tests the AgentRegistry class for agent registration, deregistration,
capability-based discovery, health status monitoring, and dynamic agent loading.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.core.agent_protocol import AgentConfig, BaseAgent
from src.core.agent_registry import AgentRegistry
from src.core.state import AgentState


class MockAgent(BaseAgent):
    """Mock agent for testing registry functionality."""

    def __init__(
        self,
        name: str,
        capabilities: list[str],
        enabled: bool = True,
        healthy: bool = True,
        current_tasks: int = 0,
        max_concurrent: int = 2,
    ):
        """Initialize mock agent with configurable parameters."""
        config = AgentConfig(
            name=name,
            capabilities=capabilities,
            enabled=enabled,
            max_concurrent_tasks=max_concurrent,
        )
        super().__init__(config)
        self._is_healthy = healthy
        self._current_tasks = current_tasks

    async def execute_task(self, state: AgentState) -> AgentState:
        """Mock task execution."""
        return state

    def set_health_status(self, healthy: bool):
        """Set agent health status for testing."""
        self._is_healthy = healthy

    def set_current_tasks(self, count: int):
        """Set current task count for testing."""
        self._current_tasks = count


class TestAgentRegistryBasicOperations:
    """Test basic agent registry operations."""

    @pytest.fixture
    def empty_registry(self):
        """Create empty agent registry for testing."""
        return AgentRegistry()

    @pytest.fixture
    def populated_registry(self):
        """Create registry with pre-registered agents."""
        registry = AgentRegistry()

        agents = [
            MockAgent("research", ["research", "analysis"]),
            MockAgent("coding", ["implementation", "coding"]),
            MockAgent("testing", ["testing", "validation"]),
            MockAgent("documentation", ["documentation", "writing"]),
        ]

        for agent in agents:
            registry.register(agent)

        return registry

    def test_registry_initialization(self, empty_registry):
        """Test registry initializes with empty state."""
        assert len(empty_registry._agents) == 0
        assert len(empty_registry._capabilities) == 0
        assert len(empty_registry._agent_modules) == 0

    def test_register_agent_success(self, empty_registry):
        """Test successful agent registration."""
        agent = MockAgent("test_agent", ["capability1", "capability2"])

        empty_registry.register(agent)

        assert "test_agent" in empty_registry._agents
        assert empty_registry._agents["test_agent"] == agent
        assert "capability1" in empty_registry._capabilities
        assert "capability2" in empty_registry._capabilities
        assert "test_agent" in empty_registry._capabilities["capability1"]
        assert "test_agent" in empty_registry._capabilities["capability2"]

    def test_register_duplicate_agent_raises_error(self, empty_registry):
        """Test registering agent with duplicate name raises ValueError."""
        agent1 = MockAgent("duplicate", ["capability1"])
        agent2 = MockAgent("duplicate", ["capability2"])

        empty_registry.register(agent1)

        with pytest.raises(ValueError, match="Agent 'duplicate' is already registered"):
            empty_registry.register(agent2)

    def test_register_disabled_agent_skipped(self, empty_registry):
        """Test disabled agents are not registered."""
        agent = MockAgent("disabled", ["capability1"], enabled=False)

        empty_registry.register(agent)

        assert "disabled" not in empty_registry._agents
        assert "capability1" not in empty_registry._capabilities

    def test_deregister_agent_success(self, populated_registry):
        """Test successful agent deregistration."""
        result = populated_registry.deregister("research")

        assert result is True
        assert "research" not in populated_registry._agents
        assert "research" not in populated_registry._capabilities.get("research", set())
        assert "research" not in populated_registry._capabilities.get("analysis", set())

    def test_deregister_nonexistent_agent(self, populated_registry):
        """Test deregistering non-existent agent returns False."""
        result = populated_registry.deregister("nonexistent")

        assert result is False

    def test_deregister_cleans_empty_capabilities(self, empty_registry):
        """Test deregistration removes empty capability sets."""
        agent = MockAgent("unique", ["unique_capability"])
        empty_registry.register(agent)

        assert "unique_capability" in empty_registry._capabilities

        empty_registry.deregister("unique")

        assert "unique_capability" not in empty_registry._capabilities

    def test_get_agent_success(self, populated_registry):
        """Test successful agent retrieval."""
        agent = populated_registry.get_agent("coding")

        assert agent is not None
        assert agent.name == "coding"
        assert "coding" in agent.capabilities

    def test_get_nonexistent_agent(self, populated_registry):
        """Test retrieving non-existent agent returns None."""
        agent = populated_registry.get_agent("nonexistent")

        assert agent is None


class TestAgentRegistryCapabilityBasedDiscovery:
    """Test capability-based agent discovery functionality."""

    @pytest.fixture
    def capability_registry(self):
        """Create registry with agents having overlapping capabilities."""
        registry = AgentRegistry()

        agents = [
            MockAgent("agent1", ["web_scraping", "data_analysis"], current_tasks=0),
            MockAgent("agent2", ["data_analysis", "reporting"], current_tasks=1),
            MockAgent("agent3", ["web_scraping", "testing"], current_tasks=0),
            MockAgent(
                "agent4", ["data_analysis"], current_tasks=2, max_concurrent=2
            ),  # At capacity
        ]

        for agent in agents:
            registry.register(agent)

        return registry

    def test_get_agent_for_capability_success(self, capability_registry):
        """Test finding agent for single capability."""
        agent = capability_registry.get_agent_for_capability("web_scraping")

        assert agent is not None
        assert "web_scraping" in agent.capabilities
        # Should select agent with lowest current task count
        assert agent.name in ["agent1", "agent3"]

    def test_get_agent_for_capability_selects_least_loaded(self, capability_registry):
        """Test capability-based selection prefers less loaded agents."""
        # Both agent1 and agent2 have data_analysis, but agent1 has 0 tasks vs agent2 with 1
        agent = capability_registry.get_agent_for_capability("data_analysis")

        assert agent is not None
        assert agent.name == "agent1"  # Should select least loaded

    def test_get_agent_for_nonexistent_capability(self, capability_registry):
        """Test finding agent for non-existent capability returns None."""
        agent = capability_registry.get_agent_for_capability("nonexistent")

        assert agent is None

    def test_get_agent_for_capability_excludes_unhealthy(self, capability_registry):
        """Test capability-based selection excludes unhealthy agents."""
        # Make agent1 unhealthy
        agent1 = capability_registry.get_agent("agent1")
        agent1.set_health_status(False)

        agent = capability_registry.get_agent_for_capability("web_scraping")

        # Should select agent3 since agent1 is unhealthy
        assert agent is not None
        assert agent.name == "agent3"

    def test_get_agent_for_capability_excludes_at_capacity(self, capability_registry):
        """Test capability-based selection excludes agents at capacity."""
        agent = capability_registry.get_agent_for_capability("data_analysis")

        # agent4 is at capacity (2 tasks, max 2), so should select agent1 or agent2
        assert agent is not None
        assert agent.name in ["agent1", "agent2"]

    def test_get_agents_for_multiple_capabilities_success(self, capability_registry):
        """Test finding agents that can handle multiple capabilities."""
        agents = capability_registry.get_agents_for_capabilities(
            ["data_analysis", "web_scraping"]
        )

        # Only agent1 has both capabilities
        assert len(agents) == 1
        assert agents[0].name == "agent1"

    def test_get_agents_for_multiple_capabilities_no_match(self, capability_registry):
        """Test finding agents for capabilities with no overlap."""
        agents = capability_registry.get_agents_for_capabilities(
            ["web_scraping", "reporting"]
        )

        # No agent has both capabilities
        assert len(agents) == 0

    def test_get_agents_for_empty_capabilities_list(self, capability_registry):
        """Test finding agents for empty capabilities list."""
        agents = capability_registry.get_agents_for_capabilities([])

        assert len(agents) == 0

    def test_get_agents_for_capabilities_excludes_unhealthy(self, capability_registry):
        """Test multi-capability selection excludes unhealthy agents."""
        # Make agent1 unhealthy
        agent1 = capability_registry.get_agent("agent1")
        agent1.set_health_status(False)

        agents = capability_registry.get_agents_for_capabilities(
            ["data_analysis", "web_scraping"]
        )

        # Should return empty list since agent1 is unhealthy
        assert len(agents) == 0


class TestAgentRegistryHealthAndStatus:
    """Test health monitoring and status reporting."""

    @pytest.fixture
    def empty_registry(self):
        """Create empty agent registry for testing."""
        return AgentRegistry()

    @pytest.fixture
    def health_registry(self):
        """Create registry with agents in various health states."""
        registry = AgentRegistry()

        agents = [
            MockAgent("healthy1", ["capability1"], healthy=True, current_tasks=1),
            MockAgent("healthy2", ["capability2"], healthy=True, current_tasks=0),
            MockAgent("unhealthy1", ["capability3"], healthy=False, current_tasks=0),
            MockAgent("unhealthy2", ["capability4"], healthy=False, current_tasks=2),
        ]

        for agent in agents:
            registry.register(agent)

        return registry

    def test_get_health_status_summary(self, health_registry):
        """Test retrieving overall health status summary."""
        status = health_registry.get_health_status()

        assert status["total_agents"] == 4
        assert status["healthy_agents"] == 2
        assert status["unhealthy_agents"] == 2
        assert status["registry_healthy"] is True  # At least one healthy agent
        assert status["total_capabilities"] == 4
        assert "agents" in status

    def test_get_health_status_individual_agents(self, health_registry):
        """Test individual agent health status reporting."""
        status = health_registry.get_health_status()

        healthy1_status = status["agents"]["healthy1"]
        assert healthy1_status["healthy"] is True
        assert healthy1_status["current_tasks"] == 1
        assert healthy1_status["enabled"] is True

        unhealthy1_status = status["agents"]["unhealthy1"]
        assert unhealthy1_status["healthy"] is False
        assert unhealthy1_status["current_tasks"] == 0

    def test_registry_unhealthy_when_all_agents_unhealthy(self, empty_registry):
        """Test registry reports unhealthy when all agents are unhealthy."""
        unhealthy_agent = MockAgent("unhealthy", ["capability"], healthy=False)
        empty_registry.register(unhealthy_agent)

        status = empty_registry.get_health_status()

        assert status["registry_healthy"] is False
        assert status["healthy_agents"] == 0
        assert status["unhealthy_agents"] == 1

    def test_registry_healthy_when_no_agents(self, empty_registry):
        """Test empty registry reports as unhealthy."""
        status = empty_registry.get_health_status()

        assert status["registry_healthy"] is False
        assert status["total_agents"] == 0
        assert status["healthy_agents"] == 0


class TestAgentRegistryListing:
    """Test agent and capability listing functionality."""

    @pytest.fixture
    def listing_registry(self):
        """Create registry for testing listing operations."""
        registry = AgentRegistry()

        agents = [
            MockAgent("alpha", ["capability_a", "capability_shared"]),
            MockAgent("beta", ["capability_b", "capability_shared"]),
            MockAgent("gamma", ["capability_c"]),
        ]

        for agent in agents:
            registry.register(agent)

        return registry

    def test_list_agents(self, listing_registry):
        """Test listing all registered agents."""
        agents = listing_registry.list_agents()

        assert len(agents) == 3
        agent_names = [agent.name for agent in agents]
        assert "alpha" in agent_names
        assert "beta" in agent_names
        assert "gamma" in agent_names

    def test_list_capabilities(self, listing_registry):
        """Test listing all available capabilities."""
        capabilities = listing_registry.list_capabilities()

        expected_capabilities = [
            "capability_a",
            "capability_b",
            "capability_c",
            "capability_shared",
        ]

        assert len(capabilities) == 4
        for capability in expected_capabilities:
            assert capability in capabilities

    def test_empty_registry_listings(self, empty_registry):
        """Test listing operations on empty registry."""
        assert len(empty_registry.list_agents()) == 0
        assert len(empty_registry.list_capabilities()) == 0


class TestAgentRegistryDynamicDiscovery:
    """Test dynamic agent discovery and loading functionality."""

    @pytest.fixture
    def mock_agent_module(self):
        """Create mock agent module for testing discovery."""
        mock_module = Mock()

        # Mock agent class that implements AgentProtocol
        mock_agent_class = Mock()
        mock_agent_class.__name__ = "TestAgent"
        mock_agent_class.__module__ = "test.module"
        mock_agent_class.__annotations__ = {"name": str}

        # Mock required methods
        mock_agent_class.execute_task = Mock()
        mock_agent_class.validate_task = Mock()
        mock_agent_class.get_config = Mock()
        mock_agent_class.get_health_status = Mock()
        mock_agent_class.cleanup = Mock()
        mock_agent_class.name = "test_agent"

        # Mock create_default factory method
        mock_instance = MockAgent("discovered_agent", ["discovered_capability"])
        mock_agent_class.create_default = Mock(return_value=mock_instance)

        # Set up module inspection
        mock_module.__path__ = ["/fake/path"]

        return mock_module, mock_agent_class, mock_instance

    def test_discover_agents_success(self, empty_registry, mock_agent_module):
        """Test successful agent discovery from module."""
        mock_module, mock_agent_class, mock_instance = mock_agent_module

        with (
            patch("importlib.import_module", return_value=mock_module),
            patch("pathlib.Path.glob", return_value=[Path("test_agent.py")]),
            patch("inspect.getmembers", return_value=[("TestAgent", mock_agent_class)]),
        ):
            count = empty_registry.discover_agents("test.agents")

            assert count == 1
            assert "discovered_agent" in empty_registry._agents
            assert "discovered_capability" in empty_registry._capabilities

    def test_discover_agents_import_error(self, empty_registry):
        """Test agent discovery handles import errors gracefully."""
        with patch(
            "importlib.import_module", side_effect=ImportError("Module not found")
        ):
            count = empty_registry.discover_agents("nonexistent.module")

            assert count == 0
            assert len(empty_registry._agents) == 0

    def test_discover_agents_in_module_success(self, empty_registry, mock_agent_module):
        """Test discovering agents in specific module."""
        mock_module, mock_agent_class, mock_instance = mock_agent_module

        with (
            patch("importlib.import_module", return_value=mock_module),
            patch("inspect.getmembers", return_value=[("TestAgent", mock_agent_class)]),
        ):
            count = empty_registry._discover_agents_in_module("test.module")

            assert count == 1
            assert "discovered_agent" in empty_registry._agents

    def test_discover_agents_in_module_no_factory_method(self, empty_registry):
        """Test discovery handles agents without create_default method."""
        mock_module = Mock()
        mock_agent_class = Mock()
        mock_agent_class.__name__ = "TestAgent"
        mock_agent_class.__module__ = "test.module"
        mock_agent_class.__annotations__ = {"name": str}

        # Required methods present
        mock_agent_class.execute_task = Mock()
        mock_agent_class.validate_task = Mock()
        mock_agent_class.get_config = Mock()
        mock_agent_class.get_health_status = Mock()
        mock_agent_class.cleanup = Mock()
        mock_agent_class.name = "test_agent"

        # No create_default method

        with (
            patch("importlib.import_module", return_value=mock_module),
            patch("inspect.getmembers", return_value=[("TestAgent", mock_agent_class)]),
        ):
            count = empty_registry._discover_agents_in_module("test.module")

            # Should skip agents without factory method
            assert count == 0

    def test_discover_agents_instantiation_failure(
        self, empty_registry, mock_agent_module
    ):
        """Test discovery handles agent instantiation failures."""
        mock_module, mock_agent_class, mock_instance = mock_agent_module
        mock_agent_class.create_default.side_effect = Exception("Instantiation failed")

        with (
            patch("importlib.import_module", return_value=mock_module),
            patch("inspect.getmembers", return_value=[("TestAgent", mock_agent_class)]),
        ):
            count = empty_registry._discover_agents_in_module("test.module")

            # Should handle instantiation failure gracefully
            assert count == 0

    def test_implements_agent_protocol_valid_agent(self, empty_registry):
        """Test _implements_agent_protocol with valid agent class."""
        mock_class = Mock()
        mock_class.execute_task = Mock()
        mock_class.validate_task = Mock()
        mock_class.get_config = Mock()
        mock_class.get_health_status = Mock()
        mock_class.cleanup = Mock()
        mock_class.name = "test"

        result = empty_registry._implements_agent_protocol(mock_class)

        assert result is True

    def test_implements_agent_protocol_missing_methods(self, empty_registry):
        """Test _implements_agent_protocol with invalid agent class."""
        mock_class = Mock()
        # Missing required methods

        result = empty_registry._implements_agent_protocol(mock_class)

        assert result is False

    def test_implements_agent_protocol_exception_handling(self, empty_registry):
        """Test _implements_agent_protocol handles exceptions gracefully."""
        mock_class = Mock()
        mock_class.execute_task = Mock(side_effect=Exception("Error"))

        result = empty_registry._implements_agent_protocol(mock_class)

        assert result is False


class TestAgentRegistryCleanup:
    """Test registry cleanup and resource management."""

    @pytest.fixture
    def cleanup_registry(self):
        """Create registry with agents for cleanup testing."""
        registry = AgentRegistry()

        agents = [
            MockAgent("agent1", ["capability1"]),
            MockAgent("agent2", ["capability2"]),
            MockAgent("agent3", ["capability3"]),
        ]

        for agent in agents:
            registry.register(agent)

        return registry

    @pytest.mark.asyncio
    async def test_cleanup_all_agents_success(self, cleanup_registry):
        """Test successful cleanup of all agents."""
        # Verify agents are registered
        assert len(cleanup_registry._agents) == 3

        await cleanup_registry.cleanup_all()

        # Verify all data structures are cleared
        assert len(cleanup_registry._agents) == 0
        assert len(cleanup_registry._capabilities) == 0
        assert len(cleanup_registry._agent_modules) == 0

    @pytest.mark.asyncio
    async def test_cleanup_handles_agent_cleanup_errors(self, cleanup_registry):
        """Test cleanup handles individual agent cleanup errors gracefully."""
        # Mock one agent to raise an error during cleanup
        agent1 = cleanup_registry.get_agent("agent1")
        agent1.cleanup = Mock(side_effect=Exception("Cleanup error"))

        # Should not raise exception
        await cleanup_registry.cleanup_all()

        # Should still clear registry data structures
        assert len(cleanup_registry._agents) == 0
        assert len(cleanup_registry._capabilities) == 0


class TestAgentRegistryEdgeCases:
    """Test edge cases and error conditions."""

    def test_register_agent_with_empty_capabilities(self, empty_registry):
        """Test registering agent with no capabilities."""
        agent = MockAgent("no_caps", [])

        empty_registry.register(agent)

        assert "no_caps" in empty_registry._agents
        # No new capabilities should be added
        assert len(empty_registry._capabilities) == 0

    def test_capability_index_consistency_after_multiple_operations(
        self, empty_registry
    ):
        """Test capability index remains consistent after multiple operations."""
        # Register agents with overlapping capabilities
        agent1 = MockAgent("agent1", ["shared", "unique1"])
        agent2 = MockAgent("agent2", ["shared", "unique2"])
        agent3 = MockAgent("agent3", ["shared"])

        empty_registry.register(agent1)
        empty_registry.register(agent2)
        empty_registry.register(agent3)

        # Verify initial state
        assert len(empty_registry._capabilities["shared"]) == 3
        assert len(empty_registry._capabilities["unique1"]) == 1
        assert len(empty_registry._capabilities["unique2"]) == 1

        # Deregister one agent
        empty_registry.deregister("agent2")

        # Verify capability index is updated correctly
        assert len(empty_registry._capabilities["shared"]) == 2
        assert "unique2" not in empty_registry._capabilities
        assert "agent2" not in empty_registry._capabilities["shared"]

    def test_get_agent_for_capability_with_all_agents_disabled(self, empty_registry):
        """Test capability search when all matching agents are disabled."""
        # Register agents but make them unhealthy/disabled
        agent1 = MockAgent("agent1", ["target_capability"], healthy=False)
        MockAgent("agent2", ["target_capability"], enabled=False)

        empty_registry.register(agent1)  # unhealthy agent will be registered
        # disabled agent won't be registered due to enabled=False

        result = empty_registry.get_agent_for_capability("target_capability")

        # Should return None since no healthy, enabled agents available
        assert result is None

    def test_concurrent_registration_deregistration(self, empty_registry):
        """Test registry handles concurrent operations safely."""
        # Simulate concurrent operations
        agent1 = MockAgent("concurrent1", ["capability1"])
        agent2 = MockAgent("concurrent2", ["capability1"])

        # Register both agents
        empty_registry.register(agent1)
        empty_registry.register(agent2)

        # Deregister one while the other remains
        empty_registry.deregister("concurrent1")

        # Verify state consistency
        assert "concurrent1" not in empty_registry._agents
        assert "concurrent2" in empty_registry._agents
        assert "concurrent1" not in empty_registry._capabilities["capability1"]
        assert "concurrent2" in empty_registry._capabilities["capability1"]

    def test_agent_module_tracking(self, empty_registry):
        """Test agent module path tracking functionality."""
        agent = MockAgent("tracked_agent", ["capability"])

        # Simulate discovery with module tracking
        empty_registry.register(agent)
        empty_registry._agent_modules["tracked_agent"] = "test.module.path"

        # Deregister should clean up module tracking
        empty_registry.deregister("tracked_agent")

        assert "tracked_agent" not in empty_registry._agent_modules


class TestAgentRegistryPerformance:
    """Test registry performance characteristics."""

    def test_large_scale_agent_registration(self, empty_registry):
        """Test registry performance with many agents."""
        # Register a large number of agents
        agent_count = 100
        capabilities_per_agent = 5

        for i in range(agent_count):
            capabilities = [
                f"capability_{j}" for j in range(i % capabilities_per_agent)
            ]
            agent = MockAgent(f"agent_{i}", capabilities)
            empty_registry.register(agent)

        # Verify all agents registered
        assert len(empty_registry._agents) == agent_count

        # Test capability lookup performance
        agent = empty_registry.get_agent_for_capability("capability_0")
        assert agent is not None

    def test_capability_search_efficiency(self, empty_registry):
        """Test efficiency of capability-based searches."""
        # Create agents with overlapping capabilities
        common_capability = "common"

        for i in range(50):
            capabilities = [common_capability, f"unique_{i}"]
            agent = MockAgent(f"agent_{i}", capabilities, current_tasks=i % 5)
            empty_registry.register(agent)

        # Search should efficiently find least loaded agent
        agent = empty_registry.get_agent_for_capability(common_capability)
        assert agent is not None
        assert agent._current_tasks == 0  # Should find the least loaded

    def test_multi_capability_search_performance(self, empty_registry):
        """Test performance of multi-capability searches."""
        # Create agents with varying capability combinations
        base_capabilities = ["web", "data", "analysis", "reporting", "testing"]

        for i in range(30):
            # Each agent gets 2-3 capabilities
            capabilities = base_capabilities[i % 3 : (i % 3) + 2]
            agent = MockAgent(f"agent_{i}", capabilities)
            empty_registry.register(agent)

        # Test multi-capability search
        agents = empty_registry.get_agents_for_capabilities(["web", "data"])

        # Should find agents efficiently
        assert len(agents) > 0
        for agent in agents:
            assert "web" in agent.capabilities
            assert "data" in agent.capabilities

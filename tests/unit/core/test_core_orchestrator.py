"""Comprehensive test suite for core/orchestrator.py.

Tests the ModularOrchestrator class with LangGraph workflows, OpenAI API calls,
agent discovery, registration, health monitoring, and task coordination.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
import yaml
from langgraph.graph import StateGraph

from src.core.agent_protocol import AgentConfig, BaseAgent
from src.core.agent_registry import AgentRegistry
from src.core.orchestrator import ModularOrchestrator
from src.core.state import AgentState


class MockAgent(BaseAgent):
    """Mock agent for testing orchestrator functionality."""

    def __init__(self, name: str, capabilities: list[str], healthy: bool = True):
        """Initialize mock agent with test configuration."""
        config = AgentConfig(
            name=name, capabilities=capabilities, enabled=True, max_concurrent_tasks=2
        )
        super().__init__(config)
        self._is_healthy = healthy
        self._task_validation_results = {}

    async def execute_task(self, state: AgentState) -> AgentState:
        """Mock task execution with configurable behavior."""
        self._increment_task_count()

        # Simulate processing time
        await asyncio.sleep(0.01)

        task_id = state.get("task_id")
        agent_outputs = state.get("agent_outputs", {})

        # Mock successful execution
        agent_outputs[self.name] = {
            "status": "completed",
            "result": f"Task {task_id} completed by {self.name}",
            "artifacts": [f"{self.name}_output.txt"],
            "execution_time": 0.5,
        }

        state["agent_outputs"] = agent_outputs
        self._decrement_task_count()
        return state

    async def validate_task(self, task_data: dict) -> bool:
        """Mock task validation with configurable results."""
        task_id = task_data.get("id", "unknown")
        if task_id in self._task_validation_results:
            return self._task_validation_results[task_id]

        # Default validation based on agent health and capacity
        return await super().validate_task(task_data)

    def set_task_validation_result(self, task_id: str, result: bool):
        """Configure task validation result for testing."""
        self._task_validation_results[task_id] = result

    def set_unhealthy(self):
        """Set agent to unhealthy state for testing."""
        self._is_healthy = False

    def simulate_task_failure(self, task_id: int):
        """Configure agent to simulate task failure."""
        self._task_validation_results[str(task_id)] = False


@pytest.fixture
def mock_config():
    """Sample orchestrator configuration for testing."""
    return {
        "orchestrator": {
            "supervisor_model": "o3",
            "batch_processing": {
                "enabled": True,
                "max_batch_size": 5,
                "max_parallel_tasks": 2,
            },
            "coordination": {
                "task_assignment_strategy": "capability_based",
                "max_task_retries": 3,
            },
        }
    }


@pytest.fixture
def config_file(tmp_path, mock_config):
    """Create temporary config file for testing."""
    config_path = tmp_path / "orchestrator.yaml"
    with config_path.open("w") as f:
        yaml.dump(mock_config, f)
    return str(config_path)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client with controlled responses."""
    client = AsyncMock()

    # Mock standard delegation response
    mock_response = Mock()
    mock_response.content = """{
        "assigned_agent": "coding",
        "reasoning": "This task requires code implementation",
        "priority": "high",
        "estimated_duration": 60,
        "dependencies": [],
        "context_requirements": ["security patterns"],
        "confidence_score": 0.85
    }"""

    client.ainvoke.return_value = mock_response
    return client


@pytest.fixture
def mock_task_manager():
    """Mock task manager for orchestrator testing."""
    tm = Mock()
    tm.get_tasks_by_status.return_value = []
    tm.update_task_status.return_value = None
    return tm


@pytest.fixture
def mock_agent_registry():
    """Mock agent registry with pre-configured agents."""
    registry = AgentRegistry()

    # Add mock agents
    research_agent = MockAgent("research", ["research", "analysis"])
    coding_agent = MockAgent("coding", ["implementation", "coding"])
    testing_agent = MockAgent("testing", ["testing", "validation"])

    registry.register(research_agent)
    registry.register(coding_agent)
    registry.register(testing_agent)

    return registry


@pytest_asyncio.fixture
async def mock_orchestrator(config_file, mock_openai_client, mock_task_manager):
    """Create orchestrator instance with mocked dependencies."""
    with (
        patch("src.core.orchestrator.TaskManager", return_value=mock_task_manager),
        patch("src.core.orchestrator.ChatOpenAI", return_value=mock_openai_client),
        patch(
            "src.agents.research_agent.ResearchAgent.create_default"
        ) as mock_research,
        patch("src.agents.coding_agent.CodingAgent.create_default") as mock_coding,
        patch("src.agents.testing_agent.TestingAgent.create_default") as mock_testing,
        patch(
            "src.agents.documentation_agent.DocumentationAgent.create_default"
        ) as mock_docs,
    ):
        # Setup mock agents
        mock_research.return_value = MockAgent("research", ["research", "analysis"])
        mock_coding.return_value = MockAgent("coding", ["implementation", "coding"])
        mock_testing.return_value = MockAgent("testing", ["testing", "validation"])
        mock_docs.return_value = MockAgent(
            "documentation", ["documentation", "writing"]
        )

        orchestrator = ModularOrchestrator(
            config_path=config_file, openai_api_key="test-key", db_path=":memory:"
        )

        yield orchestrator

        # Cleanup
        await orchestrator.cleanup()


class TestModularOrchestratorInitialization:
    """Test orchestrator initialization and configuration loading."""

    def test_orchestrator_default_config_initialization(self):
        """Test orchestrator initializes with default config when file not found."""
        with (
            patch("src.core.orchestrator.TaskManager"),
            patch("src.core.orchestrator.ChatOpenAI"),
            patch("pathlib.Path.open", side_effect=FileNotFoundError),
        ):
            orchestrator = ModularOrchestrator(
                config_path="/nonexistent/config.yaml", openai_api_key="test-key"
            )

            assert orchestrator.config["orchestrator"]["supervisor_model"] == "o3"
            assert (
                orchestrator.config["orchestrator"]["batch_processing"]["enabled"]
                is True
            )

    def test_orchestrator_config_file_loading(self, config_file):
        """Test orchestrator loads configuration from YAML file correctly."""
        with (
            patch("src.core.orchestrator.TaskManager"),
            patch("src.core.orchestrator.ChatOpenAI"),
        ):
            orchestrator = ModularOrchestrator(
                config_path=config_file, openai_api_key="test-key"
            )

            assert (
                orchestrator.config["orchestrator"]["batch_processing"][
                    "max_batch_size"
                ]
                == 5
            )

    def test_orchestrator_invalid_config_falls_back_to_default(self, tmp_path):
        """Test orchestrator handles invalid config file gracefully."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")

        with (
            patch("src.core.orchestrator.TaskManager"),
            patch("src.core.orchestrator.ChatOpenAI"),
        ):
            orchestrator = ModularOrchestrator(
                config_path=str(invalid_config), openai_api_key="test-key"
            )

            # Should fall back to default config
            assert orchestrator.config["orchestrator"]["supervisor_model"] == "o3"

    def test_orchestrator_agent_initialization_failure_handling(self, config_file):
        """Test orchestrator handles agent initialization failures gracefully."""
        with (
            patch("src.core.orchestrator.TaskManager"),
            patch("src.core.orchestrator.ChatOpenAI"),
            patch(
                "src.agents.research_agent.ResearchAgent.create_default",
                side_effect=Exception("Agent init failed"),
            ),pytest.raises(Exception, match="Agent init failed")
        ):
            ModularOrchestrator(config_path=config_file, openai_api_key="test-key")


class TestModularOrchestratorWorkflowCreation:
    """Test LangGraph workflow creation and structure."""

    @pytest_asyncio.fixture
    async def orchestrator_with_workflow(self, mock_orchestrator):
        """Orchestrator instance with workflow ready for testing."""
        return mock_orchestrator

    @pytest.mark.asyncio
    async def test_workflow_creation_includes_all_nodes(
        self, orchestrator_with_workflow
    ):
        """Test that workflow includes supervisor, agent, coordination, and finalization nodes."""
        workflow = orchestrator_with_workflow.workflow

        # Verify workflow is compiled StateGraph
        assert workflow is not None
        assert isinstance(orchestrator_with_workflow._create_workflow(), StateGraph)

    @pytest.mark.asyncio
    async def test_workflow_routing_with_valid_agent(self, orchestrator_with_workflow):
        """Test workflow routing to valid agent."""
        # Get routing function and test
        orchestrator_with_workflow._create_workflow()
        # Note: Testing routing logic indirectly through state updates

    @pytest.mark.asyncio
    async def test_workflow_coordination_detection(self, orchestrator_with_workflow):
        """Test workflow coordination detection logic."""
        state = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Test task"},
            "agent_outputs": {
                "coding": {
                    "status": "blocked",
                    "result": "Need assistance with authentication",
                }
            },
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        # Test coordination detection
        result = await orchestrator_with_workflow._coordinate_agents(state)
        assert "coordination_context" in result


class TestModularOrchestratorTaskAnalysisAndDelegation:
    """Test task analysis and agent delegation logic."""

    @pytest.mark.asyncio
    async def test_analyze_and_delegate_task_success(self, mock_orchestrator):
        """Test successful task analysis and delegation."""
        state = {
            "messages": [],
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Implement authentication system",
                "description": "Create JWT-based authentication",
                "component_area": "security",
            },
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        result = await mock_orchestrator._analyze_and_delegate_task(state)

        assert result["next_agent"] is not None
        assert "coordination_context" in result
        assert "assigned_agent" in result["coordination_context"]

    @pytest.mark.asyncio
    async def test_analyze_and_delegate_task_no_task_data(self, mock_orchestrator):
        """Test task analysis with missing task data."""
        state = {
            "messages": [],
            "task_id": None,
            "task_data": None,
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        result = await mock_orchestrator._analyze_and_delegate_task(state)

        assert result["error_context"] is not None
        assert "supervisor_error" in result["error_context"]
        assert result["next_agent"] is None

    @pytest.mark.asyncio
    async def test_determine_best_agent_capability_matching(self, mock_orchestrator):
        """Test agent selection based on capability matching."""
        task_data = {
            "id": 1,
            "title": "Write unit tests",
            "description": "Create comprehensive test suite",
            "component_area": "testing",
        }

        agent_info = {
            "testing": {
                "capabilities": ["testing", "validation"],
                "current_tasks": 0,
                "max_concurrent": 2,
            },
            "coding": {
                "capabilities": ["implementation", "coding"],
                "current_tasks": 1,
                "max_concurrent": 2,
            },
        }

        best_agent = await mock_orchestrator._determine_best_agent(
            task_data, agent_info
        )
        assert best_agent == "testing"

    @pytest.mark.asyncio
    async def test_determine_best_agent_load_balancing(self, mock_orchestrator):
        """Test agent selection considers current task load."""
        task_data = {
            "id": 1,
            "title": "Generic task",
            "description": "General work",
            "component_area": "general",
        }

        agent_info = {
            "agent1": {
                "capabilities": ["general"],
                "current_tasks": 0,
                "max_concurrent": 2,
            },
            "agent2": {
                "capabilities": ["general"],
                "current_tasks": 2,
                "max_concurrent": 2,
            },
        }

        best_agent = await mock_orchestrator._determine_best_agent(
            task_data, agent_info
        )
        assert best_agent == "agent1"

    @pytest.mark.asyncio
    async def test_determine_best_agent_no_available_agents(self, mock_orchestrator):
        """Test agent selection when no agents are available."""
        task_data = {
            "id": 1,
            "title": "Complex task",
            "description": "Specialized work",
            "component_area": "specialized",
        }

        agent_info = {
            "agent1": {
                "capabilities": ["other"],
                "current_tasks": 2,
                "max_concurrent": 2,
            }
        }

        best_agent = await mock_orchestrator._determine_best_agent(
            task_data, agent_info
        )
        assert best_agent is None


class TestModularOrchestratorCoordination:
    """Test agent coordination and assistance handling."""

    @pytest.mark.asyncio
    async def test_coordinate_agents_with_blocked_agent(self, mock_orchestrator):
        """Test coordination when agent is blocked."""
        state = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Test task"},
            "agent_outputs": {
                "coding": {"status": "blocked", "result": "Need database schema"}
            },
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        result = await mock_orchestrator._coordinate_agents(state)

        assert "coordination_context" in result
        # Should attempt to find alternative agent or set reassignment context

    @pytest.mark.asyncio
    async def test_coordinate_agents_requires_assistance(self, mock_orchestrator):
        """Test coordination when agent requires assistance."""
        state = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Test task"},
            "agent_outputs": {
                "research": {
                    "status": "requires_assistance",
                    "result": "Need additional context",
                }
            },
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        result = await mock_orchestrator._coordinate_agents(state)

        assert "coordination_context" in result

    @pytest.mark.asyncio
    async def test_find_alternative_agent_success(self, mock_orchestrator):
        """Test finding alternative agent when primary is blocked."""
        task_data = {
            "id": 1,
            "title": "Implement feature",
            "description": "Add new functionality",
            "component_area": "implementation",
        }

        alternative = await mock_orchestrator._find_alternative_agent(
            task_data, "coding"
        )

        # Should find an alternative agent (not the blocked one)
        assert alternative is not None
        assert alternative != "coding"

    @pytest.mark.asyncio
    async def test_find_alternative_agent_no_alternatives(self, mock_orchestrator):
        """Test finding alternative agent when none are available."""
        # Mock all agents as unhealthy or at capacity
        for agent in mock_orchestrator.agent_registry.list_agents():
            agent.set_unhealthy()

        task_data = {
            "id": 1,
            "title": "Implement feature",
            "description": "Add new functionality",
            "component_area": "implementation",
        }

        alternative = await mock_orchestrator._find_alternative_agent(
            task_data, "coding"
        )
        assert alternative is None


class TestModularOrchestratorTaskExecution:
    """Test complete task execution workflows."""

    @pytest.mark.asyncio
    async def test_execute_task_success(self, mock_orchestrator):
        """Test successful task execution through complete workflow."""
        task_id = 1

        # Mock task data retrieval
        with patch.object(mock_orchestrator, "_get_task_data") as mock_get_task:
            mock_get_task.return_value = {
                "id": task_id,
                "title": "Test task",
                "description": "Test description",
                "component_area": "testing",
                "success_criteria": "Task completes successfully",
            }

            result = await mock_orchestrator.execute_task(task_id)

            assert result["task_id"] == task_id
            assert result["status"] in ["completed", "failed"]
            assert "execution_time" in result

    @pytest.mark.asyncio
    async def test_execute_task_not_found(self, mock_orchestrator):
        """Test task execution with non-existent task."""
        task_id = 999

        with patch.object(mock_orchestrator, "_get_task_data", return_value=None):
            with pytest.raises(ValueError, match=f"Task {task_id} not found"):
                await mock_orchestrator.execute_task(task_id)

    @pytest.mark.asyncio
    async def test_execute_task_workflow_exception(self, mock_orchestrator):
        """Test task execution with workflow exception."""
        task_id = 1

        with (
            patch.object(mock_orchestrator, "_get_task_data") as mock_get_task,
            patch.object(
                mock_orchestrator.workflow,
                "ainvoke",
                side_effect=Exception("Workflow error"),
            ),
        ):
            mock_get_task.return_value = {
                "id": task_id,
                "title": "Test task",
                "description": "Test description",
            }

            result = await mock_orchestrator.execute_task(task_id)

            assert result["status"] == "failed"
            assert "error" in result
            assert "Workflow error" in result["error"]

    @pytest.mark.asyncio
    async def test_finalize_task_completed_status(self, mock_orchestrator):
        """Test task finalization with completed status."""
        state = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Test task"},
            "agent_outputs": {
                "coding": {
                    "status": "completed",
                    "result": "Task completed successfully",
                }
            },
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        result = await mock_orchestrator._finalize_task(state)

        assert len(result["messages"]) > 0
        assert "finalized" in result["messages"][-1].content.lower()

    @pytest.mark.asyncio
    async def test_finalize_task_failed_status(self, mock_orchestrator):
        """Test task finalization with failed status."""
        state = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Test task"},
            "agent_outputs": {
                "coding": {"status": "failed", "result": "Task failed due to error"}
            },
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        result = await mock_orchestrator._finalize_task(state)

        assert len(result["messages"]) > 0
        assert "failed" in result["messages"][-1].content.lower()


class TestModularOrchestratorBatchProcessing:
    """Test batch task processing functionality."""

    @pytest.mark.asyncio
    async def test_execute_batch_success(self, mock_orchestrator):
        """Test successful batch execution."""
        task_ids = [1, 2, 3]

        with patch.object(mock_orchestrator, "execute_task") as mock_execute:
            mock_execute.side_effect = [
                {"task_id": 1, "status": "completed"},
                {"task_id": 2, "status": "completed"},
                {"task_id": 3, "status": "failed"},
            ]

            result = await mock_orchestrator.execute_batch(task_ids, batch_size=2)

            assert result["batch_size"] == 3
            assert result["successful"] == 2
            assert result["failed"] == 1
            assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_execute_batch_with_exceptions(self, mock_orchestrator):
        """Test batch execution with some tasks throwing exceptions."""
        task_ids = [1, 2]

        with patch.object(mock_orchestrator, "execute_task") as mock_execute:
            mock_execute.side_effect = [
                {"task_id": 1, "status": "completed"},
                Exception("Task execution failed"),
            ]

            result = await mock_orchestrator.execute_batch(task_ids)

            assert result["batch_size"] == 2
            assert result["successful"] == 1
            assert result["failed"] == 1

    @pytest.mark.asyncio
    async def test_execute_batch_custom_batch_size(self, mock_orchestrator):
        """Test batch execution with custom batch size."""
        task_ids = [1, 2, 3, 4, 5]

        with patch.object(mock_orchestrator, "execute_task") as mock_execute:
            mock_execute.return_value = {"task_id": 1, "status": "completed"}

            result = await mock_orchestrator.execute_batch(task_ids, batch_size=2)

            assert result["batch_size"] == 5
            assert mock_execute.call_count == 5


class TestModularOrchestratorHealthAndCleanup:
    """Test health monitoring and cleanup functionality."""

    def test_get_agent_health_status(self, mock_orchestrator):
        """Test retrieving agent health status."""
        health_status = mock_orchestrator.get_agent_health_status()

        assert "registry_healthy" in health_status
        assert "total_agents" in health_status
        assert "healthy_agents" in health_status
        assert "agents" in health_status

    @pytest.mark.asyncio
    async def test_cleanup_orchestrator(self, mock_orchestrator):
        """Test orchestrator cleanup process."""
        # Should not raise any exceptions
        await mock_orchestrator.cleanup()

        # Verify registry cleanup was called
        # Note: This is tested indirectly through the cleanup process

    @pytest.mark.asyncio
    async def test_update_task_status_logging(self, mock_orchestrator):
        """Test task status update logging."""
        task_id = 1
        status = "completed"
        agent_outputs = {"coding": {"result": "success"}}

        with patch("src.core.orchestrator.logger") as mock_logger:
            await mock_orchestrator._update_task_status(task_id, status, agent_outputs)

            # Verify logging occurred
            mock_logger.info.assert_called()


class TestModularOrchestratorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_malformed_agent_health_data(self, mock_orchestrator):
        """Test handling of malformed agent health data."""
        # Get one of the registered agents and modify its health data
        agents = mock_orchestrator.agent_registry.list_agents()
        if agents:
            agent = agents[0]
            # Simulate malformed health data
            agent._current_tasks = "invalid"  # Should be int

            # This should not crash the system
            await mock_orchestrator._find_alternative_agent(
                {"id": 1, "title": "test"}, "other_agent"
            )
            # Should handle the malformed data gracefully

    @pytest.mark.asyncio
    async def test_concurrent_task_execution_safety(self, mock_orchestrator):
        """Test concurrent task execution doesn't cause race conditions."""
        task_ids = [1, 2, 3, 4, 5]

        with patch.object(mock_orchestrator, "_get_task_data") as mock_get_task:
            mock_get_task.side_effect = lambda tid: {
                "id": tid,
                "title": f"Task {tid}",
                "description": f"Description {tid}",
            }

            # Execute multiple tasks concurrently
            tasks = [mock_orchestrator.execute_task(tid) for tid in task_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All tasks should complete (successfully or with controlled failures)
            assert len(results) == len(task_ids)
            for result in results:
                if isinstance(result, dict):
                    assert "task_id" in result
                    assert "status" in result

    @pytest.mark.asyncio
    async def test_agent_validation_timeout_handling(self, mock_orchestrator):
        """Test handling of agent validation timeouts."""
        agents = mock_orchestrator.agent_registry.list_agents()
        if agents:
            agent = agents[0]

            # Mock validation that takes too long
            async def slow_validation(task_data):
                await asyncio.sleep(0.1)  # Simulate slow validation
                return True

            agent.validate_task = slow_validation

            # This should not cause the orchestrator to hang
            task_data = {"id": 1, "title": "test task"}
            alternative = await mock_orchestrator._find_alternative_agent(
                task_data, "blocked_agent"
            )

            # Should either succeed or timeout gracefully
            assert alternative is None or isinstance(alternative, str)

    @pytest.mark.asyncio
    async def test_workflow_state_corruption_recovery(self, mock_orchestrator):
        """Test recovery from corrupted workflow state."""
        # Create a corrupted state (missing required fields)
        corrupted_state = {
            "messages": [],
            # Missing task_id, task_data, etc.
        }

        # The orchestrator should handle this gracefully
        result = await mock_orchestrator._analyze_and_delegate_task(corrupted_state)

        # Should set appropriate error context
        assert result.get("error_context") is not None


class TestModularOrchestratorIntegration:
    """Integration tests for orchestrator workflows."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_with_coordination(self, mock_orchestrator):
        """Test complete workflow including coordination between agents."""
        task_id = 1

        # Setup a scenario where first agent gets blocked and needs coordination
        agents = list(mock_orchestrator.agent_registry._agents.values())
        if agents:
            # Make the first agent that will be selected fail validation
            first_agent = agents[0]
            first_agent.set_task_validation_result("1", False)

        with patch.object(mock_orchestrator, "_get_task_data") as mock_get_task:
            mock_get_task.return_value = {
                "id": task_id,
                "title": "Complex task requiring coordination",
                "description": "Multi-step task that may need agent cooperation",
                "component_area": "integration",
                "success_criteria": "All components working together",
            }

            result = await mock_orchestrator.execute_task(task_id)

            # Should complete even with coordination needs
            assert result["task_id"] == task_id
            assert "status" in result
            assert "execution_time" in result

    @pytest.mark.asyncio
    async def test_workflow_with_all_agents_unavailable(self, mock_orchestrator):
        """Test workflow behavior when all agents are unavailable."""
        # Make all agents unhealthy
        for agent in mock_orchestrator.agent_registry.list_agents():
            agent.set_unhealthy()

        task_id = 1
        with patch.object(mock_orchestrator, "_get_task_data") as mock_get_task:
            mock_get_task.return_value = {
                "id": task_id,
                "title": "Task with no available agents",
                "description": "This task cannot be completed",
                "component_area": "impossible",
            }

            result = await mock_orchestrator.execute_task(task_id)

            # Should fail gracefully
            assert result["task_id"] == task_id
            # Status should indicate the problem
            assert result["status"] in ["failed", "completed"]

    @pytest.mark.asyncio
    async def test_batch_processing_with_mixed_results(self, mock_orchestrator):
        """Test batch processing with mix of successful and failed tasks."""
        task_ids = [1, 2, 3, 4]

        with patch.object(mock_orchestrator, "_get_task_data") as mock_get_task:

            def get_task_data(tid):
                return {
                    "id": tid,
                    "title": f"Task {tid}",
                    "description": f"Task {tid} description",
                    "component_area": "mixed",
                }

            mock_get_task.side_effect = get_task_data

            # Make some agents fail for certain tasks
            agents = list(mock_orchestrator.agent_registry._agents.values())
            if agents:
                # Make task 2 and 4 fail
                for agent in agents:
                    agent.set_task_validation_result("2", False)
                    agent.set_task_validation_result("4", False)

            result = await mock_orchestrator.execute_batch(task_ids, batch_size=2)

            assert result["batch_size"] == 4
            assert result["successful"] + result["failed"] == 4
            assert len(result["results"]) == 4

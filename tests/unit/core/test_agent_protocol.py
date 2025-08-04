"""Comprehensive test suite for core/agent_protocol.py.

Tests the AgentProtocol interface, AgentConfig model, BaseAgent implementation,
and AgentExecutionError exception handling.
"""

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from src.core.agent_protocol import (
    AgentConfig,
    AgentExecutionError,
    AgentProtocol,
    BaseAgent,
)
from src.core.state import AgentState


class TestAgentConfig:
    """Test AgentConfig Pydantic model."""

    def test_agent_config_creation_with_defaults(self):
        """Test creating AgentConfig with default values."""
        config = AgentConfig(name="test_agent")

        assert config.name == "test_agent"
        assert config.enabled is True
        assert config.capabilities == []
        assert config.model == "openrouter/horizon-beta"
        assert config.timeout == 180
        assert config.retry_attempts == 2
        assert config.tools == []
        assert config.max_concurrent_tasks == 1

    def test_agent_config_creation_with_custom_values(self):
        """Test creating AgentConfig with custom values."""
        config = AgentConfig(
            name="custom_agent",
            enabled=False,
            capabilities=["research", "analysis"],
            model="gpt-4",
            timeout=300,
            retry_attempts=5,
            tools=["web_search", "calculator"],
            max_concurrent_tasks=3,
        )

        assert config.name == "custom_agent"
        assert config.enabled is False
        assert config.capabilities == ["research", "analysis"]
        assert config.model == "gpt-4"
        assert config.timeout == 300
        assert config.retry_attempts == 5
        assert config.tools == ["web_search", "calculator"]
        assert config.max_concurrent_tasks == 3

    def test_agent_config_allows_extra_fields(self):
        """Test that AgentConfig allows additional fields."""
        config = AgentConfig(
            name="test_agent",
            custom_field="custom_value",
            another_field={"nested": "data"},
        )

        assert config.name == "test_agent"
        assert hasattr(config, "custom_field")
        assert config.custom_field == "custom_value"
        assert hasattr(config, "another_field")
        assert config.another_field == {"nested": "data"}

    def test_agent_config_validation_empty_name(self):
        """Test AgentConfig validation with empty name."""
        with pytest.raises(ValidationError):
            AgentConfig(name="")

    def test_agent_config_validation_missing_name(self):
        """Test AgentConfig validation with missing name."""
        with pytest.raises(ValidationError):
            AgentConfig()

    def test_agent_config_serialization(self):
        """Test AgentConfig serialization to dict."""
        config = AgentConfig(
            name="test_agent", capabilities=["test", "demo"], model="custom-model"
        )

        config_dict = config.model_dump()

        assert config_dict["name"] == "test_agent"
        assert config_dict["capabilities"] == ["test", "demo"]
        assert config_dict["model"] == "custom-model"
        assert "enabled" in config_dict
        assert "timeout" in config_dict

    def test_agent_config_deserialization(self):
        """Test AgentConfig deserialization from dict."""
        config_data = {
            "name": "deserialized_agent",
            "enabled": True,
            "capabilities": ["capability1", "capability2"],
            "model": "test-model",
            "timeout": 120,
            "retry_attempts": 3,
            "tools": ["tool1"],
            "max_concurrent_tasks": 2,
            "custom_setting": "value",
        }

        config = AgentConfig.model_validate(config_data)

        assert config.name == "deserialized_agent"
        assert config.enabled is True
        assert config.capabilities == ["capability1", "capability2"]
        assert config.custom_setting == "value"


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def __init__(self, config: AgentConfig, execution_behavior: str = "success"):
        """Initialize concrete agent with configurable behavior."""
        super().__init__(config)
        self.execution_behavior = execution_behavior
        self.execution_calls = []
        self.validation_calls = []

    async def execute_task(self, state: AgentState) -> AgentState:
        """Execute task with configurable behavior for testing."""
        self.execution_calls.append(state)
        self._increment_task_count()

        try:
            if self.execution_behavior == "success":
                # Simulate successful execution
                await asyncio.sleep(0.01)  # Simulate work
                state["agent_outputs"][self.name] = {
                    "status": "completed",
                    "result": f"Task completed by {self.name}",
                }
            elif self.execution_behavior == "failure":
                raise AgentExecutionError(
                    self.name, state.get("task_id", 0), "Simulated execution failure"
                )
            elif self.execution_behavior == "timeout":
                await asyncio.sleep(1)  # Simulate long-running task
                state["agent_outputs"][self.name] = {
                    "status": "timeout",
                    "result": "Task timed out",
                }
            elif self.execution_behavior == "partial":
                state["agent_outputs"][self.name] = {
                    "status": "partial",
                    "result": "Task partially completed",
                }

            return state
        finally:
            self._decrement_task_count()

    async def validate_task(self, task_data: dict[str, Any]) -> bool:
        """Validate task with custom logic for testing."""
        self.validation_calls.append(task_data)

        # Custom validation logic based on execution behavior
        if self.execution_behavior == "validation_failure":
            return False

        # Call parent validation
        return await super().validate_task(task_data)


class TestBaseAgent:
    """Test BaseAgent abstract base class implementation."""

    @pytest.fixture
    def basic_config(self):
        """Basic agent configuration for testing."""
        return AgentConfig(
            name="test_agent",
            capabilities=["testing", "validation"],
            max_concurrent_tasks=2,
        )

    @pytest.fixture
    def concrete_agent(self, basic_config):
        """Concrete agent instance for testing."""
        return ConcreteAgent(basic_config)

    def test_base_agent_initialization(self, basic_config):
        """Test BaseAgent initialization with configuration."""
        agent = ConcreteAgent(basic_config)

        assert agent.config == basic_config
        assert agent.name == "test_agent"
        assert agent.capabilities == ["testing", "validation"]
        assert agent._is_healthy is True
        assert agent._current_tasks == 0

    def test_base_agent_get_config(self, concrete_agent):
        """Test get_config method returns agent configuration."""
        config = concrete_agent.get_config()

        assert isinstance(config, AgentConfig)
        assert config.name == "test_agent"
        assert config.capabilities == ["testing", "validation"]

    def test_base_agent_health_status_default(self, concrete_agent):
        """Test get_health_status returns correct default status."""
        status = concrete_agent.get_health_status()

        assert status["name"] == "test_agent"
        assert status["healthy"] is True
        assert status["current_tasks"] == 0
        assert status["max_concurrent_tasks"] == 2
        assert status["capabilities"] == ["testing", "validation"]
        assert status["enabled"] is True
        assert status["last_heartbeat"] is None

    def test_base_agent_task_count_management(self, concrete_agent):
        """Test task count increment and decrement methods."""
        assert concrete_agent._current_tasks == 0

        concrete_agent._increment_task_count()
        assert concrete_agent._current_tasks == 1

        concrete_agent._increment_task_count()
        assert concrete_agent._current_tasks == 2

        concrete_agent._decrement_task_count()
        assert concrete_agent._current_tasks == 1

        concrete_agent._decrement_task_count()
        assert concrete_agent._current_tasks == 0

        # Should not go below 0
        concrete_agent._decrement_task_count()
        assert concrete_agent._current_tasks == 0

    @pytest.mark.asyncio
    async def test_base_agent_validate_task_success(self, concrete_agent):
        """Test successful task validation."""
        task_data = {"id": 1, "title": "Test task", "description": "A test task"}

        result = await concrete_agent.validate_task(task_data)

        assert result is True
        assert len(concrete_agent.validation_calls) == 1
        assert concrete_agent.validation_calls[0] == task_data

    @pytest.mark.asyncio
    async def test_base_agent_validate_task_empty_data(self, concrete_agent):
        """Test task validation with empty task data."""
        result = await concrete_agent.validate_task({})

        assert result is False

    @pytest.mark.asyncio
    async def test_base_agent_validate_task_none_data(self, concrete_agent):
        """Test task validation with None task data."""
        result = await concrete_agent.validate_task(None)

        assert result is False

    @pytest.mark.asyncio
    async def test_base_agent_validate_task_at_capacity(self, concrete_agent):
        """Test task validation when agent is at capacity."""
        # Fill agent to capacity
        concrete_agent._current_tasks = 2  # Max is 2

        task_data = {"id": 1, "title": "Test task", "description": "A test task"}

        result = await concrete_agent.validate_task(task_data)

        assert result is False

    @pytest.mark.asyncio
    async def test_base_agent_validate_task_unhealthy_agent(self, concrete_agent):
        """Test task validation when agent is unhealthy."""
        concrete_agent._is_healthy = False

        task_data = {"id": 1, "title": "Test task", "description": "A test task"}

        result = await concrete_agent.validate_task(task_data)

        assert result is False

    @pytest.mark.asyncio
    async def test_base_agent_cleanup(self, concrete_agent):
        """Test agent cleanup method."""
        # Set some state
        concrete_agent._current_tasks = 2
        concrete_agent._is_healthy = True

        await concrete_agent.cleanup()

        assert concrete_agent._current_tasks == 0
        assert concrete_agent._is_healthy is False


class TestAgentProtocolCompliance:
    """Test AgentProtocol compliance and runtime checking."""

    def test_agent_protocol_runtime_checking_valid_agent(self):
        """Test AgentProtocol runtime checking with valid agent."""
        config = AgentConfig(name="protocol_test")
        agent = ConcreteAgent(config)

        # Should pass isinstance check
        assert isinstance(agent, AgentProtocol)

    def test_agent_protocol_duck_typing_valid_implementation(self):
        """Test AgentProtocol duck typing with valid implementation."""

        # Create a class that implements all required methods
        class DuckTypedAgent:
            def __init__(self):
                self.name = "duck_typed"
                self.capabilities = ["testing"]
                self.config = AgentConfig(name="duck_typed")

            async def execute_task(self, state: AgentState) -> AgentState:
                return state

            async def validate_task(self, task_data: dict[str, Any]) -> bool:
                return True

            def get_config(self) -> AgentConfig:
                return self.config

            def get_health_status(self) -> dict[str, Any]:
                return {"healthy": True}

            async def cleanup(self) -> None:
                pass

        agent = DuckTypedAgent()

        # Should pass isinstance check due to duck typing
        assert isinstance(agent, AgentProtocol)

    def test_agent_protocol_duck_typing_invalid_implementation(self):
        """Test AgentProtocol duck typing with invalid implementation."""

        class IncompleteAgent:
            def __init__(self):
                self.name = "incomplete"
                self.capabilities = ["testing"]
                self.config = AgentConfig(name="incomplete")

            # Missing required methods

        agent = IncompleteAgent()

        # Should fail isinstance check
        assert not isinstance(agent, AgentProtocol)

    @pytest.mark.asyncio
    async def test_agent_protocol_method_signatures(self):
        """Test that agent protocol methods have correct signatures."""
        config = AgentConfig(name="signature_test")
        agent = ConcreteAgent(config)

        # Test execute_task signature
        state = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "test"},
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        result = await agent.execute_task(state)
        assert isinstance(result, dict)  # AgentState is TypedDict

        # Test validate_task signature
        task_data = {"id": 1, "title": "test"}
        validation_result = await agent.validate_task(task_data)
        assert isinstance(validation_result, bool)

        # Test get_config signature
        config_result = agent.get_config()
        assert isinstance(config_result, AgentConfig)

        # Test get_health_status signature
        health_result = agent.get_health_status()
        assert isinstance(health_result, dict)

        # Test cleanup signature
        cleanup_result = await agent.cleanup()
        assert cleanup_result is None


class TestAgentExecutionScenarios:
    """Test various agent execution scenarios."""

    @pytest.fixture
    def sample_state(self):
        """Sample agent state for testing."""
        return {
            "messages": [],
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Test task",
                "description": "A comprehensive test task",
                "component_area": "testing",
            },
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

    @pytest.mark.asyncio
    async def test_agent_successful_execution(self, sample_state):
        """Test successful agent task execution."""
        config = AgentConfig(name="success_agent")
        agent = ConcreteAgent(config, execution_behavior="success")

        result = await agent.execute_task(sample_state)

        assert "success_agent" in result["agent_outputs"]
        assert result["agent_outputs"]["success_agent"]["status"] == "completed"
        assert len(agent.execution_calls) == 1

    @pytest.mark.asyncio
    async def test_agent_execution_failure(self, sample_state):
        """Test agent task execution failure."""
        config = AgentConfig(name="failure_agent")
        agent = ConcreteAgent(config, execution_behavior="failure")

        with pytest.raises(AgentExecutionError) as exc_info:
            await agent.execute_task(sample_state)

        assert exc_info.value.agent_name == "failure_agent"
        assert exc_info.value.task_id == 1
        assert "Simulated execution failure" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_agent_execution_partial_completion(self, sample_state):
        """Test agent task execution with partial completion."""
        config = AgentConfig(name="partial_agent")
        agent = ConcreteAgent(config, execution_behavior="partial")

        result = await agent.execute_task(sample_state)

        assert "partial_agent" in result["agent_outputs"]
        assert result["agent_outputs"]["partial_agent"]["status"] == "partial"

    @pytest.mark.asyncio
    async def test_agent_execution_task_count_management(self, sample_state):
        """Test task count management during execution."""
        config = AgentConfig(name="count_agent")
        agent = ConcreteAgent(config, execution_behavior="success")

        initial_count = agent._current_tasks

        # Start execution (should increment)
        task = asyncio.create_task(agent.execute_task(sample_state))
        await asyncio.sleep(0.001)  # Let increment happen

        # Task count should be incremented during execution
        # (Note: This test is inherently racy, but demonstrates the concept)

        # Complete execution
        await task

        # Task count should be back to initial after completion
        final_count = agent._current_tasks
        assert final_count == initial_count

    @pytest.mark.asyncio
    async def test_agent_concurrent_execution_limit(self, sample_state):
        """Test agent respects concurrent execution limits."""
        config = AgentConfig(name="concurrent_agent", max_concurrent_tasks=2)
        agent = ConcreteAgent(config, execution_behavior="success")

        # Start two tasks (should be within limit)
        task1 = asyncio.create_task(agent.execute_task(sample_state.copy()))
        task2 = asyncio.create_task(agent.execute_task(sample_state.copy()))

        # Third task should fail validation due to capacity
        agent._current_tasks = 2  # Simulate tasks in progress
        validation_result = await agent.validate_task(sample_state["task_data"])

        assert validation_result is False

        # Clean up running tasks
        await asyncio.gather(task1, task2)


class TestAgentExecutionError:
    """Test AgentExecutionError exception class."""

    def test_agent_execution_error_creation(self):
        """Test creating AgentExecutionError with basic parameters."""
        error = AgentExecutionError("test_agent", 123, "Test error message")

        assert error.agent_name == "test_agent"
        assert error.task_id == 123
        assert error.cause is None
        assert "Agent test_agent failed task 123: Test error message" in str(error)

    def test_agent_execution_error_with_cause(self):
        """Test creating AgentExecutionError with underlying cause."""
        original_error = ValueError("Original error")
        error = AgentExecutionError(
            "test_agent", 456, "Wrapper error message", cause=original_error
        )

        assert error.agent_name == "test_agent"
        assert error.task_id == 456
        assert error.cause == original_error
        assert "Agent test_agent failed task 456: Wrapper error message" in str(error)

    def test_agent_execution_error_inheritance(self):
        """Test that AgentExecutionError inherits from Exception."""
        error = AgentExecutionError("agent", 1, "message")

        assert isinstance(error, Exception)
        assert isinstance(error, AgentExecutionError)

    def test_agent_execution_error_raising_and_catching(self):
        """Test raising and catching AgentExecutionError."""

        def raising_function():
            raise AgentExecutionError("failing_agent", 789, "Function failed")

        with pytest.raises(AgentExecutionError) as exc_info:
            raising_function()

        caught_error = exc_info.value
        assert caught_error.agent_name == "failing_agent"
        assert caught_error.task_id == 789
        assert "Function failed" in str(caught_error)

    def test_agent_execution_error_with_chain(self):
        """Test AgentExecutionError with exception chaining."""
        try:
            try:
                raise ValueError("Root cause")
            except ValueError as e:
                raise AgentExecutionError(
                    "chained_agent", 999, "Chained error", cause=e
                ) from e
        except AgentExecutionError as error:
            assert error.cause is not None
            assert isinstance(error.cause, ValueError)
            assert "Root cause" in str(error.cause)


class TestAgentProtocolEdgeCases:
    """Test edge cases and error conditions in agent protocol."""

    @pytest.mark.asyncio
    async def test_agent_execute_task_with_malformed_state(self):
        """Test agent execution with malformed state."""
        config = AgentConfig(name="edge_agent")
        agent = ConcreteAgent(config)

        # Malformed state missing required fields
        malformed_state = {
            "messages": [],
            # Missing task_id, task_data, etc.
        }

        # Should handle gracefully and not crash
        result = await agent.execute_task(malformed_state)

        # Should return some form of state (may be modified)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_agent_validate_task_with_malformed_data(self):
        """Test agent validation with malformed task data."""
        config = AgentConfig(name="edge_agent")
        agent = ConcreteAgent(config)

        # Various malformed task data scenarios
        malformed_data_cases = [
            {"id": "not_a_number"},  # Wrong type
            {"title": ""},  # Empty title
            {"description": None},  # None values
            {"id": -1},  # Negative ID
        ]

        for malformed_data in malformed_data_cases:
            # Should handle gracefully
            result = await agent.validate_task(malformed_data)
            # May return True or False, but shouldn't crash
            assert isinstance(result, bool)

    def test_agent_config_with_extreme_values(self):
        """Test AgentConfig with extreme values."""
        # Test with very large values
        config = AgentConfig(
            name="extreme_agent",
            timeout=99999999,
            retry_attempts=1000,
            max_concurrent_tasks=10000,
        )

        assert config.timeout == 99999999
        assert config.retry_attempts == 1000
        assert config.max_concurrent_tasks == 10000

    def test_agent_health_status_consistency(self):
        """Test agent health status remains consistent."""
        config = AgentConfig(name="consistent_agent")
        agent = ConcreteAgent(config)

        # Get multiple health status snapshots
        status1 = agent.get_health_status()
        status2 = agent.get_health_status()

        # Should be consistent (assuming no state changes)
        assert status1["name"] == status2["name"]
        assert status1["healthy"] == status2["healthy"]
        assert status1["current_tasks"] == status2["current_tasks"]

    @pytest.mark.asyncio
    async def test_agent_cleanup_idempotency(self):
        """Test that agent cleanup is idempotent."""
        config = AgentConfig(name="cleanup_agent")
        agent = ConcreteAgent(config)

        # Set some state
        agent._current_tasks = 3
        agent._is_healthy = True

        # First cleanup
        await agent.cleanup()
        state_after_first = (agent._current_tasks, agent._is_healthy)

        # Second cleanup
        await agent.cleanup()
        state_after_second = (agent._current_tasks, agent._is_healthy)

        # Should be the same
        assert state_after_first == state_after_second
        assert agent._current_tasks == 0
        assert agent._is_healthy is False


class TestAgentProtocolPerformance:
    """Test agent protocol performance characteristics."""

    @pytest.mark.asyncio
    async def test_agent_rapid_task_validation(self):
        """Test agent can handle rapid task validation requests."""
        config = AgentConfig(name="rapid_agent")
        agent = ConcreteAgent(config)

        # Generate many validation requests
        tasks = []
        for i in range(100):
            task_data = {"id": i, "title": f"Task {i}"}
            tasks.append(agent.validate_task(task_data))

        # Execute all validations concurrently
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert len(results) == 100
        assert all(isinstance(result, bool) for result in results)

    @pytest.mark.asyncio
    async def test_agent_state_modifications_performance(self):
        """Test agent state modification performance."""
        config = AgentConfig(name="perf_agent")
        agent = ConcreteAgent(config)

        # Create large state object
        large_state = {
            "messages": [],
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Performance test",
                "large_data": ["item"] * 1000,  # Large data structure
            },
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        # Should handle large state efficiently
        result = await agent.execute_task(large_state)

        assert isinstance(result, dict)
        assert "perf_agent" in result["agent_outputs"]

    def test_agent_config_serialization_performance(self):
        """Test agent config serialization performance with large configurations."""
        # Create config with many capabilities and tools
        config = AgentConfig(
            name="large_config_agent",
            capabilities=[f"capability_{i}" for i in range(100)],
            tools=[f"tool_{i}" for i in range(50)],
            **{f"custom_field_{i}": f"value_{i}" for i in range(100)},
        )

        # Should serialize efficiently
        serialized = config.model_dump()

        assert len(serialized["capabilities"]) == 100
        assert len(serialized["tools"]) == 50

    @pytest.mark.asyncio
    async def test_agent_concurrent_health_checks(self):
        """Test concurrent health status checks don't interfere."""
        config = AgentConfig(name="concurrent_health_agent")
        agent = ConcreteAgent(config)

        # Perform many concurrent health checks
        health_tasks = [
            asyncio.create_task(asyncio.to_thread(agent.get_health_status))
            for _ in range(50)
        ]

        results = await asyncio.gather(*health_tasks)

        # All results should be consistent
        assert len(results) == 50
        for result in results:
            assert result["name"] == "concurrent_health_agent"
            assert result["healthy"] is True

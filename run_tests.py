#!/usr/bin/env python3
"""Simple test runner to bypass pytest configuration issues."""

import asyncio
import sys
import traceback
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_agent_config_tests():
    """Run AgentConfig tests manually."""
    print("🧪 Testing AgentConfig...")

    try:
        from pydantic import ValidationError

        from src.core.agent_protocol import AgentConfig

        # Test 1: Basic creation with defaults
        config = AgentConfig(name="test_agent")
        assert config.name == "test_agent"
        assert config.enabled is True
        assert config.capabilities == []
        assert config.model == "openrouter/horizon-beta"
        assert config.timeout == 180
        assert config.retry_attempts == 2
        assert config.tools == []
        assert config.max_concurrent_tasks == 1
        print("  ✓ test_agent_config_creation_with_defaults")

        # Test 2: Custom values
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
        print("  ✓ test_agent_config_creation_with_custom_values")

        # Test 3: Extra fields
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
        print("  ✓ test_agent_config_allows_extra_fields")

        # Test 4: Validation error
        try:
            AgentConfig(name="")
            raise AssertionError("Should have raised ValidationError")
        except ValidationError:
            print("  ✓ test_agent_config_validation_empty_name")

        # Test 5: Missing name
        try:
            AgentConfig()
            raise AssertionError("Should have raised ValidationError")
        except ValidationError:
            print("  ✓ test_agent_config_validation_missing_name")

        return True

    except Exception as e:
        print(f"  ✗ AgentConfig tests failed: {e}")
        traceback.print_exc()
        return False


def run_agent_registry_tests():
    """Run AgentRegistry tests manually."""
    print("🧪 Testing AgentRegistry...")

    try:
        from src.core.agent_protocol import AgentConfig, BaseAgent
        from src.core.agent_registry import AgentRegistry
        from src.core.state import AgentState

        # Create a concrete agent for testing
        class TestAgent(BaseAgent):
            async def execute_task(self, state: AgentState) -> AgentState:
                return state

        # Test 1: Registry initialization
        registry = AgentRegistry()
        assert len(registry._agents) == 0
        assert len(registry._capabilities) == 0
        assert len(registry._agent_modules) == 0
        print("  ✓ test_registry_initialization")

        # Test 2: Agent registration
        config = AgentConfig(
            name="test_agent", capabilities=["capability1", "capability2"]
        )
        agent = TestAgent(config)

        registry.register(agent)

        assert "test_agent" in registry._agents
        assert registry._agents["test_agent"] == agent
        assert "capability1" in registry._capabilities
        assert "capability2" in registry._capabilities
        assert "test_agent" in registry._capabilities["capability1"]
        assert "test_agent" in registry._capabilities["capability2"]
        print("  ✓ test_register_agent_success")

        # Test 3: Duplicate registration
        try:
            registry.register(agent)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "already registered" in str(e)
            print("  ✓ test_register_duplicate_agent_raises_error")

        # Test 4: Agent retrieval
        retrieved = registry.get_agent("test_agent")
        assert retrieved is not None
        assert retrieved.name == "test_agent"
        print("  ✓ test_get_agent_success")

        # Test 5: List agents
        agents = registry.list_agents()
        assert len(agents) == 1
        assert agents[0].name == "test_agent"
        print("  ✓ test_list_agents")

        # Test 6: List capabilities
        capabilities = registry.list_capabilities()
        assert len(capabilities) == 2
        assert "capability1" in capabilities
        assert "capability2" in capabilities
        print("  ✓ test_list_capabilities")

        return True

    except Exception as e:
        print(f"  ✗ AgentRegistry tests failed: {e}")
        traceback.print_exc()
        return False


def run_agent_state_tests():
    """Run AgentState tests manually."""
    print("🧪 Testing AgentState...")

    try:
        from typing import TYPE_CHECKING

        from langchain_core.messages import HumanMessage

        if TYPE_CHECKING:
            from src.core.state import AgentState

        # Test 1: Minimal state creation
        state: AgentState = {
            "messages": [],
            "task_id": None,
            "task_data": None,
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        assert state["messages"] == []
        assert state["task_id"] is None
        assert state["task_data"] is None
        assert state["agent_outputs"] == {}
        assert state["batch_id"] is None
        assert state["coordination_context"] == {}
        assert state["error_context"] is None
        assert state["next_agent"] is None
        print("  ✓ test_agent_state_creation_minimal")

        # Test 2: State with data
        messages = [HumanMessage(content="Test message")]
        task_data = {"id": 1, "title": "Test task"}
        agent_outputs = {"agent1": {"status": "completed"}}
        coordination_context = {"assigned_agent": "agent1"}
        error_context = {"error": "test error"}

        state: AgentState = {
            "messages": messages,
            "task_id": 1,
            "task_data": task_data,
            "agent_outputs": agent_outputs,
            "batch_id": "batch-123",
            "coordination_context": coordination_context,
            "error_context": error_context,
            "next_agent": "agent2",
        }

        assert state["messages"] == messages
        assert state["task_id"] == 1
        assert state["task_data"] == task_data
        assert state["agent_outputs"] == agent_outputs
        assert state["batch_id"] == "batch-123"
        assert state["coordination_context"] == coordination_context
        assert state["error_context"] == error_context
        assert state["next_agent"] == "agent2"
        print("  ✓ test_agent_state_creation_populated")

        # Test 3: State modification
        state["task_id"] = 42
        state["messages"].append(HumanMessage(content="New message"))
        state["agent_outputs"]["new_agent"] = {"result": "success"}
        state["coordination_context"]["phase"] = "execution"

        assert state["task_id"] == 42
        assert len(state["messages"]) == 2
        assert "new_agent" in state["agent_outputs"]
        assert state["coordination_context"]["phase"] == "execution"
        print("  ✓ test_agent_state_field_modification")

        return True

    except Exception as e:
        print(f"  ✗ AgentState tests failed: {e}")
        traceback.print_exc()
        return False


def run_orchestrator_basic_tests():
    """Run basic orchestrator tests without mocking external dependencies."""
    print("🧪 Testing Orchestrator basics...")

    try:
        from unittest.mock import Mock, patch

        from src.core.orchestrator import ModularOrchestrator

        # Test 1: Default configuration loading
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
            print("  ✓ test_orchestrator_default_config_initialization")

        # Test 2: Agent registry initialization
        with (
            patch("src.core.orchestrator.TaskManager"),
            patch("src.core.orchestrator.ChatOpenAI"),
            patch(
                "src.agents.research_agent.ResearchAgent.create_default"
            ) as mock_research,
            patch("src.agents.coding_agent.CodingAgent.create_default") as mock_coding,
            patch(
                "src.agents.testing_agent.TestingAgent.create_default"
            ) as mock_testing,
            patch(
                "src.agents.documentation_agent.DocumentationAgent.create_default"
            ) as mock_docs,
        ):
            # Create mock agents
            from src.core.agent_protocol import AgentConfig, BaseAgent
            from src.core.state import AgentState

            class MockAgent(BaseAgent):
                async def execute_task(self, state: AgentState) -> AgentState:
                    return state

            mock_research.return_value = MockAgent(
                AgentConfig(name="research", capabilities=["research"])
            )
            mock_coding.return_value = MockAgent(
                AgentConfig(name="coding", capabilities=["coding"])
            )
            mock_testing.return_value = MockAgent(
                AgentConfig(name="testing", capabilities=["testing"])
            )
            mock_docs.return_value = MockAgent(
                AgentConfig(name="documentation", capabilities=["documentation"])
            )

            orchestrator = ModularOrchestrator(
                openai_api_key="test-key", db_path=":memory:"
            )

            # Verify agents were registered
            agents = orchestrator.agent_registry.list_agents()
            assert len(agents) == 4
            agent_names = [agent.name for agent in agents]
            assert "research" in agent_names
            assert "coding" in agent_names
            assert "testing" in agent_names
            assert "documentation" in agent_names
            print("  ✓ test_orchestrator_agent_initialization")

        # Test 3: Health status retrieval
        with (
            patch("src.core.orchestrator.TaskManager"),
            patch("src.core.orchestrator.ChatOpenAI"),
        ):
            # Mock a simple orchestrator setup
            orchestrator = ModularOrchestrator.__new__(ModularOrchestrator)
            orchestrator.agent_registry = Mock()
            orchestrator.agent_registry.get_health_status.return_value = {
                "registry_healthy": True,
                "total_agents": 2,
                "healthy_agents": 2,
            }

            health_status = orchestrator.get_agent_health_status()

            assert health_status["registry_healthy"] is True
            assert health_status["total_agents"] == 2
            assert health_status["healthy_agents"] == 2
            print("  ✓ test_get_agent_health_status")

        return True

    except Exception as e:
        print(f"  ✗ Orchestrator basic tests failed: {e}")
        traceback.print_exc()
        return False


async def run_async_tests():
    """Run async tests manually."""
    print("🧪 Testing async functionality...")

    try:
        from src.core.agent_protocol import AgentConfig, BaseAgent
        from src.core.state import AgentState

        class AsyncTestAgent(BaseAgent):
            def __init__(self, config):
                super().__init__(config)
                self.execution_calls = []
                self.validation_calls = []

            async def execute_task(self, state: AgentState) -> AgentState:
                self.execution_calls.append(state)
                self._increment_task_count()

                try:
                    # Simulate work
                    await asyncio.sleep(0.01)

                    state["agent_outputs"][self.name] = {
                        "status": "completed",
                        "result": f"Task completed by {self.name}",
                    }
                    return state
                finally:
                    self._decrement_task_count()

            async def validate_task(self, task_data):
                self.validation_calls.append(task_data)
                return await super().validate_task(task_data)

        # Test async agent execution
        config = AgentConfig(name="async_agent")
        agent = AsyncTestAgent(config)

        state = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Test"},
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        result = await agent.execute_task(state)

        assert "async_agent" in result["agent_outputs"]
        assert result["agent_outputs"]["async_agent"]["status"] == "completed"
        assert len(agent.execution_calls) == 1
        print("  ✓ test_async_agent_execution")

        # Test validation
        validation_result = await agent.validate_task({"id": 1, "title": "Test"})
        assert validation_result is True
        assert len(agent.validation_calls) == 1
        print("  ✓ test_async_agent_validation")

        return True

    except Exception as e:
        print(f"  ✗ Async tests failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Running PyTestQA-Agent Test Suite")
    print("=" * 50)

    results = []

    # Run synchronous tests
    results.append(("AgentConfig", run_agent_config_tests()))
    results.append(("AgentRegistry", run_agent_registry_tests()))
    results.append(("AgentState", run_agent_state_tests()))
    results.append(("OrchestratorBasic", run_orchestrator_basic_tests()))

    # Run async tests
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async_result = loop.run_until_complete(run_async_tests())
    results.append(("Async functionality", async_result))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name}: {status}")

        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed + failed}, Passed: {passed}, Failed: {failed}")

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"💥 {failed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

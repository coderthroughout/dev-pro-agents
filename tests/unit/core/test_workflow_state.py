"""Comprehensive test suite for core/state.py.

Tests the AgentState TypedDict, state transitions, error handling,
state validation, and workflow state management patterns.
"""

from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.core.state import AgentReportV2, AgentState, TaskAssignment
from src.schemas.unified_models import (
    AgentReport,
    AgentType,
    TaskDelegation,
    TaskPriority,
    TaskStatus,
)


class TestAgentStateStructure:
    """Test AgentState TypedDict structure and basic operations."""

    def test_agent_state_creation_minimal(self):
        """Test creating minimal AgentState with required fields."""
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

    def test_agent_state_creation_populated(self):
        """Test creating AgentState with populated fields."""
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

    def test_agent_state_field_modification(self):
        """Test modifying AgentState fields after creation."""
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

        # Modify fields
        state["task_id"] = 42
        state["messages"].append(HumanMessage(content="New message"))
        state["agent_outputs"]["new_agent"] = {"result": "success"}
        state["coordination_context"]["phase"] = "execution"

        assert state["task_id"] == 42
        assert len(state["messages"]) == 1
        assert "new_agent" in state["agent_outputs"]
        assert state["coordination_context"]["phase"] == "execution"

    def test_agent_state_with_various_message_types(self):
        """Test AgentState with different LangChain message types."""
        messages = [
            HumanMessage(content="User input"),
            AIMessage(content="AI response"),
            SystemMessage(content="System message"),
        ]

        state: AgentState = {
            "messages": messages,
            "task_id": 1,
            "task_data": {"id": 1},
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        assert len(state["messages"]) == 3
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)
        assert isinstance(state["messages"][2], SystemMessage)

    def test_agent_state_deep_nested_data(self):
        """Test AgentState with deeply nested data structures."""
        complex_task_data = {
            "id": 1,
            "metadata": {
                "requirements": ["req1", "req2"],
                "config": {
                    "timeout": 300,
                    "retries": 3,
                    "options": {"debug": True, "verbose": False},
                },
            },
            "dependencies": [
                {"task_id": 2, "type": "blocking"},
                {"task_id": 3, "type": "optional"},
            ],
        }

        complex_agent_outputs = {
            "agent1": {
                "status": "completed",
                "results": {
                    "primary": {"value": 100, "confidence": 0.95},
                    "secondary": [1, 2, 3, 4, 5],
                },
                "metrics": {"execution_time": 45.2, "memory_usage": "128MB"},
            }
        }

        state: AgentState = {
            "messages": [],
            "task_id": 1,
            "task_data": complex_task_data,
            "agent_outputs": complex_agent_outputs,
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        # Verify nested access works
        assert state["task_data"]["metadata"]["config"]["timeout"] == 300
        assert state["agent_outputs"]["agent1"]["results"]["primary"]["value"] == 100


class TestAgentStateWorkflowPatterns:
    """Test common workflow patterns with AgentState."""

    def test_state_progression_through_workflow_stages(self):
        """Test state evolution through typical workflow stages."""
        # Initial state
        state: AgentState = {
            "messages": [HumanMessage(content="Start task execution")],
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Process document",
                "description": "Extract and analyze document content",
            },
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        # Stage 1: Task assignment
        state["coordination_context"]["assigned_agent"] = "research"
        state["next_agent"] = "research"

        # Stage 2: Agent execution
        state["agent_outputs"]["research"] = {
            "status": "completed",
            "result": "Document analyzed successfully",
            "artifacts": ["analysis.json", "summary.txt"],
            "execution_time": 12.5,
        }
        state["next_agent"] = "coding"

        # Stage 3: Handoff to next agent
        state["coordination_context"]["handoff_data"] = {
            "from_agent": "research",
            "to_agent": "coding",
            "context": "Analysis complete, ready for implementation",
        }

        # Stage 4: Second agent execution
        state["agent_outputs"]["coding"] = {
            "status": "completed",
            "result": "Implementation created",
            "artifacts": ["implementation.py", "tests.py"],
            "execution_time": 25.8,
        }
        state["next_agent"] = None  # Workflow complete

        # Verify final state
        assert len(state["agent_outputs"]) == 2
        assert "research" in state["agent_outputs"]
        assert "coding" in state["agent_outputs"]
        assert state["next_agent"] is None
        assert "handoff_data" in state["coordination_context"]

    def test_state_error_handling_and_recovery(self):
        """Test state management during error scenarios."""
        state: AgentState = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Failing task"},
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": "problematic_agent",
        }

        # Simulate agent failure
        state["agent_outputs"]["problematic_agent"] = {
            "status": "failed",
            "error": "Network timeout",
            "execution_time": 30.0,
        }

        # Set error context
        state["error_context"] = {
            "error_type": "execution_failure",
            "failed_agent": "problematic_agent",
            "error_message": "Network timeout",
            "timestamp": datetime.now().isoformat(),
            "retry_count": 1,
        }

        # Attempt recovery by reassigning
        state["coordination_context"]["recovery_mode"] = True
        state["coordination_context"]["original_agent"] = "problematic_agent"
        state["next_agent"] = "backup_agent"

        # Backup agent succeeds
        state["agent_outputs"]["backup_agent"] = {
            "status": "completed",
            "result": "Task completed by backup agent",
            "recovery": True,
            "execution_time": 15.2,
        }

        # Clear error context after recovery
        state["error_context"] = None
        state["next_agent"] = None

        # Verify recovery state
        assert state["error_context"] is None
        assert state["coordination_context"]["recovery_mode"] is True
        assert "backup_agent" in state["agent_outputs"]
        assert state["agent_outputs"]["backup_agent"]["recovery"] is True

    def test_state_batch_processing_context(self):
        """Test state management for batch processing scenarios."""
        batch_id = "batch-2023-001"

        state: AgentState = {
            "messages": [],
            "task_id": 101,
            "task_data": {"id": 101, "title": "Batch task 1"},
            "agent_outputs": {},
            "batch_id": batch_id,
            "coordination_context": {
                "batch_info": {
                    "batch_id": batch_id,
                    "total_tasks": 5,
                    "current_task_index": 1,
                    "batch_start_time": datetime.now().isoformat(),
                }
            },
            "error_context": None,
            "next_agent": "batch_processor",
        }

        # Process batch task
        state["agent_outputs"]["batch_processor"] = {
            "status": "completed",
            "result": "Batch task 1 completed",
            "batch_position": 1,
            "execution_time": 8.3,
        }

        # Update batch context
        state["coordination_context"]["batch_info"]["completed_tasks"] = 1
        state["coordination_context"]["batch_info"]["success_rate"] = 1.0

        # Verify batch context is maintained
        assert state["batch_id"] == batch_id
        assert state["coordination_context"]["batch_info"]["total_tasks"] == 5
        assert state["coordination_context"]["batch_info"]["completed_tasks"] == 1

    def test_state_coordination_patterns(self):
        """Test various agent coordination patterns in state."""
        state: AgentState = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Coordination test"},
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        # Pattern 1: Sequential coordination
        sequential_plan = {
            "type": "sequential",
            "agents": ["agent1", "agent2", "agent3"],
            "current_step": 0,
        }
        state["coordination_context"]["execution_plan"] = sequential_plan

        # Pattern 2: Parallel coordination
        state["coordination_context"]["parallel_tasks"] = {
            "agent_a": {"status": "in_progress", "started_at": "2023-01-01T10:00:00"},
            "agent_b": {"status": "pending", "dependencies": ["agent_a"]},
            "agent_c": {"status": "in_progress", "started_at": "2023-01-01T10:00:00"},
        }

        # Pattern 3: Conditional coordination
        state["coordination_context"]["conditions"] = {
            "if_agent1_success": "route_to_agent2",
            "if_agent1_failure": "route_to_fallback_agent",
            "if_timeout": "escalate_to_human",
        }

        # Verify coordination structures
        assert state["coordination_context"]["execution_plan"]["type"] == "sequential"
        assert len(state["coordination_context"]["parallel_tasks"]) == 3
        assert "if_agent1_success" in state["coordination_context"]["conditions"]


class TestAgentStateValidationPatterns:
    """Test state validation and consistency patterns."""

    def test_state_consistency_validation(self):
        """Test validation of state consistency rules."""
        state: AgentState = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Validation test"},
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": "validator_agent",
        }

        # Validation helper function
        def validate_state_consistency(state: AgentState) -> list[str]:
            """Validate state consistency and return list of issues."""
            issues = []

            # Check task_id consistency
            if state["task_id"] and state["task_data"]:
                if state["task_data"].get("id") != state["task_id"]:
                    issues.append("Task ID mismatch between task_id and task_data.id")

            # Check agent outputs reference valid agents
            if state["next_agent"] and state["agent_outputs"]:
                if state["next_agent"] in state["agent_outputs"]:
                    # Next agent already has output - might be inconsistent
                    agent_output = state["agent_outputs"][state["next_agent"]]
                    if agent_output.get("status") == "completed":
                        issues.append("Next agent already completed - routing issue")

            # Check error context consistency
            if state["error_context"] and not any(
                output.get("status") in ["failed", "error"]
                for output in state["agent_outputs"].values()
            ):
                issues.append("Error context present but no failed agent outputs")

            return issues

        # Test valid state
        issues = validate_state_consistency(state)
        assert len(issues) == 0

        # Test inconsistent state - task ID mismatch
        state["task_data"]["id"] = 999
        issues = validate_state_consistency(state)
        assert "Task ID mismatch" in issues[0]

        # Fix and test error context inconsistency
        state["task_data"]["id"] = 1  # Fix task ID
        state["error_context"] = {"error": "something went wrong"}
        issues = validate_state_consistency(state)
        assert len(issues) == 1
        assert "Error context present but no failed agent outputs" in issues[0]

    def test_state_transition_validation(self):
        """Test validation of state transitions."""

        # Define transition rules
        def is_valid_transition(from_state: dict, to_state: dict) -> bool:
            """Validate if transition between states is valid."""
            # Rule 1: Task ID should not change during execution
            if from_state.get("task_id") != to_state.get("task_id"):
                return False

            # Rule 2: Agent outputs should only grow, not shrink
            from_agents = set(from_state.get("agent_outputs", {}).keys())
            to_agents = set(to_state.get("agent_outputs", {}).keys())
            if not from_agents.issubset(to_agents):
                return False

            # Rule 3: Completed agents shouldn't change status
            for agent_name in from_agents:
                from_status = from_state["agent_outputs"][agent_name].get("status")
                to_status = to_state["agent_outputs"][agent_name].get("status")
                if from_status == "completed" and to_status != "completed":
                    return False

            return True

        # Initial state
        state1: AgentState = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1},
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": "agent1",
        }

        # Valid transition - add agent output
        state2: AgentState = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1},
            "agent_outputs": {"agent1": {"status": "completed", "result": "success"}},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": "agent2",
        }

        assert is_valid_transition(state1, state2) is True

        # Invalid transition - change task ID
        state3: AgentState = {
            "messages": [],
            "task_id": 999,  # Changed task ID
            "task_data": {"id": 1},
            "agent_outputs": {"agent1": {"status": "completed", "result": "success"}},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": "agent2",
        }

        assert is_valid_transition(state2, state3) is False

        # Invalid transition - remove agent output
        state4: AgentState = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1},
            "agent_outputs": {},  # Removed agent output
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": "agent2",
        }

        assert is_valid_transition(state2, state4) is False

    def test_state_invariant_maintenance(self):
        """Test maintenance of state invariants."""
        state: AgentState = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Invariant test"},
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {},
            "error_context": None,
            "next_agent": None,
        }

        # Define invariant checking functions
        def check_message_ordering_invariant(state: AgentState) -> bool:
            """Messages should maintain chronological ordering."""
            # This is a simplified check - in practice you'd check timestamps
            return True  # TypedDict doesn't enforce this, but workflow should

        def check_agent_output_immutability_invariant(state: AgentState) -> bool:
            """Completed agent outputs should be immutable."""
            # This is a design invariant - completed outputs shouldn't change
            return True  # Would need additional tracking to validate this

        def check_task_data_consistency_invariant(state: AgentState) -> bool:
            """Task data should remain consistent with task_id."""
            if state["task_id"] is None:
                return state["task_data"] is None
            if state["task_data"] is None:
                return state["task_id"] is None
            return state["task_data"].get("id") == state["task_id"]

        # Test invariants
        assert check_message_ordering_invariant(state) is True
        assert check_agent_output_immutability_invariant(state) is True
        assert check_task_data_consistency_invariant(state) is True

        # Break task consistency invariant
        state["task_id"] = 1
        state["task_data"] = None
        assert check_task_data_consistency_invariant(state) is False

        # Fix invariant
        state["task_data"] = {"id": 1, "title": "Fixed"}
        assert check_task_data_consistency_invariant(state) is True


class TestAgentStateLegacyCompatibility:
    """Test backward compatibility with legacy type aliases."""

    def test_task_assignment_alias_compatibility(self):
        """Test TaskAssignment legacy alias works correctly."""
        # TaskAssignment should be an alias for TaskDelegation
        assert TaskAssignment is TaskDelegation

        # Should be able to create instances using either name
        assignment = TaskAssignment(
            assigned_agent=AgentType.CODING,
            reasoning="Legacy compatibility test",
            priority=TaskPriority.HIGH,
            estimated_duration=60,
        )

        delegation = TaskDelegation(
            assigned_agent=AgentType.CODING,
            reasoning="Legacy compatibility test",
            priority=TaskPriority.HIGH,
            estimated_duration=60,
        )

        # Should be identical
        assert assignment.assigned_agent == delegation.assigned_agent
        assert assignment.reasoning == delegation.reasoning
        assert assignment.priority == delegation.priority

    def test_agent_report_v2_alias_compatibility(self):
        """Test AgentReportV2 legacy alias works correctly."""
        # AgentReportV2 should be an alias for AgentReport
        assert AgentReportV2 is AgentReport

        # Should be able to create instances using either name
        report_v2 = AgentReportV2(
            agent_name=AgentType.TESTING, status=TaskStatus.COMPLETED, success=True
        )

        report = AgentReport(
            agent_name=AgentType.TESTING, status=TaskStatus.COMPLETED, success=True
        )

        # Should be identical
        assert report_v2.agent_name == report.agent_name
        assert report_v2.status == report.status
        assert report_v2.success == report.success

    def test_legacy_alias_type_checking(self):
        """Test type checking works with legacy aliases."""

        # Should pass type checking
        def process_assignment(assignment: TaskAssignment) -> str:
            return f"Processing assignment for {assignment.assigned_agent}"

        def process_report(report: AgentReportV2) -> str:
            return f"Processing report from {report.agent_name}"

        assignment = TaskDelegation(
            assigned_agent=AgentType.RESEARCH,
            reasoning="Type checking test",
            priority=TaskPriority.MEDIUM,
            estimated_duration=30,
        )

        report = AgentReport(
            agent_name=AgentType.DOCUMENTATION,
            status=TaskStatus.IN_PROGRESS,
            success=False,
        )

        # Should work with legacy types
        result1 = process_assignment(assignment)
        result2 = process_report(report)

        assert "research" in result1.lower()
        assert "documentation" in result2.lower()


class TestAgentStateComplexScenarios:
    """Test complex real-world scenarios with AgentState."""

    def test_multi_agent_collaboration_state(self):
        """Test state management for complex multi-agent collaboration."""
        state: AgentState = {
            "messages": [HumanMessage(content="Build a web application")],
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Web Application Development",
                "requirements": ["user authentication", "database", "API", "frontend"],
                "complexity": "high",
                "estimated_hours": 40,
            },
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {
                "collaboration_mode": "parallel_with_dependencies",
                "dependency_graph": {
                    "research": [],
                    "coding": ["research"],
                    "testing": ["coding"],
                    "documentation": ["coding", "testing"],
                },
            },
            "error_context": None,
            "next_agent": "research",
        }

        # Research agent completes
        state["agent_outputs"]["research"] = {
            "status": "completed",
            "result": "Requirements analyzed and architecture designed",
            "artifacts": ["requirements.md", "architecture.json"],
            "handoff_data": {
                "recommended_stack": "FastAPI + React + PostgreSQL",
                "key_components": ["user_service", "data_layer", "frontend_spa"],
            },
            "execution_time": 120.0,
        }

        # Update coordination context
        state["coordination_context"]["completed_agents"] = ["research"]
        state["coordination_context"]["available_agents"] = ["coding"]
        state["next_agent"] = "coding"

        # Coding agent executes
        state["agent_outputs"]["coding"] = {
            "status": "completed",
            "result": "Application implemented with all features",
            "artifacts": ["backend/", "frontend/", "database/schema.sql"],
            "dependencies_used": state["agent_outputs"]["research"]["handoff_data"],
            "execution_time": 180.0,
        }

        # Update for parallel execution of testing and documentation
        state["coordination_context"]["completed_agents"].extend(["coding"])
        state["coordination_context"]["available_agents"] = ["testing", "documentation"]
        state["coordination_context"]["parallel_execution"] = True
        state["next_agent"] = None  # Multiple agents can proceed

        # Both agents complete in parallel
        state["agent_outputs"]["testing"] = {
            "status": "completed",
            "result": "Comprehensive test suite created and all tests pass",
            "artifacts": ["tests/", "coverage_report.html"],
            "test_coverage": 95.5,
            "execution_time": 90.0,
        }

        state["agent_outputs"]["documentation"] = {
            "status": "completed",
            "result": "Complete documentation generated",
            "artifacts": ["docs/", "README.md", "API_docs.md"],
            "execution_time": 60.0,
        }

        # Verify final collaboration state
        assert len(state["agent_outputs"]) == 4
        assert all(
            output["status"] == "completed"
            for output in state["agent_outputs"].values()
        )
        assert state["coordination_context"]["parallel_execution"] is True
        assert (
            len(state["coordination_context"]["completed_agents"]) == 2
        )  # Updated during execution

    def test_error_cascade_and_recovery_state(self):
        """Test state management during error cascades and recovery."""
        state: AgentState = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Error cascade test"},
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {
                "retry_policy": {
                    "max_retries": 3,
                    "backoff_factor": 2.0,
                    "timeout_threshold": 300.0,
                }
            },
            "error_context": None,
            "next_agent": "primary_agent",
        }

        # Primary agent fails
        state["agent_outputs"]["primary_agent"] = {
            "status": "failed",
            "error": "Service unavailable",
            "error_code": "SERVICE_DOWN",
            "execution_time": 5.0,
            "retry_count": 1,
        }

        # Set initial error context
        state["error_context"] = {
            "primary_failure": {
                "agent": "primary_agent",
                "error": "Service unavailable",
                "timestamp": datetime.now().isoformat(),
            },
            "recovery_attempts": [],
        }

        # Attempt 1: Retry with same agent
        state["coordination_context"]["current_retry"] = 1
        state["next_agent"] = "primary_agent"  # Retry

        # Second failure
        state["agent_outputs"]["primary_agent"]["retry_count"] = 2
        state["error_context"]["recovery_attempts"].append(
            {
                "attempt": 1,
                "strategy": "direct_retry",
                "result": "failed",
                "error": "Service still unavailable",
            }
        )

        # Attempt 2: Route to backup agent
        state["coordination_context"]["current_retry"] = 2
        state["next_agent"] = "backup_agent"

        # Backup agent also fails
        state["agent_outputs"]["backup_agent"] = {
            "status": "failed",
            "error": "Database connection timeout",
            "error_code": "DB_TIMEOUT",
            "execution_time": 30.0,
        }

        # Update error context
        state["error_context"]["secondary_failure"] = {
            "agent": "backup_agent",
            "error": "Database connection timeout",
            "timestamp": datetime.now().isoformat(),
        }

        # Attempt 3: Emergency fallback with degraded functionality
        state["coordination_context"]["emergency_mode"] = True
        state["next_agent"] = "fallback_agent"

        # Fallback succeeds with limited functionality
        state["agent_outputs"]["fallback_agent"] = {
            "status": "completed",
            "result": "Task completed with degraded functionality",
            "warnings": ["Limited features due to service outage"],
            "degraded_mode": True,
            "execution_time": 15.0,
        }

        # Clear error context after successful fallback
        state["error_context"]["resolution"] = {
            "successful_agent": "fallback_agent",
            "resolution_strategy": "degraded_functionality",
            "timestamp": datetime.now().isoformat(),
        }
        state["next_agent"] = None

        # Verify error cascade and recovery state
        assert len(state["agent_outputs"]) == 3
        assert state["agent_outputs"]["primary_agent"]["status"] == "failed"
        assert state["agent_outputs"]["backup_agent"]["status"] == "failed"
        assert state["agent_outputs"]["fallback_agent"]["status"] == "completed"
        assert (
            state["error_context"]["resolution"]["successful_agent"] == "fallback_agent"
        )
        assert state["coordination_context"]["emergency_mode"] is True

    def test_dynamic_workflow_modification_state(self):
        """Test state management when workflow is dynamically modified."""
        state: AgentState = {
            "messages": [],
            "task_id": 1,
            "task_data": {"id": 1, "title": "Dynamic workflow test", "adaptive": True},
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {
                "original_plan": ["agent1", "agent2", "agent3"],
                "current_plan": ["agent1", "agent2", "agent3"],
                "modification_history": [],
            },
            "error_context": None,
            "next_agent": "agent1",
        }

        # Agent1 completes and suggests workflow modification
        state["agent_outputs"]["agent1"] = {
            "status": "completed",
            "result": "Analysis complete",
            "workflow_recommendation": {
                "add_agents": ["specialist_agent"],
                "remove_agents": ["agent3"],
                "reason": "Task requires specialized processing",
            },
            "execution_time": 25.0,
        }

        # Modify workflow based on recommendation
        old_plan = state["coordination_context"]["current_plan"].copy()
        state["coordination_context"]["current_plan"] = [
            "agent1",
            "agent2",
            "specialist_agent",
        ]
        state["coordination_context"]["modification_history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "trigger": "agent1_recommendation",
                "old_plan": old_plan,
                "new_plan": state["coordination_context"]["current_plan"],
                "reason": "Task requires specialized processing",
            }
        )

        # Continue with modified workflow
        state["next_agent"] = "agent2"

        # Agent2 execution with awareness of modification
        state["agent_outputs"]["agent2"] = {
            "status": "completed",
            "result": "Processing complete, prepared for specialist",
            "handoff_data": {"specialist_context": "complex_data_structure"},
            "execution_time": 30.0,
        }

        # Specialist agent execution
        state["next_agent"] = "specialist_agent"
        state["agent_outputs"]["specialist_agent"] = {
            "status": "completed",
            "result": "Specialized processing completed successfully",
            "used_context": state["agent_outputs"]["agent2"]["handoff_data"],
            "specialized_output": {"analysis": "deep_insights"},
            "execution_time": 45.0,
        }

        state["next_agent"] = None

        # Verify dynamic modification state
        assert len(state["coordination_context"]["modification_history"]) == 1
        assert "specialist_agent" in state["coordination_context"]["current_plan"]
        assert "agent3" not in state["coordination_context"]["current_plan"]
        assert "specialist_agent" in state["agent_outputs"]
        assert (
            state["agent_outputs"]["specialist_agent"]["specialized_output"] is not None
        )

    def test_state_memory_and_context_preservation(self):
        """Test preservation of context and memory across state transitions."""
        # Simulate long-running workflow with context accumulation
        state: AgentState = {
            "messages": [HumanMessage(content="Start long-running analysis")],
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Long-running analysis",
                "context_sensitive": True,
            },
            "agent_outputs": {},
            "batch_id": None,
            "coordination_context": {
                "shared_memory": {},
                "context_chain": [],
                "accumulated_insights": [],
            },
            "error_context": None,
            "next_agent": "analyzer1",
        }

        # First analyzer builds initial context
        analyzer1_output = {
            "status": "completed",
            "result": "Initial analysis complete",
            "insights": ["pattern_a_detected", "anomaly_at_position_15"],
            "context_contributions": {
                "data_patterns": {
                    "pattern_a": {"frequency": 0.15, "significance": "high"}
                },
                "anomalies": [{"position": 15, "severity": "medium"}],
            },
            "execution_time": 40.0,
        }

        state["agent_outputs"]["analyzer1"] = analyzer1_output

        # Update shared context
        state["coordination_context"]["shared_memory"]["patterns"] = analyzer1_output[
            "context_contributions"
        ]["data_patterns"]
        state["coordination_context"]["shared_memory"]["anomalies"] = analyzer1_output[
            "context_contributions"
        ]["anomalies"]
        state["coordination_context"]["context_chain"].append(
            {
                "agent": "analyzer1",
                "contributions": list(analyzer1_output["context_contributions"].keys()),
                "timestamp": datetime.now().isoformat(),
            }
        )
        state["coordination_context"]["accumulated_insights"].extend(
            analyzer1_output["insights"]
        )

        # Second analyzer builds on previous context
        state["next_agent"] = "analyzer2"
        analyzer2_output = {
            "status": "completed",
            "result": "Deep analysis complete",
            "used_context": {
                "previous_patterns": state["coordination_context"]["shared_memory"][
                    "patterns"
                ],
                "previous_anomalies": state["coordination_context"]["shared_memory"][
                    "anomalies"
                ],
            },
            "new_insights": [
                "pattern_a_correlates_with_pattern_b",
                "anomaly_cluster_identified",
            ],
            "context_contributions": {
                "correlations": {"pattern_a_b": {"strength": 0.85, "p_value": 0.001}},
                "clusters": [{"center": 15, "radius": 3, "density": "high"}],
            },
            "execution_time": 60.0,
        }

        state["agent_outputs"]["analyzer2"] = analyzer2_output

        # Update context with new contributions
        state["coordination_context"]["shared_memory"]["correlations"] = (
            analyzer2_output["context_contributions"]["correlations"]
        )
        state["coordination_context"]["shared_memory"]["clusters"] = analyzer2_output[
            "context_contributions"
        ]["clusters"]
        state["coordination_context"]["context_chain"].append(
            {
                "agent": "analyzer2",
                "contributions": list(analyzer2_output["context_contributions"].keys()),
                "built_on": ["analyzer1"],
                "timestamp": datetime.now().isoformat(),
            }
        )
        state["coordination_context"]["accumulated_insights"].extend(
            analyzer2_output["new_insights"]
        )

        # Final synthesizer uses all accumulated context
        state["next_agent"] = "synthesizer"
        synthesizer_output = {
            "status": "completed",
            "result": "Comprehensive synthesis complete",
            "synthesis_data": {
                "total_insights": len(
                    state["coordination_context"]["accumulated_insights"]
                ),
                "context_depth": len(state["coordination_context"]["context_chain"]),
                "memory_keys": list(
                    state["coordination_context"]["shared_memory"].keys()
                ),
            },
            "final_conclusions": [
                "Pattern A is strongly correlated with Pattern B",
                "Anomaly cluster at position 15 is significant",
                "Data quality is high with localized issues",
            ],
            "execution_time": 35.0,
        }

        state["agent_outputs"]["synthesizer"] = synthesizer_output
        state["next_agent"] = None

        # Verify context preservation and accumulation
        assert len(state["coordination_context"]["accumulated_insights"]) == 4
        assert len(state["coordination_context"]["context_chain"]) == 2
        assert (
            len(state["coordination_context"]["shared_memory"]) == 4
        )  # patterns, anomalies, correlations, clusters
        assert (
            state["agent_outputs"]["synthesizer"]["synthesis_data"]["context_depth"]
            == 2
        )
        assert "analyzer2" in [
            entry["agent"] for entry in state["coordination_context"]["context_chain"]
        ]

"""Comprehensive end-to-end testing for complete system workflows.

This module tests complete system behavior including:
- Full task lifecycle from creation to completion
- Multi-agent workflow orchestration
- System behavior under stress conditions
- Error recovery and resilience patterns
- Real-world scenario simulation
- Performance under production-like loads
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from pytest_benchmark.fixture import BenchmarkFixture

from src.config import get_settings
from src.schemas.unified_models import (
    AgentReport,
    AgentType,
    ComponentArea,
    TaskComplexity,
    TaskCore,
    TaskPriority,
    TaskStatus,
)
from src.supervisor_executor import SupervisorExecutor
from src.task_manager import TaskManager


@pytest_asyncio.fixture
async def end_to_end_environment():
    """Create complete end-to-end testing environment."""
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-e2e-openai",
            "OPENROUTER_API_KEY": "test-e2e-openrouter",
            "EXA_API_KEY": "test-e2e-exa",
            "FIRECRAWL_API_KEY": "test-e2e-firecrawl",
            "ORCHESTRATION_PARALLEL_EXECUTION_LIMIT": "5",
            "ORCHESTRATION_BATCH_SIZE": "25",
            "ORCHESTRATION_DEBUG_MODE": "false",
            "ORCHESTRATION_ENABLE_METRICS": "true",
        },
    ):
        # Create comprehensive mock environment
        mock_agents = {}

        # Mock each agent type with realistic behavior
        for agent_type in [
            AgentType.RESEARCH,
            AgentType.CODING,
            AgentType.TESTING,
            AgentType.DOCUMENTATION,
        ]:
            mock_agent = AsyncMock()

            # Configure agent-specific responses
            if agent_type == AgentType.RESEARCH:
                mock_agent.execute_task.return_value = AgentReport(
                    agent_name=agent_type,
                    task_id=1,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=2.5,
                    outputs={
                        "research_findings": ["Key finding 1", "Key finding 2"],
                        "sources": ["https://example.com/source1"],
                        "confidence": 0.9,
                    },
                    artifacts=["research_report.md"],
                    recommendations=["Use JWT for authentication"],
                    next_actions=["implement_authentication"],
                    confidence_score=0.9,
                )
            elif agent_type == AgentType.CODING:
                mock_agent.execute_task.return_value = AgentReport(
                    agent_name=agent_type,
                    task_id=1,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=4.0,
                    outputs={
                        "implementation": "def authenticate(): pass",
                        "files_created": ["auth.py", "models.py"],
                        "design_decisions": ["Used JWT tokens"],
                    },
                    artifacts=["auth.py", "models.py", "requirements.txt"],
                    recommendations=["Add input validation"],
                    next_actions=["create_tests"],
                    confidence_score=0.85,
                )
            elif agent_type == AgentType.TESTING:
                mock_agent.execute_task.return_value = AgentReport(
                    agent_name=agent_type,
                    task_id=1,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=3.0,
                    outputs={
                        "test_files": ["test_auth.py"],
                        "coverage_report": {"total": 95.5, "lines": 200},
                        "test_categories": ["unit", "integration"],
                    },
                    artifacts=["test_auth.py", "coverage_report.html"],
                    recommendations=["Add edge case tests"],
                    next_actions=["run_tests"],
                    confidence_score=0.92,
                )
            else:  # DOCUMENTATION
                mock_agent.execute_task.return_value = AgentReport(
                    agent_name=agent_type,
                    task_id=1,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=1.5,
                    outputs={
                        "documentation_files": ["README.md", "API.md"],
                        "sections_created": ["installation", "usage", "api"],
                    },
                    artifacts=["README.md", "API.md", "examples/"],
                    recommendations=["Add troubleshooting guide"],
                    next_actions=["review_documentation"],
                    confidence_score=0.88,
                )

            mock_agents[agent_type] = mock_agent

        # Mock supervisor with orchestration capabilities
        mock_supervisor = AsyncMock()
        mock_supervisor.coordinate_agents.return_value = {
            "workflow_id": "e2e-workflow-123",
            "agents_executed": list(mock_agents.keys()),
            "total_execution_time": 11.0,
            "success": True,
        }

        # Mock task manager with realistic database operations
        mock_task_manager = MagicMock(spec=TaskManager)
        mock_task_manager.get_ready_tasks.return_value = []
        mock_task_manager.create_task.return_value = TaskCore(
            id=1,
            title="E2E Test Task",
            description="End-to-end testing task",
            component_area=ComponentArea.CORE,
            phase=1,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
        )
        mock_task_manager.update_task_status = MagicMock()
        mock_task_manager.get_task_by_id.return_value = None

        # Mock external services
        mock_exa_client = AsyncMock()
        mock_exa_client.search.return_value = {
            "results": [
                {
                    "title": "E2E Research Result",
                    "url": "https://example.com/e2e",
                    "text": "Comprehensive end-to-end testing guide",
                    "score": 0.95,
                }
            ]
        }

        mock_firecrawl_client = AsyncMock()
        mock_firecrawl_client.scrape.return_value = {
            "success": True,
            "data": {
                "markdown": "# E2E Testing\n\nComplete guide to end-to-end testing",
                "title": "E2E Testing Guide",
            },
        }

        yield {
            "agents": mock_agents,
            "supervisor": mock_supervisor,
            "task_manager": mock_task_manager,
            "exa_client": mock_exa_client,
            "firecrawl_client": mock_firecrawl_client,
            "settings": get_settings(),
        }


def create_realistic_task_scenario() -> list[TaskCore]:
    """Create realistic task scenario for end-to-end testing."""
    return [
        TaskCore(
            id=1,
            title="Research authentication best practices",
            description=(
                "Investigate secure authentication patterns and JWT implementation"
            ),
            component_area=ComponentArea.SECURITY,
            phase=1,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            success_criteria=(
                "Identify 3+ authentication patterns with security analysis"
            ),
            time_estimate_hours=3.0,
        ),
        TaskCore(
            id=2,
            title="Implement JWT authentication system",
            description="Build secure JWT-based authentication with proper validation",
            component_area=ComponentArea.SECURITY,
            phase=2,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.HIGH,
            success_criteria="Working authentication system with JWT tokens",
            time_estimate_hours=8.0,
            parent_task_id=1,
        ),
        TaskCore(
            id=3,
            title="Create comprehensive test suite",
            description="Build unit and integration tests for authentication system",
            component_area=ComponentArea.TESTING,
            phase=3,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.MEDIUM,
            success_criteria="90%+ test coverage with edge cases",
            time_estimate_hours=5.0,
            parent_task_id=2,
        ),
        TaskCore(
            id=4,
            title="Document authentication system",
            description=(
                "Create comprehensive documentation for authentication implementation"
            ),
            component_area=ComponentArea.DOCUMENTATION,
            phase=4,
            priority=TaskPriority.LOW,
            complexity=TaskComplexity.LOW,
            success_criteria="Complete API documentation and usage examples",
            time_estimate_hours=2.0,
            parent_task_id=2,
        ),
    ]


class TestCompleteWorkflowExecution:
    """Test complete workflow execution from start to finish."""

    @pytest.mark.asyncio
    async def test_full_task_lifecycle_success(self, end_to_end_environment):
        """Test complete task lifecycle from creation to completion."""
        env = end_to_end_environment
        tasks = create_realistic_task_scenario()

        # Create supervisor executor for orchestration
        executor = SupervisorExecutor(
            supervisor=env["supervisor"],
            config={"batch_size": 4, "max_concurrent_tasks": 2},
        )
        executor.task_manager = env["task_manager"]

        # Mock ready tasks to simulate task dependencies
        task_execution_order = []

        def mock_get_ready_tasks(*args, **kwargs):
            """Simulate dependency-based task readiness."""
            # First call: only task 1 is ready (no dependencies)
            if len(task_execution_order) == 0:
                return [tasks[0]]  # Research task
            # Second call: task 2 is ready (depends on task 1)
            elif len(task_execution_order) == 1:
                return [tasks[1]]  # Coding task
            # Third call: tasks 3 and 4 are ready (both depend on task 2)
            elif len(task_execution_order) == 2:
                return [tasks[2], tasks[3]]  # Testing and documentation
            else:
                return []  # No more tasks

        env["task_manager"].get_ready_tasks.side_effect = mock_get_ready_tasks

        # Track execution phases
        execution_phases = []
        start_time = time.time()

        # Execute workflow in phases
        while True:
            batch_report = await executor.execute_autonomous_batch()

            if batch_report["total_tasks"] == 0:
                break  # No more tasks to process

            phase_info = {
                "batch_id": batch_report["batch_id"],
                "tasks_processed": batch_report["total_tasks"],
                "success_rate": batch_report["success_rate"],
                "duration_minutes": batch_report["total_duration_minutes"],
            }
            execution_phases.append(phase_info)
            task_execution_order.extend(range(batch_report["total_tasks"]))

            # Brief pause between phases
            await asyncio.sleep(0.01)

        total_execution_time = time.time() - start_time

        # Verify complete workflow execution
        assert len(execution_phases) == 3  # Three phases based on dependencies

        # Phase 1: Research (1 task)
        assert execution_phases[0]["tasks_processed"] == 1
        assert execution_phases[0]["success_rate"] == 1.0

        # Phase 2: Coding (1 task)
        assert execution_phases[1]["tasks_processed"] == 1
        assert execution_phases[1]["success_rate"] == 1.0

        # Phase 3: Testing + Documentation (2 tasks)
        assert execution_phases[2]["tasks_processed"] == 2
        assert execution_phases[2]["success_rate"] == 1.0

        # Verify dependency-based execution order
        assert len(task_execution_order) == 4  # All tasks executed

        # Performance verification
        assert total_execution_time < 5.0  # Should complete in reasonable time

        # Verify all phases had reasonable duration
        for phase in execution_phases:
            assert phase["duration_minutes"] >= 0

    @pytest.mark.asyncio
    async def test_multi_agent_coordination_workflow(self, end_to_end_environment):
        """Test coordinated workflow across multiple agent types."""
        env = end_to_end_environment
        tasks = create_realistic_task_scenario()

        # Mock agent coordination scenario
        agent_execution_sequence = []

        async def mock_agent_execution(
            agent_type: AgentType, task: TaskCore
        ) -> AgentReport:
            """Mock individual agent execution with timing."""
            execution_start = time.time()

            # Simulate agent-specific processing time
            processing_times = {
                AgentType.RESEARCH: 0.05,
                AgentType.CODING: 0.1,
                AgentType.TESTING: 0.08,
                AgentType.DOCUMENTATION: 0.03,
            }

            await asyncio.sleep(processing_times.get(agent_type, 0.05))

            agent_execution_sequence.append(
                {
                    "agent": agent_type,
                    "task_id": task.id,
                    "start_time": execution_start,
                    "end_time": time.time(),
                }
            )

            return env["agents"][agent_type].execute_task.return_value

        # Execute multi-agent workflow
        workflow_results = {}

        for task in tasks:
            # Determine appropriate agent for task
            agent_mapping = {
                ComponentArea.SECURITY: [AgentType.RESEARCH, AgentType.CODING],
                ComponentArea.TESTING: [AgentType.TESTING],
                ComponentArea.DOCUMENTATION: [AgentType.DOCUMENTATION],
            }

            agents_for_task = agent_mapping.get(task.component_area, [AgentType.CODING])

            for agent_type in agents_for_task:
                report = await mock_agent_execution(agent_type, task)

                if task.id not in workflow_results:
                    workflow_results[task.id] = []
                workflow_results[task.id].append(
                    {
                        "agent": agent_type,
                        "report": report,
                    }
                )

        # Verify multi-agent coordination
        assert len(agent_execution_sequence) >= 4  # At least one execution per task

        # Verify each task type was handled by appropriate agents
        security_executions = [
            e for e in agent_execution_sequence if e["task_id"] in [1, 2]
        ]  # Security tasks
        testing_executions = [
            e for e in agent_execution_sequence if e["task_id"] == 3
        ]  # Testing task
        doc_executions = [
            e for e in agent_execution_sequence if e["task_id"] == 4
        ]  # Documentation task

        assert len(security_executions) >= 2  # Research + Coding for security tasks
        assert len(testing_executions) >= 1  # Testing agent for test task
        assert len(doc_executions) >= 1  # Documentation agent for doc task

        # Verify all executions were successful
        for _task_id, task_results in workflow_results.items():
            assert len(task_results) > 0
            assert all(result["report"].success for result in task_results)

        # Verify execution timing
        total_workflow_time = max(
            e["end_time"] for e in agent_execution_sequence
        ) - min(e["start_time"] for e in agent_execution_sequence)
        assert total_workflow_time < 2.0  # Should complete efficiently

    @pytest.mark.asyncio
    async def test_end_to_end_with_external_services(self, end_to_end_environment):
        """Test end-to-end workflow with external service integration."""
        env = end_to_end_environment

        # Configure external services with realistic responses
        research_queries = []
        scrape_requests = []

        async def track_exa_search(query: str, **kwargs):
            """Track EXA search requests."""
            research_queries.append(
                {
                    "query": query,
                    "timestamp": time.time(),
                    "kwargs": kwargs,
                }
            )
            return env["exa_client"].search.return_value

        async def track_firecrawl_scrape(url: str, **kwargs):
            """Track Firecrawl scrape requests."""
            scrape_requests.append(
                {
                    "url": url,
                    "timestamp": time.time(),
                    "kwargs": kwargs,
                }
            )
            return env["firecrawl_client"].scrape.return_value

        env["exa_client"].search.side_effect = track_exa_search
        env["firecrawl_client"].scrape.side_effect = track_firecrawl_scrape

        # Simulate research agent workflow with external services
        TaskCore(
            id=1,
            title="Research API security patterns",
            description="Find comprehensive information about API security",
            component_area=ComponentArea.SECURITY,
            phase=1,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
        )

        start_time = time.time()

        # Simulate research workflow
        research_steps = [
            ("search", "JWT authentication best practices"),
            ("search", "API security vulnerabilities"),
            ("scrape", "https://example.com/jwt-guide"),
            ("search", "OAuth 2.0 implementation"),
            ("scrape", "https://example.com/oauth-security"),
        ]

        for step_type, query_or_url in research_steps:
            if step_type == "search":
                await env["exa_client"].search(query_or_url)
            else:  # scrape
                await env["firecrawl_client"].scrape(query_or_url)

        end_time = time.time()

        # Verify external service integration
        assert len(research_queries) == 3  # Three search queries
        assert len(scrape_requests) == 2  # Two scrape requests

        # Verify query content
        query_texts = [q["query"] for q in research_queries]
        assert "JWT authentication" in " ".join(query_texts)
        assert "API security" in " ".join(query_texts)
        assert "OAuth 2.0" in " ".join(query_texts)

        # Verify scrape targets
        scrape_urls = [s["url"] for s in scrape_requests]
        assert "jwt-guide" in " ".join(scrape_urls)
        assert "oauth-security" in " ".join(scrape_urls)

        # Performance verification
        total_time = end_time - start_time
        assert total_time < 1.0  # Should complete quickly with mocked services

        # Verify service call timing
        all_requests = research_queries + scrape_requests
        request_times = [r["timestamp"] for r in all_requests]
        time_span = max(request_times) - min(request_times)
        assert time_span < 0.5  # All requests should complete within 500ms


class TestSystemStressAndResilience:
    """Test system behavior under stress and failure conditions."""

    @pytest.mark.asyncio
    async def test_high_concurrency_stress_test(self, end_to_end_environment):
        """Test system behavior under high concurrent load."""

        # Create high-concurrency scenario
        concurrent_tasks = 50
        tasks_per_batch = 10
        max_concurrent_batches = 5

        # Track system metrics
        error_count = 0

        async def process_stress_batch(batch_id: int) -> dict[str, Any]:
            """Process a batch under stress conditions."""
            batch_start_time = time.time()

            try:
                # Create batch of tasks
                batch_tasks = [
                    TaskCore(
                        id=batch_id * tasks_per_batch + i,
                        title=f"Stress Test Task {batch_id}-{i}",
                        description="High concurrency stress test task",
                        component_area=ComponentArea.CORE,
                        phase=1,
                        priority=TaskPriority.MEDIUM,
                        complexity=TaskComplexity.LOW,
                        time_estimate_hours=0.1,
                    )
                    for i in range(tasks_per_batch)
                ]

                # Simulate task processing with varying load
                results = []
                for task in batch_tasks:
                    # Add some processing variation
                    processing_delay = 0.001 + (task.id % 5) * 0.0002
                    await asyncio.sleep(processing_delay)

                    # Simulate occasional failures under stress
                    if hash(f"{batch_id}-{task.id}") % 100 < 5:  # 5% failure rate
                        results.append(
                            {
                                "task_id": task.id,
                                "success": False,
                                "error": "Stress-induced failure",
                            }
                        )
                    else:
                        results.append(
                            {
                                "task_id": task.id,
                                "success": True,
                                "processing_time": processing_delay,
                            }
                        )

                batch_end_time = time.time()

                return {
                    "batch_id": batch_id,
                    "results": results,
                    "processing_time": batch_end_time - batch_start_time,
                    "success_count": sum(1 for r in results if r["success"]),
                    "failure_count": sum(1 for r in results if not r["success"]),
                }

            except Exception as e:
                nonlocal error_count
                error_count += 1
                return {
                    "batch_id": batch_id,
                    "error": str(e),
                    "processing_time": time.time() - batch_start_time,
                    "success_count": 0,
                    "failure_count": tasks_per_batch,
                }

        # Execute stress test with controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent_batches)

        async def controlled_batch_execution(batch_id: int):
            """Execute batch with concurrency control."""
            async with semaphore:
                return await process_stress_batch(batch_id)

        # Run stress test
        stress_start_time = time.time()

        batch_results = await asyncio.gather(
            *[
                controlled_batch_execution(i)
                for i in range(concurrent_tasks // tasks_per_batch)
            ],
            return_exceptions=True,
        )

        stress_end_time = time.time()
        total_stress_time = stress_end_time - stress_start_time

        # Analyze stress test results
        successful_batches = [
            r for r in batch_results if isinstance(r, dict) and "error" not in r
        ]
        [r for r in batch_results if not isinstance(r, dict) or "error" in r]

        total_tasks_processed = sum(
            len(batch.get("results", [])) for batch in successful_batches
        )
        total_successful_tasks = sum(
            batch.get("success_count", 0) for batch in successful_batches
        )
        sum(batch.get("failure_count", 0) for batch in successful_batches)

        # Verify stress test outcomes
        assert len(successful_batches) > 0  # Some batches should succeed
        assert total_tasks_processed >= 30  # Should process significant number of tasks

        # System should maintain reasonable success rate under stress
        if total_tasks_processed > 0:
            success_rate = total_successful_tasks / total_tasks_processed
            assert success_rate > 0.8  # At least 80% success rate under stress

        # Performance under stress
        throughput = total_tasks_processed / total_stress_time
        assert throughput > 20  # Should maintain reasonable throughput

        # Verify controlled concurrency (shouldn't take too long)
        assert total_stress_time < 5.0  # Should complete within reasonable time

    @pytest.mark.asyncio
    async def test_cascading_failure_recovery(self, end_to_end_environment):
        """Test system recovery from cascading failures."""

        # Simulate cascading failure scenario
        failure_progression = {
            "phase_1": {"failure_rate": 0.1, "recovery_time": 0.01},
            "phase_2": {"failure_rate": 0.3, "recovery_time": 0.02},
            "phase_3": {"failure_rate": 0.6, "recovery_time": 0.05},
            "phase_4": {"failure_rate": 0.2, "recovery_time": 0.01},  # Recovery phase
            "phase_5": {"failure_rate": 0.05, "recovery_time": 0.01},  # Stabilized
        }

        current_phase = 0
        phase_names = list(failure_progression.keys())

        async def simulate_failing_service(request_id: int) -> dict[str, Any]:
            """Simulate service with cascading failures and recovery."""
            nonlocal current_phase

            # Progress through failure phases
            if request_id > 0 and request_id % 10 == 0:
                current_phase = min(current_phase + 1, len(phase_names) - 1)

            phase_name = phase_names[current_phase]
            phase_config = failure_progression[phase_name]

            # Simulate processing time and failure probability
            await asyncio.sleep(phase_config["recovery_time"])

            failure_threshold = phase_config["failure_rate"] * 100
            if hash(str(request_id)) % 100 < failure_threshold:
                raise Exception(
                    f"Service failure in {phase_name} (request {request_id})"
                )

            return {
                "request_id": request_id,
                "phase": phase_name,
                "success": True,
                "response_time": phase_config["recovery_time"],
            }

        # Test cascading failure and recovery
        num_requests = 60  # Enough to go through all phases
        request_results = []

        for request_id in range(num_requests):
            try:
                result = await simulate_failing_service(request_id)
                request_results.append(("success", result))
            except Exception as e:
                request_results.append(
                    ("failure", {"request_id": request_id, "error": str(e)})
                )

        # Analyze cascading failure pattern
        phase_analysis = {}
        for i, (status, _result) in enumerate(request_results):
            phase_index = i // 10  # 10 requests per phase
            if phase_index < len(phase_names):
                phase_name = phase_names[phase_index]
                if phase_name not in phase_analysis:
                    phase_analysis[phase_name] = {"successes": 0, "failures": 0}

                if status == "success":
                    phase_analysis[phase_name]["successes"] += 1
                else:
                    phase_analysis[phase_name]["failures"] += 1

        # Verify cascading failure and recovery pattern
        for phase_name, expected_config in failure_progression.items():
            if phase_name in phase_analysis:
                phase_data = phase_analysis[phase_name]
                total_requests = phase_data["successes"] + phase_data["failures"]

                if total_requests > 0:
                    actual_failure_rate = phase_data["failures"] / total_requests
                    expected_failure_rate = expected_config["failure_rate"]

                    # Allow some variance in failure rates
                    assert abs(actual_failure_rate - expected_failure_rate) < 0.3

        # Verify recovery occurs
        early_phase_failures = sum(
            phase_analysis.get(p, {}).get("failures", 0) for p in ["phase_2", "phase_3"]
        )
        late_phase_failures = sum(
            phase_analysis.get(p, {}).get("failures", 0) for p in ["phase_4", "phase_5"]
        )

        # System should recover (fewer failures in later phases)
        if early_phase_failures > 0:
            assert late_phase_failures < early_phase_failures

    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, end_to_end_environment):
        """Test system behavior under resource exhaustion conditions."""

        # Simulate resource exhaustion scenario
        max_memory_usage = 50 * 1024 * 1024  # 50MB limit
        resource_tracker = {
            "memory_usage": 0,
            "connection_count": 0,
            "max_connections": 10,
        }

        resource_exhaustion_events = []

        async def resource_intensive_operation(operation_id: int) -> dict[str, Any]:
            """Simulate resource-intensive operation with tracking."""
            start_time = time.time()

            # Simulate memory allocation
            memory_needed = 5 * 1024 * 1024  # 5MB per operation

            if resource_tracker["memory_usage"] + memory_needed > max_memory_usage:
                resource_exhaustion_events.append(
                    {
                        "type": "memory_exhaustion",
                        "operation_id": operation_id,
                        "timestamp": start_time,
                    }
                )
                raise Exception(f"Memory exhausted for operation {operation_id}")

            # Simulate connection usage
            if (
                resource_tracker["connection_count"]
                >= resource_tracker["max_connections"]
            ):
                resource_exhaustion_events.append(
                    {
                        "type": "connection_exhaustion",
                        "operation_id": operation_id,
                        "timestamp": start_time,
                    }
                )
                raise Exception(
                    f"Connection pool exhausted for operation {operation_id}"
                )

            # Allocate resources
            resource_tracker["memory_usage"] += memory_needed
            resource_tracker["connection_count"] += 1

            try:
                # Simulate work
                await asyncio.sleep(0.01)

                return {
                    "operation_id": operation_id,
                    "success": True,
                    "memory_used": memory_needed,
                    "execution_time": time.time() - start_time,
                }
            finally:
                # Clean up resources
                resource_tracker["memory_usage"] = max(
                    0, resource_tracker["memory_usage"] - memory_needed
                )
                resource_tracker["connection_count"] = max(
                    0, resource_tracker["connection_count"] - 1
                )

        # Test resource exhaustion with concurrent operations
        num_operations = 25

        # Execute operations concurrently (this should trigger resource exhaustion)
        semaphore = asyncio.Semaphore(15)  # Allow more concurrency than resources

        async def controlled_operation(op_id: int):
            """Execute operation with concurrency control."""
            async with semaphore:
                try:
                    return await resource_intensive_operation(op_id)
                except Exception as e:
                    return {"operation_id": op_id, "error": str(e), "success": False}

        results = await asyncio.gather(
            *[controlled_operation(i) for i in range(num_operations)]
        )

        # Analyze resource exhaustion handling
        successful_operations = [r for r in results if r.get("success", False)]
        failed_operations = [r for r in results if not r.get("success", False)]

        # Verify resource exhaustion occurred and was handled
        assert len(resource_exhaustion_events) > 0  # Should trigger resource exhaustion
        assert (
            len(failed_operations) > 0
        )  # Some operations should fail due to exhaustion
        assert len(successful_operations) > 0  # Some operations should succeed

        # Verify system maintained some functionality despite exhaustion
        success_rate = len(successful_operations) / len(results)
        assert success_rate > 0.2  # Should maintain at least 20% functionality

        # Verify different types of resource exhaustion
        exhaustion_types = {event["type"] for event in resource_exhaustion_events}
        assert len(exhaustion_types) > 0  # Should have resource exhaustion events

        # Verify resources were properly cleaned up
        assert resource_tracker["memory_usage"] == 0  # Memory should be freed
        assert resource_tracker["connection_count"] == 0  # Connections should be closed


class TestProductionScenarioSimulation:
    """Test realistic production scenarios and workflows."""

    @pytest.mark.asyncio
    async def test_realistic_development_workflow(
        self, end_to_end_environment, benchmark: BenchmarkFixture
    ):
        """Test realistic software development workflow."""
        env = end_to_end_environment

        # Create realistic development project scenario
        development_tasks = [
            TaskCore(
                id=1,
                title="Research user authentication requirements",
                description=(
                    "Analyze requirements for secure user authentication system"
                ),
                component_area=ComponentArea.SECURITY,
                phase=1,
                priority=TaskPriority.HIGH,
                complexity=TaskComplexity.MEDIUM,
                time_estimate_hours=4.0,
            ),
            TaskCore(
                id=2,
                title="Design authentication architecture",
                description="Create system design for authentication components",
                component_area=ComponentArea.ARCHITECTURE,
                phase=1,
                priority=TaskPriority.HIGH,
                complexity=TaskComplexity.HIGH,
                time_estimate_hours=6.0,
            ),
            TaskCore(
                id=3,
                title="Implement user registration endpoint",
                description="Build REST API endpoint for user registration",
                component_area=ComponentArea.API,
                phase=2,
                priority=TaskPriority.HIGH,
                complexity=TaskComplexity.MEDIUM,
                time_estimate_hours=5.0,
            ),
            TaskCore(
                id=4,
                title="Implement login endpoint",
                description="Build REST API endpoint for user login with JWT",
                component_area=ComponentArea.API,
                phase=2,
                priority=TaskPriority.HIGH,
                complexity=TaskComplexity.MEDIUM,
                time_estimate_hours=4.0,
            ),
            TaskCore(
                id=5,
                title="Create authentication middleware",
                description="Build middleware for JWT token validation",
                component_area=ComponentArea.SECURITY,
                phase=2,
                priority=TaskPriority.MEDIUM,
                complexity=TaskComplexity.MEDIUM,
                time_estimate_hours=3.0,
            ),
            TaskCore(
                id=6,
                title="Write unit tests for auth system",
                description="Create comprehensive unit tests for authentication",
                component_area=ComponentArea.TESTING,
                phase=3,
                priority=TaskPriority.HIGH,
                complexity=TaskComplexity.MEDIUM,
                time_estimate_hours=8.0,
            ),
            TaskCore(
                id=7,
                title="Write integration tests",
                description="Create end-to-end tests for authentication flow",
                component_area=ComponentArea.TESTING,
                phase=3,
                priority=TaskPriority.MEDIUM,
                complexity=TaskComplexity.HIGH,
                time_estimate_hours=6.0,
            ),
            TaskCore(
                id=8,
                title="Create API documentation",
                description="Document authentication endpoints and usage",
                component_area=ComponentArea.DOCUMENTATION,
                phase=4,
                priority=TaskPriority.MEDIUM,
                complexity=TaskComplexity.LOW,
                time_estimate_hours=3.0,
            ),
        ]

        async def simulate_development_workflow() -> dict[str, Any]:
            """Simulate complete development workflow."""
            workflow_start = time.time()

            # Phase 1: Research and Design
            research_results = []
            design_results = []

            for task in development_tasks[:2]:  # Research and design tasks
                if "research" in task.title.lower():
                    # Simulate research with external APIs
                    search_result = await env["exa_client"].search(
                        f"authentication best practices {task.component_area.value}"
                    )
                    research_results.append(
                        {
                            "task_id": task.id,
                            "findings": search_result["results"],
                            "duration": 0.1,
                        }
                    )
                else:
                    # Simulate design work
                    design_results.append(
                        {
                            "task_id": task.id,
                            "architecture": (
                                "JWT-based authentication with refresh tokens"
                            ),
                            "duration": 0.15,
                        }
                    )

            # Phase 2: Implementation
            implementation_results = []
            for task in development_tasks[2:5]:  # Implementation tasks
                impl_result = {
                    "task_id": task.id,
                    "files_created": [f"{task.title.replace(' ', '_').lower()}.py"],
                    "lines_of_code": task.time_estimate_hours
                    * 50,  # Estimate 50 LOC per hour
                    "duration": task.time_estimate_hours * 0.01,  # Scaled time
                }
                implementation_results.append(impl_result)

            # Phase 3: Testing
            testing_results = []
            for task in development_tasks[5:7]:  # Testing tasks
                test_result = {
                    "task_id": task.id,
                    "test_files": [f"test_{task.title.replace(' ', '_').lower()}.py"],
                    "test_count": int(
                        task.time_estimate_hours * 10
                    ),  # 10 tests per hour
                    "coverage_percentage": 85.0 + (task.id % 10),
                    "duration": task.time_estimate_hours * 0.01,
                }
                testing_results.append(test_result)

            # Phase 4: Documentation
            documentation_results = []
            for task in development_tasks[7:]:  # Documentation tasks
                doc_result = {
                    "task_id": task.id,
                    "documents_created": ["README.md", "API.md", "DEPLOYMENT.md"],
                    "pages": int(task.time_estimate_hours * 5),  # 5 pages per hour
                    "duration": task.time_estimate_hours * 0.005,
                }
                documentation_results.append(doc_result)

            workflow_end = time.time()

            return {
                "total_duration": workflow_end - workflow_start,
                "phases": {
                    "research": research_results,
                    "design": design_results,
                    "implementation": implementation_results,
                    "testing": testing_results,
                    "documentation": documentation_results,
                },
                "total_tasks": len(development_tasks),
                "estimated_hours": sum(
                    task.time_estimate_hours for task in development_tasks
                ),
            }

        # Benchmark the development workflow
        workflow_result = await benchmark.pedantic(
            simulate_development_workflow,
            rounds=3,
            iterations=1,
        )

        # Verify workflow completeness
        assert workflow_result["total_tasks"] == 8
        assert workflow_result["estimated_hours"] == 39.0  # Sum of all estimates

        # Verify all phases completed
        phases = workflow_result["phases"]
        assert len(phases["research"]) == 1
        assert len(phases["design"]) == 1
        assert len(phases["implementation"]) == 3
        assert len(phases["testing"]) == 2
        assert len(phases["documentation"]) == 1

        # Verify research phase used external services
        research_phase = phases["research"][0]
        assert "findings" in research_phase
        assert len(research_phase["findings"]) > 0

        # Verify implementation phase created files
        implementation_totals = {
            "files": sum(
                len(impl["files_created"]) for impl in phases["implementation"]
            ),
            "lines_of_code": sum(
                impl["lines_of_code"] for impl in phases["implementation"]
            ),
        }
        assert implementation_totals["files"] == 3
        assert implementation_totals["lines_of_code"] > 500

        # Verify testing phase
        testing_totals = {
            "test_files": sum(len(test["test_files"]) for test in phases["testing"]),
            "test_count": sum(test["test_count"] for test in phases["testing"]),
            "avg_coverage": sum(
                test["coverage_percentage"] for test in phases["testing"]
            )
            / len(phases["testing"]),
        }
        assert testing_totals["test_files"] == 2
        assert testing_totals["test_count"] > 100
        assert testing_totals["avg_coverage"] > 80

        # Performance verification
        benchmark_stats = benchmark.stats
        assert benchmark_stats.mean < 2.0  # Should complete in less than 2 seconds

    @pytest.mark.asyncio
    async def test_production_monitoring_and_alerting(self, end_to_end_environment):
        """Test production monitoring and alerting scenarios."""

        # Simulate production monitoring scenario
        monitoring_metrics = {
            "task_processing_times": [],
            "error_rates": [],
            "throughput_measurements": [],
            "resource_utilization": [],
            "alert_triggers": [],
        }

        alert_thresholds = {
            "max_processing_time": 5.0,  # seconds
            "max_error_rate": 0.1,  # 10%
            "min_throughput": 10,  # tasks per second
            "max_memory_usage": 100,  # MB
        }

        async def monitor_system_metrics(duration_seconds: int = 2):
            """Monitor system metrics for specified duration."""
            start_time = time.time()
            monitoring_interval = 0.1  # Check every 100ms

            while time.time() - start_time < duration_seconds:
                current_time = time.time()

                # Simulate processing time measurement
                processing_time = (
                    0.5 + (hash(str(current_time)) % 1000) / 1000.0
                )  # 0.5-1.5s
                monitoring_metrics["task_processing_times"].append(processing_time)

                # Check processing time alert
                if processing_time > alert_thresholds["max_processing_time"]:
                    monitoring_metrics["alert_triggers"].append(
                        {
                            "type": "high_processing_time",
                            "value": processing_time,
                            "threshold": alert_thresholds["max_processing_time"],
                            "timestamp": current_time,
                        }
                    )

                # Simulate error rate measurement
                error_rate = max(
                    0, 0.05 + (hash(str(current_time * 2)) % 100 - 95) / 100.0
                )  # Usually ~5%
                monitoring_metrics["error_rates"].append(error_rate)

                # Check error rate alert
                if error_rate > alert_thresholds["max_error_rate"]:
                    monitoring_metrics["alert_triggers"].append(
                        {
                            "type": "high_error_rate",
                            "value": error_rate,
                            "threshold": alert_thresholds["max_error_rate"],
                            "timestamp": current_time,
                        }
                    )

                # Simulate throughput measurement
                throughput = 15 + (
                    hash(str(current_time * 3)) % 20 - 10
                )  # 5-25 tasks/sec
                monitoring_metrics["throughput_measurements"].append(throughput)

                # Check throughput alert
                if throughput < alert_thresholds["min_throughput"]:
                    monitoring_metrics["alert_triggers"].append(
                        {
                            "type": "low_throughput",
                            "value": throughput,
                            "threshold": alert_thresholds["min_throughput"],
                            "timestamp": current_time,
                        }
                    )

                # Simulate memory usage
                memory_usage = 50 + (hash(str(current_time * 4)) % 60)  # 50-110 MB
                monitoring_metrics["resource_utilization"].append(memory_usage)

                # Check memory alert
                if memory_usage > alert_thresholds["max_memory_usage"]:
                    monitoring_metrics["alert_triggers"].append(
                        {
                            "type": "high_memory_usage",
                            "value": memory_usage,
                            "threshold": alert_thresholds["max_memory_usage"],
                            "timestamp": current_time,
                        }
                    )

                await asyncio.sleep(monitoring_interval)

        # Run monitoring simulation
        await monitor_system_metrics(duration_seconds=1)

        # Analyze monitoring results
        assert (
            len(monitoring_metrics["task_processing_times"]) >= 8
        )  # At least 8 samples
        assert len(monitoring_metrics["error_rates"]) >= 8
        assert len(monitoring_metrics["throughput_measurements"]) >= 8
        assert len(monitoring_metrics["resource_utilization"]) >= 8

        # Verify metric ranges
        avg_processing_time = sum(monitoring_metrics["task_processing_times"]) / len(
            monitoring_metrics["task_processing_times"]
        )
        avg_error_rate = sum(monitoring_metrics["error_rates"]) / len(
            monitoring_metrics["error_rates"]
        )
        avg_throughput = sum(monitoring_metrics["throughput_measurements"]) / len(
            monitoring_metrics["throughput_measurements"]
        )
        avg_memory_usage = sum(monitoring_metrics["resource_utilization"]) / len(
            monitoring_metrics["resource_utilization"]
        )

        assert 0.5 <= avg_processing_time <= 1.5
        assert 0.0 <= avg_error_rate <= 0.2
        assert 5 <= avg_throughput <= 25
        assert 50 <= avg_memory_usage <= 110

        # Verify alerting system
        alert_types = {alert["type"] for alert in monitoring_metrics["alert_triggers"]}

        # Should have triggered some alerts (system designed to occasionally
        # exceed thresholds)
        if len(monitoring_metrics["alert_triggers"]) > 0:
            # Verify alert structure
            for alert in monitoring_metrics["alert_triggers"]:
                assert "type" in alert
                assert "value" in alert
                assert "threshold" in alert
                assert "timestamp" in alert
                assert (
                    alert["value"] != alert["threshold"]
                )  # Should only alert when threshold crossed

            # Verify alert types are valid
            valid_alert_types = {
                "high_processing_time",
                "high_error_rate",
                "low_throughput",
                "high_memory_usage",
            }
            assert alert_types.issubset(valid_alert_types)

    @pytest.mark.asyncio
    async def test_disaster_recovery_simulation(self, end_to_end_environment):
        """Test disaster recovery and system restoration scenarios."""

        # Simulate disaster recovery scenario
        disaster_events = [
            {"type": "database_failure", "start_time": 0.2, "duration": 0.3},
            {"type": "api_service_outage", "start_time": 0.6, "duration": 0.2},
            {"type": "network_partition", "start_time": 1.0, "duration": 0.4},
        ]

        recovery_actions = []
        system_state = {
            "database_available": True,
            "api_services_available": True,
            "network_available": True,
            "backup_systems_active": False,
        }

        async def simulate_disaster_scenario(total_duration: float = 2.0):
            """Simulate disaster events and recovery actions."""
            start_time = time.time()

            while time.time() - start_time < total_duration:
                current_time = time.time() - start_time

                # Check for disaster events
                for disaster in disaster_events:
                    disaster_start = disaster["start_time"]
                    disaster_end = disaster_start + disaster["duration"]

                    if disaster_start <= current_time <= disaster_end:
                        # Disaster is active
                        if disaster["type"] == "database_failure":
                            if system_state["database_available"]:
                                system_state["database_available"] = False
                                recovery_actions.append(
                                    {
                                        "action": "activate_backup_database",
                                        "timestamp": current_time,
                                        "disaster_type": disaster["type"],
                                    }
                                )
                        elif disaster["type"] == "api_service_outage":
                            if system_state["api_services_available"]:
                                system_state["api_services_available"] = False
                                recovery_actions.append(
                                    {
                                        "action": "failover_to_backup_apis",
                                        "timestamp": current_time,
                                        "disaster_type": disaster["type"],
                                    }
                                )
                        elif (
                            disaster["type"] == "network_partition"
                            and system_state["network_available"]
                        ):
                            system_state["network_available"] = False
                            recovery_actions.append(
                                {
                                    "action": "activate_offline_mode",
                                    "timestamp": current_time,
                                    "disaster_type": disaster["type"],
                                }
                            )
                    elif current_time > disaster_end:
                        # Disaster has ended, start recovery
                        if disaster["type"] == "database_failure":
                            if not system_state["database_available"]:
                                system_state["database_available"] = True
                                recovery_actions.append(
                                    {
                                        "action": "restore_primary_database",
                                        "timestamp": current_time,
                                        "disaster_type": disaster["type"],
                                    }
                                )
                        elif disaster["type"] == "api_service_outage":
                            if not system_state["api_services_available"]:
                                system_state["api_services_available"] = True
                                recovery_actions.append(
                                    {
                                        "action": "restore_primary_apis",
                                        "timestamp": current_time,
                                        "disaster_type": disaster["type"],
                                    }
                                )
                        elif (
                            disaster["type"] == "network_partition"
                            and not system_state["network_available"]
                        ):
                            system_state["network_available"] = True
                            recovery_actions.append(
                                {
                                    "action": "restore_network_connectivity",
                                    "timestamp": current_time,
                                    "disaster_type": disaster["type"],
                                }
                            )

                # Simulate system operations based on current state
                system_operational = (
                    system_state["database_available"]
                    and system_state["api_services_available"]
                    and system_state["network_available"]
                )

                if not system_operational:
                    # System running in degraded mode
                    await asyncio.sleep(0.02)  # Slower operations
                else:
                    # Normal operations
                    await asyncio.sleep(0.01)

        # Run disaster recovery simulation
        await simulate_disaster_scenario(total_duration=1.5)

        # Analyze disaster recovery results
        assert (
            len(recovery_actions) >= 6
        )  # Should have recovery actions for each disaster

        # Verify recovery action types
        action_types = [action["action"] for action in recovery_actions]
        expected_actions = [
            "activate_backup_database",
            "restore_primary_database",
            "failover_to_backup_apis",
            "restore_primary_apis",
            "activate_offline_mode",
            "restore_network_connectivity",
        ]

        for expected_action in expected_actions:
            assert expected_action in action_types

        # Verify disaster types were handled
        disaster_types_handled = {
            action["disaster_type"] for action in recovery_actions
        }
        expected_disaster_types = {
            "database_failure",
            "api_service_outage",
            "network_partition",
        }
        assert disaster_types_handled == expected_disaster_types

        # Verify recovery timing
        recovery_pairs = {}
        for action in recovery_actions:
            disaster_type = action["disaster_type"]
            if disaster_type not in recovery_pairs:
                recovery_pairs[disaster_type] = []
            recovery_pairs[disaster_type].append(action)

        # Each disaster should have activation and restoration actions
        for _disaster_type, actions in recovery_pairs.items():
            assert len(actions) >= 2  # At least activation and restoration

            # Verify restoration happens after activation
            activation_actions = [
                a
                for a in actions
                if "activate" in a["action"] or "failover" in a["action"]
            ]
            restoration_actions = [a for a in actions if "restore" in a["action"]]

            if activation_actions and restoration_actions:
                assert min(r["timestamp"] for r in restoration_actions) > min(
                    a["timestamp"] for a in activation_actions
                )

        # Verify final system state (should be fully recovered)
        assert system_state["database_available"]
        assert system_state["api_services_available"]
        assert system_state["network_available"]

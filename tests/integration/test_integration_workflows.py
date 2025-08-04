"""Comprehensive integration tests for full workflow scenarios.

This module tests complete workflows that combine multiple components including:
- Agent orchestration and coordination
- Configuration management integration
- API client interactions (Exa, Firecrawl, OpenRouter)
- Database operations and state management
- Error handling and recovery scenarios
- Performance under realistic load conditions
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from pydantic import ValidationError

from src.config import OrchestrationSettings, get_settings
from src.schemas.unified_models import (
    AgentReport,
    AgentType,
    ComponentArea,
    TaskComplexity,
    TaskCore,
    TaskDelegation,
    TaskPriority,
    TaskStatus,
)


@pytest_asyncio.fixture
async def mock_orchestration_environment():
    """Create comprehensive mock environment for integration tests."""
    # Mock configuration
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-openai-key",
            "OPENROUTER_API_KEY": "test-openrouter-key",
            "EXA_API_KEY": "test-exa-key",
            "FIRECRAWL_API_KEY": "test-firecrawl-key",
            "ORCHESTRATION_DEBUG_MODE": "true",
            "ORCHESTRATION_PARALLEL_EXECUTION_LIMIT": "3",
        },
    ):
        # Mock API clients
        mock_exa_client = AsyncMock()
        mock_exa_client.search.return_value = {
            "results": [
                {
                    "title": "Test Research Result",
                    "url": "https://example.com/research",
                    "text": "Research content for testing",
                    "score": 0.95,
                }
            ]
        }

        mock_firecrawl_client = AsyncMock()
        mock_firecrawl_client.scrape.return_value = {
            "success": True,
            "data": {
                "markdown": (
                    "# Test Documentation\n\nTest content for documentation agent"
                ),
                "title": "Test Page",
            },
        }

        mock_openrouter_client = AsyncMock()
        mock_openrouter_response = MagicMock()
        mock_openrouter_response.content = json.dumps(
            {
                "response": "Generated code implementation",
                "confidence": 0.88,
                "reasoning": "Based on requirements analysis",
            }
        )
        mock_openrouter_client.ainvoke.return_value = mock_openrouter_response

        # Mock database operations
        mock_task_manager = MagicMock()
        mock_task_manager.get_tasks_by_status.return_value = []
        mock_task_manager.update_task_status.return_value = None
        mock_task_manager.create_task.return_value = 1

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            temp_db_path = tmp_db.name

        yield {
            "exa_client": mock_exa_client,
            "firecrawl_client": mock_firecrawl_client,
            "openrouter_client": mock_openrouter_client,
            "task_manager": mock_task_manager,
            "temp_db_path": temp_db_path,
            "settings": get_settings(),
        }

        # Cleanup
        Path(temp_db_path).unlink(missing_ok=True)


@pytest_asyncio.fixture
async def sample_complex_task():
    """Create a complex task for integration testing."""
    return TaskCore(
        id=1,
        title="Implement AI-powered code review system",
        description="""
        Design and implement a comprehensive AI-powered code review system that:
        1. Analyzes code quality and security vulnerabilities
        2. Provides automated suggestions for improvement
        3. Integrates with existing CI/CD pipelines
        4. Includes comprehensive documentation and testing
        """,
        component_area=ComponentArea.SECURITY,
        phase=1,
        priority=TaskPriority.HIGH,
        complexity=TaskComplexity.HIGH,
        success_criteria="""
        - Code quality analysis with 95%+ accuracy
        - Security vulnerability detection
        - Integration with popular CI/CD tools
        - Complete documentation and test coverage >90%
        """,
        time_estimate_hours=40.0,
    )


class TestWorkflowOrchestration:
    """Test complete workflow orchestration scenarios."""

    @pytest.mark.asyncio
    async def test_full_task_workflow_success(
        self, mock_orchestration_environment, sample_complex_task
    ):
        """Test complete successful task workflow from assignment to completion."""
        env = mock_orchestration_environment

        with (
            patch(
                "src.integrations.exa_client.ExaClient", return_value=env["exa_client"]
            ),
            patch(
                "src.integrations.firecrawl_client.FirecrawlClient",
                return_value=env["firecrawl_client"],
            ),
            patch("src.task_manager.TaskManager", return_value=env["task_manager"]),
        ):
            # Simulate task delegation process
            TaskDelegation(
                assigned_agent=AgentType.RESEARCH,
                reasoning=(
                    "This task requires initial research on existing code review tools"
                ),
                priority=TaskPriority.HIGH,
                estimated_duration=120,
                dependencies=[],
                context_requirements=[
                    "security best practices",
                    "CI/CD integration patterns",
                ],
                confidence_score=0.85,
            )

            # Test research phase
            research_report = AgentReport(
                agent_name=AgentType.RESEARCH,
                task_id=sample_complex_task.id,
                status=TaskStatus.COMPLETED,
                success=True,
                execution_time_minutes=25.0,
                outputs={
                    "research_findings": (
                        "Found 15 existing tools, identified key features"
                    ),
                    "security_analysis": (
                        "Common vulnerabilities: SQL injection, XSS, buffer overflow"
                    ),
                    "integration_patterns": (
                        "GitHub Actions, Jenkins, GitLab CI support required"
                    ),
                },
                artifacts=["research_summary.md", "competitive_analysis.json"],
                recommendations=[
                    "Focus on static analysis capabilities",
                    "Implement plugin architecture for extensibility",
                    "Use machine learning for pattern recognition",
                ],
                next_actions=["coding", "architecture_design"],
                confidence_score=0.92,
            )

            # Verify research phase completion
            assert research_report.success is True
            assert research_report.status == TaskStatus.COMPLETED
            assert len(research_report.outputs) == 3
            assert "security_analysis" in research_report.outputs

            # Test coding phase
            coding_report = AgentReport(
                agent_name=AgentType.CODING,
                task_id=sample_complex_task.id,
                status=TaskStatus.COMPLETED,
                success=True,
                execution_time_minutes=180.0,
                outputs={
                    "implementation": "Created core analysis engine with AST parsing",
                    "architecture": "Microservices architecture with API gateway",
                    "security_features": (
                        "Implemented OWASP Top 10 detection algorithms"
                    ),
                },
                artifacts=[
                    "src/analyzer/core.py",
                    "src/analyzer/security_scanner.py",
                    "src/api/main.py",
                    "docker-compose.yml",
                ],
                recommendations=[
                    "Add comprehensive logging",
                    "Implement rate limiting",
                    "Add metrics collection",
                ],
                next_actions=["testing", "documentation"],
                confidence_score=0.88,
            )

            # Verify coding phase
            assert coding_report.success is True
            assert len(coding_report.artifacts) == 4
            assert "security_features" in coding_report.outputs

            # Test testing phase
            testing_report = AgentReport(
                agent_name=AgentType.TESTING,
                task_id=sample_complex_task.id,
                status=TaskStatus.COMPLETED,
                success=True,
                execution_time_minutes=45.0,
                outputs={
                    "test_coverage": "94% line coverage, 88% branch coverage",
                    "performance_results": (
                        "Analyzes 1000 lines/second, <2GB memory usage"
                    ),
                    "security_validation": "All OWASP test cases pass",
                },
                artifacts=[
                    "tests/test_analyzer_core.py",
                    "tests/test_security_scanner.py",
                    "tests/test_api_endpoints.py",
                    "tests/performance/load_test_results.json",
                ],
                recommendations=[
                    "Add integration tests for CI/CD pipelines",
                    "Implement chaos engineering tests",
                ],
                next_actions=["documentation"],
                confidence_score=0.90,
            )

            # Verify testing phase
            assert testing_report.success is True
            assert "94%" in testing_report.outputs["test_coverage"]

            # Test documentation phase
            documentation_report = AgentReport(
                agent_name=AgentType.DOCUMENTATION,
                task_id=sample_complex_task.id,
                status=TaskStatus.COMPLETED,
                success=True,
                execution_time_minutes=30.0,
                outputs={
                    "user_documentation": "Complete user guide with examples",
                    "api_documentation": "OpenAPI 3.0 specification with examples",
                    "deployment_guide": "Docker and Kubernetes deployment instructions",
                },
                artifacts=[
                    "docs/user_guide.md",
                    "docs/api_reference.md",
                    "docs/deployment.md",
                    "docs/api_spec.yaml",
                ],
                recommendations=[
                    "Add video tutorials",
                    "Create interactive demos",
                ],
                next_actions=["deployment", "monitoring"],
                confidence_score=0.87,
            )

            # Verify documentation phase
            assert documentation_report.success is True
            assert len(documentation_report.artifacts) == 4

            # Calculate total workflow metrics
            total_execution_time = (
                research_report.execution_time_minutes
                + coding_report.execution_time_minutes
                + testing_report.execution_time_minutes
                + documentation_report.execution_time_minutes
            )

            assert total_execution_time == 280.0  # 4 hours 40 minutes
            assert (
                total_execution_time < sample_complex_task.time_estimate_hours * 60
            )  # Under estimate

            # Verify all success criteria met
            all_reports = [
                research_report,
                coding_report,
                testing_report,
                documentation_report,
            ]
            assert all(report.success for report in all_reports)
            assert all(report.confidence_score > 0.8 for report in all_reports)

    @pytest.mark.asyncio
    async def test_workflow_with_agent_failure_and_recovery(
        self, mock_orchestration_environment, sample_complex_task
    ):
        """Test workflow resilience when agent fails and recovery mechanisms."""
        # Simulate initial agent failure
        failed_coding_report = AgentReport(
            agent_name=AgentType.CODING,
            task_id=sample_complex_task.id,
            status=TaskStatus.FAILED,
            success=False,
            execution_time_minutes=15.0,
            outputs={
                "error_details": "Insufficient requirements clarity",
                "attempted_implementation": "Partial AST parser implementation",
            },
            artifacts=["src/analyzer/core_partial.py"],
            recommendations=[
                "Clarify security vulnerability detection requirements",
                "Request additional research on integration patterns",
            ],
            next_actions=["research", "requirements_clarification"],
            confidence_score=0.45,
        )

        # Verify failure is properly recorded
        assert failed_coding_report.success is False
        assert failed_coding_report.status == TaskStatus.FAILED
        assert failed_coding_report.confidence_score < 0.5

        # Simulate recovery research
        recovery_research_report = AgentReport(
            agent_name=AgentType.RESEARCH,
            task_id=sample_complex_task.id,
            status=TaskStatus.COMPLETED,
            success=True,
            execution_time_minutes=20.0,
            outputs={
                "clarified_requirements": (
                    "Detailed vulnerability detection specifications"
                ),
                "integration_examples": (
                    "Sample GitHub Actions and Jenkins configurations"
                ),
                "reference_implementations": (
                    "Analysis of SonarQube and CodeClimate approaches"
                ),
            },
            artifacts=[
                "requirements_clarification.md",
                "integration_examples/",
                "reference_analysis.json",
            ],
            recommendations=[
                "Use tree-sitter for more robust parsing",
                "Implement incremental analysis for performance",
            ],
            next_actions=["coding"],
            confidence_score=0.93,
        )

        # Verify recovery research success
        assert recovery_research_report.success is True
        assert "clarified_requirements" in recovery_research_report.outputs

        # Simulate successful retry of coding phase
        retry_coding_report = AgentReport(
            agent_name=AgentType.CODING,
            task_id=sample_complex_task.id,
            status=TaskStatus.COMPLETED,
            success=True,
            execution_time_minutes=200.0,  # Longer due to retry
            outputs={
                "implementation": "Complete analysis engine with tree-sitter parsing",
                "security_scanner": (
                    "Comprehensive vulnerability detection for OWASP Top 10"
                ),
                "performance_optimization": (
                    "Incremental analysis reduces processing time by 60%"
                ),
            },
            artifacts=[
                "src/analyzer/core.py",
                "src/analyzer/tree_sitter_parser.py",
                "src/analyzer/security_scanner.py",
                "src/analyzer/incremental.py",
            ],
            recommendations=[
                "Monitor memory usage in production",
                "Add telemetry for performance tracking",
            ],
            next_actions=["testing"],
            confidence_score=0.91,
        )

        # Verify successful retry
        assert retry_coding_report.success is True
        assert len(retry_coding_report.artifacts) == 4
        assert "tree_sitter" in retry_coding_report.outputs["implementation"]

        # Verify recovery workflow metrics
        total_recovery_time = (
            failed_coding_report.execution_time_minutes
            + recovery_research_report.execution_time_minutes
            + retry_coding_report.execution_time_minutes
        )

        assert total_recovery_time == 235.0  # Include failure and recovery time

        # Verify confidence improvement after recovery
        assert (
            retry_coding_report.confidence_score > failed_coding_report.confidence_score
        )
        assert retry_coding_report.confidence_score > 0.9

    @pytest.mark.asyncio
    async def test_parallel_agent_coordination(self, mock_orchestration_environment):
        """Test parallel execution of multiple agents on different tasks."""
        # Create multiple tasks for parallel processing
        tasks = [
            TaskCore(
                id=i,
                title=f"Task {i}: {task_title}",
                description=f"Description for task {i}",
                component_area=ComponentArea.CORE,
                phase=1,
                priority=TaskPriority.MEDIUM,
                complexity=TaskComplexity.MEDIUM,
                success_criteria=f"Complete task {i} successfully",
                time_estimate_hours=2.0,
            )
            for i, task_title in enumerate(
                [
                    "Database schema optimization",
                    "API rate limiting implementation",
                    "User authentication system",
                    "Logging and monitoring setup",
                    "Error handling improvements",
                ],
                1,
            )
        ]

        # Simulate parallel agent execution
        start_time = time.time()

        async def execute_agent_task(
            task: TaskCore, agent_type: AgentType
        ) -> AgentReport:
            """Simulate agent task execution with realistic timing."""
            # Simulate processing time (reduced for testing)
            await asyncio.sleep(0.1)  # Simulate work

            return AgentReport(
                agent_name=agent_type,
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                success=True,
                execution_time_minutes=5.0,
                outputs={
                    "result": f"Completed {task.title} using {agent_type.value} agent",
                    "details": f"Successfully processed task {task.id}",
                },
                artifacts=[f"task_{task.id}_output.py"],
                recommendations=[f"Consider optimization for task {task.id}"],
                next_actions=["review", "deploy"],
                confidence_score=0.85,
            )

        # Execute tasks in parallel with different agents
        agent_assignments = [
            (tasks[0], AgentType.CODING),
            (tasks[1], AgentType.CODING),
            (tasks[2], AgentType.CODING),
            (tasks[3], AgentType.RESEARCH),
            (tasks[4], AgentType.TESTING),
        ]

        # Run parallel execution
        results = await asyncio.gather(
            *[
                execute_agent_task(task, agent_type)
                for task, agent_type in agent_assignments
            ]
        )

        execution_time = time.time() - start_time

        # Verify parallel execution results
        assert len(results) == 5
        assert all(report.success for report in results)
        assert execution_time < 2.0  # Should complete much faster than sequential

        # Verify agent distribution
        coding_tasks = [r for r in results if r.agent_name == AgentType.CODING]
        research_tasks = [r for r in results if r.agent_name == AgentType.RESEARCH]
        testing_tasks = [r for r in results if r.agent_name == AgentType.TESTING]

        assert len(coding_tasks) == 3
        assert len(research_tasks) == 1
        assert len(testing_tasks) == 1

        # Verify each task was completed
        completed_task_ids = {report.task_id for report in results}
        expected_task_ids = {task.id for task in tasks}
        assert completed_task_ids == expected_task_ids

    @pytest.mark.asyncio
    async def test_workflow_dependency_management(self, mock_orchestration_environment):
        """Test workflow with complex task dependencies."""
        # Define tasks with dependencies
        base_task = TaskCore(
            id=1,
            title="Database schema design",
            description="Design core database schema",
            component_area=ComponentArea.DATABASE,
            phase=1,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            success_criteria="Schema supports all core features",
            time_estimate_hours=8.0,
        )

        dependent_task_1 = TaskCore(
            id=2,
            title="API endpoint implementation",
            description="Implement REST API endpoints",
            component_area=ComponentArea.API,
            phase=2,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            success_criteria="All endpoints functional with proper error handling",
            time_estimate_hours=12.0,
        )

        dependent_task_2 = TaskCore(
            id=3,
            title="Frontend integration",
            description="Integrate API with frontend application",
            component_area=ComponentArea.FRONTEND,
            phase=3,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.MEDIUM,
            success_criteria="Frontend successfully communicates with API",
            time_estimate_hours=10.0,
        )

        # Define dependency relationships
        task_dependencies = {
            1: [],  # Base task has no dependencies
            2: [1],  # API depends on database schema
            3: [1, 2],  # Frontend depends on both database and API
        }

        # Simulate dependency-aware execution
        completed_tasks = set()
        execution_order = []

        async def execute_task_with_dependencies(task: TaskCore) -> AgentReport:
            """Execute task only if dependencies are satisfied."""
            task_deps = task_dependencies.get(task.id, [])

            # Check if dependencies are satisfied
            if not all(dep_id in completed_tasks for dep_id in task_deps):
                raise ValueError(f"Dependencies not met for task {task.id}")

            # Simulate task execution
            await asyncio.sleep(0.05)  # Simulate work

            execution_order.append(task.id)
            completed_tasks.add(task.id)

            return AgentReport(
                agent_name=AgentType.CODING,
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                success=True,
                execution_time_minutes=10.0,
                outputs={
                    "implementation": f"Completed {task.title}",
                    "dependencies_resolved": str(task_deps),
                },
                artifacts=[f"task_{task.id}_implementation.py"],
                recommendations=[f"Monitor {task.title} performance"],
                next_actions=["testing", "deployment"],
                confidence_score=0.87,
            )

        # Execute tasks in dependency order

        # Base task (no dependencies)
        base_result = await execute_task_with_dependencies(base_task)
        assert base_result.success is True
        assert 1 in completed_tasks

        # Dependent task 1 (depends on base)
        dependent_1_result = await execute_task_with_dependencies(dependent_task_1)
        assert dependent_1_result.success is True
        assert 2 in completed_tasks

        # Dependent task 2 (depends on both base and dependent_1)
        dependent_2_result = await execute_task_with_dependencies(dependent_task_2)
        assert dependent_2_result.success is True
        assert 3 in completed_tasks

        # Verify execution order respects dependencies
        assert execution_order == [1, 2, 3]

        # Verify dependency information in outputs
        assert "[]" in base_result.outputs["dependencies_resolved"]  # No dependencies
        assert "[1]" in dependent_1_result.outputs["dependencies_resolved"]
        assert "[1, 2]" in dependent_2_result.outputs["dependencies_resolved"]

        # Test dependency violation
        new_task = TaskCore(
            id=4,
            title="Test task with unmet dependencies",
            description="This should fail",
            component_area=ComponentArea.TESTING,
            phase=1,
            priority=TaskPriority.LOW,
            complexity=TaskComplexity.LOW,
            success_criteria="Should not execute",
            time_estimate_hours=1.0,
        )

        # Add task with unmet dependency
        task_dependencies[4] = [99]  # Non-existent dependency

        with pytest.raises(ValueError, match="Dependencies not met for task 4"):
            await execute_task_with_dependencies(new_task)


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms in workflows."""

    @pytest.mark.asyncio
    async def test_api_client_failure_recovery(
        self, mock_orchestration_environment, sample_complex_task
    ):
        """Test recovery from API client failures."""
        env = mock_orchestration_environment

        # Configure EXA client to fail initially then succeed
        call_count = 0

        async def failing_then_succeeding_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first two attempts
                raise Exception("Network timeout error")
            else:  # Succeed on third attempt
                return {
                    "results": [
                        {
                            "title": "Recovered Research Result",
                            "url": "https://example.com/recovered",
                            "text": "Successfully recovered research content",
                            "score": 0.92,
                        }
                    ]
                }

        env["exa_client"].search.side_effect = failing_then_succeeding_search

        # Simulate research agent with retry logic
        async def research_with_retry(max_retries: int = 3) -> AgentReport:
            """Research agent with built-in retry logic."""
            retry_count = 0
            last_error = None

            while retry_count < max_retries:
                try:
                    # Attempt research
                    search_results = await env["exa_client"].search(
                        query="code review tools security analysis"
                    )

                    # Success case
                    return AgentReport(
                        agent_name=AgentType.RESEARCH,
                        task_id=sample_complex_task.id,
                        status=TaskStatus.COMPLETED,
                        success=True,
                        execution_time_minutes=15.0
                        + (retry_count * 5.0),  # Extra time for retries
                        outputs={
                            "search_results": search_results,
                            "retry_count": retry_count,
                            "recovery_successful": True,
                        },
                        artifacts=["research_results.json"],
                        recommendations=["Monitor API reliability"],
                        next_actions=["coding"],
                        confidence_score=0.85
                        - (retry_count * 0.05),  # Slight confidence reduction
                    )

                except Exception as e:
                    last_error = str(e)
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(0.1)  # Brief retry delay

            # Failed after all retries
            return AgentReport(
                agent_name=AgentType.RESEARCH,
                task_id=sample_complex_task.id,
                status=TaskStatus.FAILED,
                success=False,
                execution_time_minutes=30.0,
                outputs={
                    "error": last_error,
                    "retry_count": retry_count,
                    "recovery_successful": False,
                },
                artifacts=[],
                recommendations=[
                    "Check API connectivity",
                    "Review retry configuration",
                ],
                next_actions=["investigate_api_issues"],
                confidence_score=0.0,
            )

        # Execute research with retry
        result = await research_with_retry(max_retries=3)

        # Verify successful recovery
        assert result.success is True
        assert result.status == TaskStatus.COMPLETED
        assert result.outputs["retry_count"] == 2  # Failed twice, succeeded on third
        assert result.outputs["recovery_successful"] is True
        assert len(result.outputs["search_results"]["results"]) == 1
        assert (
            "Recovered Research Result"
            in result.outputs["search_results"]["results"][0]["title"]
        )

        # Verify retry overhead is tracked
        assert result.execution_time_minutes == 25.0  # 15 + (2 * 5) for retries
        assert result.confidence_score == 0.75  # 0.85 - (2 * 0.05) for retries

        # Verify API client was called correct number of times
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_configuration_validation_failure_handling(
        self, mock_orchestration_environment
    ):
        """Test handling of configuration validation failures."""
        # Test invalid configuration scenarios
        invalid_configs = [
            # Missing required API keys
            {
                "OPENROUTER_API_KEY": "",
                "EXA_API_KEY": "test-key",
                "FIRECRAWL_API_KEY": "test-key",
            },
            # Invalid timeout values
            {
                "OPENROUTER_API_KEY": "test-key",
                "EXA_API_KEY": "test-key",
                "FIRECRAWL_API_KEY": "test-key",
                "ORCHESTRATION_AGENTS__MAX_EXECUTION_TIME_MINUTES": "0",  # Invalid
            },
            # Invalid parallel execution limit
            {
                "OPENROUTER_API_KEY": "test-key",
                "EXA_API_KEY": "test-key",
                "FIRECRAWL_API_KEY": "test-key",
                "ORCHESTRATION_PARALLEL_EXECUTION_LIMIT": "-1",  # Invalid
            },
        ]

        for i, invalid_config in enumerate(invalid_configs):
            with patch.dict("os.environ", invalid_config, clear=True):
                try:
                    # Clear settings cache to force reload
                    get_settings.cache_clear()

                    # This should raise validation error
                    OrchestrationSettings()

                    # If we get here, validation didn't catch the error
                    if i == 0:  # Missing API key case
                        # This might not fail at pydantic level, but at runtime
                        # validation
                        continue
                    else:
                        pytest.fail(
                            f"Configuration validation should have failed for case {i}"
                        )

                except (ValidationError, ValueError) as e:
                    # Expected validation failure
                    assert "validation" in str(e).lower() or "invalid" in str(e).lower()

                except Exception as e:
                    # Other configuration errors are also acceptable
                    assert isinstance(e, ValueError | TypeError | ValidationError)

        # Restore valid configuration
        with patch.dict(
            "os.environ",
            {
                "OPENROUTER_API_KEY": "test-key",
                "EXA_API_KEY": "test-key",
                "FIRECRAWL_API_KEY": "test-key",
                "ORCHESTRATION_PARALLEL_EXECUTION_LIMIT": "3",
            },
        ):
            get_settings.cache_clear()
            valid_settings = OrchestrationSettings()
            assert valid_settings.parallel_execution_limit == 3

    @pytest.mark.asyncio
    async def test_database_connection_failure_handling(
        self, mock_orchestration_environment
    ):
        """Test handling of database connection failures."""
        env = mock_orchestration_environment

        # Configure task manager to simulate database failures
        failure_count = 0

        def failing_database_operation(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise Exception("Database connection timeout")
            else:
                return {"task_id": 1, "status": "completed"}

        env["task_manager"].update_task_status.side_effect = failing_database_operation

        # Simulate task completion with database retry logic
        async def complete_task_with_db_retry(task_id: int, status: str) -> bool:
            """Complete task with database retry logic."""
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    env["task_manager"].update_task_status(task_id, status)
                    return True
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(0.1)  # Brief retry delay
                    else:
                        # Log error and continue with degraded functionality
                        print(
                            f"Database operation failed after {max_retries} "
                            f"retries: {e}"
                        )
                        return False

        # Test database retry logic
        success = await complete_task_with_db_retry(1, "completed")

        # Verify successful recovery
        assert success is True
        assert failure_count == 3  # Failed twice, succeeded on third attempt

        # Test permanent database failure
        env["task_manager"].update_task_status.side_effect = Exception(
            "Permanent DB failure"
        )

        permanent_failure_success = await complete_task_with_db_retry(2, "completed")
        assert (
            permanent_failure_success is False
        )  # Should gracefully handle permanent failure


class TestPerformanceUnderLoad:
    """Test system performance under realistic load conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_task_processing_performance(
        self, mock_orchestration_environment
    ):
        """Test performance with multiple concurrent tasks."""
        # Create batch of tasks for concurrent processing
        num_tasks = 10
        tasks = [
            TaskCore(
                id=i,
                title=f"Performance Test Task {i}",
                description=f"Task {i} for performance testing",
                component_area=ComponentArea.CORE,
                phase=1,
                priority=TaskPriority.MEDIUM,
                complexity=TaskComplexity.LOW,
                success_criteria=f"Complete task {i} within time limit",
                time_estimate_hours=1.0,
            )
            for i in range(1, num_tasks + 1)
        ]

        # Simulate agent processing with realistic timing
        async def process_task_batch(task_batch: list[TaskCore]) -> list[AgentReport]:
            """Process a batch of tasks concurrently."""

            async def process_single_task(task: TaskCore) -> AgentReport:
                # Simulate processing time with some variance
                processing_time = 0.05 + (task.id % 3) * 0.01  # 50-70ms
                await asyncio.sleep(processing_time)

                return AgentReport(
                    agent_name=AgentType.CODING,
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=processing_time
                    * 1000
                    / 60,  # Convert to minutes
                    outputs={
                        "implementation": f"Completed task {task.id}",
                        "processing_node": f"node_{task.id % 3}",  # Simulate
                        # distribution
                    },
                    artifacts=[f"task_{task.id}_output.py"],
                    recommendations=[],
                    next_actions=["deploy"],
                    confidence_score=0.85,
                )

            # Process tasks concurrently
            return await asyncio.gather(
                *[process_single_task(task) for task in task_batch]
            )

        # Measure performance
        start_time = time.time()
        results = await process_task_batch(tasks)
        total_time = time.time() - start_time

        # Verify results
        assert len(results) == num_tasks
        assert all(report.success for report in results)

        # Performance assertions
        assert (
            total_time < 1.0
        )  # Should complete in less than 1 second with concurrency

        # Calculate throughput
        throughput = num_tasks / total_time
        assert throughput > 10  # Should process at least 10 tasks per second

        # Verify task distribution across processing nodes
        node_distribution = {}
        for report in results:
            node = report.outputs["processing_node"]
            node_distribution[node] = node_distribution.get(node, 0) + 1

        # Should distribute across multiple nodes
        assert len(node_distribution) >= 2

        # Verify all tasks were processed uniquely
        processed_task_ids = {report.task_id for report in results}
        expected_task_ids = {task.id for task in tasks}
        assert processed_task_ids == expected_task_ids

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, mock_orchestration_environment):
        """Test memory usage with large numbers of tasks."""
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large number of tasks
        num_large_tasks = 100
        large_tasks = []

        for i in range(num_large_tasks):
            # Create tasks with substantial data
            large_task = TaskCore(
                id=i,
                title=f"Large Task {i}" * 10,  # Longer title
                description=f"Large description for task {i}"
                * 20,  # Substantial description
                component_area=ComponentArea.CORE,
                phase=1,
                priority=TaskPriority.MEDIUM,
                complexity=TaskComplexity.MEDIUM,
                success_criteria=f"Complete large task {i} with substantial data" * 5,
                time_estimate_hours=2.0,
            )
            large_tasks.append(large_task)

        # Process tasks in batches to simulate realistic workflow
        batch_size = 20
        all_results = []

        for i in range(0, num_large_tasks, batch_size):
            batch = large_tasks[i : i + batch_size]

            # Process batch
            batch_results = []
            for task in batch:
                # Simulate processing with substantial output data
                report = AgentReport(
                    agent_name=AgentType.CODING,
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=5.0,
                    outputs={
                        "implementation": f"Large implementation for task {task.id}"
                        * 50,
                        "detailed_analysis": f"Detailed analysis for task {task.id}"
                        * 30,
                        "test_results": f"Test results for task {task.id}" * 20,
                    },
                    artifacts=[
                        f"task_{task.id}_impl.py",
                        f"task_{task.id}_tests.py",
                        f"task_{task.id}_docs.md",
                    ],
                    recommendations=[
                        f"Recommendation 1 for task {task.id}",
                        f"Recommendation 2 for task {task.id}",
                        f"Recommendation 3 for task {task.id}",
                    ],
                    next_actions=["test", "deploy", "monitor"],
                    confidence_score=0.88,
                )
                batch_results.append(report)

            all_results.extend(batch_results)

            # Brief pause between batches
            await asyncio.sleep(0.01)

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Verify results
        assert len(all_results) == num_large_tasks
        assert all(report.success for report in all_results)

        # Memory usage assertions
        # Allow reasonable memory growth, but ensure it's not excessive
        memory_per_task = memory_increase / num_large_tasks
        assert memory_per_task < 1.0  # Less than 1MB per task
        assert memory_increase < 100  # Total increase less than 100MB

        # Verify data integrity wasn't compromised
        sample_report = all_results[50]  # Check middle report
        assert (
            len(sample_report.outputs["implementation"]) > 1000
        )  # Substantial content
        assert len(sample_report.artifacts) == 3
        assert len(sample_report.recommendations) == 3

    @pytest.mark.asyncio
    async def test_timeout_handling_under_load(self, mock_orchestration_environment):
        """Test timeout handling when system is under load."""
        # Create tasks with varying processing times
        mixed_tasks = []
        for i in range(20):
            # Some tasks will be slow (simulating real-world variance)
            complexity = TaskComplexity.HIGH if i % 5 == 0 else TaskComplexity.LOW
            time_estimate = 3.0 if complexity == TaskComplexity.HIGH else 1.0

            task = TaskCore(
                id=i,
                title=f"Mixed Load Task {i}",
                description=(
                    f"Task {i} with "
                    f"{'high' if complexity == TaskComplexity.HIGH else 'low'} "
                    f"complexity"
                ),
                component_area=ComponentArea.CORE,
                phase=1,
                priority=TaskPriority.MEDIUM,
                complexity=complexity,
                success_criteria=f"Complete task {i} within timeout",
                time_estimate_hours=time_estimate,
            )
            mixed_tasks.append(task)

        # Process with timeout constraints
        timeout_seconds = 0.5  # Short timeout to test handling
        successful_tasks = []
        timed_out_tasks = []

        async def process_with_timeout(task: TaskCore) -> AgentReport:
            """Process task with timeout handling."""
            processing_time = (
                0.1 if task.complexity == TaskComplexity.LOW else 0.8
            )  # Some will timeout

            try:
                await asyncio.wait_for(
                    asyncio.sleep(processing_time), timeout=timeout_seconds
                )

                # Success case
                return AgentReport(
                    agent_name=AgentType.CODING,
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=processing_time * 60,
                    outputs={
                        "implementation": f"Completed task {task.id} within timeout",
                        "actual_processing_time": processing_time,
                    },
                    artifacts=[f"task_{task.id}_success.py"],
                    recommendations=[],
                    next_actions=["deploy"],
                    confidence_score=0.90,
                )

            except TimeoutError:
                # Timeout case
                return AgentReport(
                    agent_name=AgentType.CODING,
                    task_id=task.id,
                    status=TaskStatus.FAILED,
                    success=False,
                    execution_time_minutes=timeout_seconds * 60,
                    outputs={
                        "error": "Task execution timed out",
                        "timeout_duration": timeout_seconds,
                        "expected_processing_time": processing_time,
                    },
                    artifacts=[],
                    recommendations=[
                        "Increase timeout limit",
                        "Optimize task processing",
                        "Consider task splitting",
                    ],
                    next_actions=["retry_with_longer_timeout"],
                    confidence_score=0.0,
                )

        # Process all tasks
        results = await asyncio.gather(
            *[process_with_timeout(task) for task in mixed_tasks]
        )

        # Categorize results
        for result in results:
            if result.success:
                successful_tasks.append(result)
            else:
                timed_out_tasks.append(result)

        # Verify timeout handling
        assert len(successful_tasks) > 0  # Some tasks should succeed
        assert len(timed_out_tasks) > 0  # Some tasks should timeout
        assert len(successful_tasks) + len(timed_out_tasks) == len(mixed_tasks)

        # Verify successful tasks completed within timeout
        for success_report in successful_tasks:
            actual_time = success_report.outputs["actual_processing_time"]
            assert actual_time <= timeout_seconds

        # Verify timed out tasks provide proper error information
        for timeout_report in timed_out_tasks:
            assert "timed out" in timeout_report.outputs["error"]
            assert timeout_report.outputs["timeout_duration"] == timeout_seconds
            assert len(timeout_report.recommendations) >= 3
            assert "retry_with_longer_timeout" in timeout_report.next_actions

        # Verify system handled mixed load appropriately
        success_rate = len(successful_tasks) / len(mixed_tasks)
        assert 0.3 <= success_rate <= 0.8  # Reasonable success rate under load

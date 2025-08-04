"""Integration tests for agent workflows with mocked services.

This module tests complete agent workflows including:
- Multi-agent collaboration scenarios
- Service integration with timeout handling
- Async operation coordination
- Error propagation and recovery
- Performance and concurrency testing
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.coding_agent import CodingAgent
from src.core.agent_protocol import AgentExecutionError
from src.schemas.unified_models import TaskStatus


class TestAgentWorkflowIntegration:
    """Test complete agent workflow integration scenarios."""

    @pytest.mark.asyncio
    async def test_research_to_coding_workflow(
        self, mock_research_agent, agent_state_with_research_context
    ):
        """Test workflow from research agent to coding agent."""
        # Test research agent execution
        research_state = {
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Research authentication methods",
                "description": "Investigate secure login approaches",
                "component_area": "security",
            },
            "messages": [],
            "agent_outputs": {},
        }

        research_result = await mock_research_agent.execute_task(research_state)

        # Verify research agent output
        assert "agent_outputs" in research_result
        assert "research" in research_result["agent_outputs"]
        research_output = research_result["agent_outputs"]["research"]
        assert research_output["status"] == TaskStatus.COMPLETED

        # Create coding agent with research context
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            coding_agent = CodingAgent()

            # Mock coding response that references research findings
            mock_response = MagicMock()
            mock_response.content = """
            # Authentication Implementation
            
            Based on research findings about JWT tokens and secure authentication:
            
            ```python
            # auth.py
            import jwt
            import bcrypt
            
            def authenticate_user(username: str, password: str) -> bool:
                # Implementation based on research recommendations
                return validate_credentials(username, password)
            
            def create_jwt_token(user_id: str) -> str:
                # JWT implementation as recommended by research
                return jwt.encode({"user_id": user_id}, "secret", algorithm="HS256")
            ```
            
            ## Design Decisions
            - Implemented JWT tokens as research suggested
            - Used bcrypt for password hashing per security findings
            
            ## Dependencies
            - jwt
            - bcrypt
            
            ## Integration Notes
            - Configure JWT secret securely
            - Ensure HTTPS for token transmission
            """

            coding_agent.openrouter_client.ainvoke = AsyncMock(
                return_value=mock_response
            )

            # Execute coding task with research context
            coding_state = research_result.copy()
            coding_state["task_data"] = {
                "id": 2,
                "title": "Implement authentication system",
                "description": "Create secure login based on research",
                "component_area": "security",
            }

            coding_result = await coding_agent.execute_task(coding_state)

            # Verify coding agent used research context
            assert "agent_outputs" in coding_result
            assert "coding" in coding_result["agent_outputs"]
            coding_output = coding_result["agent_outputs"]["coding"]
            assert coding_output["status"] == "completed"

            # Verify research context was incorporated
            implementation = coding_output["output"]
            assert "jwt" in implementation["dependencies"]
            assert "bcrypt" in implementation["dependencies"]
            assert any(
                "JWT" in decision for decision in implementation["design_decisions"]
            )

    @pytest.mark.asyncio
    async def test_coding_to_testing_workflow(
        self, mock_testing_agent, agent_state_with_coding_context
    ):
        """Test workflow from coding agent to testing agent."""
        # Execute testing task with coding context
        testing_result = await mock_testing_agent.execute_task(
            agent_state_with_coding_context
        )

        # Verify testing agent output
        assert "agent_outputs" in testing_result
        assert "testing" in testing_result["agent_outputs"]
        testing_output = testing_result["agent_outputs"]["testing"]
        assert testing_output["status"] == "completed"

        # Verify tests were generated based on coding context
        output = testing_output["output"]
        assert output["testing_type"] == "comprehensive_test_suite"
        assert len(output["test_files"]) > 0
        assert len(output["test_categories"]) > 0

    @pytest.mark.asyncio
    async def test_full_agent_pipeline(
        self, agent_integration_environment, sample_task_data
    ):
        """Test complete pipeline: Research -> Coding -> Testing -> Documentation."""
        # Initialize state
        initial_state = {
            "task_id": 1,
            "task_data": sample_task_data,
            "messages": [],
            "agent_outputs": {},
        }

        # Step 1: Research Agent
        research_agent = agent_integration_environment["research_agent"]
        research_task_data = sample_task_data.copy()
        research_task_data["title"] = "Research authentication methods"
        research_state = initial_state.copy()
        research_state["task_data"] = research_task_data

        research_result = await research_agent.execute_task(research_state)
        assert "research" in research_result["agent_outputs"]

        # Step 2: Coding Agent
        coding_agent = agent_integration_environment["coding_agent"]
        coding_task_data = sample_task_data.copy()
        coding_task_data["title"] = "Implement authentication system"
        coding_state = research_result.copy()
        coding_state["task_data"] = coding_task_data

        coding_result = await coding_agent.execute_task(coding_state)
        assert "coding" in coding_result["agent_outputs"]

        # Step 3: Testing Agent
        testing_agent = agent_integration_environment["testing_agent"]
        testing_task_data = sample_task_data.copy()
        testing_task_data["title"] = "Test authentication system"
        testing_state = coding_result.copy()
        testing_state["task_data"] = testing_task_data

        testing_result = await testing_agent.execute_task(testing_state)
        assert "testing" in testing_result["agent_outputs"]

        # Step 4: Documentation Agent
        documentation_agent = agent_integration_environment["documentation_agent"]
        documentation_task_data = sample_task_data.copy()
        documentation_task_data["title"] = "Document authentication system"
        documentation_state = testing_result.copy()
        documentation_state["task_data"] = documentation_task_data

        documentation_result = await documentation_agent.execute_task(
            documentation_state
        )
        assert "documentation" in documentation_result["agent_outputs"]

        # Verify final state has all agent outputs
        final_outputs = documentation_result["agent_outputs"]
        assert len(final_outputs) == 4
        assert all(
            output["status"] in ["completed", TaskStatus.COMPLETED]
            for output in final_outputs.values()
        )


class TestAsyncOperationsAndTimeouts:
    """Test async operations and timeout scenarios."""

    @pytest.mark.asyncio
    async def test_agent_timeout_handling(self, mock_coding_agent):
        """Test agent behavior with timeout scenarios."""

        # Mock a slow operation
        async def slow_operation():
            await asyncio.sleep(0.2)  # Simulate slow API response
            mock_response = MagicMock()
            mock_response.content = "Slow response"
            return mock_response

        mock_coding_agent.openrouter_client.ainvoke = AsyncMock(
            side_effect=slow_operation
        )

        state = {
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Implement feature",
                "description": "Code implementation with timeout",
            },
            "messages": [],
            "agent_outputs": {},
        }

        # Test with timeout
        start_time = asyncio.get_event_loop().time()

        try:
            # Set a very short timeout for testing
            await asyncio.wait_for(mock_coding_agent.execute_task(state), timeout=0.1)
        except TimeoutError:
            # Expected timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            assert elapsed < 0.15  # Should timeout before slow operation completes
        else:
            pytest.fail("Expected timeout error")

    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self, agent_integration_environment):
        """Test concurrent execution of multiple agents."""
        # Create different tasks for each agent
        tasks = [
            {
                "agent": agent_integration_environment["research_agent"],
                "state": {
                    "task_id": 1,
                    "task_data": {
                        "id": 1,
                        "title": "Research topic A",
                        "description": "Research task",
                    },
                    "messages": [],
                    "agent_outputs": {},
                },
            },
            {
                "agent": agent_integration_environment["coding_agent"],
                "state": {
                    "task_id": 2,
                    "task_data": {
                        "id": 2,
                        "title": "Implement feature B",
                        "description": "Coding task",
                    },
                    "messages": [],
                    "agent_outputs": {},
                },
            },
            {
                "agent": agent_integration_environment["testing_agent"],
                "state": {
                    "task_id": 3,
                    "task_data": {
                        "id": 3,
                        "title": "Test feature C",
                        "description": "Testing task",
                    },
                    "messages": [],
                    "agent_outputs": {},
                },
            },
        ]

        # Execute agents concurrently
        start_time = asyncio.get_event_loop().time()

        async def execute_agent_task(task_info):
            return await task_info["agent"].execute_task(task_info["state"])

        results = await asyncio.gather(
            *[execute_agent_task(task) for task in tasks], return_exceptions=True
        )

        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time

        # Verify all tasks completed
        assert len(results) == 3
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Agent {i} failed with exception: {result}")

            # Each result should have agent outputs
            assert "agent_outputs" in result

        # Concurrent execution should be faster than sequential
        # (This is approximate since we're using mocks)
        assert execution_time < 1.0  # Should complete quickly with mocks

    @pytest.mark.asyncio
    async def test_agent_retry_mechanisms(self, mock_research_agent):
        """Test agent retry mechanisms on failures."""
        call_count = 0

        async def failing_then_succeeding_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count <= 2:  # Fail first 2 attempts
                raise Exception(f"Attempt {call_count} failed")

            # Succeed on 3rd attempt
            mock_result = MagicMock()
            mock_result.title = "Success Result"
            mock_result.url = "https://example.com/success"
            mock_result.text = "Successful search result"
            mock_result.score = 0.9

            mock_response = MagicMock()
            mock_response.results = [mock_result]
            return mock_response

        # Mock the search to fail then succeed
        mock_research_agent.exa_client.search = AsyncMock(
            side_effect=failing_then_succeeding_search
        )

        state = {
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Research with retry",
                "description": "Test retry mechanism",
            },
            "messages": [],
            "agent_outputs": {},
        }

        # Execute should eventually succeed after retries
        result = await mock_research_agent.execute_task(state)

        # Verify the task completed successfully
        assert "agent_outputs" in result
        assert "research" in result["agent_outputs"]
        research_output = result["agent_outputs"]["research"]
        assert research_output["status"] == TaskStatus.COMPLETED

        # Verify retry attempts were made
        assert call_count >= 3  # Should have attempted at least 3 times


class TestErrorPropagationAndRecovery:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_agent_error_propagation(self, mock_coding_agent):
        """Test how errors propagate through agent execution."""
        # Mock API error
        mock_coding_agent.openrouter_client.ainvoke = AsyncMock(
            side_effect=Exception("Critical API Error")
        )

        state = {
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Implement feature",
                "description": "This will fail",
            },
            "messages": [],
            "agent_outputs": {},
        }

        # Should raise AgentExecutionError
        with pytest.raises(AgentExecutionError) as exc_info:
            await mock_coding_agent.execute_task(state)

        # Verify error details
        assert exc_info.value.agent_name == "coding"
        assert exc_info.value.task_id == 1
        assert "Critical API Error" in str(exc_info.value)

        # Verify error was recorded in state
        assert "agent_outputs" in state
        assert "coding" in state["agent_outputs"]
        error_output = state["agent_outputs"]["coding"]
        assert error_output["status"] == "failed"

    @pytest.mark.asyncio
    async def test_partial_workflow_failure_recovery(
        self, agent_integration_environment, sample_task_data
    ):
        """Test recovery from partial workflow failures."""
        # Start with successful research
        research_agent = agent_integration_environment["research_agent"]
        state = {
            "task_id": 1,
            "task_data": sample_task_data.copy(),
            "messages": [],
            "agent_outputs": {},
        }

        research_result = await research_agent.execute_task(state)
        assert "research" in research_result["agent_outputs"]

        # Make coding agent fail
        coding_agent = agent_integration_environment["coding_agent"]
        coding_agent.openrouter_client.ainvoke = AsyncMock(
            side_effect=Exception("Coding failure")
        )

        # Attempt coding (should fail)
        with pytest.raises(AgentExecutionError):
            await coding_agent.execute_task(research_result)

        # Verify research results are preserved despite coding failure
        assert "research" in research_result["agent_outputs"]
        assert (
            research_result["agent_outputs"]["research"]["status"]
            == TaskStatus.COMPLETED
        )

        # Recovery: Fix coding agent and retry
        coding_agent.openrouter_client.ainvoke = AsyncMock(
            return_value=MagicMock(content="Fixed implementation")
        )

        # Should now succeed
        coding_result = await coding_agent.execute_task(research_result)
        assert "coding" in coding_result["agent_outputs"]
        assert coding_result["agent_outputs"]["coding"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_network_error_handling(self, mock_research_agent):
        """Test handling of network-related errors."""
        # Mock network error
        mock_research_agent.exa_client.search = AsyncMock(
            side_effect=Exception("Network connection failed")
        )

        state = {
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Research with network error",
                "description": "This will encounter network issues",
            },
            "messages": [],
            "agent_outputs": {},
        }

        # Should handle network error gracefully
        with pytest.raises(AgentExecutionError) as exc_info:
            await mock_research_agent.execute_task(state)

        assert "Network connection failed" in str(exc_info.value)

        # Verify error context is preserved
        assert "agent_outputs" in state
        error_output = state["agent_outputs"]["research"]
        assert error_output["status"] == TaskStatus.FAILED


class TestServiceIntegrationScenarios:
    """Test integration with external services."""

    @pytest.mark.asyncio
    async def test_exa_service_integration(self, mock_research_agent):
        """Test research agent integration with Exa service."""
        # Configure detailed Exa mock response
        mock_result = MagicMock()
        mock_result.title = "Advanced Authentication Patterns"
        mock_result.url = "https://auth-patterns.com/advanced"
        mock_result.text = (
            "Advanced patterns for secure authentication including OAuth 2.0, "
            "SAML, and multi-factor authentication."
        )
        mock_result.score = 0.92
        mock_result.summary = "Comprehensive authentication patterns guide"

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        mock_research_agent.exa_client.search = AsyncMock(return_value=mock_response)

        state = {
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Research advanced authentication",
                "description": "Investigate modern authentication patterns",
                "component_area": "security",
            },
            "messages": [],
            "agent_outputs": {},
        }

        result = await mock_research_agent.execute_task(state)

        # Verify Exa integration worked
        research_output = result["agent_outputs"]["research"]["outputs"]
        assert research_output["research_type"] == "web_search"
        assert len(research_output["sources_found"]) > 0

        source = research_output["sources_found"][0]
        assert source["title"] == "Advanced Authentication Patterns"
        assert source["relevance_score"] == 0.92

    @pytest.mark.asyncio
    async def test_openrouter_service_integration(self, mock_coding_agent):
        """Test coding agent integration with OpenRouter service."""
        # Configure detailed OpenRouter mock response
        mock_response = MagicMock()
        mock_response.content = """
        # Advanced Authentication Implementation
        
        Based on the requirements, here's a comprehensive authentication system:
        
        ```python
        # auth_service.py
        from typing import Optional
        import jwt
        import bcrypt
        from datetime import datetime, timedelta
        
        class AuthenticationService:
            def __init__(self, secret_key: str):
                self.secret_key = secret_key
            
            def authenticate_user(self, username: str, password: str) -> Optional[dict]:
                \"\"\"Authenticate user and return token if successful.\"\"\"
                if self._verify_credentials(username, password):
                    return self._create_token(username)
                return None
            
            def _verify_credentials(self, username: str, password: str) -> bool:
                # Implementation would verify against database
                return True
            
            def _create_token(self, username: str) -> dict:
                payload = {
                    'username': username,
                    'exp': datetime.utcnow() + timedelta(hours=24)
                }
                token = jwt.encode(payload, self.secret_key, algorithm='HS256')
                return {'token': token, 'expires_in': 86400}
        ```
        
        ## Design Decisions
        - Implemented class-based architecture for better organization
        - Used JWT with expiration for security
        - Separated credential verification from token creation
        - Added comprehensive type hints for better maintainability
        
        ## Dependencies
        - pyjwt>=2.0.0
        - bcrypt>=3.2.0
        - python>=3.8
        
        ## Integration Notes
        - Configure secret key through environment variables
        - Implement database integration for credential verification
        - Add rate limiting for production deployment
        - Consider implementing refresh token mechanism
        """

        mock_coding_agent.openrouter_client.ainvoke = AsyncMock(
            return_value=mock_response
        )

        state = {
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Implement advanced authentication",
                "description": "Create comprehensive authentication service",
                "component_area": "security",
            },
            "messages": [],
            "agent_outputs": {},
        }

        result = await mock_coding_agent.execute_task(state)

        # Verify OpenRouter integration worked
        coding_output = result["agent_outputs"]["coding"]["output"]
        assert coding_output["implementation_type"] == "code_generation"

        # Verify extracted components
        assert "auth_service.py" in coding_output["files_created"]
        assert "pyjwt" in coding_output["dependencies"]
        assert "bcrypt" in coding_output["dependencies"]
        assert len(coding_output["design_decisions"]) > 0
        assert len(coding_output["integration_notes"]) > 0

    @pytest.mark.asyncio
    async def test_service_unavailable_scenarios(self, mock_research_agent):
        """Test handling of service unavailable scenarios."""
        # Mock service unavailable error
        mock_research_agent.exa_client.search = AsyncMock(
            side_effect=Exception("Service temporarily unavailable")
        )

        state = {
            "task_id": 1,
            "task_data": {
                "id": 1,
                "title": "Research during outage",
                "description": "Test service unavailable handling",
            },
            "messages": [],
            "agent_outputs": {},
        }

        # Should handle service unavailability
        with pytest.raises(AgentExecutionError) as exc_info:
            await mock_research_agent.execute_task(state)

        assert "Service temporarily unavailable" in str(exc_info.value)

        # Verify graceful error handling
        assert "agent_outputs" in state
        error_output = state["agent_outputs"]["research"]
        assert error_output["status"] == TaskStatus.FAILED
        assert "Service temporarily unavailable" in error_output["blocking_issues"][0]


@pytest.mark.asyncio
async def test_end_to_end_workflow_performance(
    agent_integration_environment, sample_task_data
):
    """Test end-to-end workflow performance characteristics."""
    start_time = asyncio.get_event_loop().time()

    # Execute full pipeline
    agents = [
        ("research", "Research authentication methods"),
        ("coding", "Implement authentication system"),
        ("testing", "Test authentication system"),
        ("documentation", "Document authentication system"),
    ]

    state = {
        "task_id": 1,
        "task_data": sample_task_data.copy(),
        "messages": [],
        "agent_outputs": {},
    }

    for agent_name, task_title in agents:
        agent = agent_integration_environment[f"{agent_name}_agent"]
        task_data = sample_task_data.copy()
        task_data["title"] = task_title
        state["task_data"] = task_data

        agent_start = asyncio.get_event_loop().time()
        state = await agent.execute_task(state)
        agent_end = asyncio.get_event_loop().time()

        # Verify each agent completed successfully
        assert agent_name in state["agent_outputs"]
        assert state["agent_outputs"][agent_name]["status"] in [
            "completed",
            TaskStatus.COMPLETED,
        ]

        # Each agent should complete reasonably quickly (with mocks)
        agent_duration = agent_end - agent_start
        assert agent_duration < 1.0  # Should be fast with mocks

    end_time = asyncio.get_event_loop().time()
    total_duration = end_time - start_time

    # Total workflow should complete in reasonable time
    assert total_duration < 5.0  # Full pipeline should be fast with mocks

    # Verify all agents contributed to final state
    assert len(state["agent_outputs"]) == 4
    final_outputs = state["agent_outputs"]

    # Each agent should have proper output structure
    for agent_name in ["research", "coding", "testing", "documentation"]:
        assert agent_name in final_outputs
        output = final_outputs[agent_name]
        assert "status" in output
        assert "output" in output or "outputs" in output

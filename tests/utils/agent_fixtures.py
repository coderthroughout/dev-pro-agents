"""Agent-specific fixtures and mocks for external API clients.

This module provides specialized test fixtures for each agent type,
including comprehensive mocks for OpenRouter, Exa, and Firecrawl APIs.
"""

import asyncio
import os
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.coding_agent import CodingAgent
from src.agents.documentation_agent import DocumentationAgent
from src.agents.research_agent import ResearchAgent
from src.agents.testing_agent import TestingAgent


# ============================================================================
# OPENROUTER API MOCKS
# ============================================================================


@pytest.fixture
def mock_openrouter_success_response() -> MagicMock:
    """Create mock successful OpenRouter API response."""
    mock_response = MagicMock()
    mock_response.content = """
    # Implementation Response
    
    ```python
    def example_function():
        # Example implementation
        return "success"
    ```
    
    ## Design Decisions
    - Used simple approach for clarity
    - Focused on maintainability
    
    ## Dependencies
    - fastapi
    - pydantic
    
    ## Integration Notes
    - Add to main application
    - Configure environment variables
    """
    return mock_response


@pytest.fixture
def mock_openrouter_error_response():
    """Create mock OpenRouter API error response."""

    def _create_error(error_message: str = "API Error") -> Exception:
        return Exception(error_message)

    return _create_error


@pytest.fixture
def mock_openrouter_client(mock_openrouter_success_response: MagicMock) -> AsyncMock:
    """Create mock OpenRouter client with default success responses."""
    mock_client = AsyncMock()
    mock_client.ainvoke.return_value = mock_openrouter_success_response
    return mock_client


@pytest.fixture
def openrouter_test_responses():
    """Create factory for generating OpenRouter test responses."""

    def _create_response(response_type: str, **kwargs: Any) -> MagicMock:
        mock_response = MagicMock()

        if response_type == "coding":
            mock_response.content = f"""
            # {kwargs.get("title", "Code Implementation")}
            
            ```python
            # {kwargs.get("filename", "main.py")}
            {kwargs.get("code", "def example(): pass")}
            ```
            
            ## Design Decisions
            {kwargs.get("decisions", "- Used standard patterns")}
            
            ## Dependencies
            {kwargs.get("dependencies", "- python>=3.8")}
            
            ## Integration Notes
            {kwargs.get("notes", "- Standard integration")}
            """

        elif response_type == "testing":
            mock_response.content = f"""
            # {kwargs.get("title", "Test Suite")}
            
            ```python
            # {kwargs.get("test_file", "test_main.py")}
            import pytest
            
            {kwargs.get("test_code", "def test_example(): assert True")}
            ```
            
            ## Unit Tests
            {kwargs.get("unit_tests", "Basic unit test coverage")}
            
            ## Integration Tests
            {kwargs.get("integration_tests", "End-to-end testing")}
            
            ## Test Coverage
            {kwargs.get("coverage", "Target 90% coverage")}
            
            ## Test Data
            {kwargs.get("test_data", "Standard test fixtures")}
            """

        elif response_type == "documentation":
            mock_response.content = f"""
            # {kwargs.get("title", "Project Documentation")}
            
            ## Overview and Purpose
            {kwargs.get("overview", "Project overview and goals")}
            
            ## Implementation Details
            {kwargs.get("implementation", "Technical implementation details")}
            
            ```python
            {kwargs.get("code_example", "def example(): pass")}
            ```
            
            ## Usage Instructions
            {kwargs.get("usage", "1. Install dependencies\\n2. Run application")}
            
            ## API Documentation
            {kwargs.get("api_docs", "API endpoint documentation")}
            
            ## Testing Information
            {kwargs.get("testing_info", "How to run tests")}
            
            ## Troubleshooting Guide
            {kwargs.get("troubleshooting", "Common issues and solutions")}
            
            Create file: {kwargs.get("readme", "README.md")}
            See also: {kwargs.get("reference", "https://example.com/docs")}
            """

        else:
            mock_response.content = kwargs.get("content", "Generic response")

        return mock_response

    return _create_response


# ============================================================================
# EXA API MOCKS
# ============================================================================


@pytest.fixture
def mock_exa_search_result():
    """Mock Exa search result with realistic data."""
    result = MagicMock()
    result.title = "Authentication Best Practices"
    result.url = "https://example.com/auth-guide"
    result.published_date = datetime.now()
    result.author = "Security Expert"
    result.score = 0.95
    result.id = "exa-result-123"
    result.image = "https://example.com/auth-image.jpg"
    result.favicon = "https://example.com/favicon.ico"
    result.text = (
        "JWT tokens provide secure stateless authentication. "
        "Use HTTPS for transmission. Implement proper token expiration."
    )
    result.highlights = ["JWT tokens", "secure authentication", "HTTPS transmission"]
    result.highlight_scores = [0.95, 0.90, 0.85]
    result.summary = "Comprehensive guide to implementing secure authentication systems"
    return result


@pytest.fixture
def mock_exa_search_response(mock_exa_search_result):
    """Mock Exa search response with multiple results."""
    response = MagicMock()
    response.request_id = "req-123"
    response.resolved_search_type = "neural"
    response.results = [mock_exa_search_result]
    response.search_type = "neural"
    response.context = "Authentication and security research"
    response.cost_dollars = {"search": 0.01, "content": 0.005}
    return response


@pytest.fixture
def mock_exa_client(mock_exa_search_response):
    """Mock Exa client with comprehensive API coverage."""
    mock_client = AsyncMock()

    # Search method
    mock_client.search.return_value = mock_exa_search_response

    # Get contents method
    mock_client.get_contents.return_value = mock_exa_search_response.results

    # Find similar method
    mock_client.find_similar.return_value = mock_exa_search_response.results

    # Answer method
    mock_client.answer.return_value = {
        "answer": "JWT tokens are recommended for stateless authentication",
        "sources": ["https://example.com/jwt-guide"],
        "confidence": 0.9,
    }

    # Research methods
    mock_research_task = MagicMock()
    mock_research_task.id = "research-task-123"
    mock_research_task.status = "completed"
    mock_research_task.result = {
        "findings": ["JWT is secure", "Use HTTPS", "Implement rate limiting"],
        "sources": ["https://example.com/source1", "https://example.com/source2"],
    }

    mock_client.create_research_task.return_value = mock_research_task
    mock_client.get_research_task.return_value = mock_research_task
    mock_client.wait_for_research_task.return_value = mock_research_task
    mock_client.research.return_value = mock_research_task.result

    return mock_client


@pytest.fixture
def exa_test_scenarios():
    """Create factory for generating various Exa test scenarios."""

    def _create_scenario(scenario_type: str, **kwargs: Any) -> dict[str, Any]:
        scenarios = {
            "authentication_research": {
                "results": [
                    {
                        "title": "JWT Authentication Guide",
                        "url": "https://example.com/jwt",
                        "text": "JWT tokens provide secure authentication...",
                        "score": 0.95,
                    },
                    {
                        "title": "OAuth 2.0 Security",
                        "url": "https://example.com/oauth",
                        "text": "OAuth 2.0 provides secure authorization...",
                        "score": 0.88,
                    },
                ],
                "queries": ["JWT authentication", "OAuth security", "token validation"],
            },
            "api_research": {
                "results": [
                    {
                        "title": "REST API Best Practices",
                        "url": "https://example.com/rest-api",
                        "text": "RESTful APIs should follow standard HTTP methods...",
                        "score": 0.92,
                    }
                ],
                "queries": ["REST API design", "API security", "HTTP methods"],
            },
            "empty_results": {"results": [], "queries": ["obscure technical topic"]},
        }

        if scenario_type in scenarios:
            return scenarios[scenario_type]
        return kwargs

    return _create_scenario


# ============================================================================
# FIRECRAWL API MOCKS
# ============================================================================


@pytest.fixture
def mock_firecrawl_document():
    """Mock Firecrawl document with realistic content."""
    document = MagicMock()
    document.markdown = """
# Authentication Guide

## Overview
This guide covers secure authentication implementation.

## JWT Tokens
JWT (JSON Web Tokens) provide stateless authentication.

## Best Practices
- Use HTTPS for all authentication endpoints
- Implement proper token expiration
- Store secrets securely
    """
    document.html = """
<h1>Authentication Guide</h1>
<h2>Overview</h2>
<p>This guide covers secure authentication implementation.</p>
<h2>JWT Tokens</h2>
<p>JWT (JSON Web Tokens) provide stateless authentication.</p>
    """
    document.metadata = {
        "title": "Authentication Guide",
        "description": "Comprehensive authentication guide",
        "url": "https://example.com/auth-guide",
    }
    return document


@pytest.fixture
def mock_firecrawl_response(mock_firecrawl_document):
    """Mock Firecrawl scrape response."""
    response = MagicMock()
    response.success = True
    response.data = mock_firecrawl_document
    response.credits_used = 1
    response.status_code = 200
    return response


@pytest.fixture
def mock_firecrawl_client(mock_firecrawl_response):
    """Mock Firecrawl client with comprehensive API coverage."""
    mock_client = AsyncMock()

    # Scrape method
    mock_client.scrape.return_value = mock_firecrawl_response

    # Crawl method
    mock_crawl_result = MagicMock()
    mock_crawl_result.success = True
    mock_crawl_result.data = [mock_firecrawl_response.data]
    mock_client.crawl.return_value = mock_crawl_result

    # Search method
    mock_search_result = MagicMock()
    mock_search_result.success = True
    mock_search_result.data = [mock_firecrawl_response.data]
    mock_client.search.return_value = mock_search_result

    return mock_client


# ============================================================================
# AGENT-SPECIFIC FIXTURES
# ============================================================================


@pytest.fixture
def mock_coding_agent():
    """Mock CodingAgent with default behavior."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = CodingAgent()

        # Mock the OpenRouter client
        mock_response = MagicMock()
        mock_response.content = """
        # Code Implementation
        
        ```python
        # main.py
        def main():
            return "Hello, World!"
        ```
        
        ## Design Decisions
        - Simple implementation for clarity
        
        ## Dependencies
        - python>=3.8
        
        ## Integration Notes
        - Standard Python application
        """

        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)
        return agent


@pytest.fixture
def mock_research_agent(mock_exa_client, mock_firecrawl_client):
    """Mock ResearchAgent with default behavior."""
    agent = ResearchAgent()
    agent.exa_client = mock_exa_client
    agent.firecrawl_client = mock_firecrawl_client
    return agent


@pytest.fixture
def mock_testing_agent():
    """Mock TestingAgent with default behavior."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = TestingAgent()

        # Mock the OpenRouter client
        mock_response = MagicMock()
        mock_response.content = """
        # Test Suite
        
        ```python
        # test_main.py
        import pytest
        
        def test_main():
            assert main() == "Hello, World!"
        ```
        
        ## Unit Tests
        Basic functionality tests
        
        ## Integration Tests
        End-to-end workflow tests
        
        ## Test Coverage
        Target 95% code coverage
        
        ## Test Data
        Standard test fixtures
        """

        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)
        return agent


@pytest.fixture
def mock_documentation_agent():
    """Mock DocumentationAgent with default behavior."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = DocumentationAgent()

        # Mock the OpenRouter client
        mock_response = MagicMock()
        mock_response.content = """
        # Project Documentation
        
        ## Overview and Purpose
        This project provides example functionality.
        
        ## Implementation Details
        Built with Python and modern best practices.
        
        ```python
        def example():
            return "documented"
        ```
        
        ## Usage Instructions
        1. Install dependencies
        2. Run the application
        
        ## API Documentation
        Complete API reference available.
        
        ## Testing Information
        Run tests with pytest.
        
        ## Troubleshooting Guide
        Common issues and solutions.
        
        Create file: README.md
        """

        agent.openrouter_client.ainvoke = AsyncMock(return_value=mock_response)
        return agent


# ============================================================================
# AGENT STATE AND CONTEXT FIXTURES
# ============================================================================


@pytest.fixture
def agent_state_with_research_context():
    """Agent state with research context from ResearchAgent."""
    return {
        "task_id": 1,
        "task_data": {
            "id": 1,
            "title": "Implement authentication",
            "description": "Create secure login system",
            "component_area": "security",
        },
        "messages": [],
        "agent_outputs": {
            "research": {
                "output": {
                    "research_type": "web_search",
                    "key_findings": [
                        "JWT tokens provide stateless authentication",
                        "Use HTTPS for secure transmission",
                        "Implement proper token expiration",
                    ],
                    "sources_found": [
                        {
                            "title": "JWT Guide",
                            "url": "https://example.com/jwt",
                            "relevance_score": 0.95,
                        }
                    ],
                },
                "status": "completed",
            }
        },
    }


@pytest.fixture
def agent_state_with_coding_context():
    """Agent state with coding context from CodingAgent."""
    return {
        "task_id": 1,
        "task_data": {
            "id": 1,
            "title": "Test authentication system",
            "description": "Create tests for login functionality",
        },
        "messages": [],
        "agent_outputs": {
            "coding": {
                "output": {
                    "implementation_type": "code_generation",
                    "files_created": ["auth.py", "models.py"],
                    "design_decisions": [
                        "Used JWT for tokens",
                        "Implemented bcrypt hashing",
                    ],
                    "content": """
                    # auth.py
                    import jwt
                    
                    def authenticate(username, password):
                        return validate_credentials(username, password)
                    """,
                },
                "status": "completed",
            }
        },
    }


@pytest.fixture
def agent_state_with_all_contexts():
    """Agent state with context from all agents."""
    return {
        "task_id": 1,
        "task_data": {
            "id": 1,
            "title": "Document authentication system",
            "description": "Create comprehensive documentation",
        },
        "messages": [],
        "agent_outputs": {
            "research": {
                "output": {
                    "key_findings": ["JWT best practices", "Security considerations"],
                    "sources_found": [
                        {"title": "Security Guide", "url": "https://example.com"}
                    ],
                },
                "status": "completed",
            },
            "coding": {
                "output": {
                    "files_created": ["auth.py"],
                    "design_decisions": ["JWT implementation"],
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


# ============================================================================
# ERROR SCENARIO FIXTURES
# ============================================================================


@pytest.fixture
def api_error_scenarios():
    """Create factory for generating API error scenarios."""

    def _create_error(error_type: str, **kwargs):
        if error_type == "timeout":
            return TimeoutError("Request timed out")
        elif error_type == "rate_limit":
            return Exception("Rate limit exceeded")
        elif error_type == "auth_error":
            return Exception("Authentication failed")
        elif error_type == "network_error":
            return Exception("Network connection failed")
        elif error_type == "api_unavailable":
            return Exception("Service temporarily unavailable")
        else:
            return Exception(kwargs.get("message", "Unknown error"))

    return _create_error


@pytest.fixture
def agent_failure_scenarios():
    """Create factory for generating agent failure scenarios."""

    def _create_failure(failure_type: str, **kwargs) -> dict[str, Any]:
        if failure_type == "validation_failure":
            return {"can_handle": False, "reason": "Task validation failed"}
        elif failure_type == "execution_failure":
            return {
                "exception": Exception("Task execution failed"),
                "error_context": {"error_type": "execution"},
            }
        elif failure_type == "timeout_failure":
            return {
                "exception": TimeoutError("Agent timeout"),
                "error_context": {"error_type": "timeout"},
            }
        else:
            return kwargs

    return _create_failure


# ============================================================================
# INTEGRATION TEST FIXTURES
# ============================================================================


@pytest.fixture
def agent_integration_environment(
    mock_coding_agent, mock_research_agent, mock_testing_agent, mock_documentation_agent
):
    """Complete agent integration environment."""
    return {
        "coding_agent": mock_coding_agent,
        "research_agent": mock_research_agent,
        "testing_agent": mock_testing_agent,
        "documentation_agent": mock_documentation_agent,
    }


@pytest.fixture
def async_operation_scenarios():
    """Create factory for testing async operation scenarios."""

    async def _create_async_scenario(scenario_type: str, **kwargs):
        if scenario_type == "slow_operation":
            await asyncio.sleep(kwargs.get("delay", 0.1))
            return "Slow operation completed"
        elif scenario_type == "timeout_operation":
            await asyncio.sleep(kwargs.get("timeout", 1.0))
            return "Should not reach here"
        elif scenario_type == "concurrent_operations":
            tasks = []
            for _i in range(kwargs.get("count", 3)):
                tasks.append(asyncio.create_task(asyncio.sleep(0.01)))
            await asyncio.gather(*tasks)
            return "Concurrent operations completed"
        else:
            return kwargs.get("result", "Default result")

    return _create_async_scenario


# ============================================================================
# PERFORMANCE AND MONITORING FIXTURES
# ============================================================================


@pytest.fixture
def agent_performance_monitor():
    """Monitor agent performance during tests."""
    import time

    metrics = {
        "start_time": None,
        "end_time": None,
        "execution_time": None,
        "api_calls": 0,
        "errors": 0,
    }

    def start_monitoring():
        metrics["start_time"] = time.time()
        metrics["api_calls"] = 0
        metrics["errors"] = 0

    def stop_monitoring():
        metrics["end_time"] = time.time()
        metrics["execution_time"] = metrics["end_time"] - metrics["start_time"]

    def record_api_call():
        metrics["api_calls"] += 1

    def record_error():
        metrics["errors"] += 1

    def get_metrics():
        return metrics.copy()

    monitor = MagicMock()
    monitor.start = start_monitoring
    monitor.stop = stop_monitoring
    monitor.record_api_call = record_api_call
    monitor.record_error = record_error
    monitor.get_metrics = get_metrics

    return monitor

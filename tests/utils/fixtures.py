"""Comprehensive test fixtures for configuration scenarios and performance benchmarks.

This module provides reusable fixtures for:
- Configuration scenarios and validation testing
- Performance benchmarking and load testing
- Mock environments and external service simulation
- Task and workflow testing scenarios
- Error simulation and resilience testing
"""

import asyncio
import os
import time
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from src.schemas.unified_models import (
    AgentReport,
    AgentType,
    ComponentArea,
    TaskComplexity,
    TaskCore,
    TaskPriority,
    TaskStatus,
)


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def base_environment_variables():
    """Base environment variables for testing."""
    return {
        "OPENAI_API_KEY": "test-openai-key-123",
        "OPENROUTER_API_KEY": "test-openrouter-key-456",
        "EXA_API_KEY": "test-exa-key-789",
        "FIRECRAWL_API_KEY": "test-firecrawl-key-012",
        "DATABASE_URL": "sqlite:///test_database.db",
        "REDIS_URL": "redis://localhost:6379/1",
        "LOG_LEVEL": "INFO",
        "ENVIRONMENT": "test",
    }


@pytest.fixture
def orchestration_environment_variables():
    """Extended orchestration-specific environment variables."""
    return {
        "ORCHESTRATION_PARALLEL_EXECUTION_LIMIT": "8",
        "ORCHESTRATION_BATCH_SIZE": "20",
        "ORCHESTRATION_TIMEOUT_MINUTES": "30",
        "ORCHESTRATION_RETRY_ATTEMPTS": "3",
        "ORCHESTRATION_DEBUG_MODE": "true",
        "ORCHESTRATION_ENABLE_METRICS": "true",
        "ORCHESTRATION_MAX_MEMORY_MB": "1024",
        "ORCHESTRATION_ENABLE_CACHING": "false",
    }


@pytest.fixture
def agent_environment_variables():
    """Agent-specific environment variables."""
    return {
        "OPENAI_MODEL": "gpt-4o-mini",
        "OPENAI_MAX_TOKENS": "4000",
        "OPENAI_TEMPERATURE": "0.7",
        "OPENROUTER_MODEL": "meta-llama/llama-3.1-8b-instruct:free",
        "OPENROUTER_MAX_TOKENS": "8000",
        "OPENROUTER_TEMPERATURE": "0.3",
        "EXA_MAX_RESULTS": "10",
        "EXA_USE_AUTOPROMPT": "true",
        "FIRECRAWL_MAX_PAGES": "5",
        "FIRECRAWL_EXTRACT_MAIN_CONTENT": "true",
    }


@pytest.fixture
def supervisor_environment_variables():
    """Supervisor-specific environment variables."""
    return {
        "SUPERVISOR_MAX_PARALLEL_AGENTS": "4",
        "SUPERVISOR_COORDINATION_TIMEOUT": "600",
        "SUPERVISOR_ENABLE_MONITORING": "true",
        "SUPERVISOR_DECISION_THRESHOLD": "0.7",
        "SUPERVISOR_FALLBACK_AGENT": "coding",
        "SUPERVISOR_ENABLE_AGENT_HEALTH_CHECKS": "true",
    }


@pytest.fixture
def complete_test_environment(
    base_environment_variables,
    orchestration_environment_variables,
    agent_environment_variables,
    supervisor_environment_variables,
):
    """Complete test environment with all configuration variables."""
    env_vars = {}
    env_vars.update(base_environment_variables)
    env_vars.update(orchestration_environment_variables)
    env_vars.update(agent_environment_variables)
    env_vars.update(supervisor_environment_variables)

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def minimal_test_environment(base_environment_variables):
    """Minimal test environment with only required variables."""
    with patch.dict(os.environ, base_environment_variables):
        yield base_environment_variables


@pytest.fixture
def configuration_scenarios():
    """Factory for creating various configuration test scenarios."""

    def _create_config_scenario(scenario_type: str, **kwargs) -> dict[str, Any]:
        """Create configuration scenario based on type."""
        if scenario_type == "valid_complete":
            return {
                "environment": {
                    "OPENAI_API_KEY": "test-valid-openai",
                    "OPENROUTER_API_KEY": "test-valid-openrouter",
                    "EXA_API_KEY": "test-valid-exa",
                    "FIRECRAWL_API_KEY": "test-valid-firecrawl",
                    "ORCHESTRATION_PARALLEL_EXECUTION_LIMIT": "10",
                    "ORCHESTRATION_BATCH_SIZE": "50",
                },
                "expected_values": {
                    "parallel_execution_limit": 10,
                    "batch_size": 50,
                    "openai": {"api_key": "test-valid-openai"},
                    "openrouter": {"api_key": "test-valid-openrouter"},
                    "exa": {"api_key": "test-valid-exa"},
                    "firecrawl": {"api_key": "test-valid-firecrawl"},
                },
            }

        elif scenario_type == "missing_required":
            return {
                "environment": {
                    "OPENAI_API_KEY": "test-openai",
                    # Missing other required keys
                },
                "expected_errors": [
                    "OPENROUTER_API_KEY",
                    "EXA_API_KEY",
                    "FIRECRAWL_API_KEY",
                ],
            }

        elif scenario_type == "invalid_values":
            return {
                "environment": {
                    "OPENAI_API_KEY": "test-openai",
                    "OPENROUTER_API_KEY": "test-openrouter",
                    "EXA_API_KEY": "test-exa",
                    "FIRECRAWL_API_KEY": "test-firecrawl",
                    "ORCHESTRATION_PARALLEL_EXECUTION_LIMIT": "-5",  # Invalid
                    "ORCHESTRATION_BATCH_SIZE": "0",  # Invalid
                    "OPENAI_MAX_TOKENS": "not-a-number",  # Invalid
                },
                "expected_errors": [
                    "parallel_execution_limit must be positive",
                    "batch_size must be positive",
                    "max_tokens must be a valid integer",
                ],
            }

        elif scenario_type == "edge_values":
            return {
                "environment": {
                    "OPENAI_API_KEY": "test-openai",
                    "OPENROUTER_API_KEY": "test-openrouter",
                    "EXA_API_KEY": "test-exa",
                    "FIRECRAWL_API_KEY": "test-firecrawl",
                    "ORCHESTRATION_PARALLEL_EXECUTION_LIMIT": "1",  # Minimum
                    "ORCHESTRATION_BATCH_SIZE": "1000",  # Large
                    "OPENAI_MAX_TOKENS": "128000",  # Maximum
                    "OPENAI_TEMPERATURE": "2.0",  # Maximum
                },
                "expected_values": {
                    "parallel_execution_limit": 1,
                    "batch_size": 1000,
                    "openai": {"max_tokens": 128000, "temperature": 2.0},
                },
            }

        elif scenario_type == "cross_validation":
            return {
                "environment": {
                    "OPENAI_API_KEY": "test-openai",
                    "OPENROUTER_API_KEY": "test-openrouter",
                    "EXA_API_KEY": "test-exa",
                    "FIRECRAWL_API_KEY": "test-firecrawl",
                    "ORCHESTRATION_PARALLEL_EXECUTION_LIMIT": "20",
                    "SUPERVISOR_MAX_PARALLEL_AGENTS": "15",  # Should be <= parallel_execution_limit
                },
                "expected_adjustments": {
                    "parallel_execution_limit": 15,  # Adjusted to supervisor limit
                },
            }

        else:
            return kwargs

    return _create_config_scenario


# ============================================================================
# TASK AND WORKFLOW FIXTURES
# ============================================================================


@pytest.fixture
def sample_tasks():
    """Collection of sample tasks for testing."""
    return [
        TaskCore(
            id=1,
            title="Implement user authentication",
            description="Create secure user authentication system with JWT tokens",
            component_area=ComponentArea.SECURITY,
            phase=1,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            success_criteria="Working JWT authentication with proper validation",
            time_estimate_hours=6.0,
        ),
        TaskCore(
            id=2,
            title="Create user registration API",
            description="Build REST API endpoint for user registration",
            component_area=ComponentArea.API,
            phase=1,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            success_criteria="Registration endpoint with validation",
            time_estimate_hours=4.0,
        ),
        TaskCore(
            id=3,
            title="Write authentication tests",
            description="Create comprehensive test suite for authentication",
            component_area=ComponentArea.TESTING,
            phase=2,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.LOW,
            success_criteria="90%+ test coverage for auth system",
            time_estimate_hours=5.0,
            parent_task_id=1,
        ),
        TaskCore(
            id=4,
            title="Document authentication API",
            description="Create comprehensive API documentation",
            component_area=ComponentArea.DOCUMENTATION,
            phase=3,
            priority=TaskPriority.LOW,
            complexity=TaskComplexity.LOW,
            success_criteria="Complete API documentation with examples",
            time_estimate_hours=2.0,
            parent_task_id=1,
        ),
    ]


@pytest.fixture
def complex_task_scenario():
    """Complex multi-phase task scenario with dependencies."""
    return [
        # Phase 1: Research and Planning
        TaskCore(
            id=1,
            title="Research database design patterns",
            description="Investigate modern database design and optimization patterns",
            component_area=ComponentArea.DATABASE,
            phase=1,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            time_estimate_hours=8.0,
        ),
        TaskCore(
            id=2,
            title="Design system architecture",
            description="Create comprehensive system architecture blueprint",
            component_area=ComponentArea.ARCHITECTURE,
            phase=1,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.HIGH,
            time_estimate_hours=12.0,
        ),
        # Phase 2: Core Implementation
        TaskCore(
            id=3,
            title="Implement database models",
            description="Create database models and relationships",
            component_area=ComponentArea.DATABASE,
            phase=2,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            time_estimate_hours=10.0,
            parent_task_id=1,
        ),
        TaskCore(
            id=4,
            title="Build API endpoints",
            description="Create REST API endpoints for core functionality",
            component_area=ComponentArea.API,
            phase=2,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.HIGH,
            time_estimate_hours=15.0,
            parent_task_id=2,
        ),
        TaskCore(
            id=5,
            title="Implement business logic",
            description="Create core business logic and validation",
            component_area=ComponentArea.CORE,
            phase=2,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.HIGH,
            time_estimate_hours=20.0,
            parent_task_id=2,
        ),
        # Phase 3: Testing and Quality Assurance
        TaskCore(
            id=6,
            title="Create unit tests",
            description="Build comprehensive unit test suite",
            component_area=ComponentArea.TESTING,
            phase=3,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            time_estimate_hours=12.0,
            parent_task_id=5,
        ),
        TaskCore(
            id=7,
            title="Create integration tests",
            description="Build end-to-end integration tests",
            component_area=ComponentArea.TESTING,
            phase=3,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.HIGH,
            time_estimate_hours=8.0,
            parent_task_id=4,
        ),
        TaskCore(
            id=8,
            title="Performance testing",
            description="Conduct performance and load testing",
            component_area=ComponentArea.TESTING,
            phase=3,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.MEDIUM,
            time_estimate_hours=6.0,
            parent_task_id=4,
        ),
        # Phase 4: Documentation and Deployment
        TaskCore(
            id=9,
            title="Create technical documentation",
            description="Write comprehensive technical documentation",
            component_area=ComponentArea.DOCUMENTATION,
            phase=4,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.LOW,
            time_estimate_hours=8.0,
            parent_task_id=2,
        ),
        TaskCore(
            id=10,
            title="Setup deployment pipeline",
            description="Configure CI/CD and deployment infrastructure",
            component_area=ComponentArea.DEVOPS,
            phase=4,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.MEDIUM,
            time_estimate_hours=10.0,
            parent_task_id=7,
        ),
    ]


@pytest.fixture
def task_factory():
    """Factory for creating custom tasks with specified parameters."""

    def _create_task(
        task_id: int = 1,
        title: str = "Test Task",
        description: str = "Test task description",
        component_area: ComponentArea = ComponentArea.CORE,
        phase: int = 1,
        priority: TaskPriority = TaskPriority.MEDIUM,
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        time_estimate_hours: float = 2.0,
        parent_task_id: int | None = None,
        **kwargs,
    ) -> TaskCore:
        """Create a task with specified parameters."""
        return TaskCore(
            id=task_id,
            title=title,
            description=description,
            component_area=component_area,
            phase=phase,
            priority=priority,
            complexity=complexity,
            success_criteria=kwargs.get(
                "success_criteria", f"Complete {title.lower()}"
            ),
            time_estimate_hours=time_estimate_hours,
            parent_task_id=parent_task_id,
            created_at=kwargs.get("created_at", datetime.now()),
            updated_at=kwargs.get("updated_at"),
        )

    return _create_task


@pytest.fixture
def workflow_scenarios():
    """Factory for creating various workflow test scenarios."""

    def _create_workflow_scenario(scenario_type: str, **kwargs) -> dict[str, Any]:
        """Create workflow scenario based on type."""
        if scenario_type == "simple_linear":
            return {
                "tasks": [
                    {"id": 1, "title": "Task 1", "dependencies": []},
                    {"id": 2, "title": "Task 2", "dependencies": [1]},
                    {"id": 3, "title": "Task 3", "dependencies": [2]},
                ],
                "expected_execution_order": [1, 2, 3],
                "expected_phases": 3,
            }

        elif scenario_type == "parallel_execution":
            return {
                "tasks": [
                    {"id": 1, "title": "Root Task", "dependencies": []},
                    {"id": 2, "title": "Parallel Task A", "dependencies": [1]},
                    {"id": 3, "title": "Parallel Task B", "dependencies": [1]},
                    {"id": 4, "title": "Merge Task", "dependencies": [2, 3]},
                ],
                "expected_execution_order": [
                    1,
                    [2, 3],
                    4,
                ],  # Tasks 2,3 can run in parallel
                "expected_phases": 3,
            }

        elif scenario_type == "complex_dependencies":
            return {
                "tasks": [
                    {"id": 1, "title": "Foundation", "dependencies": []},
                    {"id": 2, "title": "Module A", "dependencies": [1]},
                    {"id": 3, "title": "Module B", "dependencies": [1]},
                    {"id": 4, "title": "Integration AB", "dependencies": [2, 3]},
                    {"id": 5, "title": "Module C", "dependencies": [2]},
                    {"id": 6, "title": "Final Integration", "dependencies": [4, 5]},
                ],
                "expected_execution_order": [1, [2, 3], [4, 5], 6],
                "expected_phases": 4,
            }

        elif scenario_type == "error_recovery":
            return {
                "tasks": [
                    {
                        "id": 1,
                        "title": "Task 1",
                        "dependencies": [],
                        "fail_probability": 0.0,
                    },
                    {
                        "id": 2,
                        "title": "Failing Task",
                        "dependencies": [1],
                        "fail_probability": 1.0,
                    },
                    {
                        "id": 3,
                        "title": "Recovery Task",
                        "dependencies": [1],
                        "fail_probability": 0.0,
                    },
                    {
                        "id": 4,
                        "title": "Final Task",
                        "dependencies": [3],
                        "fail_probability": 0.0,
                    },
                ],
                "expected_recovery": True,
                "expected_final_success": True,
            }

        else:
            return kwargs

    return _create_workflow_scenario


# ============================================================================
# MOCK AGENT AND SERVICE FIXTURES
# ============================================================================


@pytest.fixture
def mock_agent_reports():
    """Collection of mock agent reports for different scenarios."""
    return {
        "research_success": AgentReport(
            agent_name=AgentType.RESEARCH,
            task_id=1,
            status=TaskStatus.COMPLETED,
            success=True,
            execution_time_minutes=3.5,
            outputs={
                "research_findings": [
                    "JWT tokens provide stateless authentication",
                    "HTTPS is essential for secure token transmission",
                    "Token expiration should be configurable",
                ],
                "sources": [
                    "https://example.com/jwt-guide",
                    "https://example.com/security-best-practices",
                ],
                "confidence_score": 0.92,
            },
            artifacts=["research_report.md", "security_analysis.json"],
            recommendations=[
                "Use refresh tokens for long-lived sessions",
                "Implement rate limiting on auth endpoints",
            ],
            next_actions=["implement_jwt_authentication"],
            confidence_score=0.92,
        ),
        "coding_success": AgentReport(
            agent_name=AgentType.CODING,
            task_id=2,
            status=TaskStatus.COMPLETED,
            success=True,
            execution_time_minutes=8.2,
            outputs={
                "implementation": "Complete JWT authentication system",
                "files_created": ["auth.py", "models.py", "middleware.py"],
                "design_decisions": [
                    "Used PyJWT library for token handling",
                    "Implemented RSA256 signing algorithm",
                    "Added refresh token support",
                ],
                "code_quality_score": 8.5,
            },
            artifacts=[
                "src/auth.py",
                "src/models.py",
                "src/middleware.py",
                "requirements.txt",
            ],
            recommendations=[
                "Add input validation decorators",
                "Consider implementing token blacklisting",
            ],
            next_actions=["create_unit_tests", "setup_integration_tests"],
            confidence_score=0.88,
        ),
        "testing_success": AgentReport(
            agent_name=AgentType.TESTING,
            task_id=3,
            status=TaskStatus.COMPLETED,
            success=True,
            execution_time_minutes=5.1,
            outputs={
                "test_files": ["test_auth.py", "test_models.py", "test_integration.py"],
                "test_count": 47,
                "coverage_report": {
                    "total_coverage": 94.2,
                    "line_coverage": 96.1,
                    "function_coverage": 100.0,
                    "missing_lines": ["auth.py:145-148"],
                },
                "test_categories": ["unit", "integration", "security"],
            },
            artifacts=[
                "tests/test_auth.py",
                "tests/test_models.py",
                "tests/test_integration.py",
                "coverage_report.html",
            ],
            recommendations=[
                "Add edge case tests for token expiration",
                "Include load testing for auth endpoints",
            ],
            next_actions=["run_security_tests", "setup_ci_pipeline"],
            confidence_score=0.95,
        ),
        "documentation_success": AgentReport(
            agent_name=AgentType.DOCUMENTATION,
            task_id=4,
            status=TaskStatus.COMPLETED,
            success=True,
            execution_time_minutes=2.8,
            outputs={
                "documentation_files": ["README.md", "API.md", "DEPLOYMENT.md"],
                "sections_created": [
                    "installation",
                    "authentication_api",
                    "security_considerations",
                    "troubleshooting",
                ],
                "api_endpoints_documented": 8,
                "code_examples": 15,
            },
            artifacts=[
                "docs/README.md",
                "docs/API.md",
                "docs/DEPLOYMENT.md",
                "examples/auth_examples.py",
            ],
            recommendations=[
                "Add interactive API documentation",
                "Include video tutorials for setup",
            ],
            next_actions=["review_documentation", "publish_docs"],
            confidence_score=0.91,
        ),
        "agent_failure": AgentReport(
            agent_name=AgentType.CODING,
            task_id=5,
            status=TaskStatus.FAILED,
            success=False,
            execution_time_minutes=1.2,
            outputs={
                "error_message": "Failed to generate secure implementation",
                "error_type": "ValidationError",
                "attempted_solutions": [
                    "Tried basic JWT implementation",
                    "Attempted to add security headers",
                ],
                "partial_results": {
                    "code_snippets": ["import jwt", "def authenticate():"]
                },
            },
            artifacts=[],
            recommendations=[
                "Review security requirements",
                "Consider using established auth library",
            ],
            next_actions=["retry_with_different_approach"],
            confidence_score=0.15,
        ),
    }


@pytest.fixture
def mock_external_services():
    """Mock external services (EXA, Firecrawl, OpenRouter) with realistic responses."""

    class MockExternalServices:
        def __init__(self):
            self.exa_client = AsyncMock()
            self.firecrawl_client = AsyncMock()
            self.openrouter_client = AsyncMock()
            self._setup_mocks()

        def _setup_mocks(self):
            """Setup default mock responses."""
            # EXA search results
            self.exa_client.search.return_value = {
                "results": [
                    {
                        "title": "JWT Authentication Best Practices",
                        "url": "https://example.com/jwt-best-practices",
                        "text": "JSON Web Tokens (JWT) are a secure way to transmit information...",
                        "score": 0.95,
                        "published_date": "2024-01-15",
                        "author": "Security Expert",
                    },
                    {
                        "title": "Modern Authentication Patterns",
                        "url": "https://example.com/auth-patterns",
                        "text": "Modern web applications require robust authentication systems...",
                        "score": 0.87,
                        "published_date": "2024-02-20",
                        "author": "Web Developer",
                    },
                ]
            }

            # Firecrawl scraping results
            self.firecrawl_client.scrape.return_value = {
                "success": True,
                "data": {
                    "markdown": """# JWT Authentication Guide

## Overview
JWT (JSON Web Tokens) provide a secure, stateless way to handle authentication.

## Implementation Steps
1. Generate secret key
2. Create token signing function
3. Implement token validation
4. Add middleware for protected routes

## Security Considerations
- Use HTTPS only
- Set appropriate expiration times
- Implement refresh token rotation
""",
                    "title": "JWT Authentication Guide",
                    "description": "Complete guide to implementing JWT authentication",
                    "url": "https://example.com/jwt-guide",
                },
                "credits_used": 1,
            }

            # OpenRouter LLM responses
            self.openrouter_client.ainvoke.return_value = Mock(
                content="""# Authentication Implementation

```python
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from passlib.context import CryptContext

class AuthenticationManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.pwd_context = CryptContext(schemes=["bcrypt"])
    
    def create_token(self, user_id: str) -> str:
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
```

## Design Decisions
- Used bcrypt for password hashing
- Implemented 24-hour token expiration
- Added comprehensive error handling

## Dependencies
- PyJWT for token handling
- passlib for password hashing
- fastapi for web framework

## Integration Notes
- Add to main FastAPI application
- Configure secret key via environment variable
- Implement refresh token mechanism for production
"""
            )

        def configure_exa_scenario(self, scenario: str):
            """Configure EXA client for specific test scenarios."""
            if scenario == "empty_results":
                self.exa_client.search.return_value = {"results": []}
            elif scenario == "api_error":
                self.exa_client.search.side_effect = Exception("EXA API Error")
            elif scenario == "rate_limited":
                self.exa_client.search.side_effect = Exception("Rate limit exceeded")

        def configure_firecrawl_scenario(self, scenario: str):
            """Configure Firecrawl client for specific test scenarios."""
            if scenario == "scrape_failure":
                self.firecrawl_client.scrape.return_value = {
                    "success": False,
                    "error": "Failed to scrape URL",
                }
            elif scenario == "network_error":
                self.firecrawl_client.scrape.side_effect = Exception("Network timeout")

        def configure_openrouter_scenario(self, scenario: str):
            """Configure OpenRouter client for specific test scenarios."""
            if scenario == "model_error":
                self.openrouter_client.ainvoke.side_effect = Exception(
                    "Model unavailable"
                )
            elif scenario == "low_quality_response":
                self.openrouter_client.ainvoke.return_value = Mock(
                    content="# Incomplete Implementation\n\nSorry, I cannot complete this task."
                )

    return MockExternalServices()


# ============================================================================
# PERFORMANCE TESTING FIXTURES
# ============================================================================


@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks and thresholds for testing."""
    return {
        "configuration_loading": {
            "max_time_seconds": 0.1,
            "memory_limit_mb": 10,
            "description": "Configuration loading should be fast and memory efficient",
        },
        "task_creation": {
            "max_time_seconds": 0.01,
            "memory_limit_mb": 5,
            "throughput_per_second": 1000,
            "description": "Task creation should handle high throughput",
        },
        "batch_processing": {
            "small_batch": {
                "size": 10,
                "max_time_seconds": 1.0,
                "memory_limit_mb": 50,
            },
            "medium_batch": {
                "size": 100,
                "max_time_seconds": 5.0,
                "memory_limit_mb": 200,
            },
            "large_batch": {
                "size": 1000,
                "max_time_seconds": 30.0,
                "memory_limit_mb": 500,
            },
        },
        "concurrent_operations": {
            "low_concurrency": {
                "concurrent_tasks": 5,
                "max_time_seconds": 2.0,
                "expected_speedup": 2.0,
            },
            "medium_concurrency": {
                "concurrent_tasks": 20,
                "max_time_seconds": 5.0,
                "expected_speedup": 5.0,
            },
            "high_concurrency": {
                "concurrent_tasks": 100,
                "max_time_seconds": 15.0,
                "expected_speedup": 10.0,
            },
        },
        "api_client_performance": {
            "request_timeout": 30.0,
            "max_retry_attempts": 3,
            "rate_limit_per_minute": 60,
            "concurrent_requests": 10,
        },
        "memory_usage": {
            "baseline_mb": 100,
            "max_increase_mb": 200,
            "cleanup_efficiency": 0.8,  # Should free 80% of allocated memory
        },
    }


@pytest.fixture
def load_testing_scenarios():
    """Factory for creating load testing scenarios."""

    def _create_load_scenario(scenario_type: str, **kwargs) -> dict[str, Any]:
        """Create load testing scenario based on type."""
        if scenario_type == "gradual_ramp_up":
            return {
                "phases": [
                    {"duration": 10, "concurrent_users": 1},
                    {"duration": 20, "concurrent_users": 5},
                    {"duration": 30, "concurrent_users": 10},
                    {"duration": 20, "concurrent_users": 20},
                    {"duration": 10, "concurrent_users": 5},
                ],
                "expected_behavior": "steady_performance",
                "success_criteria": {
                    "response_time_95th_percentile": 2.0,
                    "error_rate": 0.01,
                    "throughput_degradation": 0.2,
                },
            }

        elif scenario_type == "spike_test":
            return {
                "phases": [
                    {"duration": 30, "concurrent_users": 2},
                    {"duration": 10, "concurrent_users": 50},  # Sudden spike
                    {"duration": 30, "concurrent_users": 2},
                ],
                "expected_behavior": "graceful_degradation",
                "success_criteria": {
                    "recovery_time_seconds": 30,
                    "max_error_rate_during_spike": 0.1,
                    "post_spike_performance_recovery": 0.9,
                },
            }

        elif scenario_type == "endurance_test":
            return {
                "phases": [
                    {
                        "duration": 300,
                        "concurrent_users": 10,
                    },  # 5 minutes sustained load
                ],
                "expected_behavior": "stable_performance",
                "success_criteria": {
                    "memory_leak_detection": True,
                    "performance_degradation": 0.1,
                    "resource_cleanup": True,
                },
            }

        elif scenario_type == "stress_test":
            return {
                "phases": [
                    {"duration": 60, "concurrent_users": 100},  # High load
                    {"duration": 60, "concurrent_users": 200},  # Very high load
                ],
                "expected_behavior": "controlled_failure",
                "success_criteria": {
                    "breaking_point_identification": True,
                    "graceful_failure_mode": True,
                    "recovery_capability": True,
                },
            }

        else:
            return kwargs

    return _create_load_scenario


@pytest.fixture
def performance_monitor():
    """Performance monitoring utility for tests."""

    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            self.start_times = {}
            self.memory_samples = []

        def start_timing(self, operation: str):
            """Start timing an operation."""
            self.start_times[operation] = time.time()

        def end_timing(self, operation: str) -> float:
            """End timing and return duration."""
            if operation not in self.start_times:
                raise ValueError(f"No start time recorded for operation: {operation}")

            duration = time.time() - self.start_times[operation]

            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)

            return duration

        def sample_memory(self, label: str = "default"):
            """Sample current memory usage."""
            try:
                import psutil

                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                self.memory_samples.append(
                    {"label": label, "memory_mb": memory_mb, "timestamp": time.time()}
                )
                return memory_mb
            except ImportError:
                # Fallback if psutil not available
                return 0.0

        def get_statistics(self, operation: str) -> dict[str, float]:
            """Get statistics for an operation."""
            if operation not in self.metrics:
                return {}

            durations = self.metrics[operation]
            return {
                "count": len(durations),
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "total": sum(durations),
            }

        def get_memory_statistics(self, label: str = None) -> dict[str, float]:
            """Get memory usage statistics."""
            if label:
                samples = [
                    s["memory_mb"] for s in self.memory_samples if s["label"] == label
                ]
            else:
                samples = [s["memory_mb"] for s in self.memory_samples]

            if not samples:
                return {}

            return {
                "count": len(samples),
                "mean": sum(samples) / len(samples),
                "min": min(samples),
                "max": max(samples),
                "peak_usage": max(samples) - min(samples) if len(samples) > 1 else 0,
            }

        def assert_performance(
            self, operation: str, max_time: float, min_throughput: float = None
        ):
            """Assert performance requirements are met."""
            stats = self.get_statistics(operation)
            if not stats:
                raise AssertionError(f"No metrics recorded for operation: {operation}")

            assert stats["mean"] <= max_time, (
                f"Operation {operation} took {stats['mean']:.3f}s, expected <= {max_time}s"
            )

            if min_throughput and stats["count"] > 0:
                actual_throughput = stats["count"] / stats["total"]
                assert actual_throughput >= min_throughput, (
                    f"Throughput {actual_throughput:.2f}/s, expected >= {min_throughput}/s"
                )

        def reset(self):
            """Reset all metrics."""
            self.metrics.clear()
            self.start_times.clear()
            self.memory_samples.clear()

    return PerformanceMonitor()


# ============================================================================
# ERROR AND RESILIENCE TESTING FIXTURES
# ============================================================================


@pytest.fixture
def error_scenarios():
    """Factory for creating various error scenarios."""

    def _create_error_scenario(error_type: str, **kwargs):
        """Create error scenario based on type."""
        if error_type == "network_timeout":
            return {
                "exception": TimeoutError("Network request timed out"),
                "retry_behavior": "exponential_backoff",
                "max_retries": 3,
                "recovery_expected": True,
            }

        elif error_type == "rate_limit":
            return {
                "exception": Exception("Rate limit exceeded. Retry after 60 seconds"),
                "retry_behavior": "fixed_delay",
                "delay_seconds": 60,
                "recovery_expected": True,
            }

        elif error_type == "authentication_failure":
            return {
                "exception": Exception("Authentication failed: Invalid API key"),
                "retry_behavior": "no_retry",
                "recovery_expected": False,
                "requires_manual_intervention": True,
            }

        elif error_type == "service_unavailable":
            return {
                "exception": Exception("Service temporarily unavailable"),
                "retry_behavior": "circuit_breaker",
                "circuit_breaker_threshold": 5,
                "recovery_expected": True,
            }

        elif error_type == "data_corruption":
            return {
                "exception": ValueError("Data corruption detected in task payload"),
                "retry_behavior": "no_retry",
                "recovery_expected": False,
                "requires_data_cleanup": True,
            }

        elif error_type == "memory_exhaustion":
            return {
                "exception": MemoryError("Not enough memory to complete operation"),
                "retry_behavior": "resource_cleanup",
                "recovery_expected": True,
                "requires_cleanup": True,
            }

        else:
            return {
                "exception": Exception(kwargs.get("message", "Unknown error")),
                "retry_behavior": kwargs.get("retry_behavior", "exponential_backoff"),
                "recovery_expected": kwargs.get("recovery_expected", True),
            }

    return _create_error_scenario


@pytest.fixture
def resilience_testing_patterns():
    """Patterns for testing system resilience."""
    return {
        "circuit_breaker": {
            "failure_threshold": 5,
            "recovery_timeout": 30,
            "half_open_max_calls": 3,
            "test_sequence": [
                {"phase": "normal", "success_rate": 1.0, "duration": 10},
                {"phase": "degraded", "success_rate": 0.3, "duration": 30},
                {"phase": "circuit_open", "success_rate": 0.0, "duration": 30},
                {"phase": "half_open", "success_rate": 0.8, "duration": 10},
                {"phase": "recovered", "success_rate": 1.0, "duration": 20},
            ],
        },
        "retry_with_backoff": {
            "max_retries": 3,
            "initial_delay": 1.0,
            "backoff_multiplier": 2.0,
            "max_delay": 30.0,
            "jitter": True,
        },
        "graceful_degradation": {
            "degradation_levels": [
                {"level": 0, "features": ["full_functionality"], "performance": 1.0},
                {"level": 1, "features": ["core_functionality"], "performance": 0.8},
                {"level": 2, "features": ["basic_functionality"], "performance": 0.5},
                {"level": 3, "features": ["minimal_functionality"], "performance": 0.2},
            ],
            "recovery_thresholds": [0.9, 0.7, 0.5],
        },
        "bulkhead_isolation": {
            "resource_pools": {
                "critical": {"size": 5, "isolation": True},
                "important": {"size": 10, "isolation": True},
                "normal": {"size": 20, "isolation": False},
            },
            "failure_isolation": True,
        },
    }


# ============================================================================
# ASYNC AND CONCURRENCY FIXTURES
# ============================================================================


@pytest_asyncio.fixture
async def async_test_environment():
    """Async test environment with proper cleanup."""
    # Setup async resources
    loop = asyncio.get_event_loop()

    # Create mock async services
    mock_services = {
        "task_queue": asyncio.Queue(),
        "result_queue": asyncio.Queue(),
        "semaphore": asyncio.Semaphore(10),
        "event": asyncio.Event(),
    }

    try:
        yield mock_services
    finally:
        # Cleanup async resources
        while not mock_services["task_queue"].empty():
            try:
                mock_services["task_queue"].get_nowait()
            except asyncio.QueueEmpty:
                break

        while not mock_services["result_queue"].empty():
            try:
                mock_services["result_queue"].get_nowait()
            except asyncio.QueueEmpty:
                break


@pytest.fixture
def concurrency_test_patterns():
    """Patterns for testing concurrent operations."""
    return {
        "producer_consumer": {
            "producers": 3,
            "consumers": 2,
            "queue_size": 50,
            "items_per_producer": 20,
            "processing_delay": 0.01,
        },
        "scatter_gather": {
            "worker_count": 5,
            "task_count": 25,
            "aggregation_timeout": 10.0,
            "partial_results_acceptable": True,
        },
        "pipeline_processing": {
            "stages": [
                {"name": "input", "workers": 2, "delay": 0.01},
                {"name": "transform", "workers": 3, "delay": 0.02},
                {"name": "validate", "workers": 2, "delay": 0.015},
                {"name": "output", "workers": 1, "delay": 0.005},
            ],
            "buffer_size": 10,
        },
        "resource_contention": {
            "shared_resources": 3,
            "competing_tasks": 15,
            "acquisition_timeout": 5.0,
            "hold_time_range": (0.1, 0.5),
        },
    }


# ============================================================================
# DATABASE AND STORAGE FIXTURES
# ============================================================================


@pytest.fixture
def mock_database():
    """Mock database for testing database operations."""

    class MockDatabase:
        def __init__(self):
            self.tables = {}
            self.connection_count = 0
            self.max_connections = 10
            self.operation_log = []

        def connect(self):
            """Simulate database connection."""
            if self.connection_count >= self.max_connections:
                raise Exception("Connection pool exhausted")

            self.connection_count += 1
            self.operation_log.append(
                {"operation": "connect", "timestamp": time.time()}
            )
            return MockConnection(self)

        def disconnect(self):
            """Simulate database disconnection."""
            if self.connection_count > 0:
                self.connection_count -= 1
            self.operation_log.append(
                {"operation": "disconnect", "timestamp": time.time()}
            )

        def create_table(self, table_name: str, schema: dict[str, str]):
            """Create a table."""
            self.tables[table_name] = {"schema": schema, "data": []}
            self.operation_log.append(
                {
                    "operation": "create_table",
                    "table": table_name,
                    "timestamp": time.time(),
                }
            )

        def insert(self, table_name: str, data: dict[str, Any]):
            """Insert data into table."""
            if table_name not in self.tables:
                raise ValueError(f"Table {table_name} does not exist")

            self.tables[table_name]["data"].append(data)
            self.operation_log.append(
                {
                    "operation": "insert",
                    "table": table_name,
                    "timestamp": time.time(),
                }
            )

        def select(
            self, table_name: str, where: dict[str, Any] = None
        ) -> list[dict[str, Any]]:
            """Select data from table."""
            if table_name not in self.tables:
                raise ValueError(f"Table {table_name} does not exist")

            data = self.tables[table_name]["data"]

            if where:
                filtered_data = []
                for row in data:
                    match = True
                    for key, value in where.items():
                        if row.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_data.append(row)
                data = filtered_data

            self.operation_log.append(
                {
                    "operation": "select",
                    "table": table_name,
                    "timestamp": time.time(),
                }
            )

            return data

        def get_operation_stats(self) -> dict[str, Any]:
            """Get database operation statistics."""
            operations = [log["operation"] for log in self.operation_log]
            return {
                "total_operations": len(self.operation_log),
                "operation_counts": {
                    op: operations.count(op) for op in set(operations)
                },
                "current_connections": self.connection_count,
                "max_connections": self.max_connections,
            }

    class MockConnection:
        def __init__(self, db: MockDatabase):
            self.db = db
            self.closed = False

        def execute(self, query: str, params: tuple = ()):
            """Execute a query."""
            if self.closed:
                raise Exception("Connection is closed")

            # Simulate query execution time
            time.sleep(0.001)
            return {"query": query, "params": params, "executed": True}

        def close(self):
            """Close connection."""
            if not self.closed:
                self.db.disconnect()
                self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()

    return MockDatabase()


# ============================================================================
# UTILITY FIXTURES
# ============================================================================


@pytest.fixture
def temp_config_files(tmp_path):
    """Create temporary configuration files for testing."""
    # Create agents.yaml
    agents_config = {
        "research_agent": {
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "max_tokens": 8000,
            "temperature": 0.3,
            "timeout_minutes": 15,
            "capabilities": ["web_search", "document_analysis"],
        },
        "coding_agent": {
            "model": "gpt-4o-mini",
            "max_tokens": 4000,
            "temperature": 0.1,
            "timeout_minutes": 30,
            "capabilities": ["code_generation", "code_review"],
        },
        "testing_agent": {
            "model": "gpt-4o-mini",
            "max_tokens": 4000,
            "temperature": 0.2,
            "timeout_minutes": 20,
            "capabilities": ["test_generation", "test_analysis"],
        },
        "documentation_agent": {
            "model": "gpt-4o-mini",
            "max_tokens": 4000,
            "temperature": 0.5,
            "timeout_minutes": 15,
            "capabilities": ["documentation_generation", "content_creation"],
        },
    }

    agents_file = tmp_path / "agents.yaml"
    with agents_file.open("w") as f:
        import yaml

        yaml.dump(agents_config, f)

    # Create orchestrator.yaml
    orchestrator_config = {
        "batch_processing": {
            "default_batch_size": 25,
            "max_batch_size": 100,
            "batch_timeout_minutes": 60,
        },
        "coordination": {
            "max_parallel_agents": 4,
            "agent_selection_strategy": "capability_based",
            "fallback_timeout_minutes": 10,
        },
        "error_handling": {
            "max_retry_attempts": 3,
            "retry_backoff_seconds": 2,
            "circuit_breaker_threshold": 5,
        },
        "monitoring": {
            "enable_metrics": True,
            "metrics_collection_interval": 30,
            "performance_tracking": True,
        },
        "integration": {
            "external_service_timeout": 30,
            "rate_limit_requests_per_minute": 60,
            "cache_ttl_seconds": 300,
        },
    }

    orchestrator_file = tmp_path / "orchestrator.yaml"
    with orchestrator_file.open("w") as f:
        import yaml

        yaml.dump(orchestrator_config, f)

    return {
        "agents_config_path": str(agents_file),
        "orchestrator_config_path": str(orchestrator_file),
        "config_directory": str(tmp_path),
    }


@pytest.fixture
def test_data_factory():
    """Factory for creating various test data scenarios."""

    def _create_test_data(data_type: str, **kwargs):
        """Create test data based on type."""
        if data_type == "user_authentication":
            return {
                "valid_users": [
                    {
                        "id": 1,
                        "username": "alice",
                        "email": "alice@example.com",
                        "role": "admin",
                    },
                    {
                        "id": 2,
                        "username": "bob",
                        "email": "bob@example.com",
                        "role": "user",
                    },
                    {
                        "id": 3,
                        "username": "charlie",
                        "email": "charlie@example.com",
                        "role": "user",
                    },
                ],
                "test_credentials": {
                    "alice": {
                        "password": "secure_password_123",
                        "jwt_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                    },
                    "bob": {
                        "password": "another_password_456",
                        "jwt_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                    },
                },
                "invalid_credentials": [
                    {"username": "invalid", "password": "wrong"},
                    {"username": "alice", "password": "wrong_password"},
                ],
            }

        elif data_type == "api_requests":
            return {
                "valid_requests": [
                    {
                        "method": "POST",
                        "path": "/api/tasks",
                        "headers": {"Content-Type": "application/json"},
                        "body": {
                            "title": "Test Task",
                            "description": "Test description",
                        },
                    },
                    {
                        "method": "GET",
                        "path": "/api/tasks/1",
                        "headers": {"Authorization": "Bearer token"},
                        "body": None,
                    },
                ],
                "invalid_requests": [
                    {
                        "method": "POST",
                        "path": "/api/tasks",
                        "headers": {"Content-Type": "text/plain"},
                        "body": "invalid json",
                    },
                    {
                        "method": "GET",
                        "path": "/api/tasks/invalid",
                        "headers": {},
                        "body": None,
                    },
                ],
            }

        elif data_type == "task_workflow":
            return {
                "simple_workflow": [
                    {"step": 1, "action": "research", "expected_duration": 10},
                    {"step": 2, "action": "implement", "expected_duration": 30},
                    {"step": 3, "action": "test", "expected_duration": 15},
                    {"step": 4, "action": "document", "expected_duration": 5},
                ],
                "complex_workflow": [
                    {
                        "step": 1,
                        "action": "analyze_requirements",
                        "expected_duration": 20,
                    },
                    {
                        "step": 2,
                        "action": "design_architecture",
                        "expected_duration": 40,
                    },
                    {"step": 3, "action": "implement_core", "expected_duration": 60},
                    {
                        "step": 4,
                        "action": "implement_features",
                        "expected_duration": 120,
                    },
                    {
                        "step": 5,
                        "action": "integration_testing",
                        "expected_duration": 30,
                    },
                    {
                        "step": 6,
                        "action": "performance_testing",
                        "expected_duration": 20,
                    },
                    {"step": 7, "action": "documentation", "expected_duration": 25},
                    {"step": 8, "action": "deployment", "expected_duration": 15},
                ],
            }

        else:
            return kwargs

    return _create_test_data


@pytest.fixture(scope="session")
def test_session_data():
    """Session-scoped test data that persists across all tests."""
    return {
        "session_id": f"test_session_{int(time.time())}",
        "start_time": datetime.now(),
        "test_counts": {"passed": 0, "failed": 0, "skipped": 0},
        "performance_baseline": {},
    }

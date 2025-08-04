# Dev-Pro-Agents Testing Infrastructure

This directory contains comprehensive testing infrastructure for the dev-pro-agents project, including unit tests, integration tests, and testing utilities.

## 📁 Structure

```
tests/
├── README.md              # This file - testing documentation
├── __init__.py            # Test package initialization
├── conftest.py            # Pytest fixtures and configuration
├── unit/                  # Unit tests (isolated component testing)
│   ├── core/              # Core orchestration unit tests
│   │   ├── test_agent_protocol.py
│   │   ├── test_agent_registry.py
│   │   ├── test_core_orchestrator.py
│   │   └── test_workflow_state.py
│   ├── agents/            # Individual agent unit tests
│   │   ├── test_coding_agent.py
│   │   ├── test_documentation_agent.py
│   │   ├── test_research_agent.py
│   │   └── test_testing_agent.py
│   ├── data/              # Data models and validation tests
│   │   ├── test_unified_models.py
│   │   ├── test_database_schemas.py
│   │   ├── test_data_transformations.py
│   │   └── test_task_calculations.py
│   ├── services/          # Service layer unit tests
│   │   ├── test_task_manager.py
│   │   ├── test_task_services.py
│   │   └── test_repositories.py
│   └── config/            # Configuration unit tests
│       ├── test_configuration.py
│       └── test_yaml_configuration.py
├── integration/           # Integration tests (multi-component)
│   ├── test_agent_integration.py
│   ├── test_integration_workflows.py
│   ├── test_database_operations.py
│   ├── test_supervisor_workflows.py
│   └── external/          # External service integration tests
│       ├── test_exa_integration.py
│       ├── test_firecrawl_integration.py
│       ├── test_integration_errors.py
│       └── test_integrations.py
├── e2e/                   # End-to-end tests (complete workflows)
│   ├── test_end_to_end.py
│   ├── test_cli_interface.py
│   └── test_batch_execution.py
├── performance/           # Performance and load tests
│   ├── test_performance.py
│   └── test_langgraph_agents.py
├── utils/                 # Testing utilities and helpers
│   ├── utils.py           # Testing utilities and helpers
│   ├── factories.py       # Test data factories and builders
│   ├── agent_fixtures.py  # Agent-specific fixtures
│   ├── fixtures.py        # General test fixtures
│   └── fixtures/          # Mock and fixture components
│       └── http_mocks.py
└── htmlcov/               # Coverage HTML reports (generated)
```

## 🚀 Quick Start

### Running Tests

```bash

# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only end-to-end tests
pytest tests/e2e/

# Run only performance tests
pytest tests/performance/

# Run specific test categories with markers
pytest -m "unit"
pytest -m "integration"
pytest -m "performance"

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/unit/data/test_unified_models.py

# Run specific subdirectory
pytest tests/unit/agents/

# Run with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x
```

### Test Categories

Tests are organized with markers for easy filtering:

- `@pytest.mark.unit` - Fast unit tests

- `@pytest.mark.integration` - Tests requiring external services

- `@pytest.mark.slow` - Tests that take longer to run

- `@pytest.mark.database` - Tests requiring database access

- `@pytest.mark.async` - Tests using async/await patterns

- `@pytest.mark.performance` - Performance/benchmark tests

- `@pytest.mark.smoke` - Basic functionality tests

### Environment Setup

```bash

# Install development dependencies
uv sync --group dev

# Set environment variables for integration tests
export OPENAI_API_KEY="your-key-here"
export EXA_API_KEY="your-key-here"
export FIRECRAWL_API_KEY="your-key-here"

# Skip integration tests if API keys not available
export SKIP_INTEGRATION_TESTS="true"
```

## 🏗️ Testing Components

### 1. Fixtures (`conftest.py`)

Comprehensive fixtures for all testing needs:

#### Database Fixtures

- `test_db_engine` - In-memory SQLite for testing

- `test_db_session` - Database session with rollback

- `isolated_db` - Isolated database with cleanup

- `clean_database` - Clean database state

#### Mock Fixtures

- `mock_openai_client` - OpenAI API mocking

- `mock_exa_client` - Exa search API mocking

- `mock_firecrawl_client` - Firecrawl API mocking

- `mock_httpx_client` - HTTP client mocking

- `mock_orchestrator` - Agent orchestrator mocking

#### Environment Fixtures

- `full_test_environment` - Complete test setup

- `integration_environment` - Integration test config

- `performance_monitor` - Performance tracking

### 2. Test Data Factories (`factories.py`)

Factory functions for creating test data:

#### Basic Factories
```python
from tests.utils.factories import (
    create_task_core,
    create_agent_report,
    create_task_delegation,
    create_task_execution_log
)

# Create a simple task
task = create_task_core(title="Test Task")

# Create with overrides
complex_task = create_task_core(
    complexity=TaskComplexity.VERY_HIGH,
    priority=TaskPriority.CRITICAL
)
```

#### Builder Pattern
```python
from tests.utils.factories import TaskBuilder, AgentReportBuilder

# Build complex tasks fluently
task = (TaskBuilder()
    .with_title("Critical Integration Task")
    .as_critical_task()
    .with_component_area(ComponentArea.ARCHITECTURE)
    .build())

# Build agent reports
report = (AgentReportBuilder()
    .for_agent(AgentType.CODING)
    .for_task(task.id)
    .as_successful_completion()
    .build())
```

#### Batch Factories
```python

# Create task hierarchies
tasks = create_task_hierarchy(parent_count=2, subtask_count=3)

# Create execution history
history = create_execution_history(
    task_id=1, 
    agent_types=[AgentType.RESEARCH, AgentType.CODING]
)
```

### 3. Testing Utilities (`utils.py`)

Helper classes and functions:

#### Database Management
```python
from tests.utils.utils import DatabaseTestManager, clear_database_tables

# Isolated database testing
db_manager = DatabaseTestManager()
session = db_manager.setup_database()

# ... run tests
db_manager.cleanup_database()
```

#### Async Testing
```python
from tests.utils.utils import AsyncTestHelper

# Wait for conditions
success = await AsyncTestHelper.wait_for_condition(
    lambda: check_status(), 
    timeout=10.0
)

# Run with timeout
result = await AsyncTestHelper.run_with_timeout(
    some_async_operation(), 
    timeout=5.0
)
```

#### Assertions
```python
from tests.utils.utils import AssertionHelpers

# Validate task transitions
AssertionHelpers.assert_task_status_transition(
    TaskStatus.IN_PROGRESS, 
    TaskStatus.COMPLETED
)

# Validate agent reports
AssertionHelpers.assert_agent_report_consistent(report)
```

#### Mock Creation
```python
from tests.utils.utils import MockFactory

# Create mock responses
response = MockFactory.create_mock_response(
    {"status": "success"}, 
    status_code=200
)

# Create OpenAI completion
completion = MockFactory.create_mock_openai_completion(
    "Task completed successfully"
)
```

## 📊 Coverage

Coverage is configured to:

- Require minimum 80% coverage

- Generate HTML, XML, and JSON reports

- Exclude CLI and entry point files

- Track branch coverage

### Coverage Reports

```bash

# Generate coverage report
pytest --cov=src --cov-report=html

# View HTML report
open htmlcov/index.html

# Generate XML for CI/CD
pytest --cov=src --cov-report=xml
```

### Coverage Exclusions

The following are excluded from coverage:

- Test files (`tests/*`)

- CLI entry points (`src/cli.py`, `src/supervisor.py`, `src/task_manager.py`)

- Package init files (`*/__init__.py`)

- Virtual environments and migrations

## 🔧 Configuration

### Pytest Configuration (`pyproject.toml`)

Key configuration options:

- Async mode enabled automatically

- Strict marker and config enforcement

- Coverage integration

- Warning filters

- Parallel execution support (optional)

### Coverage Configuration (`.coveragerc`)

- Branch coverage enabled

- Parallel execution support

- Multiple output formats

- Path mapping for CI/CD

## 🌐 Integration Tests

Integration tests require external API keys and are marked with `@pytest.mark.integration`.

### Setup
```bash

# Required environment variables
export OPENAI_API_KEY="sk-..."
export EXA_API_KEY="..."
export FIRECRAWL_API_KEY="..."

# Run integration tests
pytest -m "integration"
```

### Skipping Integration Tests
```bash

# Skip integration tests if APIs unavailable
export SKIP_INTEGRATION_TESTS="true"
pytest  # Will skip integration tests automatically
```

## 🚀 Performance Testing

Performance tests are marked with `@pytest.mark.performance`:

```python
@pytest.mark.performance
def test_task_processing_performance(performance_monitor):
    with performance_monitor.measure_time() as metrics:
        # Run performance-critical code
        process_large_task_batch()
    
    # Assert performance bounds
    performance_monitor.assert_performance_bounds(
        metrics["duration"],
        max_duration=5.0  # 5 seconds max
    )
```

## 🔍 Debugging Tests

### Common Debugging Techniques

```bash

# Run with pdb on failure
pytest --pdb

# Show all output (no capture)
pytest -s

# Verbose output
pytest -vv

# Show local variables on failure
pytest --tb=long

# Run specific test with debugging
pytest tests/test_specific.py::test_function -vv -s
```

### Test Data Debugging

```python

# Use factories with debug info
from tests.factories import create_task_core

task = create_task_core(title="Debug Task")
print(f"Created task: {task.model_dump_json(indent=2)}")
```

## 📝 Writing Tests

### Test Structure

```python
import pytest
from tests.utils.factories import create_task_core
from tests.utils.utils import AssertionHelpers

@pytest.mark.unit
async def test_task_creation(test_db_session):
    """Test task creation functionality."""
    # Arrange
    task_data = create_task_core(title="Test Task")
    
    # Act
    created_task = await create_task_in_database(task_data)
    
    # Assert
    assert created_task.id is not None
    AssertionHelpers.assert_task_core_valid(created_task)
```

### Best Practices

1. **Use descriptive test names**: `test_<function>_<scenario>_<expected_outcome>`
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Use appropriate markers**: `@pytest.mark.unit`, `@pytest.mark.async`, etc.
4. **Leverage factories**: Don't create test data manually
5. **Mock external dependencies**: Use provided mock fixtures
6. **Test edge cases**: Include boundary conditions and error cases
7. **Keep tests independent**: Each test should be able to run in isolation

### Async Testing

```python
@pytest.mark.async
async def test_async_operation(mock_openai_client):
    """Test async operation with mocked dependencies."""
    # Setup mock response
    mock_openai_client.ainvoke.return_value = create_mock_response()
    
    # Test async function
    result = await async_operation()
    
    # Assert results
    assert result.success is True
```

## 🚨 Troubleshooting

### Common Issues

1. **ImportError**: Ensure `uv sync --group dev` was run
2. **Database errors**: Check that test database fixtures are used
3. **Async test failures**: Verify `@pytest.mark.asyncio` is present
4. **Coverage too low**: Add tests for uncovered code paths
5. **Integration test failures**: Check API keys are set

### Debug Commands

```bash

# Show test collection without running
pytest --collect-only

# Show available markers
pytest --markers

# Show fixtures
pytest --fixtures

# Validate test configuration
pytest --help
```

## 🔄 CI/CD Integration

The testing infrastructure is designed for CI/CD:

```yaml

# Example GitHub Actions workflow

- name: Run tests
  run: |
    uv sync --group dev
    pytest --cov=src --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## 📚 Additional Resources

- [pytest documentation](https://docs.pytest.org/)

- [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)

- [coverage.py documentation](https://coverage.readthedocs.io/)

- [Factory Boy patterns](https://factoryboy.readthedocs.io/) (inspiration for our factories)

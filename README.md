# Dev Pro Agents - Multi-Agent Orchestration System

> **Advanced multi-agent orchestration framework for AI-driven development, research, and task automation**

## 🚀 Overview

Dev Pro Agents is a sophisticated multi-agent orchestration system built with LangGraph and Python that coordinates specialized AI agents to handle complex development workflows, research tasks, and automated implementations. The system features dynamic agent discovery, intelligent task routing, and comprehensive integration capabilities.

### Why Dev Pro Agents?

- **Library-First Architecture**: Built on proven open-source libraries like LangGraph, Pydantic v2.11.7, and SQLModel

- **Type-Safe Operations**: Full type safety with modern Python typing and validation

- **Production Ready**: Comprehensive error handling, logging, and monitoring

- **Extensible Design**: Easy to add new agents, capabilities, and integrations

- **Developer Friendly**: Rich CLI, comprehensive documentation, and clear APIs

## 🏗️ Architecture

The system is built using modern Python libraries and follows library-first principles:

### Core Technologies

- **🔄 LangGraph Supervisor**: Multi-agent coordination using `langgraph-supervisor`

- **📊 Pydantic v2.11.7**: Advanced data validation with computed fields, custom validators, and serialization

- **🗄️ SQLModel**: Type-safe database operations combining SQLAlchemy + Pydantic

- **🎨 Rich CLI**: Beautiful command-line interfaces with progress bars and tables

- **⚡ Async/Await**: Full asynchronous operation support for concurrent agent execution

- **🔧 Configuration Management**: YAML-based configuration with environment variable overrides

### Agent Types

- **🧑‍💻 Coding Agent**: Handles code generation, refactoring, and implementation tasks

- **📚 Documentation Agent**: Creates comprehensive documentation from code and specifications

- **🔍 Research Agent**: Performs web research using Exa API and Firecrawl for data gathering

- **🧪 Testing Agent**: Generates and executes comprehensive test suites

### Integration Capabilities

- **🌐 Web Scraping**: Firecrawl integration for intelligent web content extraction

- **🔍 Search & Research**: Exa API integration for semantic web search and research

- **🤖 LLM Providers**: OpenAI and Groq integration for diverse AI capabilities

- **📊 Task Management**: SQLite-based task tracking with dependency resolution

## ✨ Key Features

### Multi-Agent Orchestration

- **Dynamic Agent Discovery**: Automatically discover and register agents from modules

- **Capability-Based Routing**: Route tasks to agents based on required capabilities

- **Health Monitoring**: Real-time agent health status and performance tracking

- **Load Balancing**: Distribute tasks across available agents based on current workload

### Advanced Task Management

- **Dependency Resolution**: Handle complex task dependencies with cycle detection

- **Progress Tracking**: Real-time progress monitoring with detailed execution logs

- **Status Transitions**: Validated status transitions with business rule enforcement

- **Priority Queuing**: Intelligent task prioritization based on complexity and urgency

### Library-First Design

- **Pydantic v2 Models**: Advanced validation with computed fields and custom serializers

- **SQLModel Integration**: Type-safe database operations with automatic migrations

- **Configuration Management**: Hierarchical configuration with validation and defaults

- **Error Handling**: Comprehensive error handling with detailed context and recovery

### Developer Experience

- **Rich CLI Interface**: Beautiful command-line tools with interactive prompts

- **Comprehensive Logging**: Structured logging with configurable levels and outputs

- **Type Safety**: Full type hints and runtime validation

- **Testing Framework**: Comprehensive test suite with async support

## 🚀 Quick Start

### Prerequisites

- Python 3.12+

- uv package manager (recommended) or pip

### Installation

```bash

# Clone the repository
git clone https://github.com/BjornMelin/dev-pro-agents.git
cd dev-pro-agents

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Basic Usage

```bash

# Run the supervisor with default configuration
uv run supervisor

# Manage tasks interactively
uv run task-manager

# Use the main CLI interface
uv run dev-pro-agents --help

# Start a development session
uv run dev-pro-agents dev-session
```

### Configuration

Create a `.env` file in the project root:

```env

# LLM Provider Settings
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here

# Research & Web Scraping
EXA_API_KEY=your_exa_key_here
FIRECRAWL_API_KEY=your_firecrawl_key_here

# Agent Configuration
AGENT_MAX_WORKERS=4
AGENT_TIMEOUT_SECONDS=300
LOG_LEVEL=INFO
```

## 📁 Project Structure

```text
src/
├── agents/                 # Specialized AI agents
│   ├── __init__.py
│   ├── coding_agent.py     # Code generation and refactoring
│   ├── documentation_agent.py  # Documentation creation
│   ├── research_agent.py   # Web research and data gathering
│   └── testing_agent.py    # Test generation and execution
├── core/                   # Core orchestration components
│   ├── __init__.py
│   ├── agent_protocol.py   # Agent interface definitions
│   ├── agent_registry.py   # Agent discovery and management
│   ├── orchestrator.py     # Main orchestration logic
│   └── state.py           # Shared state management
├── integrations/          # External service integrations
│   ├── __init__.py
│   ├── exa_client.py      # Exa API client for research
│   └── firecrawl_client.py # Firecrawl API client for scraping
├── schemas/               # Pydantic models and validation
│   ├── __init__.py
│   ├── database.py        # Database entity models
│   ├── transformations.py # Data transformation utilities
│   └── unified_models.py  # Core business models
├── services/              # Business logic services
│   ├── __init__.py
│   └── task_service.py    # Task management service
├── repositories/          # Data access layer
│   ├── __init__.py
│   ├── base.py           # Base repository patterns
│   └── task_repository.py # Task data access
├── config/               # Configuration files
│   ├── agents.yaml       # Agent configurations
│   └── orchestrator.yaml # Orchestrator settings
├── utils/                # Utility functions
│   ├── __init__.py
│   └── task_calculations.py # Task metric calculations
├── cli.py                # Command-line interface
├── config.py             # Configuration management
├── database.py           # Database models and setup
├── supervisor.py         # LangGraph supervisor implementation
├── supervisor_executor.py # Batch execution coordinator
└── task_manager.py       # Task management interface
```

## 🔧 Configuration

### Agent Configuration (config/agents.yaml)

```yaml
agents:
  coding_agent:
    enabled: true
    max_concurrent_tasks: 2
    timeout_seconds: 300
    llm_provider: "openai"
    model: "gpt-4"
    
  research_agent:
    enabled: true
    max_concurrent_tasks: 3
    timeout_seconds: 600
    search_providers:
      - "exa"
      - "firecrawl"
```

### Orchestrator Configuration (config/orchestrator.yaml)

```yaml
orchestrator:
  max_concurrent_agents: 4
  task_timeout_seconds: 1800
  retry_attempts: 3
  health_check_interval: 30
  
logging:
  level: "INFO"
  format: "structured"
  outputs:
    - "console"
    - "file"
```

## 🧪 Development

### Running Tests

```bash

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_agents.py

# Run async tests
uv run pytest -m asyncio
```

### Code Quality

```bash

# Format code
uv run ruff format .

# Lint code
uv run ruff check . --fix

# Type checking
mypy src/
```

### Adding New Agents

1. Create a new agent class inheriting from `BaseAgent`
2. Implement required methods: `execute_task`, `validate_task`, `get_capabilities`
3. Add configuration in `config/agents.yaml`
4. Register the agent in the registry

Example:

```python
from ..core.agent_protocol import BaseAgent, AgentConfig

class CustomAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.name = "custom_agent"
        self.capabilities = ["custom_capability"]
    
    async def execute_task(self, task_data: dict) -> dict:
        # Implementation here
        return {"status": "completed", "result": "..."}
```

## 📊 Task Management

### Task Lifecycle

1. **Creation**: Tasks are created with metadata, dependencies, and priorities
2. **Validation**: Tasks are validated against agent capabilities
3. **Routing**: Tasks are routed to appropriate agents based on capabilities
4. **Execution**: Agents execute tasks with progress tracking
5. **Completion**: Results are stored with execution logs and metrics

### Task Priority Levels

- **Critical**: Must be completed immediately, blocks other work

- **High**: Important tasks that should be prioritized

- **Medium**: Standard priority tasks

- **Low**: Background tasks, completed when resources available

### Task Statuses

- **not_started**: Initial state, waiting for execution

- **in_progress**: Currently being processed by an agent

- **completed**: Successfully finished

- **failed**: Execution failed with error details

- **blocked**: Waiting for dependencies to complete

- **requires_assistance**: Needs human intervention

## 🔍 Monitoring & Observability

### Health Checks

The system provides comprehensive health monitoring:

- Agent health status and resource usage

- Task queue metrics and processing rates

- Database connection health

- External service availability

### Logging

Structured logging with multiple output formats:

- Console output with rich formatting

- File-based logging with rotation

- JSON structured logs for external systems

- Performance metrics and timing data

### Metrics

Key performance indicators tracked:

- Task completion rates and success ratios

- Agent utilization and performance metrics

- Response times and processing duration

- Error rates and failure patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Run the test suite and ensure all tests pass
5. Submit a pull request with clear description

### Development Guidelines

- Follow the existing code style and conventions

- Add comprehensive tests for new functionality

- Update documentation for any API changes

- Use type hints throughout the codebase

- Follow the library-first architectural principles

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Technical Architecture**: System design and component interactions

- **Agent Development Guide**: How to create and integrate new agents

- **Configuration Reference**: Complete configuration options and examples

- **API Documentation**: Detailed API reference with examples

- **Deployment Guide**: Production deployment recommendations

- **Troubleshooting**: Common issues and solutions

## 🔒 Security

- Environment-based configuration for sensitive data

- Input validation using Pydantic models

- Secure HTTP client configurations

- Database query parameterization

- Comprehensive error handling without information leakage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for the excellent multi-agent framework

- [Pydantic](https://github.com/pydantic/pydantic) for robust data validation

- [SQLModel](https://github.com/tiangolo/sqlmodel) for type-safe database operations

- [Rich](https://github.com/Textualize/rich) for beautiful terminal interfaces

---

> **Built with ❤️ for developers who believe in the power of AI-augmented workflows**

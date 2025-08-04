"""Test data factories and builders for dev-pro-agents testing.

Provides comprehensive factory functions and builders for creating test data
with realistic defaults and easy customization for all model types.
"""

import secrets
import uuid
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from src.schemas.database import (
    Task,
    TaskComment,
    TaskDependency,
    TaskExecutionLog,
    TaskProgress,
)
from src.schemas.unified_models import (
    AgentReport,
    AgentType,
    ComponentArea,
    DependencyType,
    TaskComplexity,
    TaskCore,
    TaskDelegation,
    TaskPriority,
    TaskStatus,
)


# ============================================================================
# BASE FACTORY UTILITIES
# ============================================================================


class FactorySequence:
    """Simple sequence generator for unique values."""

    def __init__(self, start: int = 1):
        self.current = start

    def next(self) -> int:
        value = self.current
        self.current += 1
        return value


# Global sequence generators
_task_id_seq = FactorySequence()
_execution_id_seq = FactorySequence()


def fake_uuid() -> UUID:
    """Generate a fake UUID for testing."""
    return uuid.uuid4()


def fake_datetime(
    days_ago: int = 0, hours_ago: int = 0, minutes_ago: int = 0
) -> datetime:
    """Generate a fake datetime relative to now."""
    now = datetime.now()
    delta = timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
    return now - delta


def random_choice_from_enum(enum_class):
    """Get a random choice from an enum."""
    return secrets.choice(list(enum_class))


# ============================================================================
# TASK FACTORIES
# ============================================================================


def create_task_core(
    id: int | None = None,
    title: str | None = None,
    description: str | None = None,
    component_area: ComponentArea | None = None,
    phase: int | None = None,
    priority: TaskPriority | None = None,
    complexity: TaskComplexity | None = None,
    status: TaskStatus | None = None,
    time_estimate_hours: float | None = None,
    parent_task_id: int | None = None,
    **kwargs,
) -> TaskCore:
    """Factory for creating TaskCore instances."""
    defaults = {
        "id": id or _task_id_seq.next(),
        "title": title or f"Test Task {_task_id_seq.current}",
        "description": description or "A test task for automated testing",
        "component_area": component_area or ComponentArea.TESTING,
        "phase": phase or 1,
        "priority": priority or TaskPriority.MEDIUM,
        "complexity": complexity or TaskComplexity.LOW,
        "status": status or TaskStatus.NOT_STARTED,
        "time_estimate_hours": time_estimate_hours or 2.0,
        "parent_task_id": parent_task_id,
        "success_criteria": "Task completed successfully",
        "source_document": "test_document.md",
    }
    defaults.update(kwargs)
    return TaskCore.model_validate(defaults)


def create_task_db(
    title: str | None = None,
    component_area: ComponentArea | None = None,
    priority: TaskPriority | None = None,
    status: TaskStatus | None = None,
    **kwargs,
) -> Task:
    """Factory for creating Task database entities."""
    defaults = {
        "title": title or f"DB Task {_task_id_seq.next()}",
        "description": "Database task for testing",
        "component_area": component_area or ComponentArea.TESTING,
        "priority": priority or TaskPriority.MEDIUM,
        "status": status or TaskStatus.NOT_STARTED,
        "uuid": fake_uuid(),
        "time_estimate_hours": 1.5,
    }
    defaults.update(kwargs)
    return Task.model_validate(defaults)


def create_task_dependency(
    task_id: int | None = None,
    depends_on_task_id: int | None = None,
    dependency_type: DependencyType | None = None,
    **kwargs,
) -> TaskDependency:
    """Factory for creating TaskDependency instances."""
    defaults = {
        "task_id": task_id or 1,
        "depends_on_task_id": depends_on_task_id or 2,
        "dependency_type": dependency_type or DependencyType.BLOCKS,
    }
    defaults.update(kwargs)
    return TaskDependency.model_validate(defaults)


def create_task_progress(
    task_id: int | None = None,
    progress_percentage: int | None = None,
    notes: str | None = None,
    **kwargs,
) -> TaskProgress:
    """Factory for creating TaskProgress instances."""
    defaults = {
        "task_id": task_id or 1,
        "progress_percentage": progress_percentage or 50,
        "notes": notes or "Progress update from tests",
        "updated_by": "test_system",
    }
    defaults.update(kwargs)
    return TaskProgress.model_validate(defaults)


def create_task_comment(
    task_id: int | None = None,
    comment: str | None = None,
    comment_type: str | None = None,
    **kwargs,
) -> TaskComment:
    """Factory for creating TaskComment instances."""
    defaults = {
        "task_id": task_id or 1,
        "comment": comment or "Test comment for task",
        "comment_type": comment_type or "note",
        "created_by": "test_user",
    }
    defaults.update(kwargs)
    return TaskComment.model_validate(defaults)


# ============================================================================
# EXECUTION AND LOGGING FACTORIES
# ============================================================================


def create_task_execution_log(
    task_id: int | None = None,
    agent_type: AgentType | None = None,
    status: TaskStatus | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    outputs: dict[str, Any] | None = None,
    **kwargs,
) -> TaskExecutionLog:
    """Factory for creating TaskExecutionLog instances."""
    start_dt = start_time or fake_datetime(hours_ago=1)
    defaults = {
        "task_id": task_id or 1,
        "execution_id": fake_uuid(),
        "agent_type": agent_type or AgentType.TESTING,
        "status": status or TaskStatus.COMPLETED,
        "start_time": start_dt,
        "end_time": end_time or start_dt + timedelta(minutes=30),
        "outputs": outputs or {"test_result": "success", "files_created": ["test.py"]},
        "confidence_score": 0.85,
    }
    defaults.update(kwargs)
    return TaskExecutionLog.model_validate(defaults)


def create_agent_report(
    agent_name: AgentType | None = None,
    task_id: int | None = None,
    status: TaskStatus | None = None,
    success: bool | None = None,
    execution_time_minutes: float | None = None,
    outputs: dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
    **kwargs,
) -> AgentReport:
    """Factory for creating AgentReport instances."""
    defaults = {
        "agent_name": agent_name or AgentType.TESTING,
        "task_id": task_id or 1,
        "status": status or TaskStatus.COMPLETED,
        "success": success if success is not None else True,
        "execution_time_minutes": execution_time_minutes or 15.5,
        "outputs": outputs or {"implementation": "Test implementation completed"},
        "artifacts": artifacts or ["test_file.py", "test_config.yaml"],
        "recommendations": ["Add more tests", "Consider edge cases"],
        "next_actions": ["review", "deploy"],
        "confidence_score": 0.9,
        "created_at": fake_datetime(),
    }
    defaults.update(kwargs)
    return AgentReport.model_validate(defaults)


def create_task_delegation(
    assigned_agent: AgentType | None = None,
    priority: TaskPriority | None = None,
    estimated_duration: int | None = None,
    dependencies: list[int] | None = None,
    confidence_score: float | None = None,
    **kwargs,
) -> TaskDelegation:
    """Factory for creating TaskDelegation instances."""
    defaults = {
        "assigned_agent": assigned_agent or AgentType.CODING,
        "reasoning": "Task requires implementation of new features with proper testing",
        "priority": priority or TaskPriority.MEDIUM,
        "estimated_duration": estimated_duration or 90,
        "dependencies": dependencies or [],
        "context_requirements": ["security guidelines", "coding standards"],
        "confidence_score": confidence_score or 0.8,
    }
    defaults.update(kwargs)
    return TaskDelegation.model_validate(defaults)


# ============================================================================
# BATCH FACTORIES FOR COMPLEX SCENARIOS
# ============================================================================


def create_task_hierarchy(
    parent_count: int = 1,
    subtask_count: int = 3,
    component_area: ComponentArea | None = None,
) -> list[TaskCore]:
    """Create a hierarchy of tasks with parent-child relationships."""
    tasks = []

    # Create parent tasks
    for i in range(parent_count):
        parent = create_task_core(
            title=f"Parent Task {i + 1}",
            component_area=component_area or ComponentArea.ARCHITECTURE,
            complexity=TaskComplexity.HIGH,
            time_estimate_hours=20.0,
        )
        tasks.append(parent)

        # Create subtasks
        for j in range(subtask_count):
            subtask = create_task_core(
                title=f"Subtask {j + 1} of Parent {i + 1}",
                parent_task_id=parent.id,
                component_area=component_area or ComponentArea.SERVICES,
                complexity=TaskComplexity.MEDIUM,
                time_estimate_hours=5.0,
            )
            tasks.append(subtask)

    return tasks


def create_task_dependency_chain(task_count: int = 4) -> list[TaskCore]:
    """Create a chain of dependent tasks."""
    tasks = []

    for i in range(task_count):
        task = create_task_core(
            title=f"Chain Task {i + 1}",
            priority=TaskPriority.HIGH if i == 0 else TaskPriority.MEDIUM,
            status=TaskStatus.COMPLETED
            if i < task_count - 2
            else TaskStatus.NOT_STARTED,
        )
        tasks.append(task)

    return tasks


def create_execution_history(
    task_id: int,
    agent_types: list[AgentType] | None = None,
    execution_count: int = 3,
) -> list[TaskExecutionLog]:
    """Create a history of executions for a task."""
    agent_types = agent_types or [
        AgentType.RESEARCH,
        AgentType.CODING,
        AgentType.TESTING,
    ]
    executions = []

    for i, agent in enumerate(agent_types[:execution_count]):
        start_time = fake_datetime(hours_ago=execution_count - i)
        execution = create_task_execution_log(
            task_id=task_id,
            agent_type=agent,
            start_time=start_time,
            end_time=start_time + timedelta(minutes=secrets.randbelow(106) + 15),
            status=TaskStatus.COMPLETED
            if i < execution_count - 1
            else TaskStatus.IN_PROGRESS,
        )
        executions.append(execution)

    return executions


# ============================================================================
# BUILDER PATTERN FACTORIES
# ============================================================================


class TaskBuilder:
    """Builder pattern for creating complex tasks."""

    def __init__(self):
        self.data = {}

    def with_id(self, task_id: int) -> "TaskBuilder":
        self.data["id"] = task_id
        return self

    def with_title(self, title: str) -> "TaskBuilder":
        self.data["title"] = title
        return self

    def with_component_area(self, area: ComponentArea) -> "TaskBuilder":
        self.data["component_area"] = area
        return self

    def with_priority(self, priority: TaskPriority) -> "TaskBuilder":
        self.data["priority"] = priority
        return self

    def with_complexity(self, complexity: TaskComplexity) -> "TaskBuilder":
        self.data["complexity"] = complexity
        return self

    def with_status(self, status: TaskStatus) -> "TaskBuilder":
        self.data["status"] = status
        return self

    def with_estimate(self, hours: float) -> "TaskBuilder":
        self.data["time_estimate_hours"] = hours
        return self

    def as_parent_task(self) -> "TaskBuilder":
        self.data.update(
            {
                "complexity": TaskComplexity.HIGH,
                "time_estimate_hours": 20.0,
                "priority": TaskPriority.HIGH,
            }
        )
        return self

    def as_subtask(self, parent_id: int) -> "TaskBuilder":
        self.data.update(
            {
                "parent_task_id": parent_id,
                "complexity": TaskComplexity.MEDIUM,
                "time_estimate_hours": 5.0,
            }
        )
        return self

    def as_critical_task(self) -> "TaskBuilder":
        self.data.update(
            {
                "priority": TaskPriority.CRITICAL,
                "complexity": TaskComplexity.VERY_HIGH,
                "time_estimate_hours": 40.0,
            }
        )
        return self

    def build(self) -> TaskCore:
        return create_task_core(**self.data)


class AgentReportBuilder:
    """Builder pattern for creating complex agent reports."""

    def __init__(self):
        self.data = {}

    def for_agent(self, agent_type: AgentType) -> "AgentReportBuilder":
        self.data["agent_name"] = agent_type
        return self

    def for_task(self, task_id: int) -> "AgentReportBuilder":
        self.data["task_id"] = task_id
        return self

    def with_status(self, status: TaskStatus) -> "AgentReportBuilder":
        self.data["status"] = status
        return self

    def with_success(self, success: bool) -> "AgentReportBuilder":
        self.data["success"] = success
        return self

    def with_execution_time(self, minutes: float) -> "AgentReportBuilder":
        self.data["execution_time_minutes"] = minutes
        return self

    def with_outputs(self, outputs: dict[str, Any]) -> "AgentReportBuilder":
        self.data["outputs"] = outputs
        return self

    def with_artifacts(self, artifacts: list[str]) -> "AgentReportBuilder":
        self.data["artifacts"] = artifacts
        return self

    def with_confidence(self, score: float) -> "AgentReportBuilder":
        self.data["confidence_score"] = score
        return self

    def as_successful_completion(self) -> "AgentReportBuilder":
        self.data.update(
            {
                "status": TaskStatus.COMPLETED,
                "success": True,
                "confidence_score": 0.9,
                "outputs": {"result": "Task completed successfully"},
            }
        )
        return self

    def as_failed_execution(self, error: str) -> "AgentReportBuilder":
        self.data.update(
            {
                "status": TaskStatus.FAILED,
                "success": False,
                "confidence_score": 0.3,
                "error_details": error,
                "issues_found": [error],
            }
        )
        return self

    def build(self) -> AgentReport:
        return create_agent_report(**self.data)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_minimal_task() -> TaskCore:
    """Create a minimal task for simple tests."""
    return create_task_core(
        title="Minimal Test Task",
        complexity=TaskComplexity.LOW,
        time_estimate_hours=1.0,
    )


def create_complex_task() -> TaskCore:
    """Create a complex task for advanced testing."""
    return create_task_core(
        title="Complex Integration Task",
        description=(
            "A complex task requiring multiple components and careful coordination"
        ),
        component_area=ComponentArea.ARCHITECTURE,
        complexity=TaskComplexity.VERY_HIGH,
        priority=TaskPriority.CRITICAL,
        time_estimate_hours=50.0,
        success_criteria="All components integrated and tested successfully",
    )


def create_test_suite_tasks() -> list[TaskCore]:
    """Create a realistic set of tasks for testing a full development workflow."""
    return [
        create_task_core(
            title="Set up development environment",
            component_area=ComponentArea.ENVIRONMENT,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            time_estimate_hours=4.0,
        ),
        create_task_core(
            title="Design system architecture",
            component_area=ComponentArea.ARCHITECTURE,
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.VERY_HIGH,
            time_estimate_hours=20.0,
        ),
        create_task_core(
            title="Implement core services",
            component_area=ComponentArea.SERVICES,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.HIGH,
            time_estimate_hours=15.0,
        ),
        create_task_core(
            title="Create comprehensive tests",
            component_area=ComponentArea.TESTING,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.MEDIUM,
            time_estimate_hours=8.0,
        ),
        create_task_core(
            title="Write documentation",
            component_area=ComponentArea.DOCUMENTATION,
            priority=TaskPriority.LOW,
            complexity=TaskComplexity.LOW,
            time_estimate_hours=6.0,
        ),
    ]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AgentReportBuilder",
    # Builders
    "TaskBuilder",
    "create_agent_report",
    "create_complex_task",
    "create_execution_history",
    # Convenience functions
    "create_minimal_task",
    "create_task_comment",
    # Core factories
    "create_task_core",
    "create_task_db",
    "create_task_delegation",
    "create_task_dependency",
    "create_task_dependency_chain",
    "create_task_execution_log",
    # Batch factories
    "create_task_hierarchy",
    "create_task_progress",
    "create_test_suite_tasks",
    "fake_datetime",
    # Utilities
    "fake_uuid",
    "random_choice_from_enum",
]

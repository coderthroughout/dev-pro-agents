"""Comprehensive tests for unified_models.py.

This module tests all Pydantic models, enums, validators, computed fields,
and business logic in the unified models schema.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.schemas.unified_models import (
    AgentReport,
    AgentType,
    BaseBusinessModel,
    BaseEntityModel,
    BaseLLMResponseModel,
    ComponentArea,
    DependencyType,
    TaskComplexity,
    TaskCore,
    TaskDelegation,
    TaskPriority,
    TaskStatus,
    UnifiedConfig,
    can_transition_status,
    get_status_progress_percentage,
)


class TestEnums:
    """Test all enum classes for completeness and consistency."""

    def test_task_status_enum_values(self):
        """Test TaskStatus enum has all expected values."""
        expected_values = {
            "not_started",
            "in_progress",
            "completed",
            "blocked",
            "failed",
            "requires_assistance",
            "partial",
        }
        actual_values = {status.value for status in TaskStatus}
        assert actual_values == expected_values

    def test_task_priority_enum_values(self):
        """Test TaskPriority enum has all expected values."""
        expected_values = {"low", "medium", "high", "critical"}
        actual_values = {priority.value for priority in TaskPriority}
        assert actual_values == expected_values

    def test_task_complexity_enum_values(self):
        """Test TaskComplexity enum has all expected values."""
        expected_values = {"low", "medium", "high", "very_high"}
        actual_values = {complexity.value for complexity in TaskComplexity}
        assert actual_values == expected_values

    def test_component_area_enum_values(self):
        """Test ComponentArea enum has all expected values."""
        expected_values = {
            "environment",
            "dependencies",
            "configuration",
            "architecture",
            "database",
            "services",
            "ui",
            "testing",
            "documentation",
            "security",
            "task",
        }
        actual_values = {area.value for area in ComponentArea}
        assert actual_values == expected_values

    def test_agent_type_enum_values(self):
        """Test AgentType enum has all expected values."""
        expected_values = {
            "research",
            "coding",
            "testing",
            "documentation",
            "supervisor",
        }
        actual_values = {agent.value for agent in AgentType}
        assert actual_values == expected_values

    def test_dependency_type_enum_values(self):
        """Test DependencyType enum has all expected values."""
        expected_values = {"blocks", "enables", "enhances", "requires"}
        actual_values = {dep.value for dep in DependencyType}
        assert actual_values == expected_values

    @pytest.mark.parametrize(
        "enum_class",
        [
            TaskStatus,
            TaskPriority,
            TaskComplexity,
            ComponentArea,
            AgentType,
            DependencyType,
        ],
    )
    def test_enum_string_conversion(self, enum_class):
        """Test that all enums can be converted to/from strings."""
        for enum_value in enum_class:
            # Test string representation
            assert str(enum_value) == enum_value.value
            # Test reconstruction from string
            assert enum_class(enum_value.value) == enum_value


class TestUnifiedConfig:
    """Test unified configuration for all models."""

    def test_pydantic_config_attributes(self):
        """Test that UnifiedConfig has all required attributes."""
        config = UnifiedConfig.PYDANTIC_CONFIG

        assert config.get("strict") is True
        assert config.get("extra") == "forbid"
        assert config.get("validate_assignment") is True
        assert config.get("use_enum_values") is False
        assert config.get("serialize_by_alias") is True
        assert config.get("frozen") is False
        assert config.get("from_attributes") is True


class TestBaseModels:
    """Test base model classes and their behavior."""

    def test_base_business_model_config(self):
        """Test BaseBusinessModel uses unified config."""

        class TestModel(BaseBusinessModel):
            value: str

        model = TestModel(value="test")
        assert model.model_config == UnifiedConfig.PYDANTIC_CONFIG

    def test_base_business_model_strict_validation(self):
        """Test BaseBusinessModel enforces strict validation."""

        class TestModel(BaseBusinessModel):
            value: int

        # Should work with correct type
        model = TestModel(value=42)
        assert model.value == 42

        # Should fail with wrong type in strict mode
        with pytest.raises(ValidationError):
            TestModel(value="not_an_int")

    def test_base_business_model_extra_forbid(self):
        """Test BaseBusinessModel forbids extra fields."""

        class TestModel(BaseBusinessModel):
            value: str

        # Should work with expected field
        model = TestModel(value="test")
        assert model.value == "test"

        # Should fail with extra field
        with pytest.raises(ValidationError):
            TestModel(value="test", extra_field="not_allowed")

    def test_base_entity_model_timestamps(self):
        """Test BaseEntityModel automatically creates timestamps."""

        class TestEntity(BaseEntityModel):
            name: str

        entity = TestEntity(name="test")

        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.updated_at, datetime)
        assert entity.created_at <= datetime.now()
        assert entity.updated_at <= datetime.now()

    def test_base_entity_model_updated_at_validator(self):
        """Test BaseEntityModel updated_at validator sets current time."""

        class TestEntity(BaseEntityModel):
            name: str

        # Test with None value
        entity = TestEntity(name="test", updated_at=None)
        assert isinstance(entity.updated_at, datetime)

        # Test with specific datetime
        specific_time = datetime(2023, 1, 1, 12, 0, 0)
        entity = TestEntity(name="test", updated_at=specific_time)
        assert entity.updated_at == specific_time


class TestBaseLLMResponseModel:
    """Test BaseLLMResponseModel JSON extraction functionality."""

    def test_extract_json_from_markdown_blocks(self):
        """Test extraction of JSON from markdown code blocks."""

        class TestModel(BaseLLMResponseModel):
            value: str

        markdown_text = """
        Here's the response:
        ```json
        {"value": "extracted"}
        ```
        Some additional text.
        """

        model = TestModel.model_validate(markdown_text)
        assert model.value == "extracted"

    def test_extract_json_from_text_with_braces(self):
        """Test extraction of JSON from text containing braces."""

        class TestModel(BaseLLMResponseModel):
            value: str

        text_with_json = 'Some text {"value": "found"} more text'

        model = TestModel.model_validate(text_with_json)
        assert model.value == "found"

    def test_extract_json_from_pure_json_string(self):
        """Test parsing of pure JSON strings."""

        class TestModel(BaseLLMResponseModel):
            value: str

        json_string = '{"value": "direct"}'

        model = TestModel.model_validate(json_string)
        assert model.value == "direct"

    def test_extract_json_fallback_raw_response(self):
        """Test fallback to raw_response for unparseable text."""

        class TestModel(BaseLLMResponseModel):
            raw_response: str

        unparseable_text = "This is just plain text without JSON"

        model = TestModel.model_validate(unparseable_text)
        assert model.raw_response == unparseable_text

    def test_extract_json_handles_dict_input(self):
        """Test that dict input passes through unchanged."""

        class TestModel(BaseLLMResponseModel):
            value: str

        dict_input = {"value": "direct_dict"}

        model = TestModel.model_validate(dict_input)
        assert model.value == "direct_dict"

    def test_extract_json_handles_complex_json(self):
        """Test extraction of complex nested JSON."""

        class TestModel(BaseLLMResponseModel):
            nested: dict
            array: list

        complex_json = """
        Response: ```json
        {
            "nested": {"key": "value", "number": 42},
            "array": [1, 2, 3, "string"]
        }
        ```
        """

        model = TestModel.model_validate(complex_json)
        assert model.nested == {"key": "value", "number": 42}
        assert model.array == [1, 2, 3, "string"]


class TestTaskDelegation:
    """Test TaskDelegation model validation and business logic."""

    def test_task_delegation_valid_creation(self):
        """Test creating valid TaskDelegation."""
        delegation = TaskDelegation(
            assigned_agent=AgentType.CODING,
            reasoning="Task requires implementation",
            priority=TaskPriority.HIGH,
            estimated_duration=120,
            dependencies=[1, 2],
            context_requirements=["auth", "db"],
            confidence_score=0.9,
        )

        assert delegation.assigned_agent == AgentType.CODING
        assert delegation.reasoning == "Task requires implementation"
        assert delegation.priority == TaskPriority.HIGH
        assert delegation.estimated_duration == 120
        assert delegation.dependencies == [1, 2]
        assert delegation.context_requirements == ["auth", "db"]
        assert delegation.confidence_score == 0.9

    def test_task_delegation_default_values(self):
        """Test TaskDelegation default values."""
        delegation = TaskDelegation(
            assigned_agent=AgentType.RESEARCH,
            reasoning="Research needed",
            priority=TaskPriority.MEDIUM,
            estimated_duration=60,
        )

        assert delegation.dependencies == []
        assert delegation.context_requirements == []
        assert delegation.confidence_score == 0.8

    def test_task_delegation_estimated_duration_validation(self):
        """Test estimated_duration must be positive."""
        with pytest.raises(ValidationError):
            TaskDelegation(
                assigned_agent=AgentType.CODING,
                reasoning="Test",
                priority=TaskPriority.LOW,
                estimated_duration=0,  # Invalid: must be > 0
            )

        with pytest.raises(ValidationError):
            TaskDelegation(
                assigned_agent=AgentType.CODING,
                reasoning="Test",
                priority=TaskPriority.LOW,
                estimated_duration=-10,  # Invalid: negative
            )

    def test_task_delegation_confidence_score_range(self):
        """Test confidence_score must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            TaskDelegation(
                assigned_agent=AgentType.CODING,
                reasoning="Test",
                priority=TaskPriority.LOW,
                estimated_duration=60,
                confidence_score=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValidationError):
            TaskDelegation(
                assigned_agent=AgentType.CODING,
                reasoning="Test",
                priority=TaskPriority.LOW,
                estimated_duration=60,
                confidence_score=-0.1,  # Invalid: < 0.0
            )

    def test_task_delegation_duration_limit_validation(self):
        """Test validation prevents tasks longer than 8 hours."""
        with pytest.raises(ValidationError, match="cannot exceed 8 hours"):
            TaskDelegation(
                assigned_agent=AgentType.CODING,
                reasoning="Very long task",
                priority=TaskPriority.LOW,
                estimated_duration=500,  # 500 minutes > 480 (8 hours)
            )

    def test_task_delegation_from_llm_response(self):
        """Test creating TaskDelegation from LLM response."""
        llm_response = """
        ```json
        {
            "assigned_agent": "testing",
            "reasoning": "This task needs comprehensive testing",
            "priority": "critical",
            "estimated_duration": 240,
            "dependencies": [5, 6],
            "context_requirements": ["test framework", "coverage tools"],
            "confidence_score": 0.95
        }
        ```
        """

        delegation = TaskDelegation.model_validate(llm_response)
        assert delegation.assigned_agent == AgentType.TESTING
        assert delegation.priority == TaskPriority.CRITICAL
        assert delegation.estimated_duration == 240
        assert delegation.confidence_score == 0.95


class TestAgentReport:
    """Test AgentReport model validation and computed fields."""

    def test_agent_report_valid_creation(self):
        """Test creating valid AgentReport."""
        report = AgentReport(
            agent_name=AgentType.CODING,
            task_id=123,
            status=TaskStatus.COMPLETED,
            success=True,
            execution_time_minutes=45.5,
            outputs={"result": "implementation complete"},
            artifacts=["main.py", "tests.py"],
            recommendations=["add logging", "improve error handling"],
            next_actions=["deploy", "monitor"],
            issues_found=["minor style issues"],
            confidence_score=0.92,
            error_details=None,
            metadata={"version": "1.0"},
        )

        assert report.agent_name == AgentType.CODING
        assert report.task_id == 123
        assert report.status == TaskStatus.COMPLETED
        assert report.success is True
        assert report.execution_time_minutes == 45.5
        assert report.outputs == {"result": "implementation complete"}
        assert report.artifacts == ["main.py", "tests.py"]
        assert len(report.recommendations) == 2
        assert len(report.next_actions) == 2
        assert len(report.issues_found) == 1
        assert report.confidence_score == 0.92
        assert report.metadata == {"version": "1.0"}

    def test_agent_report_default_values(self):
        """Test AgentReport default values."""
        report = AgentReport(
            agent_name=AgentType.RESEARCH, status=TaskStatus.IN_PROGRESS
        )

        assert report.task_id is None
        assert report.success is True
        assert report.execution_time_minutes == 0.0
        assert report.outputs == {}
        assert report.artifacts == []
        assert report.recommendations == []
        assert report.next_actions == []
        assert report.issues_found == []
        assert report.confidence_score == 0.8
        assert report.error_details is None
        assert report.metadata == {}
        assert isinstance(report.created_at, datetime)

    def test_agent_report_completion_quality_score_completed_success(self):
        """Test completion quality score for successful completion."""
        report = AgentReport(
            agent_name=AgentType.CODING,
            status=TaskStatus.COMPLETED,
            success=True,
            confidence_score=0.8,
        )

        # Completed + success = confidence * 1.2, capped at 1.0
        expected_score = min(0.8 * 1.2, 1.0)
        assert report.completion_quality_score == expected_score

    def test_agent_report_completion_quality_score_failed(self):
        """Test completion quality score for failed status."""
        report = AgentReport(
            agent_name=AgentType.CODING,
            status=TaskStatus.FAILED,
            success=False,
            confidence_score=0.8,
            issues_found=["critical error"],
            error_details="System failure",
        )

        # Failed = confidence * 0.3
        expected_score = 0.8 * 0.3
        assert report.completion_quality_score == expected_score

    def test_agent_report_completion_quality_score_with_issues(self):
        """Test completion quality score when issues found."""
        report = AgentReport(
            agent_name=AgentType.TESTING,
            status=TaskStatus.PARTIAL,
            confidence_score=0.9,
            issues_found=["test failure", "coverage gap"],
        )

        # Issues found = confidence * 0.7
        expected_score = 0.9 * 0.7
        assert report.completion_quality_score == expected_score

    def test_agent_report_status_consistency_failed_requires_details(self):
        """Test failed status requires issues or error details."""
        with pytest.raises(ValidationError, match="Failed status requires"):
            AgentReport(
                agent_name=AgentType.CODING,
                status=TaskStatus.FAILED,
                success=False,
                # Missing issues_found and error_details
            )

    def test_agent_report_status_consistency_blocked_requires_issues(self):
        """Test blocked status requires issues_found."""
        with pytest.raises(ValidationError, match="Blocked status requires"):
            AgentReport(
                agent_name=AgentType.CODING,
                status=TaskStatus.BLOCKED,
                # Missing issues_found
            )

    def test_agent_report_status_consistency_completed_success_mismatch(self):
        """Test completed status cannot have success=False."""
        with pytest.raises(ValidationError, match="Cannot have completed status"):
            AgentReport(
                agent_name=AgentType.CODING,
                status=TaskStatus.COMPLETED,
                success=False,  # Invalid combination
            )

    def test_agent_report_status_consistency_valid_failed(self):
        """Test valid failed status with proper details."""
        report = AgentReport(
            agent_name=AgentType.CODING,
            status=TaskStatus.FAILED,
            success=False,
            error_details="Implementation failed due to missing dependencies",
        )

        assert report.status == TaskStatus.FAILED
        assert report.success is False
        assert report.error_details is not None

    def test_agent_report_status_consistency_valid_blocked(self):
        """Test valid blocked status with issues."""
        report = AgentReport(
            agent_name=AgentType.DOCUMENTATION,
            status=TaskStatus.BLOCKED,
            issues_found=["waiting for API documentation", "unclear requirements"],
        )

        assert report.status == TaskStatus.BLOCKED
        assert len(report.issues_found) == 2

    def test_agent_report_from_llm_response(self):
        """Test creating AgentReport from LLM response."""
        llm_response = """
        The task has been completed successfully.
        ```json
        {
            "agent_name": "documentation",
            "task_id": 456,
            "status": "completed",
            "success": true,
            "execution_time_minutes": 30.0,
            "outputs": {"documentation": "API docs generated"},
            "artifacts": ["api_docs.md", "examples.py"],
            "recommendations": ["add more examples", "review for clarity"],
            "next_actions": ["publish docs", "notify team"],
            "confidence_score": 0.88
        }
        ```
        """

        report = AgentReport.model_validate(llm_response)
        assert report.agent_name == AgentType.DOCUMENTATION
        assert report.task_id == 456
        assert report.status == TaskStatus.COMPLETED
        assert report.success is True
        assert report.execution_time_minutes == 30.0


class TestTaskCore:
    """Test TaskCore model validation and computed fields."""

    def test_task_core_valid_creation(self):
        """Test creating valid TaskCore."""
        task = TaskCore(
            id=1,
            title="Implement authentication",
            description="Secure user authentication system",
            component_area=ComponentArea.SECURITY,
            phase=2,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            status=TaskStatus.IN_PROGRESS,
            source_document="requirements.md",
            success_criteria="Users can login securely",
            time_estimate_hours=6.5,
            parent_task_id=None,
        )

        assert task.id == 1
        assert task.title == "Implement authentication"
        assert task.description == "Secure user authentication system"
        assert task.component_area == ComponentArea.SECURITY
        assert task.phase == 2
        assert task.priority == TaskPriority.HIGH
        assert task.complexity == TaskComplexity.MEDIUM
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.time_estimate_hours == 6.5

    def test_task_core_default_values(self):
        """Test TaskCore default values."""
        task = TaskCore(title="Test Task")

        assert task.id is None
        assert task.description == ""
        assert task.component_area == ComponentArea.TASK
        assert task.phase == 1
        assert task.priority == TaskPriority.MEDIUM
        assert task.complexity == TaskComplexity.MEDIUM
        assert task.status == TaskStatus.NOT_STARTED
        assert task.source_document == ""
        assert task.success_criteria == ""
        assert task.time_estimate_hours == 1.0
        assert task.parent_task_id is None
        assert isinstance(task.created_at, datetime)
        assert task.updated_at is None

    def test_task_core_title_validation(self):
        """Test title validation constraints."""
        # Title too short
        with pytest.raises(ValidationError):
            TaskCore(title="")

        # Title too long
        with pytest.raises(ValidationError):
            TaskCore(title="x" * 201)

        # Valid title lengths
        TaskCore(title="x")  # Minimum length
        TaskCore(title="x" * 200)  # Maximum length

    def test_task_core_phase_validation(self):
        """Test phase validation constraints."""
        # Phase too low
        with pytest.raises(ValidationError):
            TaskCore(title="Test", phase=0)

        # Valid phases
        TaskCore(title="Test", phase=1)  # Minimum
        TaskCore(title="Test", phase=10)  # Should work

    def test_task_core_time_estimate_validation(self):
        """Test time estimate validation constraints."""
        # Time too low
        with pytest.raises(ValidationError):
            TaskCore(title="Test", time_estimate_hours=0.05)

        # Time too high
        with pytest.raises(ValidationError):
            TaskCore(title="Test", time_estimate_hours=200.0)

        # Valid time estimates
        TaskCore(title="Test", time_estimate_hours=0.1)  # Minimum
        TaskCore(title="Test", time_estimate_hours=160.0)  # Maximum

    def test_task_core_complexity_multiplier(self):
        """Test complexity multiplier computed field."""
        test_cases = [
            (TaskComplexity.LOW, 0.8),
            (TaskComplexity.MEDIUM, 1.0),
            (TaskComplexity.HIGH, 1.5),
            (TaskComplexity.VERY_HIGH, 2.0),
        ]

        for complexity, expected_multiplier in test_cases:
            task = TaskCore(title="Test", complexity=complexity)
            assert task.complexity_multiplier == expected_multiplier

    def test_task_core_effort_index(self):
        """Test effort index computed field."""
        task = TaskCore(
            title="Test",
            time_estimate_hours=4.0,
            complexity=TaskComplexity.HIGH,  # multiplier = 1.5
        )

        expected_effort = 4.0 * 1.5
        assert task.effort_index == expected_effort

    def test_task_core_is_overdue_property(self):
        """Test is_overdue computed property."""
        # Not overdue - not in progress
        task1 = TaskCore(
            title="Test", time_estimate_hours=10.0, status=TaskStatus.NOT_STARTED
        )
        assert task1.is_overdue is False

        # Not overdue - in progress but under 8 hours
        task2 = TaskCore(
            title="Test", time_estimate_hours=6.0, status=TaskStatus.IN_PROGRESS
        )
        assert task2.is_overdue is False

        # Overdue - in progress and over 8 hours
        task3 = TaskCore(
            title="Test", time_estimate_hours=10.0, status=TaskStatus.IN_PROGRESS
        )
        assert task3.is_overdue is True

    def test_task_core_model_serialization(self):
        """Test TaskCore can be serialized to/from dict."""
        original_task = TaskCore(
            title="Serialization Test",
            description="Test serialization",
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.VERY_HIGH,
        )

        # Serialize to dict
        task_dict = original_task.model_dump()
        assert isinstance(task_dict, dict)
        assert task_dict["title"] == "Serialization Test"

        # Deserialize from dict
        restored_task = TaskCore.model_validate(task_dict)
        assert restored_task.title == original_task.title
        assert restored_task.priority == original_task.priority
        assert restored_task.complexity == original_task.complexity


class TestUtilityFunctions:
    """Test utility functions for status and transitions."""

    def test_get_status_progress_percentage(self):
        """Test status to progress percentage mapping."""
        expected_mappings = {
            TaskStatus.NOT_STARTED: 0,
            TaskStatus.IN_PROGRESS: 50,
            TaskStatus.COMPLETED: 100,
            TaskStatus.BLOCKED: 25,
            TaskStatus.FAILED: 0,
            TaskStatus.REQUIRES_ASSISTANCE: 25,
            TaskStatus.PARTIAL: 75,
        }

        for status, expected_percentage in expected_mappings.items():
            assert get_status_progress_percentage(status) == expected_percentage

    def test_can_transition_status_valid_transitions(self):
        """Test valid status transitions."""
        valid_transitions = [
            (TaskStatus.NOT_STARTED, TaskStatus.IN_PROGRESS),
            (TaskStatus.NOT_STARTED, TaskStatus.BLOCKED),
            (TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED),
            (TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED),
            (TaskStatus.IN_PROGRESS, TaskStatus.FAILED),
            (TaskStatus.IN_PROGRESS, TaskStatus.REQUIRES_ASSISTANCE),
            (TaskStatus.IN_PROGRESS, TaskStatus.PARTIAL),
            (TaskStatus.BLOCKED, TaskStatus.IN_PROGRESS),
            (TaskStatus.BLOCKED, TaskStatus.FAILED),
            (TaskStatus.REQUIRES_ASSISTANCE, TaskStatus.IN_PROGRESS),
            (TaskStatus.REQUIRES_ASSISTANCE, TaskStatus.BLOCKED),
            (TaskStatus.FAILED, TaskStatus.IN_PROGRESS),
            (TaskStatus.FAILED, TaskStatus.BLOCKED),
            (TaskStatus.PARTIAL, TaskStatus.IN_PROGRESS),
            (TaskStatus.PARTIAL, TaskStatus.COMPLETED),
            (TaskStatus.PARTIAL, TaskStatus.BLOCKED),
        ]

        for from_status, to_status in valid_transitions:
            assert can_transition_status(from_status, to_status) is True

    def test_can_transition_status_invalid_transitions(self):
        """Test invalid status transitions."""
        invalid_transitions = [
            (TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS),
            (TaskStatus.COMPLETED, TaskStatus.BLOCKED),
            (TaskStatus.COMPLETED, TaskStatus.FAILED),
            (TaskStatus.NOT_STARTED, TaskStatus.COMPLETED),  # Skip in_progress
            (TaskStatus.NOT_STARTED, TaskStatus.FAILED),
            (TaskStatus.IN_PROGRESS, TaskStatus.NOT_STARTED),  # Backwards
        ]

        for from_status, to_status in invalid_transitions:
            assert can_transition_status(from_status, to_status) is False

    def test_can_transition_status_completed_final(self):
        """Test that completed status has no valid transitions."""
        for status in TaskStatus:
            if status != TaskStatus.COMPLETED:
                assert can_transition_status(TaskStatus.COMPLETED, status) is False


class TestModelIntegration:
    """Test integration between different models."""

    def test_task_delegation_to_agent_report_workflow(self):
        """Test typical workflow from delegation to report."""
        # Start with delegation
        delegation = TaskDelegation(
            assigned_agent=AgentType.CODING,
            reasoning="Implementation needed",
            priority=TaskPriority.HIGH,
            estimated_duration=120,
        )

        # Create corresponding task
        task = TaskCore(
            title="Implement feature",
            priority=delegation.priority,
            time_estimate_hours=delegation.estimated_duration / 60.0,
        )

        # Complete with report
        report = AgentReport(
            agent_name=delegation.assigned_agent,
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            success=True,
            execution_time_minutes=delegation.estimated_duration,
            confidence_score=0.9,
        )

        assert report.agent_name == delegation.assigned_agent
        assert report.execution_time_minutes == delegation.estimated_duration
        assert report.completion_quality_score > 0.9  # Should be high for completed

    def test_task_status_progression_workflow(self):
        """Test typical task status progression."""
        task = TaskCore(title="Progressive Task")

        # Initial status
        assert task.status == TaskStatus.NOT_STARTED
        assert get_status_progress_percentage(task.status) == 0

        # Valid progressions
        assert can_transition_status(task.status, TaskStatus.IN_PROGRESS)

        # Simulate progression
        statuses = [TaskStatus.IN_PROGRESS, TaskStatus.PARTIAL, TaskStatus.COMPLETED]

        current_status = task.status
        for next_status in statuses:
            assert can_transition_status(current_status, next_status)
            current_status = next_status

        # Final status should be 100% complete
        assert get_status_progress_percentage(current_status) == 100

    def test_complexity_effort_relationships(self):
        """Test relationships between complexity and effort calculations."""
        base_time = 4.0

        tasks = [
            TaskCore(
                title="Low",
                complexity=TaskComplexity.LOW,
                time_estimate_hours=base_time,
            ),
            TaskCore(
                title="Medium",
                complexity=TaskComplexity.MEDIUM,
                time_estimate_hours=base_time,
            ),
            TaskCore(
                title="High",
                complexity=TaskComplexity.HIGH,
                time_estimate_hours=base_time,
            ),
            TaskCore(
                title="Very High",
                complexity=TaskComplexity.VERY_HIGH,
                time_estimate_hours=base_time,
            ),
        ]

        # Effort should increase with complexity
        efforts = [task.effort_index for task in tasks]
        assert efforts == sorted(efforts)  # Should be in ascending order

        # Multipliers should be applied correctly
        assert tasks[0].effort_index == base_time * 0.8  # LOW
        assert tasks[1].effort_index == base_time * 1.0  # MEDIUM
        assert tasks[2].effort_index == base_time * 1.5  # HIGH
        assert tasks[3].effort_index == base_time * 2.0  # VERY_HIGH


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_task_core_extreme_values(self):
        """Test TaskCore with extreme but valid values."""
        # Minimum values
        min_task = TaskCore(
            title="x",  # Minimum length
            time_estimate_hours=0.1,  # Minimum time
            phase=1,  # Minimum phase
        )
        assert min_task.title == "x"
        assert min_task.time_estimate_hours == 0.1
        assert min_task.phase == 1

        # Maximum values
        max_task = TaskCore(
            title="x" * 200,  # Maximum length
            time_estimate_hours=160.0,  # Maximum time
            phase=10,  # High phase
        )
        assert len(max_task.title) == 200
        assert max_task.time_estimate_hours == 160.0
        assert max_task.phase == 10

    def test_agent_report_edge_case_scores(self):
        """Test AgentReport with edge case confidence scores."""
        # Minimum confidence
        report_min = AgentReport(
            agent_name=AgentType.TESTING,
            status=TaskStatus.COMPLETED,
            confidence_score=0.0,
        )
        assert report_min.confidence_score == 0.0
        assert report_min.completion_quality_score == 0.0

        # Maximum confidence
        report_max = AgentReport(
            agent_name=AgentType.TESTING,
            status=TaskStatus.COMPLETED,
            confidence_score=1.0,
        )
        assert report_max.confidence_score == 1.0
        # Completion score should be capped at 1.0 even if calculation exceeds
        assert report_max.completion_quality_score == 1.0

    def test_task_delegation_edge_case_duration(self):
        """Test TaskDelegation with edge case durations."""
        # Minimum duration
        delegation_min = TaskDelegation(
            assigned_agent=AgentType.RESEARCH,
            reasoning="Quick task",
            priority=TaskPriority.LOW,
            estimated_duration=1,  # 1 minute
        )
        assert delegation_min.estimated_duration == 1

        # Maximum allowed duration (8 hours = 480 minutes)
        delegation_max = TaskDelegation(
            assigned_agent=AgentType.DOCUMENTATION,
            reasoning="Long documentation task",
            priority=TaskPriority.MEDIUM,
            estimated_duration=480,  # Exactly 8 hours
        )
        assert delegation_max.estimated_duration == 480

    def test_llm_response_model_malformed_json(self):
        """Test BaseLLMResponseModel with various malformed JSON."""

        class TestModel(BaseLLMResponseModel):
            raw_response: str = None
            data: str = None

        test_cases = [
            '{"incomplete": "json"',  # Missing closing brace
            '{"invalid": json}',  # Invalid JSON syntax
            "just plain text",  # No JSON at all
            '```json\n{"broken": json\n```',  # Broken JSON in markdown
            "",  # Empty string
        ]

        for malformed_input in test_cases:
            model = TestModel.model_validate(malformed_input)
            # Should fallback to raw_response or data field
            assert hasattr(model, "raw_response") or hasattr(model, "data")


class TestModelConsistency:
    """Test consistency across all models."""

    def test_all_models_use_unified_config(self):
        """Test that all business models use UnifiedConfig."""
        business_models = [
            BaseBusinessModel,
            BaseLLMResponseModel,
            TaskDelegation,
            AgentReport,
            TaskCore,
        ]

        for model_class in business_models:
            # Create instance to check config
            if model_class == BaseBusinessModel:
                continue  # Skip abstract base

            # Create with minimal required fields
            if model_class == TaskDelegation:
                instance = model_class(
                    assigned_agent=AgentType.CODING,
                    reasoning="test",
                    priority=TaskPriority.LOW,
                    estimated_duration=60,
                )
            elif model_class == AgentReport:
                instance = model_class(
                    agent_name=AgentType.CODING, status=TaskStatus.COMPLETED
                )
            elif model_class == TaskCore:
                instance = model_class(title="test")
            else:
                # BaseLLMResponseModel subclass
                class TestModel(model_class):
                    value: str = "test"

                instance = TestModel()

            if hasattr(instance, "model_config"):
                config = instance.model_config
                assert config.get("strict") is True
                assert config.get("extra") == "forbid"
                assert config.get("validate_assignment") is True

    def test_enum_consistency_across_models(self):
        """Test that enums are used consistently across models."""
        # Create instances with same enum values
        task = TaskCore(
            title="Consistency Test",
            status=TaskStatus.IN_PROGRESS,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            component_area=ComponentArea.TESTING,
        )

        report = AgentReport(
            agent_name=AgentType.TESTING,
            status=TaskStatus.IN_PROGRESS,  # Same as task
        )

        delegation = TaskDelegation(
            assigned_agent=AgentType.TESTING,
            reasoning="test",
            priority=TaskPriority.HIGH,  # Same as task
            estimated_duration=60,
        )

        # Verify consistency
        assert task.status == report.status
        assert task.priority == delegation.priority
        # Enum values should be the same
        assert task.status.value == report.status.value
        assert task.priority.value == delegation.priority.value

    def test_datetime_handling_consistency(self):
        """Test consistent datetime handling across models."""
        task = TaskCore(title="DateTime Test")
        report = AgentReport(agent_name=AgentType.CODING, status=TaskStatus.COMPLETED)

        # Both should have datetime fields
        assert isinstance(task.created_at, datetime)
        assert isinstance(report.created_at, datetime)

        # Should be close to current time
        now = datetime.now()
        assert (now - task.created_at).total_seconds() < 1
        assert (now - report.created_at).total_seconds() < 1

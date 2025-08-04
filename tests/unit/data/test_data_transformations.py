"""Comprehensive tests for transformations.py schema conversion utilities.

This module tests all transformation functions, legacy compatibility,
batch operations, and data migration scenarios.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.schemas.database import Task as TaskEntity
from src.schemas.database import TaskExecutionLog, TaskProgress
from src.schemas.transformations import (
    BatchTransformer,
    LegacyCompatibilityLayer,
    SchemaTransformer,
    convert_to_core_model,
    convert_to_entity,
    validate_and_transform,
)
from src.schemas.unified_models import (
    AgentReport,
    AgentType,
    ComponentArea,
    TaskComplexity,
    TaskCore,
    TaskPriority,
    TaskStatus,
)


class TestSchemaTransformer:
    """Test SchemaTransformer central transformation hub."""

    def test_task_entity_to_core_conversion(self):
        """Test TaskEntity to TaskCore conversion."""
        entity = TaskEntity(
            title="Entity Task",
            description="Entity description",
            component_area=ComponentArea.SECURITY,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
            status=TaskStatus.IN_PROGRESS,
            time_estimate_hours=4.5,
        )

        core_model = SchemaTransformer.task_entity_to_core(entity)

        assert isinstance(core_model, TaskCore)
        assert core_model.title == entity.title
        assert core_model.description == entity.description
        assert core_model.component_area == entity.component_area
        assert core_model.priority == entity.priority
        assert core_model.complexity == entity.complexity
        assert core_model.status == entity.status
        assert core_model.time_estimate_hours == entity.time_estimate_hours

    def test_task_core_to_entity_conversion(self):
        """Test TaskCore to TaskEntity conversion."""
        core_model = TaskCore(
            title="Core Task",
            description="Core description",
            component_area=ComponentArea.TESTING,
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.HIGH,
            status=TaskStatus.PARTIAL,
            time_estimate_hours=8.0,
        )

        entity = SchemaTransformer.task_core_to_entity(core_model)

        assert isinstance(entity, TaskEntity)
        assert entity.title == core_model.title
        assert entity.description == core_model.description
        assert entity.component_area == core_model.component_area
        assert entity.priority == core_model.priority
        assert entity.complexity == core_model.complexity
        assert entity.status == core_model.status
        assert entity.time_estimate_hours == core_model.time_estimate_hours

    def test_legacy_task_table_to_core_conversion(self):
        """Test legacy TaskTable to TaskCore conversion."""
        # Create a legacy task entity (same as TaskEntity but simulating legacy)
        legacy_task = TaskEntity(
            id=123,
            title="Legacy Task",
            description="Legacy description",
            component_area=ComponentArea.DATABASE,
            phase=2,
            priority=TaskPriority.LOW,
            complexity=TaskComplexity.VERY_HIGH,
            status=TaskStatus.BLOCKED,
            source_document="legacy_doc.md",
            success_criteria="Legacy criteria",
            time_estimate_hours=16.0,
            parent_task_id=456,
        )

        core_model = SchemaTransformer.legacy_task_table_to_core(legacy_task)

        assert isinstance(core_model, TaskCore)
        assert core_model.id == 123
        assert core_model.title == "Legacy Task"
        assert core_model.description == "Legacy description"
        assert core_model.component_area == ComponentArea.DATABASE
        assert core_model.phase == 2
        assert core_model.priority == TaskPriority.LOW
        assert core_model.complexity == TaskComplexity.VERY_HIGH
        assert core_model.status == TaskStatus.BLOCKED
        assert core_model.time_estimate_hours == 16.0
        assert core_model.parent_task_id == 456

    def test_task_core_to_legacy_table_format(self):
        """Test TaskCore to legacy table format conversion."""
        core_model = TaskCore(
            id=789,
            title="Modern Task",
            description="Modern description",
            component_area=ComponentArea.ARCHITECTURE,
            phase=3,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.LOW,
            status=TaskStatus.COMPLETED,
            source_document="modern_doc.md",
            success_criteria="Modern success",
            time_estimate_hours=2.0,
            parent_task_id=101,
        )

        legacy_dict = SchemaTransformer.task_core_to_legacy_table(core_model)

        assert isinstance(legacy_dict, dict)
        assert legacy_dict["id"] == 789
        assert legacy_dict["title"] == "Modern Task"
        assert legacy_dict["description"] == "Modern description"
        assert legacy_dict["component_area"] == ComponentArea.ARCHITECTURE.value
        assert legacy_dict["phase"] == 3
        assert legacy_dict["priority"] == TaskPriority.MEDIUM.value
        assert legacy_dict["complexity"] == TaskComplexity.LOW.value
        assert legacy_dict["status"] == TaskStatus.COMPLETED.value
        assert legacy_dict["time_estimate_hours"] == 2.0
        assert legacy_dict["parent_task_id"] == 101
        assert isinstance(legacy_dict["created_at"], datetime)
        assert isinstance(legacy_dict["updated_at"], datetime)

    def test_agent_report_to_execution_log_conversion(self):
        """Test AgentReport to TaskExecutionLog conversion."""
        report = AgentReport(
            agent_name=AgentType.CODING,
            task_id=555,
            status=TaskStatus.COMPLETED,
            success=True,
            execution_time_minutes=30.0,
            outputs={"lines_of_code": 250},
            artifacts=["main.go", "test.go"],
            confidence_score=0.95,
            error_details=None,
            created_at=datetime.now(),
        )

        execution_log = SchemaTransformer.agent_report_to_execution_log(report)

        assert isinstance(execution_log, TaskExecutionLog)
        assert execution_log.task_id == 555
        assert execution_log.agent_type == AgentType.CODING
        assert execution_log.status == TaskStatus.COMPLETED
        assert execution_log.outputs == {"lines_of_code": 250}
        assert execution_log.confidence_score == 0.95

    def test_execution_log_to_agent_report_conversion(self):
        """Test TaskExecutionLog to AgentReport conversion."""
        execution_log = TaskExecutionLog(
            task_id=777,
            agent_type=AgentType.TESTING,
            status=TaskStatus.PARTIAL,
            start_time=datetime.now(),
            end_time=datetime.now(),
            outputs={"tests_run": 45, "tests_passed": 43},
            confidence_score=0.88,
        )

        agent_report = SchemaTransformer.execution_log_to_agent_report(execution_log)

        assert isinstance(agent_report, AgentReport)
        assert agent_report.task_id == 777
        assert agent_report.agent_name == AgentType.TESTING
        assert agent_report.status == TaskStatus.PARTIAL
        assert agent_report.outputs == {"tests_run": 45, "tests_passed": 43}
        assert agent_report.confidence_score == 0.88

    def test_legacy_agent_report_to_unified_conversion(self):
        """Test legacy agent report dict to unified AgentReport conversion."""
        legacy_report = {
            "agent_name": "legacy_coding_agent",
            "agent_type": "coding",
            "task_id": 999,
            "status": "in_progress",
            "success": True,
            "confidence_score": 0.75,
            "outputs": {"progress": "50%"},
            "artifacts": ["code.py"],
            "next_actions": ["test", "deploy"],
            "issues_found": ["minor warning"],
            "error_details": None,
            "execution_time_minutes": 45.0,
        }

        unified_report = SchemaTransformer.legacy_agent_report_to_unified(legacy_report)

        assert isinstance(unified_report, AgentReport)
        assert unified_report.agent_name == AgentType.CODING  # Converted from string
        assert unified_report.task_id == 999
        assert unified_report.status == TaskStatus.IN_PROGRESS
        assert unified_report.success is True
        assert unified_report.confidence_score == 0.75
        assert unified_report.outputs == {"progress": "50%"}
        assert unified_report.artifacts == ["code.py"]
        assert unified_report.next_actions == ["test", "deploy"]
        assert unified_report.issues_found == ["minor warning"]
        assert unified_report.execution_time_minutes == 45.0

    def test_legacy_agent_report_with_defaults(self):
        """Test legacy agent report conversion with minimal data."""
        minimal_legacy = {"task_id": 111}

        unified_report = SchemaTransformer.legacy_agent_report_to_unified(
            minimal_legacy
        )

        # Check defaults are applied
        assert unified_report.agent_name == AgentType.CODING  # Default fallback
        assert unified_report.status == TaskStatus.COMPLETED
        assert unified_report.success is True
        assert unified_report.confidence_score == 0.8
        assert unified_report.outputs == {}
        assert unified_report.artifacts == []
        assert isinstance(unified_report.created_at, datetime)

    def test_create_progress_from_status_change(self):
        """Test creating TaskProgress from status change."""
        task_id = 888
        old_status = TaskStatus.NOT_STARTED
        new_status = TaskStatus.IN_PROGRESS
        notes = "Task started by user"
        updated_by = "user123"

        progress = SchemaTransformer.create_progress_from_status_change(
            task_id, old_status, new_status, notes, updated_by
        )

        assert isinstance(progress, TaskProgress)
        assert progress.task_id == 888
        assert progress.progress_percentage == 50  # IN_PROGRESS = 50%
        assert progress.notes == "Task started by user"
        assert progress.updated_by == "user123"

    def test_create_progress_from_status_change_with_default_notes(self):
        """Test creating TaskProgress with default notes."""
        progress = SchemaTransformer.create_progress_from_status_change(
            task_id=222,
            old_status=TaskStatus.IN_PROGRESS,
            new_status=TaskStatus.COMPLETED,
        )

        assert progress.progress_percentage == 100  # COMPLETED = 100%
        assert "Status changed from in_progress to completed" in progress.notes
        assert progress.updated_by == "system"


class TestBatchTransformer:
    """Test BatchTransformer bulk operations."""

    def test_tasks_entity_to_core_list_conversion(self):
        """Test bulk TaskEntity to TaskCore conversion."""
        entities = [
            TaskEntity(
                title=f"Entity Task {i}",
                priority=TaskPriority.LOW if i % 2 == 0 else TaskPriority.HIGH,
                complexity=TaskComplexity.MEDIUM,
                time_estimate_hours=i * 1.5,
            )
            for i in range(5)
        ]

        core_models = BatchTransformer.tasks_entity_to_core_list(entities)

        assert len(core_models) == 5
        for i, core_model in enumerate(core_models):
            assert isinstance(core_model, TaskCore)
            assert core_model.title == f"Entity Task {i}"
            assert core_model.time_estimate_hours == i * 1.5
            if i % 2 == 0:
                assert core_model.priority == TaskPriority.LOW
            else:
                assert core_model.priority == TaskPriority.HIGH

    def test_tasks_core_to_entity_list_conversion(self):
        """Test bulk TaskCore to TaskEntity conversion."""
        core_models = [
            TaskCore(
                title=f"Core Task {i}",
                description=f"Description {i}",
                component_area=ComponentArea.TESTING
                if i < 3
                else ComponentArea.SECURITY,
                complexity=TaskComplexity.HIGH,
                time_estimate_hours=i * 2.0,
            )
            for i in range(4)
        ]

        entities = BatchTransformer.tasks_core_to_entity_list(core_models)

        assert len(entities) == 4
        for i, entity in enumerate(entities):
            assert isinstance(entity, TaskEntity)
            assert entity.title == f"Core Task {i}"
            assert entity.description == f"Description {i}"
            assert entity.complexity == TaskComplexity.HIGH
            assert entity.time_estimate_hours == i * 2.0
            if i < 3:
                assert entity.component_area == ComponentArea.TESTING
            else:
                assert entity.component_area == ComponentArea.SECURITY

    def test_migrate_legacy_tasks_to_entities(self):
        """Test migration of legacy tasks to new entity format."""
        legacy_tasks = [
            TaskEntity(
                title="Legacy Task 1",
                priority=TaskPriority.CRITICAL,
                status=TaskStatus.BLOCKED,
                phase=1,
            ),
            TaskEntity(
                title="Legacy Task 2",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PARTIAL,
                phase=2,
            ),
        ]

        migrated_entities = BatchTransformer.migrate_legacy_tasks_to_entities(
            legacy_tasks
        )

        assert len(migrated_entities) == 2
        for i, entity in enumerate(migrated_entities):
            assert isinstance(entity, TaskEntity)
            assert entity.title == f"Legacy Task {i + 1}"
            # Verify values went through core model validation/transformation
            assert entity.priority in [TaskPriority.CRITICAL, TaskPriority.MEDIUM]
            assert entity.status in [TaskStatus.BLOCKED, TaskStatus.PARTIAL]

    def test_export_tasks_for_backup(self):
        """Test exporting tasks to JSON-serializable format."""
        tasks = [
            TaskCore(
                title="Backup Task 1",
                description="First backup task",
                priority=TaskPriority.HIGH,
                complexity=TaskComplexity.LOW,
            ),
            TaskCore(
                title="Backup Task 2",
                description="Second backup task",
                priority=TaskPriority.MEDIUM,
                complexity=TaskComplexity.VERY_HIGH,
            ),
        ]

        backup_data = BatchTransformer.export_tasks_for_backup(tasks)

        assert isinstance(backup_data, list)
        assert len(backup_data) == 2

        # First task
        assert backup_data[0]["title"] == "Backup Task 1"
        assert backup_data[0]["description"] == "First backup task"
        assert backup_data[0]["priority"] == TaskPriority.HIGH.value
        assert backup_data[0]["complexity"] == TaskComplexity.LOW.value

        # Second task
        assert backup_data[1]["title"] == "Backup Task 2"
        assert backup_data[1]["priority"] == TaskPriority.MEDIUM.value
        assert backup_data[1]["complexity"] == TaskComplexity.VERY_HIGH.value

    def test_import_tasks_from_backup(self):
        """Test importing tasks from backup data."""
        backup_data = [
            {
                "title": "Imported Task 1",
                "description": "First imported task",
                "component_area": ComponentArea.DEPENDENCIES.value,
                "priority": TaskPriority.CRITICAL.value,
                "complexity": TaskComplexity.HIGH.value,
                "status": TaskStatus.NOT_STARTED.value,
                "time_estimate_hours": 6.0,
            },
            {
                "title": "Imported Task 2",
                "description": "Second imported task",
                "component_area": ComponentArea.DOCUMENTATION.value,
                "priority": TaskPriority.LOW.value,
                "complexity": TaskComplexity.MEDIUM.value,
                "status": TaskStatus.COMPLETED.value,
                "time_estimate_hours": 3.5,
            },
        ]

        imported_tasks = BatchTransformer.import_tasks_from_backup(backup_data)

        assert len(imported_tasks) == 2

        # First task
        task1 = imported_tasks[0]
        assert isinstance(task1, TaskCore)
        assert task1.title == "Imported Task 1"
        assert task1.description == "First imported task"
        assert task1.component_area == ComponentArea.DEPENDENCIES
        assert task1.priority == TaskPriority.CRITICAL
        assert task1.complexity == TaskComplexity.HIGH
        assert task1.status == TaskStatus.NOT_STARTED
        assert task1.time_estimate_hours == 6.0

        # Second task
        task2 = imported_tasks[1]
        assert task2.title == "Imported Task 2"
        assert task2.component_area == ComponentArea.DOCUMENTATION
        assert task2.status == TaskStatus.COMPLETED
        assert task2.time_estimate_hours == 3.5

    def test_empty_batch_operations(self):
        """Test batch operations with empty lists."""
        # Empty entity to core conversion
        core_models = BatchTransformer.tasks_entity_to_core_list([])
        assert core_models == []

        # Empty core to entity conversion
        entities = BatchTransformer.tasks_core_to_entity_list([])
        assert entities == []

        # Empty backup export
        backup_data = BatchTransformer.export_tasks_for_backup([])
        assert backup_data == []

        # Empty backup import
        imported_tasks = BatchTransformer.import_tasks_from_backup([])
        assert imported_tasks == []


class TestLegacyCompatibilityLayer:
    """Test LegacyCompatibilityLayer for backward compatibility."""

    def test_adapt_task_for_legacy_agent(self):
        """Test adapting TaskCore for legacy agent interfaces."""
        task = TaskCore(
            id=333,
            title="Legacy Adapter Test",
            description="Testing legacy adaptation",
            status=TaskStatus.IN_PROGRESS,
            priority=TaskPriority.HIGH,
            success_criteria="All tests pass",
            time_estimate_hours=4.0,
        )

        adapted_task = LegacyCompatibilityLayer.adapt_task_for_legacy_agent(task)

        assert isinstance(adapted_task, dict)
        assert adapted_task["id"] == 333
        assert adapted_task["title"] == "Legacy Adapter Test"
        assert adapted_task["description"] == "Testing legacy adaptation"
        assert adapted_task["status"] == TaskStatus.IN_PROGRESS.value
        assert adapted_task["priority"] == TaskPriority.HIGH.value
        assert adapted_task["success_criteria"] == "All tests pass"
        assert adapted_task["time_estimate"] == 4.0
        # These are computed fields from TaskCore
        assert "is_actionable" in adapted_task
        assert "progress_percentage" in adapted_task

    def test_adapt_legacy_agent_result_success(self):
        """Test adapting successful legacy agent result."""
        legacy_result = {
            "agent_name": "legacy_test_agent",
            "agent_type": "testing",
            "success": True,
            "confidence_score": 0.92,
            "outputs": {"coverage": "95%", "tests_passed": 142},
            "files_created": ["test_report.html", "coverage.xml"],
            "recommendations": ["increase test coverage", "add edge cases"],
        }

        task_id = 666
        agent_report = LegacyCompatibilityLayer.adapt_legacy_agent_result(
            legacy_result, task_id
        )

        assert isinstance(agent_report, AgentReport)
        assert agent_report.task_id == 666
        assert agent_report.agent_name == "legacy_test_agent"
        assert agent_report.status == TaskStatus.COMPLETED  # Default for success
        assert agent_report.success is True
        assert agent_report.confidence_score == 0.92
        assert agent_report.outputs == {"coverage": "95%", "tests_passed": 142}
        # Legacy field mapping
        assert agent_report.artifacts == ["test_report.html", "coverage.xml"]
        assert agent_report.next_actions == ["increase test coverage", "add edge cases"]

    def test_adapt_legacy_agent_result_failure(self):
        """Test adapting failed legacy agent result."""
        legacy_result = {
            "agent_name": "legacy_coding_agent",
            "agent_type": "coding",
            "success": False,
            "confidence_score": 0.3,
            "outputs": {"compilation_errors": 5},
            "errors": ["syntax error in main.py", "undefined variable"],
            "files_created": [],
        }

        task_id = 777
        agent_report = LegacyCompatibilityLayer.adapt_legacy_agent_result(
            legacy_result, task_id
        )

        assert agent_report.task_id == 777
        assert agent_report.agent_name == "legacy_coding_agent"
        assert agent_report.status == TaskStatus.FAILED
        assert agent_report.success is False
        assert agent_report.confidence_score == 0.3
        assert "syntax error in main.py" in agent_report.error_details
        assert "undefined variable" in agent_report.error_details
        assert agent_report.artifacts == []

    def test_adapt_legacy_agent_result_defaults(self):
        """Test legacy agent result with minimal data uses defaults."""
        minimal_result = {}
        task_id = 888

        agent_report = LegacyCompatibilityLayer.adapt_legacy_agent_result(
            minimal_result, task_id
        )

        # Check defaults are applied
        assert agent_report.task_id == 888
        assert agent_report.agent_name == "legacy_agent"
        assert agent_report.status == TaskStatus.COMPLETED
        assert agent_report.success is True
        assert agent_report.confidence_score == 0.8
        assert agent_report.outputs == {}
        assert agent_report.artifacts == []
        assert agent_report.next_actions == []

    def test_create_legacy_task_dict(self):
        """Test creating legacy task dictionary format."""
        task = TaskCore(
            id=999,
            title="Legacy Dict Task",
            description="Task for legacy dict creation",
            component_area=ComponentArea.SERVICES,
            phase=4,
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.VERY_HIGH,
            status=TaskStatus.REQUIRES_ASSISTANCE,
            source_document="legacy_spec.md",
            success_criteria="Legacy system integration",
            time_estimate_hours=12.0,
            parent_task_id=100,
        )

        legacy_dict = LegacyCompatibilityLayer.create_legacy_task_dict(task)

        assert isinstance(legacy_dict, dict)
        assert legacy_dict["id"] == 999
        assert legacy_dict["title"] == "Legacy Dict Task"
        assert legacy_dict["description"] == "Task for legacy dict creation"
        assert legacy_dict["component_area"] == ComponentArea.SERVICES.value
        assert legacy_dict["phase"] == 4
        assert legacy_dict["priority"] == TaskPriority.CRITICAL.value
        assert legacy_dict["complexity"] == TaskComplexity.VERY_HIGH.value
        assert legacy_dict["status"] == TaskStatus.REQUIRES_ASSISTANCE.value
        assert legacy_dict["source_document"] == "legacy_spec.md"
        assert legacy_dict["success_criteria"] == "Legacy system integration"
        assert legacy_dict["time_estimate_hours"] == 12.0
        assert legacy_dict["parent_task_id"] == 100


class TestConvenienceFunctions:
    """Test convenience functions for common transformations."""

    def test_convert_to_core_model_from_entity(self):
        """Test convert_to_core_model with TaskEntity."""
        entity = TaskEntity(
            title="Convenience Entity",
            description="Testing convenience function",
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.HIGH,
        )

        core_model = convert_to_core_model(entity)

        assert isinstance(core_model, TaskCore)
        assert core_model.title == "Convenience Entity"
        assert core_model.description == "Testing convenience function"
        assert core_model.priority == TaskPriority.MEDIUM
        assert core_model.complexity == TaskComplexity.HIGH

    def test_convert_to_core_model_from_dict(self):
        """Test convert_to_core_model with dictionary."""
        task_dict = {
            "title": "Dict Task",
            "description": "From dictionary",
            "component_area": ComponentArea.UI.value,
            "priority": TaskPriority.LOW.value,
            "complexity": TaskComplexity.MEDIUM.value,
            "status": TaskStatus.NOT_STARTED.value,
            "time_estimate_hours": 3.0,
        }

        core_model = convert_to_core_model(task_dict)

        assert isinstance(core_model, TaskCore)
        assert core_model.title == "Dict Task"
        assert core_model.description == "From dictionary"
        assert core_model.component_area == ComponentArea.UI
        assert core_model.priority == TaskPriority.LOW
        assert core_model.complexity == TaskComplexity.MEDIUM
        assert core_model.status == TaskStatus.NOT_STARTED
        assert core_model.time_estimate_hours == 3.0

    def test_convert_to_core_model_invalid_type(self):
        """Test convert_to_core_model with invalid type."""
        with pytest.raises(ValueError, match="Unsupported type for conversion"):
            convert_to_core_model("invalid string input")

    def test_convert_to_entity_from_core_model(self):
        """Test convert_to_entity with TaskCore."""
        core_model = TaskCore(
            title="Core to Entity",
            description="Converting core to entity",
            component_area=ComponentArea.CONFIGURATION,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.LOW,
            time_estimate_hours=1.5,
        )

        entity = convert_to_entity(core_model)

        assert isinstance(entity, TaskEntity)
        assert entity.title == "Core to Entity"
        assert entity.description == "Converting core to entity"
        assert entity.component_area == ComponentArea.CONFIGURATION
        assert entity.priority == TaskPriority.HIGH
        assert entity.complexity == TaskComplexity.LOW
        assert entity.time_estimate_hours == 1.5

    def test_convert_to_entity_from_dict(self):
        """Test convert_to_entity with dictionary."""
        task_dict = {
            "title": "Dict to Entity",
            "description": "From dict to entity",
            "phase": 3,
            "priority": TaskPriority.CRITICAL.value,
            "complexity": TaskComplexity.VERY_HIGH.value,
            "time_estimate_hours": 20.0,
        }

        entity = convert_to_entity(task_dict)

        assert isinstance(entity, TaskEntity)
        assert entity.title == "Dict to Entity"
        assert entity.description == "From dict to entity"
        assert entity.phase == 3
        assert entity.priority == TaskPriority.CRITICAL
        assert entity.complexity == TaskComplexity.VERY_HIGH
        assert entity.time_estimate_hours == 20.0

    def test_convert_to_entity_invalid_type(self):
        """Test convert_to_entity with invalid type."""
        with pytest.raises(ValueError, match="Unsupported type for conversion"):
            convert_to_entity(12345)

    def test_validate_and_transform_success(self):
        """Test validate_and_transform with valid data."""
        task_data = {
            "title": "Validated Task",
            "description": "Successfully validated",
            "priority": TaskPriority.MEDIUM.value,
            "complexity": TaskComplexity.HIGH.value,
            "time_estimate_hours": 5.0,
        }

        validated_task = validate_and_transform(task_data, TaskCore)

        assert isinstance(validated_task, TaskCore)
        assert validated_task.title == "Validated Task"
        assert validated_task.description == "Successfully validated"
        assert validated_task.priority == TaskPriority.MEDIUM
        assert validated_task.complexity == TaskComplexity.HIGH
        assert validated_task.time_estimate_hours == 5.0

    def test_validate_and_transform_validation_error(self):
        """Test validate_and_transform with invalid data."""
        invalid_data = {
            "title": "",  # Invalid: too short
            "priority": "invalid_priority",  # Invalid enum value
            "time_estimate_hours": -1.0,  # Invalid: negative
        }

        with pytest.raises(ValidationError):
            validate_and_transform(invalid_data, TaskCore)


class TestTransformationErrorHandling:
    """Test error handling in transformation functions."""

    def test_legacy_agent_report_with_invalid_status(self):
        """Test legacy agent report with invalid status value."""
        legacy_report = {
            "agent_type": "coding",
            "task_id": 123,
            "status": "invalid_status_value",  # Invalid status
        }

        # Should handle gracefully and use default
        unified_report = SchemaTransformer.legacy_agent_report_to_unified(legacy_report)
        assert unified_report.status == TaskStatus.COMPLETED  # Default

    def test_batch_transformer_partial_failure(self):
        """Test batch transformer with mixed valid/invalid data."""
        mixed_data = [
            {
                "title": "Valid Task 1",
                "priority": TaskPriority.HIGH.value,
                "time_estimate_hours": 2.0,
            },
            {
                "title": "",  # Invalid: empty title
                "priority": TaskPriority.LOW.value,
                "time_estimate_hours": 1.0,
            },
            {
                "title": "Valid Task 2",
                "priority": TaskPriority.MEDIUM.value,
                "time_estimate_hours": 3.0,
            },
        ]

        # First and third should succeed, second should fail
        valid_tasks = []
        errors = []

        for task_data in mixed_data:
            try:
                task = validate_and_transform(task_data, TaskCore)
                valid_tasks.append(task)
            except ValidationError as e:
                errors.append(e)

        assert len(valid_tasks) == 2
        assert len(errors) == 1
        assert valid_tasks[0].title == "Valid Task 1"
        assert valid_tasks[1].title == "Valid Task 2"

    def test_transformation_type_consistency(self):
        """Test transformation maintains type consistency."""
        original_core = TaskCore(
            title="Type Consistency Test",
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MEDIUM,
        )

        # Core -> Entity -> Core round trip
        entity = convert_to_entity(original_core)
        restored_core = convert_to_core_model(entity)

        assert isinstance(original_core.priority, type(restored_core.priority))
        assert isinstance(original_core.complexity, type(restored_core.complexity))
        assert isinstance(original_core.status, type(restored_core.status))
        assert original_core.priority == restored_core.priority
        assert original_core.complexity == restored_core.complexity
        assert original_core.status == restored_core.status


class TestTransformationIntegration:
    """Test integration scenarios between different transformation functions."""

    def test_complete_migration_workflow(self):
        """Test complete migration from legacy to modern format."""
        # Start with legacy data
        legacy_tasks = [
            TaskEntity(
                title="Legacy Migration Task 1",
                description="First legacy task",
                priority=TaskPriority.HIGH,
                complexity=TaskComplexity.LOW,
                status=TaskStatus.NOT_STARTED,
            ),
            TaskEntity(
                title="Legacy Migration Task 2",
                description="Second legacy task",
                priority=TaskPriority.MEDIUM,
                complexity=TaskComplexity.HIGH,
                status=TaskStatus.IN_PROGRESS,
            ),
        ]

        # Step 1: Migrate to new entities
        migrated_entities = BatchTransformer.migrate_legacy_tasks_to_entities(
            legacy_tasks
        )

        # Step 2: Convert to core models
        core_models = BatchTransformer.tasks_entity_to_core_list(migrated_entities)

        # Step 3: Export for backup
        backup_data = BatchTransformer.export_tasks_for_backup(core_models)

        # Step 4: Import from backup (simulate restore)
        restored_tasks = BatchTransformer.import_tasks_from_backup(backup_data)

        # Verify complete round trip
        assert len(restored_tasks) == 2
        assert restored_tasks[0].title == "Legacy Migration Task 1"
        assert restored_tasks[1].title == "Legacy Migration Task 2"
        assert restored_tasks[0].priority == TaskPriority.HIGH
        assert restored_tasks[1].priority == TaskPriority.MEDIUM

    def test_agent_report_transformation_chain(self):
        """Test chain of agent report transformations."""
        # Start with legacy report
        legacy_report = {
            "agent_type": "research",
            "task_id": 456,
            "success": True,
            "outputs": {"findings": ["fact1", "fact2"]},
            "files_created": ["research.md"],
            "confidence_score": 0.87,
        }

        # Step 1: Convert to unified AgentReport
        unified_report = SchemaTransformer.legacy_agent_report_to_unified(legacy_report)

        # Step 2: Convert to TaskExecutionLog
        execution_log = SchemaTransformer.agent_report_to_execution_log(unified_report)

        # Step 3: Convert back to AgentReport
        restored_report = SchemaTransformer.execution_log_to_agent_report(execution_log)

        # Verify key data preserved through transformation chain
        assert restored_report.task_id == 456
        assert restored_report.outputs == {"findings": ["fact1", "fact2"]}
        assert restored_report.confidence_score == 0.87
        assert restored_report.success is True

    def test_legacy_compatibility_workflow(self):
        """Test complete legacy compatibility workflow."""
        # Modern TaskCore
        modern_task = TaskCore(
            id=789,
            title="Modern Task for Legacy",
            description="Testing legacy compatibility",
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.VERY_HIGH,
            status=TaskStatus.PARTIAL,
        )

        # Step 1: Adapt for legacy agent
        legacy_format = LegacyCompatibilityLayer.adapt_task_for_legacy_agent(
            modern_task
        )

        # Step 2: Simulate legacy agent processing (returns old format)
        legacy_result = {
            "agent_name": "legacy_processor",
            "task_id": legacy_format["id"],
            "success": True,
            "outputs": {"processed": True},
            "files_created": ["output.txt"],
        }

        # Step 3: Adapt legacy result back to modern format
        modern_report = LegacyCompatibilityLayer.adapt_legacy_agent_result(
            legacy_result, modern_task.id
        )

        # Verify compatibility maintained
        assert modern_report.task_id == modern_task.id
        assert modern_report.success is True
        assert modern_report.outputs == {"processed": True}
        assert modern_report.artifacts == ["output.txt"]

    def test_status_progress_tracking_workflow(self):
        """Test workflow combining status changes and progress tracking."""
        task_id = 555

        # Simulate task progression
        status_changes = [
            (TaskStatus.NOT_STARTED, TaskStatus.IN_PROGRESS, "Task started"),
            (TaskStatus.IN_PROGRESS, TaskStatus.PARTIAL, "Partial completion"),
            (TaskStatus.PARTIAL, TaskStatus.COMPLETED, "Task finished"),
        ]

        progress_records = []
        for old_status, new_status, notes in status_changes:
            progress = SchemaTransformer.create_progress_from_status_change(
                task_id, old_status, new_status, notes, "workflow_test"
            )
            progress_records.append(progress)

        # Verify progression
        assert len(progress_records) == 3
        assert progress_records[0].progress_percentage == 50  # IN_PROGRESS
        assert progress_records[1].progress_percentage == 75  # PARTIAL
        assert progress_records[2].progress_percentage == 100  # COMPLETED

        # Verify notes
        assert "Task started" in progress_records[0].notes
        assert "Partial completion" in progress_records[1].notes
        assert "Task finished" in progress_records[2].notes

    def test_data_validation_across_transformations(self):
        """Test data validation is maintained across all transformations."""
        # Create task with boundary values
        boundary_task = TaskCore(
            title="x",  # Minimum length
            time_estimate_hours=0.1,  # Minimum time
            phase=1,  # Minimum phase
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.VERY_HIGH,
        )

        # Transform through different formats
        entity = convert_to_entity(boundary_task)
        legacy_dict = LegacyCompatibilityLayer.create_legacy_task_dict(boundary_task)
        back_to_core = convert_to_core_model(entity)

        # Verify boundary values preserved
        assert entity.title == "x"
        assert entity.time_estimate_hours == 0.1
        assert entity.phase == 1

        assert legacy_dict["title"] == "x"
        assert legacy_dict["time_estimate_hours"] == 0.1
        assert legacy_dict["phase"] == 1

        assert back_to_core.title == "x"
        assert back_to_core.time_estimate_hours == 0.1
        assert back_to_core.phase == 1

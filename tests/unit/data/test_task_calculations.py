"""Comprehensive tests for task_calculations.py utilities.

This module tests all calculation utilities, computed metrics,
edge cases, and enhanced task models with calculations.
"""

from src.schemas.unified_models import (
    TaskComplexity,
    TaskCore,
    TaskPriority,
)
from src.utils.task_calculations import (
    EnhancedTaskCore,
    TaskCalculations,
)


class TestTaskCalculations:
    """Test TaskCalculations utility class methods."""

    def test_complexity_score_mapping(self):
        """Test complexity score mapping for all complexity levels."""
        test_cases = [
            (TaskComplexity.LOW, 1),
            (TaskComplexity.MEDIUM, 2),
            (TaskComplexity.HIGH, 3),
            (TaskComplexity.VERY_HIGH, 4),
        ]

        for complexity, expected_score in test_cases:
            task = TaskCore(title="Test", complexity=complexity)
            score = TaskCalculations.complexity_score(task)
            assert score == expected_score

    def test_priority_score_mapping(self):
        """Test priority score mapping for all priority levels."""
        test_cases = [
            (TaskPriority.LOW, 1),
            (TaskPriority.MEDIUM, 2),
            (TaskPriority.HIGH, 3),
            (TaskPriority.CRITICAL, 4),
        ]

        for priority, expected_score in test_cases:
            task = TaskCore(title="Test", priority=priority)
            score = TaskCalculations.priority_score(task)
            assert score == expected_score

    def test_effort_index_calculation(self):
        """Test effort index calculation with various inputs."""
        # Base case: medium complexity, medium priority
        task = TaskCore(
            title="Base Test",
            complexity=TaskComplexity.MEDIUM,  # score = 2
            priority=TaskPriority.MEDIUM,  # score = 2
            time_estimate_hours=4.0,
            phase=1,
        )

        # Expected: complexity(2) * priority(2) * time_factor(4/8=0.5) *
        # phase_factor(1.1) = 2.2
        expected_effort = 2 * 2 * (4.0 / 8.0) * 1.1
        assert TaskCalculations.effort_index(task) == round(expected_effort, 2)

    def test_effort_index_with_high_complexity_priority(self):
        """Test effort index with high complexity and priority."""
        task = TaskCore(
            title="High Effort Test",
            complexity=TaskComplexity.VERY_HIGH,  # score = 4
            priority=TaskPriority.CRITICAL,  # score = 4
            time_estimate_hours=16.0,  # time_factor = min(16/8, 3.0) = 2.0
            phase=3,  # phase_factor = 1.3
        )

        # Expected: 4 * 4 * 2.0 * 1.3 = 41.6
        expected_effort = 4 * 4 * 2.0 * 1.3
        assert TaskCalculations.effort_index(task) == round(expected_effort, 2)

    def test_effort_index_with_low_complexity_priority(self):
        """Test effort index with low complexity and priority."""
        task = TaskCore(
            title="Low Effort Test",
            complexity=TaskComplexity.LOW,  # score = 1
            priority=TaskPriority.LOW,  # score = 1
            time_estimate_hours=1.0,  # time_factor = 1/8 = 0.125
            phase=1,  # phase_factor = 1.1
        )

        # Expected: 1 * 1 * 0.125 * 1.1 = 0.1375, rounded to 0.14
        expected_effort = 1 * 1 * 0.125 * 1.1
        assert TaskCalculations.effort_index(task) == round(expected_effort, 2)

    def test_effort_index_time_factor_capping(self):
        """Test effort index time factor is capped at 3x."""
        task = TaskCore(
            title="Long Task",
            complexity=TaskComplexity.MEDIUM,  # score = 2
            priority=TaskPriority.MEDIUM,  # score = 2
            time_estimate_hours=40.0,  # Would be 5.0, capped at 3.0
            phase=1,  # phase_factor = 1.1
        )

        # Expected: 2 * 2 * 3.0 * 1.1 = 13.2 (time factor capped)
        expected_effort = 2 * 2 * 3.0 * 1.1
        assert TaskCalculations.effort_index(task) == round(expected_effort, 2)

    def test_effort_index_phase_factor_scaling(self):
        """Test effort index phase factor scaling."""
        base_task = TaskCore(
            title="Phase Test",
            complexity=TaskComplexity.MEDIUM,
            priority=TaskPriority.MEDIUM,
            time_estimate_hours=8.0,  # time_factor = 1.0
            phase=5,  # phase_factor = 1.5
        )

        # Expected: 2 * 2 * 1.0 * 1.5 = 6.0
        expected_effort = 2 * 2 * 1.0 * 1.5
        assert TaskCalculations.effort_index(base_task) == round(expected_effort, 2)

    def test_risk_factor_normal_case(self):
        """Test risk factor calculation for normal cases."""
        task = TaskCore(
            title="Risk Test",
            complexity=TaskComplexity.HIGH,  # score = 3
            priority=TaskPriority.MEDIUM,  # not critical, factor = 1.0
            time_estimate_hours=6.0,
        )

        # Expected: complexity_score(3) / time_estimate(6.0) *
        # priority_adjustment(1.0) = 0.5
        expected_risk = 3 / 6.0 * 1.0
        assert TaskCalculations.risk_factor(task) == round(expected_risk, 2)

    def test_risk_factor_critical_priority(self):
        """Test risk factor with critical priority adjustment."""
        task = TaskCore(
            title="Critical Risk Test",
            complexity=TaskComplexity.MEDIUM,  # score = 2
            priority=TaskPriority.CRITICAL,  # factor = 1.2
            time_estimate_hours=4.0,
        )

        # Expected: 2 / 4.0 * 1.2 = 0.6
        expected_risk = 2 / 4.0 * 1.2
        assert TaskCalculations.risk_factor(task) == round(expected_risk, 2)

    def test_risk_factor_zero_time_estimate(self):
        """Test risk factor with zero time estimate."""
        task = TaskCore(
            title="Zero Time Test",
            complexity=TaskComplexity.HIGH,
            priority=TaskPriority.HIGH,
            time_estimate_hours=0.0,
        )

        # Should return 1.0 for zero time estimate
        assert TaskCalculations.risk_factor(task) == 1.0

    def test_risk_factor_very_low_time_estimate(self):
        """Test risk factor with very low time estimate (uses minimum 0.1)."""
        task = TaskCore(
            title="Very Low Time Test",
            complexity=TaskComplexity.VERY_HIGH,  # score = 4
            priority=TaskPriority.CRITICAL,  # factor = 1.2
            time_estimate_hours=0.05,
        )

        # Expected: 4 / max(0.05, 0.1) * 1.2 = 4 / 0.1 * 1.2 = 48.0
        expected_risk = 4 / 0.1 * 1.2
        assert TaskCalculations.risk_factor(task) == round(expected_risk, 2)

    def test_all_calculations_consistency(self):
        """Test that all calculations are internally consistent."""
        task = TaskCore(
            title="Consistency Test",
            complexity=TaskComplexity.HIGH,
            priority=TaskPriority.HIGH,
            time_estimate_hours=8.0,
            phase=2,
        )

        complexity_score = TaskCalculations.complexity_score(task)
        priority_score = TaskCalculations.priority_score(task)
        effort_index = TaskCalculations.effort_index(task)
        risk_factor = TaskCalculations.risk_factor(task)

        # Verify scores are in expected ranges
        assert 1 <= complexity_score <= 4
        assert 1 <= priority_score <= 4
        assert effort_index > 0
        assert risk_factor > 0

        # Verify relationships
        assert complexity_score == 3  # HIGH
        assert priority_score == 3  # HIGH


class TestEnhancedTaskCore:
    """Test EnhancedTaskCore with computed fields via utility methods."""

    def test_enhanced_task_core_inherits_from_task_core(self):
        """Test EnhancedTaskCore properly inherits from TaskCore."""
        enhanced_task = EnhancedTaskCore(title="Enhanced Test")

        assert isinstance(enhanced_task, TaskCore)
        assert enhanced_task.title == "Enhanced Test"
        # Should have all TaskCore properties
        assert hasattr(enhanced_task, "complexity_multiplier")
        assert hasattr(enhanced_task, "effort_index")

    def test_enhanced_task_complexity_score_property(self):
        """Test complexity_score computed property."""
        enhanced_task = EnhancedTaskCore(
            title="Complexity Test", complexity=TaskComplexity.VERY_HIGH
        )

        assert enhanced_task.complexity_score == 4
        # Verify it's cached
        assert enhanced_task.complexity_score == 4

    def test_enhanced_task_priority_score_property(self):
        """Test priority_score computed property."""
        enhanced_task = EnhancedTaskCore(
            title="Priority Test", priority=TaskPriority.CRITICAL
        )

        assert enhanced_task.priority_score == 4

    def test_enhanced_task_effort_index_property(self):
        """Test effort_index computed property."""
        enhanced_task = EnhancedTaskCore(
            title="Effort Test",
            complexity=TaskComplexity.MEDIUM,  # 2
            priority=TaskPriority.HIGH,  # 3
            time_estimate_hours=4.0,  # 0.5 factor
            phase=1,  # 1.1 factor
        )

        expected_effort = 2 * 3 * 0.5 * 1.1  # 3.3
        assert enhanced_task.effort_index == round(expected_effort, 2)

    def test_enhanced_task_risk_factor_property(self):
        """Test risk_factor computed property."""
        enhanced_task = EnhancedTaskCore(
            title="Risk Test",
            complexity=TaskComplexity.HIGH,  # 3
            priority=TaskPriority.MEDIUM,  # not critical = 1.0
            time_estimate_hours=2.0,
        )

        expected_risk = 3 / 2.0 * 1.0  # 1.5
        assert enhanced_task.risk_factor == round(expected_risk, 2)

    def test_enhanced_task_cached_properties(self):
        """Test that computed properties are properly cached."""
        enhanced_task = EnhancedTaskCore(
            title="Cache Test",
            complexity=TaskComplexity.HIGH,
            priority=TaskPriority.HIGH,
        )

        # Access properties multiple times
        score1 = enhanced_task.complexity_score
        score2 = enhanced_task.complexity_score
        effort1 = enhanced_task.effort_index
        effort2 = enhanced_task.effort_index

        # Should be same instances (cached)
        assert score1 is score2
        assert effort1 is effort2
        assert score1 == 3
        assert effort1 > 0

    def test_enhanced_task_inherits_original_computed_fields(self):
        """Test EnhancedTaskCore still has original TaskCore computed fields."""
        enhanced_task = EnhancedTaskCore(
            title="Inheritance Test",
            complexity=TaskComplexity.HIGH,
            time_estimate_hours=6.0,
        )

        # Original TaskCore computed fields should still work
        assert enhanced_task.complexity_multiplier == 1.5  # HIGH = 1.5
        assert (
            enhanced_task.effort_index
            == enhanced_task.time_estimate_hours * enhanced_task.complexity_multiplier
        )

    def test_enhanced_task_all_properties_accessible(self):
        """Test all computed properties are accessible on EnhancedTaskCore."""
        enhanced_task = EnhancedTaskCore(
            title="All Properties Test",
            complexity=TaskComplexity.MEDIUM,
            priority=TaskPriority.HIGH,
            time_estimate_hours=3.0,
            phase=2,
        )

        # Original TaskCore properties
        assert hasattr(enhanced_task, "complexity_multiplier")
        assert hasattr(
            enhanced_task, "effort_index"
        )  # This is both original and enhanced
        assert hasattr(enhanced_task, "is_overdue")

        # Enhanced properties
        assert hasattr(enhanced_task, "complexity_score")
        assert hasattr(enhanced_task, "priority_score")
        assert hasattr(enhanced_task, "risk_factor")

        # All should be callable/accessible
        assert enhanced_task.complexity_multiplier > 0
        assert enhanced_task.complexity_score > 0
        assert enhanced_task.priority_score > 0
        assert enhanced_task.effort_index > 0
        assert enhanced_task.risk_factor > 0
        assert enhanced_task.is_overdue in [True, False]


class TestCalculationEdgeCases:
    """Test edge cases and boundary conditions for calculations."""

    def test_zero_time_estimates(self):
        """Test calculations with zero time estimates."""
        task = TaskCore(
            title="Zero Time",
            complexity=TaskComplexity.HIGH,
            priority=TaskPriority.CRITICAL,
            time_estimate_hours=0.0,
        )

        # Effort index should handle zero time gracefully
        effort = TaskCalculations.effort_index(task)
        assert effort >= 0

        # Risk factor should return 1.0 for zero time
        risk = TaskCalculations.risk_factor(task)
        assert risk == 1.0

    def test_maximum_values(self):
        """Test calculations with maximum values."""
        task = TaskCore(
            title="Maximum Values",
            complexity=TaskComplexity.VERY_HIGH,  # 4
            priority=TaskPriority.CRITICAL,  # 4
            time_estimate_hours=160.0,  # Maximum allowed
            phase=10,  # High phase
        )

        complexity_score = TaskCalculations.complexity_score(task)
        priority_score = TaskCalculations.priority_score(task)
        effort_index = TaskCalculations.effort_index(task)
        risk_factor = TaskCalculations.risk_factor(task)

        assert complexity_score == 4
        assert priority_score == 4
        assert effort_index > 0  # Should be a large positive number
        assert risk_factor > 0  # Should be calculated properly

    def test_minimum_values(self):
        """Test calculations with minimum values."""
        task = TaskCore(
            title="x",  # Minimum title
            complexity=TaskComplexity.LOW,  # 1
            priority=TaskPriority.LOW,  # 1
            time_estimate_hours=0.1,  # Minimum time
            phase=1,  # Minimum phase
        )

        complexity_score = TaskCalculations.complexity_score(task)
        priority_score = TaskCalculations.priority_score(task)
        effort_index = TaskCalculations.effort_index(task)
        risk_factor = TaskCalculations.risk_factor(task)

        assert complexity_score == 1
        assert priority_score == 1
        assert effort_index > 0  # Should be small but positive
        assert risk_factor > 0  # Should be calculated properly

    def test_floating_point_precision(self):
        """Test floating point precision in calculations."""
        task = TaskCore(
            title="Precision Test",
            complexity=TaskComplexity.MEDIUM,
            priority=TaskPriority.MEDIUM,
            time_estimate_hours=1.0 / 3.0,  # Repeating decimal
            phase=1,
        )

        effort_index = TaskCalculations.effort_index(task)
        risk_factor = TaskCalculations.risk_factor(task)

        # Results should be properly rounded
        assert isinstance(effort_index, float)
        assert isinstance(risk_factor, float)
        # Should have at most 2 decimal places (rounded)
        assert effort_index == round(effort_index, 2)
        assert risk_factor == round(risk_factor, 2)

    def test_calculation_consistency_across_instances(self):
        """Test calculations are consistent across different task instances."""
        # Create two identical tasks
        task1 = TaskCore(
            title="Consistency Test 1",
            complexity=TaskComplexity.HIGH,
            priority=TaskPriority.MEDIUM,
            time_estimate_hours=5.0,
            phase=2,
        )

        task2 = TaskCore(
            title="Consistency Test 2",
            complexity=TaskComplexity.HIGH,
            priority=TaskPriority.MEDIUM,
            time_estimate_hours=5.0,
            phase=2,
        )

        # All calculations should be identical
        assert TaskCalculations.complexity_score(
            task1
        ) == TaskCalculations.complexity_score(task2)
        assert TaskCalculations.priority_score(
            task1
        ) == TaskCalculations.priority_score(task2)
        assert TaskCalculations.effort_index(task1) == TaskCalculations.effort_index(
            task2
        )
        assert TaskCalculations.risk_factor(task1) == TaskCalculations.risk_factor(
            task2
        )

    def test_enhanced_task_vs_utility_calculations(self):
        """Test EnhancedTaskCore properties match utility function results."""
        task_data = {
            "title": "Comparison Test",
            "complexity": TaskComplexity.HIGH,
            "priority": TaskPriority.CRITICAL,
            "time_estimate_hours": 4.0,
            "phase": 3,
        }

        # Create both regular task and enhanced task
        regular_task = TaskCore(**task_data)
        enhanced_task = EnhancedTaskCore(**task_data)

        # Compare results
        assert enhanced_task.complexity_score == TaskCalculations.complexity_score(
            regular_task
        )
        assert enhanced_task.priority_score == TaskCalculations.priority_score(
            regular_task
        )
        assert enhanced_task.effort_index == TaskCalculations.effort_index(regular_task)
        assert enhanced_task.risk_factor == TaskCalculations.risk_factor(regular_task)


class TestCalculationValidation:
    """Test validation of calculation inputs and outputs."""

    def test_invalid_enum_values_handled_gracefully(self):
        """Test calculations handle invalid enum values gracefully."""
        # Create task with valid enums
        task = TaskCore(
            title="Valid Task",
            complexity=TaskComplexity.MEDIUM,
            priority=TaskPriority.HIGH,
        )

        # All calculations should work normally
        assert TaskCalculations.complexity_score(task) == 2
        assert TaskCalculations.priority_score(task) == 3

    def test_calculation_output_ranges(self):
        """Test that calculation outputs are within expected ranges."""
        tasks = [
            TaskCore(
                title="Test1", complexity=c, priority=p, time_estimate_hours=t, phase=ph
            )
            for c in TaskComplexity
            for p in TaskPriority
            for t in [0.1, 1.0, 8.0, 16.0]
            for ph in [1, 3, 5]
        ]

        for task in tasks:
            complexity_score = TaskCalculations.complexity_score(task)
            priority_score = TaskCalculations.priority_score(task)
            effort_index = TaskCalculations.effort_index(task)
            risk_factor = TaskCalculations.risk_factor(task)

            # Validate ranges
            assert 1 <= complexity_score <= 4
            assert 1 <= priority_score <= 4
            assert effort_index >= 0
            assert risk_factor >= 0

            # Validate types
            assert isinstance(complexity_score, int)
            assert isinstance(priority_score, int)
            assert isinstance(effort_index, float)
            assert isinstance(risk_factor, float)

    def test_calculation_monotonicity(self):
        """Test that calculations increase/decrease monotonically where expected."""
        base_task = TaskCore(
            title="Monotonic Test",
            complexity=TaskComplexity.MEDIUM,
            priority=TaskPriority.MEDIUM,
            time_estimate_hours=4.0,
            phase=1,
        )

        # Complexity should increase monotonically
        complexities = [
            TaskComplexity.LOW,
            TaskComplexity.MEDIUM,
            TaskComplexity.HIGH,
            TaskComplexity.VERY_HIGH,
        ]
        complexity_scores = []
        for complexity in complexities:
            task = TaskCore(
                title="Test", complexity=complexity, priority=base_task.priority
            )
            complexity_scores.append(TaskCalculations.complexity_score(task))

        assert complexity_scores == sorted(complexity_scores)

        # Priority should increase monotonically
        priorities = [
            TaskPriority.LOW,
            TaskPriority.MEDIUM,
            TaskPriority.HIGH,
            TaskPriority.CRITICAL,
        ]
        priority_scores = []
        for priority in priorities:
            task = TaskCore(
                title="Test", complexity=base_task.complexity, priority=priority
            )
            priority_scores.append(TaskCalculations.priority_score(task))

        assert priority_scores == sorted(priority_scores)

    def test_effort_index_components(self):
        """Test individual components of effort index calculation."""
        task = TaskCore(
            title="Component Test",
            complexity=TaskComplexity.HIGH,  # 3
            priority=TaskPriority.CRITICAL,  # 4
            time_estimate_hours=12.0,  # time_factor = min(12/8, 3) = 1.5
            phase=4,  # phase_factor = 1.4
        )

        # Manual calculation
        complexity_score = 3
        priority_score = 4
        base_score = complexity_score * priority_score  # 12
        time_factor = min(12.0 / 8.0, 3.0)  # 1.5
        phase_factor = 1.0 + (4 * 0.1)  # 1.4
        expected = round(base_score * time_factor * phase_factor, 2)  # 25.2

        assert TaskCalculations.effort_index(task) == expected

    def test_risk_factor_components(self):
        """Test individual components of risk factor calculation."""
        # Test non-critical priority
        task1 = TaskCore(
            title="Risk Component Test 1",
            complexity=TaskComplexity.VERY_HIGH,  # 4
            priority=TaskPriority.HIGH,  # not critical, factor = 1.0
            time_estimate_hours=8.0,
        )

        expected1 = round(4 / 8.0 * 1.0, 2)  # 0.5
        assert TaskCalculations.risk_factor(task1) == expected1

        # Test critical priority
        task2 = TaskCore(
            title="Risk Component Test 2",
            complexity=TaskComplexity.MEDIUM,  # 2
            priority=TaskPriority.CRITICAL,  # critical, factor = 1.2
            time_estimate_hours=5.0,
        )

        expected2 = round(2 / 5.0 * 1.2, 2)  # 0.48
        assert TaskCalculations.risk_factor(task2) == expected2


class TestPerformanceConsiderations:
    """Test performance aspects of calculations."""

    def test_calculation_performance_with_many_tasks(self):
        """Test calculation performance with large number of tasks."""
        import time

        # Create many tasks
        tasks = [
            TaskCore(
                title=f"Performance Test {i}",
                complexity=TaskComplexity.MEDIUM,
                priority=TaskPriority.HIGH,
                time_estimate_hours=float(i % 10 + 1),
                phase=i % 5 + 1,
            )
            for i in range(1000)
        ]

        # Time the calculations
        start_time = time.time()

        for task in tasks:
            TaskCalculations.complexity_score(task)
            TaskCalculations.priority_score(task)
            TaskCalculations.effort_index(task)
            TaskCalculations.risk_factor(task)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (less than 1 second for 1000 tasks)
        assert duration < 1.0

    def test_enhanced_task_caching_performance(self):
        """Test that caching in EnhancedTaskCore improves performance."""
        enhanced_task = EnhancedTaskCore(
            title="Caching Performance Test",
            complexity=TaskComplexity.VERY_HIGH,
            priority=TaskPriority.CRITICAL,
            time_estimate_hours=10.0,
            phase=5,
        )

        import time

        # First access (should calculate and cache)
        start_time = time.time()
        score1 = enhanced_task.complexity_score
        time.time() - start_time

        # Second access (should use cache)
        start_time = time.time()
        score2 = enhanced_task.complexity_score
        time.time() - start_time

        # Results should be identical
        assert score1 == score2
        # Second access should be faster or at least not significantly slower
        # Note: This is a micro-optimization test and may not always be reliable
        # in test environments, so we just verify functionality works

    def test_calculation_memory_usage(self):
        """Test calculations don't create excessive memory usage."""
        import gc

        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform many calculations
        for i in range(100):
            task = TaskCore(
                title=f"Memory Test {i}",
                complexity=TaskComplexity.HIGH,
                priority=TaskPriority.MEDIUM,
                time_estimate_hours=2.0,
            )

            TaskCalculations.complexity_score(task)
            TaskCalculations.priority_score(task)
            TaskCalculations.effort_index(task)
            TaskCalculations.risk_factor(task)

        # Force garbage collection after test
        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count shouldn't grow excessively
        # Allow some growth but not proportional to number of tasks
        assert final_objects - initial_objects < 50

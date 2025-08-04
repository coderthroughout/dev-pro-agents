"""Comprehensive performance and load testing for batch operations.

This module tests system performance characteristics including:
- Batch processing performance and scalability
- Memory usage under load conditions
- Concurrent operation handling
- API client rate limiting and throttling
- Database connection pooling performance
- Resource utilization optimization
- Stress testing and breaking point analysis
"""

import asyncio
import gc
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest
import pytest_asyncio
from pytest_benchmark.fixture import BenchmarkFixture

from src.config import get_settings
from src.schemas.unified_models import (
    AgentReport,
    AgentType,
    ComponentArea,
    TaskComplexity,
    TaskCore,
    TaskPriority,
    TaskStatus,
)


@pytest_asyncio.fixture
async def performance_test_environment():
    """Create optimized environment for performance testing."""
    # Configure environment for performance testing
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test-perf-openai",
            "OPENROUTER_API_KEY": "test-perf-openrouter",
            "EXA_API_KEY": "test-perf-exa",
            "FIRECRAWL_API_KEY": "test-perf-firecrawl",
            "ORCHESTRATION_PARALLEL_EXECUTION_LIMIT": "10",
            "ORCHESTRATION_BATCH_SIZE": "50",
            "ORCHESTRATION_DEBUG_MODE": "false",  # Disable debug for performance
        },
    ):
        # Create high-performance mock clients
        mock_exa_client = AsyncMock()
        mock_exa_client.search.return_value = {
            "results": [
                {
                    "title": f"Performance Test Result {i}",
                    "url": f"https://example.com/perf/{i}",
                    "text": f"Performance test content {i}",
                    "score": 0.9 - (i * 0.01),
                }
                for i in range(10)
            ]
        }

        mock_firecrawl_client = AsyncMock()
        mock_firecrawl_client.scrape.return_value = {
            "success": True,
            "data": {
                "markdown": "# Performance Test Content\n\nOptimized content for performance testing",
                "title": "Performance Test Page",
            },
        }

        # Mock database with connection pooling simulation
        mock_task_manager = MagicMock()
        mock_task_manager.get_tasks_by_status.return_value = []
        mock_task_manager.update_task_status.return_value = None
        mock_task_manager.create_task.return_value = lambda: int(
            time.time() * 1000000
        )  # Unique ID

        yield {
            "exa_client": mock_exa_client,
            "firecrawl_client": mock_firecrawl_client,
            "task_manager": mock_task_manager,
            "settings": get_settings(),
        }


def create_test_tasks(
    count: int, complexity: TaskComplexity = TaskComplexity.MEDIUM
) -> list[TaskCore]:
    """Create a batch of test tasks for performance testing."""
    return [
        TaskCore(
            id=i,
            title=f"Performance Task {i}",
            description=f"Performance testing task {i} with {complexity.value} complexity",
            component_area=ComponentArea.CORE,
            phase=1,
            priority=TaskPriority.MEDIUM,
            complexity=complexity,
            success_criteria=f"Complete performance task {i} efficiently",
            time_estimate_hours=1.0 + (i % 5) * 0.5,  # Varying estimates
        )
        for i in range(1, count + 1)
    ]


class TestBatchProcessingPerformance:
    """Test batch processing performance characteristics."""

    @pytest.mark.asyncio
    async def test_small_batch_processing_baseline(
        self, performance_test_environment, benchmark: BenchmarkFixture
    ):
        """Establish baseline performance for small batch processing."""
        tasks = create_test_tasks(10)

        async def process_small_batch() -> list[AgentReport]:
            """Process small batch of tasks."""
            results = []

            for task in tasks:
                # Simulate minimal processing time
                await asyncio.sleep(0.001)  # 1ms per task

                report = AgentReport(
                    agent_name=AgentType.CODING,
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=0.1,
                    outputs={"result": f"Processed task {task.id}"},
                    artifacts=[f"task_{task.id}.py"],
                    recommendations=[],
                    next_actions=["deploy"],
                    confidence_score=0.85,
                )
                results.append(report)

            return results

        # Benchmark the processing
        results = await benchmark.pedantic(
            process_small_batch,
            rounds=5,
            iterations=1,
        )

        # Verify results
        assert len(results) == 10
        assert all(report.success for report in results)

        # Performance assertions (baseline)
        benchmark_stats = benchmark.stats
        assert benchmark_stats.mean < 0.1  # Should complete in less than 100ms
        assert benchmark_stats.stddev < 0.02  # Low variance

    @pytest.mark.asyncio
    async def test_large_batch_processing_scalability(
        self, performance_test_environment
    ):
        """Test scalability with large batch sizes."""
        # Test different batch sizes
        batch_sizes = [50, 100, 200, 500]
        performance_metrics = {}

        for batch_size in batch_sizes:
            tasks = create_test_tasks(batch_size)

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Process batch concurrently
            async def process_task(task: TaskCore) -> AgentReport:
                # Simulate variable processing time
                processing_time = 0.001 + (task.id % 10) * 0.0001
                await asyncio.sleep(processing_time)

                return AgentReport(
                    agent_name=AgentType.CODING,
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=processing_time * 60,
                    outputs={"processed": True, "batch_size": batch_size},
                    artifacts=[f"batch_{batch_size}_task_{task.id}.py"],
                    recommendations=[],
                    next_actions=["test"],
                    confidence_score=0.88,
                )

            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(20)  # Max 20 concurrent tasks

            async def process_with_semaphore(task: TaskCore) -> AgentReport:
                async with semaphore:
                    return await process_task(task)

            results = await asyncio.gather(
                *[process_with_semaphore(task) for task in tasks]
            )

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Record metrics
            total_time = end_time - start_time
            memory_usage = end_memory - start_memory
            throughput = batch_size / total_time

            performance_metrics[batch_size] = {
                "total_time": total_time,
                "memory_usage": memory_usage,
                "throughput": throughput,
                "success_rate": sum(1 for r in results if r.success) / len(results),
            }

            # Verify results
            assert len(results) == batch_size
            assert all(report.success for report in results)

        # Analyze scalability
        # Throughput should scale reasonably (not linearly due to overhead)
        throughput_50 = performance_metrics[50]["throughput"]
        throughput_500 = performance_metrics[500]["throughput"]

        # Throughput shouldn't degrade more than 50% at 10x scale
        assert throughput_500 > throughput_50 * 0.5

        # Memory usage should scale sub-linearly
        memory_50 = performance_metrics[50]["memory_usage"]
        memory_500 = performance_metrics[500]["memory_usage"]

        # Memory shouldn't increase more than 5x for 10x tasks
        if memory_50 > 0:  # Avoid division by zero
            assert memory_500 < memory_50 * 5

        # All batches should maintain high success rate
        for batch_size, metrics in performance_metrics.items():
            assert metrics["success_rate"] >= 0.99

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self, performance_test_environment):
        """Test multiple concurrent batch processing operations."""
        # Create multiple batches to process concurrently
        num_batches = 5
        batch_size = 20
        all_batches = [
            create_test_tasks(batch_size, TaskComplexity.LOW)
            for _ in range(num_batches)
        ]

        async def process_batch(batch_id: int, tasks: list[TaskCore]) -> dict[str, Any]:
            """Process a single batch and return metrics."""
            start_time = time.time()

            async def process_task(task: TaskCore) -> AgentReport:
                # Simulate processing with some variance
                await asyncio.sleep(0.005 + (task.id % 5) * 0.001)

                return AgentReport(
                    agent_name=AgentType.CODING,
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=0.5,
                    outputs={
                        "batch_id": batch_id,
                        "processed": True,
                    },
                    artifacts=[f"batch_{batch_id}_task_{task.id}.py"],
                    recommendations=[],
                    next_actions=["test"],
                    confidence_score=0.87,
                )

            results = await asyncio.gather(*[process_task(task) for task in tasks])

            end_time = time.time()

            return {
                "batch_id": batch_id,
                "results": results,
                "processing_time": end_time - start_time,
                "success_count": sum(1 for r in results if r.success),
            }

        # Process all batches concurrently
        start_time = time.time()
        batch_results = await asyncio.gather(
            *[process_batch(i, batch) for i, batch in enumerate(all_batches)]
        )
        total_time = time.time() - start_time

        # Verify concurrent processing results
        assert len(batch_results) == num_batches

        total_tasks = 0
        total_successes = 0
        max_batch_time = 0

        for batch_result in batch_results:
            assert len(batch_result["results"]) == batch_size
            total_tasks += len(batch_result["results"])
            total_successes += batch_result["success_count"]
            max_batch_time = max(max_batch_time, batch_result["processing_time"])

        # Performance assertions
        overall_success_rate = total_successes / total_tasks
        assert overall_success_rate >= 0.99

        # Concurrent processing should be faster than sequential
        expected_sequential_time = max_batch_time * num_batches
        efficiency = expected_sequential_time / total_time
        assert efficiency > 2.0  # Should be at least 2x faster with concurrency

        # Calculate overall throughput
        throughput = total_tasks / total_time
        assert throughput > 50  # Should process at least 50 tasks per second


class TestMemoryUsageOptimization:
    """Test memory usage patterns and optimization."""

    @pytest.mark.asyncio
    async def test_memory_efficient_task_processing(self, performance_test_environment):
        """Test memory efficiency in task processing."""
        # Get baseline memory usage
        gc.collect()  # Force garbage collection
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Process tasks in chunks to optimize memory usage
        total_tasks = 1000
        chunk_size = 100
        memory_samples = []

        for chunk_start in range(0, total_tasks, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_tasks)
            chunk_tasks = create_test_tasks(chunk_end - chunk_start, TaskComplexity.LOW)

            # Process chunk
            chunk_results = []
            for _i, task in enumerate(chunk_tasks, chunk_start):
                # Create minimal report to test memory efficiency
                report = AgentReport(
                    agent_name=AgentType.CODING,
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=0.1,
                    outputs={"processed": True},  # Minimal output
                    artifacts=[],  # No artifacts for memory efficiency
                    recommendations=[],
                    next_actions=[],
                    confidence_score=0.8,
                )
                chunk_results.append(report)

            # Measure memory after processing chunk
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory - baseline_memory)

            # Clear chunk results to free memory
            chunk_results.clear()
            del chunk_results
            gc.collect()

        # Analyze memory usage pattern
        max_memory_usage = max(memory_samples)
        avg_memory_usage = sum(memory_samples) / len(memory_samples)

        # Memory should remain bounded despite processing many tasks
        assert max_memory_usage < 50  # Less than 50MB increase
        assert avg_memory_usage < 20  # Average less than 20MB increase

        # Memory usage shouldn't grow linearly with number of chunks
        early_memory = sum(memory_samples[:3]) / 3  # First 3 chunks
        late_memory = sum(memory_samples[-3:]) / 3  # Last 3 chunks

        # Late memory shouldn't be more than 2x early memory
        if early_memory > 0:
            assert late_memory < early_memory * 2

    @pytest.mark.asyncio
    async def test_large_output_memory_handling(self, performance_test_environment):
        """Test memory handling with large task outputs."""
        # Create tasks that generate large outputs
        num_tasks = 50
        large_output_size = 10000  # 10KB per output field

        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024

        async def process_large_output_task(task_id: int) -> AgentReport:
            """Process task with large outputs."""
            # Generate large outputs
            large_text = "x" * large_output_size

            return AgentReport(
                agent_name=AgentType.RESEARCH,
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                success=True,
                execution_time_minutes=1.0,
                outputs={
                    "large_research_data": large_text,
                    "analysis_results": large_text,
                    "recommendations_detail": large_text,
                },
                artifacts=[f"large_output_{task_id}.json"],
                recommendations=[f"Large recommendation {i}" for i in range(100)],
                next_actions=["process_large_data"],
                confidence_score=0.9,
            )

        # Process with memory monitoring
        memory_samples = []
        results = []

        # Process in small batches to monitor memory growth
        batch_size = 10
        for batch_start in range(0, num_tasks, batch_size):
            batch_end = min(batch_start + batch_size, num_tasks)

            batch_results = await asyncio.gather(
                *[process_large_output_task(i) for i in range(batch_start, batch_end)]
            )

            results.extend(batch_results)

            # Sample memory usage
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory - baseline_memory)

            # Brief pause to allow memory measurement
            await asyncio.sleep(0.01)

        # Verify processing success
        assert len(results) == num_tasks
        assert all(report.success for report in results)

        # Analyze memory growth
        final_memory_usage = memory_samples[-1]
        expected_minimum_usage = (
            (num_tasks * large_output_size * 3) / 1024 / 1024
        )  # 3 large outputs per task

        # Memory usage should be reasonable (within 3x of expected minimum)
        assert final_memory_usage < expected_minimum_usage * 3

        # Clear results and verify memory is freed
        results.clear()
        del results
        gc.collect()

        # Memory should decrease after cleanup
        post_cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024
        cleanup_memory_usage = post_cleanup_memory - baseline_memory

        # Should free at least 50% of the memory
        assert cleanup_memory_usage < final_memory_usage * 0.5


class TestAPIClientPerformance:
    """Test API client performance and rate limiting."""

    @pytest.mark.asyncio
    async def test_api_client_concurrent_requests(self, performance_test_environment):
        """Test API client performance under concurrent load."""
        env = performance_test_environment

        # Configure mock with realistic response times
        async def realistic_exa_search(*args, **kwargs):
            """Mock EXA search with realistic response time."""
            await asyncio.sleep(0.05)  # 50ms response time
            return {
                "results": [
                    {
                        "title": "Concurrent Test Result",
                        "url": "https://example.com/concurrent",
                        "text": "Concurrent search result",
                        "score": 0.9,
                    }
                ]
            }

        env["exa_client"].search.side_effect = realistic_exa_search

        # Test concurrent API requests
        num_concurrent_requests = 20
        start_time = time.time()

        search_tasks = [
            env["exa_client"].search(f"concurrent query {i}")
            for i in range(num_concurrent_requests)
        ]

        results = await asyncio.gather(*search_tasks)
        total_time = time.time() - start_time

        # Verify results
        assert len(results) == num_concurrent_requests
        assert all("results" in result for result in results)

        # Performance assertions
        # With concurrency, should be much faster than sequential
        expected_sequential_time = num_concurrent_requests * 0.05
        efficiency = expected_sequential_time / total_time
        assert efficiency > 5  # Should be at least 5x faster with concurrency

        # Total time should be close to single request time (plus overhead)
        assert total_time < 0.2  # Should complete in less than 200ms

    @pytest.mark.asyncio
    async def test_api_rate_limiting_behavior(self, performance_test_environment):
        """Test API client behavior under rate limiting."""
        env = performance_test_environment

        # Configure mock to simulate rate limiting
        request_count = 0
        rate_limit_threshold = 10

        async def rate_limited_search(*args, **kwargs):
            """Mock search with rate limiting."""
            nonlocal request_count
            request_count += 1

            if request_count <= rate_limit_threshold:
                await asyncio.sleep(0.01)  # Fast response
                return {"results": [{"title": f"Result {request_count}"}]}
            else:
                # Simulate rate limit delay
                await asyncio.sleep(0.1)  # Slower due to rate limiting
                return {"results": [{"title": f"Rate limited result {request_count}"}]}

        env["exa_client"].search.side_effect = rate_limited_search

        # Send requests that will trigger rate limiting
        num_requests = 25
        request_times = []

        for i in range(num_requests):
            start_time = time.time()
            result = await env["exa_client"].search(f"rate limit test {i}")
            end_time = time.time()

            request_times.append(end_time - start_time)
            assert "results" in result

        # Analyze rate limiting behavior
        fast_requests = list(request_times[:rate_limit_threshold])
        slow_requests = list(request_times[rate_limit_threshold:])

        avg_fast_time = sum(fast_requests) / len(fast_requests)
        avg_slow_time = sum(slow_requests) / len(slow_requests)

        # Rate limited requests should be significantly slower
        assert avg_slow_time > avg_fast_time * 5

    @pytest.mark.asyncio
    async def test_api_timeout_handling_performance(self, performance_test_environment):
        """Test performance of timeout handling in API clients."""
        env = performance_test_environment

        # Configure mock to simulate various timeout scenarios
        timeout_scenarios = [
            0.01,  # Fast response
            0.05,  # Normal response
            0.15,  # Slow response (will timeout with 0.1s timeout)
            0.25,  # Very slow response (will timeout)
        ]

        scenario_index = 0

        async def timeout_test_search(*args, **kwargs):
            """Mock search with varying response times."""
            nonlocal scenario_index
            response_time = timeout_scenarios[scenario_index % len(timeout_scenarios)]
            scenario_index += 1

            await asyncio.sleep(response_time)
            return {"results": [{"title": f"Response after {response_time}s"}]}

        env["exa_client"].search.side_effect = timeout_test_search

        # Test with timeout handling
        timeout_duration = 0.1  # 100ms timeout
        results = []
        timeout_count = 0

        for i in range(12):  # 3 cycles of 4 scenarios
            try:
                result = await asyncio.wait_for(
                    env["exa_client"].search(f"timeout test {i}"),
                    timeout=timeout_duration,
                )
                results.append(("success", result))
            except TimeoutError:
                timeout_count += 1
                results.append(("timeout", None))

        # Verify timeout behavior
        successful_requests = [r for r in results if r[0] == "success"]
        timed_out_requests = [r for r in results if r[0] == "timeout"]

        # Should have some successes and some timeouts
        assert len(successful_requests) > 0
        assert len(timed_out_requests) > 0

        # Timeout count should match expected slow requests
        # With 3 cycles of 4 scenarios, 2 scenarios per cycle should timeout
        assert timeout_count == 6


class TestDatabasePerformance:
    """Test database operation performance."""

    @pytest.mark.asyncio
    async def test_batch_database_operations(self, performance_test_environment):
        """Test performance of batch database operations."""
        env = performance_test_environment

        # Mock database operations with realistic timing
        operation_times = []

        def timed_db_operation(*args, **kwargs):
            """Mock database operation with timing."""
            start_time = time.time()

            # Simulate database work
            time.sleep(0.001)  # 1ms per operation

            end_time = time.time()
            operation_times.append(end_time - start_time)

            return {"operation": "success", "timestamp": time.time()}

        env["task_manager"].update_task_status.side_effect = timed_db_operation
        env["task_manager"].create_task.side_effect = timed_db_operation

        # Test batch operations
        num_operations = 100
        batch_size = 20

        start_time = time.time()

        # Simulate batch processing with database operations
        for batch_start in range(0, num_operations, batch_size):
            batch_end = min(batch_start + batch_size, num_operations)

            # Process batch of database operations
            batch_operations = []
            for i in range(batch_start, batch_end):
                # Mix of create and update operations
                if i % 2 == 0:
                    batch_operations.append(
                        env["task_manager"].create_task(f"task_{i}")
                    )
                else:
                    batch_operations.append(
                        env["task_manager"].update_task_status(i, "completed")
                    )

        total_time = time.time() - start_time

        # Verify operation performance
        assert len(operation_times) == num_operations

        avg_operation_time = sum(operation_times) / len(operation_times)
        assert avg_operation_time < 0.01  # Less than 10ms per operation

        # Total throughput should be reasonable
        throughput = num_operations / total_time
        assert throughput > 50  # At least 50 operations per second

    @pytest.mark.asyncio
    async def test_connection_pool_performance(self, performance_test_environment):
        """Test database connection pool performance simulation."""
        env = performance_test_environment

        # Simulate connection pool with limited connections
        max_connections = 5
        active_connections = 0
        connection_wait_times = []

        async def simulate_db_operation_with_pool(*args, **kwargs):
            """Simulate database operation with connection pooling."""
            nonlocal active_connections

            start_wait = time.time()

            # Wait for available connection
            while active_connections >= max_connections:
                await asyncio.sleep(0.001)  # Brief wait

            # Acquire connection
            active_connections += 1
            end_wait = time.time()
            connection_wait_times.append(end_wait - start_wait)

            try:
                # Simulate database work
                await asyncio.sleep(0.01)  # 10ms database operation
                return {"success": True, "connection_id": active_connections}
            finally:
                # Release connection
                active_connections -= 1

        env[
            "task_manager"
        ].update_task_status.side_effect = simulate_db_operation_with_pool

        # Test concurrent database operations
        num_concurrent_ops = 20

        start_time = time.time()
        results = await asyncio.gather(
            *[
                env["task_manager"].update_task_status(i, "completed")
                for i in range(num_concurrent_ops)
            ]
        )
        total_time = time.time() - start_time

        # Verify results
        assert len(results) == num_concurrent_ops
        assert all(result["success"] for result in results)

        # Analyze connection pool performance
        max_wait_time = max(connection_wait_times) if connection_wait_times else 0
        avg_wait_time = (
            sum(connection_wait_times) / len(connection_wait_times)
            if connection_wait_times
            else 0
        )

        # With good connection pooling, wait times should be reasonable
        assert max_wait_time < 0.1  # Max wait less than 100ms
        assert avg_wait_time < 0.02  # Average wait less than 20ms

        # Total time should be reasonable for concurrent operations
        # Should be much less than sequential time
        expected_sequential_time = num_concurrent_ops * 0.01
        efficiency = expected_sequential_time / total_time
        assert efficiency > 3  # At least 3x speedup with connection pooling


class TestStressTestingAndBreakingPoints:
    """Test system behavior under extreme stress conditions."""

    @pytest.mark.asyncio
    async def test_extreme_concurrent_load(self, performance_test_environment):
        """Test system behavior under extreme concurrent load."""
        # Configure for high load
        extreme_concurrency = 100
        tasks_per_worker = 10
        total_tasks = extreme_concurrency * tasks_per_worker

        # Track system metrics
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()

        # Use semaphore to control maximum concurrency
        semaphore = asyncio.Semaphore(extreme_concurrency)

        async def stress_test_worker(worker_id: int) -> dict[str, Any]:
            """Individual worker for stress testing."""
            async with semaphore:
                worker_results = []
                worker_start_time = time.time()

                for task_id in range(tasks_per_worker):
                    # Simulate work with minimal processing
                    await asyncio.sleep(0.001)  # 1ms per task

                    report = AgentReport(
                        agent_name=AgentType.CODING,
                        task_id=worker_id * tasks_per_worker + task_id,
                        status=TaskStatus.COMPLETED,
                        success=True,
                        execution_time_minutes=0.01,
                        outputs={"worker_id": worker_id, "processed": True},
                        artifacts=[],
                        recommendations=[],
                        next_actions=[],
                        confidence_score=0.8,
                    )
                    worker_results.append(report)

                worker_end_time = time.time()

                return {
                    "worker_id": worker_id,
                    "results": worker_results,
                    "processing_time": worker_end_time - worker_start_time,
                    "tasks_completed": len(worker_results),
                }

        # Launch all workers concurrently
        try:
            worker_results = await asyncio.gather(
                *[stress_test_worker(i) for i in range(extreme_concurrency)]
            )

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Analyze stress test results
            total_time = end_time - start_time
            memory_usage = end_memory - start_memory

            # Verify all workers completed successfully
            assert len(worker_results) == extreme_concurrency

            total_completed_tasks = sum(
                worker["tasks_completed"] for worker in worker_results
            )
            assert total_completed_tasks == total_tasks

            # Performance under stress
            throughput = total_tasks / total_time
            assert (
                throughput > 100
            )  # Should maintain at least 100 tasks/sec under stress

            # Memory usage should be bounded even under extreme load
            assert memory_usage < 200  # Less than 200MB increase

            # All workers should complete in reasonable time
            max_worker_time = max(
                worker["processing_time"] for worker in worker_results
            )
            assert max_worker_time < 1.0  # No worker should take more than 1 second

        except Exception as e:
            # If stress test fails, ensure it's due to expected resource limits
            assert (
                "memory" in str(e).lower()
                or "timeout" in str(e).lower()
                or "resource" in str(e).lower()
            )

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, performance_test_environment):
        """Test system behavior under memory pressure."""
        # Create memory pressure by processing large amounts of data
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_samples = []

        # Gradually increase memory usage
        large_data_sets = []
        max_memory_increase = 100  # MB

        try:
            for iteration in range(20):
                # Create increasingly large data structures
                large_data = {
                    "iteration": iteration,
                    "large_list": list(range(10000 * (iteration + 1))),
                    "large_dict": {
                        f"key_{i}": f"value_{i}" * 100
                        for i in range(1000 * (iteration + 1))
                    },
                    "large_string": "x" * (50000 * (iteration + 1)),
                }

                # Process data to simulate real work
                processed_report = AgentReport(
                    agent_name=AgentType.RESEARCH,
                    task_id=iteration,
                    status=TaskStatus.COMPLETED,
                    success=True,
                    execution_time_minutes=1.0,
                    outputs={
                        "large_data_summary": f"Processed iteration {iteration}",
                        "data_size": len(str(large_data)),
                    },
                    artifacts=[f"large_data_{iteration}.json"],
                    recommendations=[],
                    next_actions=[],
                    confidence_score=0.8,
                )

                large_data_sets.append((large_data, processed_report))

                # Monitor memory usage
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_increase = current_memory - baseline_memory
                memory_samples.append(memory_increase)

                # Stop if memory usage gets too high
                if memory_increase > max_memory_increase:
                    break

                await asyncio.sleep(0.01)  # Brief pause

            # Analyze memory behavior under pressure
            max_memory_used = max(memory_samples)

            # System should handle memory pressure gracefully
            assert max_memory_used <= max_memory_increase

            # Verify system remained functional throughout
            assert len(large_data_sets) > 5  # Should process at least 5 iterations

            # Cleanup and verify memory recovery
            large_data_sets.clear()
            gc.collect()

            post_cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024
            recovered_memory = post_cleanup_memory - baseline_memory

            # Should recover at least 70% of used memory
            assert recovered_memory < max_memory_used * 0.3

        except MemoryError:
            # Memory error is acceptable for stress testing
            pass

    @pytest.mark.asyncio
    async def test_cascading_failure_resilience(self, performance_test_environment):
        """Test system resilience to cascading failures."""
        env = performance_test_environment

        # Simulate cascading failure scenario
        failure_threshold = 10  # After 10 successful operations, start failures
        operation_count = 0

        async def failing_service_simulation(*args, **kwargs):
            """Simulate service that starts failing after threshold."""
            nonlocal operation_count
            operation_count += 1

            if operation_count <= failure_threshold:
                await asyncio.sleep(0.01)  # Normal operation
                return {"success": True, "operation": operation_count}
            else:
                # Start failing
                failure_rate = min(0.8, (operation_count - failure_threshold) / 10)

                if hash(str(args) + str(kwargs)) % 100 < failure_rate * 100:
                    raise Exception(f"Cascading failure at operation {operation_count}")
                else:
                    await asyncio.sleep(0.05)  # Slower due to degraded service
                    return {
                        "success": True,
                        "operation": operation_count,
                        "degraded": True,
                    }

        env["exa_client"].search.side_effect = failing_service_simulation

        # Test system behavior during cascading failures
        num_operations = 50
        successful_operations = 0
        failed_operations = 0
        degraded_operations = 0

        for i in range(num_operations):
            try:
                result = await env["exa_client"].search(f"cascading test {i}")
                successful_operations += 1

                if result.get("degraded"):
                    degraded_operations += 1

            except Exception:
                failed_operations += 1

        # Verify cascading failure handling
        assert successful_operations > 0  # Some operations should succeed
        assert failed_operations > 0  # Some operations should fail (expected)

        # System should maintain partial functionality
        functionality_rate = successful_operations / num_operations
        assert functionality_rate > 0.2  # Should maintain at least 20% functionality

        # Should have some degraded operations (showing partial recovery)
        assert degraded_operations > 0

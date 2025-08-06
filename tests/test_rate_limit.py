"""Tests for rate limiting and concurrency utilities."""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import logging

from invoice_pdf.core.rate_limit import (
    CapacityLimiter,
    RetryError,
    retry_with_backoff,
    with_capacity_limit,
    with_retry,
    RateLimitedExecutor,
    create_gemini_executor,
    create_pdf_executor
)


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return MagicMock(spec=logging.Logger)


class TestCapacityLimiter:
    """Test CapacityLimiter functionality."""
    
    def test_capacity_limiter_init(self):
        """Test CapacityLimiter initialization."""
        limiter = CapacityLimiter(5)
        assert limiter.total_tokens == 5
        assert limiter.available_tokens <= 5
        assert limiter.borrowed_tokens >= 0
    
    @pytest.mark.asyncio
    async def test_capacity_limiter_context_manager(self):
        """Test CapacityLimiter as async context manager."""
        limiter = CapacityLimiter(2)
        initial_available = limiter.available_tokens
        
        async with limiter:
            # Should have one less token available
            assert limiter.available_tokens == initial_available - 1
            assert limiter.borrowed_tokens == 1
        
        # Should be back to original after exit
        assert limiter.available_tokens == initial_available
        assert limiter.borrowed_tokens == 0
    
    @pytest.mark.asyncio
    async def test_capacity_limiter_concurrent_access(self):
        """Test concurrent access to capacity limiter."""
        limiter = CapacityLimiter(2)
        results = []
        
        async def worker(worker_id: int):
            async with limiter:
                results.append(f"started_{worker_id}")
                await asyncio.sleep(0.1)  # Small delay
                results.append(f"finished_{worker_id}")
        
        # Start 3 workers with capacity of 2
        tasks = [worker(i) for i in range(3)]
        await asyncio.gather(*tasks)
        
        # All should complete
        assert len(results) == 6
        assert results.count("started_0") == 1
        assert results.count("finished_0") == 1


class TestRetryWithBackoff:
    """Test retry_with_backoff functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_operation(self, mock_logger):
        """Test successful operation without retries."""
        async def successful_op():
            return "success"
        
        result = await retry_with_backoff(successful_op, logger=mock_logger)
        assert result == "success"
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_operation_with_retries(self, mock_logger):
        """Test operation that fails then succeeds."""
        call_count = 0
        
        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await retry_with_backoff(
                flaky_op, 
                max_retries=3, 
                logger=mock_logger,
                operation_name="test_op"
            )
        
        assert result == "success"
        assert call_count == 3
        assert mock_logger.warning.call_count == 2  # 2 retries before success
    
    @pytest.mark.asyncio
    async def test_exhausted_retries(self, mock_logger):
        """Test operation that fails all retry attempts."""
        async def failing_op():
            raise ValueError("Persistent error")
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(RetryError) as exc_info:
                await retry_with_backoff(
                    failing_op,
                    max_retries=2,
                    logger=mock_logger,
                    operation_name="failing_test"
                )
        
        # Verify RetryError contains expected information
        error = exc_info.value
        assert error.operation_name == "failing_test"
        assert error.attempts == 2
        assert isinstance(error.last_exception, ValueError)
        
        assert mock_logger.warning.call_count == 1  # One retry
        mock_logger.error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_custom_retry_exceptions(self, mock_logger):
        """Test custom retry exception filtering."""
        async def selective_fail():
            raise KeyError("Not retryable")
        
        with pytest.raises(RetryError) as exc_info:
            await retry_with_backoff(
                selective_fail,
                retry_exceptions=(ValueError,),  # Only retry ValueError
                logger=mock_logger
            )
        
        # Should not retry KeyError, should raise RetryError immediately
        error = exc_info.value
        assert isinstance(error.last_exception, KeyError)
        assert error.attempts == 1  # No retries for non-retryable exception
        mock_logger.warning.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_backoff_calculation(self, mock_logger):
        """Test exponential backoff calculation."""
        async def failing_op():
            raise ValueError("Always fails")
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RetryError):
                await retry_with_backoff(
                    failing_op,
                    max_retries=3,
                    base_delay=1.0,
                    max_delay=5.0,
                    jitter_range=0.0,  # No jitter for predictable testing
                    logger=mock_logger
                )
        
        # Should have called sleep with exponential backoff
        calls = mock_sleep.call_args_list
        assert len(calls) == 2  # 2 retries
        
        # First retry: base_delay * 2^0 = 1.0
        assert calls[0][0][0] == 1.0
        
        # Second retry: base_delay * 2^1 = 2.0
        assert calls[1][0][0] == 2.0


class TestDecorators:
    """Test decorator functions."""
    
    @pytest.mark.asyncio
    async def test_with_capacity_limit_decorator(self):
        """Test capacity limiting decorator."""
        limiter = CapacityLimiter(1)
        
        @with_capacity_limit(limiter)
        async def limited_func(value: str) -> str:
            return f"processed_{value}"
        
        result = await limited_func("test")
        assert result == "processed_test"
    
    @pytest.mark.asyncio
    async def test_with_retry_decorator(self, mock_logger):
        """Test retry decorator."""
        call_count = 0
        
        @with_retry(max_retries=2, operation_name="decorated_op")
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Flaky error")
            return "success"
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await flaky_func()
        
        assert result == "success"
        assert call_count == 2


class TestRateLimitedExecutor:
    """Test RateLimitedExecutor functionality."""
    
    @pytest.mark.asyncio
    async def test_executor_successful_operation(self):
        """Test executor with successful operation."""
        executor = RateLimitedExecutor(capacity=2, max_retries=2)
        
        async def test_op():
            return "executor_success"
        
        result = await executor.execute(test_op, operation_name="test")
        assert result == "executor_success"
    
    @pytest.mark.asyncio
    async def test_executor_with_retries(self):
        """Test executor with retry logic."""
        executor = RateLimitedExecutor(capacity=2, max_retries=3)
        call_count = 0
        
        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Executor flaky error")
            return "executor_retry_success"
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await executor.execute(flaky_op, operation_name="executor_test")
        
        assert result == "executor_retry_success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_executor_stats(self):
        """Test executor statistics."""
        executor = RateLimitedExecutor(capacity=5)
        stats = executor.stats
        
        assert "available_capacity" in stats
        assert "borrowed_capacity" in stats
        assert "total_capacity" in stats
        assert stats["total_capacity"] == 5
        assert stats["available_capacity"] <= 5
        assert stats["borrowed_capacity"] >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_executor_operations(self):
        """Test executor with concurrent operations."""
        executor = RateLimitedExecutor(capacity=2)
        results = []
        
        async def concurrent_op(op_id: int):
            await asyncio.sleep(0.1)
            results.append(f"op_{op_id}")
            return f"result_{op_id}"
        
        # Run 4 operations with capacity of 2
        tasks = [
            executor.execute(lambda: concurrent_op(i), operation_name=f"concurrent_{i}")
            for i in range(4)
        ]
        
        completed_results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(completed_results) == 4
        assert len(results) == 4
        assert all(result.startswith("result_") for result in completed_results)


class TestFactoryFunctions:
    """Test factory functions for common executors."""
    
    def test_create_gemini_executor(self):
        """Test Gemini executor factory."""
        executor = create_gemini_executor(quota_limit=15)
        
        assert executor.limiter.total_tokens == 15
        assert executor.max_retries == 3
        assert executor.base_delay == 2.0
        assert executor.max_delay == 10.0
        assert executor.jitter_range == 3.0
    
    def test_create_pdf_executor(self):
        """Test PDF executor factory."""
        executor = create_pdf_executor(fd_limit=100)
        
        assert executor.limiter.total_tokens == 100
        assert executor.max_retries == 2
        assert executor.base_delay == 1.0
        assert executor.max_delay == 5.0
        assert executor.jitter_range == 1.0


@pytest.mark.asyncio
async def test_integration_scenario():
    """Integration test combining capacity limiting and retry logic."""
    executor = RateLimitedExecutor(capacity=2, max_retries=3)
    success_count = 0
    
    async def integration_op(op_id: int):
        nonlocal success_count
        # Simulate some operations failing
        if op_id % 3 == 0 and success_count < 2:
            success_count += 1
            raise ValueError(f"Transient error for op_{op_id}")
        return f"integration_success_{op_id}"
    
    # Run multiple operations concurrently
    with patch('asyncio.sleep', new_callable=AsyncMock):
        tasks = [
            executor.execute(
                lambda i=i: integration_op(i), 
                operation_name=f"integration_{i}"
            )
            for i in range(6)
        ]
        
        results = await asyncio.gather(*tasks)
    
    # Should have some successful results
    successful_results = [r for r in results if r is not None]
    assert len(successful_results) > 0
    assert all("integration_success_" in str(r) for r in successful_results)
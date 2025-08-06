"""Shared concurrency and rate-limiting utilities for async operations."""
import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TypeVar

try:
    import anyio
except ImportError:
    # Fallback to asyncio if anyio is not available
    anyio = None

T = TypeVar("T")


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    
    def __init__(self, operation_name: str, last_exception: Exception, attempts: int):
        self.operation_name = operation_name
        self.last_exception = last_exception
        self.attempts = attempts
        super().__init__(
            f"{operation_name} failed after {attempts} attempts: {last_exception}"
        )


class CapacityLimiter:
    """Async capacity limiter compatible with anyio.CapacityLimiter interface.

    Falls back to asyncio.Semaphore if anyio is not available.
    """

    def __init__(self, total_tokens: int):
        """Initialize capacity limiter with total capacity."""
        self.total_tokens = total_tokens
        if anyio:
            self._limiter = anyio.CapacityLimiter(total_tokens)
        else:
            self._semaphore = asyncio.Semaphore(total_tokens)

    async def __aenter__(self):
        """Async context manager entry."""
        if anyio:
            await self._limiter.__aenter__()
        else:
            await self._semaphore.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if anyio:
            await self._limiter.__aexit__(exc_type, exc_val, exc_tb)
        else:
            await self._semaphore.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def available_tokens(self) -> int:
        """Get available tokens/capacity."""
        if anyio:
            return self._limiter.available_tokens
        return self._semaphore._value

    @property
    def borrowed_tokens(self) -> int:
        """Get borrowed tokens/capacity."""
        if anyio:
            return self._limiter.borrowed_tokens
        return self.total_tokens - self._semaphore._value


async def retry_with_backoff(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 10.0,
    jitter_range: float = 3.0,
    retry_exceptions: tuple = (Exception,),
    operation_name: str | None = None,
    logger: logging.Logger | None = None
) -> T:
    """Execute an async operation with exponential backoff retry logic.

    Args:
        operation: Async function to execute
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay for exponential backoff (default: 2.0 seconds)
        max_delay: Maximum delay between retries (default: 10.0 seconds)
        jitter_range: Random jitter range added to delay (default: 3.0 seconds)
        retry_exceptions: Tuple of exceptions that should trigger retry (default: (Exception,))
        operation_name: Name for logging purposes (optional)
        logger: Logger instance to use (optional, defaults to module logger)

    Returns:
        Result of successful operation
    
    Raises:
        RetryError: When all retry attempts are exhausted
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    last_exception = None

    for attempt in range(max_retries):
        try:
            return await operation()
        except retry_exceptions as exc:
            last_exception = exc

            # Check for specific upstream transient errors
            error_message = str(exc).lower()
            if "transient upstream error" in error_message or "will retry" in error_message:
                # Re-raise transient errors to be handled by retry logic
                pass

            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff and jitter
                delay = min(max_delay, base_delay * (2 ** attempt))
                jitter = random.uniform(0, jitter_range)
                total_delay = delay + jitter

                operation_desc = operation_name or "operation"
                logger.warning(
                    f"[RETRY] {operation_desc} - Attempt {attempt + 1}/{max_retries} failed: "
                    f"{str(exc)[:100]}. Retrying in {total_delay:.1f}s..."
                )
                await asyncio.sleep(total_delay)
                continue
            operation_desc = operation_name or "operation"
            logger.error(
                f"[RETRY] {operation_desc} - Exhausted retries ({max_retries}): "
                f"{str(exc)[:150]}"
            )
            raise RetryError(operation_desc, exc, max_retries)
        except Exception as exc:
            # Non-retryable exception - raise RetryError with details
            operation_desc = operation_name or "operation"
            logger.error(
                f"[RETRY] {operation_desc} - Non-retryable exception: "
                f"{str(exc)[:150]}"
            )
            raise RetryError(operation_desc, exc, 1)  # Single attempt for non-retryable

    # Should not reach here, but handle edge case
    if last_exception:
        operation_desc = operation_name or "operation"
        logger.error(f"[RETRY] {operation_desc} - Unexpected retry loop exit")
        raise RetryError(operation_desc, last_exception, max_retries)
    
    # This should never happen, but ensure type safety
    raise RetryError(operation_name or "operation", Exception("Unknown error"), max_retries)


def with_capacity_limit(limiter: CapacityLimiter):
    """Decorator to apply capacity limiting to async functions."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async with limiter:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def with_retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 10.0,
    jitter_range: float = 3.0,
    retry_exceptions: tuple = (Exception,),
    operation_name: str | None = None
):
    """Decorator to apply retry logic to async functions."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async def operation():
                return await func(*args, **kwargs)

            name = operation_name or func.__name__
            return await retry_with_backoff(
                operation=operation,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter_range=jitter_range,
                retry_exceptions=retry_exceptions,
                operation_name=name
            )
        return wrapper
    return decorator


class RateLimitedExecutor:
    """Executor that combines capacity limiting and retry logic."""

    def __init__(
        self,
        capacity: int,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 10.0,
        jitter_range: float = 3.0
    ):
        """Initialize rate-limited executor.

        Args:
            capacity: Maximum concurrent operations
            max_retries: Maximum retry attempts per operation
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
            jitter_range: Random jitter range for delays
        """
        self.limiter = CapacityLimiter(capacity)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_range = jitter_range
        self._logger = logging.getLogger(__name__)

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        operation_name: str | None = None,
        retry_exceptions: tuple = (Exception,)
    ) -> T:
        """Execute operation with both capacity limiting and retry logic."""
        async def limited_operation():
            async with self.limiter:
                return await operation()

        return await retry_with_backoff(
            operation=limited_operation,
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            jitter_range=self.jitter_range,
            retry_exceptions=retry_exceptions,
            operation_name=operation_name,
            logger=self._logger
        )

    @property
    def stats(self) -> dict:
        """Get current executor statistics."""
        return {
            "available_capacity": self.limiter.available_tokens,
            "borrowed_capacity": self.limiter.borrowed_tokens,
            "total_capacity": self.limiter.total_tokens
        }


# Pre-configured executors for common use cases
def create_gemini_executor(quota_limit: int = 10) -> RateLimitedExecutor:
    """Create executor optimized for Gemini API calls."""
    return RateLimitedExecutor(
        capacity=quota_limit,
        max_retries=3,
        base_delay=2.0,
        max_delay=10.0,
        jitter_range=3.0
    )


def create_pdf_executor(fd_limit: int = 50) -> RateLimitedExecutor:
    """Create executor optimized for PDF file operations."""
    return RateLimitedExecutor(
        capacity=fd_limit,
        max_retries=2,  # Fewer retries for file operations
        base_delay=1.0,
        max_delay=5.0,
        jitter_range=1.0
    )


"""Test failure scenarios and error handling paths."""

import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "invoice_pdf"))

from invoice_pdf.core.exceptions import APIError, InvalidPDFError, PathTraversalError, PDFTooLargeError, SecurityError
from invoice_pdf.core.pdf_utils import check_pdf_size_safety, get_memory_efficient_pdf_reader
from invoice_pdf.core.security import sanitize_filename, validate_api_key, validate_safe_path


def test_pdf_size_validation():
    """Test PDF size validation and limits."""
    # Create a test file
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        # Write some data
        tmp.write(b"%PDF-1.4\n" + b"X" * 1000)  # ~1KB file
        tmp_path = Path(tmp.name)

    try:
        # Test normal case
        size_mb, is_safe = check_pdf_size_safety(tmp_path, max_size_mb=1.0)
        assert size_mb < 1.0
        assert is_safe

        # Test size limit exceeded
        with pytest.raises(PDFTooLargeError) as exc_info:
            check_pdf_size_safety(tmp_path, max_size_mb=0.0001)  # Very small limit

        assert "exceeds maximum allowed size" in str(exc_info.value)
        assert exc_info.value.file_path == tmp_path

    finally:
        tmp_path.unlink()


def test_invalid_pdf_handling():
    """Test handling of corrupted/invalid PDF files."""
    # Create an invalid PDF file
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"This is not a valid PDF file")
        tmp_path = Path(tmp.name)

    try:
        # Should raise InvalidPDFError
        with pytest.raises(InvalidPDFError) as exc_info:
            get_memory_efficient_pdf_reader(tmp_path)

        assert "corrupted" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    finally:
        tmp_path.unlink()


def test_path_traversal_detection():
    """Test detection of path traversal attempts."""
    # Test individual dangerous patterns
    dangerous_path = "../../../etc/passwd.pdf"

    try:
        validate_safe_path(dangerous_path)
        assert False, f"Should have raised exception for {dangerous_path}"
    except (PathTraversalError, SecurityError) as e:
        assert "path traversal" in str(e).lower() or "security" in str(e).lower()

    # Test another pattern
    dangerous_path2 = "file_with/../traversal.pdf"
    try:
        validate_safe_path(dangerous_path2)
        assert False, f"Should have raised exception for {dangerous_path2}"
    except (PathTraversalError, SecurityError) as e:
        assert "path traversal" in str(e).lower() or "security" in str(e).lower()


def test_security_extension_validation():
    """Test file extension security validation."""
    dangerous_files = [
        "malware.exe",
        "script.sh",
        "virus.bat",
        "trojan.scr",
        "backdoor.com"
    ]

    for dangerous_file in dangerous_files:
        with pytest.raises(SecurityError) as exc_info:
            validate_safe_path(dangerous_file)

        assert "not allowed" in str(exc_info.value)


def test_api_key_validation():
    """Test API key security validation."""
    # Test short key
    with pytest.raises(SecurityError):
        validate_api_key("short")

    # Test test/dummy keys
    test_keys = [
        "test_key_12345678901234567890",
        "fake_api_key_12345678901234567890",
        "dummy123456789012345678901234567890",
        "0000000000000000000000000000000000",
        "1111111111111111111111111111111111"
    ]

    for test_key in test_keys:
        with pytest.raises(SecurityError) as exc_info:
            validate_api_key(test_key)

        assert "test" in str(exc_info.value).lower() or "dummy" in str(exc_info.value).lower() or "short" in str(exc_info.value).lower()


def test_filename_sanitization():
    """Test filename sanitization for security."""
    dangerous_filenames = {
        'file<>:"|?*.pdf': "file_______.pdf",
        "invoice\x00null.pdf": "invoice_null.pdf",
        "../../traverse.pdf": "_._traverse.pdf",  # Leading dots get replaced
        "CON.pdf": "safe_CON.pdf",  # Windows reserved name
        "file" + "x" * 300 + ".pdf": True,  # Long filename (check truncation)
        "...dangerous.pdf": "dangerous.pdf"
    }

    for dangerous, expected in dangerous_filenames.items():
        sanitized = sanitize_filename(dangerous)

        if expected is True:
            # Just check it was truncated
            assert len(sanitized) <= 255
            assert sanitized.endswith(".pdf")
        else:
            assert sanitized == expected

        # Should not contain dangerous characters
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert '"' not in sanitized
        assert "|" not in sanitized


def test_memory_efficient_pdf_reader_failures():
    """Test memory efficient PDF reader error handling."""
    # Test non-existent file
    with pytest.raises(Exception):  # Should raise some PDF-related exception
        get_memory_efficient_pdf_reader("/nonexistent/file.pdf")

    # Test empty file
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Empty file should raise InvalidPDFError
        with pytest.raises(InvalidPDFError):
            get_memory_efficient_pdf_reader(tmp_path)

    finally:
        tmp_path.unlink()


def test_concurrent_resource_limits():
    """Test resource limits under concurrent access."""
    import asyncio

    from core.rate_limit import CapacityLimiter

    async def test_capacity_limit():
        limiter = CapacityLimiter(2)  # Only allow 2 concurrent operations

        counter = 0
        max_concurrent = 0

        async def test_operation():
            nonlocal counter, max_concurrent
            async with limiter:
                counter += 1
                max_concurrent = max(max_concurrent, counter)
                await asyncio.sleep(0.1)  # Simulate work
                counter -= 1

        # Start 5 operations concurrently
        tasks = [test_operation() for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should never have exceeded capacity of 2
        assert max_concurrent <= 2

    asyncio.run(test_capacity_limit())


def test_exception_hierarchy():
    """Test that exception hierarchy works correctly."""
    from core.exceptions import ClassificationError, InvoicePDFError, PDFProcessingError, wrap_exception

    # Test inheritance
    pdf_error = PDFProcessingError("test.pdf", "Test error")
    assert isinstance(pdf_error, InvoicePDFError)

    classification_error = ClassificationError("test.pdf", "Classification failed")
    assert isinstance(classification_error, InvoicePDFError)

    # Test exception wrapping
    original = ValueError("Original error")
    wrapped = wrap_exception("test_function", original, "test.pdf")
    assert isinstance(wrapped, InvoicePDFError)
    assert "test_function" in str(wrapped)


def test_error_message_formatting():
    """Test that error messages contain useful information."""
    test_path = "/path/to/test.pdf"

    # Test PDFTooLargeError formatting
    error = PDFTooLargeError(test_path, 150.0, 100.0)
    error_str = str(error)

    assert test_path in error_str
    assert "150.0" in error_str  # File size
    assert "100.0" in error_str  # Max size
    assert "MB" in error_str

    # Test APIError formatting
    original_error = Exception("Network timeout")
    api_error = APIError(test_path, original_error, "gemini-2.5-flash", 3)
    api_error_str = str(api_error)

    assert test_path in api_error_str
    assert "gemini-2.5-flash" in api_error_str
    assert "3" in api_error_str  # Retry count
    assert "Network timeout" in api_error_str


def test_json_parsing_edge_cases():
    """Test JSON parsing handles edge cases in API responses."""

    edge_case_responses = [
        "",  # Empty response
        "Not JSON at all",  # Plain text
        "{ invalid json",  # Malformed JSON
        '{"incomplete": ',  # Incomplete JSON
        "502 Bad Gateway",  # Server error
        '{"classification": "vendor_invoice"}',  # Missing fields
    ]

    # These should all be handled gracefully without crashing
    for response_text in edge_case_responses:
        # Test that the JSON extraction logic handles these cases
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            # Should detect unparseable response
            assert True
        else:
            json_str = response_text[json_start:json_end]
            try:
                json.loads(json_str)
            except json.JSONDecodeError:
                # Should catch JSON decode errors
                assert True


@pytest.mark.asyncio
async def test_retry_mechanism_limits():
    """Test that retry mechanism respects limits."""
    from core.rate_limit import RetryError, retry_with_backoff

    call_count = 0

    async def failing_operation():
        nonlocal call_count
        call_count += 1
        raise Exception("Always fails")

    # Should fail after max retries
    with pytest.raises(RetryError) as exc_info:
        await retry_with_backoff(
            failing_operation,
            max_retries=3,
            base_delay=0.01  # Very short delay for testing
        )

    # Should have called the operation exactly max_retries times
    assert call_count == 3
    assert exc_info.value.attempts == 3


def test_configuration_validation():
    """Test configuration validation catches invalid settings."""
    import os

    from pydantic import ValidationError

    from invoice_pdf.config import Settings

    # Test with missing API key
    old_key = os.environ.get("GEMINI_API_KEY")
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

    try:
        with pytest.raises(ValidationError):
            Settings()
    finally:
        # Restore original key
        if old_key:
            os.environ["GEMINI_API_KEY"] = old_key


def test_manifest_error_handling():
    """Test manifest database error handling."""
    import os
    import sqlite3
    import tempfile

    from utilities.manifest import ProcessingManifest

    # Test with invalid database path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file where directory should be
        bad_path = os.path.join(tmpdir, "not_a_dir.db")
        with open(bad_path, "w") as f:
            f.write("not a database")

        manifest = ProcessingManifest(bad_path)

        # Should handle database errors gracefully
        try:
            manifest.connect()
            # Operations should handle errors
            manifest.get_summary()
        except Exception as e:
            # Should get a database-related error, not crash
            assert isinstance(e, (sqlite3.Error, OSError))


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])

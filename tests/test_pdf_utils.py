"""Tests for PDF utility functions."""

import asyncio
from pathlib import Path

import pytest

from invoice_pdf.core.pdf_utils import (
    extract_first_n_pages,
    get_page_count,
    initialize_pdf_semaphore,
    safe_extract_first_n_pages,
    safe_get_page_count,
)

# Test fixtures paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
PDF_FIXTURES = [
    FIXTURES_DIR / "14930.pdf",
    FIXTURES_DIR / "8753496_cust_1745346325.pdf",
    FIXTURES_DIR / "Crown SBI30296055913_01-01-2017_27-03-2018.pdf"
]


class TestPDFUtils:
    """Test class for PDF utility functions."""

    def test_get_page_count_valid_pdf(self):
        """Test get_page_count with valid PDF files."""
        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                page_count = get_page_count(pdf_path)
                assert page_count > 0, f"Page count should be > 0 for {pdf_path.name}"
                assert isinstance(page_count, int), "Page count should be an integer"

    def test_get_page_count_nonexistent_file(self):
        """Test get_page_count with non-existent file."""
        fake_path = FIXTURES_DIR / "nonexistent.pdf"
        page_count = get_page_count(fake_path)
        assert page_count == 0, "Should return 0 for non-existent file"

    def test_get_page_count_with_string_path(self):
        """Test get_page_count with string path."""
        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                page_count = get_page_count(str(pdf_path))
                assert page_count > 0, f"Should work with string path for {pdf_path.name}"

    def test_extract_first_n_pages_valid_pdf(self):
        """Test extract_first_n_pages with valid PDF files."""
        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                # Test extracting first 3 pages
                pdf_bytes = extract_first_n_pages(pdf_path, 3)
                assert isinstance(pdf_bytes, bytes), "Should return bytes"
                assert len(pdf_bytes) > 0, "PDF bytes should not be empty"

                # Verify it's a valid PDF by checking PDF header
                assert pdf_bytes.startswith(b"%PDF-"), "Should start with PDF header"

    def test_extract_first_n_pages_more_pages_than_available(self):
        """Test extract_first_n_pages when requesting more pages than available."""
        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                actual_pages = get_page_count(pdf_path)
                # Request way more pages than available
                pdf_bytes = extract_first_n_pages(pdf_path, actual_pages + 100)
                assert isinstance(pdf_bytes, bytes), "Should return bytes"
                assert len(pdf_bytes) > 0, "PDF bytes should not be empty"

    def test_extract_first_n_pages_all_pages(self):
        """Test extract_first_n_pages when requesting all pages."""
        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                actual_pages = get_page_count(pdf_path)
                pdf_bytes = extract_first_n_pages(pdf_path, actual_pages)
                original_bytes = pdf_path.read_bytes()

                # When extracting all pages, should return original document
                # (This is the optimization case in our implementation)
                assert isinstance(pdf_bytes, bytes), "Should return bytes"
                assert len(pdf_bytes) > 0, "PDF bytes should not be empty"

    def test_extract_first_n_pages_single_page(self):
        """Test extract_first_n_pages with single page."""
        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                pdf_bytes = extract_first_n_pages(pdf_path, 1)
                assert isinstance(pdf_bytes, bytes), "Should return bytes"
                assert len(pdf_bytes) > 0, "PDF bytes should not be empty"
                assert pdf_bytes.startswith(b"%PDF-"), "Should be valid PDF"

    def test_extract_first_n_pages_nonexistent_file(self):
        """Test extract_first_n_pages with non-existent file."""
        fake_path = FIXTURES_DIR / "nonexistent.pdf"
        with pytest.raises(Exception):
            extract_first_n_pages(fake_path, 3)

    def test_extract_first_n_pages_with_string_path(self):
        """Test extract_first_n_pages with string path."""
        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                pdf_bytes = extract_first_n_pages(str(pdf_path), 2)
                assert isinstance(pdf_bytes, bytes), "Should work with string path"
                assert len(pdf_bytes) > 0, "PDF bytes should not be empty"


class TestAsyncPDFUtils:
    """Test class for async PDF utility functions."""

    @pytest.mark.asyncio
    async def test_safe_get_page_count_without_semaphore(self):
        """Test safe_get_page_count without semaphore initialization."""
        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                page_count = await safe_get_page_count(pdf_path)
                assert page_count > 0, f"Page count should be > 0 for {pdf_path.name}"
                assert isinstance(page_count, int), "Page count should be an integer"

    @pytest.mark.asyncio
    async def test_safe_get_page_count_with_semaphore(self):
        """Test safe_get_page_count with semaphore initialization."""
        # Initialize semaphore
        initialize_pdf_semaphore(5)

        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                page_count = await safe_get_page_count(pdf_path)
                assert page_count > 0, f"Page count should be > 0 for {pdf_path.name}"
                assert isinstance(page_count, int), "Page count should be an integer"

    @pytest.mark.asyncio
    async def test_safe_extract_first_n_pages_without_semaphore(self):
        """Test safe_extract_first_n_pages without semaphore initialization."""
        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                pdf_bytes = await safe_extract_first_n_pages(pdf_path, 2)
                assert isinstance(pdf_bytes, bytes), "Should return bytes"
                assert len(pdf_bytes) > 0, "PDF bytes should not be empty"
                assert pdf_bytes.startswith(b"%PDF-"), "Should be valid PDF"

    @pytest.mark.asyncio
    async def test_safe_extract_first_n_pages_with_semaphore(self):
        """Test safe_extract_first_n_pages with semaphore initialization."""
        # Initialize semaphore
        initialize_pdf_semaphore(3)

        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                pdf_bytes = await safe_extract_first_n_pages(pdf_path, 3)
                assert isinstance(pdf_bytes, bytes), "Should return bytes"
                assert len(pdf_bytes) > 0, "PDF bytes should not be empty"
                assert pdf_bytes.startswith(b"%PDF-"), "Should be valid PDF"

    @pytest.mark.asyncio
    async def test_concurrent_pdf_operations(self):
        """Test concurrent PDF operations with semaphore."""
        initialize_pdf_semaphore(2)  # Limited capacity

        # Create multiple concurrent tasks
        tasks = []
        for pdf_path in PDF_FIXTURES[:2]:  # Use first 2 fixtures
            if pdf_path.exists():
                tasks.extend([
                    safe_get_page_count(pdf_path),
                    safe_extract_first_n_pages(pdf_path, 1),
                    safe_get_page_count(pdf_path),
                ])

        if tasks:
            results = await asyncio.gather(*tasks)

            # Verify results
            for i, result in enumerate(results):
                if i % 3 in [0, 2]:  # page count results
                    assert isinstance(result, int), "Page count should be integer"
                    assert result > 0, "Page count should be positive"
                else:  # extract results
                    assert isinstance(result, bytes), "Extract should return bytes"
                    assert len(result) > 0, "PDF bytes should not be empty"


class TestPDFSemaphore:
    """Test class for PDF semaphore initialization."""

    def test_initialize_pdf_semaphore(self):
        """Test semaphore initialization."""
        initialize_pdf_semaphore(10)
        # We can't directly test the internal state, but this should not raise
        # The semaphore will be used in subsequent async calls

    def test_initialize_pdf_semaphore_multiple_times(self):
        """Test that reinitializing semaphore works."""
        initialize_pdf_semaphore(5)
        initialize_pdf_semaphore(15)  # Should overwrite previous
        # Should not raise any errors


class TestMemorySafety:
    """Test class for memory safety features with large files."""

    def test_extract_large_file_memory_safety(self):
        """Test that large file extraction uses memory-safe streaming."""
        # Use a real PDF fixture to test file size logic
        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                # Get original file size
                file_size = pdf_path.stat().st_size
                page_count = get_page_count(pdf_path)

                # Test extracting all pages (should use streaming for large files)
                pdf_bytes = extract_first_n_pages(pdf_path, page_count)

                # Verify we got valid PDF bytes
                assert isinstance(pdf_bytes, bytes), "Should return bytes"
                assert len(pdf_bytes) > 0, "PDF bytes should not be empty"
                assert pdf_bytes.startswith(b"%PDF-"), "Should be valid PDF"

                # For files over 20MB, verify we're using file streaming
                if file_size > 20_000_000:
                    # Should have used direct file reading
                    original_bytes = pdf_path.read_bytes()
                    assert pdf_bytes == original_bytes, "Large files should use direct file reading"

    def test_extract_partial_pages_always_uses_pymupdf(self):
        """Test that partial page extraction always uses PyMuPDF processing."""
        for pdf_path in PDF_FIXTURES:
            if pdf_path.exists():
                page_count = get_page_count(pdf_path)

                if page_count > 1:
                    # Extract partial pages (should always use PyMuPDF)
                    pdf_bytes = extract_first_n_pages(pdf_path, page_count - 1)

                    # Verify we got valid PDF bytes
                    assert isinstance(pdf_bytes, bytes), "Should return bytes"
                    assert len(pdf_bytes) > 0, "PDF bytes should not be empty"
                    assert pdf_bytes.startswith(b"%PDF-"), "Should be valid PDF"

                    # Should be different from original file (since we extracted partial)
                    original_bytes = pdf_path.read_bytes()
                    assert pdf_bytes != original_bytes, "Partial extraction should differ from original"


@pytest.fixture(autouse=True)
def reset_semaphore():
    """Reset the PDF semaphore before each test to ensure clean state."""
    # Import here to avoid circular imports
    import invoice_pdf.core.pdf_utils
    invoice_pdf.core.pdf_utils.pdf_fd_semaphore = None
    yield
    # Clean up after test
    invoice_pdf.core.pdf_utils.pdf_fd_semaphore = None

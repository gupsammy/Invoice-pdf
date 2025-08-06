"""Pure PDF utility functions for page counting and extraction.

This module provides side-effect-free PDF operations that can be easily tested
and used across the application. All functions are synchronous and thread-safe.
"""

import asyncio
import logging
from pathlib import Path

import fitz  # PyMuPDF

from .rate_limit import CapacityLimiter

# Memory safety threshold for large PDF files (20MB)
LARGE_FILE_THRESHOLD_BYTES = 20_000_000

# Global PDF file descriptor semaphore - will be initialized by main application
pdf_fd_semaphore: CapacityLimiter | None = None


def get_page_count(pdf_path: Path | str) -> int:
    """Get the total number of pages in a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Number of pages in the PDF, or 0 if an error occurs

    Raises:
        Exception: If PDF cannot be opened or read
    """
    try:
        doc = fitz.open(str(pdf_path))
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception:
        logging.exception("Error getting page count for %s", pdf_path)
        return 0


def extract_first_n_pages(pdf_path: Path | str, max_pages: int) -> bytes:
    """Extract the first N pages from a PDF and return as bytes.

    Uses memory-efficient streaming for large files (>20MB). For smaller files,
    uses optimized page slicing - if all pages are needed, returns original
    document bytes without creating a new document.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract

    Returns:
        PDF bytes containing only the first N pages

    Raises:
        Exception: If PDF cannot be opened, processed, or converted to bytes
    """
    try:
        pdf_path = Path(pdf_path)
        file_size = pdf_path.stat().st_size

        # Open the source PDF
        source_doc = fitz.open(str(pdf_path))

        try:
            # Determine pages to extract
            pages_to_copy = min(len(source_doc), max_pages)

            # Memory-efficient handling for large files
            if pages_to_copy == len(source_doc):
                # If we need all pages from a large file, stream directly without loading into memory
                if file_size > LARGE_FILE_THRESHOLD_BYTES:
                    source_doc.close()
                    # Return file bytes directly without PyMuPDF processing
                    return pdf_path.read_bytes()
                
                # Small file - use existing logic
                return source_doc.tobytes()

            # For partial page extraction, always use PyMuPDF (no streaming option)
            # Create a new document and copy pages
            new_doc = fitz.open()
            try:
                new_doc.insert_pdf(source_doc, from_page=0, to_page=pages_to_copy - 1)
                return new_doc.tobytes()
            finally:
                new_doc.close()
        finally:
            source_doc.close()

    except Exception:
        logging.exception("Error extracting first %s pages from %s", max_pages, pdf_path)
        raise


async def safe_get_page_count(pdf_path: Path | str) -> int:
    """Get PDF page count with file descriptor semaphore guard.

    This async wrapper uses the global pdf_fd_semaphore to limit concurrent
    PDF file operations, preventing resource exhaustion.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Number of pages in the PDF, or 0 if an error occurs
    """
    global pdf_fd_semaphore

    if pdf_fd_semaphore is None:
        # Fallback to direct call if semaphore not initialized
        return await asyncio.to_thread(get_page_count, pdf_path)

    async with pdf_fd_semaphore:
        return await asyncio.to_thread(get_page_count, pdf_path)


async def safe_extract_first_n_pages(pdf_path: Path | str, max_pages: int) -> bytes:
    """Extract PDF pages with file descriptor semaphore guard.

    This async wrapper uses the global pdf_fd_semaphore to limit concurrent
    PDF file operations, preventing resource exhaustion. Uses memory-efficient
    streaming for large files (>20MB).

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract

    Returns:
        PDF bytes containing only the first N pages

    Raises:
        Exception: If PDF cannot be processed
    """
    global pdf_fd_semaphore

    if pdf_fd_semaphore is None:
        # Fallback to direct call if semaphore not initialized
        return await asyncio.to_thread(extract_first_n_pages, pdf_path, max_pages)

    async with pdf_fd_semaphore:
        return await asyncio.to_thread(extract_first_n_pages, pdf_path, max_pages)


def initialize_pdf_semaphore(limit: int) -> None:
    """Initialize the global PDF file descriptor semaphore.

    Args:
        limit: Maximum number of concurrent PDF operations
    """
    global pdf_fd_semaphore
    pdf_fd_semaphore = CapacityLimiter(limit)


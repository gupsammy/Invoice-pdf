"""Pure PDF utility functions for page counting and extraction.

This module provides side-effect-free PDF operations that can be easily tested
and used across the application. All functions are synchronous and thread-safe.
"""

import asyncio
import logging
from pathlib import Path

import fitz  # PyMuPDF

from .rate_limit import CapacityLimiter

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
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception:
        logging.exception("Error getting page count for %s", pdf_path)
        return 0


def extract_first_n_pages(pdf_path: Path | str, max_pages: int) -> bytes:
    """Extract the first N pages from a PDF and return as bytes.

    Uses optimized page slicing - if all pages are needed, returns original
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
        # Open the source PDF
        source_doc = fitz.open(str(pdf_path))

        # Determine pages to extract
        pages_to_copy = min(len(source_doc), max_pages)

        if pages_to_copy == len(source_doc):
            # If we need all pages, just return the original document as bytes
            pdf_bytes = source_doc.tobytes()
            source_doc.close()
            return pdf_bytes

        # Create a new document and copy pages
        new_doc = fitz.open()
        new_doc.insert_pdf(source_doc, from_page=0, to_page=pages_to_copy - 1)
        pdf_bytes = new_doc.tobytes()

        # Clean up
        new_doc.close()
        source_doc.close()

        return pdf_bytes

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
    PDF file operations, preventing resource exhaustion.

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


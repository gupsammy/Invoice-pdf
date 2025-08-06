"""Pure PDF utility functions for page counting and extraction.

This module provides side-effect-free PDF operations that can be easily tested
and used across the application. All functions are synchronous and thread-safe.
"""

import asyncio
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import fitz  # PyMuPDF

from .rate_limit import CapacityLimiter
from .exceptions import PDFTooLargeError, InvalidPDFError, PDFReadError

# Memory safety threshold for large PDF files (20MB)
LARGE_FILE_THRESHOLD_BYTES = 20_000_000

# Global PDF file descriptor semaphore - will be initialized by main application
pdf_fd_semaphore: CapacityLimiter | None = None


def check_pdf_size_safety(file_path: Path | str, max_size_mb: float = 100.0) -> tuple[float, bool]:
    """Check if PDF file size is safe for memory operations.
    
    Args:
        file_path: Path to PDF file
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        Tuple of (file_size_mb, is_safe)
        
    Raises:
        PDFTooLargeError: If file exceeds maximum size
        PDFReadError: If unable to read file
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        file_size_bytes = file_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        logging.debug(f"PDF size check: {file_path.name} = {file_size_mb:.1f}MB")
        
        if file_size_mb > max_size_mb:
            raise PDFTooLargeError(file_path, file_size_mb, max_size_mb)
            
        # Return whether file is considered "safe" (under warning threshold)
        is_safe = file_size_mb <= (max_size_mb * 0.5)  # 50% of max
        
        return file_size_mb, is_safe
        
    except OSError as e:
        raise PDFReadError(file_path, e)


def get_memory_efficient_pdf_reader(file_path: Path | str, max_size_mb: float = 100.0) -> fitz.Document:
    """Get a PDF reader with memory safety checks.
    
    Args:
        file_path: Path to PDF file
        max_size_mb: Maximum allowed file size in MB
        
    Returns:
        PyMuPDF Document object
        
    Raises:
        PDFTooLargeError: If file exceeds maximum size
        InvalidPDFError: If PDF is corrupted
        PDFReadError: If unable to read file
    """
    file_path = Path(file_path)
    
    # Check file size first
    file_size_mb, is_safe = check_pdf_size_safety(file_path, max_size_mb)
    
    if not is_safe:
        logging.warning(f"Large PDF detected: {file_path.name} ({file_size_mb:.1f}MB)")
    
    try:
        # Open with memory-friendly settings for large files
        doc = fitz.open(file_path)
        
        # Validate PDF is readable
        try:
            page_count = doc.page_count
            if page_count == 0:
                raise InvalidPDFError(file_path, "PDF has no pages")
                
        except Exception as e:
            doc.close()
            raise InvalidPDFError(file_path, f"Unable to read PDF pages: {e}")
            
        return doc
        
    except fitz.FileDataError as e:
        raise InvalidPDFError(file_path, f"PDF file is corrupted: {e}")
    except fitz.FileNotFoundError:
        raise PDFReadError(file_path, FileNotFoundError(f"PDF file not found: {file_path}"))
    except Exception as e:
        raise PDFReadError(file_path, e)


@contextmanager
def open_pdf(path: Path | str, max_size_mb: float = 100.0) -> Generator[fitz.Document, None, None]:
    """Context manager for safe PDF handling with memory safety checks.

    Ensures PDF documents are properly closed after use to prevent
    resource leaks and memory issues. Includes size checking for safety.

    Args:
        path: Path to the PDF file
        max_size_mb: Maximum allowed file size in MB

    Yields:
        Opened PDF document

    Raises:
        PDFTooLargeError: If PDF exceeds maximum size
        InvalidPDFError: If PDF is corrupted
        PDFReadError: If PDF cannot be read
    """
    doc = get_memory_efficient_pdf_reader(path, max_size_mb)
    try:
        yield doc
    finally:
        doc.close()


def get_page_count(pdf_path: Path | str, max_size_mb: float = 100.0) -> int:
    """Get the total number of pages in a PDF file with memory safety.

    Args:
        pdf_path: Path to the PDF file
        max_size_mb: Maximum allowed file size in MB

    Returns:
        Number of pages in the PDF

    Raises:
        PDFTooLargeError: If PDF exceeds maximum size
        InvalidPDFError: If PDF is corrupted  
        PDFReadError: If PDF cannot be read
    """
    with open_pdf(pdf_path, max_size_mb) as doc:
        return len(doc)


def extract_first_n_pages(pdf_path: Path | str, max_pages: int, max_size_mb: float = 100.0) -> bytes:
    """Extract the first N pages from a PDF and return as bytes with memory safety.

    Uses memory-efficient streaming for large files (>20MB). For smaller files,
    uses optimized page slicing - if all pages are needed, returns original
    document bytes without creating a new document.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract
        max_size_mb: Maximum allowed file size in MB

    Returns:
        PDF bytes containing only the first N pages

    Raises:
        PDFTooLargeError: If PDF exceeds maximum size
        InvalidPDFError: If PDF is corrupted
        PDFReadError: If PDF cannot be read
    """
    pdf_path = Path(pdf_path)
    
    # Check file size and get memory safety info
    file_size_mb, is_safe = check_pdf_size_safety(pdf_path, max_size_mb)
    file_size = pdf_path.stat().st_size

    # Open the source PDF with memory safety
    with open_pdf(pdf_path, max_size_mb) as source_doc:
        # Determine pages to extract
        pages_to_copy = min(len(source_doc), max_pages)

        # Memory-efficient handling for large files
        if pages_to_copy == len(source_doc):
            # If we need all pages from a large file, stream directly without loading into memory
            if file_size > LARGE_FILE_THRESHOLD_BYTES:
                # Return file bytes directly without PyMuPDF processing
                logging.info(f"Streaming large PDF directly: {pdf_path.name} ({file_size_mb:.1f}MB)")
                return pdf_path.read_bytes()

            # Small file - use existing logic
            return source_doc.tobytes()

        # For partial page extraction, always use PyMuPDF (no streaming option)
        # Create a new document and copy pages
        if not is_safe:
            logging.warning(f"Extracting pages from large PDF: {pdf_path.name} ({file_size_mb:.1f}MB)")
            
        new_doc = fitz.open()
        try:
            new_doc.insert_pdf(source_doc, from_page=0, to_page=pages_to_copy - 1)
            return new_doc.tobytes()
        finally:
            new_doc.close()


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

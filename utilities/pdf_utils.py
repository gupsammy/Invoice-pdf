"""
PDF utility functions for the invoice processing system.

This module contains pure synchronous PDF processing helpers that don't require
semaphore management or async context.
"""

import logging
import fitz  # PyMuPDF

from config_adv import MAX_CLASSIFICATION_PAGES


def get_pdf_page_count(pdf_path: str) -> int:
    """Get the total number of pages in a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception as e:
        logging.exception(f"Error getting page count for {pdf_path}: {e!s}")
        return 0


def extract_first_n_pages_pdf(pdf_path: str, max_pages: int = MAX_CLASSIFICATION_PAGES) -> bytes:
    """
    Extract the first N pages from a PDF and return as bytes using optimized page slicing.
    
    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract (default: 7)
        
    Returns:
        PDF bytes containing only the first N pages
    """
    try:
        # Open the source PDF
        source_doc = fitz.open(pdf_path)

        # Determine pages to extract
        pages_to_copy = min(len(source_doc), max_pages)

        if pages_to_copy == len(source_doc):
            # If we need all pages, just return the original document as bytes
            pdf_bytes = source_doc.tobytes()
            source_doc.close()
            return pdf_bytes

        # Create a new document and copy pages
        new_doc = fitz.open()
        new_doc.insert_pdf(source_doc, from_page=0, to_page=pages_to_copy-1)
        pdf_bytes = new_doc.tobytes()
        new_doc.close()
        source_doc.close()
        return pdf_bytes

    except Exception as e:
        logging.exception(f"Error extracting first {max_pages} pages from {pdf_path}: {e!s}")
        raise
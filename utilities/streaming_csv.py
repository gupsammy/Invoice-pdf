"""
Streaming CSV Writer utility for memory-efficient CSV generation.

This module provides a StreamingCSVWriter class that writes CSV rows 
immediately to disk instead of accumulating them in memory.
"""

import asyncio
import csv
from pathlib import Path
from typing import Any


class StreamingCSVWriter:
    """
    Memory-efficient CSV writer that streams results directly to disk.
    
    Instead of accumulating results in memory before writing, this class 
    writes each row immediately, reducing memory usage for large datasets.
    """

    def __init__(self, file_path: str, fieldnames: list[str], append: bool = False):
        """
        Initialize the streaming CSV writer.
        
        Args:
            file_path: Path where to write the CSV file
            fieldnames: List of CSV column names
            append: If True, append to existing file instead of overwriting
        """
        self.file_path = Path(file_path)
        self.fieldnames = fieldnames
        self.lock = asyncio.Lock()
        self.append_mode = append

        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine if we should write header
        write_header = True
        if append and self.file_path.exists():
            # In append mode, only write header if file is empty
            write_header = self.file_path.stat().st_size == 0

        # Open file in appropriate mode
        mode = "a" if append else "w"
        self.file = open(self.file_path, mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        
        if write_header:
            self.writer.writeheader()
            self.file.flush()

        self._rows_written = 0

    async def write_row(self, row: dict[str, Any]):
        """
        Write a single row to the CSV file asynchronously.
        
        Args:
            row: Dictionary containing row data
        """
        async with self.lock:
            # Ensure all fieldnames are present with default values
            complete_row = {field: row.get(field, "") for field in self.fieldnames}
            self.writer.writerow(complete_row)
            self.file.flush()
            self._rows_written += 1

    def write_row_sync(self, row: dict[str, Any]):
        """
        Write a single row to the CSV file synchronously.
        
        Args:
            row: Dictionary containing row data
        """
        # Ensure all fieldnames are present with default values
        complete_row = {field: row.get(field, "") for field in self.fieldnames}
        self.writer.writerow(complete_row)
        self.file.flush()
        self._rows_written += 1

    def close(self):
        """Close the CSV file."""
        if hasattr(self, "file") and not self.file.closed:
            self.file.close()

    @property
    def rows_written(self) -> int:
        """Return the number of rows written so far."""
        return self._rows_written

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.close()

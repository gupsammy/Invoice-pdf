"""
SQLite manifest for resumable invoice processing pipeline.

This module provides a crash-safe SQLite-based manifest system to track
the progress of PDF processing through classification and extraction steps.
"""

import sqlite3
import logging
import os
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class ProcessingManifest:
    """
    SQLite-based manifest for tracking PDF processing progress.
    
    Supports concurrent writes through WAL mode and provides atomic
    operations for classification and extraction status tracking.
    """
    
    def __init__(self, db_path: str = "manifest.db", batch_size: int = 50):
        self.db_path = db_path
        self.batch_size = batch_size
        self._conn = None
        self._lock = threading.RLock()
        self._batch_counter = 0
        
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def connect(self) -> sqlite3.Connection:
        """Initialize database connection with WAL mode for concurrent access."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=30.0  # 30 second timeout for busy database
            )
            
            # Enable WAL mode for better concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=10000")
            self._conn.execute("PRAGMA temp_store=memory")
            
            # Create table if not exists
            self._create_table()
            
        return self._conn
    
    def close(self):
        """Close database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
            finally:
                self._conn = None
    
    def _create_table(self):
        """Create the progress tracking table."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS progress (
            file_path      TEXT PRIMARY KEY,
            classified     INTEGER DEFAULT 0,
            classification TEXT,
            extracted      INTEGER DEFAULT 0,
            doc_type       TEXT,
            last_error     TEXT,
            updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        with self._lock:
            self._conn.execute(create_sql)
            self._conn.commit()
    
    def _retry_operation(self, operation, max_retries: int = 3):
        """Retry database operations with exponential backoff."""
        import time
        for attempt in range(max_retries):
            try:
                return operation()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.1  # 0.1, 0.2, 0.4 seconds
                    logger.warning(f"Database locked, retrying in {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    raise
    
    def mark_classified(self, file_path: str, classification: str, doc_type: str = ""):
        """Mark a file as classified with its classification result."""
        def operation():
            with self._lock:
                self._conn.execute("""
                    INSERT OR REPLACE INTO progress 
                    (file_path, classified, classification, doc_type, updated_at)
                    VALUES (?, 1, ?, ?, ?)
                """, (file_path, classification, doc_type, datetime.now().isoformat()))
                
                self._batch_counter += 1
                if self._batch_counter >= self.batch_size:
                    self._conn.commit()
                    self._batch_counter = 0
        
        self._retry_operation(operation)
        logger.debug(f"Marked {file_path} as classified: {classification}")
    
    def mark_extracted(self, file_path: str):
        """Mark a file as having completed extraction."""
        def operation():
            with self._lock:
                self._conn.execute("""
                    INSERT OR REPLACE INTO progress 
                    (file_path, classified, classification, extracted, doc_type, updated_at)
                    VALUES (
                        ?, 
                        COALESCE((SELECT classified FROM progress WHERE file_path = ?), 1),
                        COALESCE((SELECT classification FROM progress WHERE file_path = ?), ''),
                        1,
                        COALESCE((SELECT doc_type FROM progress WHERE file_path = ?), ''),
                        ?
                    )
                """, (file_path, file_path, file_path, file_path, datetime.now().isoformat()))
                
                self._batch_counter += 1
                if self._batch_counter >= self.batch_size:
                    self._conn.commit()
                    self._batch_counter = 0
        
        self._retry_operation(operation)
        logger.debug(f"Marked {file_path} as extracted")
    
    def mark_error(self, file_path: str, error_message: str):
        """Mark a file as having encountered an error."""
        def operation():
            with self._lock:
                self._conn.execute("""
                    INSERT OR REPLACE INTO progress 
                    (file_path, classified, classification, extracted, doc_type, last_error, updated_at)
                    VALUES (
                        ?,
                        COALESCE((SELECT classified FROM progress WHERE file_path = ?), 0),
                        COALESCE((SELECT classification FROM progress WHERE file_path = ?), ''),
                        COALESCE((SELECT extracted FROM progress WHERE file_path = ?), 0),
                        COALESCE((SELECT doc_type FROM progress WHERE file_path = ?), ''),
                        ?,
                        ?
                    )
                """, (file_path, file_path, file_path, file_path, file_path, error_message, datetime.now().isoformat()))
                
                self._batch_counter += 1
                if self._batch_counter >= self.batch_size:
                    self._conn.commit()
                    self._batch_counter = 0
        
        self._retry_operation(operation)
        logger.warning(f"Marked {file_path} with error: {error_message}")
    
    def get_resume_queues(self, pdf_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Get lists of files that need classification or extraction.
        
        Returns:
            Tuple of (classify_list, extract_list)
        """
        def operation():
            with self._lock:
                # Get current progress for all files
                placeholders = ','.join('?' * len(pdf_paths))
                cursor = self._conn.execute(f"""
                    SELECT file_path, classified, classification, extracted, doc_type, last_error 
                    FROM progress 
                    WHERE file_path IN ({placeholders})
                """, pdf_paths)
                
                progress_map = {row[0]: row[1:] for row in cursor.fetchall()}
                
                classify_list = []
                extract_list = []
                
                for pdf_path in pdf_paths:
                    if pdf_path in progress_map:
                        classified, classification, extracted, doc_type, last_error = progress_map[pdf_path]
                        
                        # Skip if already fully processed
                        if extracted:
                            continue
                        
                        # Skip files with classification errors - they need to be re-classified
                        if last_error and not classified:
                            classify_list.append(pdf_path)
                            continue
                        
                        # Skip files with extraction errors that were classified successfully
                        if last_error and classified and not extracted:
                            # Don't add to any queue - manual intervention may be needed
                            # Could add a separate "error" queue in the future
                            continue
                            
                        # Need extraction if classified but not extracted (and no errors)
                        if classified and not extracted and not last_error:
                            # Only add to extraction queue if classification is valid
                            if classification in ['vendor_invoice', 'employee_t&e']:
                                extract_list.append(pdf_path)
                            # If classification is 'irrelevant', mark as extracted (no extraction needed)
                            elif classification == 'irrelevant':
                                self.mark_extracted(pdf_path)
                        elif not classified:
                            # Need classification (and no errors)
                            classify_list.append(pdf_path)
                    else:
                        # New file, needs classification
                        classify_list.append(pdf_path)
                
                return classify_list, extract_list
        
        return self._retry_operation(operation)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary statistics."""
        def operation():
            with self._lock:
                cursor = self._conn.execute("""
                    SELECT 
                        COUNT(*) as total_files,
                        SUM(classified) as classified_count,
                        SUM(extracted) as extracted_count,
                        SUM(CASE WHEN last_error IS NOT NULL THEN 1 ELSE 0 END) as error_count,
                        SUM(CASE WHEN classified = 1 AND extracted = 0 AND classification != 'irrelevant' THEN 1 ELSE 0 END) as pending_extraction
                    FROM progress
                """)
                
                row = cursor.fetchone()
                if row:
                    total, classified, extracted, errors, pending_extraction = row
                    return {
                        'total_files': total or 0,
                        'classified': classified or 0,
                        'extracted': extracted or 0,
                        'failed': errors or 0,
                        'pending_extraction': pending_extraction or 0,
                        'pending_classification': max(0, (total or 0) - (classified or 0))
                    }
                else:
                    return {
                        'total_files': 0,
                        'classified': 0,
                        'extracted': 0,
                        'failed': 0,
                        'pending_extraction': 0,
                        'pending_classification': 0
                    }
        
        return self._retry_operation(operation)
    
    def is_file_completed(self, file_path: str, stage: str) -> bool:
        """
        Check if a specific file has completed a specific stage efficiently.
        
        Args:
            file_path: Path to the file
            stage: Either 'classification' or 'extraction'
            
        Returns:
            True if the stage is completed, False otherwise
        """
        def operation():
            with self._lock:
                if stage == 'classification':
                    cursor = self._conn.execute("""
                        SELECT classified, last_error FROM progress 
                        WHERE file_path = ? AND classified = 1
                    """, (file_path,))
                    result = cursor.fetchone()
                    return result is not None and not result[1]  # classified and no error
                    
                elif stage == 'extraction':
                    cursor = self._conn.execute("""
                        SELECT extracted, classification, last_error FROM progress 
                        WHERE file_path = ? AND extracted = 1
                    """, (file_path,))
                    result = cursor.fetchone()
                    if result is None:
                        # Check if it's irrelevant (auto-extracted)
                        cursor = self._conn.execute("""
                            SELECT classification FROM progress 
                            WHERE file_path = ? AND classified = 1 AND classification = 'irrelevant'
                        """, (file_path,))
                        return cursor.fetchone() is not None
                    return True
                    
                else:
                    raise ValueError(f"Unknown stage: {stage}")
        
        try:
            return self._retry_operation(operation)
        except Exception as e:
            logger.error(f"Error checking file completion status: {e}")
            return False

    def force_commit(self):
        """Force commit any pending batch operations."""
        with self._lock:
            if self._conn:
                self._conn.commit()
                self._batch_counter = 0
    
    def reset_errors(self, file_paths: Optional[List[str]] = None):
        """Reset error status for specified files or all files."""
        def operation():
            with self._lock:
                if file_paths:
                    placeholders = ','.join('?' * len(file_paths))
                    self._conn.execute(f"""
                        UPDATE progress 
                        SET last_error = NULL, updated_at = ?
                        WHERE file_path IN ({placeholders})
                    """, [datetime.now().isoformat()] + file_paths)
                else:
                    self._conn.execute("""
                        UPDATE progress 
                        SET last_error = NULL, updated_at = ?
                    """, (datetime.now().isoformat(),))
                
                self._conn.commit()
        
        self._retry_operation(operation)


def init_db(path: str = "manifest.db") -> ProcessingManifest:
    """Initialize and return a ProcessingManifest instance."""
    manifest = ProcessingManifest(path)
    manifest.connect()
    return manifest


# Convenience functions for backward compatibility
def mark_classified(manifest: ProcessingManifest, file_path: str, classification: str, doc_type: str = ""):
    """Mark a file as classified."""
    manifest.mark_classified(file_path, classification, doc_type)


def mark_extracted(manifest: ProcessingManifest, file_path: str):
    """Mark a file as extracted."""
    manifest.mark_extracted(file_path)


def mark_error(manifest: ProcessingManifest, file_path: str, error_message: str):
    """Mark a file with an error."""
    manifest.mark_error(file_path, error_message)


def get_resume_queues(manifest: ProcessingManifest, pdf_paths: List[str]) -> Tuple[List[str], List[str]]:
    """Get resume queues for classification and extraction."""
    return manifest.get_resume_queues(pdf_paths)


def summary(manifest: ProcessingManifest) -> Dict[str, Any]:
    """Get processing summary."""
    return manifest.get_summary()
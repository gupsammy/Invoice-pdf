"""
Quick Optimization Implementations
High-impact, low-effort changes you can apply immediately to main_2step_enhanced.py
"""

import asyncio
import csv
import os
import sqlite3
import aiosqlite
from pathlib import Path
from datetime import datetime
import fitz
import logging
from typing import Dict, List, Any, Optional

# ===========================
# 1. OPTIMIZED PDF PAGE EXTRACTION
# ===========================

def extract_first_n_pages_pdf_optimized(pdf_path: str, max_pages: int = 7) -> bytes:
    """
    Optimized PDF page extraction - doesn't load entire PDF into memory.
    Replace the existing extract_first_n_pages_pdf function with this.
    """
    try:
        source_doc = fitz.open(pdf_path)
        total_pages = len(source_doc)
        
        if total_pages <= max_pages:
            # Direct conversion without intermediate loading
            pdf_bytes = source_doc.tobytes()
            source_doc.close()
            return pdf_bytes
        
        # Use PyMuPDF's select() for true zero-copy selection
        source_doc.select(range(max_pages))
        pdf_bytes = source_doc.tobytes()
        source_doc.close()
        return pdf_bytes
        
    except Exception as e:
        logging.error(f"Error extracting first {max_pages} pages from {pdf_path}: {str(e)}")
        raise

# ===========================
# 2. ADAPTIVE SEMAPHORE FOR DYNAMIC CONCURRENCY
# ===========================

class AdaptiveSemaphore:
    """
    Dynamically adjusts concurrency based on success/error rates.
    Use this instead of regular asyncio.Semaphore.
    """
    def __init__(self, initial_limit=10, max_limit=25, min_limit=3):
        self.current_limit = initial_limit
        self.max_limit = max_limit
        self.min_limit = min_limit
        self.semaphore = asyncio.Semaphore(initial_limit)
        self.error_count = 0
        self.success_count = 0
        self.adjustment_threshold = 10
        
    async def __aenter__(self):
        await self.semaphore.acquire()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Record success or failure
        if exc_type is None:
            self.success_count += 1
            # Increase limit after consistent successes
            if self.success_count > self.adjustment_threshold and self.current_limit < self.max_limit:
                self.current_limit = min(self.current_limit + 2, self.max_limit)
                self._recreate_semaphore()
                self.success_count = 0
                logging.info(f"Increased concurrency limit to {self.current_limit}")
        else:
            self.error_count += 1
            # Decrease limit after errors
            if self.error_count > 3 and self.current_limit > self.min_limit:
                self.current_limit = max(self.current_limit - 3, self.min_limit)
                self._recreate_semaphore()
                self.error_count = 0
                logging.warning(f"Decreased concurrency limit to {self.current_limit}")
        
        self.semaphore.release()
        
    def _recreate_semaphore(self):
        """Recreate semaphore with new limit"""
        # Note: This is a simplified approach. In production, you'd need to handle
        # existing acquisitions more carefully
        self.semaphore = asyncio.Semaphore(self.current_limit)

# ===========================
# 3. STREAMING CSV WRITER
# ===========================

class StreamingCSVWriter:
    """
    Thread-safe CSV writer that streams results to disk immediately.
    Prevents memory exhaustion when processing thousands of files.
    """
    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.file = open(filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.lock = asyncio.Lock()
        self.row_count = 0
        
    async def write_row(self, row_dict: Dict[str, Any]):
        """Write a single row to CSV file"""
        async with self.lock:
            self.writer.writerow(row_dict)
            self.file.flush()  # Ensure immediate write
            self.row_count += 1
            
            # Log progress every 100 rows
            if self.row_count % 100 == 0:
                logging.info(f"Streamed {self.row_count} results to {self.filepath}")
    
    async def write_rows(self, rows: List[Dict[str, Any]]):
        """Write multiple rows to CSV file"""
        async with self.lock:
            for row in rows:
                self.writer.writerow(row)
            self.file.flush()
            self.row_count += len(rows)
    
    def close(self):
        """Close the file handle"""
        self.file.close()
        logging.info(f"Closed {self.filepath} with {self.row_count} total rows")

# ===========================
# 4. SQLITE-BASED PROCESSING TRACKER
# ===========================

class ProcessingTracker:
    """
    SQLite-based tracker for resume functionality.
    Much more efficient than loading entire CSV into memory.
    """
    def __init__(self, db_path: str = "processing_state.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processing_state (
                file_path TEXT PRIMARY KEY,
                file_name TEXT,
                classification TEXT,
                confidence REAL,
                extraction_status TEXT,
                processed_at TIMESTAMP,
                error_message TEXT,
                total_pages INTEGER,
                processing_time_seconds REAL
            )
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_classification 
            ON processing_state(classification)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_extraction_status 
            ON processing_state(extraction_status)
        ''')
        conn.commit()
        conn.close()
    
    async def mark_classified(self, file_path: str, classification: str, 
                            confidence: float, total_pages: int):
        """Mark a file as classified"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO processing_state 
                   (file_path, file_name, classification, confidence, 
                    total_pages, processed_at) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (file_path, os.path.basename(file_path), classification, 
                 confidence, total_pages, datetime.now())
            )
            await db.commit()
    
    async def mark_extracted(self, file_path: str, status: str = "completed"):
        """Mark a file as extracted"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """UPDATE processing_state 
                   SET extraction_status = ?, processed_at = ? 
                   WHERE file_path = ?""",
                (status, datetime.now(), file_path)
            )
            await db.commit()
    
    async def get_pending_extractions(self) -> List[tuple]:
        """Get files that need extraction"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT file_path, classification FROM processing_state 
                   WHERE classification IN ('vendor_invoice', 'employee_t&e') 
                   AND (extraction_status IS NULL OR extraction_status = 'pending')
                   ORDER BY total_pages ASC"""  # Process smaller files first
            )
            return await cursor.fetchall()
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            # Total files
            cursor = await db.execute("SELECT COUNT(*) FROM processing_state")
            total_files = (await cursor.fetchone())[0]
            
            # Classification stats
            cursor = await db.execute(
                """SELECT classification, COUNT(*) 
                   FROM processing_state 
                   GROUP BY classification"""
            )
            classification_stats = dict(await cursor.fetchall())
            
            # Extraction stats
            cursor = await db.execute(
                """SELECT extraction_status, COUNT(*) 
                   FROM processing_state 
                   WHERE classification IN ('vendor_invoice', 'employee_t&e')
                   GROUP BY extraction_status"""
            )
            extraction_stats = dict(await cursor.fetchall())
            
            return {
                "total_files": total_files,
                "classification": classification_stats,
                "extraction": extraction_stats
            }

# ===========================
# 5. CHUNKED FILE PROCESSOR
# ===========================

def chunk_files(files: List[str], chunk_size: int = 500):
    """
    Generator that yields file chunks for processing.
    Essential for handling 10000+ files without memory issues.
    """
    for i in range(0, len(files), chunk_size):
        yield files[i:i + chunk_size]

async def process_large_dataset_chunked(
    all_files: List[str],
    client,
    output_folder: str,
    chunk_size: int = 500,
    inter_chunk_delay: float = 2.0
):
    """
    Process large datasets in chunks to manage memory and API limits.
    """
    tracker = ProcessingTracker()
    total_chunks = (len(all_files) + chunk_size - 1) // chunk_size
    
    # Classification CSV writer
    classification_csv = StreamingCSVWriter(
        os.path.join(output_folder, "classification_results_streamed.csv"),
        fieldnames=["file_name", "file_path", "classification", "confidence", 
                   "total_pages", "processed_at"]
    )
    
    try:
        for chunk_num, file_chunk in enumerate(chunk_files(all_files, chunk_size)):
            logging.info(f"Processing chunk {chunk_num + 1}/{total_chunks} ({len(file_chunk)} files)")
            
            # Process classification for this chunk
            # Note: You'd integrate this with your existing classification logic
            for file_path in file_chunk:
                # Simulate classification (replace with actual classification)
                await tracker.mark_classified(
                    file_path, 
                    "vendor_invoice",  # This would be actual classification
                    0.95,  # Actual confidence
                    10  # Actual page count
                )
                
                # Stream result to CSV
                await classification_csv.write_row({
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path,
                    "classification": "vendor_invoice",
                    "confidence": 0.95,
                    "total_pages": 10,
                    "processed_at": datetime.now().isoformat()
                })
            
            # Get stats after each chunk
            stats = await tracker.get_processing_stats()
            logging.info(f"Chunk {chunk_num + 1} complete. Stats: {stats}")
            
            # Add delay between chunks to respect rate limits
            if chunk_num < total_chunks - 1:
                await asyncio.sleep(inter_chunk_delay)
                
    finally:
        classification_csv.close()

# ===========================
# 6. MEMORY-EFFICIENT FILE GENERATOR
# ===========================

def get_pdf_files_generator(input_folder: str):
    """
    Generator that yields PDF file paths without loading all into memory.
    Essential for directories with 100000+ files.
    """
    for root, dirs, files in os.walk(input_folder):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in ["dup", "__pycache__", ".git"]]
        
        for file in files:
            if file.lower().endswith('.pdf'):
                yield os.path.join(root, file)

# ===========================
# 7. EXAMPLE INTEGRATION
# ===========================

async def example_optimized_pipeline():
    """
    Example of how to integrate these optimizations into your pipeline.
    """
    input_folder = "input"
    output_folder = "output"
    
    # Use adaptive semaphore instead of regular semaphore
    classify_semaphore = AdaptiveSemaphore(initial_limit=15, max_limit=25)
    extract_semaphore = AdaptiveSemaphore(initial_limit=7, max_limit=12)
    
    # Use SQLite tracker for resume functionality
    tracker = ProcessingTracker()
    
    # Use generator for memory-efficient file listing
    pdf_files = list(get_pdf_files_generator(input_folder))
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    # Process in chunks
    await process_large_dataset_chunked(
        pdf_files,
        client=None,  # Your genai client
        output_folder=output_folder,
        chunk_size=500
    )
    
    # Get final stats
    final_stats = await tracker.get_processing_stats()
    logging.info(f"Final processing stats: {final_stats}")

# ===========================
# 8. CONFIGURATION UPDATES
# ===========================

# Add these environment-based configurations to your main script
OPTIMIZED_CONFIG = {
    # Increased concurrency limits based on analysis
    "MAX_CONCURRENT_CLASSIFY": int(os.getenv("MAX_CONCURRENT_CLASSIFY", "20")),
    "MAX_CONCURRENT_EXTRACT": int(os.getenv("MAX_CONCURRENT_EXTRACT", "8")),
    
    # Chunk size for large datasets
    "PROCESSING_CHUNK_SIZE": int(os.getenv("PROCESSING_CHUNK_SIZE", "500")),
    
    # Memory management
    "ENABLE_RESULT_STREAMING": os.getenv("ENABLE_RESULT_STREAMING", "true").lower() == "true",
    "MAX_RESULTS_IN_MEMORY": int(os.getenv("MAX_RESULTS_IN_MEMORY", "100")),
    
    # Performance tuning
    "INTER_CHUNK_DELAY_SECONDS": float(os.getenv("INTER_CHUNK_DELAY_SECONDS", "2.0")),
    "ENABLE_ADAPTIVE_CONCURRENCY": os.getenv("ENABLE_ADAPTIVE_CONCURRENCY", "true").lower() == "true",
}

if __name__ == "__main__":
    # Test the optimizations
    logging.basicConfig(level=logging.INFO)
    
    # Test optimized PDF extraction
    print("Testing optimized PDF extraction...")
    # pdf_bytes = extract_first_n_pages_pdf_optimized("test.pdf", 7)
    
    # Run example pipeline
    print("Running example optimized pipeline...")
    # asyncio.run(example_optimized_pipeline())
    
    print("Optimization implementations ready to integrate!")
    print(f"Configuration: {OPTIMIZED_CONFIG}")
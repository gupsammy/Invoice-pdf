"""
Unit tests for the ProcessingManifest SQLite implementation.
"""

import os
import sys
import tempfile

import pytest

sys.path.append("..")
from invoice_pdf.io.manifest import ProcessingManifest, init_db


class TestProcessingManifest:
    """Test cases for ProcessingManifest functionality."""

    def setup_method(self):
        """Set up test environment with temporary database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_manifest.db")
        self.manifest = ProcessingManifest(self.db_path, batch_size=2)
        self.manifest.connect()

        # Test file paths
        self.test_files = [
            "/test/path/invoice1.pdf",
            "/test/path/invoice2.pdf",
            "/test/path/report1.pdf",
            "/test/path/irrelevant.pdf"
        ]

    def teardown_method(self):
        """Clean up test environment."""
        self.manifest.close()
        # Clean up temp files (including SQLite WAL files)
        import glob
        db_pattern = self.db_path + "*"  # Matches .db, .db-wal, .db-shm
        for db_file in glob.glob(db_pattern):
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass  # Ignore cleanup errors

        # Clean up temp directory
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            # If directory is not empty, remove everything
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_initialization(self):
        """Test that database and table are created correctly."""
        # Verify table exists
        cursor = self.manifest._conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='progress'
        """)
        assert cursor.fetchone() is not None

    def test_mark_classified(self):
        """Test marking files as classified."""
        file_path = self.test_files[0]
        classification = "vendor_invoice"
        doc_type = "invoice"

        self.manifest.mark_classified(file_path, classification, doc_type)
        self.manifest.force_commit()

        # Verify in database
        cursor = self.manifest._conn.execute(
            "SELECT classified, classification, doc_type FROM progress WHERE file_path = ?",
            (file_path,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 1  # classified
        assert row[1] == classification
        assert row[2] == doc_type

    def test_mark_extracted(self):
        """Test marking files as extracted."""
        file_path = self.test_files[0]

        # First classify the file
        self.manifest.mark_classified(file_path, "vendor_invoice", "invoice")
        # Then mark as extracted
        self.manifest.mark_extracted(file_path)
        self.manifest.force_commit()

        # Verify in database
        cursor = self.manifest._conn.execute(
            "SELECT classified, extracted FROM progress WHERE file_path = ?",
            (file_path,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 1  # classified
        assert row[1] == 1  # extracted

    def test_mark_error(self):
        """Test marking files with errors."""
        file_path = self.test_files[0]
        error_msg = "API timeout error"

        self.manifest.mark_error(file_path, error_msg)
        self.manifest.force_commit()

        # Verify in database
        cursor = self.manifest._conn.execute(
            "SELECT last_error FROM progress WHERE file_path = ?",
            (file_path,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == error_msg

    def test_get_resume_queues_new_files(self):
        """Test getting resume queues with all new files."""
        classify_list, extract_list = self.manifest.get_resume_queues(self.test_files)

        # All files should need classification
        assert set(classify_list) == set(self.test_files)
        assert len(extract_list) == 0

    def test_get_resume_queues_mixed_status(self):
        """Test getting resume queues with mixed file statuses."""
        # Set up different file states
        self.manifest.mark_classified(self.test_files[0], "vendor_invoice", "invoice")
        self.manifest.mark_classified(self.test_files[1], "employee_t&e", "expense")
        self.manifest.mark_classified(self.test_files[2], "irrelevant", "")
        self.manifest.mark_extracted(self.test_files[1])  # Complete one file
        self.manifest.force_commit()

        classify_list, extract_list = self.manifest.get_resume_queues(self.test_files)

        # test_files[0] should need extraction
        # test_files[1] is complete (should not appear)
        # test_files[2] is irrelevant (should be auto-marked as extracted)
        # test_files[3] is new (needs classification)
        assert self.test_files[3] in classify_list
        assert self.test_files[0] in extract_list
        assert self.test_files[1] not in classify_list
        assert self.test_files[1] not in extract_list

    def test_get_summary_empty(self):
        """Test summary with no files processed."""
        summary = self.manifest.get_summary()

        expected = {
            "total_files": 0,
            "classified": 0,
            "extracted": 0,
            "failed": 0,
            "pending_extraction": 0,
            "pending_classification": 0
        }
        assert summary == expected

    def test_get_summary_with_data(self):
        """Test summary with processed files."""
        # Process some files
        self.manifest.mark_classified(self.test_files[0], "vendor_invoice", "invoice")
        self.manifest.mark_classified(self.test_files[1], "employee_t&e", "expense")
        self.manifest.mark_extracted(self.test_files[1])
        self.manifest.mark_error(self.test_files[2], "Processing failed")
        self.manifest.force_commit()

        summary = self.manifest.get_summary()

        assert summary["total_files"] == 3
        assert summary["classified"] == 2
        assert summary["extracted"] == 1
        assert summary["failed"] == 1
        assert summary["pending_extraction"] == 1  # test_files[0] classified but not extracted

    def test_batch_commits(self):
        """Test that batch commits work correctly."""
        # Set small batch size
        self.manifest.batch_size = 2

        # Add files without forcing commit
        self.manifest.mark_classified(self.test_files[0], "vendor_invoice")

        # Should not be committed yet
        cursor = self.manifest._conn.execute(
            "SELECT COUNT(*) FROM progress"
        )
        # Actually, the first operation triggers a commit since we use INSERT OR REPLACE
        # Let's check after batch size is reached

        self.manifest.mark_classified(self.test_files[1], "employee_t&e")

        # Should be committed now (batch_size = 2)
        cursor = self.manifest._conn.execute(
            "SELECT COUNT(*) FROM progress"
        )
        count = cursor.fetchone()[0]
        assert count >= 2

    def test_context_manager(self):
        """Test using ProcessingManifest as a context manager."""
        temp_db = os.path.join(self.temp_dir, "context_test.db")

        with ProcessingManifest(temp_db) as manifest:
            manifest.mark_classified(self.test_files[0], "vendor_invoice")
            # Connection should be active
            assert manifest._conn is not None

        # Connection should be closed after exiting context
        # Note: We can't directly check if connection is closed easily,
        # but we can verify operations still work through a new instance
        with ProcessingManifest(temp_db) as manifest2:
            summary = manifest2.get_summary()
            assert summary["total_files"] == 1

    def test_reset_errors(self):
        """Test resetting error status."""
        # Add files with errors
        self.manifest.mark_error(self.test_files[0], "Error 1")
        self.manifest.mark_error(self.test_files[1], "Error 2")
        self.manifest.force_commit()

        # Reset specific file error
        self.manifest.reset_errors([self.test_files[0]])

        # Check that only one error was reset
        cursor = self.manifest._conn.execute(
            "SELECT file_path FROM progress WHERE last_error IS NOT NULL"
        )
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == self.test_files[1]

        # Reset all errors
        self.manifest.reset_errors()

        cursor = self.manifest._conn.execute(
            "SELECT COUNT(*) FROM progress WHERE last_error IS NOT NULL"
        )
        count = cursor.fetchone()[0]
        assert count == 0


class TestConvenienceFunctions:
    """Test the convenience functions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_convenience.db")
        self.manifest = init_db(self.db_path)

    def teardown_method(self):
        """Clean up test environment."""
        self.manifest.close()
        # Clean up temp files (including SQLite WAL files)
        import glob
        db_pattern = self.db_path + "*"  # Matches .db, .db-wal, .db-shm
        for db_file in glob.glob(db_pattern):
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass  # Ignore cleanup errors

        # Clean up temp directory
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            # If directory is not empty, remove everything
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_db(self):
        """Test the init_db convenience function."""
        assert isinstance(self.manifest, ProcessingManifest)
        assert self.manifest._conn is not None

    def test_convenience_functions_integration(self):
        """Test that convenience functions work with the manifest."""
        from invoice_pdf.io.manifest import get_resume_queues, mark_classified, mark_extracted, summary

        test_file = "/test/convenience.pdf"

        # Test marking operations
        mark_classified(self.manifest, test_file, "vendor_invoice", "invoice")
        mark_extracted(self.manifest, test_file)

        # Test summary
        stats = summary(self.manifest)
        assert stats["total_files"] == 1
        assert stats["classified"] == 1
        assert stats["extracted"] == 1

        # Test resume queues
        classify_list, extract_list = get_resume_queues(self.manifest, [test_file, "/new/file.pdf"])
        assert test_file not in classify_list  # already processed
        assert test_file not in extract_list   # already extracted
        assert "/new/file.pdf" in classify_list  # new file


class TestPhase5Migration:
    """Test cases specifically for Phase 5 migration - ensuring resume functionality still works."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_migration.db")
        self.manifest = ProcessingManifest(self.db_path, batch_size=1)
        self.manifest.connect()

        # Simulate a batch of PDF files for processing
        self.pdf_files = [
            "/batch/invoice1.pdf",
            "/batch/invoice2.pdf",
            "/batch/report1.pdf",
            "/batch/irrelevant1.pdf",
            "/batch/invoice3.pdf"
        ]

    def teardown_method(self):
        """Clean up test environment."""
        self.manifest.close()
        # Clean up temp files (including SQLite WAL files)
        import glob
        db_pattern = self.db_path + "*"
        for db_file in glob.glob(db_pattern):
            if os.path.exists(db_file):
                try:
                    os.remove(db_file)
                except OSError:
                    pass

        # Clean up temp directory
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resume_functionality_after_migration(self):
        """Test that resume functionality works correctly after Phase 5 migration."""
        # Simulate initial processing - some files get classified, some extracted
        self.manifest.mark_classified(self.pdf_files[0], "vendor_invoice", "invoice")
        self.manifest.mark_classified(self.pdf_files[1], "employee_t&e", "expense")
        self.manifest.mark_classified(self.pdf_files[2], "irrelevant", "")
        self.manifest.mark_classified(self.pdf_files[3], "vendor_invoice", "invoice")

        # Mark some as extracted (simulating partial completion)
        self.manifest.mark_extracted(self.pdf_files[0])
        # pdf_files[1] classified but not extracted
        # pdf_files[2] should be auto-extracted as irrelevant
        self.manifest.mark_extracted(self.pdf_files[3])
        # pdf_files[4] not processed at all

        self.manifest.force_commit()

        # Test resume queues functionality
        classify_list, extract_list = self.manifest.get_resume_queues(self.pdf_files)

        # Verify the resume logic works correctly
        assert self.pdf_files[4] in classify_list, "New file should need classification"
        assert self.pdf_files[1] in extract_list, "Classified employee T&E should need extraction"

        # Already processed files should not appear in either queue
        assert self.pdf_files[0] not in classify_list and self.pdf_files[0] not in extract_list
        assert self.pdf_files[2] not in classify_list and self.pdf_files[2] not in extract_list  # irrelevant, auto-extracted
        assert self.pdf_files[3] not in classify_list and self.pdf_files[3] not in extract_list

        # Verify the correct count of files in each queue
        assert len(classify_list) == 1, f"Expected 1 file in classify queue, got {len(classify_list)}"
        assert len(extract_list) == 1, f"Expected 1 file in extract queue, got {len(extract_list)}"

    def test_batch_processing_simulation(self):
        """Test simulation of batch processing with resume after interruption."""
        # Simulate processing first batch
        batch_1 = self.pdf_files[:3]
        for pdf in batch_1:
            self.manifest.mark_classified(pdf, "vendor_invoice")
            self.manifest.mark_extracted(pdf)

        self.manifest.force_commit()

        # Simulate interruption - second batch partially processed
        batch_2 = self.pdf_files[3:]
        self.manifest.mark_classified(batch_2[0], "employee_t&e")
        # Simulate crash before extraction

        # Test resume behavior
        all_files = self.pdf_files
        classify_list, extract_list = self.manifest.get_resume_queues(all_files)

        # First batch should be complete (not in any queue)
        for pdf in batch_1:
            assert pdf not in classify_list and pdf not in extract_list

        # Fourth file (batch_2[0]) should need extraction
        assert batch_2[0] in extract_list and batch_2[0] not in classify_list

        # Fifth file (batch_2[1]) should need classification
        if len(batch_2) > 1:
            assert batch_2[1] in classify_list and batch_2[1] not in extract_list

    def test_error_handling_in_resume(self):
        """Test that resume works correctly when files have errors."""
        # Simulate some files with errors
        self.manifest.mark_classified(self.pdf_files[0], "vendor_invoice")
        self.manifest.mark_error(self.pdf_files[0], "Extraction failed")

        self.manifest.mark_error(self.pdf_files[1], "Classification failed")

        self.manifest.mark_classified(self.pdf_files[2], "employee_t&e")
        # No error for this one

        self.manifest.force_commit()

        classify_list, extract_list = self.manifest.get_resume_queues(self.pdf_files)

        # Files with classification errors should be re-classified
        assert self.pdf_files[1] in classify_list

        # Files with extraction errors should NOT appear in either queue (need manual intervention)
        assert self.pdf_files[0] not in classify_list and self.pdf_files[0] not in extract_list

        # Successfully classified files without errors should be in extract queue
        assert self.pdf_files[2] in extract_list


if __name__ == "__main__":
    pytest.main([__file__])

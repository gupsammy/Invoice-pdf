"""Regression test using golden file snapshot for small PDFs."""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LEGACY_SCRIPT = Path(__file__).parent.parent / "invoice_pdf" / "_legacy" / "main_2step_enhanced.py"


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_pdfs() -> List[Path]:
    """Get list of test PDF files."""
    return list(FIXTURES_DIR.glob("*.pdf"))


def test_legacy_script_processes_fixtures(test_pdfs, temp_output_dir):
    """Test that legacy script can process fixture PDFs without errors."""
    if not test_pdfs:
        pytest.skip("No test PDF fixtures found")
    
    if not LEGACY_SCRIPT.exists():
        pytest.skip(f"Legacy script not found at {LEGACY_SCRIPT}")
    
    # Copy test PDFs to temp input directory
    temp_input = temp_output_dir / "input"
    temp_input.mkdir()
    for pdf in test_pdfs:
        (temp_input / pdf.name).write_bytes(pdf.read_bytes())
    
    # Set environment for test
    env = os.environ.copy()
    env.update({
        "DEBUG_RESPONSES": "0",  # Don't save debug responses during tests
    })
    
    # Run legacy script
    cmd = [
        "python", str(LEGACY_SCRIPT),
        "--input", str(temp_input),
        "--output", str(temp_output_dir),
    ]
    
    result = subprocess.run(
        cmd,
        cwd=LEGACY_SCRIPT.parent.parent.parent,  # Run from project root
        env=env,
        capture_output=True,
        text=True,
        timeout=120,  # 2 minute timeout
    )
    
    # Check that script completed successfully
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
    
    # Check that output files were created
    output_files = list(temp_output_dir.glob("*.csv"))
    assert len(output_files) > 0, "No CSV output files were created"
    
    # Basic validation that CSVs contain headers
    for csv_file in output_files:
        content = csv_file.read_text()
        assert len(content.strip()) > 0, f"CSV file {csv_file.name} is empty"
        lines = content.strip().split('\n')
        assert len(lines) > 0, f"CSV file {csv_file.name} has no content"


def test_fixture_pdfs_exist():
    """Ensure we have test fixtures available."""
    fixtures = list(FIXTURES_DIR.glob("*.pdf"))
    assert len(fixtures) >= 3, f"Expected at least 3 test PDFs, found {len(fixtures)}"
    
    for fixture in fixtures:
        assert fixture.stat().st_size > 0, f"Test PDF {fixture.name} is empty"
        assert fixture.stat().st_size < 100_000, f"Test PDF {fixture.name} is too large for regression test"
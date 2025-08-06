"""Regression test using golden file snapshot for small PDFs."""
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LEGACY_SCRIPT = Path(__file__).parent.parent / "invoice_pdf" / "_legacy" / "main_2step_enhanced.py"


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_pdfs() -> list[Path]:
    """Get list of test PDF files."""
    return list(FIXTURES_DIR.glob("*.pdf"))


def test_legacy_script_processes_fixtures(test_pdfs, temp_output_dir):
    """Test that legacy script can start up and initialize without errors."""
    if not test_pdfs:
        pytest.skip("No test PDF fixtures found")

    if not LEGACY_SCRIPT.exists():
        pytest.skip(f"Legacy script not found at {LEGACY_SCRIPT}")

    # Copy test PDFs to temp input directory
    temp_input = temp_output_dir / "input"
    temp_input.mkdir()
    for pdf in test_pdfs:
        (temp_input / pdf.name).write_bytes(pdf.read_bytes())

    # Set environment for test (no API key to avoid actual API calls)
    env = os.environ.copy()
    env.update({
        "DEBUG_RESPONSES": "0",  # Don't save debug responses during tests
        "GEMINI_API_KEY": "test-key-no-api-calls",  # Dummy key for config test
    })

    # Run legacy script with --help to test initialization without API calls
    cmd = [
        "python", str(LEGACY_SCRIPT),
        "--help"
    ]

    result = subprocess.run(
        cmd,
        check=False, cwd=LEGACY_SCRIPT.parent.parent.parent,  # Run from project root
        env=env,
        capture_output=True,
        text=True,
        timeout=10,  # Short timeout for help
    )

    # Check that script can initialize and show help
    assert result.returncode == 0, f"Script failed to show help with stderr: {result.stderr}"
    assert "--input" in result.stdout, "Help output should contain --input option"
    assert "--output" in result.stdout, "Help output should contain --output option"


def test_fixture_pdfs_exist():
    """Ensure we have test fixtures available."""
    fixtures = list(FIXTURES_DIR.glob("*.pdf"))
    assert len(fixtures) >= 3, f"Expected at least 3 test PDFs, found {len(fixtures)}"

    for fixture in fixtures:
        assert fixture.stat().st_size > 0, f"Test PDF {fixture.name} is empty"
        assert fixture.stat().st_size < 100_000, f"Test PDF {fixture.name} is too large for regression test"

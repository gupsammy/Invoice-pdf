"""Integration tests with mocked Gemini API responses."""

import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "invoice_pdf"))



# Load mock responses
FIXTURES_DIR = Path(__file__).parent / "fixtures"
with open(FIXTURES_DIR / "mock_responses.json") as f:
    MOCK_RESPONSES = json.load(f)


class MockGeminiResponse:
    """Mock response from Gemini API."""

    def __init__(self, text: str):
        self.text = text


@pytest.fixture
def sample_pdf_bytes():
    """Sample PDF bytes for testing."""
    # Minimal valid PDF structure
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f 
0000000015 00000 n 
0000000074 00000 n 
0000000131 00000 n 
trailer
<< /Size 4 /Root 1 0 R >>
startxref
190
%%EOF"""


@pytest.fixture
def temp_pdf_file(sample_pdf_bytes):
    """Create a temporary PDF file for testing."""
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(sample_pdf_bytes)
        tmp_path = Path(tmp.name)

    yield tmp_path

    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()


@pytest.fixture
def mock_genai_client():
    """Mock Gemini AI client."""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    return client


@pytest.mark.asyncio
async def test_classify_document_vendor_invoice(temp_pdf_file, mock_genai_client):
    """Test classification of vendor invoice document."""
    from _legacy.main_2step_enhanced import classify_document_async
    from core.rate_limit import CapacityLimiter

    # Setup mock response
    mock_response = MOCK_RESPONSES["classification"]["vendor_invoice"]
    mock_genai_client.aio.models.generate_content.return_value = MockGeminiResponse(
        json.dumps(mock_response)
    )

    # Create capacity limiter
    capacity_limiter = CapacityLimiter(1)

    # Test classification
    result = await classify_document_async(
        str(temp_pdf_file),
        mock_genai_client,
        capacity_limiter,
        "/tmp"
    )

    assert result is not None
    assert result["classification"] == "vendor_invoice"
    assert result["confidence"] == 0.95
    assert "vendor letterhead" in result["key_indicators"]


@pytest.mark.asyncio
async def test_classify_document_employee_te(temp_pdf_file, mock_genai_client):
    """Test classification of employee T&E document."""
    from _legacy.main_2step_enhanced import classify_document_async
    from core.rate_limit import CapacityLimiter

    # Setup mock response
    mock_response = MOCK_RESPONSES["classification"]["employee_te"]
    mock_genai_client.aio.models.generate_content.return_value = MockGeminiResponse(
        json.dumps(mock_response)
    )

    capacity_limiter = CapacityLimiter(1)

    result = await classify_document_async(
        str(temp_pdf_file),
        mock_genai_client,
        capacity_limiter,
        "/tmp"
    )

    assert result is not None
    assert result["classification"] == "employee_t&e"
    assert result["confidence"] == 0.92
    assert "employee code" in result["key_indicators"]


@pytest.mark.asyncio
async def test_classify_document_irrelevant(temp_pdf_file, mock_genai_client):
    """Test classification of irrelevant document."""
    from _legacy.main_2step_enhanced import classify_document_async
    from core.rate_limit import CapacityLimiter

    # Setup mock response
    mock_response = MOCK_RESPONSES["classification"]["irrelevant"]
    mock_genai_client.aio.models.generate_content.return_value = MockGeminiResponse(
        json.dumps(mock_response)
    )

    capacity_limiter = CapacityLimiter(1)

    result = await classify_document_async(
        str(temp_pdf_file),
        mock_genai_client,
        capacity_limiter,
        "/tmp"
    )

    assert result is not None
    assert result["classification"] == "irrelevant"
    assert result["confidence"] == 0.88


@pytest.mark.asyncio
async def test_extract_vendor_invoice(temp_pdf_file, mock_genai_client):
    """Test extraction from vendor invoice."""
    from _legacy.main_2step_enhanced import extract_document_data_async
    from core.rate_limit import CapacityLimiter

    # Setup mock response
    mock_response = MOCK_RESPONSES["extraction"]["vendor_invoice"]
    mock_genai_client.aio.models.generate_content.return_value = MockGeminiResponse(
        json.dumps(mock_response)
    )

    capacity_limiter = CapacityLimiter(1)

    result = await extract_document_data_async(
        str(temp_pdf_file),
        "vendor_invoice",
        mock_genai_client,
        capacity_limiter,
        "/tmp"
    )

    assert result is not None
    assert result["vendor_name"] == "Tech Solutions Inc."
    assert result["invoice_number"] == "INV-2024-001"
    assert result["total_amount"] == 11800.00
    assert len(result["line_items"]) == 1


@pytest.mark.asyncio
async def test_extract_employee_te(temp_pdf_file, mock_genai_client):
    """Test extraction from employee T&E document."""
    from _legacy.main_2step_enhanced import extract_document_data_async
    from core.rate_limit import CapacityLimiter

    # Setup mock response
    mock_response = MOCK_RESPONSES["extraction"]["employee_te"]
    mock_genai_client.aio.models.generate_content.return_value = MockGeminiResponse(
        json.dumps(mock_response)
    )

    capacity_limiter = CapacityLimiter(1)

    result = await extract_document_data_async(
        str(temp_pdf_file),
        "employee_t&e",
        mock_genai_client,
        capacity_limiter,
        "/tmp"
    )

    assert result is not None
    assert result["employee_name"] == "John Smith"
    assert result["employee_code"] == "EMP001"
    assert result["total_amount"] == 5500.00
    assert len(result["expense_categories"]) == 3


@pytest.mark.asyncio
async def test_classify_api_failure_retry(temp_pdf_file, mock_genai_client):
    """Test classification handles API failures with retry."""
    from _legacy.main_2step_enhanced import classify_document_async
    from core.rate_limit import CapacityLimiter

    # Setup mock to fail then succeed
    side_effects = [
        Exception("Transient upstream error – will retry"),
        MockGeminiResponse(json.dumps(MOCK_RESPONSES["classification"]["vendor_invoice"]))
    ]
    mock_genai_client.aio.models.generate_content.side_effect = side_effects

    capacity_limiter = CapacityLimiter(1)

    result = await classify_document_async(
        str(temp_pdf_file),
        mock_genai_client,
        capacity_limiter,
        "/tmp"
    )

    # Should succeed after retry
    assert result is not None
    assert result["classification"] == "vendor_invoice"


@pytest.mark.asyncio
async def test_classify_exhausted_retries(temp_pdf_file, mock_genai_client):
    """Test classification fails after exhausting retries."""
    from _legacy.main_2step_enhanced import classify_document_with_metadata
    from core.rate_limit import CapacityLimiter

    # Setup mock to always fail
    mock_genai_client.aio.models.generate_content.side_effect = Exception("Persistent API error")

    capacity_limiter = CapacityLimiter(1)

    # This should catch the APIError and return None for backward compatibility
    pdf_path, result = await classify_document_with_metadata(
        str(temp_pdf_file),
        mock_genai_client,
        capacity_limiter,
        "/tmp"
    )

    assert pdf_path == str(temp_pdf_file)
    assert result is None  # Should return None after catching APIError


@pytest.mark.asyncio
async def test_classify_invalid_json_response(temp_pdf_file, mock_genai_client):
    """Test classification handles invalid JSON responses."""
    from _legacy.main_2step_enhanced import classify_document_with_metadata
    from core.rate_limit import CapacityLimiter

    # Setup mock with invalid JSON
    mock_genai_client.aio.models.generate_content.return_value = MockGeminiResponse(
        "This is not valid JSON response from the API"
    )

    capacity_limiter = CapacityLimiter(1)

    # Should catch InvalidAPIResponseError and return None
    pdf_path, result = await classify_document_with_metadata(
        str(temp_pdf_file),
        mock_genai_client,
        capacity_limiter,
        "/tmp"
    )

    assert pdf_path == str(temp_pdf_file)
    assert result is None


@pytest.mark.asyncio
async def test_extract_api_failure(temp_pdf_file, mock_genai_client):
    """Test extraction handles API failures."""
    from _legacy.main_2step_enhanced import extract_document_with_metadata
    from core.rate_limit import CapacityLimiter

    # Setup mock to fail
    mock_genai_client.aio.models.generate_content.side_effect = Exception("API Error")

    capacity_limiter = CapacityLimiter(1)

    # Should catch APIError and return None
    (pdf_path, doc_type), result = await extract_document_with_metadata(
        str(temp_pdf_file),
        "vendor_invoice",
        mock_genai_client,
        capacity_limiter,
        "/tmp"
    )

    assert pdf_path == str(temp_pdf_file)
    assert doc_type == "vendor_invoice"
    assert result is None


@pytest.mark.asyncio
async def test_concurrent_classification(temp_pdf_file, mock_genai_client):
    """Test concurrent classification operations."""
    import asyncio

    from _legacy.main_2step_enhanced import classify_document_with_metadata
    from core.rate_limit import CapacityLimiter

    # Setup mock response
    mock_response = MOCK_RESPONSES["classification"]["vendor_invoice"]
    mock_genai_client.aio.models.generate_content.return_value = MockGeminiResponse(
        json.dumps(mock_response)
    )

    capacity_limiter = CapacityLimiter(2)  # Allow 2 concurrent operations

    # Run multiple classifications concurrently
    tasks = []
    for i in range(3):
        task = classify_document_with_metadata(
            str(temp_pdf_file),
            mock_genai_client,
            capacity_limiter,
            "/tmp"
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # All should succeed
    for pdf_path, result in results:
        assert pdf_path == str(temp_pdf_file)
        assert result is not None
        assert result["classification"] == "vendor_invoice"


@pytest.mark.asyncio
async def test_pdf_processing_error_handling(mock_genai_client):
    """Test handling of PDF processing errors."""
    from _legacy.main_2step_enhanced import classify_document_with_metadata
    from core.rate_limit import CapacityLimiter

    # Test with non-existent file
    capacity_limiter = CapacityLimiter(1)

    pdf_path, result = await classify_document_with_metadata(
        "/nonexistent/file.pdf",
        mock_genai_client,
        capacity_limiter,
        "/tmp"
    )

    assert pdf_path == "/nonexistent/file.pdf"
    assert result is None  # Should handle PDF processing error gracefully


def test_mock_response_structure():
    """Test that mock responses have the expected structure."""
    # Test classification responses
    for doc_type in ["vendor_invoice", "employee_te", "irrelevant"]:
        response = MOCK_RESPONSES["classification"][doc_type]

        # Required fields
        assert "classification" in response
        assert "confidence" in response
        assert "reasoning" in response
        assert "key_indicators" in response
        assert "classification_model" in response
        assert "total_pages_in_pdf" in response
        assert "pages_analyzed" in response

        # Confidence should be between 0 and 1
        assert 0 <= response["confidence"] <= 1

    # Test extraction responses
    vendor_response = MOCK_RESPONSES["extraction"]["vendor_invoice"]
    assert "vendor_name" in vendor_response
    assert "invoice_number" in vendor_response
    assert "total_amount" in vendor_response

    employee_response = MOCK_RESPONSES["extraction"]["employee_te"]
    assert "employee_name" in employee_response
    assert "employee_code" in employee_response
    assert "total_amount" in employee_response

"""Test integration between models and legacy code."""
import pytest

from invoice_pdf.core.models import (
    ClassificationResult,
    DocumentType,
    classification_result_to_dict,
)


def test_classification_result_legacy_dict_compatibility():
    """Test that ClassificationResult can be created and converted to legacy dict."""
    # Create a ClassificationResult as the legacy code would
    result = ClassificationResult(
        file_name="test.pdf",
        file_path="/path/to/test.pdf",
        classification=DocumentType.VENDOR_INVOICE,
        confidence=0.85,
        reasoning="Document contains vendor letterhead",
        key_indicators=["invoice_number", "vendor_info"],
        classification_model="gemini-2.5-flash",
        total_pages_in_pdf=3,
        pages_analyzed=3,
        has_vendor_letterhead=True,
        appears_financial=True
    )
    
    # Convert to dict as legacy code expects
    legacy_dict = classification_result_to_dict(result)
    
    # Verify all expected fields are present
    assert legacy_dict["file_name"] == "test.pdf"
    assert legacy_dict["classification"] == "vendor_invoice"
    assert legacy_dict["confidence"] == 0.85
    assert legacy_dict["has_vendor_letterhead"] == True
    assert legacy_dict["appears_financial"] == True
    
    # Should be serializable (CSV writers need this)
    import json
    json_str = json.dumps(legacy_dict, default=str)
    assert "test.pdf" in json_str


def test_preprocessing_failure_result_creation():
    """Test preprocessing failure result creation as it would be used in legacy code."""
    from invoice_pdf._legacy.main_2step_enhanced import create_preprocessing_failure_result
    
    # This should return a ClassificationResult
    result = create_preprocessing_failure_result("/path/to/failed.pdf", "Cannot read PDF")
    
    # Check attributes instead of isinstance due to potential module loading issues
    assert hasattr(result, 'classification')
    assert hasattr(result, 'confidence')
    assert hasattr(result, 'file_name')
    assert result.classification.value == "processing_failed"
    assert result.confidence == 1.0
    assert "Cannot read PDF" in result.reasoning
    assert result.total_pages_in_pdf == 0
    
    # Should convert to legacy dict properly
    legacy_dict = classification_result_to_dict(result)
    assert legacy_dict["classification"] == "processing_failed"
    assert legacy_dict["file_name"] == "failed.pdf"


def test_pydantic_validation_in_classification():
    """Test that pydantic validation works for classification results."""
    # Valid confidence should work
    result = ClassificationResult(
        file_name="test.pdf",
        file_path="/test.pdf",
        classification=DocumentType.VENDOR_INVOICE,
        confidence=0.5,
        reasoning="Test",
        classification_model="test-model",
        total_pages_in_pdf=1,
        pages_analyzed=1
    )
    assert result.confidence == 0.5
    
    # Invalid confidence should be caught
    with pytest.raises(ValueError):
        ClassificationResult(
            file_name="test.pdf",
            file_path="/test.pdf",
            classification=DocumentType.VENDOR_INVOICE,
            confidence=1.5,  # Invalid: > 1.0
            reasoning="Test",
            classification_model="test-model",
            total_pages_in_pdf=1,
            pages_analyzed=1
        )


def test_enum_handling_in_classification():
    """Test that enum values work properly."""
    result = ClassificationResult(
        file_name="test.pdf",
        file_path="/test.pdf",
        classification="vendor_invoice",  # String input
        confidence=0.9,
        reasoning="Test",
        classification_model="test-model",
        total_pages_in_pdf=1,
        pages_analyzed=1
    )
    
    # Should convert to proper enum
    assert result.classification == DocumentType.VENDOR_INVOICE
    assert isinstance(result.classification, DocumentType)
    
    # Should serialize back to string
    legacy_dict = classification_result_to_dict(result)
    assert legacy_dict["classification"] == "vendor_invoice"


def test_backwards_compatibility_with_existing_data():
    """Test that existing data structures can create models."""
    legacy_classification_data = {
        "file_name": "invoice123.pdf",
        "file_path": "/invoices/invoice123.pdf",
        "classification": "employee_t&e",
        "confidence": 0.92,
        "reasoning": "Employee expense form detected",
        "key_indicators": ["employee_codes", "travel_dates"],
        "classification_model": "gemini-2.5-flash",
        "total_pages_in_pdf": 5,
        "pages_analyzed": 5,
        "has_employee_codes": True,
        "has_travel_dates": True,
        "appears_financial": True,
        "primary_document_type": "employee_expense_form"
    }
    
    # Should work with existing data
    result = ClassificationResult(**legacy_classification_data)
    
    assert result.file_name == "invoice123.pdf"
    assert result.classification == DocumentType.EMPLOYEE_TE
    assert result.has_employee_codes == True
    assert result.has_travel_dates == True
    
    # Convert back should preserve all data
    converted = classification_result_to_dict(result)
    assert converted["file_name"] == "invoice123.pdf"
    assert converted["classification"] == "employee_t&e"
    assert converted["has_employee_codes"] == True
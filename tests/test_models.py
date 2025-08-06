"""Tests for data models."""
import json

import pytest

from invoice_pdf.core.models import (
    ClassificationResult,
    CurrencyCode,
    DocumentType,
    EmployeeExtractionResult,
    InvoiceType,
    ProcessingResult,
    RegistrationNumber,
    VendorExtractionResult,
    classification_result_to_dict,
)


@pytest.fixture
def sample_classification_data():
    """Sample classification data matching legacy format."""
    return {
        "file_name": "test_invoice.pdf",
        "file_path": "/path/to/test_invoice.pdf",
        "classification": "vendor_invoice",
        "confidence": 0.95,
        "reasoning": "Document contains vendor letterhead and invoice numbers",
        "key_indicators": ["invoice_number", "vendor_letterhead", "tax_information"],
        "classification_model": "gemini-2.5-flash",
        "total_pages_in_pdf": 3,
        "pages_analyzed": 3,
        "classification_notes": "Clear vendor invoice with GST details",
        "has_employee_codes": False,
        "has_vendor_letterhead": True,
        "has_invoice_numbers": True,
        "has_travel_dates": False,
        "appears_financial": True,
        "has_amount_calculations": True,
        "has_tax_information": True,
        "contains_multiple_doc_types": False,
        "primary_document_type": "vendor_invoice"
    }


@pytest.fixture
def sample_vendor_extraction_data():
    """Sample vendor extraction data."""
    return {
        "file_name": "vendor_invoice.pdf",
        "file_path": "/path/to/vendor_invoice.pdf",
        "extraction_model": "gemini-2.5-pro",
        "document_type_processed": "vendor_invoice",
        "total_pages_in_pdf": 2,
        "document_status": {
            "readable": True,
            "contains_invoices": True,
            "document_type": "vendor_invoice",
            "multiple_documents": False,
            "orientation_issues": False
        },
        "extracted_data": [{
            "data_source": "vendor_invoice",
            "issuer": "ABC Corporation",
            "consignor": "ABC Corporation",
            "consignee": "XYZ Ltd",
            "vendor_name": "ABC Corporation",
            "original_vendor_name": "",
            "invoice_type": "invoice",
            "pan": "ABCDE1234F",
            "registration_numbers": [
                {"type": "GSTIN", "value": "27ABCDE1234F1Z5"}
            ],
            "invoice_date": "2024-01-15",
            "document_number": "PO123",
            "invoice_number": "INV-2024-001",
            "description": "Software licensing fees",
            "basic_amount": 10000.0,
            "tax_amount": 1800.0,
            "total_amount": 11800.0,
            "currency_code": "INR",
            "original_amount": None,
            "exchange_rate": None,
            "amount_calculated": False,
            "calculation_method": "extracted_from_total",
            "is_main_invoice": True,
            "page_numbers": [1, 2]
        }],
        "processing_notes": "Successfully extracted all required fields"
    }


@pytest.fixture
def sample_employee_extraction_data():
    """Sample employee extraction data."""
    return {
        "file_name": "employee_te.pdf",
        "file_path": "/path/to/employee_te.pdf",
        "extraction_model": "gemini-2.5-pro",
        "document_type_processed": "employee_t&e",
        "total_pages_in_pdf": 4,
        "document_status": {
            "readable": True,
            "contains_invoices": True,
            "document_type": "employee_reimbursement",
            "multiple_documents": False,
            "orientation_issues": False
        },
        "extracted_data": [{
            "data_source": "employee_reimbursement",
            "employee_name": "John Doe",
            "employee_code": "EMP001",
            "department": "Sales",
            "invoice_date": "2024-01-20",
            "description": "Business trip to Mumbai - client meetings",
            "basic_amount": 5000.0,
            "tax_amount": 0.0,
            "total_amount": 5000.0,
            "currency_code": "INR",
            "original_amount": None,
            "amount_calculated": False,
            "calculation_method": "extracted_from_total",
            "page_numbers": [1]
        }],
        "processing_notes": "Employee expense form processed successfully"
    }


def test_classification_result_creation(sample_classification_data):
    """Test ClassificationResult model creation and validation."""
    result = ClassificationResult(**sample_classification_data)

    assert result.file_name == "test_invoice.pdf"
    assert result.classification == DocumentType.VENDOR_INVOICE
    assert result.confidence == 0.95
    assert result.total_pages_in_pdf == 3
    assert result.pages_analyzed == 3
    assert result.has_vendor_letterhead == True
    assert result.has_tax_information == True


def test_classification_result_json_roundtrip(sample_classification_data):
    """Test JSON serialization and deserialization."""
    result = ClassificationResult(**sample_classification_data)

    # Convert to JSON and back
    json_str = result.model_dump_json()
    result_dict = json.loads(json_str)
    result_restored = ClassificationResult(**result_dict)

    assert result.file_name == result_restored.file_name
    assert result.classification == result_restored.classification
    assert result.confidence == result_restored.confidence


def test_classification_result_to_dict_conversion(sample_classification_data):
    """Test conversion to legacy dict format."""
    result = ClassificationResult(**sample_classification_data)
    legacy_dict = classification_result_to_dict(result)

    # Check that all original fields are present
    assert legacy_dict["file_name"] == "test_invoice.pdf"
    assert legacy_dict["classification"] == "vendor_invoice"
    assert legacy_dict["confidence"] == 0.95
    assert legacy_dict["has_vendor_letterhead"] == True


def test_vendor_extraction_result_creation(sample_vendor_extraction_data):
    """Test VendorExtractionResult model creation."""
    result = VendorExtractionResult(**sample_vendor_extraction_data)

    assert result.file_name == "vendor_invoice.pdf"
    assert result.extraction_model == "gemini-2.5-pro"
    assert result.document_status.readable == True
    assert len(result.extracted_data) == 1

    extract_data = result.extracted_data[0]
    assert extract_data.vendor_name == "ABC Corporation"
    assert extract_data.total_amount == 11800.0
    assert extract_data.currency_code == CurrencyCode.INR
    assert extract_data.invoice_type == InvoiceType.INVOICE


def test_vendor_extraction_json_roundtrip(sample_vendor_extraction_data):
    """Test vendor extraction JSON roundtrip."""
    result = VendorExtractionResult(**sample_vendor_extraction_data)

    # Convert to JSON and back
    json_str = result.model_dump_json()
    result_dict = json.loads(json_str)
    result_restored = VendorExtractionResult(**result_dict)

    assert result.file_name == result_restored.file_name
    assert result.extracted_data[0].vendor_name == result_restored.extracted_data[0].vendor_name
    assert result.extracted_data[0].total_amount == result_restored.extracted_data[0].total_amount


def test_employee_extraction_result_creation(sample_employee_extraction_data):
    """Test EmployeeExtractionResult model creation."""
    result = EmployeeExtractionResult(**sample_employee_extraction_data)

    assert result.file_name == "employee_te.pdf"
    assert result.extraction_model == "gemini-2.5-pro"
    assert result.document_status.readable == True
    assert len(result.extracted_data) == 1

    extract_data = result.extracted_data[0]
    assert extract_data.employee_name == "John Doe"
    assert extract_data.employee_code == "EMP001"
    assert extract_data.total_amount == 5000.0


def test_employee_extraction_json_roundtrip(sample_employee_extraction_data):
    """Test employee extraction JSON roundtrip."""
    result = EmployeeExtractionResult(**sample_employee_extraction_data)

    # Convert to JSON and back
    json_str = result.model_dump_json()
    result_dict = json.loads(json_str)
    result_restored = EmployeeExtractionResult(**result_dict)

    assert result.file_name == result_restored.file_name
    assert result.extracted_data[0].employee_name == result_restored.extracted_data[0].employee_name
    assert result.extracted_data[0].total_amount == result_restored.extracted_data[0].total_amount


def test_processing_result_composite():
    """Test ProcessingResult composite model."""
    classification = ClassificationResult(
        file_name="test.pdf",
        file_path="/test.pdf",
        classification=DocumentType.VENDOR_INVOICE,
        confidence=0.9,
        reasoning="Test reasoning",
        classification_model="test-model",
        total_pages_in_pdf=1,
        pages_analyzed=1
    )

    result = ProcessingResult(
        file_name="test.pdf",
        file_path="/test.pdf",
        classification_result=classification
    )

    assert result.is_successful == True
    assert result.document_type == DocumentType.VENDOR_INVOICE


def test_legacy_compatibility_functions(sample_classification_data):
    """Test legacy compatibility conversion functions."""
    result = ClassificationResult(**sample_classification_data)
    legacy_dict = classification_result_to_dict(result)

    # Should contain all expected legacy fields
    expected_fields = [
        "file_name", "file_path", "classification", "confidence",
        "has_vendor_letterhead", "has_invoice_numbers", "appears_financial"
    ]

    for field in expected_fields:
        assert field in legacy_dict


def test_validation_errors():
    """Test model validation errors."""
    # Test invalid confidence range
    with pytest.raises(ValueError):
        ClassificationResult(
            file_name="test.pdf",
            file_path="/test.pdf",
            classification=DocumentType.VENDOR_INVOICE,
            confidence=1.5,  # Invalid: > 1.0
            reasoning="Test",
            classification_model="test",
            total_pages_in_pdf=1,
            pages_analyzed=1
        )

    # Test pages analyzed exceeding total pages
    result = ClassificationResult(
        file_name="test.pdf",
        file_path="/test.pdf",
        classification=DocumentType.VENDOR_INVOICE,
        confidence=0.9,
        reasoning="Test",
        classification_model="test",
        total_pages_in_pdf=5,
        pages_analyzed=10  # Will be corrected to 5
    )
    assert result.pages_analyzed == 5


def test_registration_number_model():
    """Test RegistrationNumber model."""
    reg = RegistrationNumber(type="GSTIN", value="27ABCDE1234F1Z5")

    assert reg.type == "GSTIN"
    assert reg.value == "27ABCDE1234F1Z5"

    # JSON roundtrip
    json_str = reg.model_dump_json()
    reg_dict = json.loads(json_str)
    reg_restored = RegistrationNumber(**reg_dict)

    assert reg.type == reg_restored.type
    assert reg.value == reg_restored.value


def test_enum_values():
    """Test enum value handling."""
    # Test DocumentType enum
    assert DocumentType.VENDOR_INVOICE.value == "vendor_invoice"
    assert DocumentType.EMPLOYEE_TE.value == "employee_t&e"

    # Test InvoiceType enum
    assert InvoiceType.INVOICE.value == "invoice"
    assert InvoiceType.DEBIT_NOTE.value == "debit_note"

    # Test CurrencyCode enum
    assert CurrencyCode.INR.value == "INR"
    assert CurrencyCode.USD.value == "USD"


def test_model_defaults():
    """Test model default values."""
    # Test minimal ClassificationResult
    result = ClassificationResult(
        file_name="test.pdf",
        file_path="/test.pdf",
        classification=DocumentType.VENDOR_INVOICE,
        confidence=0.9,
        reasoning="Test reasoning",
        classification_model="test-model",
        total_pages_in_pdf=1,
        pages_analyzed=1
    )

    # Check defaults
    assert result.key_indicators == []
    assert result.classification_notes == ""
    assert result.has_employee_codes == False
    assert result.primary_document_type == "unknown"


def test_backwards_compatibility_dict_creation():
    """Test creating models from legacy dict format (backwards compatibility)."""
    legacy_data = {
        "file_name": "legacy.pdf",
        "file_path": "/legacy.pdf",
        "classification": "vendor_invoice",
        "confidence": 0.8,
        "reasoning": "Legacy test",
        "classification_model": "legacy-model",
        "total_pages_in_pdf": 2,
        "pages_analyzed": 2,
        "has_vendor_letterhead": True,
        "appears_financial": True
    }

    # Should work with legacy dict
    result = ClassificationResult(**legacy_data)
    assert result.file_name == "legacy.pdf"
    assert result.classification == DocumentType.VENDOR_INVOICE
    assert result.has_vendor_letterhead == True

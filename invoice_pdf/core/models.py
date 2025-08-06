"""Canonical data models for Invoice PDF processing."""
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Document classification types."""
    VENDOR_INVOICE = "vendor_invoice"
    EMPLOYEE_TE = "employee_t&e"
    IRRELEVANT = "irrelevant"
    PROCESSING_FAILED = "processing_failed"


class InvoiceType(str, Enum):
    """Invoice type classifications."""
    INVOICE = "invoice"
    DEBIT_NOTE = "debit_note"
    CREDIT_NOTE = "credit_note"
    RECEIPT = "receipt"
    GRN = "grn"
    WAYBILL = "waybill"
    CHALLAN = "challan"
    SERVICE_ORDER = "service_order"
    PROFESSIONAL_BILL = "professional_bill"


class CurrencyCode(str, Enum):
    """Common currency codes."""
    INR = "INR"
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


class CalculationMethod(str, Enum):
    """Amount calculation methods."""
    SUM_OF_LINE_ITEMS = "sum_of_line_items"
    EXTRACTED_FROM_TOTAL = "extracted_from_total"
    CALCULATED_FROM_COMPONENTS = "calculated_from_components"
    SUM_OF_CATEGORIES = "sum_of_categories"


class RegistrationNumber(BaseModel):
    """Business registration number."""
    type: str = Field(..., description="Registration type (GST, VAT, CST, TIN, GSTIN)")
    value: str = Field(..., description="Registration number value")


class DocumentCharacteristics(BaseModel):
    """Document characteristics for classification."""
    has_employee_codes: bool = Field(default=False)
    has_vendor_letterhead: bool = Field(default=False)
    has_invoice_numbers: bool = Field(default=False)
    has_travel_dates: bool = Field(default=False)
    appears_financial: bool = Field(default=False)
    has_amount_calculations: bool = Field(default=False)
    has_tax_information: bool = Field(default=False)
    contains_multiple_doc_types: bool = Field(default=False)
    primary_document_type: str = Field(..., description="Primary document type identified")


class DocumentStatus(BaseModel):
    """Status of document processing."""
    readable: bool = Field(..., description="Whether document is readable")
    contains_invoices: bool = Field(..., description="Whether document contains invoice data")
    document_type: str = Field(..., description="Detected document type")
    multiple_documents: bool = Field(default=False, description="Whether PDF contains multiple documents")
    orientation_issues: bool = Field(default=False, description="Whether document has orientation issues")


class ClassificationResult(BaseModel):
    """Result of document classification step."""
    # File metadata
    file_name: str = Field(..., description="PDF file name")
    file_path: str = Field(..., description="Full path to PDF file")
    
    # Classification results
    classification: DocumentType = Field(..., description="Document classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    reasoning: str = Field(..., description="Reasoning for classification")
    key_indicators: List[str] = Field(default_factory=list, description="Key indicators found")
    
    # Document characteristics
    document_characteristics: Optional[DocumentCharacteristics] = None
    
    # Processing metadata
    classification_model: str = Field(..., description="Model used for classification")
    total_pages_in_pdf: int = Field(..., ge=0, description="Total pages in PDF")
    pages_analyzed: int = Field(..., ge=0, description="Number of pages analyzed")
    classification_notes: str = Field(default="", description="Additional classification notes")
    
    # Legacy compatibility fields
    has_employee_codes: bool = Field(default=False)
    has_vendor_letterhead: bool = Field(default=False)
    has_invoice_numbers: bool = Field(default=False)
    has_travel_dates: bool = Field(default=False)
    appears_financial: bool = Field(default=False)
    has_amount_calculations: bool = Field(default=False)
    has_tax_information: bool = Field(default=False)
    contains_multiple_doc_types: bool = Field(default=False)
    primary_document_type: str = Field(default="unknown")
    
    @field_validator('pages_analyzed')
    def pages_analyzed_not_exceed_total(cls, v, info):
        """Ensure pages analyzed doesn't exceed total pages."""
        total_pages = info.data.get('total_pages_in_pdf', 0)
        if v > total_pages:
            return total_pages
        return v


class VendorExtractionData(BaseModel):
    """Vendor invoice extraction data."""
    data_source: str = Field(default="vendor_invoice")
    issuer: str = Field(..., description="Entity that issued/created the invoice")
    consignor: str = Field(..., description="Entity providing/shipping goods")
    consignee: str = Field(..., description="Entity receiving goods/services")
    vendor_name: str = Field(..., description="Primary vendor/service provider name")
    original_vendor_name: str = Field(default="", description="Original name if converted to English")
    invoice_type: InvoiceType = Field(..., description="Type of invoice document")
    pan: str = Field(default="", description="Vendor PAN number")
    registration_numbers: List[RegistrationNumber] = Field(default_factory=list)
    invoice_date: Optional[str] = Field(None, description="Invoice issue date (YYYY-MM-DD)")
    document_number: str = Field(default="", description="PO number or document reference")
    invoice_number: str = Field(..., description="Invoice/bill number")
    description: str = Field(..., description="Goods/services description")
    basic_amount: Optional[float] = Field(None, description="Base amount before taxes")
    tax_amount: Optional[float] = Field(None, description="Total tax amount")
    total_amount: float = Field(..., description="Total invoice value")
    currency_code: CurrencyCode = Field(default=CurrencyCode.INR)
    original_amount: Optional[float] = Field(None, description="Amount in original currency")
    exchange_rate: Optional[float] = Field(None, description="Exchange rate if provided")
    amount_calculated: bool = Field(default=False, description="Whether amount was calculated")
    calculation_method: Optional[CalculationMethod] = Field(None)
    is_main_invoice: bool = Field(default=True, description="Whether this is the main invoice")
    page_numbers: List[int] = Field(default_factory=list, description="Page numbers where data was found")


class EmployeeExtractionData(BaseModel):
    """Employee T&E extraction data."""
    data_source: str = Field(default="employee_reimbursement")
    employee_name: str = Field(..., description="Employee name (person being reimbursed)")
    employee_code: str = Field(..., description="Employee ID or code")
    department: str = Field(default="", description="Employee department or cost center")
    invoice_date: Optional[str] = Field(None, description="Travel end date or submission date (YYYY-MM-DD)")
    description: str = Field(..., description="Travel purpose, destinations, and expense summary")
    basic_amount: Optional[float] = Field(None, description="Total expense amount before deductions")
    tax_amount: Optional[float] = Field(None, description="Any tax amounts or service charges")
    total_amount: float = Field(..., description="Total reimbursement amount")
    currency_code: CurrencyCode = Field(default=CurrencyCode.INR)
    original_amount: Optional[float] = Field(None, description="Amount in original currency")
    amount_calculated: bool = Field(default=False, description="Whether amount was calculated")
    calculation_method: Optional[CalculationMethod] = Field(None)
    page_numbers: List[int] = Field(default_factory=list, description="Page numbers where data was found")


class VendorExtractionResult(BaseModel):
    """Result of vendor invoice extraction step."""
    # File metadata
    file_name: str = Field(..., description="PDF file name")
    file_path: str = Field(..., description="Full path to PDF file")
    
    # Processing metadata
    extraction_model: str = Field(..., description="Model used for extraction")
    document_type_processed: str = Field(..., description="Document type that was processed")
    total_pages_in_pdf: int = Field(..., ge=0, description="Total pages in PDF")
    
    # Document status
    document_status: DocumentStatus
    
    # Extracted data
    extracted_data: List[VendorExtractionData] = Field(default_factory=list)
    
    # Processing notes
    processing_notes: str = Field(default="", description="Detailed processing notes")


class EmployeeExtractionResult(BaseModel):
    """Result of employee T&E extraction step."""
    # File metadata
    file_name: str = Field(..., description="PDF file name")
    file_path: str = Field(..., description="Full path to PDF file")
    
    # Processing metadata
    extraction_model: str = Field(..., description="Model used for extraction")
    document_type_processed: str = Field(..., description="Document type that was processed")
    total_pages_in_pdf: int = Field(..., ge=0, description="Total pages in PDF")
    
    # Document status
    document_status: DocumentStatus
    
    # Extracted data
    extracted_data: List[EmployeeExtractionData] = Field(default_factory=list)
    
    # Processing notes
    processing_notes: str = Field(default="", description="Detailed processing notes")


class ProcessingError(BaseModel):
    """Error information for failed processing."""
    error_message: str = Field(..., description="Error description")
    stage: str = Field(..., description="Processing stage where error occurred")
    timestamp: datetime = Field(default_factory=datetime.now, description="When error occurred")


class ProcessingResult(BaseModel):
    """Union type for all processing results."""
    file_name: str = Field(..., description="PDF file name")
    file_path: str = Field(..., description="Full path to PDF file")
    classification_result: Optional[ClassificationResult] = None
    vendor_extraction_result: Optional[VendorExtractionResult] = None
    employee_extraction_result: Optional[EmployeeExtractionResult] = None
    processing_error: Optional[ProcessingError] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if processing was successful."""
        return self.processing_error is None
    
    @property
    def document_type(self) -> Optional[DocumentType]:
        """Get document type from classification result."""
        if self.classification_result:
            return self.classification_result.classification
        return None


# Utility functions for legacy compatibility
def classification_result_to_dict(result: ClassificationResult) -> Dict[str, Any]:
    """Convert ClassificationResult to legacy dict format."""
    data = result.model_dump()
    
    # Flatten document_characteristics for legacy compatibility
    if result.document_characteristics:
        chars = result.document_characteristics.model_dump()
        for key, value in chars.items():
            if key not in data:  # Don't override direct fields
                data[key] = value
    
    return data


def vendor_extraction_result_to_dict(result: VendorExtractionResult) -> Dict[str, Any]:
    """Convert VendorExtractionResult to legacy dict format."""
    data = result.model_dump()
    
    # Flatten for legacy CSV compatibility
    if result.extracted_data:
        # Take first extraction data for legacy compatibility
        extract_data = result.extracted_data[0].model_dump()
        data.update(extract_data)
    
    # Add flattened document_status fields
    status = result.document_status.model_dump()
    for key, value in status.items():
        data[key] = value
    
    return data


def employee_extraction_result_to_dict(result: EmployeeExtractionResult) -> Dict[str, Any]:
    """Convert EmployeeExtractionResult to legacy dict format."""
    data = result.model_dump()
    
    # Flatten for legacy CSV compatibility
    if result.extracted_data:
        # Take first extraction data for legacy compatibility
        extract_data = result.extracted_data[0].model_dump()
        data.update(extract_data)
    
    # Add flattened document_status fields
    status = result.document_status.model_dump()
    for key, value in status.items():
        data[key] = value
    
    return data
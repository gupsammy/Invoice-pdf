"""Exception hierarchy for Invoice PDF processing."""

from pathlib import Path
from typing import Any, Optional


class InvoicePDFError(Exception):
    """Base exception for all Invoice PDF processing errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class PDFProcessingError(InvoicePDFError):
    """Base class for PDF processing related errors."""
    
    def __init__(
        self, 
        file_path: Path | str, 
        message: str, 
        original_error: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        self.file_path = Path(file_path)
        self.original_error = original_error
        
        full_message = f"PDF processing failed for {self.file_path}: {message}"
        if original_error:
            full_message += f" (Original error: {original_error})"
            
        super().__init__(full_message, details)


class PDFTooLargeError(PDFProcessingError):
    """Raised when a PDF file exceeds the maximum allowed size."""
    
    def __init__(
        self, 
        file_path: Path | str, 
        file_size_mb: float, 
        max_size_mb: float
    ) -> None:
        message = f"PDF size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
        super().__init__(
            file_path, 
            message,
            details={"file_size_mb": file_size_mb, "max_size_mb": max_size_mb}
        )


class InvalidPDFError(PDFProcessingError):
    """Raised when a PDF file is corrupted or invalid."""
    
    def __init__(
        self, 
        file_path: Path | str, 
        reason: str = "PDF file is corrupted or invalid"
    ) -> None:
        super().__init__(file_path, reason)


class PDFReadError(PDFProcessingError):
    """Raised when unable to read or parse a PDF file."""
    
    def __init__(
        self, 
        file_path: Path | str, 
        original_error: Exception
    ) -> None:
        message = "Unable to read PDF file"
        super().__init__(file_path, message, original_error)


class ClassificationError(InvoicePDFError):
    """Base class for document classification errors."""
    
    def __init__(
        self, 
        file_path: Path | str, 
        message: str, 
        model_used: Optional[str] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        self.file_path = Path(file_path)
        self.model_used = model_used
        self.original_error = original_error
        
        full_message = f"Classification failed for {self.file_path}: {message}"
        if model_used:
            full_message += f" (Model: {model_used})"
        if original_error:
            full_message += f" (Original error: {original_error})"
            
        details = {"file_path": str(self.file_path)}
        if model_used:
            details["model_used"] = model_used
            
        super().__init__(full_message, details)


class APIError(ClassificationError):
    """Raised when API calls to Gemini fail."""
    
    def __init__(
        self, 
        file_path: Path | str, 
        api_error: Exception, 
        model_used: Optional[str] = None,
        retry_count: int = 0
    ) -> None:
        message = f"API call failed after {retry_count} retries"
        super().__init__(file_path, message, model_used, api_error)
        self.retry_count = retry_count


class InvalidAPIResponseError(ClassificationError):
    """Raised when API returns invalid or unparseable response."""
    
    def __init__(
        self, 
        file_path: Path | str, 
        response_text: str, 
        model_used: Optional[str] = None,
        parsing_error: Optional[Exception] = None
    ) -> None:
        message = f"API returned invalid response: {response_text[:100]}..."
        super().__init__(file_path, message, model_used, parsing_error)
        self.response_text = response_text


class ExtractionError(InvoicePDFError):
    """Base class for data extraction errors."""
    
    def __init__(
        self, 
        file_path: Path | str, 
        document_type: str,
        message: str, 
        model_used: Optional[str] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        self.file_path = Path(file_path)
        self.document_type = document_type
        self.model_used = model_used
        self.original_error = original_error
        
        full_message = f"Extraction failed for {self.file_path} ({document_type}): {message}"
        if model_used:
            full_message += f" (Model: {model_used})"
        if original_error:
            full_message += f" (Original error: {original_error})"
            
        details = {
            "file_path": str(self.file_path),
            "document_type": document_type
        }
        if model_used:
            details["model_used"] = model_used
            
        super().__init__(full_message, details)


class ValidationError(InvoicePDFError):
    """Raised when extracted data fails validation."""
    
    def __init__(
        self, 
        file_path: Path | str, 
        field_name: str, 
        field_value: Any, 
        validation_rule: str
    ) -> None:
        self.file_path = Path(file_path)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule
        
        message = f"Validation failed for {file_path} field '{field_name}': {validation_rule}"
        details = {
            "file_path": str(self.file_path),
            "field_name": field_name,
            "field_value": field_value,
            "validation_rule": validation_rule
        }
        
        super().__init__(message, details)


class SecurityError(InvoicePDFError):
    """Base class for security-related errors."""
    
    def __init__(
        self, 
        message: str, 
        security_check: str,
        file_path: Optional[Path | str] = None
    ) -> None:
        self.security_check = security_check
        self.file_path = Path(file_path) if file_path else None
        
        full_message = f"Security check failed ({security_check}): {message}"
        details = {"security_check": security_check}
        if file_path:
            details["file_path"] = str(file_path)
            
        super().__init__(full_message, details)


class PathTraversalError(SecurityError):
    """Raised when path traversal attack is detected."""
    
    def __init__(self, attempted_path: str) -> None:
        message = f"Path traversal attempt detected: {attempted_path}"
        super().__init__(message, "path_traversal", attempted_path)
        self.attempted_path = attempted_path


class ResourceLimitError(SecurityError):
    """Raised when resource limits are exceeded."""
    
    def __init__(
        self, 
        resource_type: str, 
        current_usage: float, 
        limit: float, 
        unit: str = ""
    ) -> None:
        unit_str = f" {unit}" if unit else ""
        message = f"{resource_type} usage ({current_usage}{unit_str}) exceeds limit ({limit}{unit_str})"
        super().__init__(message, "resource_limit")
        
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        self.unit = unit


class ConfigurationError(InvoicePDFError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, setting_name: str, issue: str) -> None:
        message = f"Configuration error for '{setting_name}': {issue}"
        super().__init__(message, {"setting_name": setting_name, "issue": issue})
        self.setting_name = setting_name
        self.issue = issue


def wrap_exception(
    func_name: str, 
    original_error: Exception, 
    file_path: Optional[Path | str] = None,
    context: Optional[dict[str, Any]] = None
) -> InvoicePDFError:
    """Wrap generic exceptions in our typed hierarchy."""
    
    context = context or {}
    
    # Handle specific exception types
    if isinstance(original_error, (FileNotFoundError, PermissionError)):
        return PDFProcessingError(
            file_path or "unknown",
            f"File system error in {func_name}",
            original_error,
            context
        )
    
    if isinstance(original_error, (MemoryError, OSError)):
        return ResourceLimitError(
            "memory" if isinstance(original_error, MemoryError) else "system",
            0,  # Unknown current usage
            0,  # Unknown limit
            "bytes"
        )
    
    # Generic wrapper
    return InvoicePDFError(
        f"Unexpected error in {func_name}: {original_error}",
        {**context, "function": func_name, "original_error": str(original_error)}
    )


# Export all exceptions for easy imports
__all__ = [
    "InvoicePDFError",
    "PDFProcessingError",
    "PDFTooLargeError", 
    "InvalidPDFError",
    "PDFReadError",
    "ClassificationError",
    "APIError",
    "InvalidAPIResponseError",
    "ExtractionError",
    "ValidationError",
    "SecurityError",
    "PathTraversalError",
    "ResourceLimitError",
    "ConfigurationError",
    "wrap_exception"
]
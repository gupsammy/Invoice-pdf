"""Security utilities for safe file and path operations."""

import os
import re
from pathlib import Path

from .exceptions import PathTraversalError, ResourceLimitError, SecurityError


def validate_safe_path(file_path: str | Path, allowed_extensions: tuple[str, ...] = (".pdf",)) -> Path:
    """Validate that a file path is safe and within allowed bounds.
    
    Args:
        file_path: Path to validate
        allowed_extensions: Tuple of allowed file extensions (default: PDF only)
        
    Returns:
        Validated Path object
        
    Raises:
        PathTraversalError: If path contains traversal attempts
        SecurityError: If path is outside allowed bounds or has invalid extension
    """
    # Convert to Path object for easier manipulation
    path = Path(file_path).resolve()  # resolve() normalizes and makes absolute

    # Check for path traversal attempts in original string
    path_str = str(file_path)
    dangerous_patterns = [
        "..",           # Parent directory references
        "~",            # Home directory references
        "//",           # Double slashes
        "\\\\",         # Double backslashes (Windows)
        "/./",          # Current directory references
        "\\.\\",        # Current directory references (Windows)
    ]

    for pattern in dangerous_patterns:
        if pattern in path_str:
            raise PathTraversalError(path_str)

    # Additional checks for suspicious characters
    if re.search(r'[<>:"|?*\x00-\x1f]', path_str):
        raise SecurityError(f"Invalid characters in path: {path_str}", "invalid_characters", path_str)

    # Check file extension
    if allowed_extensions and path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
        raise SecurityError(
            f"File extension '{path.suffix}' not allowed. Allowed: {allowed_extensions}",
            "invalid_extension",
            path_str
        )

    # Ensure the path doesn't try to escape common restricted areas
    path_parts = path.parts
    restricted_parts = {
        "..", ".", "~", "/proc", "/sys", "/dev", "/etc",
        "c:\\windows", "c:\\system32", "c:\\program files"
    }

    for part in path_parts:
        if part.lower() in restricted_parts:
            raise PathTraversalError(path_str)

    return path


def validate_resource_limits(
    current_memory_mb: float = 0,
    max_memory_mb: float = 500,
    current_file_handles: int = 0,
    max_file_handles: int = 100
) -> None:
    """Validate that current resource usage is within safe limits.
    
    Args:
        current_memory_mb: Current memory usage in MB
        max_memory_mb: Maximum allowed memory usage in MB
        current_file_handles: Current open file handle count  
        max_file_handles: Maximum allowed file handles
        
    Raises:
        ResourceLimitError: If any resource limit is exceeded
    """
    if current_memory_mb > max_memory_mb:
        raise ResourceLimitError(
            "memory", current_memory_mb, max_memory_mb, "MB"
        )

    if current_file_handles > max_file_handles:
        raise ResourceLimitError(
            "file_handles", current_file_handles, max_file_handles, "handles"
        )


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize a filename for safe filesystem operations.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed filename length
        
    Returns:
        Sanitized filename safe for filesystem operations
        
    Raises:
        SecurityError: If filename cannot be safely sanitized
    """
    if not filename or not filename.strip():
        raise SecurityError("Empty filename provided", "empty_filename")

    # Remove/replace dangerous characters
    # Keep alphanumeric, dots, hyphens, underscores, and spaces
    sanitized = re.sub(r"[^a-zA-Z0-9._\-\s]", "_", filename)

    # Remove multiple consecutive dots (potential traversal)
    sanitized = re.sub(r"\.{2,}", ".", sanitized)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")

    # Ensure filename isn't too long
    if len(sanitized) > max_length:
        # Preserve extension if present
        path = Path(sanitized)
        stem = path.stem[:max_length - len(path.suffix) - 1]
        sanitized = f"{stem}{path.suffix}"

    # Check for reserved names (Windows)
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }

    if Path(sanitized).stem.upper() in reserved_names:
        sanitized = f"safe_{sanitized}"

    if not sanitized:
        raise SecurityError("Filename could not be sanitized safely", "unsanitizable_filename")

    return sanitized


def check_disk_space(path: str | Path, required_mb: float) -> bool:
    """Check if sufficient disk space is available.
    
    Args:
        path: Path to check (can be file or directory)
        required_mb: Required space in MB
        
    Returns:
        True if sufficient space is available
        
    Raises:
        SecurityError: If unable to check disk space
    """
    try:
        path = Path(path)

        # Find existing parent directory to check space
        check_path = path if path.is_dir() else path.parent
        while not check_path.exists() and check_path.parent != check_path:
            check_path = check_path.parent

        # Get disk usage
        statvfs = os.statvfs(check_path)

        # Calculate available space in MB
        available_bytes = statvfs.f_bavail * statvfs.f_frsize
        available_mb = available_bytes / (1024 * 1024)

        return available_mb >= required_mb

    except Exception as e:
        raise SecurityError(f"Unable to check disk space: {e}", "disk_space_check", str(path))


def validate_api_key(api_key: str, min_length: int = 20) -> None:
    """Validate API key format for basic security checks.
    
    Args:
        api_key: API key to validate
        min_length: Minimum required key length
        
    Raises:
        SecurityError: If API key appears invalid or insecure
    """
    if not api_key or not api_key.strip():
        raise SecurityError("Empty API key provided", "empty_api_key")

    api_key = api_key.strip()

    if len(api_key) < min_length:
        raise SecurityError(
            f"API key too short (minimum {min_length} characters)",
            "short_api_key"
        )

    # Check for obviously fake/test keys
    test_patterns = [
        r"^test",
        r"^fake",
        r"^dummy",
        r"^example",
        r"^[0]+$",  # All zeros
        r"^[1]+$",  # All ones
        r"^(abc|123)+$",  # Repeated simple patterns
    ]

    for pattern in test_patterns:
        if re.match(pattern, api_key.lower()):
            raise SecurityError(
                f"API key appears to be a test/dummy key: {api_key[:10]}...",
                "test_api_key"
            )


def create_secure_temp_path(base_dir: str | Path, prefix: str = "invoice_pdf_") -> Path:
    """Create a secure temporary file path within the specified directory.
    
    Args:
        base_dir: Base directory for temporary files
        prefix: Prefix for temporary filename
        
    Returns:
        Secure temporary file path
        
    Raises:
        SecurityError: If temporary path cannot be created securely
    """
    import uuid

    base_dir = Path(base_dir)

    # Validate base directory
    validated_base = validate_safe_path(base_dir, allowed_extensions=())

    # Ensure base directory exists and is writable
    if not validated_base.exists():
        try:
            validated_base.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise SecurityError(f"Cannot create temp directory {validated_base}: {e}", "temp_dir_creation")

    if not os.access(validated_base, os.W_OK):
        raise SecurityError(f"Temp directory not writable: {validated_base}", "temp_dir_not_writable")

    # Create unique filename
    unique_id = str(uuid.uuid4()).replace("-", "")[:12]
    safe_prefix = sanitize_filename(prefix)
    temp_filename = f"{safe_prefix}{unique_id}"

    return validated_base / temp_filename


# Export main functions
__all__ = [
    "check_disk_space",
    "create_secure_temp_path",
    "sanitize_filename",
    "validate_api_key",
    "validate_resource_limits",
    "validate_safe_path"
]

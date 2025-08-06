"""Configuration management for Invoice PDF processing."""
import os
from pathlib import Path

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized configuration for Invoice PDF processing."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # API Configuration
    gemini_api_key: str = Field(..., description="Gemini API key for document processing")
    use_vertex_ai: bool = Field(default=False, description="Use Vertex AI instead of standard Gemini API")
    google_cloud_project: str = Field(default="not-set", description="Google Cloud project for Vertex AI")
    google_cloud_location: str = Field(default="not-set", description="Google Cloud location for Vertex AI")

    # Processing Configuration
    processing_chunk_size: int = Field(default=500, description="Number of files to process in each chunk")
    quota_limit: int = Field(default=10, description="API concurrency limit")
    pdf_fd_semaphore_limit: int = Field(default=50, description="PDF file descriptor limit")

    # Model Configuration
    classification_model: str = Field(default="gemini-2.5-flash", description="Model for document classification")
    extraction_model: str = Field(default="gemini-2.5-pro", description="Model for data extraction")
    max_classification_pages: int = Field(default=7, description="Maximum pages for classification step")
    max_extraction_pages: int = Field(default=20, description="Maximum pages for extraction step")

    # Retry Configuration
    retry_max_attempts: int = Field(default=3, description="Maximum retry attempts per operation")
    retry_base_delay: float = Field(default=2.0, description="Base delay for exponential backoff")
    retry_max_delay: float = Field(default=10.0, description="Maximum delay between retries")
    retry_jitter_range: float = Field(default=3.0, description="Jitter range for retry delays")

    # Debug Configuration
    debug_responses: bool = Field(default=False, description="Save API responses for debugging")

    # File Paths
    input_directory: Path | None = Field(default=None, description="Input directory containing PDFs")
    output_directory: Path | None = Field(default=None, description="Output directory for results")

    @field_validator("gemini_api_key")
    @classmethod
    def api_key_must_not_be_empty(cls, v):
        """Ensure API key is provided."""
        if not v or v.strip() == "":
            raise ValueError("GEMINI_API_KEY must be provided")
        return v

    @field_validator("use_vertex_ai", mode="before")
    @classmethod
    def parse_vertex_ai_flag(cls, v):
        """Parse vertex AI flag from string."""
        if isinstance(v, str):
            return v.lower() == "true"
        return v

    @field_validator("debug_responses", mode="before")
    @classmethod
    def parse_debug_flag(cls, v):
        """Parse debug flag from string."""
        if isinstance(v, str):
            return v == "1" or v.lower() == "true"
        return v

    @classmethod
    def from_env(cls) -> "Settings":
        """Create Settings instance from environment variables."""
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            use_vertex_ai=os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true",
            google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT", "not-set"),
            google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "not-set"),
            processing_chunk_size=int(os.getenv("PROCESSING_CHUNK_SIZE", "500")),
            debug_responses=os.getenv("DEBUG_RESPONSES", "0") == "1",
        )

    @property
    def api_client_kwargs(self) -> dict:
        """Get API client configuration."""
        if self.use_vertex_ai:
            return {
                "vertexai_project": self.google_cloud_project,
                "vertexai_location": self.google_cloud_location,
            }
        return {"api_key": self.gemini_api_key}

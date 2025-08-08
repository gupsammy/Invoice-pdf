"""
Configuration constants for the advanced invoice processing system.

This module contains shared constants used across multiple modules.
Runtime knobs and secrets remain in main_2step_enhanced_adv.py to reduce coupling.
"""

# ===========================
# SHARED CONFIGURATION CONSTANTS
# ===========================

# Folder Configuration
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
LOGS_FOLDER = "logs"
JSON_RESPONSES_FOLDER = "json_responses"

# File Configuration
CLASSIFICATION_CSV_FILE = "classification_results.csv"
VENDOR_EXTRACTION_CSV_FILE = "vendor_extraction_results.csv"
EMPLOYEE_EXTRACTION_CSV_FILE = "employee_extraction_results.csv"
COMBINED_CSV_FILE = "combined_enhanced_results.csv"
FAILED_CSV_FILE = "failed_enhanced_files.csv"
LOG_FILE = "invoice_extraction_2step_enhanced.log"

# Enhanced 2-Step Configuration
MAX_CLASSIFICATION_PAGES = 7  # Increased from 5 to 7 pages
MAX_EXTRACTION_PAGES = 20     # Only process files with ≤20 pages for extraction (split if >20)
CLASSIFICATION_MODEL = "gemini-2.5-flash"
EXTRACTION_MODEL = "gemini-2.5-pro"

# Concurrency Configuration
MAX_CONCURRENT_CLASSIFY = 5   # Max concurrent classification tasks
MAX_CONCURRENT_EXTRACT = 3    # Max concurrent extraction tasks
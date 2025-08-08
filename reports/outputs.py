"""
Report and output generation utilities for the invoice processing system.

This module contains all CSV writing, Excel creation, and summary report
generation functions centralized from the main processing script.
"""

import csv
import os
import logging
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.styles import Font, PatternFill

# Import required configuration constants
from config_adv import (
    CLASSIFICATION_CSV_FILE, VENDOR_EXTRACTION_CSV_FILE, EMPLOYEE_EXTRACTION_CSV_FILE,
    FAILED_CSV_FILE
)


# TODO: Functions will be moved from main_2step_enhanced_adv.py in subsequent steps
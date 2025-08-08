"""
JSON utility functions for the invoice processing system.

This module contains JSON parsing and repair helpers that centralize
the JSON repair logic used in both classification and extraction.
"""

import json
import re
import unicodedata
from typing import Any


def try_parse_or_repair_json(json_str: str) -> dict[str, Any]:
    """
    Attempt to parse JSON string, applying repair strategies if initial parsing fails.
    
    This function centralizes the JSON repair logic that was previously duplicated
    in both classification and extraction functions.
    
    Args:
        json_str: The JSON string to parse
        
    Returns:
        Parsed JSON data as dictionary
        
    Raises:
        json.JSONDecodeError: If JSON cannot be parsed even after repair attempts
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as first_err:
        # Enhanced JSON repair with multiple strategies
        repaired_json = json_str
        
        # Strategy 1: Clean Unicode and normalize characters
        repaired_json = unicodedata.normalize('NFKD', repaired_json)
        repaired_json = ''.join(c for c in repaired_json if ord(c) < 127 or c in '{}[]":,')
        
        # Strategy 1.5: Fix missing colons after keys (most common error)
        # Pattern: "key" value -> "key": value
        repaired_json = re.sub(r'"([^"]+)"\s+(["\d\[\{])', r'"\1": \2', repaired_json)
        
        # Strategy 1.6: Fix missing commas between object elements
        # Pattern: "value"\n    "key" -> "value",\n    "key"
        repaired_json = re.sub(r'("(?:[^"\\]|\\.)*")\s*\n\s*("(?:[^"\\]|\\.)*"\s*:)', r'\1,\n    \2', repaired_json)
        
        # Strategy 2: Fix missing commas between array elements  
        # Pattern: ]\n    [ -> ],\n    [
        repaired_json = re.sub(r']\s*\n\s*\[', '],\n    [', repaired_json)
        
        # Strategy 3: Fix array endings with junk characters
        # Pattern: ] junk ] -> ]
        repaired_json = re.sub(r']\s*[^\,\s\]]*\s*]', ']', repaired_json)
        
        # Strategy 4: Fix quote endings with trailing text
        # Pattern: "text" junk" -> "text"
        repaired_json = re.sub(r'"([^"]*)"[^,"}]*"', r'"\1"', repaired_json)
        
        # Strategy 5: Remove parenthetical text after quoted strings
        # Pattern: "text" (explanation) -> "text"
        repaired_json = re.sub(r'"([^"]*)" \([^)]*\)', r'"\1"', repaired_json)
        
        # Truncate after the last }
        last_brace = repaired_json.rfind('}')
        if last_brace != -1:
            repaired_json = repaired_json[:last_brace + 1]
        
        # Strategy 6: Fix missing commas in arrays
        repaired_json = re.sub(r'"\s*\n\s*"', '",\n    "', repaired_json)
        
        return json.loads(repaired_json)  # may raise; let it propagate for caller handling
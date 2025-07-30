"""
Prompts module for invoice extraction.
Contains system instructions and prompts for the Gemini API.
"""

SYSTEM_INSTRUCTION = """You are provided with a PDF invoice. Extract the following information and return a well-structured JSON object:

1. vendor_name: Full legal name of the vendor/supplier
2. pan: PAN number of the vendor 
3. registration_numbers: Array of objects with "type" (e.g., GST, VAT, CST, TIN, GSTIN) and "value" fields
4. invoice_date: Date in YYYY-MM-DD format when possible
5. document_number: PO/document number or reference 
6. invoice_number: Invoice/bill number
7. description: Brief description of goods/services
8. basic_amount: Base amount before taxes
9. tax_amount: Total tax amount (sum of all taxes).
10. total_amount: Total invoice value including taxes

Notes:
- For missing values, use null (not empty strings)
- If multiple items are listed, combine descriptions and sum amounts
- For registration numbers, capture all visible identifiers
- Extract data from letterhead and body text as needed

Return only the JSON object without additional comments."""
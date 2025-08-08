"""
Enhanced 2-Step Prompts module for invoice extraction - Version 2.
Contains corrected classification and specialized extraction prompts with enhanced features.
"""

# ===========================
# STEP 1: CORRECTED ENHANCED CLASSIFICATION PROMPT (Flash Model - First 7 Pages)
# ===========================

ENHANCED_CLASSIFICATION_PROMPT_V2 = """You are a document classification expert. Analyze the provided PDF document (first 7 pages only) and classify it into one of three categories based on the PRIMARY document content, not supporting attachments.

## CLASSIFICATION CATEGORIES (classify based on the MAIN/PRIMARY document):

1. **vendor_invoice**: Business-to-business invoices and billing documents WITH ACTUAL TRANSACTION DATA:
   - Standard vendor invoices and bills with vendor/customer details, amounts, and dates
   - Service provider invoices with billing information
   - Professional service bills (legal, consulting, etc.) with transaction details
   - Debit notes, credit notes, receipts with billing information
   - Goods Received Note (GRN) with financial data
   - Airway waybills with charges and party details
   - Beneficiary payment advice with amounts
   - Purchase orders with detailed billing information
   - Challans with financial transactions
   - **IMPORTANT**: Must contain substantive business transaction data (vendor details, customer details, amounts, dates, line items). Documents with only "Invoice" titles but no actual invoice content should be classified as `irrelevant`

2. **employee_t&e**: Employee Travel & Expense Reports ONLY (NOT other HR forms):
   - Travel & Expense (T&E) reports and forms with travel dates and expense categories
   - TADA (Travel Allowance & Daily Allowance) forms with travel details
   - Employee expense claim forms with reimbursement requests for business travel and expenses
   - Business travel expense summaries with supporting receipts/invoices as attachments
   - Employee reimbursement documents with employee codes/names FOR TRAVEL AND BUSINESS EXPENSES
   - Employee business expense submissions for approval/payment with expense categories
   - **IMPORTANT**: Only classify as employee_t&e if the document contains actual travel expenses, business expenses, or reimbursement claims. Supporting invoices/receipts attached to employee T&E forms should NOT change the classification - the main T&E form determines the classification.

3. **irrelevant**: Documents that are not financial transactions or do not contain billable/reimbursable amounts:
   - Management reports, dashboards, analytics without billing
   - Policy documents, guidelines, operational manuals
   - Quotations without confirmed billing (proposals only)
   - Marketing materials, presentations, brochures
   - Legal contracts without billing components (terms only)
   - Payslips and salary documents
   - Bank statements (informational only)
   - Provident Fund (PF) documents
   - Identity documents (passport, PAN, etc.)
   - Form No 12BB
   - Full and Final Settlement Statement (F&F)
   - Exit Form or Exit Clearance Form
   - Salary Advance Form
   - Job Application Forms
   - Employment Offer Letters
   - Employee Joining Forms
   - Employee Data Forms
   - Performance Appraisal Forms
   - Leave Application Forms
   - Resignation Letters
   - HR Policy Documents
   - Employment Agreements (without billing)
   - Training Certificates
   - Employee ID Documents
   - **Accounting and Bookkeeping Documents**: 
     - Ledger account statements showing running balances and debit/credit entries
     - Account confirmation statements and balance confirmations
     - General ledger reports and accounting summaries
     - Customer/vendor balance reconciliation statements
     - Financial statements and trial balances
     - Any document titled with "Ledger", "Account Statement", "Balance Confirmation", or similar accounting terminology
     - Documents showing tabular debit/credit columns with running balance calculations (these are accounting records, not invoices)
   - **EPFO and Statutory Receipts**: 
     - EPFO (Employees' Provident Fund Organisation or EPF) challans and receipts
     - Documents with "COMBINED CHALLAN OF A/C NO." headers
     - Documents mentioning "EMPLOYEES' PROVIDENT FUND ORGANISATION"
     - ESI, PF contribution receipts and government statutory payments
   - **Test files and placeholders**: Documents containing only titles like "Invoice-1", "Invoice-2" without actual business transaction data (no vendor details, amounts, dates, or substantive content)
   - Pure informational content without financial transactions

## CRITICAL CLASSIFICATION LOGIC:

### Primary Document Identification:
1. **Analyze the document comprehensively** - examine all available pages (up to 7) to identify the main content and document purpose
2. **Employee T&E Priority**: If the document contains a "Travel & Expenses Statement" or employee reimbursement form with travel/business expenses, classify as `employee_t&e` regardless of supporting vendor receipts
3. **Vendor Invoice Priority**: If the document contains vendor invoices/bills between businesses, classify as `vendor_invoice`
4. **Multi-document Analysis**: Look throughout the document for invoices - they may appear on later pages, not just the first page
5. **Supporting Document Logic**: Receipts, bills, and invoices that are attached as supporting documentation to an employee expense report should NOT change the classification from `employee_t&e`
6. **Goods Receipt Notes**: GRNs with financial data should be classified as `vendor_invoice`, even if they appear early in the document

### Mixed Document Handling:
- **Employee T&E with receipts**: Main document = Employee expense form → `employee_t&e`
- **Vendor invoice with receipts**: Main document = Vendor invoice → `vendor_invoice`  
- **Multiple separate vendor invoices**: No employee forms → `vendor_invoice`
- **Multiple separate employee reports**: No vendor invoices → `employee_t&e`

## ENHANCED ANALYSIS INSTRUCTIONS:

1. **Comprehensive Document Analysis**: Examine the entire document (all available pages) to identify the main content and purpose - invoices may appear on any page, not just the first
2. **Key Identifier Recognition**: Look for specific indicators throughout the document:
   - **Employee T&E Indicators**: "Travel & Expenses Statement", employee names, employee codes, travel dates, daily allowances, expense categories, travel purpose, business expense claims, reimbursement requests
   - **Vendor Invoice Indicators**: Company letterheads, "INVOICE", "BILL", "TAX INVOICE", "GOODS RECEIPT NOTE" (GRN), vendor business names, GST/VAT numbers, invoice numbers, "To:" and "From:" business addresses, actual amounts, line items, tax details
   - **Irrelevant Indicators**: Dashboard headers, policy statements, quotations without acceptance, **HR forms that are not travel/expense related**, **test/placeholder files with only titles**, **accounting documents with "Ledger", "Account Statement", "Balance Confirmation" in titles**, **documents with debit/credit columns and running balances**, **EPFO documents with "EMPLOYEES' PROVIDENT FUND ORGANISATION" or "COMBINED CHALLAN" headers**
3. **Document Structure Priority**: 
   - **Comprehensive Page Analysis**: Examine all pages for invoices - don't stop at the first page if there are business transactions on later pages
   - **Title/Header Analysis**: What do the document titles throughout the pages say?
   - **Content Flow**: Is this an employee submitting expenses or a business billing another business?
   - **Supporting vs Primary**: Are the vendor documents supporting an employee's expense claim, or are they standalone business invoices?
   - **Invoice Detection**: Look for patterns like "TAX INVOICE", "GOODS RECEIPT NOTE", "BILL", or business-to-business transaction data throughout the document
4. **Orientation Handling**: 
   - **CRITICAL**: Pages may be rotated 90°, 180°, or 270° - use mental rotation to read content
   - **Try multiple reading angles** before making classification decisions
   - **Note orientation issues** but continue with classification
5. **Confidence Assessment**: Rate classification confidence from 0.1 to 1.0 based on clarity of primary document identification

## OUTPUT FORMAT:

Return ONLY a JSON object with this exact structure:

```json
{
  "classification": "vendor_invoice|employee_t&e|irrelevant",
  "confidence": 0.1-1.0,
  "key_indicators": ["list", "of", "specific", "indicators", "found", "in", "primary", "document"],
  "document_characteristics": {
    "has_employee_codes": boolean,
    "has_vendor_letterhead": boolean,
    "has_invoice_numbers": boolean,
    "has_travel_dates": boolean,
    "appears_financial": boolean,
    "has_amount_calculations": boolean,
    "has_tax_information": boolean,
    "contains_multiple_doc_types": boolean,
    "primary_document_type": "employee_expense_form|vendor_invoice|dashboard|policy|other"
  },
  "total_pages_analyzed": number,
  "classification_notes": "Notes about primary vs supporting document distinction and any orientation issues"
}
```

## CLASSIFICATION EXAMPLES:

### Employee T&E with Supporting Receipts:
- **Primary**: "Travel & Expenses Statement" for John Doe
- **Supporting**: Fuel receipts, hotel bills, courier waybills
- **Classification**: `employee_t&e` (confidence: 0.9)
- **Reasoning**: Main document is employee expense form; receipts are supporting evidence

### Vendor Invoice Collection:
- **Primary**: Business invoice from Company A to Company B
- **Additional**: More invoices between businesses
- **Classification**: `vendor_invoice` (confidence: 0.9)
- **Reasoning**: All documents are business-to-business billing

### Mixed Business Documents:
- **Primary**: Vendor invoice from Company A (may appear on any page, not just first)
- **Supporting**: Related delivery notes, payment receipts
- **Classification**: `vendor_invoice` (confidence: 0.8)
- **Reasoning**: Main transactional document is vendor invoice (check all pages for invoices)

### Documents with GRNs and Later Invoices:
- **Page 1**: Goods Receipt Note with financial data
- **Page 7**: Tax Invoice with detailed billing
- **Classification**: `vendor_invoice` (confidence: 0.9)
- **Reasoning**: Document contains business invoices throughout, not just on first page

## SPECIAL HANDLING INSTRUCTIONS:

1. **Rotated Content**: Apply mental rotation for 90°, 180°, 270° rotated pages
2. **Multi-Language**: Handle documents with non-English content by looking for structural patterns
3. **Poor Quality**: Make reasonable inferences from document structure and numerical patterns
4. **Primary Document Focus**: Always base classification on the main document, not supporting materials
5. **Employee T&E vs HR Forms**: Distinguish carefully between:
   - **employee_t&e**: Travel expense forms, business expense claims, reimbursement requests with expense categories
   - **irrelevant**: Other employee forms like salary advance, exit forms, joining forms, appraisals, leave applications

Return ONLY the JSON object without additional text or explanations."""

# ===========================
# STEP 2A: CORRECTED EMPLOYEE REIMBURSEMENT EXTRACTION PROMPT (Pro Model)
# ===========================

ENHANCED_EMPLOYEE_REIMBURSEMENT_EXTRACTION_PROMPT_V2 = """You are a specialized data extraction expert for employee expense reports and travel reimbursements. Your task is to extract detailed financial data from Travel & Expense (T&E) reports, TADA forms, and employee reimbursement documents with enhanced accuracy and completeness.

## PROCESSING INSTRUCTIONS:

### Document Analysis Strategy:
1. **Primary Source Identification**: Focus on the main T&E report or expense summary page (usually page 1)
2. **Employee Information Extraction**: Identify employee details, codes, departments, and identification numbers
3. **Travel Period Analysis**: Extract travel dates, destinations, and business purposes
4. **Amount Prioritization**: Extract total approved/claimed amounts from main expense reports
5. **Supporting Document Logic**: Ignore individual receipts, hotel bills, transport tickets that support the main expense report
6. **Multi-Employee Handling**: If multiple employees in same document, create separate entries for each

### Key Data Points to Extract:
- **Employee Name**: Use employee name (the person being reimbursed)
- **Employee Details**: Employee codes, names, departments, employee IDs
- **Travel Information**: Travel dates, destinations, business purposes, project codes
- **Expense Categories**: Travel, accommodation, meals, local transport, miscellaneous expenses
- **Financial Amounts**: Total claimed, approved, and reimbursable amounts (prioritize in original currency)
- **Reference Numbers**: Employee ID, expense report numbers, approval codes, manager signatures

### Enhanced T&E Processing Rules:

1. **Employee Name**: ALWAYS use employee name as employee_name field
2. **Document Number Priority**: Use employee ID or employee code as document_number
3. **Date Selection**: Use the last date of travel period or report submission date as invoice_date
4. **Amount Focus**: Extract total reimbursement amount from summary, NOT individual receipt amounts
5. **Department Information**: Include employee department when available
6. **Currency Handling**: Record amounts in original currency when specified, note currency code
7. **Approval Status**: Note if expenses are approved, pending, or rejected
8. **Amount Calculation Logic**: Indicate if amounts are totaled/calculated vs directly extracted

### Amount Extraction Strategy:
1. **Look for Summary Totals**: Check for "Total Amount", "Total Reimbursement", "Grand Total", "Total Payable"
2. **Calculate if Necessary**: If no total shown, sum individual expense categories
3. **Mark Calculation**: Use `amount_calculated` field to indicate if amounts were calculated vs extracted

## CRITICAL EXTRACTION RULES:

### Amount Extraction Priority:
1. **Total Reimbursement Amount** (highest priority)
2. **Total Approved Amount** (if different from claimed)
3. **Total Claimed Amount** (if no approval amounts shown)
4. **Sum of expense categories** (only if totals not available - mark as calculated)

### Employee Information Priority:
1. **Employee ID/Code** (use as employee_code)
2. **Full Employee Name** (use as employee_name)
3. **Department/Cost Center** (include in description)
4. **Manager/Approver Name** (note in processing_notes)

### International Travel Handling:
- **Currency Detection**: Look for foreign currency amounts (USD, EUR, GBP, etc.)
- **Original Currency Priority**: Record amounts in original currency when available
- **Conversion Notes**: Note any currency conversions in processing_notes
- **Exchange Rates**: Capture exchange rates if provided

### Orientation and Quality Handling:
- **CRITICAL**: Pages may be rotated 90°, 180°, or 270° - use mental rotation to read content
- **Try multiple reading strategies** before marking as unreadable
- **Note orientation issues** in processing_notes but continue with extraction
- **Poor Quality Handling**: Make reasonable inferences from document structure

## OUTPUT FORMAT:

Return a JSON object with this exact structure:

```json
{
  "document_status": {
    "readable": boolean,
    "contains_invoices": boolean,
    "document_type": "employee_reimbursement",
    "multiple_documents": boolean,
    "orientation_issues": boolean
  },
  "extracted_data": [
    {
      "data_source": "travel_expense_report",
      "employee_name": "Employee Name (the person being reimbursed)",
      "employee_code": "Employee ID or code",
      "department": "Employee department or cost center",
      "invoice_date": "YYYY-MM-DD (travel end date or submission date)",
      "description": "Travel purpose, destinations, and expense summary",
      "basic_amount": "Total expense amount before any deductions",
      "tax_amount": "Any tax amounts or service charges",
      "total_amount": "Total reimbursement amount (highest priority)",
      "currency_code": "INR|USD|EUR|GBP|etc (original currency)",
      "original_amount": "Amount in original currency if different",
      "amount_calculated": boolean,
      "calculation_method": "sum_of_categories|extracted_from_total|null",
      "page_numbers": [list of page numbers where data was found]
    }
  ],
  "processing_notes": "Detailed notes about extraction process, currency handling, approvals, orientation issues, calculation methods, etc."
}
```

## EMPLOYEE REIMBURSEMENT SCENARIOS:

### Single Employee T&E Report:
- Extract employee name as employee_name
- Use travel end date or report date as invoice_date
- Sum total reimbursement amounts from summary or calculate if needed
- Include travel purpose and destinations in description

### Multiple Employee Reports in Same PDF:
- Create separate entries for each employee
- Mark multiple_documents as true
- Include distinct page_numbers for each employee's data
- Maintain separate currency handling for each employee

### Employee T&E with Mixed Receipts:
- Focus ONLY on the employee expense form data
- Ignore supporting vendor receipts (fuel, hotel, courier, etc.)
- Extract total amounts from employee expense summary
- Note in processing_notes that supporting receipts were ignored

### International Travel Expenses:
- Record amounts in original foreign currency when available
- Include currency_code (USD, EUR, etc.)
- Note exchange rates and conversions in processing_notes
- Prioritize original currency amounts over converted amounts

### Complex Multi-Category Expenses:
- Focus on total approved/reimbursable amounts
- If no total available, sum expense categories and mark as calculated
- Summarize expense categories in description
- Note calculation methods in processing_notes
- Handle per-diem vs actual expense distinctions

## ENHANCED VALIDATION RULES:

1. **Mandatory Fields**: employee_name (employee), total_amount, employee_code (employee ID)
2. **Currency Consistency**: Ensure currency_code matches the amounts extracted
3. **Date Validation**: invoice_date should be travel end date or later (submission date)
4. **Amount Logic**: total_amount should be the approved reimbursement, not individual receipt sums
5. **Employee Focus**: Always extract employee as the "vendor" being reimbursed
6. **Calculation Tracking**: Always indicate if amounts were calculated vs directly extracted

## SPECIAL HANDLING:

### Mixed Employee and Vendor Documents:
- If document contains both employee expenses AND separate vendor invoices, extract ONLY employee expense data in this step
- Note presence of separate vendor invoices in processing_notes for separate processing

### Supporting Receipt Handling:
- IGNORE individual hotel, transport, meal receipts that support the main expense report
- Extract ONLY from expense summary or consolidated expense forms
- Note in processing_notes if supporting receipts were ignored

### Currency and International Considerations:
- **Multi-Currency Support**: Handle documents with multiple currencies
- **Original Currency Priority**: Always record in original currency when available
- **Conversion Tracking**: Note any currency conversions performed
- **Per-Diem vs Actual**: Distinguish between per-diem allowances and actual expense reimbursements

### Amount Calculation and Confidence:
- **High Confidence**: When extracting from clear "Total Amount" fields
- **Medium Confidence**: When extracting from summary sections
- **Calculated Amounts**: When summing individual categories, mark as calculated
- **Processing Notes**: Always explain calculation methodology

Extract comprehensive employee expense data while maintaining focus on reimbursement transactions and avoiding duplication from supporting documentation.

Return ONLY the JSON object without additional comments."""

# ===========================
# STEP 2B: CORRECTED VENDOR INVOICE EXTRACTION PROMPT (Pro Model)
# ===========================

ENHANCED_VENDOR_INVOICE_EXTRACTION_PROMPT_V2 = """You are a specialized data extraction expert for vendor invoices and business billing documents. Your task is to extract detailed financial data from vendor invoices, service provider bills, and B2B billing documents with enhanced accuracy, duplicate prevention, main document detection, and comprehensive party identification.

## PROCESSING INSTRUCTIONS:

### Document Analysis Strategy:
1. **Main Invoice Identification**: Identify the primary invoice/bill vs supporting documents or sub-contractor invoices
2. **Party Identification**: Extract complete details for invoice issuer, consignor, and consignee
3. **Invoice Classification**: Determine the type of billing document (invoice, debit note, credit note, etc.)
4. **Financial Data Analysis**: Extract itemized amounts, taxes, and totals with currency detection
5. **Duplicate Prevention**: Identify and avoid extracting duplicate invoices within the same document
6. **Sub-Contractor Logic**: Distinguish main invoices from sub-contractor/supporting invoices

### Enhanced Vendor Processing Rules:

1. **Main Document Priority**: Always extract from the primary invoice, not supporting receipts or sub-contractor bills
2. **Party Identification**: Identify issuer (who sends), consignor (who ships), and consignee (who receives)
3. **Vendor Name Handling**: Convert non-English vendor names to English when recognizable, otherwise keep original
4. **Invoice Type Classification**: Categorize as invoice, debit note, credit note, receipt, GRN, waybill, challan, etc.
5. **Main vs Sub-Contractor**: For files like GOLIVE with main contractor + sub-contractors, extract ONLY the main invoice
6. **Currency Detection**: Identify and record original currency for international transactions
7. **Amount Calculation**: Indicate if amounts were calculated vs directly extracted

### Key Data Points to Extract:
- **Party Information**: 
  - **Issuer**: Who issued/sent the invoice (usually the vendor/service provider)
  - **Consignor**: Who is shipping/providing goods (may be same as issuer)
  - **Consignee**: Who is receiving goods/services (usually the customer)
- **Vendor Information**: Full legal names (English preferred), business addresses, contact details
- **Registration Numbers**: GST, VAT, CST, TIN, PAN, and other tax identifiers
- **Invoice Classification**: Type of document (invoice, debit note, credit note, etc.)
- **Financial Breakdown**: Base amounts, tax calculations, total values in original currency
- **International Handling**: Original currency amounts, exchange rates if provided
- **Service/Product Details**: Descriptions of goods or services with proper English translation when needed

### Main Invoice Detection Logic:
1. **Primary Contractor Rule**: In files with main contractor + sub-contractors, extract ONLY the main contractor invoice
2. **Amount Hierarchy**: Main invoices typically have the largest amounts and summarize sub-contractor costs
3. **Document Position**: Main invoices typically appear first or have the most comprehensive details
4. **Reference Logic**: Main invoices often reference or summarize sub-contractor invoices
5. **Contract Relationship**: Main invoice is typically between the primary service provider and end customer

### Multi-Document and Duplicate Handling:
- **Multiple Separate Invoices**: Extract each distinct vendor invoice separately (different parties, different transactions)
- **Main + Sub-Contractors**: Extract ONLY the main contractor invoice, ignore sub-contractor invoices
- **Duplicate Detection**: Compare invoice numbers, dates, and amounts to identify duplicates
- **Triplicate Copies**: Extract only once per unique invoice (ignore duplicate/triplicate copies)
- **Mixed Content**: Distinguish between main invoices and supporting documentation
- **Summary vs Detail**: Prioritize summary sheets when they contain consolidated information

## CRITICAL EXTRACTION RULES:

### Main Invoice Detection Strategy:
1. **Contract Relationship**: Look for the main service provider to end customer relationship
2. **Amount Analysis**: Main invoices typically consolidate sub-contractor costs
3. **Reference Checking**: Main invoices may reference purchase orders or client contracts
4. **Document Quality**: Main invoices typically have more formal letterheads and complete details
5. **Page Position**: Main invoices often appear first in document sequence

### Party Identification Rules:
1. **Issuer**: The entity that created and sent the invoice (look for "From:", letterhead, invoice issuer)
2. **Consignor**: The entity providing/shipping goods (may be same as issuer, look for "Shipped by:")
3. **Consignee**: The entity receiving goods/services (look for "To:", "Bill To:", customer details)
4. **Address Analysis**: Use address information to distinguish between parties
5. **Role Clarification**: Sometimes one entity plays multiple roles (issuer = consignor)

### International Transaction Handling:
1. **Original Currency Priority**: Always record amounts in original currency (USD, EUR, GBP, etc.)
2. **Currency Code Requirement**: Include currency_code field for all amounts
3. **Conversion Notes**: Note any currency conversions in processing_notes
4. **Exchange Rate Capture**: Record exchange rates when provided
5. **Multi-Currency Documents**: Handle documents with multiple currencies separately

### Vendor Name Enhancement:
1. **English Conversion**: Convert recognizable non-English vendor names to English equivalents
2. **Name Standardization**: Use consistent vendor name formats
3. **Original Preservation**: Keep original name in processing_notes if converted
4. **Corporate Suffixes**: Include Pvt Ltd, LLC, Inc, GmbH, etc. in vendor names

### Invoice Type Classification:
- **Standard Invoice**: Regular vendor bill for goods/services
- **Debit Note**: Additional charges or corrections
- **Credit Note**: Refunds or adjustments  
- **Receipt**: Payment confirmation (usually supporting document)
- **GRN**: Goods Received Note with billing
- **Waybill**: Shipping document with charges
- **Challan**: Delivery document with billing
- **Service Order**: Service billing document
- **Professional Bill**: Legal, consulting, or professional service billing

### Amount Calculation and Confidence:
1. **Direct Extraction**: When amounts are clearly shown in total fields
2. **Calculated Amounts**: When summing line items or calculating from components
3. **Confidence Indicators**: Mark calculation method and confidence level
4. **Validation**: Cross-check calculated amounts against any provided totals

### Orientation and Quality Handling:
- **CRITICAL**: Pages may be rotated 90°, 180°, or 270° - use mental rotation to read content
- **Multiple Attempts**: Try different reading strategies before marking as unreadable
- **Quality Inference**: Make reasonable inferences from document structure and patterns
- **Processing Notes**: Document any orientation issues or quality concerns

## OUTPUT FORMAT:

Return a JSON object with this exact structure:

```json
{
  "document_status": {
    "readable": boolean,
    "contains_invoices": boolean,
    "document_type": "vendor_invoice",
    "multiple_documents": boolean,
    "orientation_issues": boolean
  },
  "extracted_data": [
    {
      "data_source": "vendor_invoice",
      "issuer": "Entity that issued/created the invoice",
      "consignor": "Entity providing/shipping goods (may be same as issuer)",
      "consignee": "Entity receiving goods/services",
      "vendor_name": "Primary vendor/service provider name (English when possible)",
      "original_vendor_name": "Original name if converted to English",
      "invoice_type": "invoice|debit_note|credit_note|receipt|grn|waybill|challan|service_order|professional_bill",
      "pan": "Vendor PAN number",
      "registration_numbers": [
        {"type": "GST|VAT|CST|TIN|GSTIN", "value": "registration number"}
      ],
      "invoice_date": "YYYY-MM-DD (invoice issue date, not due date)",
      "document_number": "PO number or document reference", 
      "invoice_number": "Invoice/bill number (main identifier)",
      "description": "Goods/services description in English when possible",
      "basic_amount": "Base amount before taxes in original currency",
      "tax_amount": "Total tax amount in original currency",
      "total_amount": "Total invoice value in original currency",
      "currency_code": "INR|USD|EUR|GBP|etc (original currency)",
      "original_amount": "Amount in original currency if document shows conversion",
      "exchange_rate": "Exchange rate if provided",
      "amount_calculated": boolean,
      "calculation_method": "sum_of_line_items|extracted_from_total|calculated_from_components|null",
      "is_main_invoice": boolean,
      "page_numbers": [list of page numbers where main invoice data was found]
    }
  ],
  "processing_notes": "Detailed extraction notes including main vs sub-contractor logic, duplicate handling, currency conversions, vendor name translations, calculation methods, party identification logic, orientation issues, etc."
}
```

## VENDOR INVOICE SCENARIOS:

### Standard Vendor Invoice:
- Extract from main invoice page with vendor letterhead
- Identify issuer (vendor), consignor (shipper), consignee (customer)
- Use invoice issue date as invoice_date (not payment due date)
- Include all visible tax registration numbers
- Sum line items for accurate totals, prefer summary amounts

### Main Contractor with Sub-Contractors (GOLIVE-type):
- Identify the main contractor invoice (usually page 1, largest amount)
- Extract ONLY the main contractor invoice details
- Mark is_main_invoice as true
- Note sub-contractor invoices in processing_notes but DO NOT extract them
- Focus on the primary service provider to end customer relationship

### International Vendor Invoice:
- Record amounts in original foreign currency (USD, EUR, etc.)
- Include currency_code field
- Note exchange rates if provided
- Identify issuer, consignor, consignee across international boundaries
- Convert vendor name to English if clearly identifiable
- Preserve original vendor name in processing_notes

### Multi-Invoice PDF (Separate Transactions):
- Identify distinct separate transactions between different parties
- Extract each as separate invoice entries
- Compare invoice numbers, dates, amounts to avoid duplicates
- Mark multiple_documents as true
- Create separate entries for genuinely different transactions

### Invoice with Supporting Documents:
- Extract from main invoice pages only
- Ignore supporting receipts, delivery notes, or payment receipts
- Avoid double-counting amounts already included in main invoice
- Note supporting documents ignored in processing_notes
- Focus on the primary billing relationship

### Service Provider Professional Bill:
- Focus on service description and billing period
- Extract professional service details with English descriptions
- Include hourly rates, fixed fees, or retainer amounts
- Capture service delivery dates and project references
- Identify client-service provider relationship clearly

### Waybill/Shipping Invoice:
- Identify shipper (consignor), carrier (issuer), receiver (consignee)
- Extract shipping charges, handling fees, and total costs
- Include shipment details and reference numbers
- Handle multi-party logistics relationships
- Note any freight forwarding arrangements

## DUPLICATE PREVENTION LOGIC:

### Invoice Comparison Criteria:
1. **Invoice Number Match**: Same invoice number = potential duplicate
2. **Amount Verification**: Same total amount strengthens duplicate probability
3. **Date Consistency**: Same invoice date confirms duplicate
4. **Vendor Matching**: Same vendor with same invoice details = duplicate
5. **Page Analysis**: Multiple pages showing same invoice data

### Main vs Sub-Contractor Detection:
1. **Amount Analysis**: Main invoices typically have larger amounts
2. **Reference Checking**: Main invoices may reference sub-contractor work
3. **Relationship Mapping**: Main contractor invoices end customer, sub-contractors invoice main contractor
4. **Contract Context**: Look for primary service agreements vs sub-contractor arrangements
5. **Document Quality**: Main invoices typically have more formal presentation

### Duplicate Handling Actions:
- **Single Extraction**: Extract only once per unique invoice
- **Page Preference**: Prefer earlier pages or pages with complete data
- **Quality Priority**: Choose page with clearest/most complete information
- **Documentation**: Note duplicate pages found in processing_notes

## ENHANCED VALIDATION RULES:

1. **Mandatory Fields**: issuer, consignee, vendor_name, invoice_number, total_amount, currency_code
2. **Party Validation**: Ensure issuer, consignor, consignee are properly identified
3. **Currency Consistency**: Ensure currency_code matches the extracted amounts
4. **Date Logic**: invoice_date should be issue date, not due date or payment date
5. **Amount Validation**: total_amount should equal basic_amount + tax_amount when both available
6. **Registration Numbers**: Include all visible tax/business registration identifiers
7. **Main Invoice Logic**: Mark is_main_invoice appropriately based on document analysis

## SPECIAL HANDLING INSTRUCTIONS:

### Multi-Language Documents:
- **Vendor Name Translation**: Convert to English when clearly identifiable
- **Description Translation**: Provide English descriptions when possible
- **Original Preservation**: Keep original text in processing_notes when translated
- **Currency Recognition**: Identify currency symbols regardless of language

### Poor Quality or Rotated Documents:
- **Orientation Handling**: Apply mental rotation for 90°, 180°, 270° rotated content
- **Quality Inference**: Make reasonable inferences from document structure and patterns
- **Multiple Attempts**: Try different reading strategies before marking as unreadable
- **Partial Extraction**: Extract available data even if some fields are unclear

### International Transaction Specifics:
- **Currency Priority**: Always use original transaction currency
- **Conversion Handling**: Note any currency conversions shown in document
- **Multi-Currency**: Handle documents showing multiple currencies
- **Rate Documentation**: Capture exchange rates when provided

### Complex Multi-Party Scenarios:
- **Freight Forwarding**: Handle shipper, forwarder, consignee relationships
- **Drop Shipping**: Identify manufacturer, retailer, end customer relationships
- **Service Outsourcing**: Distinguish main service provider from sub-contractors
- **International Trade**: Handle importer, exporter, shipping agent relationships

Extract comprehensive vendor invoice data while maintaining accuracy, preventing duplicates, properly identifying all parties, and supporting international transactions with proper currency handling.

Return ONLY the JSON object without additional comments."""


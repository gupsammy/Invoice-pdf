# Invoice Data Extractor

This tool extracts structured data from PDF invoices using Google's Gemini AI and exports the results to CSV format.

## Features

- Multi-threaded processing for improved performance
- Error handling and logging
- Progress bar for tracking processing status
- Comprehensive data extraction from various invoice formats

## Setup Instructions

1. Clone or download this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
4. Create an `input` folder in the same directory and place your PDF invoices there

## Usage

Simply run the script:

```
python invoice_extractor.py
```

The script will:

1. Process all PDFs in the `input` folder
2. Extract invoice data using Gemini AI
3. Output results to `output.csv`
4. Create a log file with processing details

## Output Format

The CSV output contains the following fields:

- Name of the vendor
- PAN
- VAT/CST/Other registration number (formatted as type:value pairs)
- Invoice Date
- PO No/Document No
- Invoice no
- Description of goods/service
- Basic Amount
- Tax Amount
- Invoice Value
- File Name

## Troubleshooting

- Check the log file for detailed error messages
- Ensure your PDF files are readable and not password-protected
- Verify that your Gemini API key is valid and properly set in the `.env` file
- For handling specific invoice formats that aren't extracting well, you may need to modify the system prompt

## Requirements

- Python 3.7+
- Google Generative AI API access
- Dependencies listed in requirements.txt

import os
import json
import csv
import time
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Import Google Generative AI
from google import genai
from google.genai import types

# Constants for retry logic
_MAX_EXTRACTION_RETRIES = 3  # Max retries for a single file extraction attempt
_RETRY_DELAY_BASE_SECONDS = 10  # Base delay for retries, used with exponential backoff

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("invoice_extraction.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in a .env file.")

def extract_invoice_data(pdf_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract invoice data from a PDF file using Gemini API, with retries.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing the extracted invoice data or None if extraction fails after retries.
    """
    try:
        client = genai.Client(api_key=API_KEY)
        pdf_path_obj = Path(pdf_path)
        pdf_bytes = pdf_path_obj.read_bytes()
        
        system_instruction = """You are provided with a PDF invoice. Extract the following information and return a well-structured JSON object:

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

        contents = [
            types.Part.from_bytes(
                data=pdf_bytes,
                mime_type='application/pdf',
            ),
            "Extract details from this pdf"
        ]
        config = types.GenerateContentConfig(system_instruction=system_instruction)
        
        for attempt in range(_MAX_EXTRACTION_RETRIES):
            try:
                logging.info(f"Extraction attempt {attempt + 1}/{_MAX_EXTRACTION_RETRIES} for {pdf_path}...")
                response = client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=config
                )
                
                json_responses_dir = "json_responses"
                os.makedirs(json_responses_dir, exist_ok=True)
                pdf_filename_stem = pdf_path_obj.stem
                json_log_filename = f"{pdf_filename_stem}_response_attempt_{attempt+1}.txt"
                json_log_path = os.path.join(json_responses_dir, json_log_filename)
                with open(json_log_path, 'w', encoding='utf-8') as f_json_log:
                    f_json_log.write(response.text)
                    
                response_text = response.text
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    logging.warning(f"No JSON object found in response for {pdf_path} (attempt {attempt + 1}). Raw response snippet: {response_text[:200]}")
                    # Check if this non-JSON response is due to a known server error
                    if any(phrase in response_text for phrase in ["502 Bad Gateway", "Service Unavailable", "server error"]):
                        if attempt < _MAX_EXTRACTION_RETRIES - 1:
                            delay = _RETRY_DELAY_BASE_SECONDS * (2**attempt)
                            logging.info(f"Retrying {pdf_path} in {delay}s due to server error in response content...")
                            time.sleep(delay)
                            continue # To the next attempt in the loop
                        else:
                            logging.error(f"All {_MAX_EXTRACTION_RETRIES} retries failed for {pdf_path} due to persistent server error in response content.")
                            return None
                    else: # Non-JSON response, not identified as a retryable server error
                        logging.error(f"Unrecognized non-JSON response for {pdf_path}, not retrying this file attempt.")
                        return None # Stop attempts for this file if response is fundamentally not JSON and not a server error

                json_str = response_text[json_start:json_end]
                invoice_data = json.loads(json_str)
                logging.info(f"Successfully extracted data for {pdf_path} on attempt {attempt + 1}.")
                return invoice_data # Success

            except json.JSONDecodeError as jde:
                logging.error(f"JSONDecodeError for {pdf_path} (attempt {attempt + 1}): {str(jde)}. Response snippet: {response_text[:200]}")
                # Generally, JSONDecodeError means content is malformed, not typically a transient server issue.
                # So, we don't retry for this specific error within this file's attempts.
                return None
            except (types.generation_types.BlockedPromptException, types.generation_types.StopCandidateException) as specific_api_error:
                logging.error(f"API policy/content error for {pdf_path} (attempt {attempt + 1}): {str(specific_api_error)}. Not retrying this file.")
                return None # Non-retryable API policy error
            except Exception as e:
                error_str = str(e)
                logging.warning(f"Extraction attempt {attempt + 1}/{_MAX_EXTRACTION_RETRIES} for {pdf_path} failed: {error_str[:200]}")
                
                is_retryable_error = any(phrase in error_str.lower() for phrase in 
                                         ["502 bad gateway", "service unavailable", "server error", 
                                          "internal error", "transient error", "rate limit exceeded", "timeout"])
                
                if is_retryable_error:
                    if attempt < _MAX_EXTRACTION_RETRIES - 1:
                        delay = _RETRY_DELAY_BASE_SECONDS * (2**attempt) # Exponential backoff
                        logging.info(f"Retrying {pdf_path} in {delay} seconds due to: {error_str[:100]}")
                        time.sleep(delay)
                    else:
                        logging.error(f"All {_MAX_EXTRACTION_RETRIES} retries failed for {pdf_path}. Last error: {error_str[:200]}")
                        return None # All retries for this file exhausted
                else:
                    logging.error(f"Non-retryable error during extraction for {pdf_path}: {error_str[:200]}. Not retrying this file.")
                    return None # Non-retryable error, stop attempts for this file.
        
        # If loop completes, it means all retries were exhausted for retryable errors
        logging.error(f"Failed to extract data from {pdf_path} after {_MAX_EXTRACTION_RETRIES} attempts due to persistent retryable issues.")
        return None

    except Exception as e: # Catch errors in initial setup (file reading, client init)
        logging.error(f"Outer error setting up extraction for {pdf_path}: {str(e)}")
        return None

def process_pdf_file(pdf_path: str) -> Optional[Dict[str, Any]]:
    """
    Process a single PDF file and return the extracted data with the filename.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with extracted data and filename or None if extraction fails
    """
    try:
        filename = os.path.basename(pdf_path)
        # extract_invoice_data now handles its own retries
        data = extract_invoice_data(pdf_path) 
        
        if data:
            data["file_name"] = filename
            return data
        else:
            # extract_invoice_data returning None means it failed after its retries
            logging.warning(f"Processing failed for {pdf_path} after all internal retries.")
            return None
    except Exception as e:
        logging.error(f"Error in process_pdf_file for {pdf_path}: {str(e)}")
        return None

def format_registration_numbers(reg_numbers: List[Dict[str, str]]) -> str:
    """
    Format registration numbers as comma-separated type:value pairs.
    
    Args:
        reg_numbers: List of dictionaries with type and value keys
        
    Returns:
        Formatted string of registration numbers
    """
    if not reg_numbers:
        return ""
    
    return ", ".join([f"{reg.get('type', 'N/A')}:{reg.get('value', 'N/A')}" for reg in reg_numbers])

def save_to_csv(data_list: List[Dict[str, Any]], output_file: str = "output.csv") -> None:
    """
    Save the extracted data to a CSV file.
    
    Args:
        data_list: List of dictionaries containing the extracted data
        output_file: Path to the output CSV file
    """
    if not data_list:
        logging.warning("No data to save to CSV as data_list is empty.")
        # Create an empty CSV with headers if no data
        # return # Or create empty CSV with headers
    
    fieldnames = [
        "Name of the vendor", "PAN", "VAT / CST / Other registration number",
        "Invoice Date", "PO No/ Document No", "Invoice no", 
        "Description of goods / service", "Basic Amount", "Tax Amount", 
        "Invoice Value", "File Name"
    ]
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            if not data_list:
                logging.info(f"No data to write to {output_file}, but header is written.")
                return

            for item in data_list:
                reg_numbers = format_registration_numbers(item.get("registration_numbers", []))
                row = {
                    "Name of the vendor": item.get("vendor_name"),
                    "PAN": item.get("pan"),
                    "VAT / CST / Other registration number": reg_numbers,
                    "Invoice Date": item.get("invoice_date"),
                    "PO No/ Document No": item.get("document_number"),
                    "Invoice no": item.get("invoice_number"),
                    "Description of goods / service": item.get("description"),
                    "Basic Amount": item.get("basic_amount"),
                    "Tax Amount": item.get("tax_amount"),
                    "Invoice Value": item.get("total_amount"),
                    "File Name": item.get("file_name", "") # Ensure file_name is present
                }
                writer.writerow(row)
                
        logging.info(f"Data successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving data to CSV '{output_file}': {str(e)}")

def save_failed_files_to_csv(failed_file_paths: List[str], output_csv_file: str = "failed_extraction_files.csv"):
    """
    Save the list of permanently failed file names to a CSV.
    """
    if not failed_file_paths:
        logging.info("No permanently failed files to save to CSV.")
        return

    failed_filenames = [os.path.basename(path) for path in failed_file_paths]
    
    try:
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Failed File Name"]) # Header
            for filename in failed_filenames:
                writer.writerow([filename])
        logging.info(f"List of {len(failed_filenames)} permanently failed files saved to {output_csv_file}")
    except Exception as e:
        logging.error(f"Error saving failed files list to CSV '{output_csv_file}': {str(e)}")

def main():
    """
    Main function to process all PDF files in the input folder with multi-level retries.
    """
    input_folder = "input"
    output_csv_file = "output.csv"
    json_responses_folder = "json_responses" 
    
    MAX_GLOBAL_RETRY_PASSES = 2  # e.g., 2 retry passes after the initial pass (total 3 attempts for a batch)
    GLOBAL_RETRY_BATCH_DELAY_SECONDS = 30 
    INDIVIDUAL_PROCESSING_DELAY_SECONDS = 1 
    
    # Clear and then ensure creation of json_responses directory
    if os.path.exists(json_responses_folder):
        try:
            shutil.rmtree(json_responses_folder)
            logging.info(f"Successfully cleared existing json_responses folder: {json_responses_folder}")
        except Exception as e:
            logging.error(f"Error clearing json_responses folder '{json_responses_folder}': {str(e)}. Please check permissions or manually delete it.")
            # Optionally, decide if script should exit if clearing fails, for now it will continue and try to create/use it.

    os.makedirs(json_responses_folder, exist_ok=True)
    os.makedirs(input_folder, exist_ok=True)
    
    if not os.path.exists(input_folder): # Should be redundant due to makedirs
        logging.error(f"Input folder '{input_folder}' does not exist and could not be created.")
        return
    
    initial_pdf_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                         if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(input_folder, f))]
    
    if not initial_pdf_files:
        logging.warning(f"No PDF files found in '{input_folder}'")
        return
    
    logging.info(f"Found {len(initial_pdf_files)} PDF files for initial processing.")
    
    master_processed_data_list = [] 
    files_requiring_processing = list(initial_pdf_files) 
    
    # Loop for initial pass (pass 0) and then global retry passes
    # Total attempts for a batch = 1 (initial) + MAX_GLOBAL_RETRY_PASSES
    for pass_num in range(MAX_GLOBAL_RETRY_PASSES + 1):
        if not files_requiring_processing:
            logging.info("All pending files have been processed successfully or no files were left for this pass.")
            break 

        current_batch_size = len(files_requiring_processing)
        pass_description = f"Initial Pass ({current_batch_size} files)"
        if pass_num > 0:
            pass_description = f"Global Retry Pass {pass_num}/{MAX_GLOBAL_RETRY_PASSES} ({current_batch_size} files)"
            logging.info(f"--- Starting {pass_description} ---")
            logging.info(f"Waiting {GLOBAL_RETRY_BATCH_DELAY_SECONDS} seconds before this retry pass...")
            time.sleep(GLOBAL_RETRY_BATCH_DELAY_SECONDS)
        else: 
            logging.info(f"--- Starting {pass_description} ---")

        successfully_processed_this_pass_data = []
        failed_in_this_pass_paths = []
        
        max_workers = 4  # Consider making this configurable or adjusting based on API limits
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pdf_file, pdf_path): pdf_path 
                       for pdf_path in files_requiring_processing}
            
            with tqdm(total=len(futures), desc=pass_description) as pbar:
                for future in as_completed(futures):
                    pdf_file_path_for_future = futures[future]
                    try:
                        result_data = future.result() # process_pdf_file returns data or None
                        if result_data:
                            successfully_processed_this_pass_data.append(result_data)
                        else:
                            failed_in_this_pass_paths.append(pdf_file_path_for_future)
                    except Exception as e_outer_loop:
                        logging.error(f"Unhandled error in main processing loop for {pdf_file_path_for_future} during {pass_description}: {str(e_outer_loop)}")
                        failed_in_this_pass_paths.append(pdf_file_path_for_future)
                    finally:
                        pbar.update(1)
                        time.sleep(INDIVIDUAL_PROCESSING_DELAY_SECONDS) 
        
        master_processed_data_list.extend(successfully_processed_this_pass_data)
        files_requiring_processing = failed_in_this_pass_paths 

        if not files_requiring_processing:
            logging.info(f"All files from this batch were processed successfully in {pass_description}.")
        else:
            logging.info(f"{len(files_requiring_processing)} files failed in {pass_description} and will be handled in the next pass if attempts remain.")

    permanently_failed_file_paths = files_requiring_processing

    logging.info(f"Total successfully processed files: {len(master_processed_data_list)} out of {len(initial_pdf_files)} initial files.")
    save_to_csv(master_processed_data_list, output_csv_file)

    if permanently_failed_file_paths:
        logging.warning(f"Number of permanently failed files after all passes: {len(permanently_failed_file_paths)}")
        save_failed_files_to_csv(permanently_failed_file_paths, "failed_extraction_files.csv")
    else:
        logging.info("No files permanently failed extraction after all passes.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
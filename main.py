import os
import json
import csv
import time
import logging
import shutil
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Import Google Generative AI
from google import genai
from google.genai import types

# Constants for retry logic and async configuration
_MAX_EXTRACTION_RETRIES = 3  # Max retries for a single file extraction attempt
_RETRY_DELAY_BASE_SECONDS = 10  # Base delay for retries, used with exponential backoff
MAX_CONCURRENT_API_CALLS = 5  # Max concurrent API calls for async parallelization

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

async def extract_invoice_data_async(
    pdf_path: str,
    client: "genai.Client", 
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """
    Asynchronously extract invoice data from a PDF file using Gemini API, with retries.
    
    Args:
        pdf_path: Path to the PDF file
        client: Shared genai client
        semaphore: Semaphore to control concurrent API calls
        
    Returns:
        Dictionary containing the extracted invoice data or None if extraction fails after retries.
    """
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
    
    async with semaphore:
        for attempt in range(_MAX_EXTRACTION_RETRIES):
            try:
                logging.info(f"[{pdf_path_obj.name}] Attempt {attempt + 1}/{_MAX_EXTRACTION_RETRIES} – requesting Gemini ...")
                response = await client.aio.models.generate_content(
                    model="gemini-2.5-pro",
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
                    
                resp_txt = response.text
                json_start = resp_txt.find('{')
                json_end = resp_txt.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    # Non-JSON or malformed response – decide if transient
                    if any(err in resp_txt for err in ["502", "Service Unavailable", "server error"]):
                        raise RuntimeError("Transient upstream error – will retry")
                    logging.error(f"[{pdf_path_obj.name}] Un-parsable response – giving up without retry: {resp_txt[:120]}")
                    return None

                invoice_data = json.loads(resp_txt[json_start:json_end])
                logging.info(f"[{pdf_path_obj.name}] Data extraction successful.")
                return invoice_data

            except json.JSONDecodeError as jde:
                logging.error(f"[{pdf_path_obj.name}] JSON decode failed: {str(jde)} – response excerpt: {resp_txt[:120]}")
                return None  # Usually unrecoverable
            except Exception as exc:
                if attempt < _MAX_EXTRACTION_RETRIES - 1:
                    delay = _RETRY_DELAY_BASE_SECONDS * (2 ** attempt)
                    logging.warning(f"[{pdf_path_obj.name}] Error – {str(exc)[:100]}. Retrying in {delay}s ...")
                    await asyncio.sleep(delay)
                    continue
                logging.error(f"[{pdf_path_obj.name}] Exhausted retries after persistent errors: {str(exc)[:150]}")
                return None

# This function is now replaced by direct async calls in main()

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

async def main():
    """
    Async main function to process all PDF files in the input folder with async parallelization.
    """
    input_folder = "input"
    output_csv_file = "output.csv"
    json_responses_folder = "json_responses"

    MAX_GLOBAL_RETRY_PASSES = 2
    GLOBAL_RETRY_BATCH_DELAY_SECONDS = 30

    # Housekeeping – clear old response artefacts
    if os.path.exists(json_responses_folder):
        try:
            shutil.rmtree(json_responses_folder)
            logging.info(f"Cleared {json_responses_folder} directory.")
        except Exception as e:
            logging.error(f"Unable to clear {json_responses_folder}: {str(e)}")
    os.makedirs(json_responses_folder, exist_ok=True)
    os.makedirs(input_folder, exist_ok=True)

    pdf_files_initial = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(input_folder, f))
    ]
    if not pdf_files_initial:
        logging.warning(f"No PDF files present in '{input_folder}'. Exiting.")
        return

    logging.info(f"🔍 Discovered {len(pdf_files_initial)} PDFs to process.")
    logging.info(f"⚙️  Configuration: Max concurrent API calls = {MAX_CONCURRENT_API_CALLS}")
    logging.info(f"🤖 Using model: gemini-2.5-pro via Vertex AI")

    # Shared genai client (re-used across requests)
    # When using Vertex AI, don't pass API key - use application default credentials
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
        client = genai.Client()  # Uses application default credentials
        logging.info("🔐 Using Vertex AI with Application Default Credentials")
    else:
        client = genai.Client(api_key=API_KEY)
        logging.info("🔐 Using regular Gemini API with API key")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

    processed_data: List[Dict[str, Any]] = []
    pending_files: List[str] = list(pdf_files_initial)

    for pass_idx in range(MAX_GLOBAL_RETRY_PASSES + 1):
        if not pending_files:
            break

        pass_desc = "Initial Pass" if pass_idx == 0 else f"Retry Pass {pass_idx}/{MAX_GLOBAL_RETRY_PASSES}"
        if pass_idx > 0:
            logging.info(f"Waiting {GLOBAL_RETRY_BATCH_DELAY_SECONDS}s before {pass_desc} ...")
            await asyncio.sleep(GLOBAL_RETRY_BATCH_DELAY_SECONDS)
        logging.info(f"--- Starting {pass_desc} ({len(pending_files)} files) ---")

        # Kick off tasks concurrently with controlled concurrency inside extract function.
        task_to_path = {
            asyncio.create_task(extract_invoice_data_async(f, client, semaphore)): f for f in pending_files
        }

        current_pass_success: List[Dict[str, Any]] = []
        failed_this_pass: List[str] = []

        # Use tqdm for progress tracking with detailed descriptions
        with tqdm(
            total=len(task_to_path),
            desc=f"{pass_desc}",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            # Create a list to track tasks and their corresponding paths
            tasks = list(task_to_path.keys())
            paths = list(task_to_path.values())
            
            # Use gather to await all tasks and track completion (maintains parallelism)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for task, pdf_path, result in zip(tasks, paths, results):
                filename = os.path.basename(pdf_path)
                
                try:
                    if isinstance(result, Exception):
                        # Task raised an exception
                        failed_this_pass.append(pdf_path)
                        pbar.set_postfix_str(f"💥 {filename}")
                        logging.error(f"[ERROR] {filename} - task exception: {str(result)[:100]}")
                    elif result:
                        # Task completed successfully with data
                        result["file_name"] = filename
                        current_pass_success.append(result)
                        pbar.set_postfix_str(f"✅ {filename}")
                        logging.info(f"[SUCCESS] {filename} - extraction completed")
                    else:
                        # Task completed but returned None (failed extraction)
                        failed_this_pass.append(pdf_path)
                        pbar.set_postfix_str(f"❌ {filename}")
                        logging.warning(f"[FAILED] {filename} - extraction failed")
                except Exception as e:
                    failed_this_pass.append(pdf_path)
                    pbar.set_postfix_str(f"💥 {filename}")
                    logging.error(f"[ERROR] {filename} - unexpected error: {str(e)[:100]}")
                finally:
                    pbar.update(1)
        
        # Summary logging for this pass
        success_count = len(current_pass_success)
        failed_count = len(failed_this_pass)
        logging.info(f"📊 {pass_desc} Summary: {success_count} successful, {failed_count} failed")
        
        if success_count > 0:
            logging.info(f"✅ Successfully processed files: {[os.path.basename(item['file_name']) for item in current_pass_success]}")
        if failed_count > 0:
            logging.warning(f"❌ Failed files: {[os.path.basename(f) for f in failed_this_pass]}")

        processed_data.extend(current_pass_success)
        pending_files = failed_this_pass

    # Final summary and results
    total_files = len(pdf_files_initial)
    successful_files = len(processed_data)
    failed_files = len(pending_files)
    
    logging.info(f"🎯 FINAL RESULTS:")
    logging.info(f"   📁 Total files: {total_files}")
    logging.info(f"   ✅ Successful: {successful_files} ({successful_files/total_files*100:.1f}%)")
    logging.info(f"   ❌ Failed: {failed_files} ({failed_files/total_files*100:.1f}%)")

    # Persist results
    save_to_csv(processed_data, output_csv_file)

    if pending_files:
        logging.warning(f"📋 {len(pending_files)} files permanently failed after all retries. See failed_extraction_files.csv.")
        save_failed_files_to_csv(pending_files, "failed_extraction_files.csv")
    else:
        logging.info("🎉 All files processed successfully!")

if __name__ == "__main__":
    # Log configuration
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
    if use_vertex:
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "not-set")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "not-set")
        logging.info(f"🚀 Starting with Vertex AI - Project: {project}, Location: {location}")
    else:
        logging.info("🚀 Starting with regular Gemini API")
    
    start_time = time.time()
    asyncio.run(main())
    elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
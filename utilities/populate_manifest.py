#!/usr/bin/env python3
"""
Populate manifest database from existing CSV results.

This script reads the classification and extraction CSV files and populates
the SQLite manifest database so that resume mode can work properly.
"""

import csv
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utilities.manifest import ProcessingManifest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def populate_manifest_from_csvs(output_folder: str, manifest_path: str):
    """
    Populate the manifest database from existing CSV results.
    
    Args:
        output_folder: Path to the output folder containing CSV files
        manifest_path: Path to the manifest database file
    """
    logging.info(f"🔄 Populating manifest database: {manifest_path}")
    logging.info(f"📁 Reading CSV files from: {output_folder}")
    
    # Initialize manifest
    manifest = ProcessingManifest(manifest_path)
    manifest.connect()
    
    # CSV file paths
    classification_csv = os.path.join(output_folder, "classification_results.csv")
    vendor_csv = os.path.join(output_folder, "vendor_extraction_results.csv")
    employee_csv = os.path.join(output_folder, "employee_extraction_results.csv")
    
    classification_count = 0
    extraction_count = 0
    error_count = 0
    
    # Process classification results
    if os.path.exists(classification_csv):
        logging.info("📊 Processing classification results...")
        
        with open(classification_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row.get('File Path', '')
                classification = row.get('Classification', '')
                
                if not file_path or not classification:
                    continue
                
                # Check if this is a processing failure
                if classification == 'processing_failed':
                    error_message = row.get('Classification Notes', 'Processing failed during classification')
                    manifest.mark_error(file_path, error_message)
                    error_count += 1
                else:
                    # Mark as successfully classified
                    manifest.mark_classified(file_path, classification, classification)
                    classification_count += 1
                    
                    # For irrelevant files, mark as extracted (no extraction needed)
                    if classification == 'irrelevant':
                        manifest.mark_extracted(file_path)
                        extraction_count += 1
        
        logging.info(f"✅ Processed {classification_count} classifications and {error_count} errors")
    else:
        logging.warning(f"❌ Classification CSV not found: {classification_csv}")
    
    # Track which files have been extracted
    extracted_files = set()
    
    # Process vendor extraction results
    if os.path.exists(vendor_csv):
        logging.info("🏢 Processing vendor extraction results...")
        vendor_extracted = 0
        
        with open(vendor_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row.get('file_path', '')
                if file_path and file_path not in extracted_files:
                    manifest.mark_extracted(file_path)
                    extracted_files.add(file_path)
                    vendor_extracted += 1
        
        logging.info(f"✅ Processed {vendor_extracted} vendor extractions")
    else:
        logging.warning(f"❌ Vendor CSV not found: {vendor_csv}")
    
    # Process employee extraction results
    if os.path.exists(employee_csv):
        logging.info("👤 Processing employee extraction results...")
        employee_extracted = 0
        
        with open(employee_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row.get('file_path', '')
                if file_path and file_path not in extracted_files:
                    manifest.mark_extracted(file_path)
                    extracted_files.add(file_path)
                    employee_extracted += 1
        
        logging.info(f"✅ Processed {employee_extracted} employee extractions")
    else:
        logging.warning(f"❌ Employee CSV not found: {employee_csv}")
    
    extraction_count += len(extracted_files)
    
    # Force commit all changes
    manifest.force_commit()
    
    # Get summary statistics
    summary = manifest.get_summary()
    
    logging.info("📊 Manifest Population Summary:")
    logging.info(f"   📁 Total files: {summary['total_files']}")
    logging.info(f"   🏷️  Classified: {summary['classified']}")
    logging.info(f"   ✅ Extracted: {summary['extracted']}")
    logging.info(f"   ❌ Failed: {summary['failed']}")
    logging.info(f"   ⏳ Pending classification: {summary['pending_classification']}")
    logging.info(f"   ⏳ Pending extraction: {summary['pending_extraction']}")
    
    manifest.close()
    logging.info("🎉 Manifest population complete!")
    
    return summary

def main():
    """Main function to populate manifest from CSV results."""
    # Default paths
    output_folder = "output/16"
    manifest_path = "output/manifest.db"
    
    # Check if output folder exists
    if not os.path.exists(output_folder):
        logging.error(f"❌ Output folder not found: {output_folder}")
        sys.exit(1)
    
    # Check if at least classification CSV exists
    classification_csv = os.path.join(output_folder, "classification_results.csv")
    if not os.path.exists(classification_csv):
        logging.error(f"❌ Classification CSV not found: {classification_csv}")
        sys.exit(1)
    
    try:
        summary = populate_manifest_from_csvs(output_folder, manifest_path)
        
        # Print next steps
        logging.info("")
        logging.info("🚀 Next Steps:")
        logging.info("   1. Verify the manifest database is populated correctly")
        logging.info("   2. Run the resume command:")
        logging.info("      python main_2step_enhanced_adv.py --input input/16 --resume --no-tui")
        logging.info("")
        
        # Calculate expected resume work
        if summary['pending_classification'] > 0 or summary['pending_extraction'] > 0:
            logging.info(f"📋 Expected resume work:")
            if summary['pending_classification'] > 0:
                logging.info(f"   🏷️  {summary['pending_classification']} files need classification")
            if summary['pending_extraction'] > 0:
                logging.info(f"   ✅ {summary['pending_extraction']} files need extraction")
        else:
            logging.info("✨ All files appear to be processed! Resume mode should complete quickly.")
            
    except Exception as e:
        logging.error(f"❌ Error populating manifest: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# Invoice-PDF Pipeline – Troubleshooting & Fix Proposal

_Last updated: 2025-08-06_

---

## 1. Symptoms you observed

- **Logs show plenty of `… - Success` lines** for both classification and extraction.
- Final summary nevertheless reports **`0 classified`, `0 extracted`** and raises warnings:
  - `No vendor extraction results to save.`
  - `No employee extraction results to save.`
- `json_responses/` stays empty unless `DEBUG_RESPONSES=1` is set.
- CSV files in `output/` are created but contain empty rows only.

---

## 2. Root-cause analysis

| #     | Description                                                                                                                                                                                                                                                                                                                                                            | Source lines                                   |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| **A** | _Header/field-name mismatch_ – `StreamingCSVWriter` writes each row with<br>`{field: row.get(field,'') …}`. All field names come from<br>`get_vendor_extraction_csv_config()` / `get_employee_extraction_csv_config()` and are Title-Case (`"File Name"`, …) while the runtime `result` dicts are snake-case (`file_name`, …). → Every lookup fails → blank CSV lines. | `main_2step_enhanced.py` 1757-1796 + 1149-1176 |
| **B** | After chunk processing we intentionally do `classification_results = []` and `extraction_results = []` to free memory **before** the final reporting section. Consequently every later statistic as well as the secondary CSV write based on those lists sees nothing.                                                                                                 | `main_2step_enhanced.py` 2045-2049             |
| **C** | Raw Gemini JSON responses are saved only when the environment variable `DEBUG_RESPONSES` is set (`SAVE_RESPONSES = …`). That is expected behaviour – not a bug – but worth noting.                                                                                                                                                                                     | `main_2step_enhanced.py` 65, 308-317, 546-548  |

---

## 3. Minimal-impact fix

### 3.1 Stream the correct columns

1. **Add two helper mappers** (one for vendor, one for employee) that transform each `result` dict into a dict whose keys match the Title-Case field names.
2. Replace the current loop inside `process_chunk_with_streaming_csv` with calls to those mappers.

_Pseudo-patch_

```python
# after get_vendor_extraction_csv_config()

def map_vendor_row(result: dict):
    doc_st = result.get("document_status", {})
    entries = result.get("extracted_data", []) or [{}]
    for e in entries:
        yield {
            "File Name"         : result.get("file_name", ""),
            "File Path"         : result.get("file_path", ""),
            "Document Type"     : result.get("document_type_processed", ""),
            "Readable"          : doc_st.get("readable", ""),
            # … fill remaining columns …
            "Invoice Number"    : e.get("invoice_number", ""),
            "Basic Amount"      : e.get("basic_amount", ""),
            "Total Amount"      : e.get("total_amount", ""),
            "Total Pages In PDF": result.get("total_pages_in_pdf", ""),
            "Processing Notes"  : result.get("processing_notes", "")
        }
```

Same idea for `map_employee_row`.

3. Update the streaming section:

```python
for res in extraction_results:
    if res["document_type_processed"] == "vendor_invoice":
        for row in map_vendor_row(res):
            await vendor_writer.write_row(row)
    elif res["document_type_processed"] == "employee_t&e":
        for row in map_employee_row(res):
            await employee_writer.write_row(row)
```

### 3.2 Retain in-memory result lists until the very end

Simply **remove** lines 2045-2049 _or_ move them below the final report + Excel generation.

### 3.3 (Optional) Always keep raw responses while debugging

```bash
export DEBUG_RESPONSES=1   # before running the pipeline
```

---

## 4. Expected outcome after fix

- Streaming writers populate `vendor_extraction_results.csv` and `employee_extraction_results.csv` with real data – visible already during processing.
- Final summary shows the correct classified / extracted counts.
- Excel & markdown reports are complete.
- No more “No … extraction results to save.” warnings.

---

## 5. Verification checklist

1. Run pipeline on a small sample (e.g., 5 PDFs) ✔️
2. Inspect live CSVs – rows should populate immediately ✔️
3. Ensure final log reports non-zero counts ✔️
4. Open produced Excel workbook – data should appear in employee / vendor sheets ✔️

---

## 6. Possible future improvements

- Validate field names automatically and raise if mismatched (early-fail).
- Switch to dataclass / pydantic objects for stronger typing between stages.
- Consider writing directly to SQLite instead of CSV for easier querying.

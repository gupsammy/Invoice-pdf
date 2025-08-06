# Implementation Plan: Phase-0 Optimisations for Large-Batch PDF Processing

This plan turns the **"Phase-0 checklist"** (low-hanging fruit, ≤1 day of work) into concrete engineering tasks. Each task block contains:

• Objective & rationale  
• Code locations  
• Exact changes / snippets  
• Expected effort & test procedure

> Scope = keep a **single API-key (10 QPS)**, handle 1 000–10 000 PDFs on a single workstation.

---

## 0 – Preparation

| Step | Action                                                                                                       |
| ---- | ------------------------------------------------------------------------------------------------------------ |
| 0.1  | **Create a feature branch** `perf/phase0-optimisations`                                                      |
| 0.2  | `pip install --upgrade "google-genai[aiohttp]"` into the venv & add to `requirements.txt`                    |
| 0.3  | Set two env flags in `.env.example`: `DEBUG_RESPONSES` (default 0) and `PROCESSING_CHUNK_SIZE` (default 500) |

---

## 1 – Global Quota Semaphore & Pipeline Overlap

### 1.1 Replace twin semaphores with one

**Files**: `main_2step_enhanced.py`

```python
# OLD (lines ~1461-1464)
classify_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLASSIFY)
extract_semaphore  = asyncio.Semaphore(MAX_CONCURRENT_EXTRACT)
```

```python
# NEW
QUOTA_LIMIT = 10  # single-key AFC limit
quota_semaphore = asyncio.Semaphore(QUOTA_LIMIT)
```

Refactor function signatures (`classify_document_async`, `extract_document_data_async`, wrappers, etc.) to receive `quota_semaphore` instead of individual ones.

### 1.2 Enable overlap

At the end of the classification stage **do not wait** for all tasks before starting extraction:

```python
# Create classification tasks, then immediately start extraction tasks for each result
# Use asyncio.Queue or simple list append while classification futures resolve.
```

Effort: 2 hrs. Test with 30 PDFs → confirm log shows mixed classification/extraction interleaving and never >10 concurrent requests (check timestamps).

---

## 2 – Response Dump Flag

### 2.1 Environment toggle

Add at top-level constants section:

```python
SAVE_RESPONSES = os.getenv("DEBUG_RESPONSES", "0") == "1"
```

### 2.2 Wrap the file-write blocks

```python
if SAVE_RESPONSES or not success:
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(response.text)
```

Apply in both classification & extraction loops.

Effort: 20 min. Test with flag 0/1 and inspect `json_responses/*` directory size.

---

## 3 – Switch to aiohttp Transport

### 3.1 Install extra

(covered in preparation)

### 3.2 Instantiate client with custom `http_options`

```python
from google.genai import types
import aiohttp

http_options = types.HttpOptions(
    async_client_args={
        "connector": aiohttp.TCPConnector(limit=50, limit_per_host=10),
        "timeout": aiohttp.ClientTimeout(total=60),
    }
)

client = genai.Client(api_key=API_KEY, http_options=http_options)
```

Effort: 30 min. Benchmark single call latency before/after.

---

## 4 – Thread-off PDF Slicing

### 4.1 Update call sites

```python
pdf_bytes = await asyncio.to_thread(
    extract_first_n_pages_pdf, pdf_path, MAX_CLASSIFICATION_PAGES)
```

(Same for extraction truncation.)

### 4.2 Guard file-descriptor count

Add module-level semaphore `pdf_fd_sem = asyncio.Semaphore(50)` and acquire it around `fitz.open` if OS limit is hit during stress test.

Effort: 45 min. Validate no event-loop blockage via `asyncio.get_running_loop().time()` deltas.

---

## 5 – Streaming CSV Writers

### 5.1 Utility class

Create `utilities/streaming_csv.py`:

```python
class StreamingCSVWriter:
    def __init__(self, path, fieldnames):
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.lock = asyncio.Lock()

    async def write_row(self, row: dict):
        async with self.lock:
            self.writer.writerow(row)
            self.file.flush()

    def close(self):
        self.file.close()
```

### 5.2 Integrate

• Instantiate one writer per CSV before loops.  
• Replace list-append with `await csv_writer.write_row(row_dict)` immediately after parsing.

Effort: 1 hr. Memory before/after with 5 000 PDFs should drop from ~1.2 GB → <200 MB.

---

## 6 – Chunked Processing

### 6.1 Helper

```python
def chunker(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]
```

### 6.2 Main loop change

Around the existing `pdf_files` list:

```python
for chunk in chunker(pdf_files, int(os.getenv("PROCESSING_CHUNK_SIZE", "500"))):
    await process_one_chunk(chunk)
    gc.collect()   # let memory free
```

`process_one_chunk` is basically the current `main()` body minus discovery.

Effort: 1 hr.

---

## 7 – Resume Logic Fix

Update glob in `get_missing_extraction_files`:

```python
pattern = f"{stem}_extraction_{doc_type}_attempt_*.txt"
if any(fnmatch.filter(os.listdir(subdir), pattern)):
    extracted_files.add(file_path)
```

Effort: 15 min. Unit-test with manually copied attempt_2 files.

---

## 8 – Misc Minor Fixes

- Add jitter to retry back-off (`delay = min(10, 2**attempt + random.uniform(0,3))`).
- Cache `json_str` once inside malformed-JSON repair block.

Effort: 10 min.

---

## Acceptance Criteria

1. **Functional parity** – Results CSV / summary identical for same input set.
2. **Throughput** – 1000-file synthetic batch completes in ≤65 min on dev machine (≈3.9 s/file).
3. **Memory** – RSS ≤1.5 GB at any point during run.
4. **Disk output** – `json_responses/` size ≤5 % of previous size when `DEBUG_RESPONSES` off.

---

## Roll-out Plan

1. PR into `perf/phase0-optimisations`, run unit tests & a 30-file smoke test.
2. Merge → trigger nightly big-batch job against 3 000-PDF corpus.
3. Monitor logs for 429/500 rates; adjust `PROCESSING_CHUNK_SIZE` if memory still climbs.
4. After stable run, tag **v1.1.0** and update README performance section.

---

_© 2025 – Invoice-pdf optimisation plan_

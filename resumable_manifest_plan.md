# Resumable Processing – Manifest & Resume Strategy

This document proposes **a unified, low-overhead resume mechanism** for `main_2step_enhanced.py` that works regardless of when the pipeline is interrupted (before, during or after classification / extraction) and requires **only one flag** (`--resume`).

---

## 0. High-Level Requirements

1. A single _resume_ flow must:
   - Skip **already-classified** files.
   - Skip **already-extracted** files.
   - Continue pending work without redoing anything.
2. Never delete JSON response artefacts automatically.
3. Add **minimal performance overhead** (the pipeline is already slow due to API calls).
4. Support 10k+ PDFs and parallel workers without corruption.

---

## 1. Candidate Strategies

| #   | Strategy                                                    | Pros                                                                                                                                                                                                                            | Cons                                                                                                                                                                                 |
| --- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | **CSV manifest** (reuse existing CSVs)                      | – No extra file<br> – Human-readable                                                                                                                                                                                            | – CSV write is non-atomic ⇒ may miss last row during crash<br> – Updating a specific row costs O(N) rewrite<br> – Hard to mark per-file multi-state (classified & extracted) cleanly |
| 2   | **Per-file sidecar JSON** (`file.pdf.status.json`)          | – Trivial JSON write per file<br> – Independent, atomic                                                                                                                                                                         | – 2× file-count = 4k+ FS entries → noticeable FS overhead on HDD / network drives<br> – Listing/reading many tiny files slows resume scan                                            |
| 3   | **Monolithic NDJSON log** (`manifest.ndjson`)               | – Append-only (= O(1) write) & crash-safe<br> – Simple to parse                                                                                                                                                                 | – To compute latest state we must read full file (O(N)) each resume; acceptable for 10k but grows linearly                                                                           |
| 4   | **Lightweight SQLite DB** (`manifest.db`) **(recommended)** | – Atomic commits, ACID, built-in Python `sqlite3` (no deps)<br> – O(1) lookup/update per file<br> – Handles concurrent writes from many coroutines (SQLite WAL mode)<br> – Compact single file<br> – Easy queries for analytics | – Slightly more code (schema + helper)                                                                                                                                               |

---

## 2. Recommended Approach – SQLite Manifest

### 2.1 Schema (single table)

```sql
CREATE TABLE IF NOT EXISTS progress (
    file_path      TEXT PRIMARY KEY,
    classified     INTEGER DEFAULT 0,   -- 0 = not yet, 1 = done
    classification TEXT,               -- vendor_invoice / employee_t&e / irrelevant / fail
    extracted      INTEGER DEFAULT 0,   -- 0 = not yet, 1 = done (only if relevant)
    doc_type       TEXT,               -- vendor_invoice / employee_t&e
    last_error     TEXT,
    updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

_Indexes_: `PRIMARY KEY` is enough, lookups are by `file_path`.

### 2.2 Write Logic

| Event                                  | SQL                                                                                                                                                                                                                                                    |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Classification success                 | `INSERT INTO progress (file_path, classified, classification, doc_type, updated_at) VALUES (...) ON CONFLICT(file_path) DO UPDATE SET classified=1, classification=excluded.classification, doc_type=excluded.doc_type, updated_at=CURRENT_TIMESTAMP;` |
| Classification failure (non-retryable) | same as above with `classification='processing_failed'`                                                                                                                                                                                                |
| Extraction success                     | `UPDATE progress SET extracted=1, updated_at=CURRENT_TIMESTAMP WHERE file_path=?;`                                                                                                                                                                     |
| Extraction failure                     | `UPDATE progress SET last_error=?, updated_at=CURRENT_TIMESTAMP WHERE file_path=?;`                                                                                                                                                                    |

_Commit frequency_: commit once per individual update (cost ≪ network/api latency) or batch every 100 updates inside the worker.

### 2.3 Resume Algorithm (`--resume`)

1. Open `manifest.db` (create if missing).
2. **Classification queue** = every `*.pdf` **not** in table **OR** `classified=0`.
3. **Extraction queue** = every row where `classified=1 AND extracted=0 AND classification IN ('vendor_invoice','employee_t&e')`.
4. Run two workers exactly like today using those filtered lists.
   - Newly discovered PDFs (added after last run) are automatically picked up.

### 2.4 Migration / Backwards Compatibility

- First time: scan existing `classification_results.csv` and `json_responses` to seed rows (optional).
- After migration, old CSVs continue to be produced; manifest is _additional_.

---

## 3. Implementation Steps

1. **Create `manifest.py` helper**
   - Functions: `init_db(path)`, `mark_classified(file_path, classification, doc_type)`, `mark_extracted(...)`, `get_resume_queues(pdf_paths)`.
2. **Update `setup_processing_environment()`**
   - Do **not** delete `json_responses` directories.
   - Always open/initialise manifest (path inside `output/<folder>/manifest.db`).
3. **Classification Worker**
   - Before sending file to API, _re-check_ `classified` flag to avoid double-work in race.
   - After success/failure call `manifest.mark_classified()`.
4. **Extraction Worker**
   - Similar: consult manifest first, update afterwards.
5. **`--resume` flag semantics**
   - Default behaviour = fresh run (redo everything). `--resume` → skip completed tasks using manifest.
6. **Remove JSON-deletion branch**.
7. **Phase-in commit batching** (e.g., `PRAGMA journal_mode=WAL;` + `conn.commit()` every 50 ops) to keep I/O negligible.

---

## 4. Alternative Quick-Win (if you prefer not to touch SQLite)

- Keep existing CSVs but **write them in _append_ mode** and add a `status` column (`classified` / `extracted`). On resume, parse the CSV with a streaming reader and build the two queues. Complexity is lower than SQLite but lookups become O(N) each restart.

---

## 5. Expected Overhead

| Item                          | Cost                       |
| ----------------------------- | -------------------------- |
| SQLite connection open        | ~2 ms                      |
| Single `INSERT … ON CONFLICT` | ~150 µs                    |
| Commit every 100 ops          | Negligible (<2 ms per 100) |
| Resume queue query            | <50 ms for 10k rows        |

Overhead is **orders of magnitude smaller** than one Gemini API call, so throughput is unaffected.

---

## 6. Deliverables

1. `utilities/manifest.py` helper module.
2. Modifications in `main_2step_enhanced.py` (≈ 60 LOC net change).
3. Unit tests for manifest functions.
4. Update `README.md` with new flag description.

---

### Once implemented you can safely “pause” the run (Ctrl-C) and relaunch with `--resume` – the pipeline will pick up exactly where it left off, with no extra waiting.

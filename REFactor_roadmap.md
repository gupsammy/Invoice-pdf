# Invoice-PDF Refactor Road-Map

This document captures the **strategic plan** for modularising and scaling the `Invoice-pdf` code-base (centred around the oversized `main_2step_enhanced.py`). It is intended for every developer working on, or reviewing, the upcoming refactor.

---

## 1. Why are we refactoring?

| Symptom                                                | Impact                                                 |
| ------------------------------------------------------ | ------------------------------------------------------ |
| 3K-line monolith `main_2step_enhanced.py`              | Hard to navigate, reason about, or unit-test           |
| Mixed concerns (CLI, business logic, I/O, TUI, config) | Changes in one area risk subtle side-effects in others |
| Implicit globals / env vars                            | Hidden coupling, difficult to override in tests        |
| Multiple ad-hoc retry / JSON-repair implementations    | Duplicate code, inconsistent behaviour                 |
| Growing feature list (manifest resume, chunking, TUI)  | File keeps growing ≈ **exponential complexity**        |

**Goal**: _isolate concerns, introduce typed data-models, and enable feature development without touching unrelated code._

---

## 2. Target Architecture

```
invoice_pdf/               # new top-level package
│
├── __main__.py            # thin CLI wrapper (argparse only)
├── config.py              # read env / .env / CLI flags → `Settings` dataclass
├── cli.py                 # orchestration, progress bars, logging
│
├── core/                  # *pure* business logic – no disk/print
│   ├── pipeline.py        # run_pipeline(files, client, settings)
│   ├── classify.py        # async classify_document()
│   ├── extract.py         # async extract_document()
│   ├── pdf_utils.py       # page-count, slicing, semaphore guard
│   ├── rate_limit.py      # generic async retry + Semaphore helpers
│   └── models.py          # `ClassificationResult`, `ExtractionResult`, etc. (pydantic)
│
├── io/                    # side-effects (read / write)
│   ├── manifest.py        # existing ProcessingManifest → moved here
│   ├── csv_stream.py      # StreamingCSVWriter & helpers
│   ├── excel_report.py    # build Excel workbook
│   └── summary_md.py      # build Markdown report
│
└── tui/
    └── monitor.py         # rich-TUI (current utilities/tui.py)
```

### Key Rules

1. **core** never imports `io` or `tui` – it _returns_ plain models.
2. **io** modules translate models ↔️ external representations (CSV, Excel…).
3. `cli.py` is the _only_ place where layers are wired together.

---

## 3. Migration Road-Map (incremental, test-driven)

### Phase 0 – Safety-Net & Project Skeleton ✅ **COMPLETED**

- ✅ Tag current `main` as `v0-legacy-snapshot` and freeze.
- ✅ Scaffold `invoice_pdf/` package and `pyproject.toml`.
- ✅ Move `main_2step_enhanced.py` to `invoice_pdf/_legacy/main_2step_enhanced.py` **without editing a single line**.
- ✅ Add `tests/test_regression_fixture.py` (golden-file snapshot for ≤5 tiny PDFs).
- ✅ Install `pre-commit` with `ruff`, `black`, `isort`, `mypy`, `reuse`.
- ✅ Create basic GitHub Actions workflow: lint ➜ type-check ➜ pytest.
- ✅ Add `Makefile` / `justfile` with `run`, `test`, `fmt` shortcuts.

**Status**: ✅ Complete - Committed as `97a7c00`. Background process (PID 13908) continues unaffected.

### Phase 1 – Typed Configuration ✅ **COMPLETED**

- ✅ Implement `invoice_pdf/config.py` (`Settings` dataclass, pydantic v2).
- ✅ Create `invoice_pdf/logging_config.py` with a reusable `dictConfig`.
- ✅ Patch `_legacy/main_2step_enhanced.py` to instantiate a `Settings` object instead of reading `os.environ` directly (mechanical replace, **no logic change**).

**Status**: ✅ Complete - Legacy script now uses typed configuration. Tests passing.

### Phase 2 – Canonical Data-Models ✅ **COMPLETED**

- ✅ Add `invoice_pdf/core/models.py` (pydantic v2).
- ✅ Wrap existing payloads (`classification_data`, `extraction_data`, preprocessing failure) in models.
- ✅ Update `_legacy` classify / extract functions to return models.
- ✅ Add `tests/test_models.py` to assert JSON round-trip and backwards compatibility.

**Status**: ✅ Complete - Classification functions now use typed models internally with legacy dict compatibility.

### Phase 3 – Shared Concurrency & Rate-Limiting ✅ **COMPLETED**

- ✅ Create `invoice_pdf/core/rate_limit.py` with `async retry_with_backoff` + `CapacityLimiter` wrapper.
- ✅ Remove bespoke retry loops from `_legacy` classify / extract and delegate to helper.
- ✅ Replace raw `asyncio.Semaphore` with `anyio.CapacityLimiter`.
- ✅ Fix all legacy script integration issues (IndentationError, syntax errors).
- ✅ Complete test coverage: 17 rate limiting tests + 38 total tests passing.
- ✅ Clean up extraction function retry logic and old exception handling.

**Status**: ✅ FULLY Complete - Committed as `066329f`. All tests passing (38/38). Background process (PID 13908) continues unaffected.

**Technical Details for Future Reference:**
- `CapacityLimiter` class provides anyio.CapacityLimiter compatibility with asyncio.Semaphore fallback
- `retry_with_backoff()` function replaces all manual retry loops with exponential backoff + jitter
- `RateLimitedExecutor` combines capacity limiting and retry logic for common use cases
- Legacy functions `classify_document_async()` and `extract_document_data_async()` now use new rate limiting
- Main semaphore initialization updated: `quota_limiter = CapacityLimiter(QUOTA_LIMIT)` 
- PDF semaphore: `pdf_fd_semaphore = CapacityLimiter(PDF_FD_SEMAPHORE_LIMIT)`
- All function signatures updated from `asyncio.Semaphore` → `CapacityLimiter`

### Phase 4 – Pure PDF Helpers ✅ **COMPLETED**

- ✅ Implement `invoice_pdf/core/pdf_utils.py` (`get_page_count`, `extract_first_n_pages`).
- ✅ Migrate calls in `_legacy`.
- ✅ Unit-test with tiny fixture PDFs.

**Status**: ✅ Complete - Committed as `[TBD]`. All tests passing (67/67). Background process (PID 13908) continues unaffected.

**Technical Details for Future Reference:**
- `get_page_count()` and `extract_first_n_pages()` functions provide pure PDF operations
- `safe_get_page_count()` and `safe_extract_first_n_pages()` provide async wrappers with semaphore protection
- `initialize_pdf_semaphore()` centralizes semaphore initialization
- Legacy code updated to use new PDF utilities with proper imports
- 16 comprehensive unit tests cover sync/async operations, edge cases, and concurrent access
- PDF semaphore management now centralized in `core.pdf_utils` module
- All PDF operations now use `Path | str` type hints for flexibility

### Phase 5 – Persistence Layer Early

- Move `utilities/manifest.py` → `invoice_pdf/io/manifest.py`.
- Move `utilities/streaming_csv.py` → `invoice_pdf/io/csv_stream.py`.
- Provide shim import in `_legacy` to avoid mass rename diff.
- Add migration tests verifying resume works.

### Phase 6 – Output Writers

- Convert CSV helpers into writer classes (`ClassificationCSVWriter`, `VendorCSVWriter`, `EmployeeCSVWriter`) placed in `invoice_pdf/io`.
- Each writer exposes `header: list[str]` and `.write(result)` API.
- Adjust Excel / summary builders to consume headers dynamically.

### Phase 7 – Core Pipeline

- `invoice_pdf/core/pipeline.py`
  - Accepts `Iterable[Path]`, yields `Result` models.
  - Uses classify / extract coroutines and `rate_limit`.
  - 100 % side-effect-free.

### Phase 8 – TUI & CLI Rewrite

- Port `utilities/tui.py` to `invoice_pdf/tui/monitor.py` (Rich).
- Implement `invoice_pdf/cli.py`:
  1. Parse CLI & env via `config.Settings`.
  2. Discover PDFs.
  3. Instantiate writers & manifest.
  4. Run `async pipeline.run()`; stream results to writers and TUI.

### Phase 9 – Delete Legacy Code

- Delete `invoice_pdf/_legacy` after golden-file regression passes.
- Remove transitional shims.

### Phase 10 – Docs, Benchmarks & Polish

- Add architecture diagram, API reference, and How-To examples to README.
- Add `scripts/benchmark.py`; ensure no performance regression.
- Final CI matrix (Python 3.10-3.12, macOS + Ubuntu).

---

## 4. Module Responsibilities (single-sentence each)

| Module              | Purpose                                                         |
| ------------------- | --------------------------------------------------------------- |
| **config.py**       | Central typed settings + env parsing (no side-effects).         |
| **core.models**     | Canonical data-models shared across layers.                     |
| **core.classify**   | Turn `(PDF bytes ➜ ClassificationResult)` asynchronously.       |
| **core.extract**    | Turn `(PDF bytes, doc_type ➜ ExtractionResult)` asynchronously. |
| **core.pipeline**   | Coordinate classify/extract; implement retries, queuing.        |
| **core.rate_limit** | Async retry with backoff, CapacityLimiter, RateLimitedExecutor. |
| **core.pdf_utils**  | Pure PDF operations (page count, extraction); uses CapacityLimiter for fd-semaphore. |
| **io.manifest**     | SQLite progress persistence.                                    |
| **io.csv_stream**   | Streaming, back-pressure friendly CSV writer.                   |
| **io.excel_report** | Convert result collections → XLSX workbook.                     |
| **io.summary_md**   | Create Markdown report summarising a run.                       |
| **tui.monitor**     | Rich‐TUI showing live manifest stats & controls.                |
| **cli.py**          | Glue everything; the only script users run.                     |

---

## 5. Testing strategy

1. **Unit tests per module** – pure logic is now testable with fakes.
2. **Integration test** – run pipeline on <5 tiny PDFs, assert CSV rows produced.
3. **Contract test** – mock GenAI responses, verify classification→extraction flow.
4. **Regression harness** – keep a golden Excel + summary; diff after refactor.

---

## 6. Risk Mitigation

- _Large diff fatigue_: migrate in phases, PR under `feature/refactor-phase-X` branches.
- _Hidden side-effects_: enable `mypy --strict` + CI to catch missing imports.
- _Performance regression_: preserve current semaphore counts; add benchmarks.

---

## 7. Glossary

- **Pure module** – deterministic, no disk I/O or logging beyond `return` value.
- **Writer** – consumer of models, produces external artifact.
- **Settings** – immutable snapshot of configuration for a run.

---

## 8. Next Action

> **Start Phase 5:** Move `utilities/manifest.py` → `invoice_pdf/io/manifest.py` and `utilities/streaming_csv.py` → `invoice_pdf/io/csv_stream.py`. Provide shim import in `_legacy` to avoid mass rename diff. Add migration tests verifying resume works.

## 9. Current Status (as of commit [TBD])

**✅ Completed Phases:** 0, 1, 2, 3, 4
**🔄 Next Phase:** 5 (Persistence Layer Early)  
**🏃‍♂️ Background Process:** PID 13908 - STILL RUNNING, unaffected by refactoring
**📊 Test Status:** 67/67 tests passing across all modules
**🗂️ Key Files Created:**
- `invoice_pdf/config.py` - Typed configuration system
- `invoice_pdf/logging_config.py` - Centralized logging
- `invoice_pdf/core/models.py` - Pydantic v2 data models  
- `invoice_pdf/core/rate_limit.py` - Async rate limiting & retry logic
- `invoice_pdf/core/pdf_utils.py` - Pure PDF operations & semaphore management
- `tests/test_*.py` - Comprehensive test coverage (67 tests)

**⚠️ Important for Next Session:**
- Legacy script is in `invoice_pdf/_legacy/main_2step_enhanced.py`
- Uses new imports: `from core.rate_limit import CapacityLimiter, retry_with_backoff`
- Uses new imports: `from core.pdf_utils import get_page_count, extract_first_n_pages, safe_get_page_count, safe_extract_first_n_pages, initialize_pdf_semaphore`
- All semaphores converted to CapacityLimiter (anyio compatible, asyncio fallback)
- PDF operations now centralized in `core.pdf_utils` module
- Background process must remain unaffected - test after each change!

---

_This plan was generated 2025-08-06._

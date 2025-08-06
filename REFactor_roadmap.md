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

## 3. Migration Phases (incremental & safe)

### Phase 0 ✔︎ – _Preparation_

- Freeze current `develop` branch.
- Add this `REFactor_roadmap.md`.

### Phase 1 – _Introduce typed models_

1. Create `invoice_pdf/core/models.py` using `pydantic`.
2. Wrap existing dicts (`classification_data`, `extraction_data`, failure dict) into models.
3. Update `classify_document_async` & `extract_document_data_async` to return model instances (`.dict()` for callers).
4. Add `tests/test_models.py` verifying JSON-round-trip.

### Phase 2 – _Extract PDF helpers_

- Move `get_pdf_page_count`, `extract_first_n_pages_pdf`, semaphore logic into `core/pdf_utils.py`.
- Replace direct imports in code.
- Unit-test with fake PDFs.

### Phase 3 – _Move Manifest & StreamingCSV_

- Physically move `utilities/manifest.py` → `io/manifest.py` (update imports).
- Same for `utilities/streaming_csv.py` → `io/csv_stream.py`.
- No behavioural change expected; run tests.

### Phase 4 – _Split Output Generators_

- Convert `save_vendor_extraction_results_to_csv`, `save_employee…`, etc. into classes under `io/`.
- Public interface: `writer.write(result: VendorExtraction)`.
- Migrate calls in main flow.

### Phase 5 – _Create `core/pipeline.py`_

- Extract **pure** orchestration logic:
  - pipeline overlap (queues), retry wrapper.
  - Accept `Iterable[Path]`, return `AsyncIterator[Result]`.
- Remove logging & disk writes – caller handles.

### Phase 6 – _Rewrite CLI Layer_

- `cli.py` now:
  1. Parses CLI & env via `config.Settings`.
  2. Discovers PDFs.
  3. Instantiates writers & manifest.
  4. Runs `async run_pipeline()`; streams results to writers.
  5. Drives TUI (optional).

### Phase 7 – _Delete legacy modules_

- Remove `main_2step_enhanced.py` and `utilities/tui.py` once feature parity is confirmed.

### Phase 8 – _Refinement & Docs_

- Add README diagrams & examples.
- CI pipeline: flake8 + pytest.

---

## 4. Module Responsibilities (single-sentence each)

| Module              | Purpose                                                         |
| ------------------- | --------------------------------------------------------------- |
| **config.py**       | Central typed settings + env parsing (no side-effects).         |
| **core.models**     | Canonical data-models shared across layers.                     |
| **core.classify**   | Turn `(PDF bytes ➜ ClassificationResult)` asynchronously.       |
| **core.extract**    | Turn `(PDF bytes, doc_type ➜ ExtractionResult)` asynchronously. |
| **core.pipeline**   | Coordinate classify/extract; implement retries, queuing.        |
| **core.pdf_utils**  | Pure PDF operations; protected by fd-semaphore.                 |
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

> **Start Phase 1:** Scaffold `core/models.py`, migrate `classification_data` & `extraction_data` to strongly-typed objects. Once merged, proceed to Phase 2.

---

_This plan was generated 2025-08-06._

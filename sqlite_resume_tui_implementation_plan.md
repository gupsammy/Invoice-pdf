# Feature Branch: SQLite Manifest + Interactive TUI Pause⁄Resume

> Branch name `feature/sqlite-resume-tui`
>
> Goal: Deliver a **single-flag resumable pipeline** powered by a crash-safe
> SQLite manifest **plus** an ultra-lightweight Text-UI that lets the operator
> pause, resume, and monitor progress without killing the process.

---

## 1 High-Level Design

```
┌──────────────────┐        INSERT / UPDATE           ┌────────────────────┐
│  workers (async) │ ────────────────► manifest.db ──►│  progress lens     │
└──────────────────┘                                   └────────────────────┘
       ▲  ▲                                                   ▲     ▲
       │  │  live counts via pub/sub                          │     │
       │  └────────────────────────────────────────────────────┘     │
       │                                                            │
       └─────── pause_event / resume_event  ◄─────────  TUI (rich)
```

- **Manifest Layer** – central source-of-truth; atomic SQL writes keep track
  of every fileʼs state.
- **Pipeline Layer** – classification / extraction workers consult manifest
  **before** doing work and **after** completing.
- **TUI Layer** – runs in its own asyncio task; leverages `rich` or
  `textual` to render progress and handle hot-keys (`p`=pause, `r`=resume,
  `q`=quit gracefully). It toggles an `asyncio.Event()` consumed by workers.

---

## 2 SQLite Manifest

### 2.1 Schema (unchanged from previous doc)

```sql
CREATE TABLE IF NOT EXISTS progress (
    file_path      TEXT PRIMARY KEY,
    classified     INTEGER DEFAULT 0,
    classification TEXT,
    extracted      INTEGER DEFAULT 0,
    doc_type       TEXT,
    last_error     TEXT,
    updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

- WAL mode (`PRAGMA journal_mode=WAL;`) for write concurrency.

### 2.2 Helper API (`utilities/manifest.py`)

```python
init_db(path) -> sqlite3.Connection
mark_classified(path, classification, doc_type)
mark_extracted(path)
mark_error(path, err)
get_resume_queues(pdf_paths) -> (classify_list, extract_list)
summary() -> dict  # counts for TUI
```

- Each helper does a single SQL statement with `conn.commit()` batched every
  `N` operations (configurable, default = 50).

### 2.3 Resuming Logic

- New CLI flag: `--resume` (boolean). On startup:
  1. Scan _filesystem_ for all PDFs (like today).
  2. Query manifest for state → build two lists.
  3. Skip any file already `extracted=1`.
  4. If classification missing but extraction done (shouldnʼt happen), log
     warning and mark `classified=1`.

---

## 3 Pause ⁄ Resume Mechanism

### 3.1 Internal Event

```python
pause_event = asyncio.Event()
pause_event.set()  # allowed to run
```

Workers wrap long-running steps:

```python
await pause_event.wait()
# do classification / extraction call
```

### 3.2 TUI Controls (`utilities/tui.py`)

- **Library**: `rich` (native asyncio support via `rich.live`). No extra
  dependency if we already have `rich` for logs; otherwise add to
  `requirements.txt` (`rich>=13.7`).
- **Layout**: single panel showing counts (files processed, queued, failed,
  pause status, ETA). Key handling:
  - `p` → pause (clears event)
  - `r` → resume (sets event)
  - `q` → graceful shutdown (sets a global `shutdown_event` so workers finish
    current task, then exit)

---

## 4 Code Changes

| Step | File                             | ± LOC       | Notes                                                                |
| ---- | -------------------------------- | ----------- | -------------------------------------------------------------------- |
| 1    | `utilities/manifest.py`          | +180        | new helper module                                                    |
| 2    | `main_2step_enhanced.py`         | +≈70 / −≈40 | integrate manifest API; remove JSON-deletion; add pause-event checks |
| 3    | `utilities/tui.py`               | +140        | live dashboard + key handler                                         |
| 4    | `requirements.txt`               | +1          | `rich`                                                               |
| 5    | tests (`tests/test_manifest.py`) | +120        | unit tests                                                           |

Total ≈ 550 LOC new/changed.

---

## 5 Branch & Merge Plan

1. `git checkout -b feature/sqlite-resume-tui`
2. Commit in _atomic_ vertical slices:
   1. **manifest module + unit tests**
   2. **integrate manifest into classification only** (flag-guarded)
   3. **extend to extraction path**
   4. **remove JSON deletion + unify `--resume` flag**
   5. **add TUI skeleton** (read-only)
   6. **add pause/resume control**
3. CI: run existing pipeline tests + new manifest tests.
4. PR description links to `sqlite_resume_tui_implementation_plan.md`.
5. Once approved, squash-merge and delete branch.

---

## 6 Operator UX

```
$ python main_2step_enhanced.py --input big_folder --resume
┌───────────────────────────────┐
│  ▶ Processing…  [r]=resume  │
│                               │
│  Classified:      1450 / 2004 │
│  Extracted:        920 / 1103 │
│  Failed:             7        │
│  State:  RUNNING              │
└───────────────────────────────┘
```

_Press `p`_ → state changes to **PAUSED**; workers block quickly after finishing current in-flight API call. _Press `r`_ to continue, _`q`_ to exit cleanly (manifest persists).

---

## 7 Performance Impact

- Manifest write < 0.2 ms vs. one Gemini API call ~800 ms → **<0.05 %**.
- TUI refresh every 0.5 s; negligible.

---

## 8 Risks & Mitigations

| Risk                                 | Mitigation                                     |
| ------------------------------------ | ---------------------------------------------- |
| Concurrent writes cause DB locks     | SQLite WAL + retry loop (exponential back-off) |
| Operator kills process while DB busy | WAL ensures durability; next start recovers    |
| TUI not wanted in headless mode      | Disable auto-launch with `--no-tui` flag       |

---

### Ready to start coding 🚀

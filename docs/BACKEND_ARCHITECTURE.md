# PaperMate Backend Architecture

This document describes the backend boundaries that should guide future changes.
PaperMate remains a Streamlit app, but backend behavior is organized into
explicit layers so UI changes, workflow changes, task execution, and retrieval
quality can be modified independently.

## Layer Map

```text
app.py
  Streamlit rendering, session state, buttons, forms, toasts, page navigation

src/application/
  User-facing use cases that orchestrate services without importing Streamlit
  - paper_workflow.py: upload, duplicate detection, parse/index scheduling
  - study_workflow.py: question answering, translation jobs, card jobs/saves

src/*_service.py and src/*_pipeline.py
  Domain services and pipelines
  - auth_service/team_service/paper_service/job_service/literature_card_service
  - pdf_parser/chunker/rag_pipeline/card_pipeline/markdown_translator

src/retrieval/
  Query planning, vector/BM25 retrieval, RRF, rerank, evidence expansion,
  context building, and retrieval evaluation helpers

scripts/worker.py
  Background job runtime for parse/index/translate/card/eval jobs

src/db.py
  SQLite schema, additive migrations, and persistence helpers
```

## Dependency Rules

- `app.py` may call `src.application.*` and read UI-friendly service queries, but
  should not build multi-step backend workflows inline.
- `src/application/*` may orchestrate services, pipelines, and jobs, but must not
  import Streamlit or mutate `st.session_state`.
- `src/*_service.py` should own one domain concern: permissions, papers, jobs,
  cards, feedback, or persistence.
- `scripts/worker.py` should own runtime execution only. Business scheduling
  belongs in application services; durable job state belongs in `job_service.py`.
- External model/API retry policy should go through `src.external_call` unless a
  provider requires a documented exception.

## Job Runtime Contract

Jobs are durable SQLite records. Worker processes claim queued jobs with a
`worker_id`, set `locked_at`, `heartbeat_at`, and `lease_expires_at`, and keep
the lease alive while long work is running. If a worker dies, another worker can
return expired leases to the queue.

Failure handling:

- Worker failures call `fail_job(..., auto_retry=True)`.
- Retryable jobs are returned to `queued` with `next_run_at` set by exponential
  backoff.
- Final failures clear the worker lease fields and set `status='failed'`.
- Parse, index, and translation paper statuses are mirrored back to `papers`.

## External API Contract

LLM, embedding, and VLM clients share `src.external_call.RetryPolicy`.

Default retry settings are configured by:

```env
EXTERNAL_API_MAX_ATTEMPTS=2
EXTERNAL_API_RETRY_BASE_SECONDS=1
EXTERNAL_API_RETRY_MAX_SECONDS=8
```

Retry policy is intentionally bounded. It is meant to smooth transient 429/5xx
or network failures, not to hide persistent configuration errors.

## Verification

Run the backend smoke gate before changing core backend behavior:

```bash
python scripts/verify_backend.py
```

The smoke gate includes `scripts/test_architecture_boundaries.py`, which checks
that backend modules do not import Streamlit and that `app.py` does not directly
import selected low-level workflow functions that belong behind the application
layer.

Run retrieval-focused local checks after changing `src/retrieval/`, chunking,
index text, reranking, or context building:

```bash
python scripts/verify_backend.py --suite retrieval
```

Run all local non-network checks before a larger refactor:

```bash
python scripts/verify_backend.py --suite all
```

Run a local deployment diagnostic before troubleshooting startup, queue, or
configuration issues:

```bash
python scripts/doctor_backend.py
python scripts/doctor_backend.py --json
```

The doctor checks runtime directory writeability, SQLite schema and pragmas,
queue state, supported providers, and whether required secret-bearing settings
are present. It only reports whether keys are configured, never the key values.

For real indexed papers, run retrieval evaluation with a JSONL seed set:

```bash
python scripts/eval_retrieval.py data/retrieval_seed.jsonl --disable-llm-rerank
```

Network/API smoke checks such as `scripts/test_deepseek.py` and
`scripts/test_mineru.py` are intentionally not part of the default backend gate.

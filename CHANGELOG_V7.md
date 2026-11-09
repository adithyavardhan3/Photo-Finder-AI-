# Photo Finder AI v7 — targeted indexing update

## Changes in this package
- Preserves the existing PySide6 UI and YuNet + SFace embedding search architecture.
- Recursive scan recognizes JPEG, PNG, BMP, TIFF, WebP and common RAW extensions (CR2, CR3, NEF, ARW, DNG, ORF, RW2, RAF, PEF, SRW).
- Optional RAW decoding via `rawpy`; RAW files are reported as errors and the scan continues if rawpy is missing or a RAW file is damaged.
- Incremental index skips files whose size, modification timestamp and status are unchanged, including successfully processed photos with zero detected faces.
- Adds a visible “Scan for New Photos” action that uses the same incremental scan path.
- Adds year, customer-folder, containing-folder and file-type metadata columns to SQLite with additive migrations.
- Keeps embedding persistence and batched commits from v6; the UI indexing worker remains a background QThread.
- Adds scanner regression test for common RAW extensions.

## Validation performed in this environment
- `python -m compileall -q .` passed.
- `PYTHONPATH=. pytest -q` passed: 2 scanner tests.
- No 50,000-photo benchmark or end-to-end GUI/model test was run in this environment.

## Known limitations / not yet implemented
- RAW decoding is optional and requires installing `rawpy`; embedded preview extraction is not implemented.
- The scanner currently builds a path list to show determinate progress; for very large libraries this uses memory proportional to file count.
- Year/customer metadata is inferred from path components; it is not a dedicated interactive filter yet.
- No filesystem watcher, content hash verification, deleted-file reconciliation, similarity-based RAW/edited grouping, match feedback UI, or benchmark harness has been implemented in this update.
- The existing face search still loads all embeddings into memory for exhaustive cosine search; FAISS has not been adopted or benchmark-justified.
- Model weights are downloaded on first use unless already cached; for a fully offline deployment, place verified model files in `~/PhotoFinderAI/models` before disconnecting.
- Commercial redistribution rights for the exact model weights and dependencies must be reviewed before commercial release.

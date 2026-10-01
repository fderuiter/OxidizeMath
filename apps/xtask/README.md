# Automation Runner (xtask)

This directory contains the workspace automation runner, which is responsible for executing various quality gates and legacy utility tasks.

## Commands

### Active Quality Gates
These commands run automated checks in git hooks or CI.

| Command | Operational Role | Status |
|---|---|---|
| `check-file-lengths` | Verifies file-length constraints. | Active |
| `check-staged-duplicates` | Checks for duplicated files in staging. | Active |
| `lint-all` | Runs formatting, clippy, file-length, and staged duplicate checks. | Active |
| `test-features` | Runs tests for various cargo features. | Active |
| `traceability` | Generates and checks the traceability report. | Active |
| `verify-records` | Verifies structural records. | Active |
| `verify-suite` | Runs the high-integrity verification suite. | Active |

### Legacy Utility Tasks
These commands are used for manual repository maintenance or legacy workflows.

| Command | Operational Role | Status |
|---|---|---|
| `compile-papers` | Compiles academic papers using Tectonic. | Legacy |
| `regenerate-baseline` | Regenerates the public API baseline. | Legacy |
| `setup` | Installs the pre-commit hook and runs initial checks. | Legacy |

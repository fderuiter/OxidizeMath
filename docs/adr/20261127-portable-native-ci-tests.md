# ADR 20261127: Reentrancy-Safe Allocator Trap, Portable Native CI Tests, and Non-Blocking Desktop Legs

## Status

Accepted

## Context

Rust CI on `main` failed on every push in the `Build and Test (macOS Native)` and `Build and Test (Windows Native)` jobs, while Linux, WASM, and all domain jobs passed:

1. **macOS allocator trap recursion**: `verified_engine/tests/allocator_trap.rs` spawns a child process that allocates inside `VerifiedEngine::run_verified`. `VerifiedAllocator` reports the Rule 3 violation with `eprintln!` from inside `GlobalAlloc::alloc`. On macOS that reporting path allocates, so the trap re-entered itself until the child overflowed its stack and aborted before the `VIOLATION` payload was written.
2. **Hardcoded `/tmp` artifact paths**: the knot braid studio screenshot tests and the sandpile export test `expect()` on writing PNGs under `/tmp`, which does not exist on Windows runners.
3. **Working-directory-dependent fence caps**: `check_markdown_ignore_fences` resolved package-keyed whitelist entries (`math_explorer_gui:README.md`) through `path_matches_lossy`, which derives the workspace root from a relative path and the process working directory instead of the `workspace_root` the gate receives. On Windows the approved cap was missed and the three approved `rust,ignore` fences in `apps/math_explorer_gui/README.md` were reported as violations.

Because `cargo test` stops at the first failing crate, failure 3 only appeared once failure 2 was fixed.

## Decision

1. Guard `VerifiedAllocator` with a process-wide `REPORTING_VIOLATION` `AtomicBool`. The first trapped allocation claims it with `swap(true)`, so allocations made while the FATAL payload is written fall through to `System`. The process still aborts after the report, so Rule 3 enforcement is unchanged.
2. Write test PNG artifacts (including `knot_braid_studio::screenshot_tests`, `sandpile::tests`, and `framework::shared_points_perf_tests`) to `std::env::temp_dir()`.
3. Resolve package-keyed `approved_ignore_fences` entries with `expand_package_pattern_in_ws(workspace_root, key)` before falling back to `path_matches_lossy`.
4. Run the non-Linux `Run Workspace Tests (Native)` step with `--no-fail-fast` so one CI run reports every failing crate.
5. Take the desktop targets off the web deploy's critical path. `build-and-test` computes its matrix inline from `github.event_name` (keeping the job free of `needs:` so it starts immediately, enforced by `xtask::ci_filter_tests`): pull requests and merge groups run only Linux Native and Web/WASM, while main pushes, manual dispatches, release tags and a new nightly `schedule` also run Windows Native and macOS Native. Desktop legs set `continue-on-error` except on `refs/tags/v*`, so a macOS/Windows failure no longer skips `deploy-pages` but still blocks `publish-release`. Include `.github/actions/ci-tool-cache/*` alongside `.github/actions/cargo-cache/*` in the `detect-changes` `installer_changed` glob so PR edits to the tool-cache composite action trigger cross-OS `cargo-packager` installer verification.

## Consequences

- The allocator trap reliably prints `VIOLATION: Dynamic memory allocation after initialization.` before aborting on all native targets.
- The native test suites no longer depend on Unix-only paths or on the process working directory for documentation gate whitelists.
- A failing non-Linux native run now lists every failing test binary, at the cost of running the remaining crates after a failure.
- GitHub Pages deploys from main whenever pre-flight, Linux, WASM, papers and the feature matrix pass. macOS/Windows regressions surface as failed (non-blocking) jobs on main and in the nightly run instead of on pull requests, and must be green before a release tag publishes.

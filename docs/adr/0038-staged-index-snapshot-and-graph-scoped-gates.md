# ADR 0038: Staged Git Index Snapshot Isolation and Dependency-Graph Pre-Commit Gates

## Status

Accepted (2026-11-22)

## Context

Following ADR 0034, `cargo run -p xtask -- lint-staged` accelerated local commits by scoping static checks and Clippy to paths in `git diff --cached`. However, auditing `lint-staged` against partial staging, nested directory execution, and multi-crate edits revealed five remaining correctness gaps:

1. **Working-tree vs. Git index skew:** `rustfmt`, `unified_verification`, `check-markdown-links`, and `cargo clippy` read files directly from the working tree rather than the staged Git index. Staging a bad edit and reverting the working tree allowed broken code or secrets to bypass `.githooks/pre-commit`, while leaving valid WIP unstaged edits on disk blocked clean partial commits (`git add -p`).
2. **Working-directory fragility:** `apps/xtask/src/staged.rs`, `apps/xtask/src/gates.rs`, and `crates/oxidize_core/src/workspace.rs` relied on relative paths (`Cargo.toml`, `../../Cargo.toml`, `target/debug/unified_verification`), causing divergent behavior when invoked from nested crate directories such as `crates/pure_math` or `math_explorer_gui`.
3. **Static 3-crate web list and missing reverse dependents:** `WEB_PACKAGE_NAMES` hardcoded only three crates (`oxidize_core`, `egui_plot`, `math_explorer_gui`) instead of the 17 workspace crates compiled into `math_explorer_gui` on `wasm32-unknown-unknown`, and `lint-staged` did not expand modified library crates to reverse dependents or escalate root configuration edits (`Cargo.toml`, `Cargo.lock`, `rust-toolchain*`, `.cargo/*`, `.clippy.toml`, `.rustfmt.toml`, `verification_whitelist.toml`) to full-workspace checks.
4. **Incomplete offline static coverage on staged commits:** `batch-staged` skipped `check-profiles`, `check-docs`, AST clone detection (`check-duplicates`), and citation/module `traceability`, and `check-markdown-links` skipped validation when commits deleted or renamed tracked documentation targets without editing `.md` files.
5. **Duplicate workflow lint dispatch:** Staging `.github/` files triggered `lint-workflows` twice—once inside `unified_verification batch-staged` and a second time via `xtask lint-staged`.

## Decision

1. **Workspace-Root Anchoring Across `oxidize_core`, `unified_verification`, and `xtask`:**
   - Add bounded ancestor discovery (`find_workspace_root` and `find_workspace_root_from`) in `crates/oxidize_core/src/workspace.rs` and anchor `get_workspace_members`, traceability runners, `unified_verification` binary lookup, and `xtask` staged runners to the workspace root.

2. **Reusable Staged Git Index Snapshot (`<target_dir>/lint-staged-snapshot`):**
   - Materialize exact staged index blobs via `git --no-optional-locks ls-files --stage -z` and chunked `git --no-optional-locks cat-file --batch` into `<target_dir>/lint-staged-snapshot` without modifying, stashing, or locking the developer working tree or `.git/index`.
   - Persist `<target_dir>/lint-staged-snapshot.manifest` (`oid`, `mode`, `len`, `mtime`) so unchanged files preserve filesystem timestamps across runs, prune deleted/untracked files from the snapshot, forward `CARGO_TARGET_DIR` to share warm Cargo build artifacts, and resolve `GIT_DIR` / `GIT_INDEX_FILE` to absolute paths when executing verification commands inside the snapshot directory.

3. **Dynamic Dependency-Graph Expansion and Root Config Escalation:**
   - Parse workspace `Cargo.toml` manifests (`apps/xtask/src/staged/graph.rs`) to dynamically compute all 17 web-compiled workspace crates reachable from `math_explorer_gui/Cargo.toml` and expand modified crates across transitive normal/build reverse dependents plus direct dev-dependents.
   - Escalate staged edits to root manifests, lockfiles, toolchain configs, or lint configs to workspace-wide `HOST_CLIPPY_ARGS` and `WASM_CLIPPY_ARGS` from `apps/xtask/src/gates.rs`.

4. **Expanded Offline Static Gates and Single Workflow Dispatch:**
   - Expand `unified_verification batch-staged` to enforce `verify-records --staged`, `check-profiles`, `check-docs`, single-parse `AstCache` clone detection (`check-duplicates`), and in-process `traceability` alongside file-length, entropy, and policy-audit checks while keeping `check-vulnerabilities` and `check-api-drift` deferred to `xtask lint-all`.
   - Escalate `xtask` Markdown link validation to scan all repository `.md` files in the snapshot whenever tracked paths are deleted or renamed, and execute `lint-workflows` once from `xtask` when `.github/` paths are staged (removing the duplicate call in `crates/unified_verification/src/batch.rs`).

## Verification Commands

```bash
# Run snapshot-backed pre-commit gates from workspace root or any nested crate
cargo run -p xtask -- lint-staged

# Run unit and integration tests for workspace discovery, batch-staged, and staged planning
cargo test -p oxidize_core -p unified_verification -p xtask

# Run full PR-readiness verification suite
cargo run -p xtask -- lint-all
```

## Consequences

- **Positive:** Pre-commit verification evaluates the exact staged commit snapshot regardless of unstaged working-tree edits or current working directory, reuses warm Cargo artifacts across runs, catches reverse-dependent and `wasm32` breakages across all 17 web crates, and blocks profile, documentation, clone, traceability, and broken-link regressions offline before commit creation.
- **Positive:** Eliminates duplicate `.github/` workflow linting during pre-commit hooks while preserving sub-minute incremental commit ergonomics by keeping OSV network checks and API baseline drift in `xtask lint-all`.
- **Neutral:** `<target_dir>/lint-staged-snapshot` maintains a lightweight checkout-sized mirror inside Cargo's gitignored `target/` directory.

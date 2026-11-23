# ADR 0047: Precompiled Verification Patterns, AST Source Prefetch, and Zero-Allocation Scans

## Status

Accepted

## Context

`unified_verification` runs across pre-commit hooks and CI gates (`batch-static` and `batch-ast`) to audit workspace documentation, public API facade parity, entropy/secret hygiene, and Cargo policy compliance. Profiling warm runs of `unified_verification` surfaced four avoidable bottlenecks:

1. `doc_audit` and `doc_audit/facade_imports` recompiled identical regular expressions inside per-crate and per-file loops rather than sharing a scoped pattern struct like `ReportPatterns` in `crates/unified_verification/src/report.rs`.
2. `api_drift/facade` and `api_drift/facade_errors` re-parsed public item signatures across three separate passes while probing all four candidate suffixes blindly, and cold `AstCache` walks read source files sequentially on a single thread.
3. `entropy_guard` scanned the ~340 KB machine-generated `crates/unified_verification/api_baseline.json`, allocated heap `HashMap<char, usize>` tables and lowercase copies per string literal, and invoked `Path::canonicalize()` for every regular file before checking the allowlist.
4. `policy_audit` allocated `Ident::to_string()` on every visited identifier in `DependencyVisitor` and `DeadCodeVisitor`, cloned `DeadCodeEntry` vectors per crate, and extracted/parsed Rust doctests even after all declared dependencies in `Cargo.toml` had already matched in the crate AST.

Per [ADR 0034](0034-scoped-pre-commit-and-single-parse-ast-verification.md), `AstCache` must preserve single-parse AST semantics on the owner thread because `syn::File` spans (`proc_macro2::Span`) are `!Send`.

## Decision

1. **Scoped Regex Structs in `doc_audit`**:
   - Introduce `DocAuditPatterns` in [`crates/unified_verification/src/doc_audit.rs`](../../crates/unified_verification/src/doc_audit.rs) and `FacadeImportPatterns` in [`crates/unified_verification/src/doc_audit/facade_imports.rs`](../../crates/unified_verification/src/doc_audit/facade_imports.rs), compiling regexes once per audit pass and threading borrowed `&Regex` references into inner helpers while keeping public wrapper functions intact.
2. **Prefix-Filtered Signature Cache and Scoped `AstCache` Source Prefetch**:
   - Extract [`crates/unified_verification/src/api_drift/facade_signatures.rs`](../../crates/unified_verification/src/api_drift/facade_signatures.rs) to select candidate completion suffixes by signature prefix (`pub fn`, `pub struct`, `pub const`, `pub union`, `pub trait`, `impl`, `pub use`, `pub type`) and build a shared `ParsedSignatureMap` once per `check_facade_completeness_in_dir_with_cache` invocation across [`facade.rs`](../../crates/unified_verification/src/api_drift/facade.rs) and [`facade_errors.rs`](../../crates/unified_verification/src/api_drift/facade_errors.rs).
   - Add `AstCache::preload_sources` in [`crates/oxidize_core/src/traceability/cache.rs`](../../crates/oxidize_core/src/traceability/cache.rs) using `std::thread::scope` worker chunks on native targets (`#[cfg(not(target_arch = "wasm32"))]`) and `HashMap::entry` single-lookup insertion, keeping `syn::parse_file` execution strictly on the cache owner thread and invoking `preload_sources` from [`crates/unified_verification/src/ast_visitor.rs`](../../crates/unified_verification/src/ast_visitor.rs).
3. **Allowlist-First Syscall Reduction and Stack-Allocated Entropy Checks**:
   - Add `crates/unified_verification/api_baseline.json` to `[entropy_guard].allowlist` in [`verification_whitelist.toml`](../../verification_whitelist.toml), check allowlists before filesystem canonicalization, and canonicalize each scan root once in [`crates/unified_verification/src/entropy_guard/file_filter.rs`](../../crates/unified_verification/src/entropy_guard/file_filter.rs) so regular files avoid per-file `canonicalize()` syscalls while symlinks still canonicalize for deduplication.
   - Replace `.to_lowercase()` in `is_integrity_digest` with `eq_ignore_ascii_case` and compute Shannon entropy for ASCII tokens via a `[0u32; 128]` stack frequency table in [`crates/unified_verification/src/entropy_guard.rs`](../../crates/unified_verification/src/entropy_guard.rs) with UTF-8 `HashMap` fallback.
4. **Zero-Allocation Identifier Matching and Doctest Short-Circuiting in `policy_audit`**:
   - Match `syn::Ident` directly against `&str` (`ident == dep.as_str()`) in [`crates/unified_verification/src/policy_audit/dependencies.rs`](../../crates/unified_verification/src/policy_audit/dependencies.rs), borrow `&[DeadCodeEntry]` slices in [`crates/unified_verification/src/policy_audit.rs`](../../crates/unified_verification/src/policy_audit.rs), and skip [`crates/unified_verification/src/policy_audit/doctests.rs`](../../crates/unified_verification/src/policy_audit/doctests.rs) once all declared dependencies are accounted for.

## Consequences

- **Positive**: Substantially reduces wall-clock runtime across `batch-static`, `check-docs`, `check-api-drift`, `check-entropy`, `test-policy-audit`, and `batch-ast` without weakening any verification invariant or increasing `AstCache::parse_calls()`.
- **Positive**: Keeps `oxidize_core` compatible with `wasm32-unknown-unknown` via a sequential `#[cfg(target_arch = "wasm32")]` fallback in `AstCache::preload_sources` and avoids `unsafe` or `!Send` cross-thread AST sharing.
- **Trade-off**: Adding `AstCache::preload_sources` extends the public API of `oxidize_core::traceability::AstCache`, requiring a synchronized regeneration of `crates/unified_verification/api_baseline.json` via `cargo run -p xtask -- regenerate-baseline`.

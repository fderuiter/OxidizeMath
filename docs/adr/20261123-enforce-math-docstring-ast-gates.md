# ADR 20261123: Enforce Single-Pass AST Gates for 4-Part Math Docstrings and Retire Library `missing_docs` Caps

## Status

Accepted (extends [ADR 0005](0005-active-documentation-standard.md), [ADR 0013](0013-scientific-metadata-documentation-standard.md), and [ADR 0044](0044-ratchet-missing-docs-and-core-api-documentation.md))

## Context

1. **Grandfathered Library `missing_docs` Exemptions**: Following [ADR 0044](0044-ratchet-missing-docs-and-core-api-documentation.md), `[missing_docs.files]` in [`verification_whitelist.toml`](../../verification_whitelist.toml) still exempted `176` library files (`985` outer `#[allow(missing_docs)]` attributes, covering `160` public files in `api_baseline.json`), while `audit_file_missing_docs` allowed item-level `#[allow(missing_docs)]` on structs, enums, traits, fields, and functions whenever a file cap was present.
2. **Unverified Mathematical Preconditions in IDE Tooltips**: Neither `unified_verification check-docs` nor `batch-static` / `batch-staged` verified that public mathematical models and solvers across the `8` mathematical crates (`math_commons`, `pure_math`, `domain_ai`, `domain_applied`, `domain_biology`, `domain_climate`, `domain_epidemiology`, `domain_physics`) document governing equations, literature citations, numerical stability bounds, and validity domains.
3. **500-Line Source Ceiling vs. Rich Mathematical Docstrings**: Near-limit library modules (`465–499` lines) could not inline multi-section LaTeX specifications and runnable `math_explorer::` doctests without breaching `check_file_lengths`.

## Decision

1. **Canonical Mathematical Specifications (`docs/math_specs/`)**:
   - Author canonical specifications under [`docs/math_specs/`](../math_specs/README.md) for all `8` mathematical crates anchored to BibTeX keys in [`papers/references.bib`](../../papers/references.bib) and manuscripts in `papers/`, alongside external `#[doc = include_str!("...")]` snippets under `docs/math_specs/snippets/` for near-limit Rust modules.
2. **Single-Pass 4-Part Math Contract AST Gate (`doc_audit::math_contracts`)**:
   - Add a `syn` AST verification pass in `crates/unified_verification/src/doc_audit/math_contracts.rs` sharing `AstCache` across `check-docs`, `batch-ast`, `batch-static`, and `batch-staged`.
   - Resolve `///`, `#[cfg_attr(..., doc = ...)]`, and `#[doc = include_str!("...")]` docstrings and fail closed with exact `file:line` diagnostics whenever any public mathematical solver or model omits:
     1. Governing `LaTeX` equation (`$...$` / `$$...$$`) or `#[embed_theory(...)]`,
     2. Peer-reviewed `DOI`/`arXiv` identifier or `docs/math_specs/` citation,
     3. `# Stability` section,
     4. `# Domain of Validity` section.
3. **Reject Item-Level `missing_docs` Bypasses in Library Crates & Retire All 176 Library Caps**:
   - Update `crates/unified_verification/src/policy_audit/missing_docs.rs` to reject item-level `#[allow(missing_docs)]` / `#[expect(missing_docs)]` attributes on structs, enums, variants, traits, fields, and functions across library crates per `CONTRIBUTING.md`, leaving binary-only `math_explorer_gui` files under existing caps.
   - Document all `176` whitelisted library files with runnable facade doctests and remove all `985` library caps from `[missing_docs.files]` in `verification_whitelist.toml`.
4. **Cross-Artifact Markdown Link Verification (`xtask check-markdown-links`)**:
   - Extend `apps/xtask/src/markdown_links.rs` to verify relative file links inside Rust docstrings (`///` / `//!`), `#[doc = include_str!("...")]` targets, `docs/math_specs/`, and `papers/references.bib#<key>` BibTeX fragment anchors.

## Consequences

- **Positive**: `0` library files remain under `[missing_docs.files]`; IDE hover tooltips across all `8` mathematical crates display complete equations, citations, stability bounds, validity domains, and runnable `math_explorer::` examples; `.githooks/pre-commit`, `xtask check-docs`, and `xtask lint-all` fail closed on missing contract sections or broken cross-artifact links while keeping every Rust source file at or below `500` lines.
- **Neutral**: Public signatures in `crates/unified_verification/api_baseline.json`, zero-allocation `VerifiedAllocator` hot loops, and binary-only `math_explorer_gui` caps remain unchanged.

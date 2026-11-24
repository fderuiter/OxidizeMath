# ADR 20261123: Ratcheted AST Gate for Mathematical Rustdoc, Domain Specs, and BibTeX Citations

## Status

Accepted

## Date

2026-11-23

## Context

Prior to this decision, three verification gaps weakened theory-to-code provenance across the 8 mathematical crates (`math_commons`, `pure_math`, `domain_ai`, `domain_applied`, `domain_biology`, `domain_climate`, `domain_epidemiology`, and `domain_physics`):

1. **Mislinked Domain Registry Entries and Unverified `spec:` Targets**: Nineteen `[links]` entries in `traceability.toml` mapped domains (including `biology`, `climate`, `epidemiology`, `fluid_dynamics`, `quantum`, and `battery_degradation`) to unrelated `.tex` manuscripts simply to satisfy orphan checks, while `spec:` entries such as `spec:transformer_v1` and `spec:wave_optics_caustics_v1` bypassed file existence and bibliographic validation in `TraceabilityEngine`.
2. **Unchecked BibTeX Persistent Identifiers and Free-Form `#[derive(Theory)]` Citations**: `papers/references.bib` lacked machine-verified `doi` or `eprint` checks on cited keys, and `#[derive(Theory)]` in `crates/verified_engine_macros/src/theory.rs` accepted arbitrary prose strings in `#[theory(citation = "...")]` instead of verified BibTeX keys.
3. **No Structural Rustdoc Contract on Public Mathematical APIs**: While `[missing_docs.files]` ratcheted `#[allow(missing_docs)]` counts downward, public functions, structs, enums, traits, and methods in the 8 math crates could pass CI with one-line summaries omitting governing equations, verified citations, numerical stability / conditioning bounds, or domains of validity.

## Decision

1. **Verified Domain Specifications (`docs/math_specs/`) and Registry Rebinding**:
   - Establish `docs/math_specs/*.md` containing 22 domain specifications (`transformer_v1`, `wave_optics_caustics_v1`, `pure_math_v1`, and the 19 rebound domain specs), each documenting governing LaTeX equations, numerical stability / error bounds, domain of validity, and verified BibTeX citations (`@<key>` / `bib:<key>`).
   - Rebind the 19 mislinked `.tex` entries in `traceability.toml` to their corresponding `spec:<domain>_v1` specifications while retaining all 29 `.tex` manuscripts on their genuine domain links.
2. **Zero-Dependency BibTeX Parser and `TraceabilityEngine` Spec Verification (`oxidize_core::bibtex`)**:
   - Add `oxidize_core::bibtex` (`BibtexDatabase`, `BibtexEntry`, `extract_tex_cite_keys`, `extract_doc_bib_keys`, `extract_doc_spec_ids`, and `validate_math_spec_markdown`) requiring every cited BibTeX entry in `papers/references.bib` and `docs/math_specs/*.md` to exist and declare a non-empty `doi` or `eprint` identifier.
   - Extend `TraceabilityEngine` (`crates/oxidize_core/src/traceability/spec_verify.rs`) to validate `docs/math_specs/<id>.md` files and cited BibTeX entries during workspace, build-script, and subdirectory scans.
3. **Compile-Time `#[derive(Theory)]` BibTeX Validation**:
   - Share `oxidize_core/src/bibtex/parser.rs` inside `crates/verified_engine_macros/src/bibtex_check.rs` (caching parsed `papers/references.bib` entries in a `OnceLock`) so `#[derive(Theory)]` rejects unknown, prose, or `doi`/`eprint`-missing `#[theory(citation = "...")]` attributes with span-accurate compile errors while preserving `TheoryDescribable` trait signatures in `scientific_metadata`.
   - Migrate all 15 production `#[derive(Theory)]` models to verified BibTeX keys in `papers/references.bib`.
4. **Single-Pass AST 4-Part Math Rustdoc Gate and Downward-Only Ratchet**:
   - Add `crates/unified_verification/src/math_doc_contract.rs` visiting public `fn`, `struct`, `enum`, `trait`, trait methods, and inherent `pub fn` items across the 8 math crates while reusing `AstCache` so every `.rs` file in the workspace is parsed by `syn::parse_file` at most once.
   - Require all 4 contract parts on new or migrated public math items: (1) LaTeX equation (`$$...$$`, `$...$`, or `#[embed_theory(...)]`), (2) verified BibTeX citation key (`@<key>` / `bib:<key>`) with `doi`/`eprint` plus verified `spec:<id>` or `papers/<name>.tex` target, (3) numerical stability / conditioning / error bounds, and (4) domain of validity / preconditions.
   - Enforce downward-only per-file legacy caps (`>= 1`) under `[math_doc_contract.files]` in `verification_whitelist.toml` (with all 15 migrated/pilot files across all 8 math crates held at cap `0`), wired into `check-docs`, `check-math-docs`, `batch-static`, `lint-all`, `test-policy-audit`, and `lint-staged`.
5. **Traceability Portal & Theory Context Portal `spec:` Rendering (`math_explorer_gui`)**:
   - Update `TraceabilityTab` (`math_explorer_gui/src/tabs/traceability.rs` and `web_snapshot.rs`) and `theory_links` (`math_explorer_gui/src/framework/theory_links.rs`) to render verified `docs/math_specs/<id>.md` specification links and resolve exact BibTeX citation keys to DOI/arXiv hyperlinks without false orphan errors or raw `spec:` token leakage.

## Consequences

- Every public item added to the 8 mathematical crates, every `#[derive(Theory)]` model, every `docs/math_specs/*.md` specification, and every `.tex` manuscript citation is verified against `papers/references.bib` (`doi` or `eprint`) in a single `AstCache` pass without adding external dependencies to `apps/xtask` or `verified_engine_macros`.
- Legacy math files cannot accrue new undocumented or uncited public items (`actual > cap` fails) and must lower or remove their `[math_doc_contract.files]` entry whenever violations are remediated (`actual < cap` fails).

# 11. Unified AST Citation Binding, Macro Expansion, and CLI Enforcement

## Status
Accepted

## Context
Theoretical mathematical models across domain crates required unified citation binding, AST attribute traversal for function impls and traits, and strict enforcement of traceability metrics in CLI verification tools.

## Decision
1. Expanded `AstVisitor` in `oxidize_core` to traverse `syn::ImplItemFn` and `syn::TraitItemFn` in addition to free functions, enabling comprehensive citation and attribute auditing across impl blocks and traits.
2. Refactored function attribute parsing into `process_fn_attrs` to parse doc citations (`/// [cite:model]`) and macro attributes (`#[verified]`, `#[embed_theory]`, `verified(opt_out)`).
3. Added missing doc citations (`// [cite:model_name]`) to verified domain source files to achieve 100% semantic integrity verification for theoretical models.
4. Updated `traceability_cli.rs` to print dashboard metrics prior to error evaluation and fail hard on any unverified semantic status or assertion density deficits.

## Consequences
- Impl and trait methods are audited for theoretical citation binding and verification attributes.
- Traceability CLI enforces 100% semantic integrity and minimum assertion density across workspace crates.

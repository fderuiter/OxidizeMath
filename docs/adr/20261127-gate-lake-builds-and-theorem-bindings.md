# ADR 20261127: Gate Mandatory Lake Builds, Scaled-Rational Lean 4 Proofs, and Certified Theorem Bindings

## Status

Accepted

## Context

Prior to this decision, `Tier6_Formal` verification in OxidizeMath computed SHA-256 digests and scanned `.lean` source text for `sorry`/`admit`/`axiom` keywords without invoking `lake build` or inspecting the compiled Lean 4 kernel environment. As a consequence:

1. Four of the six `.lean` files under `proofs/OxidizeMath/` contained latent Lean `4.14.0` syntax and tactic errors (`end`, `by exact ...; ...`, and linear `omega` obligations), and all six modeled invariants over plain integers rather than scaled rationals with IEEE-754 significand rounding slack.
2. `formal_proof_verification!` only checked file existence rather than binding specific compiled Lean theorem identifiers to matching Rust modules (leaving `bessel.rs` misbound to `IntervalSoundness.lean` while `CopulaFrechet.lean` and `SymplecticComposition.lean` lacked module bindings).
3. `MaturityTier::Certified` in `traceability.toml` did not require `Tier6_Formal` theorem bindings, `ReportMode::Check` (`check-certificate --check` / `batch-static`) skipped deterministic and pyramid parity comparisons while retaining an `OXIDIZE_VERIFY_SUITE_WRITE` environment-variable bypass, and `TraceabilityTab` displayed unconditional `sorry-free` labels.

## Decision

1. **Scaled-Rational & Rounding-Slack Lean 4.14.0 Proofs (`proofs/OxidizeMath/*.lean`)**:
   - Reformulate all six Lean 4 modules (`IntervalSoundness.lean`, `CompartmentalConservation.lean`, `CopulaFrechet.lean`, `QuantumUnitarity.lean`, `SurvivalBounds.lean`, and `SymplecticComposition.lean`) over exact scaled rationals (`sig / S` at positive IEEE-754 significand scale `0 < S`) and explicit outward-rounded / discrete step rounding slack using core Lean `4.14.0` (`Init`, zero `Mathlib4` external dependencies).
2. **Mandatory `lake build` & Kernel Axiom Inspection (`scientific_verification::lean_proofs`)**:
   - Require `lake build` in `proofs/` and inspect compiled `.olean` module constants via `lake env lean --stdin` (`Lean.collectAxioms`, `Lean.isExtern`, `Lean.Compiler.getImplementedBy?`).
   - Fail closed whenever `lake` is absent, `lake build` exits non-zero, or any proof introduces `sorryAx`, user `axiom`/`opaque` declarations, `@[extern]`, or `@[implemented_by]` outside standard Lean axioms (`propext`, `Quot.sound`, `Classical.choice`).
3. **AST Theorem Binding Extraction & `Certified` Qualification Gate**:
   - Extend `formal_proof_verification!` (`pure_math` and `math_commons`) and `oxidize_core::ast_visitor` to require and extract `proof = "..."` and `theorems = ["...", ...]`.
   - Rebind `IntervalSoundness.lean` to `math_commons`, `CopulaFrechet.lean` to `clinical_trials_statistics`, and `SymplecticComposition.lean` to `pure_math`, while replacing the misplaced `IntervalSoundness.lean` macro in `bessel.rs` with a Bessel enclosure check.
   - Require `Tier6_Formal` theorem bindings to compiled `.olean` theorems for every `MaturityTier::Certified` crate and module in `oxidize_core::traceability::qualification` and `unified_verification check-maturity`, demoting unproven `Certified` targets (`domain_biology`, `engine_opt`, and 14 unproven modules) to `Validated` in `traceability.toml`.
4. **Certificate Parity & `TraceabilityTab` Surface**:
   - Extend `VerificationCertificate` and `IntegrityReport` to a 15-key schema adding `lean4_proof_bindings`, run deterministic AST and pyramid/proof-binding parity checks inside `ReportMode::Check`, and remove `OXIDIZE_VERIFY_SUITE_WRITE`.
   - Install cached `elan` (`leanprover/lean4:v4.14.0`) and run `lake build` in `.github/workflows/ci.yml` (`pre-flight` and native `build-and-test`), and add `elan`/`lean`/`lake` toolchain probes to `xtask setup`.
   - Replace unconditional `sorry-free` labels in `TraceabilityTab` with WCAG 2.2 AA `[Kernel Verified]` / `[Unverified]` badges, linked Rust modules, and expandable theorem bindings.

## Consequences

- `lake build` in `proofs/` compiles cleanly in ~3 seconds under `leanprover/lean4:v4.14.0` with zero external package downloads and zero unproven axioms.
- Tampering with any `.lean` file, theorem binding, or `Verification_Certificate.json` field causes `verify-tier6`, `check-maturity`, `batch-static`, and `check-certificate --check` to fail closed.
- Every `Certified` crate and module in `traceability.toml` is backed by compiled Lean 4 kernel theorems surfaced transparently in `TraceabilityTab`.

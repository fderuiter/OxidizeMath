# ADR 0028: Finite Survival Time Invariant and Total-Order Kaplan-Meier Estimation

## Status
Accepted

## Context
Clinical trials survival analysis exhibited three robustness hazards across `crates/domain_applied` and `math_explorer_gui`:
1. **Non-Finite `SurvivalTime` Construction**: `SurvivalTime::new` in `crates/domain_applied/src/applied/clinical_trials/types.rs` checked `t.is_nan() || t < 0.0`, accepting `f64::INFINITY` (such as `"inf, 1"` parsed via `str::parse::<f64>()` in preset imports or interactive text input, or `SurvivalTime::exp` overflow) and leaving `-0.0` unnormalized.
2. **Intransitive Comparator and Non-Terminating Tie Loop in `kaplan_meier`**: `kaplan_meier` in `crates/domain_applied/src/applied/clinical_trials/survival_analysis.rs` passed an epsilon comparator (`(ta - tb).abs() < TOLERANCE_STANDARD => Ordering::Equal`) to `slice::sort_unstable_by`. Because epsilon equivalence is not transitive (e.g., `0.0`, `6e-10`, and `1.2e-9` with `TOLERANCE_STANDARD = 1e-9`), Rust's standard library sort panicked on near-tie inputs. Furthermore, the step-grouping loop relied on `(obs[i].time - t).abs() < TOLERANCE_STANDARD` to increment `i`; when `t` was `f64::INFINITY`, `inf - inf` evaluated to `NaN`, stalling `i` in an infinite `Vec::push` loop.
3. **Unbounded GUI Observation Input**: `SurvivalAnalysisTool::recalculate` in `math_explorer_gui/src/tabs/clinical_trials/survival.rs` parsed multi-line text and JSON presets without capping observation row count.

## Decision
1. **Strict Finite Validation and `-0.0` Normalization in `SurvivalTime::new`**: Update `SurvivalTime::new` to reject `!t.is_finite() || t < 0.0` with `ClinicalTrialError::InvalidData` and canonicalize `-0.0` to `+0.0`. Because all `SurvivalTime` arithmetic methods delegate to `Self::new`, arithmetic overflow to `±Infinity` or `NaN` now deterministically returns `Err`.
2. **Total-Order Sorting and Unconditional Loop Progress in `kaplan_meier`**: Sort observations with `a.time.as_f64().total_cmp(&b.time.as_f64())` to guarantee a strict total order for `sort_unstable_by`. Record the anchor observation `obs[i]` and increment `i` unconditionally before grouping subsequent observations within `math_commons::registry::TOLERANCE_STANDARD`, ensuring monotonic loop termination in bounded iterations (NASA Power of 10 Rule 2).
3. **Bounded Observation Cap in `SurvivalAnalysisTool`**: Enforce `math_commons::registry::MAX_ITERATIONS` (`5000`) on non-empty observation rows in `SurvivalAnalysisTool::recalculate`, clearing partial state and populating `error_message` when exceeded so interactive edits, session loads, and preset imports fail safely.

## Consequences
- **Positive**: Infinite, `NaN`, negative, and overflowing survival times are rejected at the type boundary, near-tie datasets sort without panics, `kaplan_meier` guarantees strict index progress on every outer iteration, and `SurvivalAnalysisTool` rejects oversized or non-finite presets cleanly.
- **Neutral**: All public signatures in `crates/unified_verification/api_baseline.json` and survival probability curves for valid finite datasets and starter presets remain unchanged.

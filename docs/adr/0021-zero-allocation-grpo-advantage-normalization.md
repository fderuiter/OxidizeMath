# 21. Zero-Allocation Slice Advantage Normalization in GRPO

## Status
Accepted

## Context
In `crates/domain_applied/src/applied/grpo/formulas.rs`, `response_level_advantage` evaluated the z-score advantage of an individual response by converting `rewards: &[f64]` to `statrs::statistics::Data` via `rewards.to_vec()`. This caused a dynamic heap allocation on every single response evaluation. When calculating advantages across a group of $G$ responses (such as during rollout evaluations in reinforcement learning and in `math_explorer/examples/grpo_demo.rs`), callers repeatedly invoked `response_level_advantage`, causing $G$ heap allocations and $O(G^2)$ redundant statistical passes over the reward group.

## Decision
1. Refactored `response_level_advantage` to compute sample statistics (mean and Bessel-corrected sample standard deviation $N-1$) directly over borrowed slices `&[f64]` without heap allocations.
2. Handled edge cases safely:
   - Empty slices return `0.0` advantage.
   - Single-element groups and constant reward groups produce `0.0` standard deviation and divide safely via `EPSILON_NORM`.
3. Added `normalize_group_advantages(rewards: &[f64], out: &mut [f64])` to normalize full groups into a pre-allocated buffer slice in one pass in $O(G)$ time with zero heap allocations.
4. Added batch helpers `batch_response_level_advantages(rewards: &[f64]) -> Vec<f64>` and alias `batch_group_advantages` for convenient $O(G)$ batch evaluations.
5. Updated `math_explorer/examples/grpo_demo.rs` to demonstrate zero-allocation group advantage normalization into a pre-allocated buffer.
6. Preserved the public function signature of `response_level_advantage` to maintain API stability and exact numerical parity.
7. Added unit and regression tests in `crates/domain_applied/tests/grpo.rs` verifying zero heap allocations (via `VerifiedAllocator` and `VerifiedEngine::run_verified`), edge cases, and numerical equivalence.

## Consequences
- Eliminates heap allocation in `response_level_advantage` and per-response repeated statistics recalculations in rollout loops.
- Provides an $O(G)$ zero-allocation batch normalization API for group relative policy optimization.
- Retains numerical stability and exact equivalence with previous z-score advantage calculations.
- Satisfies all high-integrity verification gates and API drift checks.

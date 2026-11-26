# ADR 20261123: Log-Space HMM Solvers, Zero-Allocation Baum-Welch EM, Validated Serde, and Markov Sequence Workbench

## Status
Accepted (extends [ADR 0022](0022-unify-preset-state-serialization.md), [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md), and [ADR 0037](0037-glicko2-frozen-period-tournament-simulator.md))

## Context
`pure_math::statistics::markov` provides discrete-time absorbing Markov chains (`MarkovChain`), continuous-time generator chains (`ContinuousMarkovChain`), and discrete hidden Markov models (`HiddenMarkovModel`). Four gaps blocked analytical sequence inference and interactive browser exploration:

1. **Numerical Underflow and Likelihood Corruption in `HiddenMarkovModel`**:
   - [`viterbi.rs`](../../crates/03_domains/pure_math/src/pure_math/statistics/markov/hmm/viterbi.rs) multiplied raw probabilities `v[(t - 1, i)] * A[(i, j)] * B[(j, y_t)]`, underflowing to `0.0` around ~500 steps on the 3-state market regime HMM (`markov_demo.rs`) and collapsing `argmax` ties to state `0`, while storing `usize` backpointers through `T::from_usize` / `to_usize`.
   - [`forward_backward.rs`](../../crates/03_domains/pure_math/src/pure_math/statistics/markov/hmm/forward_backward.rs) normalized each forward row `alpha.row(t)` to sum to `1.0` without accumulating per-step scaling factors `s_t`, causing `HiddenMarkovModel::forward` to return `1.0` for every reachable sequence.
2. **Missing Unsupervised Parameter Estimation**: `HiddenMarkovModel` had no Baum-Welch Expectation-Maximization solver or preallocated workspace API compatible with `VerifiedAllocator` zero-allocation verification.
3. **Missing Constructor-Validated Serialization**: `HiddenMarkovModel`, `ContinuousMarkovChain`, `MarkovChain`, and `StateType` lacked `serde::{Serialize, Deserialize}` implementations enforcing stochastic row sums, valid generator diagonals, non-empty dimensions, and `NaN`-free parameters.
4. **No Interactive GUI Workbench**: None of the Markov solvers were exposed in `math_explorer_gui` (`GameTheoryTab`), leaving users unable to decode observation sequences, inspect reliability curves `P(t) = exp(Gt)`, or compute expected possession values `EPV = B w` without running CLI examples.

## Decision
1. **Repair Log-Domain Viterbi and Scaled Forward Log-Likelihoods**:
   - Rewrite [`viterbi.rs`](../../crates/03_domains/pure_math/src/pure_math/statistics/markov/hmm/viterbi.rs) around `ViterbiWorkspace<T>` and `HiddenMarkovModel::viterbi_into` using exact `ln` transitions/emissions (`-inf` for exact zero probabilities) and `Vec<usize>` backpointers so 600-step and 10,000-step sequences decode with zero heap allocations and zero underflow.
   - Add `HiddenMarkovModel::log_likelihood` ($\sum_t \ln s_t$) and repair `HiddenMarkovModel::forward` ($\exp(\sum_t \ln s_t)$) in [`forward_backward.rs`](../../crates/03_domains/pure_math/src/pure_math/statistics/markov/hmm/forward_backward.rs), guarding zero scaling factors in `backward_probabilities`.
2. **Add Zero-Allocation Dirichlet-Smoothed Baum-Welch EM**:
   - Implement `BaumWelchConfig<T>`, `BaumWelchReport<T>`, and `BaumWelchWorkspace<T>` in [`baum_welch_types.rs`](../../crates/03_domains/pure_math/src/pure_math/statistics/markov/hmm/baum_welch_types.rs) and `fit_baum_welch_into` / `fit_baum_welch` in [`baum_welch.rs`](../../crates/03_domains/pure_math/src/pure_math/statistics/markov/hmm/baum_welch.rs).
   - Enforce symmetric Dirichlet pseudocount smoothing (`dirichlet_alpha >= 0`), convergence tolerance, iteration caps, and parameter snapshot rollback if floating-point rounding ever perturbs monotonic non-decreasing log-likelihood.
3. **Enforce Constructor-Validated `serde` Deserialization**:
   - Implement `Serialize` and strict `#[serde(deny_unknown_fields)]` `Deserialize` across `HiddenMarkovModel<T>`, `ContinuousMarkovChain<T>` ([`ctmc/mod.rs`](../../crates/03_domains/pure_math/src/pure_math/statistics/markov/ctmc/mod.rs)), `MarkovChain<T>` ([`dtmc/mod.rs`](../../crates/03_domains/pure_math/src/pure_math/statistics/markov/dtmc/mod.rs)), and `StateType`, delegating every deserialized payload through validated constructors so non-stochastic rows, positive generator diagonals, mismatched absorbing states, empty `0x0` matrices, and `NaN`/`Inf` entries fail immediately with `MarkovError`.
4. **Ship `MarkovSequenceWorkbenchTool` in `GameTheoryTab`**:
   - Add [`presets/markov.rs`](../../apps/math_explorer_gui/src/presets/markov.rs) and [`tabs/game_theory/markov/mod.rs`](../../apps/math_explorer_gui/src/tabs/game_theory/markov/mod.rs) supporting HMM Regime & Baum-Welch, CTMC Reliability, and Absorbing DTMC Valuation modes with `ExportCapabilities::plot()`, atomic UTF-8 and BOM-prefixed `.csv`/`.json` import and takeaway export via [`web_io.rs`](../../apps/math_explorer_gui/src/web_io.rs), a proactive `4,096`-byte `URL_JSON_BOUNDS` share-link budget warning, and linked theory documentation in [`papers/clinical_trials_statistics.tex`](../../papers/clinical_trials_statistics.tex).

```text
Updated GUI inventory totals:
- InteractiveTools: 77 (55 plot + 7 raster + 15 non-figure)
- StarterPresetMetadata cards: 112
```

## Consequences
- **Positive**: Long-horizon HMM sequences decode accurately without numerical collapse, Baum-Welch EM and Viterbi execute with zero steady-state heap allocations (`VerifiedAllocator`), serialized Markov models cannot bypass stochastic invariants, and all three Markov families are interactively accessible 100% client-side within the `< 3 MB` compressed WASM budget.
- **Negative / Trade-offs**: Expanding the public `pure_math::statistics::markov` API (`ViterbiWorkspace`, `BaumWelchConfig`, `BaumWelchReport`, `BaumWelchWorkspace`, `log_likelihood`, `viterbi_into`, `fit_baum_welch_into`, `fit_baum_welch`) requires regenerating `api_baseline.json` and maintaining modular submodule boundaries below the 500-line file cap.

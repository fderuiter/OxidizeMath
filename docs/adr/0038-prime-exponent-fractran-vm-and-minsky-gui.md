# ADR 0038: Prime-Exponent FRACTRAN Virtual Machine and Minsky Counter-Machine GUI

## Status
Accepted (extends [ADR 0023](0023-separate-metadata-ids-from-display-labels.md), [ADR 0024](0024-deferred-modal-exports-and-plot-primitives.md), and [ADR 0034](0034-curate-math-explorer-facade-exports-and-doctests.md))

## Context
[`README.md`](../../README.md) and [`crates/pure_math/src/pure_math/algorithmic_information/mod.rs`](../../crates/pure_math/src/pure_math/algorithmic_information/mod.rs) advertise FRACTRAN universal Minsky register machines, whereas [`crates/pure_math/src/pure_math/algorithmic_information/fractran.rs`](../../crates/pure_math/src/pure_math/algorithmic_information/fractran.rs) previously exposed only inert `FractranProgram` and `FractranState` structs with zero execution semantics, [`papers/algorithmic_information_rust.tex`](../../papers/algorithmic_information_rust.tex) omitted the `fractran` module, and [`math_explorer_gui/src/tabs/number_theory/mod.rs`](../../math_explorer_gui/src/tabs/number_theory/mod.rs) provided no interactive FRACTRAN explorer.

Executing Conway's universal `PRIMEGAME` program by naively multiplying arbitrary-precision integers allocates heap memory on every step and rapidly overflows 64-bit integers ($N$ exceeds $2^{64}$ before generating early primes), violating the zero-allocation hot-loop contract enforced by [`crates/verified_engine/src/allocator.rs`](../../crates/verified_engine/src/allocator.rs).

## Decision
1. **Preallocated Prime-Exponent Register VM in `pure_math`**:
   - Preserve `FractranProgram`, `FractranProgram::new`, and `FractranState` fields intact in [`crates/pure_math/src/pure_math/algorithmic_information/fractran.rs`](../../crates/pure_math/src/pure_math/algorithmic_information/fractran.rs) for source compatibility with [`crates/unified_verification/api_baseline.json`](../../crates/unified_verification/api_baseline.json).
   - Factor exact `dashu::rational::RBig` numerators and denominators from [`crates/pure_math/src/math_types.rs`](../../crates/pure_math/src/math_types.rs) (via [`crates/pure_math/src/pure_math/algorithmic_information/fractran_factor.rs`](../../crates/pure_math/src/pure_math/algorithmic_information/fractran_factor.rs)) into a sorted `u64` prime basis and flat `u32` numerator/denominator exponent tables inside `FractranCompiledVm`, rejecting non-positive (`<= 0`) fractions and capping synchronous trial division at `MAX_SMOOTH_PRIME = 10_000`.
   - Implement `#[verified_engine::verified]` `FractranCompiledVm::step_registers` and `power_of_two_exponent` using in-place `u32` vector subtraction and `u32::checked_add` overflow guards with zero dynamic heap allocations.
   - Provide `FractranState::decode`, `FractranState::transitions`, `FractranState::from_vm_heuristic`, and `FractranPreset` (`PrimeGame`, `Mult`, `Add`, `Fibonacci`).
2. **Interactive `FractranMachineTool` and Minsky Graph in `math_explorer_gui`**:
   - Implement `FractranMachineTool` across [`math_explorer_gui/src/tabs/number_theory/fractran_machine/mod.rs`](../../math_explorer_gui/src/tabs/number_theory/fractran_machine/mod.rs) and [`math_explorer_gui/src/tabs/number_theory/fractran_machine/diagram.rs`](../../math_explorer_gui/src/tabs/number_theory/fractran_machine/diagram.rs) and register it in [`math_explorer_gui/src/tabs/number_theory/mod.rs`](../../math_explorer_gui/src/tabs/number_theory/mod.rs).
   - Wire `PlaybackTransport::in_thread` ([`math_explorer_gui/src/framework/playback.rs`](../../math_explorer_gui/src/framework/playback.rs)) and cap `"Run Until Power of 2"` breakpoints at `4096` steps per frame, pausing playback on $2^k$ hits, halts, `u32::MAX` register overflow, or reaching the frame cap.
   - Render live firing-rule highlights, `egui_plot::Plot` prime-register bar charts with `.csv`, `.json`, `.svg`, `.png`, and `.gif` exports, and an `AccessKit`-focusable Minsky counter-machine graph (`Role::GraphicsDocument`).
3. **Traceability, Paper Documentation, and Cross-Target Verification**:
   - Document the `fractran` module and Minsky counter-machine correspondence in [`papers/algorithmic_information_rust.tex`](../../papers/algorithmic_information_rust.tex).
   - Add `VerifiedAllocator::lock_allocations()` zero-allocation `PRIMEGAME` prime-generation tests in [`crates/pure_math/tests/algorithmic_information.rs`](../../crates/pure_math/tests/algorithmic_information.rs), native GUI export/roundtrip tests in [`math_explorer_gui/src/tabs/number_theory/fractran_machine/tests.rs`](../../math_explorer_gui/src/tabs/number_theory/fractran_machine/tests.rs), and `wasm32-unknown-unknown` browser tests in [`math_explorer_gui/src/tabs/number_theory/fractran_machine/wasm_tests.rs`](../../math_explorer_gui/src/tabs/number_theory/fractran_machine/wasm_tests.rs).

## Consequences
- **Positive**: `FractranCompiledVm::step_registers` executes universal FRACTRAN programs such as Conway's `PRIMEGAME` with zero heap allocations per step, avoiding arbitrary-precision integer blow-up while preserving exact prime-exponent semantics up to `u32::MAX` per prime register.
- **Positive**: Users can step or fast-forward `PRIMEGAME`, `MULT`, `ADD`, `FIBONACCI`, and custom smooth fraction programs in `NumberTheoryTab` with keyboard-accessible Minsky state diagrams, URL hash sharing, and figure/dataset exports on desktop and WebAssembly.
- **Neutral**: Custom user fractions entered in the synchronous UI must factor over primes `<= 10_000` (`MAX_SMOOTH_PRIME`) so compilation never stalls the main UI thread.

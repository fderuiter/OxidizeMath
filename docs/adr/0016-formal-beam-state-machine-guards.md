# ADR 0016: Formal BeamState Machine with Validation Guards

## Status

Accepted

## Context
In radiotherapy respiratory gating, `GatingLogic` previously controlled linear accelerator beam delivery using an unencapsulated public boolean field (`is_beam_on`). Floating-point evaluations compared predicted amplitude against threshold boundaries without finite-value guards. Under IEEE 754 semantics, comparisons against `NaN` evaluate to `false`, causing the system to fail open and maintain radiation delivery during radar sensor phase-lock loss or numerical corruption. Furthermore, absence of formal lifecycle states allowed arbitrary external mutation and unvalidated jumps between dormant and beam-emitting states.

## Decision
We refactored `GatingLogic` into an encapsulated, deterministic finite state machine with strict input validation:
1. **Explicit Lifecycle State Machine (`BeamState`)**: Formulated formal operational states: `Standby`, `Armed`, `BeamActive`, `BeamHold`, and `SafetyInterlock`. State is privately encapsulated within `GatingLogic`.
2. **Transition Validation and Interlock Latching**: Implemented lifecycle methods `arm()`, `disarm()`, `trigger_interlock()`, and `reset_interlock()`. Illegal transitions (such as jumping directly from `Standby` or `SafetyInterlock` to `BeamActive`) are rejected with `GatingError`. Once entered, `SafetyInterlock` latches permanently until an explicit `reset_interlock()` command returns the system to `Standby`.
3. **Strict Floating-Point Guards**: Guarded `evaluate(&mut self, amplitude: f64, velocity: f64) -> BeamState` with `is_finite()` validation across all input signals and parameters. Any non-finite float immediately trips the controller to `SafetyInterlock`, deactivating radiation beam delivery synchronously.
4. **Hysteresis Enforcement in Ready States**: Enforced Schmidt trigger hysteresis boundaries ($A_{pred} < T - H \implies \text{BeamActive}$, $A_{pred} > T + H \implies \text{BeamHold}$) strictly when the controller is in ready operational states (`Armed`, `BeamActive`, `BeamHold`). Calls to `evaluate()` in `Standby` or `SafetyInterlock` are prevented from emitting radiation.
5. **Zero-Allocation Latency Compliance**: Preserved zero heap allocations in the real-time evaluation path to satisfy LINAC sub-microsecond latency requirements.

## Consequences
- **Safety**: Unintended radiation emission is eliminated when sensor inputs become corrupted or non-finite.
- **Encapsulation**: State manipulation is strictly mediated by validated lifecycle methods and read-only query accessors (`state()`, `is_beam_active()`).
- **Compatibility**: Preserved Schmidt trigger displacement thresholds and backward-compatible status queries for existing clinical integration tests.

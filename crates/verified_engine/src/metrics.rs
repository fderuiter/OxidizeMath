use std::cell::Cell;

/// Snapshot of runtime execution complexity counters recorded on the current thread during a
/// verified run.
///
/// # Example
/// ```
/// use verified_engine::metrics::{
///     ComplexityMetrics, disable_verification, enable_verification, get_metrics,
///     increment_arithmetic, increment_calls, increment_memory_loads, reset_metrics,
/// };
///
/// enable_verification();
/// reset_metrics();
/// increment_calls();
/// increment_arithmetic(4);
/// increment_memory_loads(2);
///
/// let snapshot: ComplexityMetrics = get_metrics();
/// disable_verification();
///
/// assert_eq!(snapshot.function_calls, 1);
/// assert_eq!(snapshot.arithmetic_ops, 4);
/// assert_eq!(snapshot.memory_loads, 2);
/// ```
#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ComplexityMetrics {
    /// Total binary arithmetic and bitwise operations executed inside `#[verified]` blocks.
    pub arithmetic_ops: u64,
    /// Total slice/array indexing load operations executed inside `#[verified]` blocks.
    pub memory_loads: u64,
    /// Total invocations of `#[verified]` functions recorded on the current thread.
    pub function_calls: u64,
}

impl ComplexityMetrics {
    /// Constructs a zeroed [`ComplexityMetrics`] counter snapshot usable in `const` contexts.
    pub const fn new() -> Self {
        Self {
            arithmetic_ops: 0,
            memory_loads: 0,
            function_calls: 0,
        }
    }
}

thread_local! {
    /// Thread-local accumulator of binary arithmetic operations in active verification regions.
    pub static ARITHMETIC_OPS: Cell<u64> = const { Cell::new(0) };
    /// Thread-local accumulator of indexing memory loads in active verification regions.
    pub static MEMORY_LOADS: Cell<u64> = const { Cell::new(0) };
    /// Thread-local accumulator of `#[verified]` function entries in active verification regions.
    pub static FUNCTION_CALLS: Cell<u64> = const { Cell::new(0) };
    /// Thread-local flag indicating whether verification instrumentation and allocator checks are active.
    pub static VERIFICATION_MODE: Cell<bool> = const { Cell::new(false) };
}

/// Enables runtime verification mode and metric accumulation on the calling thread.
pub fn enable_verification() {
    VERIFICATION_MODE.set(true);
}

/// Disables runtime verification mode on the calling thread.
pub fn disable_verification() {
    VERIFICATION_MODE.set(false);
}

/// Returns `true` when verification instrumentation is active on the calling thread.
pub fn is_verification_enabled() -> bool {
    VERIFICATION_MODE.get()
}

/// Resets all thread-local complexity counters (`ARITHMETIC_OPS`, `MEMORY_LOADS`, `FUNCTION_CALLS`) to zero.
pub fn reset_metrics() {
    ARITHMETIC_OPS.set(0);
    MEMORY_LOADS.set(0);
    FUNCTION_CALLS.set(0);
}

/// Reads the current thread-local counters into a [`ComplexityMetrics`] snapshot.
pub fn get_metrics() -> ComplexityMetrics {
    ComplexityMetrics {
        arithmetic_ops: ARITHMETIC_OPS.get(),
        memory_loads: MEMORY_LOADS.get(),
        function_calls: FUNCTION_CALLS.get(),
    }
}

/// Adds `count` arithmetic operations to the calling thread's counter when verification is enabled.
#[inline(always)]
pub fn increment_arithmetic(count: u64) {
    if is_verification_enabled() {
        ARITHMETIC_OPS.set(ARITHMETIC_OPS.get() + count);
    }
}

/// Adds `count` indexed memory loads to the calling thread's counter when verification is enabled.
#[inline(always)]
pub fn increment_memory_loads(count: u64) {
    if is_verification_enabled() {
        MEMORY_LOADS.set(MEMORY_LOADS.get() + count);
    }
}

/// Increments the calling thread's `#[verified]` function call counter by one when verification is enabled.
#[inline(always)]
pub fn increment_calls() {
    if is_verification_enabled() {
        FUNCTION_CALLS.set(FUNCTION_CALLS.get() + 1);
    }
}

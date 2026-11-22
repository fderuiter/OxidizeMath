use std::cell::Cell;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub struct ComplexityMetrics {
    #[allow(missing_docs)]
    pub arithmetic_ops: u64,
    #[allow(missing_docs)]
    pub memory_loads: u64,
    #[allow(missing_docs)]
    pub function_calls: u64,
}

impl ComplexityMetrics {
    #[allow(missing_docs)]
    pub const fn new() -> Self {
        Self {
            arithmetic_ops: 0,
            memory_loads: 0,
            function_calls: 0,
        }
    }
}

thread_local! {
    pub static ARITHMETIC_OPS: Cell<u64> = const { Cell::new(0) };
    pub static MEMORY_LOADS: Cell<u64> = const { Cell::new(0) };
    pub static FUNCTION_CALLS: Cell<u64> = const { Cell::new(0) };
    pub static VERIFICATION_MODE: Cell<bool> = const { Cell::new(false) };
}

#[allow(missing_docs)]
pub fn enable_verification() {
    VERIFICATION_MODE.set(true);
}

#[allow(missing_docs)]
pub fn disable_verification() {
    VERIFICATION_MODE.set(false);
}

#[allow(missing_docs)]
pub fn is_verification_enabled() -> bool {
    VERIFICATION_MODE.get()
}

#[allow(missing_docs)]
pub fn reset_metrics() {
    ARITHMETIC_OPS.set(0);
    MEMORY_LOADS.set(0);
    FUNCTION_CALLS.set(0);
}

#[allow(missing_docs)]
pub fn get_metrics() -> ComplexityMetrics {
    ComplexityMetrics {
        arithmetic_ops: ARITHMETIC_OPS.get(),
        memory_loads: MEMORY_LOADS.get(),
        function_calls: FUNCTION_CALLS.get(),
    }
}

#[inline(always)]
#[allow(missing_docs)]
pub fn increment_arithmetic(count: u64) {
    if is_verification_enabled() {
        ARITHMETIC_OPS.set(ARITHMETIC_OPS.get() + count);
    }
}

#[inline(always)]
#[allow(missing_docs)]
pub fn increment_memory_loads(count: u64) {
    if is_verification_enabled() {
        MEMORY_LOADS.set(MEMORY_LOADS.get() + count);
    }
}

#[inline(always)]
#[allow(missing_docs)]
pub fn increment_calls() {
    if is_verification_enabled() {
        FUNCTION_CALLS.set(FUNCTION_CALLS.get() + 1);
    }
}

#[cfg(feature = "std")]
use core::cell::Cell;
#[cfg(feature = "std")]
use core::sync::atomic::{AtomicUsize, Ordering};

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
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
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

/// Process-wide count of threads currently executing with verification enabled.
#[cfg(feature = "std")]
static ACTIVE_VERIFIERS: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "std")]
std::thread_local! {
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
#[cfg(feature = "std")]
pub fn enable_verification() {
    if !VERIFICATION_MODE.replace(true) {
        ACTIVE_VERIFIERS.fetch_add(1, Ordering::SeqCst);
    }
}

/// No-op verification toggle in `#![no_std]` builds where thread-local storage is unavailable.
#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn enable_verification() {}

/// Disables runtime verification mode on the calling thread.
#[cfg(feature = "std")]
pub fn disable_verification() {
    if VERIFICATION_MODE.replace(false) {
        ACTIVE_VERIFIERS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// No-op verification toggle in `#![no_std]` builds where thread-local storage is unavailable.
#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn disable_verification() {}

/// Returns `true` when verification instrumentation is active on the calling thread.
#[cfg(feature = "std")]
#[inline(always)]
pub fn is_verification_enabled() -> bool {
    ACTIVE_VERIFIERS.load(Ordering::Relaxed) != 0 && VERIFICATION_MODE.get()
}

/// Returns `false` in `#![no_std]` builds without `std` thread-local counters.
#[cfg(not(feature = "std"))]
#[inline(always)]
pub const fn is_verification_enabled() -> bool {
    false
}

/// Resets all thread-local complexity counters (`ARITHMETIC_OPS`, `MEMORY_LOADS`, `FUNCTION_CALLS`) to zero.
#[cfg(feature = "std")]
pub fn reset_metrics() {
    ARITHMETIC_OPS.set(0);
    MEMORY_LOADS.set(0);
    FUNCTION_CALLS.set(0);
}

/// No-op counter reset in `#![no_std]` builds.
#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn reset_metrics() {}

/// Reads the current thread-local counters into a [`ComplexityMetrics`] snapshot.
#[cfg(feature = "std")]
pub fn get_metrics() -> ComplexityMetrics {
    ComplexityMetrics {
        arithmetic_ops: ARITHMETIC_OPS.get(),
        memory_loads: MEMORY_LOADS.get(),
        function_calls: FUNCTION_CALLS.get(),
    }
}

/// Returns a zeroed [`ComplexityMetrics`] snapshot in `#![no_std]` builds.
#[cfg(not(feature = "std"))]
#[inline(always)]
pub const fn get_metrics() -> ComplexityMetrics {
    ComplexityMetrics::new()
}

/// Adds `count` arithmetic operations to the calling thread's counter when verification is enabled.
#[cfg(feature = "std")]
#[inline(always)]
pub fn increment_arithmetic(count: u64) {
    if is_verification_enabled() {
        ARITHMETIC_OPS.set(ARITHMETIC_OPS.get() + count);
    }
}

/// No-op arithmetic counter increment in `#![no_std]` builds.
#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn increment_arithmetic(_count: u64) {}

/// Adds `count` indexed memory loads to the calling thread's counter when verification is enabled.
#[cfg(feature = "std")]
#[inline(always)]
pub fn increment_memory_loads(count: u64) {
    if is_verification_enabled() {
        MEMORY_LOADS.set(MEMORY_LOADS.get() + count);
    }
}

/// No-op memory-load counter increment in `#![no_std]` builds.
#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn increment_memory_loads(_count: u64) {}

/// Increments the calling thread's `#[verified]` function call counter by one when verification is enabled.
#[cfg(feature = "std")]
#[inline(always)]
pub fn increment_calls() {
    if is_verification_enabled() {
        FUNCTION_CALLS.set(FUNCTION_CALLS.get() + 1);
    }
}

/// No-op function-call counter increment in `#![no_std]` builds.
#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn increment_calls() {}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::engine::VerifiedEngine;
    use std::sync::Mutex;

    static TEST_SERIAL_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_idempotent_enable_disable_and_atomic_count() {
        let _guard = TEST_SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable_verification();
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 0);
        assert!(!is_verification_enabled());

        // Redundant disable on inactive thread must not underflow zero.
        disable_verification();
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 0);

        // Repeated enable on same thread must increment counter only once.
        enable_verification();
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 1);
        assert!(is_verification_enabled());
        enable_verification();
        enable_verification();
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 1);

        disable_verification();
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 0);
        assert!(!is_verification_enabled());
        disable_verification();
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_run_verified_panic_unwind_preserves_atomic_count() {
        let _guard = TEST_SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable_verification();
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 0);

        let unwind_enable = std::panic::catch_unwind(|| {
            VerifiedEngine::run_verified(|| {
                enable_verification();
                assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 1);
                panic!("simulated failure after nested enable");
            });
        });
        assert!(unwind_enable.is_err());
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 0);
        assert!(!is_verification_enabled());

        let unwind_disable = std::panic::catch_unwind(|| {
            VerifiedEngine::run_verified(|| {
                disable_verification();
                assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 0);
                panic!("simulated failure after manual disable");
            });
        });
        assert!(unwind_disable.is_err());
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 0);
        assert!(!is_verification_enabled());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_multithreaded_active_verifiers_isolation() {
        use std::sync::{Arc, Barrier};
        let _guard = TEST_SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable_verification();
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 0);

        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let w_entered = Arc::clone(&entered);
        let w_release = Arc::clone(&release);

        let handle = std::thread::spawn(move || {
            enable_verification();
            enable_verification();
            reset_metrics();
            increment_calls();
            increment_arithmetic(5);
            w_entered.wait();
            w_release.wait();
            let snap = get_metrics();
            disable_verification();
            snap
        });

        entered.wait();
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 1);
        assert!(!is_verification_enabled());
        reset_metrics();
        increment_arithmetic(99);
        assert_eq!(get_metrics().arithmetic_ops, 0);

        release.wait();
        let worker_snap = handle.join().expect("worker joined");
        assert_eq!(worker_snap.function_calls, 1);
        assert_eq!(worker_snap.arithmetic_ops, 5);
        assert_eq!(ACTIVE_VERIFIERS.load(Ordering::SeqCst), 0);
    }
}

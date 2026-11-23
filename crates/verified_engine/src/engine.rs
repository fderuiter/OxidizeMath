use crate::allocator::{lock_allocations, unlock_allocations};
use crate::metrics::{
    ComplexityMetrics, disable_verification, enable_verification, get_metrics, reset_metrics,
};

/// Logs an external native C++ library dependency for workspace traceability audits.
pub fn register_cpp_dependency(dep: &str) {
    println!("[Traceability] Tracking external C++ dependency: {}", dep);
}

/// High-integrity runtime execution harness enforcing post-initialization memory locks and
/// collecting thread-local [`ComplexityMetrics`].
///
/// # Invariants
/// - Enables thread-local verification mode, resets counters, and engages [`lock_allocations`]
///   prior to invoking the target closure (NASA Power of 10 Rule 3).
/// - Unlocks allocations and disables verification mode even if the closure panics, before
///   resuming stack unwinding via [`std::panic::resume_unwind`].
///
/// # Example
/// ```
/// use verified_engine::allocator::MEMORY_LOCK;
/// use verified_engine::engine::VerifiedEngine;
/// use verified_engine::verified;
/// use std::sync::atomic::Ordering;
///
/// #[verified]
/// fn harmonic_potential(k: f64, displacement: f64) -> f64 {
///     0.5 * k * displacement * displacement
/// }
///
/// let (energy, metrics) = VerifiedEngine::run_verified(|| harmonic_potential(4.0, 3.0));
/// assert_eq!(energy, 18.0);
/// assert_eq!(metrics.function_calls, 1);
/// assert!(metrics.arithmetic_ops >= 3);
/// assert!(!MEMORY_LOCK.load(Ordering::SeqCst));
///
/// // Panic unwinding cleanly restores allocation and verification state.
/// let unwind_res = std::panic::catch_unwind(|| {
///     VerifiedEngine::run_verified(|| panic!("divergent step"));
/// });
/// assert!(unwind_res.is_err());
/// assert!(!MEMORY_LOCK.load(Ordering::SeqCst));
/// ```
pub struct VerifiedEngine;

impl VerifiedEngine {
    /// Executes `f` inside a zero-allocation verified region and returns its output paired with
    /// collected [`ComplexityMetrics`].
    pub fn run_verified<F, R>(f: F) -> (R, ComplexityMetrics)
    where
        F: FnOnce() -> R,
    {
        enable_verification();
        reset_metrics();
        // NASA Power of 10 Rule 3: No dynamic memory allocation after initialization.
        // We assume initialization is done outside of `run_verified`, so we lock here.
        lock_allocations();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

        unlock_allocations();
        let metrics = get_metrics();
        disable_verification();

        match result {
            Ok(r) => (r, metrics),
            Err(e) => std::panic::resume_unwind(e),
        }
    }
}

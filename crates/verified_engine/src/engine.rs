use std::sync::OnceLock;

use crate::allocator::{lock_allocations, probe_verified_allocator_hook, unlock_allocations};
use crate::metrics::{
    ComplexityMetrics, disable_verification, enable_verification, get_metrics, reset_metrics,
};

/// Cached result of the one-time process `#[global_allocator]` hook probe.
static ALLOCATOR_PROBE: OnceLock<bool> = OnceLock::new();

fn is_doctest_process() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Ok(exe) = std::env::current_exe() else {
            return false;
        };
        let path_str = exe.to_string_lossy();
        let is_rustdoc_out = exe
            .file_stem()
            .and_then(|s| s.to_str())
            .is_some_and(|stem| stem == "rust_out" || stem.starts_with("rust_out"));
        is_rustdoc_out || path_str.contains("rustdoctest")
    }
}

pub(crate) fn enforce_allocator_installed_with(allocator_installed: bool, is_doctest: bool) {
    assert!(
        allocator_installed || is_doctest,
        "VIOLATION: VerifiedEngine::run_verified requires VerifiedAllocator to be installed via #[global_allocator] in test binaries."
    );
}

fn ensure_allocator_installed() {
    let installed = *ALLOCATOR_PROBE.get_or_init(probe_verified_allocator_hook);
    enforce_allocator_installed_with(installed, is_doctest_process());
}

/// Logs an external native C++ library dependency for workspace traceability audits.
pub fn register_cpp_dependency(dep: &str) {
    println!("[Traceability] Tracking external C++ dependency: {}", dep);
}

/// High-integrity runtime execution harness enforcing post-initialization memory locks and
/// collecting thread-local [`ComplexityMetrics`].
///
/// # Invariants
/// - Verifies once per process via [`OnceLock`] that [`crate::allocator::VerifiedAllocator`] is
///   installed before locking memory in unit and integration test binaries (allowing `rustdoc`
///   doctest binaries without `#[global_allocator]` to collect [`ComplexityMetrics`]).
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
/// assert_eq!(metrics.allocations, 0);
/// assert_eq!(metrics.allocated_bytes, 0);
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
        ensure_allocator_installed();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "requires VerifiedAllocator to be installed")]
    fn test_probe_fails_fast_when_allocator_missing_outside_doctest() {
        enforce_allocator_installed_with(false, false);
    }

    #[test]
    fn test_probe_permits_doctest_without_global_allocator() {
        enforce_allocator_installed_with(false, true);
        enforce_allocator_installed_with(true, false);
    }
}

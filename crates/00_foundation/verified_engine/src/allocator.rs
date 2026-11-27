use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, Ordering};

use verified_engine_macros::runtime_violation;

/// Global allocator wrapper enforcing NASA Power of 10 Rule 3 (no post-initialization heap
/// allocation inside verified execution regions).
///
/// When installed via `#[global_allocator]` and both [`MEMORY_LOCK`] and
/// [`crate::metrics::is_verification_enabled()`] are active, any `alloc`, `alloc_zeroed`, or
/// `realloc` call emits a structured `FATAL` JSON payload and aborts the process.
pub struct VerifiedAllocator;

/// Process-wide atomic flag indicating whether dynamic heap allocations are locked.
pub static MEMORY_LOCK: AtomicBool = AtomicBool::new(false);

/// Engages the global allocation lock (`MEMORY_LOCK = true`) using sequentially consistent ordering.
///
/// # Example
/// ```
/// use std::sync::atomic::Ordering;
/// use verified_engine::allocator::{MEMORY_LOCK, lock_allocations, unlock_allocations};
/// use verified_engine::metrics::is_verification_enabled;
///
/// assert!(!is_verification_enabled());
/// lock_allocations();
/// assert!(MEMORY_LOCK.load(Ordering::SeqCst));
/// unlock_allocations();
/// assert!(!MEMORY_LOCK.load(Ordering::SeqCst));
/// ```
pub fn lock_allocations() {
    MEMORY_LOCK.store(true, Ordering::SeqCst);
}

/// Releases the global allocation lock (`MEMORY_LOCK = false`) using sequentially consistent ordering.
pub fn unlock_allocations() {
    MEMORY_LOCK.store(false, Ordering::SeqCst);
}

/// Set once a violation starts being reported. Writing the FATAL payload to stderr can itself
/// allocate (lazy stderr/thread-local setup on macOS), so allocations made while reporting are
/// passed straight to `System` instead of re-entering the trap and overflowing the stack.
static REPORTING_VIOLATION: AtomicBool = AtomicBool::new(false);

/// Returns `true` when this allocation must trip the Rule 3 trap, claiming the single report slot.
#[inline]
fn should_trap() -> bool {
    MEMORY_LOCK.load(Ordering::SeqCst)
        && crate::metrics::is_verification_enabled()
        && !REPORTING_VIOLATION.swap(true, Ordering::SeqCst)
}

unsafe impl GlobalAlloc for VerifiedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if should_trap() {
            runtime_violation!("VIOLATION: Dynamic memory allocation after initialization.");
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if should_trap() {
            runtime_violation!("VIOLATION: Dynamic memory allocation after initialization.");
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if should_trap() {
            runtime_violation!("VIOLATION: Dynamic memory reallocation after initialization.");
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

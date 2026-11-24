use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use verified_engine_macros::runtime_violation;

/// Global allocator wrapper enforcing NASA Power of 10 Rule 3 (no post-initialization heap
/// allocation inside verified execution regions) while recording exact per-thread allocation
/// counts and byte totals.
///
/// When installed via `#[global_allocator]` and both [`MEMORY_LOCK`] and
/// [`crate::metrics::is_verification_enabled()`] are active, any `alloc`, `alloc_zeroed`, or
/// `realloc` call emits a structured `FATAL` JSON payload and aborts the process.
pub struct VerifiedAllocator;

/// Process-wide atomic flag indicating whether dynamic heap allocations are locked.
pub static MEMORY_LOCK: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// Re-entrancy guard preventing recursive TLS accounting inside `GlobalAlloc`.
    static IN_ALLOC_HOOK: Cell<bool> = const { Cell::new(false) };
    /// Thread-local total heap allocation calls observed by [`VerifiedAllocator`].
    pub static THREAD_ALLOC_COUNT: Cell<u64> = const { Cell::new(0) };
    /// Thread-local total heap bytes requested through [`VerifiedAllocator`].
    pub static THREAD_ALLOC_BYTES: Cell<u64> = const { Cell::new(0) };
    /// Thread-local flag temporarily bypassing `MEMORY_LOCK` during the one-time installation probe.
    static PROBE_BYPASS_LOCK: Cell<bool> = const { Cell::new(false) };
}

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

/// Returns the `(allocations, allocated_bytes)` recorded on the calling thread by [`VerifiedAllocator`].
#[inline]
pub fn thread_allocation_stats() -> (u64, u64) {
    let count = THREAD_ALLOC_COUNT.try_with(Cell::get).unwrap_or(0);
    let bytes = THREAD_ALLOC_BYTES.try_with(Cell::get).unwrap_or(0);
    (count, bytes)
}

/// Resets the calling thread's [`VerifiedAllocator`] allocation count and byte accumulators to zero.
#[inline]
pub fn reset_thread_allocation_stats() {
    let _ = THREAD_ALLOC_COUNT.try_with(|c| c.set(0));
    let _ = THREAD_ALLOC_BYTES.try_with(|c| c.set(0));
}

/// Performs a single 1-byte probe allocation to detect whether [`VerifiedAllocator`] is installed
/// as the process `#[global_allocator]`.
pub(crate) fn probe_verified_allocator_hook() -> bool {
    let _ = PROBE_BYPASS_LOCK.try_with(|c| c.set(true));
    let (before_count, _) = thread_allocation_stats();
    let probe = std::hint::black_box(Box::new(0u8));
    std::hint::black_box(*probe);
    drop(probe);
    let (after_count, _) = thread_allocation_stats();
    let _ = PROBE_BYPASS_LOCK.try_with(|c| c.set(false));
    after_count > before_count
}

impl VerifiedAllocator {
    #[inline(always)]
    fn check_and_record(bytes: usize, is_realloc: bool) {
        let bypass = PROBE_BYPASS_LOCK.try_with(Cell::get).unwrap_or(false);
        if !bypass
            && MEMORY_LOCK.load(Ordering::SeqCst)
            && crate::metrics::is_verification_enabled()
            && !std::thread::panicking()
        {
            if is_realloc {
                runtime_violation!("VIOLATION: Dynamic memory reallocation after initialization.");
            } else {
                runtime_violation!("VIOLATION: Dynamic memory allocation after initialization.");
            }
        }
        let _ = IN_ALLOC_HOOK.try_with(|in_hook| {
            if !in_hook.replace(true) {
                let bytes_u64 = bytes as u64;
                let _ = THREAD_ALLOC_COUNT.try_with(|c| c.set(c.get().saturating_add(1)));
                let _ = THREAD_ALLOC_BYTES.try_with(|c| c.set(c.get().saturating_add(bytes_u64)));
                if !bypass {
                    crate::metrics::increment_allocations(bytes_u64);
                }
                in_hook.set(false);
            }
        });
    }
}

unsafe impl GlobalAlloc for VerifiedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::check_and_record(layout.size(), false);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::check_and_record(layout.size(), false);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::check_and_record(new_size, true);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

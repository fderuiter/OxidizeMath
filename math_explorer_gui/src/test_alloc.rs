#![cfg_attr(any(), verified(opt_out = "tests"))]

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};

static PEAK_TRACKING_ACTIVE: AtomicBool = AtomicBool::new(false);
static LIVE_HEAP_BYTES: AtomicIsize = AtomicIsize::new(0);
static PEAK_LIVE_HEAP_BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ON_MEASURED_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Delegates global allocation in GUI tests directly to [`verified_engine::allocator::VerifiedAllocator`]
/// so thread-local allocation counts and byte totals are recorded without size or alignment filters,
/// while also tracking peak live heap bytes for streaming export tests.
pub(crate) struct CountingVerifiedAllocator;

impl CountingVerifiedAllocator {
    #[inline]
    fn record_live_delta(delta: isize) {
        if !PEAK_TRACKING_ACTIVE.load(Ordering::Relaxed)
            || !ON_MEASURED_THREAD.try_with(Cell::get).unwrap_or(false)
        {
            return;
        }
        let prev = LIVE_HEAP_BYTES.fetch_add(delta, Ordering::Relaxed);
        let curr = prev.saturating_add(delta);
        if curr > 0 {
            PEAK_LIVE_HEAP_BYTES.fetch_max(curr as usize, Ordering::Relaxed);
        }
    }
}

unsafe impl GlobalAlloc for CountingVerifiedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { verified_engine::allocator::VerifiedAllocator.alloc(layout) };
        if !ptr.is_null() {
            Self::record_live_delta(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        Self::record_live_delta(-(layout.size() as isize));
        unsafe { verified_engine::allocator::VerifiedAllocator.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { verified_engine::allocator::VerifiedAllocator.alloc_zeroed(layout) };
        if !ptr.is_null() {
            Self::record_live_delta(layout.size() as isize);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let next =
            unsafe { verified_engine::allocator::VerifiedAllocator.realloc(ptr, layout, new_size) };
        if !next.is_null() {
            Self::record_live_delta(new_size as isize - layout.size() as isize);
        }
        next
    }
}

struct TrackingResetGuard;

impl Drop for TrackingResetGuard {
    fn drop(&mut self) {
        PEAK_TRACKING_ACTIVE.store(false, Ordering::SeqCst);
        let _ = ON_MEASURED_THREAD.try_with(|c| c.set(false));
    }
}

/// Executes `f` under `VERIFIED_ALLOC_TEST_MUTEX` while reading exact thread-local allocation
/// counts and requested byte totals from [`verified_engine::allocator::thread_allocation_stats`].
pub(crate) fn measure_coord_series_allocs<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    let _lock = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (before_count, before_bytes) = verified_engine::allocator::thread_allocation_stats();
    let out = f();
    let (after_count, after_bytes) = verified_engine::allocator::thread_allocation_stats();
    let count = after_count.saturating_sub(before_count) as usize;
    let bytes = after_bytes.saturating_sub(before_bytes) as usize;
    (out, count, bytes)
}

/// Executes `f` under `VERIFIED_ALLOC_TEST_MUTEX` and returns `(output, peak_live_heap_bytes)` on the measured thread.
pub(crate) fn measure_peak_live_heap_bytes<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let _lock = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    ON_MEASURED_THREAD.with(|c| c.set(true));
    LIVE_HEAP_BYTES.store(0, Ordering::SeqCst);
    PEAK_LIVE_HEAP_BYTES.store(0, Ordering::SeqCst);
    PEAK_TRACKING_ACTIVE.store(true, Ordering::SeqCst);
    let _reset = TrackingResetGuard;
    let out = f();
    drop(_reset);
    let peak = PEAK_LIVE_HEAP_BYTES.load(Ordering::SeqCst);
    (out, peak)
}

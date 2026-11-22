#![cfg_attr(any(), verified(opt_out = "tests"))]

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const MIN_SERIES_ALLOC_BYTES: usize = 64 * 1024;
const F64_ALIGN: usize = std::mem::align_of::<f64>();

static TRACKING_ACTIVE: AtomicBool = AtomicBool::new(false);
static COORD_SERIES_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static COORD_SERIES_ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ON_MEASURED_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Wraps `VerifiedAllocator` in unit tests to count `f64`-aligned coordinate buffer allocations.
pub(crate) struct CountingVerifiedAllocator;

impl CountingVerifiedAllocator {
    #[inline]
    fn record_if_tracked(align: usize, size: usize) {
        if TRACKING_ACTIVE.load(Ordering::Relaxed)
            && align >= F64_ALIGN
            && size >= MIN_SERIES_ALLOC_BYTES
            && ON_MEASURED_THREAD.try_with(Cell::get).unwrap_or(false)
        {
            COORD_SERIES_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            COORD_SERIES_ALLOC_BYTES.fetch_add(size, Ordering::Relaxed);
        }
    }
}

unsafe impl GlobalAlloc for CountingVerifiedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::record_if_tracked(layout.align(), layout.size());
        unsafe { verified_engine::allocator::VerifiedAllocator.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { verified_engine::allocator::VerifiedAllocator.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::record_if_tracked(layout.align(), layout.size());
        unsafe { verified_engine::allocator::VerifiedAllocator.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::record_if_tracked(layout.align(), new_size);
        unsafe { verified_engine::allocator::VerifiedAllocator.realloc(ptr, layout, new_size) }
    }
}

struct TrackingResetGuard;

impl Drop for TrackingResetGuard {
    fn drop(&mut self) {
        TRACKING_ACTIVE.store(false, Ordering::SeqCst);
        let _ = ON_MEASURED_THREAD.try_with(|c| c.set(false));
    }
}

/// Executes `f` under `VERIFIED_ALLOC_TEST_MUTEX` while counting `f64`-aligned allocations `>= 64 KiB` on the calling thread.
pub(crate) fn measure_coord_series_allocs<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    let _lock = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    ON_MEASURED_THREAD.with(|c| c.set(true));
    COORD_SERIES_ALLOC_COUNT.store(0, Ordering::SeqCst);
    COORD_SERIES_ALLOC_BYTES.store(0, Ordering::SeqCst);
    TRACKING_ACTIVE.store(true, Ordering::SeqCst);
    let _reset = TrackingResetGuard;
    let out = f();
    drop(_reset);
    let count = COORD_SERIES_ALLOC_COUNT.load(Ordering::SeqCst);
    let bytes = COORD_SERIES_ALLOC_BYTES.load(Ordering::SeqCst);
    (out, count, bytes)
}

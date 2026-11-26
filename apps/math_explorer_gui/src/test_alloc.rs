#![cfg_attr(any(), verified(opt_out = "tests"))]

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const MIN_SERIES_ALLOC_BYTES: usize = 64 * 1024;
const F64_ALIGN: usize = std::mem::align_of::<f64>();

static TRACKING_ACTIVE: AtomicBool = AtomicBool::new(false);
static COORD_SERIES_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static COORD_SERIES_ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_TRACKING_ACTIVE: AtomicBool = AtomicBool::new(false);
static LIVE_HEAP_BYTES: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);
static PEAK_LIVE_HEAP_BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ON_MEASURED_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Wraps `VerifiedAllocator` in unit tests to count `f64`-aligned coordinate buffer allocations
/// and track cross-thread peak live heap bytes.
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
        Self::record_if_tracked(layout.align(), layout.size());
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
        Self::record_if_tracked(layout.align(), layout.size());
        let ptr = unsafe { verified_engine::allocator::VerifiedAllocator.alloc_zeroed(layout) };
        if !ptr.is_null() {
            Self::record_live_delta(layout.size() as isize);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::record_if_tracked(layout.align(), new_size);
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
        TRACKING_ACTIVE.store(false, Ordering::SeqCst);
        PEAK_TRACKING_ACTIVE.store(false, Ordering::SeqCst);
        let _ = ON_MEASURED_THREAD.try_with(|c| c.set(false));
    }
}

/// Executes `f` under `VERIFIED_ALLOC_TEST_MUTEX` while counting `f64`-aligned allocations `>= 64 KiB` on the calling thread.
#[cfg(any(feature = "epidemiology", feature = "pure_math"))]
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

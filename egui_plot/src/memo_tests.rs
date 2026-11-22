//! Counting-allocator, memoization hit/invalidation, and streaming intersection allocation tests.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use egui::{Id, Pos2, RawInput, Rect, Vec2b, vec2};

use crate::{
    AccessState, ExportPrimitive, FitType, HLine, Line, Plot, PlotBounds, PlotItem, PlotMemory,
    PlotPoint, PlotPoints, PlotTransform, SharedPoints, clear_last_plot_datasets,
    detect_critical_points, get_last_plot_primitives, get_last_plot_shared_datasets,
};

struct CountingSystemAllocator;

static ALLOC_LOCK: Mutex<()> = Mutex::new(());
static TRACK_COORD_ALLOCS: AtomicBool = AtomicBool::new(false);
static COORD_ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static COORD_ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ON_MEASURED_THREAD: Cell<bool> = const { Cell::new(false) };
}

/// Minimum size (64 KiB) and `f64` alignment (`align >= 8`) matching any multi-thousand-point
/// `[f64; 2]`, `PlotPoint`, or `merged_knots: Vec<f64>` coordinate allocation while ignoring
/// 4-byte-aligned `egui::Pos2` paint buffers.
const COORD_SERIES_MIN_BYTES: usize = 64 * 1024;
const F64_ALIGN: usize = std::mem::align_of::<f64>();

#[inline]
fn record_coord_alloc_if_tracked(align: usize, size: usize) {
    if TRACK_COORD_ALLOCS.load(Ordering::Relaxed)
        && align >= F64_ALIGN
        && size >= COORD_SERIES_MIN_BYTES
        && ON_MEASURED_THREAD.try_with(Cell::get).unwrap_or(false)
    {
        COORD_ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        COORD_ALLOC_BYTES.fetch_add(size, Ordering::Relaxed);
    }
}

// SAFETY: Delegates all allocation operations directly to `std::alloc::System` with identical
// `Layout` and pointer arguments while recording atomic counters.
unsafe impl GlobalAlloc for CountingSystemAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_coord_alloc_if_tracked(layout.align(), layout.size());
        // SAFETY: Forwarded directly to `System::alloc`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Forwarded directly to `System::dealloc`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_coord_alloc_if_tracked(layout.align(), layout.size());
        // SAFETY: Forwarded directly to `System::alloc_zeroed`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_coord_alloc_if_tracked(layout.align(), new_size);
        // SAFETY: Forwarded directly to `System::realloc`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_TEST_ALLOCATOR: CountingSystemAllocator = CountingSystemAllocator;

struct ThreadTrackingGuard;

impl Drop for ThreadTrackingGuard {
    fn drop(&mut self) {
        TRACK_COORD_ALLOCS.store(false, Ordering::SeqCst);
        let _ = ON_MEASURED_THREAD.try_with(|c| c.set(false));
    }
}

fn measure_coord_allocs<R>(f: impl FnOnce() -> R) -> (R, usize, usize) {
    ON_MEASURED_THREAD.with(|c| c.set(true));
    COORD_ALLOC_COUNT.store(0, Ordering::SeqCst);
    COORD_ALLOC_BYTES.store(0, Ordering::SeqCst);
    TRACK_COORD_ALLOCS.store(true, Ordering::SeqCst);
    let _reset = ThreadTrackingGuard;
    let out = f();
    drop(_reset);
    (
        out,
        COORD_ALLOC_COUNT.load(Ordering::SeqCst),
        COORD_ALLOC_BYTES.load(Ordering::SeqCst),
    )
}

fn fixed_input() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(960.0, 540.0))),
        ..Default::default()
    }
}

#[test]
fn idle_frames_400k_borrowed_points_zero_series_clones_and_shared_handles() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let point_count = 400_002_usize;
    let borrowed: Vec<PlotPoint> = (0..point_count)
        .map(|i| {
            let x = i as f64 * 1e-4;
            PlotPoint::new(x, (x * 0.25).sin())
        })
        .collect();

    let ctx = egui::Context::default();
    let plot_id = Id::new("perf_400k_borrowed_plot");

    // Warm up frames 1 & 2 so PlotMemory and axis thickness stabilize.
    for _ in 0..2 {
        clear_last_plot_datasets();
        let _ = ctx.run(fixed_input(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = Plot::new(plot_id).id(plot_id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Borrowed 400k", PlotPoints::Borrowed(&borrowed)));
                });
            });
        });
    }

    let first_shared = get_last_plot_shared_datasets();
    assert_eq!(first_shared.len(), 1);
    assert_eq!(first_shared[0].1.len(), point_count);
    let before = PlotMemory::load(&ctx, plot_id)
        .expect("PlotMemory present")
        .runtime_cache
        .counters();
    assert_eq!(before.series_misses, 1);
    assert_eq!(before.fit_hits, 0);
    assert_eq!(before.fit_misses, 0);

    // Run 3 idle frames under the counting allocator.
    let (_, coord_allocs, coord_bytes) = measure_coord_allocs(|| {
        for _ in 0..3 {
            clear_last_plot_datasets();
            let _ = ctx.run(fixed_input(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let _ = Plot::new(plot_id).id(plot_id).show(ui, |plot_ui| {
                        plot_ui.line(Line::new("Borrowed 400k", PlotPoints::Borrowed(&borrowed)));
                    });
                });
            });
        }
    });

    assert_eq!(
        coord_allocs, 0,
        "Expected zero full-series f64 coordinate allocations on idle frames, got {coord_allocs} ({coord_bytes} B)"
    );

    let after = PlotMemory::load(&ctx, plot_id)
        .expect("PlotMemory present")
        .runtime_cache
        .counters();
    assert_eq!(after.series_misses, before.series_misses);
    assert_eq!(after.series_hits, before.series_hits + 3);
    assert_eq!(after.poi_misses, before.poi_misses);
    assert_eq!(after.poi_hits, before.poi_hits + 3);

    let latest_shared = get_last_plot_shared_datasets();
    assert!(first_shared[0].1.ptr_eq(&latest_shared[0].1));
    let prims = get_last_plot_primitives();
    match &prims[0] {
        ExportPrimitive::Polyline { points, .. } => {
            assert!(first_shared[0].1.ptr_eq(points));
        }
        other => panic!("Expected Polyline primitive, got {other:?}"),
    }
}

#[test]
fn streaming_monotone_intersection_allocates_zero_merged_knots_buffer() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let n = 15_000_usize; // 2 * 15_000 * 8B = 240 KiB > 64 KiB if merged_knots were allocated
    let a: SharedPoints = (0..n)
        .map(|i| {
            let x = i as f64 * 0.001;
            [x, (x * 2.0).sin()]
        })
        .collect();
    let b: SharedPoints = (0..n)
        .map(|i| {
            let x = i as f64 * 0.001 + 0.0003;
            [x, (x * 2.0).cos() * 0.5]
        })
        .collect();
    let items: Vec<Box<dyn PlotItem>> =
        vec![Box::new(Line::new("A", a)), Box::new(Line::new("B", b))];
    let transform = PlotTransform::new(
        Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0)),
        PlotBounds::from_min_max([0.0, -2.0], [15.0, 2.0]),
        false,
    );

    let (pois, coord_allocs, coord_bytes) =
        measure_coord_allocs(|| detect_critical_points(&items, &transform));
    assert!(!pois.is_empty());
    assert_eq!(
        coord_allocs, 0,
        "Streaming monotone intersection should allocate zero >=64KiB knot vectors, got {coord_allocs} ({coord_bytes} B)"
    );
}

#[test]
fn fit_and_poi_caches_hit_when_idle_and_invalidate_on_series_bounds_or_log_changes() {
    let ctx = egui::Context::default();
    let plot_id = Id::new("memo_invalidation_plot");
    let mut series_data: Vec<[f64; 2]> = (1..=40)
        .map(|i| {
            let x = i as f64 * 0.25;
            [x, 1.5 * x + 0.5]
        })
        .collect();

    ctx.data_mut(|d| {
        d.insert_temp(
            plot_id,
            AccessState {
                fit_model: Some(FitType::Linear),
                show_fit_overlay: true,
                show_fit_residuals: true,
                ..Default::default()
            },
        );
    });

    let run_with_shared = |ctx: &egui::Context, shared: &SharedPoints, hline_y: f64, max_x: f64| {
        let _ = ctx.run(fixed_input(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = Plot::new(plot_id)
                    .id(plot_id)
                    .include_x(max_x)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("Linear Signal", shared));
                        plot_ui.hline(HLine::new("Ref", hline_y));
                    });
            });
        });
    };

    let shared_initial = SharedPoints::from_slice(&series_data);
    for _ in 0..2 {
        run_with_shared(&ctx, &shared_initial, 5.0, 12.0);
    }

    let c0 = PlotMemory::load(&ctx, plot_id)
        .unwrap()
        .runtime_cache
        .counters();

    // Idle frame: series_hits, fit_hits, and poi_hits all increment; zero new misses.
    run_with_shared(&ctx, &shared_initial, 5.0, 12.0);
    let c_idle = PlotMemory::load(&ctx, plot_id)
        .unwrap()
        .runtime_cache
        .counters();
    assert_eq!(c_idle.series_hits, c0.series_hits + 1);
    assert_eq!(c_idle.series_misses, c0.series_misses);
    assert_eq!(c_idle.fit_hits, c0.fit_hits + 1);
    assert_eq!(c_idle.fit_misses, c0.fit_misses);
    assert_eq!(c_idle.poi_hits, c0.poi_hits + 1);
    assert_eq!(c_idle.poi_misses, c0.poi_misses);

    // 1. Bounds change (`include_x` 12.0 -> 25.0): POI cache invalidates; FitResult model hits & resamples curve.
    run_with_shared(&ctx, &shared_initial, 5.0, 25.0);
    let c_bounds = PlotMemory::load(&ctx, plot_id)
        .unwrap()
        .runtime_cache
        .counters();
    assert_eq!(c_bounds.poi_misses, c_idle.poi_misses + 1);
    assert_eq!(c_bounds.fit_misses, c_idle.fit_misses);
    assert_eq!(c_bounds.fit_hits, c_idle.fit_hits + 1);

    // 2. Log flag change (`log_x = true`): invalidates both FitResult and CriticalPoint caches.
    if let Some(mut mem) = PlotMemory::load(&ctx, plot_id) {
        mem.log_x = true;
        mem.auto_bounds = Vec2b::new(true, true);
        mem.store(&ctx, plot_id);
    }
    run_with_shared(&ctx, &shared_initial, 5.0, 25.0);
    let c_log = PlotMemory::load(&ctx, plot_id)
        .unwrap()
        .runtime_cache
        .counters();
    assert_eq!(c_log.fit_misses, c_bounds.fit_misses + 1);
    assert_eq!(c_log.poi_misses, c_bounds.poi_misses + 1);

    // 3. Series mutation: invalidates series, fit, and POI caches.
    series_data[10][1] += 3.0;
    let shared_mutated = SharedPoints::from_slice(&series_data);
    run_with_shared(&ctx, &shared_mutated, 5.0, 25.0);
    let c_mut = PlotMemory::load(&ctx, plot_id)
        .unwrap()
        .runtime_cache
        .counters();
    assert_eq!(c_mut.series_misses, c_log.series_misses + 1);
    assert_eq!(c_mut.fit_misses, c_log.fit_misses + 1);
    assert_eq!(c_mut.poi_misses, c_log.poi_misses + 1);
}

#[test]
fn idle_frames_with_memoized_fit_poi_and_data_table_zero_series_clones() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let series: Vec<PlotPoint> = (0..50_000)
        .map(|i| {
            let x = i as f64 * 0.001;
            PlotPoint::new(x, 0.4 * x + (x * 0.5).sin())
        })
        .collect();
    let plot_id = Id::new("memo_fit_poi_datatable_alloc_plot");
    let ctx = egui::Context::default();
    ctx.data_mut(|d| {
        d.insert_temp(
            plot_id,
            AccessState {
                fit_model: Some(FitType::Linear),
                show_fit_overlay: true,
                show_fit_residuals: true,
                show_table: true,
                ..Default::default()
            },
        );
    });

    for _ in 0..3 {
        clear_last_plot_datasets();
        let _ = ctx.run(fixed_input(), |c| {
            egui::CentralPanel::default().show(c, |ui| {
                let _ = Plot::new(plot_id).id(plot_id).show(ui, |pui| {
                    pui.line(Line::new("Signal", PlotPoints::Borrowed(&series)));
                });
            });
        });
    }

    let (_, coord_allocs, coord_bytes) = measure_coord_allocs(|| {
        for _ in 0..5 {
            clear_last_plot_datasets();
            let _ = ctx.run(fixed_input(), |c| {
                egui::CentralPanel::default().show(c, |ui| {
                    let _ = Plot::new(plot_id).id(plot_id).show(ui, |pui| {
                        pui.line(Line::new("Signal", PlotPoints::Borrowed(&series)));
                    });
                });
            });
        }
    });
    assert_eq!(
        coord_allocs, 0,
        "Expected 0 >=64 KiB f64 allocations across 5 idle frames with Fit + Residuals + POI + DataTable active, got {coord_allocs} ({coord_bytes} B)"
    );
}

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{
    atlas_regime_color, atlas_scalar_color, clamp_atlas_dimensions, AtlasAxisPair, AtlasAxisSpec,
    AtlasCell, AtlasPassMeta, AtlasRenderRequest, AtlasRunner, ParameterAtlasController,
    RegimeCartographer, ATLAS_STRIDES, MAX_ATLAS_SIDE,
};
use crate::async_sim::SimulationRunner;
use eframe::egui;
use scientific_metadata::theory::ParameterConstraint;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

#[cfg(any(feature = "biology", feature = "epidemiology"))]
#[path = "domain_tests.rs"]
mod domain_tests;

pub(super) fn wcag_contrast_on_black(color: egui::Color32) -> f64 {
    let to_lin = |ch: u8| {
        let s = f64::from(ch) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let lum = 0.2126 * to_lin(color.r()) + 0.7152 * to_lin(color.g()) + 0.0722 * to_lin(color.b());
    (lum + 0.05) / 0.05
}

struct CountingCartographer {
    axes: AtlasAxisPair,
    eval_count: Arc<AtomicUsize>,
    bump_after: Option<(usize, Arc<AtomicU64>, u64)>,
    spin_per_eval: Duration,
}

impl CountingCartographer {
    fn new(eval_count: Arc<AtomicUsize>) -> Self {
        let axis = |key: &'static str, label: &'static str, max: f64| AtlasAxisSpec {
            key,
            label,
            constraint: ParameterConstraint {
                min: 0.0,
                max,
                step: 0.01,
            },
        };
        Self {
            axes: AtlasAxisPair {
                x: axis("p1", "Param X", 1.0),
                y: axis("p2", "Param Y", 2.0),
            },
            eval_count,
            bump_after: None,
            spin_per_eval: Duration::ZERO,
        }
    }
}

impl RegimeCartographer for CountingCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let prev = self.eval_count.fetch_add(1, Ordering::Relaxed);
        if let Some((limit, ref gen_atom, next_gen)) = self.bump_after {
            if prev + 1 == limit {
                gen_atom.store(next_gen, Ordering::Release);
            }
        }
        if !self.spin_per_eval.is_zero() {
            let t0 = crate::web_io::PlatformInstant::now();
            while t0.elapsed() < self.spin_per_eval {
                std::hint::spin_loop();
            }
        }
        let val = 0.5 * p1 + 0.25 * p2;
        AtlasCell {
            regime: u8::from(val >= 0.5),
            primary: val,
            secondary: p1 - p2,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        atlas_scalar_color(cell.primary, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        if regime == 0 {
            "Subcritical"
        } else {
            "Supercritical"
        }
    }

    fn primary_metric_label(&self) -> &'static str {
        "Synthetic Gain"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!("p1={p1:.2}, p2={p2:.2} -> {:.3}", cell.primary)
    }
}

#[test]
fn test_atlas_axis_spec_and_coord_projection_clamping() {
    assert_eq!(clamp_atlas_dimensions(0, 64), (0, 0));
    assert_eq!(clamp_atlas_dimensions(96, 64), (96, 64));
    let (cw, ch) = clamp_atlas_dimensions(512, 256);
    assert_eq!((cw, ch), (MAX_ATLAS_SIDE, MAX_ATLAS_SIDE / 2));

    let count = Arc::new(AtomicUsize::new(0));
    let axes = CountingCartographer::new(count).axes();
    assert!((axes.x.clamp(-5.0) - 0.0).abs() < 1e-12);
    assert!((axes.y.clamp(99.0) - 2.0).abs() < 1e-12);
    assert!((axes.x.clamp(f64::NAN) - 0.5).abs() < 1e-12);

    let (p1, p2) = axes.pixel_to_coords(10, 20, 32, 32);
    assert_eq!(axes.coords_to_pixel(p1, p2, 32, 32), (10, 20));
}

#[test]
fn test_progressive_strides_8_4_2_1_converge_with_coarse_sample_reuse() {
    let eval_count = Arc::new(AtomicUsize::new(0));
    let cart: Arc<dyn RegimeCartographer> =
        Arc::new(CountingCartographer::new(Arc::clone(&eval_count)));
    let latest_gen = Arc::new(AtomicU64::new(1));
    let req = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 32,
        height: 32,
        high_contrast: false,
        cartographer: Arc::clone(&cart),
    }));
    let mut runner = AtlasRunner::new(req, latest_gen);

    let expected = [
        (8, 16, false),
        (4, 64, false),
        (2, 256, false),
        (1, 1024, true),
    ];
    for (stride, total_evals, done) in expected {
        runner.step_without_budget();
        let snap = runner.get_snapshot();
        let meta = snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<AtlasPassMeta>())
            .expect("AtlasPassMeta attached");
        assert_eq!(meta.stride, stride);
        assert_eq!(meta.is_final, done);
        assert_eq!(eval_count.load(Ordering::Relaxed), total_evals);
    }

    let final_snap = runner.get_snapshot();
    let final_meta = final_snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<AtlasPassMeta>())
        .expect("final meta");
    for sy in 0..32 {
        for sx in 0..32 {
            let (p1, p2) = cart.axes().pixel_to_coords(sx, sy, 32, 32);
            assert_eq!(final_meta.cells[sy * 32 + sx], cart.evaluate_cell(p1, p2));
        }
    }
}

#[test]
fn test_atomic_generation_bump_cancels_inflight_pass_mid_slice() {
    let eval_count = Arc::new(AtomicUsize::new(0));
    let latest_gen = Arc::new(AtomicU64::new(1));
    let mut cart = CountingCartographer::new(Arc::clone(&eval_count));
    cart.bump_after = Some((20, Arc::clone(&latest_gen), 2));
    let cart_arc: Arc<dyn RegimeCartographer> = Arc::new(cart);
    let shared = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 128,
        height: 128,
        high_contrast: false,
        cartographer: Arc::clone(&cart_arc),
    }));
    let mut runner = AtlasRunner::new(Arc::clone(&shared), Arc::clone(&latest_gen));
    shared.write().unwrap().generation = 2;

    runner.step();
    let calls = eval_count.load(Ordering::Relaxed);
    assert!(
        calls <= 48,
        "stale 8x pass (256 cells) should abort at 16-sample check after generation bump, got {calls}"
    );
    let snap = runner.get_snapshot();
    let meta = snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<AtlasPassMeta>())
        .expect("meta");
    assert_eq!(meta.generation, 2);
    assert!(!meta.is_final);
}

#[test]
fn test_cooperative_slice_budget_yields_and_resumes_across_steps() {
    let eval_count = Arc::new(AtomicUsize::new(0));
    let mut cart = CountingCartographer::new(Arc::clone(&eval_count));
    cart.spin_per_eval = Duration::from_micros(600);
    let latest_gen = Arc::new(AtomicU64::new(1));
    let shared = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 16,
        height: 16,
        high_contrast: false,
        cartographer: Arc::new(cart),
    }));
    let mut runner = AtlasRunner::new(shared, latest_gen);

    runner.step();
    assert!(
        !runner.is_complete(),
        "600us * 256 samples > 8ms slice budget must yield cooperatively before 1x completion"
    );
    for _ in 0..40 {
        if runner.is_complete() {
            break;
        }
        runner.step();
    }
    assert!(runner.is_complete());
    assert_eq!(eval_count.load(Ordering::Relaxed), 256);
}

#[test]
fn test_atlas_runner_zero_heap_allocations_under_verified_engine() {
    let eval_count = Arc::new(AtomicUsize::new(0));
    let latest_gen = Arc::new(AtomicU64::new(1));
    let shared = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 32,
        height: 32,
        high_contrast: true,
        cartographer: Arc::new(CountingCartographer::new(eval_count)),
    }));
    let mut runner = AtlasRunner::new(shared, latest_gen);

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        for _ in 0..ATLAS_STRIDES.len() {
            runner.step();
        }
    });
    assert!(runner.is_complete());
}

#[test]
fn test_high_contrast_atlas_colors_meet_wcag_3_to_1_on_black() {
    for i in 0..=100 {
        let t = f64::from(i) / 100.0;
        let c = atlas_scalar_color(t, true);
        let ratio = wcag_contrast_on_black(c);
        assert!(
            ratio >= 3.0,
            "HighContrast scalar color at t={t:.2} had contrast {ratio:.2} < 3.0"
        );
        for slot in 0..4 {
            let rc = atlas_regime_color(slot, 4, t, true);
            let r_ratio = wcag_contrast_on_black(rc);
            assert!(
                r_ratio >= 3.0,
                "HighContrast regime slot {slot} at t={t:.2} had contrast {r_ratio:.2} < 3.0"
            );
        }
    }
}

#[test]
fn test_controller_lazy_worker_visibility_and_aria_teleport() {
    let mut ctrl = ParameterAtlasController::new(vec![("p1", "p2"), ("p2", "p1")]);
    assert!(ctrl.has_default_persisted_state());
    assert!(!ctrl.is_visible());
    assert!(ctrl.set_axis_keys("unknown", "axis").is_err());
    ctrl.set_axis_keys("p2", "p1").expect("valid pair");
    assert!(!ctrl.has_default_persisted_state());
    ctrl.set_axis_keys("p1", "p2")
        .expect("restore default pair");

    let ctx = egui::Context::default();
    ctrl.set_visible(true);
    let count = Arc::new(AtomicUsize::new(0));
    let _ = ctx.run(egui::RawInput::default(), |egui_ctx| {
        egui::CentralPanel::default().show(egui_ctx, |ui| {
            let c_ref = Arc::clone(&count);
            let _ = ctrl.show_panel(ui, "test_atlas", 0.25, 0.75, 7, |_| {
                Arc::new(CountingCartographer::new(c_ref))
            });
        });
    });

    let ev = ctrl
        .teleport_to(Some(&ctx), 9.0, -3.0)
        .expect("teleport event");
    assert_eq!((ev.x_key, ev.y_key), ("p1", "p2"));
    assert!((ev.x_val - 1.0).abs() < 1e-12);
    assert!((ev.y_val - 0.0).abs() < 1e-12);
    let announced = crate::accessibility::with_live_region(|r| r.slots[r.active_slot].clone());
    assert!(announced.contains("Parameter atlas teleported"));
}

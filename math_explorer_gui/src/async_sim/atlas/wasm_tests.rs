#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{
    atlas_regime_color, atlas_scalar_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasPassMeta,
    AtlasRenderRequest, AtlasRunner, ParameterAtlasController, RegimeCartographer, ATLAS_STRIDES,
};
use crate::async_sim::SimulationRunner;
use crate::framework::state_bounds::ToolStateBounds;
use crate::framework::InteractiveTool;
use eframe::egui;
use scientific_metadata::theory::ParameterConstraint;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

struct WasmTestCartographer {
    evals: Arc<AtomicUsize>,
    cancel_gen: Option<(usize, Arc<AtomicU64>)>,
}

impl RegimeCartographer for WasmTestCartographer {
    fn axes(&self) -> AtlasAxisPair {
        let spec = |key: &'static str, label: &'static str| AtlasAxisSpec {
            key,
            label,
            constraint: ParameterConstraint {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
        };
        AtlasAxisPair {
            x: spec("u", "Axis U"),
            y: spec("v", "Axis V"),
        }
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let idx = self.evals.fetch_add(1, Ordering::Relaxed);
        if let Some((limit, ref atom)) = self.cancel_gen {
            if idx + 1 == limit {
                atom.store(2, Ordering::Release);
            }
        }
        AtlasCell {
            regime: u8::from(p1 + p2 > 1.0),
            primary: (0.5 * (p1 + p2)).clamp(0.0, 1.0),
            secondary: p1 - p2,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        atlas_scalar_color(cell.primary, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        if regime == 0 {
            "Low"
        } else {
            "High"
        }
    }

    fn primary_metric_label(&self) -> &'static str {
        "Sum"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!("{p1:.2},{p2:.2}={:.2}", cell.primary)
    }
}

#[wasm_bindgen_test]
fn wasm_atlas_progressive_8_4_2_1_convergence_and_cancellation() {
    let evals = Arc::new(AtomicUsize::new(0));
    let latest_gen = Arc::new(AtomicU64::new(1));
    let req = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 32,
        height: 32,
        high_contrast: true,
        cartographer: Arc::new(WasmTestCartographer {
            evals: Arc::clone(&evals),
            cancel_gen: None,
        }),
    }));
    let mut runner = AtlasRunner::new(Arc::clone(&req), Arc::clone(&latest_gen));
    for expected_stride in ATLAS_STRIDES {
        while !runner.is_complete() {
            runner.step();
            let snap = runner.get_snapshot();
            let meta = snap
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<AtlasPassMeta>())
                .expect("pass meta");
            if meta.stride == expected_stride {
                break;
            }
        }
    }
    while !runner.is_complete() {
        runner.step();
    }
    assert_eq!(evals.load(Ordering::Relaxed), 1024);

    let cancel_evals = Arc::new(AtomicUsize::new(0));
    let cancel_atom = Arc::new(AtomicU64::new(1));
    let cancel_req = Arc::new(RwLock::new(AtlasRenderRequest {
        generation: 1,
        width: 128,
        height: 128,
        high_contrast: false,
        cartographer: Arc::new(WasmTestCartographer {
            evals: Arc::clone(&cancel_evals),
            cancel_gen: Some((18, Arc::clone(&cancel_atom))),
        }),
    }));
    let mut cancel_runner = AtlasRunner::new(Arc::clone(&cancel_req), Arc::clone(&cancel_atom));
    cancel_req.write().unwrap().generation = 2;
    cancel_runner.step();
    assert!(cancel_evals.load(Ordering::Relaxed) <= 48);
}

#[wasm_bindgen_test]
fn wasm_atlas_high_contrast_wcag_and_aria_live_teleport() {
    crate::accessibility::init_accessibility_bridge();
    let to_lin = |ch: u8| {
        let s = f64::from(ch) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    for step in 0..=40 {
        let t = f64::from(step) / 40.0;
        let c = atlas_regime_color((step % 4) as u8, 4, t, true);
        let lum = 0.2126 * to_lin(c.r()) + 0.7152 * to_lin(c.g()) + 0.0722 * to_lin(c.b());
        let ratio = (lum + 0.05) / 0.05;
        assert!(ratio >= 3.0);
    }

    let mut ctrl = ParameterAtlasController::new(vec![("u", "v")]);
    ctrl.set_visible(true);
    let ctx = egui::Context::default();
    let evals = Arc::new(AtomicUsize::new(0));
    let _ = ctx.run(egui::RawInput::default(), |egui_ctx| {
        egui::CentralPanel::default().show(egui_ctx, |ui| {
            let e_ref = Arc::clone(&evals);
            let _ = ctrl.show_panel(ui, "wasm_atlas", 0.2, 0.8, 1, |_| {
                Arc::new(WasmTestCartographer {
                    evals: e_ref,
                    cancel_gen: None,
                })
            });
        });
    });
    let ev = ctrl.teleport_to(None, 0.75, 0.35).expect("wasm teleport");
    assert!((ev.x_val - 0.75).abs() < 1e-12);
    let live = crate::accessibility::with_live_region(|r| r.slots[r.active_slot].clone());
    assert!(live.contains("Parameter atlas teleported"));
}

#[cfg(all(feature = "biology", feature = "epidemiology"))]
#[wasm_bindgen_test]
fn wasm_flagship_state_bounds_and_washout_reseed() {
    use crate::async_sim::SimulationRunner;
    use crate::tabs::epidemiology::sir::SirTool;
    use crate::tabs::morphogenesis::{grid_u_variance, MorphogenesisTool};

    let mut sir = SirTool::default();
    sir.atlas.set_visible(true);
    let sir_json = sir.save_state().expect("sir json");
    ToolStateBounds::DEFAULT.validate_json(&sir_json).unwrap();

    let morpho = MorphogenesisTool::default();
    {
        let mut p = morpho.params.write().unwrap();
        p.a = 0.6;
        p.b = 0.2;
        p.d_v = 40.0;
        p.dt = 0.002;
    }
    let mut runner = MorphogenesisTool::create_runner(&morpho.params, 20, 20);
    runner.system.u_mut().fill(0.8);
    runner.system.v_mut().fill(0.2 / 0.64);
    runner.step();
    assert!(grid_u_variance(runner.system.u()) < 1e-8);
    {
        let mut p = morpho.params.write().unwrap();
        p.a = 0.1;
        p.b = 0.9;
    }
    for _ in 0..60 {
        runner.step();
    }
    assert!(grid_u_variance(runner.system.u()) > 1e-4);
}

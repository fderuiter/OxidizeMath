use super::billiards_tool::HyperbolicBilliardsTool;
use super::disk_tool::HyperbolicDiskTool;
use super::sync_wasm_hyperbolic_route;
use crate::app::MathExplorerApp;
use crate::framework::InteractiveTool;
use math_explorer::num_complex::Complex64;
use math_explorer::pure_math::differential_geometry::hyperbolic::{
    HyperbolicModel, POINCARE_DISK_RADIUS_MAX,
};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_hyperbolic_url_hydration_and_mobius_pan_invariants() {
    let doc = web_sys::window()
        .and_then(|win| win.document())
        .expect("browser window document must exist");
    let created_canvas = if doc.get_element_by_id("the_canvas_id").is_none() {
        let canvas = doc
            .create_element("canvas")
            .expect("canvas element creation must succeed");
        canvas
            .set_attribute("id", "the_canvas_id")
            .expect("id attribute must be set");
        if let Some(body) = doc.body() {
            let _ = body.append_child(&canvas);
        }
        Some(canvas)
    } else {
        None
    };

    let disk_hash = sync_wasm_hyperbolic_route(Some("HyperbolicDiskTool"));
    assert_eq!(
        disk_hash,
        "#tab=hyperbolic_geometry&tool=HyperbolicDiskTool"
    );
    if let Some(canvas) = doc.get_element_by_id("the_canvas_id") {
        assert_eq!(
            canvas.get_attribute("data-hyperbolic-route").as_deref(),
            Some("#tab=hyperbolic_geometry&tool=HyperbolicDiskTool")
        );
    }
    if let Some(canvas) = created_canvas {
        canvas.remove();
    }

    let mut app = MathExplorerApp::default();
    let hydrated = app
        .apply_share_location_strings(&disk_hash, "", None)
        .expect("URL hydration for HyperbolicDiskTool must succeed");
    assert!(hydrated);
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "hyperbolic_geometry");
    assert_eq!(
        app.tabs[app.selected_tab].active_tool_id(),
        Some("HyperbolicDiskTool")
    );

    let mut disk = HyperbolicDiskTool::default();
    for step in 0..40_usize {
        let ang = (step as f64) * 0.15;
        disk.apply_mobius_pan(Complex64::new(0.05 * ang.cos(), 0.05 * ang.sin()));
    }
    disk.model = HyperbolicModel::Klein;
    assert!(disk.has_finite_parameters());
    for poly in disk.tiling.polygons() {
        for &v in poly.active_vertices() {
            assert!(v.norm() <= POINCARE_DISK_RADIUS_MAX);
        }
    }
}

#[wasm_bindgen_test]
fn wasm_hyperbolic_billiards_lyapunov_divergence_and_url_roundtrip() {
    assert_eq!(sync_wasm_hyperbolic_route(None), "#tab=hyperbolic_geometry");
    let billiards_hash = sync_wasm_hyperbolic_route(Some("HyperbolicBilliardsTool"));
    let mut app = MathExplorerApp::default();
    assert!(app
        .apply_share_location_strings(&billiards_hash, "", None)
        .expect("URL hydration for HyperbolicBilliardsTool must succeed"));
    assert_eq!(
        app.tabs[app.selected_tab].active_tool_id(),
        Some("HyperbolicBilliardsTool")
    );

    let mut tool = HyperbolicBilliardsTool::default();
    for _ in 0..50 {
        tool.step_simulation();
    }
    assert!(tool.system.primary_bounces() >= 2);
    let latest = tool.system.latest_sample();
    assert!(latest.log_ratio.is_finite() && latest.log_ratio > 0.5);
}

// [cite:modular_polynomials_review]

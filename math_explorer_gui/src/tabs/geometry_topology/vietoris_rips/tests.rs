#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::state::{SculptBrush, VrSourceMode};
use super::*;
use egui_plot::PlotPoints;

#[test]
fn test_vietoris_rips_default_initialization() {
    let tool = VietorisRipsTool::default();
    assert_eq!(tool.mode, VrSourceMode::Points2D);
    assert_eq!(tool.cached_vertices.len(), tool.points.len());
    assert_eq!(tool.cached_vertices.len(), 20);
    assert_eq!(tool.cached_edges.len(), tool.complex.count_simplices(1));
    assert_eq!(tool.cached_triangles.len(), tool.complex.count_simplices(2));
    assert!(!tool.barcode_intervals.is_empty());
    assert!(tool.error_msg.is_none());
}

#[test]
fn test_vietoris_rips_parameter_updates() {
    let mut tool = VietorisRipsTool {
        radius: 15.0,
        ..Default::default()
    };

    tool.update_complex();
    assert_eq!(tool.cached_edges.len(), tool.complex.count_simplices(1));
    assert_eq!(tool.cached_triangles.len(), tool.complex.count_simplices(2));
    assert_eq!(tool.cached_vertices.len(), tool.points.len());
    assert!(!tool.cached_edges.is_empty());
    assert!(!tool.cached_triangles.is_empty());
    assert_eq!(tool.beta_0, 1);
    assert_eq!(tool.beta_1, 0);

    tool.radius = 0.0;
    tool.update_complex();
    assert_eq!(tool.cached_edges.len(), 0);
    assert_eq!(tool.cached_triangles.len(), 0);
    assert_eq!(tool.cached_vertices.len(), tool.points.len());
    assert_eq!(tool.beta_0, 20);
}

#[test]
fn test_vietoris_rips_error_state_cache_clearing() {
    let mut tool = VietorisRipsTool::default();
    assert!(!tool.cached_vertices.is_empty());

    tool.radius = -1.0;
    tool.update_complex();
    assert!(tool.error_msg.is_some());
    assert!(tool.cached_triangles.is_empty());
    assert!(tool.cached_edges.is_empty());
    assert!(tool.cached_vertices.is_empty());

    let mut tool2 = VietorisRipsTool::default();
    tool2.points.clear();
    tool2.update_complex();
    assert!(tool2.error_msg.is_some());
    assert!(tool2.cached_triangles.is_empty());
    assert!(tool2.cached_edges.is_empty());
    assert!(tool2.cached_vertices.is_empty());
}

#[test]
fn test_vietoris_rips_borrowed_plot_points_slice_variants() {
    let mut tool = VietorisRipsTool {
        radius: 15.0,
        ..Default::default()
    };
    tool.update_complex();

    let v_slice: &[PlotPoint] = &tool.cached_vertices[..];
    let v_plot: PlotPoints<'_> = v_slice.into();
    assert!(matches!(v_plot, PlotPoints::Borrowed(_)));
    assert_eq!(v_plot.points().len(), tool.cached_vertices.len());

    let edge = tool.cached_edges.first().expect("edges non-empty");
    let e_plot: PlotPoints<'_> = (&edge[..]).into();
    assert!(matches!(e_plot, PlotPoints::Borrowed(_)));
    assert_eq!(e_plot.points().len(), 2);

    let tri = tool.cached_triangles.first().expect("triangles non-empty");
    let t_plot: PlotPoints<'_> = (&tri[..]).into();
    assert!(matches!(t_plot, PlotPoints::Borrowed(_)));
    assert_eq!(t_plot.points().len(), 3);
}

#[test]
fn test_vietoris_rips_try_load_state_bounds_and_64_point_cap() {
    let mut tool = VietorisRipsTool::default();
    let before = tool.complex_builds;

    assert!(tool
        .try_load_state(r#"{"radius":1.5,"points":[]}"#)
        .is_err());
    assert!(tool
        .try_load_state(r#"{"radius":1.5,"points":[[0.0,"NaN"]]}"#)
        .is_err());
    let over_64 = serde_json::json!({
        "radius": 2.0,
        "points": (0..(MAX_VIETORIS_RIPS_POINTS + 1)).map(|i| [i as f64 * 0.1, 0.0]).collect::<Vec<_>>()
    });
    assert!(tool.try_load_state(&over_64.to_string()).is_err());
    assert!(tool
        .try_load_state(r#"{"mode":"points_3d","radius":4.0,"points_3d":[[0.0,0.0,0.0]],"camera":{"pitch":0.0,"yaw":0.0,"zoom":0.0}}"#)
        .is_err());
    assert_eq!(tool.complex_builds, before);

    let exact_64 = serde_json::json!({
        "radius": 1.2,
        "points": (0..MAX_VIETORIS_RIPS_POINTS).map(|i| {
            let a = std::f64::consts::TAU * (i as f64 / MAX_VIETORIS_RIPS_POINTS as f64);
            [5.0 + 4.0 * a.cos(), 5.0 + 4.0 * a.sin()]
        }).collect::<Vec<_>>()
    });
    assert!(tool.try_load_state(&exact_64.to_string()).is_ok());
    assert_eq!(tool.complex_builds, before + 1);
    assert_eq!(tool.points.len(), 64);
    assert_eq!(tool.beta_1, 1);
}

#[test]
fn test_vietoris_rips_3d_lorenz_and_takens_sweeps() {
    let mut tool = VietorisRipsTool::default();
    let lorenz_cfg = r#"{"mode":"lorenz_3d","radius":7.5,"chaos":{"sigma":10.0,"rho":28.0,"beta":2.6666666666666665,"dt":0.01,"landmark_count":64,"takens_lag":8},"camera":{"pitch":0.35,"yaw":0.55,"zoom":1.0}}"#;
    tool.try_load_state(lorenz_cfg).expect("load 3d lorenz");
    assert_eq!(tool.mode, VrSourceMode::Lorenz3D);
    assert_eq!(tool.points_3d.len(), 64);
    assert!(!tool.barcode_intervals.is_empty());

    // Sweep epsilon across butterfly wing scale and verify beta_1 == 2 emerges.
    let mut saw_two_wings = false;
    for step in 12..=22 {
        let r = step as f64 * 0.5;
        let (b0, b1) = tool.recompute_filtration_zero_alloc(r).unwrap();
        if b0 == 1 && b1 == 2 {
            saw_two_wings = true;
            break;
        }
    }
    assert!(
        saw_two_wings,
        "Lorenz 3D landmarks must lock onto beta_1 = 2"
    );

    // Sweep Takens lag tau from collapsed diagonal (tau = 1) to unfolded attractor (tau = 10).
    let takens_cfg = r#"{"mode":"takens_3d","radius":7.0,"chaos":{"sigma":10.0,"rho":28.0,"beta":2.6666666666666665,"dt":0.01,"landmark_count":64,"takens_lag":1},"camera":{"pitch":0.45,"yaw":-0.6,"zoom":1.0}}"#;
    tool.try_load_state(takens_cfg).expect("load takens");
    let max_pers_tau1 = tool
        .barcode_intervals
        .iter()
        .filter(|iv| iv.dimension == 1)
        .map(|iv| iv.persistence())
        .fold(0.0_f64, f64::max);
    tool.chaos.takens_lag = 10;
    tool.rebuild_chaos_landmarks().unwrap();
    let max_pers_tau10 = tool
        .barcode_intervals
        .iter()
        .filter(|iv| iv.dimension == 1)
        .map(|iv| iv.persistence())
        .fold(0.0_f64, f64::max);
    assert!(max_pers_tau10 > max_pers_tau1);
}

#[test]
fn test_vietoris_rips_2d_and_3d_sculpt_brushes_and_bounds() {
    let mut tool = VietorisRipsTool::default();
    let square = r#"{"radius":1.2,"points":[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]}"#;
    tool.try_load_state(square).unwrap();
    assert_eq!(tool.beta_1, 1);

    // Insert a center bridge vertex that triangulates and fills the 1-cycle!
    assert!(tool.apply_sculpt_2d(SculptBrush::Add, [0.5, 0.5], 0.5, false));
    assert_eq!(tool.points.len(), 5);
    assert_eq!(tool.beta_1, 0);

    // Grab center vertex near [0.5, 0.5] and drag outside the square so the hole re-opens.
    assert!(tool.apply_sculpt_2d(SculptBrush::Move, [0.52, 0.52], 0.5, false));
    assert!(tool.apply_sculpt_2d(SculptBrush::Move, [5.0, 5.0], 0.5, true));
    assert_eq!(tool.beta_1, 1);

    // Erase never drops below 1 vertex.
    for _ in 0..10 {
        let p = [tool.points[0].x, tool.points[0].y];
        let _ = tool.apply_sculpt_2d(SculptBrush::Erase, p, 2.0, false);
    }
    assert_eq!(tool.points.len(), 1);

    // Verify 3D projection/unprojection roundtrip and 3D brush sculpting.
    let pts3 = vec![
        Point3D::new(0.0, 0.0, 0.0),
        Point3D::new(4.0, 0.0, 0.0),
        Point3D::new(4.0, 4.0, 0.0),
        Point3D::new(0.0, 4.0, 0.0),
    ];
    tool.load_validated_3d_points(pts3, 4.5).unwrap();
    assert_eq!(tool.beta_1, 1);
    let (proj0, depth0) = tool.project_3d_with_depth(tool.points_3d[0]);
    let roundtrip = tool.unproject_3d(proj0, depth0);
    assert!((roundtrip.x - tool.points_3d[0].x).abs() < 1e-9);
    assert!((roundtrip.y - tool.points_3d[0].y).abs() < 1e-9);
    assert!((roundtrip.z - tool.points_3d[0].z).abs() < 1e-9);
    assert!(tool.apply_sculpt_3d_projected(SculptBrush::Add, [0.0, 0.0], 1.5, false));
    assert_eq!(tool.points_3d.len(), 5);
}

#[test]
fn test_vietoris_rips_zero_allocation_hot_path_under_verified_allocator() {
    let mut tool = VietorisRipsTool {
        mode: VrSourceMode::Lorenz3D,
        radius: 7.5,
        ..Default::default()
    };
    tool.rebuild_chaos_landmarks().unwrap();

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((b0, _b1), _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        tool.recompute_filtration_zero_alloc(8.0)
            .expect("zero-alloc 3d filtration")
    });
    assert_eq!(b0, 1);
}

#[cfg(feature = "physics")]
#[test]
fn test_attractor_plotter_invariants_and_one_click_handoff_to_vietoris_rips() {
    let mut plotter = crate::tabs::chaos::attractors::AttractorPlotter::default();
    let lyap = plotter
        .current_lyapunov_exponent()
        .expect("finite lorenz_lyapunov");
    assert!(
        lyap > 0.2,
        "Standard chaotic Lorenz rho=28 must have positive lambda_1, got {lyap}"
    );
    let (c_eps, d2) = plotter
        .current_correlation_invariants()
        .expect("finite correlation invariants");
    assert!(c_eps.is_finite() && c_eps > 0.0);
    assert!(d2.is_finite() && d2 > 0.5);

    // Subcritical rho = 14 must yield negative Lyapunov exponent.
    plotter.system.rho = 14.0;
    let sub_lyap = plotter
        .current_lyapunov_exponent()
        .expect("finite subcritical lyapunov");
    assert!(
        sub_lyap < 0.0,
        "Subcritical Lorenz rho=14 must have negative lambda_1, got {sub_lyap}"
    );

    // Restore chaotic rho = 28 and execute one-click handoff into MathExplorerApp.
    plotter.system.rho = 28.0;
    let ctx = egui::Context::default();
    let t0 = crate::web_io::PlatformInstant::now();
    let payload = plotter
        .send_landmarks_to_vietoris_rips(Some(&ctx))
        .expect("handoff succeeds");
    assert_eq!(payload.points.len(), MAX_VIETORIS_RIPS_POINTS);

    let mut app = crate::app::MathExplorerApp::default();
    crate::app::domain_bar::poll_cross_tab_tool_handoffs(&mut app, &ctx);
    assert_eq!(app.tabs[app.selected_tab].name(), "Geometry & Topology");

    let mut vr = VietorisRipsTool::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| vr.show_inside(ui));
    });
    assert_eq!(vr.mode, VrSourceMode::Points3D);
    assert_eq!(vr.points_3d.len(), MAX_VIETORIS_RIPS_POINTS);
    assert!(vr.chaos.lyapunov_exponent().is_some_and(|v| v.is_finite()));
    assert!(vr.chaos.correlation_sum().is_some_and(|v| v.is_finite()));
    assert!(
        crate::web_io::PlatformInstant::now()
            .duration_since(t0)
            .as_secs_f64()
            < 5.0
    );
}

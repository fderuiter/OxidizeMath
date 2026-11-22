//! Unit and contract tests for [`super::LatticeBoltzmannTool`], `floor()` brush hits,
//! `625`-byte `base64url` obstacle masks, aerodynamic telemetry, and GIF wake animation.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::export_figure::render_tool_gif_with_fallback;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::InteractiveTool;
use crate::url_state::{
    encode_base64url, encode_share_fragment, parse_share_location, validate_url_preset_json,
    MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES,
};
use eframe::egui;
use egui_plot::export::ExportOptions;
use math_commons::math_kernel::types::flatten_2d_index;

#[test]
fn test_floor_brush_hits_all_four_subcell_quadrants_and_clamps_bounds() {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(100.0, 50.0));
    // Target lattice cell (gx = 30, gy = 25), rendered at screen row py = 49 - 25 = 24 ([24.0, 25.0)).
    for (sx, sy) in [(30.05, 24.05), (30.95, 24.05), (30.05, 24.95), (30.95, 24.95)] {
        let mapped = screen_pos_to_cell(egui::pos2(sx, sy), rect);
        assert_eq!(
            mapped,
            Some((30, 25)),
            "Sub-cell pointer ({sx}, {sy}) must floor to exact cell (30, 25)"
        );
    }

    assert_eq!(lattice_coord_to_cell(-2.5, -0.1), Some((0, 0)));
    assert_eq!(lattice_coord_to_cell(105.7, 99.9), Some((99, 49)));
    assert_eq!(lattice_coord_to_cell(f64::NAN, 10.0), None);
}

#[test]
fn test_inlet_guard_brush_radii_and_solid_area_budget_cap() {
    let mut grid = vec![false; LBM_CELL_COUNT];
    // Radius 1 paints a single exact cell when x >= 2, and refuses x < 2.
    assert_eq!(apply_brush_with_budget(&mut grid, 0, 25, 1, true, 100), 0);
    assert_eq!(apply_brush_with_budget(&mut grid, 1, 25, 1, true, 100), 0);
    assert_eq!(apply_brush_with_budget(&mut grid, 2, 25, 1, true, 100), 1);
    assert!(grid[flatten_2d_index(2, 25, LBM_WIDTH)]);

    // Large radius near inlet never sets x < 2.
    let count = apply_brush_with_budget(&mut grid, 2, 25, 5, true, 500);
    assert!(count > 10);
    for y in 0..LBM_HEIGHT {
        assert!(!grid[flatten_2d_index(0, y, LBM_WIDTH)]);
        assert!(!grid[flatten_2d_index(1, y, LBM_WIDTH)]);
    }

    // Budget cap strictly prevents adding beyond `area_budget`, while erasing still works.
    let mut capped = vec![false; LBM_CELL_COUNT];
    let used = apply_brush_with_budget(&mut capped, 50, 25, 5, true, 25);
    assert_eq!(used, 25);
    assert_eq!(count_solid_cells(&capped), 25);
    let after_extra = apply_brush_with_budget(&mut capped, 70, 25, 5, true, 25);
    assert_eq!(after_extra, 25);
    let after_erase = apply_brush_with_budget(&mut capped, 50, 25, 3, false, 25);
    assert!(after_erase < 25);
}

#[test]
fn test_obstacle_mask_625_byte_834_char_roundtrip_and_strict_length_rejection() {
    let mut sculpted = streamlined_airfoil_obstacle_grid();
    apply_brush_with_budget(&mut sculpted, 60, 15, 2, true, DEFAULT_AREA_BUDGET);

    let encoded = encode_obstacle_mask(&sculpted);
    assert_eq!(encoded.len(), LBM_MASK_B64_CHARS);
    let decoded = decode_obstacle_mask(&encoded).expect("625-byte mask must decode");
    assert_eq!(decoded, sculpted);

    let short_b64 = encode_base64url(&[0_u8; 624]);
    assert!(decode_obstacle_mask(&short_b64).is_err());

    let long_b64 = encode_base64url(&[0_u8; 626]);
    assert!(decode_obstacle_mask(&long_b64).is_err());
}

#[test]
fn test_url_share_link_roundtrip_stays_within_bounds_and_legacy_presets_load() {
    let mut tool = LatticeBoltzmannTool::default();
    tool.try_load_state(r#"{"viscosity": 0.035}"#)
        .expect("legacy single-field viscosity config must remain backward-compatible");
    assert!((tool.viscosity - 0.035).abs() < 1e-12);
    assert_eq!(tool.obstacles, cylinder_obstacle_grid());

    // Sculpt a custom asymmetric obstacle and round-trip through the browser share-link codec.
    tool.view_mode = LbmViewMode::Vorticity;
    tool.area_budget = 220;
    tool.brush_radius = 3;
    tool.sculpt_at_cell(48, 22, true);
    let expected_mask = tool.obstacles.clone();

    let preset_json = tool
        .export_preset_json()
        .expect("export_preset_json should succeed");
    assert!(preset_json.len() <= MAX_DECODED_URL_STATE_BYTES);
    validate_url_preset_json(&preset_json).expect("URL JSON bounds must pass");

    let fragment = encode_share_fragment("fluid_dynamics", Some("LatticeBoltzmannTool"), Some(&preset_json))
        .expect("encode_share_fragment must fit within 8 KiB");
    assert!(fragment.len() <= MAX_URL_FRAGMENT_BYTES);

    let parsed = parse_share_location(&fragment, "")
        .expect("parse_share_location must succeed")
        .expect("share params present");
    let mut restored = LatticeBoltzmannTool::default();
    restored
        .import_preset_json(parsed.state_json.as_deref().unwrap())
        .expect("restored tool must import URL state");
    assert_eq!(restored.obstacles, expected_mask);
    assert_eq!(restored.view_mode, LbmViewMode::Vorticity);
    assert_eq!(restored.area_budget, 220);
}

#[test]
fn test_aerodynamics_zero_guards_and_streamlined_beats_bluff_plate_drag() {
    let empty = vec![false; LBM_CELL_COUNT];
    let zero_h = compute_aerodynamics((1.5, -0.4), 0.10, 0.02, &empty, 10);
    assert_eq!(zero_h.cd, 0.0);
    assert_eq!(zero_h.cl, 0.0);

    let cyl = cylinder_obstacle_grid();
    let zero_u = compute_aerodynamics((1.5, -0.4), 0.0, 0.02, &cyl, 10);
    assert_eq!(zero_u.cd, 0.0);
    assert_eq!(zero_u.cl, 0.0);

    let visc = 0.015;
    let u0 = 0.10;
    let bluff_mask = bluff_plate_obstacle_grid();
    let airfoil_mask = streamlined_airfoil_obstacle_grid();
    let mut bluff_solver = build_wind_tunnel_solver(visc, u0, &bluff_mask);
    let mut airfoil_solver = build_wind_tunnel_solver(visc, u0, &airfoil_mask);

    let mut bluff_cd_sum = 0.0;
    let mut airfoil_cd_sum = 0.0;
    for step in 1..=200 {
        bluff_solver.set_inlet(0, 0, LBM_INLET_GUARD_COLS, LBM_HEIGHT, u0, 0.0);
        bluff_solver.step();
        airfoil_solver.set_inlet(0, 0, LBM_INLET_GUARD_COLS, LBM_HEIGHT, u0, 0.0);
        airfoil_solver.step();
        if step > 100 {
            bluff_cd_sum +=
                compute_aerodynamics(bluff_solver.net_obstacle_force(), u0, visc, &bluff_mask, step)
                    .cd;
            airfoil_cd_sum += compute_aerodynamics(
                airfoil_solver.net_obstacle_force(),
                u0,
                visc,
                &airfoil_mask,
                step,
            )
            .cd;
        }
    }
    let avg_bluff_cd = bluff_cd_sum / 100.0;
    let avg_airfoil_cd = airfoil_cd_sum / 100.0;
    assert!(
        avg_bluff_cd > avg_airfoil_cd && avg_airfoil_cd > 0.0,
        "Bluff-plate C_D ({avg_bluff_cd:.3}) must exceed streamlined airfoil C_D ({avg_airfoil_cd:.3})"
    );
}

#[test]
fn test_runner_finite_state_guard_halts_on_invalid_params_and_recovers() {
    let cfg = Arc::new(RwLock::new(LbmSharedConfig {
        viscosity: -0.5,
        inflow_speed: 0.10,
        steps_per_frame: 2,
        view_mode: LbmViewMode::Speed,
        high_contrast: false,
        obstacles: cylinder_obstacle_grid(),
        mask_generation: 0,
    }));
    let lat = Arc::new(RwLock::new(LatticeState::new(LBM_WIDTH, LBM_HEIGHT)));
    let mut runner = LbmRunner::new(Arc::clone(&cfg), lat);
    crate::async_sim::SimulationRunner::step(&mut runner);
    assert!(FiniteStateGuard::diverged(&runner));
    assert!(FiniteStateGuard::diagnostic_report(&runner).is_some());

    cfg.write().unwrap().viscosity = DEFAULT_VISCOSITY;
    FiniteStateGuard::reset_parameters(&mut runner);
    assert!(!FiniteStateGuard::diverged(&runner));
    assert!(FiniteStateGuard::diagnostic_report(&runner).is_none());
}

#[test]
fn test_multiframe_gif_export_retains_sculpted_obstacles_and_animates_wake() {
    let mut tool = LatticeBoltzmannTool::default();
    tool.apply_obstacle_preset(bluff_plate_obstacle_grid());
    tool.view_mode = LbmViewMode::Vorticity;
    tool.push_shared_config();
    tool.sculpt_at_cell(45, 25, true);

    let frame0 = tool.export_raster_image().expect("frame 0 raster");
    tool.step_simulation();
    let frame1 = tool.export_raster_image().expect("frame 1 raster");
    tool.step_simulation();
    let frame2 = tool.export_raster_image().expect("frame 2 raster");

    // Sculpted cell (gx = 45, gy = 25 -> py = 49 - 25 = 24) stays solid across frames.
    let solid_px_idx = flatten_2d_index(45, 24, LBM_WIDTH);
    assert_eq!(frame0.pixels[solid_px_idx], frame1.pixels[solid_px_idx]);
    assert_eq!(frame1.pixels[solid_px_idx], frame2.pixels[solid_px_idx]);
    // Fluid wake pixels evolve across frames.
    assert_ne!(
        frame0.pixels, frame2.pixels,
        "Stepped GIF frames must animate fluid wake around sculpted obstacles"
    );

    let opts = ExportOptions {
        width: 200,
        height: 100,
        ..Default::default()
    };
    let gif_bytes = render_tool_gif_with_fallback(&mut tool, &[], &opts, 4, 12)
        .expect("render_tool_gif_with_fallback must produce valid GIF");
    assert!(gif_bytes.starts_with(b"GIF89a") || gif_bytes.starts_with(b"GIF87a"));
    assert!(gif_bytes.len() > 256);
}

#[test]
fn test_export_wind_tunnel_verification_previews() {
    let visc = 0.012;
    let u0 = 0.11;
    let bluff_mask = bluff_plate_obstacle_grid();
    let airfoil_mask = streamlined_airfoil_obstacle_grid();
    let mut bluff = build_wind_tunnel_solver(visc, u0, &bluff_mask);
    let mut airfoil = build_wind_tunnel_solver(visc, u0, &airfoil_mask);
    let mut bluff_series = Vec::with_capacity(60);
    let mut airfoil_series = Vec::with_capacity(60);

    for step in 1..=300 {
        // Tiny initial vertical perturbation seeds classic Karman vortex shedding behind bluff plate.
        let uy_seed = if step <= 25 { 0.004 } else { 0.0 };
        bluff.set_inlet(0, 0, LBM_INLET_GUARD_COLS, LBM_HEIGHT, u0, uy_seed);
        bluff.step();
        airfoil.set_inlet(0, 0, LBM_INLET_GUARD_COLS, LBM_HEIGHT, u0, uy_seed);
        airfoil.step();
        if step >= 60 && step % 4 == 0 {
            let b_cd = compute_aerodynamics(bluff.net_obstacle_force(), u0, visc, &bluff_mask, step).cd;
            let a_cd =
                compute_aerodynamics(airfoil.net_obstacle_force(), u0, visc, &airfoil_mask, step).cd;
            bluff_series.push([step as f64, b_cd]);
            airfoil_series.push([step as f64, a_cd]);
        }
    }

    let scale_opts = ExportOptions::default();
    let bluff_img = egui_plot::export_primitives::scale_color_image(
        &render_lbm_color_image(&bluff, u0, LbmViewMode::Vorticity, true),
        800,
        400,
        &scale_opts,
    );
    let airfoil_img = egui_plot::export_primitives::scale_color_image(
        &render_lbm_color_image(&airfoil, u0, LbmViewMode::Speed, true),
        800,
        400,
        &scale_opts,
    );
    let bluff_png = egui_plot::export::color32_image_to_png(&bluff_img).expect("bluff PNG");
    let airfoil_png = egui_plot::export::color32_image_to_png(&airfoil_img).expect("airfoil PNG");
    let _ = std::fs::write("/tmp/lbm_bluff_plate_vorticity.png", &bluff_png);
    let _ = std::fs::write("/tmp/lbm_streamlined_airfoil_speed.png", &airfoil_png);

    let datasets = vec![
        ("Bluff Plate C_D(t)".to_string(), bluff_series),
        ("Streamlined Airfoil C_D(t)".to_string(), airfoil_series),
    ];
    let chart_png =
        egui_plot::export::export_custom_png(&datasets, &scale_opts).expect("C_D chart PNG");
    let _ = std::fs::write("/tmp/lbm_drag_telemetry_comparison.png", &chart_png);
    assert!(bluff_png.starts_with(b"\x89PNG") && chart_png.starts_with(b"\x89PNG"));
}

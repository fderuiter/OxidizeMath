//! Native unit, physics-coupling, export, and WCAG keyboard/pointer UI tests
//! for [`super::MriKSpaceScannerTool`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::phantom::{spin_echo_weight_from_bloch, synthesize_phantom, synthesize_shepp_logan};
use super::state::{
    decode_mask_hex, KSpaceBrushMode, KSpaceMaskFilter, KSpaceTrajectory, MriKSpaceScannerState,
    PhantomSource,
};
use super::trajectory::{build_parametric_mask, evaluate_bin_scanner_physics};
use super::{MriKSpaceScannerTool, KSPACE_CENTER, KSPACE_DIM, KSPACE_LEN};
use crate::async_sim::SimulationRunner;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use crate::tabs::mri::{MriRunner, MriSharedParams, MriTab};
use crate::tabs::ExplorerTab;
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::Vector3;
use std::sync::Arc;

fn test_raw_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 860.0),
        )),
        ..Default::default()
    }
}

fn find_accesskit_node_rect_and_id(
    output: &egui::FullOutput,
    target_label: &str,
) -> Option<(egui::Id, egui::Rect)> {
    let update = output.platform_output.accesskit_update.as_ref()?;
    for (nid, node) in &update.nodes {
        if node.label() == Some(target_label) {
            let b = node.bounds()?;
            let rect = egui::Rect::from_min_max(
                egui::pos2(b.x0 as f32, b.y0 as f32),
                egui::pos2(b.x1 as f32, b.y1 as f32),
            );
            return Some((egui::Id::new(nid.0), rect));
        }
    }
    None
}

#[test]
fn test_sibling_registration_and_quad_panel_rendering() {
    let mut tab = MriTab::default();
    assert_eq!(tab.active_tool_id(), Some("MriTool"));
    assert_eq!(tab.framework.available_tools.len(), 3);

    assert!(tab.select_tool_by_name("MriKSpaceScannerTool"));
    assert_eq!(tab.active_tool_id(), Some("MriKSpaceScannerTool"));

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut frame = eframe::Frame::_new_kittest();
    let output = ctx.run(test_raw_input(), |ctx| {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });
    for expected in [
        "Ground-Truth Spin-Echo MRI Phantom",
        "Interactive 2D k-Space Frequency Canvas",
        "Reconstructed 2D Inverse FFT Magnitude Image",
        "Spatial Reconstruction Absolute Error Heatmap",
    ] {
        assert!(
            find_accesskit_node_rect_and_id(&output, expected).is_some(),
            "Missing quad-panel AccessKit node '{expected}'"
        );
    }
}

#[test]
fn test_full_cartesian_64x64_accuracy_and_latency_budget() {
    let mut tool = MriKSpaceScannerTool::default();
    assert_eq!(tool.recon.sampled_bins, KSPACE_LEN);
    assert!(
        tool.recon.normalized_l2_error < 1e-5,
        "Full 64x64 reconstruction normalized L2 error {} exceeded 1e-5",
        tool.recon.normalized_l2_error
    );
    assert!(tool.recon.psnr_db >= 99.0);

    // Warm up FFT planner once, then verify stroke + recompute stays well below 16 ms.
    let _ = tool.paint_at_bin(KSPACE_CENTER, KSPACE_CENTER);
    tool.state.brush_mode = KSpaceBrushMode::Paint;
    let start = PlatformInstant::now();
    assert!(tool.paint_at_bin(KSPACE_CENTER, KSPACE_CENTER));
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert!(
        elapsed_ms < 16.0,
        "Single 64x64 k-space stroke + 2D FFT took {elapsed_ms:.2} ms (budget < 16 ms)"
    );
    assert!(tool.recon.normalized_l2_error < 1e-5);
}

#[test]
fn test_bloch_spin_echo_tr_te_compartment_contrast() {
    let short_tr = spin_echo_weight_from_bloch(1.0, 2.4, 0.32, 0.4, 0.04);
    let long_tr = spin_echo_weight_from_bloch(1.0, 2.4, 0.32, 3.5, 0.04);
    assert!(
        long_tr > short_tr * 2.0,
        "Long TR ({long_tr}) must recover substantially more longitudinal CSF signal than short TR ({short_tr})"
    );

    let short_te = spin_echo_weight_from_bloch(0.95, 0.35, 0.045, 2.0, 0.01);
    let long_te = spin_echo_weight_from_bloch(0.95, 0.35, 0.045, 2.0, 0.20);
    assert!(
        short_te > long_te * 5.0,
        "Long TE ({long_te}) must attenuate short-T2 tissue relative to short TE ({short_te})"
    );

    let pd_like = MriKSpaceScannerState {
        tr_s: 0.5,
        te_s: 0.015,
        ..Default::default()
    };
    let t2_weighted = MriKSpaceScannerState {
        tr_s: 4.0,
        te_s: 0.18,
        ..pd_like.clone()
    };
    assert_ne!(synthesize_shepp_logan(&pd_like), synthesize_shepp_logan(&t2_weighted));
}

#[test]
fn test_turing_phantom_and_lorenz_chaotic_vs_nyquist_aliasing() {
    let dc_phys = evaluate_bin_scanner_physics(KSPACE_CENTER, KSPACE_CENTER);
    assert!(dc_phys.k_mag_per_m < 1e-9);
    assert!(dc_phys.reference_phase_rad.abs() < 1e-9);

    let edge_phys = evaluate_bin_scanner_physics(KSPACE_CENTER, KSPACE_DIM - 1);
    assert!(edge_phys.k_mag_per_m > 100.0);
    assert!(edge_phys.reference_phase_rad.abs() > 1.0);

    let mut state = MriKSpaceScannerState {
        phantom: PhantomSource::TuringMorphogenesis,
        trajectory: KSpaceTrajectory::LorenzChaotic,
        lorenz_steps: 2200,
        turing_steps: 160,
        ..Default::default()
    };
    let phantom = synthesize_phantom(&state);
    assert_eq!(phantom.len(), KSPACE_LEN);
    assert!(phantom.iter().any(|&v| v > 0.25));

    let lorenz_mask = build_parametric_mask(&state);
    let mut quadrants = [0_usize; 4];
    for r in 0..KSPACE_DIM {
        for c in 0..KSPACE_DIM {
            if lorenz_mask[flatten_2d_index(c, r, KSPACE_DIM)]
                && r != KSPACE_CENTER
                && c != KSPACE_CENTER
            {
                let q = usize::from(r > KSPACE_CENTER) * 2 + usize::from(c > KSPACE_CENTER);
                quadrants[q] += 1;
            }
        }
    }
    assert!(quadrants.iter().all(|&count| count > 25));

    state.trajectory = KSpaceTrajectory::CartesianLines;
    state.cartesian_stride = 2;
    let cart_mask = build_parametric_mask(&state);
    assert_eq!(cart_mask.iter().filter(|&&b| b).count(), KSPACE_LEN / 2);
}

#[test]
fn test_keyhole_highpass_and_deterministic_seeded_random_masks() {
    let mut keyhole_tool = MriKSpaceScannerTool::default();
    keyhole_tool.state.mask_filter = KSpaceMaskFilter::Keyhole;
    keyhole_tool.state.cutoff_radius = 10;
    keyhole_tool.mark_parametric_mask_dirty();
    keyhole_tool.recompute_if_dirty(false);

    let mut highpass_tool = MriKSpaceScannerTool::default();
    highpass_tool.state.mask_filter = KSpaceMaskFilter::HighPass;
    highpass_tool.state.cutoff_radius = 10;
    highpass_tool.mark_parametric_mask_dirty();
    highpass_tool.recompute_if_dirty(false);

    assert!(
        keyhole_tool.recon.psnr_db > highpass_tool.recon.psnr_db + 6.0,
        "Keyhole DC preservation ({:.2} dB) must exceed center-erased HighPass ({:.2} dB)",
        keyhole_tool.recon.psnr_db,
        highpass_tool.recon.psnr_db
    );

    let mut rand_a = MriKSpaceScannerState {
        mask_filter: KSpaceMaskFilter::SeededRandom,
        random_keep_ratio: 0.4,
        rng_seed: 12345,
        ..Default::default()
    };
    let mask_a1 = build_parametric_mask(&rand_a);
    let mask_a2 = build_parametric_mask(&rand_a);
    assert_eq!(mask_a1, mask_a2);
    rand_a.rng_seed = 99999;
    assert_ne!(mask_a1, build_parametric_mask(&rand_a));
}

#[test]
fn test_tactile_pointer_brush_and_hex_mask_roundtrip() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = MriKSpaceScannerTool::default();
    tool.state.brush_mode = KSpaceBrushMode::Erase;
    tool.state.brush_radius = 5;

    let out1 = ctx.run(test_raw_input(), |ctx| tool.show(ctx));
    let (_, canvas_rect) =
        find_accesskit_node_rect_and_id(&out1, "Interactive 2D k-Space Frequency Canvas")
            .expect("interactive k-space canvas mounted");
    let center = canvas_rect.center();

    let mut click_in = test_raw_input();
    click_in.events = vec![
        egui::Event::PointerMoved(center),
        egui::Event::PointerButton {
            pos: center,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos: center,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(click_in, |ctx| tool.show(ctx));

    assert!(tool.recon.sampled_bins < KSPACE_LEN);
    assert!(tool.recon.psnr_db < 30.0);
    let hex = tool.state.custom_mask_hex.as_ref().expect("custom hex recorded");
    assert!(hex.starts_with("0x") && hex.len() == 1026);
    assert_eq!(decode_mask_hex(hex).unwrap(), tool.active_mask);

    let saved = tool.save_state().unwrap();
    let mut restored = MriKSpaceScannerTool::default();
    restored.try_load_state(&saved).unwrap();
    assert_eq!(restored.active_mask, tool.active_mask);
    assert!(restored.try_load_state(r#"{"unknown":1}"#).is_err());
}

#[test]
fn test_wcag_keyboard_reticle_arrow_navigation_and_live_announcements() {
    crate::accessibility::clear_test_announcements();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = MriKSpaceScannerTool::default();

    let out = ctx.run(test_raw_input(), |ctx| tool.show(ctx));
    assert!(
        find_accesskit_node_rect_and_id(&out, "Interactive 2D k-Space Frequency Canvas").is_some()
    );
    let canvas_id = tool.kspace_canvas_id.expect("kspace_canvas_id recorded");
    ctx.memory_mut(|m| m.request_focus(canvas_id));
    let _ = ctx.run(test_raw_input(), |ctx| tool.show(ctx));

    let mut nav_in = test_raw_input();
    nav_in.events = vec![
        egui::Event::Key {
            key: egui::Key::ArrowRight,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::Key {
            key: egui::Key::Space,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(nav_in, |ctx| tool.show(ctx));

    assert_eq!(tool.reticle_col, KSPACE_CENTER + 1);
    assert!(!tool.active_mask[flatten_2d_index(KSPACE_CENTER + 1, KSPACE_CENTER, KSPACE_DIM)]);
    assert_eq!(tool.focused_canvas_id(), Some(canvas_id));

    let msgs = crate::accessibility::all_announcements();
    assert!(msgs.iter().any(|(m, p)| p == "polite" && m.contains("PSNR")));
}

#[test]
fn test_three_starter_presets_and_plot_raster_exports() {
    let mut tab = MriTab::default();
    for (title, preset_json) in [
        (
            "Keyhole Contrast Preservation",
            super::theory::PRESET_MRI_KSPACE_KEYHOLE,
        ),
        (
            "Cartesian 2x Nyquist Ghosting",
            super::theory::PRESET_MRI_KSPACE_NYQUIST_GHOSTING,
        ),
        (
            "Lorenz Chaotic Compressed Sensing",
            super::theory::PRESET_MRI_KSPACE_LORENZ_CS,
        ),
    ] {
        tab.framework
            .apply_starter_card(title, Some("MriKSpaceScannerTool"), Some(preset_json))
            .unwrap();
        assert_eq!(tab.active_tool_id(), Some("MriKSpaceScannerTool"));
    }

    tab.framework.export_options.width = 256;
    tab.framework.export_options.height = 256;
    tab.framework.export_gif_frames = 4;
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    for fmt in [
        ExportFormat::Csv,
        ExportFormat::Json,
        ExportFormat::Png,
        ExportFormat::Svg,
        ExportFormat::Gif,
    ] {
        let _ = ctx.run(test_raw_input(), |ctx| {
            tab.framework.queue_export(fmt);
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| tab.show_inside(ui, &mut frame));
        });
        assert_eq!(tab.framework.export_status_severity, ExportStatusSeverity::Success);
    }
}

#[test]
fn test_mri_runner_zero_allocation_preserved_with_sibling_tool() {
    let params = Arc::new(std::sync::RwLock::new(MriSharedParams {
        t1: 1.0,
        t2: 0.2,
        b_field: Vector3::new(0.0, 0.0, 1.0e-7),
        time_scale: 2.0,
    }));
    let mut runner = MriRunner::new(params, Vector3::new(1.0, 0.0, 0.0), 1.0);
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        runner.step();
    });
}

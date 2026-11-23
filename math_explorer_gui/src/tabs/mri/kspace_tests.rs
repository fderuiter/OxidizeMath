#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::kspace_runner::MriKSpaceRunner;
use super::kspace_types::{
    KSpaceMaskPreset, MriKSpaceSharedParams, MriKSpaceToolState, PhantomPreset, StudioBrushMode,
    MAX_STUDIO_STROKES, PRESET_MRI_KSPACE_STUDIO,
};
use super::{MriKSpaceTool, MriTab};
use crate::async_sim::{SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::InteractiveTool;
use crate::tabs::ExplorerTab;
use crate::url_state::{
    encode_share_fragment, parse_share_location, validate_url_preset_json,
    MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES,
};
use eframe::egui;
use std::sync::{Arc, RwLock};

fn build_runner(cfg: MriKSpaceToolState) -> MriKSpaceRunner {
    let params = Arc::new(RwLock::new(MriKSpaceSharedParams {
        config: cfg,
        high_contrast: false,
    }));
    MriKSpaceRunner::new(params)
}

#[test]
fn test_mri_tab_registers_both_tools_and_preserves_legacy_state() {
    let mut tab = MriTab::default();
    assert_eq!(tab.active_tool_id(), Some("MriTool"));
    assert!(tab.select_tool_by_name("MriKSpaceStudio"));
    assert_eq!(tab.active_tool_id(), Some("MriKSpaceStudio"));

    tab.framework
        .active_tool
        .as_mut()
        .expect("studio active")
        .import_preset_json(PRESET_MRI_KSPACE_STUDIO)
        .expect("studio preset valid");

    // Legacy pre-framework state must restore MriTool even when Studio was active.
    let legacy_json = r#"{"paused":true,"time_scale":2.5}"#;
    tab.load_state(legacy_json);
    assert_eq!(tab.active_tool_id(), Some("MriTool"));
    let mri_state_str = tab
        .framework
        .active_tool
        .as_ref()
        .and_then(|t| t.save_state())
        .expect("MriTool state serialized");
    let mri_state: super::MriToolState = serde_json::from_str(&mri_state_str).unwrap();
    assert!((mri_state.time_scale - 2.5).abs() < 1e-9);
    assert_eq!(mri_state.state, Some(SimWorkerState::Ready));
}

#[test]
fn test_shepp_logan_and_turing_t1_vs_t2_contrast_inversion() {
    for preset in [
        PhantomPreset::SheppLogan,
        PhantomPreset::TuringSpots,
        PhantomPreset::TuringStripes,
    ] {
        let runner = build_runner(MriKSpaceToolState {
            phantom: preset,
            ..Default::default()
        });
        assert!(runner.recon_mag.iter().any(|&v| v > 0.2));
    }

    let t1w = build_runner(MriKSpaceToolState {
        phantom: PhantomPreset::TuringSpots,
        flip_angle_deg: 90.0,
        tr: 0.35,
        te: 0.010,
        ..Default::default()
    });
    let act_idx = (0..t1w.rho.len())
        .find(|&i| t1w.rho[i] > 0.1 && t1w.t1[i] < 0.75)
        .expect("activator-rich short-T1 voxel present");
    let inh_idx = (0..t1w.rho.len())
        .find(|&i| t1w.rho[i] > 0.1 && t1w.t1[i] > 1.75)
        .expect("inhibitor-rich long-T1 voxel present");
    assert!(
        t1w.recon_mag[act_idx] > t1w.recon_mag[inh_idx] * 1.4,
        "Short-TR/short-TE T1w must highlight short-T1 activator foci"
    );

    let t2w = build_runner(MriKSpaceToolState {
        phantom: PhantomPreset::TuringSpots,
        flip_angle_deg: 90.0,
        tr: 4.5,
        te: 0.14,
        ..Default::default()
    });
    assert!(
        t2w.recon_mag[inh_idx] > t2w.recon_mag[act_idx] * 1.4,
        "Long-TR/long-TE T2w must invert contrast in favor of long-T2 inhibitor matrix"
    );
}

#[test]
fn test_central_kspace_erasure_drops_interior_contrast_preserving_edges() {
    for phantom in [PhantomPreset::SheppLogan, PhantomPreset::TuringSpots] {
        let full = build_runner(MriKSpaceToolState {
            phantom,
            mask_preset: KSpaceMaskPreset::Full,
            ..Default::default()
        });
        let mut erased = build_runner(MriKSpaceToolState {
            phantom,
            mask_preset: KSpaceMaskPreset::Full,
            brush_mode: StudioBrushMode::KSpaceMaskErase,
            ..Default::default()
        });
        erased.process_command(SimCommand::ApplyBrush {
            cx: 32,
            cy: 32,
            r: 7,
            is_obstacle: true,
        });

        let n = full.size;
        let (mut full_int, mut erased_int, mut full_edge, mut erased_edge) = (0.0, 0.0, 0.0, 0.0);
        for r in 4..(n - 4) {
            for c in 4..(n - 4) {
                let idx = r * n + c;
                let grad_rho = (full.rho[idx + 1] - full.rho[idx - 1])
                    .hypot(full.rho[idx + n] - full.rho[idx - n]);
                if full.rho[idx] > 0.15 && grad_rho < 0.04 {
                    full_int += full.recon_mag[idx];
                    erased_int += erased.recon_mag[idx];
                } else if grad_rho > 0.16 {
                    full_edge += full.recon_mag[idx];
                    erased_edge += erased.recon_mag[idx];
                }
            }
        }
        assert!(erased_int < full_int * 0.55, "{phantom:?} interior drop");
        assert!(
            erased_edge / erased_int.max(1e-9) > (full_edge / full_int) * 1.8,
            "{phantom:?} edge-to-interior boost"
        );
    }
}

#[test]
fn test_kspace_rf_spike_induces_global_herringbone_striping() {
    let clean = build_runner(MriKSpaceToolState::default());
    let mut spiked = build_runner(MriKSpaceToolState {
        brush_mode: StudioBrushMode::KSpaceRfSpike,
        ..Default::default()
    });
    spiked.process_command(SimCommand::ApplyBrush {
        cx: 41,
        cy: 25,
        r: 1,
        is_obstacle: true,
    });

    let mut bg_clean_max = 0.0_f64;
    let mut bg_spiked_min = f64::INFINITY;
    let mut bg_spiked_max = 0.0_f64;
    for (i, &rho) in clean.rho.iter().enumerate() {
        if rho == 0.0 {
            bg_clean_max = bg_clean_max.max(clean.recon_mag[i]);
            bg_spiked_min = bg_spiked_min.min(spiked.recon_mag[i]);
            bg_spiked_max = bg_spiked_max.max(spiked.recon_mag[i]);
        }
    }
    assert!(bg_clean_max < 1e-6);
    assert!(bg_spiked_max > 0.25, "RF spike must modulate global image");
    let clean_interior_diff: Vec<f64> = (0..clean.rho.len())
        .filter(|&i| clean.rho[i] > 0.2)
        .map(|i| spiked.recon_mag[i] - clean.recon_mag[i])
        .collect();
    let min_d = clean_interior_diff
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_d = clean_interior_diff
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_d - min_d > 0.25,
        "Herringbone striping must oscillate across slice"
    );
}

#[test]
fn test_delta_b0_pocket_dropout_and_180_spin_echo_refocusing_rescue() {
    let mut runner = build_runner(MriKSpaceToolState {
        phantom: PhantomPreset::TuringSpots,
        te: 0.006,
        selected_voxel: [32, 32],
        brush_mode: StudioBrushMode::SpatialDeltaB0Pocket,
        ..Default::default()
    });
    let idx = 32 * runner.size + 32;
    let baseline_mag = runner.recon_mag[idx];
    assert!(baseline_mag > 0.25);

    runner.process_command(SimCommand::ApplyBrush {
        cx: 32,
        cy: 32,
        r: 4,
        is_obstacle: true,
    });
    // Advance 17 ticks from TE = 6 ms to tau = 40 ms so phase dispersion causes strong T2* dropout.
    for _ in 0..17 {
        runner.step();
    }
    let dephased_mag = runner.recon_mag[idx];
    let dephased_net_xy = runner.net_voxel_spin([32, 32]).xy().norm();
    assert!(
        dephased_mag < baseline_mag * 0.55,
        "ΔB0 pocket must suffer T2* dropout before refocusing ({dephased_mag} vs {baseline_mag})"
    );

    // Trigger 180° Hahn spin-echo pulse at t = tau = 40 ms, then advance 20 ticks to TE_echo = 2*tau = 80 ms.
    runner.process_command(SimCommand::Custom("refocus_180".to_string()));
    assert!(runner.echo_tau.is_some());
    for _ in 0..20 {
        runner.step();
    }
    let refocused_mag = runner.recon_mag[idx];
    let refocused_net_xy = runner.net_voxel_spin([32, 32]).xy().norm();
    assert!(
        refocused_mag > dephased_mag * 1.45 && refocused_net_xy > dephased_net_xy * 1.45,
        "180° refocusing must rescue T2* pocket signal ({refocused_mag} vs {dephased_mag})"
    );
}

#[test]
fn test_runner_step_is_zero_allocation_on_64_and_128_grids() {
    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    for size in [64_usize, 128] {
        let mut runner = build_runner(MriKSpaceToolState {
            grid_size: size,
            selected_voxel: [size / 2, size / 2],
            evolve_turing: size == 64,
            ..Default::default()
        });
        runner.step();
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            runner.step();
        });
    }
}

#[test]
fn test_ui_bloch_3d_viewport_brushes_and_compact_url_state() {
    let mut tool = MriKSpaceTool::default();
    for i in 0..(MAX_STUDIO_STROKES + 4) {
        tool.apply_brush_at_grid(StudioBrushMode::KSpaceMaskErase, (i % 64) as i32, 20);
    }
    assert_eq!(tool.config.strokes.len(), MAX_STUDIO_STROKES);

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            tool.show_inside(ui);
        });
    });
    assert!(tool.export_raster_image().is_some());
    assert!(!tool.export_datasets().is_empty());

    let preset_json = tool.export_preset_json().expect("preset JSON exported");
    assert!(preset_json.len() <= MAX_DECODED_URL_STATE_BYTES);
    validate_url_preset_json(&preset_json).expect("within 4 KiB URL bounds");
    let fragment =
        encode_share_fragment("mri-physics", Some("MriKSpaceStudio"), Some(&preset_json))
            .expect("share fragment encoded within 4 KiB budget");
    assert!(fragment.len() <= MAX_URL_FRAGMENT_BYTES);
    let parsed = parse_share_location(&fragment, "")
        .expect("valid fragment")
        .expect("share params parsed");
    assert_eq!(parsed.tool.as_deref(), Some("MriKSpaceStudio"));
    assert_eq!(parsed.state_json.as_deref(), Some(preset_json.as_str()));
}

#[test]
fn test_runner_pass_latency_budget_64_and_128() {
    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for size in [64_usize, 128] {
        let mut runner = build_runner(MriKSpaceToolState {
            grid_size: size,
            selected_voxel: [size / 2, size / 2],
            ..Default::default()
        });
        runner.step();
        let iters = 20_u32;
        let start = crate::web_io::PlatformInstant::now();
        for _ in 0..iters {
            runner.step();
        }
        let avg_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
        eprintln!("MriKSpaceRunner {size}x{size} average pass latency: {avg_us:.1} us");
        #[cfg(not(debug_assertions))]
        assert!(
            avg_us < 1000.0,
            "{size}x{size} pass took {avg_us:.1} us (exceeds 1000 us < 1 ms budget)"
        );
    }
}

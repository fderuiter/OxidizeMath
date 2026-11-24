#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::scene::{
    SplatLensConfig, SplatLensMode, VolumetricSplatPreset, CAMERA_DISTANCE, MAX_DYNAMIC_SPLATS,
    VIEWPORT_HALF_EXTENT,
};
use super::*;
use crate::volumetric_scene::lensing::project_and_rasterize_lensed_workspace;
use crate::volumetric_scene::{VolumetricSplatWorkspace, SPLAT_GRID_SIZE as VOL_SPLAT_GRID_SIZE};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

#[test]
fn test_lensed_accretion_disk_workspace_zero_alloc_verified() {
    let (scene, _grads) = super::scene::build_preset_scene_and_gradients(
        VolumetricSplatPreset::RelativisticAccretionDisk,
    );
    assert!(
        scene.gaussians.len() <= MAX_DYNAMIC_SPLATS && scene.gaussians.len() >= 200,
        "Expected bounded accretion disk splats <= {MAX_DYNAMIC_SPLATS}, got {}",
        scene.gaussians.len()
    );
    let mut ws = VolumetricSplatWorkspace::with_capacity(MAX_DYNAMIC_SPLATS);
    assert_eq!(
        ws.pixel_buffer.len(),
        VOL_SPLAT_GRID_SIZE * VOL_SPLAT_GRID_SIZE
    );
    ws.scene = scene;
    let total_src = ws.scene.gaussians.len();
    let cam = Camera3D::new(0.42, 0.55, 1.0);
    let lens = SplatLensConfig {
        mode: SplatLensMode::Schwarzschild,
        center: [0.0, 0.0],
        einstein_radius: 0.90,
        viewport_rs: 0.24,
        peak_orbital_beta: 0.48,
        ..Default::default()
    };

    let telemetry = {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (tel, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            project_and_rasterize_lensed_workspace(
                &mut ws,
                &cam,
                CAMERA_DISTANCE,
                VIEWPORT_HALF_EXTENT,
                [0.0, 0.0],
                1.0,
                64,
                &lens,
            )
        });
        assert!(metrics.function_calls > 0);
        tel
    };

    assert!(
        ws.projected_splats.len() > total_src,
        "Schwarzschild dual images must emit > {total_src} projected splats, got {}",
        ws.projected_splats.len()
    );
    assert!(ws.projected_splats.len() <= 2 * MAX_DYNAMIC_SPLATS);
    assert!(telemetry.christoffel_gamma_r00 > 0.0 && telemetry.christoffel_gamma_r00.is_finite());
    assert!((telemetry.shadow_radius - 1.5 * lens.viewport_rs).abs() < 1.0e-9);
}

#[test]
fn test_photon_sphere_shadow_einstein_clamp_and_beaming_extremes() {
    let mut tool = GaussianSplattingTool::default();
    tool.set_scene_preset(VolumetricSplatPreset::RelativisticAccretionDisk);
    tool.lens.mode = SplatLensMode::Schwarzschild;
    tool.lens.center = [0.0, 0.0];
    tool.lens.einstein_radius = 0.95;
    tool.lens.viewport_rs = 0.32;
    tool.lens.peak_orbital_beta = 0.95;

    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1080.0, 720.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |sub| tool.show(sub));

    assert!(
        tool.projected_splats.iter().all(|g| {
            g.mean.x.is_finite()
                && g.mean.y.is_finite()
                && g.opacity.is_finite()
                && g.conic[(0, 0)].is_finite()
        }),
        "All lensed dual Gaussian2D splats must remain strictly finite across 1.5 R_s & theta_E"
    );
    let max_color = tool
        .projected_splats
        .iter()
        .map(|g| g.color.x + g.color.y + g.color.z)
        .fold(0.0_f64, f64::max);
    let min_color = tool
        .projected_splats
        .iter()
        .map(|g| g.color.x + g.color.y + g.color.z)
        .fold(f64::INFINITY, f64::min);
    assert!(
        max_color > min_color * 1.25,
        "Approaching accretion-disk side must be Doppler/synchrotron boosted relative to receding side"
    );
}

#[test]
fn test_interactive_lens_drag_caustic_overlays_url_and_exports() {
    let mut tool = GaussianSplattingTool::default();
    tool.set_scene_preset(VolumetricSplatPreset::RelativisticAccretionDisk);
    tool.lens.mode = SplatLensMode::Schwarzschild;
    tool.lens.show_caustic_overlay = true;
    tool.drag_lens_to([0.42, -0.31]);
    assert!((tool.lens.center[0] - 0.42).abs() < 1.0e-9);
    assert!((tool.lens.center[1] + 0.31).abs() < 1.0e-9);

    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1080.0, 720.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |sub| tool.show(sub));

    let rn = tool
        .lens_telemetry
        .nfw_critical_radius
        .expect("NFW caustic");
    let rb = tool
        .lens_telemetry
        .burkert_critical_radius
        .expect("Burkert caustic");
    assert!(
        (rn - rb).abs() > 1.0e-3,
        "NFW cusp and Burkert core caustics must differ"
    );

    let saved = tool.save_state().expect("lensed state serializes");
    let mut restored = GaussianSplattingTool::default();
    restored
        .try_load_state(&saved)
        .expect("lensed state round-trips");
    assert_eq!(
        restored.scene_preset,
        VolumetricSplatPreset::RelativisticAccretionDisk
    );
    assert_eq!(restored.lens.mode, SplatLensMode::Schwarzschild);
    assert!((restored.lens.center[0] - 0.42).abs() < 1.0e-6);

    assert_valid_splat_exports(&tool);
}

fn assert_valid_splat_exports(tool: &GaussianSplattingTool) {
    let png = tool.export_lensed_png_bytes().expect("PNG export");
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n") && png.len() > 256);
    let svg = tool.export_lensed_svg_string().expect("SVG export");
    assert!(svg.contains("<svg") && svg.contains("Einstein Ring"));
    let csv = tool.export_lensed_csv_string().expect("CSV export");
    assert!(csv.starts_with(
        "index,lens_mode,mean_x,mean_y,depth,opacity,color_r,color_g,color_b,conic_xx,conic_xy,conic_yy,shadow_radius,nfw_crit_r,burkert_crit_r\n"
    ));
    assert!(csv.lines().count() > 200);
    let ply = tool
        .export_splat_ply_3d()
        .expect("PLY hook")
        .expect("PLY bytes");
    assert!(ply.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
    assert!(tool.export_raster_image().is_some());
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
fn wasm_lensed_gaussian_splatting_roundtrip_and_exports() {
    let mut tool = GaussianSplattingTool::default();
    tool.set_scene_preset(VolumetricSplatPreset::RelativisticAccretionDisk);
    tool.lens.mode = SplatLensMode::Schwarzschild;
    tool.drag_lens_to([0.25, 0.15]);

    let state = tool.save_state().expect("wasm save_state");
    let mut reload = GaussianSplattingTool::default();
    reload.try_load_state(&state).expect("wasm try_load_state");
    assert_eq!(reload.lens.mode, SplatLensMode::Schwarzschild);
    assert_valid_splat_exports(&reload);
}

#[test]
fn test_viewport_kpc_frame_roundtrip_critical_alignment_and_legacy_serde_default() {
    use super::scene::{kpc_to_viewport, viewport_to_kpc};
    use crate::volumetric_scene::lensing::{
        evaluate_lens_mode_outcome, DEFAULT_VIEWPORT_KPC_SCALE,
    };
    use math_explorer::constants::si::SOLAR_MASS;
    use math_explorer::physics::astrophysics::{
        halo_thin_lens_sample, tangential_critical_radius_kpc,
    };
    use math_explorer::physics::high_energy::general_relativity::SchwarzschildBlackHole;
    use math_scalars::ViewportLength;
    use nalgebra::{Point2, Vector2};

    let legacy_json = r#"{"mode":"nfw_cusp","center":[0.2,-0.1],"halo_log10_mass":11.4,"halo_scale_radius":0.55,"log10_sigma_crit":8.15}"#;
    let cfg: SplatLensConfig = serde_json::from_str(legacy_json).expect("legacy JSON deserializes");
    assert!((cfg.viewport_kpc_scale - DEFAULT_VIEWPORT_KPC_SCALE).abs() < 1.0e-12);

    let vp = ViewportLength::new(1.375);
    let kpc = viewport_to_kpc(vp, cfg.viewport_kpc_scale);
    assert!((kpc.get() - 1.375 * DEFAULT_VIEWPORT_KPC_SCALE).abs() < 1.0e-12);
    assert!((kpc_to_viewport(kpc, cfg.viewport_kpc_scale).get() - vp.get()).abs() < 1.0e-12);

    let (nfw, burk, sigma_crit) = cfg.matched_halos();
    let r_search = viewport_to_kpc(ViewportLength::new(6.0), cfg.viewport_kpc_scale);
    let rn_kpc =
        tangential_critical_radius_kpc(None, &nfw, sigma_crit, r_search).expect("NFW crit");
    let rn_vp = kpc_to_viewport(rn_kpc, cfg.viewport_kpc_scale);
    let sample_at_crit = halo_thin_lens_sample(
        viewport_to_kpc(rn_vp, cfg.viewport_kpc_scale),
        None,
        &nfw,
        sigma_crit,
    );
    assert!(sample_at_crit.lambda_tangential.abs() < 1.0e-6);

    let bh = SchwarzschildBlackHole::new(1.0e8 * SOLAR_MASS.get()).unwrap();
    let mut masked = 0usize;
    let out = evaluate_lens_mode_outcome(
        &bh,
        &cfg,
        Vector2::new(cfg.center[0], cfg.center[1]),
        &nfw,
        &burk,
        sigma_crit,
        Point2::new(cfg.center[0] + rn_vp.get(), cfg.center[1]),
        1.0,
        &mut masked,
    );
    let deflected = out.primary.expect("primary halo image").mean;
    assert!((deflected.x - (cfg.center[0] + 2.0 * rn_vp.get())).abs() < 1.0e-5);
}

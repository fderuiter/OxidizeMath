#![cfg_attr(any(), verified(opt_out = "tests"))]

use eframe::egui;
use math_explorer::fit::FitModel;

use super::*;
use crate::audio::frame::LIMITER_CEILING_LINEAR;
use crate::framework::InteractiveTool;

#[test]
fn default_studio_populates_power_law_fit_and_exports_on_frame_one() {
    let studio = AbelianSandpileStudio::default();
    assert_eq!(studio.name(), "Abelian Sandpile Studio");
    assert_eq!(studio.domain(), "solid_state");
    assert!(studio.has_finite_parameters());

    let fit = studio
        .latest_fit
        .as_ref()
        .expect("default critical warmup must populate FitModel::PowerLaw");
    assert_eq!(fit.fit_model, FitModel::PowerLaw);
    assert!(
        fit.r_squared >= 0.90,
        "expected default FitModel::PowerLaw R^2 >= 0.90, got {:.4}",
        fit.r_squared
    );

    let img = studio
        .export_raster_image()
        .expect("raster image available immediately");
    assert_eq!(img.size, [31, 31]);
    assert!(img.pixels.iter().any(|&c| c != egui::Color32::BLACK));

    let prims = studio.export_plot_primitives();
    assert!(prims.len() >= 2);
    let datasets = studio.export_datasets();
    assert_eq!(datasets.len(), 3);
    assert!(!datasets[0].1.is_empty());
}

#[test]
fn presets_group_proof_actions_and_state_bounds_roundtrip() {
    let mut studio = AbelianSandpileStudio::default();

    studio.apply_preset(SandpilePreset::CreutzIdentityCrystal63);
    assert_eq!(studio.state.grid_size, 63);
    assert!(studio.burning_result.is_recurrent);
    assert_eq!(studio.burning_result.burned_cells, 63 * 63);

    studio.apply_preset(SandpilePreset::AbelianGroupIdentityProof);
    assert!(studio.group_status.contains("Verified (r + e)° == r"));
    studio.invert_current_state_action();
    assert!(studio.group_status.contains("true"));

    let saved = studio.save_state().expect("serializes cleanly");
    let mut restored = AbelianSandpileStudio::default();
    restored
        .try_load_state(&saved)
        .expect("loads serialized state");
    assert_eq!(restored.state, studio.state);

    // Reject out-of-bounds > 128 grid and unknown fields.
    let bad_dim = saved.replace("\"grid_size\":25", "\"grid_size\":256");
    assert!(restored.try_load_state(&bad_dim).is_err());
    let unknown_field = saved.replace("\"rng_seed\":1990", "\"rng_seed\":1990,\"extra\":1");
    assert!(restored.try_load_state(&unknown_field).is_err());
}

#[test]
fn sonification_frame_respects_minus_three_dbfs_ceiling_and_renders_headless_ui() {
    let frame = audio::sandpile_sonification_frame(120, 4_500, 900, 1_024, -0.28);
    assert!(frame.is_audible());
    assert!(frame.master_gain <= LIMITER_CEILING_LINEAR + 1.0e-6);
    for v in &frame.voices[..(frame.active_voices as usize)] {
        assert!((20.0..=8_000.0).contains(&v.freq_hz));
        assert!((0.0..=1.0).contains(&v.gain));
    }

    let mut studio = AbelianSandpileStudio::default();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            studio.show_inside(ui);
        });
    });
    assert!(studio.texture.is_some());
}

fn save_upscaled_color_image(img: &egui::ColorImage, scale: usize, path: &str) {
    let w = img.size[0] * scale;
    let h = img.size[1] * scale;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        let src_y = y / scale;
        for x in 0..w {
            let c = img.pixels[src_y * img.size[0] + (x / scale)];
            rgba.extend_from_slice(&[c.r(), c.g(), c.b(), c.a()]);
        }
    }
    let file = std::fs::File::create(path).expect("create PNG artifact");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().expect("write PNG header");
    writer.write_image_data(&rgba).expect("write PNG pixels");
}

#[test]
fn deferred_exports_and_visual_artifacts_succeed() {
    let mut studio = AbelianSandpileStudio::default();
    let opts = egui_plot::export::ExportOptions::default();
    let plot_png = egui_plot::export::export_custom_png(&studio.export_datasets(), &opts)
        .expect("power-law plot PNG export");
    std::fs::write("/tmp/sandpile_avalanche_power_law.png", &plot_png).expect("write plot PNG");

    studio.apply_preset(SandpilePreset::CreutzIdentityCrystal63);
    let crystal = studio.export_raster_image().expect("63x63 crystal image");
    save_upscaled_color_image(&crystal, 8, "/tmp/sandpile_creutz_identity_63x63.png");

    studio.state.view_mode = SandpileViewMode::DharBurningWaves;
    studio.state.colormap = SandpileColormap::BurningWavePhase;
    let burn_img = studio.export_raster_image().expect("burning wave image");
    save_upscaled_color_image(&burn_img, 8, "/tmp/sandpile_dhar_burning_waves.png");

    let mut fw = crate::framework::SimulationFramework::new("solid_state");
    fw.active_tool = Some(Box::new(studio));
    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |ctx| {
        fw.show_export_modal = true;
        fw.queue_export(crate::framework::ExportFormat::Svg);
        fw.show(ctx, "sandpile_export_verify");
    });
    let svg = fw
        .export_queue
        .last_exported_content
        .as_deref()
        .expect("SVG generated");
    assert!(svg.contains("<rect"));
}

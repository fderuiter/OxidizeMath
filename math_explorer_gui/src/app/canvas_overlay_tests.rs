//! Automated WCAG 2.2 AA contrast, two-tone overlay, vector state marker, and `AccessKit`
//! plot/canvas description tests across `ThemeMode::Dark`, `ThemeMode::Light`, and `ThemeMode::HighContrast`.

use crate::app::palette::color_guard_tests::{
    audit_source_for_color32_violations, strip_cfg_test_items_and_comments,
};
use crate::app::{contrast_ratio, contrasting_text_color, SemanticPalette, ThemeMode};
use eframe::egui::{self, Color32};

const ALL_THEMES: [ThemeMode; 3] = [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast];

const fn assert_copy_impl<T: Copy>() {}

fn assert_theme_border_and_two_tone_contrast(mode: ThemeMode) {
    let v = mode.visuals();
    let p = mode.palette();
    for bg in [
        v.panel_fill,
        v.window_fill,
        v.extreme_bg_color,
        p.surface_fill,
    ] {
        let cr = contrast_ratio(p.graph_edge, bg);
        assert!(
            cr >= 3.0,
            "{mode:?} graph_edge vs {bg:?} achieved {cr:.2}:1 (< 3.0:1)"
        );
    }
    for primary in [p.accent, p.info, p.warning, p.node_stroke] {
        let halo = contrasting_text_color(primary);
        for r in (0..=255_u8).step_by(17) {
            for g in (0..=255_u8).step_by(17) {
                for b in (0..=255_u8).step_by(51) {
                    let px = Color32::from_rgb(r, g, b);
                    let best = contrast_ratio(px, primary).max(contrast_ratio(px, halo));
                    assert!(
                        best >= 3.0,
                        "{mode:?} two-tone ({primary:?}, {halo:?}) over {px:?} achieved {best:.2}:1 (< 3.0:1)"
                    );
                }
            }
        }
    }
}

fn assert_theme_simulation_token_contrast(mode: ThemeMode) {
    let p = mode.palette();
    for up in [1_i8, 2_i8] {
        for down in [-1_i8, -2_i8] {
            let cr = contrast_ratio(p.ising_spin_color(up), p.ising_spin_color(down));
            assert!(
                cr >= 3.0,
                "{mode:?} Ising spin {up} vs {down} achieved {cr:.2}:1 (< 3.0:1)"
            );
        }
    }
    if mode == ThemeMode::HighContrast {
        for s in [-2_i8, -1, 1, 2] {
            assert!(contrast_ratio(p.ising_spin_color(s), Color32::BLACK) >= 3.0);
        }
        for step in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!(contrast_ratio(p.attention_ramp_color(step), Color32::BLACK) >= 3.0);
        }
    }
    assert!(contrast_ratio(p.attention_ramp_color(0.0), p.attention_ramp_color(1.0)) >= 3.0);
    assert!(contrast_ratio(p.obstacle_dark, p.obstacle_light) >= 3.0);
    for k in 0..=100_u32 {
        let t = f64::from(k) / 100.0;
        let rgb = if mode == ThemeMode::HighContrast {
            math_commons::math_kernel::colormap::heatmap_color_high_contrast(t)
        } else {
            math_commons::math_kernel::colormap::heatmap_color(t)
        };
        let fluid = SemanticPalette::colormap_to_color32(rgb);
        let cr = contrast_ratio(fluid, p.obstacle_rim_color(fluid));
        assert!(
            cr >= 3.0,
            "{mode:?} obstacle rim vs fluid(t={t:.2}) achieved {cr:.2}:1 (< 3.0:1)"
        );
    }
}

#[test]
fn test_semantic_palette_is_copy_and_tokens_meet_three_theme_contrast() {
    assert_copy_impl::<SemanticPalette>();
    for mode in ALL_THEMES {
        assert_theme_border_and_two_tone_contrast(mode);
        assert_theme_simulation_token_contrast(mode);
    }
}

#[test]
fn test_strip_cfg_test_items_preserves_lines_and_catches_post_test_regressions() {
    let synthetic = "\
pub fn before_fn() -> u32 { 1 }
#[cfg(test)]
fn inline_test_helper() {
    let _allowed_in_test = egui::Color32::YELLOW;
    let _gray_in_test = egui::Color32::from_gray(50);
}
pub fn prod_after_test(r: u8, g: u8, b: u8) {
    let _bad1 = egui::Color32::from_gray(50);
    let _bad2 = egui::Color32::from_rgb(r, g, b);
    let _bad3 = egui::Color32::from_rgba_unmultiplied(10, 20, 30, 40);
    let _bad4 = egui::Color32::BLACK;
    let _bad5 = egui::Color32::WHITE;
}
";
    let stripped = strip_cfg_test_items_and_comments(synthetic);
    assert_eq!(stripped.lines().count(), synthetic.lines().count());
    assert!(!stripped.contains("Color32::YELLOW"));

    let violations = audit_source_for_color32_violations(
        "math_explorer_gui/src/tabs/fluid_dynamics/lattice_boltzmann/render.rs",
        synthetic,
    );
    assert!(violations
        .iter()
        .any(|v| v.contains(":8: Color32::from_gray(")));
    assert!(violations
        .iter()
        .any(|v| v.contains(":9: Color32::from_rgb(")));
    assert!(violations
        .iter()
        .any(|v| v.contains(":10: Color32::from_rgba_unmultiplied(literal)")));
    assert!(violations.iter().any(|v| v.contains(":11: Color32::BLACK")));
    assert!(violations.iter().any(|v| v.contains(":12: Color32::WHITE")));
}

fn collect_accesskit_texts(out: &egui::FullOutput) -> Vec<String> {
    let mut texts = Vec::new();
    if let Some(ref update) = out.platform_output.accesskit_update {
        for (_, node) in &update.nodes {
            if let Some(lbl) = node.label() {
                texts.push(lbl.to_string());
            }
            if let Some(desc) = node.description() {
                texts.push(desc.to_string());
            }
            if let Some(val) = node.value() {
                texts.push(val.to_string());
            }
        }
    }
    texts
}

#[test]
fn test_egui_plot_raster_descriptions_populate_accesskit_nodes() {
    for mode in ALL_THEMES {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        ctx.set_visuals(mode.visuals());
        let img = egui::ColorImage::new([4, 4], vec![mode.palette().accent; 16]);
        let tex = ctx.load_texture("plot_raster_tex", img, egui::TextureOptions::NEAREST);

        let explicit_out = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                egui_plot::Plot::new("explicit_desc_plot")
                    .title("Explicit Raster Plot")
                    .description("Peak intensity 1.0; blue-to-red ramp")
                    .show(ui, |plot_ui| {
                        plot_ui.image(egui_plot::PlotImage::new(
                            "img",
                            tex.id(),
                            egui_plot::PlotPoint::new(2.0, 2.0),
                            [4.0, 4.0],
                        ));
                    });
            });
        });
        let texts = collect_accesskit_texts(&explicit_out);
        assert!(
            texts
                .iter()
                .any(|t| t.contains("Peak intensity 1.0; blue-to-red ramp")),
            "{mode:?} Plot explicit description missing from AccessKit tree: {texts:?}"
        );

        let fallback_out = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                egui_plot::Plot::new("fallback_desc_plot")
                    .title("Fallback Raster Plot")
                    .show(ui, |plot_ui| {
                        plot_ui.image(egui_plot::PlotImage::new(
                            "img",
                            tex.id(),
                            egui_plot::PlotPoint::new(2.0, 2.0),
                            [4.0, 4.0],
                        ));
                    });
            });
        });
        let fb_texts = collect_accesskit_texts(&fallback_out);
        assert!(
            fb_texts.iter().any(|t| t.contains("Layers: img; bounds")),
            "{mode:?} Plot fallback raster description missing from AccessKit tree: {fb_texts:?}"
        );
    }
}

#[cfg(feature = "pure_math")]
fn assert_pure_math_ulam_overlay_and_accesskit(ctx: &egui::Context, mode: ThemeMode) {
    use crate::framework::InteractiveTool;
    let mut ulam = crate::tabs::number_theory::prime_spiral::PrimeSpiralWidget::default();
    let ulam_texts = collect_accesskit_texts(&ctx.run(egui::RawInput::default(), |c| ulam.show(c)));
    assert!(
        ulam_texts
            .iter()
            .any(|t| t.contains("m=1 framed origin") && t.contains("two-tone ring")),
        "{mode:?} PrimeSpiral AccessKit missing key: {ulam_texts:?}"
    );

    let (mut lookup, mut ulam_px) = (Vec::new(), Vec::new());
    crate::tabs::number_theory::prime_spiral::PrimeSpiralWidget::populate_ulam_pixels(
        31,
        mode,
        &mut lookup,
        &mut ulam_px,
    );
    assert_eq!(ulam_px.len(), 31 * 31);
    assert!(ulam_px.iter().any(|&c| c != mode.palette().surface_fill));
}

#[cfg(feature = "physics")]
fn assert_physics_ising_and_dose_accesskit(ctx: &egui::Context, mode: ThemeMode) {
    use crate::framework::InteractiveTool;
    let mut ising = crate::tabs::solid_state::ising::IsingModelTool::default();
    if ising.pin_cells.len() >= 2 {
        ising.pin_cells[0] = 1;
        ising.pin_cells[1] = 2;
    }
    let ising_texts =
        collect_accesskit_texts(&ctx.run(egui::RawInput::default(), |c| ising.show(c)));
    assert!(
        ising_texts
            .iter()
            .any(|t| t.contains("Pin +1 gold '+'") && t.contains("Pin -1 teal 'x'")),
        "{mode:?} Ising AccessKit missing pin markers: {ising_texts:?}"
    );

    let mut dose = crate::tabs::medical::dose::DoseCalculationTool {
        imrt_mode: true,
        ..Default::default()
    };
    let dose_texts = collect_accesskit_texts(&ctx.run(egui::RawInput::default(), |c| dose.show(c)));
    assert!(dose_texts
        .iter()
        .any(|t| t.contains("PTV target green contour")
            && t.contains("OAR organ-at-risk red contour")));
}

#[cfg(feature = "ai")]
fn assert_ai_attention_and_gridworld_accesskit(ctx: &egui::Context, mode: ThemeMode) {
    use crate::framework::InteractiveTool;
    let mut att = crate::tabs::ai::attention_maps::AttentionMapsTool::default();
    let att_texts = collect_accesskit_texts(&ctx.run(egui::RawInput::default(), |c| att.show(c)));
    assert!(
        att_texts
            .iter()
            .any(|t| t.contains("Color key: 3-stop luminance ramp")),
        "{mode:?} AttentionMaps AccessKit missing color key: {att_texts:?}"
    );

    let mut gw = crate::tabs::ai::grid_world::GridWorldTool::default();
    let gw_texts = collect_accesskit_texts(&ctx.run(egui::RawInput::default(), |c| gw.show(c)));
    assert!(
        gw_texts
            .iter()
            .any(|t| t.contains("Start (") && t.contains("[S]") && t.contains("Goal (")),
        "{mode:?} GridWorld AccessKit missing state badges: {gw_texts:?}"
    );
}

#[cfg(feature = "biology")]
fn assert_biology_morphogenesis_accesskit(ctx: &egui::Context, mode: ThemeMode) {
    use crate::framework::InteractiveTool;
    let mut morpho = crate::tabs::morphogenesis::MorphogenesisTool::default();
    morpho.step_simulation();
    let m_texts = collect_accesskit_texts(&ctx.run(egui::RawInput::default(), |c| morpho.show(c)));
    assert!(
        m_texts
            .iter()
            .any(|t| t.contains("concentration ramp") && t.contains("Turing")),
        "{mode:?} Morphogenesis AccessKit missing summary: {m_texts:?}"
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "physics"))]
fn assert_lbm_theme_sync_and_accesskit(
    ctx: &egui::Context,
    lbm: &mut crate::async_sim::unified::UnifiedSimTool<
        crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified,
    >,
    mode: ThemeMode,
) {
    use crate::framework::InteractiveTool;
    use crate::tabs::fluid_dynamics::lattice_boltzmann::render::is_lbm_obstacle_pixel;

    let _ = ctx.run(egui::RawInput::default(), |c| lbm.show(c));
    let mut matched_theme = false;
    for _ in 0..400 {
        let _ = lbm.controller.update();
        if let Some(img) = lbm.export_raster_image() {
            if is_lbm_obstacle_pixel(img.pixels[24 * 100 + 30], mode) {
                matched_theme = true;
                assert!(!img.pixels.contains(&Color32::from_gray(50)));
                break;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(matched_theme, "{mode:?} LBM failed to sync obstacle theme");
    let lbm_texts = collect_accesskit_texts(&ctx.run(egui::RawInput::default(), |c| lbm.show(c)));
    assert!(
        lbm_texts
            .iter()
            .any(|t| t.contains("hatch") && t.contains("Color/shape key")),
        "{mode:?} LBM AccessKit summary missing hatch/color key: {lbm_texts:?}"
    );
}

#[test]
fn test_simulations_and_raster_plots_across_all_three_themes() {
    #[cfg(all(not(target_arch = "wasm32"), feature = "physics"))]
    let mut lbm = crate::async_sim::unified::UnifiedSimTool::<
        crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified,
    >::new();

    for mode in ALL_THEMES {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        ctx.set_visuals(mode.visuals());
        #[cfg(feature = "pure_math")]
        assert_pure_math_ulam_overlay_and_accesskit(&ctx, mode);
        #[cfg(feature = "physics")]
        assert_physics_ising_and_dose_accesskit(&ctx, mode);
        #[cfg(feature = "ai")]
        assert_ai_attention_and_gridworld_accesskit(&ctx, mode);
        #[cfg(feature = "biology")]
        assert_biology_morphogenesis_accesskit(&ctx, mode);
        #[cfg(all(not(target_arch = "wasm32"), feature = "physics"))]
        assert_lbm_theme_sync_and_accesskit(&ctx, &mut lbm, mode);
    }
}

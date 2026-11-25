use super::maturity_badge::{
    badge_colors_for_tier, badge_display_text, render_maturity_badge, resolve_tool_maturity_tier,
};
#[cfg(all(feature = "pure_math", feature = "climate"))]
use super::{SimulationFramework, ToolMetadata};
use crate::app::theme::{contrast_ratio, ThemeMode};
use eframe::egui;
use scientific_metadata::theory::MaturityTier;

fn shape_contains(shape: &egui::epaint::Shape, needle: &str) -> bool {
    match shape {
        egui::epaint::Shape::Text(t) => t.galley.text().contains(needle),
        egui::epaint::Shape::Vec(items) => items.iter().any(|s| shape_contains(s, needle)),
        _ => false,
    }
}

#[test]
fn test_maturity_badge_colors_meet_wcag_22_aa_across_all_themes() {
    for mode in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        let visuals = mode.visuals();
        for tier in [
            MaturityTier::Experimental,
            MaturityTier::Validated,
            MaturityTier::Certified,
        ] {
            let colors = badge_colors_for_tier(&visuals, tier);
            let text_ratio = contrast_ratio(colors.fg_text, colors.bg_fill);
            assert!(
                text_ratio >= 4.5,
                "{mode:?} {tier:?} text contrast {text_ratio:.2} < 4.5"
            );
            let border_ratio = contrast_ratio(colors.border_stroke, visuals.panel_fill);
            assert!(
                border_ratio >= 3.0,
                "{mode:?} {tier:?} border contrast {border_ratio:.2} < 3.0"
            );
        }
    }
}

#[test]
fn test_tool_maturity_resolution_distinguishes_certified_and_experimental_models() {
    assert_eq!(
        resolve_tool_maturity_tier("OdeSolverTool", "analysis", Some(MaturityTier::Certified)),
        MaturityTier::Certified
    );
    assert_eq!(
        resolve_tool_maturity_tier("CeraTool", "climate", Some(MaturityTier::Experimental)),
        MaturityTier::Experimental
    );
    assert_eq!(
        resolve_tool_maturity_tier(
            "FavoritismTool",
            "favoritism",
            Some(MaturityTier::Experimental)
        ),
        MaturityTier::Experimental
    );
    assert_eq!(
        resolve_tool_maturity_tier("WinRatioWorkbenchTool", "clinical_trials", None),
        MaturityTier::Validated
    );
    assert_eq!(
        resolve_tool_maturity_tier("WinRatioTool", "clinical_trials", None),
        MaturityTier::Validated
    );
    assert_eq!(
        resolve_tool_maturity_tier("ClinicalTrialsTool", "clinical_trials", None),
        MaturityTier::Certified
    );
    assert_eq!(
        resolve_tool_maturity_tier("RiemannianPdeStudio", "geometry_topology", None),
        MaturityTier::Certified
    );
    assert_eq!(
        resolve_tool_maturity_tier("DtmcTensorWorkbenchTool", "stochastic_processes", None),
        MaturityTier::Validated
    );
    assert_eq!(
        resolve_tool_maturity_tier("CausticsRayBenchTool", "optics", None),
        MaturityTier::Certified
    );
    assert_eq!(
        resolve_tool_maturity_tier("PortfolioCopulaWorkbenchTool", "financial_math", None),
        MaturityTier::Validated
    );
    assert_eq!(
        resolve_tool_maturity_tier("HyperbolicDiskTool", "hyperbolic_geometry", None),
        MaturityTier::Validated
    );
    assert_eq!(
        resolve_tool_maturity_tier("CalibrationWorkbenchTool", "ai", None),
        MaturityTier::Validated
    );
    assert_eq!(
        resolve_tool_maturity_tier("QuasicrystalDiffractionTool", "solid_state", None),
        MaturityTier::Certified
    );
    assert_eq!(
        resolve_tool_maturity_tier("BcsSuperconductivityTool", "solid_state", None),
        MaturityTier::Certified
    );
}

#[test]
fn test_render_maturity_badge_populates_text_and_hover_a11y() {
    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(640.0, 480.0),
        )),
        ..Default::default()
    };
    let out = ctx.run(raw, |ctx| {
        ctx.enable_accesskit();
        egui::CentralPanel::default().show(ctx, |ui| {
            render_maturity_badge(ui, MaturityTier::Certified, false, "ODE Solvers");
            render_maturity_badge(ui, MaturityTier::Experimental, true, "CERA Model");
        });
    });
    assert!(out.shapes.iter().any(|cs| shape_contains(
        &cs.shape,
        badge_display_text(MaturityTier::Certified, false)
    )));
    assert!(out.shapes.iter().any(|cs| {
        shape_contains(
            &cs.shape,
            badge_display_text(MaturityTier::Experimental, true),
        )
    }));
}

#[cfg(all(feature = "pure_math", feature = "climate"))]
fn collect_texts(shape: &egui::epaint::Shape, out: &mut Vec<String>) {
    match shape {
        egui::epaint::Shape::Text(t) => out.push(t.galley.text().to_string()),
        egui::epaint::Shape::Vec(items) => {
            for s in items {
                collect_texts(s, out);
            }
        }
        _ => {}
    }
}

#[cfg(all(feature = "pure_math", feature = "climate"))]
#[test]
fn test_simulation_framework_and_theory_portal_render_maturity_badges() {
    let tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>
        .into_iter()
        .filter(|m| m.id == "OdeSolverTool" || m.id == "CeraTool")
        .collect();
    assert_eq!(tools.len(), 2);

    let mut framework = SimulationFramework::new("analysis");
    framework.available_tools = tools;
    framework.selected_tool_index = None;
    let ode_idx = framework
        .available_tools
        .iter()
        .position(|m| m.id == "OdeSolverTool")
        .expect("OdeSolverTool registered");
    framework.select_tool(ode_idx);
    framework.show_theory_portal = true;

    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let out = ctx.run(raw, |ctx| {
        framework.show(ctx, "maturity_portal_test");
    });

    let mut texts = Vec::new();
    for cs in &out.shapes {
        collect_texts(&cs.shape, &mut texts);
    }
    assert!(
        texts.iter().any(|t| t.contains("[CERTIFIED]")),
        "Theory Context Portal should render [CERTIFIED] for OdeSolverTool; got: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("[EXP]")),
        "Tool selector should render compact [EXP] badge for CeraTool; got: {texts:?}"
    );
}

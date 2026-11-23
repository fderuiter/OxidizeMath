#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::app::MathExplorerApp;
use crate::framework::{InteractiveTool, ToolMetadata};
use eframe::egui::{self, accesskit::Role};

fn test_screen() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

fn rect_overlap_area(a: egui::Rect, b: egui::Rect) -> f32 {
    let inter = a.intersect(b);
    if inter.is_positive() {
        inter.area()
    } else {
        0.0
    }
}

fn collect_role_rects(out: &egui::FullOutput, role: Role) -> Vec<(String, egui::Rect)> {
    let Some(ak) = out.platform_output.accesskit_update.as_ref() else {
        return Vec::new();
    };
    ak.nodes
        .iter()
        .filter(|(_, n)| n.role() == role)
        .filter_map(|(_, n)| {
            let b = n.bounds()?;
            let rect = egui::Rect::from_min_max(
                egui::pos2(b.x0 as f32, b.y0 as f32),
                egui::pos2(b.x1 as f32, b.y1 as f32),
            );
            Some((n.label().unwrap_or_default().to_owned(), rect))
        })
        .collect()
}

fn assert_no_2d_plot_chrome(tool: &mut dyn InteractiveTool, tool_label: &str) {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let _ = ctx.run(test_screen(), |c| tool.show(c));
    let out = ctx.run(test_screen(), |c| tool.show(c));
    let buttons = collect_role_rects(&out, Role::Button);
    for forbidden in [
        "Collapse plot toolbar",
        "Plot grid and axis settings",
        "Zoom in",
        "Zoom out",
        "Reset plot bounds",
    ] {
        assert!(
            !buttons.iter().any(|(lbl, _)| lbl == forbidden),
            "3D viewport tool '{tool_label}' leaked 2D plot chrome button '{forbidden}'"
        );
    }
}

#[test]
#[cfg(all(feature = "physics", feature = "pure_math", feature = "ai"))]
fn test_all_6_plot_3d_viewport_tools_hide_2d_plot_chrome() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let expected_3d_ids = [
        "AttractorPlotter",
        "CrystalViewer",
        "SpinVisualizer",
        "SurfaceViewer",
        "CurvatureHeatmap",
        "Gaussian Splatting",
    ];
    let mut verified = 0_usize;
    for target_id in expected_3d_ids {
        let meta = inventory::iter::<ToolMetadata>
            .into_iter()
            .find(|m| m.id == target_id)
            .unwrap_or_else(|| panic!("Missing registered 3D ToolMetadata '{target_id}'"));
        let mut tool = (meta.build)();
        assert_no_2d_plot_chrome(&mut *tool, target_id);
        verified += 1;
    }
    assert_eq!(verified, 6);
}

const PLOT_CHROME_BUTTON_LABELS: &[&str] = &[
    "Collapse plot toolbar",
    "Plot grid and axis settings",
    "Zoom in",
    "Zoom out",
    "Reset plot bounds",
    "Save PNG",
    "Export CSV",
    "Export JSON",
    "Table View",
    "Fit Curve",
];

fn assert_output_zero_legend_button_overlap(out: &egui::FullOutput, context_name: &str) {
    let checkboxes = collect_role_rects(out, Role::CheckBox);
    let buttons = collect_role_rects(out, Role::Button);
    for (cl, cr) in &checkboxes {
        for (bl, br) in &buttons {
            let is_plot_btn =
                PLOT_CHROME_BUTTON_LABELS.contains(&bl.as_str()) || bl.contains("Settings");
            if is_plot_btn {
                assert_eq!(
                    rect_overlap_area(*cr, *br),
                    0.0,
                    "'{context_name}' CheckBox '{cl}' ({cr:?}) overlapped plot Button '{bl}' ({br:?})"
                );
            }
        }
    }
}

#[test]
fn test_all_registered_tools_zero_legend_button_overlap() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    for meta in inventory::iter::<ToolMetadata> {
        let mut tool = (meta.build)();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let _ = ctx.run(test_screen(), |c| tool.show(c));
        let out = ctx.run(test_screen(), |c| tool.show(c));
        assert_output_zero_legend_button_overlap(&out, meta.id);
    }
}

#[test]
fn test_all_app_tabs_zero_legend_button_overlap() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    for idx in 0..app.tabs.len() {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        app.select_tab(&ctx, idx);
        let _ = ctx.run(test_screen(), |c| {
            eframe::App::update(&mut app, c, &mut frame);
        });
        let out = ctx.run(test_screen(), |c| {
            eframe::App::update(&mut app, c, &mut frame);
        });
        assert_output_zero_legend_button_overlap(&out, app.tabs[idx].name());
    }
}

#[test]
#[cfg(feature = "climate")]
fn test_real_gistemp_anomalies_degree4_fit_accuracy() {
    let series = math_explorer::climate::dataset::get_temperature_anomalies();
    assert_eq!(series.len(), 144);
    let fit = egui_plot::fit_model_with_axes(
        &series,
        egui_plot::FitType::Quartic,
        "GISTEMP Temperature Anomaly",
        false,
        false,
        false,
    )
    .expect("ADR 0031 GISTEMP Degree 4 fit must succeed");
    assert!(
        (fit.r_squared - 0.9158).abs() < 0.001,
        "GISTEMP 1880..=2023 Degree 4 R^2 expected ~0.9158, got {}",
        fit.r_squared
    );
    assert!(!fit.formula.contains("0.0000"));
}

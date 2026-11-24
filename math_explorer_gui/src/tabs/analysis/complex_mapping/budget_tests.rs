#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::cache::MAX_CONFORMAL_SYNC_INSTRUCTIONS;
use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::framework::SimulationFramework;
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use crate::web_io::PlatformInstant;

/// Builds a holomorphic custom expression `w = f(z)` with ~199 bytecode instructions.
fn heavy_holomorphic_expr() -> String {
    let mut terms = vec!["z".to_string()];
    for idx in 1..=22 {
        terms.push(format!("0.02 * sin({idx} * z + a)"));
    }
    terms.join(" + ")
}

fn collect_rendered_texts(shapes: &[egui::epaint::Shape], out: &mut Vec<String>) {
    for shape in shapes {
        match shape {
            egui::epaint::Shape::Text(t) => out.push(t.galley.job.text.clone()),
            egui::epaint::Shape::Vec(inner) => collect_rendered_texts(inner, out),
            _ => {}
        }
    }
}

#[test]
fn test_complex_mapping_caps_contour_work_and_announces_budget_once() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    clear_test_announcements();

    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        grid_density: crate::presets::MAX_INTERACTIVE_GRID_SIZE,
        ..Default::default()
    };
    tool.math_input.set_expression(&heavy_holomorphic_expr());
    assert!(tool.math_input.is_valid());
    let inst = tool.math_input.instruction_count();
    assert!(
        inst >= 150,
        "expected >= 150 bytecode instructions, got {inst}"
    );

    let (eff_density, eff_res, notice) = tool.effective_contour_grid();
    assert!(notice.is_some());
    let total_ops = inst
        .saturating_mul(2)
        .saturating_mul(eff_density)
        .saturating_mul(eff_res);
    assert!(
        total_ops <= MAX_CONFORMAL_SYNC_INSTRUCTIONS,
        "contour work {total_ops} exceeded cap {MAX_CONFORMAL_SYNC_INSTRUCTIONS}"
    );

    let ctx = egui::Context::default();
    let t0 = PlatformInstant::now();
    let out = ctx.run(egui::RawInput::default(), |c| tool.show(c));
    assert!(t0.elapsed().as_secs_f64() < 1.0);
    assert!(tool.cache.budget_notice.is_some());

    let shapes: Vec<_> = out.shapes.into_iter().map(|cs| cs.shape).collect();
    let mut texts = Vec::new();
    collect_rendered_texts(&shapes, &mut texts);
    assert!(
        texts
            .iter()
            .any(|t| t.contains("Contour work budget applied")),
        "expected inline contour budget notice in {texts:?}"
    );

    let announced = all_announcements();
    assert!(announced
        .iter()
        .any(|(m, p)| p == "polite" && m.contains("Contour work budget applied")));
    let count_after_first = announced.len();
    let _ = ctx.run(egui::RawInput::default(), |c| tool.show(c));
    assert_eq!(
        all_announcements().len(),
        count_after_first,
        "contour budget notice must not spam live region on idle frames"
    );

    assert_starter_presets_and_export_png(&ctx, tool);
}

fn assert_starter_presets_and_export_png(ctx: &egui::Context, tool: ComplexMappingTool) {
    let mut starter = ComplexMappingTool::default();
    starter.import_preset_json(STARTER_PRESET_JSON).unwrap();
    assert!(starter.cache.budget_notice.is_none());
    starter
        .import_preset_json(JOUKOWSKY_POLE_PRESET_JSON)
        .unwrap();
    assert!(starter.cache.budget_notice.is_none());

    let mut export_fw = SimulationFramework::new("__complex_budget_png__");
    export_fw.active_tool = Some(Box::new(tool));
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(960.0, 540.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |c| {
        export_fw.show_export_modal = true;
        export_fw.queue_export(crate::framework::ExportFormat::Png);
        export_fw.show(c, "cm_budget_fw");
    });
    let png = export_fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("export budgeted complex mapping contours PNG");
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
    let _ = std::fs::write("/tmp/complex_mapping_budget_contours.png", png);
}

#[test]
fn test_complex_mapping_rejects_oversized_url_and_state_custom_formulas() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    clear_test_announcements();

    let mut abusive = String::from("z");
    for _ in 0..140 {
        abusive.push_str("+sin(z)");
    }
    let url_json = format!(
        r#"{{"metadata":{{"schema_version":"1.0","title":"AbusiveComplex","domain":"analysis","timestamp":"2026-11-23T00:00:00Z"}},"config":{{"function":"Custom","grid_min":-2.0,"grid_max":2.0,"grid_density":100,"inspection_point":[1.0,0.5],"custom_expr":"{abusive}"}}}}"#
    );
    assert!(url_json.len() <= MAX_DECODED_URL_STATE_BYTES);

    let mut fw = SimulationFramework::new("analysis");
    fw.active_tool = Some(Box::new(ComplexMappingTool::default()));
    let err = fw
        .import_url_preset_json(Some("ComplexMappingTool"), Some(&url_json))
        .expect_err("expected URL hydration with > 256 AST nodes to fail");
    assert!(err.contains("Invalid complex mapping formula"));
    assert!(all_announcements()
        .iter()
        .any(|(m, p)| p == "assertive" && m.contains("Invalid complex mapping formula")));
}

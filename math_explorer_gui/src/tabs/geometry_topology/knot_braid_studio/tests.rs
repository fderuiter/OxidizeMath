#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::model::{DiagramEditOp, StudioFamily};
use super::presets::{
    BORROMEAN_RINGS_PRESET_JSON, CHIRAL_TREFOIL_PRESET_JSON, FIGURE_EIGHT_ISOTOPY_PRESET_JSON,
};
use super::viewport::diagram_theme_colors;
use super::KnotBraidStudioTool;
use crate::app::theme::{contrast_ratio, ThemeMode};
use crate::framework::InteractiveTool;
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use eframe::egui;
use std::collections::HashMap;

#[test]
fn presets_and_url_bounds_roundtrip_cleanly() {
    for raw in [
        CHIRAL_TREFOIL_PRESET_JSON,
        FIGURE_EIGHT_ISOTOPY_PRESET_JSON,
        BORROMEAN_RINGS_PRESET_JSON,
    ] {
        assert!(raw.len() <= MAX_DECODED_URL_STATE_BYTES);
        let mut tool = KnotBraidStudioTool::default();
        tool.import_preset_json(raw).expect("valid preset envelope");
        assert!(tool.has_finite_parameters());
        let saved = tool.save_state().expect("serialize state");
        assert!(saved.len() <= MAX_DECODED_URL_STATE_BYTES);

        let mut restored = KnotBraidStudioTool::default();
        restored.try_load_state(&saved).expect("restore state");
        assert_eq!(restored.save_state(), Some(saved));
    }

    let mut tool = KnotBraidStudioTool::default();
    let oversized = format!(
        r#"{{"tube_radius":0.14,"family":"{}""#,
        "A".repeat(MAX_DECODED_URL_STATE_BYTES + 32)
    );
    assert!(tool.try_load_state(&oversized).is_err());
}

#[test]
fn chiral_trefoils_borromean_linking_and_fox_burau_parity() {
    let right = KnotBraidStudioTool::default();
    assert_eq!(right.model.alexander_fox, right.model.alexander_burau);
    assert_eq!(right.model.is_chiral, Some(true));

    let mut left = KnotBraidStudioTool::default();
    left.import_preset_json(CHIRAL_TREFOIL_PRESET_JSON)
        .expect("load chiral left trefoil");
    assert_eq!(left.model.alexander_fox, right.model.alexander_fox);
    assert_ne!(left.model.jones_half_power, right.model.jones_half_power);

    let mut borromean = KnotBraidStudioTool::default();
    borromean
        .import_preset_json(BORROMEAN_RINGS_PRESET_JSON)
        .expect("load Borromean rings");
    assert_eq!(
        borromean.model.linking_matrix,
        vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]
    );
    assert!(borromean
        .model
        .jones_half_power
        .as_ref()
        .is_some_and(|j| j.coeffs.len() > 1));
}

#[test]
fn keyboard_navigation_and_reidemeister_moves_preserve_invariants_and_update_3d_writhe() {
    let mut tool = KnotBraidStudioTool {
        family: StudioFamily::ArtinBraid,
        braid_strands: 3,
        braid_generators: vec![1, 2, 1, 2],
        ..KnotBraidStudioTool::default()
    };
    tool.cached_key = 0;
    tool.sync_model();

    let base_alex = tool.model.alexander_fox.clone();
    let base_jones = tool.model.jones_half_power.clone();
    let base_writhe = tool.model.writhe_diagram;

    // Keyboard cycle crossings and arcs.
    assert!(tool.handle_diagram_key(egui::Key::ArrowRight));
    assert_eq!(tool.selected_crossing, 1);
    assert!(tool.handle_diagram_key(egui::Key::ArrowLeft));
    assert_eq!(tool.selected_crossing, 0);
    assert!(tool.handle_diagram_key(egui::Key::S));
    assert_eq!(tool.selected_segment, 1);

    // Keyboard R-III triangle slide on sigma_1 sigma_2 sigma_1.
    assert!(tool.handle_diagram_key(egui::Key::Num3));
    assert_eq!(tool.model.alexander_fox, base_alex);
    assert_eq!(tool.model.jones_half_power, base_jones);
    assert_eq!(tool.model.writhe_diagram, base_writhe);

    // Keyboard R-I twist (+) updates diagram writhe and lifted 3D Gauss writhe while keeping Jones invariant.
    let writhe_3d_before: f64 = tool.model.writhe_gauss_3d.iter().sum();
    assert!(tool.handle_diagram_key(egui::Key::Num1));
    let writhe_3d_after: f64 = tool.model.writhe_gauss_3d.iter().sum();
    assert_eq!(tool.model.writhe_diagram, base_writhe + 1);
    assert!((writhe_3d_after - writhe_3d_before).abs() > 0.05);
    assert_eq!(tool.model.alexander_fox, base_alex);
    assert_eq!(tool.model.jones_half_power, base_jones);

    // Untwist the newly added crossing via R-I remove.
    tool.selected_crossing = tool.model.diagram.crossings.len() - 1;
    assert!(tool.handle_diagram_key(egui::Key::U));
    assert_eq!(tool.model.writhe_diagram, base_writhe);

    // R-II bigon poke & cancel preserve both polynomial invariants and writhe.
    tool.push_reidemeister_op(DiagramEditOp::RiiAdd {
        over_seg: 0,
        under_seg: 2,
    })
    .expect("R-II poke");
    assert_eq!(tool.model.alexander_fox, base_alex);
    assert_eq!(tool.model.jones_half_power, base_jones);
    let c_len = tool.model.diagram.crossings.len();
    tool.push_reidemeister_op(DiagramEditOp::RiiRemove {
        c1: c_len - 2,
        c2: c_len - 1,
    })
    .expect("R-II cancel");
    assert_eq!(tool.model.jones_half_power, base_jones);
}

#[test]
fn crossing_cap_above_sixteen_keeps_alexander_and_writhe_active() {
    let mut tool = KnotBraidStudioTool {
        family: StudioFamily::ArtinBraid,
        braid_strands: 2,
        braid_generators: vec![1; 17],
        ..KnotBraidStudioTool::default()
    };
    tool.cached_key = 0;
    tool.sync_model();

    assert_eq!(tool.model.diagram.crossing_count(), 17);
    assert!(tool.model.crossing_cap_notice.is_some());
    assert!(tool.model.kauffman_bracket.is_none());
    assert!(tool.model.jones_half_power.is_none());
    assert!(tool.model.alexander_fox.is_some());
    assert_eq!(tool.model.alexander_fox, tool.model.alexander_burau);
    assert_eq!(tool.model.writhe_diagram, 17);
}

#[test]
fn wcag_aa_contrast_and_headless_render_across_all_themes() {
    let mut tool = KnotBraidStudioTool::default();
    for mode in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        let visuals = mode.visuals();
        let colors = diagram_theme_colors(&visuals);
        assert!(contrast_ratio(colors.focus_ring, colors.canvas_bg) >= 4.5);
        assert!(contrast_ratio(colors.label_text, colors.canvas_bg) >= 4.5);
        assert!(contrast_ratio(colors.positive_text, colors.badge_bg) >= 4.5);
        assert!(contrast_ratio(colors.negative_text, colors.badge_bg) >= 4.5);
        assert!(contrast_ratio(colors.border_stroke, colors.canvas_bg) >= 3.0);
        for &series_col in &colors.component_colors {
            assert!(contrast_ratio(series_col, colors.canvas_bg) >= 4.5);
        }

        let ctx = egui::Context::default();
        mode.apply_to_context(&ctx);
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
        });
    }
}

#[test]
fn exported_3d_tubular_mesh_is_watertight_and_orientable() {
    let mut tool = KnotBraidStudioTool::default();
    tool.import_preset_json(BORROMEAN_RINGS_PRESET_JSON)
        .expect("load Borromean rings");
    let mesh = tool
        .export_mesh_3d()
        .expect("3D solid capability")
        .expect("valid tubular mesh");
    assert!(!mesh.vertices.is_empty());
    assert_eq!(mesh.indices.len() % 3, 0);

    let mut half_edges: HashMap<(usize, usize), usize> = HashMap::new();
    for &[a, b, c] in mesh.indices.as_chunks::<3>().0 {
        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            assert_ne!(u, v);
            *half_edges.entry((u, v)).or_insert(0) += 1;
        }
    }
    for (&(u, v), &cnt) in &half_edges {
        assert_eq!(cnt, 1, "duplicate directed half-edge ({u}, {v})");
        assert_eq!(half_edges.get(&(v, u)).copied(), Some(1));
    }

    let obj = oxidize_core::mesh::export_mesh_to_obj_string(&mesh).expect("OBJ export");
    assert!(obj.contains("v ") && obj.contains("f "));

    let stl_bin = oxidize_core::mesh::export_mesh_to_stl_binary(&mesh).expect("binary STL export");
    let tri_count = mesh.indices.len() / 3;
    assert_eq!(stl_bin.len(), 84 + 50 * tri_count);

    let stl_ascii = oxidize_core::mesh::export_mesh_to_stl_ascii(&mesh).expect("ASCII STL export");
    assert!(stl_ascii.starts_with("solid ") && stl_ascii.contains("facet normal"));
}

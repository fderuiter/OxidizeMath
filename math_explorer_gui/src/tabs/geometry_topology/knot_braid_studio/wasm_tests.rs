#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::{BORROMEAN_RINGS_PRESET_JSON, CHIRAL_TREFOIL_PRESET_JSON};
use super::KnotBraidStudioTool;
use crate::framework::InteractiveTool;
use eframe::egui;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_knot_braid_studio_render_reidemeister_and_export_roundtrip() {
    let mut tool = KnotBraidStudioTool::default();
    tool.import_preset_json(CHIRAL_TREFOIL_PRESET_JSON)
        .expect("wasm chiral trefoil load");

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    assert!(tool.has_finite_parameters());
    assert_eq!(tool.model.is_chiral, Some(true));

    // Apply keyboard Reidemeister I twist and verify state round-trip + mesh export on wasm32.
    let orig_jones = tool.model.jones_half_power.clone();
    assert!(tool.handle_diagram_key(egui::Key::Num1));
    assert_eq!(tool.model.jones_half_power, orig_jones);

    let saved = tool.save_state().expect("wasm save_state");
    let mut restored = KnotBraidStudioTool::default();
    restored.try_load_state(&saved).expect("wasm restore");
    assert_eq!(restored.save_state(), Some(saved));

    restored
        .import_preset_json(BORROMEAN_RINGS_PRESET_JSON)
        .expect("wasm Borromean preset");
    let mesh = restored
        .export_mesh_3d()
        .expect("wasm mesh option")
        .expect("wasm mesh result");
    assert!(!mesh.vertices.is_empty());
    assert_eq!(mesh.indices.len() % 3, 0);

    let mut half_edges = std::collections::HashMap::new();
    for &[a, b, c] in mesh.indices.as_chunks::<3>().0 {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *half_edges.entry((u, v)).or_insert(0usize) += 1;
        }
    }
    for (&(u, v), &cnt) in &half_edges {
        assert_eq!(cnt, 1);
        assert_eq!(half_edges.get(&(v, u)).copied(), Some(1));
    }

    let stl_bin = oxidize_core::mesh::export_mesh_to_stl_binary(&mesh).expect("wasm STL export");
    assert_eq!(stl_bin.len(), 84 + 50 * (mesh.indices.len() / 3));
    let obj = oxidize_core::mesh::export_mesh_to_obj_string(&mesh).expect("wasm OBJ export");
    assert!(obj.contains("v ") && obj.contains("f "));
}

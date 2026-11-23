#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::data_bus::read_shared_graph;
use eframe::egui;

#[test]
fn test_signed_edge_brushes_hub_stamp_and_sever_stroke_publish_to_bus() {
    let ctx = egui::Context::default();
    let mut tool = GraphEditorTool {
        mode: EditorMode::HubStamp,
        hub_spokes: 4,
        brush_weight: 2.5,
        ..Default::default()
    };

    // 1. Stamp a 4-spoke excitatory hub at (200, 200)
    tool.stamp_hub_at(Pos2::new(200.0, 200.0));
    assert_eq!(tool.node_positions.len(), 5);
    assert_eq!(tool.edges.len(), 4);
    assert!(tool.edges.iter().all(|&(_, _, w)| (w - 2.5).abs() < 1e-9));

    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
    let g1 = read_shared_graph(&ctx).expect("hub stamp published to WorkspaceDataBus");
    assert_eq!(g1.nodes.len(), 5);
    assert_eq!(g1.edges.len(), 4);

    // 2. Paint an inhibitory -w edge between spoke 1 and spoke 2
    tool.mode = EditorMode::InhibitoryBrush;
    tool.brush_weight = 1.75;
    assert!(tool.upsert_signed_edge(1, 2, tool.active_signed_weight()));
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
    let g2 = read_shared_graph(&ctx).expect("inhibitory edge published");
    assert!(g2.revision > g1.revision);
    assert!(g2.edges.iter().any(|&(_, _, w)| (w + 1.75).abs() < 1e-9));

    // 3. Sever edges intersecting vertical cut line near hub -> spokes
    let severed =
        tool.sever_edges_along_segment(Pos2::new(235.0, 120.0), Pos2::new(235.0, 280.0), true);
    assert!(severed >= 1);
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
    let g3 = read_shared_graph(&ctx).expect("sever stroke published");
    assert!(g3.revision > g2.revision);
    assert_eq!(g3.edges.len(), tool.edges.len());

    // 4. Undo restores severed edges and advances bus revision
    tool.undo();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
    let g4 = read_shared_graph(&ctx).expect("undo published");
    assert!(g4.revision > g3.revision);
    assert_eq!(g4.edges.len(), 5);
}

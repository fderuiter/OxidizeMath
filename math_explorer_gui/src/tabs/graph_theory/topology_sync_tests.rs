#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::GraphTheoryTab;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::tabs::ExplorerTab;
use eframe::egui;
use math_explorer::pure_math::graph_theory::graph_io::TopologyFormat;

fn verify_imported_sync_across_tools(
    tab: &mut GraphTheoryTab,
    expected_nodes: usize,
    expected_deg: usize,
) {
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    for tool_id in [
        "GraphEditorTool",
        "AlgorithmVisualizerTool",
        "NetworkMetricsTool",
    ] {
        assert!(tab.select_tool_by_name(tool_id));
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                tab.show_inside(ui, &mut frame);
            });
        });
        let doc = tab
            .framework()
            .unwrap()
            .active_tool
            .as_ref()
            .unwrap()
            .export_topology_document()
            .unwrap()
            .unwrap();
        assert_eq!(
            doc.nodes.len(),
            expected_nodes,
            "tool {tool_id} node mismatch"
        );
        if tool_id == "NetworkMetricsTool" {
            assert_eq!(
                doc.attributes.get("degeneracy").map(String::as_str),
                Some(expected_deg.to_string().as_str())
            );
            let vc_size: usize = doc
                .attributes
                .get("vertex_cover_size")
                .unwrap()
                .parse()
                .unwrap();
            assert!(vc_size >= 1 && vc_size <= expected_nodes);
            let covered_ids: std::collections::HashSet<&str> = doc
                .nodes
                .iter()
                .filter(|n| n.attributes.get("in_vertex_cover").map(String::as_str) == Some("true"))
                .map(|n| n.id.as_str())
                .collect();
            for edge in &doc.edges {
                assert!(
                    covered_ids.contains(edge.source.as_str())
                        || covered_ids.contains(edge.target.as_str()),
                    "Uncovered edge {} -- {}",
                    edge.source,
                    edge.target
                );
            }
            let tikz = doc.serialize(TopologyFormat::Tikz).unwrap();
            assert!(tikz.contains("\\begin{tikzpicture}"));
        }
    }
}

#[test]
fn test_dot_graphml_and_csv_imports_sync_across_graph_theory_tools() {
    let mut tab = GraphTheoryTab::default();

    clear_test_announcements();
    let dot_k4 = "\u{feff}graph K4 {\n  0 -- 1 [weight=2.0];\n  0 -- 2;\n  0 -- 3;\n  1 -- 2;\n  1 -- 3;\n  2 -- 3;\n}\n";
    tab.import_topology_text("k4.dot", dot_k4)
        .expect("DOT K4 import succeeds");
    assert!(all_announcements()
        .iter()
        .any(|(m, p)| m.contains("Imported 4 nodes and 6 edges") && p == "polite"));
    verify_imported_sync_across_tools(&mut tab, 4, 3);

    let exported_graphml = tab
        .shared_topology()
        .unwrap()
        .serialize(TopologyFormat::GraphMl)
        .unwrap();
    tab.import_topology_text("k4.graphml", &exported_graphml)
        .expect("GraphML roundtrip import succeeds");
    verify_imported_sync_across_tools(&mut tab, 4, 3);

    let csv_star =
        "source\ttarget\tweight\ncenter\tleaf1\t1.0\ncenter\tleaf2\t2.5\ncenter\tleaf3\t1.5\n";
    tab.import_topology_text("star.tsv", csv_star)
        .expect("TSV star import succeeds");
    verify_imported_sync_across_tools(&mut tab, 4, 1);
}

#[test]
fn test_graph_theory_import_rejects_xxe_and_oversized_inputs_cleanly() {
    let mut tab = GraphTheoryTab::default();
    clear_test_announcements();
    let xxe = "<?xml version=\"1.0\"?><!DOCTYPE graphml [<!ENTITY xxe \"boom\">]><graphml/>";
    assert!(tab.import_topology_text("bad.graphml", xxe).is_err());
    assert!(all_announcements().iter().any(|(_, p)| p == "assertive"));

    let huge = "a".repeat(2 * 1024 * 1024 + 64);
    assert!(tab.import_topology_text("huge.dot", &huge).is_err());
}

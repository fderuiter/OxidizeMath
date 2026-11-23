#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::deferred_export_audit_tests::run_deferred_export_frame;
use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements};
use eframe::egui;

#[cfg(feature = "pure_math")]
pub(super) fn verify_topology_capability(
    meta_id: &str,
    tool: &dyn InteractiveTool,
    caps: ExportCapabilities,
) -> bool {
    if !caps.topology {
        assert!(
            !caps.topology_disabled_reason.is_empty(),
            "{meta_id} disables topology export without an explanatory reason"
        );
        return false;
    }
    assert!(
        caps.topology_disabled_reason.is_empty(),
        "{meta_id} advertises topology export but has non-empty disabled reason"
    );
    let doc = tool
        .export_topology_document()
        .unwrap_or_else(|| panic!("{meta_id} must implement export_topology_document"))
        .unwrap_or_else(|e| panic!("{meta_id} topology generation failed: {e}"));
    assert!(!doc.name.is_empty(), "{meta_id} emitted unnamed topology");
    true
}

#[cfg(not(feature = "pure_math"))]
pub(super) fn verify_topology_capability(
    meta_id: &str,
    _tool: &dyn InteractiveTool,
    caps: ExportCapabilities,
) -> bool {
    assert!(
        !caps.topology && !caps.topology_disabled_reason.is_empty(),
        "{meta_id} cannot advertise topology export without pure_math"
    );
    false
}

fn verify_single_topology_tool_exports(meta_id: &str, tool: Box<dyn InteractiveTool>) {
    let mut fw = SimulationFramework::new("__audit_topology__");
    fw.active_tool = Some(tool);
    for (fmt, marker) in [
        (ExportFormat::TopologyDot, "graph "),
        (ExportFormat::TopologyGraphMl, "<graphml"),
        (ExportFormat::TopologyTikz, "\\begin{tikzpicture}"),
        (ExportFormat::TopologySvg, "<svg"),
        (ExportFormat::TopologyCsv, "source,target,weight"),
    ] {
        clear_test_announcements();
        run_deferred_export_frame(&mut fw, fmt, false);
        assert_eq!(
            fw.export_status_severity,
            ExportStatusSeverity::Success,
            "{meta_id} failed {fmt:?}: {:?}",
            fw.export_status_message
        );
        let text = fw
            .export_queue
            .last_exported_content
            .as_deref()
            .unwrap_or_else(|| panic!("{meta_id} missing content for {fmt:?}"));
        assert!(
            text.contains(marker),
            "{meta_id} {fmt:?} missing {marker:?}: {text}"
        );
        assert!(all_announcements()
            .iter()
            .any(|(msg, prio)| msg.contains("saved successfully") && prio == "polite"));
    }

    let ctx = egui::Context::default();
    clear_test_announcements();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        fw.queue_topology_clipboard_export(ExportFormat::TopologyDot);
        fw.show(ctx, "audit_topology_clip");
    });
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let clip = fw
        .export_queue
        .last_clipboard_content
        .as_deref()
        .expect("clipboard topology recorded");
    assert!(clip.contains("graph "));
}

#[cfg(all(
    feature = "pure_math",
    feature = "epidemiology",
    feature = "biology",
    feature = "applied"
))]
#[test]
fn test_all_seven_topology_tools_deferred_exports_and_clipboard() {
    let topology_ids = [
        "GraphEditorTool",
        "AlgorithmVisualizerTool",
        "NetworkMetricsTool",
        "NetworkPropagationTool",
        "NeuralNetworkVizTool",
        "FractranMachineTool",
        "BraessNetworkParadoxTool",
    ];
    for id in topology_ids {
        let meta = inventory::iter::<crate::framework::ToolMetadata>
            .into_iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("Missing registered topology tool {id}"));
        verify_single_topology_tool_exports(id, (meta.build)());
    }

    let mut editor = crate::tabs::graph_theory::graph_editor::GraphEditorTool::default();
    editor
        .try_load_state(
            r#"{"node_positions":{"0":[120.0,140.0],"1":[240.0,80.0],"2":[240.0,200.0],"3":[360.0,140.0]},"edges":[[0,1,1.0],[1,3,1.0],[0,2,1.0],[2,3,1.0],[1,2,0.0]],"next_node_id":4,"congestion":{"enabled":true,"source_node":0,"sink_node":3,"total_demand":1.0,"shortcut_enabled":true}}"#,
        )
        .expect("valid Braess-enabled GraphEditorTool state");
    let caps = editor.export_capabilities();
    assert!(caps.dataset && caps.figure && caps.topology);
    verify_single_topology_tool_exports("GraphEditorTool(Braess)", Box::new(editor));
}

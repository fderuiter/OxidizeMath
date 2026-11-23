use crate::framework::InteractiveTool;
use eframe::egui::{self, vec2, Context, Event, Key, Modifiers, Pos2, RawInput, Rect};

fn screen_raw() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    }
}

fn tab_raw() -> RawInput {
    let mut raw = screen_raw();
    raw.events.push(Event::Key {
        key: Key::Tab,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    raw
}

fn assert_tool_canvas_tab_focus_and_accesskit_label(
    tool: &mut dyn InteractiveTool,
    expected_label_substring: &str,
) {
    let ctx = Context::default();
    ctx.enable_accesskit();

    let _ = ctx.run(screen_raw(), |ctx| tool.show(ctx));
    let out = ctx.run(screen_raw(), |ctx| tool.show(ctx));
    let update = out
        .platform_output
        .accesskit_update
        .unwrap_or_else(|| panic!("{} must emit AccessKit update", tool.name()));

    let matching_nodes: Vec<egui::accesskit::NodeId> = update
        .nodes
        .iter()
        .filter_map(|(id, node)| {
            let is_canvas_or_image = matches!(
                node.role(),
                egui::accesskit::Role::GraphicsDocument | egui::accesskit::Role::Image
            );
            let label = node.label().unwrap_or_default();
            let supports_focus = node.supports_action(egui::accesskit::Action::Focus);
            (is_canvas_or_image
                && !label.trim().is_empty()
                && label.contains(expected_label_substring)
                && supports_focus)
                .then_some(*id)
        })
        .collect();

    assert!(
        !matching_nodes.is_empty(),
        "Tool '{}' must expose a focusable GraphicsDocument/Image AccessKit node containing '{}'",
        tool.name(),
        expected_label_substring
    );

    // Step Tab across all controls until keyboard focus lands on one of the target canvas/image nodes.
    let mut reached_via_tab = false;
    for _ in 0..80 {
        let tab_out = ctx.run(tab_raw(), |ctx| tool.show(ctx));
        if let Some(upd) = tab_out.platform_output.accesskit_update {
            if matching_nodes.contains(&upd.focus) {
                reached_via_tab = true;
                break;
            }
        }
    }

    assert!(
        reached_via_tab,
        "Tool '{}' canvas/image widget must be reachable via Key::Tab traversal",
        tool.name()
    );
}

#[cfg(feature = "ai")]
#[test]
fn test_ai_custom_canvases_receive_tab_focus_and_accesskit_labels() {
    let mut attention = crate::tabs::ai::attention_maps::AttentionMapsTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(&mut attention, "Attention Weights");

    let mut q_table = crate::tabs::ai::q_table_inspector::QTableInspectorTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(&mut q_table, "Q-Table Heatmap Visualization");

    let mut grid_world = crate::tabs::ai::grid_world::GridWorldTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(&mut grid_world, "Grid World Visualization");
}

#[cfg(feature = "pure_math")]
#[test]
fn test_pure_math_custom_canvases_and_rasters_receive_tab_focus_and_labels() {
    let mut algo =
        crate::tabs::graph_theory::algorithm_visualizer::AlgorithmVisualizerTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut algo,
        "Graph Theory Algorithm Visualization",
    );

    let mut metrics = crate::tabs::graph_theory::network_metrics::NetworkMetricsTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(&mut metrics, "Network Metrics Visualization");

    let mut spiral = crate::tabs::number_theory::prime_spiral::PrimeSpiralWidget::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(&mut spiral, "Ulam Prime Spiral");
}

#[cfg(feature = "biology")]
#[test]
fn test_neuroscience_canvas_receives_tab_focus_and_accesskit_label() {
    let mut neuro = crate::tabs::neuroscience::neural_network_viz::NeuralNetworkVizTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(&mut neuro, "Neural Network Visualization");

    let mut sheet = crate::tabs::neuroscience::excitable_sheet::ExcitableSheetTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut sheet,
        "2D Excitable Sheet & Virtual Electrodes Visualization",
    );
}

#[cfg(feature = "physics")]
#[test]
fn test_medical_bioelectric_canvas_receives_tab_focus_and_accesskit_label() {
    let mut bio = crate::tabs::medical::bioelectric_leads::BioelectricLeadTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut bio,
        "Bioelectric 3-Lead Dipole Mapper Visualization",
    );
}

#[cfg(feature = "epidemiology")]
#[test]
fn test_epidemiology_canvas_receives_tab_focus_and_accesskit_label() {
    let mut prop =
        crate::tabs::epidemiology::network_propagation::NetworkPropagationTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut prop,
        "Network Propagation Visualization",
    );
}

#[cfg(feature = "physics")]
#[test]
fn test_physics_raster_tools_receive_tab_focus_and_alt_text() {
    let mut fractals = crate::tabs::chaos::fractals::FractalViewer::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(&mut fractals, "Mandelbrot");

    let mut ising = crate::tabs::solid_state::ising::IsingModelTool::default();
    // Allow background Ising worker to publish its initial spin lattice snapshot.
    for _ in 0..40 {
        if ising.worker.latest_snapshot().is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert_tool_canvas_tab_focus_and_accesskit_label(&mut ising, "2D Ising Model spin lattice");

    let mut radar = crate::tabs::medical::radar_gating::RadarGatingTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut radar,
        "60 GHz MIMO Radar Room Visualization",
    );
}

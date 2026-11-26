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

    let mut closed_loop = crate::tabs::ai::closed_loop_control::ClosedLoopControlTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut closed_loop,
        "Closed-Loop Phase & Perturbation Visualization",
    );
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

    let mut hyp_disk = crate::tabs::hyperbolic_geometry::disk_tool::HyperbolicDiskTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut hyp_disk,
        "Hyperbolic Disk Tessellation Canvas",
    );

    let mut hyp_billiards =
        crate::tabs::hyperbolic_geometry::billiards_tool::HyperbolicBilliardsTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut hyp_billiards,
        "Hyperbolic Geodesic Billiards Canvas",
    );

    let mut fractran = crate::tabs::number_theory::fractran_machine::FractranMachineTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut fractran,
        "FRACTRAN Minsky Counter-Machine Graph",
    );

    let mut chladni = crate::tabs::analysis::chladni_wave_tank::ChladniWaveTankTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut chladni,
        "2D Chladni Plate & Wave Tank Visualization",
    );

    let mut knot =
        crate::tabs::geometry_topology::knot_braid_studio::KnotBraidStudioTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(&mut knot, "2D Planar Link Diagram Canvas");
}

#[cfg(feature = "biology")]
#[test]
fn test_neuroscience_canvas_receives_tab_focus_and_accesskit_label() {
    let mut neuro = crate::tabs::neuroscience::neural_network_viz::NeuralNetworkVizTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(&mut neuro, "Neural Network Visualization");

    let mut lenia = crate::tabs::morphogenesis::LeniaTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut lenia,
        "Lenia Studio Continuous Automaton Canvas",
    );

    let mut botanical = crate::tabs::morphogenesis::PhyllotaxisLSystemTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut botanical,
        "Botanical Phyllotaxis and L-System Morphogenesis Canvas",
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
    let mut hamiltonian = crate::tabs::chaos::hamiltonian::HamiltonianChaosLab::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut hamiltonian,
        "Poincare Surface of Section",
    );

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

    let mut optics = crate::tabs::optics_waves::OpticsWaveTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(&mut optics, "2D Wave & Caustics Playground");

    let mut radar = crate::tabs::medical::radar_gating::RadarGatingTool::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut radar,
        "60 GHz MIMO Radar Room Visualization",
    );

    let mut sandpile = crate::tabs::solid_state::sandpile::AbelianSandpileStudio::default();
    assert_tool_canvas_tab_focus_and_accesskit_label(
        &mut sandpile,
        "2D Abelian Sandpile Lattice & Recurrent Crystal",
    );
}

fn key_raw(key: Key) -> RawInput {
    let mut raw = screen_raw();
    raw.events.push(Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    raw
}

fn find_and_focus_plot3d_document(
    ctx: &Context,
    tool: &mut dyn InteractiveTool,
    expected_title: &str,
) {
    let _ = ctx.run(screen_raw(), |c| tool.show(c));
    let out = ctx.run(screen_raw(), |c| tool.show(c));
    let update = out
        .platform_output
        .accesskit_update
        .unwrap_or_else(|| panic!("{} must emit AccessKit update", tool.name()));

    let target_id = update
        .nodes
        .iter()
        .find_map(|(id, node)| {
            let label = node.label().unwrap_or_default();
            let value = node.value().unwrap_or_default();
            (node.role() == egui::accesskit::Role::GraphicsDocument
                && label.contains("Interactive Plot (")
                && label.contains(expected_title)
                && value.contains("3D View")
                && value.contains("Pitch")
                && value.contains("Yaw")
                && value.contains("Zoom")
                && node.supports_action(egui::accesskit::Action::Focus))
            .then_some(*id)
        })
        .unwrap_or_else(|| {
            panic!(
                "Tool '{}' missing focusable Role::GraphicsDocument titled '{}' with 3D value",
                tool.name(),
                expected_title
            )
        });

    let mut reached_via_tab = false;
    for _ in 0..140 {
        let tab_out = ctx.run(tab_raw(), |c| tool.show(c));
        if tab_out
            .platform_output
            .accesskit_update
            .is_some_and(|upd| upd.focus == target_id)
        {
            reached_via_tab = true;
            break;
        }
    }
    assert!(
        reached_via_tab,
        "Tool '{}' Plot3DViewport ('{}') must be reachable via Key::Tab",
        tool.name(),
        expected_title
    );
}

fn assert_plot3d_tool_graphics_document_and_world_xyz(
    tool: &mut dyn InteractiveTool,
    expected_title: &str,
) {
    let ctx = Context::default();
    ctx.enable_accesskit();
    find_and_focus_plot3d_document(&ctx, tool, expected_title);

    crate::accessibility::clear_test_announcements();
    let _ = ctx.run(key_raw(Key::ArrowRight), |c| tool.show(c));
    let _ = ctx.run(key_raw(Key::ArrowDown), |c| tool.show(c));
    let anns = crate::accessibility::all_announcements();
    assert!(
        anns.iter().any(|(msg, prio)| {
            prio == "polite" && msg.contains("Point X:") && msg.contains("Y:") && msg.contains("Z:")
        }),
        "Tool '{}' arrow keys did not announce world (X, Y, Z) point; got: {anns:?}",
        tool.name()
    );
    assert!(
        !anns
            .iter()
            .any(|(msg, _)| msg.starts_with("Dataset ") && !msg.contains("Z:")),
        "Tool '{}' fell back to 2D projected point announcement: {anns:?}",
        tool.name()
    );

    crate::accessibility::clear_test_announcements();
    let _ = ctx.run(key_raw(Key::Home), |c| tool.show(c));
    let home_anns = crate::accessibility::all_announcements();
    assert!(
        home_anns
            .iter()
            .any(|(msg, prio)| prio == "polite" && msg.contains("Camera view reset")),
        "Tool '{}' Key::Home did not announce Camera3D reset; got: {home_anns:?}",
        tool.name()
    );
    assert!(
        !home_anns
            .iter()
            .any(|(msg, _)| msg.contains("Plot bounds reset")),
        "Tool '{}' leaked 2D 'Plot bounds reset' on Key::Home",
        tool.name()
    );
}

#[cfg(feature = "pure_math")]
#[test]
fn test_pure_math_plot3d_tools_expose_graphics_document_values_and_world_xyz() {
    let mut surface = crate::tabs::geometry_topology::surface_viewer::SurfaceViewer::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut surface, "Surface");

    let mut curvature =
        crate::tabs::geometry_topology::curvature_heatmap::CurvatureHeatmap::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut curvature, "Curvature");

    let mut riemannian =
        crate::tabs::geometry_topology::riemannian_studio::RiemannianPdeStudio::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(
        &mut riemannian,
        "3D Riemannian PDE Manifold Viewport",
    );

    let mut knot =
        crate::tabs::geometry_topology::knot_braid_studio::KnotBraidStudioTool::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut knot, "Knot Braid 3D");

    let mut vr = crate::tabs::geometry_topology::vietoris_rips::VietorisRipsTool::default();
    vr.chaos.playing = false;
    let pts = vec![
        math_explorer::pure_math::statistics::tda::Point3D::new(-2.0, -1.0, 0.5),
        math_explorer::pure_math::statistics::tda::Point3D::new(2.0, -1.0, -0.5),
        math_explorer::pure_math::statistics::tda::Point3D::new(0.0, 2.5, 1.5),
    ];
    vr.load_validated_3d_points(pts, 6.0)
        .expect("valid 3D cloud");
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut vr, "Vietoris Rips 3D");
}

#[cfg(all(feature = "pure_math", feature = "biology"))]
#[test]
fn test_biology_plot3d_tools_expose_graphics_document_values_and_world_xyz() {
    let mut morpho =
        crate::tabs::geometry_topology::surface_morphogenesis::SurfaceMorphogenesisTool::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut morpho, "Surface Morphogenesis");

    let mut cortical =
        crate::tabs::neuroscience::cortical_morphometry::CorticalMorphometryTool::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut cortical, "Cortical Morphometry");
}

#[cfg(feature = "physics")]
#[test]
fn test_physics_plot3d_tools_expose_graphics_document_values_and_world_xyz() {
    let mut crystal = crate::tabs::solid_state::crystal_viewer::CrystalViewer::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut crystal, "Crystal");

    let mut attractor = crate::tabs::chaos::attractors::AttractorPlotter::default();
    for _ in 0..8 {
        attractor.step_simulation();
    }
    attractor.execution_state = crate::tabs::chaos::attractors::SolverExecutionState::Paused;
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut attractor, "Attractor");

    let mut spin = crate::tabs::quantum::spin_viz::SpinVisualizer::default();
    spin.step_once();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut spin, "3D Quantum Bloch Sphere");

    let mut volumetric =
        crate::tabs::quantum::volumetric_studio::VolumetricFieldStudioTool::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut volumetric, "Volumetric Field Studio");

    let mut geodesic = crate::tabs::relativity::GeodesicPlaygroundTool::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut geodesic, "Relativity Flamm 3D");

    let mut mri = crate::tabs::mri::kspace_tool::MriKSpaceTool::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut mri, "MRI Voxel Bloch 3D");
}

#[cfg(feature = "ai")]
#[test]
fn test_ai_plot3d_tools_expose_graphics_document_values_and_world_xyz() {
    let mut splat = crate::tabs::ai::gaussian_splatting::GaussianSplattingTool::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut splat, "Gaussian Splatting");

    let mut landscape = crate::tabs::ai::loss_landscape::LossLandscapeTool::default();
    assert_plot3d_tool_graphics_document_and_world_xyz(&mut landscape, "Loss Landscape 3D");
}

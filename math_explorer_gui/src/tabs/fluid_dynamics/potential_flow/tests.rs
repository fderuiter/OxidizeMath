#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{ShaderDialect, SimulationFramework};
use std::collections::BTreeMap;

fn run_frame(ctx: &egui::Context, tool: &mut PotentialFlowTool) {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(960.0, 640.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(input, |c| tool.show(c));
}

#[test]
fn potential_flow_prebuilds_default_buffers_and_reuses_across_idle_frames() {
    let mut tool = PotentialFlowTool::default();
    assert_eq!(tool.grid_revision, 1);
    assert_eq!(tool.cached_revision, 1);
    assert!(!tool.origins.is_empty());
    assert_eq!(tool.origins.len(), tool.tips.len());
    assert!(tool.origins.capacity() >= (tool.grid_size + 1).pow(2));
    assert!(tool.tips.capacity() >= (tool.grid_size + 1).pow(2));

    {
        let borrowed_origins = PlotPoints::Borrowed(&tool.origins);
        let borrowed_tips = PlotPoints::Borrowed(&tool.tips);
        assert!(matches!(borrowed_origins, PlotPoints::Borrowed(_)));
        assert!(matches!(borrowed_tips, PlotPoints::Borrowed(_)));
        assert_eq!(borrowed_origins.points().len(), tool.origins.len());
    }

    let origins_ptr = tool.origins.as_ptr();
    let tips_ptr = tool.tips.as_ptr();
    let origins_cap = tool.origins.capacity();
    let ctx = egui::Context::default();
    run_frame(&ctx, &mut tool);
    run_frame(&ctx, &mut tool);

    assert_eq!(tool.grid_revision, 1);
    assert_eq!(tool.cached_revision, 1);
    assert_eq!(tool.origins.as_ptr(), origins_ptr);
    assert_eq!(tool.tips.as_ptr(), tips_ptr);
    assert_eq!(tool.origins.capacity(), origins_cap);
}

#[test]
fn potential_flow_clamps_high_grid_state_and_invalidates_on_load_add_and_clear() {
    let ctx = egui::Context::default();
    let mut tool = PotentialFlowTool::default();
    let high_grid = r#"{"elements":[{"element_type":"Uniform","strength":10.0,"x":0.0,"y":0.0,"angle":0.0}],"grid_size":1000,"view_range":10.0}"#;
    tool.try_load_state(high_grid)
        .expect("grid_size 1000 should load and clamp to interactive cap");
    assert_eq!(tool.grid_size, crate::presets::MAX_INTERACTIVE_GRID_SIZE);
    assert!(tool.has_finite_parameters());
    assert_ne!(tool.cached_revision, tool.grid_revision);

    run_frame(&ctx, &mut tool);
    assert_eq!(tool.cached_revision, tool.grid_revision);
    assert_eq!(tool.origins.len(), (tool.grid_size + 1).pow(2));

    let rev_before_add = tool.grid_revision;
    tool.selected_element_type = ElementType::Vortex;
    tool.new_element_strength = 25.0;
    tool.add_element();
    assert!(tool.grid_revision > rev_before_add);
    assert_ne!(tool.cached_revision, tool.grid_revision);
    run_frame(&ctx, &mut tool);
    assert_eq!(tool.cached_revision, tool.grid_revision);

    let cap_before_clear = tool.origins.capacity();
    tool.clear();
    assert!(tool.origins.is_empty() && tool.tips.is_empty());
    assert_eq!(tool.origins.capacity(), cap_before_clear);
    assert_ne!(tool.cached_revision, tool.grid_revision);
    run_frame(&ctx, &mut tool);
    assert_eq!(tool.cached_revision, tool.grid_revision);
    assert!(tool.origins.is_empty() && tool.tips.is_empty());
}

#[test]
fn potential_flow_grid_size_and_view_range_preserve_or_grow_capacity_cleanly() {
    let ctx = egui::Context::default();
    let mut tool = PotentialFlowTool {
        elements: vec![SerializedFlowElement {
            element_type: ElementType::Uniform,
            strength: 10.0,
            x: 0.0,
            y: 0.0,
            angle: 0.0,
        }],
        ..Default::default()
    };
    tool.rebuild_field();
    run_frame(&ctx, &mut tool);
    assert_eq!(tool.origins.len(), (20_usize + 1).pow(2));

    tool.grid_size = 40;
    run_frame(&ctx, &mut tool);
    assert_eq!(tool.cached_revision, tool.grid_revision);
    assert_eq!(tool.origins.len(), (40_usize + 1).pow(2));
    assert!(tool.origins.capacity() >= (40_usize + 1).pow(2));

    let grown_origins_ptr = tool.origins.as_ptr();
    let grown_tips_ptr = tool.tips.as_ptr();
    let grown_cap = tool.origins.capacity();

    tool.grid_size = 12;
    run_frame(&ctx, &mut tool);
    assert_eq!(tool.origins.len(), (12_usize + 1).pow(2));
    assert_eq!(tool.origins.as_ptr(), grown_origins_ptr);
    assert_eq!(tool.tips.as_ptr(), grown_tips_ptr);
    assert_eq!(tool.origins.capacity(), grown_cap);

    let prev_rev = tool.cached_revision;
    tool.view_range = 25.0;
    run_frame(&ctx, &mut tool);
    assert!(tool.cached_revision > prev_rev);
    assert!((tool.origins[0].x - (-25.0)).abs() < 1e-9);
    assert_eq!(tool.origins.as_ptr(), grown_origins_ptr);
}

#[test]
fn potential_flow_skips_source_singularity_and_rebuilds_in_place_without_allocating() {
    let mut tool = PotentialFlowTool {
        elements: vec![SerializedFlowElement {
            element_type: ElementType::Source,
            strength: 15.0,
            x: 0.0,
            y: 0.0,
            angle: 0.0,
        }],
        grid_size: 10,
        view_range: 5.0,
        ..Default::default()
    };
    tool.rebuild_field();
    tool.rebuild_arrows();

    assert_eq!(tool.origins.len(), (10_usize + 1).pow(2) - 1);
    assert!(tool
        .origins
        .iter()
        .chain(tool.tips.iter())
        .all(|p| p.x.is_finite() && p.y.is_finite()));

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            tool.reconcile_arrows();
            tool.grid_size = 8;
            tool.reconcile_arrows();
            assert_eq!(tool.origins.len(), (8_usize + 1).pow(2) - 1);
            let borrowed_o = PlotPoints::Borrowed(&tool.origins);
            let borrowed_t = PlotPoints::Borrowed(&tool.tips);
            std::hint::black_box((borrowed_o.points().len(), borrowed_t.points().len()));
        });
    }
}

#[test]
fn test_default_cylinder_streamlines_stagnation_dye_and_probe() {
    let mut tool = PotentialFlowTool::default();
    assert!(!tool.cached_streamlines.is_empty());
    assert!(!tool.cached_polylines.is_empty());
    assert_eq!(tool.cached_stagnation.len(), 2);
    assert_eq!(tool.dye_streaks.len(), DYE_EMITTER_COUNT);
    assert!(tool.dye_streaks.iter().any(|s| s.trail.len() > 2));

    let readout = tool.evaluate_probe();
    assert!(readout.bernoulli_pressure.is_finite());
    assert!(readout.convective_acceleration.x.is_finite());

    let head_before = tool.dye_streaks[0].head;
    tool.step_simulation();
    assert_ne!(tool.dye_streaks[0].head, head_before);
}

#[test]
fn test_lagally_pressure_drag_and_canvas_accessibility() {
    let mut oval = PotentialFlowTool {
        elements: presets::rankine_oval_elements(),
        ..Default::default()
    };
    oval.rebuild_field();
    let oval_obs: BTreeMap<_, _> = oval.observables().into_iter().collect();
    assert_eq!(oval_obs["pressure_drag"], 0.0);
    assert_eq!(oval_obs["stagnation_count"], 2.0);

    // Removing the balancing Sink creates a net Source Q=75 -> Lagally drag |U_inf * Q| = 750
    oval.elements.pop();
    oval.rebuild_field();
    let half_body_obs: BTreeMap<_, _> = oval.observables().into_iter().collect();
    assert!((half_body_obs["pressure_drag"] - 750.0).abs() < 1e-6);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(960.0, 640.0),
            )),
            ..Default::default()
        },
        |c| oval.show(c),
    );
    assert!(out.platform_output.accesskit_update.is_some());
}

#[test]
fn test_potential_flow_brush_sculpt_and_erase() {
    use crate::framework::{BrushMode, BrushStroke, InputMode, InteractionContext};
    use std::collections::HashSet;

    let mut tool = PotentialFlowTool::default();
    let ctx = egui::Context::default();
    let mut dummy_resp = None;
    let _ = ctx.run(egui::RawInput::default(), |c| {
        egui::Window::new("dummy").show(c, |ui| {
            dummy_resp = Some(ui.label("probe"));
        });
    });
    let resp = dummy_resp.expect("dummy response");
    let count_before = tool.elements.len();
    let inject_ctx = InteractionContext {
        pointer_pos: Some(egui::pos2(2.0, 2.0)),
        delta: egui::Vec2::ZERO,
        is_dragging: true,
        is_clicked: false,
        response: &resp,
        input_mode: InputMode::Mouse,
        multi_touch: None,
        keys_down: HashSet::new(),
        modifiers: egui::Modifiers::NONE,
        brush_stroke: Some(BrushStroke {
            start: [2.0, 2.0],
            end: [2.0, 2.0],
            radius: 1.0,
            mode: BrushMode::FlowSource,
            drag_vector: [0.0, 0.0],
        }),
    };
    tool.on_brush(&inject_ctx);
    assert_eq!(tool.elements.len(), count_before + 1);

    let erase_ctx = InteractionContext {
        brush_stroke: Some(BrushStroke {
            start: [2.0, 2.0],
            end: [2.0, 2.0],
            radius: 1.5,
            mode: BrushMode::Erase,
            drag_vector: [0.0, 0.0],
        }),
        ..inject_ctx
    };
    tool.on_brush(&erase_ctx);
    assert_eq!(tool.elements.len(), count_before);
}

#[test]
fn test_presets_observables_state_roundtrip_and_glsl_export() {
    let mut tool = PotentialFlowTool {
        elements: presets::lifting_cylinder_elements(),
        show_stagnation_points: false,
        ..Default::default()
    };
    tool.rebuild_field();
    assert_eq!(tool.cached_stagnation.len(), 2);
    assert!(tool.cached_stagnation.iter().all(|p| p[1] < -0.5));

    let obs: BTreeMap<_, _> = tool.observables().into_iter().collect();
    assert!((obs["kutta_joukowski_lift"] - 1256.637).abs() < 1.0);
    assert_eq!(obs["pressure_drag"], 0.0);
    assert_eq!(obs["stagnation_count"], 2.0);

    // Supercritical circulation masks interior dipole stagnation point so exterior count is 1.0
    tool.elements[2].strength = -300.0;
    tool.rebuild_field();
    let sup_obs: BTreeMap<_, _> = tool.observables().into_iter().collect();
    assert_eq!(sup_obs["stagnation_count"], 1.0);

    let saved = tool.save_state().expect("save_state");
    let mut restored = PotentialFlowTool::default();
    restored.try_load_state(&saved).expect("load_state");
    assert_eq!(restored.elements, tool.elements);
    assert!(!restored.show_stagnation_points);

    let glsl = tool
        .export_glsl_shader(ShaderDialect::WebGl2Es300)
        .expect("WebGL2 shader");
    assert!(glsl.contains("flow_vortex"));
    assert!(glsl.contains("float psi = -(gamma * INV_TWO_PI) * 0.5 * log(r2);"));
}

#[test]
fn test_canvas_keyboard_nudging_stamping_and_deletion() {
    let ctx = egui::Context::default();
    let mut tool = PotentialFlowTool {
        selected_element_type: ElementType::Source,
        new_element_strength: 35.0,
        focused_reticle: [1.0, 2.0],
        ..Default::default()
    };

    let initial_len = tool.elements.len();
    tool.apply_keyboard_commands(&ctx, [0.0, 0.0], true, false, false, false);
    assert_eq!(tool.elements.len(), initial_len + 1);
    assert_eq!(tool.selected_element, Some(initial_len));

    tool.apply_keyboard_commands(&ctx, [0.5, -0.5], false, false, false, false);
    assert!((tool.elements[initial_len].x - 1.5).abs() < 1e-6);
    assert!((tool.elements[initial_len].y - 1.5).abs() < 1e-6);

    tool.apply_keyboard_commands(&ctx, [-1.0, 0.5], false, false, false, true);
    assert!((tool.probe_pos[0] - (-5.0)).abs() < 1e-6);

    tool.apply_keyboard_commands(&ctx, [0.0, 0.0], false, true, false, false);
    assert_eq!(tool.elements.len(), initial_len);

    let mut fw = SimulationFramework::new("fluid_dynamics");
    fw.select_tool_by_name("PotentialFlowTool");
    let _ = ctx.run(egui::RawInput::default(), |c| fw.show(c, "pf_test"));
}

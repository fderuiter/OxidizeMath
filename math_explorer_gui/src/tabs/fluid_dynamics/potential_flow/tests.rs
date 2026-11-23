#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;

fn raw_screen(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 900.0),
        )),
        events,
        ..Default::default()
    }
}

fn key_press_events(key: egui::Key) -> Vec<egui::Event> {
    vec![
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: false,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]
}

#[test]
fn test_canvas_handle_drag_updates_singularity_and_clamps_bounds() {
    let ctx = egui::Context::default();
    let mut tool = PotentialFlowTool::default();
    let _ = ctx.run(raw_screen(Vec::new()), |c| tool.show(c));
    let _ = ctx.run(raw_screen(Vec::new()), |c| tool.show(c));
    let (_, doublet_screen) = tool
        .cached_handles
        .iter()
        .copied()
        .find(|(t, _)| *t == DragTarget::Element(1))
        .expect("Doublet handle cached on canvas");

    let target_screen = doublet_screen + egui::vec2(60.0, -45.0);
    let _ = ctx.run(
        raw_screen(vec![
            egui::Event::PointerMoved(doublet_screen),
            egui::Event::PointerButton {
                pos: doublet_screen,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]),
        |c| tool.show(c),
    );
    assert_eq!(tool.active_drag, Some(DragTarget::Element(1)));

    let _ = ctx.run(
        raw_screen(vec![egui::Event::PointerMoved(target_screen)]),
        |c| {
            tool.show(c);
        },
    );
    assert!(tool.elements[1].x > 0.1 && tool.elements[1].y > 0.1);

    tool.apply_drag_target(DragTarget::Element(1), 999.0, -999.0);
    assert_eq!(tool.elements[1].x, FLOW_POS_MAX);
    assert_eq!(tool.elements[1].y, FLOW_POS_MIN);
}

#[test]
fn test_click_to_stamp_and_wheel_strength_tuning() {
    let ctx = egui::Context::default();
    let mut tool = PotentialFlowTool {
        stamp_mode: true,
        selected_element_type: ElementType::Vortex,
        new_element_strength: 35.0,
        ..Default::default()
    };
    let _ = ctx.run(raw_screen(Vec::new()), |c| tool.show(c));
    let _ = ctx.run(raw_screen(Vec::new()), |c| tool.show(c));
    let initial_len = tool.elements.len();
    let (_, doublet_pos) = tool
        .cached_handles
        .iter()
        .copied()
        .find(|(t, _)| *t == DragTarget::Element(1))
        .expect("Doublet handle cached");
    let empty_pos = doublet_pos + egui::vec2(90.0, -70.0);

    let _ = ctx.run(
        raw_screen(vec![
            egui::Event::PointerMoved(empty_pos),
            egui::Event::PointerButton {
                pos: empty_pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]),
        |c| tool.show(c),
    );
    let _ = ctx.run(
        raw_screen(vec![egui::Event::PointerButton {
            pos: empty_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]),
        |c| tool.show(c),
    );
    assert_eq!(tool.elements.len(), initial_len + 1);
    assert_eq!(
        tool.elements.last().unwrap().element_type,
        ElementType::Vortex
    );

    let (_, vortex_screen) = tool
        .cached_handles
        .iter()
        .copied()
        .find(|(t, _)| *t == DragTarget::Element(initial_len))
        .expect("Stamped vortex handle present");
    let scroll_input = raw_screen(vec![
        egui::Event::PointerMoved(vortex_screen),
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, 12.0),
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    let _ = ctx.run(scroll_input, |c| tool.show(c));
    assert!((tool.elements[initial_len].strength - 37.5).abs() < 1e-6);
}

#[test]
fn test_keyboard_reticle_nudge_stamp_delete_and_unlocked_tab() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = PotentialFlowTool::default();
    tool.brush_kit.interaction_mode = crate::framework::BrushInteractionMode::Pan;
    let _ = ctx.run(raw_screen(Vec::new()), |c| tool.show(c));
    assert!(tool.focused_canvas_id().is_none());
    let canvas_id = tool.last_canvas_id.expect("canvas id recorded");
    ctx.memory_mut(|m| m.request_focus(canvas_id));
    let _ = ctx.run(raw_screen(Vec::new()), |c| tool.show(c));
    assert_eq!(tool.focused_canvas_id(), Some(canvas_id));
    assert_eq!(ctx.memory(|m| m.focused()), Some(canvas_id));

    let orig_reticle = tool.reticle;
    let _ = ctx.run(raw_screen(key_press_events(egui::Key::ArrowRight)), |c| {
        tool.show(c)
    });
    assert!((tool.reticle[0] - (orig_reticle[0] + 0.5)).abs() < 1e-6);
    assert_eq!(ctx.memory(|m| m.focused()), Some(canvas_id));
    let live_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert!(live_msg.as_deref().is_some_and(|m| m.contains("Reticle")));

    let before_count = tool.elements.len();
    let _ = ctx.run(raw_screen(key_press_events(egui::Key::Enter)), |c| {
        tool.show(c)
    });
    assert_eq!(tool.elements.len(), before_count + 1);

    let _ = ctx.run(raw_screen(key_press_events(egui::Key::Delete)), |c| {
        tool.show(c)
    });
    assert_eq!(tool.elements.len(), before_count);

    let _ = ctx.run(raw_screen(key_press_events(egui::Key::Tab)), |c| {
        tool.show(c)
    });
    assert_ne!(ctx.memory(|m| m.focused()), Some(canvas_id));
}

#[test]
fn test_step_simulation_advances_rk4_dye_and_presets_load() {
    let mut tool = PotentialFlowTool::default();
    for preset in [
        crate::starter_gallery::PRESET_LIFTING_CYLINDER,
        crate::starter_gallery::PRESET_RANKINE_OVAL,
        crate::starter_gallery::PRESET_VORTEX_STREET,
    ] {
        tool.import_preset_json(preset)
            .expect("Aerodynamic preset must load");
        assert!(tool.has_finite_parameters());
        let before_head = tool.dye_probes[0].head;
        tool.step_simulation();
        let after_head = tool.dye_probes[0].head;
        assert_ne!(before_head, after_head);
        let readout = tool
            .active_probe_readout()
            .expect("Probe readout available");
        assert!(readout.bernoulli_pressure.is_finite());
        assert!(readout.convective_acceleration.x.is_finite());
    }
}

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

    // Shrinking grid_size clears and repopulates without reallocating.
    tool.grid_size = 12;
    run_frame(&ctx, &mut tool);
    assert_eq!(tool.origins.len(), (12_usize + 1).pow(2));
    assert_eq!(tool.origins.as_ptr(), grown_origins_ptr);
    assert_eq!(tool.tips.as_ptr(), grown_tips_ptr);
    assert_eq!(tool.origins.capacity(), grown_cap);

    // Updating view_range invalidates and updates coordinates in-place.
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

    // (0.0, 0.0) lands on the central grid node where Source::velocity returns zero.
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
        });
    }
}

#[test]
fn test_cylinder_streamlines_stagnation_and_kutta_lift_observables() {
    let mut tool = PotentialFlowTool::default();
    let (streamlines, stag) = tool.ensure_cached_contours();
    assert!(
        !streamlines.is_empty(),
        "streamline contours must be generated"
    );
    assert_eq!(
        stag.len(),
        2,
        "symmetric cylinder has 2 exterior stagnation points"
    );

    let mut spinning = PotentialFlowTool::default();
    spinning.elements.push(SerializedFlowElement {
        element_type: ElementType::Vortex,
        strength: -40.0 * PI,
        x: 0.0,
        y: 0.0,
        angle: 0.0,
    });
    spinning.rebuild_field();
    let obs = spinning.flow_observables();
    let lift = obs
        .iter()
        .find(|(k, _)| *k == "kutta_joukowski_lift")
        .unwrap()
        .1;
    let drag = obs.iter().find(|(k, _)| *k == "pressure_drag").unwrap().1;
    let stag_count = obs
        .iter()
        .find(|(k, _)| *k == "stagnation_count")
        .unwrap()
        .1;
    assert!(lift > 1250.0 && drag.abs() < 1e-9);
    assert_eq!(stag_count, 2.0);

    spinning.elements[2].strength = -300.0;
    spinning.rebuild_field();
    let sup_stag = spinning
        .flow_observables()
        .into_iter()
        .find(|(k, _)| *k == "stagnation_count")
        .unwrap()
        .1;
    assert_eq!(
        sup_stag, 1.0,
        "supercritical spin detaches a single exterior saddle"
    );
}

#[test]
fn test_spatial_brush_sculpt_mode_keyboard_stamp_and_erase() {
    let ctx = egui::Context::default();
    let mut tool = PotentialFlowTool::default();
    tool.brush_kit.interaction_mode = crate::framework::BrushInteractionMode::Sculpt;
    tool.brush_kit.mode = crate::framework::BrushMode::FlowVortex;
    tool.brush_kit.radius = 1.0;
    let initial_len = tool.elements.len();

    let _ = ctx.run(raw_screen(Vec::new()), |c| tool.show(c));
    let canvas_id = tool.last_canvas_id.expect("canvas id recorded");
    ctx.memory_mut(|m| m.request_focus(canvas_id));
    let _ = ctx.run(raw_screen(Vec::new()), |c| tool.show(c));
    assert_eq!(tool.focused_canvas_id(), Some(canvas_id));

    for _ in 0..3 {
        let _ = ctx.run(raw_screen(key_press_events(egui::Key::ArrowRight)), |c| {
            tool.show(c)
        });
    }
    let _ = ctx.run(raw_screen(key_press_events(egui::Key::Space)), |c| {
        tool.show(c)
    });
    assert_eq!(tool.elements.len(), initial_len + 1);
    let stamped = tool.elements.last().unwrap();
    assert_eq!(stamped.element_type, ElementType::Vortex);
    assert!((stamped.x - 3.0).abs() < 1e-6);

    tool.brush_kit.mode = crate::framework::BrushMode::Erase;
    let _ = ctx.run(raw_screen(key_press_events(egui::Key::Enter)), |c| {
        tool.show(c)
    });
    assert_eq!(tool.elements.len(), initial_len);
}

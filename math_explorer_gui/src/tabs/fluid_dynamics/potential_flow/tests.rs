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
    let _ = ctx.run(raw_screen(Vec::new()), |c| tool.show(c));
    let canvas_id = tool.focused_canvas_id().expect("canvas id recorded");
    ctx.memory_mut(|m| m.request_focus(canvas_id));
    let _ = ctx.run(raw_screen(Vec::new()), |c| tool.show(c));
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

#[test]
fn potential_flow_clamps_high_grid_state_and_caches_static_arrow_field() {
    let mut tool = PotentialFlowTool::default();
    let high_grid = r#"{"elements":[{"element_type":"Uniform","strength":10.0,"x":0.0,"y":0.0,"angle":0.0}],"grid_size":1000,"view_range":10.0}"#;
    tool.try_load_state(high_grid)
        .expect("grid_size 1000 should load and clamp to interactive cap");
    assert_eq!(tool.grid_size, crate::presets::MAX_INTERACTIVE_GRID_SIZE);
    assert!(tool.has_finite_parameters());

    let mut live_tool = PotentialFlowTool::default();
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| live_tool.show(c));
    let _ = ctx.run(egui::RawInput::default(), |c| live_tool.show(c));
    assert_eq!(live_tool.cache.eval_count, 1);

    live_tool.grid_size = 24;
    let _ = ctx.run(egui::RawInput::default(), |c| live_tool.show(c));
    assert_eq!(live_tool.cache.eval_count, 2);
}

#[test]
fn test_cylinder_streamlines_stagnation_and_kutta_lift_observables() {
    let mut tool = PotentialFlowTool::default();
    let (streamlines, stag) = tool.ensure_cached_contours();
    assert!(!streamlines.is_empty(), "streamline contours must be generated");
    assert_eq!(stag.len(), 2, "symmetric cylinder has 2 exterior stagnation points");

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
    assert_eq!(sup_stag, 1.0, "supercritical spin detaches a single exterior saddle");
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
    let canvas_id = tool.focused_canvas_id().expect("canvas id recorded");
    ctx.memory_mut(|m| m.request_focus(canvas_id));
    let _ = ctx.run(raw_screen(Vec::new()), |c| tool.show(c));

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

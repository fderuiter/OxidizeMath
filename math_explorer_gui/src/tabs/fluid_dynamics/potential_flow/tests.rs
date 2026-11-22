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

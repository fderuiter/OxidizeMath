use crate::app::ui_panels::render_help_window;
use crate::command_palette::CommandPalette;
use crate::starter_gallery::{show_starter_gallery_overlay, StarterCard};
use eframe::egui::{self, vec2, Context, Event, Id, Key, Modifiers, Pos2, RawInput, Rect};
use egui_plot::{
    show_export_modal, CallbackExportSource, ExportModalConfig, ExportModalState, Legend, Line,
    Plot, PlotMemory,
};

fn screen_raw() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    }
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

struct DummyTab(&'static str);

impl crate::tabs::ExplorerTab for DummyTab {
    fn name(&self) -> &'static str {
        self.0
    }
    fn show(&mut self, _ctx: &Context, _frame: &mut eframe::Frame) {}
}

#[test]
fn test_command_palette_escape_tab_containment_and_focus_restore() {
    let ctx = Context::default();
    let tabs: Vec<Box<dyn crate::tabs::ExplorerTab>> =
        vec![Box::new(DummyTab("Physics")), Box::new(DummyTab("AI"))];
    let mut palette = CommandPalette::default();

    let step = |raw: RawInput, palette: &mut CommandPalette| -> (Id, Id) {
        let mut trigger = Id::NULL;
        let mut other_bg = Id::NULL;
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                trigger = ui.button("Open Palette").id;
                other_bg = ui.button("Background Button").id;
            });
            let _ = palette.show(ctx, &tabs);
        });
        (trigger, other_bg)
    };

    let (trigger_id, other_bg_id) = step(screen_raw(), &mut palette);
    ctx.memory_mut(|m| m.request_focus(trigger_id));
    let _ = step(screen_raw(), &mut palette);
    assert_eq!(ctx.memory(|m| m.focused()), Some(trigger_id));

    palette.open();
    let _ = step(screen_raw(), &mut palette);
    let _ = step(screen_raw(), &mut palette);
    assert!(palette.is_open);
    assert!(ctx.memory(|m| m.focused()).is_some());
    assert_ne!(ctx.memory(|m| m.focused()), Some(trigger_id));

    for _ in 0..6 {
        let _ = step(key_raw(Key::Tab), &mut palette);
        let f = ctx.memory(|m| m.focused());
        assert_ne!(f, Some(trigger_id));
        assert_ne!(f, Some(other_bg_id));
    }

    let _ = step(key_raw(Key::Escape), &mut palette);
    assert!(!palette.is_open);
    assert_eq!(ctx.memory(|m| m.focused()), Some(trigger_id));
}

#[test]
fn test_command_palette_enter_selection_restores_focus() {
    let ctx = Context::default();
    let tabs: Vec<Box<dyn crate::tabs::ExplorerTab>> = vec![Box::new(DummyTab("Physics"))];
    let mut palette = CommandPalette::default();
    let mut trigger_id = Id::NULL;

    let _ = ctx.run(screen_raw(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            trigger_id = ui.button("Trigger").id;
        });
    });
    ctx.memory_mut(|m| m.request_focus(trigger_id));
    let _ = ctx.run(screen_raw(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            trigger_id = ui.button("Trigger").id;
        });
    });

    palette.open();
    let _ = ctx.run(screen_raw(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            trigger_id = ui.button("Trigger").id;
        });
        let _ = palette.show(ctx, &tabs);
    });

    let mut selected = None;
    let _ = ctx.run(key_raw(Key::Enter), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            trigger_id = ui.button("Trigger").id;
        });
        selected = palette.show(ctx, &tabs);
    });
    assert!(selected.is_some());
    assert!(!palette.is_open);
    assert_eq!(ctx.memory(|m| m.focused()), Some(trigger_id));
}

#[test]
fn test_starter_gallery_escape_tab_containment_and_focus_restore() {
    let ctx = Context::default();
    let cards = vec![StarterCard {
        id: "card_a",
        title: "Starter A",
        description: "Demo card",
        category: "Demo",
        icon: "🚀",
        domain: "ai",
        tool_name: None,
        preset_json: Some("{}"),
    }];
    let mut is_open = false;

    let step = |raw: RawInput, open: &mut bool| -> (Id, Id) {
        let mut trigger = Id::NULL;
        let mut bg = Id::NULL;
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                trigger = ui.button("Starter Gallery").id;
                bg = ui.button("Background Control").id;
            });
            let _ = show_starter_gallery_overlay(ctx, "pane0", &cards, open);
        });
        (trigger, bg)
    };

    let (trigger_id, bg_id) = step(screen_raw(), &mut is_open);
    ctx.memory_mut(|m| m.request_focus(trigger_id));
    let _ = step(screen_raw(), &mut is_open);

    is_open = true;
    let _ = step(screen_raw(), &mut is_open);
    let _ = step(screen_raw(), &mut is_open);
    assert!(is_open);
    assert!(ctx.memory(|m| m.focused()).is_some());
    assert_ne!(ctx.memory(|m| m.focused()), Some(trigger_id));

    for _ in 0..6 {
        let _ = step(key_raw(Key::Tab), &mut is_open);
        assert_ne!(ctx.memory(|m| m.focused()), Some(trigger_id));
        assert_ne!(ctx.memory(|m| m.focused()), Some(bg_id));
    }

    let _ = step(key_raw(Key::Escape), &mut is_open);
    assert!(!is_open);
    assert_eq!(ctx.memory(|m| m.focused()), Some(trigger_id));
}

#[test]
fn test_keyboard_shortcuts_modal_escape_tab_containment_and_focus_restore() {
    let ctx = Context::default();
    let mut show_help = false;

    let step = |raw: RawInput, open: &mut bool| -> (Id, Id) {
        let mut trigger = Id::NULL;
        let mut bg = Id::NULL;
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                trigger = ui.button("Help ?").id;
                bg = ui.button("Other Toolbar Button").id;
            });
            render_help_window(ctx, open);
        });
        (trigger, bg)
    };

    let (trigger_id, bg_id) = step(screen_raw(), &mut show_help);
    ctx.memory_mut(|m| m.request_focus(trigger_id));
    let _ = step(screen_raw(), &mut show_help);

    show_help = true;
    let _ = step(screen_raw(), &mut show_help);
    let _ = step(screen_raw(), &mut show_help);
    assert!(show_help);
    assert!(ctx.memory(|m| m.focused()).is_some());
    assert_ne!(ctx.memory(|m| m.focused()), Some(trigger_id));

    for _ in 0..5 {
        let _ = step(key_raw(Key::Tab), &mut show_help);
        assert_ne!(ctx.memory(|m| m.focused()), Some(trigger_id));
        assert_ne!(ctx.memory(|m| m.focused()), Some(bg_id));
    }

    let _ = step(key_raw(Key::Escape), &mut show_help);
    assert!(!show_help);
    assert_eq!(ctx.memory(|m| m.focused()), Some(trigger_id));
}

#[test]
fn test_export_high_res_modal_and_multi_pane_isolation() {
    let pane_a = Context::default();
    let pane_b = Context::default();
    let mut state_a = ExportModalState::default();
    let config_a = ExportModalConfig {
        window_title: "Export High-Res Plot - pane_0",
        id: Id::new("pane_0_export"),
        png_filename: "a.png",
        svg_filename: "a.svg",
        gif_filename: "a.gif",
    };

    let run_both = |raw_a: RawInput, raw_b: RawInput, state_a: &mut ExportModalState| -> (Id, Id) {
        let mut ta = Id::NULL;
        let mut bb = Id::NULL;
        let _ = pane_a.run(raw_a, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ta = ui.button("Export Pane A").id;
            });
            let mut src = CallbackExportSource {
                datasets_fn: || vec![("S".to_string(), vec![[0.0, 1.0], [1.0, 2.0]])],
                sample_frames_fn: |_| Vec::new(),
            };
            show_export_modal(ctx, state_a.bindings(), &config_a, &mut src);
        });
        let _ = pane_b.run(raw_b, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                bb = ui.button("Control Pane B").id;
            });
        });
        (ta, bb)
    };

    let (trigger_a, button_b) = run_both(screen_raw(), screen_raw(), &mut state_a);
    pane_a.memory_mut(|m| m.request_focus(trigger_a));
    pane_b.memory_mut(|m| m.request_focus(button_b));
    let _ = run_both(screen_raw(), screen_raw(), &mut state_a);

    state_a.open = true;
    let _ = run_both(screen_raw(), screen_raw(), &mut state_a);
    let _ = run_both(screen_raw(), screen_raw(), &mut state_a);
    assert!(pane_a.memory(|m| m.focused()).is_some());
    assert_ne!(pane_a.memory(|m| m.focused()), Some(trigger_a));
    assert_eq!(
        pane_b.memory(|m| m.focused()),
        Some(button_b),
        "Opening modal in Pane A must never steal focus in Pane B"
    );

    let _ = run_both(key_raw(Key::Escape), screen_raw(), &mut state_a);
    assert!(!state_a.open);
    assert_eq!(pane_a.memory(|m| m.focused()), Some(trigger_a));
    assert_eq!(pane_b.memory(|m| m.focused()), Some(button_b));
}

fn accesskit_action_raw(
    action: egui::accesskit::Action,
    target: egui::accesskit::NodeId,
) -> RawInput {
    let mut raw = screen_raw();
    raw.events.push(Event::AccessKitActionRequest(
        egui::accesskit::ActionRequest {
            action,
            target,
            data: None,
        },
    ));
    raw
}

#[test]
fn test_legend_checkbox_accesskit_action_click_and_keyboard() {
    let ctx = Context::default();
    ctx.enable_accesskit();
    let plot_id = Id::new("accesskit_legend_plot");

    let run = |raw: RawInput| {
        ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(plot_id)
                    .id(plot_id)
                    .legend(Legend::default())
                    .show_navigation_controls(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("Series Alpha", vec![[0.0, 1.0], [1.0, 3.0]]));
                    });
            });
        })
    };

    let out = run(screen_raw());
    let update = out
        .platform_output
        .accesskit_update
        .expect("AccessKit enabled");
    let checkbox_node_id = update
        .nodes
        .iter()
        .find_map(|(id, node)| {
            (node.role() == egui::accesskit::Role::CheckBox
                && node.label().is_some_and(|l| l.contains("Series Alpha")))
            .then_some(*id)
        })
        .expect("LegendEntry must expose Role::CheckBox in AccessKit");

    // 1. AccessKit Action::Click hides the series.
    let _ = run(accesskit_action_raw(
        egui::accesskit::Action::Click,
        checkbox_node_id,
    ));
    assert!(!PlotMemory::load(&ctx, plot_id)
        .unwrap()
        .hidden_items
        .is_empty());

    // 2. Focus via AccessKit + Space toggles series back on, Enter toggles it off again.
    let _ = run(accesskit_action_raw(
        egui::accesskit::Action::Focus,
        checkbox_node_id,
    ));
    let _ = run(key_raw(Key::Space));
    assert!(PlotMemory::load(&ctx, plot_id)
        .unwrap()
        .hidden_items
        .is_empty());

    let _ = run(key_raw(Key::Enter));
    assert!(!PlotMemory::load(&ctx, plot_id)
        .unwrap()
        .hidden_items
        .is_empty());
}

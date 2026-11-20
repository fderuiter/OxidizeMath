use egui::{
    AreaState, Color32, Context, Event, FullOutput, Id, Key, Modifiers, Pos2, RawInput, Rect,
    Shape, Stroke, Tooltip, Vec2b, vec2,
};

use super::{AccessibleTooltipExt as _, contrast_ratio, focus_ring_stroke};
use crate::{
    AccessState, CoordinatesFormatter, Corner, ExportDataSource, ExportFrame, ExportModalConfig,
    ExportModalState, Legend, Line, Plot, PlotMemory, show_export_modal,
};

fn base_input() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(900.0, 650.0))),
        ..Default::default()
    }
}

fn key_raw(key: Key, modifiers: Modifiers) -> RawInput {
    RawInput {
        events: vec![Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }],
        modifiers,
        ..base_input()
    }
}

fn hover_raw(pos: Pos2) -> RawInput {
    RawInput {
        events: vec![Event::PointerMoved(pos)],
        ..base_input()
    }
}

fn live_message(ctx: &Context) -> Option<String> {
    ctx.data(|d| d.get_temp::<String>(Id::new("aria_live_message")))
}

fn contains_rect_stroke(shape: &Shape, expected_color: Color32) -> bool {
    match shape {
        Shape::Rect(r) => r.stroke.color == expected_color && r.stroke.width >= 2.0,
        Shape::Vec(v) => v.iter().any(|s| contains_rect_stroke(s, expected_color)),
        _ => false,
    }
}

fn contains_circle_marker(shape: &Shape) -> bool {
    match shape {
        Shape::Circle(_) => true,
        Shape::Vec(v) => v.iter().any(contains_circle_marker),
        _ => false,
    }
}

fn contains_galley_substring(shape: &Shape, needle: &str) -> bool {
    match shape {
        Shape::Text(t) => t.galley.text().contains(needle),
        Shape::Vec(v) => v.iter().any(|s| contains_galley_substring(s, needle)),
        _ => false,
    }
}

fn run_multi_series_plot(ctx: &Context, plot_id: Id, raw: RawInput) -> (FullOutput, Option<Id>) {
    let mut hovered_item = None;
    let out = ctx.run(raw, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let res = Plot::new(plot_id)
                .id(plot_id)
                .legend(Legend::default())
                .show_navigation_controls(false)
                .coordinates_formatter(Corner::LeftBottom, CoordinatesFormatter::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Alpha", vec![[0.0, 1.0], [2.0, 4.0], [4.0, 9.0]]));
                    plot_ui.line(Line::new("Beta", vec![[0.0, 2.0], [2.0, 5.0]]));
                    plot_ui.line(Line::new("Gamma", vec![[0.0, 3.0], [2.0, 6.0]]));
                });
            hovered_item = res.hovered_plot_item;
        });
    });
    (out, hovered_item)
}

#[test]
fn test_legend_space_enter_focus_ring_and_live_announcements() {
    let ctx = Context::default();
    let plot_id = Id::new("a11y_legend_toggle");
    let _ = run_multi_series_plot(&ctx, plot_id, base_input());

    // Tab stops: 1=Plot, 2=X-axis, 3=Y-axis, 4=Wrench, 5=First LegendEntry ("Alpha").
    for _ in 0..5 {
        let _ = run_multi_series_plot(&ctx, plot_id, key_raw(Key::Tab, Modifiers::NONE));
    }
    let (focused_out, hovered_item) = run_multi_series_plot(&ctx, plot_id, base_input());
    assert!(
        hovered_item.is_some(),
        "Focused LegendEntry must highlight its corresponding plot series"
    );
    let expected_ring =
        focus_ring_stroke(&ctx.style().visuals, ctx.style().visuals.extreme_bg_color);
    assert!(
        focused_out
            .shapes
            .iter()
            .any(|cs| contains_rect_stroke(&cs.shape, expected_ring.color)),
        "Focused LegendEntry must paint a visible focus ring"
    );

    let _ = run_multi_series_plot(&ctx, plot_id, key_raw(Key::Space, Modifiers::NONE));
    assert_eq!(live_message(&ctx).as_deref(), Some("Series Alpha hidden"));
    assert_eq!(
        PlotMemory::load(&ctx, plot_id).unwrap().hidden_items.len(),
        1
    );

    let _ = run_multi_series_plot(&ctx, plot_id, key_raw(Key::Enter, Modifiers::NONE));
    assert_eq!(live_message(&ctx).as_deref(), Some("Series Alpha shown"));
    assert!(
        PlotMemory::load(&ctx, plot_id)
            .unwrap()
            .hidden_items
            .is_empty()
    );
}

#[test]
fn test_legend_alt_enter_isolates_and_restores_all_series() {
    let ctx = Context::default();
    let plot_id = Id::new("a11y_legend_isolate");
    let _ = run_multi_series_plot(&ctx, plot_id, base_input());

    // Focus 2nd LegendEntry ("Beta") on 6th Tab.
    for _ in 0..6 {
        let _ = run_multi_series_plot(&ctx, plot_id, key_raw(Key::Tab, Modifiers::NONE));
    }
    let beta_focus = ctx.memory(|m| m.focused()).expect("Beta focused");

    let _ = run_multi_series_plot(&ctx, plot_id, key_raw(Key::Enter, Modifiers::ALT));
    let mem_isolated = PlotMemory::load(&ctx, plot_id).unwrap();
    assert_eq!(
        mem_isolated.hidden_items.len(),
        2,
        "Alt+Enter must hide Alpha and Gamma while keeping Beta visible"
    );
    assert_eq!(live_message(&ctx).as_deref(), Some("Isolated series Beta"));
    assert_eq!(ctx.memory(|m| m.focused()), Some(beta_focus));

    let _ = run_multi_series_plot(&ctx, plot_id, key_raw(Key::Enter, Modifiers::ALT));
    let mem_restored = PlotMemory::load(&ctx, plot_id).unwrap();
    assert!(
        mem_restored.hidden_items.is_empty(),
        "Second Alt+Enter must restore all legend series"
    );
    assert_eq!(live_message(&ctx).as_deref(), Some("All series shown"));
}

#[test]
fn test_plot_widget_info_series_count_and_focus_border_contrast() {
    let ctx = Context::default();
    ctx.enable_accesskit();
    let plot_id = Id::new("a11y_canvas_info");

    let _ = run_multi_series_plot(&ctx, plot_id, base_input());
    let (out, _) = run_multi_series_plot(&ctx, plot_id, key_raw(Key::Tab, Modifiers::NONE));

    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit update");
    let doc_label = ak
        .nodes
        .iter()
        .find_map(|(_, n)| {
            (n.role() == egui::accesskit::Role::GraphicsDocument).then(|| n.label())?
        })
        .expect("GraphicsDocument node must expose a descriptive label");
    assert!(doc_label.contains("Interactive Plot"));
    assert!(doc_label.contains("3 series"));

    let mut low_contrast = egui::Visuals::dark();
    low_contrast.selection.stroke = Stroke::new(1.0_f32, low_contrast.extreme_bg_color);
    for visuals in [egui::Visuals::dark(), egui::Visuals::light(), low_contrast] {
        ctx.set_visuals(visuals.clone());
        let ring = focus_ring_stroke(&visuals, visuals.extreme_bg_color);
        assert!(
            contrast_ratio(ring.color, visuals.extreme_bg_color) >= 3.0,
            "Focus border must meet WCAG 3:1 non-text contrast"
        );
        let (themed_out, _) = run_multi_series_plot(&ctx, plot_id, base_input());
        assert!(
            themed_out
                .shapes
                .iter()
                .any(|cs| contains_rect_stroke(&cs.shape, ring.color)),
            "Focused Plot must paint its >=3:1 focus border"
        );
    }
}

fn step_focus_lock_frame(ctx: &Context, plot_id: Id, raw: RawInput) -> (FullOutput, Id, Id) {
    let mut before_id = Id::NULL;
    let mut plot_resp_id = Id::NULL;
    let out = ctx.run(raw, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            before_id = ui.button("Before Plot").id;
            let res = Plot::new(plot_id)
                .id(plot_id)
                .coordinates_formatter(Corner::LeftBottom, CoordinatesFormatter::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Wave", vec![[1.0, 2.0], [3.0, 6.0]]));
                    plot_ui.line(Line::new("Ref", vec![[1.0, 5.0], [3.0, 8.0]]));
                });
            plot_resp_id = res.response.id;
        });
    });
    (out, before_id, plot_resp_id)
}

#[test]
fn test_plot_focus_locks_arrow_keys_without_trapping_tab_and_draws_crosshairs() {
    let ctx = Context::default();
    let plot_id = Id::new("a11y_focus_lock_and_crosshair");

    let (_, before_id, plot_resp_id) = step_focus_lock_frame(&ctx, plot_id, base_input());
    ctx.memory_mut(|m| m.request_focus(plot_resp_id));
    let (focused_out, _, _) = step_focus_lock_frame(&ctx, plot_id, base_input());
    assert!(
        focused_out
            .shapes
            .iter()
            .any(|cs| contains_circle_marker(&cs.shape))
    );
    assert!(
        focused_out
            .shapes
            .iter()
            .any(|cs| contains_galley_substring(&cs.shape, "x:")),
        "Focused Plot must render corner coordinates for the active point"
    );

    for arrow in [
        Key::ArrowRight,
        Key::ArrowDown,
        Key::ArrowLeft,
        Key::ArrowUp,
    ] {
        let _ = step_focus_lock_frame(&ctx, plot_id, key_raw(arrow, Modifiers::NONE));
        assert_eq!(
            ctx.memory(|m| m.focused()),
            Some(plot_resp_id),
            "{arrow:?} must remain locked on focused Plot without moving focus"
        );
    }

    let _ = step_focus_lock_frame(&ctx, plot_id, key_raw(Key::Tab, Modifiers::NONE));
    assert_ne!(ctx.memory(|m| m.focused()), Some(plot_resp_id));

    ctx.memory_mut(|m| m.request_focus(plot_resp_id));
    let _ = step_focus_lock_frame(&ctx, plot_id, base_input());
    let _ = step_focus_lock_frame(&ctx, plot_id, key_raw(Key::Tab, Modifiers::SHIFT));
    let _ = step_focus_lock_frame(&ctx, plot_id, base_input());
    assert_eq!(ctx.memory(|m| m.focused()), Some(before_id));
}

#[test]
fn test_focused_plot_zoom_pan_and_home_reset_keys() {
    let ctx = Context::default();
    let plot_id = Id::new("a11y_viewport_keys");

    let run = |raw: RawInput| -> Id {
        let mut plot_resp_id = Id::NULL;
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                plot_resp_id = Plot::new(plot_id)
                    .id(plot_id)
                    .allow_drag(Vec2b::new(true, false))
                    .allow_scroll(Vec2b::new(false, false))
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("Curve", vec![[-10.0, -5.0], [10.0, 5.0]]));
                    })
                    .response
                    .id;
            });
        });
        plot_resp_id
    };

    let plot_resp_id = run(base_input());
    ctx.memory_mut(|m| m.request_focus(plot_resp_id));
    let _ = run(base_input());
    let initial = *PlotMemory::load(&ctx, plot_id).unwrap().transform.bounds();

    let _ = run(key_raw(Key::Plus, Modifiers::NONE));
    let after_plus = *PlotMemory::load(&ctx, plot_id).unwrap().transform.bounds();
    assert!(after_plus.width() < initial.width() * 0.85);

    let _ = run(key_raw(Key::Equals, Modifiers::NONE));
    let after_eq = *PlotMemory::load(&ctx, plot_id).unwrap().transform.bounds();
    assert!(after_eq.width() < after_plus.width() * 0.85);

    let _ = run(key_raw(Key::Minus, Modifiers::NONE));
    let after_minus = *PlotMemory::load(&ctx, plot_id).unwrap().transform.bounds();
    assert!(after_minus.width() > after_eq.width() * 1.15);

    let _ = run(key_raw(Key::ArrowRight, Modifiers::SHIFT));
    let after_pan_x = *PlotMemory::load(&ctx, plot_id).unwrap().transform.bounds();
    assert!(after_pan_x.min[0] > after_minus.min[0]);
    let state = ctx.data(|d| d.get_temp::<AccessState>(plot_id)).unwrap();
    assert_eq!(
        state.focused_point, 0,
        "Shift+ArrowRight must pan, not step point"
    );

    let _ = run(key_raw(Key::ArrowUp, Modifiers::SHIFT));
    let after_locked_y = *PlotMemory::load(&ctx, plot_id).unwrap().transform.bounds();
    assert!((after_locked_y.min[1] - after_pan_x.min[1]).abs() < 1e-6);

    let _ = run(key_raw(Key::Home, Modifiers::NONE));
    let after_home = *PlotMemory::load(&ctx, plot_id).unwrap().transform.bounds();
    assert!((after_home.width() - initial.width()).abs() < 1e-4);
    assert_eq!(live_message(&ctx).as_deref(), Some("Plot bounds reset"));
}

#[test]
fn test_toolbar_and_wrench_buttons_expose_descriptive_names() {
    let ctx = Context::default();
    ctx.enable_accesskit();
    let plot_id = Id::new("a11y_toolbar_names");

    let out = ctx.run(base_input(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new(plot_id).id(plot_id).show(ui, |plot_ui| {
                plot_ui.line(Line::new("SinglePt", vec![[0.0, 0.0]]));
            });
        });
    });

    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let labels: Vec<&str> = ak
        .nodes
        .iter()
        .filter(|(_, n)| n.role() == egui::accesskit::Role::Button)
        .filter_map(|(_, n)| n.label())
        .collect();
    for expected in [
        "Zoom in",
        "Zoom out",
        "Reset plot bounds",
        "Plot grid and axis settings",
        "Fit Curve",
    ] {
        assert!(
            labels.contains(&expected),
            "Missing button label '{expected}' in {labels:?}"
        );
    }
}

#[test]
fn test_accessible_tooltip_focus_hover_persistence_and_escape_dismissal() {
    let ctx = Context::default();

    let step = |raw: RawInput| -> Id {
        let mut btn_id = Id::NULL;
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                btn_id = super::focusable_action_button(
                    ui,
                    false,
                    egui::Button::new("Disabled Export"),
                    Some("Disabled Export"),
                )
                .accessible_tooltip("Requires at least 2 points")
                .id;
            });
        });
        btn_id
    };

    let btn_id = step(base_input());
    ctx.memory_mut(|m| m.request_focus(btn_id));
    let _ = step(base_input());
    assert!(Tooltip::was_tooltip_open_last_frame(&ctx, btn_id));
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("Requires at least 2 points")
    );

    let tooltip_rect = AreaState::load(&ctx, Tooltip::tooltip_id(btn_id, 0))
        .expect("Tooltip AreaState")
        .rect();
    ctx.memory_mut(|m| m.surrender_focus(btn_id));
    let _ = step(hover_raw(tooltip_rect.center()));
    assert!(
        Tooltip::was_tooltip_open_last_frame(&ctx, btn_id),
        "Moving pointer onto open tooltip must keep it visible (SCR39)"
    );

    ctx.memory_mut(|m| m.request_focus(btn_id));
    let _ = step(base_input());
    let _ = step(key_raw(Key::Escape, Modifiers::NONE));
    assert!(
        !Tooltip::was_tooltip_open_last_frame(&ctx, btn_id),
        "Escape must immediately dismiss open tooltip"
    );
    assert_eq!(
        ctx.memory(|m| m.focused()),
        Some(btn_id),
        "Escape tooltip dismissal must not surrender button focus"
    );
}

struct DisabledExportSource;

impl ExportDataSource for DisabledExportSource {
    fn export_datasets(&mut self) -> Vec<(String, Vec<[f64; 2]>)> {
        Vec::new()
    }

    fn sample_gif_frames(&mut self, _: usize) -> Vec<ExportFrame> {
        Vec::new()
    }

    fn figure_capability(&self) -> (bool, &'static str) {
        (false, "Figure export unavailable for symbolic view")
    }
}

fn step_disabled_export_modal(
    ctx: &Context,
    state: &mut ExportModalState,
    cfg: &ExportModalConfig<'_>,
    raw: RawInput,
) {
    let _ = ctx.run(raw, |ctx| {
        let mut src = DisabledExportSource;
        show_export_modal(ctx, state.bindings(), cfg, &mut src);
    });
}

#[test]
fn test_disabled_export_modal_tooltip_dismisses_on_escape_without_closing_modal() {
    let ctx = Context::default();
    let mut state = ExportModalState {
        open: true,
        ..Default::default()
    };
    let cfg = ExportModalConfig {
        window_title: "Export Test",
        id: Id::new("disabled_export_modal_test"),
        png_filename: "t.png",
        svg_filename: "t.svg",
        gif_filename: "t.gif",
    };

    step_disabled_export_modal(&ctx, &mut state, &cfg, base_input());
    step_disabled_export_modal(&ctx, &mut state, &cfg, base_input());
    for _ in 0..16 {
        step_disabled_export_modal(&ctx, &mut state, &cfg, key_raw(Key::Tab, Modifiers::NONE));
        if live_message(&ctx).as_deref() == Some("Figure export unavailable for symbolic view") {
            break;
        }
    }
    let focused_btn = ctx
        .memory(|m| m.focused())
        .expect("Disabled Save PNG focused");
    assert!(Tooltip::was_tooltip_open_last_frame(&ctx, focused_btn));

    step_disabled_export_modal(
        &ctx,
        &mut state,
        &cfg,
        key_raw(Key::Escape, Modifiers::NONE),
    );
    assert!(!Tooltip::was_tooltip_open_last_frame(&ctx, focused_btn));
    assert!(
        state.open,
        "First Escape must dismiss tooltip while keeping modal open"
    );
    assert_eq!(ctx.memory(|m| m.focused()), Some(focused_btn));

    step_disabled_export_modal(
        &ctx,
        &mut state,
        &cfg,
        key_raw(Key::Escape, Modifiers::NONE),
    );
    assert!(
        !state.open,
        "Second Escape with tooltip already dismissed must close modal"
    );
}

#[test]
fn test_snapped_point_tooltip_anchored_hover_persistent_and_escape_dismissible() {
    let ctx = Context::default();
    let plot_id = Id::new("a11y_point_tooltip");

    let run = |raw: RawInput| -> (Id, Pos2) {
        let mut plot_resp_id = Id::NULL;
        let mut target_screen_pos = Pos2::ZERO;
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let res = Plot::new(plot_id).id(plot_id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Trajectory", vec![[0.0, 0.0], [5.0, 5.0]]));
                });
                plot_resp_id = res.response.id;
                target_screen_pos = res
                    .transform
                    .position_from_point(&crate::PlotPoint::new(0.0, 0.0));
            });
        });
        (plot_resp_id, target_screen_pos)
    };

    let (plot_resp_id, _) = run(base_input());
    ctx.memory_mut(|m| m.request_focus(plot_resp_id));
    let _ = run(base_input());
    let (_, target_screen_pos) = run(base_input());
    assert!(Tooltip::was_tooltip_open_last_frame(&ctx, plot_resp_id));

    let popup_rect = AreaState::load(&ctx, Tooltip::tooltip_id(plot_resp_id, 0))
        .expect("Point tooltip AreaState")
        .rect();
    assert!(
        popup_rect.expand(40.0).contains(target_screen_pos),
        "Point tooltip {popup_rect:?} must anchor near snapped point {target_screen_pos:?}"
    );

    let _ = run(key_raw(Key::Escape, Modifiers::NONE));
    assert!(!Tooltip::was_tooltip_open_last_frame(&ctx, plot_resp_id));
    assert_eq!(ctx.memory(|m| m.focused()), Some(plot_resp_id));

    let _ = run(key_raw(Key::ArrowRight, Modifiers::NONE));
    assert!(
        Tooltip::was_tooltip_open_last_frame(&ctx, plot_resp_id),
        "Stepping to next point must reopen snapped-point tooltip"
    );

    let next_popup = AreaState::load(&ctx, Tooltip::tooltip_id(plot_resp_id, 0))
        .unwrap()
        .rect();
    ctx.memory_mut(|m| m.surrender_focus(plot_resp_id));
    let _ = run(hover_raw(next_popup.center()));
    assert!(
        Tooltip::was_tooltip_open_last_frame(&ctx, plot_resp_id),
        "Hovering over open snapped-point tooltip must keep it open"
    );
}

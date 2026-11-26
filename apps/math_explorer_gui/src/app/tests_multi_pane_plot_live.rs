#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::viewport::focus::focus_pane_interactive;
use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements, with_live_region};
use eframe::egui;
use eframe::egui::accesskit::{Action, ActionRequest, NodeId, Role};
use eframe::App;

fn run_app_frame(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let mut frame = eframe::Frame::_new_kittest();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1600.0, 1800.0));
    ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            events,
            ..Default::default()
        },
        |c| app.update(c, &mut frame),
    )
}

fn key_event(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }
}

fn focus_node_event(target: NodeId) -> egui::Event {
    egui::Event::AccessKitActionRequest(ActionRequest {
        action: Action::Focus,
        target,
        data: None,
    })
}

fn setup_sir_multi_pane(
    ctx: &egui::Context,
    layout: ViewportLayout,
    target_pane: usize,
) -> Option<(MathExplorerApp, egui::FullOutput)> {
    let mut app = MathExplorerApp::default();
    let epi_idx = app.tabs.iter().position(|t| t.name() == "Epidemiology")?;
    app.set_layout(layout);
    for pane in 0..layout.pane_count() {
        app.set_pane_tab(pane, epi_idx);
        let fw = app.pane_tab_mut(pane)?.framework_mut()?;
        assert!(fw.select_tool_by_name("SirTool"));
    }
    focus_pane_interactive(&mut app, ctx, target_pane);
    let _ = run_app_frame(ctx, &mut app, Vec::new());
    let out = run_app_frame(ctx, &mut app, Vec::new());
    Some((app, out))
}

fn pane_region_bounds(out: &egui::FullOutput, pane_label: &str) -> egui::Rect {
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit tree");
    let (_, node) = ak
        .nodes
        .iter()
        .find(|(_, n)| n.role() == Role::Region && n.label() == Some(pane_label))
        .unwrap_or_else(|| panic!("Missing region {pane_label}"));
    let b = node.bounds().expect("Region bounds");
    egui::Rect::from_min_max(
        egui::pos2(b.x0 as f32, b.y0 as f32),
        egui::pos2(b.x1 as f32, b.y1 as f32),
    )
}

fn find_node_in_rect(
    out: &egui::FullOutput,
    pane_rect: egui::Rect,
    role: Role,
    label_substr: Option<&str>,
) -> NodeId {
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    ak.nodes
        .iter()
        .find_map(|(id, n)| {
            if n.role() != role {
                return None;
            }
            if let Some(sub) = label_substr {
                if !n.label().is_some_and(|l| l.contains(sub)) {
                    return None;
                }
            }
            let b = n.bounds()?;
            let center = egui::pos2(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32);
            pane_rect.contains(center).then_some(*id)
        })
        .unwrap_or_else(|| panic!("Missing {role:?} ({label_substr:?}) inside {pane_rect:?}"))
}

fn assert_root_ctx_and_twin_slot(
    ctx: &egui::Context,
    key: &'static str,
    priority: &'static str,
    expected_substr: &str,
    layout: ViewportLayout,
) {
    let ctx_msg = ctx
        .data(|d| d.get_temp::<String>(egui::Id::new(key)))
        .unwrap_or_default();
    assert!(
        ctx_msg.contains(expected_substr),
        "{layout:?}: root ctx {key} '{ctx_msg}' missing '{expected_substr}'"
    );
    with_live_region(|region| {
        let active = &region.slots[region.active_slot];
        assert!(
            active.contains(expected_substr),
            "{layout:?}: TwinSlotLiveRegion active slot '{active}' missing '{expected_substr}'"
        );
    });
    assert!(
        all_announcements()
            .iter()
            .any(|(m, p)| m.contains(expected_substr) && p == priority),
        "{layout:?}: {priority} announcement missing '{expected_substr}'"
    );
}

fn verify_plot_canvas_keys(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    out: &egui::FullOutput,
    pane_rect: egui::Rect,
    layout: ViewportLayout,
) {
    let plot_node = find_node_in_rect(out, pane_rect, Role::GraphicsDocument, Some("series"));
    let _ = run_app_frame(ctx, app, vec![focus_node_event(plot_node)]);

    for (key, expected) in [
        (egui::Key::ArrowRight, "Point X:"),
        (egui::Key::Plus, "Plot zoomed in"),
        (egui::Key::Home, "Plot bounds reset"),
    ] {
        clear_test_announcements();
        let _ = run_app_frame(ctx, app, vec![key_event(key, egui::Modifiers::NONE)]);
        assert_root_ctx_and_twin_slot(ctx, "aria_live_message", "polite", expected, layout);
    }
}

fn verify_legend_and_toolbar_and_assertive(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    out: &egui::FullOutput,
    pane_rect: egui::Rect,
    layout: ViewportLayout,
) {
    let legend_node = find_node_in_rect(out, pane_rect, Role::CheckBox, Some("Susceptible"));
    let _ = run_app_frame(ctx, app, vec![focus_node_event(legend_node)]);

    clear_test_announcements();
    let _ = run_app_frame(
        ctx,
        app,
        vec![key_event(egui::Key::Space, egui::Modifiers::NONE)],
    );
    assert_root_ctx_and_twin_slot(ctx, "aria_live_message", "polite", "Series", layout);

    clear_test_announcements();
    let _ = run_app_frame(
        ctx,
        app,
        vec![key_event(egui::Key::Enter, egui::Modifiers::ALT)],
    );
    assert_root_ctx_and_twin_slot(
        ctx,
        "aria_live_message",
        "polite",
        "Isolated series",
        layout,
    );

    let zoom_btn = find_node_in_rect(out, pane_rect, Role::Button, Some("Zoom in"));
    clear_test_announcements();
    let _ = run_app_frame(ctx, app, vec![focus_node_event(zoom_btn)]);
    assert_root_ctx_and_twin_slot(ctx, "aria_live_message", "polite", "Zoom in", layout);
    let focused_zoom = ctx.memory(|m| m.focused());
    assert!(focused_zoom.is_some());
    let _ = run_app_frame(
        ctx,
        app,
        vec![key_event(egui::Key::Escape, egui::Modifiers::NONE)],
    );
    assert_eq!(ctx.memory(|m| m.focused()), focused_zoom);

    clear_test_announcements();
    egui_plot::emit_plot_live_message(ctx, format!("Invalid CSV in {layout:?} pane"), true);
    assert_root_ctx_and_twin_slot(
        ctx,
        "aria_live_assertive_message",
        "assertive",
        "Invalid CSV",
        layout,
    );
}

fn verify_multi_pane_plot_announcements(layout: ViewportLayout, target_pane: usize) {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let Some((mut app, out)) = setup_sir_multi_pane(&ctx, layout, target_pane) else {
        return;
    };
    let pane_label = layout.pane_labels()[target_pane];
    let pane_rect = pane_region_bounds(&out, pane_label);
    verify_plot_canvas_keys(&ctx, &mut app, &out, pane_rect, layout);
    verify_legend_and_toolbar_and_assertive(&ctx, &mut app, &out, pane_rect, layout);
}

#[test]
fn test_split_horizontal_vertical_and_quad_grid_plot_live_region_parity() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    verify_multi_pane_plot_announcements(ViewportLayout::SplitHorizontal, 1);
    verify_multi_pane_plot_announcements(ViewportLayout::SplitVertical, 1);
    verify_multi_pane_plot_announcements(ViewportLayout::QuadGrid, 3);
}

#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::app::{MathExplorerApp, ViewportLayout};
use eframe::egui::{
    self, accesskit, vec2, Context, Event, FullOutput, Id, Key, Modifiers, PointerButton, Pos2,
    RawInput, Rect,
};
use eframe::App;

pub(super) fn screen_rect() -> Rect {
    Rect::from_min_size(Pos2::ZERO, vec2(2000.0, 3000.0))
}

pub(super) fn target_pane_rect(ctx: &Context, pane_idx: usize) -> Rect {
    ctx.data(|d| d.get_temp::<Vec<Rect>>(Id::new("LAST_PANE_RECTS")))
        .and_then(|rects| rects.get(pane_idx).copied())
        .unwrap_or_else(screen_rect)
}

pub(super) fn run_app(ctx: &Context, app: &mut MathExplorerApp, events: Vec<Event>) -> FullOutput {
    let mut frame = eframe::Frame::_new_kittest();
    let raw = RawInput {
        screen_rect: Some(screen_rect()),
        events,
        ..Default::default()
    };
    ctx.run(raw, |c| app.update(c, &mut frame))
}

pub(super) fn key_evt(key: Key, modifiers: Modifiers) -> Event {
    Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }
}

pub(super) fn cmd_mods() -> Modifiers {
    if cfg!(target_os = "macos") {
        Modifiers::MAC_CMD | Modifiers::COMMAND
    } else {
        Modifiers::CTRL | Modifiers::COMMAND
    }
}

pub(super) fn ak_action_evt(action: accesskit::Action, target: accesskit::NodeId) -> Event {
    Event::AccessKitActionRequest(accesskit::ActionRequest {
        action,
        target,
        data: None,
    })
}

pub(super) fn find_ak_nodes(
    out: &FullOutput,
    role: accesskit::Role,
    label_substr: &str,
) -> Vec<(accesskit::NodeId, Rect)> {
    let Some(ak) = out.platform_output.accesskit_update.as_ref() else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for (id, node) in &ak.nodes {
        if node.role() != role {
            continue;
        }
        let matches_label = match node.label() {
            Some(lbl) => lbl.contains(label_substr),
            None => label_substr.is_empty(),
        };
        if matches_label {
            let rect = match node.bounds() {
                Some(b) => Rect::from_min_max(
                    egui::pos2(b.x0 as f32, b.y0 as f32),
                    egui::pos2(b.x1 as f32, b.y1 as f32),
                ),
                None => Rect::NOTHING,
            };
            found.push((*id, rect));
        }
    }
    found.sort_by(|a, b| {
        a.1.min
            .y
            .total_cmp(&b.1.min.y)
            .then_with(|| a.1.min.x.total_cmp(&b.1.min.x))
    });
    found
}

pub(super) fn find_ak_node_in_pane(
    _ctx: &Context,
    out: &FullOutput,
    pane_rect: Rect,
    role: accesskit::Role,
    label_substr: &str,
) -> accesskit::NodeId {
    find_ak_nodes(out, role, label_substr)
        .into_iter()
        .find_map(|(nid, ak_rect)| pane_rect.contains(ak_rect.center()).then_some(nid))
        .unwrap_or_else(|| {
            panic!("Expected AccessKit {role:?} '{label_substr}' inside pane {pane_rect:?}")
        })
}

pub(super) fn focus_ak_node(
    ctx: &Context,
    app: &mut MathExplorerApp,
    target: accesskit::NodeId,
) -> Id {
    let _ = run_app(
        ctx,
        app,
        vec![ak_action_evt(accesskit::Action::Focus, target)],
    );
    ctx.memory(|m| m.focused())
        .expect("AccessKit Action::Focus must focus target widget")
}

pub(super) fn click_pos(ctx: &Context, app: &mut MathExplorerApp, pos: Pos2) -> FullOutput {
    let _ = run_app(
        ctx,
        app,
        vec![
            Event::PointerMoved(pos),
            Event::PointerButton {
                pos,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: Modifiers::NONE,
            },
        ],
    );
    run_app(
        ctx,
        app,
        vec![
            Event::PointerMoved(pos),
            Event::PointerButton {
                pos,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: Modifiers::NONE,
            },
        ],
    )
}

pub(super) fn find_text_centers(out: &FullOutput, needle: &str) -> Vec<Pos2> {
    let mut centers = Vec::new();
    let mut stack: Vec<&egui::epaint::Shape> = out.shapes.iter().map(|c| &c.shape).collect();
    while let Some(shape) = stack.pop() {
        match shape {
            egui::epaint::Shape::Text(ts) => {
                if ts.galley.job.text.contains(needle) {
                    centers.push(ts.pos + ts.galley.rect.center().to_vec2());
                }
            }
            egui::epaint::Shape::Vec(nested) => {
                for inner in nested.iter().rev() {
                    stack.push(inner);
                }
            }
            _ => {}
        }
    }
    centers.sort_by(|a, b| a.y.total_cmp(&b.y).then_with(|| a.x.total_cmp(&b.x)));
    centers
}

pub(super) fn has_rendered_text(out: &FullOutput, needle: &str) -> bool {
    !find_text_centers(out, needle).is_empty()
}

pub(super) fn find_domain_tab(app: &MathExplorerApp, domain: &str) -> usize {
    app.tabs
        .iter()
        .position(|t| t.framework().is_some_and(|fw| fw.domain == domain))
        .unwrap_or_else(|| panic!("Expected SimulationFramework tab for domain '{domain}'"))
}

fn configure_ode_custom(app: &mut MathExplorerApp, pane: usize, analysis_idx: usize) {
    app.set_pane_tab(pane, analysis_idx);
    let fw = app.pane_tab_mut(pane).unwrap().framework_mut().unwrap();
    assert!(fw.select_tool_by_name("OdeSolverTool"));
    let json = fw.active_tool.as_ref().unwrap().save_state().unwrap();
    let custom = json.replace("\"Exponential\"", "\"Custom\"");
    fw.active_tool.as_mut().unwrap().load_state(&custom);
}

fn configure_training_monitor(app: &mut MathExplorerApp, pane: usize, ai_idx: usize) {
    app.set_pane_tab(pane, ai_idx);
    let fw = app.pane_tab_mut(pane).unwrap().framework_mut().unwrap();
    assert!(fw.select_tool_by_name("TrainingMonitorTool"));
}

fn configure_sir_tool(app: &mut MathExplorerApp, pane: usize, epi_idx: usize) {
    app.set_pane_tab(pane, epi_idx);
    let fw = app.pane_tab_mut(pane).unwrap().framework_mut().unwrap();
    assert!(fw.select_tool_by_name("SirTool"));
}

pub(super) fn configure_layout_with_canary_panes(
    app: &mut MathExplorerApp,
    layout: ViewportLayout,
) -> usize {
    let epi_idx = find_domain_tab(app, "epidemiology");
    let analysis_idx = find_domain_tab(app, "analysis");
    let ai_idx = find_domain_tab(app, "ai");

    app.set_layout(layout);
    match layout {
        ViewportLayout::Single => {
            configure_sir_tool(app, 0, epi_idx);
            app.focus_pane(0);
            0
        }
        ViewportLayout::SplitHorizontal => {
            configure_ode_custom(app, 0, analysis_idx);
            configure_sir_tool(app, 1, epi_idx);
            app.focus_pane(1);
            1
        }
        ViewportLayout::SplitVertical => {
            configure_training_monitor(app, 0, ai_idx);
            configure_sir_tool(app, 1, epi_idx);
            app.focus_pane(1);
            1
        }
        ViewportLayout::QuadGrid => {
            configure_ode_custom(app, 0, analysis_idx);
            configure_training_monitor(app, 1, ai_idx);
            configure_ode_custom(app, 2, analysis_idx);
            configure_sir_tool(app, 3, epi_idx);
            app.focus_pane(3);
            3
        }
    }
}

pub(super) fn snapshot_all_pane_states(
    app: &MathExplorerApp,
    exclude_plot_states_pane: Option<usize>,
) -> Vec<(usize, String)> {
    let count = app.viewport_layout.pane_count();
    (0..count)
        .map(|idx| {
            let mut state = app
                .pane_tab(idx)
                .and_then(|t| t.save_state())
                .unwrap_or_default();
            if exclude_plot_states_pane == Some(idx) {
                if let Ok(mut val) = serde_json::from_str::<serde_json::Value>(&state) {
                    if let Some(obj) = val.as_object_mut() {
                        obj.remove("plot_states");
                        state = serde_json::to_string(&val).unwrap_or(state);
                    }
                }
            }
            (idx, state)
        })
        .collect()
}

pub(super) fn copied_texts(out: &FullOutput) -> Vec<String> {
    out.platform_output
        .commands
        .iter()
        .filter_map(|cmd| match cmd {
            egui::OutputCommand::CopyText(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

pub(super) fn secondary_click_pos(
    ctx: &Context,
    app: &mut MathExplorerApp,
    pos: Pos2,
) -> FullOutput {
    let _ = run_app(
        ctx,
        app,
        vec![
            Event::PointerMoved(pos),
            Event::PointerButton {
                pos,
                button: PointerButton::Secondary,
                pressed: true,
                modifiers: Modifiers::NONE,
            },
        ],
    );
    run_app(
        ctx,
        app,
        vec![
            Event::PointerMoved(pos),
            Event::PointerButton {
                pos,
                button: PointerButton::Secondary,
                pressed: false,
                modifiers: Modifiers::NONE,
            },
        ],
    )
}

pub(super) fn has_close_viewport_cmd(out: &FullOutput) -> bool {
    out.viewport_output
        .get(&egui::ViewportId::ROOT)
        .is_some_and(|vp| vp.commands.contains(&egui::ViewportCommand::Close))
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Multi-pane and cross-tab [`crate::framework::CrossToolBus`] routing for [`crate::app::MathExplorerApp`].

use crate::app::{domain_bar::select_primary_tab, MathExplorerApp};
use crate::framework::CrossToolEnvelope;
use eframe::egui;

/// Polls instance-scoped [`crate::framework::CrossToolBus`] queues across all tabs and split panes.
pub fn poll_cross_tab_tool_handoffs(app: &mut MathExplorerApp, ctx: &egui::Context) {
    let pane_count = app.viewport_layout.pane_count();
    let primary_tab_idx = app.active_primary_tab_index();
    let mut actions: Vec<(Option<usize>, CrossToolEnvelope)> = Vec::new();
    let mut live_msgs: Vec<(Option<usize>, CrossToolEnvelope)> = Vec::new();

    for (tab_idx, tab) in app.tabs.iter_mut().enumerate() {
        if let Some(fw) = tab.framework_mut() {
            fw.collect_tool_bus_outbox(Some(ctx));
            let src_pane = (tab_idx == primary_tab_idx).then_some(0);
            for env in fw.cross_tool_bus.drain_outgoing_actions() {
                actions.push((src_pane, env));
            }
            if let Some(live) = fw.cross_tool_bus.take_outgoing_live() {
                live_msgs.push((src_pane, live));
            }
        }
    }

    for pane_idx in 1..pane_count {
        if let Some(slot) = app.pane_slots.slot_mut(pane_idx) {
            if let Some((_, ref mut tab)) = slot.active {
                if let Some(fw) = tab.framework_mut() {
                    fw.collect_tool_bus_outbox(Some(ctx));
                    for env in fw.cross_tool_bus.drain_outgoing_actions() {
                        actions.push((Some(pane_idx), env));
                    }
                    if let Some(live) = fw.cross_tool_bus.take_outgoing_live() {
                        live_msgs.push((Some(pane_idx), live));
                    }
                }
            }
        }
    }

    for (src_pane, env) in actions {
        route_action_envelope(app, ctx, pane_count, src_pane, env);
    }
    for (src_pane, live) in live_msgs {
        route_live_envelope(app, ctx, pane_count, src_pane, &live);
    }
}

fn tab_matches_envelope(tab: &dyn crate::tabs::ExplorerTab, env: &CrossToolEnvelope) -> bool {
    if let Some(target_tab) = env.target_tab {
        return tab.name().eq_ignore_ascii_case(target_tab);
    }
    if let Some(target_tool) = env.target_tool {
        return tab.framework().is_some_and(|fw| {
            fw.available_tools.iter().any(|m| {
                m.id.eq_ignore_ascii_case(target_tool)
                    || m.display_name.eq_ignore_ascii_case(target_tool)
            })
        });
    }
    false
}

fn deliver_action_to_tab(
    tab: &mut dyn crate::tabs::ExplorerTab,
    ctx: &egui::Context,
    env: CrossToolEnvelope,
) {
    if env.focus_target {
        if let Some(target_tool) = env.target_tool {
            let _ = tab.select_tool_by_name(target_tool);
        }
    }
    if let Some(fw) = tab.framework_mut() {
        fw.cross_tool_bus.enqueue_incoming_action(env);
        let _ = fw.deliver_tool_bus_inbox(Some(ctx));
    }
}

fn route_action_envelope(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    pane_count: usize,
    src_pane: Option<usize>,
    env: CrossToolEnvelope,
) {
    for pane_idx in 1..pane_count {
        if src_pane == Some(pane_idx) {
            continue;
        }
        let matches = app
            .pane_slots
            .slot(pane_idx)
            .and_then(|s| s.active.as_ref())
            .is_some_and(|(_, t)| tab_matches_envelope(t.as_ref(), &env));
        if matches {
            if let Some(slot) = app.pane_slots.slot_mut(pane_idx) {
                if let Some((_, ref mut tab)) = slot.active {
                    deliver_action_to_tab(tab.as_mut(), ctx, env);
                }
            }
            return;
        }
    }

    let primary_idx = app.active_primary_tab_index();
    if src_pane != Some(0)
        && app
            .tabs
            .get(primary_idx)
            .is_some_and(|t| tab_matches_envelope(t.as_ref(), &env))
    {
        if let Some(tab) = app.tabs.get_mut(primary_idx) {
            deliver_action_to_tab(tab.as_mut(), ctx, env);
        }
        return;
    }

    if let Some(tab_idx) = app
        .tabs
        .iter()
        .position(|t| tab_matches_envelope(t.as_ref(), &env))
    {
        if env.focus_target {
            select_primary_tab(app, ctx, tab_idx);
        }
        if let Some(tab) = app.tabs.get_mut(tab_idx) {
            deliver_action_to_tab(tab.as_mut(), ctx, env);
        }
    }
}

fn route_live_envelope(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    pane_count: usize,
    src_pane: Option<usize>,
    live: &CrossToolEnvelope,
) {
    let primary_idx = app.active_primary_tab_index();
    if src_pane != Some(0) {
        if let Some(tab) = app.tabs.get_mut(primary_idx) {
            if live.target_tab.is_none() || tab_matches_envelope(tab.as_ref(), live) {
                if let Some(fw) = tab.framework_mut() {
                    fw.cross_tool_bus.set_incoming_live(live.clone());
                    let _ = fw.deliver_tool_bus_inbox(Some(ctx));
                }
            }
        }
    }
    for pane_idx in 1..pane_count {
        if src_pane == Some(pane_idx) {
            continue;
        }
        if let Some(slot) = app.pane_slots.slot_mut(pane_idx) {
            if let Some((_, ref mut tab)) = slot.active {
                if live.target_tab.is_none() || tab_matches_envelope(tab.as_ref(), live) {
                    if let Some(fw) = tab.framework_mut() {
                        fw.cross_tool_bus.set_incoming_live(live.clone());
                        let _ = fw.deliver_tool_bus_inbox(Some(ctx));
                    }
                }
            }
        }
    }
}

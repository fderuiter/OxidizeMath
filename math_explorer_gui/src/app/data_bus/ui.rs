//! Accessible UI controls and WCAG 2.2 AA banners for [`super::WorkspaceDataBus`] consumers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{read_pane_sync_meta, write_pane_sync_meta};
#[cfg(any(
    test,
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics"
))]
use crate::app::contrasting_text_color;
#[cfg(any(
    test,
    feature = "pure_math",
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics"
))]
use crate::app::SemanticPalette;
use eframe::egui;

#[cfg(any(
    test,
    feature = "pure_math",
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics"
))]
fn current_pane_and_tab(ctx: &egui::Context) -> (usize, usize) {
    let pane = ctx
        .data(|d| {
            d.get_temp::<usize>(egui::Id::new(
                crate::app::viewport::focus::CURRENT_RENDERING_PANE_KEY,
            ))
        })
        .unwrap_or(0)
        .min(3);
    let tab = ctx
        .data(|d| d.get_temp::<usize>(egui::Id::new(("pane_assigned_tab_idx", pane))))
        .unwrap_or(0);
    (pane, tab)
}

/// Synchronizes a consumer tool's `follow_workspace_graph` flag with any pane-header toggle click
/// and records active pane capability so `pane_chrome` can surface a header toggle button.
#[cfg(any(
    test,
    feature = "pure_math",
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics"
))]
pub fn sync_consumer_follow_flag(ctx: &egui::Context, follow: &mut bool, tool_label: &str) -> bool {
    let (pane, tab_idx) = current_pane_and_tab(ctx);
    let mut meta = read_pane_sync_meta(ctx);
    let mut changed = false;
    if let Some(override_val) = meta.pending_toggle[pane].take() {
        if *follow != override_val {
            *follow = override_val;
            changed = true;
            announce_sync_toggle(ctx, tool_label, *follow);
        }
    }
    meta.active_status[pane] = Some((tab_idx, *follow));
    write_pane_sync_meta(ctx, meta);
    changed
}

/// Emits a polite live-region screen-reader announcement when workspace graph sync toggles.
#[cfg(any(
    test,
    feature = "pure_math",
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics"
))]
pub fn announce_sync_toggle(ctx: &egui::Context, tool_label: &str, enabled: bool) {
    let msg = if enabled {
        format!("{tool_label}: Follow Workspace Graph enabled")
    } else {
        format!("{tool_label}: Follow Workspace Graph paused; retaining local graph snapshot")
    };
    crate::accessibility::announce_live_status(ctx, &msg);
}

/// Renders the accessible `Follow Workspace Graph` sidebar toggle and revision status badge.
///
/// Returns `true` if the user toggled the checkbox during this frame.
#[cfg(any(
    test,
    feature = "pure_math",
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics"
))]
pub fn render_follow_workspace_toggle(
    ui: &mut egui::Ui,
    follow: &mut bool,
    tool_label: &str,
    applied_revision: Option<u64>,
) -> bool {
    let header_changed = sync_consumer_follow_flag(ui.ctx(), follow, tool_label);
    let mut clicked = false;
    let palette = SemanticPalette::from_visuals(ui.visuals());

    ui.horizontal_wrapped(|ui| {
        let resp = ui
            .checkbox(follow, "Follow Workspace Graph")
            .on_hover_text(
                "Subscribe to live signed-graph edits from Graph Editor across panes, or uncheck to isolate this pane.",
            );
        if resp.changed() {
            clicked = true;
            let (pane, tab_idx) = current_pane_and_tab(ui.ctx());
            let mut meta = read_pane_sync_meta(ui.ctx());
            meta.active_status[pane] = Some((tab_idx, *follow));
            write_pane_sync_meta(ui.ctx(), meta);
            announce_sync_toggle(ui.ctx(), tool_label, *follow);
        }

        let (badge_text, color) = match (*follow, applied_revision) {
            (true, Some(rev)) if rev > 0 => (format!("Live rev #{rev}"), palette.success),
            (true, _) => ("Awaiting graph".to_string(), palette.muted_text),
            (false, _) => ("Isolated".to_string(), palette.warning),
        };
        ui.label(egui::RichText::new(badge_text).small().strong().color(color));
    });

    header_changed || clicked
}

/// Renders a polite WCAG 2.2 AA node-budget warning banner when a shared graph exceeds `budget`.
#[cfg(any(
    test,
    feature = "applied",
    feature = "biology",
    feature = "epidemiology",
    feature = "physics"
))]
pub fn render_node_budget_banner(
    ui: &mut egui::Ui,
    source_nodes: Option<usize>,
    budget: usize,
    unit_label: &str,
) {
    let Some(total) = source_nodes.filter(|&n| n > budget) else {
        return;
    };
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let bg = ui.visuals().faint_bg_color;
    let fg = contrasting_text_color(bg);
    let msg = format!(
        "Polite limit: clamped shared graph from {total} nodes to {budget} {unit_label} for real-time stability."
    );

    let announce_key = egui::Id::new(("node_budget_announced", unit_label));
    let prev = ui.ctx().data(|d| d.get_temp::<usize>(announce_key));
    if prev != Some(total) {
        ui.ctx().data_mut(|d| d.insert_temp(announce_key, total));
        crate::accessibility::announce_live_status(ui.ctx(), &msg);
    }

    egui::Frame::group(ui.style())
        .fill(bg)
        .stroke(egui::Stroke::new(1.5_f32, palette.warning))
        .inner_margin(egui::Margin::same(6))
        .show(ui, |banner_ui| {
            let resp = banner_ui.label(egui::RichText::new(&msg).small().color(fg));
            banner_ui.ctx().accesskit_node_builder(resp.id, |b| {
                b.set_role(egui::accesskit::Role::Status);
                b.set_live(egui::accesskit::Live::Polite);
            });
        });
}

/// Renders a compact `Follow Workspace Graph` button in `pane_chrome` headers when the pane's
/// active tool participates in workspace graph coupling.
pub(crate) fn render_pane_header_graph_sync(
    header_ui: &mut egui::Ui,
    ctx: &egui::Context,
    pane_idx: usize,
    assigned_tab_idx: usize,
    header_width: f32,
) {
    if pane_idx >= 4 || header_width < 220.0 {
        return;
    }
    let mut meta = read_pane_sync_meta(ctx);
    let Some((recorded_tab, is_following)) = meta.active_status[pane_idx] else {
        return;
    };
    if recorded_tab != assigned_tab_idx {
        return;
    }

    header_ui.separator();
    let label = if header_width >= 360.0 {
        if is_following {
            "🔗 Follow Graph: On"
        } else {
            "⏸ Follow Graph: Off"
        }
    } else if is_following {
        "🔗 Sync"
    } else {
        "⏸ Isolated"
    };
    let resp = header_ui
        .selectable_label(is_following, label)
        .on_hover_text("Toggle Follow Workspace Graph for this pane");
    if resp.clicked() {
        let next = !is_following;
        meta.pending_toggle[pane_idx] = Some(next);
        meta.active_status[pane_idx] = Some((assigned_tab_idx, next));
        write_pane_sync_meta(ctx, meta);
    }
}

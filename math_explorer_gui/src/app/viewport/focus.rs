#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::MathExplorerApp;
use eframe::egui;

/// Temp context key storing a requested pane index (`0..pane_count`) to focus on this frame.
pub(crate) const PENDING_PANE_FOCUS_KEY: &str = "PENDING_PANE_FOCUS";

/// Temp context key storing the authoritative active `focused_pane` index during rendering.
pub(crate) const ACTIVE_FOCUSED_PANE_KEY: &str = "ACTIVE_FOCUSED_PANE";

/// Temp context key storing the pane index currently rendering its child UI.
pub(crate) const CURRENT_RENDERING_PANE_KEY: &str = "CURRENT_RENDERING_PANE";

/// Temp context key indicating a shell menu, modal, popup, or split sash holds focus.
pub(crate) const SHELL_MODAL_OR_SASH_FOCUSED_KEY: &str = "SHELL_MODAL_OR_SASH_FOCUSED";

/// Temp context key storing the pane index last resolved at the end of the previous frame.
const LAST_RESOLVED_FOCUSED_PANE_KEY: &str = "LAST_RESOLVED_FOCUSED_PANE";

/// Returns the WCAG 2.2 SC 2.4.7 / Technique G195 high-contrast border color for the focused pane.
#[inline]
pub fn pane_focus_border_color(visuals: &egui::Visuals) -> egui::Color32 {
    crate::app::SemanticPalette::from_visuals(visuals).accent
}

/// Paints either the two-tone high-contrast focus border (WCAG G195) or the subtle unfocused border.
pub(super) fn paint_pane_border(
    painter: &egui::Painter,
    pane_rect: egui::Rect,
    is_focused: bool,
    visuals: &egui::Visuals,
) {
    if is_focused {
        let focus_width = visuals.selection.stroke.width.max(2.0_f32);
        painter.rect_stroke(
            pane_rect.shrink(focus_width),
            0.0,
            egui::Stroke::new(1.0_f32, visuals.panel_fill),
            egui::StrokeKind::Inside,
        );
        painter.rect_stroke(
            pane_rect,
            0.0,
            egui::Stroke::new(focus_width, pane_focus_border_color(visuals)),
            egui::StrokeKind::Inside,
        );
    } else {
        painter.rect_stroke(
            pane_rect,
            0.0,
            visuals.widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Inside,
        );
    }
}

/// Announces the newly focused pane region and active simulation module via [`crate::accessibility::TwinSlotLiveRegion`].
pub fn announce_pane_focus(app: &MathExplorerApp, ctx: &egui::Context, pane_idx: usize) {
    let pane_count = app.viewport_layout.pane_count();
    let label = app
        .viewport_layout
        .pane_labels()
        .get(pane_idx)
        .copied()
        .unwrap_or("Main Viewport");
    let tab_idx = if pane_idx == 0 {
        app.active_primary_tab_index()
    } else {
        app.pane_tabs.get(pane_idx).copied().unwrap_or(0)
    };
    let tab_name = app.tabs.get(tab_idx).map_or("No module", |t| t.name());
    let msg = if pane_count > 1 {
        format!(
            "Focused {label} (Pane {} of {pane_count}): {tab_name}",
            pane_idx + 1
        )
    } else {
        format!("Focused {label}: {tab_name}")
    };
    crate::accessibility::announce_live_status(ctx, msg);
}

/// Moves `focused_pane` and widget focus to `target_pane` and announces the switch.
pub fn focus_pane_interactive(app: &mut MathExplorerApp, ctx: &egui::Context, target_pane: usize) {
    let count = app.viewport_layout.pane_count();
    if target_pane >= count {
        return;
    }
    app.focused_pane = target_pane;
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(PENDING_PANE_FOCUS_KEY), target_pane);
        d.insert_temp(egui::Id::new(ACTIVE_FOCUSED_PANE_KEY), target_pane);
    });
    announce_pane_focus(app, ctx, target_pane);
    ctx.request_repaint();
}

/// Cycles `focused_pane` forward (`F6`) or backward (`Shift+F6`) across active panes.
pub fn cycle_focused_pane(app: &mut MathExplorerApp, ctx: &egui::Context, reverse: bool) {
    let count = app.viewport_layout.pane_count();
    let current = app.active_focused_pane();
    let next = if count <= 1 {
        0
    } else if reverse {
        (current + count - 1) % count
    } else {
        (current + 1) % count
    };
    focus_pane_interactive(app, ctx, next);
}

/// Registers `F6` and `Shift+F6` in the command registry and dispatches pane cycling shortcuts.
pub(crate) fn register_and_handle_pane_focus_shortcuts(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
) {
    egui_plot::commands::CommandRegistryData::register(
        ctx,
        "Focus Next Pane",
        "Move keyboard focus to the next viewport pane",
        egui_plot::commands::CommandTrigger::Key(egui::Key::F6),
        false,
        "Viewport",
    );
    egui_plot::commands::CommandRegistryData::register(
        ctx,
        "Focus Previous Pane",
        "Move keyboard focus to the previous viewport pane",
        egui_plot::commands::CommandTrigger::Shortcut(egui::Modifiers::SHIFT, egui::Key::F6),
        false,
        "Viewport",
    );

    let modal_blocked = app.command_palette.is_open
        || app.show_help_menu
        || egui::Popup::is_any_open(ctx)
        || ctx.memory(|m| m.top_modal_layer().is_some());
    if modal_blocked {
        return;
    }

    let shift_f6 = ctx.input_mut(|i| i.consume_key(egui::Modifiers::SHIFT, egui::Key::F6));
    if shift_f6 {
        cycle_focused_pane(app, ctx, true);
        return;
    }
    let next_f6 = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::F6));
    if next_f6 {
        cycle_focused_pane(app, ctx, false);
    }
}

/// Synchronizes `app.focused_pane` with the currently focused widget in `ctx` and reports
/// whether a shell bar, popup, modal, or split sash currently owns focus.
pub(super) fn sync_focused_pane_from_widget_focus(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    pane_rects: &[egui::Rect],
    sash_focused: bool,
    has_pending_pane_focus: bool,
) -> bool {
    let modal_or_popup = app.is_dragging_divider
        || app.dragging_quad_axes != (false, false)
        || sash_focused
        || app.command_palette.is_open
        || app.show_help_menu
        || egui::Popup::is_any_open(ctx)
        || ctx.memory(|m| m.top_modal_layer().is_some());

    let Some(fid) = ctx.memory(|m| m.focused()) else {
        return modal_or_popup;
    };
    if super::sash::is_any_sash_id(fid) {
        return true;
    }
    let Some(resp) = ctx.read_response(fid) else {
        return modal_or_popup;
    };
    if resp.layer_id.order != egui::Order::Background {
        return true;
    }
    if let Some(idx) = pane_rects
        .iter()
        .position(|&rect| response_belongs_to_pane(&resp, rect))
    {
        if !modal_or_popup && !has_pending_pane_focus {
            app.focus_pane(idx);
        }
        modal_or_popup
    } else {
        true
    }
}

/// Prepares `egui::Memory` so the first focusable widget rendered inside the target pane acquires focus.
pub(super) fn begin_pending_pane_focus(ctx: &egui::Context, should_focus_this_pane: bool) {
    if !should_focus_this_pane {
        return;
    }
    ctx.memory_mut(|m| {
        if let Some(prev) = m.focused() {
            m.surrender_focus(prev);
        }
        m.move_focus(egui::FocusDirection::Next);
    });
}

/// Clears any unconsumed `FocusDirection::Next` after rendering a target pane so focus cannot leak.
pub(super) fn finish_pending_pane_focus(ctx: &egui::Context, should_focus_this_pane: bool) {
    if should_focus_this_pane {
        ctx.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
    }
}

pub(crate) fn response_belongs_to_pane(resp: &egui::Response, pane_rect: egui::Rect) -> bool {
    if resp.interact_rect.is_positive() {
        pane_rect.expand(0.5).contains(resp.interact_rect.center())
    } else {
        pane_rect.contains(resp.rect.center())
    }
}

fn surrender_widget_focus_outside_rect(ctx: &egui::Context, target_rect: egui::Rect) {
    let Some(fid) = ctx.memory(|m| m.focused()) else {
        return;
    };
    if super::sash::is_any_sash_id(fid) {
        return;
    }
    let inside = ctx.read_response(fid).is_some_and(|r| {
        r.layer_id.order != egui::Order::Background || response_belongs_to_pane(&r, target_rect)
    });
    if !inside {
        ctx.memory_mut(|m| m.surrender_focus(fid));
    }
}

/// Resolves pointer/widget pane focus for the current multi-pane frame and updates temp focus keys.
pub(super) fn resolve_frame_pane_focus(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    pane_rects: &[egui::Rect],
    sash_focused: bool,
) -> Option<usize> {
    let raw_input = ctx.input(|i| i.raw.clone());
    let pending_focus =
        ctx.data_mut(|d| d.remove_temp::<usize>(egui::Id::new(PENDING_PANE_FOCUS_KEY)));
    if pending_focus.is_none() {
        app.update_focused_pane_from_input(ctx, &raw_input, pane_rects);
    }
    let current_target = app.active_focused_pane();
    let last_resolved =
        ctx.data(|d| d.get_temp::<usize>(egui::Id::new(LAST_RESOLVED_FOCUSED_PANE_KEY)));
    if let Some(&target_rect) = last_resolved
        .filter(|&prev| prev != current_target)
        .and_then(|_| pane_rects.get(current_target))
    {
        surrender_widget_focus_outside_rect(ctx, target_rect);
    }

    let shell_focused = sync_focused_pane_from_widget_focus(
        app,
        ctx,
        pane_rects,
        sash_focused,
        pending_focus.is_some(),
    );
    let focused_idx = app.active_focused_pane();
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(ACTIVE_FOCUSED_PANE_KEY), focused_idx);
        d.insert_temp(
            egui::Id::new(SHELL_MODAL_OR_SASH_FOCUSED_KEY),
            shell_focused,
        );
    });
    pending_focus
}

/// Records the final resolved focused pane at the end of a multi-pane frame.
pub(super) fn record_resolved_pane_focus(ctx: &egui::Context, focused_idx: usize) {
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new(LAST_RESOLVED_FOCUSED_PANE_KEY), focused_idx);
        d.insert_temp(egui::Id::new(ACTIVE_FOCUSED_PANE_KEY), focused_idx);
    });
}

fn is_pane_scoped_event(ev: &egui::Event) -> bool {
    matches!(
        ev,
        egui::Event::Key { .. }
            | egui::Event::Text(_)
            | egui::Event::Ime(_)
            | egui::Event::Copy
            | egui::Event::Cut
            | egui::Event::Paste(_)
    )
}

/// Runs `f` while temporarily masking keyboard, text, IME, and clipboard events when `pane_idx` is inactive.
pub(super) fn with_pane_scoped_input<R>(
    ctx: &egui::Context,
    pane_idx: usize,
    f: impl FnOnce() -> R,
) -> R {
    let allow_input = ctx.data(|d| {
        let shell_focused = d
            .get_temp::<bool>(egui::Id::new(SHELL_MODAL_OR_SASH_FOCUSED_KEY))
            .unwrap_or(false);
        let active_pane = d
            .get_temp::<usize>(egui::Id::new(ACTIVE_FOCUSED_PANE_KEY))
            .unwrap_or(0);
        !shell_focused && pane_idx == active_pane
    });
    if allow_input {
        return f();
    }

    let (stashed_events, stashed_keys) = ctx.input_mut(|i| {
        let mut kept = Vec::with_capacity(i.events.len());
        let mut stashed = Vec::new();
        for ev in i.events.drain(..) {
            if is_pane_scoped_event(&ev) {
                stashed.push(ev);
            } else {
                kept.push(ev);
            }
        }
        i.events = kept;
        let keys = std::mem::take(&mut i.keys_down);
        (stashed, keys)
    });

    let out = f();

    if !stashed_events.is_empty() || !stashed_keys.is_empty() {
        ctx.input_mut(|i| {
            i.events.extend(stashed_events);
            i.keys_down.extend(stashed_keys);
        });
    }
    out
}

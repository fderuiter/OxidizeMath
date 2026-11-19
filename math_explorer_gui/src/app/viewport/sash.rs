#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ViewportLayout;
use crate::accessibility::announce_status;
use crate::app::MathExplorerApp;
use eframe::egui;

/// Minimum allowed split ratio (`20%`).
pub const MIN_SPLIT_RATIO: f32 = 0.20;

/// Maximum allowed split ratio (`80%`).
pub const MAX_SPLIT_RATIO: f32 = 0.80;

/// Default equal-halves split ratio (`50%`).
pub const DEFAULT_SPLIT_RATIO: f32 = 0.50;

/// Coarse keyboard adjustment step (`5%`).
const COARSE_STEP: f64 = 0.05;

/// Fine Shift+Arrow keyboard adjustment step (`1%`).
const FINE_STEP: f64 = 0.01;

/// Returns the deterministic widget [`egui::Id`] for the dual-pane viewport split sash.
///
/// # Example
///
/// ```ignore
/// let id = split_sash_id();
/// ```
pub fn split_sash_id() -> egui::Id {
    egui::Id::new("viewport_split_sash")
}

/// Per-frame interaction outcome for the dual-pane split sash widget.
#[derive(Debug, Clone, Copy)]
pub(super) struct SashFrameState {
    pub is_hovered: bool,
    pub has_focus: bool,
    pub consumed_keys: bool,
}

/// Returns the high-contrast focus outline color for the active [`egui::Visuals`].
pub(super) fn focus_outline_color(visuals: &egui::Visuals) -> egui::Color32 {
    if visuals.dark_mode {
        egui::Color32::from_rgb(255, 214, 10)
    } else {
        egui::Color32::from_rgb(0, 72, 186)
    }
}

/// Returns `true` when `event` is a sash keyboard resize or reset key press.
pub(super) fn is_sash_adjustment_event(event: &egui::Event) -> bool {
    let egui::Event::Key {
        key,
        pressed: true,
        modifiers,
        ..
    } = event
    else {
        return false;
    };
    if modifiers.ctrl || modifiers.mac_cmd || modifiers.command || modifiers.alt {
        return false;
    }
    match key {
        egui::Key::ArrowLeft
        | egui::Key::ArrowRight
        | egui::Key::ArrowUp
        | egui::Key::ArrowDown => true,
        egui::Key::Home | egui::Key::Enter => !modifiers.shift,
        _ => false,
    }
}

fn clamp_ratio_f64(ratio: f64) -> f32 {
    const { assert!(MIN_SPLIT_RATIO < MAX_SPLIT_RATIO) };
    debug_assert!(ratio.is_finite());
    let clamped = ratio.clamp(f64::from(MIN_SPLIT_RATIO), f64::from(MAX_SPLIT_RATIO));
    let rounded_bp = (clamped * 10_000.0).round() / 10_000.0;
    (rounded_bp as f32).clamp(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO)
}

fn sash_accessible_label(layout: ViewportLayout) -> &'static str {
    match layout {
        ViewportLayout::SplitHorizontal => "Horizontal viewport split divider",
        ViewportLayout::SplitVertical => "Vertical viewport split divider",
        ViewportLayout::Single | ViewportLayout::QuadGrid => "Viewport split divider",
    }
}

fn announce_split_ratio(split_ratio: f32) {
    let clamped = split_ratio.clamp(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO);
    let pct = (clamped * 100.0).round() as i32;
    debug_assert!((20..=80).contains(&pct));
    announce_status(&format!("Split ratio {pct}%"));
}

fn apply_keyboard_adjustments(ctx: &egui::Context, split_ratio: &mut f32) -> (bool, bool, bool) {
    let start_ratio = (*split_ratio).clamp(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO);
    let mut current = f64::from(start_ratio);
    let mut consumed_any = false;
    let mut arrow_pressed = false;

    ctx.input_mut(|input| {
        input.events.retain(|event| {
            if !is_sash_adjustment_event(event) {
                return true;
            }
            let egui::Event::Key { key, modifiers, .. } = event else {
                return true;
            };
            let step = if modifiers.shift {
                FINE_STEP
            } else {
                COARSE_STEP
            };
            match key {
                egui::Key::ArrowLeft | egui::Key::ArrowUp => {
                    current = (current - step)
                        .clamp(f64::from(MIN_SPLIT_RATIO), f64::from(MAX_SPLIT_RATIO));
                    arrow_pressed = true;
                }
                egui::Key::ArrowRight | egui::Key::ArrowDown => {
                    current = (current + step)
                        .clamp(f64::from(MIN_SPLIT_RATIO), f64::from(MAX_SPLIT_RATIO));
                    arrow_pressed = true;
                }
                egui::Key::Home | egui::Key::Enter => {
                    current = f64::from(DEFAULT_SPLIT_RATIO);
                }
                _ => return true,
            }
            consumed_any = true;
            false
        });
    });

    if !consumed_any {
        return (false, false, false);
    }

    let updated = clamp_ratio_f64(current);
    let changed = (updated - start_ratio).abs() > 1e-6;
    *split_ratio = updated;
    (changed, true, arrow_pressed)
}

fn apply_accesskit_actions(ctx: &egui::Context, sash_id: egui::Id, split_ratio: &mut f32) -> bool {
    use egui::accesskit::{Action, ActionData};

    let start_ratio = (*split_ratio).clamp(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO);
    let mut current = f64::from(start_ratio);
    let mut requested = false;

    ctx.input(|input| {
        let inc = input.num_accesskit_action_requests(sash_id, Action::Increment);
        let dec = input.num_accesskit_action_requests(sash_id, Action::Decrement);
        if inc > 0 || dec > 0 {
            let net = inc as f64 - dec as f64;
            current = (current + net * COARSE_STEP)
                .clamp(f64::from(MIN_SPLIT_RATIO), f64::from(MAX_SPLIT_RATIO));
            requested = true;
        }
        for data in input.accesskit_action_requests(sash_id, Action::SetValue) {
            if let Some(ActionData::NumericValue(val)) = &data.data {
                let normalized = if *val > 1.0 { *val / 100.0 } else { *val };
                current = normalized.clamp(f64::from(MIN_SPLIT_RATIO), f64::from(MAX_SPLIT_RATIO));
                requested = true;
            }
        }
    });

    if !requested {
        return false;
    }

    let updated = clamp_ratio_f64(current);
    let changed = (updated - start_ratio).abs() > 1e-6;
    *split_ratio = updated;
    changed
}

fn handle_pointer_drag(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    sash_resp: &egui::Response,
    total_rect: egui::Rect,
    divider_rect: egui::Rect,
) -> (bool, bool) {
    let hover_pos = ctx.input(|i| i.pointer.hover_pos());
    let is_hovered = sash_resp.hovered() || hover_pos.is_some_and(|pos| divider_rect.contains(pos));
    let is_down = ctx.input(|i| i.pointer.primary_down());
    let was_dragging = app.is_dragging_divider;
    let drag_start_key = sash_resp.id.with("drag_start_ratio");

    if is_hovered && ctx.input(|i| i.pointer.any_pressed()) {
        if !app.is_dragging_divider {
            ctx.data_mut(|d| d.insert_temp(drag_start_key, app.split_ratio));
        }
        app.is_dragging_divider = true;
        sash_resp.request_focus();
    }
    if !is_down {
        app.is_dragging_divider = false;
    }
    app.split_view_state.is_dragging_divider = app.is_dragging_divider;

    let mut changed = false;
    if app.is_dragging_divider {
        let prev = app.split_ratio;
        let delta = ctx.input(|i| i.pointer.delta());
        let ratio_delta = match app.viewport_layout {
            ViewportLayout::SplitHorizontal if total_rect.width() > 0.0 => {
                delta.x / total_rect.width()
            }
            ViewportLayout::SplitVertical if total_rect.height() > 0.0 => {
                delta.y / total_rect.height()
            }
            _ => 0.0,
        };
        app.split_ratio = (app.split_ratio + ratio_delta).clamp(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO);
        changed = (app.split_ratio - prev).abs() > 1e-6;
    } else if was_dragging {
        let start_ratio = ctx.data_mut(|d| {
            let val = d.get_temp::<f32>(drag_start_key);
            d.remove::<f32>(drag_start_key);
            val
        });
        if start_ratio.is_some_and(|r| (app.split_ratio - r).abs() > 1e-4) {
            announce_split_ratio(app.split_ratio);
        }
    }

    if is_hovered || app.is_dragging_divider {
        ctx.set_cursor_icon(match app.viewport_layout {
            ViewportLayout::SplitHorizontal => egui::CursorIcon::ResizeColumn,
            ViewportLayout::SplitVertical => egui::CursorIcon::ResizeRow,
            _ => egui::CursorIcon::Default,
        });
    }

    (is_hovered, changed)
}

fn register_sash_accessibility(
    ui: &egui::Ui,
    sash_resp: &egui::Response,
    layout: ViewportLayout,
    split_ratio: f32,
) {
    let clamped = split_ratio.clamp(MIN_SPLIT_RATIO, MAX_SPLIT_RATIO);
    let pct_f64 = (f64::from(clamped) * 100.0).round();
    let value_text = format!("{}%", pct_f64 as i32);
    let label = sash_accessible_label(layout);

    let mut info = egui::WidgetInfo::labeled(egui::WidgetType::Other, ui.is_enabled(), label);
    info.value = Some(pct_f64);
    info.current_text_value = Some(value_text.clone());

    sash_resp.widget_info({
        let info = info.clone();
        move || info.clone()
    });
    ui.ctx().data_mut(|d| {
        d.insert_temp(sash_resp.id.with("sash_widget_info"), info);
    });

    ui.ctx().accesskit_node_builder(sash_resp.id, |builder| {
        use egui::accesskit::{Action, Orientation, Role};

        builder.set_role(Role::Splitter);
        builder.set_label(label);
        builder.set_value(value_text);
        builder.set_numeric_value(pct_f64);
        builder.set_min_numeric_value(20.0);
        builder.set_max_numeric_value(80.0);
        builder.set_numeric_value_step(5.0);
        builder.set_orientation(match layout {
            ViewportLayout::SplitHorizontal => Orientation::Vertical,
            ViewportLayout::SplitVertical => Orientation::Horizontal,
            ViewportLayout::Single | ViewportLayout::QuadGrid => Orientation::Vertical,
        });
        builder.add_action(Action::SetValue);
        if pct_f64 < 80.0 {
            builder.add_action(Action::Increment);
        }
        if pct_f64 > 20.0 {
            builder.add_action(Action::Decrement);
        }
    });
}

/// Allocates and processes pointer, keyboard, and AccessKit interactions for the split sash.
pub(super) fn interact_split_sash(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    total_rect: egui::Rect,
    divider_rect: egui::Rect,
) -> SashFrameState {
    let sash_id = split_sash_id();
    let mut sash_resp = ui.interact(divider_rect, sash_id, egui::Sense::click_and_drag());

    let (is_hovered, drag_changed) =
        handle_pointer_drag(app, ctx, &sash_resp, total_rect, divider_rect);

    let mut ratio_changed = drag_changed;
    let mut consumed_keys = false;

    if sash_resp.has_focus() {
        ctx.memory_mut(|m| {
            m.set_focus_lock_filter(
                sash_id,
                egui::EventFilter {
                    tab: false,
                    horizontal_arrows: true,
                    vertical_arrows: true,
                    escape: false,
                },
            );
        });

        let modal_open =
            app.command_palette.is_open || app.show_help_menu || egui::Popup::is_any_open(ctx);
        if !modal_open {
            let (kb_changed, kb_consumed, arrow_pressed) =
                apply_keyboard_adjustments(ctx, &mut app.split_ratio);
            consumed_keys = kb_consumed;
            if arrow_pressed {
                ctx.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            }
            if kb_changed {
                ratio_changed = true;
                announce_split_ratio(app.split_ratio);
            }
        }
    }

    if apply_accesskit_actions(ctx, sash_id, &mut app.split_ratio) {
        ratio_changed = true;
        announce_split_ratio(app.split_ratio);
    }

    if ratio_changed {
        sash_resp.mark_changed();
        ctx.request_repaint();
    }

    register_sash_accessibility(ui, &sash_resp, app.viewport_layout, app.split_ratio);

    SashFrameState {
        is_hovered,
        has_focus: sash_resp.has_focus(),
        consumed_keys,
    }
}

/// Paints the split divider bar, center handle grip, and high-contrast keyboard focus outline.
pub(super) fn paint_split_sash(
    ui: &mut egui::Ui,
    total_rect: egui::Rect,
    divider_rect: egui::Rect,
    layout: ViewportLayout,
    is_dragging: bool,
    state: &SashFrameState,
) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let divider_color = if is_dragging {
        palette.accent
    } else if state.is_hovered || state.has_focus {
        palette.graph_edge
    } else {
        palette.surface_fill
    };
    ui.painter().rect_filled(divider_rect, 0.0, divider_color);

    let center = divider_rect.center();
    let handle_color = if is_dragging || state.is_hovered || state.has_focus {
        crate::app::contrasting_text_color(divider_color)
    } else {
        palette.muted_text
    };
    let (p1, p2) = if layout == ViewportLayout::SplitHorizontal {
        (
            center - egui::vec2(0.0, 16.0),
            center + egui::vec2(0.0, 16.0),
        )
    } else {
        (
            center - egui::vec2(16.0, 0.0),
            center + egui::vec2(16.0, 0.0),
        )
    };
    ui.painter()
        .line_segment([p1, p2], egui::Stroke::new(2.0_f32, handle_color));

    if state.has_focus {
        let focus_rect = divider_rect.expand(1.5).intersect(total_rect);
        let outline_color = focus_outline_color(ui.visuals());
        ui.painter().rect_stroke(
            focus_rect,
            1.0,
            egui::Stroke::new(2.0_f32, outline_color),
            egui::StrokeKind::Inside,
        );
    }
}

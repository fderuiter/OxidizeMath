//! Internal WCAG 2.2 AA accessibility helpers for `egui_plot`.
//!
//! Provides high-contrast focus strokes (SC 2.4.7 / SC 1.4.11), focusable
//! `aria-disabled` action buttons (SC 2.1.1 / SC 4.1.2), and focus-triggered,
//! hover-persistent, `Escape`-dismissible tooltips (SC 1.4.13 / SCR39).

use egui::{
    AreaState, Button, Color32, Context, Id, Key, LayerId, Modifiers, PopupAnchor, Pos2, Response,
    RichText, Sense, Stroke, Tooltip, Ui, WidgetInfo, WidgetText, WidgetType,
};

#[derive(Clone, Default)]
pub(crate) struct PointTooltipState {
    pub anchor_pos: Pos2,
    pub point: [f64; 2],
    pub title: String,
    pub body_lines: Vec<(String, bool)>,
    pub active_last_frame: bool,
    pub dismissed: bool,
    pub dismissed_anchor: Option<Pos2>,
    pub dismissed_point: Option<[f64; 2]>,
    pub announced_text: Option<String>,
}

fn srgb_channel_to_linear(channel: u8) -> f32 {
    let normalized = f32::from(channel) / 255.0;
    if normalized <= 0.04045 {
        normalized / 12.92
    } else {
        ((normalized + 0.055) / 1.055).powf(2.4)
    }
}

fn relative_luminance(color: Color32) -> f32 {
    let [r, g, b, _] = color.to_srgba_unmultiplied();
    0.2126 * srgb_channel_to_linear(r)
        + 0.7152 * srgb_channel_to_linear(g)
        + 0.0722 * srgb_channel_to_linear(b)
}

/// Computes the WCAG 2.2 relative luminance contrast ratio between two colors.
#[must_use]
pub(crate) fn contrast_ratio(fg: Color32, bg: Color32) -> f32 {
    let lum_a = relative_luminance(fg);
    let lum_b = relative_luminance(bg);
    let lighter = lum_a.max(lum_b);
    let darker = lum_a.min(lum_b);
    (lighter + 0.05) / (darker + 0.05)
}

/// Returns a focus border stroke derived from `visuals.selection.stroke`
/// guaranteed to achieve at least 3:1 contrast against `bg`.
#[must_use]
pub(crate) fn focus_ring_stroke(visuals: &egui::Visuals, bg: Color32) -> Stroke {
    let mut stroke = visuals.selection.stroke;
    stroke.width = stroke.width.max(2.0);
    if contrast_ratio(stroke.color, bg) < 3.0 {
        let white_contrast = contrast_ratio(Color32::WHITE, bg);
        let black_contrast = contrast_ratio(Color32::BLACK, bg);
        stroke.color = if white_contrast >= black_contrast {
            Color32::WHITE
        } else {
            Color32::BLACK
        };
    }
    stroke
}

fn layer_may_handle_escape(ctx: &Context, layer_id: LayerId) -> bool {
    match ctx.memory(|m| m.top_modal_layer()) {
        None => true,
        Some(top_modal) => top_modal == layer_id,
    }
}

fn is_tooltip_area_hovered(ctx: &Context, parent_id: Id, margin: f32) -> bool {
    if !Tooltip::was_tooltip_open_last_frame(ctx, parent_id) {
        return false;
    }
    let tooltip_id = Tooltip::tooltip_id(parent_id, 0);
    let Some(area) = AreaState::load(ctx, tooltip_id) else {
        return false;
    };
    ctx.input(|i| {
        i.pointer
            .hover_pos()
            .is_some_and(|pos| area.rect().expand(margin).contains(pos))
    })
}

fn attach_button_semantics(response: &Response, enabled: bool, a11y_label: Option<&str>) {
    if let Some(label) = a11y_label {
        let owned = label.to_owned();
        response.widget_info({
            let info_label = owned.clone();
            move || WidgetInfo::labeled(WidgetType::Button, enabled, info_label.clone())
        });
        response.ctx.accesskit_node_builder(response.id, |builder| {
            builder.set_role(egui::accesskit::Role::Button);
            builder.set_label(owned);
            if !enabled {
                builder.set_disabled();
            }
        });
    } else if !enabled {
        response.ctx.accesskit_node_builder(response.id, |builder| {
            builder.set_disabled();
        });
    }
}

/// Renders an action button that remains keyboard-focusable (`aria-disabled`)
/// when `enabled` is `false` and `ui.is_enabled()` is `true`, allowing keyboard
/// users to inspect and `Escape`-dismiss disabled-reason tooltips.
pub(crate) fn focusable_action_button(
    ui: &mut Ui,
    enabled: bool,
    button: Button<'_>,
    a11y_label: Option<&str>,
) -> Response {
    let response = if enabled || !ui.is_enabled() {
        ui.add_enabled(enabled, button)
    } else {
        ui.scope(|ui| {
            let noninteractive = ui.visuals().widgets.noninteractive;
            ui.visuals_mut().widgets.inactive = noninteractive;
            ui.visuals_mut().widgets.hovered = noninteractive;
            ui.add(button.sense(Sense::focusable_noninteractive()))
        })
        .inner
    };
    attach_button_semantics(&response, enabled && ui.is_enabled(), a11y_label);
    response
}

/// Extension trait adding WCAG 2.2 SC 1.4.13 compliant tooltips to [`Response`].
pub(crate) trait AccessibleTooltipExt {
    /// Displays a tooltip on pointer hover or keyboard focus that stays open
    /// while hovering the popup and dismisses cleanly on `Key::Escape`.
    fn accessible_tooltip(self, text: impl Into<WidgetText>) -> Self;
}

impl AccessibleTooltipExt for Response {
    fn accessible_tooltip(self, text: impl Into<WidgetText>) -> Self {
        let dismissed_id = self.id.with("a11y_tooltip_dismissed");
        let announced_id = self.id.with("a11y_tooltip_announced");
        let was_open = Tooltip::was_tooltip_open_last_frame(&self.ctx, self.id);
        let dismissed = self
            .ctx
            .data(|d| d.get_temp::<bool>(dismissed_id).unwrap_or(false));
        let trigger_hovered = self.hovered() || self.contains_pointer();
        let popup_hovered = is_tooltip_area_hovered(&self.ctx, self.id, 8.0);
        let escape_on_focused_loss = was_open
            && !dismissed
            && self.lost_focus()
            && self.ctx.input(|i| i.key_pressed(Key::Escape));

        if (trigger_hovered || self.has_focus() || popup_hovered || escape_on_focused_loss)
            && !dismissed
            && layer_may_handle_escape(&self.ctx, self.layer_id)
            && self
                .ctx
                .input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape))
        {
            if self.lost_focus() {
                self.request_focus();
            }
            self.ctx.data_mut(|d| {
                d.insert_temp(dismissed_id, true);
                d.insert_temp(announced_id, false);
            });
            return self;
        }

        let active = trigger_hovered || self.has_focus() || popup_hovered;
        if !active {
            if dismissed {
                self.ctx.data_mut(|d| d.insert_temp(dismissed_id, false));
            }
            self.ctx.data_mut(|d| d.insert_temp(announced_id, false));
            return self;
        }
        if dismissed {
            return self;
        }

        render_and_announce_widget_tooltip(&self, text.into(), announced_id);
        self
    }
}

fn render_and_announce_widget_tooltip(
    response: &Response,
    widget_text: WidgetText,
    announced_id: Id,
) {
    let plain = widget_text.text().to_owned();
    let already_announced = response
        .ctx
        .data(|d| d.get_temp::<bool>(announced_id).unwrap_or(false));
    if !already_announced && !plain.is_empty() {
        crate::emit_plot_live_message(&response.ctx, plain, false);
        response.ctx.data_mut(|d| d.insert_temp(announced_id, true));
    }
    let width = response.ctx.style().spacing.tooltip_width;
    Tooltip::for_widget(response).gap(6.0).show(|ui| {
        ui.set_max_width(width);
        ui.add(egui::Label::new(widget_text));
    });
}

/// Checks whether the point ruler tooltip is currently kept alive by pointer
/// hover over the popup or dismissed via `Key::Escape` on the current frame.
pub(crate) fn preprocess_point_tooltip_frame(
    plot_area_response: &Response,
    plot_hovered_or_focused: bool,
) -> Option<PointTooltipState> {
    let state_id = plot_area_response.id.with("a11y_point_tooltip_state");
    let mut state = plot_area_response
        .ctx
        .data(|d| d.get_temp::<PointTooltipState>(state_id))
        .unwrap_or_default();
    let popup_hovered =
        is_tooltip_area_hovered(&plot_area_response.ctx, plot_area_response.id, 10.0);
    let lost_focus_on_escape = state.active_last_frame
        && !state.dismissed
        && plot_area_response.lost_focus()
        && plot_area_response.ctx.input(|i| i.key_pressed(Key::Escape));

    if state.active_last_frame
        && !state.dismissed
        && (plot_hovered_or_focused || popup_hovered || lost_focus_on_escape)
        && layer_may_handle_escape(&plot_area_response.ctx, plot_area_response.layer_id)
        && plot_area_response
            .ctx
            .input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape))
    {
        if plot_area_response.lost_focus() {
            plot_area_response.request_focus();
        }
        state.dismissed = true;
        state.dismissed_anchor = Some(state.anchor_pos);
        state.dismissed_point = Some(state.point);
        state.active_last_frame = false;
        state.announced_text = None;
        plot_area_response
            .ctx
            .data_mut(|d| d.insert_temp(state_id, state));
        return None;
    }

    if !plot_hovered_or_focused && !popup_hovered {
        state.dismissed = false;
        state.dismissed_anchor = None;
        state.dismissed_point = None;
        state.active_last_frame = false;
        state.announced_text = None;
        plot_area_response
            .ctx
            .data_mut(|d| d.insert_temp(state_id, state));
        return None;
    }

    (popup_hovered && !state.dismissed && state.active_last_frame).then_some(state)
}

/// Renders a point tooltip anchored at `anchor_pos` (snapped data point screen
/// coordinates) unless dismissed by `Escape`, updating hover-persistence state.
pub(crate) fn show_snapped_point_tooltip(
    plot_area_response: &Response,
    anchor_pos: Pos2,
    point: [f64; 2],
    title: &str,
    body_lines: &[(String, bool)],
    announce_on_open: bool,
) {
    let state_id = plot_area_response.id.with("a11y_point_tooltip_state");
    let mut state = plot_area_response
        .ctx
        .data(|d| d.get_temp::<PointTooltipState>(state_id))
        .unwrap_or_default();

    let anchor_moved = state
        .dismissed_anchor
        .is_some_and(|prev| prev.distance(anchor_pos) > 2.5);
    let point_changed = state.dismissed_point.is_some_and(|prev| {
        (prev[0] - point[0]).abs() > 1e-12 || (prev[1] - point[1]).abs() > 1e-12
    });
    if anchor_moved || point_changed {
        state.dismissed = false;
        state.dismissed_anchor = None;
        state.dismissed_point = None;
    }
    if state.dismissed {
        state.active_last_frame = false;
        plot_area_response
            .ctx
            .data_mut(|d| d.insert_temp(state_id, state));
        return;
    }

    state.anchor_pos = anchor_pos;
    state.point = point;
    state.title = title.to_owned();
    state.body_lines = body_lines.to_vec();
    state.active_last_frame = true;

    let summary = format_point_tooltip_summary(title, body_lines);
    if announce_on_open && state.announced_text.as_deref() != Some(summary.as_str()) {
        if !summary.is_empty() {
            crate::emit_plot_live_message(&plot_area_response.ctx, summary.clone(), false);
        }
        state.announced_text = Some(summary);
    }

    plot_area_response
        .ctx
        .data_mut(|d| d.insert_temp(state_id, state));
    paint_anchored_point_popup(plot_area_response, anchor_pos, title, body_lines);
}

fn format_point_tooltip_summary(title: &str, body_lines: &[(String, bool)]) -> String {
    let mut parts: Vec<&str> = Vec::with_capacity(body_lines.len() + 1);
    if !title.is_empty() {
        parts.push(title);
    }
    for (line, _) in body_lines {
        if !line.is_empty() {
            parts.push(line.as_str());
        }
    }
    parts.join(", ")
}

fn paint_anchored_point_popup(
    plot_area_response: &Response,
    anchor_pos: Pos2,
    title: &str,
    body_lines: &[(String, bool)],
) {
    let tooltip_width = plot_area_response.ctx.style().spacing.tooltip_width;
    let mut tooltip = Tooltip::always_open(
        plot_area_response.ctx.clone(),
        plot_area_response.layer_id,
        plot_area_response.id,
        PopupAnchor::Position(anchor_pos),
    );
    tooltip.popup = tooltip.popup.width(tooltip_width);
    tooltip.gap(12.0).show(|ui| {
        ui.set_max_width(tooltip_width);
        if !title.is_empty() {
            ui.label(RichText::new(title).strong());
            if !body_lines.is_empty() {
                ui.add_space(2.0);
            }
        }
        for (line, monospace) in body_lines {
            if *monospace {
                ui.label(RichText::new(line).monospace());
            } else {
                ui.label(line);
            }
        }
    });
}

#[cfg(test)]
#[path = "a11y_tests.rs"]
mod tests;

//! Internal WCAG 2.2 AA accessibility helpers for `egui_plot`.
//!
//! Provides high-contrast focus strokes (SC 2.4.7 / SC 1.4.11), focusable
//! `aria-disabled` action buttons (SC 2.1.1 / SC 4.1.2), and focus-triggered,
//! hover-persistent, `Escape`-dismissible tooltips (SC 1.4.13 / SCR39).

use egui::{
    AreaState, Button, Color32, Context, EventFilter, FocusDirection, Id, Key, LayerId, Modifiers,
    PopupAnchor, Pos2, Rect, Response, RichText, Sense, Stroke, StrokeKind, Tooltip, Ui, UiBuilder,
    WidgetInfo, WidgetText, WidgetType,
};

#[derive(Clone, Default)]
pub(crate) struct PointTooltipState {
    pub anchor_pos: Pos2,
    pub point: [f64; 2],
    pub world_point: Option<[f64; 3]>,
    pub suppress_rulers: bool,
    pub title: String,
    pub body_lines: Vec<(String, bool)>,
    pub active_last_frame: bool,
    pub dismissed: bool,
    pub dismissed_anchor: Option<Pos2>,
    pub dismissed_point: Option<[f64; 2]>,
    pub dismissed_world_point: Option<[f64; 3]>,
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
pub fn contrast_ratio(fg: Color32, bg: Color32) -> f32 {
    let lum_a = relative_luminance(fg);
    let lum_b = relative_luminance(bg);
    let lighter = lum_a.max(lum_b);
    let darker = lum_a.min(lum_b);
    (lighter + 0.05) / (darker + 0.05)
}

/// Composites a premultiplied `fg` color (including additive fills) over an opaque `bg` color.
#[must_use]
pub(crate) fn composite_over(fg: Color32, bg: Color32) -> Color32 {
    if fg == Color32::TRANSPARENT {
        return bg;
    }
    if fg.a() == u8::MAX {
        return fg;
    }
    let inv_a = 1.0 - f32::from(fg.a()) / 255.0;
    let r = (f32::from(fg.r()) + f32::from(bg.r()) * inv_a)
        .round()
        .clamp(0.0, 255.0) as u8;
    let g = (f32::from(fg.g()) + f32::from(bg.g()) * inv_a)
        .round()
        .clamp(0.0, 255.0) as u8;
    let b = (f32::from(fg.b()) + f32::from(bg.b()) * inv_a)
        .round()
        .clamp(0.0, 255.0) as u8;
    Color32::from_rgb(r, g, b)
}

/// Returns a focus border stroke derived from `visuals.selection.stroke`
/// guaranteed to achieve at least 3:1 contrast against `bg`.
#[must_use]
pub fn focus_ring_stroke(visuals: &egui::Visuals, bg: Color32) -> Stroke {
    let resolved_bg = composite_over(bg, visuals.window_fill);
    let mut stroke = visuals.selection.stroke;
    stroke.width = stroke.width.max(2.0);
    if contrast_ratio(stroke.color, resolved_bg) < 3.0 {
        let white_contrast = contrast_ratio(Color32::WHITE, resolved_bg);
        let black_contrast = contrast_ratio(Color32::BLACK, resolved_bg);
        stroke.color = if white_contrast >= black_contrast {
            Color32::WHITE
        } else {
            Color32::BLACK
        };
    }
    stroke
}

/// Returns `preferred` when it achieves at least 3:1 contrast against `bg`,
/// falling back to [`focus_ring_stroke`]'s guaranteed `>= 3:1` color otherwise.
#[must_use]
pub(crate) fn accessible_marker_color(
    visuals: &egui::Visuals,
    preferred: Color32,
    bg: Color32,
) -> Color32 {
    if preferred == Color32::TRANSPARENT {
        return Color32::TRANSPARENT;
    }
    let resolved_bg = composite_over(bg, visuals.window_fill);
    if contrast_ratio(preferred, resolved_bg) >= 3.0 {
        preferred
    } else {
        focus_ring_stroke(visuals, resolved_bg).color
    }
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
    let effective_enabled = enabled && ui.is_enabled();
    attach_button_semantics(&response, effective_enabled, a11y_label);
    if response.has_focus() {
        let ring = focus_ring_stroke(ui.visuals(), ui.visuals().window_fill);
        ui.painter()
            .rect_stroke(response.rect, 2.0, ring, StrokeKind::Inside);
    }
    response
}

pub(crate) fn lock_horizontal_arrows_for_tab(ctx: &Context, id: Id) {
    ctx.memory_mut(|m| {
        m.set_focus_lock_filter(
            id,
            EventFilter {
                tab: false,
                horizontal_arrows: true,
                vertical_arrows: false,
                escape: false,
            },
        );
    });
}

/// Renders a keyboard-focusable `Role::Tab` button with `aria-selected` state,
/// horizontal arrow key locking for `TabList` roving, and a `>= 3:1` focus ring.
pub(crate) fn focusable_tab_button(ui: &mut Ui, selected: bool, label: &str) -> Response {
    let response = focusable_action_button(ui, true, Button::selectable(selected, label), None);
    let enabled = ui.is_enabled();
    let owned = label.to_owned();
    response.widget_info({
        let info_label = owned.clone();
        move || {
            WidgetInfo::selected(
                WidgetType::SelectableLabel,
                enabled,
                selected,
                info_label.clone(),
            )
        }
    });
    response.ctx.accesskit_node_builder(response.id, |builder| {
        builder.set_role(egui::accesskit::Role::Tab);
        builder.set_label(owned);
        builder.set_selected(selected);
        builder.clear_toggled();
    });
    if response.has_focus() {
        lock_horizontal_arrows_for_tab(&response.ctx, response.id);
        let bg = if selected {
            ui.visuals().selection.bg_fill
        } else {
            ui.visuals().window_fill
        };
        let ring = focus_ring_stroke(ui.visuals(), bg);
        ui.painter()
            .rect_stroke(response.rect, 2.0, ring, StrokeKind::Inside);
    }
    response
}

/// Renders a modal action button that stays keyboard-focusable as `aria-disabled`
/// when `!enabled`, exposing `disabled_reason` via an `Escape`-dismissible tooltip.
pub(crate) fn modal_action_button(
    ui: &mut Ui,
    enabled: bool,
    label: &str,
    disabled_reason: &str,
) -> Response {
    let mut response = focusable_action_button(ui, enabled, Button::new(label), Some(label));
    if !enabled && !disabled_reason.is_empty() {
        response = response.accessible_tooltip(disabled_reason);
    }
    response
}

/// Renders a modal status message label tagged with `Role::Status` and polite/assertive live metadata.
pub(crate) fn status_message_label(ui: &mut Ui, msg: &str, is_err: bool) -> Response {
    let base = if is_err {
        ui.visuals().error_fg_color
    } else {
        ui.visuals().selection.stroke.color
    };
    let fg = accessible_marker_color(ui.visuals(), base, ui.visuals().window_fill);
    let response = ui.colored_label(fg, msg);
    let owned = msg.to_owned();
    response.ctx.accesskit_node_builder(response.id, |builder| {
        builder.set_role(egui::accesskit::Role::Status);
        builder.set_label(owned.clone());
        builder.set_value(owned);
        builder.set_live(if is_err {
            egui::accesskit::Live::Assertive
        } else {
            egui::accesskit::Live::Polite
        });
        builder.set_live_atomic();
    });
    response
}

/// Handles `ArrowLeft`/`ArrowRight` roving selection across a rendered `Role::TabList`.
pub(crate) fn step_tablist_on_arrow_keys(ui: &Ui, tab_responses: &[Response]) -> Option<usize> {
    let n = tab_responses.len();
    if n < 2 {
        return None;
    }
    let cur = tab_responses.iter().position(Response::has_focus)?;
    let next = if ui
        .ctx()
        .input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowRight))
    {
        (cur + 1) % n
    } else if ui
        .ctx()
        .input_mut(|i| i.consume_key(Modifiers::NONE, Key::ArrowLeft))
    {
        (cur + n - 1) % n
    } else {
        return None;
    };
    let target_id = tab_responses[next].id;
    ui.ctx().memory_mut(|m| {
        m.move_focus(FocusDirection::None);
        m.request_focus(target_id);
    });
    lock_horizontal_arrows_for_tab(ui.ctx(), target_id);
    ui.ctx().request_repaint();
    Some(next)
}

pub(crate) fn set_accesskit_bounds(ctx: &Context, id: Id, rect: Rect) {
    ctx.accesskit_node_builder(id, |builder| {
        builder.set_bounds(egui::accesskit::Rect {
            x0: f64::from(rect.min.x),
            y0: f64::from(rect.min.y),
            x1: f64::from(rect.max.x),
            y1: f64::from(rect.max.y),
        });
    });
}

pub(crate) fn register_table_node(
    ctx: &Context,
    table_id: Id,
    label: &str,
    row_count: usize,
    column_count: usize,
) {
    let owned = label.to_owned();
    ctx.accesskit_node_builder(table_id, |builder| {
        builder.set_role(egui::accesskit::Role::Table);
        builder.set_label(owned);
        builder.set_row_count(row_count);
        builder.set_column_count(column_count);
    });
}

pub(crate) fn register_row_node(
    ui: &mut Ui,
    resp: &Response,
    label: String,
    row_index: usize,
    selected: Option<bool>,
) {
    let enabled = ui.is_enabled();
    resp.widget_info({
        let info_label = label.clone();
        move || match selected {
            Some(sel) => WidgetInfo::selected(WidgetType::Other, enabled, sel, info_label.clone()),
            None => WidgetInfo::labeled(WidgetType::Other, enabled, info_label.clone()),
        }
    });
    ui.ctx().accesskit_node_builder(resp.id, |builder| {
        builder.set_role(egui::accesskit::Role::Row);
        builder.set_label(label);
        builder.set_row_index(row_index);
        builder.clear_toggled();
        if let Some(sel) = selected {
            builder.set_selected(sel);
        }
    });
    if resp.has_focus() {
        ui.memory_mut(|m| {
            m.set_focus_lock_filter(
                resp.id,
                EventFilter {
                    tab: false,
                    horizontal_arrows: false,
                    vertical_arrows: true,
                    escape: false,
                },
            );
        });
    }
}

pub(crate) fn scope_table_row(
    ui: &mut Ui,
    table_id: Id,
    row_salt: impl std::hash::Hash,
    size: egui::Vec2,
    sense: Sense,
    row_index: usize,
    add_cells: impl FnOnce(&mut Ui, Rect, &Response) -> (String, Option<bool>),
) -> Response {
    let mut row_meta: Option<(String, Option<bool>)> = None;
    let ir = ui.scope_builder(
        UiBuilder::new()
            .id_salt(row_salt)
            .sense(sense)
            .accessibility_parent(table_id),
        |row_ui| {
            let (_, rect) = row_ui.allocate_space(size);
            let pre_resp = row_ui.response();
            row_meta = Some(add_cells(row_ui, rect, &pre_resp));
        },
    );
    let (label, selected) = row_meta.unwrap_or_default();
    register_row_node(ui, &ir.response, label, row_index, selected);
    ir.response
}

pub(crate) fn register_cell_node(
    row_ui: &Ui,
    cell_id: Id,
    cell_rect: Rect,
    role: egui::accesskit::Role,
    col_index: usize,
    label: String,
    focusable: bool,
    row_bg: Color32,
) -> Response {
    let sense = if focusable {
        Sense::focusable_noninteractive()
    } else {
        Sense::hover()
    };
    let resp = row_ui.interact(cell_rect, cell_id, sense);
    let enabled = row_ui.is_enabled();
    resp.widget_info({
        let info_label = label.clone();
        move || WidgetInfo::labeled(WidgetType::Label, enabled, info_label.clone())
    });
    row_ui.ctx().accesskit_node_builder(resp.id, |builder| {
        builder.set_role(role);
        builder.set_label(label);
        builder.clear_value();
        builder.set_column_index(col_index);
    });
    if focusable && resp.has_focus() {
        let ring = focus_ring_stroke(row_ui.visuals(), row_bg);
        row_ui
            .painter()
            .rect_stroke(cell_rect, 2.0, ring, StrokeKind::Inside);
    }
    resp
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
        state.dismissed_world_point = state.world_point;
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
        state.dismissed_world_point = None;
        state.active_last_frame = false;
        state.announced_text = None;
        plot_area_response
            .ctx
            .data_mut(|d| d.insert_temp(state_id, state));
        return None;
    }

    (popup_hovered && !state.dismissed && state.active_last_frame).then_some(state)
}

pub(crate) struct SnappedTooltipArgs<'a> {
    pub anchor_pos: Pos2,
    pub point: [f64; 2],
    pub world_point: Option<[f64; 3]>,
    pub suppress_rulers: bool,
    pub title: &'a str,
    pub body_lines: &'a [(String, bool)],
    pub announce_on_open: bool,
}

fn world_point_changed(prev: Option<[f64; 3]>, curr: Option<[f64; 3]>) -> bool {
    match (prev, curr) {
        (Some(a), Some(b)) => {
            (a[0] - b[0]).abs() > 1e-12
                || (a[1] - b[1]).abs() > 1e-12
                || (a[2] - b[2]).abs() > 1e-12
        }
        (None, None) => false,
        _ => true,
    }
}

/// Renders a point tooltip anchored at `anchor_pos` (snapped data point screen
/// coordinates) unless dismissed by `Escape`, updating hover-persistence state.
pub(crate) fn show_snapped_point_tooltip(
    plot_area_response: &Response,
    args: SnappedTooltipArgs<'_>,
) {
    let state_id = plot_area_response.id.with("a11y_point_tooltip_state");
    let mut state = plot_area_response
        .ctx
        .data(|d| d.get_temp::<PointTooltipState>(state_id))
        .unwrap_or_default();

    let anchor_moved = state
        .dismissed_anchor
        .is_some_and(|prev| prev.distance(args.anchor_pos) > 2.5);
    let point_changed = state.dismissed_point.is_some_and(|prev| {
        (prev[0] - args.point[0]).abs() > 1e-12
            || (prev[1] - args.point[1]).abs() > 1e-12
            || world_point_changed(state.dismissed_world_point, args.world_point)
    });
    if anchor_moved || point_changed {
        state.dismissed = false;
        state.dismissed_anchor = None;
        state.dismissed_point = None;
        state.dismissed_world_point = None;
    }
    if state.dismissed {
        state.active_last_frame = false;
        plot_area_response
            .ctx
            .data_mut(|d| d.insert_temp(state_id, state));
        return;
    }

    state.anchor_pos = args.anchor_pos;
    state.point = args.point;
    state.world_point = args.world_point;
    state.suppress_rulers = args.suppress_rulers;
    state.title = args.title.to_owned();
    state.body_lines = args.body_lines.to_vec();
    state.active_last_frame = true;

    let summary = format_point_tooltip_summary(args.title, args.body_lines);
    if args.announce_on_open && state.announced_text.as_deref() != Some(summary.as_str()) {
        if !summary.is_empty() {
            crate::emit_plot_live_message(&plot_area_response.ctx, summary.clone(), false);
        }
        state.announced_text = Some(summary);
    }

    plot_area_response
        .ctx
        .data_mut(|d| d.insert_temp(state_id, state));
    paint_anchored_point_popup(
        plot_area_response,
        args.anchor_pos,
        args.title,
        args.body_lines,
    );
}

/// Formats a polite ARIA live announcement for a focused or snapped critical point.
#[must_use]
pub(crate) fn format_critical_point_live_summary(
    poi: &crate::poi::CriticalPoint,
    index: usize,
    total: usize,
) -> String {
    format!(
        "Critical point {} of {}: {} ({}), X: {:.4}, Y: {:.4}",
        index + 1,
        total,
        poi.kind.label(),
        poi.series_summary(),
        poi.point[0],
        poi.point[1]
    )
}

/// Renders an Escape-dismissible tooltip anchored at a critical point marker and optionally announces it.
pub(crate) fn show_critical_point_tooltip(
    plot_area_response: &Response,
    anchor_pos: Pos2,
    poi: &crate::poi::CriticalPoint,
    announce_on_open: bool,
    include_pin_hint: bool,
) {
    let mut body_lines = vec![
        (format!("x = {:.6}", poi.point[0]), true),
        (format!("y = {:.6}", poi.point[1]), true),
    ];
    if include_pin_hint {
        body_lines.push(("Click marker to pin callout".to_owned(), false));
    }
    show_snapped_point_tooltip(
        plot_area_response,
        SnappedTooltipArgs {
            anchor_pos,
            point: poi.point,
            world_point: None,
            suppress_rulers: false,
            title: &poi.tooltip_title(),
            body_lines: &body_lines,
            announce_on_open,
        },
    );
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

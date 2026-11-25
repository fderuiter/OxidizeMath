//! Shared simulation canvas accessibility registration, high-contrast `graph_edge` borders,
//! two-tone focus/hover/reticle overlays (WCAG 2.2 Technique G195), and inline swatch legends.

use crate::app::{contrasting_text_color, SemanticPalette};
use eframe::egui::{self, Color32, Pos2, Rect, Response, Stroke, StrokeKind};

/// Attaches `WidgetInfo` and `AccessKit` node metadata (`role`, `label`, and optional `value` /
/// `description`) to a custom canvas or visualization widget.
pub fn register_canvas_accessibility(
    ui: &egui::Ui,
    response: &Response,
    role: egui::accesskit::Role,
    label: &str,
    value: Option<&str>,
) {
    let mut info = egui::WidgetInfo::labeled(egui::WidgetType::Other, ui.is_enabled(), label);
    if let Some(val) = value {
        info.current_text_value = Some(val.to_string());
    }
    response.widget_info({
        let info = info.clone();
        move || info.clone()
    });
    ui.ctx().data_mut(|d| {
        d.insert_temp(response.id.with("canvas_widget_info"), info);
    });
    ui.ctx().accesskit_node_builder(response.id, |builder| {
        builder.set_role(role);
        builder.set_label(label);
        if let Some(val) = value {
            builder.set_value(val);
            builder.set_description(val);
        }
    });
}

/// Returns the semantic `>= 3.0:1` simulation canvas border stroke (`palette.graph_edge`).
#[inline]
#[must_use]
pub fn canvas_border_stroke(visuals: &egui::Visuals) -> Stroke {
    let palette = SemanticPalette::from_visuals(visuals);
    Stroke::new(1.25_f32, palette.graph_edge)
}

/// Strokes `rect` with the semantic `palette.graph_edge` border (`>= 3.0:1` across all themes).
pub fn paint_canvas_border(ui: &egui::Ui, rect: Rect) {
    ui.painter().rect_stroke(
        rect,
        0.0_f32,
        canvas_border_stroke(ui.visuals()),
        StrokeKind::Inside,
    );
}

/// Paints a two-tone rectangular frame pairing `primary` with an inner contrasting halo
/// (`contrasting_text_color(primary)`), guaranteeing `>= 3.0:1` contrast over any pixel.
pub fn paint_two_tone_rect_frame(
    painter: &egui::Painter,
    rect: Rect,
    rounding: f32,
    primary: Color32,
    primary_width: f32,
) {
    let halo = contrasting_text_color(primary);
    painter.rect_stroke(
        rect,
        rounding,
        Stroke::new(primary_width + 2.0_f32, halo),
        StrokeKind::Inside,
    );
    painter.rect_stroke(
        rect,
        rounding,
        Stroke::new(primary_width, primary),
        StrokeKind::Inside,
    );
}

/// Paints a two-tone reticle frame centered around `rect` (`Middle`/`Outside` halo + `primary`),
/// preserving visibility on small raster cells without obscuring single-pixel contents.
pub fn paint_two_tone_reticle(
    painter: &egui::Painter,
    rect: Rect,
    rounding: f32,
    primary: Color32,
) {
    let halo = contrasting_text_color(primary);
    painter.rect_stroke(
        rect,
        rounding,
        Stroke::new(3.5_f32, halo),
        StrokeKind::Middle,
    );
    painter.rect_stroke(
        rect,
        rounding,
        Stroke::new(1.8_f32, primary),
        StrokeKind::Middle,
    );
}

/// Paints a two-tone circle marker (`contrasting_text_color(primary)` halo + `primary` core).
pub fn paint_two_tone_circle(
    painter: &egui::Painter,
    center: Pos2,
    radius: f32,
    primary: Color32,
    width: f32,
) {
    let halo = contrasting_text_color(primary);
    painter.circle_stroke(center, radius, Stroke::new(width + 2.0_f32, halo));
    painter.circle_stroke(center, radius, Stroke::new(width, primary));
}

/// Paints a two-tone line segment (`contrasting_text_color(primary)` halo + `primary` core).
pub fn paint_two_tone_line(
    painter: &egui::Painter,
    points: [Pos2; 2],
    primary: Color32,
    width: f32,
) {
    let halo = contrasting_text_color(primary);
    painter.line_segment(points, Stroke::new(width + 2.0_f32, halo));
    painter.line_segment(points, Stroke::new(width, primary));
}

/// Paints a two-tone focus ring inside `response.rect` when the widget holds keyboard focus.
pub fn paint_canvas_focus_ring(ui: &egui::Ui, response: &Response) {
    if response.has_focus() {
        let primary = SemanticPalette::from_visuals(ui.visuals()).accent;
        paint_two_tone_rect_frame(ui.painter(), response.rect, 2.0_f32, primary, 2.0_f32);
    }
}

/// Locks horizontal and vertical arrow keys to `id` while leaving `Tab` and `Escape` unlocked.
pub fn lock_arrow_keys_for_id(ctx: &egui::Context, id: egui::Id) {
    ctx.memory_mut(|m| {
        m.set_focus_lock_filter(
            id,
            egui::EventFilter {
                tab: false,
                horizontal_arrows: true,
                vertical_arrows: true,
                escape: false,
            },
        );
    });
}

/// Locks horizontal and vertical arrow keys to `response.id` strictly while focused,
/// leaving `Tab` and `Escape` unlocked so keyboard traversal is never trapped.
pub fn lock_canvas_arrow_keys(ui: &egui::Ui, response: &Response) {
    if response.has_focus() {
        if response.gained_focus() {
            ui.ctx().request_repaint();
        }
        lock_arrow_keys_for_id(ui.ctx(), response.id);
    }
}

/// Renders a compact legend swatch pairing `fill`, an optional non-color `glyph` inside the swatch,
/// a `graph_edge` outline, and a readable `label`.
pub fn draw_legend_swatch(ui: &mut egui::Ui, fill: Color32, glyph: Option<&str>, label: &str) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let (swatch_rect, _) =
            ui.allocate_exact_size(egui::vec2(14.0_f32, 14.0_f32), egui::Sense::hover());
        ui.painter().rect_filled(swatch_rect, 2.0_f32, fill);
        ui.painter().rect_stroke(
            swatch_rect,
            2.0_f32,
            Stroke::new(1.0_f32, palette.graph_edge),
            StrokeKind::Inside,
        );
        if let Some(sym) = glyph {
            ui.painter().text(
                swatch_rect.center(),
                egui::Align2::CENTER_CENTER,
                sym,
                egui::FontId::proportional(10.0),
                contrasting_text_color(fill),
            );
        }
        ui.small(label);
    });
}

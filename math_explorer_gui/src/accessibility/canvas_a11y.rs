#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Canvas and custom visualization `AccessKit` registration, focus rings, and arrow-key focus locks.

/// Attaches `WidgetInfo` and `AccessKit` node metadata (`role`, `label`, and optional `value`)
/// to a custom canvas or visualization widget.
pub fn register_canvas_accessibility(
    ui: &eframe::egui::Ui,
    response: &eframe::egui::Response,
    role: eframe::egui::accesskit::Role,
    label: &str,
    value: Option<&str>,
) {
    let mut info =
        eframe::egui::WidgetInfo::labeled(eframe::egui::WidgetType::Other, ui.is_enabled(), label);
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
        }
    });
}

/// Paints a high-contrast focus ring inside `response.rect` when the widget holds keyboard focus.
pub fn paint_canvas_focus_ring(ui: &eframe::egui::Ui, response: &eframe::egui::Response) {
    if response.has_focus() {
        let color = crate::app::SemanticPalette::from_visuals(ui.visuals()).accent;
        ui.painter().rect_stroke(
            response.rect,
            2.0_f32,
            eframe::egui::Stroke::new(2.0_f32, color),
            eframe::egui::StrokeKind::Inside,
        );
    }
}

/// Locks horizontal and vertical arrow keys to `response.id` strictly while focused,
/// leaving `Tab` and `Escape` unlocked so keyboard traversal is never trapped.
pub fn lock_canvas_arrow_keys(ui: &eframe::egui::Ui, response: &eframe::egui::Response) {
    if response.has_focus() {
        if response.gained_focus() {
            ui.ctx().request_repaint();
        }
        ui.memory_mut(|m| {
            m.set_focus_lock_filter(
                response.id,
                eframe::egui::EventFilter {
                    tab: false,
                    horizontal_arrows: true,
                    vertical_arrows: true,
                    escape: false,
                },
            );
        });
    }
}

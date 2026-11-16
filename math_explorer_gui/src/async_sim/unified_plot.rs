#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::{SimCommand, StateSnapshot};
use eframe::egui;
use egui_plot::{Line, Plot, PlotImage, PlotPoint, PlotPoints};

pub fn draw_image_plot_helper<F>(
    name: &str,
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    texture: &egui::TextureHandle,
    mut on_brush: F,
) where
    F: FnMut(SimCommand),
{
    let width = texture.size()[0] as f32;
    let height = texture.size()[1] as f32;

    Plot::new(format!("{}_plot_img", name))
        .data_aspect(1.0)
        .view_aspect(1.0)
        .show_grid(false)
        .show_axes([false, false])
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .show(ui, |plot_ui| {
            plot_ui.image(PlotImage::new(
                format!("{}_image", name),
                texture.id(),
                PlotPoint::new(width as f64 / 2.0, height as f64 / 2.0),
                [width, height],
            ));

            if plot_ui.response().hovered()
                && ctx.input(|i| i.pointer.primary_down() || i.pointer.secondary_down())
            {
                if let Some(pos) = plot_ui.pointer_coordinate() {
                    let grid_x = pos.x.round() as i32;
                    let grid_y = pos.y.round() as i32;
                    on_brush(SimCommand::ApplyBrush {
                        cx: grid_x,
                        cy: grid_y,
                        r: 2,
                        is_obstacle: ctx.input(|i| i.pointer.primary_down()),
                    });
                }
            }
        });
}

pub fn draw_line_plot_helper(name: &str, ui: &mut egui::Ui, snapshot: &StateSnapshot) {
    if snapshot.custom_data.is_empty() {
        return;
    }
    let mut points = Vec::new();
    for i in (0..snapshot.custom_data.len()).step_by(2) {
        if i + 1 < snapshot.custom_data.len() {
            points.push([snapshot.custom_data[i], snapshot.custom_data[i + 1]]);
        }
    }
    let line = Line::new(name, PlotPoints::from_iter(points))
        .color(egui::Color32::from_rgb(100, 200, 255));

    Plot::new(format!("{}_plot_line", name))
        .view_aspect(2.0)
        .show(ui, |plot_ui| {
            plot_ui.line(line);
        });
}

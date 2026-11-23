#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Shared 3D viewport wrapper around `egui_plot::Plot` hiding 2D rulers and locking bounds.

use eframe::egui;

use super::{Camera3D, CAMERA_3D_HELPER_TEXT};

/// Shared 3D viewport wrapper around `egui_plot::Plot` that hides frozen 2D axes/grid,
/// disables 2D pan/zoom gestures, locks symmetric viewport bounds so `Camera3D::zoom`
/// never gets cancelled by auto-bounds, and delegates 3D gestures to [`Camera3D`].
pub struct Plot3DViewport<'a> {
    id_source: egui::Id,
    plot_key: String,
    title: Option<String>,
    half_extent: f64,
    helper_text: Option<&'a str>,
}

impl<'a> Plot3DViewport<'a> {
    /// Creates a 3D viewport with locked symmetric half-extent in projected plot units.
    #[must_use]
    pub fn new(id_source: impl egui_plot::PlotIdSource, half_extent: f64) -> Self {
        let (id_source, plot_key, title) = id_source.into_plot_id_key_and_title();
        Self {
            id_source,
            plot_key,
            title,
            half_extent: half_extent.max(0.1),
            helper_text: Some(CAMERA_3D_HELPER_TEXT),
        }
    }

    /// Overrides the helper caption displayed beneath the 3D canvas.
    #[must_use]
    pub fn helper_text(mut self, text: &'a str) -> Self {
        self.helper_text = Some(text);
        self
    }

    /// Renders the locked 3D plot, processes [`Camera3D`] gestures, and draws optional helper copy.
    pub fn show<'p, R>(
        self,
        ui: &mut egui::Ui,
        camera: &mut Camera3D,
        build_fn: impl FnOnce(&mut egui_plot::PlotUi<'p>) -> R,
    ) -> egui_plot::PlotResponse<R> {
        let avail = ui.available_size_before_wrap();
        let label_reserve = if self.helper_text.is_some() {
            24.0_f32
        } else {
            0.0_f32
        };
        let plot_height = (avail.y - label_reserve).max(64.0_f32);
        let aspect = if plot_height > 0.0_f32 {
            (avail.x / plot_height).max(1.0_f32) as f64
        } else {
            1.0
        };
        let x_half = self.half_extent * aspect;
        let y_half = self.half_extent;

        let mut captured_plot_id = None;
        let mut plot = egui_plot::Plot::new(self.id_source)
            .plot_key(self.plot_key)
            .height(plot_height)
            .data_aspect(1.0)
            .allow_2d_analysis(false)
            .allow_csv_import(false)
            .show_x(false)
            .show_y(false)
            .show_axes(false)
            .show_grid(false)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_axis_zoom_drag(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .allow_double_click_reset(false)
            .show_navigation_controls(false)
            .default_x_bounds(-x_half, x_half)
            .default_y_bounds(-y_half, y_half);
        if let Some(title) = self.title {
            plot = plot.title(title);
        }
        let plot_response = plot.show(ui, |plot_ui| {
            plot_ui.set_plot_bounds_x(-x_half..=x_half);
            plot_ui.set_plot_bounds_y(-y_half..=y_half);
            captured_plot_id = Some(plot_ui.plot_id());
            build_fn(plot_ui)
        });

        let rect_h = plot_response.response.rect.height().max(1.0_f32);
        camera.pan_units_per_px = ((2.0 * y_half) as f32) / rect_h;
        camera.handle_interaction_with_plot(&plot_response.response, ui, captured_plot_id);

        if let Some(text) = self.helper_text {
            ui.label(text);
        }

        plot_response
    }
}

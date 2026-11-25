#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Shared 3D viewport wrapper around `egui_plot::Plot` hiding 2D rulers and locking bounds.

use eframe::egui;

use crate::camera::{Camera3D, CAMERA_3D_HELPER_TEXT};

/// Shared 3D viewport wrapper around `egui_plot::Plot` that activates `scene_3d_mode`,
/// disables 2D pan/zoom gestures, locks symmetric viewport bounds so `Camera3D::zoom`
/// never gets cancelled by auto-bounds, and delegates 3D gestures to [`Camera3D`].
pub struct Plot3DViewport<'a> {
    id_source: egui::Id,
    plot_key: String,
    title: Option<String>,
    status_text: Option<String>,
    half_extent: f64,
    helper_text: Option<&'a str>,
    allow_camera_drag: bool,
    allow_primary_orbit: bool,
}

fn humanize_plot_key(raw: &str) -> Option<String> {
    let words: Vec<String> = raw
        .split(|c: char| !c.is_alphanumeric())
        .filter(|part| !part.is_empty() && !part.eq_ignore_ascii_case("plot"))
        .map(|part| {
            if part.eq_ignore_ascii_case("3d") {
                "3D".to_owned()
            } else {
                let mut chars = part.chars();
                match chars.next() {
                    Some(first) => {
                        let mut s = first.to_uppercase().collect::<String>();
                        s.push_str(chars.as_str());
                        s
                    }
                    None => String::new(),
                }
            }
        })
        .collect();
    (!words.is_empty()).then(|| words.join(" "))
}

fn base_3d_plot(
    id_source: egui::Id,
    plot_height: f32,
    x_half: f64,
    y_half: f64,
) -> egui_plot::Plot<'static> {
    egui_plot::Plot::new(id_source)
        .height(plot_height)
        .data_aspect(1.0)
        .scene_3d_mode(true)
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
        .show_toolbar(false)
        .show_settings_button(false)
        .default_x_bounds(-x_half, x_half)
        .default_y_bounds(-y_half, y_half)
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
            status_text: None,
            half_extent: half_extent.max(0.1),
            helper_text: Some(CAMERA_3D_HELPER_TEXT),
            allow_camera_drag: true,
            allow_primary_orbit: true,
        }
    }

    /// Sets an explicit human-readable display title on the `Role::GraphicsDocument` node.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        let raw = title.into();
        let trimmed = raw.trim();
        self.title = (!trimmed.is_empty()).then(|| trimmed.to_owned());
        self
    }

    /// Appends tool-specific telemetry or status text to the 3D orientation `AccessKit` `value`.
    #[must_use]
    pub fn status_text(mut self, status: impl Into<String>) -> Self {
        let raw = status.into();
        let trimmed = raw.trim();
        self.status_text = (!trimmed.is_empty()).then(|| trimmed.to_owned());
        self
    }

    /// Overrides the helper caption displayed beneath the 3D canvas.
    #[must_use]
    pub fn helper_text(mut self, text: &'a str) -> Self {
        self.helper_text = Some(text);
        self
    }

    /// Controls whether pointer drag and double-click gestures orbit, pan, or reset [`Camera3D`].
    #[must_use]
    pub fn allow_camera_drag(mut self, allow: bool) -> Self {
        self.allow_camera_drag = allow;
        self
    }

    /// Controls whether unmodified primary pointer drags orbit [`Camera3D`] (`true` in Orbit mode, `false` in Sculpt mode).
    #[must_use]
    pub fn allow_primary_orbit(mut self, enabled: bool) -> Self {
        self.allow_primary_orbit = enabled;
        self
    }

    /// Computes `(plot_height, x_half, y_half)` for an available container size and half-extent.
    #[must_use]
    pub fn compute_half_extents(
        avail: egui::Vec2,
        half_extent: f64,
        has_helper: bool,
    ) -> (f32, f64, f64) {
        let clamped_half = half_extent.max(0.1);
        let label_reserve = if has_helper { 24.0_f32 } else { 0.0_f32 };
        let plot_height = (avail.y - label_reserve).max(64.0_f32);
        let aspect = if plot_height > 0.0_f32 {
            (avail.x / plot_height).max(1.0_f32) as f64
        } else {
            1.0
        };
        (plot_height, clamped_half * aspect, clamped_half)
    }

    /// Renders the locked 3D plot, processes [`Camera3D`] gestures, and draws optional helper copy.
    pub fn show<'p, R>(
        self,
        ui: &mut egui::Ui,
        camera: &mut Camera3D,
        build_fn: impl FnOnce(&mut egui_plot::PlotUi<'p>) -> R,
    ) -> egui_plot::PlotResponse<R> {
        crate::accessibility::ensure_plot_live_bridge();
        let avail = ui.available_size_before_wrap();
        let (plot_height, x_half, y_half) =
            Self::compute_half_extents(avail, self.half_extent, self.helper_text.is_some());

        let resolved_title = self.title.or_else(|| humanize_plot_key(&self.plot_key));
        let status_ref = self.status_text.as_deref();
        let initial_value = camera.orientation_value_text(status_ref);

        let mut captured_plot_id = None;
        let mut plot = base_3d_plot(self.id_source, plot_height, x_half, y_half)
            .plot_key(self.plot_key)
            .a11y_value(initial_value);
        if let Some(title) = resolved_title {
            plot = plot.title(title);
        }
        let plot_response = plot.show(ui, |plot_ui| {
            plot_ui.set_plot_bounds_x(-x_half..=x_half);
            plot_ui.set_plot_bounds_y(-y_half..=y_half);
            captured_plot_id = Some(plot_ui.plot_id());
            build_fn(plot_ui)
        });

        let pre_cam = *camera;
        let rect_h = plot_response.response.rect.height().max(1.0_f32);
        camera.pan_units_per_px = ((2.0 * y_half) as f32) / rect_h;
        camera.handle_interaction_with_options(
            &plot_response.response,
            ui,
            captured_plot_id,
            self.allow_camera_drag,
            self.allow_primary_orbit,
        );
        if *camera != pre_cam {
            let updated_value = camera.orientation_value_text(status_ref);
            ui.ctx()
                .accesskit_node_builder(plot_response.response.id, |builder| {
                    builder.set_value(updated_value);
                });
        }

        if let Some(text) = self.helper_text {
            ui.label(text);
        }

        plot_response
    }
}

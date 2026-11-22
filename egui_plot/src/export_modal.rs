//! Shared high-resolution plot and animation export modal UI.

use crate::export::{
    AspectRatioOption, DatasetFrame, ExportDataset, ExportOptions, MAX_EXPORT_DIMENSION,
    copy_image_to_clipboard_with_callback, copy_text_to_clipboard_with_callback, export_custom_gif,
    export_custom_image, export_custom_png, export_custom_svg, save_file_dialog,
};
use crate::export_primitives::{
    MAX_SVG_CONTOUR_LEVELS, MIN_SVG_CONTOUR_LEVELS, TEXTURE_SIZE_PRESETS,
};

/// Consistent section header rendered inside the export modal window.
pub const MODAL_HEADER_TEXT: &str = "Offscreen High-Res Render Settings";

const DARK_BG_COLOR: egui::Color32 = egui::Color32::from_rgb(18, 18, 18);

#[cfg(not(target_arch = "wasm32"))]
const INITIAL_CLIPBOARD_STATUS: &str = "Copied plot image to clipboard!";
#[cfg(target_arch = "wasm32")]
const INITIAL_CLIPBOARD_STATUS: &str = "Copying plot image to clipboard...";

/// Persistent state for the shared high-resolution plot export modal.
///
/// # Example
///
/// ```
/// use egui_plot::ExportModalState;
///
/// let state = ExportModalState::default();
/// assert!(!state.open);
/// assert_eq!(state.options.width, 1920);
/// assert_eq!(state.options.height, 1080);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ExportModalState {
    /// Whether the export modal window is currently open.
    pub open: bool,
    /// Offscreen rendering options (dimensions, scale, thickness, opacity, background, aspect, log axes).
    pub options: ExportOptions,
    /// Whether the dark background theme is selected (`true` for dark, `false` for light).
    pub bg_dark: bool,
    /// Frames per second for animated GIF export.
    pub gif_fps: u32,
    /// Number of animation frame steps to sample for GIF export.
    pub gif_frames: u32,
    /// Status or error message from the most recent export or clipboard operation.
    pub status_message: Option<String>,
}

impl Default for ExportModalState {
    fn default() -> Self {
        Self {
            open: false,
            options: ExportOptions::default(),
            bg_dark: true,
            gif_fps: 15,
            gif_frames: 20,
            status_message: None,
        }
    }
}

impl ExportModalState {
    /// Borrow this state's fields as [`ExportModalBindings`] for [`show_export_modal`].
    #[must_use]
    pub fn bindings(&mut self) -> ExportModalBindings<'_> {
        ExportModalBindings {
            open: &mut self.open,
            options: &mut self.options,
            bg_dark: &mut self.bg_dark,
            gif_fps: &mut self.gif_fps,
            gif_frames: &mut self.gif_frames,
            status_message: &mut self.status_message,
        }
    }
}

/// Mutable bindings into export modal state fields.
#[derive(Debug)]
pub struct ExportModalBindings<'a> {
    /// Mutable flag controlling modal visibility.
    pub open: &'a mut bool,
    /// Mutable offscreen render options.
    pub options: &'a mut ExportOptions,
    /// Mutable dark/light background toggle.
    pub bg_dark: &'a mut bool,
    /// Mutable GIF frame rate setting.
    pub gif_fps: &'a mut u32,
    /// Mutable GIF frame step count setting.
    pub gif_frames: &'a mut u32,
    /// Mutable status feedback message.
    pub status_message: &'a mut Option<String>,
}

/// Caller-scoped configuration for [`show_export_modal`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExportModalConfig<'a> {
    /// Title shown in the `egui::Window` title bar.
    pub window_title: &'a str,
    /// Caller-scoped widget ID used to namespace combo boxes and async clipboard status.
    pub id: egui::Id,
    /// Default filename suggested when saving PNG images.
    pub png_filename: &'a str,
    /// Default filename suggested when saving SVG vector figures.
    pub svg_filename: &'a str,
    /// Default filename suggested when saving animated GIFs.
    pub gif_filename: &'a str,
}

/// Action triggered by clicking one of the export buttons inside [`show_export_modal`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportModalAction {
    /// Save high-resolution PNG figure to disk.
    SavePng,
    /// Save vector SVG figure to disk.
    SaveSvg,
    /// Copy vector SVG markup to the system clipboard.
    CopySvg,
    /// Save multi-frame animated GIF to disk.
    SaveGif,
    /// Copy high-resolution PNG figure to the system clipboard.
    CopyClipboard,
    /// Save borderless 16-bit grayscale heightmap PNG to disk.
    SaveHeightmap16Png,
    /// Save borderless tangent-space normal-map PNG to disk.
    SaveNormalMapPng,
}

/// Mutable bindings for 2D scalar-field contour and borderless 16-bit texture controls.
#[derive(Debug)]
pub struct SpatialExportBindings<'a> {
    /// Number of isoline contour levels (`1..=16`) for SVG contour extraction.
    pub contour_levels: &'a mut usize,
    /// Borderless power-of-two square size (`256`, `512`, `1024`, `2048`) for texture exports.
    pub texture_size: &'a mut u32,
    /// Surface gradient strength multiplier for normal-map generation.
    pub normal_strength: &'a mut f32,
    /// Whether contour threshold count controls should be displayed.
    pub show_contour_controls: bool,
    /// Whether 16-bit heightmap and normal-map controls and actions should be displayed.
    pub show_texture_controls: bool,
}

/// Provider interface supplying plot datasets and sampled GIF frames on demand.
pub trait ExportDataSource {
    /// Collect current 2D plot datasets for static PNG, SVG, or clipboard export.
    fn export_datasets(&mut self) -> Vec<ExportDataset>;

    /// Sample multi-frame datasets for animated GIF export.
    fn sample_gif_frames(&mut self, num_frames: usize) -> Vec<DatasetFrame>;

    /// Returns `(enabled, disabled_hover_reason)` for figure export buttons.
    fn figure_capability(&self) -> (bool, &'static str) {
        (true, "")
    }

    /// Optional status label color override (defaults to `Color32::GREEN` when `None`).
    fn status_color(&self) -> Option<egui::Color32> {
        None
    }

    /// Optional hook to intercept modal button clicks (e.g. to queue deferred end-of-frame exports).
    fn intercept_action(&mut self, _action: ExportModalAction) -> bool {
        false
    }

    /// Optional mutable bindings enabling 2D contour and borderless 16-bit texture controls.
    fn spatial_bindings(&mut self) -> Option<SpatialExportBindings<'_>> {
        None
    }
}

/// Closure-backed [`ExportDataSource`] pairing a dataset provider and a frame sampler.
#[derive(Clone, Copy, Debug)]
pub struct CallbackExportSource<D, F> {
    /// Callback invoked to retrieve static plot datasets.
    pub datasets_fn: D,
    /// Callback invoked with `num_frames` to sample animation frames for GIF export.
    pub sample_frames_fn: F,
}

impl<D, F> ExportDataSource for CallbackExportSource<D, F>
where
    D: FnMut() -> Vec<ExportDataset>,
    F: FnMut(usize) -> Vec<DatasetFrame>,
{
    fn export_datasets(&mut self) -> Vec<ExportDataset> {
        (self.datasets_fn)()
    }

    fn sample_gif_frames(&mut self, num_frames: usize) -> Vec<DatasetFrame> {
        (self.sample_frames_fn)(num_frames)
    }
}

/// Construct the temporary storage ID used for async clipboard completion messages.
#[must_use]
pub fn clipboard_status_id(scope_id: egui::Id) -> egui::Id {
    scope_id.with("async_clipboard_status")
}

/// Poll and drain any pending async clipboard status message for `scope_id`.
pub fn poll_clipboard_status(
    ctx: &egui::Context,
    scope_id: egui::Id,
    status_message: &mut Option<String>,
) -> bool {
    let status_id = clipboard_status_id(scope_id);
    if let Some(msg) = ctx.data_mut(|d| d.remove_temp::<String>(status_id)) {
        *status_message = Some(msg);
        true
    } else {
        false
    }
}

/// Calculate target output pixel `(width, height)` from base dimensions, scale, and aspect ratio.
///
/// # Example
///
/// ```
/// use egui_plot::{AspectRatioOption, ExportOptions, calculate_target_dimensions};
///
/// let opts = ExportOptions {
///     width: 1920,
///     height: 1080,
///     scale: 2.0,
///     aspect_ratio: AspectRatioOption::SixteenNine,
///     ..Default::default()
/// };
/// assert_eq!(calculate_target_dimensions(&opts), (3840, 2160));
/// ```
#[must_use]
pub fn calculate_target_dimensions(options: &ExportOptions) -> (u32, u32) {
    let base_w = f64::from(options.width.max(1));
    let base_h = f64::from(options.height.max(1));
    let scale = if options.scale.is_finite() && options.scale > 0.0 {
        f64::from(options.scale)
    } else {
        1.0
    };
    let calc_w = ((base_w * scale).round() as u32).max(1);
    let mut calc_h = ((base_h * scale).round() as u32).max(1);
    if let Some(ratio) = options
        .aspect_ratio
        .ratio()
        .filter(|&r| r.is_finite() && r > 0.0)
    {
        calc_h = ((f64::from(calc_w) / f64::from(ratio)).round() as u32).max(1);
    }
    (calc_w, calc_h)
}

/// Return `true` when either calculated dimension exceeds [`MAX_EXPORT_DIMENSION`].
#[must_use]
pub fn exceeds_max_dimension(width: u32, height: u32) -> bool {
    width > MAX_EXPORT_DIMENSION || height > MAX_EXPORT_DIMENSION
}

fn aspect_ratio_label(aspect: AspectRatioOption) -> &'static str {
    match aspect {
        AspectRatioOption::Free => "Custom / Free",
        AspectRatioOption::SixteenNine => "16:9 (Widescreen)",
        AspectRatioOption::FourThree => "4:3 (Standard)",
        AspectRatioOption::OneOne => "1:1 (Square)",
    }
}

fn render_aspect_ratio_combo(
    ui: &mut egui::Ui,
    scope_id: egui::Id,
    aspect: &mut AspectRatioOption,
) {
    ui.horizontal(|ui| {
        ui.label("Aspect Ratio:");
        egui::ComboBox::from_id_salt(scope_id.with("export_aspect"))
            .selected_text(aspect_ratio_label(*aspect))
            .show_ui(ui, |ui| {
                for candidate in [
                    AspectRatioOption::SixteenNine,
                    AspectRatioOption::FourThree,
                    AspectRatioOption::OneOne,
                    AspectRatioOption::Free,
                ] {
                    ui.selectable_value(aspect, candidate, aspect_ratio_label(candidate));
                }
            });
    });
}

fn render_modal_settings(
    ui: &mut egui::Ui,
    scope_id: egui::Id,
    bindings: &mut ExportModalBindings<'_>,
) {
    ui.horizontal(|ui| {
        ui.label("Resolution Scale:");
        ui.selectable_value(&mut bindings.options.scale, 1.0, "1x (1080p)");
        ui.selectable_value(&mut bindings.options.scale, 2.0, "2x (4K)");
        ui.selectable_value(&mut bindings.options.scale, 4.0, "4x");
    });

    render_aspect_ratio_combo(ui, scope_id, &mut bindings.options.aspect_ratio);

    ui.horizontal(|ui| {
        ui.label("Line Thickness:");
        ui.add(egui::Slider::new(
            &mut bindings.options.line_thickness,
            0.5..=10.0,
        ));
    });

    ui.horizontal(|ui| {
        ui.label("Background Opacity:");
        ui.add(egui::Slider::new(
            &mut bindings.options.background_opacity,
            0.0..=1.0,
        ));
    });

    ui.horizontal(|ui| {
        ui.label("Background Color:");
        ui.selectable_value(bindings.bg_dark, true, "Dark");
        ui.selectable_value(bindings.bg_dark, false, "Light");
    });

    bindings.options.background_color = if *bindings.bg_dark {
        DARK_BG_COLOR
    } else {
        egui::Color32::WHITE
    };

    ui.horizontal(|ui| {
        ui.label("GIF FPS:");
        ui.add(egui::Slider::new(bindings.gif_fps, 1..=60));
    });

    ui.horizontal(|ui| {
        ui.label("GIF Frame Steps:");
        ui.add(egui::Slider::new(bindings.gif_frames, 5..=60));
    });
}

fn render_spatial_settings(ui: &mut egui::Ui, spatial: &mut SpatialExportBindings<'_>) {
    if !spatial.show_contour_controls && !spatial.show_texture_controls {
        return;
    }
    ui.separator();
    ui.label(egui::RichText::new("2D Contours & 16-Bit Material Maps").strong());
    if spatial.show_contour_controls {
        ui.horizontal(|ui| {
            ui.label("Contour Thresholds:");
            ui.add(egui::Slider::new(
                spatial.contour_levels,
                MIN_SVG_CONTOUR_LEVELS..=MAX_SVG_CONTOUR_LEVELS,
            ));
        });
    }
    if spatial.show_texture_controls {
        ui.horizontal(|ui| {
            ui.label("Borderless Size:");
            for preset in TEXTURE_SIZE_PRESETS {
                ui.selectable_value(spatial.texture_size, preset, format!("{preset}x{preset}"));
            }
        });
        ui.horizontal(|ui| {
            ui.label("Normal Strength:");
            ui.add(egui::Slider::new(spatial.normal_strength, 0.1..=16.0));
        });
    }
}

fn render_dimension_and_status(
    ui: &mut egui::Ui,
    options: &ExportOptions,
    status_message: &Option<String>,
    status_color: Option<egui::Color32>,
) {
    let (warn_color, ok_color) = if ui.visuals().dark_mode {
        (
            egui::Color32::from_rgb(245, 194, 66),
            egui::Color32::from_rgb(74, 214, 122),
        )
    } else {
        (
            egui::Color32::from_rgb(146, 84, 0),
            egui::Color32::from_rgb(15, 115, 54),
        )
    };
    let (calc_w, calc_h) = calculate_target_dimensions(options);
    if exceeds_max_dimension(calc_w, calc_h) {
        ui.label(
            egui::RichText::new(format!(
                "⚠️ Capped at max dimension {}x{} px",
                MAX_EXPORT_DIMENSION, MAX_EXPORT_DIMENSION
            ))
            .color(warn_color),
        );
    } else {
        ui.label(format!("Target Canvas Size: {} x {} px", calc_w, calc_h));
    }

    if let Some(msg) = status_message {
        ui.label(egui::RichText::new(msg).color(status_color.unwrap_or(ok_color)));
    }
}

fn handle_save_png(
    options: &ExportOptions,
    filename: &str,
    source: &mut impl ExportDataSource,
    status_message: &mut Option<String>,
) {
    let datasets = source.export_datasets();
    match export_custom_png(&datasets, options) {
        Ok(png_bytes) => {
            save_file_dialog(filename, &png_bytes, "PNG Image", &["png"]);
            *status_message = Some("PNG image saved successfully.".to_string());
        }
        Err(e) => {
            *status_message = Some(format!("Export failed: {e}"));
        }
    }
}

fn handle_save_svg(
    options: &ExportOptions,
    filename: &str,
    source: &mut impl ExportDataSource,
    status_message: &mut Option<String>,
) {
    let datasets = source.export_datasets();
    match export_custom_svg(&datasets, options) {
        Ok(svg_text) => {
            save_file_dialog(filename, svg_text.as_bytes(), "SVG Vector Image", &["svg"]);
            *status_message = Some("SVG image saved successfully.".to_string());
        }
        Err(e) => {
            *status_message = Some(format!("Export failed: {e}"));
        }
    }
}

fn handle_save_gif(
    options: &ExportOptions,
    gif_fps: u32,
    gif_frames: u32,
    filename: &str,
    source: &mut impl ExportDataSource,
    status_message: &mut Option<String>,
) {
    let sampled_frames = source.sample_gif_frames(gif_frames as usize);
    match export_custom_gif(&sampled_frames, options, gif_fps) {
        Ok(gif_bytes) => {
            save_file_dialog(filename, &gif_bytes, "GIF Animation", &["gif"]);
            *status_message = Some("GIF animation saved successfully.".to_string());
        }
        Err(e) => {
            *status_message = Some(format!("GIF export failed: {e}"));
        }
    }
}

fn handle_copy_clipboard(
    ctx: &egui::Context,
    scope_id: egui::Id,
    options: &ExportOptions,
    source: &mut impl ExportDataSource,
    status_message: &mut Option<String>,
) {
    let datasets = source.export_datasets();
    match export_custom_image(&datasets, options) {
        Ok(color_img) => {
            let cb_ctx = ctx.clone();
            let status_id = clipboard_status_id(scope_id);
            match copy_image_to_clipboard_with_callback(&color_img, move |res| {
                cb_ctx.data_mut(|d| {
                    let msg = match res {
                        Ok(_) => "Copied plot image to clipboard!".to_string(),
                        Err(e) => format!("Clipboard copy failed: {e}"),
                    };
                    d.insert_temp(status_id, msg);
                });
                cb_ctx.request_repaint();
            }) {
                Ok(_) => {
                    *status_message = Some(INITIAL_CLIPBOARD_STATUS.to_string());
                }
                Err(e) => {
                    *status_message = Some(format!("Clipboard copy failed: {e}"));
                }
            }
        }
        Err(e) => {
            *status_message = Some(format!("Render failed: {e}"));
        }
    }
}

fn handle_copy_svg(
    ctx: &egui::Context,
    scope_id: egui::Id,
    options: &ExportOptions,
    source: &mut impl ExportDataSource,
    status_message: &mut Option<String>,
) {
    let datasets = source.export_datasets();
    match export_custom_svg(&datasets, options) {
        Ok(svg_text) => {
            let cb_ctx = ctx.clone();
            let status_id = clipboard_status_id(scope_id);
            match copy_text_to_clipboard_with_callback(&svg_text, move |res| {
                cb_ctx.data_mut(|d| {
                    let msg = match res {
                        Ok(()) => "Copied SVG markup to clipboard!".to_string(),
                        Err(e) => format!("SVG clipboard copy failed: {e}"),
                    };
                    d.insert_temp(status_id, msg);
                });
                cb_ctx.request_repaint();
            }) {
                Ok(()) => {
                    *status_message = Some("Copied SVG markup to clipboard!".to_string());
                }
                Err(e) => {
                    *status_message = Some(format!("SVG clipboard copy failed: {e}"));
                }
            }
        }
        Err(e) => {
            *status_message = Some(format!("SVG render failed: {e}"));
        }
    }
}

fn modal_fig_button(ui: &mut egui::Ui, enabled: bool, label: &str, disabled_reason: &str) -> bool {
    use crate::a11y::AccessibleTooltipExt as _;

    let mut resp =
        crate::a11y::focusable_action_button(ui, enabled, egui::Button::new(label), Some(label));
    if !enabled && !disabled_reason.is_empty() {
        resp = resp.accessible_tooltip(disabled_reason);
    }
    enabled && resp.clicked()
}

fn render_texture_action_row(
    ui: &mut egui::Ui,
    enabled: bool,
    reason: &str,
    source: &mut impl ExportDataSource,
) {
    let has_textures = source
        .spatial_bindings()
        .is_some_and(|s| s.show_texture_controls);
    if !has_textures {
        return;
    }
    ui.horizontal(|ui| {
        if modal_fig_button(ui, enabled, "🗻 Save 16-bit Heightmap", reason) {
            let _ = source.intercept_action(ExportModalAction::SaveHeightmap16Png);
        }
        if modal_fig_button(ui, enabled, "🧭 Save Normal Map", reason) {
            let _ = source.intercept_action(ExportModalAction::SaveNormalMapPng);
        }
    });
}

fn render_action_buttons(
    ui: &mut egui::Ui,
    bindings: &mut ExportModalBindings<'_>,
    config: &ExportModalConfig<'_>,
    source: &mut impl ExportDataSource,
) -> bool {
    let mut close_requested = false;
    let (enabled, reason) = source.figure_capability();
    ui.horizontal_wrapped(|ui| {
        if modal_fig_button(ui, enabled, "💾 Save PNG", reason)
            && !source.intercept_action(ExportModalAction::SavePng)
        {
            handle_save_png(
                bindings.options,
                config.png_filename,
                source,
                bindings.status_message,
            );
        }
        if modal_fig_button(ui, enabled, "💾 Save SVG", reason)
            && !source.intercept_action(ExportModalAction::SaveSvg)
        {
            handle_save_svg(
                bindings.options,
                config.svg_filename,
                source,
                bindings.status_message,
            );
        }
        if modal_fig_button(ui, enabled, "📋 Copy SVG", reason)
            && !source.intercept_action(ExportModalAction::CopySvg)
        {
            handle_copy_svg(
                ui.ctx(),
                config.id,
                bindings.options,
                source,
                bindings.status_message,
            );
        }
        if modal_fig_button(ui, enabled, "🎬 Save GIF", reason)
            && !source.intercept_action(ExportModalAction::SaveGif)
        {
            handle_save_gif(
                bindings.options,
                *bindings.gif_fps,
                *bindings.gif_frames,
                config.gif_filename,
                source,
                bindings.status_message,
            );
        }
        if modal_fig_button(ui, enabled, "📋 Copy to Clipboard", reason)
            && !source.intercept_action(ExportModalAction::CopyClipboard)
        {
            handle_copy_clipboard(
                ui.ctx(),
                config.id,
                bindings.options,
                source,
                bindings.status_message,
            );
        }
        if ui.button("Close").clicked() {
            close_requested = true;
        }
    });
    render_texture_action_row(ui, enabled, reason, source);
    close_requested
}

/// Poll async clipboard status for `config.id` and render the shared high-res export modal when open.
pub fn show_export_modal(
    ctx: &egui::Context,
    mut bindings: ExportModalBindings<'_>,
    config: &ExportModalConfig<'_>,
    source: &mut impl ExportDataSource,
) {
    poll_clipboard_status(ctx, config.id, bindings.status_message);
    let status_color = source.status_color();
    let modal_id = config.id.with("export_modal_window");
    let mut modal_open = *bindings.open;
    crate::modal_lifecycle::ModalLifecycle::new(modal_id)
        .title(config.window_title)
        .default_width(420.0)
        .show(ctx, &mut modal_open, |ui, frame_ctx| {
            ui.label(egui::RichText::new(MODAL_HEADER_TEXT).strong());
            ui.separator();
            render_modal_settings(ui, config.id, &mut bindings);
            if let Some(mut spatial) = source.spatial_bindings() {
                render_spatial_settings(ui, &mut spatial);
            }
            ui.separator();
            render_dimension_and_status(
                ui,
                bindings.options,
                bindings.status_message,
                status_color,
            );
            ui.separator();
            if render_action_buttons(ui, &mut bindings, config, source) {
                frame_ctx.close();
            }
        });
    *bindings.open = modal_open;
}

#[cfg(test)]
#[path = "export_modal_tests.rs"]
mod tests;

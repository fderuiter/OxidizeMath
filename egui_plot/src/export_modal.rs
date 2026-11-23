//! Shared high-resolution plot and animation export modal UI.

#[path = "export_modal_gif.rs"]
mod export_modal_gif;

pub use export_modal_gif::ModalGifJobHandle;

use crate::export::{
    AspectRatioOption, DatasetFrame, ExportDataset, ExportOptions, MAX_EXPORT_DIMENSION,
    copy_image_to_clipboard_with_callback, export_custom_image, export_custom_png,
    export_custom_svg, save_file_dialog,
};

const MODAL_HEADER_TEXT: &str = "Offscreen High-Res Render Settings";

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
    /// Active standalone GIF export job handle (skipped during serialization).
    #[cfg_attr(feature = "serde", serde(skip, default))]
    pub gif_job: ModalGifJobHandle,
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
            gif_job: ModalGifJobHandle::default(),
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
            gif_job: Some(&mut self.gif_job),
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
    /// Optional mutable handle to a standalone GIF export job.
    pub gif_job: Option<&'a mut ModalGifJobHandle>,
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
    /// Save multi-frame animated GIF to disk.
    SaveGif,
    /// Copy high-resolution PNG figure to the system clipboard.
    CopyClipboard,
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

    /// Optional `(completed_frames, total_frames)` progress reported by an external GIF job manager.
    fn active_gif_progress(&self) -> Option<(usize, usize)> {
        None
    }

    /// Optional hook to cancel an external GIF export job, returning `true` if a job was canceled.
    fn cancel_gif_export(&mut self) -> bool {
        false
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

fn poll_clipboard_status(
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

#[must_use]
fn calculate_target_dimensions(options: &ExportOptions) -> (u32, u32) {
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

#[must_use]
fn exceeds_max_dimension(width: u32, height: u32) -> bool {
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
        ui.add(
            egui::Slider::new(bindings.gif_frames, 5..=60).clamping(egui::SliderClamping::Edits),
        );
    });
}

fn render_dimension_and_status(
    ui: &mut egui::Ui,
    options: &ExportOptions,
    status_message: &Option<String>,
    status_color: Option<egui::Color32>,
    gif_progress: Option<(usize, usize)>,
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

    if let Some((completed, total)) = gif_progress {
        let denom = total.max(1) as f32;
        let fraction = (completed as f32 / denom).clamp(0.0, 1.0);
        ui.add(
            egui::ProgressBar::new(fraction)
                .text(format!("GIF Progress: {completed} / {total} frames")),
        );
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

fn start_standalone_gif_job(
    ctx: &egui::Context,
    scope_id: egui::Id,
    bindings: &mut ExportModalBindings<'_>,
    source: &mut impl ExportDataSource,
) {
    let sampled_frames = source.sample_gif_frames((*bindings.gif_frames).max(1) as usize);
    if let Some(job) = bindings.gif_job.as_deref_mut() {
        job.start(
            ctx,
            sampled_frames,
            bindings.options,
            *bindings.gif_fps,
            bindings.status_message,
        );
    } else {
        let fallback_id = scope_id.with("standalone_modal_gif_job");
        let mut handle = ctx
            .data_mut(|d| d.get_temp::<ModalGifJobHandle>(fallback_id))
            .unwrap_or_default();
        handle.start(
            ctx,
            sampled_frames,
            bindings.options,
            *bindings.gif_fps,
            bindings.status_message,
        );
        if handle.is_active() {
            ctx.data_mut(|d| d.insert_temp(fallback_id, handle));
        }
    }
}

fn current_modal_gif_progress(
    ctx: &egui::Context,
    scope_id: egui::Id,
    bindings: &ExportModalBindings<'_>,
    source: &impl ExportDataSource,
) -> Option<(usize, usize)> {
    if let Some(ext) = source.active_gif_progress() {
        return Some(ext);
    }
    if let Some(job) = bindings.gif_job.as_deref() {
        return job.progress();
    }
    let fallback_id = scope_id.with("standalone_modal_gif_job");
    ctx.data(|d| d.get_temp::<ModalGifJobHandle>(fallback_id))
        .and_then(|h| h.progress())
}

fn cancel_modal_gif_job(
    ctx: &egui::Context,
    scope_id: egui::Id,
    bindings: &mut ExportModalBindings<'_>,
    source: &mut impl ExportDataSource,
) {
    if source.cancel_gif_export() {
        ctx.request_repaint();
        return;
    }
    let canceled = if let Some(job) = bindings.gif_job.as_deref_mut() {
        job.cancel()
    } else {
        let fallback_id = scope_id.with("standalone_modal_gif_job");
        ctx.data_mut(|d| d.remove_temp::<ModalGifJobHandle>(fallback_id))
            .is_some_and(|mut h| h.cancel())
    };
    if canceled {
        let msg = "GIF export canceled.".to_string();
        crate::emit_plot_live_message(ctx, msg.clone(), false);
        *bindings.status_message = Some(msg);
        ctx.request_repaint();
    }
}

fn poll_standalone_gif_job(
    ctx: &egui::Context,
    scope_id: egui::Id,
    filename: &str,
    bindings: &mut ExportModalBindings<'_>,
) {
    if let Some(job) = bindings.gif_job.as_deref_mut() {
        job.poll(ctx, filename, bindings.status_message);
        return;
    }
    let fallback_id = scope_id.with("standalone_modal_gif_job");
    let Some(mut handle) = ctx.data_mut(|d| d.remove_temp::<ModalGifJobHandle>(fallback_id)) else {
        return;
    };
    handle.poll(ctx, filename, bindings.status_message);
    if handle.is_active() {
        ctx.data_mut(|d| d.insert_temp(fallback_id, handle));
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

fn modal_fig_button(ui: &mut egui::Ui, enabled: bool, label: &str, disabled_reason: &str) -> bool {
    use crate::a11y::AccessibleTooltipExt as _;

    let mut resp =
        crate::a11y::focusable_action_button(ui, enabled, egui::Button::new(label), Some(label));
    if !enabled && !disabled_reason.is_empty() {
        resp = resp.accessible_tooltip(disabled_reason);
    }
    enabled && resp.clicked()
}

fn render_gif_and_clipboard_buttons(
    ui: &mut egui::Ui,
    gif_active: bool,
    bindings: &mut ExportModalBindings<'_>,
    config: &ExportModalConfig<'_>,
    source: &mut impl ExportDataSource,
) {
    let (enabled, reason) = source.figure_capability();
    let gif_enabled = enabled && !gif_active;
    let gif_reason = if !enabled {
        reason
    } else {
        "GIF export already in progress"
    };
    if modal_fig_button(ui, gif_enabled, "🎬 Save GIF", gif_reason)
        && !gif_active
        && !source.intercept_action(ExportModalAction::SaveGif)
    {
        start_standalone_gif_job(ui.ctx(), config.id, bindings, source);
    }
    if gif_active && ui.button("⏹ Cancel GIF").clicked() {
        cancel_modal_gif_job(ui.ctx(), config.id, bindings, source);
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
}

fn render_action_buttons(
    ui: &mut egui::Ui,
    gif_active: bool,
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
        render_gif_and_clipboard_buttons(ui, gif_active, bindings, config, source);
        if ui.button("Close").clicked() {
            close_requested = true;
        }
    });
    close_requested
}

/// Poll async clipboard and GIF status for `config.id` and render the shared high-res export modal when open.
pub fn show_export_modal(
    ctx: &egui::Context,
    mut bindings: ExportModalBindings<'_>,
    config: &ExportModalConfig<'_>,
    source: &mut impl ExportDataSource,
) {
    poll_clipboard_status(ctx, config.id, bindings.status_message);
    poll_standalone_gif_job(ctx, config.id, config.gif_filename, &mut bindings);
    let status_color = source.status_color();
    let modal_id = config.id.with("export_modal_window");
    let mut modal_open = *bindings.open;
    crate::modal_lifecycle::ModalLifecycle::new(modal_id)
        .title(config.window_title)
        .default_width(380.0)
        .show(ctx, &mut modal_open, |ui, frame_ctx| {
            ui.label(egui::RichText::new(MODAL_HEADER_TEXT).strong());
            ui.separator();
            render_modal_settings(ui, config.id, &mut bindings);
            ui.separator();
            let gif_progress = current_modal_gif_progress(ctx, config.id, &bindings, source);
            render_dimension_and_status(
                ui,
                bindings.options,
                bindings.status_message,
                status_color,
                gif_progress,
            );
            ui.separator();
            if render_action_buttons(ui, gif_progress.is_some(), &mut bindings, config, source) {
                frame_ctx.close();
            }
        });
    *bindings.open = modal_open;
}

#[cfg(test)]
#[path = "export_modal_tests.rs"]
mod tests;

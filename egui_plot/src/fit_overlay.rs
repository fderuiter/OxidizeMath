//! Hoverable fitted curve & residual stem series injection, badge rendering, and Fit Curve menus.

use std::sync::Arc;

use egui::{Color32, Rect, RichText, Stroke, Ui, vec2};

use crate::{
    AccessState, SharedPoints,
    a11y::{self, AccessibleTooltipExt as _},
    fit::{FitResult, FitType},
    items::{Line, LineStyle, MarkerShape, PlotItem, Points},
    runtime_cache::{CachedItemSeries, PlotRuntimeCache},
    transform::PlotTransform,
};

/// Computes or reuses the memoized active regression fit (if enabled) and appends hoverable
/// fitted curve and optional vertical residual stem series directly into `items`.
///
/// When `access_state.fit_model` is `None`, skips regression fitting and curve sampling.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_and_inject_fit_items<'a>(
    ui: &Ui,
    items: &mut Vec<Box<dyn PlotItem + 'a>>,
    base_series: &[CachedItemSeries],
    runtime_cache: &mut PlotRuntimeCache,
    access_state: &mut AccessState,
    transform: &PlotTransform,
    log_x: bool,
    log_y: bool,
) -> Option<(Arc<FitResult>, Color32)> {
    let valid_first = base_series
        .iter()
        .enumerate()
        .find(|(_, entry)| entry.points.len() >= 2)
        .map(|(idx, _)| idx);
    if let Some(first_idx) = valid_first
        && base_series
            .get(access_state.fit_dataset_idx)
            .is_none_or(|entry| entry.points.len() < 2)
    {
        access_state.fit_dataset_idx = first_idx;
    }

    let Some(fit_type) = access_state.fit_model else {
        runtime_cache.clear_fit();
        return None;
    };
    let target = base_series.get(access_state.fit_dataset_idx)?;
    let (fit_opt, curve_pts) = runtime_cache.resolve_fit_and_curve(
        &target.points,
        &target.name,
        fit_type,
        log_x,
        log_y,
        access_state.fit_log_mode,
        access_state.show_fit_overlay,
        transform,
    );
    let fit_res = fit_opt?;
    let accent = if target.color == Color32::TRANSPARENT {
        ui.visuals().widgets.active.bg_fill
    } else {
        target.color
    };

    if access_state.show_fit_overlay {
        let disp_name = display_dataset_name(&target.name, access_state.fit_dataset_idx);
        if curve_pts.len() >= 2 {
            let line_name = format!("Fit ({}): {disp_name}", fit_res.fit_type.short_label());
            let line = Line::new(line_name, curve_pts)
                .color(accent)
                .width(2.5_f32)
                .style(LineStyle::Dashed { length: 8.0 });
            items.push(Box::new(line));
        }
        if access_state.show_fit_residuals && !fit_res.residuals.is_empty() {
            let res_color = residual_stem_color(ui, accent);
            let stems = Points::new(format!("{disp_name} Residuals"), fit_res.residuals.clone())
                .color(res_color)
                .radius(3.2_f32)
                .shape(MarkerShape::Circle)
                .stems(0.0_f32);
            items.push(Box::new(stems));
        }
    }
    Some((fit_res, accent))
}

fn residual_stem_color(ui: &Ui, accent: Color32) -> Color32 {
    if ui.visuals().dark_mode {
        Color32::from_rgb(
            accent.r().saturating_add(45),
            accent.g().saturating_add(25),
            210,
        )
    } else {
        Color32::from_rgb(185, 68, 42)
    }
}

fn display_dataset_name(raw: &str, idx: usize) -> String {
    if raw.trim().is_empty() {
        format!("Dataset {}", idx + 1)
    } else {
        raw.to_owned()
    }
}

/// Paints the floating equation + `R²` + `RMSE` summary badge when a curve fit overlay is active.
pub(crate) fn paint_fit_formula_badge(
    ui: &mut Ui,
    plot_rect: Rect,
    fit_res: &FitResult,
    dataset_idx: usize,
    accent_color: Color32,
) {
    let badge_pos = plot_rect.min + vec2(12.0_f32, 36.0_f32);
    let badge_rect = Rect::from_min_size(badge_pos, vec2(276.0_f32, 74.0_f32));
    ui.scope_builder(egui::UiBuilder::new().max_rect(badge_rect), |ui| {
        egui::Frame::window(ui.style())
            .fill(ui.visuals().window_fill.gamma_multiply(0.88_f32))
            .stroke(Stroke::new(1.5_f32, accent_color))
            .corner_radius(6.0_f32)
            .inner_margin(6.0_f32)
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    let name_display = display_dataset_name(&fit_res.dataset_name, dataset_idx);
                    let mode_tag = if fit_res.log_mode { " [Log Mode]" } else { "" };
                    ui.label(
                        RichText::new(format!("Fit: {name_display}{mode_tag}"))
                            .strong()
                            .color(accent_color),
                    );
                    ui.label(RichText::new(&fit_res.formula).monospace().size(12.5_f32));
                    let stats = format!(
                        "R² = {:.4}   RMSE = {}",
                        fit_res.r_squared,
                        format_metric_compact(fit_res.rmse)
                    );
                    ui.label(
                        RichText::new(stats)
                            .small()
                            .color(ui.visuals().text_color()),
                    );
                });
            });
    });
}

fn format_metric_compact(v: f64) -> String {
    if !v.is_finite() {
        "N/A".to_owned()
    } else if v == 0.0 {
        "0.0000".to_owned()
    } else if !(1e-3..1e5).contains(&v.abs()) {
        format!("{v:.3e}")
    } else {
        format!("{v:.4}")
    }
}

/// Renders the shared `Fit Curve` menu items for both the top toolbar and context menu.
pub(crate) fn render_fit_curve_menu(
    ui: &mut Ui,
    base_datasets: &[(String, SharedPoints, Color32)],
    access_state: &mut AccessState,
    in_context_menu: bool,
) {
    let valid: Vec<(usize, &str, usize)> = base_datasets
        .iter()
        .enumerate()
        .filter(|(_, (_, pts, _))| pts.len() >= 2)
        .map(|(i, (name, pts, _))| (i, name.as_str(), pts.len()))
        .collect();
    if valid.is_empty() {
        let reason = "Requires a dataset with at least 2 points for curve fitting";
        for preset in FitType::MENU_PRESETS {
            let lbl = preset.label();
            let _ = a11y::focusable_action_button(ui, false, egui::Button::new(lbl), Some(lbl))
                .accessible_tooltip(reason);
        }
        return;
    }
    if let Some(&(first_idx, _, _)) = valid.first()
        && !valid
            .iter()
            .any(|&(idx, _, _)| idx == access_state.fit_dataset_idx)
    {
        access_state.fit_dataset_idx = first_idx;
    }
    render_dataset_selector(ui, &valid, access_state);
    ui.label(RichText::new("Fit Model:").small().weak());
    for preset in FitType::MENU_PRESETS {
        let selected = access_state.fit_model == Some(preset);
        if ui.selectable_label(selected, preset.label()).clicked() {
            access_state.fit_model = Some(preset);
            access_state.show_fit_overlay = true;
            ui.close();
        }
    }
    ui.separator();
    ui.checkbox(&mut access_state.show_fit_residuals, "Show Residuals");
    let log_supported = access_state
        .fit_model
        .is_none_or(FitType::supports_log_mode);
    ui.add_enabled_ui(log_supported, |ui| {
        ui.checkbox(&mut access_state.fit_log_mode, "Log Mode (Transformed)")
            .on_hover_text("Fit Exponential or Power Law via linearized logarithmic least squares");
    });
    if in_context_menu && access_state.fit_model.is_some() {
        ui.separator();
        if ui.button("Clear Fit Overlay").clicked() {
            access_state.fit_model = None;
            ui.close();
        }
    }
}

fn render_dataset_selector(
    ui: &mut Ui,
    valid: &[(usize, &str, usize)],
    access_state: &mut AccessState,
) {
    if valid.len() <= 1 {
        return;
    }
    ui.label(RichText::new("Select Dataset:").small().weak());
    for &(idx, name, len) in valid {
        let label = display_dataset_name(name, idx);
        if ui
            .selectable_label(
                access_state.fit_dataset_idx == idx,
                format!("{label} ({len} pts)"),
            )
            .clicked()
        {
            access_state.fit_dataset_idx = idx;
        }
    }
    ui.separator();
}

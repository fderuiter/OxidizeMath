//! Hoverable fitted curve & residual stem series injection, badge rendering, and Fit Curve menus.

use std::hash::{Hash, Hasher};

use egui::{Color32, Rect, RichText, Stroke, Ui, vec2};

use crate::{
    AccessState,
    a11y::{self, AccessibleTooltipExt as _},
    fit::{FitResult, FitType, fit_model_with_axes},
    items::{Line, LineStyle, MarkerShape, PlotItem, Points},
    transform::PlotTransform,
};

pub(crate) type BaseDatasetTuple = (String, Vec<[f64; 2]>, Color32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FitCacheKey {
    pub(crate) fit_type: FitType,
    pub(crate) dataset_idx: usize,
    pub(crate) log_x: bool,
    pub(crate) log_y: bool,
    pub(crate) log_mode: bool,
    pub(crate) dataset_hash: u64,
}

fn hash_fit_dataset(name: &str, points: &[[f64; 2]]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    points.len().hash(&mut hasher);
    for pt in points {
        pt[0].to_bits().hash(&mut hasher);
        pt[1].to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Computes the active regression fit (if enabled) and appends hoverable fitted curve and
/// optional vertical residual stem series directly into `items` prior to dataset extraction.
pub(crate) fn compute_and_inject_fit_items<'a>(
    ui: &Ui,
    items: &mut Vec<Box<dyn PlotItem + 'a>>,
    base_datasets: &[BaseDatasetTuple],
    access_state: &mut AccessState,
    transform: &PlotTransform,
    log_x: bool,
    log_y: bool,
) -> Option<(FitResult, Color32)> {
    let valid_first = base_datasets
        .iter()
        .enumerate()
        .find(|(_, (_, pts, _))| pts.len() >= 2)
        .map(|(idx, _)| idx);
    if let Some(first_idx) = valid_first
        && base_datasets
            .get(access_state.fit_dataset_idx)
            .is_none_or(|(_, pts, _)| pts.len() < 2)
    {
        access_state.fit_dataset_idx = first_idx;
    }

    let fit_type = access_state.fit_model?;
    if !access_state.show_fit_overlay {
        return None;
    }
    let (ds_name, ds_pts, ds_color) = base_datasets.get(access_state.fit_dataset_idx)?;
    let log_mode = access_state.fit_log_mode && fit_type.supports_log_mode();
    let cache_key = FitCacheKey {
        fit_type,
        dataset_idx: access_state.fit_dataset_idx,
        log_x,
        log_y,
        log_mode,
        dataset_hash: hash_fit_dataset(ds_name, ds_pts),
    };
    let fit_res = if let Some((ref prev_key, ref prev_res)) = access_state.cached_fit
        && *prev_key == cache_key
    {
        prev_res.clone()?
    } else {
        let solved = fit_model_with_axes(ds_pts, fit_type, ds_name, log_x, log_y, log_mode);
        access_state.cached_fit = Some((cache_key, solved.clone()));
        solved?
    };
    let accent = if *ds_color == Color32::TRANSPARENT {
        ui.visuals().widgets.active.bg_fill
    } else {
        *ds_color
    };

    let disp_name = display_dataset_name(ds_name, access_state.fit_dataset_idx);
    let curve_pts = sample_fit_curve(&fit_res, transform);
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
    Some((fit_res, accent))
}

fn sample_fit_curve(fit_res: &FitResult, transform: &PlotTransform) -> Vec<[f64; 2]> {
    let bounds = transform.bounds();
    let (x_min, x_max) = (bounds.min[0], bounds.max[0]);
    if !x_min.is_finite() || !x_max.is_finite() || x_max <= x_min {
        return Vec::new();
    }
    let count = 200_usize;
    let use_log_x = transform.log_axis().x && x_min > 0.0;
    let (lo_log, hi_log) = if use_log_x {
        (x_min.log10(), x_max.log10())
    } else {
        (0.0, 0.0)
    };
    let mut out = Vec::with_capacity(count + 1);
    for i in 0..=count {
        let t = i as f64 / (count as f64);
        let x = if use_log_x {
            10.0_f64.powf(lo_log + t * (hi_log - lo_log))
        } else {
            x_min + t * (x_max - x_min)
        };
        let y = fit_res.eval(x);
        if x.is_finite() && y.is_finite() {
            out.push([x, y]);
        }
    }
    out
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
    import_toast: &mut Option<(String, bool)>,
) {
    let badge_pos = plot_rect.min + vec2(12.0_f32, 36.0_f32);
    let badge_width = 280.0_f32.min((plot_rect.width() - 24.0_f32).max(160.0_f32));
    let badge_height = 126.0_f32.min((plot_rect.height() - 44.0_f32).max(65.0_f32));
    let badge_rect = Rect::from_min_size(badge_pos, vec2(badge_width, badge_height));
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
                    crate::fit_export::render_fit_overlay_actions(ui, fit_res, import_toast);
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
    base_datasets: &[BaseDatasetTuple],
    access_state: &mut AccessState,
    in_context_menu: bool,
) {
    let valid: Vec<(usize, &String, &Vec<[f64; 2]>)> = base_datasets
        .iter()
        .enumerate()
        .filter(|(_, (_, pts, _))| pts.len() >= 2)
        .map(|(i, (name, pts, _))| (i, name, pts))
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
    valid: &[(usize, &String, &Vec<[f64; 2]>)],
    access_state: &mut AccessState,
) {
    if valid.len() <= 1 {
        return;
    }
    ui.label(RichText::new("Select Dataset:").small().weak());
    for &(idx, name, pts) in valid {
        let label = display_dataset_name(name, idx);
        if ui
            .selectable_label(
                access_state.fit_dataset_idx == idx,
                format!("{label} ({} pts)", pts.len()),
            )
            .clicked()
        {
            access_state.fit_dataset_idx = idx;
        }
    }
    ui.separator();
}

//! Hoverable fitted curve & residual stem series injection, badge rendering, Fit Curve menus,
//! and custom regression callback registration.

use std::sync::{Arc, RwLock};

use egui::{Color32, Rect, RichText, Stroke, Ui, vec2};

use crate::{
    AccessState, SharedPoints,
    a11y::{self, AccessibleTooltipExt as _},
    fit::{
        CustomFitCallback, CustomFitEvaluator, FitResult, FitType, MAX_FIT_PARAMS,
        fit_custom_evaluator,
    },
    items::{Line, LineStyle, MarkerShape, PlotItem, Points},
    runtime_cache::{CachedItemSeries, FitRequest, PlotRuntimeCache},
    transform::PlotTransform,
};

static CUSTOM_FIT_CALLBACK: RwLock<Option<CustomFitCallback>> = RwLock::new(None);

/// Registers an application-level custom formula compiler callback for `FitType::Custom` overlays.
pub fn register_custom_fit_callback(
    callback: impl Fn(&str) -> Result<Arc<dyn CustomFitEvaluator>, String> + Send + Sync + 'static,
) {
    if let Ok(mut slot) = CUSTOM_FIT_CALLBACK.write() {
        *slot = Some(Arc::new(callback));
    }
}

/// Returns `true` when an application-level custom regression callback has been registered.
#[must_use]
pub fn has_custom_fit_callback() -> bool {
    CUSTOM_FIT_CALLBACK
        .read()
        .ok()
        .is_some_and(|slot| slot.is_some())
}

pub(crate) fn custom_fit_callback_ptr() -> usize {
    CUSTOM_FIT_CALLBACK
        .read()
        .ok()
        .and_then(|slot| {
            slot.as_ref()
                .map(|cb| Arc::as_ptr(cb) as *const () as usize)
        })
        .unwrap_or(0)
}

/// Compiles `formula` via the registered callback and fits it to `points` using 6-parameter LM.
pub fn fit_custom_with_registered_callback(
    points: &[[f64; 2]],
    formula: &str,
    initial_seeds: &[f64],
    dataset_name: &str,
) -> Result<FitResult, String> {
    let cb = CUSTOM_FIT_CALLBACK
        .read()
        .ok()
        .and_then(|slot| slot.clone())
        .ok_or_else(|| "No custom regression callback registered".to_owned())?;
    let evaluator = cb(formula)?;
    fit_custom_evaluator(points, evaluator, formula, initial_seeds, dataset_name)
}

fn parse_seed_list(raw: &str) -> Result<Vec<f64>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for tok in trimmed.split([',', ';', ' ']).filter(|s| !s.is_empty()) {
        let val: f64 = tok
            .parse()
            .map_err(|_| format!("Invalid initial seed '{tok}'"))?;
        if !val.is_finite() {
            return Err("Initial seeds must be finite".to_owned());
        }
        out.push(val);
        if out.len() > MAX_FIT_PARAMS {
            return Err(format!("At most {MAX_FIT_PARAMS} seeds are supported"));
        }
    }
    Ok(out)
}

pub(crate) fn solve_custom_fit(
    ds_pts: &[[f64; 2]],
    ds_name: &str,
    formula: &str,
    seeds_text: &str,
) -> Result<FitResult, String> {
    let seeds = parse_seed_list(seeds_text)?;
    fit_custom_with_registered_callback(ds_pts, formula, &seeds, ds_name)
}

fn select_valid_dataset_index(base_series: &[CachedItemSeries], access_state: &mut AccessState) {
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
}

/// Computes or reuses the memoized active regression fit (if enabled) and appends hoverable
/// fitted curve and optional vertical residual stem series directly into `items`.
///
/// When `access_state.fit_model` is `None` or hidden, skips regression fitting and curve sampling.
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
    select_valid_dataset_index(base_series, access_state);

    let Some(fit_type) = access_state.fit_model else {
        access_state.custom_fit_error = None;
        runtime_cache.clear_fit();
        return None;
    };
    if !access_state.show_fit_overlay {
        return None;
    }
    let target = base_series.get(access_state.fit_dataset_idx)?;
    let log_mode = access_state.fit_log_mode && fit_type.supports_log_mode();
    let cb_ptr = if fit_type == FitType::Custom {
        custom_fit_callback_ptr()
    } else {
        0
    };
    let (fit_opt, curve_pts, custom_err) = runtime_cache.resolve_fit_and_curve(&FitRequest {
        ds_pts: &target.points,
        ds_name: &target.name,
        fit_type,
        log_x,
        log_y,
        fit_log_mode: log_mode,
        custom_formula: &access_state.custom_fit_formula,
        custom_seeds_text: &access_state.custom_fit_seeds_text,
        custom_callback_ptr: cb_ptr,
        need_curve: true,
        transform,
    });
    access_state.custom_fit_error = custom_err;
    let fit_res = fit_opt?;
    let accent = if target.color == Color32::TRANSPARENT {
        ui.visuals().widgets.active.bg_fill
    } else {
        target.color
    };
    inject_fit_series(
        ui,
        items,
        &target.name,
        access_state,
        &fit_res,
        curve_pts,
        accent,
    );
    Some((fit_res, accent))
}

fn inject_fit_series<'a>(
    ui: &Ui,
    items: &mut Vec<Box<dyn PlotItem + 'a>>,
    ds_name: &str,
    access_state: &AccessState,
    fit_res: &FitResult,
    curve_pts: SharedPoints,
    accent: Color32,
) {
    let disp_name = display_dataset_name(ds_name, access_state.fit_dataset_idx);
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
    let badge_rect = Rect::from_min_size(badge_pos, vec2(296.0_f32, 76.0_f32));
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
    base_datasets: &[CachedItemSeries],
    access_state: &mut AccessState,
    in_context_menu: bool,
) {
    let valid: Vec<(usize, &str, usize)> = base_datasets
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.points.len() >= 2)
        .map(|(i, entry)| (i, entry.name.as_str(), entry.points.len()))
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
    render_custom_fit_controls(ui, access_state);
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

fn render_custom_fit_controls(ui: &mut Ui, access_state: &mut AccessState) {
    if !has_custom_fit_callback() {
        return;
    }
    let selected = access_state.fit_model == Some(FitType::Custom);
    if ui
        .selectable_label(selected, FitType::Custom.label())
        .clicked()
    {
        access_state.fit_model = Some(FitType::Custom);
        access_state.show_fit_overlay = true;
    }
    if access_state.fit_model == Some(FitType::Custom) {
        ui.add_space(4.0);
        ui.label(RichText::new("Formula y = f(x) (up to 6 params):").small());
        ui.add(
            egui::TextEdit::singleline(&mut access_state.custom_fit_formula)
                .desired_width(210.0)
                .font(egui::TextStyle::Monospace),
        );
        ui.label(RichText::new("Initial Seeds (optional CSV):").small());
        ui.add(
            egui::TextEdit::singleline(&mut access_state.custom_fit_seeds_text)
                .hint_text("e.g. 1.0, 0.5, 0.0")
                .desired_width(210.0),
        );
        if let Some(ref err) = access_state.custom_fit_error {
            ui.colored_label(ui.visuals().error_fg_color, err);
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

//! Hoverable fitted curve & residual stem series injection, wrapping badge / error banner
//! rendering, `Fit Curve` menus, and custom regression callback registration.

use std::{
    hash::{Hash, Hasher},
    sync::{Arc, RwLock},
};

use egui::{Color32, Pos2, Rect, RichText, Stroke, Ui, pos2, vec2};

use crate::{
    AccessState, Corner,
    a11y::{self, AccessibleTooltipExt as _},
    emit_plot_live_message,
    fit::{
        CustomFitCallback, CustomFitEvaluator, FitResult, FitType, MAX_FIT_PARAMS,
        fit_custom_evaluator, fit_model_with_axes,
    },
    items::{Line, LineStyle, MarkerShape, PlotItem, Points},
    transform::PlotTransform,
};

pub(crate) type BaseDatasetTuple = (String, Vec<[f64; 2]>, Color32);

/// Outcome of evaluating the active curve-fit request for the current frame.
#[derive(Clone, Debug)]
pub(crate) enum FitOverlayStatus {
    /// No fit model is currently selected.
    Inactive,
    /// Fit model is selected, but the overlay is hidden (`show_fit_overlay == false`).
    Hidden,
    /// Fit converged and produced a valid [`FitResult`].
    Succeeded {
        result: Box<FitResult>,
        accent_color: Color32,
    },
    /// Fit was requested by the user, but failed on the active dataset.
    Failed {
        fit_type: FitType,
        dataset_name: String,
        reason: String,
    },
}

impl FitOverlayStatus {
    #[must_use]
    pub(crate) const fn can_toggle_visibility(&self) -> bool {
        matches!(self, Self::Succeeded { .. } | Self::Hidden)
    }

    #[must_use]
    pub(crate) const fn did_inject_items(&self) -> bool {
        matches!(self, Self::Succeeded { .. })
    }
}

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FitCacheKey {
    pub(crate) fit_type: FitType,
    pub(crate) dataset_idx: usize,
    pub(crate) log_x: bool,
    pub(crate) log_y: bool,
    pub(crate) log_mode: bool,
    pub(crate) dataset_hash: u64,
    pub(crate) custom_hash: u64,
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

fn hash_custom_fit_config(access_state: &AccessState, fit_type: FitType) -> u64 {
    if fit_type != FitType::Custom {
        return 0;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    access_state.custom_fit_formula.hash(&mut hasher);
    access_state.custom_fit_seeds_text.hash(&mut hasher);
    has_custom_fit_callback().hash(&mut hasher);
    hasher.finish()
}

/// Computes the active regression fit (if enabled) and appends hoverable fitted curve and
/// optional vertical residual stem series directly into `items` BEFORE legend construction.
pub(crate) fn compute_and_inject_fit_items<'a>(
    ui: &Ui,
    items: &mut Vec<Box<dyn PlotItem + 'a>>,
    base_datasets: &[BaseDatasetTuple],
    access_state: &mut AccessState,
    transform: &PlotTransform,
    log_x: bool,
    log_y: bool,
) -> FitOverlayStatus {
    select_valid_dataset_index(base_datasets, access_state);

    let Some(fit_type) = access_state.fit_model else {
        access_state.last_fit_error_announced = None;
        return FitOverlayStatus::Inactive;
    };
    if !access_state.show_fit_overlay {
        return FitOverlayStatus::Hidden;
    }
    let Some((ds_name, ds_pts, ds_color)) = base_datasets.get(access_state.fit_dataset_idx) else {
        return FitOverlayStatus::Failed {
            fit_type,
            dataset_name: "Selected series".to_owned(),
            reason: "No visible series data available for curve fitting.".to_owned(),
        };
    };
    let disp_name = display_dataset_name(ds_name, access_state.fit_dataset_idx);
    let Some(fit_res) =
        resolve_cached_fit(ds_pts, fit_type, &disp_name, log_x, log_y, access_state)
    else {
        let reason = access_state
            .custom_fit_error
            .clone()
            .filter(|_| fit_type == FitType::Custom)
            .unwrap_or_else(|| diagnose_fit_failure(ds_pts, fit_type, log_x, log_y));
        return FitOverlayStatus::Failed {
            fit_type,
            dataset_name: disp_name,
            reason,
        };
    };

    access_state.last_fit_error_announced = None;
    let accent = resolve_fit_accent_color(ui, *ds_color);
    append_fitted_series(
        ui,
        items,
        &fit_res,
        &disp_name,
        accent,
        transform,
        access_state,
    );
    FitOverlayStatus::Succeeded {
        result: Box::new(fit_res),
        accent_color: accent,
    }
}

fn select_valid_dataset_index(base_datasets: &[BaseDatasetTuple], access_state: &mut AccessState) {
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
}

fn resolve_cached_fit(
    ds_pts: &[[f64; 2]],
    fit_type: FitType,
    disp_name: &str,
    log_x: bool,
    log_y: bool,
    access_state: &mut AccessState,
) -> Option<FitResult> {
    let log_mode = access_state.fit_log_mode && fit_type.supports_log_mode();
    let cache_key = FitCacheKey {
        fit_type,
        dataset_idx: access_state.fit_dataset_idx,
        log_x,
        log_y,
        log_mode,
        dataset_hash: hash_fit_dataset(disp_name, ds_pts),
        custom_hash: hash_custom_fit_config(access_state, fit_type),
    };
    if let Some((ref prev_key, ref prev_res)) = access_state.cached_fit
        && *prev_key == cache_key
    {
        return prev_res.clone();
    }
    let solved = resolve_active_fit(
        ds_pts,
        disp_name,
        fit_type,
        access_state,
        log_x,
        log_y,
        log_mode,
    );
    access_state.cached_fit = Some((cache_key, solved.clone()));
    solved
}

fn resolve_active_fit(
    ds_pts: &[[f64; 2]],
    ds_name: &str,
    fit_type: FitType,
    access_state: &mut AccessState,
    log_x: bool,
    log_y: bool,
    log_mode: bool,
) -> Option<FitResult> {
    if fit_type == FitType::Custom {
        let seeds = match parse_seed_list(&access_state.custom_fit_seeds_text) {
            Ok(s) => s,
            Err(err) => {
                access_state.custom_fit_error = Some(err);
                return None;
            }
        };
        match fit_custom_with_registered_callback(
            ds_pts,
            &access_state.custom_fit_formula,
            &seeds,
            ds_name,
        ) {
            Ok(res) => {
                access_state.custom_fit_error = None;
                Some(res)
            }
            Err(err) => {
                access_state.custom_fit_error = Some(err);
                None
            }
        }
    } else {
        access_state.custom_fit_error = None;
        fit_model_with_axes(ds_pts, fit_type, ds_name, log_x, log_y, log_mode).ok()
    }
}

fn resolve_fit_accent_color(ui: &Ui, ds_color: Color32) -> Color32 {
    if ds_color == Color32::TRANSPARENT {
        ui.visuals().widgets.active.bg_fill
    } else {
        ds_color
    }
}

fn append_fitted_series<'a>(
    ui: &Ui,
    items: &mut Vec<Box<dyn PlotItem + 'a>>,
    fit_res: &FitResult,
    disp_name: &str,
    accent: Color32,
    transform: &PlotTransform,
    access_state: &AccessState,
) {
    let curve_pts = sample_fit_curve(fit_res, transform);
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

fn diagnose_fit_failure(
    points: &[[f64; 2]],
    fit_type: FitType,
    log_x: bool,
    log_y: bool,
) -> String {
    let min_pts = match fit_type {
        FitType::Quadratic | FitType::Logistic => 3,
        FitType::Cubic | FitType::Sinusoidal => 4,
        FitType::Quartic => 5,
        _ => 2,
    };
    let finite: Vec<[f64; 2]> = points
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect();
    if finite.len() < min_pts {
        return format!(
            "Requires at least {min_pts} finite points (found {}).",
            finite.len()
        );
    }
    let needs_pos_x = log_x || matches!(fit_type, FitType::PowerLaw | FitType::Logarithmic);
    if needs_pos_x && finite.iter().filter(|p| p[0] > 0.0).count() < min_pts {
        return "Requires strictly positive X values (x > 0) for logarithmic/power domain."
            .to_owned();
    }
    if log_y && finite.iter().filter(|p| p[1] > 0.0).count() < min_pts {
        return "Requires strictly positive Y values (y > 0) on logarithmic Y-axis.".to_owned();
    }
    if matches!(fit_type, FitType::Exponential | FitType::PowerLaw) {
        let pos_y = finite.iter().filter(|p| p[1] > 0.0).count();
        let neg_y = finite.iter().filter(|p| p[1] < 0.0).count();
        if pos_y < 2 && neg_y < 2 {
            return "Requires at least 2 non-zero Y values of matching sign.".to_owned();
        }
    }
    let (x_min, x_max) = finite
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
            (lo.min(p[0]), hi.max(p[0]))
        });
    if (x_max - x_min).abs() <= 1e-15 {
        return "Cannot fit constant-X dataset (zero X variance).".to_owned();
    }
    format!(
        "{} solver did not converge on this dataset.",
        fit_type.short_label()
    )
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

/// Placement context for the floating curve-fit summary frame or error banner.
pub(crate) struct FitOverlayLayout {
    pub plot_rect: Rect,
    pub legend_placement: Option<(Corner, Rect)>,
    pub right_controls_rect: Option<Rect>,
    pub dataset_idx: usize,
}

/// Paints either the wrapping curve-fit formula summary or the inline failed-fit diagnostic banner.
pub(crate) fn paint_fit_status_overlay(
    ui: &mut Ui,
    layout: &FitOverlayLayout,
    status: &FitOverlayStatus,
    access_state: &mut AccessState,
) {
    match status {
        FitOverlayStatus::Inactive | FitOverlayStatus::Hidden => {}
        FitOverlayStatus::Succeeded {
            result,
            accent_color,
        } => {
            if access_state.show_fit_overlay {
                let avail_rect = compute_fit_overlay_rect(layout);
                paint_succeeded_fit_frame(
                    ui,
                    avail_rect,
                    result,
                    layout.dataset_idx,
                    *accent_color,
                );
            }
        }
        FitOverlayStatus::Failed {
            fit_type,
            dataset_name,
            reason,
        } => {
            announce_fit_failure_once(ui, access_state, *fit_type, dataset_name, reason);
            let avail_rect = compute_fit_overlay_rect(layout);
            paint_failed_fit_banner(
                ui,
                avail_rect,
                *fit_type,
                dataset_name,
                reason,
                access_state,
            );
        }
    }
}

fn compute_fit_overlay_rect(layout: &FitOverlayLayout) -> Rect {
    const PAD: f32 = 8.0;
    let plot_rect = layout.plot_rect;
    let x_min = plot_rect.min.x + PAD;
    let mut y_min = plot_rect.min.y + PAD;
    let ctrl_left = layout
        .right_controls_rect
        .map_or(plot_rect.max.x - PAD, |r| r.min.x - PAD);
    let mut x_max = ctrl_left;

    if let Some((corner, lrect)) = layout.legend_placement {
        match corner {
            Corner::LeftTop => {
                y_min = (lrect.max.y + PAD).min(plot_rect.max.y - PAD);
                if layout
                    .right_controls_rect
                    .is_none_or(|ctrl| y_min > ctrl.max.y + 4.0)
                {
                    x_max = plot_rect.max.x - PAD;
                }
            }
            Corner::RightTop => {
                let left_of_legend = lrect.min.x - PAD;
                if left_of_legend - x_min >= 180.0 {
                    x_max = x_max.min(left_of_legend);
                } else {
                    let below_ctrl = layout
                        .right_controls_rect
                        .map_or(lrect.max.y, |c| lrect.max.y.max(c.max.y));
                    y_min = (below_ctrl + PAD).min(plot_rect.max.y - PAD);
                    x_max = plot_rect.max.x - PAD;
                }
            }
            Corner::LeftBottom | Corner::RightBottom => {}
        }
    }

    let safe_max = Pos2::new(
        x_max.max(x_min + 64.0),
        (plot_rect.max.y - PAD).max(y_min + 24.0),
    );
    Rect::from_min_max(pos2(x_min, y_min), safe_max)
}

fn paint_succeeded_fit_frame(
    ui: &mut Ui,
    avail_rect: Rect,
    fit_res: &FitResult,
    dataset_idx: usize,
    accent_color: Color32,
) {
    let max_frame_w = avail_rect.width().clamp(64.0, 384.0);
    let wrap_w = (max_frame_w - 16.0).max(48.0);
    let mut overlay_ui = ui.new_child(egui::UiBuilder::new().max_rect(avail_rect));
    overlay_ui.set_clip_rect(avail_rect.intersect(ui.clip_rect()));
    overlay_ui.scope(|ui| {
        ui.set_max_width(max_frame_w);
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
        egui::Frame::window(ui.style())
            .fill(ui.visuals().window_fill.gamma_multiply(0.92_f32))
            .stroke(Stroke::new(1.5_f32, accent_color))
            .corner_radius(6.0_f32)
            .inner_margin(6.0_f32)
            .show(ui, |ui| {
                ui.set_max_width(wrap_w);
                ui.vertical(|ui| {
                    let name_display = display_dataset_name(&fit_res.dataset_name, dataset_idx);
                    let mode_tag = if fit_res.log_mode { " [Log Mode]" } else { "" };
                    ui.label(
                        RichText::new(format!("Fit: {name_display}{mode_tag}"))
                            .strong()
                            .color(accent_color),
                    );
                    ui.add(
                        egui::Label::new(
                            RichText::new(&fit_res.formula).monospace().size(12.0_f32),
                        )
                        .wrap(),
                    );
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

fn announce_fit_failure_once(
    ui: &Ui,
    access_state: &mut AccessState,
    fit_type: FitType,
    dataset_name: &str,
    reason: &str,
) {
    let key = (fit_type, access_state.fit_dataset_idx, reason.to_owned());
    if access_state.last_fit_error_announced.as_ref() != Some(&key) {
        access_state.last_fit_error_announced = Some(key);
        let msg = format!(
            "Curve fit failed ({}) on {dataset_name}: {reason}",
            fit_type.short_label()
        );
        emit_plot_live_message(ui.ctx(), msg, true);
    }
}

fn paint_failed_fit_banner(
    ui: &mut Ui,
    avail_rect: Rect,
    fit_type: FitType,
    dataset_name: &str,
    reason: &str,
    access_state: &mut AccessState,
) {
    let err_color = if ui.visuals().dark_mode {
        Color32::from_rgb(255, 118, 118)
    } else {
        Color32::from_rgb(180, 28, 28)
    };
    let max_frame_w = avail_rect.width().clamp(64.0, 360.0);
    let wrap_w = (max_frame_w - 16.0).max(48.0);
    let mut overlay_ui = ui.new_child(egui::UiBuilder::new().max_rect(avail_rect));
    overlay_ui.set_clip_rect(avail_rect.intersect(ui.clip_rect()));
    overlay_ui.scope(|ui| {
        ui.set_max_width(max_frame_w);
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
        egui::Frame::window(ui.style())
            .fill(ui.visuals().window_fill.gamma_multiply(0.94_f32))
            .stroke(Stroke::new(1.5_f32, err_color))
            .corner_radius(6.0_f32)
            .inner_margin(6.0_f32)
            .show(ui, |ui| {
                ui.set_max_width(wrap_w);
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(format!("⚠ Fit Failed ({})", fit_type.short_label()))
                            .strong()
                            .color(err_color),
                    );
                    ui.add(egui::Label::new(format!("{dataset_name}: {reason}")).wrap());
                    let clear_btn = egui::Button::new("Clear Fit").min_size(vec2(72.0, 24.0));
                    if a11y::focusable_action_button(
                        ui,
                        true,
                        clear_btn,
                        Some("Clear failed curve fit"),
                    )
                    .accessible_tooltip("Clear failed curve fit")
                    .clicked()
                    {
                        access_state.fit_model = None;
                        access_state.cached_fit = None;
                        access_state.custom_fit_error = None;
                        access_state.last_fit_error_announced = None;
                    }
                });
            });
    });
}

fn format_metric_compact(v: f64) -> String {
    if !v.is_finite() {
        "N/A".to_owned()
    } else if v == 0.0 {
        "0.0000".to_owned()
    } else {
        crate::format::format_fit_param(v)
    }
}

/// Renders the shared `Fit Curve` menu items for both the docked header and context menu.
pub(crate) fn render_fit_curve_menu(
    ui: &mut Ui,
    base_datasets: &[BaseDatasetTuple],
    access_state: &mut AccessState,
    in_context_menu: bool,
    fit_status: &FitOverlayStatus,
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
        render_context_fit_actions(ui, access_state, fit_status);
    }
}

fn render_context_fit_actions(
    ui: &mut Ui,
    access_state: &mut AccessState,
    fit_status: &FitOverlayStatus,
) {
    ui.separator();
    if fit_status.can_toggle_visibility() {
        let toggle_text = if access_state.show_fit_overlay {
            "Hide Fit"
        } else {
            "Show Fit"
        };
        if ui.button(toggle_text).clicked() {
            access_state.show_fit_overlay = !access_state.show_fit_overlay;
            ui.close();
        }
    }
    if ui.button("Clear Fit").clicked() {
        access_state.fit_model = None;
        access_state.cached_fit = None;
        access_state.custom_fit_error = None;
        access_state.last_fit_error_announced = None;
        ui.close();
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

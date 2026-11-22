//! Free-parameter discovery and `AnimatedSlider` bank for `MathExpressionInputBar`.

use crate::reflective_ui::slider::{
    load_slider_settings, resolve_scoped_slider_id, save_slider_settings,
};
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use std::collections::BTreeMap;
use std::ops::RangeInclusive;

/// Maximum number of free parameter sliders synthesized per formula scope.
pub const MAX_SYNTHESIZED_PARAMS: usize = 6;

const HARD_PARAM_MIN: f64 = -1e4;
const HARD_PARAM_MAX: f64 = 1e4;

/// Per-parameter soft `[min, max]` and `step` configuration stored in [`ParameterSliderBank`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParameterRangeConfig {
    /// Soft lower bound for the parameter slider.
    pub min: f64,
    /// Soft upper bound for the parameter slider.
    pub max: f64,
    /// Step size for the parameter slider (`0.0` = continuous).
    pub step: f64,
}

/// Manages discovered free parameter values, custom per-parameter bounds, inline evaluation slices,
/// and [`AnimatedSlider`] UI across formula sandboxes.
#[derive(Clone, Debug)]
pub struct ParameterSliderBank {
    active_names: Vec<String>,
    values: BTreeMap<String, f64>,
    bounds_overrides: BTreeMap<String, ParameterRangeConfig>,
    packed_values: [f64; MAX_SYNTHESIZED_PARAMS],
    default_value: f64,
    min_bound: f64,
    max_bound: f64,
}

impl Default for ParameterSliderBank {
    fn default() -> Self {
        Self::new()
    }
}

impl ParameterSliderBank {
    /// Creates a new parameter bank defaulting newly discovered parameters to `1.0` in `-5.0..=5.0`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_names: Vec::new(),
            values: BTreeMap::new(),
            bounds_overrides: BTreeMap::new(),
            packed_values: [1.0; MAX_SYNTHESIZED_PARAMS],
            default_value: 1.0,
            min_bound: -5.0,
            max_bound: 5.0,
        }
    }

    /// Configures default slider range bounds and default parameter value.
    #[must_use]
    pub fn with_range(mut self, range: RangeInclusive<f64>, default_value: f64) -> Self {
        self.min_bound =
            sanitize_param(*range.start(), -5.0).min(sanitize_param(*range.end(), 5.0));
        self.max_bound =
            sanitize_param(*range.start(), -5.0).max(sanitize_param(*range.end(), 5.0));
        self.default_value = sanitize_param(default_value, 1.0);
        self.repack();
        self
    }

    /// Synchronizes active parameter slots (up to [`MAX_SYNTHESIZED_PARAMS`]) while preserving
    /// previously edited values and custom `[min, max, step]` bounds across formula changes.
    pub fn sync_names(&mut self, names: &[String]) {
        self.active_names.clear();
        for name in names.iter().take(MAX_SYNTHESIZED_PARAMS) {
            if !self.active_names.iter().any(|existing| existing == name) {
                self.values
                    .entry(name.clone())
                    .or_insert(self.default_value);
                self.active_names.push(name.clone());
            }
        }
        self.repack();
    }

    /// Returns the ordered list of active free parameter names.
    #[must_use]
    pub fn active_names(&self) -> &[String] {
        &self.active_names
    }

    /// Returns a stack-backed slice of active parameter values aligned with [`math_commons::ast::CompiledExpr`] slots.
    #[must_use]
    pub fn param_slice(&self) -> &[f64] {
        &self.packed_values[..self.active_names.len()]
    }

    /// Returns the effective `[min, max, step]` configuration for `name`.
    #[must_use]
    pub fn param_bounds(&self, name: &str) -> ParameterRangeConfig {
        self.bounds_overrides
            .get(name)
            .copied()
            .unwrap_or(ParameterRangeConfig {
                min: self.min_bound,
                max: self.max_bound,
                step: 0.0,
            })
    }

    /// Returns `true` when `name` has custom `[min, max, step]` bounds stored in this bank.
    #[must_use]
    pub fn has_custom_bounds(&self, name: &str) -> bool {
        self.bounds_overrides.contains_key(name)
    }

    /// Sets custom soft `[min, max]` bounds and `step` override for parameter `name`.
    pub fn set_param_bounds(&mut self, name: impl Into<String>, min: f64, max: f64, step: f64) {
        let a = sanitize_param(min, self.min_bound);
        let b = sanitize_param(max, self.max_bound);
        let lo = a.min(b);
        let hi = a.max(b);
        let clean_step = if step.is_finite() && step > 0.0 {
            step.min((hi - lo).abs().max(1.0))
        } else {
            0.0
        };
        self.bounds_overrides.insert(
            name.into(),
            ParameterRangeConfig {
                min: lo,
                max: hi,
                step: clean_step,
            },
        );
    }

    /// Clears custom soft bounds for `name`, reverting to the bank's default range.
    pub fn clear_param_bounds(&mut self, name: &str) {
        self.bounds_overrides.remove(name);
    }

    /// Exports only the currently active parameter values for compact JSON / `#state=` URLs.
    #[must_use]
    pub fn export_active_map(&self) -> BTreeMap<String, f64> {
        let mut out = BTreeMap::new();
        for (idx, name) in self.active_names.iter().enumerate() {
            out.insert(
                name.clone(),
                sanitize_param(self.packed_values[idx], self.default_value),
            );
        }
        out
    }

    /// Replaces stored parameter values from a deserialized preset or URL map while retaining custom bounds.
    pub fn set_params_map(&mut self, map: &BTreeMap<String, f64>) {
        self.values.clear();
        for (k, &v) in map {
            self.values
                .insert(k.clone(), sanitize_param(v, self.default_value));
        }
        for name in &self.active_names {
            self.values
                .entry(name.clone())
                .or_insert(self.default_value);
        }
        self.repack();
    }

    /// Renders an [`AnimatedSlider`] row for each active free parameter.
    ///
    /// Returns `true` if any parameter was dragged, edited, or animated this frame.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        if self.active_names.is_empty() {
            return false;
        }

        let mut changed = false;
        ui.add_space(4.0);
        ui.label(egui::RichText::new("Parameters:").weak().small());
        let names = self.active_names.clone();
        for (idx, name) in names.iter().enumerate() {
            changed |= self.render_param_row(ui, idx, name);
        }
        changed
    }

    fn render_param_row(&mut self, ui: &mut egui::Ui, idx: usize, name: &str) -> bool {
        let mut val = sanitize_param(self.packed_values[idx], self.default_value);
        let salt = egui::Id::new((ui.id(), "math_param_slider", name));
        let slider_id = resolve_scoped_slider_id(ui, Some(salt), name);

        let mut mem_settings = load_slider_settings(ui.ctx(), slider_id);
        if mem_settings.min.is_none() && mem_settings.max.is_none() && mem_settings.step.is_none() {
            if let Some(ovr) = self.bounds_overrides.get(name) {
                mem_settings.min = Some(ovr.min);
                mem_settings.max = Some(ovr.max);
                if ovr.step > 0.0 {
                    mem_settings.step = Some(ovr.step);
                }
                save_slider_settings(ui.ctx(), slider_id, mem_settings);
            }
        }

        let resp = ui.add(
            AnimatedSlider::new(&mut val, self.min_bound..=self.max_bound)
                .hard_bounds(HARD_PARAM_MIN..=HARD_PARAM_MAX)
                .text(name)
                .id_salt(salt),
        );

        let after_settings = load_slider_settings(ui.ctx(), slider_id);
        if after_settings.min.is_some()
            || after_settings.max.is_some()
            || after_settings.step.is_some()
        {
            self.set_param_bounds(
                name.to_string(),
                after_settings.min.unwrap_or(self.min_bound),
                after_settings.max.unwrap_or(self.max_bound),
                after_settings.step.unwrap_or(0.0),
            );
        } else {
            self.bounds_overrides.remove(name);
        }

        if resp.changed() {
            let clean = sanitize_param(val, self.default_value);
            self.packed_values[idx] = clean;
            self.values.insert(name.to_string(), clean);
            return true;
        }
        false
    }

    fn repack(&mut self) {
        for (idx, name) in self.active_names.iter().enumerate() {
            let raw = self.values.get(name).copied().unwrap_or(self.default_value);
            let clean = sanitize_param(raw, self.default_value);
            self.packed_values[idx] = clean;
            self.values.insert(name.clone(), clean);
        }
    }
}

fn sanitize_param(val: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(HARD_PARAM_MIN, HARD_PARAM_MAX)
    } else {
        fallback
    }
}

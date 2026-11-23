//! Free-parameter discovery and `AnimatedSlider` bank for `MathExpressionInputBar`.

use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use std::collections::BTreeMap;
use std::ops::RangeInclusive;

/// Maximum number of free parameter sliders synthesized per formula scope.
pub const MAX_SYNTHESIZED_PARAMS: usize = 6;

/// Display metadata and domain bounds for a named synthesized parameter slider.
#[derive(Clone, Debug)]
pub struct ParamSliderSpec {
    label: String,
    min: f64,
    max: f64,
    default_value: f64,
}

/// Manages discovered free parameter values, inline evaluation slices, and `AnimatedSlider` UI.
#[derive(Clone, Debug)]
pub struct ParameterSliderBank {
    active_names: Vec<String>,
    values: BTreeMap<String, f64>,
    specs: BTreeMap<String, ParamSliderSpec>,
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
            specs: BTreeMap::new(),
            packed_values: [1.0; MAX_SYNTHESIZED_PARAMS],
            default_value: 1.0,
            min_bound: -5.0,
            max_bound: 5.0,
        }
    }

    /// Configures custom slider range bounds and default parameter value.
    #[must_use]
    pub fn with_range(mut self, range: RangeInclusive<f64>, default_value: f64) -> Self {
        self.min_bound = (*range.start()).min(*range.end());
        self.max_bound = (*range.start()).max(*range.end());
        self.default_value = sanitize_param(default_value, 1.0);
        self.repack();
        self
    }

    /// Registers custom display label, slider range, and default value for a named parameter.
    pub fn register_param_spec(
        &mut self,
        name: &str,
        label: &str,
        range: RangeInclusive<f64>,
        default_value: f64,
    ) {
        let min = (*range.start()).min(*range.end());
        let max = (*range.start()).max(*range.end());
        let clean_default = sanitize_param(default_value, self.default_value).clamp(min, max);
        self.specs.insert(
            name.to_string(),
            ParamSliderSpec {
                label: label.to_string(),
                min,
                max,
                default_value: clean_default,
            },
        );
        self.values.entry(name.to_string()).or_insert(clean_default);
        self.repack();
    }

    fn default_for(&self, name: &str) -> f64 {
        self.specs
            .get(name)
            .map_or(self.default_value, |s| s.default_value)
    }

    /// Synchronizes active parameter slots (up to [`MAX_SYNTHESIZED_PARAMS`]) while preserving
    /// previously edited values across formula changes.
    pub fn sync_names(&mut self, names: &[String]) {
        self.active_names.clear();
        for name in names.iter().take(MAX_SYNTHESIZED_PARAMS) {
            if !self.active_names.iter().any(|existing| existing == name) {
                let fallback = self.default_for(name);
                self.values.entry(name.clone()).or_insert(fallback);
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

    /// Looks up the active parameter value assigned to `name`.
    #[must_use]
    pub fn active_value(&self, name: &str) -> Option<f64> {
        let idx = self.active_names.iter().position(|n| n == name)?;
        Some(self.packed_values[idx])
    }

    /// Updates a single stored parameter value without clearing other remembered parameters.
    pub fn set_param(&mut self, name: &str, value: f64) {
        let fallback = self.default_for(name);
        self.values
            .insert(name.to_string(), sanitize_param(value, fallback));
        self.repack();
    }

    /// Clamps active parameter values to `range` (or registered parameter spec bounds),
    /// replacing non-finite values with each parameter's default.
    pub fn clamp_active_values(&mut self, range: RangeInclusive<f64>) {
        let lo = (*range.start()).min(*range.end());
        let hi = (*range.start()).max(*range.end());
        for name in &self.active_names {
            let fallback = self.default_for(name);
            let (p_lo, p_hi) = self
                .specs
                .get(name)
                .map_or((lo, hi), |spec| (spec.min, spec.max));
            let cur = self.values.get(name).copied().unwrap_or(fallback);
            let clamped = if cur.is_finite() {
                cur.clamp(p_lo, p_hi)
            } else {
                fallback
            };
            self.values.insert(name.clone(), clamped);
        }
        self.repack();
    }

    /// Computes a deterministic hash of active parameter names and values for geometry caching.
    #[must_use]
    pub fn state_fingerprint(&self) -> u64 {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.active_names.hash(&mut h);
        for &val in self.param_slice() {
            let bits = if val == 0.0 { 0u64 } else { val.to_bits() };
            bits.hash(&mut h);
        }
        h.finish()
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

    /// Replaces stored parameter values from a deserialized preset or URL map.
    pub fn set_params_map(&mut self, map: &BTreeMap<String, f64>) {
        self.values.clear();
        for (k, &v) in map {
            let fallback = self.default_for(k);
            self.values.insert(k.clone(), sanitize_param(v, fallback));
        }
        for name in &self.active_names {
            let fallback = self.default_for(name);
            self.values.entry(name.clone()).or_insert(fallback);
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
        for (idx, name) in self.active_names.iter().enumerate() {
            let spec = self.specs.get(name);
            let fallback = spec.map_or(self.default_value, |s| s.default_value);
            let (base_lo, base_hi) =
                spec.map_or((self.min_bound, self.max_bound), |s| (s.min, s.max));
            let display_label = spec.map_or(name.as_str(), |s| s.label.as_str());
            let mut val = sanitize_param(self.packed_values[idx], fallback);
            let lo = base_lo.min(val);
            let hi = base_hi.max(val);
            let resp = ui.add(
                AnimatedSlider::new(&mut val, lo..=hi)
                    .text(display_label)
                    .id_salt((ui.id(), "math_param_slider", name)),
            );
            if resp.changed() {
                let clean = sanitize_param(val, fallback);
                self.packed_values[idx] = clean;
                self.values.insert(name.clone(), clean);
                changed = true;
            }
        }
        changed
    }

    fn repack(&mut self) {
        for (idx, name) in self.active_names.iter().enumerate() {
            let fallback = self.default_for(name);
            let raw = self.values.get(name).copied().unwrap_or(fallback);
            let clean = sanitize_param(raw, fallback);
            self.packed_values[idx] = clean;
            self.values.insert(name.clone(), clean);
        }
    }
}

fn sanitize_param(val: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(-1e4, 1e4)
    } else {
        fallback
    }
}

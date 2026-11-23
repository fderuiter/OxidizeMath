#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{
    WavePotentialPreset, DEFAULT_CUSTOM_EXPR, DEFAULT_DT, DEFAULT_K0, DEFAULT_SIGMA,
    DEFAULT_STEPS_PER_FRAME, DEFAULT_V0, DEFAULT_WIDTH, DEFAULT_X0, DT_MAX, DT_MIN,
    DT_RECOVERY_MIN, GRID_POINTS, K0_MAX, K0_MIN, MAX_CUSTOM_EXPR_BYTES, MAX_STEPS_PER_FRAME,
    POTENTIAL_CLAMP_MAX, POTENTIAL_CLAMP_MIN, SIGMA_MAX, SIGMA_MIN, V0_MAX, V0_MIN, WIDTH_MAX,
    WIDTH_MIN, X0_MAX, X0_MIN,
};
use super::WaveSimulatorTool;
use crate::async_sim::SimWorkerState;
use std::hash::{Hash, Hasher};

fn default_x0() -> f64 {
    DEFAULT_X0
}
fn default_k0() -> f64 {
    DEFAULT_K0
}
fn default_sigma() -> f64 {
    DEFAULT_SIGMA
}
fn default_v0() -> f64 {
    DEFAULT_V0
}
fn default_width() -> f64 {
    DEFAULT_WIDTH
}
fn default_dt() -> f64 {
    DEFAULT_DT
}
fn default_steps_per_frame() -> usize {
    DEFAULT_STEPS_PER_FRAME
}
fn default_custom_expr() -> String {
    DEFAULT_CUSTOM_EXPR.to_string()
}

/// Truncates `text` at the largest valid UTF-8 char boundary at or below `max_bytes`.
#[must_use]
pub fn truncate_utf8_bytes(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[derive(serde::Serialize)]
struct WaveSimStateRef<'a> {
    preset: WavePotentialPreset,
    x0: f64,
    k0: f64,
    sigma: f64,
    v0: f64,
    width: f64,
    dt: f64,
    steps_per_frame: usize,
    custom_expr: &'a str,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    sparse_potential: &'a [(u16, f64)],
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WaveSimState {
    #[serde(default)]
    preset: Option<WavePotentialPreset>,
    #[serde(default)]
    potential_type: Option<f64>,
    #[serde(default = "default_x0")]
    x0: f64,
    #[serde(default = "default_k0")]
    k0: f64,
    #[serde(default = "default_sigma")]
    sigma: f64,
    #[serde(default = "default_v0")]
    v0: f64,
    #[serde(default = "default_width")]
    width: f64,
    #[serde(default = "default_dt")]
    dt: f64,
    #[serde(default = "default_steps_per_frame")]
    steps_per_frame: usize,
    #[serde(default = "default_custom_expr")]
    custom_expr: String,
    #[serde(default)]
    sparse_potential: Vec<(u16, f64)>,
}

impl WaveSimState {
    fn resolve_preset(&self) -> Result<WavePotentialPreset, String> {
        if let Some(p) = self.preset {
            return Ok(p);
        }
        if let Some(pt) = self.potential_type {
            if !pt.is_finite() {
                return Err("WaveSimulator potential_type must be finite".to_string());
            }
            return Ok(WavePotentialPreset::from_legacy_potential_type(pt));
        }
        Err("WaveSimulator state must specify 'preset' or legacy 'potential_type'".to_string())
    }

    fn validate(&self) -> Result<WavePotentialPreset, String> {
        let preset = self.resolve_preset()?;
        if ![self.x0, self.k0, self.sigma, self.v0, self.width, self.dt]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err("WaveSimulator parameters must be finite numbers".to_string());
        }
        if !(X0_MIN..=X0_MAX).contains(&self.x0)
            || !(K0_MIN..=K0_MAX).contains(&self.k0)
            || !(SIGMA_MIN..=SIGMA_MAX).contains(&self.sigma)
            || !(V0_MIN..=V0_MAX).contains(&self.v0)
            || !(WIDTH_MIN..=WIDTH_MAX).contains(&self.width)
            || !(DT_RECOVERY_MIN..=DT_MAX).contains(&self.dt)
        {
            return Err("WaveSimulator numeric parameters are outside safe bounds".to_string());
        }
        if !(1..=MAX_STEPS_PER_FRAME).contains(&self.steps_per_frame) {
            return Err("WaveSimulator steps_per_frame must be in 1..=100".to_string());
        }
        if self.custom_expr.len() > MAX_CUSTOM_EXPR_BYTES {
            return Err(format!(
                "WaveSimulator custom_expr length ({} bytes) exceeds {} byte limit",
                self.custom_expr.len(),
                MAX_CUSTOM_EXPR_BYTES
            ));
        }
        if self.sparse_potential.len() > GRID_POINTS
            || self.sparse_potential.iter().any(|&(idx, val)| {
                usize::from(idx) >= GRID_POINTS
                    || !val.is_finite()
                    || !(POTENTIAL_CLAMP_MIN..=POTENTIAL_CLAMP_MAX).contains(&val)
            })
        {
            return Err("WaveSimulator sparse_potential contains invalid entries".to_string());
        }
        Ok(preset)
    }
}

impl WaveSimulatorTool {
    pub(crate) fn compute_state_revision(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut hasher);
        (self.preset as u8).hash(&mut hasher);
        self.x0.to_bits().hash(&mut hasher);
        self.k0.to_bits().hash(&mut hasher);
        self.sigma.to_bits().hash(&mut hasher);
        self.v0.to_bits().hash(&mut hasher);
        self.width.to_bits().hash(&mut hasher);
        self.dt.to_bits().hash(&mut hasher);
        self.steps_per_frame.hash(&mut hasher);
        self.math_input.raw_text.hash(&mut hasher);
        for &(idx, val) in &self.sparse_potential {
            idx.hash(&mut hasher);
            val.to_bits().hash(&mut hasher);
        }
        hasher.finish().max(1)
    }

    pub(crate) fn parameters_are_valid(&self) -> bool {
        [self.x0, self.k0, self.sigma, self.v0, self.width, self.dt]
            .iter()
            .all(|v| v.is_finite())
            && (X0_MIN..=X0_MAX).contains(&self.x0)
            && (K0_MIN..=K0_MAX).contains(&self.k0)
            && (SIGMA_MIN..=SIGMA_MAX).contains(&self.sigma)
            && (V0_MIN..=V0_MAX).contains(&self.v0)
            && (WIDTH_MIN..=WIDTH_MAX).contains(&self.width)
            && (DT_RECOVERY_MIN..=DT_MAX).contains(&self.dt)
            && (1..=MAX_STEPS_PER_FRAME).contains(&self.steps_per_frame)
            && self.math_input.raw_text.len() <= MAX_CUSTOM_EXPR_BYTES
            && self.sparse_potential.len() <= GRID_POINTS
            && self.sparse_potential.iter().all(|&(idx, val)| {
                usize::from(idx) < GRID_POINTS
                    && val.is_finite()
                    && (POTENTIAL_CLAMP_MIN..=POTENTIAL_CLAMP_MAX).contains(&val)
            })
            && self.worker.state() != SimWorkerState::Faulted
    }

    pub(crate) fn serialize_state(&self) -> Option<String> {
        let bounded_expr = truncate_utf8_bytes(&self.math_input.raw_text, MAX_CUSTOM_EXPR_BYTES);
        serde_json::to_string(&WaveSimStateRef {
            preset: self.preset,
            x0: self.x0,
            k0: self.k0,
            sigma: self.sigma,
            v0: self.v0,
            width: self.width,
            dt: self.dt,
            steps_per_frame: self.steps_per_frame,
            custom_expr: bounded_expr,
            sparse_potential: &self.sparse_potential,
        })
        .ok()
    }

    pub(crate) fn deserialize_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: WaveSimState =
            serde_json::from_str(state).map_err(|e| format!("Invalid WaveSimulator state: {e}"))?;
        let preset = parsed.validate()?;

        self.preset = preset;
        self.x0 = parsed.x0;
        self.k0 = parsed.k0;
        self.sigma = parsed.sigma;
        self.v0 = parsed.v0;
        self.width = parsed.width;
        self.dt = parsed.dt;
        self.steps_per_frame = parsed.steps_per_frame;
        self.sparse_potential = parsed.sparse_potential;
        self.math_input.set_expression(&parsed.custom_expr);
        if let Some(expr) = self.math_input.expr() {
            self.last_valid_expr = expr.clone();
        }
        self.sync_config_and_reset(None);
        Ok(())
    }

    pub(crate) fn clamp_wave_parameters(&mut self) {
        self.x0 = if self.x0.is_finite() {
            self.x0.clamp(X0_MIN, X0_MAX)
        } else {
            DEFAULT_X0
        };
        self.k0 = if self.k0.is_finite() {
            self.k0.clamp(K0_MIN, K0_MAX)
        } else {
            DEFAULT_K0
        };
        self.sigma = if self.sigma.is_finite() {
            self.sigma.clamp(SIGMA_MIN, SIGMA_MAX)
        } else {
            DEFAULT_SIGMA
        };
        self.v0 = if self.v0.is_finite() {
            self.v0.clamp(V0_MIN, V0_MAX)
        } else {
            DEFAULT_V0
        };
        self.width = if self.width.is_finite() {
            self.width.clamp(WIDTH_MIN, WIDTH_MAX)
        } else {
            DEFAULT_WIDTH
        };
        self.dt = if self.dt.is_finite() && self.dt > 0.0 {
            self.dt.clamp(DT_MIN, DT_MAX)
        } else {
            DEFAULT_DT
        };
        self.steps_per_frame = self.steps_per_frame.clamp(1, MAX_STEPS_PER_FRAME);
        self.sparse_potential
            .retain(|&(idx, val)| usize::from(idx) < GRID_POINTS && val.is_finite());
        for (_, val) in &mut self.sparse_potential {
            *val = val.clamp(POTENTIAL_CLAMP_MIN, POTENTIAL_CLAMP_MAX);
        }
        if self.math_input.raw_text.len() > MAX_CUSTOM_EXPR_BYTES {
            let truncated =
                truncate_utf8_bytes(&self.math_input.raw_text, MAX_CUSTOM_EXPR_BYTES).to_string();
            self.math_input.set_expression(&truncated);
            if let Some(expr) = self.math_input.expr() {
                self.last_valid_expr = expr.clone();
            }
        }
        self.sync_config_and_reset(None);
    }
}

//! Presets, URL-safe state schema, [`TheoryDescribable`], and `StarterPresetMetadata` cards
//! for [`super::BiochemicalDynamicsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// View selector inside [`super::BiochemicalDynamicsTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KineticsPlotView {
    /// Side-by-side enzyme saturation curves ($v(\[S\])$) and 1D reaction-diffusion spatial profiles.
    #[default]
    CoupledSplit,
    /// Dedicated Michaelis-Menten vs Hill cooperative velocity curves.
    SaturationCurve,
    /// Dedicated 1D activator-substrate `ReactionDiffusionSystem` spatial concentration wave.
    SpatialReactionDiffusion,
}

impl KineticsPlotView {
    /// All selectable plot views.
    pub const ALL: [Self; 3] = [
        Self::CoupledSplit,
        Self::SaturationCurve,
        Self::SpatialReactionDiffusion,
    ];

    /// Human-readable label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::CoupledSplit => "Coupled Kinetics + Spatial RD",
            Self::SaturationCurve => "Michaelis-Menten vs Hill",
            Self::SpatialReactionDiffusion => "1D Reaction-Diffusion Wave",
        }
    }
}

/// Compact `< 4 KiB` URL-safe state for [`super::BiochemicalDynamicsTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BiochemicalDynamicsState {
    /// Active plot layout mode.
    pub view: KineticsPlotView,
    /// Maximum enzymatic velocity $V_{\max}$ (`0.1..=20.0`).
    pub v_max: f64,
    /// Half-saturation Michaelis constant $K_m$ (`0.05..=10.0`).
    pub k_m: f64,
    /// Cooperative Hill exponent $n$ (`0.5..=8.0`).
    pub hill_n: f64,
    /// Maximum substrate concentration $\[S\]_{\max}$ swept on the kinetics abscissa (`1.0..=40.0`).
    pub max_substrate: f64,
    /// Ambient substrate reservoir $s_0$ in the 1D reaction-diffusion chamber (`0.2..=10.0`).
    pub substrate_reservoir: f64,
    /// Substrate replenishment rate $k_{\mathrm{feed}}$ (`0.01..=2.0`).
    pub feed_rate: f64,
    /// Activator clearance rate $k_{\mathrm{deg}}$ (`0.01..=2.0`).
    pub degradation_rate: f64,
    /// Activator diffusion coefficient $D_u$ (`0.005..=2.0`).
    pub diffusion_activator: f64,
    /// Substrate diffusion coefficient $D_s$ (`0.005..=2.0`).
    pub diffusion_substrate: f64,
    /// 1D spatial grid resolution (`16..=256`).
    pub grid_points: usize,
    /// Integration step $\Delta t$ (`0.001..=0.1`).
    pub dt: f64,
    /// Whether continuous reaction-diffusion stepping is active.
    #[serde(default)]
    pub playing: bool,
}

impl Default for BiochemicalDynamicsState {
    fn default() -> Self {
        Self {
            view: KineticsPlotView::CoupledSplit,
            v_max: 2.4,
            k_m: 1.2,
            hill_n: 2.5,
            max_substrate: 8.0,
            substrate_reservoir: 2.2,
            feed_rate: 0.30,
            degradation_rate: 0.45,
            diffusion_activator: 0.08,
            diffusion_substrate: 0.40,
            grid_points: 96,
            dt: 0.02,
            playing: false,
        }
    }
}

impl BiochemicalDynamicsState {
    /// Validates all parameters against `URL_JSON_BOUNDS` and physical positivity invariants.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("v_max", self.v_max, 0.1, 20.0),
            ("k_m", self.k_m, 0.05, 10.0),
            ("hill_n", self.hill_n, 0.5, 8.0),
            ("max_substrate", self.max_substrate, 1.0, 40.0),
            ("substrate_reservoir", self.substrate_reservoir, 0.2, 10.0),
            ("feed_rate", self.feed_rate, 0.01, 2.0),
            ("degradation_rate", self.degradation_rate, 0.01, 2.0),
            ("diffusion_activator", self.diffusion_activator, 0.005, 2.0),
            ("diffusion_substrate", self.diffusion_substrate, 0.005, 2.0),
            ("dt", self.dt, 0.001, 0.1),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Biochemical parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if !(16..=256).contains(&self.grid_points) {
            return Err(format!(
                "grid_points {} must be within 16..=256",
                self.grid_points
            ));
        }
        Ok(())
    }
}

/// [`TheoryDescribable`] metadata for Michaelis-Menten, Hill cooperativity, and reaction-diffusion.
#[derive(Clone, Debug, Default)]
pub struct BiochemicalTheory;

impl TheoryDescribable for BiochemicalTheory {
    fn theory_description(&self) -> String {
        "Models enzymatic catalysis from hyperbolic Michaelis-Menten saturation v = V_max*[S]/(K_m + [S]) \
         to cooperative allosteric Hill kinetics v = V_max*[S]^n/(K_m^n + [S]^n), coupled directly \
         into a zero-allocation two-species 1D Fickian reaction-diffusion PDE system."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Michaelis-Menten and cooperative Hill enzyme reaction velocity coupled with one-dimensional \
         activator-substrate Fickian reaction-diffusion."
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Michaelis, L., & Menten, M. L. (1913). Die Kinetik der Invertinwirkung. Biochem. Z., 49, \
         333-369; Hill, A. V. (1910). The possible effects of the aggregation of the molecules of \
         haemoglobin on its dissociation curves. J. Physiol., 40, iv-vii."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Allosteric Cooperativity".to_string(),
            "Hill exponent n > 1 produces sigmoidal ultrasensitive switching around half-saturation K_m."
                .to_string(),
        );
        map.insert(
            "Spatial Reaction-Diffusion".to_string(),
            "Couples fast-diffusing substrate replenishment with cooperative autocatalytic turnover."
                .to_string(),
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"v([S]) = \frac{V_{\max}[S]^n}{K_m^n + [S]^n},\quad \frac{\partial c_i}{\partial t} = D_i \nabla^2 c_i + R_i(\mathbf{c})"
                .to_string(),
        )
    }
}

/// Starter preset JSON showcasing high-cooperativity sigmoidal Hill switching ($n = 4$).
pub const PRESET_ALLOSTERIC_HILL: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Allosteric Hill Cooperativity (Hemoglobin n=4)",
    "domain": "biochemical",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "view": "CoupledSplit",
    "v_max": 3.2,
    "k_m": 1.6,
    "hill_n": 4.0,
    "max_substrate": 8.0,
    "substrate_reservoir": 2.6,
    "feed_rate": 0.35,
    "degradation_rate": 0.50,
    "diffusion_activator": 0.06,
    "diffusion_substrate": 0.45,
    "grid_points": 96,
    "dt": 0.02,
    "playing": false
  }
}"#;

/// Starter preset JSON showcasing a propagating autocatalytic reaction-diffusion front.
pub const PRESET_AUTOCATALYTIC_WAVE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Autocatalytic Reaction-Diffusion Pulse",
    "domain": "biochemical",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "view": "SpatialReactionDiffusion",
    "v_max": 3.8,
    "k_m": 1.1,
    "hill_n": 3.2,
    "max_substrate": 10.0,
    "substrate_reservoir": 3.0,
    "feed_rate": 0.28,
    "degradation_rate": 0.42,
    "diffusion_activator": 0.12,
    "diffusion_substrate": 0.60,
    "grid_points": 128,
    "dt": 0.015,
    "playing": true
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "biochemical_allosteric_hill",
        title: "Allosteric Hill Cooperativity",
        description: "v = V_max·[S]ⁿ / (K_mⁿ + [S]ⁿ): Compares tetrameric n=4 ultrasensitive Hill cooperativity against Michaelis-Menten catalysis.",
        category: "Biochemical Kinetics",
        icon: "🧪",
        domain: "biochemical",
        tool_name: Some("BiochemicalDynamicsTool"),
        preset_generator: || PRESET_ALLOSTERIC_HILL,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "biochemical_autocatalytic_wave",
        title: "Autocatalytic Reaction-Diffusion Pulse",
        description: "∂u/∂t = D_u∇²u + s·V_max·uⁿ/(K_mⁿ + uⁿ) - k_deg·u: Cooperative enzyme wave front with substrate depletion.",
        category: "Biochemical Kinetics",
        icon: "⚗",
        domain: "biochemical",
        tool_name: Some("BiochemicalDynamicsTool"),
        preset_generator: || PRESET_AUTOCATALYTIC_WAVE,
    }
}

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::applied::cannibalism::{CannibalismKernel, JuvenileAdultParams};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Minimum permitted number of age grid cells for interactive PDE grids.
pub const MIN_AGE_BINS: usize = 16;

/// Maximum permitted number of age grid cells for interactive PDE grids.
pub const MAX_AGE_BINS: usize = 512;

/// Structural JSON bounds enforced on untrusted `PopulationStructureTool` payloads.
pub const CANNIBALISM_STATE_BOUNDS: crate::framework::ToolStateBounds =
    crate::framework::ToolStateBounds::new(65_536, 8, MAX_AGE_BINS);

/// Active scientific viewport layout inside [`super::PopulationStructureTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PopulationViewMode {
    /// Four-panel synchronized view showing Lexis heatmap, age pyramid, Hopf phase, and 2D ODE.
    #[default]
    Dashboard,
    /// Full-bleed Lexis `(a, t)` cohort density heatmap (`egui::ColorImage`).
    LexisHeatmap,
    /// Age-structured density distribution `n(t, a)` and juvenile cohort overlay.
    AgePyramid,
    /// `(J(t), A(t))` juvenile-adult Hopf limit-cycle phase portrait.
    PhasePortrait,
    /// Side-by-side time trajectories comparing lumped 2D ODE and age-structured PDE totals.
    OdeComparison,
}

impl PopulationViewMode {
    /// Short user-facing label for selector controls.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Dashboard => "2×2 Dashboard",
            Self::LexisHeatmap => "Lexis (a, t) Heatmap",
            Self::AgePyramid => "Age Pyramid n(t, a)",
            Self::PhasePortrait => "Hopf Phase (J, A)",
            Self::OdeComparison => "2D ODE vs PDE",
        }
    }
}

/// Complete validated configuration schema for [`super::PopulationStructureTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopulationStructureState {
    /// Number of uniform age nodes (`16..=512`).
    pub age_bins: usize,
    /// Maximum age horizon `a_max` (`2.0..=20.0`).
    pub max_age: f64,
    /// Juvenile maturation threshold `alpha` (`0.5..=10.0`).
    pub maturation_age: f64,
    /// Maximum juvenile age `a_vuln` vulnerable to cannibalism (`0.2..=8.0`).
    pub vulnerable_age_max: f64,
    /// Requested integration step size `dt` (`0.002..=0.5`).
    pub dt: f64,
    /// Baseline natural mortality `nu_0` (`0.01..=2.0`).
    pub base_mortality: f64,
    /// Cannibalistic attack rate (`0.0..=2.0`).
    pub attack_rate: f64,
    /// Ricker saturation damping `sigma` (`0.0..=5.0`).
    pub saturation_rate: f64,
    /// Peak adult fecundity rate `beta_0` (`0.5..=50.0`).
    pub fecundity_rate: f64,
    /// Lifeboat cannibalistic energy survival reduction (`0.0..=1.0`).
    pub energy_gain: f64,
    /// Lumped 2D ODE cannibalism loss term `phi` (`0.0..=1.0`).
    pub ode_phi: f64,
    /// Active scientific viewport selection.
    pub view_mode: PopulationViewMode,
}

impl Default for PopulationStructureState {
    fn default() -> Self {
        Self {
            age_bins: 64,
            max_age: 8.0,
            maturation_age: 2.0,
            vulnerable_age_max: 1.6,
            dt: 0.04,
            base_mortality: 0.20,
            attack_rate: 0.22,
            saturation_rate: 0.25,
            fecundity_rate: 18.0,
            energy_gain: 0.08,
            ode_phi: 0.12,
            view_mode: PopulationViewMode::Dashboard,
        }
    }
}

fn finite_clamp(val: f64, fallback: f64, min: f64, max: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback.clamp(min, max)
    }
}

impl PopulationStructureState {
    /// Validates `16..=512` age-grid bounds and finite domain ranges for all parameters.
    pub fn validate(&self) -> Result<(), String> {
        if !(MIN_AGE_BINS..=MAX_AGE_BINS).contains(&self.age_bins) {
            return Err(format!(
                "Parameter 'age_bins' = {} must be within [{MIN_AGE_BINS}, {MAX_AGE_BINS}]",
                self.age_bins
            ));
        }
        let checks = [
            ("max_age", self.max_age, 2.0, 20.0),
            ("maturation_age", self.maturation_age, 0.5, 10.0),
            ("vulnerable_age_max", self.vulnerable_age_max, 0.2, 8.0),
            ("dt", self.dt, 0.002, 0.5),
            ("base_mortality", self.base_mortality, 0.01, 2.0),
            ("attack_rate", self.attack_rate, 0.0, 2.0),
            ("saturation_rate", self.saturation_rate, 0.0, 5.0),
            ("fecundity_rate", self.fecundity_rate, 0.5, 50.0),
            ("energy_gain", self.energy_gain, 0.0, 1.0),
            ("ode_phi", self.ode_phi, 0.0, 1.0),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if self.maturation_age > self.max_age {
            return Err("maturation_age must not exceed max_age".to_string());
        }
        if self.vulnerable_age_max > self.maturation_age {
            return Err("vulnerable_age_max must not exceed maturation_age".to_string());
        }
        Ok(())
    }

    /// Returns a sanitized copy with `age_bins` clamped to `16..=512` and finite rate bounds.
    #[must_use]
    pub fn clamped(&self) -> Self {
        let def = Self::default();
        let max_age = finite_clamp(self.max_age, def.max_age, 2.0, 20.0);
        let maturation_age =
            finite_clamp(self.maturation_age, def.maturation_age, 0.5, 10.0).min(max_age);
        let vulnerable_age_max = finite_clamp(
            self.vulnerable_age_max,
            def.vulnerable_age_max,
            0.2,
            8.0,
        )
        .min(maturation_age);
        Self {
            age_bins: self.age_bins.clamp(MIN_AGE_BINS, MAX_AGE_BINS),
            max_age,
            maturation_age,
            vulnerable_age_max,
            dt: finite_clamp(self.dt, def.dt, 0.002, 0.5),
            base_mortality: finite_clamp(self.base_mortality, def.base_mortality, 0.01, 2.0),
            attack_rate: finite_clamp(self.attack_rate, def.attack_rate, 0.0, 2.0),
            saturation_rate: finite_clamp(self.saturation_rate, def.saturation_rate, 0.0, 5.0),
            fecundity_rate: finite_clamp(self.fecundity_rate, def.fecundity_rate, 0.5, 50.0),
            energy_gain: finite_clamp(self.energy_gain, def.energy_gain, 0.0, 1.0),
            ode_phi: finite_clamp(self.ode_phi, def.ode_phi, 0.0, 1.0),
            view_mode: self.view_mode,
        }
    }

    /// Hashes persisted configuration fields into `hasher` for idle-safe revision tracking.
    pub fn hash_into<H: Hasher>(&self, hasher: &mut H) {
        self.age_bins.hash(hasher);
        self.max_age.to_bits().hash(hasher);
        self.maturation_age.to_bits().hash(hasher);
        self.vulnerable_age_max.to_bits().hash(hasher);
        self.dt.to_bits().hash(hasher);
        self.base_mortality.to_bits().hash(hasher);
        self.attack_rate.to_bits().hash(hasher);
        self.saturation_rate.to_bits().hash(hasher);
        self.fecundity_rate.to_bits().hash(hasher);
        self.energy_gain.to_bits().hash(hasher);
        self.ode_phi.to_bits().hash(hasher);
        self.view_mode.hash(hasher);
    }

    /// Builds a domain [`CannibalismKernel`] from this state configuration.
    #[must_use]
    pub fn to_mvf_kernel(&self) -> CannibalismKernel {
        CannibalismKernel::new(
            self.base_mortality,
            0.03,
            self.attack_rate * 2.5,
            self.vulnerable_age_max,
            self.saturation_rate,
            self.fecundity_rate * 0.45,
            self.maturation_age,
            55.0,
        )
    }

    /// Builds domain [`JuvenileAdultParams`] from this state configuration.
    #[must_use]
    pub fn to_juvenile_adult_params(&self) -> JuvenileAdultParams {
        JuvenileAdultParams {
            maturation_age: self.maturation_age,
            juvenile_mortality: self.base_mortality,
            attack_rate: self.attack_rate,
            vulnerability_max_age: self.vulnerable_age_max,
            adult_mortality: (self.base_mortality * 4.5).clamp(0.2, 3.0),
            energy_survival_gain: self.energy_gain,
            adult_fecundity: self.fecundity_rate,
            cannibal_fecundity_gain: 0.12,
            adult_carrying_scale: 80.0,
        }
    }
}

const PRESET_HOPF_CYCLES: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Dwarf & Giant Hopf Cycles",
    "domain": "cannibalism",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "age_bins": 80,
    "max_age": 8.0,
    "maturation_age": 2.0,
    "vulnerable_age_max": 1.7,
    "dt": 0.035,
    "base_mortality": 0.18,
    "attack_rate": 0.28,
    "saturation_rate": 0.20,
    "fecundity_rate": 22.0,
    "energy_gain": 0.05,
    "ode_phi": 0.14,
    "view_mode": "Dashboard"
  }
}"#;

const PRESET_LEXIS_WAVES: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lexis Cohort Ridges",
    "domain": "cannibalism",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "age_bins": 96,
    "max_age": 9.0,
    "maturation_age": 2.2,
    "vulnerable_age_max": 1.8,
    "dt": 0.04,
    "base_mortality": 0.16,
    "attack_rate": 0.32,
    "saturation_rate": 0.18,
    "fecundity_rate": 24.0,
    "energy_gain": 0.04,
    "ode_phi": 0.10,
    "view_mode": "LexisHeatmap"
  }
}"#;

const PRESET_LIFEBOAT_REGIME: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lifeboat Persistence Regime",
    "domain": "cannibalism",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "age_bins": 72,
    "max_age": 10.0,
    "maturation_age": 2.4,
    "vulnerable_age_max": 1.5,
    "dt": 0.04,
    "base_mortality": 0.28,
    "attack_rate": 0.19,
    "saturation_rate": 0.30,
    "fecundity_rate": 16.0,
    "energy_gain": 0.28,
    "ode_phi": 0.08,
    "view_mode": "PhasePortrait"
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PopulationStructureTool",
        display_name: "Population Structure & Cannibalism",
        domain: "cannibalism",
        tags: &[
            "cannibalism",
            "pde",
            "mckendrick",
            "age structure",
            "hopf",
            "applied math",
        ],
        build: || Box::new(super::PopulationStructureTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "cannibalism_hopf_cycles",
        title: "Dwarf & Giant Hopf Cycles",
        description: "Supercritical adult-on-juvenile cannibalism triggering periodic single-cohort dominance waves.",
        category: "Population Dynamics",
        icon: "🐟",
        domain: "cannibalism",
        tool_name: Some("PopulationStructureTool"),
        preset_generator: || PRESET_HOPF_CYCLES,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "cannibalism_lexis_waves",
        title: "Lexis Cohort Ridges",
        description: "Diagonal characteristic cohort ridges propagating across the (a, t) Lexis plane under Diekmann saturation.",
        category: "Population Dynamics",
        icon: "🌊",
        domain: "cannibalism",
        tool_name: Some("PopulationStructureTool"),
        preset_generator: || PRESET_LEXIS_WAVES,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "cannibalism_lifeboat_regime",
        title: "Lifeboat Persistence Regime",
        description: "van den Bosch lifeboat energy feedback sustaining adult survival when juvenile recruitment drops.",
        category: "Population Dynamics",
        icon: "🛟",
        domain: "cannibalism",
        tool_name: Some("PopulationStructureTool"),
        preset_generator: || PRESET_LIFEBOAT_REGIME,
    }
}

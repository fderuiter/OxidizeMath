#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::applied::cannibalism::{
    AgeStructuredConfig, PredationRegime, TriboliumLpaModel,
};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Population cannibalism regime explored in [`super::CannibalismTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CannibalismRegime {
    /// Diekmann et al. (1986) adult-on-juvenile cannibalism producing Hopf cohort waves.
    #[default]
    DiekmannHopf,
    /// Costantino et al. (1997) *Tribolium* egg and callow pupa predation producing period-doubling chaos.
    TriboliumChaos,
}

impl CannibalismRegime {
    /// Human-readable regime title rendered in radio buttons and status summaries.
    pub const fn label(self) -> &'static str {
        match self {
            Self::DiekmannHopf => "Diekmann Hopf Cohort Waves",
            Self::TriboliumChaos => "Tribolium Egg & Callow Chaos",
        }
    }
}

/// Complete validated state schema for [`super::CannibalismTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CannibalismToolState {
    /// Active ecological cannibalism model regime.
    pub regime: CannibalismRegime,
    /// Adult fecundity rate `beta0` (`5.0..=200.0`).
    pub beta0: f64,
    /// Adult-on-juvenile cannibalism attack rate `c` (`0.05..=3.0`).
    pub c_cannibalism: f64,
    /// Adult-on-pupa callow cannibalism coefficient `c_pa` (`0.0..=2.0`).
    pub c_pa: f64,
    /// Baseline per-capita natural mortality rate `mu0` (`0.02..=1.5`).
    pub mu0: f64,
    /// Number of upwind characteristic age bins (`32..=256`).
    pub grid_size: usize,
    /// Number of PDE characteristic steps advanced per animation frame (`1..=20`).
    pub steps_per_frame: usize,
    /// Number of parameter slices in the bifurcation sweep (`16..=80`).
    pub sweep_points: usize,
    /// Whether live cohort transport animation is actively running.
    pub is_playing: bool,
}

impl Default for CannibalismToolState {
    fn default() -> Self {
        Self {
            regime: CannibalismRegime::DiekmannHopf,
            beta0: 20.0,
            c_cannibalism: 0.5,
            c_pa: 0.35,
            mu0: 0.4,
            grid_size: 100,
            steps_per_frame: 4,
            sweep_points: 36,
            is_playing: false,
        }
    }
}

impl CannibalismToolState {
    /// Validates that every numeric parameter is finite and within safe domain bounds.
    pub fn validate(&self) -> Result<(), String> {
        let float_checks = [
            ("beta0", self.beta0, 5.0, 200.0),
            ("c_cannibalism", self.c_cannibalism, 0.05, 3.0),
            ("c_pa", self.c_pa, 0.0, 2.0),
            ("mu0", self.mu0, 0.02, 1.5),
        ];
        for (name, val, min, max) in float_checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Cannibalism parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if !(32..=256).contains(&self.grid_size) {
            return Err(format!(
                "Age grid size {} must lie within [32, 256]",
                self.grid_size
            ));
        }
        if !(1..=20).contains(&self.steps_per_frame) {
            return Err(format!(
                "Steps per frame {} must lie within [1, 20]",
                self.steps_per_frame
            ));
        }
        if !(16..=80).contains(&self.sweep_points) {
            return Err(format!(
                "Bifurcation sweep points {} must lie within [16, 80]",
                self.sweep_points
            ));
        }
        Ok(())
    }

    /// Clamps any out-of-range parameters back into safe domain bounds.
    pub fn clamp_in_place(&mut self) {
        self.beta0 = if self.beta0.is_finite() {
            self.beta0.clamp(5.0, 200.0)
        } else {
            20.0
        };
        self.c_cannibalism = if self.c_cannibalism.is_finite() {
            self.c_cannibalism.clamp(0.05, 3.0)
        } else {
            0.5
        };
        self.c_pa = if self.c_pa.is_finite() {
            self.c_pa.clamp(0.0, 2.0)
        } else {
            0.35
        };
        self.mu0 = if self.mu0.is_finite() {
            self.mu0.clamp(0.02, 1.5)
        } else {
            0.4
        };
        self.grid_size = self.grid_size.clamp(32, 256);
        self.steps_per_frame = self.steps_per_frame.clamp(1, 20);
        self.sweep_points = self.sweep_points.clamp(16, 80);
    }

    /// Hashes every persisted field into `hasher` for deterministic state-revision tracking.
    pub fn hash_into<H: Hasher>(&self, hasher: &mut H) {
        self.regime.hash(hasher);
        self.beta0.to_bits().hash(hasher);
        self.c_cannibalism.to_bits().hash(hasher);
        self.c_pa.to_bits().hash(hasher);
        self.mu0.to_bits().hash(hasher);
        self.grid_size.hash(hasher);
        self.steps_per_frame.hash(hasher);
        self.sweep_points.hash(hasher);
        self.is_playing.hash(hasher);
    }

    /// Maps the validated GUI state onto a zero-allocation [`AgeStructuredConfig`].
    pub fn to_pde_config(&self) -> AgeStructuredConfig {
        match self.regime {
            CannibalismRegime::DiekmannHopf => {
                let (vulnerable_age, handling_time) = if self.beta0 >= 30.0 {
                    (0.75, 0.03)
                } else {
                    (1.5, 0.0)
                };
                AgeStructuredConfig {
                    grid_size: self.grid_size,
                    max_age: 5.0,
                    maturation_age: 2.0,
                    natural_mortality: self.mu0,
                    beta0: self.beta0,
                    regime: PredationRegime::Diekmann {
                        vulnerable_age,
                        attack_rate: self.c_cannibalism,
                        handling_time,
                    },
                }
            }
            CannibalismRegime::TriboliumChaos => {
                let mut cfg = AgeStructuredConfig::tribolium_cycles(self.c_pa);
                cfg.grid_size = self.grid_size;
                cfg.beta0 = self.beta0;
                cfg.natural_mortality = self.mu0;
                cfg
            }
        }
    }

    /// Maps the validated GUI state onto a discrete [`TriboliumLpaModel`].
    pub fn to_lpa_model(&self) -> TriboliumLpaModel {
        TriboliumLpaModel::two_cycle_preset(self.c_pa)
    }
}

/// Starter preset 1: Supercritical Diekmann Hopf cohort waves (`beta0 = 45.0`).
pub const PRESET_DIEKMANN_HOPF_WAVES: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Cannibalism",
    "domain": "cannibalism",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "regime": "DiekmannHopf",
    "beta0": 45.0,
    "c_cannibalism": 1.0,
    "c_pa": 0.35,
    "mu0": 0.4,
    "grid_size": 100,
    "steps_per_frame": 4,
    "sweep_points": 40,
    "is_playing": false
  }
}"#;

/// Starter preset 2: *Tribolium* flour beetle callow predation 2-cycles (`c_pa = 0.7`).
pub const PRESET_TRIBOLIUM_TWO_CYCLES: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Cannibalism",
    "domain": "cannibalism",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "regime": "TriboliumChaos",
    "beta0": 140.0,
    "c_cannibalism": 0.5,
    "c_pa": 0.7,
    "mu0": 0.2,
    "grid_size": 80,
    "steps_per_frame": 4,
    "sweep_points": 44,
    "is_playing": false
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CannibalismTool",
        display_name: "Cannibalism",
        domain: "cannibalism",
        tags: &[
            "cannibalism",
            "mckendrick",
            "pde",
            "lexis",
            "tribolium",
            "bifurcation",
        ],
        build: || Box::new(super::CannibalismTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "cannibalism_diekmann_hopf",
        title: "Diekmann Hopf Cohort Waves",
        description: "Supercritical McKendrick-von Foerster cannibalism (beta0 = 45.0) generating diagonal missing-generation troughs on the Lexis carpet.",
        category: "Population Dynamics",
        icon: "🌊",
        domain: "cannibalism",
        tool_name: Some("CannibalismTool"),
        preset_generator: || PRESET_DIEKMANN_HOPF_WAVES,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "cannibalism_tribolium_two_cycles",
        title: "Tribolium 2-Cycle Bifurcation",
        description: "Flour beetle pupa-adult callow predation (c_pa = 0.7) driving period-doubling 2-cycles across PDE and LPA dynamics.",
        category: "Population Dynamics",
        icon: "🪲",
        domain: "cannibalism",
        tool_name: Some("CannibalismTool"),
        preset_generator: || PRESET_TRIBOLIUM_TWO_CYCLES,
    }
}

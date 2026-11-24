//! Presets, URL-safe state schema, [`TheoryDescribable`], and `StarterPresetMetadata` cards
//! for [`super::PhyllotaxisLSystemTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::biology::lsystem::BotanicalLSystemPreset;
use math_explorer::biology::phyllotaxis::{DivergenceAnglePreset, GOLDEN_ANGLE_DEG, MAX_PRIMORDIA};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Studio viewport display mode inside [`super::PhyllotaxisLSystemTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BotanicalStudioMode {
    /// Side-by-side Vogel phyllotaxis disc and 2D L-system vascular shoot canvas.
    #[default]
    DualSplit,
    /// Dedicated Vogel phyllotaxis primordia and parastichy spiral canvas.
    PhyllotaxisOnly,
    /// Dedicated L-system 2D bracketed turtle shoot canvas.
    LSystemOnly,
}

impl BotanicalStudioMode {
    /// All studio layout modes.
    pub const ALL: [Self; 3] = [Self::DualSplit, Self::PhyllotaxisOnly, Self::LSystemOnly];

    /// Human-readable mode label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::DualSplit => "Dual Studio",
            Self::PhyllotaxisOnly => "Phyllotaxis Disc",
            Self::LSystemOnly => "L-System Canopy",
        }
    }
}

/// Compact `< 4 KiB` URL-safe state for [`super::PhyllotaxisLSystemTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhyllotaxisLSystemState {
    /// Active viewport layout mode.
    pub mode: BotanicalStudioMode,
    /// Selected phyllotactic divergence preset.
    pub divergence_preset: DivergenceAnglePreset,
    /// Divergence angle $\alpha$ in degrees (`10.0..=350.0`).
    pub divergence_deg: f64,
    /// Primordia count $N$ (`8..=4096`).
    pub primordia_count: usize,
    /// Radial Vogel scale $c$ (`0.2..=5.0`).
    pub radial_scale: f64,
    /// Whether to overlay opposing $(F_k, F_{k+1})$ parastichy spirals on the Vogel disc.
    pub show_parastichies: bool,
    /// Selected L-system grammar preset.
    pub lsystem_preset: BotanicalLSystemPreset,
    /// Target L-system derivation depth (`1..=11`).
    pub generations: usize,
    /// Turtle branching angle $\delta$ in degrees (`5.0..=120.0`).
    pub branch_angle_deg: f64,
    /// Primary parametric contraction ratio $r_1$ (`0.35..=0.95`).
    pub contraction_r1: f64,
    /// Lateral parametric contraction ratio $r_2$ (`0.35..=0.95`).
    pub contraction_r2: f64,
}

impl Default for PhyllotaxisLSystemState {
    fn default() -> Self {
        Self {
            mode: BotanicalStudioMode::DualSplit,
            divergence_preset: DivergenceAnglePreset::GoldenAngle,
            divergence_deg: GOLDEN_ANGLE_DEG,
            primordia_count: 256,
            radial_scale: 1.0,
            show_parastichies: true,
            lsystem_preset: BotanicalLSystemPreset::HerbaceousShoot,
            generations: 5,
            branch_angle_deg: 25.0,
            contraction_r1: 0.78,
            contraction_r2: 0.68,
        }
    }
}

impl PhyllotaxisLSystemState {
    /// Validates all parameters against `URL_JSON_BOUNDS` and botanical domain invariants.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("divergence_deg", self.divergence_deg, 10.0, 350.0),
            ("radial_scale", self.radial_scale, 0.2, 5.0),
            ("branch_angle_deg", self.branch_angle_deg, 5.0, 120.0),
            ("contraction_r1", self.contraction_r1, 0.35, 0.95),
            ("contraction_r2", self.contraction_r2, 0.35, 0.95),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Botanical parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if !(8..=MAX_PRIMORDIA).contains(&self.primordia_count) {
            return Err(format!(
                "primordia_count {} must be within 8..={}",
                self.primordia_count, MAX_PRIMORDIA
            ));
        }
        if !(1..=11).contains(&self.generations) {
            return Err(format!(
                "generations {} must be within 1..=11",
                self.generations
            ));
        }
        Ok(())
    }
}

/// [`TheoryDescribable`] metadata for botanical phyllotaxis and L-system morphogenesis.
#[derive(Clone, Debug, Default)]
pub struct BotanicalTheory;

impl TheoryDescribable for BotanicalTheory {
    fn theory_description(&self) -> String {
        "Combines Vogel's planar phyllotaxis spiral (r_n = c*sqrt(n), theta_n = n*alpha) with \
         deterministic D0L and parametric Lindenmayer vascular branching grammars. At the golden \
         divergence angle alpha = 2*pi*(2 - phi) approx 137.5078 deg, primordia pack uniformly \
         into opposing consecutive Fibonacci parastichy families (F_k, F_{k+1}), whereas rational \
         turn fractions collapse into radial orthostichy spokes."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Vogel golden-angle phyllotaxis spiral r sub n equals c root n with opposing Fibonacci \
         parastichy spirals paired with parallel L-system bracketed turtle branching."
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Vogel, H. (1979). A better way to construct the sunflower head. Mathematical Biosciences, \
         44(3-4), 179-189; Prusinkiewicz, P., & Lindenmayer, A. (1990). The Algorithmic Beauty of Plants."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Vogel Phyllotaxis".to_string(),
            "Equal-area primordia placement r_n = c*sqrt(n), theta_n = n*alpha exhibiting Fibonacci (F_k, F_{k+1}) parastichy transitions.".to_string(),
        );
        map.insert(
            "Bracketed L-Systems".to_string(),
            "Zero-allocation double-buffered string rewriting interpreted by a 2D meristem turtle stack.".to_string(),
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"r_n = c\sqrt{n},\quad \theta_n = n\alpha_{\varphi},\quad \alpha_{\varphi} = 2\pi(2 - \varphi) \approx 137.5078^\circ"
                .to_string(),
        )
    }
}

/// Starter preset JSON for golden-angle sunflower parastichies (`(21, 34)` Fibonacci spirals).
pub const PRESET_GOLDEN_SUNFLOWER: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Golden-Angle Sunflower Parastichies",
    "domain": "biology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "DualSplit",
    "divergence_preset": "GoldenAngle",
    "divergence_deg": 137.50776405003785,
    "primordia_count": 377,
    "radial_scale": 1.0,
    "show_parastichies": true,
    "lsystem_preset": "HerbaceousShoot",
    "generations": 5,
    "branch_angle_deg": 25.0,
    "contraction_r1": 0.78,
    "contraction_r2": 0.68
  }
}"#;

/// Starter preset JSON for parametric dichotomous vascular canopy and Lucas phyllotaxis.
pub const PRESET_PARAMETRIC_CANOPY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Parametric Vascular Canopy",
    "domain": "biology",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "DualSplit",
    "divergence_preset": "LucasAngle",
    "divergence_deg": 99.50155281280884,
    "primordia_count": 220,
    "radial_scale": 1.15,
    "show_parastichies": true,
    "lsystem_preset": "ParametricDichotomousTree",
    "generations": 8,
    "branch_angle_deg": 30.0,
    "contraction_r1": 0.75,
    "contraction_r2": 0.68
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "phyllotaxis_golden_sunflower",
        title: "Golden-Angle Sunflower Parastichies",
        description: "r_n = c√n, θ_n = n·137.5078°: Opposing (21, 34) Fibonacci parastichy spirals and D0L herbaceous shoot.",
        category: "Botanical Morphogenesis",
        icon: "🌻",
        domain: "biology",
        tool_name: Some("PhyllotaxisLSystemTool"),
        preset_generator: || PRESET_GOLDEN_SUNFLOWER,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lsystem_parametric_canopy",
        title: "Parametric Vascular Canopy",
        description: "A(l) → F(l)[+A(r₁l)][-A(r₂l)]: Asymmetric dichotomous vascular branching paired with Lucas noble phyllotaxis.",
        category: "Botanical Morphogenesis",
        icon: "🌿",
        domain: "biology",
        tool_name: Some("PhyllotaxisLSystemTool"),
        preset_generator: || PRESET_PARAMETRIC_CANOPY,
    }
}

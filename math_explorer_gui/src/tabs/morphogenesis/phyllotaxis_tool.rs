#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use math_explorer::biology::phyllotaxis::{
    LSystemConfig, LSystemEngine, LSystemPreset, NutrientCoupling, NutrientKineticsKind,
    PhyllotaxisConfig, PhyllotaxisState, GOLDEN_ANGLE_DEG,
};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

fn default_lsys_iters() -> usize {
    4
}

fn default_branch_angle() -> f64 {
    25.0
}

/// Serializable enzyme kinetics choice for [`PhyllotaxisLSystemTool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum PhyllotaxisKineticsChoice {
    /// Non-cooperative hyperbolic Michaelis-Menten saturation ($n = 1$).
    #[default]
    MichaelisMenten,
    /// Cooperative sigmoidal Hill saturation ($n > 0$).
    Hill,
}

impl PhyllotaxisKineticsChoice {
    pub(crate) fn to_domain(self) -> NutrientKineticsKind {
        match self {
            Self::MichaelisMenten => NutrientKineticsKind::MichaelisMenten,
            Self::Hill => NutrientKineticsKind::Hill,
        }
    }
}

/// Serializable L-system grammar preset choice for [`PhyllotaxisLSystemTool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum LSystemPresetChoice {
    /// Axial monopodial shoot (`X -> F+[[X]-X]-F[-FX]+X`).
    #[default]
    BotanicalShoot,
    /// Sympodial bush/tree (`F -> FF+[+F-F-F]-[-F+F+F]`).
    SympodialTree,
    /// Bilateral fern frond (`X -> F[-X][+X]FX`).
    MonopodialFern,
}

impl LSystemPresetChoice {
    pub(crate) fn to_domain(self) -> LSystemPreset {
        match self {
            Self::BotanicalShoot => LSystemPreset::BotanicalShoot,
            Self::SympodialTree => LSystemPreset::SympodialTree,
            Self::MonopodialFern => LSystemPreset::MonopodialFern,
        }
    }
}

/// Complete validated URL and preset state schema for [`PhyllotaxisLSystemTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhyllotaxisToolState {
    /// Divergence angle $\alpha$ in degrees (`45.0..=180.0`).
    pub divergence_angle_deg: f64,
    /// Radial packing scale constant $c$ (`0.05..=2.0`).
    pub scale_c: f64,
    /// Active primordia (floret) count (`5..=512`).
    pub primordia_count: usize,
    /// Nutrient substrate concentration `[S]` (`0.0..=25.0`).
    pub substrate: f64,
    /// Maximum kinetic uptake velocity $V_{\max}$ (`0.05..=25.0`).
    pub v_max: f64,
    /// Half-saturation constant $K_m$ (`0.05..=25.0`).
    pub k_m: f64,
    /// Hill cooperativity exponent $n$ (`0.2..=8.0`).
    pub hill_n: f64,
    /// Enzyme saturation kinetics family.
    #[serde(default)]
    pub kinetics: PhyllotaxisKineticsChoice,
    /// Botanical L-system rewriting preset.
    #[serde(default)]
    pub lsystem_preset: LSystemPresetChoice,
    /// Bounded parallel L-system derivation depth (`1..=6`).
    #[serde(default = "default_lsys_iters")]
    pub lsystem_iterations: usize,
    /// Base branching angle $\delta_0$ in degrees (`5.0..=60.0`).
    #[serde(default = "default_branch_angle")]
    pub branch_angle_deg: f64,
}

impl Default for PhyllotaxisToolState {
    fn default() -> Self {
        Self {
            divergence_angle_deg: GOLDEN_ANGLE_DEG,
            scale_c: 0.45,
            primordia_count: 220,
            substrate: 2.5,
            v_max: 1.0,
            k_m: 1.0,
            hill_n: 2.5,
            kinetics: PhyllotaxisKineticsChoice::MichaelisMenten,
            lsystem_preset: LSystemPresetChoice::BotanicalShoot,
            lsystem_iterations: 4,
            branch_angle_deg: 25.0,
        }
    }
}

impl PhyllotaxisToolState {
    /// Validates finiteness and domain bounds for all phyllotaxis, L-system, and kinetics parameters.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("divergence_angle_deg", self.divergence_angle_deg, 45.0, 180.0),
            ("scale_c", self.scale_c, 0.05, 2.0),
            ("substrate", self.substrate, 0.0, 25.0),
            ("v_max", self.v_max, 0.05, 25.0),
            ("k_m", self.k_m, 0.05, 25.0),
            ("hill_n", self.hill_n, 0.2, 8.0),
            ("branch_angle_deg", self.branch_angle_deg, 5.0, 60.0),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Phyllotaxis parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if !(5..=512).contains(&self.primordia_count) {
            return Err(format!(
                "Phyllotaxis primordia_count {} must be within 5..=512",
                self.primordia_count
            ));
        }
        if !(1..=6).contains(&self.lsystem_iterations) {
            return Err(format!(
                "L-system iterations {} must be within 1..=6",
                self.lsystem_iterations
            ));
        }
        Ok(())
    }
}

/// Interactive tool for golden-angle Vogel phyllotaxis and nutrient-coupled L-system morphogenesis.
pub struct PhyllotaxisLSystemTool {
    pub(crate) state: PhyllotaxisToolState,
    pub(crate) phyllotaxis: PhyllotaxisState,
    pub(crate) lsystem: LSystemEngine,
    pub(crate) theory_cfg: PhyllotaxisConfig,
}

impl Default for PhyllotaxisLSystemTool {
    fn default() -> Self {
        let state = PhyllotaxisToolState::default();
        let cfg = PhyllotaxisConfig {
            divergence_angle_deg: state.divergence_angle_deg,
            scale_c: state.scale_c,
            base_initiation_rate: 12.0,
        };
        let nutrient = NutrientCoupling::default();
        let phyllotaxis = PhyllotaxisState::new(512, state.primordia_count, cfg, nutrient);
        let mut lsystem = LSystemEngine::new(16_384, 16_384, 128);
        lsystem.generate(LSystemConfig::default(), &nutrient);
        Self {
            state,
            phyllotaxis,
            lsystem,
            theory_cfg: cfg,
        }
    }
}

impl PhyllotaxisLSystemTool {
    /// Synchronizes preallocated [`PhyllotaxisState`] and [`LSystemEngine`] from `self.state`.
    pub fn recompute(&mut self) {
        let cfg = PhyllotaxisConfig {
            divergence_angle_deg: self.state.divergence_angle_deg,
            scale_c: self.state.scale_c,
            base_initiation_rate: 12.0,
        };
        let nutrient = NutrientCoupling::new(
            self.state.kinetics.to_domain(),
            self.state.substrate.max(0.0),
            self.state.v_max.max(0.05),
            self.state.k_m.max(0.05),
            self.state.hill_n.max(0.2),
        )
        .unwrap_or_default();

        self.phyllotaxis.set_nutrient(nutrient);
        self.phyllotaxis.set_config(cfg);
        self.phyllotaxis.set_active_count(self.state.primordia_count);

        let lsys_cfg = LSystemConfig {
            preset: self.state.lsystem_preset.to_domain(),
            iterations: self.state.lsystem_iterations.clamp(1, 6),
            step_length: 0.35,
            branch_angle_deg: self.state.branch_angle_deg,
        };
        self.lsystem.generate(lsys_cfg, &nutrient);
    }

    /// Advances developmental plastochron growth in-place.
    pub fn advance_plastochrons(&mut self, dt: f64) {
        self.phyllotaxis.step(dt);
        self.state.primordia_count = self.phyllotaxis.active_count().max(5);
    }
}

impl FiniteStateGuard for PhyllotaxisLSystemTool {
    fn is_finite_state(&self) -> bool {
        self.has_finite_parameters()
            && self
                .phyllotaxis
                .primordia()
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        None
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        self.state.scale_c = (self.state.scale_c * 0.5).clamp(0.05, 2.0);
        self.recompute();
    }

    fn clamp_parameters(&mut self) {
        let sanitize = |v: f64, lo: f64, hi: f64, def: f64| {
            if v.is_nan() { def } else { v.clamp(lo, hi) }
        };
        let d = PhyllotaxisToolState::default();
        self.state.divergence_angle_deg =
            sanitize(self.state.divergence_angle_deg, 45.0, 180.0, d.divergence_angle_deg);
        self.state.scale_c = sanitize(self.state.scale_c, 0.05, 2.0, d.scale_c);
        self.state.primordia_count = self.state.primordia_count.clamp(5, 512);
        self.state.substrate = sanitize(self.state.substrate, 0.0, 25.0, d.substrate);
        self.state.v_max = sanitize(self.state.v_max, 0.05, 25.0, d.v_max);
        self.state.k_m = sanitize(self.state.k_m, 0.05, 25.0, d.k_m);
        self.state.hill_n = sanitize(self.state.hill_n, 0.2, 8.0, d.hill_n);
        self.state.lsystem_iterations = self.state.lsystem_iterations.clamp(1, 6);
        self.state.branch_angle_deg =
            sanitize(self.state.branch_angle_deg, 5.0, 60.0, d.branch_angle_deg);
        self.recompute();
    }

    fn reset_parameters(&mut self) {
        self.state = PhyllotaxisToolState::default();
        self.recompute();
    }
}

impl InteractiveTool for PhyllotaxisLSystemTool {
    fn name(&self) -> &'static str {
        "Phyllotaxis & L-Systems"
    }

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let florets: Vec<[f64; 2]> = self
            .phyllotaxis
            .primordia()
            .iter()
            .map(|p| [p.x, p.y])
            .collect();
        let shoot: Vec<[f64; 2]> = self
            .lsystem
            .segments()
            .iter()
            .flat_map(|s| [[s[0], s[1]], [s[2], s[3]]])
            .collect();
        vec![
            ("Vogel Primordia".to_string(), florets),
            ("L-System Shoot".to_string(), shoot),
        ]
    }

    fn step_simulation(&mut self) {
        self.advance_plastochrons(0.25);
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.draw_controls(ui);
        self.draw_canvas(ui);
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory_cfg
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.state.divergence_angle_deg.to_bits().hash(&mut h);
        self.state.scale_c.to_bits().hash(&mut h);
        self.state.primordia_count.hash(&mut h);
        self.state.substrate.to_bits().hash(&mut h);
        self.state.v_max.to_bits().hash(&mut h);
        self.state.k_m.to_bits().hash(&mut h);
        self.state.hill_n.to_bits().hash(&mut h);
        self.state.kinetics.hash(&mut h);
        self.state.lsystem_preset.hash(&mut h);
        self.state.lsystem_iterations.hash(&mut h);
        self.state.branch_angle_deg.to_bits().hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: PhyllotaxisToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Phyllotaxis state: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.recompute();
        Ok(())
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.divergence_angle_deg.is_finite()
            && self.state.scale_c.is_finite()
            && self.state.substrate.is_finite()
            && self.state.v_max.is_finite()
            && self.state.k_m.is_finite()
            && self.state.hill_n.is_finite()
            && self.state.branch_angle_deg.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PhyllotaxisLSystemTool",
        display_name: "Phyllotaxis & L-Systems",
        domain: "biology",
        tags: &[
            "phyllotaxis",
            "golden-angle",
            "fibonacci",
            "l-systems",
            "botany",
            "kinetics",
        ],
        build: || Box::new(PhyllotaxisLSystemTool::default()),
    }
}

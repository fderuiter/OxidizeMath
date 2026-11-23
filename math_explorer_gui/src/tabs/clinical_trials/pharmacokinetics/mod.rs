//! Interactive Pharmacokinetics (ADME) regimen planner for `ClinicalTrialsTab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod export;
pub mod metrics;
pub mod model;
pub mod persistence;
pub mod plot;
pub mod ui;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod ui_tests;

pub use persistence::STARTER_PRESET_JSON;

use crate::framework::{
    ExportCapabilities, InteractiveTool, ToolStateBounds, TraceSnapshotManager,
};
use eframe::egui;
use egui_plot::PlotPoint;

/// Maximum number of scheduled doses in a multi-dose regimen.
pub const MAX_DOSES: usize = 64;
/// Minimum curve sample count across the simulation horizon.
pub const MIN_SAMPLE_POINTS: usize = 64;
/// Default curve sample count across the simulation horizon.
pub const DEFAULT_SAMPLE_POINTS: usize = 501;
/// Maximum curve sample count across the simulation horizon.
pub const MAX_SAMPLE_POINTS: usize = 2_000;
/// Maximum Newton-Raphson iterations when calibrating `ka` from target `T_max`.
pub const SOLVE_KA_MAX_ITER: usize = 100;
/// Concentration floor (`mg/L`) used for finite base-10 logarithmic axis projection.
pub const LOG10_CONC_FLOOR: f64 = 1.0e-4;
/// Upper slider and state bound for dose amounts (`mg`).
pub const MAX_DOSE_MG: f64 = 5_000.0;
/// Lower bound for positive first-order rate constants (`1/h`).
pub const MIN_RATE: f64 = 0.005;
/// Upper bound for absorption rate constants (`1/h`).
pub const MAX_KA: f64 = 10.0;
/// Upper bound for elimination rate constants (`1/h`).
pub const MAX_KE: f64 = 2.0;
/// Lower bound for apparent volume of distribution (`L`).
pub const MIN_VOLUME_L: f64 = 1.0;
/// Upper bound for apparent volume of distribution (`L`).
pub const MAX_VOLUME_L: f64 = 500.0;
/// Lower bound for dosing interval (`h`).
pub const MIN_INTERVAL_H: f64 = 1.0;
/// Upper bound for dosing interval (`h`).
pub const MAX_INTERVAL_H: f64 = 168.0;
/// Lower bound for simulation time horizon (`h`).
pub const MIN_HORIZON_H: f64 = 4.0;
/// Upper bound for simulation time horizon (`h`).
pub const MAX_HORIZON_H: f64 = 336.0;
/// Upper bound for delayed second-pulse lag time (`h`).
pub const MAX_LAG_H: f64 = 24.0;
/// Upper bound for target single-dose peak time `T_max` (`h`).
pub const MAX_TARGET_TMAX_H: f64 = 48.0;
/// Upper bound for minimum effective concentration (`mg/L`).
pub const MAX_MEC: f64 = 500.0;
/// Upper bound for minimum toxic concentration (`mg/L`).
pub const MAX_MTC: f64 = 1_000.0;

/// Drug entity composition mode (achiral single entity vs chiral enantiomer mixture).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EntityMode {
    /// Single chemical entity modeled via `BatemanModel`.
    SingleEntity,
    /// Two-isomer chiral mixture modeled via `EnantiomerModel`.
    Enantiomer,
}

impl EntityMode {
    /// Human-readable display label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SingleEntity => "Single Entity",
            Self::Enantiomer => "Chiral Enantiomer",
        }
    }
}

/// Oral drug release profile (immediate release vs biphasic extended release).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ReleaseMode {
    /// Single first-order absorption pulse.
    ImmediateRelease,
    /// Biphasic immediate + delayed pulse (`TwoPulseModel`).
    TwoPulse,
}

impl ReleaseMode {
    /// Human-readable display label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ImmediateRelease => "Immediate Release (IR)",
            Self::TwoPulse => "Biphasic Extended Release (Two-Pulse XR)",
        }
    }
}

/// Strategy for configuring primary absorption rate constant `ka`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AbsorptionMode {
    /// Directly specify `ka` via slider.
    DirectKa,
    /// Root-solve `ka` from target `T_max` and `ke` via `solve_ka`.
    SolveFromTMax,
}

/// Resolved four-way pharmacokinetic formulation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormulationKind {
    /// Single-entity immediate release (`BatemanModel`).
    SingleImmediate,
    /// Single-entity biphasic extended release (`TwoPulseModel<BatemanModel>`).
    SingleTwoPulse,
    /// Chiral immediate release (`EnantiomerModel`).
    ChiralImmediate,
    /// Chiral biphasic extended release (`TwoPulseModel<EnantiomerModel>`).
    ChiralTwoPulse,
}

/// Interactive ADME pharmacokinetics and multi-dose regimen planning workbench.
pub struct PharmacokineticsTool {
    /// Single-entity vs chiral mixture selector.
    pub entity_mode: EntityMode,
    /// Immediate-release vs biphasic extended-release selector.
    pub release_mode: ReleaseMode,
    /// Direct `ka` vs root-solved `ka` from `T_max`.
    pub absorption_mode: AbsorptionMode,
    /// Maintenance dose amount (`mg`).
    pub dose_mg: f64,
    /// Oral bioavailability fraction `F` (`0.0..=1.0`).
    pub bioavailability: f64,
    /// Primary (or `d`-isomer) absorption rate `ka` (`1/h`).
    pub ka: f64,
    /// Primary (or `d`-isomer) elimination rate `ke` (`1/h`).
    pub ke: f64,
    /// Apparent volume of distribution `V_d` (`L`).
    pub volume_l: f64,
    /// Target peak time `T_max` (`h`) when `AbsorptionMode::SolveFromTMax` is active.
    pub target_t_max: f64,
    /// Whether slot `0` uses a distinct loading dose `loading_dose_mg`.
    pub use_loading_dose: bool,
    /// Loading dose administered at `t = 0` (`mg`).
    pub loading_dose_mg: f64,
    /// Dosing interval `tau` (`h`).
    pub interval_hours: f64,
    /// Total scheduled dose slots (`1..=64`).
    pub n_doses: usize,
    /// Sorted 0-based indices of skipped dose slots (`< n_doses`).
    pub missed_slots: Vec<usize>,
    /// Total plasma simulation horizon (`h`).
    pub horizon_hours: f64,
    /// Number of uniform time samples (`64..=2000`).
    pub sample_points: usize,
    /// Second-pulse lag delay (`h`) for `TwoPulseModel`.
    pub lag_time_hours: f64,
    /// Fraction of dose released in first pulse (`0.0..=1.0`).
    pub f1: f64,
    /// Fraction of dose released in delayed pulse (`0.0..=1.0`).
    pub f2: f64,
    /// Fraction of `d`-isomer in chiral mixture (`0.0..=1.0`).
    pub f_d: f64,
    /// Fraction of `l`-isomer in chiral mixture (`0.0..=1.0`).
    pub f_l: f64,
    /// Absorption rate constant for `l`-isomer (`1/h`).
    pub ka_l: f64,
    /// Elimination rate constant for `l`-isomer (`1/h`).
    pub ke_l: f64,
    /// Minimum effective concentration (`mg/L`).
    pub mec: f64,
    /// Minimum toxic concentration (`mg/L`).
    pub mtc: f64,
    /// Toggle base-10 logarithmic Y-axis projection.
    pub log10_y: bool,

    pub(crate) compiled: Option<model::CompiledFormulation>,
    pub(crate) effective_ka: f64,
    pub(crate) solver_error: Option<String>,
    pub(crate) total_points: Vec<PlotPoint>,
    pub(crate) d_points: Vec<PlotPoint>,
    pub(crate) l_points: Vec<PlotPoint>,
    pub(crate) plot_total_points: Vec<PlotPoint>,
    pub(crate) plot_d_points: Vec<PlotPoint>,
    pub(crate) plot_l_points: Vec<PlotPoint>,
    pub(crate) metrics: metrics::PkMetrics,
    pub(crate) snapshot_manager: TraceSnapshotManager,
    pub(crate) card_export: export::CardExportState,
    pub(crate) last_param_summary: String,
    pub(crate) revision: u64,
}

impl Default for PharmacokineticsTool {
    fn default() -> Self {
        let mut tool = Self {
            entity_mode: EntityMode::SingleEntity,
            release_mode: ReleaseMode::ImmediateRelease,
            absorption_mode: AbsorptionMode::DirectKa,
            dose_mg: 100.0,
            bioavailability: 0.85,
            ka: 1.2,
            ke: 0.10,
            volume_l: 40.0,
            target_t_max: 2.5,
            use_loading_dose: false,
            loading_dose_mg: 200.0,
            interval_hours: 12.0,
            n_doses: 6,
            missed_slots: Vec::with_capacity(MAX_DOSES),
            horizon_hours: 72.0,
            sample_points: DEFAULT_SAMPLE_POINTS,
            lag_time_hours: 4.0,
            f1: 0.5,
            f2: 0.5,
            f_d: 0.75,
            f_l: 0.25,
            ka_l: 1.0,
            ke_l: 0.06,
            mec: 1.5,
            mtc: 5.0,
            log10_y: false,
            compiled: None,
            effective_ka: 1.2,
            solver_error: None,
            total_points: Vec::with_capacity(MAX_SAMPLE_POINTS),
            d_points: Vec::with_capacity(MAX_SAMPLE_POINTS),
            l_points: Vec::with_capacity(MAX_SAMPLE_POINTS),
            plot_total_points: Vec::with_capacity(MAX_SAMPLE_POINTS),
            plot_d_points: Vec::with_capacity(MAX_SAMPLE_POINTS),
            plot_l_points: Vec::with_capacity(MAX_SAMPLE_POINTS),
            metrics: metrics::PkMetrics::default(),
            snapshot_manager: TraceSnapshotManager::new(),
            card_export: export::CardExportState::default(),
            last_param_summary: String::new(),
            revision: 1,
        };
        tool.recalculate();
        tool
    }
}

impl PharmacokineticsTool {
    /// Resolves the active four-way formulation variant.
    #[must_use]
    pub const fn formulation(&self) -> FormulationKind {
        match (self.entity_mode, self.release_mode) {
            (EntityMode::SingleEntity, ReleaseMode::ImmediateRelease) => {
                FormulationKind::SingleImmediate
            }
            (EntityMode::SingleEntity, ReleaseMode::TwoPulse) => FormulationKind::SingleTwoPulse,
            (EntityMode::Enantiomer, ReleaseMode::ImmediateRelease) => {
                FormulationKind::ChiralImmediate
            }
            (EntityMode::Enantiomer, ReleaseMode::TwoPulse) => FormulationKind::ChiralTwoPulse,
        }
    }

    /// Formats a concise parameter summary for `TraceSnapshotManager` labels.
    #[must_use]
    pub fn param_summary(&self) -> String {
        let load = if self.use_loading_dose {
            format!(", D_load={:.0}mg", self.loading_dose_mg)
        } else {
            String::new()
        };
        let missed = if self.missed_slots.is_empty() {
            String::new()
        } else {
            format!(", missed={:?}", self.missed_slots)
        };
        format!(
            "D={:.0}mg{}, tau={:.0}h, N={}{}",
            self.dose_mg, load, self.interval_hours, self.n_doses, missed
        )
    }

    /// Rebuilds the superposition models and evaluates concentration curves and PK metrics.
    pub fn recalculate(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
        self.last_param_summary = self.param_summary();
        self.compile_models();
        self.evaluate_sampled_regimen();
    }
}

impl InteractiveTool for PharmacokineticsTool {
    fn name(&self) -> &'static str {
        "Pharmacokinetics (ADME)"
    }

    fn domain(&self) -> &'static str {
        "clinical_trials"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        export::collect_export_datasets(self)
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn state_revision(&self) -> u64 {
        self.compute_state_revision()
    }

    fn has_finite_parameters(&self) -> bool {
        self.check_finite_parameters()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_all_parameters()
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_full_state()
    }

    fn save_recovery_state(&self) -> Option<String> {
        self.serialize_recovery_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_pk_state(state, false)
    }

    fn try_load_recovery_state(&mut self, state: &str) -> Result<(), String> {
        self.load_pk_state(state, true)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        ui::show_pharmacokinetics_inside(self, ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl scientific_metadata::theory::TheoryDescribable for PharmacokineticsTool {
    fn theory_description(&self) -> String {
        format!(
            "One-compartment oral ADME Bateman kinetics $C(t) = \\frac{{F D k_a}}{{V_d (k_a - k_e)}}(e^{{-k_e t}} - e^{{-k_a t}})$ with multi-dose linear superposition $C_{{\\Sigma}}(t) = \\sum_{{i}} C(t - t_i)$, biphasic two-pulse release $f_1 C(t) + f_2 C(t - t_{{\\text{{lag}}}})$, and chiral isomer mixtures ($F = {:.2}$, $D = {:.1}$ mg, $k_a = {:.3}\\text{{ h}}^{{-1}}$, $k_e = {:.3}\\text{{ h}}^{{-1}}$, $V_d = {:.1}$ L).",
            self.bioavailability, self.dose_mg, self.effective_ka, self.ke, self.volume_l
        )
    }

    fn phonetic_description(&self) -> String {
        "One-compartment Bateman pharmacokinetics and multi-dose ADME superposition theory".into()
    }

    fn theory_citation(&self) -> String {
        "Clinical Pharmacokinetics and Pharmacodynamics: Concepts and Applications (Rowland & Tozer, 2011); Über eine Erweiterung der Theorie des radioaktiven Zerfalls (Bateman, 1910)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Bateman Function".into(),
            "First-order extravascular absorption and mono-exponential elimination model".into(),
        );
        map.insert(
            "Linear Superposition".into(),
            "Multi-dose accumulation across maintenance, loading, and missed-dose schedules".into(),
        );
        map.insert(
            "Two-Pulse & Chiral ADME".into(),
            "Biphasic extended-release pulses and stereoselective d/l-enantiomer plasma curves"
                .into(),
        );
        map
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PharmacokineticsTool",
        display_name: "Pharmacokinetics (ADME)",
        domain: "clinical_trials",
        tags: &["adme", "pharmacokinetics", "bateman", "dosing", "enantiomer"],
        build: || Box::new(PharmacokineticsTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "pharmacokinetics_adme_starter",
        title: "Pharmacokinetics (ADME)",
        description: "C(t) = F·D·ka / [Vd(ka - ke)] · (e^(-ke·t) - e^(-ka·t)): Multi-dose ADME regimen planner with therapeutic windows.",
        category: "Pharmacokinetics",
        icon: "💊",
        domain: "clinical_trials",
        tool_name: Some("PharmacokineticsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]

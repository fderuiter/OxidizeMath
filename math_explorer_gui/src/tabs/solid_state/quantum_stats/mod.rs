//! Quantum statistics, canonical thermodynamics, Debye phonons, and Thomas-Fermi screening workbench.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod export;
pub mod model;
mod presets;
pub mod ui;

#[cfg(test)]
mod tests;

use super::datasheet::DatasheetExportState;
use crate::framework::{
    CalloutManager, CalloutSpec, ExportCapabilities, InteractiveTool, ToolStateBounds,
    TraceSnapshotManager,
};
use eframe::egui;
use math_explorer::physics::solid_state::types::{ElectronVolts, Kelvin};
use math_explorer::physics::stat_mech::quantum_stats::{calculate_occupancy_ev, BoseEinstein};
use std::collections::HashMap;

pub(crate) const MIN_TEMP_K: f64 = 1.0;
pub(crate) const MAX_TEMP_K: f64 = 1200.0;
pub(crate) const MIN_MU_EV: f64 = -0.50;
pub(crate) const MAX_MU_EV: f64 = 1.00;
pub(crate) const MIN_EMIN_EV: f64 = 0.0;
pub(crate) const MAX_EMIN_EV: f64 = 0.40;
pub(crate) const MIN_ESPAN_EV: f64 = 0.10;
pub(crate) const MAX_ESPAN_EV: f64 = 1.50;
pub(crate) const MIN_SOUND_V: f64 = 0.5;
pub(crate) const MAX_SOUND_V: f64 = 12.0;
pub(crate) const MIN_OPTICAL_MEV: f64 = 5.0;
pub(crate) const MAX_OPTICAL_MEV: f64 = 120.0;
pub(crate) const MIN_THETA_D_K: f64 = 40.0;
pub(crate) const MAX_THETA_D_K: f64 = 1000.0;
pub(crate) const MIN_K_TF: f64 = 0.10;
pub(crate) const MAX_K_TF: f64 = 4.00;
pub(crate) const MIN_EXCHANGE_MEV: f64 = 0.5;
pub(crate) const MAX_EXCHANGE_MEV: f64 = 50.0;

/// Quantum statistical ensemble distribution focus mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QuantumStatsMode {
    /// Fermi-Dirac + Maxwell-Boltzmann (with Bose-Einstein overlay for $E > \mu$).
    FermiDiracCompare,
    /// Strict Bose-Einstein bosonic occupancy (enforces $\mu < E_{\min}$ condensate domain).
    BoseEinstein,
    /// Classical Maxwell-Boltzmann thermal occupancy.
    MaxwellBoltzmann,
}

impl QuantumStatsMode {
    /// Human-readable label for UI selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::FermiDiracCompare => "Fermi-Dirac & Ensemble Overlay",
            Self::BoseEinstein => "Bose-Einstein (Bosonic Mode)",
            Self::MaxwellBoltzmann => "Maxwell-Boltzmann (Classical)",
        }
    }
}

/// Active plot view in [`QuantumStatsPhononTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QuantumPhononView {
    /// Quantum statistical occupancy $f(E)$ across $(E, \mu, T)$.
    QuantumOccupancy,
    /// Acoustic/optical phonon & magnon dispersions and Debye $T^3$ heat capacity.
    PhononAndDebye,
    /// Thomas-Fermi dielectric $\epsilon(q)$ and Yukawa screened potential $V(r)$.
    ScreeningAndYukawa,
}

impl QuantumPhononView {
    /// Human-readable plot tab label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::QuantumOccupancy => "Quantum Occupancy f(E)",
            Self::PhononAndDebye => "Phonons, Magnons & Debye C_V",
            Self::ScreeningAndYukawa => "Thomas-Fermi ε(q) & Yukawa V(r)",
        }
    }
}

/// Derived canonical ensemble, phonon, screening, and spin observables.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuantumStatsMetrics {
    /// Thermal energy $k_B T$ in meV.
    pub thermal_energy_mev: f64,
    /// Canonical partition function $Z$ across the discrete 16-level spectrum.
    pub partition_z: f64,
    /// Helmholtz free energy $F$ in eV.
    pub free_energy_ev: f64,
    /// Canonical internal energy $U = \langle E \rangle$ in eV.
    pub internal_energy_ev: f64,
    /// Canonical entropy $S$ in meV/K.
    pub entropy_mev_per_k: f64,
    /// Low-temperature Debye molar heat capacity $C_V / (N k_B)$ dimensionless per atom.
    pub debye_cv_per_nk: f64,
    /// Screening length $\lambda_{\text{TF}} = 1 / k_{\text{TF}}$ in Å.
    pub screening_length_a: f64,
    /// Ferromagnetic Heisenberg dimer ground energy $H$ in meV.
    pub heisenberg_dimer_mev: f64,
    /// Bosonic canonical commutation $[b_0, b_0^\dagger]$ norm (`1.0`).
    pub boson_ccr_norm: f64,
    /// Seeded 1D lattice random-walk diffusion coefficient $D$.
    pub lattice_diffusion_d: f64,
}

/// Interactive workbench for quantum statistical mechanics, phonons, and electrostatic screening.
pub struct QuantumStatsPhononTool {
    pub(crate) temperature_k: f64,
    pub(crate) chemical_potential_ev: f64,
    pub(crate) energy_min_ev: f64,
    pub(crate) energy_span_ev: f64,
    pub(crate) sound_velocity_eva: f64,
    pub(crate) optical_omega_mev: f64,
    pub(crate) debye_theta_k: f64,
    pub(crate) k_tf_inv_a: f64,
    pub(crate) exchange_j_mev: f64,
    pub(crate) stats_mode: QuantumStatsMode,
    pub(crate) plot_view: QuantumPhononView,
    pub(crate) metrics: QuantumStatsMetrics,
    pub(crate) stat_mech_error: Option<String>,
    pub(crate) fd_points: Vec<[f64; 2]>,
    pub(crate) be_points: Vec<[f64; 2]>,
    pub(crate) mb_points: Vec<[f64; 2]>,
    pub(crate) acoustic_points: Vec<[f64; 2]>,
    pub(crate) optical_points: Vec<[f64; 2]>,
    pub(crate) magnon_points: Vec<[f64; 2]>,
    pub(crate) debye_cv_points: Vec<[f64; 2]>,
    pub(crate) dielectric_points: Vec<[f64; 2]>,
    pub(crate) yukawa_points: Vec<[f64; 2]>,
    pub(crate) snapshot_manager: TraceSnapshotManager,
    pub(crate) callout_manager: CalloutManager,
    pub(crate) datasheet_export: DatasheetExportState,
    pub(crate) last_param_summary: String,
    pub(crate) revision: u64,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QuantumStatsState {
    temperature_k: f64,
    chemical_potential_ev: f64,
    energy_min_ev: f64,
    energy_span_ev: f64,
    sound_velocity_eva: f64,
    optical_omega_mev: f64,
    debye_theta_k: f64,
    k_tf_inv_a: f64,
    exchange_j_mev: f64,
    stats_mode: QuantumStatsMode,
    plot_view: QuantumPhononView,
}

impl Default for QuantumStatsPhononTool {
    fn default() -> Self {
        let mut tool = Self {
            temperature_k: 300.0,
            chemical_potential_ev: 0.25,
            energy_min_ev: 0.0,
            energy_span_ev: 0.60,
            sound_velocity_eva: 3.5,
            optical_omega_mev: 36.0,
            debye_theta_k: 343.0,
            k_tf_inv_a: 1.25,
            exchange_j_mev: 8.0,
            stats_mode: QuantumStatsMode::FermiDiracCompare,
            plot_view: QuantumPhononView::QuantumOccupancy,
            metrics: QuantumStatsMetrics::default(),
            stat_mech_error: None,
            fd_points: Vec::with_capacity(121),
            be_points: Vec::with_capacity(121),
            mb_points: Vec::with_capacity(121),
            acoustic_points: Vec::with_capacity(81),
            optical_points: Vec::with_capacity(81),
            magnon_points: Vec::with_capacity(81),
            debye_cv_points: Vec::with_capacity(65),
            dielectric_points: Vec::with_capacity(81),
            yukawa_points: Vec::with_capacity(81),
            snapshot_manager: TraceSnapshotManager::default(),
            callout_manager: CalloutManager::default(),
            datasheet_export: DatasheetExportState::default(),
            last_param_summary: String::new(),
            revision: 1,
        };
        tool.recalculate();
        tool
    }
}

impl QuantumStatsPhononTool {
    pub(crate) fn param_summary(&self) -> String {
        format!(
            "T={:.1}K, μ={:.3}eV, Θ_D={:.0}K, k_TF={:.2}Å⁻¹",
            self.temperature_k, self.chemical_potential_ev, self.debye_theta_k, self.k_tf_inv_a
        )
    }
}

impl InteractiveTool for QuantumStatsPhononTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Quantum Statistics & Phonons"
    }
    fn domain(&self) -> &'static str {
        "solid_state"
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        export::collect_export_datasets(self)
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("thermal_energy_mev", self.metrics.thermal_energy_mev),
            ("partition_z", self.metrics.partition_z),
            ("free_energy_ev", self.metrics.free_energy_ev),
            ("internal_energy_ev", self.metrics.internal_energy_ev),
            ("debye_cv_per_nk", self.metrics.debye_cv_per_nk),
        ]
    }

    fn sync_callouts(&mut self, callouts: &[CalloutSpec]) {
        self.callout_manager.sync_specs(callouts);
    }

    fn has_finite_parameters(&self) -> bool {
        self.temperature_k.is_finite()
            && (MIN_TEMP_K..=MAX_TEMP_K).contains(&self.temperature_k)
            && self.chemical_potential_ev.is_finite()
            && (MIN_MU_EV..=MAX_MU_EV).contains(&self.chemical_potential_ev)
            && self.energy_min_ev.is_finite()
            && (MIN_EMIN_EV..=MAX_EMIN_EV).contains(&self.energy_min_ev)
            && self.energy_span_ev.is_finite()
            && (MIN_ESPAN_EV..=MAX_ESPAN_EV).contains(&self.energy_span_ev)
            && self.sound_velocity_eva.is_finite()
            && self.optical_omega_mev.is_finite()
            && self.debye_theta_k.is_finite()
            && self.k_tf_inv_a.is_finite()
            && self.exchange_j_mev.is_finite()
            && self.stat_mech_error.is_none()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.temperature_k = if self.temperature_k.is_finite() {
            self.temperature_k.clamp(MIN_TEMP_K, MAX_TEMP_K)
        } else {
            300.0
        };
        self.energy_min_ev = if self.energy_min_ev.is_finite() {
            self.energy_min_ev.clamp(MIN_EMIN_EV, MAX_EMIN_EV)
        } else {
            0.0
        };
        self.energy_span_ev = if self.energy_span_ev.is_finite() {
            self.energy_span_ev.clamp(MIN_ESPAN_EV, MAX_ESPAN_EV)
        } else {
            0.60
        };
        self.chemical_potential_ev = if self.chemical_potential_ev.is_finite() {
            self.chemical_potential_ev.clamp(MIN_MU_EV, MAX_MU_EV)
        } else {
            0.0
        };
        if self.stats_mode == QuantumStatsMode::BoseEinstein
            && self.chemical_potential_ev >= self.energy_min_ev
        {
            self.chemical_potential_ev = (self.energy_min_ev - 0.02).clamp(MIN_MU_EV, MAX_MU_EV);
        }
        self.sound_velocity_eva = self.sound_velocity_eva.clamp(MIN_SOUND_V, MAX_SOUND_V);
        self.optical_omega_mev = self
            .optical_omega_mev
            .clamp(MIN_OPTICAL_MEV, MAX_OPTICAL_MEV);
        self.debye_theta_k = self.debye_theta_k.clamp(MIN_THETA_D_K, MAX_THETA_D_K);
        self.k_tf_inv_a = self.k_tf_inv_a.clamp(MIN_K_TF, MAX_K_TF);
        self.exchange_j_mev = self
            .exchange_j_mev
            .clamp(MIN_EXCHANGE_MEV, MAX_EXCHANGE_MEV);
        self.recalculate();
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&QuantumStatsState {
            temperature_k: self.temperature_k,
            chemical_potential_ev: self.chemical_potential_ev,
            energy_min_ev: self.energy_min_ev,
            energy_span_ev: self.energy_span_ev,
            sound_velocity_eva: self.sound_velocity_eva,
            optical_omega_mev: self.optical_omega_mev,
            debye_theta_k: self.debye_theta_k,
            k_tf_inv_a: self.k_tf_inv_a,
            exchange_j_mev: self.exchange_j_mev,
            stats_mode: self.stats_mode,
            plot_view: self.plot_view,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: QuantumStatsState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid QuantumStats state JSON: {e}"))?;
        validate_quantum_stats_state(&s)?;
        self.temperature_k = s.temperature_k;
        self.chemical_potential_ev = s.chemical_potential_ev;
        self.energy_min_ev = s.energy_min_ev;
        self.energy_span_ev = s.energy_span_ev;
        self.sound_velocity_eva = s.sound_velocity_eva;
        self.optical_omega_mev = s.optical_omega_mev;
        self.debye_theta_k = s.debye_theta_k;
        self.k_tf_inv_a = s.k_tf_inv_a;
        self.exchange_j_mev = s.exchange_j_mev;
        self.stats_mode = s.stats_mode;
        self.plot_view = s.plot_view;
        self.recalculate();
        self.revision = self.revision.wrapping_add(1).max(1);
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        ui::show_quantum_stats_inside(self, parent_ui);
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

fn validate_quantum_stats_state(s: &QuantumStatsState) -> Result<(), String> {
    if !s.temperature_k.is_finite() || !(MIN_TEMP_K..=MAX_TEMP_K).contains(&s.temperature_k) {
        return Err(format!("temperature_k out of [{MIN_TEMP_K}, {MAX_TEMP_K}]"));
    }
    if !s.chemical_potential_ev.is_finite()
        || !(MIN_MU_EV..=MAX_MU_EV).contains(&s.chemical_potential_ev)
    {
        return Err(format!(
            "chemical_potential_ev out of [{MIN_MU_EV}, {MAX_MU_EV}]"
        ));
    }
    if !s.energy_min_ev.is_finite() || !(MIN_EMIN_EV..=MAX_EMIN_EV).contains(&s.energy_min_ev) {
        return Err(format!(
            "energy_min_ev out of [{MIN_EMIN_EV}, {MAX_EMIN_EV}]"
        ));
    }
    if !s.energy_span_ev.is_finite() || !(MIN_ESPAN_EV..=MAX_ESPAN_EV).contains(&s.energy_span_ev) {
        return Err(format!(
            "energy_span_ev out of [{MIN_ESPAN_EV}, {MAX_ESPAN_EV}]"
        ));
    }
    if !s.sound_velocity_eva.is_finite()
        || !(MIN_SOUND_V..=MAX_SOUND_V).contains(&s.sound_velocity_eva)
        || !s.optical_omega_mev.is_finite()
        || !(MIN_OPTICAL_MEV..=MAX_OPTICAL_MEV).contains(&s.optical_omega_mev)
        || !s.debye_theta_k.is_finite()
        || !(MIN_THETA_D_K..=MAX_THETA_D_K).contains(&s.debye_theta_k)
        || !s.k_tf_inv_a.is_finite()
        || !(MIN_K_TF..=MAX_K_TF).contains(&s.k_tf_inv_a)
        || !s.exchange_j_mev.is_finite()
        || !(MIN_EXCHANGE_MEV..=MAX_EXCHANGE_MEV).contains(&s.exchange_j_mev)
    {
        return Err("Phonon, Debye, screening, or exchange parameter out of bounds".into());
    }
    if s.stats_mode == QuantumStatsMode::BoseEinstein && s.chemical_potential_ev >= s.energy_min_ev
    {
        let err = calculate_occupancy_ev(
            BoseEinstein,
            ElectronVolts::new(s.energy_min_ev),
            ElectronVolts::new(s.chemical_potential_ev),
            Kelvin::new(s.temperature_k),
        )
        .expect_err("mu >= E_min must fail for Bose-Einstein");
        return Err(err.to_string());
    }
    Ok(())
}

impl scientific_metadata::theory::TheoryDescribable for QuantumStatsPhononTool {
    fn theory_description(&self) -> String {
        format!(
            "Quantum statistical occupancy $f(E) = [\\exp((E - \\mu)/k_B T) \\pm 1]^{{-1}}$ in physical CODATA SI units ($k_B T = {:.2}\\text{{ meV}}$ at $T = {:.1}\\text{{ K}}$), canonical ensemble $(Z = {:.3}, F = {:.3}\\text{{ eV}}, U = {:.3}\\text{{ eV}})$, Debye $T^3$ lattice heat capacity ($\\Theta_D = {:.0}\\text{{ K}}$), and Thomas-Fermi/Yukawa screening ($k_{{\\text{{TF}}}} = {:.2}\\,\\text{{\\AA}}^{{-1}}$).",
            self.metrics.thermal_energy_mev,
            self.temperature_k,
            self.metrics.partition_z,
            self.metrics.free_energy_ev,
            self.metrics.internal_energy_ev,
            self.debye_theta_k,
            self.k_tf_inv_a
        )
    }
    fn phonetic_description(&self) -> String {
        "Quantum statistical mechanics, Fermi Dirac and Bose Einstein occupancy, Debye phonons, and Thomas Fermi Yukawa screening".into()
    }
    fn theory_citation(&self) -> String {
        "Solid State Physics (Ashcroft & Mermin, 1976); Statistical Physics (Landau & Lifshitz, 1980)".into()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "QuantumStats".into(),
            "Fermi-Dirac, Bose-Einstein (mu < E), and Maxwell-Boltzmann occupancies in CODATA eV/Kelvin".into(),
        );
        map.insert(
            "DebyePhonons".into(),
            "Acoustic omega = v_s k, Einstein optical branch, Heisenberg magnons, and Debye T^3 specific heat".into(),
        );
        map.insert(
            "Screening".into(),
            "Thomas-Fermi dielectric epsilon(q) = 1 + k_TF^2 / q^2 and Yukawa screened potential exp(-k_TF r)/r".into(),
        );
        map
    }
}

// [cite:mmwave_radiotherapy_setup]

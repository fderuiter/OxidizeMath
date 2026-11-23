//! Finite-temperature BCS superconductivity, Bogoliubov quasiparticle, and NIS tunneling workbench.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod export;
pub mod model;
pub mod ui;

#[cfg(test)]
mod tests;

use super::datasheet::DatasheetExportState;
use crate::framework::{
    CalloutManager, CalloutSpec, ExportCapabilities, InteractiveTool, ToolStateBounds,
    TraceSnapshotManager,
};
use eframe::egui;
use std::collections::HashMap;

pub(crate) const MIN_TEMP_K: f64 = 0.0;
pub(crate) const MAX_TEMP_K: f64 = 40.0;
pub(crate) const MIN_DEBYE_MEV: f64 = 4.0;
pub(crate) const MAX_DEBYE_MEV: f64 = 80.0;
pub(crate) const MIN_COUPLING_NV: f64 = 0.08;
pub(crate) const MAX_COUPLING_NV: f64 = 0.60;
pub(crate) const MIN_DOS_PER_EV: f64 = 0.2;
pub(crate) const MAX_DOS_PER_EV: f64 = 5.0;
pub(crate) const MIN_GAMMA_MEV: f64 = 0.002;
pub(crate) const MAX_GAMMA_MEV: f64 = 0.50;
pub(crate) const MIN_FROHLICH_ALPHA: f64 = 0.05;
pub(crate) const MAX_FROHLICH_ALPHA: f64 = 2.0;

/// Active plot inspection lens in [`BcsSuperconductivityTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BcsPlotView {
    /// Self-consistent superconducting gap $\Delta(T)$ vs temperature $T$.
    GapVsTemperature,
    /// Bogoliubov quasiparticle dispersion $E_k$ and coherence probabilities $(u_k^2, v_k^2)$.
    QuasiparticleAndCoherence,
    /// Dynes-broadened NIS differential conductance $dI/dV$ and Fröhlich coupling $|M_q|^2$.
    NisTunnelingAndCoupling,
}

impl BcsPlotView {
    /// Human-readable tab label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::GapVsTemperature => "Δ(T) Gap Phase Curve",
            Self::QuasiparticleAndCoherence => "Bogoliubov E_k & (u_k², v_k²)",
            Self::NisTunnelingAndCoupling => "NIS dI/dV & Fröhlich |M_q|²",
        }
    }
}

///Derived BCS thermodynamic and spectroscopic observables at the current operating point.
#[derive(Debug, Clone, Copy, Default)]
pub struct BcsSummaryMetrics {
    /// Zero-temperature gap $\Delta(0)$ in meV.
    pub delta_zero_mev: f64,
    /// Operating-temperature gap $\Delta(T)$ in meV.
    pub delta_t_mev: f64,
    /// Critical temperature $T_c$ in Kelvin.
    pub tc_kelvin: f64,
    /// Dimensionless weak-coupling ratio $2\Delta(0) / (k_B T_c)$.
    pub universal_ratio: f64,
    /// Zero-bias NIS normalized differential conductance $G_{NS}(0) / G_{NN}$.
    pub zero_bias_conductance: f64,
    /// Fröhlich electron-phonon matrix element $|M(q_D)|^2$ at $q_D = 1.5\text{ \AA}^{-1}$ in $\text{eV}^2$.
    pub frohlich_mq_sq_ev2: f64,
    /// Canonical anticommutation $\{c_{0\uparrow}, c_{0\uparrow}^\dagger\}$ norm verification (`1.0`).
    pub cooper_car_norm: f64,
}

/// Interactive workbench for finite-temperature BCS superconductivity and NIS tunneling spectroscopy.
pub struct BcsSuperconductivityTool {
    pub(crate) temperature_k: f64,
    pub(crate) debye_energy_mev: f64,
    pub(crate) coupling_nv: f64,
    pub(crate) dos_per_ev: f64,
    pub(crate) dynes_gamma_mev: f64,
    pub(crate) frohlich_alpha: f64,
    pub(crate) plot_view: BcsPlotView,
    pub(crate) metrics: BcsSummaryMetrics,
    pub(crate) solver_error: Option<String>,
    pub(crate) gap_vs_temp_points: Vec<[f64; 2]>,
    pub(crate) dispersion_points: Vec<[f64; 2]>,
    pub(crate) normal_disp_points: Vec<[f64; 2]>,
    pub(crate) u_sq_points: Vec<[f64; 2]>,
    pub(crate) v_sq_points: Vec<[f64; 2]>,
    pub(crate) nis_didv_points: Vec<[f64; 2]>,
    pub(crate) dynes_dos_points: Vec<[f64; 2]>,
    pub(crate) frohlich_points: Vec<[f64; 2]>,
    pub(crate) snapshot_manager: TraceSnapshotManager,
    pub(crate) callout_manager: CalloutManager,
    pub(crate) datasheet_export: DatasheetExportState,
    pub(crate) last_param_summary: String,
    pub(crate) revision: u64,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BcsState {
    temperature_k: f64,
    debye_energy_mev: f64,
    coupling_nv: f64,
    dos_per_ev: f64,
    dynes_gamma_mev: f64,
    frohlich_alpha: f64,
    plot_view: BcsPlotView,
}

impl Default for BcsSuperconductivityTool {
    fn default() -> Self {
        let mut tool = Self {
            temperature_k: 4.2,
            debye_energy_mev: 23.7,
            coupling_nv: 0.32,
            dos_per_ev: 1.45,
            dynes_gamma_mev: 0.04,
            frohlich_alpha: 0.45,
            plot_view: BcsPlotView::GapVsTemperature,
            metrics: BcsSummaryMetrics::default(),
            solver_error: None,
            gap_vs_temp_points: Vec::with_capacity(65),
            dispersion_points: Vec::with_capacity(121),
            normal_disp_points: Vec::with_capacity(121),
            u_sq_points: Vec::with_capacity(121),
            v_sq_points: Vec::with_capacity(121),
            nis_didv_points: Vec::with_capacity(121),
            dynes_dos_points: Vec::with_capacity(121),
            frohlich_points: Vec::with_capacity(65),
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

impl BcsSuperconductivityTool {
    pub(crate) fn param_summary(&self) -> String {
        format!(
            "T={:.2}K, ħω_D={:.1}meV, N(0)V={:.2}, Γ={:.3}meV",
            self.temperature_k, self.debye_energy_mev, self.coupling_nv, self.dynes_gamma_mev
        )
    }
}

impl InteractiveTool for BcsSuperconductivityTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "BCS Superconductivity"
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
            ("delta_t_mev", self.metrics.delta_t_mev),
            ("delta_zero_mev", self.metrics.delta_zero_mev),
            ("tc_kelvin", self.metrics.tc_kelvin),
            ("universal_ratio", self.metrics.universal_ratio),
            ("zero_bias_conductance", self.metrics.zero_bias_conductance),
        ]
    }

    fn sync_callouts(&mut self, callouts: &[CalloutSpec]) {
        self.callout_manager.sync_specs(callouts);
    }

    fn has_finite_parameters(&self) -> bool {
        self.temperature_k.is_finite()
            && (MIN_TEMP_K..=MAX_TEMP_K).contains(&self.temperature_k)
            && self.debye_energy_mev.is_finite()
            && (MIN_DEBYE_MEV..=MAX_DEBYE_MEV).contains(&self.debye_energy_mev)
            && self.coupling_nv.is_finite()
            && (MIN_COUPLING_NV..=MAX_COUPLING_NV).contains(&self.coupling_nv)
            && self.dos_per_ev.is_finite()
            && (MIN_DOS_PER_EV..=MAX_DOS_PER_EV).contains(&self.dos_per_ev)
            && self.dynes_gamma_mev.is_finite()
            && (MIN_GAMMA_MEV..=MAX_GAMMA_MEV).contains(&self.dynes_gamma_mev)
            && self.frohlich_alpha.is_finite()
            && (MIN_FROHLICH_ALPHA..=MAX_FROHLICH_ALPHA).contains(&self.frohlich_alpha)
            && self.solver_error.is_none()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.temperature_k = if self.temperature_k.is_finite() {
            self.temperature_k.clamp(MIN_TEMP_K, MAX_TEMP_K)
        } else {
            4.2
        };
        self.debye_energy_mev = if self.debye_energy_mev.is_finite() {
            self.debye_energy_mev.clamp(MIN_DEBYE_MEV, MAX_DEBYE_MEV)
        } else {
            23.7
        };
        self.coupling_nv = if self.coupling_nv.is_finite() {
            self.coupling_nv.clamp(MIN_COUPLING_NV, MAX_COUPLING_NV)
        } else {
            0.32
        };
        self.dos_per_ev = if self.dos_per_ev.is_finite() {
            self.dos_per_ev.clamp(MIN_DOS_PER_EV, MAX_DOS_PER_EV)
        } else {
            1.45
        };
        self.dynes_gamma_mev = if self.dynes_gamma_mev.is_finite() {
            self.dynes_gamma_mev.clamp(MIN_GAMMA_MEV, MAX_GAMMA_MEV)
        } else {
            0.04
        };
        self.frohlich_alpha = if self.frohlich_alpha.is_finite() {
            self.frohlich_alpha
                .clamp(MIN_FROHLICH_ALPHA, MAX_FROHLICH_ALPHA)
        } else {
            0.45
        };
        self.recalculate();
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&BcsState {
            temperature_k: self.temperature_k,
            debye_energy_mev: self.debye_energy_mev,
            coupling_nv: self.coupling_nv,
            dos_per_ev: self.dos_per_ev,
            dynes_gamma_mev: self.dynes_gamma_mev,
            frohlich_alpha: self.frohlich_alpha,
            plot_view: self.plot_view,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: BcsState =
            serde_json::from_str(state).map_err(|e| format!("Invalid BCS state JSON: {e}"))?;
        validate_bcs_state(&s)?;
        self.temperature_k = s.temperature_k;
        self.debye_energy_mev = s.debye_energy_mev;
        self.coupling_nv = s.coupling_nv;
        self.dos_per_ev = s.dos_per_ev;
        self.dynes_gamma_mev = s.dynes_gamma_mev;
        self.frohlich_alpha = s.frohlich_alpha;
        self.plot_view = s.plot_view;
        self.recalculate();
        self.revision = self.revision.wrapping_add(1).max(1);
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        ui::show_bcs_inside(self, parent_ui);
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

fn validate_bcs_state(s: &BcsState) -> Result<(), String> {
    if !s.temperature_k.is_finite() || !(MIN_TEMP_K..=MAX_TEMP_K).contains(&s.temperature_k) {
        return Err(format!(
            "temperature_k out of [{MIN_TEMP_K}, {MAX_TEMP_K}]: {}",
            s.temperature_k
        ));
    }
    if !s.debye_energy_mev.is_finite()
        || !(MIN_DEBYE_MEV..=MAX_DEBYE_MEV).contains(&s.debye_energy_mev)
    {
        return Err(format!(
            "debye_energy_mev out of [{MIN_DEBYE_MEV}, {MAX_DEBYE_MEV}]"
        ));
    }
    if !s.coupling_nv.is_finite() || !(MIN_COUPLING_NV..=MAX_COUPLING_NV).contains(&s.coupling_nv) {
        return Err(format!(
            "coupling_nv out of [{MIN_COUPLING_NV}, {MAX_COUPLING_NV}]"
        ));
    }
    if !s.dos_per_ev.is_finite() || !(MIN_DOS_PER_EV..=MAX_DOS_PER_EV).contains(&s.dos_per_ev) {
        return Err(format!(
            "dos_per_ev out of [{MIN_DOS_PER_EV}, {MAX_DOS_PER_EV}]"
        ));
    }
    if !s.dynes_gamma_mev.is_finite()
        || !(MIN_GAMMA_MEV..=MAX_GAMMA_MEV).contains(&s.dynes_gamma_mev)
    {
        return Err(format!(
            "dynes_gamma_mev out of [{MIN_GAMMA_MEV}, {MAX_GAMMA_MEV}]"
        ));
    }
    if !s.frohlich_alpha.is_finite()
        || !(MIN_FROHLICH_ALPHA..=MAX_FROHLICH_ALPHA).contains(&s.frohlich_alpha)
    {
        return Err(format!(
            "frohlich_alpha out of [{MIN_FROHLICH_ALPHA}, {MAX_FROHLICH_ALPHA}]"
        ));
    }
    Ok(())
}

impl scientific_metadata::theory::TheoryDescribable for BcsSuperconductivityTool {
    fn theory_description(&self) -> String {
        format!(
            "Bardeen-Cooper-Schrieffer (BCS) finite-temperature gap equation $1 = N(0)V \\int_0^{{\\hbar\\omega_D}} \\frac{{\\tanh(E_k / 2k_B T)}}{{E_k}}\\,d\\xi_k$ with $E_k = \\sqrt{{\\xi_k^2 + \\Delta(T)^2}}$. At $T = {:.2}\\text{{ K}}$, $\\Delta(T) = {:.3}\\text{{ meV}}$, $T_c = {:.2}\\text{{ K}}$, and $2\\Delta(0)/(k_B T_c) = {:.3}$.",
            self.temperature_k, self.metrics.delta_t_mev, self.metrics.tc_kelvin, self.metrics.universal_ratio
        )
    }
    fn phonetic_description(&self) -> String {
        "B C S finite temperature superconductivity, Bogoliubov quasiparticles, and Dynes N I S tunneling conductance".into()
    }
    fn theory_citation(&self) -> String {
        "Bardeen, Cooper & Schrieffer, Phys. Rev. 108, 1175 (1957); Dynes et al., Phys. Rev. Lett. 41, 1509 (1978)".into()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "GapEquation".into(),
            "Self-consistent finite-T BCS Cooper-pair gap Delta(T) vanishing above T_c".into(),
        );
        map.insert(
            "Bogoliubov".into(),
            "Quasiparticle dispersion E_k = sqrt(xi_k^2 + Delta^2) and coherence weights u_k^2 + v_k^2 = 1".into(),
        );
        map.insert(
            "DynesNIS".into(),
            "Thermally convolved Dynes density-of-states differential tunneling conductance dI/dV"
                .into(),
        );
        map
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BcsSuperconductivityTool",
        display_name: "BCS Superconductivity",
        domain: "solid_state",
        tags: &["superconductivity", "bcs", "bogoliubov", "nis-tunneling", "frohlich"],
        build: || Box::new(BcsSuperconductivityTool::default()),
    }
}

const BCS_NIOBIUM_NIS_PRESET: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Niobium Cryogenic NIS Tunneling",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "temperature_k": 1.8,
    "debye_energy_mev": 23.7,
    "coupling_nv": 0.32,
    "dos_per_ev": 1.45,
    "dynes_gamma_mev": 0.03,
    "frohlich_alpha": 0.45,
    "plot_view": "NisTunnelingAndCoupling"
  }
}"#;

const BCS_TRANSITION_PRESET: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "BCS Gap Phase Curve & Bogoliubov Quasiparticles",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "temperature_k": 4.2,
    "debye_energy_mev": 27.5,
    "coupling_nv": 0.28,
    "dos_per_ev": 1.25,
    "dynes_gamma_mev": 0.04,
    "frohlich_alpha": 0.38,
    "plot_view": "GapVsTemperature"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bcs_niobium_nis_starter",
        title: "Niobium Cryogenic NIS Tunneling",
        description: "Dynes-broadened NIS dI/dV coherence peaks and Fröhlich electron-phonon vertex at 1.8 K.",
        category: "Condensed Matter",
        icon: "🧲",
        domain: "solid_state",
        tool_name: Some("BcsSuperconductivityTool"),
        preset_generator: || BCS_NIOBIUM_NIS_PRESET,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bcs_normal_transition_starter",
        title: "BCS Gap Δ(T) & Critical Temperature T_c",
        description: "Finite-temperature self-consistent BCS energy gap Δ(T) closing smoothly at T_c with 2Δ(0)/(k_B T_c) ≈ 3.53.",
        category: "Condensed Matter",
        icon: "❄",
        domain: "solid_state",
        tool_name: Some("BcsSuperconductivityTool"),
        preset_generator: || BCS_TRANSITION_PRESET,
    }
}

// [cite:mmwave_radiotherapy_setup]

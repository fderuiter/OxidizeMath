#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Standard Model Flavor, Electroweak, QCD Running & MSW Neutrino Lab (`StandardModelTool`).

use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds};
use eframe::egui;
use math_explorer::error::StandardModelError;
use math_explorer::physics::standard_model::{
    flavor::{construct_ckm, jarlskog_invariant, unitarity_triangle_apex},
    gauge::GaugeCouplings,
    higgs::{boson_masses, HiggsPotential},
    neutrinos::{
        matter_potential_ev2, msw_oscillation_probabilities, pmns_oscillation_probabilities,
        NeutrinoFlavor, PmnsParameters,
    },
    qcd::running_coupling,
};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

const MZ_REF_GEV: f64 = 91.1876;

/// Serializable neutrino flavor selector for [`StandardModelTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GuiNeutrinoFlavor {
    /// Electron flavor ($\nu_e$).
    Electron,
    /// Muon flavor ($\nu_\mu$).
    #[default]
    Muon,
    /// Tau flavor ($\nu_\tau$).
    Tau,
}

impl GuiNeutrinoFlavor {
    /// Converts GUI enum to kernel [`NeutrinoFlavor`].
    pub const fn to_kernel(self) -> NeutrinoFlavor {
        match self {
            Self::Electron => NeutrinoFlavor::Electron,
            Self::Muon => NeutrinoFlavor::Muon,
            Self::Tau => NeutrinoFlavor::Tau,
        }
    }

    /// Short display label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Electron => "nu_e (Electron)",
            Self::Muon => "nu_mu (Muon)",
            Self::Tau => "nu_tau (Tau)",
        }
    }
}

/// Active plot view in [`StandardModelTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SmPlotView {
    /// Three-flavor MSW matter and vacuum oscillation probabilities vs $L/E$ ($\text{km/GeV}$).
    #[default]
    MswOscillations,
    /// 1-loop QCD running coupling $\alpha_s(Q)$ vs $\log_{10}(Q/\text{GeV})$ down to the Landau pole.
    QcdRunning,
    /// Rescaled CKM Unitarity Triangle in the $(\bar{\rho}, \bar{\eta})$ complex plane.
    CkmUnitarityTriangle,
}

impl SmPlotView {
    /// Display label for UI selectors.
    pub const fn label(self) -> &'static str {
        match self {
            Self::MswOscillations => "3-Flavor PMNS + MSW Matter Oscillations vs L/E",
            Self::QcdRunning => "1-Loop QCD Running Coupling alpha_s(Q) & Landau Pole",
            Self::CkmUnitarityTriangle => "CKM Unitarity Triangle (rho_bar, eta_bar)",
        }
    }
}

/// Serializable state for [`StandardModelTool`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StandardModelState {
    /// Hypercharge coupling $g_1$.
    pub g1: f64,
    /// Weak isospin coupling $g_2$.
    pub g2: f64,
    /// Strong gauge coupling $g_s$.
    pub gs: f64,
    /// Higgs quadratic mass parameter $\mu^2 < 0$ ($\text{GeV}^2$).
    pub higgs_mu2: f64,
    /// Higgs quartic self-coupling $\lambda > 0$.
    pub higgs_lambda: f64,
    /// CKM Cabibbo angle $\theta_{12}$ (degrees).
    pub ckm_theta12_deg: f64,
    /// CKM angle $\theta_{23}$ (degrees).
    pub ckm_theta23_deg: f64,
    /// CKM angle $\theta_{13}$ (degrees).
    pub ckm_theta13_deg: f64,
    /// CKM CP-violating phase $\delta$ (degrees).
    pub ckm_delta_deg: f64,
    /// Reference strong coupling $\alpha_s(M_Z)$.
    pub alpha_s_mz: f64,
    /// Number of active quark flavors $n_f$.
    pub active_flavors: f64,
    /// Probe momentum transfer scale $Q$ ($\text{GeV}$).
    pub probe_scale_gev: f64,
    /// Initial neutrino flavor produced at the source.
    pub initial_flavor: GuiNeutrinoFlavor,
    /// PMNS solar mixing angle $\theta_{12}$ (degrees).
    pub pmns_theta12_deg: f64,
    /// PMNS atmospheric mixing angle $\theta_{23}$ (degrees).
    pub pmns_theta23_deg: f64,
    /// PMNS reactor mixing angle $\theta_{13}$ (degrees).
    pub pmns_theta13_deg: f64,
    /// PMNS CP phase $\delta_{\text{CP}}$ (degrees).
    pub pmns_delta_cp_deg: f64,
    /// Mass ordering toggle (`false` = Normal Ordering, `true` = Inverted Ordering).
    pub inverted_ordering: bool,
    /// Antineutrino beam toggle (`true` flips sign of Wolfenstein $A_{CC}$ and conjugates $U_{\text{PMNS}}$).
    pub antineutrino: bool,
    /// Neutrino beam energy $E$ ($\text{GeV}$).
    pub neutrino_energy_gev: f64,
    /// Propagation baseline $L$ ($\text{km}$).
    pub baseline_km: f64,
    /// Constant crust/mantle/core matter density $\rho$ ($\text{g/cm}^3$).
    pub matter_density_g_cm3: f64,
    /// Electron fraction $Y_e = N_e / (N_p + N_n)$.
    pub electron_fraction: f64,
    /// Selected plot mode.
    pub plot_view: SmPlotView,
}

impl Default for StandardModelState {
    fn default() -> Self {
        Self {
            g1: 0.357,
            g2: 0.652,
            gs: 1.221,
            higgs_mu2: -7840.0,
            higgs_lambda: 0.1293,
            ckm_theta12_deg: 13.04,
            ckm_theta23_deg: 2.38,
            ckm_theta13_deg: 0.201,
            ckm_delta_deg: 68.8,
            alpha_s_mz: 0.1179,
            active_flavors: 5.0,
            probe_scale_gev: 91.1876,
            initial_flavor: GuiNeutrinoFlavor::Muon,
            pmns_theta12_deg: 33.41,
            pmns_theta23_deg: 49.10,
            pmns_theta13_deg: 8.54,
            pmns_delta_cp_deg: 197.0,
            inverted_ordering: false,
            antineutrino: false,
            neutrino_energy_gev: 2.5,
            baseline_km: 1300.0,
            matter_density_g_cm3: 2.8,
            electron_fraction: 0.5,
            plot_view: SmPlotView::MswOscillations,
        }
    }
}

/// Live derived Standard Model observables for UI display.
#[derive(Debug, Clone)]
pub(crate) struct SmObservables {
    pub(crate) sin2_theta_w: f64,
    pub(crate) cos_theta_w: f64,
    pub(crate) vev_gev: f64,
    pub(crate) m_w_gev: f64,
    pub(crate) m_z_gev: f64,
    pub(crate) ckm_jarlskog: f64,
    pub(crate) ckm_apex: (f64, f64),
    pub(crate) qcd_alpha_probe: Result<f64, StandardModelError>,
    pub(crate) landau_pole_gev: f64,
    pub(crate) msw_probs: [f64; 3],
    pub(crate) vac_probs: [f64; 3],
    pub(crate) matter_acc_ev2: f64,
}

/// Interactive Standard Model electroweak, CKM, QCD running, and MSW neutrino explorer.
pub struct StandardModelTool {
    pub(crate) state: StandardModelState,
    pub(crate) revision: u64,
}

impl Default for StandardModelTool {
    fn default() -> Self {
        Self {
            state: StandardModelState::default(),
            revision: 1,
        }
    }
}

impl StandardModelTool {
    pub(crate) fn clamp_in_place(&mut self) {
        let s = &mut self.state;
        s.g1 = fclamp(s.g1, 0.05, 1.5, 0.357);
        s.g2 = fclamp(s.g2, 0.05, 1.8, 0.652);
        s.gs = fclamp(s.gs, 0.1, 3.0, 1.221);
        s.higgs_mu2 = fclamp(s.higgs_mu2, -30_000.0, -50.0, -7840.0);
        s.higgs_lambda = fclamp(s.higgs_lambda, 0.01, 2.0, 0.1293);
        s.ckm_theta12_deg = fclamp(s.ckm_theta12_deg, 0.1, 45.0, 13.04);
        s.ckm_theta23_deg = fclamp(s.ckm_theta23_deg, 0.05, 30.0, 2.38);
        s.ckm_theta13_deg = fclamp(s.ckm_theta13_deg, 0.01, 15.0, 0.201);
        s.ckm_delta_deg = fclamp(s.ckm_delta_deg, 0.0, 360.0, 68.8);
        s.alpha_s_mz = fclamp(s.alpha_s_mz, 0.05, 0.25, 0.1179);
        s.active_flavors = fclamp(s.active_flavors, 0.0, 6.0, 5.0);
        s.probe_scale_gev = fclamp(s.probe_scale_gev, 0.05, 2000.0, 91.1876);
        s.pmns_theta12_deg = fclamp(s.pmns_theta12_deg, 0.0, 90.0, 33.41);
        s.pmns_theta23_deg = fclamp(s.pmns_theta23_deg, 0.0, 90.0, 49.10);
        s.pmns_theta13_deg = fclamp(s.pmns_theta13_deg, 0.0, 45.0, 8.54);
        s.pmns_delta_cp_deg = fclamp(s.pmns_delta_cp_deg, 0.0, 360.0, 197.0);
        s.neutrino_energy_gev = fclamp(s.neutrino_energy_gev, 0.1, 40.0, 2.5);
        s.baseline_km = fclamp(s.baseline_km, 1.0, 13_000.0, 1300.0);
        s.matter_density_g_cm3 = fclamp(s.matter_density_g_cm3, 0.0, 15.0, 2.8);
        s.electron_fraction = fclamp(s.electron_fraction, 0.1, 0.9, 0.5);
    }

    pub(crate) fn pmns_params(&self) -> PmnsParameters {
        let mut p = if self.state.inverted_ordering {
            PmnsParameters::pdg_inverted_ordering()
        } else {
            PmnsParameters::pdg_normal_ordering()
        };
        p.theta12 = self.state.pmns_theta12_deg.to_radians();
        p.theta23 = self.state.pmns_theta23_deg.to_radians();
        p.theta13 = self.state.pmns_theta13_deg.to_radians();
        p.delta_cp = self.state.pmns_delta_cp_deg.to_radians();
        p
    }

    pub(crate) fn landau_pole_scale_gev(&self) -> f64 {
        let b0 = (33.0 - 2.0 * self.state.active_flavors) / (12.0 * PI);
        if b0 <= 1e-9 {
            return 0.0;
        }
        MZ_REF_GEV * (-1.0 / (2.0 * b0 * self.state.alpha_s_mz)).exp()
    }

    fn evaluate_neutrino_observables(&self) -> ([f64; 3], [f64; 3], f64) {
        let s = &self.state;
        let pmns = self.pmns_params();
        let flv = s.initial_flavor.to_kernel();
        let msw_probs = msw_oscillation_probabilities(
            &pmns,
            flv,
            s.baseline_km,
            s.neutrino_energy_gev,
            s.matter_density_g_cm3,
            s.electron_fraction,
            s.antineutrino,
        )
        .unwrap_or([1.0, 0.0, 0.0]);
        let vac_probs = pmns_oscillation_probabilities(
            &pmns,
            flv,
            s.baseline_km,
            s.neutrino_energy_gev,
            s.antineutrino,
        )
        .unwrap_or([1.0, 0.0, 0.0]);
        let matter_acc_ev2 = matter_potential_ev2(
            s.neutrino_energy_gev,
            s.matter_density_g_cm3,
            s.electron_fraction,
            s.antineutrino,
        )
        .unwrap_or(0.0);
        (msw_probs, vac_probs, matter_acc_ev2)
    }

    pub(crate) fn evaluate_observables(&self) -> SmObservables {
        let s = &self.state;
        let couplings = GaugeCouplings::new(s.g1, s.g2, s.gs);
        let (cos_w, sin_w) = couplings.weak_mixing_angle();
        let higgs = HiggsPotential {
            mu2: s.higgs_mu2,
            lambda: s.higgs_lambda,
        };
        let vev_gev = higgs.vev().unwrap_or(0.0);
        let (m_w_gev, m_z_gev) = boson_masses(vev_gev, &couplings);

        let ckm = construct_ckm(
            s.ckm_theta12_deg.to_radians(),
            s.ckm_theta23_deg.to_radians(),
            s.ckm_theta13_deg.to_radians(),
            s.ckm_delta_deg.to_radians(),
        );
        let ckm_jarlskog = jarlskog_invariant(&ckm);
        let ckm_apex = unitarity_triangle_apex(&ckm).unwrap_or((0.0, 0.0));

        let qcd_alpha_probe = running_coupling(
            MZ_REF_GEV,
            s.alpha_s_mz,
            s.probe_scale_gev,
            s.active_flavors,
        );
        let (msw_probs, vac_probs, matter_acc_ev2) = self.evaluate_neutrino_observables();

        SmObservables {
            sin2_theta_w: sin_w * sin_w,
            cos_theta_w: cos_w,
            vev_gev,
            m_w_gev,
            m_z_gev,
            ckm_jarlskog,
            ckm_apex,
            qcd_alpha_probe,
            landau_pole_gev: self.landau_pole_scale_gev(),
            msw_probs,
            vac_probs,
            matter_acc_ev2,
        }
    }
}

fn fclamp(val: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(lo, hi)
    } else {
        fallback
    }
}

impl FiniteStateGuard for StandardModelTool {
    fn is_finite_state(&self) -> bool {
        let s = &self.state;
        s.g1.is_finite()
            && s.g2.is_finite()
            && s.gs.is_finite()
            && s.higgs_mu2.is_finite()
            && s.higgs_lambda.is_finite()
            && s.ckm_theta12_deg.is_finite()
            && s.ckm_theta23_deg.is_finite()
            && s.ckm_theta13_deg.is_finite()
            && s.ckm_delta_deg.is_finite()
            && s.alpha_s_mz.is_finite()
            && s.active_flavors.is_finite()
            && s.probe_scale_gev.is_finite()
            && s.pmns_theta12_deg.is_finite()
            && s.pmns_theta23_deg.is_finite()
            && s.pmns_theta13_deg.is_finite()
            && s.pmns_delta_cp_deg.is_finite()
            && s.neutrino_energy_gev.is_finite()
            && s.baseline_km.is_finite()
            && s.matter_density_g_cm3.is_finite()
            && s.electron_fraction.is_finite()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) {
        self.clamp_in_place();
        self.revision = self.revision.wrapping_add(1);
    }

    fn reset_parameters(&mut self) {
        self.state = StandardModelState::default();
        self.revision = self.revision.wrapping_add(1);
    }
}

impl InteractiveTool for StandardModelTool {
    fn name(&self) -> &'static str {
        "Standard Model & Neutrinos"
    }

    fn domain(&self) -> &'static str {
        "subatomic"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, raw: &str) -> Result<(), String> {
        self.state_bounds().validate_json(raw)?;
        let loaded: StandardModelState =
            serde_json::from_str(raw).map_err(|e| format!("Invalid Standard Model JSON: {e}"))?;
        self.state = loaded;
        self.clamp_in_place();
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.compute_plot_datasets()
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_standard_model_tool(ui);
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

impl scientific_metadata::theory::TheoryDescribable for StandardModelTool {
    fn theory_description(&self) -> String {
        "Simulates SU(3)_C x SU(2)_L x U(1)_Y electroweak symmetry breaking (Weinberg angle, Higgs VEV, W/Z boson masses), CKM quark mixing and Jarlskog CP violation, 1-loop QCD asymptotic freedom and infrared Landau pole scaling, and three-flavor PMNS + Wolfenstein-Mikheyev-Smirnov (MSW) matter neutrino oscillations.".into()
    }

    fn phonetic_description(&self) -> String {
        "Electroweak Higgs symmetry breaking, CKM unitarity triangle, QCD asymptotic freedom, and MSW neutrino oscillations".into()
    }

    fn theory_citation(&self) -> String {
        "Weinberg (1967) Phys. Rev. Lett. 19, 1264 (10.1103/PhysRevLett.19.1264); Gross & Wilczek (1973) (10.1103/PhysRevLett.30.1343); Wolfenstein (1978) (10.1103/PhysRevD.17.2369)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "StandardModelTool",
        display_name: "Standard Model & Neutrinos",
        domain: "subatomic",
        tags: &["standard_model", "higgs", "ckm", "qcd", "pmns", "msw", "neutrinos"],
        build: || Box::new(StandardModelTool::default()),
    }
}

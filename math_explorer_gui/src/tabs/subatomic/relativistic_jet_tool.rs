#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Relativistic Observer & Synchrotron Jet Lab (`RelativisticJetTool`).

use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds};
use eframe::egui;
use math_explorer::physics::high_energy::{
    apparent_transverse_beta, calculate_lorentz_factor, doppler_beaming_factor,
    fluid_dynamics::{shock_compression_ratio, specific_enthalpy},
    radiation::{inverse_compton_spectral_index, synchrotron_power},
    relativistic_doppler_factor,
    statistics::li_ma_significance,
    FourVector, C,
};
use serde::{Deserialize, Serialize};

/// Plot mode selectable in [`RelativisticJetTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum JetPlotView {
    /// Relativistic Doppler factor $\delta(\theta)$ and apparent transverse speed $\beta_{\text{app}}(\theta)$.
    #[default]
    AngularBeamingAndSuperluminal,
    /// Lorentz factor $\gamma(\beta)$ and normalized synchrotron emission vs speed $\beta = v/c$.
    LorentzAndSynchrotronVsBeta,
}

impl JetPlotView {
    /// Display label for UI combo boxes.
    pub const fn label(self) -> &'static str {
        match self {
            Self::AngularBeamingAndSuperluminal => {
                "Doppler Factor delta(theta) & Apparent Speed beta_app(theta)"
            }
            Self::LorentzAndSynchrotronVsBeta => {
                "Lorentz Factor gamma(beta) & Normalized Synchrotron Power"
            }
        }
    }
}

/// Serializable state for [`RelativisticJetTool`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelativisticJetState {
    /// Bulk jet speed fraction $\beta = v / c \in [0.0, 0.995]$.
    pub beta: f64,
    /// Observer line-of-sight viewing angle $\theta \in [0.0, 180.0]$ (degrees).
    pub viewing_angle_deg: f64,
    /// Particle rest mass $m$ (kg) used for 4-momentum shell validation.
    pub rest_mass_kg: f64,
    /// Dimensionless comoving $x$-momentum $p_x / (m c)$ (or $p_x / c$ when $m = 0$).
    pub comoving_px_factor: f64,
    /// Magnetic field energy density $U_B$ ($\text{J/m}^3$).
    pub magnetic_energy_density_ub: f64,
    /// Relativistic electron power-law index $p > 1$ ($dN/dE \propto E^{-p}$).
    pub electron_power_law_p: f64,
    /// Steady continuous jet (`k = 2`) vs discrete moving plasmoid (`k = 3`).
    pub continuous_jet: bool,
    /// Relativistic fluid adiabatic index $\hat{\gamma} > 1$ ($4/3$ for ultrarelativistic plasma).
    pub adiabatic_index: f64,
    /// Comoving rest-mass density $\rho$ ($\text{kg/m}^3$).
    pub fluid_density: f64,
    /// Dimensionless thermal pressure ratio $P / (\rho c^2) \ge 0$.
    pub pressure_over_rho_c2: f64,
    /// On-source photon count $N_{\text{on}}$.
    pub photon_n_on: u32,
    /// Off-source background photon count $N_{\text{off}}$.
    pub photon_n_off: u32,
    /// On/off exposure ratio $\alpha > 0$.
    pub exposure_alpha: f64,
    /// Active plot view.
    pub plot_view: JetPlotView,
}

impl Default for RelativisticJetState {
    fn default() -> Self {
        Self {
            beta: 0.96,
            viewing_angle_deg: 10.0,
            rest_mass_kg: 1.0,
            comoving_px_factor: 0.5,
            magnetic_energy_density_ub: 1.5,
            electron_power_law_p: 2.4,
            continuous_jet: true,
            adiabatic_index: 4.0 / 3.0,
            fluid_density: 1.0,
            pressure_over_rho_c2: 0.25,
            photon_n_on: 185,
            photon_n_off: 220,
            exposure_alpha: 0.25,
            plot_view: JetPlotView::AngularBeamingAndSuperluminal,
        }
    }
}

/// Live derived relativistic observer, radiation, fluid, and Li-Ma observables.
#[derive(Debug, Clone, Copy)]
pub(crate) struct JetObservables {
    pub(crate) gamma: f64,
    pub(crate) doppler_delta: f64,
    pub(crate) beta_apparent: f64,
    pub(crate) beaming_amplification: f64,
    pub(crate) comoving_p: FourVector,
    pub(crate) boosted_p: FourVector,
    pub(crate) momentum_shell_valid: bool,
    pub(crate) sync_power_watts: f64,
    pub(crate) ic_spectral_index: f64,
    pub(crate) shock_ratio: f64,
    pub(crate) enthalpy_over_c2: f64,
    pub(crate) li_ma_sigma: f64,
}

/// Interactive Relativistic Observer, Synchrotron Jet, Shock Hydro, and Li-Ma Lab.
pub struct RelativisticJetTool {
    pub(crate) state: RelativisticJetState,
    pub(crate) revision: u64,
}

impl Default for RelativisticJetTool {
    fn default() -> Self {
        Self {
            state: RelativisticJetState::default(),
            revision: 1,
        }
    }
}

impl RelativisticJetTool {
    pub(crate) fn clamp_in_place(&mut self) {
        let s = &mut self.state;
        s.beta = jclamp(s.beta, 0.0, 0.995, 0.96);
        s.viewing_angle_deg = jclamp(s.viewing_angle_deg, 0.0, 180.0, 10.0);
        s.rest_mass_kg = jclamp(s.rest_mass_kg, 0.0, 50.0, 1.0);
        s.comoving_px_factor = jclamp(s.comoving_px_factor, -5.0, 5.0, 0.5);
        s.magnetic_energy_density_ub = jclamp(s.magnetic_energy_density_ub, 1e-4, 200.0, 1.5);
        s.electron_power_law_p = jclamp(s.electron_power_law_p, 1.05, 5.0, 2.4);
        s.adiabatic_index = jclamp(s.adiabatic_index, 1.05, 2.5, 4.0 / 3.0);
        s.fluid_density = jclamp(s.fluid_density, 0.01, 100.0, 1.0);
        s.pressure_over_rho_c2 = jclamp(s.pressure_over_rho_c2, 0.0, 10.0, 0.25);
        s.photon_n_on = s.photon_n_on.min(500_000);
        s.photon_n_off = s.photon_n_off.min(500_000);
        s.exposure_alpha = jclamp(s.exposure_alpha, 0.02, 5.0, 0.25);
    }

    pub(crate) fn evaluate_observables(&self) -> JetObservables {
        let s = &self.state;
        let gamma = calculate_lorentz_factor(s.beta * C).unwrap_or(1.0);
        let theta_rad = s.viewing_angle_deg.to_radians();
        let doppler_delta = relativistic_doppler_factor(s.beta, theta_rad).unwrap_or(1.0);
        let beta_apparent = apparent_transverse_beta(s.beta, theta_rad).unwrap_or(0.0);
        let ic_spectral_index =
            inverse_compton_spectral_index(s.electron_power_law_p).unwrap_or(0.7);
        let beaming_amplification =
            doppler_beaming_factor(doppler_delta, ic_spectral_index, s.continuous_jet)
                .unwrap_or(1.0);

        let scale_mc = if s.rest_mass_kg > 1e-12 {
            s.rest_mass_kg * C
        } else {
            C
        };
        let px = s.comoving_px_factor * scale_mc;
        let mc = s.rest_mass_kg * C;
        let e_over_c = (mc * mc + px * px).sqrt();
        let comoving_p = FourVector::new(e_over_c, px, 0.0, 0.0);
        let boosted_p = comoving_p
            .lorentz_boost(-s.beta, 0.0, 0.0)
            .unwrap_or(comoving_p);
        let momentum_shell_valid = comoving_p
            .is_valid_momentum(s.rest_mass_kg)
            .unwrap_or(false)
            && boosted_p.is_valid_momentum(s.rest_mass_kg).unwrap_or(false);

        let sync_power_watts =
            synchrotron_power(s.magnetic_energy_density_ub, s.beta, gamma).unwrap_or(0.0);
        let shock_ratio = shock_compression_ratio(s.adiabatic_index).unwrap_or(4.0);
        let pressure_pa = s.pressure_over_rho_c2 * s.fluid_density * C * C;
        let enthalpy_over_c2 =
            specific_enthalpy(s.adiabatic_index, pressure_pa, s.fluid_density).unwrap_or(1.0);
        let li_ma_sigma = li_ma_significance(
            f64::from(s.photon_n_on),
            f64::from(s.photon_n_off),
            s.exposure_alpha,
        )
        .unwrap_or(0.0);

        JetObservables {
            gamma,
            doppler_delta,
            beta_apparent,
            beaming_amplification,
            comoving_p,
            boosted_p,
            momentum_shell_valid,
            sync_power_watts,
            ic_spectral_index,
            shock_ratio,
            enthalpy_over_c2,
            li_ma_sigma,
        }
    }
}

fn jclamp(val: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(lo, hi)
    } else {
        fallback
    }
}

impl FiniteStateGuard for RelativisticJetTool {
    fn is_finite_state(&self) -> bool {
        let s = &self.state;
        s.beta.is_finite()
            && s.viewing_angle_deg.is_finite()
            && s.rest_mass_kg.is_finite()
            && s.comoving_px_factor.is_finite()
            && s.magnetic_energy_density_ub.is_finite()
            && s.electron_power_law_p.is_finite()
            && s.adiabatic_index.is_finite()
            && s.fluid_density.is_finite()
            && s.pressure_over_rho_c2.is_finite()
            && s.exposure_alpha.is_finite()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) {
        self.clamp_in_place();
        self.revision = self.revision.wrapping_add(1);
    }

    fn reset_parameters(&mut self) {
        self.state = RelativisticJetState::default();
        self.revision = self.revision.wrapping_add(1);
    }
}

impl InteractiveTool for RelativisticJetTool {
    fn name(&self) -> &'static str {
        "Relativistic Jet & Observer Lab"
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
        let loaded: RelativisticJetState =
            serde_json::from_str(raw).map_err(|e| format!("Invalid Relativistic Jet JSON: {e}"))?;
        self.state = loaded;
        self.clamp_in_place();
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.compute_jet_datasets()
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_jet_tool(ui);
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

impl scientific_metadata::theory::TheoryDescribable for RelativisticJetTool {
    fn theory_description(&self) -> String {
        "Models special-relativistic FourVector Lorentz boosts preserving p^mu p_mu = -(mc)^2, relativistic Doppler beaming delta = [gamma(1 - beta cos theta)]^(-1), apparent superluminal motion beta_app, synchrotron & inverse-Compton emission, Rankine-Hugoniot shock compression, relativistic specific enthalpy, and Li & Ma (1983) gamma-ray detection significance.".into()
    }

    fn phonetic_description(&self) -> String {
        "Relativistic Lorentz boosts, Doppler jet beaming, synchrotron power, shock hydrodynamics, and Li-Ma significance".into()
    }

    fn theory_citation(&self) -> String {
        "Rees (1966) Nature 211, 468 (10.1038/211468a0); Li & Ma (1983) ApJ 272, 317 (10.1086/161295)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "RelativisticJetTool",
        display_name: "Relativistic Jet & Observer Lab",
        domain: "subatomic",
        tags: &["relativity", "doppler", "synchrotron", "blazar", "lorentz", "li_ma"],
        build: || Box::new(RelativisticJetTool::default()),
    }
}

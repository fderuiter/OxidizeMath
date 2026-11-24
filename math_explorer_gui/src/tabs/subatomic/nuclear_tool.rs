#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Nuclear Stability & Decay/Reactions interactive tool (`NuclearStabilityTool`).

use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata, ToolStateBounds};
use eframe::egui;
use math_explorer::constants::{LIGHT_SPEED, NEUTRON_MASS, PROTON_MASS};
use math_explorer::physics::nuclear::{
    decay, models::shell, properties, reactions, AtomicNumber, BatemanChain, LiquidDropBreakdown,
    LiquidDropModel, MassNumber,
};
use serde::{Deserialize, Serialize};

/// Plot mode selectable in [`NuclearStabilityTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NuclearPlotMode {
    /// Four-species Bateman decay-chain population curves $N_k(t)$.
    #[default]
    BatemanDecayChain,
    /// Breit-Wigner nuclear resonance cross-section profile $\sigma(E)$.
    BreitWignerResonance,
    /// Weizsäcker binding energy per nucleon $B(Z, A)/A$ along the current isobar $A = Z + N$.
    IsobarParabola,
}

impl NuclearPlotMode {
    /// Human-readable label for UI combo boxes.
    pub const fn label(self) -> &'static str {
        match self {
            Self::BatemanDecayChain => "Bateman 4-Step Decay Chain N_k(t)",
            Self::BreitWignerResonance => "Breit-Wigner Resonance sigma(E)",
            Self::IsobarParabola => "Weizsacker Isobar Curve B(Z, A)/A",
        }
    }
}

/// Serializable state for [`NuclearStabilityTool`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NuclearStabilityState {
    /// Selected proton count $Z \in [1, 96]$.
    pub selected_z: u32,
    /// Selected neutron count $N \in [0, 150]$.
    pub selected_n: u32,
    /// Shell-model orbital angular momentum quantum number $l \in [0, 6]$.
    pub orbital_l: u32,
    /// Spin-orbit alignment (`true` => $j = l + 1/2$, `false` => $j = |l - 1/2|$).
    pub spin_aligned: bool,
    /// Alpha-particle tunneling speed fraction $\beta_\alpha = v / c \in [0.01, 0.25]$.
    pub alpha_beta: f64,
    /// Compound-nucleus resonance energy $E_{\text{res}}$ (MeV).
    pub resonance_energy_mev: f64,
    /// Compound-nucleus decay width $\Gamma$ (MeV).
    pub resonance_gamma_mev: f64,
    /// Incident projectile probe energy $E$ (MeV).
    pub probe_energy_mev: f64,
    /// Parent half-life $T_{1/2,1}$ (s) in the 4-step Bateman chain.
    pub half_life_1: f64,
    /// First daughter half-life $T_{1/2,2}$ (s).
    pub half_life_2: f64,
    /// Grand-daughter half-life $T_{1/2,3}$ (s), decaying to stable $X_4$.
    pub half_life_3: f64,
    /// Decay simulation horizon $t_{\max}$ (s).
    pub max_decay_time: f64,
    /// Active lower plot view.
    pub plot_mode: NuclearPlotMode,
}

impl Default for NuclearStabilityState {
    fn default() -> Self {
        Self {
            selected_z: 26,
            selected_n: 30,
            orbital_l: 3,
            spin_aligned: true,
            alpha_beta: 0.055,
            resonance_energy_mev: 8.0,
            resonance_gamma_mev: 1.4,
            probe_energy_mev: 8.0,
            half_life_1: 14.0,
            half_life_2: 7.0,
            half_life_3: 18.0,
            max_decay_time: 70.0,
            plot_mode: NuclearPlotMode::BatemanDecayChain,
        }
    }
}

/// Live derived nuclear observables for the currently selected $(Z, N)$ nuclide.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NuclideObservables {
    pub(crate) mass_a: u32,
    pub(crate) breakdown: LiquidDropBreakdown,
    pub(crate) radius_fm: f64,
    pub(crate) density_fm3: f64,
    pub(crate) spin_orbit_j: f64,
    pub(crate) spin_orbit_shift: f64,
    pub(crate) gamow_eta: f64,
    pub(crate) alpha_q_mev: f64,
    pub(crate) remaining_parent_pct: f64,
    pub(crate) resonance_sigma: f64,
}

/// Interactive `(Z, N)` nuclide chart, Weizsäcker breakdown, Bateman decay, and resonance explorer.
pub struct NuclearStabilityTool {
    pub(crate) state: NuclearStabilityState,
    pub(crate) revision: u64,
    pub(crate) focused_canvas: Option<egui::Id>,
}

impl Default for NuclearStabilityTool {
    fn default() -> Self {
        Self {
            state: NuclearStabilityState::default(),
            revision: 1,
            focused_canvas: None,
        }
    }
}

impl NuclearStabilityTool {
    #[cfg(test)]
    pub(crate) fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    pub(crate) fn clamp_in_place(&mut self) {
        let s = &mut self.state;
        s.selected_z = s.selected_z.clamp(1, 96);
        s.selected_n = s.selected_n.clamp(0, 150);
        s.orbital_l = s.orbital_l.clamp(0, 6);
        s.alpha_beta = finite_clamp(s.alpha_beta, 0.01, 0.25, 0.055);
        s.resonance_energy_mev = finite_clamp(s.resonance_energy_mev, 0.5, 30.0, 8.0);
        s.resonance_gamma_mev = finite_clamp(s.resonance_gamma_mev, 0.1, 10.0, 1.4);
        s.probe_energy_mev = finite_clamp(s.probe_energy_mev, 0.1, 35.0, 8.0);
        s.half_life_1 = finite_clamp(s.half_life_1, 0.5, 150.0, 14.0);
        s.half_life_2 = finite_clamp(s.half_life_2, 0.5, 150.0, 7.0);
        s.half_life_3 = finite_clamp(s.half_life_3, 0.5, 150.0, 18.0);
        s.max_decay_time = finite_clamp(s.max_decay_time, 5.0, 300.0, 70.0);
    }

    pub(crate) fn evaluate_observables(&self) -> Option<NuclideObservables> {
        let s = &self.state;
        let z = AtomicNumber::new(s.selected_z);
        let mass_a = s.selected_z + s.selected_n;
        let a = MassNumber::new(mass_a).ok()?;
        let ldm = LiquidDropModel::new();
        let breakdown = ldm.binding_energy_breakdown(z, a).ok()?;
        let radius_fm = properties::calculate_radius(a);
        let density_fm3 = properties::calculate_nucleon_density(a).ok()?;

        let l_f = f64::from(s.orbital_l);
        let j_f = if s.orbital_l == 0 || s.spin_aligned {
            l_f + 0.5
        } else {
            l_f - 0.5
        };
        let spin_orbit_shift = shell::spin_orbit_coupling(l_f, 0.5, j_f);

        let z_daughter = AtomicNumber::new(s.selected_z.saturating_sub(2).max(1));
        let gamow_eta =
            decay::gamow_factor(z_daughter, AtomicNumber::new(2), s.alpha_beta * LIGHT_SPEED)
                .unwrap_or(0.0);
        let alpha_q_mev = Self::compute_alpha_q_value(&ldm, s.selected_z, s.selected_n, &breakdown);
        let remaining_parent_pct =
            decay::calculate_remaining(100.0, s.half_life_1, 0.5 * s.max_decay_time).unwrap_or(0.0);
        let resonance_sigma = reactions::breit_wigner(
            s.probe_energy_mev,
            s.resonance_energy_mev,
            s.resonance_gamma_mev,
        )
        .unwrap_or(0.0);

        Some(NuclideObservables {
            mass_a,
            breakdown,
            radius_fm,
            density_fm3,
            spin_orbit_j: j_f,
            spin_orbit_shift,
            gamow_eta,
            alpha_q_mev,
            remaining_parent_pct,
            resonance_sigma,
        })
    }

    fn compute_alpha_q_value(
        ldm: &LiquidDropModel,
        z: u32,
        n: u32,
        parent_be: &LiquidDropBreakdown,
    ) -> f64 {
        if z <= 2 || n < 2 {
            return 0.0;
        }
        let Ok(a_d) = MassNumber::new((z - 2) + (n - 2)) else {
            return 0.0;
        };
        let Ok(be_daughter) = ldm.binding_energy_breakdown(AtomicNumber::new(z - 2), a_d) else {
            return 0.0;
        };
        let m_parent = f64::from(z) * PROTON_MASS + f64::from(n) * NEUTRON_MASS - parent_be.total;
        let m_daughter =
            f64::from(z - 2) * PROTON_MASS + f64::from(n - 2) * NEUTRON_MASS - be_daughter.total;
        let m_alpha = 2.0 * PROTON_MASS + 2.0 * NEUTRON_MASS - 28.295_66;
        reactions::q_value(&[m_parent], &[m_daughter, m_alpha])
    }

    pub(crate) fn compute_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.state.plot_mode {
            NuclearPlotMode::BatemanDecayChain => self.bateman_datasets(),
            NuclearPlotMode::BreitWignerResonance => self.breit_wigner_datasets(),
            NuclearPlotMode::IsobarParabola => self.isobar_datasets(),
        }
    }

    fn bateman_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let s = &self.state;
        let Ok(chain) = BatemanChain::<4>::from_half_lives(
            [
                Some(s.half_life_1),
                Some(s.half_life_2),
                Some(s.half_life_3),
                None,
            ],
            [100.0, 0.0, 0.0, 0.0],
        ) else {
            return Vec::new();
        };
        let samples = 96usize;
        let mut curves = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for i in 0..samples {
            let t = s.max_decay_time * (i as f64) / ((samples - 1) as f64);
            if let Ok(pops) = chain.populations_at(t) {
                for k in 0..4 {
                    curves[k].push([t, pops[k]]);
                }
            }
        }
        let [c0, c1, c2, c3] = curves;
        vec![
            ("Parent N_1(t)".into(), c0),
            ("Daughter N_2(t)".into(), c1),
            ("Granddaughter N_3(t)".into(), c2),
            ("Stable Endpoint N_4(t)".into(), c3),
        ]
    }

    fn breit_wigner_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let s = &self.state;
        let e_min = (s.resonance_energy_mev - 4.0 * s.resonance_gamma_mev).max(0.05);
        let e_max = s.resonance_energy_mev + 4.0 * s.resonance_gamma_mev;
        let samples = 96usize;
        let mut pts = Vec::with_capacity(samples);
        for i in 0..samples {
            let e = e_min + (e_max - e_min) * (i as f64) / ((samples - 1) as f64);
            if let Ok(sigma) =
                reactions::breit_wigner(e, s.resonance_energy_mev, s.resonance_gamma_mev)
            {
                pts.push([e, sigma]);
            }
        }
        vec![("Breit-Wigner Resonance Profile".into(), pts)]
    }

    fn isobar_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mass_a = (self.state.selected_z + self.state.selected_n).max(2);
        let Ok(a) = MassNumber::new(mass_a) else {
            return Vec::new();
        };
        let ldm = LiquidDropModel::new();
        let mut pts = Vec::new();
        for z in 1..mass_a {
            if let Ok(b) = ldm.binding_energy_breakdown(AtomicNumber::new(z), a) {
                if b.per_nucleon > -2.0 {
                    pts.push([f64::from(z), b.per_nucleon]);
                }
            }
        }
        vec![(format!("Isobar A={mass_a} B(Z,A)/A (MeV)"), pts)]
    }
}

fn finite_clamp(val: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(lo, hi)
    } else {
        fallback
    }
}

impl FiniteStateGuard for NuclearStabilityTool {
    fn is_finite_state(&self) -> bool {
        let s = &self.state;
        s.alpha_beta.is_finite()
            && s.resonance_energy_mev.is_finite()
            && s.resonance_gamma_mev.is_finite()
            && s.probe_energy_mev.is_finite()
            && s.half_life_1.is_finite()
            && s.half_life_2.is_finite()
            && s.half_life_3.is_finite()
            && s.max_decay_time.is_finite()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) {
        self.clamp_in_place();
        self.revision = self.revision.wrapping_add(1);
    }

    fn reset_parameters(&mut self) {
        self.state = NuclearStabilityState::default();
        self.revision = self.revision.wrapping_add(1);
    }
}

impl InteractiveTool for NuclearStabilityTool {
    fn name(&self) -> &'static str {
        "Nuclear Stability & Decay"
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
        let loaded: NuclearStabilityState = serde_json::from_str(raw)
            .map_err(|e| format!("Invalid Nuclear Stability JSON: {e}"))?;
        self.state = loaded;
        self.clamp_in_place();
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.compute_datasets()
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_nuclear_tool(ui);
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

impl scientific_metadata::theory::TheoryDescribable for NuclearStabilityTool {
    fn theory_description(&self) -> String {
        "Evaluates the Weizsacker Semi-Empirical Mass Formula B(Z, A) = a_V A - a_S A^(2/3) - a_C Z(Z-1) A^(-1/3) - a_sym (A-2Z)^2/A +- delta(Z,N), nuclear shell spin-orbit coupling <L.S>, Gamow alpha-tunneling factors, reaction Q-values, Breit-Wigner resonances, and stack-allocated Bateman linear decay chains.".into()
    }

    fn phonetic_description(&self) -> String {
        "Weizsacker semi-empirical mass formula, Gamow tunneling, Breit-Wigner resonance, and Bateman decay kinetics".into()
    }

    fn theory_citation(&self) -> String {
        "von Weizsacker (1935) Z. Phys. 96, 431 (10.1007/BF01337700); Bateman (1910) Proc. Cambridge Philos. Soc. 15, 423; Breit & Wigner (1936) Phys. Rev. 49, 519 (10.1103/PhysRev.49.519)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    ToolMetadata {
        id: "NuclearStabilityTool",
        display_name: "Nuclear Stability & Decay",
        domain: "subatomic",
        tags: &["nuclear", "weizsacker", "bateman", "resonance", "gamow", "isotopes"],
        build: || Box::new(NuclearStabilityTool::default()),
    }
}

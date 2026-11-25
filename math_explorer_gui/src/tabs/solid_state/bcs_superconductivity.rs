#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:solid_state]

#[path = "bcs_superconductivity/ui.rs"]
mod ui;

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::error::SolidStateError;
use math_explorer::physics::solid_state::bcs::{BcsGapSolver, IsotropicBCSModel};
use math_explorer::physics::solid_state::types::ElectronVolts;

/// Focused solid-state physics panels inside [`BcsSuperconductivityTool`].
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum BcsSubView {
    /// Isotropic s-wave BCS gap equation, Bogoliubov quasiparticles $\pm E_k$, and coherence factors $u_k^2, v_k^2$.
    #[default]
    GapAndCoherence,
    /// Debye acoustic / Einstein optical phonon branches, $T^3$ heat capacity, and Fröhlich coupling $|M_q|^2$.
    PhononsAndDebye,
    /// Static Thomas-Fermi dielectric function $\epsilon(q)$ and Yukawa vs bare Coulomb potential.
    ThomasFermiScreening,
}

impl BcsSubView {
    /// Human-readable selector label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::GapAndCoherence => "BCS Gap & Bogoliubov Coherence",
            Self::PhononsAndDebye => "Phonons, Debye C_V & Fröhlich",
            Self::ThomasFermiScreening => "Thomas-Fermi & Yukawa Screening",
        }
    }
}

/// Serializable URL/preset state for [`BcsSuperconductivityTool`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BcsSuperconductivityState {
    pub view: BcsSubView,
    pub pairing_potential: f64,
    pub debye_cutoff: f64,
    pub half_bandwidth: f64,
    pub mixing_param: f64,
    pub sound_velocity: f64,
    pub einstein_freq: f64,
    pub debye_temp: f64,
    pub frohlich_alpha: f64,
    pub k_tf: f64,
}

impl Default for BcsSuperconductivityState {
    fn default() -> Self {
        Self {
            view: BcsSubView::GapAndCoherence,
            pairing_potential: 0.45,
            debye_cutoff: 2.5,
            half_bandwidth: 4.0,
            mixing_param: 0.5,
            sound_velocity: 2.2,
            einstein_freq: 4.5,
            debye_temp: 340.0,
            frohlich_alpha: 1.2,
            k_tf: 1.4,
        }
    }
}

impl BcsSuperconductivityState {
    pub(crate) fn is_valid(&self) -> bool {
        self.pairing_potential.is_finite()
            && (0.02..=2.0).contains(&self.pairing_potential)
            && self.debye_cutoff.is_finite()
            && (0.2..=8.0).contains(&self.debye_cutoff)
            && self.half_bandwidth.is_finite()
            && (0.5..=10.0).contains(&self.half_bandwidth)
            && self.mixing_param.is_finite()
            && (0.05..=1.0).contains(&self.mixing_param)
            && self.sound_velocity.is_finite()
            && (0.2..=8.0).contains(&self.sound_velocity)
            && self.einstein_freq.is_finite()
            && (0.2..=10.0).contains(&self.einstein_freq)
            && self.debye_temp.is_finite()
            && (20.0..=1000.0).contains(&self.debye_temp)
            && self.frohlich_alpha.is_finite()
            && (0.05..=6.0).contains(&self.frohlich_alpha)
            && self.k_tf.is_finite()
            && (0.05..=5.0).contains(&self.k_tf)
    }

    pub(crate) fn clamp(&mut self) {
        self.pairing_potential = self.pairing_potential.clamp(0.02, 2.0);
        self.debye_cutoff = self.debye_cutoff.clamp(0.2, 8.0);
        self.half_bandwidth = self.half_bandwidth.clamp(0.5, 10.0);
        self.mixing_param = self.mixing_param.clamp(0.05, 1.0);
        self.sound_velocity = self.sound_velocity.clamp(0.2, 8.0);
        self.einstein_freq = self.einstein_freq.clamp(0.2, 10.0);
        self.debye_temp = self.debye_temp.clamp(20.0, 1000.0);
        self.frohlich_alpha = self.frohlich_alpha.clamp(0.05, 6.0);
        self.k_tf = self.k_tf.clamp(0.05, 5.0);
    }
}

/// Interactive BCS superconductivity, lattice phonon, and Thomas-Fermi screening explorer.
pub struct BcsSuperconductivityTool {
    pub state: BcsSuperconductivityState,
    pub(crate) bcs_model: IsotropicBCSModel,
    pub(crate) solver: BcsGapSolver,
    pub(crate) converged_gap_ev: f64,
    pub(crate) convergence_warning: Option<String>,
    pub(crate) state_rev: u64,
}

impl Default for BcsSuperconductivityTool {
    fn default() -> Self {
        let state = BcsSuperconductivityState::default();
        let bcs_model = IsotropicBCSModel::with_uniform_window(
            160,
            state.half_bandwidth,
            state.pairing_potential,
            state.debye_cutoff,
        );
        let solver = BcsGapSolver::new()
            .with_max_iterations(400)
            .with_tolerance(1e-8)
            .with_mixing_param(state.mixing_param);
        let mut tool = Self {
            state,
            bcs_model,
            solver,
            converged_gap_ev: 0.0,
            convergence_warning: None,
            state_rev: 1,
        };
        tool.solve_bcs_gap();
        tool
    }
}

impl BcsSuperconductivityTool {
    /// Updates the preallocated [`IsotropicBCSModel`] and solves the self-consistent gap with zero heap allocations.
    pub fn solve_bcs_gap(&mut self) {
        self.bcs_model.update_uniform_window(
            self.state.half_bandwidth,
            self.state.pairing_potential,
            self.state.debye_cutoff,
        );
        self.solver.mixing_param = self.state.mixing_param;
        let guess = ElectronVolts::new((0.08 * self.state.debye_cutoff).max(1e-3));
        match self.solver.solve(&self.bcs_model, guess) {
            Ok(delta) => {
                self.converged_gap_ev = (delta / ElectronVolts::new(1.0)).max(0.0);
                self.convergence_warning = None;
            }
            Err(SolidStateError::ConvergenceFailure(iters, last_delta)) => {
                self.converged_gap_ev = (last_delta / ElectronVolts::new(1.0)).max(0.0);
                self.convergence_warning = Some(format!(
                    "Weak-coupling slow convergence after {iters} iterations (Δ ≈ {:.4} eV)",
                    self.converged_gap_ev
                ));
            }
            Err(err) => {
                self.converged_gap_ev = 0.0;
                self.convergence_warning = Some(format!("BCS solver diagnostic: {err}"));
            }
        }
    }

    pub(crate) fn mark_changed(&mut self) {
        self.state_rev = self.state_rev.wrapping_add(1);
        self.solve_bcs_gap();
    }
}

impl InteractiveTool for BcsSuperconductivityTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "BCS Superconductivity & Phonons"
    }
    fn domain(&self) -> &'static str {
        "solid_state"
    }
    fn state_revision(&self) -> u64 {
        self.state_rev
    }
    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds {
            max_bytes: 4_096,
            max_depth: 4,
            max_array_len: 16,
        }
    }
    fn has_finite_parameters(&self) -> bool {
        self.state.is_valid()
    }
    fn clamp_parameters(&mut self) -> bool {
        self.state.clamp();
        self.mark_changed();
        true
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: BcsSuperconductivityState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid BcsSuperconductivity state: {e}"))?;
        if !parsed.is_valid() {
            return Err("BcsSuperconductivity parameters out of finite bounds".into());
        }
        self.state = parsed;
        self.mark_changed();
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.draw_controls_panel(parent_ui);
        self.draw_central_plots(parent_ui);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BcsSuperconductivityTool",
        display_name: "BCS Superconductivity & Phonons",
        domain: "solid_state",
        tags: &["bcs", "superconductivity", "phonons", "debye", "screening", "yukawa"],
        build: || Box::new(BcsSuperconductivityTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for BcsSuperconductivityTool {
    fn theory_description(&self) -> String {
        match self.state.view {
            BcsSubView::GapAndCoherence => format!(
                "Isotropic s-wave BCS pairing ($V = {:.2}$ eV, Debye cutoff {:.2} eV) yielding self-consistent gap $\\Delta = {:.4}$ eV, Bogoliubov quasiparticle dispersion $E_k = \\sqrt{{\\xi_k^2 + \\Delta^2}}$, and coherence factors $u_k^2 + v_k^2 = 1$.",
                self.state.pairing_potential, self.state.debye_cutoff, self.converged_gap_ev
            ),
            BcsSubView::PhononsAndDebye => format!(
                "Acoustic Debye dispersion $\\omega = v_s k$ ($v_s = {:.2}$), optical Einstein mode $\\omega_E = {:.2}$, low-temperature Debye heat capacity $C_V \\propto (T/\\Theta_D)^3$ ($\\Theta_D = {:.0}$ K), and Fröhlich electron-phonon coupling.",
                self.state.sound_velocity, self.state.einstein_freq, self.state.debye_temp
            ),
            BcsSubView::ThomasFermiScreening => format!(
                "Static Thomas-Fermi dielectric screening $\\epsilon(q) = 1 + k_{{TF}}^2 / q^2$ and exponentially screened Yukawa potential $V(r) = \\exp(-k_{{TF}} r)/r$ ($k_{{TF}} = {:.2}$).",
                self.state.k_tf
            ),
        }
    }
    fn phonetic_description(&self) -> String {
        "BCS superconductivity gap equation, Bogoliubov coherence factors, Debye phonons, and Thomas-Fermi screening theory".into()
    }
    fn theory_citation(&self) -> String {
        "Theory of Superconductivity (Bardeen, Cooper & Schrieffer, 1957); Introduction to Solid State Physics (Kittel, 2004); Solid State Physics (Ashcroft & Mermin, 1976)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "BcsGap".into(),
            "Self-consistent Cooper pairing gap Delta and Bogoliubov quasiparticle dispersion"
                .into(),
        );
        map.insert(
            "DebyePhonons".into(),
            "Acoustic/optical lattice vibrations and T^3 low-temperature Debye heat capacity"
                .into(),
        );
        map.insert(
            "ThomasFermi".into(),
            "Static free-electron dielectric function and real-space Yukawa potential".into(),
        );
        map
    }
}

const BCS_GAP_STARTER_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "BCS Superconducting Gap & Coherence",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "view": "GapAndCoherence",
    "pairing_potential": 0.48,
    "debye_cutoff": 2.5,
    "half_bandwidth": 4.0,
    "mixing_param": 0.5,
    "sound_velocity": 2.2,
    "einstein_freq": 4.5,
    "debye_temp": 340.0,
    "frohlich_alpha": 1.2,
    "k_tf": 1.4
  }
}"#;

const BCS_PHONONS_STARTER_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Debye Phonons & Thomas-Fermi Screening",
    "domain": "solid_state",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "view": "PhononsAndDebye",
    "pairing_potential": 0.45,
    "debye_cutoff": 2.5,
    "half_bandwidth": 4.0,
    "mixing_param": 0.5,
    "sound_velocity": 2.4,
    "einstein_freq": 5.0,
    "debye_temp": 380.0,
    "frohlich_alpha": 1.5,
    "k_tf": 1.6
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bcs_gap_coherence_starter",
        title: "BCS Superconducting Gap & Coherence",
        description: "Self-consistent isotropic BCS energy gap Δ, Bogoliubov quasiparticle branches ±E_k, and coherence factors u_k², v_k².",
        category: "Condensed Matter",
        icon: "❄",
        domain: "solid_state",
        tool_name: Some("BcsSuperconductivityTool"),
        preset_generator: || BCS_GAP_STARTER_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bcs_phonons_screening_starter",
        title: "Debye Phonons & Thomas-Fermi Screening",
        description: "Acoustic/optical phonon dispersions, T³ Debye heat capacity, Fröhlich coupling, and Yukawa screening.",
        category: "Condensed Matter",
        icon: "🔊",
        domain: "solid_state",
        tool_name: Some("BcsSuperconductivityTool"),
        preset_generator: || BCS_PHONONS_STARTER_JSON,
    }
}

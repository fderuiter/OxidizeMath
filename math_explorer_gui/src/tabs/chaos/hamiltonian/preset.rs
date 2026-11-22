//! Bounded JSON state serialization, URL share state validation, and Starter Gallery cards.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    ComparatorView, HamiltonianChaosLab, HamiltonianModelKind, SymplecticSolverChoice, MAX_ORBITS,
};
use crate::starter_gallery::StarterPresetMetadata;

/// Compact parameter and seed-ensemble snapshot for [`HamiltonianChaosLab`].
///
/// Storing canonical 2D Poincaré seeds (`<= 16` pairs) rather than full trajectory
/// histories keeps snapshots under `600` bytes so stateless URL fragments fit
/// comfortably inside `URL_JSON_BOUNDS` (`4 KiB`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HamiltonianLabState {
    /// Fixed integration time step $\Delta t \in [0.0005, 0.1]$.
    pub dt: f64,
    /// Constant Hamiltonian energy shell $E$.
    pub energy: f64,
    /// Selected conservative Hamiltonian model.
    pub model: HamiltonianModelKind,
    /// Selected symplectic integrator.
    pub solver: SymplecticSolverChoice,
    /// Active secondary comparator plot view.
    pub comparator_view: ComparatorView,
    /// Sub-steps integrated per UI frame (`1..=128`).
    pub steps_per_frame: usize,
    /// Whether the explicit `RungeKutta4` drift comparator is displayed.
    pub show_rk4_ghost: bool,
    /// Whether continuous playback is active.
    pub is_playing: bool,
    /// Crosshair cursor `(q, p)` on the Poincaré plane.
    pub cursor_qp: [f64; 2],
    /// Canonical `(q, p)` Poincaré seeds (`len() <= 16`).
    pub seeds: Vec<[f64; 2]>,
}

const fn model_energy_limits(model: HamiltonianModelKind) -> (f64, f64) {
    match model {
        HamiltonianModelKind::HenonHeiles => (0.02, 0.1665),
        HamiltonianModelKind::DoublePendulum => (4.0, 52.0),
        HamiltonianModelKind::FigureEightThreeBody => (0.5, 2.5),
    }
}

const fn model_cursor_limits(model: HamiltonianModelKind) -> [f64; 4] {
    match model {
        HamiltonianModelKind::HenonHeiles => [-0.62, 0.88, -0.62, 0.62],
        HamiltonianModelKind::DoublePendulum => [
            -std::f64::consts::PI,
            std::f64::consts::PI,
            -8.5,
            8.5,
        ],
        HamiltonianModelKind::FigureEightThreeBody => [-1.25, 1.25, -1.25, 1.25],
    }
}

impl HamiltonianChaosLab {
    /// Re-evaluates existing orbit seeds on a newly adjusted energy shell.
    pub fn rebuild_existing_seeds(&mut self) {
        let saved_seeds: Vec<[f64; 2]> = self.orbits.iter().map(|s| s.seed_qp).collect();
        self.clear_orbits();
        for qp in saved_seeds {
            let _ = self.seed_orbit_no_warmup(qp);
        }
        if !self.orbits.is_empty() {
            self.step_batch(240);
        }
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    pub(super) fn serialize_lab_state(&self) -> Option<String> {
        let snapshot = HamiltonianLabState {
            dt: self.dt,
            energy: self.energy,
            model: self.model,
            solver: self.solver,
            comparator_view: self.comparator_view,
            steps_per_frame: self.steps_per_frame,
            show_rk4_ghost: self.show_rk4_ghost,
            is_playing: self.is_playing,
            cursor_qp: self.cursor_qp,
            seeds: self.orbits.iter().map(|s| s.seed_qp).collect(),
        };
        serde_json::to_string(&snapshot).ok()
    }

    pub(super) fn deserialize_lab_state(&mut self, raw: &str) -> Result<(), String> {
        let parsed: HamiltonianLabState = serde_json::from_str(raw)
            .map_err(|e| format!("Invalid HamiltonianChaosLab state: {e}"))?;
        validate_lab_state(&parsed)?;

        let prev_model = self.model;
        let prev_energy = self.energy;
        self.model = parsed.model;
        self.energy = parsed.energy;

        // Pre-validate every seed on the target energy shell before committing state.
        let mut temp_lab_ok = true;
        for &qp in &parsed.seeds {
            if !self.can_seed_on_shell(parsed.model, parsed.energy, qp) {
                temp_lab_ok = false;
                break;
            }
        }
        if !temp_lab_ok {
            self.model = prev_model;
            self.energy = prev_energy;
            return Err("One or more Poincare seeds lie outside the target energy shell".into());
        }

        self.dt = parsed.dt;
        self.solver = parsed.solver;
        self.comparator_view = parsed.comparator_view;
        self.steps_per_frame = parsed.steps_per_frame;
        self.show_rk4_ghost = parsed.show_rk4_ghost;
        self.is_playing = parsed.is_playing;
        self.cursor_qp = parsed.cursor_qp;

        self.clear_orbits();
        for qp in parsed.seeds {
            let _ = self.seed_orbit_no_warmup(qp);
        }
        if !self.orbits.is_empty() {
            self.step_batch(320);
        }
        self.state_rev = self.state_rev.wrapping_add(1);
        Ok(())
    }

    fn can_seed_on_shell(&self, model: HamiltonianModelKind, energy: f64, qp: [f64; 2]) -> bool {
        if !qp[0].is_finite() || !qp[1].is_finite() {
            return false;
        }
        match model {
            HamiltonianModelKind::HenonHeiles => {
                self.henon.seed_on_poincare_section(energy, qp[0], qp[1]).is_some()
            }
            HamiltonianModelKind::DoublePendulum => self
                .pendulum
                .seed_on_poincare_section(energy, qp[0], qp[1])
                .is_some(),
            HamiltonianModelKind::FigureEightThreeBody => {
                (-0.45..=0.45).contains(&qp[0]) && (-0.45..=0.45).contains(&qp[1])
            }
        }
    }
}

fn validate_lab_state(parsed: &HamiltonianLabState) -> Result<(), String> {
    if !parsed.dt.is_finite() || !(0.0005..=0.1).contains(&parsed.dt) {
        return Err(format!("dt out of [0.0005, 0.1]: {}", parsed.dt));
    }
    let (e_min, e_max) = model_energy_limits(parsed.model);
    if !parsed.energy.is_finite() || !(e_min..=e_max).contains(&parsed.energy) {
        return Err(format!(
            "energy out of [{e_min}, {e_max}] for {:?}: {}",
            parsed.model, parsed.energy
        ));
    }
    if parsed.model == HamiltonianModelKind::DoublePendulum
        && parsed.solver != SymplecticSolverChoice::GaussLegendre4
    {
        return Err(
            "DoublePendulum requires GaussLegendre4 implicit symplectic solver".to_string(),
        );
    }
    if !(1..=128).contains(&parsed.steps_per_frame) {
        return Err(format!(
            "steps_per_frame out of 1..=128: {}",
            parsed.steps_per_frame
        ));
    }
    if parsed.seeds.len() > MAX_ORBITS {
        return Err(format!(
            "seeds length {} exceeds MAX_ORBITS ({MAX_ORBITS})",
            parsed.seeds.len()
        ));
    }
    let [q_min, q_max, p_min, p_max] = model_cursor_limits(parsed.model);
    if !parsed.cursor_qp[0].is_finite()
        || !parsed.cursor_qp[1].is_finite()
        || !(q_min..=q_max).contains(&parsed.cursor_qp[0])
        || !(p_min..=p_max).contains(&parsed.cursor_qp[1])
    {
        return Err("cursor_qp out of Poincare viewport bounds".to_string());
    }
    Ok(())
}

/// Starter preset: Hénon-Heiles mixed KAM stability islands and stochastic sea at $E = 0.125$.
pub const PRESET_HENON_HEILES_KAM: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Henon-Heiles KAM Islands & Chaotic Sea",
    "domain": "chaos",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "dt": 0.02,
    "energy": 0.125,
    "model": "HenonHeiles",
    "solver": "Yoshida4",
    "comparator_view": "EnergyDrift",
    "steps_per_frame": 48,
    "show_rk4_ghost": true,
    "is_playing": false,
    "cursor_qp": [0.15, 0.0],
    "seeds": [[0.1, 0.0], [0.3, 0.0], [-0.18, 0.1], [0.0, 0.3], [-0.27, 0.0], [0.42, 0.0]]
  }
}"#;

/// Starter preset: Chaotic non-separable Double Pendulum with implicit `GaussLegendre4` vs `RungeKutta4`.
pub const PRESET_DOUBLE_PENDULUM_DRIFT: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Double Pendulum Symplectic vs RK4 Drift",
    "domain": "chaos",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "dt": 0.02,
    "energy": 32.0,
    "model": "DoublePendulum",
    "solver": "GaussLegendre4",
    "comparator_view": "EnergyDrift",
    "steps_per_frame": 32,
    "show_rk4_ghost": true,
    "is_playing": false,
    "cursor_qp": [0.35, 0.0],
    "seeds": [[0.2, 0.0], [1.1, 0.0], [-0.85, 1.4], [1.8, 0.0]]
  }
}"#;

/// Starter preset: Planar 3-body Chenciner-Montgomery Figure-8 choreography under `Yoshida4`.
pub const PRESET_FIGURE_EIGHT_CHOREOGRAPHY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "3-Body Figure-8 Choreography",
    "domain": "chaos",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "dt": 0.01,
    "energy": 1.287,
    "model": "FigureEightThreeBody",
    "solver": "Yoshida4",
    "comparator_view": "ConfigurationSpace",
    "steps_per_frame": 40,
    "show_rk4_ghost": true,
    "is_playing": false,
    "cursor_qp": [0.0, 0.0],
    "seeds": [[0.0, 0.0], [0.03, -0.015]]
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "henon_heiles_kam_islands",
        title: "Hénon-Heiles KAM Islands",
        description: "Mixed regular KAM tori and stochastic chaos on H = 1/8 with 4th-order Yoshida symplectic integration.",
        category: "Hamiltonian Chaos",
        icon: "🪐",
        domain: "chaos",
        tool_name: Some("HamiltonianChaosLab"),
        preset_generator: || PRESET_HENON_HEILES_KAM,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "double_pendulum_symplectic_drift",
        title: "Double Pendulum Symplectic vs RK4",
        description: "Non-separable canonical double pendulum comparing implicit Gauss-Legendre 4th order against secular RK4 drift.",
        category: "Hamiltonian Chaos",
        icon: "⚖️",
        domain: "chaos",
        tool_name: Some("HamiltonianChaosLab"),
        preset_generator: || PRESET_DOUBLE_PENDULUM_DRIFT,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "figure_eight_three_body",
        title: "3-Body Figure-8 Choreography",
        description: "Planar three-body Chenciner-Montgomery periodic figure-8 orbit preserved by Yoshida 4th-order splitting.",
        category: "Celestial Mechanics",
        icon: "♾️",
        domain: "chaos",
        tool_name: Some("HamiltonianChaosLab"),
        preset_generator: || PRESET_FIGURE_EIGHT_CHOREOGRAPHY,
    }
}

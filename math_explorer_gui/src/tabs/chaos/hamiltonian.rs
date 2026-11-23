//! Symplectic Hamiltonian Chaos and Poincaré Section Lab (`HamiltonianChaosLab`).
//!
//! Combines zero-allocation symplectic integrators (`StormerVerlet`, `Yoshida4`, `GaussLegendre4`)
//! with cubic-Hermite Poincaré surface-of-section detection across `HenonHeiles`,
//! `DoublePendulum`, and planar 3-body Figure-8 choreographies, alongside a live explicit
//! `RungeKutta4` energy-drift comparator.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:chaos_theory]

use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata};
use eframe::egui;
use math_explorer::physics::astrophysics::nbody::FigureEightThreeBody;
use math_explorer::physics::chaos::{
    double_pendulum::DoublePendulum, henon_heiles::HenonHeiles, poincare::PoincareSection,
};
use math_explorer::pure_math::analysis::ode::{
    ArrayState, GaussLegendre4, RungeKutta4, StormerVerlet, SymplecticMethod, Yoshida4,
};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

#[path = "hamiltonian/canvas.rs"]
mod canvas;
#[path = "hamiltonian/export.rs"]
mod export;
#[path = "hamiltonian/ghost.rs"]
mod ghost;
#[path = "hamiltonian/preset.rs"]
pub mod preset;
#[path = "hamiltonian/simulation.rs"]
mod simulation;
#[path = "hamiltonian/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "hamiltonian/tests.rs"]
mod tests;

/// Maximum number of simultaneously seeded orbits on the Poincaré section.
pub const MAX_ORBITS: usize = 16;

/// Maximum number of Poincaré section intersection points retained per orbit.
pub const MAX_SECTION_POINTS: usize = 4_096;

/// Maximum number of ghost `RungeKutta4` Poincaré points retained for drift visualization.
pub const MAX_GHOST_SECTION_POINTS: usize = 512;

/// Maximum number of energy-drift time series samples retained for the comparator plot.
pub const MAX_DRIFT_SAMPLES: usize = 512;

/// Maximum number of real-space configuration trail points retained per orbit.
pub const MAX_TRAIL_POINTS: usize = 240;

/// Per-frame wall-clock integration budget in milliseconds (sustains 60 FPS on desktop & WASM).
pub const FRAME_BUDGET_MS: f64 = 8.0;

/// Conservative Hamiltonian benchmark system selectable in [`HamiltonianChaosLab`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HamiltonianModelKind {
    /// Separable 2-DOF Hénon-Heiles galactic potential exhibiting KAM stability islands and chaos.
    HenonHeiles,
    /// Non-separable 2-DOF canonical planar double pendulum.
    DoublePendulum,
    /// Planar 3-body gravitational Chenciner-Montgomery Figure-8 choreography (`ArrayState<12>`).
    FigureEightThreeBody,
}

impl HamiltonianModelKind {
    /// User-facing display label for selector menus.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HenonHeiles => "Hénon-Heiles (KAM Islands)",
            Self::DoublePendulum => "Double Pendulum (Non-Separable)",
            Self::FigureEightThreeBody => "3-Body Figure-8 Choreography",
        }
    }

    /// Axis labels `(q_label, p_label)` for the Poincaré surface of section.
    #[must_use]
    pub const fn section_axes(self) -> (&'static str, &'static str) {
        match self {
            Self::HenonHeiles => ("q₂", "p₂"),
            Self::DoublePendulum => ("θ₂ (rad)", "p₂"),
            Self::FigureEightThreeBody => ("x₁", "p_x₁"),
        }
    }
}

/// Serializable symplectic integrator selector for [`HamiltonianChaosLab`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SymplecticSolverChoice {
    /// Second-order velocity Störmer-Verlet splitting.
    StormerVerlet,
    /// Fourth-order Yoshida symmetric triple-jump composition.
    Yoshida4,
    /// Fourth-order 2-stage implicit Gauss-Legendre Runge-Kutta.
    GaussLegendre4,
}

impl SymplecticSolverChoice {
    /// User-facing display label for selector menus.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::StormerVerlet => "Störmer-Verlet (2nd Order)",
            Self::Yoshida4 => "Yoshida 4th Order",
            Self::GaussLegendre4 => "Gauss-Legendre 4th Order (Implicit)",
        }
    }

    /// Converts the GUI enum to the `pure_math` [`SymplecticMethod`] descriptor.
    #[must_use]
    pub const fn to_method(self) -> SymplecticMethod {
        match self {
            Self::StormerVerlet => SymplecticMethod::StormerVerlet,
            Self::Yoshida4 => SymplecticMethod::Yoshida4,
            Self::GaussLegendre4 => SymplecticMethod::GaussLegendre4,
        }
    }
}

/// Secondary right-hand plot mode alongside the Poincaré section canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ComparatorView {
    /// Plots relative Hamiltonian error $|\Delta H / H_0|$ (%) for Symplectic vs `RungeKutta4`.
    EnergyDrift,
    /// Plots real-space configuration trajectories alongside the decaying `RungeKutta4` twin.
    ConfigurationSpace,
}

impl ComparatorView {
    /// User-facing label for plot mode tabs.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::EnergyDrift => "Relative Energy Drift (%)",
            Self::ConfigurationSpace => "Configuration Trajectory",
        }
    }
}

/// Bounded per-orbit trajectory slot in the seeded ensemble (`<= MAX_ORBITS`).
#[derive(Debug, Clone)]
pub struct OrbitSlot {
    /// Canonical `(q, p)` seed coordinate on the Poincaré surface of section.
    pub seed_qp: [f64; 2],
    /// Palette series index for consistent coloring across canvas and plots.
    pub color_index: usize,
    /// Exact initial Hamiltonian energy $H_0$.
    pub initial_energy: f64,
    /// Latest relative symplectic energy error $|\Delta H / H_0| \times 100$.
    pub last_rel_err_pct: f64,
    pub(crate) state4: ArrayState<4>,
    pub(crate) prev4: ArrayState<4>,
    pub(crate) deriv4: ArrayState<4>,
    pub(crate) prev_deriv4: ArrayState<4>,
    pub(crate) state12: ArrayState<12>,
    pub(crate) prev12: ArrayState<12>,
    pub(crate) deriv12: ArrayState<12>,
    pub(crate) prev_deriv12: ArrayState<12>,
    /// Captured `(q, p)` section crossings, ring-capped at [`MAX_SECTION_POINTS`].
    pub section_points: Vec<[f64; 2]>,
    pub(crate) section_write_idx: usize,
    /// Recent real-space `[x, y]` configuration positions, capped at [`MAX_TRAIL_POINTS`].
    pub trail: Vec<[f64; 2]>,
}

/// Synchronized explicit `RungeKutta4` comparator tracking energy drift and phase decay.
#[derive(Debug, Clone, Default)]
pub struct GhostComparator {
    /// Whether an active orbit twin is initialized.
    pub active: bool,
    /// Initial Hamiltonian energy $H_0$ shared with the tracked symplectic orbit.
    pub initial_energy: f64,
    /// Elapsed simulation time for the tracked comparator twin.
    pub elapsed_time: f64,
    pub(crate) step_count: usize,
    pub(crate) state4: ArrayState<4>,
    pub(crate) prev4: ArrayState<4>,
    pub(crate) deriv4: ArrayState<4>,
    pub(crate) prev_deriv4: ArrayState<4>,
    pub(crate) state12: ArrayState<12>,
    /// Decaying `RungeKutta4` Poincaré crossings, capped at [`MAX_GHOST_SECTION_POINTS`].
    pub section_points: Vec<[f64; 2]>,
    pub(crate) section_write_idx: usize,
    /// Recent configuration trail of the `RungeKutta4` twin.
    pub trail: Vec<[f64; 2]>,
    /// Time series `[t, rel_err_pct]` for the symplectic solver.
    pub symplectic_drift: Vec<[f64; 2]>,
    /// Time series `[t, rel_err_pct]` for explicit `RungeKutta4`.
    pub rk4_drift: Vec<[f64; 2]>,
    /// Latest symplectic relative energy error (%).
    pub symplectic_err_pct: f64,
    /// Latest `RungeKutta4` relative energy error (%).
    pub rk4_err_pct: f64,
}

/// Interactive Symplectic Hamiltonian Chaos & Poincaré Section Lab.
pub struct HamiltonianChaosLab {
    /// Active conservative Hamiltonian model.
    pub model: HamiltonianModelKind,
    /// Selected symplectic integrator.
    pub solver: SymplecticSolverChoice,
    /// Selected secondary plot mode.
    pub comparator_view: ComparatorView,
    /// Integration time step $\Delta t$.
    pub dt: f64,
    /// Target Hamiltonian energy shell $E$ (for 2-DOF section seeding).
    pub energy: f64,
    /// Number of integration sub-steps executed per UI frame.
    pub steps_per_frame: usize,
    /// Whether the explicit `RungeKutta4` comparator twin is enabled.
    pub show_rk4_ghost: bool,
    /// Whether continuous frame stepping is active.
    pub is_playing: bool,
    /// Keyboard / pointer crosshair cursor `(q, p)` on the Poincaré plane.
    pub cursor_qp: [f64; 2],
    /// Seeded symplectic orbit ensemble (`len() <= MAX_ORBITS`).
    pub orbits: Vec<OrbitSlot>,
    /// Synchronized explicit `RungeKutta4` comparator twin.
    pub ghost: GhostComparator,
    /// Elapsed simulation time $t$.
    pub sim_time: f64,
    /// Last measured integration frame wall duration in milliseconds.
    pub last_frame_ms: f64,
    pub(crate) state_rev: u64,
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) last_status_note: Option<String>,
    pub(crate) henon: HenonHeiles,
    pub(crate) pendulum: DoublePendulum,
    pub(crate) three_body: FigureEightThreeBody,
    pub(crate) verlet2: StormerVerlet<ArrayState<2>>,
    pub(crate) yoshida2: Yoshida4<ArrayState<2>>,
    pub(crate) verlet6: StormerVerlet<ArrayState<6>>,
    pub(crate) yoshida6: Yoshida4<ArrayState<6>>,
    pub(crate) gl4_4: GaussLegendre4<ArrayState<4>>,
    pub(crate) rk4_4: RungeKutta4<ArrayState<4>>,
    pub(crate) gl4_12: GaussLegendre4<ArrayState<12>>,
    pub(crate) rk4_12: RungeKutta4<ArrayState<12>>,
    pub(crate) poincare4: PoincareSection<4>,
    pub(crate) poincare12: PoincareSection<12>,
}

impl Default for HamiltonianChaosLab {
    fn default() -> Self {
        let z2 = ArrayState::<2>::zeros();
        let z4 = ArrayState::<4>::zeros();
        let z6 = ArrayState::<6>::zeros();
        let z12 = ArrayState::<12>::zeros();
        let mut lab = Self {
            model: HamiltonianModelKind::HenonHeiles,
            solver: SymplecticSolverChoice::Yoshida4,
            comparator_view: ComparatorView::EnergyDrift,
            dt: 0.015,
            energy: 0.125,
            steps_per_frame: 32,
            show_rk4_ghost: true,
            is_playing: false,
            cursor_qp: [0.12, 0.0],
            orbits: Vec::with_capacity(MAX_ORBITS),
            ghost: GhostComparator::default(),
            sim_time: 0.0,
            last_frame_ms: 0.0,
            state_rev: 1,
            focused_canvas: None,
            last_status_note: None,
            henon: HenonHeiles::default(),
            pendulum: DoublePendulum::default(),
            three_body: FigureEightThreeBody::default(),
            verlet2: StormerVerlet::new(&z2),
            yoshida2: Yoshida4::new(&z2),
            verlet6: StormerVerlet::new(&z6),
            yoshida6: Yoshida4::new(&z6),
            gl4_4: GaussLegendre4::new(&z4),
            rk4_4: RungeKutta4::new(&z4),
            gl4_12: GaussLegendre4::new(&z12),
            rk4_12: RungeKutta4::new(&z12),
            poincare4: PoincareSection::<4>::henon_heiles(),
            poincare12: PoincareSection::<12>::figure_eight(),
        };
        lab.seed_default_showcase();
        lab
    }
}

impl TheoryDescribable for HamiltonianChaosLab {
    fn theory_description(&self) -> String {
        format!(
            "Symplectic integration ($dp \\wedge dq$ invariant) on {} using {} ($\\Delta t = {:.4}$, $E = {:.4}$). Explicit Runge-Kutta methods violate Liouville volume preservation on curved energy shells, causing secular drift, whereas symplectic maps preserve a nearby shadow Hamiltonian.",
            self.model.label(),
            self.solver.label(),
            self.dt,
            self.energy
        )
    }

    fn phonetic_description(&self) -> String {
        format!(
            "Symplectic Hamiltonian chaos lab running {} with {} integrator at time step {:.4}.",
            self.model.label(),
            self.solver.label(),
            self.dt
        )
    }

    fn theory_citation(&self) -> String {
        "Hairer, E., Lubich, C., & Wanner, G. (2006). Geometric Numerical Integration (2nd ed.). Springer; Yoshida, H. (1990); Hénon, M., & Heiles, C. (1964); Chenciner, A., & Montgomery, R. (2000).".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "poincare".to_string(),
            "Cubic-Hermite zero-crossing Poincaré surface of section revealing KAM stability islands and stochastic seas.".to_string(),
        );
        map.insert(
            "symplectic".to_string(),
            "Canonical structure-preserving integrators: Störmer-Verlet, 4th-order Yoshida, and 4th-order implicit Gauss-Legendre.".to_string(),
        );
        map
    }
}

impl InteractiveTool for HamiltonianChaosLab {
    fn name(&self) -> &'static str {
        "Hamiltonian Chaos Lab"
    }

    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn has_finite_parameters(&self) -> bool {
        self.dt.is_finite()
            && self.energy.is_finite()
            && self.cursor_qp[0].is_finite()
            && self.cursor_qp[1].is_finite()
            && self.orbits_are_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_model_parameters();
        true
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.collect_export_primitives()
    }

    fn step_simulation(&mut self) {
        self.step_batch(self.steps_per_frame.clamp(1, 128));
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_lab_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_lab_state(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        if self.is_playing {
            self.step_frame_with_budget();
            ui.ctx().request_repaint();
        }
        self.render_side_panel(ui);
        self.render_central_workspace(ui);
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

inventory::submit! {
    ToolMetadata {
        id: "HamiltonianChaosLab",
        display_name: "Hamiltonian Chaos Lab",
        domain: "chaos",
        tags: &["hamiltonian", "symplectic", "poincare", "kam", "pendulum", "nbody"],
        build: || Box::new(HamiltonianChaosLab::default()),
    }
}

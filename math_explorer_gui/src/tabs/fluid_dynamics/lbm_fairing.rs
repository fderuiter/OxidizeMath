//! Preallocated Lattice Boltzmann fairing drag evaluator, obstacle stamps, and `lbm_low_drag_fairing` challenge.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[allow(unused_imports)]
pub use crate::framework::fairing_mask::{
    apply_stroke_to_fairing_mask, decode_mask_hex, encode_fairing_mask_rle, encode_mask_hex,
    inspect_mask_geometry, restore_inlet_and_core_mask, FairingStamp, LBM_CELL_COUNT,
    LBM_CORE_RADIUS, LBM_CORE_X, LBM_CORE_Y, LBM_CYLINDER_RADIUS, LBM_HEIGHT, LBM_INLET_COLS,
    LBM_MASK_BYTES, LBM_MIN_SOLID_CELLS, LBM_REF_DIAMETER, LBM_RHO_0, LBM_U_IN, LBM_WIDTH,
};
use crate::framework::{ChallengeEvaluation, ChallengeMetadata};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    compute_drag_coefficient, BgkCollision, LatticeBoltzmannD2Q9,
};
use serde::{Deserialize, Serialize};

/// Deterministic BGK + bounce-back steps for zero-allocation drag evaluation.
pub const LBM_EVAL_STEPS: usize = 120;

/// Canonical challenge ID for the LBM low-drag fairing inverse problem.
pub const CHALLENGE_LBM_FAIRING: &str = "lbm_low_drag_fairing";
/// Locked kinematic viscosity `nu` for `lbm_low_drag_fairing`.
pub const CHALLENGE_LBM_VISCOSITY: f64 = 0.02;

/// Serializable state payload for [`super::lattice_boltzmann::LbmUnified`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LbmSerializedState {
    /// Kinematic viscosity `nu`.
    pub viscosity: f64,
    /// Optional hex-encoded `100 x 50` obstacle bitmask.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask_hex: Option<String>,
}

/// Quantitative hydrodynamic metrics for a sculpted obstacle mask.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LbmFairingMetrics {
    /// Momentum-exchange horizontal drag force `F_D`.
    pub f_drag: f64,
    /// Momentum-exchange vertical lift force `F_L`.
    pub f_lift: f64,
    /// Dimensionless drag coefficient `C_D`.
    pub c_d: f64,
    /// Calibrated bluff-cylinder baseline drag coefficient `C_D_cylinder`.
    pub c_d_cylinder: f64,
    /// Total solid obstacle cells on the grid.
    pub n_solid: usize,
    /// Whether all cells inside `R_core = 4` around `(30, 25)` remain solid.
    pub core_intact: bool,
    /// Score `S_lbm` clamped to `0.0..=100.0`.
    pub score: f64,
    /// Whether `N_solid >= 65`, `core_intact`, and `C_D <= 0.82 * C_D_cylinder`.
    pub passed: bool,
}

/// Preallocated zero-heap-allocation LBM fairing evaluator.
pub struct LbmFairingEvaluator {
    solver: LatticeBoltzmannD2Q9<BgkCollision>,
    c_d_cylinder: f64,
    pub latest: LbmFairingMetrics,
}

static CYL_CD: std::sync::OnceLock<f64> = std::sync::OnceLock::new();

impl Default for LbmFairingEvaluator {
    fn default() -> Self {
        let tau = 3.0 * CHALLENGE_LBM_VISCOSITY + 0.5;
        let solver = LatticeBoltzmannD2Q9::new(LBM_WIDTH, LBM_HEIGHT, tau);
        let c_d_cylinder = CYL_CD.get().copied().unwrap_or(0.0);
        Self {
            solver,
            c_d_cylinder,
            latest: LbmFairingMetrics {
                f_drag: 0.0,
                f_lift: 0.0,
                c_d: c_d_cylinder,
                c_d_cylinder,
                n_solid: 81,
                core_intact: true,
                score: 0.0,
                passed: false,
            },
        }
    }
}

impl LbmFairingEvaluator {
    /// Lazily resolves `C_D_cylinder` in place using `self.solver` with zero heap allocations.
    pub fn ensure_baseline(&mut self) -> f64 {
        if self.c_d_cylinder <= 0.0 {
            let cd = *CYL_CD.get_or_init(|| {
                let mut cyl_mask = [false; LBM_CELL_COUNT];
                FairingStamp::BluffCylinder.write_mask(&mut cyl_mask);
                self.simulate_mask_forces(&cyl_mask, CHALLENGE_LBM_VISCOSITY)
                    .2
                    .max(1e-6)
            });
            self.c_d_cylinder = cd;
            if self.latest.c_d_cylinder <= 0.0 {
                self.latest.c_d_cylinder = cd;
                if self.latest.c_d <= 0.0 {
                    self.latest.c_d = cd;
                }
            }
        }
        self.c_d_cylinder
    }

    fn simulate_mask_forces(
        &mut self,
        mask: &[bool; LBM_CELL_COUNT],
        viscosity: f64,
    ) -> (f64, f64, f64) {
        let clean_visc = viscosity.clamp(0.00333, 0.5);
        self.solver.collision_model.tau = 3.0 * clean_visc + 0.5;
        self.solver.init_equilibrium();
        for y in 0..LBM_HEIGHT {
            for x in 0..LBM_WIDTH {
                let is_obs = x >= LBM_INLET_COLS && mask[flatten_2d_index(x, y, LBM_WIDTH)];
                self.solver.set_obstacle(x, y, is_obs);
            }
        }
        self.solver
            .set_inlet(0, 0, LBM_WIDTH, LBM_HEIGHT, LBM_U_IN, 0.0);

        let avg_window = 20_usize;
        let mut sum_fd = 0.0;
        let mut sum_fl = 0.0;
        for step_idx in 0..LBM_EVAL_STEPS {
            self.solver
                .set_inlet(0, 0, LBM_INLET_COLS, LBM_HEIGHT, LBM_U_IN, 0.0);
            self.solver.step();
            if step_idx + avg_window >= LBM_EVAL_STEPS {
                let (fd, fl) = self.solver.hydrodynamic_forces();
                sum_fd += fd;
                sum_fl += fl;
            }
        }
        let f_drag = sum_fd / avg_window as f64;
        let f_lift = sum_fl / avg_window as f64;
        let c_d = compute_drag_coefficient(f_drag, LBM_RHO_0, LBM_U_IN, LBM_REF_DIAMETER);
        (f_drag, f_lift, c_d)
    }

    /// Evaluates `mask` in place with zero dynamic heap allocations.
    pub fn evaluate_mask_in_place(
        &mut self,
        mask: &[bool; LBM_CELL_COUNT],
        viscosity: f64,
    ) -> LbmFairingMetrics {
        let cd_cyl = self.ensure_baseline();
        let (n_solid, core_intact) = inspect_mask_geometry(mask);
        let (f_drag, f_lift, c_d) = self.simulate_mask_forces(mask, viscosity);
        let raw_score = 100.0 * (cd_cyl - c_d) / (0.30 * cd_cyl);
        let geom_valid = n_solid >= LBM_MIN_SOLID_CELLS && core_intact;
        let score = if geom_valid {
            raw_score.clamp(0.0, 100.0)
        } else {
            0.0
        };
        let passed = geom_valid && c_d <= 0.82 * cd_cyl;
        let metrics = LbmFairingMetrics {
            f_drag,
            f_lift,
            c_d,
            c_d_cylinder: cd_cyl,
            n_solid,
            core_intact,
            score,
            passed,
        };
        self.latest = metrics;
        metrics
    }

    /// Formats a [`ChallengeEvaluation`] from the latest evaluated fairing metrics.
    #[must_use]
    pub fn challenge_evaluation(&self) -> ChallengeEvaluation {
        let m = &self.latest;
        let ratio = m.c_d / m.c_d_cylinder.max(1e-9);
        let core_label = if m.core_intact { "OK" } else { "BREACHED" };
        ChallengeEvaluation::new(
            m.score,
            m.passed,
            "C_D / C_D,cyl",
            ratio,
            format!(
                "C_D = {:.3} ({:.1}% of cyl {:.3}, need ≤ 82%), N_solid = {}/{} | Core R=4: {}",
                m.c_d,
                ratio * 100.0,
                m.c_d_cylinder,
                m.n_solid,
                LBM_MIN_SOLID_CELLS,
                core_label
            ),
        )
    }
}

fn generate_lbm_challenge_baseline() -> String {
    let mut mask = [false; LBM_CELL_COUNT];
    FairingStamp::BluffCylinder.write_mask(&mut mask);
    serde_json::to_string(&LbmSerializedState {
        viscosity: CHALLENGE_LBM_VISCOSITY,
        mask_hex: Some(encode_mask_hex(&mask)),
    })
    .unwrap_or_default()
}

fn clamp_lbm_challenge_state(raw_json: &str) -> Result<String, String> {
    let mut parsed: LbmSerializedState =
        serde_json::from_str(raw_json).map_err(|e| format!("Invalid LBM challenge JSON: {e}"))?;
    parsed.viscosity = CHALLENGE_LBM_VISCOSITY;
    let mut mask = [false; LBM_CELL_COUNT];
    if let Some(hex) = &parsed.mask_hex {
        decode_mask_hex(hex, &mut mask)?;
    } else {
        FairingStamp::BluffCylinder.write_mask(&mut mask);
    }
    restore_inlet_and_core_mask(&mut mask);
    parsed.mask_hex = Some(encode_mask_hex(&mask));
    serde_json::to_string(&parsed).map_err(|e| format!("LBM serialize error: {e}"))
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_LBM_FAIRING,
        title: "Low-Drag Fairing Sculpting",
        description: "Sculpt or stamp an aerodynamic fairing around the protected R_core = 4 structural core at (30, 25) with N_solid ≥ 65 cells that cuts bounce-back drag to C_D ≤ 0.82 · C_D_cylinder.",
        category: "Computational Fluids",
        icon: "🌊",
        domain: "fluid_dynamics",
        tool_name: "LatticeBoltzmannTool",
        locked_inputs: "Grid = 100×50, U_in = 0.1, ν = 0.02 (τ = 0.56), Core (30, 25) R_core = 4",
        tunable_budget: "Obstacle mask (stamps, mouse brush, or keyboard cursor) with N_solid ≥ 65",
        target_summary: "N_solid ≥ 65, R_core = 4 intact, and C_D ≤ 0.82 · C_D_cylinder",
        baseline_state_generator: generate_lbm_challenge_baseline,
        clamp_locked_state: clamp_lbm_challenge_state,
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]

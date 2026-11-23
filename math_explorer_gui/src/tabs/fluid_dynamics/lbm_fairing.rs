//! Preallocated Lattice Boltzmann fairing drag evaluator, obstacle stamps, and `lbm_low_drag_fairing` challenge.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ChallengeEvaluation, ChallengeMetadata};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
    compute_drag_coefficient, BgkCollision, LatticeBoltzmannD2Q9,
};
use serde::{Deserialize, Serialize};

/// Canonical LBM grid width.
pub const LBM_WIDTH: usize = 100;
/// Canonical LBM grid height.
pub const LBM_HEIGHT: usize = 50;
/// Total cell count (`100 * 50 = 5,000`).
pub const LBM_CELL_COUNT: usize = LBM_WIDTH * LBM_HEIGHT;
/// Packed byte length of the `5,000`-cell obstacle bitmask (`625` bytes).
pub const LBM_MASK_BYTES: usize = LBM_CELL_COUNT / 8;

/// Inlet column count along the left boundary (`x in 0..2`).
pub const LBM_INLET_COLS: usize = 2;
/// Horizontal inlet velocity `U_in` in lattice units.
pub const LBM_U_IN: f64 = 0.1;
/// Reference lattice density `rho_0`.
pub const LBM_RHO_0: f64 = 1.0;

/// Protected structural core X center.
pub const LBM_CORE_X: i32 = 30;
/// Protected structural core Y center.
pub const LBM_CORE_Y: i32 = 25;
/// Protected structural core radius `R_core = 4`.
pub const LBM_CORE_RADIUS: i32 = 4;
/// Baseline bluff cylinder radius `R_cyl = 5`.
pub const LBM_CYLINDER_RADIUS: i32 = 5;
/// Reference bluff-cylinder diameter `D_ref = 11.0` lattice units.
pub const LBM_REF_DIAMETER: f64 = 11.0;
/// Minimum required solid obstacle cells for `lbm_low_drag_fairing`.
pub const LBM_MIN_SOLID_CELLS: usize = 65;
/// Deterministic BGK + bounce-back steps for zero-allocation drag evaluation.
pub const LBM_EVAL_STEPS: usize = 120;

/// Canonical challenge ID for the LBM low-drag fairing inverse problem.
pub const CHALLENGE_LBM_FAIRING: &str = "lbm_low_drag_fairing";
/// Locked kinematic viscosity `nu` for `lbm_low_drag_fairing`.
pub const CHALLENGE_LBM_VISCOSITY: f64 = 0.02;

/// Built-in keyboard-accessible obstacle fairing stamps around `(30, 25)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FairingStamp {
    /// Baseline bluff circular cylinder (`R = 5`, `N_solid = 81`).
    BluffCylinder,
    /// Streamlined nose + tapered aft fairing hugging `R_core = 4`.
    TeardropFairing,
    /// Forebody symmetric wedge + `R_core = 4` cylinder core.
    SymmetricWedge,
}

impl FairingStamp {
    /// Display label used for buttons and ARIA status announcements.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::BluffCylinder => "Bluff Cylinder",
            Self::TeardropFairing => "Teardrop Fairing",
            Self::SymmetricWedge => "Symmetric Wedge",
        }
    }

    /// Worker command action string for synchronizing [`super::lattice_boltzmann::LbmUnified`].
    #[must_use]
    pub const fn command_name(self) -> &'static str {
        match self {
            Self::BluffCylinder => "Stamp: Bluff Cylinder",
            Self::TeardropFairing => "Stamp: Teardrop Fairing",
            Self::SymmetricWedge => "Stamp: Symmetric Wedge",
        }
    }

    /// Parses a worker custom command back into a [`FairingStamp`].
    #[must_use]
    pub fn from_command(action: &str) -> Option<Self> {
        match action {
            "Stamp: Bluff Cylinder" => Some(Self::BluffCylinder),
            "Stamp: Teardrop Fairing" => Some(Self::TeardropFairing),
            "Stamp: Symmetric Wedge" => Some(Self::SymmetricWedge),
            _ => None,
        }
    }

    /// Populates `mask` (`100 x 50`) with the geometric obstacle cells of `self`.
    pub fn write_mask(self, mask: &mut [bool; LBM_CELL_COUNT]) {
        mask.fill(false);
        for y in 0..LBM_HEIGHT {
            for x in LBM_INLET_COLS..LBM_WIDTH {
                let dx = x as i32 - LBM_CORE_X;
                let dy = y as i32 - LBM_CORE_Y;
                let in_core = dx * dx + dy * dy <= LBM_CORE_RADIUS * LBM_CORE_RADIUS;
                let in_shape = match self {
                    Self::BluffCylinder => {
                        dx * dx + dy * dy <= LBM_CYLINDER_RADIUS * LBM_CYLINDER_RADIUS
                    }
                    Self::TeardropFairing => {
                        let nose = (dx == -5 && dy.abs() <= 1) || (dx == -4 && dy.abs() <= 2);
                        let tail = ((4..=5).contains(&dx) && dy.abs() <= 2)
                            || ((6..=7).contains(&dx) && dy.abs() <= 1)
                            || (dx == 8 && dy == 0);
                        in_core || nose || tail
                    }
                    Self::SymmetricWedge => {
                        let fore = (dx == -6 && dy == 0)
                            || (dx == -5 && dy.abs() <= 1)
                            || (dx == -4 && dy.abs() <= 2)
                            || (dx == -3 && dy.abs() <= 3);
                        let aft =
                            ((4..=5).contains(&dx) && dy.abs() <= 2) || (dx == 6 && dy.abs() <= 1);
                        in_core || fore || aft
                    }
                };
                mask[flatten_2d_index(x, y, LBM_WIDTH)] = in_shape;
            }
        }
    }
}

/// Restores fluid inlet columns (`x < 2`) and solid core (`R_core = 4`) inside `mask`.
pub fn restore_inlet_and_core_mask(mask: &mut [bool; LBM_CELL_COUNT]) {
    for y in 0..LBM_HEIGHT {
        for x in 0..LBM_WIDTH {
            let idx = flatten_2d_index(x, y, LBM_WIDTH);
            if x < LBM_INLET_COLS {
                mask[idx] = false;
            } else {
                let dx = x as i32 - LBM_CORE_X;
                let dy = y as i32 - LBM_CORE_Y;
                if dx * dx + dy * dy <= LBM_CORE_RADIUS * LBM_CORE_RADIUS {
                    mask[idx] = true;
                }
            }
        }
    }
}

/// Applies an interpolated [`crate::framework::BrushStroke`] to `mask`, returning `true` if any cell changed.
pub fn apply_stroke_to_fairing_mask(
    mask: &mut [bool; LBM_CELL_COUNT],
    stroke: &crate::framework::BrushStroke,
    lock_core: bool,
) -> bool {
    let is_obs = match stroke.mode {
        crate::framework::BrushMode::Solid => true,
        crate::framework::BrushMode::Erase => false,
        _ => return false,
    };
    let (x0, y0) =
        crate::framework::SpatialBrushKit::plot_to_grid_y_up(stroke.start, LBM_WIDTH, LBM_HEIGHT);
    let (x1, y1) =
        crate::framework::SpatialBrushKit::plot_to_grid_y_up(stroke.end, LBM_WIDTH, LBM_HEIGHT);
    let r = (stroke.radius.round() as i32).clamp(0, 16);
    let mut changed = false;
    crate::framework::rasterize_disk_stroke(
        (x0, y0),
        (x1, y1),
        r,
        (LBM_WIDTH, LBM_HEIGHT),
        |gx, gy| {
            if gx < LBM_INLET_COLS {
                return;
            }
            let dx = gx as i32 - LBM_CORE_X;
            let dy = gy as i32 - LBM_CORE_Y;
            if !is_obs && lock_core && dx * dx + dy * dy <= LBM_CORE_RADIUS * LBM_CORE_RADIUS {
                return;
            }
            let idx = flatten_2d_index(gx, gy, LBM_WIDTH);
            if mask[idx] != is_obs {
                mask[idx] = is_obs;
                changed = true;
            }
        },
    );
    changed
}

/// Encodes `mask` into run-length encoded `[value, count]` pairs for `SculptStatePayload`.
#[must_use]
pub fn encode_fairing_mask_rle(mask: &[bool; LBM_CELL_COUNT]) -> Vec<[u32; 2]> {
    let bytes: Vec<u8> = mask.iter().map(|&b| u8::from(b)).collect();
    crate::framework::encode_rle_mask(&bytes)
}

/// Counts total solid cells and verifies every cell within `R_core = 4` of `(30, 25)` is solid.
#[must_use]
pub fn inspect_mask_geometry(mask: &[bool; LBM_CELL_COUNT]) -> (usize, bool) {
    let mut n_solid = 0_usize;
    let mut core_intact = true;
    for y in 0..LBM_HEIGHT {
        for x in 0..LBM_WIDTH {
            let is_obs = mask[flatten_2d_index(x, y, LBM_WIDTH)];
            if is_obs {
                n_solid += 1;
            }
            let dx = x as i32 - LBM_CORE_X;
            let dy = y as i32 - LBM_CORE_Y;
            if dx * dx + dy * dy <= LBM_CORE_RADIUS * LBM_CORE_RADIUS && !is_obs {
                core_intact = false;
            }
        }
    }
    (n_solid, core_intact)
}

/// Encodes a `5,000`-cell obstacle mask into `1,250` lowercase hex characters.
#[must_use]
pub fn encode_mask_hex(mask: &[bool; LBM_CELL_COUNT]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(LBM_MASK_BYTES * 2);
    for byte_idx in 0..LBM_MASK_BYTES {
        let mut b = 0_u8;
        for bit in 0..8 {
            if mask[byte_idx * 8 + bit] {
                b |= 1_u8 << bit;
            }
        }
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Decodes a `1,250`-character hex string into `mask`.
pub fn decode_mask_hex(hex: &str, mask: &mut [bool; LBM_CELL_COUNT]) -> Result<(), String> {
    if hex.len() != LBM_MASK_BYTES * 2 {
        return Err(format!(
            "Invalid LBM obstacle mask length {} (expected {})",
            hex.len(),
            LBM_MASK_BYTES * 2
        ));
    }
    let bytes = hex.as_bytes();
    for byte_idx in 0..LBM_MASK_BYTES {
        let hi = decode_nibble(bytes[byte_idx * 2])?;
        let lo = decode_nibble(bytes[byte_idx * 2 + 1])?;
        let packed = (hi << 4) | lo;
        for bit in 0..8 {
            mask[byte_idx * 8 + bit] = ((packed >> bit) & 1) != 0;
        }
    }
    Ok(())
}

fn decode_nibble(ch: u8) -> Result<u8, String> {
    match ch {
        b'0'..=b'9' => Ok(ch - b'0'),
        b'a'..=b'f' => Ok(ch - b'a' + 10),
        b'A'..=b'F' => Ok(ch - b'A' + 10),
        _ => Err(format!(
            "Invalid hex character '{}' in LBM mask",
            ch as char
        )),
    }
}

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

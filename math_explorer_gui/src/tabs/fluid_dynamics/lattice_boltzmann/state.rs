//! State schema, `625`-byte `base64url` obstacle mask codec, brush geometry,
//! preset shapes, and aerodynamic telemetry helpers for [`super::LatticeBoltzmannTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;

/// Wind-tunnel lattice width in cells.
pub const LBM_WIDTH: usize = 100;
/// Wind-tunnel lattice height in cells.
pub const LBM_HEIGHT: usize = 50;
/// Total cell count (`100 * 50 = 5,000`).
pub const LBM_CELL_COUNT: usize = LBM_WIDTH * LBM_HEIGHT;
/// Exact packed byte length of the `100 x 50` obstacle bitmask (`5,000 / 8 = 625`).
pub const LBM_MASK_BYTES: usize = LBM_CELL_COUNT / 8;
/// Exact unpadded `base64url` character length of [`LBM_MASK_BYTES`] (`ceil(625 * 4 / 3) = 834`).
pub const LBM_MASK_B64_CHARS: usize = 834;
/// Number of left-edge inlet columns (`x < 2`) guarded against solid obstacles.
pub const LBM_INLET_GUARD_COLS: usize = 2;

/// Minimum stable BGK kinematic viscosity (`tau = 0.51`).
pub const VISCOSITY_MIN: f64 = 0.00333;
/// Maximum BGK kinematic viscosity (`tau = 2.0`).
pub const VISCOSITY_MAX: f64 = 0.5;
/// Default kinematic viscosity matching legacy starter presets.
pub const DEFAULT_VISCOSITY: f64 = 0.02;

/// Minimum wind-tunnel horizontal inflow speed in lattice units.
pub const INFLOW_SPEED_MIN: f64 = 0.0;
/// Maximum stable low-Mach wind-tunnel horizontal inflow speed in lattice units.
pub const INFLOW_SPEED_MAX: f64 = 0.18;
/// Default horizontal inflow speed in lattice units (`Ma ≈ 0.17`).
pub const DEFAULT_INFLOW_SPEED: f64 = 0.10;

/// Minimum solver sub-steps executed per worker frame.
pub const STEPS_PER_FRAME_MIN: usize = 1;
/// Maximum solver sub-steps executed per worker frame.
pub const STEPS_PER_FRAME_MAX: usize = 24;
/// Default solver sub-steps executed per worker frame.
pub const DEFAULT_STEPS_PER_FRAME: usize = 6;

/// Minimum obstacle sculpting brush radius (`1` = single-cell precision).
pub const BRUSH_RADIUS_MIN: i32 = 1;
/// Maximum obstacle sculpting brush radius.
pub const BRUSH_RADIUS_MAX: i32 = 5;
/// Default obstacle sculpting brush radius.
pub const DEFAULT_BRUSH_RADIUS: i32 = 2;

/// Minimum configurable solid-cell area budget.
pub const AREA_BUDGET_MIN: usize = 10;
/// Maximum configurable solid-cell area budget.
pub const AREA_BUDGET_MAX: usize = 1_500;
/// Default solid-cell area budget for minimum-drag sculpting challenges.
pub const DEFAULT_AREA_BUDGET: usize = 250;

/// Scalar field rendered on the wind-tunnel canvas.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LbmViewMode {
    /// Velocity magnitude `|u|` rendered via sequential heatmap.
    #[default]
    Speed,
    /// Signed central-difference curl `dv_y/dx - dv_x/dy` rendered via diverging colormap.
    Vorticity,
}

/// Primary-pointer brush action on the wind-tunnel canvas.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum LbmBrushMode {
    /// Paint solid bounce-back cells up to the active area budget.
    #[default]
    Solid,
    /// Erase solid cells back to fluid.
    Erase,
}

/// Live hydrodynamic force and dimensionless telemetry reported by the LBM solver.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct LbmTelemetry {
    /// Instantaneous horizontal fluid-on-obstacle drag force `F_x`.
    pub drag_force: f64,
    /// Instantaneous vertical fluid-on-obstacle lift force `F_y`.
    pub lift_force: f64,
    /// Dimensionless drag coefficient `C_D = F_x / (0.5 * U^2 * D)`.
    pub cd: f64,
    /// Dimensionless lift coefficient `C_L = F_y / (0.5 * U^2 * D)`.
    pub cl: f64,
    /// Reynolds number `Re = U * D / viscosity`.
    pub reynolds: f64,
    /// Total count of solid obstacle cells in the tunnel.
    pub solid_cells: usize,
    /// Projected vertical obstacle height `D` in lattice cells.
    pub frontal_height: usize,
    /// Cumulative LBM steps executed since initialization or reset.
    pub step_count: usize,
}

fn default_inflow_speed() -> f64 {
    DEFAULT_INFLOW_SPEED
}

fn default_steps_per_frame() -> usize {
    DEFAULT_STEPS_PER_FRAME
}

fn default_brush_radius() -> i32 {
    DEFAULT_BRUSH_RADIUS
}

fn default_area_budget() -> usize {
    DEFAULT_AREA_BUDGET
}

fn default_obstacle_mask() -> String {
    encode_obstacle_mask(&cylinder_obstacle_grid())
}

/// Serializable preset and URL state for [`super::LatticeBoltzmannTool`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LbmToolState {
    /// Kinematic viscosity `nu = (tau - 0.5) / 3` (required so empty `{}` fails validation).
    pub viscosity: f64,
    /// Horizontal inlet velocity `U` in lattice units.
    #[serde(default = "default_inflow_speed")]
    pub inflow_speed: f64,
    /// Solver steps per background worker tick.
    #[serde(default = "default_steps_per_frame")]
    pub steps_per_frame: usize,
    /// Sculpting brush radius in `1..=5`.
    #[serde(default = "default_brush_radius")]
    pub brush_radius: i32,
    /// Maximum allowed solid obstacle cell count.
    #[serde(default = "default_area_budget")]
    pub area_budget: usize,
    /// Active scalar visualization mode.
    #[serde(default)]
    pub view_mode: LbmViewMode,
    /// Packed `625`-byte (`834`-char `base64url`) obstacle bitmask.
    #[serde(default = "default_obstacle_mask")]
    pub obstacle_mask: String,
    /// Optional `1,250`-char hex obstacle bitmask accepted from `lbm_low_drag_fairing` challenge baselines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask_hex: Option<String>,
}

impl LbmToolState {
    /// Validates parameter finiteness, discrete bounds, `625`-byte mask integrity, and area budget.
    pub fn validate(&self) -> Result<Vec<bool>, String> {
        if !self.viscosity.is_finite() || self.viscosity <= 0.0 {
            return Err(format!(
                "viscosity ({}) must be finite and strictly positive",
                self.viscosity
            ));
        }
        if !self.inflow_speed.is_finite() || self.inflow_speed < 0.0 {
            return Err(format!(
                "inflow_speed ({}) must be finite and non-negative",
                self.inflow_speed
            ));
        }
        if !(STEPS_PER_FRAME_MIN..=STEPS_PER_FRAME_MAX).contains(&self.steps_per_frame) {
            return Err(format!(
                "steps_per_frame ({}) out of range [{STEPS_PER_FRAME_MIN}, {STEPS_PER_FRAME_MAX}]",
                self.steps_per_frame
            ));
        }
        if !(BRUSH_RADIUS_MIN..=BRUSH_RADIUS_MAX).contains(&self.brush_radius) {
            return Err(format!(
                "brush_radius ({}) out of range [{BRUSH_RADIUS_MIN}, {BRUSH_RADIUS_MAX}]",
                self.brush_radius
            ));
        }
        if !(AREA_BUDGET_MIN..=AREA_BUDGET_MAX).contains(&self.area_budget) {
            return Err(format!(
                "area_budget ({}) out of range [{AREA_BUDGET_MIN}, {AREA_BUDGET_MAX}]",
                self.area_budget
            ));
        }
        let obstacles = if let Some(hex) = &self.mask_hex {
            let mut arr = [false; LBM_CELL_COUNT];
            super::super::lbm_fairing::decode_mask_hex(hex, &mut arr)?;
            for (idx, cell) in arr.iter_mut().enumerate() {
                if idx % LBM_WIDTH < LBM_INLET_GUARD_COLS {
                    *cell = false;
                }
            }
            arr.to_vec()
        } else {
            decode_obstacle_mask(&self.obstacle_mask)?
        };
        let solid = count_solid_cells(&obstacles);
        if solid > self.area_budget {
            return Err(format!(
                "Decoded obstacle mask has {solid} solid cells, exceeding area_budget ({})",
                self.area_budget
            ));
        }
        Ok(obstacles)
    }
}

/// Packs a `100 x 50` obstacle slice into `625` bytes (LSB-first per cell index `y * 100 + x`).
#[must_use]
pub fn pack_obstacle_bits(obstacles: &[bool]) -> [u8; LBM_MASK_BYTES] {
    let mut packed = [0_u8; LBM_MASK_BYTES];
    for (idx, &is_solid) in obstacles.iter().take(LBM_CELL_COUNT).enumerate() {
        let x = idx % LBM_WIDTH;
        if is_solid && x >= LBM_INLET_GUARD_COLS {
            packed[idx / 8] |= 1_u8 << (idx % 8);
        }
    }
    packed
}

/// Encodes a `100 x 50` obstacle grid into an `834`-character unpadded `base64url` string.
#[must_use]
pub fn encode_obstacle_mask(obstacles: &[bool]) -> String {
    let packed = pack_obstacle_bits(obstacles);
    let encoded = crate::url_state::encode_base64url(&packed);
    debug_assert_eq!(encoded.len(), LBM_MASK_B64_CHARS);
    encoded
}

/// Decodes a `base64url` obstacle mask, strictly requiring `625` decoded bytes and clearing `x < 2`.
pub fn decode_obstacle_mask(encoded: &str) -> Result<Vec<bool>, String> {
    if encoded.len() < LBM_MASK_B64_CHARS || encoded.len() > LBM_MASK_B64_CHARS + 2 {
        return Err(format!(
            "Invalid LBM obstacle mask length: expected {LBM_MASK_B64_CHARS} base64url chars, got {}",
            encoded.len()
        ));
    }
    let bytes = crate::url_state::decode_base64url(encoded, LBM_MASK_BYTES)?;
    if bytes.len() != LBM_MASK_BYTES {
        return Err(format!(
            "Invalid LBM obstacle mask length: expected {LBM_MASK_BYTES} bytes, got {} bytes",
            bytes.len()
        ));
    }
    let mut obstacles = vec![false; LBM_CELL_COUNT];
    for (idx, slot) in obstacles.iter_mut().enumerate() {
        let x = idx % LBM_WIDTH;
        if x >= LBM_INLET_GUARD_COLS {
            *slot = ((bytes[idx / 8] >> (idx % 8)) & 1) != 0;
        }
    }
    Ok(obstacles)
}

/// Counts solid cells in the obstacle mask.
#[must_use]
pub fn count_solid_cells(obstacles: &[bool]) -> usize {
    obstacles.iter().filter(|&&b| b).count()
}

/// Computes projected frontal obstacle height `D` (distinct rows containing any solid cell).
#[must_use]
pub fn projected_obstacle_height(obstacles: &[bool]) -> usize {
    (0..LBM_HEIGHT)
        .filter(|&y| {
            let row_start = flatten_2d_index(0, y, LBM_WIDTH);
            obstacles[row_start..(row_start + LBM_WIDTH)]
                .iter()
                .any(|&solid| solid)
        })
        .count()
}

/// Computes dimensionless `C_D`, `C_L`, and `Re` from bounce-back forces (`0.0` when `D == 0` or `U == 0`).
#[must_use]
pub fn compute_aerodynamics(
    force: (f64, f64),
    inflow_speed: f64,
    viscosity: f64,
    obstacles: &[bool],
    step_count: usize,
) -> LbmTelemetry {
    let solid_cells = count_solid_cells(obstacles);
    let frontal_height = projected_obstacle_height(obstacles);
    let drag_force = if force.0.is_finite() { force.0 } else { 0.0 };
    let lift_force = if force.1.is_finite() { force.1 } else { 0.0 };
    if frontal_height == 0 || !inflow_speed.is_finite() || inflow_speed <= 0.0 {
        return LbmTelemetry {
            drag_force,
            lift_force,
            cd: 0.0,
            cl: 0.0,
            reynolds: 0.0,
            solid_cells,
            frontal_height,
            step_count,
        };
    }
    let d = frontal_height as f64;
    let denom = 0.5 * inflow_speed * inflow_speed * d;
    let cd = if denom > 0.0 { drag_force / denom } else { 0.0 };
    let cl = if denom > 0.0 { lift_force / denom } else { 0.0 };
    let reynolds = if viscosity.is_finite() && viscosity > 0.0 {
        inflow_speed * d / viscosity
    } else {
        0.0
    };
    LbmTelemetry {
        drag_force,
        lift_force,
        cd,
        cl,
        reynolds,
        solid_cells,
        frontal_height,
        step_count,
    }
}

/// Converts continuous lattice coordinates `(lx, ly)` into clamped cell indices via `floor()`.
#[must_use]
pub fn lattice_coord_to_cell(lx: f64, ly: f64) -> Option<(usize, usize)> {
    if !lx.is_finite() || !ly.is_finite() {
        return None;
    }
    let max_x = (LBM_WIDTH - 1) as f64;
    let max_y = (LBM_HEIGHT - 1) as f64;
    let gx = lx.floor().clamp(0.0, max_x) as usize;
    let gy = ly.floor().clamp(0.0, max_y) as usize;
    Some((gx, gy))
}

/// Maps a screen pointer position inside `rect` to `(gx, gy)` using `floor()` and vertical image flip.
#[must_use]
pub fn screen_pos_to_cell(pos: egui::Pos2, rect: egui::Rect) -> Option<(usize, usize)> {
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return None;
    }
    let lx = (f64::from(pos.x - rect.left()) / f64::from(rect.width())) * (LBM_WIDTH as f64);
    let ly = (f64::from(rect.bottom() - pos.y) / f64::from(rect.height())) * (LBM_HEIGHT as f64);
    lattice_coord_to_cell(lx, ly)
}

/// Applies a circular brush (`dx^2 + dy^2 < r^2`, `r in 1..=5`) respecting `x >= 2` and `area_budget`.
pub fn apply_brush_with_budget(
    obstacles: &mut [bool],
    cx: usize,
    cy: usize,
    radius: i32,
    paint_solid: bool,
    area_budget: usize,
) -> usize {
    let r = radius.clamp(BRUSH_RADIUS_MIN, BRUSH_RADIUS_MAX);
    let r_sq = r * r;
    let mut solid_count = count_solid_cells(obstacles);
    let cx_i = cx as i32;
    let cy_i = cy as i32;

    for y in (cy_i - r + 1)..=(cy_i + r - 1) {
        for x in (cx_i - r + 1)..=(cx_i + r - 1) {
            if x < LBM_INLET_GUARD_COLS as i32
                || x >= LBM_WIDTH as i32
                || y < 0
                || y >= LBM_HEIGHT as i32
            {
                continue;
            }
            let dx = x - cx_i;
            let dy = y - cy_i;
            if dx * dx + dy * dy >= r_sq {
                continue;
            }
            let idx = flatten_2d_index(x as usize, y as usize, LBM_WIDTH);
            if paint_solid {
                if !obstacles[idx] && solid_count < area_budget {
                    obstacles[idx] = true;
                    solid_count += 1;
                }
            } else if obstacles[idx] {
                obstacles[idx] = false;
                solid_count = solid_count.saturating_sub(1);
            }
        }
    }
    solid_count
}

/// Trims trailing solid cells if `area_budget` is lowered below the current solid count.
pub fn enforce_area_budget(obstacles: &mut [bool], area_budget: usize) {
    for (idx, cell) in obstacles.iter_mut().enumerate() {
        if idx % LBM_WIDTH < LBM_INLET_GUARD_COLS {
            *cell = false;
        }
    }
    let mut solid = count_solid_cells(obstacles);
    if solid <= area_budget {
        return;
    }
    for cell in obstacles.iter_mut().rev() {
        if *cell {
            *cell = false;
            solid -= 1;
            if solid <= area_budget {
                break;
            }
        }
    }
}

/// Default circular cylinder obstacle (`cx = 28`, `cy = 25`, `r = 5`, `81` solid cells, `D = 11`).
#[must_use]
pub fn cylinder_obstacle_grid() -> Vec<bool> {
    let mut grid = vec![false; LBM_CELL_COUNT];
    for y in 0..LBM_HEIGHT {
        for x in LBM_INLET_GUARD_COLS..LBM_WIDTH {
            let dx = x as i32 - 28;
            let dy = y as i32 - 25;
            if dx * dx + dy * dy <= 25 {
                grid[flatten_2d_index(x, y, LBM_WIDTH)] = true;
            }
        }
    }
    grid
}

/// Vertical flat bluff-plate obstacle (`x in 26..=28`, `y in 18..=32`, `45` solid cells, `D = 15`).
#[must_use]
pub fn bluff_plate_obstacle_grid() -> Vec<bool> {
    let mut grid = vec![false; LBM_CELL_COUNT];
    for y in 18..=32 {
        for x in 26..=28 {
            grid[flatten_2d_index(x, y, LBM_WIDTH)] = true;
        }
    }
    grid
}

/// Symmetric NACA-inspired streamlined airfoil (`x in 18..=46`, max height `D = 11`, `<= 195` cells).
#[must_use]
pub fn streamlined_airfoil_obstacle_grid() -> Vec<bool> {
    let mut grid = vec![false; LBM_CELL_COUNT];
    let x_lead = 18.0_f64;
    let chord = 28.0_f64;
    let half_max = 5.2_f64;
    for x in 18..=46 {
        let s = ((x as f64) - x_lead) / chord;
        let profile = 5.0
            * (0.2969 * s.sqrt() - 0.1260 * s - 0.3516 * s * s + 0.2843 * s.powi(3)
                - 0.1015 * s.powi(4));
        let half_t = (half_max * profile).max(0.0);
        for y in 0..LBM_HEIGHT {
            let dy = ((y as f64) - 25.0).abs();
            if dy <= half_t {
                grid[flatten_2d_index(x, y, LBM_WIDTH)] = true;
            }
        }
    }
    grid
}

//! Canonical `100 x 50` Lattice Boltzmann fairing stamps, brush rasterization, and `1,250`-char hex mask codec.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_commons::math_kernel::types::flatten_2d_index;
use serde::{Deserialize, Serialize};

/// Canonical LBM grid width (`100` cells).
pub const LBM_WIDTH: usize = 100;
/// Canonical LBM grid height (`50` cells).
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

/// Built-in keyboard-accessible obstacle fairing stamps around `(30, 25)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FairingStamp {
    /// Baseline bluff circular cylinder (`R = 5`, `N_solid = 81`).
    #[default]
    BluffCylinder,
    /// Streamlined nose + tapered aft fairing hugging `R_core = 4`.
    TeardropFairing,
    /// Forebody symmetric wedge + `R_core = 4` cylinder core.
    SymmetricWedge,
}

impl FairingStamp {
    /// Ordered array of all built-in fairing stamps.
    pub const ALL: [FairingStamp; 3] = [
        Self::BluffCylinder,
        Self::TeardropFairing,
        Self::SymmetricWedge,
    ];

    /// Display label used for buttons and ARIA status announcements.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::BluffCylinder => "Bluff Cylinder",
            Self::TeardropFairing => "Teardrop Fairing",
            Self::SymmetricWedge => "Symmetric Wedge",
        }
    }

    /// Worker command action string for synchronizing LBM workers.
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

/// Applies an interpolated [`super::BrushStroke`] to `mask`, returning `true` if any cell changed.
pub fn apply_stroke_to_fairing_mask(
    mask: &mut [bool; LBM_CELL_COUNT],
    stroke: &super::BrushStroke,
    lock_core: bool,
) -> bool {
    let is_obs = match stroke.mode {
        super::BrushMode::Solid => true,
        super::BrushMode::Erase => false,
        _ => return false,
    };
    let (x0, y0) = super::SpatialBrushKit::plot_to_grid_y_up(stroke.start, LBM_WIDTH, LBM_HEIGHT);
    let (x1, y1) = super::SpatialBrushKit::plot_to_grid_y_up(stroke.end, LBM_WIDTH, LBM_HEIGHT);
    let r = (stroke.radius.round() as i32).clamp(0, 16);
    let mut changed = false;
    super::rasterize_disk_stroke((x0, y0), (x1, y1), r, (LBM_WIDTH, LBM_HEIGHT), |gx, gy| {
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
    });
    changed
}

/// Toggles a single cell `(gx, gy)` inside `mask` while protecting inlet columns (`gx < 2`).
pub fn toggle_fairing_cell(
    mask: &mut [bool; LBM_CELL_COUNT],
    gx: usize,
    gy: usize,
    lock_core: bool,
) -> Option<bool> {
    if !(LBM_INLET_COLS..LBM_WIDTH).contains(&gx) || gy >= LBM_HEIGHT {
        return None;
    }
    let idx = flatten_2d_index(gx, gy, LBM_WIDTH);
    let next = !mask[idx];
    let dx = gx as i32 - LBM_CORE_X;
    let dy = gy as i32 - LBM_CORE_Y;
    if !next && lock_core && dx * dx + dy * dy <= LBM_CORE_RADIUS * LBM_CORE_RADIUS {
        return None;
    }
    mask[idx] = next;
    Some(next)
}

/// Encodes `mask` into run-length encoded `[value, count]` pairs for `SculptStatePayload`.
#[must_use]
pub fn encode_fairing_mask_rle(mask: &[bool; LBM_CELL_COUNT]) -> Vec<[u32; 2]> {
    let bytes: Vec<u8> = mask.iter().map(|&b| u8::from(b)).collect();
    super::encode_rle_mask(&bytes)
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

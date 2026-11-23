//! Quantized acoustic cell roles, `< 4 KiB` bounded RLE codec, and preset label/pin helpers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::analysis::pde::wave_2d::{apply_point_clamp, Wave2DSolver};

use super::{ClampLayout, TankPreset, WaveColormap, GRID_SIZE};
use crate::framework::{decode_rle_mask, encode_rle_mask};

/// Upper bound on RLE run pairs serialized into `sculpt_rle` to stay strictly `< 4 KiB` URL JSON.
pub(crate) const MAX_SCULPT_RLE_RUNS: usize = 400;
const MAX_SCULPT_RLE_BYTES: usize = 2_850;

impl TankPreset {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::SquareChladniPlate => "Square Chladni Plate",
            Self::CircularDrumhead => "Circular Drumhead",
            Self::DoubleSlitInterference => "Double-Slit Interference",
            Self::AcousticLens => "Acoustic Lens",
        }
    }
}

impl ClampLayout {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::None => "Free / Perimeter Only",
            Self::CenterPin => "Center Nodal Pin",
            Self::QuadSymmetry => "Fourfold Symmetry Pins",
            Self::EdgeMidpoints => "Edge Midpoint Clamps",
        }
    }
}

impl WaveColormap {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::PerceptualThermal => "Perceptual Thermal",
            Self::BipolarPhase => "Bipolar Wave Phase",
            Self::WcagHighContrast => "WCAG High Contrast",
        }
    }
}

/// Quantized `u8` cell medium role (`0` baseline, `1` rigid clamp, `2..=7` graded speed/stiffness).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub(crate) enum AcousticCellRole {
    #[default]
    Baseline = 0,
    RigidClamp = 1,
    DeepSlowLens = 2,
    MediumSlowLens = 3,
    MildSlowLens = 4,
    SpruceRib = 5,
    CarbonBrace = 6,
    PhononicScatterer = 7,
}

impl AcousticCellRole {
    pub(crate) const fn from_u8(byte: u8) -> Self {
        match byte {
            1 => Self::RigidClamp,
            2 => Self::DeepSlowLens,
            3 => Self::MediumSlowLens,
            4 => Self::MildSlowLens,
            5 => Self::SpruceRib,
            6 => Self::CarbonBrace,
            7 => Self::PhononicScatterer,
            _ => Self::Baseline,
        }
    }

    /// Relative wave/flexural speed scale $s(x, y) = 1 / n(x, y)$.
    pub(crate) const fn speed_scale(self) -> f64 {
        match self {
            Self::Baseline | Self::RigidClamp => 1.0,
            Self::DeepSlowLens => 0.40,
            Self::MediumSlowLens => 0.555_555_6,
            Self::MildSlowLens => 0.740_740_7,
            Self::SpruceRib => 1.25,
            Self::CarbonBrace => 1.538_461_5,
            Self::PhononicScatterer => 0.333_333_3,
        }
    }

    /// Quantizes a continuous refractive index $n(x, y)$ into a discrete `u8` role.
    #[cfg(feature = "physics")]
    pub(crate) fn from_refractive_index(n: f64) -> Self {
        if !n.is_finite() || (n - 1.0).abs() < 0.12 {
            Self::Baseline
        } else if n >= 2.75 {
            Self::PhononicScatterer
        } else if n >= 2.15 {
            Self::DeepSlowLens
        } else if n >= 1.55 {
            Self::MediumSlowLens
        } else if n > 1.12 {
            Self::MildSlowLens
        } else if n <= 0.72 {
            Self::CarbonBrace
        } else {
            Self::SpruceRib
        }
    }
}

pub(crate) fn is_valid_sculpt_rle(runs: &[[u32; 2]]) -> bool {
    if runs.is_empty() {
        return true;
    }
    if runs.len() > 1_024 {
        return false;
    }
    let expected = GRID_SIZE * GRID_SIZE;
    let mut sum = 0_usize;
    for &[val, count] in runs {
        if val > 7 || count == 0 {
            return false;
        }
        let Some(next) = sum.checked_add(count as usize) else {
            return false;
        };
        if next > expected {
            return false;
        }
        sum = next;
    }
    sum == expected
}

pub(crate) fn decode_sculpt_rle_into(runs: &[[u32; 2]], mask: &mut Vec<u8>) {
    let total = GRID_SIZE * GRID_SIZE;
    if let Ok(decoded) = decode_rle_mask(runs, total) {
        *mask = decoded;
    } else {
        mask.clear();
        mask.resize(total, 0);
    }
}

fn rle_fits_url_budget(runs: &[[u32; 2]]) -> bool {
    if runs.len() > MAX_SCULPT_RLE_RUNS {
        return false;
    }
    let est_bytes: usize = runs
        .iter()
        .map(|&[_v, c]| 6 + c.max(1).ilog10() as usize)
        .sum();
    est_bytes <= MAX_SCULPT_RLE_BYTES
}

pub(crate) fn encode_bounded_sculpt_rle(mask: &[u8], width: usize, height: usize) -> Vec<[u32; 2]> {
    if mask.iter().all(|&b| b == 0) {
        return Vec::new();
    }
    let direct = encode_rle_mask(mask);
    if rle_fits_url_budget(&direct) {
        return direct;
    }
    for (bx_step, by_step) in [(2, 2), (4, 2), (8, 2), (16, 2), (32, 1), (64, 1), (128, 1)] {
        let coarsened = coarsen_mask_blocks(mask, width, height, bx_step, by_step);
        if coarsened.iter().all(|&b| b == 0) {
            continue;
        }
        let runs = encode_rle_mask(&coarsened);
        if rle_fits_url_budget(&runs) {
            return runs;
        }
    }
    Vec::new()
}

fn coarsen_mask_blocks(
    mask: &[u8],
    width: usize,
    height: usize,
    bx_step: usize,
    by_step: usize,
) -> Vec<u8> {
    let mut out = vec![0_u8; width * height];
    let mut by = 0;
    while by < height {
        let y_end = (by + by_step).min(height);
        let mut bx = 0;
        while bx < width {
            let x_end = (bx + bx_step).min(width);
            let dominant = dominant_block_role(mask, width, bx, x_end, by, y_end);
            for y in by..y_end {
                for x in bx..x_end {
                    out[flatten_2d_index(x, y, width)] = dominant;
                }
            }
            bx += bx_step;
        }
        by += by_step;
    }
    out
}

fn dominant_block_role(
    mask: &[u8],
    width: usize,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
) -> u8 {
    let mut counts = [0_u16; 8];
    for y in y0..y1 {
        for x in x0..x1 {
            let r = mask[flatten_2d_index(x, y, width)].min(7) as usize;
            counts[r] = counts[r].saturating_add(1);
        }
    }
    let mut best_role = 0_u8;
    let mut best_cnt = 0_u16;
    for (role, &cnt) in counts.iter().enumerate().skip(1) {
        if cnt > best_cnt {
            best_cnt = cnt;
            best_role = role as u8;
        }
    }
    if best_cnt > 0 && usize::from(best_cnt) * 8 >= (x1 - x0) * (y1 - y0) {
        best_role
    } else {
        0
    }
}

pub(crate) fn apply_preset_clamp_pins(layout: ClampLayout, wall_mask: &mut [f64]) {
    let (pins, radius): (&[(f64, f64)], f64) = match layout {
        ClampLayout::None => (&[], 0.0),
        ClampLayout::CenterPin => (&[(0.5, 0.5)], 2.0),
        ClampLayout::QuadSymmetry => (&[(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)], 1.8),
        ClampLayout::EdgeMidpoints => (&[(0.5, 0.04), (0.5, 0.96), (0.04, 0.5), (0.96, 0.5)], 2.2),
    };
    for &(cx, cy) in pins {
        apply_point_clamp(wall_mask, GRID_SIZE, GRID_SIZE, cx, cy, radius);
    }
}

pub(crate) fn replay_sculpt_mask_on_solver(sculpt_mask: &[u8], solver: &mut Wave2DSolver) {
    let n = sculpt_mask
        .len()
        .min(solver.wall_mask.len())
        .min(solver.wave_speed_mask.len());
    for (idx, &raw) in sculpt_mask.iter().enumerate().take(n) {
        let role = AcousticCellRole::from_u8(raw);
        match role {
            AcousticCellRole::Baseline => {}
            AcousticCellRole::RigidClamp => {
                solver.wall_mask[idx] = 0.0;
            }
            _ => {
                solver.wall_mask[idx] = 1.0;
                solver.wave_speed_mask[idx] = role.speed_scale();
            }
        }
    }
}

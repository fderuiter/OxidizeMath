//! Custom PTV/OAR raster contour sculpting, compact RLE mask codecs,
//! `dice_similarity_coefficient` evaluation, and vector boundary extraction.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    voxel_coord_cm, IMRT_GRID_DIM, IMRT_OAR_CENTER, IMRT_OAR_RADIUS_CM, IMRT_PHANTOM_SIZE_CM,
    IMRT_TUMOR_CENTER, IMRT_TUMOR_RADIUS_CM, IMRT_VOXEL_COUNT,
};
use crate::framework::{
    decode_rle_mask, encode_rle_mask, rasterize_disk_stroke, BrushMode, BrushStroke,
};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::medical::optical_motion::validation::dice_similarity_coefficient;

/// Resets `ptv_mask` and `oar_mask` to the canonical circular phantom geometry
/// and rebuilds `tumor_indices` and `organ_indices` in place with zero heap allocations.
pub fn reset_circular_contours(
    ptv_mask: &mut [bool; IMRT_VOXEL_COUNT],
    oar_mask: &mut [bool; IMRT_VOXEL_COUNT],
    tumor_indices: &mut Vec<usize>,
    organ_indices: &mut Vec<usize>,
) {
    let rt2 = IMRT_TUMOR_RADIUS_CM * IMRT_TUMOR_RADIUS_CM;
    let ro2 = IMRT_OAR_RADIUS_CM * IMRT_OAR_RADIUS_CM;
    for y in 0..IMRT_GRID_DIM {
        let py = voxel_coord_cm(y);
        for x in 0..IMRT_GRID_DIM {
            let px = voxel_coord_cm(x);
            let idx = flatten_2d_index(x, y, IMRT_GRID_DIM);
            let dt2 = (px - IMRT_TUMOR_CENTER[0]).powi(2) + (py - IMRT_TUMOR_CENTER[1]).powi(2);
            let do2 = (px - IMRT_OAR_CENTER[0]).powi(2) + (py - IMRT_OAR_CENTER[1]).powi(2);
            ptv_mask[idx] = dt2 <= rt2;
            oar_mask[idx] = !ptv_mask[idx] && do2 <= ro2;
        }
    }
    rebuild_indices_from_masks(ptv_mask, oar_mask, tumor_indices, organ_indices);
}

/// Rebuilds `tumor_indices` and `organ_indices` in place from `ptv_mask` and `oar_mask`
/// without reallocating pre-reserved `Vec` capacities.
pub fn rebuild_indices_from_masks(
    ptv_mask: &[bool; IMRT_VOXEL_COUNT],
    oar_mask: &[bool; IMRT_VOXEL_COUNT],
    tumor_indices: &mut Vec<usize>,
    organ_indices: &mut Vec<usize>,
) {
    tumor_indices.clear();
    organ_indices.clear();
    for idx in 0..IMRT_VOXEL_COUNT {
        if ptv_mask[idx] {
            tumor_indices.push(idx);
        } else if oar_mask[idx] {
            organ_indices.push(idx);
        }
    }
}

/// Maps phantom plot coordinates `(x_cm, y_cm)` in `[-10.0, 10.0]^2` to integer grid indices `(gx, gy)`.
#[must_use]
pub fn plot_cm_to_grid(pt: [f64; 2]) -> (i32, i32) {
    let step = IMRT_PHANTOM_SIZE_CM / IMRT_GRID_DIM as f64;
    let max_idx = (IMRT_GRID_DIM - 1) as i32;
    let gx = (((pt[0] + 0.5 * IMRT_PHANTOM_SIZE_CM) / step).floor() as i32).clamp(0, max_idx);
    let gy = (((pt[1] + 0.5 * IMRT_PHANTOM_SIZE_CM) / step).floor() as i32).clamp(0, max_idx);
    (gx, gy)
}

/// Applies a [`BrushStroke`] (`PtvTarget`, `OrganAtRisk`, or `Erase`) to the `64 x 64` masks
/// and updates index lists, returning `true` if at least one voxel changed.
pub fn paint_contour_stroke(
    ptv_mask: &mut [bool; IMRT_VOXEL_COUNT],
    oar_mask: &mut [bool; IMRT_VOXEL_COUNT],
    tumor_indices: &mut Vec<usize>,
    organ_indices: &mut Vec<usize>,
    stroke: &BrushStroke,
) -> bool {
    if !matches!(
        stroke.mode,
        BrushMode::PtvTarget | BrushMode::OrganAtRisk | BrushMode::Erase
    ) {
        return false;
    }
    let step_cm = IMRT_PHANTOM_SIZE_CM / IMRT_GRID_DIM as f64;
    let start_g = plot_cm_to_grid(stroke.start);
    let end_g = plot_cm_to_grid(stroke.end);
    let r_cells = ((stroke.radius.max(0.0) / step_cm).round() as i32).clamp(0, 24);
    let mut changed = false;

    rasterize_disk_stroke(
        start_g,
        end_g,
        r_cells,
        (IMRT_GRID_DIM, IMRT_GRID_DIM),
        |gx, gy| {
            let idx = flatten_2d_index(gx, gy, IMRT_GRID_DIM);
            let (next_ptv, next_oar) = match stroke.mode {
                BrushMode::PtvTarget => (true, false),
                BrushMode::OrganAtRisk => (false, true),
                BrushMode::Erase => (false, false),
                _ => (ptv_mask[idx], oar_mask[idx]),
            };
            if ptv_mask[idx] != next_ptv || oar_mask[idx] != next_oar {
                ptv_mask[idx] = next_ptv;
                oar_mask[idx] = next_oar;
                changed = true;
            }
        },
    );

    if changed {
        rebuild_indices_from_masks(ptv_mask, oar_mask, tumor_indices, organ_indices);
    }
    changed
}

/// Compact `[value, run_length]` run-length encoding for a `64 x 64` contour mask.
pub type RleMaskRuns = Vec<[u32; 2]>;

/// Encodes a boolean `64 x 64` mask via `encode_rle_mask` into compact `[value, run_length]` pairs.
#[must_use]
pub fn encode_bool_mask_rle(mask: &[bool; IMRT_VOXEL_COUNT]) -> RleMaskRuns {
    let mut bytes = [0_u8; IMRT_VOXEL_COUNT];
    for (dst, &flag) in bytes.iter_mut().zip(mask.iter()) {
        *dst = u8::from(flag);
    }
    encode_rle_mask(&bytes)
}

/// Decodes `[value, run_length]` pairs via `decode_rle_mask` into a boolean `64 x 64` mask.
pub fn decode_bool_mask_rle(
    runs: &[[u32; 2]],
    dst: &mut [bool; IMRT_VOXEL_COUNT],
) -> Result<(), String> {
    let decoded = decode_rle_mask(runs, IMRT_VOXEL_COUNT)?;
    for (slot, byte) in dst.iter_mut().zip(decoded) {
        if byte > 1 {
            return Err(format!("Boolean contour RLE cell value {byte} exceeds 1"));
        }
        *slot = byte == 1;
    }
    Ok(())
}

/// Computes the verified `dice_similarity_coefficient` between the local `90%` prescription
/// isodose mask (`dose >= 0.90 * prescription_gy` within the `7`-voxel ICRU conformity ring
/// around `ptv_mask`) and `ptv_mask` with zero heap allocations.
pub fn compute_isodose_dice(
    dose_slice: &[f64],
    ptv_mask: &[bool; IMRT_VOXEL_COUNT],
    isodose_buf: &mut [bool; IMRT_VOXEL_COUNT],
    prescription_gy: f64,
) -> f64 {
    let Some((x0, x1, y0, y1)) = ptv_roi_bounds(ptv_mask, 7) else {
        isodose_buf.fill(false);
        return 0.0;
    };
    let threshold = (0.90 * prescription_gy).max(0.5);
    for y in 0..IMRT_GRID_DIM {
        let in_y = y >= y0 && y <= y1;
        for x in 0..IMRT_GRID_DIM {
            let idx = flatten_2d_index(x, y, IMRT_GRID_DIM);
            let d = dose_slice.get(idx).copied().unwrap_or(0.0);
            isodose_buf[idx] = in_y && x >= x0 && x <= x1 && d.is_finite() && d >= threshold;
        }
    }
    let dsc = dice_similarity_coefficient(&isodose_buf[..], &ptv_mask[..]);
    if dsc.is_finite() {
        dsc.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn ptv_roi_bounds(
    ptv_mask: &[bool; IMRT_VOXEL_COUNT],
    margin: usize,
) -> Option<(usize, usize, usize, usize)> {
    let mut min_x = IMRT_GRID_DIM;
    let mut max_x = 0_usize;
    let mut min_y = IMRT_GRID_DIM;
    let mut max_y = 0_usize;
    let mut any = false;
    for y in 0..IMRT_GRID_DIM {
        for x in 0..IMRT_GRID_DIM {
            if ptv_mask[flatten_2d_index(x, y, IMRT_GRID_DIM)] {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                any = true;
            }
        }
    }
    any.then(|| {
        (
            min_x.saturating_sub(margin),
            (max_x + margin).min(IMRT_GRID_DIM - 1),
            min_y.saturating_sub(margin),
            (max_y + margin).min(IMRT_GRID_DIM - 1),
        )
    })
}

/// Computes the centroid `(x_cm, y_cm)` of active mask indices, falling back to `fallback`.
#[must_use]
pub fn mask_centroid_cm(indices: &[usize], fallback: [f64; 2]) -> [f64; 2] {
    if indices.is_empty() {
        return fallback;
    }
    let mut sx = 0.0;
    let mut sy = 0.0;
    for &idx in indices {
        sx += voxel_coord_cm(idx % IMRT_GRID_DIM);
        sy += voxel_coord_cm(idx / IMRT_GRID_DIM);
    }
    let inv = 1.0 / (indices.len() as f64);
    [sx * inv, sy * inv]
}

/// Extracts perimeter cell-edge line segments in phantom coordinates (shifted by `shift_x_cm`)
/// for crisp overlay rendering of arbitrary sculpted PTV/OAR masks.
#[must_use]
pub fn extract_mask_boundary_segments(
    mask: &[bool; IMRT_VOXEL_COUNT],
    shift_x_cm: f64,
) -> Vec<[[f64; 2]; 2]> {
    let step = IMRT_PHANTOM_SIZE_CM / IMRT_GRID_DIM as f64;
    let half = 0.5 * step;
    let mut segs = Vec::new();
    for y in 0..IMRT_GRID_DIM {
        let cy = voxel_coord_cm(y);
        for x in 0..IMRT_GRID_DIM {
            if !mask[flatten_2d_index(x, y, IMRT_GRID_DIM)] {
                continue;
            }
            let cx = voxel_coord_cm(x) + shift_x_cm;
            let x0 = cx - half;
            let x1 = cx + half;
            let y0 = cy - half;
            let y1 = cy + half;
            if x == 0 || !mask[flatten_2d_index(x - 1, y, IMRT_GRID_DIM)] {
                segs.push([[x0, y0], [x0, y1]]);
            }
            if x + 1 == IMRT_GRID_DIM || !mask[flatten_2d_index(x + 1, y, IMRT_GRID_DIM)] {
                segs.push([[x1, y0], [x1, y1]]);
            }
            if y == 0 || !mask[flatten_2d_index(x, y - 1, IMRT_GRID_DIM)] {
                segs.push([[x0, y0], [x1, y0]]);
            }
            if y + 1 == IMRT_GRID_DIM || !mask[flatten_2d_index(x, y + 1, IMRT_GRID_DIM)] {
                segs.push([[x0, y1], [x1, y1]]);
            }
        }
    }
    segs
}

// [cite:favorite_child]

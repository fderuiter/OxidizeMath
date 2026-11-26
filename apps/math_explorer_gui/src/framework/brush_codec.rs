//! Compact RLE 2D mask and 1D sparse-potential codecs plus integer Bresenham
//! stroke interpolation for [`super::spatial_brush::SpatialBrushKit`].

/// Maximum grid cell count accepted when decoding persisted RLE 2D masks.
pub const MAX_RLE_MASK_CELLS: usize = 262_144;

/// Encodes a row-major `u8` cell mask into `[value, run_length]` pairs.
///
/// Returns an empty vector if all cells are `0`, keeping default tool states compact.
#[must_use]
pub fn encode_rle_mask(cells: &[u8]) -> Vec<[u32; 2]> {
    if cells.is_empty() || cells.iter().all(|&c| c == 0) {
        return Vec::new();
    }
    let mut runs: Vec<[u32; 2]> = Vec::new();
    let mut current = u32::from(cells[0]);
    let mut count: u32 = 1;

    for &byte in &cells[1..] {
        let val = u32::from(byte);
        if val == current && count < u32::MAX {
            count += 1;
        } else {
            runs.push([current, count]);
            current = val;
            count = 1;
        }
    }
    runs.push([current, count]);
    runs
}

/// Decodes `[value, run_length]` RLE pairs into a `u8` mask of `expected_len` cells.
///
/// An empty `runs` slice decodes to `vec![0; expected_len]`.
pub fn decode_rle_mask(runs: &[[u32; 2]], expected_len: usize) -> Result<Vec<u8>, String> {
    if expected_len > MAX_RLE_MASK_CELLS {
        return Err(format!(
            "Requested RLE mask size {expected_len} exceeds maximum {MAX_RLE_MASK_CELLS}"
        ));
    }
    if runs.is_empty() {
        return Ok(vec![0_u8; expected_len]);
    }
    let mut out = Vec::with_capacity(expected_len);
    for &[val, count] in runs {
        if val > u32::from(u8::MAX) {
            return Err(format!("RLE cell value {val} exceeds u8 range"));
        }
        let next_len = out
            .len()
            .checked_add(count as usize)
            .ok_or_else(|| "RLE run length overflow".to_string())?;
        if next_len > expected_len {
            return Err(format!(
                "RLE decoded length {next_len} exceeds expected grid size {expected_len}"
            ));
        }
        out.resize(next_len, val as u8);
    }
    if out.len() != expected_len {
        return Err(format!(
            "RLE decoded length {} does not match expected grid size {expected_len}",
            out.len()
        ));
    }
    Ok(out)
}

/// Encodes differences between `base` and `current` 1D potentials as quantized sparse `(index, value)` tuples.
#[must_use]
pub fn encode_sparse_potential(base: &[f64], current: &[f64], eps: f64) -> Vec<(u16, f64)> {
    let n = base.len().min(current.len()).min(usize::from(u16::MAX));
    let mut out = Vec::new();
    for i in 0..n {
        if (current[i] - base[i]).abs() > eps && current[i].is_finite() {
            let quantized = (current[i] * 10_000.0).round() / 10_000.0;
            out.push((i as u16, quantized));
        }
    }
    out
}

/// Applies sparse `(index, value)` overrides onto `target`, validating indices and finiteness.
pub fn apply_sparse_potential(target: &mut [f64], deltas: &[(u16, f64)]) -> Result<(), String> {
    for &(idx, val) in deltas {
        if !val.is_finite() {
            return Err(format!(
                "Non-finite sparse potential value at index {idx}: {val}"
            ));
        }
        let i = usize::from(idx);
        if i >= target.len() {
            return Err(format!(
                "Sparse potential index {i} out of bounds for grid length {}",
                target.len()
            ));
        }
        target[i] = val;
    }
    Ok(())
}

/// Computes integer grid cells along the segment `(x0, y0) -> (x1, y1)` using Bresenham's algorithm.
///
/// Consecutive points change by at most `1` along each axis, guaranteeing zero skipped grid cells
/// even on rapid diagonal drags.
#[must_use]
pub fn bresenham_line(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let capacity = (dx.max(-dy) as usize).saturating_add(1).min(8192);
    let mut pts = Vec::with_capacity(capacity);
    let mut x = x0;
    let mut y = y0;

    for _ in 0..8192 {
        pts.push((x, y));
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
    pts
}

/// Sweeps a circular disk of `radius` along the Bresenham line `start -> end`
/// and invokes `visit(gx, gy)` for every in-bounds `(0..bounds.0, 0..bounds.1)` cell.
pub fn rasterize_disk_stroke(
    start: (i32, i32),
    end: (i32, i32),
    radius: i32,
    bounds: (usize, usize),
    mut visit: impl FnMut(usize, usize),
) {
    let r = radius.max(0);
    let r_sq = r * r;
    let w = bounds.0 as i32;
    let h = bounds.1 as i32;
    for (cx, cy) in bresenham_line(start.0, start.1, end.0, end.1) {
        for y in (cy - r)..=(cy + r) {
            if y < 0 || y >= h {
                continue;
            }
            for x in (cx - r)..=(cx + r) {
                if x >= 0 && x < w && (x - cx).pow(2) + (y - cy).pow(2) <= r_sq {
                    visit(x as usize, y as usize);
                }
            }
        }
    }
}

/// Interpolates a contiguous 1D inclusive index span `[lo, hi]` covering `i0..=i1` expanded by `half_width`.
#[must_use]
pub fn interpolate_1d_span(
    i0: usize,
    i1: usize,
    half_width: usize,
    len: usize,
) -> Option<(usize, usize)> {
    if len == 0 {
        return None;
    }
    let last = len - 1;
    let lo = i0.min(i1).saturating_sub(half_width).min(last);
    let hi = i0.max(i1).saturating_add(half_width).min(last);
    Some((lo, hi))
}

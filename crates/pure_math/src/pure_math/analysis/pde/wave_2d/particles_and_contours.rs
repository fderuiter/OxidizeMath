//! Fixed-capacity Chladni sand particle tracer and 2D Marching-Squares zero-contour extractor.

// [cite:advanced_linear_algebra]

use math_commons::math_kernel::types::flatten_2d_index;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

/// Default Chladni sand particle capacity (`4,000` grains).
pub const DEFAULT_SAND_PARTICLES: usize = 4_000;

/// Fixed-capacity Chladni sand grain ensemble drifting down the 2D wave-energy gradient.
#[derive(Debug, Clone)]
pub struct ChladniSandSystem {
    capacity: usize,
    pos_x: Vec<f64>,
    pos_y: Vec<f64>,
    vel_x: Vec<f64>,
    vel_y: Vec<f64>,
    rng: OxidizeRng,
}

impl ChladniSandSystem {
    /// Pre-allocates `capacity` sand grains seeded deterministically via [`OxidizeRng`].
    pub fn new(capacity: usize, seed: u64, width: usize, height: usize, wall_mask: &[f64]) -> Self {
        assert!(capacity >= 1, "Particle capacity must be >= 1");
        assert!(width >= 4 && height >= 4, "Grid dimensions must be >= 4");
        let mut sys = Self {
            capacity,
            pos_x: vec![0.0; capacity],
            pos_y: vec![0.0; capacity],
            vel_x: vec![0.0; capacity],
            vel_y: vec![0.0; capacity],
            rng: OxidizeRng::new(seed),
        };
        sys.reseed(seed, width, height, wall_mask);
        sys
    }

    /// Deterministically scatters all grains across active (`wall_mask > 0.5`) plate cells.
    pub fn reseed(&mut self, seed: u64, width: usize, height: usize, wall_mask: &[f64]) {
        assert_eq!(wall_mask.len(), width * height);
        self.rng = OxidizeRng::new(seed);
        let x_span = (width - 3) as f64;
        let y_span = (height - 3) as f64;
        for i in 0..self.capacity {
            let mut px = 1.0 + next_unit_f64(&mut self.rng) * x_span;
            let mut py = 1.0 + next_unit_f64(&mut self.rng) * y_span;
            let mut tries = 0_usize;
            while tries < 16 {
                let gx = (px.round() as usize).min(width - 1);
                let gy = (py.round() as usize).min(height - 1);
                if wall_mask[flatten_2d_index(gx, gy, width)] > 0.5 {
                    break;
                }
                px = 1.0 + next_unit_f64(&mut self.rng) * x_span;
                py = 1.0 + next_unit_f64(&mut self.rng) * y_span;
                tries += 1;
            }
            self.pos_x[i] = px;
            self.pos_y[i] = py;
            self.vel_x[i] = 0.0;
            self.vel_y[i] = 0.0;
        }
    }

    /// Number of tracked sand grains.
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        self.capacity
    }

    /// Returns `true` when zero grains are allocated.
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        self.capacity == 0
    }

    /// Continuous horizontal cell coordinates $x_p \in [1, W - 2]$.
    #[verified_engine::verified]
    pub fn positions_x(&self) -> &[f64] {
        &self.pos_x
    }

    /// Continuous vertical cell coordinates $y_p \in [1, H - 2]$.
    #[verified_engine::verified]
    pub fn positions_y(&self) -> &[f64] {
        &self.pos_y
    }

    /// Drifts all sand grains down $-\nabla E(x, y)$ with amplitude-scaled agitation in zero allocations.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step(
        &mut self,
        energy_field: &[f64],
        wall_mask: &[f64],
        width: usize,
        height: usize,
        mobility: f64,
        agitation: f64,
        dt: f64,
    ) {
        assert_eq!(energy_field.len(), width * height);
        assert_eq!(wall_mask.len(), width * height);
        let max_e = max_slice_value(energy_field);
        if max_e <= 1.0e-18 || dt <= 0.0 {
            return;
        }
        let inv_max_e = 1.0 / max_e;
        let max_x = (width - 2) as f64;
        let max_y = (height - 2) as f64;

        for i in 0..self.capacity {
            let px = self.pos_x[i].clamp(1.0, max_x - 1.0e-4);
            let py = self.pos_y[i].clamp(1.0, max_y - 1.0e-4);
            let (e_norm, gx, gy) =
                sample_normalized_energy_grad(energy_field, width, px, py, inv_max_e);
            let amp = e_norm.sqrt();
            let jx = (next_unit_f64(&mut self.rng) * 2.0 - 1.0) * agitation * amp;
            let jy = (next_unit_f64(&mut self.rng) * 2.0 - 1.0) * agitation * amp;
            let vx = 0.65 * self.vel_x[i] - mobility * gx + jx;
            let vy = 0.65 * self.vel_y[i] - mobility * gy + jy;
            let cand_x = (px + vx * dt).clamp(1.0, max_x - 1.0e-4);
            let cand_y = (py + vy * dt).clamp(1.0, max_y - 1.0e-4);
            let cx = (cand_x.round() as usize).min(width - 1);
            let cy = (cand_y.round() as usize).min(height - 1);
            if wall_mask[flatten_2d_index(cx, cy, width)] > 0.5 {
                self.pos_x[i] = cand_x;
                self.pos_y[i] = cand_y;
                self.vel_x[i] = vx;
                self.vel_y[i] = vy;
            } else {
                self.vel_x[i] = -0.3 * vx;
                self.vel_y[i] = -0.3 * vy;
            }
        }
    }

    /// Computes mean normalized wave energy sampled at the current sand grain positions.
    #[verified_engine::verified]
    pub fn mean_normalized_energy(&self, energy_field: &[f64], width: usize) -> f64 {
        let max_e = max_slice_value(energy_field);
        if max_e <= 1.0e-18 || self.capacity == 0 {
            return 0.0;
        }
        let inv_max = 1.0 / max_e;
        let mut sum = 0.0;
        for i in 0..self.capacity {
            let (e_norm, _, _) = sample_normalized_energy_grad(
                energy_field,
                width,
                self.pos_x[i],
                self.pos_y[i],
                inv_max,
            );
            sum += e_norm;
        }
        sum / (self.capacity as f64)
    }
}

#[verified_engine::verified]
fn next_unit_f64(rng: &mut OxidizeRng) -> f64 {
    let bits = rng.next_u64() >> 11;
    (bits as f64) * (1.0 / ((1_u64 << 53) as f64))
}

#[verified_engine::verified]
fn max_slice_value(field: &[f64]) -> f64 {
    let mut m = 0.0_f64;
    for &v in field {
        if v.is_finite() && v > m {
            m = v;
        }
    }
    m
}

#[verified_engine::verified]
fn sample_normalized_energy_grad(
    energy: &[f64],
    width: usize,
    px: f64,
    py: f64,
    inv_max_e: f64,
) -> (f64, f64, f64) {
    let ix = px.floor() as usize;
    let iy = py.floor() as usize;
    let tx = px - (ix as f64);
    let ty = py - (iy as f64);
    let e00 = energy[flatten_2d_index(ix, iy, width)] * inv_max_e;
    let e10 = energy[flatten_2d_index(ix + 1, iy, width)] * inv_max_e;
    let e01 = energy[flatten_2d_index(ix, iy + 1, width)] * inv_max_e;
    let e11 = energy[flatten_2d_index(ix + 1, iy + 1, width)] * inv_max_e;

    let val = (1.0 - ty) * ((1.0 - tx) * e00 + tx * e10) + ty * ((1.0 - tx) * e01 + tx * e11);
    let gx = (1.0 - ty) * (e10 - e00) + ty * (e11 - e01);
    let gy = (1.0 - tx) * (e01 - e00) + tx * (e11 - e10);
    (val.max(0.0), gx, gy)
}

/// Pre-allocated 2D Marching-Squares zero-level nodal contour extractor.
#[derive(Debug, Clone)]
pub struct MarchingSquares2D {
    max_segments: usize,
    segments: Vec<[[f64; 2]; 2]>,
}

impl MarchingSquares2D {
    /// Pre-allocates storage for up to `max_segments` contour line segments.
    #[verified_engine::verified]
    pub fn new(max_segments: usize) -> Self {
        let cap = max_segments.max(64);
        Self {
            max_segments: cap,
            segments: Vec::with_capacity(cap),
        }
    }

    /// Extracted line segments `[[x0, y0], [x1, y1]]` in grid-cell coordinates.
    #[verified_engine::verified]
    pub fn segments(&self) -> &[[[f64; 2]; 2]] {
        &self.segments
    }

    /// Extracts `field == 0.0` isoline segments across active cells without dynamic allocation.
    #[verified_engine::verified]
    pub fn extract_zero_contours(
        &mut self,
        field: &[f64],
        wall_mask: &[f64],
        width: usize,
        height: usize,
        amplitude_floor: f64,
    ) -> &[[[f64; 2]; 2]] {
        assert_eq!(field.len(), width * height);
        assert_eq!(wall_mask.len(), width * height);
        self.segments.clear();
        for y in 0..(height.saturating_sub(1)) {
            for x in 0..(width.saturating_sub(1)) {
                if self.segments.len() + 2 > self.max_segments {
                    return &self.segments;
                }
                self.process_cell(field, wall_mask, width, x, y, amplitude_floor);
            }
        }
        &self.segments
    }

    #[verified_engine::verified]
    fn process_cell(
        &mut self,
        field: &[f64],
        wall_mask: &[f64],
        width: usize,
        x: usize,
        y: usize,
        floor: f64,
    ) {
        let i00 = flatten_2d_index(x, y, width);
        let i10 = flatten_2d_index(x + 1, y, width);
        let i11 = flatten_2d_index(x + 1, y + 1, width);
        let i01 = flatten_2d_index(x, y + 1, width);
        if wall_mask[i00] <= 0.5
            || wall_mask[i10] <= 0.5
            || wall_mask[i11] <= 0.5
            || wall_mask[i01] <= 0.5
        {
            return;
        }
        let v00 = field[i00];
        let v10 = field[i10];
        let v11 = field[i11];
        let v01 = field[i01];
        let peak = v00.abs().max(v10.abs()).max(v11.abs()).max(v01.abs());
        if peak <= floor {
            return;
        }
        let mask = ((v00 >= 0.0) as u8)
            | (((v10 >= 0.0) as u8) << 1)
            | (((v11 >= 0.0) as u8) << 2)
            | (((v01 >= 0.0) as u8) << 3);
        if mask == 0 || mask == 15 {
            return;
        }
        let xf = x as f64;
        let yf = y as f64;
        let e_bottom = [xf + interp_zero(v00, v10), yf];
        let e_right = [xf + 1.0, yf + interp_zero(v10, v11)];
        let e_top = [xf + interp_zero(v01, v11), yf + 1.0];
        let e_left = [xf, yf + interp_zero(v00, v01)];
        self.push_case_segments(mask, e_bottom, e_right, e_top, e_left);
    }

    #[verified_engine::verified]
    fn push_case_segments(&mut self, mask: u8, b: [f64; 2], r: [f64; 2], t: [f64; 2], l: [f64; 2]) {
        match mask {
            1 | 14 => self.segments.push([l, b]),
            2 | 13 => self.segments.push([b, r]),
            3 | 12 => self.segments.push([l, r]),
            4 | 11 => self.segments.push([r, t]),
            5 => {
                self.segments.push([l, t]);
                self.segments.push([b, r]);
            }
            6 | 9 => self.segments.push([b, t]),
            7 | 8 => self.segments.push([l, t]),
            10 => {
                self.segments.push([l, b]);
                self.segments.push([r, t]);
            }
            _ => {}
        }
    }
}

#[verified_engine::verified]
fn interp_zero(a: f64, b: f64) -> f64 {
    let denom = a - b;
    if denom.abs() < 1.0e-15 {
        0.5
    } else {
        (a / denom).clamp(0.0, 1.0)
    }
}

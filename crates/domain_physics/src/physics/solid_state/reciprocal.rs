//! Reciprocal-lattice geometry, Brillouin zone polygons, and zero-allocation kinematic Bragg diffraction.
//!
//! Evaluates reciprocal basis vectors $\mathbf{b}_i$, Wigner-Seitz first Brillouin zones,
//! crystallographic $(h,k,l)$ geometric structure factors, and preallocated 2D structure-factor
//! intensity maps $S(\mathbf{q}) = \frac{1}{N} \left|\sum_{j=1}^N e^{-i \mathbf{q}\cdot\mathbf{r}_j}\right|^2$.

// [cite:solid_state]

use super::lattice::UnitCell;
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::Vector3;
use std::f64::consts::{PI, TAU};

/// Maximum 2D reciprocal grid side length ($G \times G$) for interactive and verified evaluations.
pub const MAX_BRAGG_GRID: usize = 192;

/// Computes the three 3D reciprocal lattice basis vectors $[\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3]$
/// satisfying $\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi \delta_{ij}$.
///
/// Returns `None` if the direct unit-cell triple product volume vanishes.
#[must_use]
#[verified_engine::verified]
pub fn reciprocal_basis_3d(direct: &[Vector3<f64>; 3]) -> Option<[Vector3<f64>; 3]> {
    let c23 = direct[1].cross(&direct[2]);
    let volume = direct[0].dot(&c23);
    if volume.abs() < math_commons::registry::TOLERANCE_HIGH {
        return None;
    }
    let scale = TAU / volume;
    Some([
        c23 * scale,
        direct[2].cross(&direct[0]) * scale,
        direct[0].cross(&direct[1]) * scale,
    ])
}

/// Computes the two 2D reciprocal lattice basis vectors $[\mathbf{b}_1, \mathbf{b}_2]$
/// satisfying $\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi \delta_{ij}$.
///
/// Returns `None` if the 2D unit-cell area determinant vanishes.
#[must_use]
#[verified_engine::verified]
pub fn reciprocal_basis_2d(direct: &[[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let det = direct[0][0] * direct[1][1] - direct[0][1] * direct[1][0];
    if det.abs() < math_commons::registry::TOLERANCE_HIGH {
        return None;
    }
    let scale = TAU / det;
    Some([
        [direct[1][1] * scale, -direct[1][0] * scale],
        [-direct[0][1] * scale, direct[0][0] * scale],
    ])
}

/// Returns the 1D first Brillouin zone interval $[-\pi/a, +\pi/a]$ for lattice constant `a > 0`.
#[must_use]
#[verified_engine::verified]
pub fn brillouin_zone_1d(a: f64) -> Option<[f64; 2]> {
    if !a.is_finite() || a <= math_commons::registry::TOLERANCE_HIGH {
        return None;
    }
    let k_edge = PI / a;
    Some([-k_edge, k_edge])
}

/// Computes the closed Wigner-Seitz first Brillouin zone polygon in 2D reciprocal space
/// by clipping against Bragg bisector half-planes $\mathbf{q} \cdot \mathbf{G} \le \frac{1}{2}|\mathbf{G}|^2$.
///
/// Writes closed vertices (`out[count - 1] == out[0]`) into `out` and returns total vertex count.
#[verified_engine::verified]
pub fn first_brillouin_zone_2d(
    reciprocal_basis: &[[f64; 2]; 2],
    out: &mut [[f64; 2]; 16],
) -> usize {
    let b1 = reciprocal_basis[0];
    let b2 = reciprocal_basis[1];
    let bound = 2.0 * (b1[0].hypot(b1[1]) + b2[0].hypot(b2[1])).max(1.0);
    let mut poly = [[0.0_f64; 2]; 16];
    poly[0] = [-bound, -bound];
    poly[1] = [bound, -bound];
    poly[2] = [bound, bound];
    poly[3] = [-bound, bound];
    let mut len = 4_usize;

    for h in -2_i32..=2 {
        for k in -2_i32..=2 {
            if h == 0 && k == 0 {
                continue;
            }
            let gx = f64::from(h) * b1[0] + f64::from(k) * b2[0];
            let gy = f64::from(h) * b1[1] + f64::from(k) * b2[1];
            let rhs = 0.5 * (gx * gx + gy * gy);
            len = clip_polygon_halfplane(&poly, len, [gx, gy], rhs, out);
            if len == 0 {
                return 0;
            }
            poly[..len].copy_from_slice(&out[..len]);
        }
    }
    len = dedup_polygon_vertices(&poly, len, out);
    if len > 0 && len < out.len() {
        out[len] = out[0];
        return len + 1;
    }
    len
}

fn dedup_polygon_vertices(poly: &[[f64; 2]; 16], len: usize, out: &mut [[f64; 2]; 16]) -> usize {
    let mut unique = 0_usize;
    for &pt in &poly[..len] {
        let is_dup =
            unique > 0 && (pt[0] - out[unique - 1][0]).hypot(pt[1] - out[unique - 1][1]) < 1e-8;
        let wraps_first = unique > 1 && (pt[0] - out[0][0]).hypot(pt[1] - out[0][1]) < 1e-8;
        if !is_dup && !wraps_first && unique < out.len() - 1 {
            out[unique] = pt;
            unique += 1;
        }
    }
    unique
}

fn clip_polygon_halfplane(
    poly: &[[f64; 2]; 16],
    len: usize,
    normal: [f64; 2],
    rhs: f64,
    out: &mut [[f64; 2]; 16],
) -> usize {
    if len == 0 {
        return 0;
    }
    let mut count = 0_usize;
    for i in 0..len {
        let cur = poly[i];
        let next = poly[(i + 1) % len];
        let d_cur = cur[0] * normal[0] + cur[1] * normal[1] - rhs;
        let d_next = next[0] * normal[0] + next[1] * normal[1] - rhs;
        let in_cur = d_cur <= 1e-11;
        let in_next = d_next <= 1e-11;
        if in_cur && count < out.len() - 1 {
            out[count] = cur;
            count += 1;
        }
        if in_cur != in_next && count < out.len() - 1 {
            let denom = d_cur - d_next;
            if denom.abs() > 1e-15 {
                let t = (d_cur / denom).clamp(0.0, 1.0);
                out[count] = [
                    cur[0] + t * (next[0] - cur[0]),
                    cur[1] + t * (next[1] - cur[1]),
                ];
                count += 1;
            }
        }
    }
    count
}

/// Builds a closed symmetric $n$-gon Jones / pseudo-Brillouin zone boundary from fundamental Bragg radius $q^*$.
#[verified_engine::verified]
pub fn pseudo_brillouin_zone_polygon(
    rotational_order: usize,
    q_fundamental: f64,
    out: &mut [[f64; 2]; 16],
) -> usize {
    let sides = match rotational_order {
        5 => 10,
        n if (4..=12).contains(&n) => n,
        _ => 4,
    };
    let apothem = 0.5 * q_fundamental.max(0.1);
    let circum = apothem / (PI / (sides as f64)).cos();
    for (k, slot) in out.iter_mut().enumerate().take(sides) {
        let theta = (PI / (sides as f64)) + TAU * (k as f64) / (sides as f64);
        let (s, c) = theta.sin_cos();
        *slot = [circum * c, circum * s];
    }
    out[sides] = out[0];
    sides + 1
}

/// Evaluates the crystallographic geometric structure factor intensity $|F_{hkl}|^2$
/// at integer Miller indices $(h, k, l)$ on a 3D [`UnitCell`].
///
/// Reproduces exact selection / extinction rules:
/// - Simple Cubic: $|F_{hkl}|^2 = 1$ for all $(h,k,l)$.
/// - BCC: $|F_{hkl}|^2 = 4$ when $h + k + l$ is even, and $0$ (extinct) when odd.
/// - FCC: $|F_{hkl}|^2 = 16$ when $(h,k,l)$ have unmixed parity, and $0$ (extinct) otherwise.
#[must_use]
#[verified_engine::verified]
pub fn unit_cell_structure_factor(cell: &UnitCell, hkl: [i32; 3]) -> f64 {
    let Some(recip) = reciprocal_basis_3d(&cell.lattice_vectors) else {
        return 0.0;
    };
    let g =
        recip[0] * f64::from(hkl[0]) + recip[1] * f64::from(hkl[1]) + recip[2] * f64::from(hkl[2]);
    let (mut sum_re, mut sum_im) = (0.0_f64, 0.0_f64);
    for pos in &cell.atomic_positions {
        let phase = -g.dot(pos);
        let (s, c) = phase.sin_cos();
        sum_re += c;
        sum_im += s;
    }
    sum_re * sum_re + sum_im * sum_im
}

/// Evaluates the kinematic structure factor $S(\mathbf{q}) = \frac{1}{N}\left|\sum_{j=1}^N e^{-i\mathbf{q}\cdot\mathbf{r}_j}\right|^2$
/// at a single 2D wavevector $\mathbf{q} = (q_x, q_y)$ with zero dynamic heap allocations.
#[must_use]
#[verified_engine::verified]
pub fn structure_factor_2d(points: &[[f64; 2]], q: [f64; 2]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let (re, im) = accumulate_structure_amplitude_2d(points, q);
    (re * re + im * im) / (points.len() as f64)
}

fn accumulate_structure_amplitude_2d(points: &[[f64; 2]], q: [f64; 2]) -> (f64, f64) {
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for p in points {
        let phase = -(q[0] * p[0] + q[1] * p[1]);
        let (s, c) = phase.sin_cos();
        re += c;
        im += s;
    }
    (re, im)
}

/// Evaluates the 3D kinematic structure factor $S(\mathbf{q}) = \frac{1}{N}\left|\sum_{j=1}^N e^{-i\mathbf{q}\cdot\mathbf{r}_j}\right|^2$
/// with zero dynamic heap allocations.
#[must_use]
#[verified_engine::verified]
pub fn structure_factor_3d(points: &[Vector3<f64>], q: Vector3<f64>) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let (mut re, mut im) = (0.0_f64, 0.0_f64);
    for p in points {
        let phase = -q.dot(p);
        let (s, c) = phase.sin_cos();
        re += c;
        im += s;
    }
    (re * re + im * im) / (points.len() as f64)
}

/// Preallocated zero-allocation solver workspace for 2D kinematic Bragg diffraction maps:
/// $$S(q_x, q_y) = \frac{1}{N}\left|\sum_{j=1}^N e^{-i(q_x x_j + q_y y_j)}\right|^2$$
///
/// Reference: Shechtman et al. (1984), DOI: `10.1103/PhysRevLett.53.1951`,
/// [`shechtman1984metallic`](../../../../../papers/references.bib#shechtman1984metallic);
/// Levine & Steinhardt (1984), DOI: `10.1103/PhysRevLett.53.2477`,
/// [`levine1984quasicrystals`](../../../../../papers/references.bib#levine1984quasicrystals);
/// see [`docs/math_specs/domain_physics.md`](../../../../../docs/math_specs/domain_physics.md).
///
/// # Stability
/// Separable 1D complex phasor recurrences evaluate $e^{-i q_x x_j}$ and $e^{-i q_y y_j}$ across
/// bounded grid lengths $G \le 192$ with $O(G \varepsilon_{\text{mach}})$ phase drift and `0` heap allocations.
///
/// # Domain of Validity
/// Finite 2D atomic coordinates, reciprocal half-widths $q_{\max} \ge 0.5$, and square grid sizes
/// $8 \le G \le 192$ (`MAX_BRAGG_GRID`).
#[derive(Debug, Clone)]
pub struct BraggWorkspace {
    max_grid: usize,
    grid_size: usize,
    q_max: f64,
    amp_re: Vec<f64>,
    amp_im: Vec<f64>,
    intensities: Vec<f64>,
    phase_x_re: Vec<f64>,
    phase_x_im: Vec<f64>,
    peak_intensity: f64,
}

impl Default for BraggWorkspace {
    fn default() -> Self {
        Self::new(MAX_BRAGG_GRID)
    }
}

impl BraggWorkspace {
    /// Creates a preallocated workspace supporting square reciprocal grids up to `max_grid` ($\le 192$).
    #[must_use]
    pub fn new(max_grid: usize) -> Self {
        let cap = max_grid.clamp(8, MAX_BRAGG_GRID);
        let total = cap * cap;
        Self {
            max_grid: cap,
            grid_size: 0,
            q_max: 1.0,
            amp_re: vec![0.0; total],
            amp_im: vec![0.0; total],
            intensities: vec![0.0; total],
            phase_x_re: vec![0.0; cap],
            phase_x_im: vec![0.0; cap],
            peak_intensity: 0.0,
        }
    }

    /// Grid side dimension $G$ of the most recently evaluated reciprocal map.
    #[must_use]
    #[verified_engine::verified]
    pub fn grid_size(&self) -> usize {
        self.grid_size
    }

    /// Half-width $q_{\max}$ of the evaluated reciprocal domain $[-q_{\max}, +q_{\max}]^2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn q_max(&self) -> f64 {
        self.q_max
    }

    /// Maximum finite $S(\mathbf{q})$ intensity (excluding the forward $\mathbf{q} \approx \mathbf{0}$ beam when `suppress_central_peak` was set).
    #[must_use]
    #[verified_engine::verified]
    pub fn peak_intensity(&self) -> f64 {
        self.peak_intensity
    }

    /// Row-major slice of length `grid_size * grid_size` containing $S(q_x, q_y)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn intensities(&self) -> &[f64] {
        let total = self.grid_size * self.grid_size;
        &self.intensities[..total]
    }

    /// Evaluates $S(q_x, q_y)$ across $[-q_{\max}, +q_{\max}]^2$ on a `grid_size x grid_size` grid
    /// with zero dynamic heap allocations inside `VerifiedEngine::run_verified`.
    #[verified_engine::verified]
    pub fn evaluate_grid_verified(
        &mut self,
        points: &[[f64; 2]],
        grid_size: usize,
        q_max: f64,
        suppress_central_peak: bool,
    ) {
        let g = grid_size.clamp(8, self.max_grid);
        let q_lim = q_max.abs().max(0.5);
        self.grid_size = g;
        self.q_max = q_lim;
        evaluate_separable_bragg_grid(self, points, g, q_lim, suppress_central_peak);
    }
}

fn evaluate_separable_bragg_grid(
    ws: &mut BraggWorkspace,
    points: &[[f64; 2]],
    g: usize,
    q_lim: f64,
    suppress_central_peak: bool,
) {
    let total = g * g;
    ws.amp_re[..total].fill(0.0);
    ws.amp_im[..total].fill(0.0);
    ws.intensities[..total].fill(0.0);
    ws.peak_intensity = 0.0;
    if points.is_empty() {
        return;
    }

    let dq = (2.0 * q_lim) / ((g - 1) as f64);
    let q_min = -q_lim;
    for &point in points {
        accumulate_separable_atom(ws, point, g, q_min, dq);
    }
    finalize_bragg_intensities(ws, points.len(), g, q_min, dq, suppress_central_peak);
}

fn accumulate_separable_atom(
    ws: &mut BraggWorkspace,
    point: [f64; 2],
    g: usize,
    q_min: f64,
    dq: f64,
) {
    let [rx, ry] = point;
    let (s0x, c0x) = (-q_min * rx).sin_cos();
    let (dsx, dcx) = (-dq * rx).sin_cos();
    let (mut px_re, mut px_im) = (c0x, s0x);
    for col in 0..g {
        ws.phase_x_re[col] = px_re;
        ws.phase_x_im[col] = px_im;
        let next_re = px_re * dcx - px_im * dsx;
        let next_im = px_re * dsx + px_im * dcx;
        px_re = next_re;
        px_im = next_im;
    }

    let (s0y, c0y) = (-q_min * ry).sin_cos();
    let (dsy, dcy) = (-dq * ry).sin_cos();
    let (mut py_re, mut py_im) = (c0y, s0y);
    for row in 0..g {
        for col in 0..g {
            let idx = flatten_2d_index(col, row, g);
            let x_re = ws.phase_x_re[col];
            let x_im = ws.phase_x_im[col];
            ws.amp_re[idx] += py_re * x_re - py_im * x_im;
            ws.amp_im[idx] += py_re * x_im + py_im * x_re;
        }
        let next_re = py_re * dcy - py_im * dsy;
        let next_im = py_re * dsy + py_im * dcy;
        py_re = next_re;
        py_im = next_im;
    }
}

fn finalize_bragg_intensities(
    ws: &mut BraggWorkspace,
    num_points: usize,
    g: usize,
    q_min: f64,
    dq: f64,
    suppress_central_peak: bool,
) {
    let inv_n = 1.0 / (num_points as f64);
    let core_radius_sq = (1.6 * dq).powi(2);
    let mut peak = 0.0_f64;
    for row in 0..g {
        let qy = q_min + (row as f64) * dq;
        for col in 0..g {
            let qx = q_min + (col as f64) * dq;
            let idx = flatten_2d_index(col, row, g);
            if suppress_central_peak && (qx * qx + qy * qy) <= core_radius_sq {
                ws.intensities[idx] = 0.0;
                continue;
            }
            let s_val = (ws.amp_re[idx].powi(2) + ws.amp_im[idx].powi(2)) * inv_n;
            ws.intensities[idx] = s_val;
            if s_val > peak {
                peak = s_val;
            }
        }
    }
    ws.peak_intensity = peak;
}

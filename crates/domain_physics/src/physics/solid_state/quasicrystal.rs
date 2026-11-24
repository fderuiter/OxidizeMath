//! Higher-dimensional cut-and-project quasicrystal generators and perpendicular-space phason dynamics.
//!
//! Generates 1D Fibonacci ($\mathbb{Z}^2$), 2D 5-fold and 10-fold Penrose ($\mathbb{Z}^5$),
//! 2D 8-fold Ammann-Beenker ($\mathbb{Z}^4$), and 2D 12-fold Stampfli ($\mathbb{Z}[\zeta_{12}] \cong \mathbb{Z}^4$)
//! point clouds with continuous phason shifts and discrete vertex flip detection.

// [cite:solid_state]

use std::f64::consts::{FRAC_1_SQRT_2, GOLDEN_RATIO as STD_GOLDEN_RATIO, PI, TAU};

/// Golden ratio $\tau = (1 + \sqrt{5}) / 2$.
pub const GOLDEN_RATIO: f64 = STD_GOLDEN_RATIO;
/// Maximum physical patch radius (in lattice units) to bound higher-D lattice scans on `wasm32`.
pub const MAX_PATCH_RADIUS: f64 = 6.0;
/// Maximum preallocated site capacity for [`QuasicrystalWorkspace`].
pub const MAX_QUASICRYSTAL_SITES: usize = math_commons::registry::MAX_GRID_SIZE;
/// Maximum preallocated nearest-neighbor tile edge capacity for [`QuasicrystalWorkspace`].
pub const MAX_QUASICRYSTAL_EDGES: usize = 3 * math_commons::registry::MAX_GRID_SIZE;

/// Forbidden and aperiodic rotational symmetry classes generated via cut-and-project windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QuasicrystalSymmetry {
    /// 1D Fibonacci chain projected from $\mathbb{Z}^2$ at golden-ratio slope $\tau^{-1}$.
    Fibonacci1D,
    /// 2D 5-fold pentagonal Penrose projection from $\mathbb{Z}^5$ on hyperplane $\sum n_j = 1$.
    Penrose5Fold,
    /// 2D 8-fold octagonal Ammann-Beenker projection from $\mathbb{Z}^4$.
    AmmannBeenker8Fold,
    /// 2D 10-fold decagonal Penrose projection from $\mathbb{Z}^5$ on hyperplane $\sum n_j = 0$.
    Penrose10Fold,
    /// 2D 12-fold dodecagonal Stampfli projection from cyclotomic $\mathbb{Z}[\zeta_{12}] \cong \mathbb{Z}^4$.
    Stampfli12Fold,
}

impl QuasicrystalSymmetry {
    /// Rotational symmetry order $n$ of the kinematic diffraction pattern ($2$ for 1D Fibonacci).
    #[must_use]
    #[verified_engine::verified]
    pub const fn rotational_order(self) -> usize {
        match self {
            Self::Fibonacci1D => 2,
            Self::Penrose5Fold => 5,
            Self::AmmannBeenker8Fold => 8,
            Self::Penrose10Fold => 10,
            Self::Stampfli12Fold => 12,
        }
    }

    /// Characteristic nearest-neighbor tile edge length in physical parallel space.
    #[must_use]
    #[verified_engine::verified]
    pub fn characteristic_bond_length(self) -> f64 {
        match self {
            Self::Fibonacci1D => 1.0,
            Self::Penrose5Fold | Self::Penrose10Fold => (2.0_f64 / 5.0_f64).sqrt(),
            Self::AmmannBeenker8Fold | Self::Stampfli12Fold => FRAC_1_SQRT_2,
        }
    }

    /// Characteristic fundamental reciprocal wavevector magnitude $q^*$ of intense primary Bragg rings.
    #[must_use]
    #[verified_engine::verified]
    pub fn fundamental_wavevector(self) -> f64 {
        match self {
            Self::Fibonacci1D => {
                let norm = (1.0 + GOLDEN_RATIO * GOLDEN_RATIO).sqrt();
                TAU * (GOLDEN_RATIO + 1.0) / norm
            }
            Self::Penrose5Fold | Self::Penrose10Fold => {
                TAU * (2.0_f64 / 5.0_f64).sqrt() * GOLDEN_RATIO
            }
            Self::AmmannBeenker8Fold => TAU * (1.0 + FRAC_1_SQRT_2),
            Self::Stampfli12Fold => TAU * (1.0 + 3.0_f64.sqrt()) * FRAC_1_SQRT_2 * 0.5,
        }
    }
}

/// Single projected quasicrystal vertex carrying parallel, perpendicular, and phason-flip state.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ProjectedSite {
    /// Physical parallel-space coordinates $(x_\parallel, y_\parallel)$.
    pub pos_parallel: [f64; 2],
    /// Internal perpendicular-space coordinates $(x_\perp, y_\perp)$.
    pub pos_perp: [f64; 2],
    /// True when this site entered the acceptance window relative to `reference_phason`.
    pub flipped: bool,
}

/// Bounded parameters controlling cut-and-project patch generation and phason window shifts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuasicrystalPatchConfig {
    /// Target quasicrystal symmetry and higher-dimensional embedding.
    pub symmetry: QuasicrystalSymmetry,
    /// Circular aperture radius $R_\parallel \in [1.0, 6.0]$ in parallel space.
    pub patch_radius: f64,
    /// Acceptance-window scale factor in $[0.5, 1.4]$.
    pub window_scale: f64,
    /// Active perpendicular-space phason shift $\boldsymbol{\phi}_\perp = (\phi_x, \phi_y)$.
    pub phason_shift: [f64; 2],
    /// Reference perpendicular-space phason shift used to highlight discrete tile flips.
    pub reference_phason: [f64; 2],
}

impl Default for QuasicrystalPatchConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            symmetry: QuasicrystalSymmetry::Penrose5Fold,
            patch_radius: 4.0,
            window_scale: 1.0,
            phason_shift: [0.0, 0.0],
            reference_phason: [0.0, 0.0],
        }
    }
}

impl QuasicrystalPatchConfig {
    /// Returns a copy with all continuous parameters clamped to safe responsive bounds.
    #[must_use]
    #[verified_engine::verified]
    pub fn sanitized(self) -> Self {
        Self {
            symmetry: self.symmetry,
            patch_radius: self.patch_radius.clamp(1.0, MAX_PATCH_RADIUS),
            window_scale: self.window_scale.clamp(0.5, 1.4),
            phason_shift: [
                self.phason_shift[0].clamp(-2.0, 2.0),
                self.phason_shift[1].clamp(-2.0, 2.0),
            ],
            reference_phason: [
                self.reference_phason[0].clamp(-2.0, 2.0),
                self.reference_phason[1].clamp(-2.0, 2.0),
            ],
        }
    }
}

/// Preallocated zero-allocation workspace for cut-and-project quasicrystal generation.
#[derive(Debug, Clone)]
pub struct QuasicrystalWorkspace {
    sites: Vec<ProjectedSite>,
    parallel_coords: Vec<[f64; 2]>,
    edges: Vec<([f64; 2], [f64; 2])>,
    site_count: usize,
    edge_count: usize,
    /// Number of vertices accepted under `phason_shift` that were rejected at `reference_phason`.
    pub flipped_in_count: usize,
    /// Number of vertices rejected under `phason_shift` that were accepted at `reference_phason`.
    pub flipped_out_count: usize,
}

impl Default for QuasicrystalWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl QuasicrystalWorkspace {
    /// Allocates fixed-capacity buffers up to [`MAX_QUASICRYSTAL_SITES`] and [`MAX_QUASICRYSTAL_EDGES`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            sites: vec![ProjectedSite::default(); MAX_QUASICRYSTAL_SITES],
            parallel_coords: vec![[0.0; 2]; MAX_QUASICRYSTAL_SITES],
            edges: vec![([0.0; 2], [0.0; 2]); MAX_QUASICRYSTAL_EDGES],
            site_count: 0,
            edge_count: 0,
            flipped_in_count: 0,
            flipped_out_count: 0,
        }
    }

    /// Active slice of projected quasicrystal vertices from the most recent generation.
    #[must_use]
    #[verified_engine::verified]
    pub fn sites(&self) -> &[ProjectedSite] {
        &self.sites[..self.site_count]
    }

    /// Active slice of 2D parallel-space coordinates suitable for [`super::reciprocal::BraggWorkspace`].
    #[must_use]
    #[verified_engine::verified]
    pub fn parallel_points(&self) -> &[[f64; 2]] {
        &self.parallel_coords[..self.site_count]
    }

    /// Active slice of nearest-neighbor rhomb/tile bond segments in parallel space.
    #[must_use]
    #[verified_engine::verified]
    pub fn edges(&self) -> &[([f64; 2], [f64; 2])] {
        &self.edges[..self.edge_count]
    }

    /// Populates projected sites and phason-flip counts with zero dynamic heap allocations.
    #[verified_engine::verified]
    pub fn generate_into(&mut self, config: &QuasicrystalPatchConfig) {
        let cfg = config.sanitized();
        self.site_count = 0;
        self.edge_count = 0;
        self.flipped_in_count = 0;
        self.flipped_out_count = 0;

        match cfg.symmetry {
            QuasicrystalSymmetry::Fibonacci1D => self.generate_fibonacci_1d(&cfg),
            QuasicrystalSymmetry::Penrose5Fold => self.generate_penrose(&cfg, 1, false),
            QuasicrystalSymmetry::Penrose10Fold => self.generate_penrose(&cfg, 0, true),
            QuasicrystalSymmetry::AmmannBeenker8Fold => self.generate_ammann_beenker(&cfg),
            QuasicrystalSymmetry::Stampfli12Fold => self.generate_stampfli(&cfg),
        }
        self.populate_bond_edges(cfg.symmetry);
    }

    fn push_site(&mut self, pos_parallel: [f64; 2], pos_perp: [f64; 2], flipped: bool) {
        if self.site_count < self.sites.len() {
            self.sites[self.site_count] = ProjectedSite {
                pos_parallel,
                pos_perp,
                flipped,
            };
            self.parallel_coords[self.site_count] = pos_parallel;
            self.site_count += 1;
        }
        if flipped {
            self.flipped_in_count += 1;
        }
    }

    fn generate_fibonacci_1d(&mut self, cfg: &QuasicrystalPatchConfig) {
        let norm = (1.0 + GOLDEN_RATIO * GOLDEN_RATIO).sqrt();
        let cos_t = GOLDEN_RATIO / norm;
        let sin_t = 1.0 / norm;
        let half_w = 0.5 * cfg.window_scale * (cos_t + sin_t);
        let max_perp = half_w + cfg.phason_shift[0].abs().max(cfg.reference_phason[0].abs());
        let bound = (cfg.patch_radius.hypot(max_perp).ceil() as i32) + 1;

        for n1 in -bound..=bound {
            for n2 in -bound..=bound {
                let f1 = f64::from(n1);
                let f2 = f64::from(n2);
                let x_par = f1 * cos_t + f2 * sin_t;
                if x_par.abs() > cfg.patch_radius {
                    continue;
                }
                let x_perp = -f1 * sin_t + f2 * cos_t;
                let in_cur = (x_perp - cfg.phason_shift[0]).abs() <= half_w + 1e-12;
                let in_ref = (x_perp - cfg.reference_phason[0]).abs() <= half_w + 1e-12;
                if in_cur {
                    self.push_site([x_par, 0.0], [x_perp, 0.0], !in_ref);
                } else if in_ref {
                    self.flipped_out_count += 1;
                }
            }
        }
    }

    fn generate_penrose(
        &mut self,
        cfg: &QuasicrystalPatchConfig,
        sum_target: i32,
        symmetric: bool,
    ) {
        let scale = (2.0_f64 / 5.0_f64).sqrt();
        let mut e_par = [[0.0_f64; 2]; 5];
        let mut e_perp = [[0.0_f64; 2]; 5];
        let mut normals = [[0.0_f64; 2]; 5];
        for j in 0..5 {
            let angle = TAU * (j as f64) / 5.0;
            let (s1, c1) = angle.sin_cos();
            let (s2, c2) = (2.0 * angle).sin_cos();
            e_par[j] = [scale * c1, scale * s1];
            e_perp[j] = [scale * c2, scale * s2];
            normals[j] = [c1, s1];
        }
        let r_w = 0.85 * cfg.window_scale;
        let circum = r_w / (PI / 5.0).cos();
        let bound = ((cfg.patch_radius.powi(2) + (circum + phason_pad(cfg)).powi(2) + 0.25)
            .sqrt()
            .ceil() as i32)
            + 1;
        let r_sq = cfg.patch_radius * cfg.patch_radius + 1e-12;

        for n0 in -bound..=bound {
            for n1 in -bound..=bound {
                for n2 in -bound..=bound {
                    for n3 in -bound..=bound {
                        let n4 = sum_target - n0 - n1 - n2 - n3;
                        if n4.abs() > bound {
                            continue;
                        }
                        let (par, perp) = project_coords(&[n0, n1, n2, n3, n4], &e_par, &e_perp);
                        if par[0] * par[0] + par[1] * par[1] <= r_sq {
                            self.accept_site(par, perp, cfg, &normals, r_w, symmetric);
                        }
                    }
                }
            }
        }
    }

    fn generate_ammann_beenker(&mut self, cfg: &QuasicrystalPatchConfig) {
        let mut e_par = [[0.0_f64; 2]; 4];
        let mut e_perp = [[0.0_f64; 2]; 4];
        let mut normals = [[0.0_f64; 2]; 4];
        for j in 0..4 {
            let a1 = PI * (j as f64) / 4.0;
            let (s1, c1) = a1.sin_cos();
            let (s3, c3) = (3.0 * a1).sin_cos();
            e_par[j] = [FRAC_1_SQRT_2 * c1, FRAC_1_SQRT_2 * s1];
            e_perp[j] = [FRAC_1_SQRT_2 * c3, FRAC_1_SQRT_2 * s3];
            normals[j] = [c1, s1];
        }
        let r_w = 0.5 * (1.0 + FRAC_1_SQRT_2 * 2.0_f64.sqrt()) * FRAC_1_SQRT_2 * cfg.window_scale;
        let circum = r_w / (PI / 8.0).cos();
        let bound = (cfg.patch_radius.hypot(circum + phason_pad(cfg)).ceil() as i32) + 1;
        self.scan_4d_lattice(cfg, bound, &e_par, &e_perp, &normals, r_w);
    }

    fn generate_stampfli(&mut self, cfg: &QuasicrystalPatchConfig) {
        let mut e_par = [[0.0_f64; 2]; 4];
        let mut e_perp = [[0.0_f64; 2]; 4];
        for j in 0..4 {
            let a1 = PI * (j as f64) / 6.0;
            let (s1, c1) = a1.sin_cos();
            let (s5, c5) = (5.0 * a1).sin_cos();
            e_par[j] = [FRAC_1_SQRT_2 * c1, FRAC_1_SQRT_2 * s1];
            e_perp[j] = [FRAC_1_SQRT_2 * c5, FRAC_1_SQRT_2 * s5];
        }
        let mut normals = [[0.0_f64; 2]; 6];
        for (k, normal) in normals.iter_mut().enumerate() {
            let (s, c) = (PI * (k as f64) / 6.0).sin_cos();
            *normal = [c, s];
        }
        let r_w = 0.85 * cfg.window_scale;
        let circum = r_w / (PI / 12.0).cos();
        let bound = ((2.0 * (cfg.patch_radius.powi(2) + (circum + phason_pad(cfg)).powi(2)))
            .sqrt()
            .ceil() as i32)
            + 1;
        self.scan_4d_lattice(cfg, bound, &e_par, &e_perp, &normals, r_w);
    }

    fn scan_4d_lattice(
        &mut self,
        cfg: &QuasicrystalPatchConfig,
        bound: i32,
        e_par: &[[f64; 2]; 4],
        e_perp: &[[f64; 2]; 4],
        normals: &[[f64; 2]],
        r_w: f64,
    ) {
        let r_sq = cfg.patch_radius * cfg.patch_radius + 1e-12;
        for n0 in -bound..=bound {
            for n1 in -bound..=bound {
                for n2 in -bound..=bound {
                    for n3 in -bound..=bound {
                        let (par, perp) = project_coords(&[n0, n1, n2, n3], e_par, e_perp);
                        if par[0] * par[0] + par[1] * par[1] <= r_sq {
                            self.accept_site(par, perp, cfg, normals, r_w, true);
                        }
                    }
                }
            }
        }
    }

    fn accept_site(
        &mut self,
        par: [f64; 2],
        perp: [f64; 2],
        cfg: &QuasicrystalPatchConfig,
        normals: &[[f64; 2]],
        r_w: f64,
        symmetric: bool,
    ) {
        let in_cur = in_poly_window(perp, cfg.phason_shift, normals, r_w, symmetric);
        let in_ref = in_poly_window(perp, cfg.reference_phason, normals, r_w, symmetric);
        if in_cur {
            self.push_site(par, perp, !in_ref);
        } else if in_ref {
            self.flipped_out_count += 1;
        }
    }

    fn populate_bond_edges(&mut self, symmetry: QuasicrystalSymmetry) {
        if symmetry == QuasicrystalSymmetry::Fibonacci1D || self.site_count < 2 {
            return;
        }
        let target = symmetry.characteristic_bond_length();
        let min_sq = (target * 0.92).powi(2);
        let max_sq = (target * 1.08).powi(2);
        for i in 0..self.site_count {
            let a = self.parallel_coords[i];
            for j in (i + 1)..self.site_count {
                let b = self.parallel_coords[j];
                let dx = a[0] - b[0];
                let dy = a[1] - b[1];
                let d_sq = dx * dx + dy * dy;
                if d_sq >= min_sq && d_sq <= max_sq && self.edge_count < self.edges.len() {
                    self.edges[self.edge_count] = (a, b);
                    self.edge_count += 1;
                }
            }
        }
    }
}

fn project_coords(idx: &[i32], e_par: &[[f64; 2]], e_perp: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    let (mut px, mut py, mut qx, mut qy) = (0.0, 0.0, 0.0, 0.0);
    for ((&n, par), perp) in idx.iter().zip(e_par.iter()).zip(e_perp.iter()) {
        let v = f64::from(n);
        px += v * par[0];
        py += v * par[1];
        qx += v * perp[0];
        qy += v * perp[1];
    }
    ([px, py], [qx, qy])
}

fn phason_pad(cfg: &QuasicrystalPatchConfig) -> f64 {
    cfg.phason_shift[0]
        .hypot(cfg.phason_shift[1])
        .max(cfg.reference_phason[0].hypot(cfg.reference_phason[1]))
}

fn in_poly_window(
    pos_perp: [f64; 2],
    shift: [f64; 2],
    normals: &[[f64; 2]],
    r_w: f64,
    symmetric: bool,
) -> bool {
    let (dx, dy, limit) = (pos_perp[0] - shift[0], pos_perp[1] - shift[1], r_w + 1e-12);
    normals.iter().all(|n| {
        let proj = dx * n[0] + dy * n[1];
        if symmetric {
            proj.abs() <= limit
        } else {
            proj <= limit
        }
    })
}

/// Writes the closed perpendicular-space acceptance polygon vertices into `out` and returns vertex count.
#[verified_engine::verified]
pub fn acceptance_window_polygon(
    symmetry: QuasicrystalSymmetry,
    window_scale: f64,
    phason_shift: [f64; 2],
    out: &mut [[f64; 2]; 14],
) -> usize {
    let scale = window_scale.clamp(0.5, 1.4);
    let (sides, apothem, angle_offset) = match symmetry {
        QuasicrystalSymmetry::Fibonacci1D => {
            let norm = (1.0 + GOLDEN_RATIO * GOLDEN_RATIO).sqrt();
            let hw = 0.5 * scale * ((GOLDEN_RATIO + 1.0) / norm);
            out[0] = [phason_shift[0] - hw, -0.15];
            out[1] = [phason_shift[0] + hw, -0.15];
            out[2] = [phason_shift[0] + hw, 0.15];
            out[3] = [phason_shift[0] - hw, 0.15];
            out[4] = out[0];
            return 5;
        }
        QuasicrystalSymmetry::Penrose5Fold => (5_usize, 0.85 * scale, PI / 5.0),
        QuasicrystalSymmetry::Penrose10Fold => (10_usize, 0.85 * scale, PI / 10.0),
        QuasicrystalSymmetry::AmmannBeenker8Fold => {
            let rw = 0.5 * (1.0 + FRAC_1_SQRT_2 * 2.0_f64.sqrt()) * FRAC_1_SQRT_2 * scale;
            (8_usize, rw, PI / 8.0)
        }
        QuasicrystalSymmetry::Stampfli12Fold => (12_usize, 0.85 * scale, PI / 12.0),
    };
    let circum = apothem / (PI / (sides as f64)).cos();
    for (k, slot) in out.iter_mut().enumerate().take(sides) {
        let theta = angle_offset + TAU * (k as f64) / (sides as f64);
        let (s, c) = theta.sin_cos();
        *slot = [phason_shift[0] + circum * c, phason_shift[1] + circum * s];
    }
    out[sides] = out[0];
    sides + 1
}

/// Convenience helper allocating and returning projected quasicrystal sites for `config`.
#[must_use]
#[verified_engine::verified]
pub fn generate_quasicrystal_patch(config: &QuasicrystalPatchConfig) -> Vec<ProjectedSite> {
    let mut ws = QuasicrystalWorkspace::new();
    ws.generate_into(config);
    ws.sites().to_vec()
}

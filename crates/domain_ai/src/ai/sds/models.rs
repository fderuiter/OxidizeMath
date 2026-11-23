//! Concrete zero-allocation volumetric NeRF and analytical score-distillation priors.
//!
//! Provides [`AnalyticalVoxelNeRF`] (implementing [`NeRFModel`] with stack-only trilinear
//! interpolation and [`DifferentiableNeRF`] with exact volumetric transmittance backpropagation),
//! alongside [`PromptPrototypeEncoder`] and [`AnalyticalScorePrior`] (implementing
//! [`super::score::TextEncoder`] and [`super::score::DiffusionModel`] via closed-form
//! Tweedie Gaussian diffusion score estimation).

use crate::ai::sds::rendering::{NeRFModel, RayBundle};
use crate::ai::sds::training::DifferentiableNeRF;
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::{DMatrix, Vector3};

pub use crate::ai::sds::priors::{AnalyticalScorePrior, PromptPrototypeEncoder};

const MAX_BACKWARD_SAMPLES: usize = 32;

#[inline]
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[inline]
fn voxel_flat_index(ix: usize, iy: usize, iz: usize, res: usize) -> usize {
    let yz = flatten_2d_index(iy, iz, res);
    flatten_2d_index(ix, yz, res)
}

/// Active voxel sample exported for 3D viewport inspection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelPointSample {
    /// World-space voxel center in $[-E, E]^3$.
    pub position: Vector3<f64>,
    /// Evaluated non-negative volume density $\sigma$.
    pub density: f64,
    /// Evaluated RGB albedo in $[0, 1]^3$.
    pub color: Vector3<f64>,
}

/// Differentiable dense trilinear voxel radiance field over $[-E, E]^3$.
///
/// Parameters are stored in a `(grid_res^3, 4)` [`DMatrix<f64>`] where column `0` holds the
/// density logit $\theta_\sigma$ and columns `1..=3` hold RGB color logits $(\theta_r, \theta_g, \theta_b)$.
#[derive(Debug, Clone)]
pub struct AnalyticalVoxelNeRF {
    grid_res: usize,
    half_extent: f64,
    density_scale: f64,
    t_near: f64,
    t_far: f64,
    backward_samples: usize,
    params: DMatrix<f64>,
}

impl AnalyticalVoxelNeRF {
    /// Constructs a smooth low-density initial voxel grid centered at the origin.
    #[verified_engine::verified]
    pub fn new(grid_res: usize, half_extent: f64) -> Self {
        assert!(grid_res >= 2, "grid_res must be at least 2");
        assert!(half_extent > 0.0, "half_extent must be positive");
        let n_voxels = grid_res * grid_res * grid_res;
        let mut params = DMatrix::from_element(n_voxels, 4, -2.2);
        let inv = 2.0 / ((grid_res - 1) as f64);
        for iz in 0..grid_res {
            for iy in 0..grid_res {
                for ix in 0..grid_res {
                    let px = -1.0 + (ix as f64) * inv;
                    let py = -1.0 + (iy as f64) * inv;
                    let pz = -1.0 + (iz as f64) * inv;
                    let r2 = px * px + py * py + pz * pz;
                    let v = voxel_flat_index(ix, iy, iz, grid_res);
                    params[(v, 0)] = -1.6 - 1.8 * r2;
                    params[(v, 1)] = -0.2 * px;
                    params[(v, 2)] = 0.2 * py;
                    params[(v, 3)] = 0.15 - 0.2 * pz;
                }
            }
        }
        Self {
            grid_res,
            half_extent,
            density_scale: 3.0,
            t_near: 1.2,
            t_far: 4.2,
            backward_samples: 16,
            params,
        }
    }

    /// Configures the ray integration interval and sample count used by [`DifferentiableNeRF::backward`].
    #[verified_engine::verified]
    pub fn with_ray_march_bounds(mut self, t_near: f64, t_far: f64, samples: usize) -> Self {
        assert!(t_far > t_near, "t_far must exceed t_near");
        self.t_near = t_near;
        self.t_far = t_far;
        self.backward_samples = samples.clamp(2, MAX_BACKWARD_SAMPLES);
        self
    }

    /// Voxel lattice resolution along each axis.
    pub fn grid_res(&self) -> usize {
        self.grid_res
    }

    /// Half-extent $E$ of the world-space bounding cube $[-E, E]^3$.
    pub fn half_extent(&self) -> f64 {
        self.half_extent
    }

    /// Borrows the `(grid_res^3, 4)` trainable parameter tensor.
    pub fn params(&self) -> &DMatrix<f64> {
        &self.params
    }

    /// Replaces the trainable parameter tensor after validating dimensions.
    #[verified_engine::verified]
    pub fn set_params(&mut self, next_params: DMatrix<f64>) {
        assert_eq!(next_params.shape(), self.params.shape());
        self.params = next_params;
    }

    /// Collects voxels whose density exceeds `min_density` for 3D point-cloud visualization.
    pub fn active_voxels(&self, min_density: f64) -> Vec<VoxelPointSample> {
        let mut out = Vec::new();
        let res = self.grid_res;
        let step = (2.0 * self.half_extent) / ((res - 1) as f64);
        for iz in 0..res {
            for iy in 0..res {
                for ix in 0..res {
                    let v = voxel_flat_index(ix, iy, iz, res);
                    let sigma = softplus(self.params[(v, 0)]) * self.density_scale;
                    if sigma >= min_density {
                        let pos = Vector3::new(
                            -self.half_extent + (ix as f64) * step,
                            -self.half_extent + (iy as f64) * step,
                            -self.half_extent + (iz as f64) * step,
                        );
                        let color = Vector3::new(
                            sigmoid(self.params[(v, 1)]),
                            sigmoid(self.params[(v, 2)]),
                            sigmoid(self.params[(v, 3)]),
                        );
                        out.push(VoxelPointSample {
                            position: pos,
                            density: sigma,
                            color,
                        });
                    }
                }
            }
        }
        out
    }

    #[inline]
    fn trilinear_cell(&self, pos: &Vector3<f64>) -> Option<([usize; 8], [f64; 8])> {
        let e = self.half_extent;
        if pos.x.abs() > e || pos.y.abs() > e || pos.z.abs() > e {
            return None;
        }
        let max_coord = (self.grid_res - 1) as f64;
        let inv_span = max_coord / (2.0 * e);
        let gx = ((pos.x + e) * inv_span).clamp(0.0, max_coord);
        let gy = ((pos.y + e) * inv_span).clamp(0.0, max_coord);
        let gz = ((pos.z + e) * inv_span).clamp(0.0, max_coord);

        let ix0 = (gx.floor() as usize).min(self.grid_res - 1);
        let iy0 = (gy.floor() as usize).min(self.grid_res - 1);
        let iz0 = (gz.floor() as usize).min(self.grid_res - 1);
        let ix1 = (ix0 + 1).min(self.grid_res - 1);
        let iy1 = (iy0 + 1).min(self.grid_res - 1);
        let iz1 = (iz0 + 1).min(self.grid_res - 1);

        let tx = gx - (ix0 as f64);
        let ty = gy - (iy0 as f64);
        let tz = gz - (iz0 as f64);

        let xs = [ix0, ix1];
        let ys = [iy0, iy1];
        let zs = [iz0, iz1];
        let wx = [1.0 - tx, tx];
        let wy = [1.0 - ty, ty];
        let wz = [1.0 - tz, tz];

        let mut indices = [0_usize; 8];
        let mut weights = [0.0_f64; 8];
        let mut k = 0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    indices[k] = voxel_flat_index(xs[dx], ys[dy], zs[dz], self.grid_res);
                    weights[k] = wx[dx] * wy[dy] * wz[dz];
                    k += 1;
                }
            }
        }
        Some((indices, weights))
    }

    #[inline]
    fn dir_shade(dir: &Vector3<f64>) -> f64 {
        0.92 + 0.08 * (0.5 * (1.0 - dir.z.clamp(-1.0, 1.0)))
    }

    fn extract_ray_rgb_grad(
        bundle: &RayBundle,
        image_grad: &DMatrix<f64>,
        ray_idx: usize,
    ) -> Vector3<f64> {
        let num_rays = bundle.rays.len();
        if image_grad.nrows() == num_rays && image_grad.ncols() >= 3 {
            return Vector3::new(
                image_grad[(ray_idx, 0)],
                image_grad[(ray_idx, 1)],
                image_grad[(ray_idx, 2)],
            );
        }
        let h = bundle.height.max(1);
        let x = ray_idx / h;
        let y = ray_idx % h;
        if y < image_grad.nrows() && x < image_grad.ncols() {
            let g = image_grad[(y, x)];
            Vector3::new(g, g, g)
        } else {
            Vector3::zeros()
        }
    }

    fn accumulate_ray_backward(
        &self,
        origin: &Vector3<f64>,
        direction: &Vector3<f64>,
        grad_rgb: &Vector3<f64>,
        grad_params: &mut DMatrix<f64>,
    ) {
        let n_s = self.backward_samples.clamp(2, MAX_BACKWARD_SAMPLES);
        let delta = (self.t_far - self.t_near) / (n_s as f64);
        let shade = Self::dir_shade(direction);
        let trace = self.trace_ray_samples(origin, direction, n_s, delta, shade);

        let mut suffix_color = Vector3::zeros();
        for s in (0..n_s).rev() {
            if trace.inside[s] {
                let d_c_d_sigma = delta
                    * (trace.trans[s] * (1.0 - trace.alphas[s]) * trace.colors[s] - suffix_color);
                let g_raw_sigma =
                    grad_rgb.dot(&d_c_d_sigma) * sigmoid(trace.raw_sigmas[s]) * self.density_scale;
                let sc = trace.sig_colors[s];
                let rw = trace.ray_weights[s] * shade;
                let g_rgb = Vector3::new(
                    grad_rgb.x * rw * sc.x * (1.0 - sc.x),
                    grad_rgb.y * rw * sc.y * (1.0 - sc.y),
                    grad_rgb.z * rw * sc.z * (1.0 - sc.z),
                );
                let (idxs, ws) = &trace.cells[s];
                for k in 0..8 {
                    let (v, w) = (idxs[k], ws[k]);
                    grad_params[(v, 0)] += w * g_raw_sigma;
                    grad_params[(v, 1)] += w * g_rgb.x;
                    grad_params[(v, 2)] += w * g_rgb.y;
                    grad_params[(v, 3)] += w * g_rgb.z;
                }
            }
            suffix_color += trace.ray_weights[s] * trace.colors[s];
        }
    }

    fn trace_ray_samples(
        &self,
        origin: &Vector3<f64>,
        direction: &Vector3<f64>,
        n_s: usize,
        delta: f64,
        shade: f64,
    ) -> RayBackwardTrace {
        let mut tr = RayBackwardTrace::default();
        let mut cur_t = 1.0_f64;
        for s in 0..n_s {
            tr.trans[s] = cur_t;
            let pos = origin + (self.t_near + ((s as f64) + 0.5) * delta) * direction;
            if let Some((idxs, ws)) = self.trilinear_cell(&pos) {
                tr.inside[s] = true;
                tr.cells[s] = (idxs, ws);
                let (mut rs, mut rc) = (0.0, Vector3::zeros());
                for k in 0..8 {
                    let (v, w) = (idxs[k], ws[k]);
                    rs += w * self.params[(v, 0)];
                    rc += w * Vector3::new(
                        self.params[(v, 1)],
                        self.params[(v, 2)],
                        self.params[(v, 3)],
                    );
                }
                tr.raw_sigmas[s] = rs;
                let sigma = softplus(rs) * self.density_scale;
                let sc = Vector3::new(sigmoid(rc.x), sigmoid(rc.y), sigmoid(rc.z));
                tr.sig_colors[s] = sc;
                tr.colors[s] = sc * shade;
                let alpha = 1.0 - (-sigma * delta).exp();
                tr.alphas[s] = alpha;
                tr.ray_weights[s] = cur_t * alpha;
                cur_t *= 1.0 - alpha;
            }
        }
        tr
    }
}

struct RayBackwardTrace {
    cells: [([usize; 8], [f64; 8]); MAX_BACKWARD_SAMPLES],
    inside: [bool; MAX_BACKWARD_SAMPLES],
    raw_sigmas: [f64; MAX_BACKWARD_SAMPLES],
    sig_colors: [Vector3<f64>; MAX_BACKWARD_SAMPLES],
    colors: [Vector3<f64>; MAX_BACKWARD_SAMPLES],
    alphas: [f64; MAX_BACKWARD_SAMPLES],
    trans: [f64; MAX_BACKWARD_SAMPLES],
    ray_weights: [f64; MAX_BACKWARD_SAMPLES],
}

impl Default for RayBackwardTrace {
    fn default() -> Self {
        Self {
            cells: [([0; 8], [0.0; 8]); MAX_BACKWARD_SAMPLES],
            inside: [false; MAX_BACKWARD_SAMPLES],
            raw_sigmas: [0.0; MAX_BACKWARD_SAMPLES],
            sig_colors: [Vector3::zeros(); MAX_BACKWARD_SAMPLES],
            colors: [Vector3::zeros(); MAX_BACKWARD_SAMPLES],
            alphas: [0.0; MAX_BACKWARD_SAMPLES],
            trans: [0.0; MAX_BACKWARD_SAMPLES],
            ray_weights: [0.0; MAX_BACKWARD_SAMPLES],
        }
    }
}

impl NeRFModel for AnalyticalVoxelNeRF {
    #[verified_engine::verified]
    fn query(&self, pos: &Vector3<f64>, dir: &Vector3<f64>) -> (f64, Vector3<f64>) {
        debug_assert!(self.grid_res >= 2);
        let Some((idxs, ws)) = self.trilinear_cell(pos) else {
            return (0.0, Vector3::zeros());
        };
        let mut raw_sigma = 0.0;
        let mut raw_r = 0.0;
        let mut raw_g = 0.0;
        let mut raw_b = 0.0;
        for k in 0..8 {
            let v = idxs[k];
            let w = ws[k];
            raw_sigma += w * self.params[(v, 0)];
            raw_r += w * self.params[(v, 1)];
            raw_g += w * self.params[(v, 2)];
            raw_b += w * self.params[(v, 3)];
        }
        let sigma = softplus(raw_sigma) * self.density_scale;
        let shade = Self::dir_shade(dir);
        let color = Vector3::new(
            sigmoid(raw_r) * shade,
            sigmoid(raw_g) * shade,
            sigmoid(raw_b) * shade,
        );
        (sigma, color)
    }
}

impl DifferentiableNeRF for AnalyticalVoxelNeRF {
    #[verified_engine::verified]
    fn backward(&self, bundle: &RayBundle, image_grad: &DMatrix<f64>) -> DMatrix<f64> {
        assert_eq!(self.params.ncols(), 4);
        let mut grad_params = DMatrix::zeros(self.params.nrows(), 4);
        for (ray_idx, ray) in bundle.rays.iter().enumerate() {
            let grad_rgb = Self::extract_ray_rgb_grad(bundle, image_grad, ray_idx);
            if grad_rgb.norm_squared() > 0.0 {
                self.accumulate_ray_backward(
                    &ray.origin,
                    &ray.direction,
                    &grad_rgb,
                    &mut grad_params,
                );
            }
        }
        grad_params
    }
}

/// Converts a rendered `(height, width)` RGB image into a `(height * width, 3)` latent tensor
/// aligned with the column-major ray ordering of [`RayBundle`].
#[verified_engine::verified]
pub fn rgb_image_to_latent(image: &DMatrix<Vector3<f64>>) -> DMatrix<f64> {
    let height = image.nrows();
    let width = image.ncols();
    let num_pixels = width * height;
    assert_eq!(num_pixels, image.len());
    let mut latent = DMatrix::zeros(num_pixels, 3);
    for x in 0..width {
        for y in 0..height {
            let idx = flatten_2d_index(y, x, height);
            let px = image[(y, x)];
            latent[(idx, 0)] = px.x;
            latent[(idx, 1)] = px.y;
            latent[(idx, 2)] = px.z;
        }
    }
    latent
}

#[cfg(test)]
#[path = "tests_models.rs"]
mod tests;

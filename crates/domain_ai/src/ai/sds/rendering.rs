use nalgebra::{DMatrix, Matrix4, Vector3};
use rand::Rng;
use std::f64::consts::PI;

/// Represents a single ray: r(t) = o + t * d
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// World-space camera ray origin $\mathbf{o} \in \mathbb{R}^3$.
    pub origin: Vector3<f64>,
    /// Normalized world-space viewing direction $\mathbf{d} \in \mathbb{S}^2$.
    pub direction: Vector3<f64>,
}

/// A bundle of rays corresponding to an image grid.
#[derive(Debug, Clone, PartialEq)]
pub struct RayBundle {
    /// Column-major array of camera rays covering `width * height` pixels.
    pub rays: Vec<Ray>,
    /// Horizontal image resolution in pixels.
    pub width: usize,
    /// Vertical image resolution in pixels.
    pub height: usize,
}

/// A persistent rendering context for zero-allocation rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderContext {
    /// Scratch RGB pixel buffer of length `width * height`.
    pub image_data: Vec<Vector3<f64>>,
    /// Stratified ray-march sample depths $t_i \in [t_{\mathrm{near}}, t_{\mathrm{far}}]$.
    pub ts: Vec<f64>,
    /// Sampled volume densities $\sigma_i \ge 0$ along the active ray.
    pub densities: Vec<f64>,
    /// Sampled emitted RGB radiances $\mathbf{c}_i$ along the active ray.
    pub colors: Vec<Vector3<f64>>,
    /// Adjacent sample intervals $\delta_i = t_{i+1} - t_i$ along the active ray.
    pub deltas: Vec<f64>,
    /// Cached image width in pixels.
    pub width: usize,
    /// Cached image height in pixels.
    pub height: usize,
    /// Cached number of stratified samples per ray.
    pub n_samples: usize,
}

impl RenderContext {
    /// Creates an empty zero-capacity rendering context.
    pub fn new() -> Self {
        Self {
            image_data: Vec::new(),
            ts: Vec::new(),
            densities: Vec::new(),
            colors: Vec::new(),
            deltas: Vec::new(),
            width: 0,
            height: 0,
            n_samples: 0,
        }
    }

    /// Resizes scratch buffers to accommodate a `width` by `height` image with `n_samples` per ray.
    pub fn resize(&mut self, width: usize, height: usize, n_samples: usize) {
        if self.width != width || self.height != height {
            self.image_data.resize(width * height, Vector3::zeros());
            self.width = width;
            self.height = height;
        }
        if self.n_samples != n_samples {
            self.ts.resize(n_samples, 0.0);
            self.densities.resize(n_samples, 0.0);
            self.colors.resize(n_samples, Vector3::zeros());
            self.deltas.resize(n_samples, 0.0);
            self.n_samples = n_samples;
        }
    }
}

impl Default for RenderContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Module 1.1: Ray Bundle Generation
#[verified_engine::verified]
pub fn generate_ray_bundle(
    pose: &Matrix4<f64>,
    width: usize,
    height: usize,
    fov_y: f64,
) -> RayBundle {
    let mut rays = Vec::with_capacity(width * height);
    let aspect_ratio = width as f64 / height as f64;

    let half_height = (fov_y / 2.0).tan();
    let half_width = aspect_ratio * half_height;

    let origin = pose.transform_point(&nalgebra::Point3::origin()).coords;

    // Iterate in column-major order (x then y)
    for x in 0..width {
        for y in 0..height {
            let u = (x as f64 + 0.5) / width as f64;
            let v = (y as f64 + 0.5) / height as f64;

            let screen_x = (2.0 * u - 1.0) * half_width;
            let screen_y = (1.0 - 2.0 * v) * half_height;

            let dir_cam = Vector3::new(screen_x, screen_y, -1.0);
            let dir_world = pose.transform_vector(&dir_cam).normalize();

            rays.push(Ray {
                origin,
                direction: dir_world,
            });
        }
    }

    RayBundle {
        rays,
        width,
        height,
    }
}

/// Module 1.2: Stratified Sampling
#[verified_engine::verified]
pub fn stratified_sampling(t_near: f64, t_far: f64, n_samples: usize, out: &mut [f64]) {
    let mut rng = oxidize_core::rng::OxidizeRng::default();
    stratified_sampling_with_rng(t_near, t_far, n_samples, &mut rng, out)
}

/// Module 1.2: Stratified Sampling using an injected RNG.
#[verified_engine::verified]
pub fn stratified_sampling_with_rng<R: Rng + ?Sized>(
    t_near: f64,
    t_far: f64,
    n_samples: usize,
    rng: &mut R,
    out: &mut [f64],
) {
    let bin_size = (t_far - t_near) / n_samples as f64;
    for (i, item) in out.iter_mut().enumerate().take(n_samples) {
        let bin_start = t_near + i as f64 * bin_size;
        *item = bin_start + rng.r#gen::<f64>() * bin_size;
    }
}

/// Module 1.3: MLP Query & Positional Encoding (Helper)
#[verified_engine::verified]
pub fn positional_encoding(p: f64, l: usize) -> Vec<f64> {
    let mut encoded = Vec::with_capacity(2 * l);
    for i in 0..l {
        let freq = 2.0_f64.powi(i as i32) * PI;
        encoded.push((freq * p).sin());
        encoded.push((freq * p).cos());
    }
    encoded
}

/// Continuous volumetric Neural Radiance Field (NeRF) query interface $(\mathbf{x}, \mathbf{d}) \mapsto (\sigma, \mathbf{c})$.
///
/// Integrated along camera rays $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ via classical optical quadrature:
///
/// $$\hat{\mathbf{C}}(\mathbf{r}) = \sum_{i=1}^{N} T_i \left(1 - \exp(-\sigma_i \delta_i)\right) \mathbf{c}_i, \qquad T_i = \exp\!\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$$
///
/// Reference: Mildenhall et al. (2020), arXiv:2003.08934; see
/// [`docs/math_specs/domain_ai.md`](../../../../../docs/math_specs/domain_ai.md).
///
/// # Stability
/// Non-negative volume density $\sigma_i \ge 0$ and positive step lengths $\delta_i > 0$ guarantee
/// $1 - \exp(-\sigma_i \delta_i) \in [0, 1)$ and monotone transmittance decay $T_{i+1} \le T_i$.
///
/// # Domain of Validity
/// Finite 3D query coordinates $\mathbf{x} \in \mathbb{R}^3$, unit viewing directions
/// $\|\mathbf{d}\|_2 = 1$, non-negative densities $\sigma \ge 0$, and bounded radiance $\mathbf{c} \in [0, 1]^3$.
pub trait NeRFModel {
    /// Evaluates volume density $\sigma(\mathbf{x}) \ge 0$ and emitted RGB radiance $\mathbf{c}(\mathbf{x}, \mathbf{d})$ at `pos` and `dir`.
    #[verified_engine::verified]
    fn query(&self, pos: &Vector3<f64>, dir: &Vector3<f64>) -> (f64, Vector3<f64>);
}

/// Module 1.4: Volume Integration (Compositing) Helper
#[verified_engine::verified]
pub fn volume_integration(
    densities: &[f64],
    colors: &[Vector3<f64>],
    deltas: &[f64],
) -> Vector3<f64> {
    let mut final_color = Vector3::zeros();
    let mut transmittance = 1.0;

    for i in 0..densities.len() {
        let sigma = densities[i];
        let color = colors[i];
        let delta = deltas[i];

        let alpha = 1.0 - (-sigma * delta).exp();
        let weight = transmittance * alpha;
        final_color += weight * color;
        transmittance *= 1.0 - alpha;

        if transmittance < 1e-4 {
            break;
        }
    }

    final_color
}

/// Module 1.4: Volume Integration (Full Image)
#[verified_engine::verified]
pub fn render_image<M: NeRFModel + ?Sized>(
    ctx: &mut RenderContext,
    bundle: &RayBundle,
    model: &M,
    t_near: f64,
    t_far: f64,
    n_samples: usize,
) -> DMatrix<Vector3<f64>> {
    let mut rng = oxidize_core::rng::OxidizeRng::default();
    render_image_with_rng(ctx, bundle, model, t_near, t_far, n_samples, &mut rng)
}

/// Module 1.4: Volume Integration (Full Image) using an injected RNG.
#[verified_engine::verified]
pub fn render_image_with_rng<M: NeRFModel + ?Sized, R: Rng + ?Sized>(
    ctx: &mut RenderContext,
    bundle: &RayBundle,
    model: &M,
    t_near: f64,
    t_far: f64,
    n_samples: usize,
    rng: &mut R,
) -> DMatrix<Vector3<f64>> {
    ctx.resize(bundle.width, bundle.height, n_samples);

    // Explicitly lock the global allocator for zero-allocation compliance during the hot path
    verified_engine::allocator::lock_allocations();

    struct LockGuard;
    impl Drop for LockGuard {
        fn drop(&mut self) {
            verified_engine::allocator::unlock_allocations();
        }
    }
    let _guard = LockGuard;

    // Hot path starts
    for (ray_idx, ray) in bundle.rays.iter().enumerate() {
        stratified_sampling_with_rng(t_near, t_far, n_samples, rng, &mut ctx.ts);

        for i in 0..n_samples {
            let t = ctx.ts[i];
            let pos = ray.origin + t * ray.direction;

            let (sigma, c) = model.query(&pos, &ray.direction);

            ctx.densities[i] = sigma;
            ctx.colors[i] = c;

            let next_t = if i < n_samples - 1 {
                ctx.ts[i + 1]
            } else if i > 0 {
                t + (t - ctx.ts[i - 1])
            } else {
                t + 1.0
            };
            ctx.deltas[i] = next_t - t;
        }

        ctx.image_data[ray_idx] = volume_integration(&ctx.densities, &ctx.colors, &ctx.deltas);
    }
    // Hot path ends

    drop(_guard);

    DMatrix::from_vec(bundle.height, bundle.width, ctx.image_data.clone())
}

#[cfg(test)]
#[path = "tests_rendering.rs"]
mod tests;

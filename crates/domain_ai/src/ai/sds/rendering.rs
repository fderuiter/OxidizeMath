use crate::error::AIError;
use math_commons::error::MathError;
use nalgebra::{DMatrix, Matrix4, Vector3};
use rand::Rng;
use std::f64::consts::PI;

/// Represents a single ray: r(t) = o + t * d
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    #[allow(missing_docs)]
    pub origin: Vector3<f64>,
    #[allow(missing_docs)]
    pub direction: Vector3<f64>,
}

/// A bundle of rays corresponding to an image grid.
pub struct RayBundle {
    #[allow(missing_docs)]
    pub rays: Vec<Ray>,
    #[allow(missing_docs)]
    pub width: usize,
    #[allow(missing_docs)]
    pub height: usize,
}

/// A persistent rendering context for zero-allocation rendering.
pub struct RenderContext {
    #[allow(missing_docs)]
    pub image_data: Vec<Vector3<f64>>,
    #[allow(missing_docs)]
    pub ts: Vec<f64>,
    #[allow(missing_docs)]
    pub densities: Vec<f64>,
    #[allow(missing_docs)]
    pub colors: Vec<Vector3<f64>>,
    #[allow(missing_docs)]
    pub deltas: Vec<f64>,
    #[allow(missing_docs)]
    pub width: usize,
    #[allow(missing_docs)]
    pub height: usize,
    #[allow(missing_docs)]
    pub n_samples: usize,
}

impl RenderContext {
    #[allow(missing_docs)]
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

    /// Resizes internal pixel and ray-sample buffers with checked `width * height` multiplication.
    #[verified_engine::verified(alloc_ok = "pre-allocates RenderContext pixel and sample buffers")]
    pub fn try_resize(
        &mut self,
        width: usize,
        height: usize,
        n_samples: usize,
    ) -> Result<(), AIError> {
        let pixels = width
            .checked_mul(height)
            .ok_or(MathError::CapacityOverflow {
                context: "RenderContext::try_resize width * height",
            })?;
        if self.width != width || self.height != height {
            self.image_data.resize(pixels, Vector3::zeros());
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
        Ok(())
    }

    #[allow(missing_docs)]
    pub fn resize(&mut self, width: usize, height: usize, n_samples: usize) {
        if self.try_resize(width, height, n_samples).is_err() {
            self.image_data.clear();
            self.width = 0;
            self.height = 0;
            self.n_samples = 0;
        }
    }
}

impl Default for RenderContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Checked ray-bundle generator rejecting zero dimensions, non-finite poses/FOVs, and `width * height` overflow.
#[verified_engine::verified(alloc_ok = "allocates viewport RayBundle")]
pub fn try_generate_ray_bundle(
    pose: &Matrix4<f64>,
    width: usize,
    height: usize,
    fov_y: f64,
) -> Result<RayBundle, AIError> {
    if width == 0 || height == 0 {
        return Err(AIError::InvalidParameter {
            name: "viewport_dimension".to_string(),
            value: width.min(height) as f64,
        });
    }
    if !fov_y.is_finite() || fov_y <= 0.0 || fov_y >= PI {
        return Err(AIError::InvalidParameter {
            name: "fov_y".to_string(),
            value: fov_y,
        });
    }
    if !pose.iter().all(|v| v.is_finite()) {
        return Err(AIError::InvalidParameter {
            name: "pose".to_string(),
            value: f64::NAN,
        });
    }
    let total_rays = width
        .checked_mul(height)
        .ok_or(MathError::CapacityOverflow {
            context: "generate_ray_bundle width * height",
        })?;

    let mut rays = Vec::with_capacity(total_rays);
    let aspect_ratio = width as f64 / height as f64;
    let half_height = (fov_y / 2.0).tan();
    let half_width = aspect_ratio * half_height;
    let origin = pose.transform_point(&nalgebra::Point3::origin()).coords;

    for x in 0..width {
        let u = (x as f64 + 0.5) / width as f64;
        let screen_x = (2.0 * u - 1.0) * half_width;
        for y in 0..height {
            let v = (y as f64 + 0.5) / height as f64;
            let screen_y = (1.0 - 2.0 * v) * half_height;
            let dir_cam = Vector3::new(screen_x, screen_y, -1.0);
            let raw_dir = pose.transform_vector(&dir_cam);
            let norm = raw_dir.norm();
            let dir_world = if norm > 1e-12 && norm.is_finite() {
                raw_dir / norm
            } else {
                Vector3::new(0.0, 0.0, -1.0)
            };
            rays.push(Ray {
                origin,
                direction: dir_world,
            });
        }
    }

    Ok(RayBundle {
        rays,
        width,
        height,
    })
}

/// Module 1.1: Ray Bundle Generation
#[verified_engine::verified(alloc_ok = "allocates viewport RayBundle")]
pub fn generate_ray_bundle(
    pose: &Matrix4<f64>,
    width: usize,
    height: usize,
    fov_y: f64,
) -> RayBundle {
    try_generate_ray_bundle(pose, width, height, fov_y).unwrap_or(RayBundle {
        rays: Vec::new(),
        width,
        height,
    })
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
    if n_samples == 0 || out.is_empty() || !t_near.is_finite() || !t_far.is_finite() {
        return;
    }
    let bin_size = (t_far - t_near) / n_samples as f64;
    for (i, item) in out.iter_mut().enumerate().take(n_samples) {
        let bin_start = t_near + i as f64 * bin_size;
        *item = bin_start + rng.r#gen::<f64>() * bin_size;
    }
}

/// Checked positional encoding rejecting non-finite scalar coordinates and `2 * l` overflow.
#[verified_engine::verified(alloc_ok = "allocates 2L sinusoidal positional encoding vector")]
pub fn try_positional_encoding(p: f64, l: usize) -> Result<Vec<f64>, AIError> {
    if !p.is_finite() {
        return Err(AIError::InvalidParameter {
            name: "p".to_string(),
            value: p,
        });
    }
    let cap = 2usize.checked_mul(l).ok_or(MathError::CapacityOverflow {
        context: "positional_encoding 2 * l",
    })?;
    let mut encoded = Vec::with_capacity(cap);
    for i in 0..l {
        let exp = i32::try_from(i).unwrap_or(i32::MAX);
        let freq = 2.0_f64.powi(exp) * PI;
        let angle = freq * p;
        if angle.is_finite() {
            encoded.push(angle.sin());
            encoded.push(angle.cos());
        } else {
            encoded.push(0.0);
            encoded.push(1.0);
        }
    }
    Ok(encoded)
}

/// Module 1.3: MLP Query & Positional Encoding (Helper)
#[verified_engine::verified(alloc_ok = "allocates 2L sinusoidal positional encoding vector")]
pub fn positional_encoding(p: f64, l: usize) -> Vec<f64> {
    try_positional_encoding(p, l).unwrap_or_default()
}

/// Module 1.3: MLP Query Interface
pub trait NeRFModel {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn query(&self, pos: &Vector3<f64>, dir: &Vector3<f64>) -> (f64, Vector3<f64>);
}

/// Module 1.4: Volume Integration (Compositing) Helper
#[verified_engine::verified]
pub fn volume_integration(
    densities: &[f64],
    colors: &[Vector3<f64>],
    deltas: &[f64],
) -> Vector3<f64> {
    let n = densities.len().min(colors.len()).min(deltas.len());
    let mut final_color = Vector3::zeros();
    let mut transmittance = 1.0;

    for i in 0..n {
        let sigma = densities[i];
        let color = colors[i];
        let delta = deltas[i];
        if !sigma.is_finite() || !delta.is_finite() || !color.iter().all(|c| c.is_finite()) {
            continue;
        }

        let alpha = 1.0 - (-sigma.max(0.0) * delta.max(0.0)).exp();
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
#[verified_engine::verified(alloc_ok = "resizes context and returns owned DMatrix image")]
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
#[verified_engine::verified(
    alloc_ok = "resizes context before hot loop and builds output DMatrix after unlock"
)]
pub fn render_image_with_rng<M: NeRFModel + ?Sized, R: Rng + ?Sized>(
    ctx: &mut RenderContext,
    bundle: &RayBundle,
    model: &M,
    t_near: f64,
    t_far: f64,
    n_samples: usize,
    rng: &mut R,
) -> DMatrix<Vector3<f64>> {
    if ctx
        .try_resize(bundle.width, bundle.height, n_samples)
        .is_err()
        || bundle.width == 0
        || bundle.height == 0
    {
        return DMatrix::from_element(0, 0, Vector3::zeros());
    }
    if n_samples == 0 || bundle.rays.len() != ctx.image_data.len() {
        ctx.image_data.fill(Vector3::zeros());
        return DMatrix::from_vec(bundle.height, bundle.width, ctx.image_data.clone());
    }

    verified_engine::allocator::lock_allocations();
    struct LockGuard;
    impl Drop for LockGuard {
        fn drop(&mut self) {
            verified_engine::allocator::unlock_allocations();
        }
    }
    let _guard = LockGuard;

    for (ray_idx, ray) in bundle.rays.iter().enumerate() {
        stratified_sampling_with_rng(t_near, t_far, n_samples, rng, &mut ctx.ts);
        for i in 0..n_samples {
            let t = ctx.ts[i];
            let pos = ray.origin + t * ray.direction;
            let (sigma, c) = model.query(&pos, &ray.direction);
            ctx.densities[i] = sigma;
            ctx.colors[i] = c;
            let next_t = if i + 1 < n_samples {
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

    drop(_guard);
    DMatrix::from_vec(bundle.height, bundle.width, ctx.image_data.clone())
}

#[cfg(test)]
#[path = "tests_rendering.rs"]
mod tests;

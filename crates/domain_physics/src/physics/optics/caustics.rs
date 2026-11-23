//! Pre-allocated RK4 Hamiltonian ray tracer and geometric caustic cusp detector.
//!
//! Integrates Hamiltonian ray equations in optical path parameter $\tau$:
//! $$\frac{d\mathbf{r}}{d\tau} = \mathbf{p}, \qquad \frac{d\mathbf{p}}{d\tau} = \frac{1}{2}\nabla(n^2)$$
//! using [`RungeKutta4`] on stack-resident [`ArrayState<4>`] vectors, coupled with exact vector
//! Snell refraction at discrete positive- and negative-index interfaces and ray-tube Jacobian
//! sign-flip detection for geometric caustics.

use super::refraction::{OpticalMediumConfig, OpticsPreset, snell_refract_vector};
use math_commons::math_kernel::types::flatten_2d_index;
use pure_math::pure_math::analysis::ode::{ArrayState, OdeSystem, RungeKutta4, Solver};

// [cite:optics]

impl OdeSystem<ArrayState<4>> for OpticalMediumConfig {
    #[verified_engine::verified]
    fn derivative(&self, _tau: f64, state: &ArrayState<4>) -> ArrayState<4> {
        let [gx, gy] = self.grad_n_squared(state.0[0], state.0[1]);
        ArrayState([state.0[2], state.0[3], 0.5 * gx, 0.5 * gy])
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _tau: f64, state: &ArrayState<4>, out: &mut ArrayState<4>) {
        let [gx, gy] = self.grad_n_squared(state.0[0], state.0[1]);
        out.0[0] = state.0[2];
        out.0[1] = state.0[3];
        out.0[2] = 0.5 * gx;
        out.0[3] = 0.5 * gy;
    }
}

/// Zero-allocation RK4 Hamiltonian ray bundle and caustic envelope tracer.
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::optics::caustics::HamiltonianRayTracer;
/// use math_explorer::physics::optics::refraction::{OpticalMediumConfig, OpticsPreset};
///
/// let medium = OpticalMediumConfig {
///     preset: OpticsPreset::LuneburgLens,
///     ..OpticalMediumConfig::default()
/// };
/// let mut tracer = HamiltonianRayTracer::new(32, 240, 256);
/// tracer.trace_in_place(&medium, 15, 0.005);
/// assert_eq!(tracer.ray_count(), 15);
/// assert!(!tracer.caustic_points().is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct HamiltonianRayTracer {
    solver: RungeKutta4<ArrayState<4>>,
    max_rays: usize,
    max_steps_per_ray: usize,
    stride: usize,
    ray_count: usize,
    points: Vec<[f64; 2]>,
    dirs: Vec<[f64; 2]>,
    ray_lengths: Vec<usize>,
    caustic_points: Vec<[f64; 2]>,
    caustic_count: usize,
}

impl HamiltonianRayTracer {
    /// Pre-allocates trajectory and caustic buffers for up to `max_rays` rays.
    #[verified_engine::verified]
    pub fn new(max_rays: usize, max_steps_per_ray: usize, max_caustic_points: usize) -> Self {
        let safe_rays = max_rays.max(2);
        let safe_steps = max_steps_per_ray.max(8);
        let stride = safe_steps + 1;
        let total = safe_rays * stride;
        Self {
            solver: RungeKutta4::new(&ArrayState([0.0; 4])),
            max_rays: safe_rays,
            max_steps_per_ray: safe_steps,
            stride,
            ray_count: 0,
            points: vec![[0.0; 2]; total],
            dirs: vec![[1.0, 0.0]; total],
            ray_lengths: vec![0; safe_rays],
            caustic_points: vec![[0.0; 2]; max_caustic_points.max(16)],
            caustic_count: 0,
        }
    }

    /// Returns the number of active rays from the most recent `trace_in_place` call.
    #[inline]
    pub fn ray_count(&self) -> usize {
        self.ray_count
    }

    /// Returns the `[x, y]` trajectory slice for ray `ray_idx`.
    #[inline]
    pub fn ray_points(&self, ray_idx: usize) -> &[[f64; 2]] {
        if ray_idx >= self.ray_count {
            return &[];
        }
        let start = flatten_2d_index(0, ray_idx, self.stride);
        let len = self.ray_lengths[ray_idx];
        &self.points[start..start + len]
    }

    /// Returns the detected `[x, y]` caustic envelope and focus points.
    #[inline]
    pub fn caustic_points(&self) -> &[[f64; 2]] {
        &self.caustic_points[..self.caustic_count]
    }

    /// Traces a single ray initialized at `(x0, y0)` with unit direction `dir0` in place.
    #[verified_engine::verified]
    pub fn trace_single_ray_in_place(
        &mut self,
        medium: &OpticalMediumConfig,
        ray_idx: usize,
        origin: [f64; 2],
        dir0: [f64; 2],
        d_tau: f64,
    ) {
        if ray_idx >= self.max_rays {
            return;
        }
        let norm0 = (dir0[0] * dir0[0] + dir0[1] * dir0[1]).sqrt().max(1e-12);
        let ux = dir0[0] / norm0;
        let uy = dir0[1] / norm0;
        let n0 = medium.refractive_index_at(origin[0], origin[1]).abs();
        let mut state = ArrayState([origin[0], origin[1], n0 * ux, n0 * uy]);
        let base = flatten_2d_index(0, ray_idx, self.stride);
        self.points[base] = origin;
        self.dirs[base] = [ux, uy];
        let mut count = 1_usize;
        let step_tau = d_tau.max(1e-4);

        for s in 1..=self.max_steps_per_ray {
            let prev = state;
            let reg0 = medium.discrete_region_id(prev.0[0], prev.0[1]);
            self.solver
                .step(medium, (s as f64) * step_tau, &mut state, step_tau);
            if !Self::in_domain(medium, state.0[0], state.0[1])
                || medium.is_obstacle_at(state.0[0], state.0[1], step_tau)
            {
                break;
            }
            let reg1 = medium.discrete_region_id(state.0[0], state.0[1]);
            if reg0 != reg1 {
                state = self.refract_across_interface(medium, prev, step_tau, reg0);
            }
            let p_norm = (state.0[2] * state.0[2] + state.0[3] * state.0[3])
                .sqrt()
                .max(1e-12);
            let slot = flatten_2d_index(s, ray_idx, self.stride);
            self.points[slot] = [state.0[0], state.0[1]];
            self.dirs[slot] = [state.0[2] / p_norm, state.0[3] / p_norm];
            count += 1;
        }
        self.ray_lengths[ray_idx] = count;
    }

    /// Traces a full preset-appropriate ray bundle and updates caustic points with zero heap allocation.
    #[verified_engine::verified]
    pub fn trace_in_place(
        &mut self,
        medium: &OpticalMediumConfig,
        requested_rays: usize,
        d_tau: f64,
    ) {
        let count = requested_rays.clamp(2, self.max_rays);
        self.ray_count = count;
        for r in 0..count {
            let frac = if count > 1 {
                (r as f64) / ((count - 1) as f64)
            } else {
                0.5
            };
            let (origin, dir) = Self::initial_ray_for_preset(medium, frac);
            self.trace_single_ray_in_place(medium, r, origin, dir, d_tau);
        }
        self.detect_caustics_in_place(medium);
    }

    #[inline]
    #[verified_engine::verified]
    fn in_domain(medium: &OpticalMediumConfig, x: f64, y: f64) -> bool {
        x >= 0.0 && x <= medium.domain_width && y >= 0.0 && y <= medium.domain_height
    }

    #[verified_engine::verified]
    fn initial_ray_for_preset(medium: &OpticalMediumConfig, frac: f64) -> ([f64; 2], [f64; 2]) {
        let u = 2.0 * frac - 1.0; // [-1, 1]
        let [_cx, cy] = medium.lens_center();
        match medium.preset {
            OpticsPreset::LuneburgLens | OpticsPreset::BiconvexDroplet => {
                let half_span = 0.88 * medium.lens_radius;
                ([0.08 * medium.domain_width, cy + u * half_span], [1.0, 0.0])
            }
            OpticsPreset::DoubleSlit => {
                let half_span = (0.5 * medium.slit_separation + medium.aperture_width)
                    .min(0.40 * medium.domain_height);
                ([0.08 * medium.domain_width, cy + u * half_span], [1.0, 0.0])
            }
            OpticsPreset::MaxwellFishEye => {
                let src = medium.source_position();
                let theta = u * 0.95;
                (src, [theta.cos(), theta.sin()])
            }
            OpticsPreset::VeselagoLens => {
                let src = medium.source_position();
                let theta = u * 0.62;
                (src, [theta.cos(), theta.sin()])
            }
        }
    }

    #[verified_engine::verified]
    fn refract_across_interface(
        &mut self,
        medium: &OpticalMediumConfig,
        prev: ArrayState<4>,
        step_tau: f64,
        reg0: u8,
    ) -> ArrayState<4> {
        let mut lo = 0.0_f64;
        let mut hi = 1.0_f64;
        for _ in 0..14 {
            let mid = 0.5 * (lo + hi);
            let mut probe = prev;
            self.solver.step(medium, 0.0, &mut probe, mid * step_tau);
            if medium.discrete_region_id(probe.0[0], probe.0[1]) == reg0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let mut before = prev;
        self.solver.step(medium, 0.0, &mut before, lo * step_tau);
        let mut after = prev;
        self.solver.step(medium, 0.0, &mut after, hi * step_tau);

        let p_norm = (before.0[2] * before.0[2] + before.0[3] * before.0[3])
            .sqrt()
            .max(1e-12);
        let inc_unit = [before.0[2] / p_norm, before.0[3] / p_norm];
        let bx = 0.5 * (before.0[0] + after.0[0]);
        let by = 0.5 * (before.0[1] + after.0[1]);
        let n1 = medium.refractive_index_at(before.0[0], before.0[1]);
        let n2 = medium.refractive_index_at(after.0[0], after.0[1]);
        let normal = medium.interface_normal(bx, by, inc_unit);
        let refr_unit = snell_refract_vector(inc_unit, normal, n1, n2);
        let target_n = if (refr_unit[0] * normal[0] + refr_unit[1] * normal[1]) < 0.0 {
            n1.abs()
        } else {
            n2.abs()
        };
        let mut post = ArrayState([
            bx + 1e-6 * refr_unit[0],
            by + 1e-6 * refr_unit[1],
            target_n * refr_unit[0],
            target_n * refr_unit[1],
        ]);
        let rem_tau = (1.0 - hi).max(0.0) * step_tau;
        if rem_tau > 0.0 {
            self.solver.step(medium, 0.0, &mut post, rem_tau);
        }
        post
    }

    #[verified_engine::verified]
    fn detect_caustics_in_place(&mut self, medium: &OpticalMediumConfig) {
        self.caustic_count = 0;
        let max_out = self.caustic_points.len();
        if self.ray_count < 2 {
            return;
        }
        let max_sep_sq = 0.08 * 0.08 * (medium.domain_height * medium.domain_height);
        for r in 0..(self.ray_count - 1) {
            let len = self.ray_lengths[r].min(self.ray_lengths[r + 1]);
            if len < 3 {
                continue;
            }
            let mut prev_j = self.ray_tube_jacobian(r, 1);
            for s in 2..len {
                let curr_j = self.ray_tube_jacobian(r, s);
                if prev_j * curr_j <= 0.0 && (prev_j != 0.0 || curr_j != 0.0) {
                    let i0 = flatten_2d_index(s, r, self.stride);
                    let i1 = flatten_2d_index(s, r + 1, self.stride);
                    let p0 = self.points[i0];
                    let p1 = self.points[i1];
                    let dx = p1[0] - p0[0];
                    let dy = p1[1] - p0[1];
                    if dx * dx + dy * dy <= max_sep_sq && self.caustic_count < max_out {
                        self.caustic_points[self.caustic_count] =
                            [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])];
                        self.caustic_count += 1;
                    }
                }
                prev_j = curr_j;
            }
        }
    }

    #[inline]
    #[verified_engine::verified]
    fn ray_tube_jacobian(&self, ray_idx: usize, step_idx: usize) -> f64 {
        let i0 = flatten_2d_index(step_idx, ray_idx, self.stride);
        let i1 = flatten_2d_index(step_idx, ray_idx + 1, self.stride);
        let p0 = self.points[i0];
        let p1 = self.points[i1];
        let d0 = self.dirs[i0];
        let d1 = self.dirs[i1];
        let vx = 0.5 * (d0[0] + d1[0]);
        let vy = 0.5 * (d0[1] + d1[1]);
        vx * (p1[1] - p0[1]) - vy * (p1[0] - p0[0])
    }
}

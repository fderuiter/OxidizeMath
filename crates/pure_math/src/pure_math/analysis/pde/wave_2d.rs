//! Pre-allocated 2D Finite-Difference Time-Domain (FDTD) scalar and dispersive wave solver.
//!
//! Solves the 2D inhomogeneous damped wave equation on a structured Cartesian grid:
//!
//! $$\frac{n(x, y)^2}{c_0^2} \frac{\partial^2 u}{\partial t^2} + \sigma(x, y)\frac{n(x, y)^2}{c_0^2}\frac{\partial u}{\partial t} = \nabla^2 u + S(x, y, t)$$
//!
//! using [`StencilOperators::central_diff_2nd_2d`] in positive-index media and a passive
//! matched-Drude auxiliary formulation when negative-index metamaterial cells ($n < 0$) are
//! present. All grid and auxiliary buffers are allocated once at construction so stepping runs
//! with zero dynamic heap allocations under `VerifiedAllocator`.

use super::fused_stepper::StencilOperators;
use math_commons::math_kernel::types::{StepSize, flatten_2d_index};

// [cite:optics]

const MIN_ABS_INDEX: f64 = 0.05;
const CFL_SAFETY: f64 = 0.95;

/// Spatial geometry of a coherent harmonic wave source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveSourceKind {
    /// Vertical line source at column `x_index` spanning rows `y_min..=y_max`.
    PlaneWave {
        /// Grid column index of the coherent line source.
        x_index: usize,
        /// Inclusive lower grid row index.
        y_min: usize,
        /// Inclusive upper grid row index.
        y_max: usize,
    },
    /// Localized soft point source centered at `(x, y)`.
    PointSource {
        /// Grid column index of the point source.
        x: usize,
        /// Grid row index of the point source.
        y: usize,
    },
}

/// Time-harmonic soft source specification for [`WaveField2D`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveSource2D {
    /// Spatial source geometry.
    pub kind: WaveSourceKind,
    /// Driving frequency $f$ in cycles per unit time ($\omega = 2\pi f$).
    pub frequency: f64,
    /// Peak source injection amplitude.
    pub amplitude: f64,
    /// Phase offset $\phi_0$ in radians.
    pub phase: f64,
}

impl WaveSource2D {
    /// Evaluates the smoothly ramped harmonic source value at simulation time `t`.
    #[inline]
    #[verified_engine::verified]
    pub fn evaluate_at_time(&self, t: f64) -> f64 {
        if self.amplitude == 0.0 || self.frequency <= 0.0 {
            return 0.0;
        }
        let omega = std::f64::consts::TAU * self.frequency;
        let ramp_tau = (1.5 / self.frequency).max(1e-6);
        let ratio = t / ramp_tau;
        let envelope = 1.0 - (-ratio * ratio).exp();
        self.amplitude * envelope * (omega * t + self.phase).sin()
    }
}

/// Grid and boundary configuration for [`WaveField2D`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveFieldConfig2D {
    /// Number of grid columns (`>= 8`).
    pub width: usize,
    /// Number of grid rows (`>= 8`).
    pub height: usize,
    /// Horizontal grid spacing $\Delta x > 0$.
    pub dx: f64,
    /// Vertical grid spacing $\Delta y > 0$.
    pub dy: f64,
    /// Reference background wave speed $c_0 > 0$.
    pub c0: f64,
    /// Sponge absorbing boundary thickness in grid cells.
    pub sponge_width: usize,
    /// Peak damping coefficient $\sigma_{\max} \ge 0$ at the outer boundary.
    pub sponge_max_damping: f64,
    /// Exponential moving-average factor $\beta \in (0, 1]$ for `irradiance` accumulation.
    pub irradiance_decay: f64,
}

impl Default for WaveFieldConfig2D {
    fn default() -> Self {
        Self {
            width: 128,
            height: 128,
            dx: 1.0 / 128.0,
            dy: 1.0 / 128.0,
            c0: 1.0,
            sponge_width: 10,
            sponge_max_damping: 22.0,
            irradiance_decay: 0.04,
        }
    }
}

/// Pre-allocated 2D FDTD scalar and dispersive metamaterial wave solver.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::math_kernel::types::flatten_2d_index;
/// use math_explorer::pure_math::analysis::pde::wave::{
///     WaveField2D, WaveFieldConfig2D, WaveSource2D, WaveSourceKind,
/// };
///
/// let config = WaveFieldConfig2D {
///     width: 64,
///     height: 64,
///     dx: 1.0 / 64.0,
///     dy: 1.0 / 64.0,
///     ..WaveFieldConfig2D::default()
/// };
/// let mut field = WaveField2D::new(config);
/// let src = WaveSource2D {
///     kind: WaveSourceKind::PointSource { x: 32, y: 32 },
///     frequency: 8.0,
///     amplitude: 25.0,
///     phase: 0.0,
/// };
/// let dt = field.cfl_dt();
/// field.step(dt, Some(&src));
/// assert!(field.field()[flatten_2d_index(32, 32, 64)].is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct WaveField2D {
    config: WaveFieldConfig2D,
    ops: StencilOperators,
    u_prev: Vec<f64>,
    u_curr: Vec<f64>,
    u_next: Vec<f64>,
    irradiance: Vec<f64>,
    refractive_index: Vec<f64>,
    wave_speed_sq: Vec<f64>,
    damping: Vec<f64>,
    obstacle: Vec<u8>,
    hx: Vec<f64>,
    hy: Vec<f64>,
    jz: Vec<f64>,
    mx: Vec<f64>,
    my: Vec<f64>,
    omega_p_sq: Vec<f64>,
    has_negative_index: bool,
    drude_omega_ref: f64,
    max_wave_speed: f64,
    time: f64,
    step_count: usize,
}

impl WaveField2D {
    /// Allocates a new 2D FDTD wave solver initialized to a homogeneous $n = 1$ medium at rest.
    #[verified_engine::verified]
    pub fn new(config: WaveFieldConfig2D) -> Self {
        assert!(
            config.width >= 8 && config.height >= 8,
            "WaveField2D dimensions must be at least 8x8"
        );
        assert!(
            config.dx > 0.0 && config.dy > 0.0 && config.c0 > 0.0,
            "WaveField2D spacings dx, dy and speed c0 must be positive"
        );
        let len = config.width * config.height;
        let ops = StencilOperators::new_2d(StepSize(config.dx), StepSize(config.dy));
        let c0_sq = config.c0 * config.c0;
        let mut solver = Self {
            config,
            ops,
            u_prev: vec![0.0; len],
            u_curr: vec![0.0; len],
            u_next: vec![0.0; len],
            irradiance: vec![0.0; len],
            refractive_index: vec![1.0; len],
            wave_speed_sq: vec![c0_sq; len],
            damping: vec![0.0; len],
            obstacle: vec![0; len],
            hx: vec![0.0; len],
            hy: vec![0.0; len],
            jz: vec![0.0; len],
            mx: vec![0.0; len],
            my: vec![0.0; len],
            omega_p_sq: vec![0.0; len],
            has_negative_index: false,
            drude_omega_ref: std::f64::consts::TAU * 10.0,
            max_wave_speed: config.c0,
            time: 0.0,
            step_count: 0,
        };
        solver.rebuild_sponge_damping();
        solver
    }

    /// Returns the grid configuration.
    #[inline]
    pub fn config(&self) -> &WaveFieldConfig2D {
        &self.config
    }

    /// Returns the current scalar wave displacement slice $u(x, y, t)$.
    #[inline]
    pub fn field(&self) -> &[f64] {
        &self.u_curr
    }

    /// Returns the exponentially smoothed wave irradiance slice $\langle u^2 \rangle(x, y)$.
    #[inline]
    pub fn irradiance(&self) -> &[f64] {
        &self.irradiance
    }

    /// Returns the signed refractive index slice $n(x, y)$.
    #[inline]
    pub fn refractive_index(&self) -> &[f64] {
        &self.refractive_index
    }

    /// Returns the obstacle mask slice (`1` = Dirichlet zero wall, `0` = transparent cell).
    #[inline]
    pub fn obstacle_mask(&self) -> &[u8] {
        &self.obstacle
    }

    /// Returns the current simulation time $t$ and step count.
    #[inline]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Returns the total number of executed FDTD steps.
    #[inline]
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Returns whether negative-index metamaterial cells ($n < 0$) are active.
    #[inline]
    pub fn has_negative_index(&self) -> bool {
        self.has_negative_index
    }

    /// Clears wave displacement, auxiliary Drude currents, and irradiance buffers in place.
    #[verified_engine::verified]
    pub fn clear_state(&mut self) {
        self.u_prev.fill(0.0);
        self.u_curr.fill(0.0);
        self.u_next.fill(0.0);
        self.irradiance.fill(0.0);
        self.hx.fill(0.0);
        self.hy.fill(0.0);
        self.jz.fill(0.0);
        self.mx.fill(0.0);
        self.my.fill(0.0);
        self.time = 0.0;
        self.step_count = 0;
    }

    /// Sets the reference driving frequency $f_0$ used to tune negative-index Drude cells.
    #[verified_engine::verified]
    pub fn set_reference_frequency(&mut self, frequency: f64) {
        let safe_f = frequency.abs().max(0.1);
        self.drude_omega_ref = std::f64::consts::TAU * safe_f;
        self.recompute_medium_tables();
    }

    /// Mutable access to refractive index and obstacle slices, followed by table refresh.
    #[verified_engine::verified]
    pub fn configure_medium_in_place<F>(&mut self, mut builder: F)
    where
        F: FnMut(usize, usize, &WaveFieldConfig2D, &mut f64, &mut u8),
    {
        let w = self.config.width;
        let h = self.config.height;
        for y in 0..h {
            for x in 0..w {
                let idx = flatten_2d_index(x, y, w);
                let mut n_val = 1.0;
                let mut obs = 0_u8;
                builder(x, y, &self.config, &mut n_val, &mut obs);
                self.refractive_index[idx] = n_val;
                self.obstacle[idx] = obs;
            }
        }
        self.recompute_medium_tables();
    }

    /// Computes the maximum stable 2D Courant-Friedrichs-Lewy time step $\Delta t_{\text{CFL}}$.
    #[inline]
    #[verified_engine::verified]
    pub fn cfl_dt(&self) -> f64 {
        let inv_h_norm = (self.ops.inv_dx_sq + self.ops.inv_dy_sq).sqrt();
        let raw_cfl = 1.0 / (self.max_wave_speed * inv_h_norm);
        if self.has_negative_index {
            let max_kp = self
                .omega_p_sq
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v))
                .sqrt();
            let drude_limit = if max_kp > 0.0 {
                1.2 / (max_kp * self.config.c0)
            } else {
                raw_cfl
            };
            CFL_SAFETY * raw_cfl.min(drude_limit)
        } else {
            CFL_SAFETY * raw_cfl
        }
    }

    /// Clamps a candidate time step `dt` into $(0, \Delta t_{\text{CFL}}]$.
    #[inline]
    #[verified_engine::verified]
    pub fn clamp_dt(&self, dt: f64) -> f64 {
        let limit = self.cfl_dt();
        if !dt.is_finite() || dt <= 0.0 {
            limit
        } else {
            dt.min(limit)
        }
    }

    /// Advances the 2D wavefield by one CFL-clamped step in place with zero heap allocation.
    #[verified_engine::verified]
    pub fn step(&mut self, requested_dt: f64, source: Option<&WaveSource2D>) -> f64 {
        let dt = self.clamp_dt(requested_dt);
        if self.has_negative_index {
            self.step_drude_metamaterial(dt, source);
        } else {
            self.step_scalar_leapfrog(dt, source);
        }
        self.accumulate_irradiance();
        self.time += dt;
        self.step_count += 1;
        dt
    }

    /// Returns true if all current field and irradiance values are finite.
    #[verified_engine::verified]
    pub fn is_finite(&self) -> bool {
        self.u_curr.iter().all(|v| v.is_finite()) && self.irradiance.iter().all(|v| v.is_finite())
    }

    #[verified_engine::verified]
    fn rebuild_sponge_damping(&mut self) {
        let w = self.config.width;
        let h = self.config.height;
        let sw = self.config.sponge_width;
        let sigma_max = self.config.sponge_max_damping.max(0.0);
        if sw == 0 || sigma_max == 0.0 {
            self.damping.fill(0.0);
            return;
        }
        let inv_sw = 1.0 / (sw as f64);
        for y in 0..h {
            let dy_edge = y.min(h - 1 - y);
            for x in 0..w {
                let dx_edge = x.min(w - 1 - x);
                let dist = dx_edge.min(dy_edge);
                let idx = flatten_2d_index(x, y, w);
                if dist < sw {
                    let depth = (sw - dist) as f64 * inv_sw;
                    self.damping[idx] = sigma_max * depth * depth;
                } else {
                    self.damping[idx] = 0.0;
                }
            }
        }
    }

    #[verified_engine::verified]
    fn recompute_medium_tables(&mut self) {
        let c0 = self.config.c0;
        let k0 = self.drude_omega_ref / c0;
        let mut max_v = c0;
        let mut neg_found = false;
        for idx in 0..self.refractive_index.len() {
            let raw_n = self.refractive_index[idx];
            let abs_n = raw_n.abs().max(MIN_ABS_INDEX);
            let sign_n = if raw_n < 0.0 { -1.0 } else { 1.0 };
            let clamped_n = sign_n * abs_n;
            self.refractive_index[idx] = clamped_n;
            let v = c0 / abs_n;
            if v > max_v {
                max_v = v;
            }
            self.wave_speed_sq[idx] = v * v;
            if clamped_n < 0.0 {
                neg_found = true;
                self.omega_p_sq[idx] = k0 * k0 * (1.0 - clamped_n);
            } else {
                self.omega_p_sq[idx] = 0.0;
            }
        }
        self.has_negative_index = neg_found;
        self.max_wave_speed = if neg_found { max_v.max(c0) } else { max_v };
    }

    #[verified_engine::verified]
    fn step_scalar_leapfrog(&mut self, dt: f64, source: Option<&WaveSource2D>) {
        let w = self.config.width;
        let h = self.config.height;
        let dt_sq = dt * dt;
        let src_val = source.map_or(0.0, |s| s.evaluate_at_time(self.time));
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = flatten_2d_index(x, y, w);
                if self.obstacle[idx] != 0 {
                    self.u_next[idx] = 0.0;
                    continue;
                }
                let curr = self.u_curr[idx];
                let lap = self.ops.central_diff_2nd_2d(
                    curr,
                    self.u_curr[idx - 1],
                    self.u_curr[idx + 1],
                    self.u_curr[idx - w],
                    self.u_curr[idx + w],
                );
                let s_term = Self::source_weight(source, x, y, src_val);
                let alpha = 0.5 * self.damping[idx] * dt;
                let numer = 2.0 * curr - (1.0 - alpha) * self.u_prev[idx]
                    + dt_sq * self.wave_speed_sq[idx] * (lap + s_term);
                self.u_next[idx] = numer / (1.0 + alpha);
            }
        }
        std::mem::swap(&mut self.u_prev, &mut self.u_curr);
        std::mem::swap(&mut self.u_curr, &mut self.u_next);
    }

    #[inline]
    #[verified_engine::verified]
    pub(super) fn source_weight(
        source: Option<&WaveSource2D>,
        x: usize,
        y: usize,
        src_val: f64,
    ) -> f64 {
        let Some(src) = source else {
            return 0.0;
        };
        match src.kind {
            WaveSourceKind::PlaneWave {
                x_index,
                y_min,
                y_max,
            } => {
                if x == x_index && y >= y_min && y <= y_max {
                    src_val
                } else {
                    0.0
                }
            }
            WaveSourceKind::PointSource { x: sx, y: sy } => {
                let dx = x as isize - sx as isize;
                let dy = y as isize - sy as isize;
                let r2 = dx * dx + dy * dy;
                if r2 == 0 {
                    src_val
                } else if r2 <= 2 {
                    src_val * (-0.75 * (r2 as f64)).exp()
                } else {
                    0.0
                }
            }
        }
    }

    #[verified_engine::verified]
    fn accumulate_irradiance(&mut self) {
        let beta = self.config.irradiance_decay.clamp(1e-4, 1.0);
        let keep = 1.0 - beta;
        for idx in 0..self.irradiance.len() {
            let u = self.u_curr[idx];
            self.irradiance[idx] = keep * self.irradiance[idx] + beta * (u * u);
        }
    }
}

#[path = "wave_drude.rs"]
mod drude;

#[cfg(test)]
#[path = "wave_2d_tests.rs"]
mod tests;

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{VrChaosConfig, VrSourceMode};
use super::MAX_CHAOS_LANDMARKS;
use crate::web_io::PlatformInstant;
use math_explorer::pure_math::statistics::tda::{
    farthest_point_landmarks_into, takens_delay_embedding_3d, Point3D,
};

const WARMUP_STEPS: usize = 600;
const TRACE_LEN: usize = 1600;

/// Live 3D Lorenz / Takens trajectory state and cached chaos invariants.
pub struct VrChaosState {
    /// Prandtl parameter `σ`.
    pub sigma: f64,
    /// Rayleigh parameter `ρ` (strange attractor emerges above Hopf bifurcation `ρ ≈ 24.74`).
    pub rho: f64,
    /// Geometric parameter `β`.
    pub beta: f64,
    /// Integration step `dt`.
    pub dt: f64,
    /// Number of farthest-point landmarks extracted (`4..=64`).
    pub landmark_count: usize,
    /// Takens embedding delay lag `τ` (`1..=50`).
    pub takens_lag: usize,
    /// Whether live trajectory streaming is actively advancing per frame.
    pub playing: bool,
    /// Monotonic sequence counter for external `AttractorRunner` live stream frames.
    pub live_stream_seq: u64,
    /// Cross-platform timestamp of the most recent live stream update (`wasm32`-safe).
    pub last_stream_tick: PlatformInstant,
    current_state: Point3D,
    trajectory_buf: Vec<Point3D>,
    scalar_trace: Vec<f64>,
    min_dist_sq: Vec<f64>,
    landmark_buf: Vec<Point3D>,
    #[cfg(feature = "physics")]
    lyapunov_key: Option<(u64, u64, u64, u64)>,
    cached_lyapunov: Option<f64>,
    cached_correlation_sum: Option<f64>,
    cached_correlation_dim: Option<f64>,
}

impl Default for VrChaosState {
    fn default() -> Self {
        let mut s = Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            dt: 0.01,
            landmark_count: MAX_CHAOS_LANDMARKS,
            takens_lag: 8,
            playing: false,
            live_stream_seq: 0,
            last_stream_tick: PlatformInstant::now(),
            current_state: Point3D::new(1.0, 1.0, 1.0),
            trajectory_buf: Vec::with_capacity(TRACE_LEN),
            scalar_trace: Vec::with_capacity(TRACE_LEN),
            min_dist_sq: vec![f64::INFINITY; TRACE_LEN],
            landmark_buf: vec![Point3D::new(0.0, 0.0, 0.0); MAX_CHAOS_LANDMARKS],
            #[cfg(feature = "physics")]
            lyapunov_key: None,
            cached_lyapunov: None,
            cached_correlation_sum: None,
            cached_correlation_dim: None,
        };
        s.regenerate_trajectory();
        s
    }
}

impl VrChaosState {
    /// Extracts serializable chaos configuration.
    pub fn to_config(&self) -> VrChaosConfig {
        VrChaosConfig {
            sigma: self.sigma,
            rho: self.rho,
            beta: self.beta,
            dt: self.dt,
            landmark_count: self.landmark_count,
            takens_lag: self.takens_lag,
        }
    }

    /// Applies validated configuration and regenerates the deterministic trajectory buffer.
    pub fn apply_config(&mut self, cfg: VrChaosConfig) {
        self.sigma = cfg.sigma;
        self.rho = cfg.rho;
        self.beta = cfg.beta;
        self.dt = cfg.dt;
        self.landmark_count = cfg.landmark_count.clamp(4, MAX_CHAOS_LANDMARKS);
        self.takens_lag = cfg.takens_lag.clamp(1, 50);
        self.regenerate_trajectory();
    }

    /// Checks whether all chaos parameters are finite.
    pub fn has_finite_parameters(&self) -> bool {
        self.sigma.is_finite()
            && self.rho.is_finite()
            && self.beta.is_finite()
            && self.dt.is_finite()
    }

    /// Returns cached largest Lyapunov exponent `λ₁` (available when `physics` feature is enabled).
    pub fn lyapunov_exponent(&self) -> Option<f64> {
        self.cached_lyapunov
    }

    /// Returns cached Grassberger-Procaccia correlation sum `C(ε)` at active filtration radius.
    pub fn correlation_sum(&self) -> Option<f64> {
        self.cached_correlation_sum
    }

    /// Returns cached log-log Grassberger-Procaccia correlation dimension estimate `D₂`.
    pub fn correlation_dimension_estimate(&self) -> Option<f64> {
        self.cached_correlation_dim
    }

    /// Re-integrates the Lorenz attractor from `(1, 1, 1)` for deterministic preset & slider sweeps.
    pub fn regenerate_trajectory(&mut self) {
        self.trajectory_buf.clear();
        self.scalar_trace.clear();
        let dt = self.dt.clamp(0.001, 0.05);
        let mut pt = Point3D::new(1.0, 1.0, 1.0);
        for _ in 0..WARMUP_STEPS {
            pt = step_lorenz_point(pt, dt, self.sigma, self.rho, self.beta);
            if !pt.is_finite() {
                pt = Point3D::new(1.0, 1.0, 1.0);
                break;
            }
        }
        for _ in 0..TRACE_LEN {
            pt = step_lorenz_point(pt, dt, self.sigma, self.rho, self.beta);
            if !pt.is_finite() {
                break;
            }
            self.trajectory_buf.push(pt);
            self.scalar_trace.push(pt.x);
        }
        if self.trajectory_buf.is_empty() {
            self.trajectory_buf.push(Point3D::new(1.0, 1.0, 1.0));
            self.scalar_trace.push(1.0);
        }
        self.current_state = *self.trajectory_buf.last().unwrap();
    }

    /// Advances the live trajectory by `steps` ODE ticks without heap allocations.
    pub fn advance_live_steps(&mut self, steps: usize) {
        self.last_stream_tick = PlatformInstant::now();
        let dt = self.dt.clamp(0.001, 0.05);
        for _ in 0..steps.clamp(1, 32) {
            let next = step_lorenz_point(self.current_state, dt, self.sigma, self.rho, self.beta);
            if !next.is_finite() {
                self.playing = false;
                break;
            }
            self.current_state = next;
            if self.trajectory_buf.len() >= TRACE_LEN {
                self.trajectory_buf.remove(0);
                self.scalar_trace.remove(0);
            }
            self.trajectory_buf.push(next);
            self.scalar_trace.push(next.x);
        }
    }

    /// Extracts up to `landmark_count` 3D points for either `Lorenz3D` or `Takens3D`.
    pub fn build_landmarks(&mut self, mode: VrSourceMode) -> Result<Vec<Point3D>, String> {
        let count = self.landmark_count.clamp(4, MAX_CHAOS_LANDMARKS);
        match mode {
            VrSourceMode::Takens3D => {
                let lag = self.takens_lag.clamp(1, 50);
                let embedded = takens_delay_embedding_3d(&self.scalar_trace, lag)
                    .map_err(|e| format!("Takens embedding error: {e:?}"))?;
                if self.min_dist_sq.len() < embedded.points.len() {
                    self.min_dist_sq
                        .resize(embedded.points.len(), f64::INFINITY);
                }
                let written = farthest_point_landmarks_into(
                    &embedded.points,
                    count,
                    &mut self.min_dist_sq,
                    &mut self.landmark_buf,
                )
                .map_err(|e| format!("Landmark error: {e:?}"))?;
                Ok(self.landmark_buf[..written].to_vec())
            }
            _ => {
                if self.min_dist_sq.len() < self.trajectory_buf.len() {
                    self.min_dist_sq
                        .resize(self.trajectory_buf.len(), f64::INFINITY);
                }
                let written = farthest_point_landmarks_into(
                    &self.trajectory_buf,
                    count,
                    &mut self.min_dist_sq,
                    &mut self.landmark_buf,
                )
                .map_err(|e| format!("Landmark error: {e:?}"))?;
                Ok(self.landmark_buf[..written].to_vec())
            }
        }
    }

    /// Refreshes `lorenz_lyapunov` and `correlation_dimension` invariants (gated on `physics`).
    pub fn refresh_invariants(&mut self, active_3d: &[Point3D], radius: f64) {
        #[cfg(feature = "physics")]
        {
            let key = (
                self.sigma.to_bits(),
                self.rho.to_bits(),
                self.beta.to_bits(),
                self.dt.to_bits(),
            );
            if self.lyapunov_key != Some(key) {
                self.cached_lyapunov =
                    compute_physics_lyapunov(self.sigma, self.rho, self.beta, self.dt);
                self.lyapunov_key = Some(key);
            }
            let (c_sum, d2) = compute_physics_correlation(active_3d, &self.trajectory_buf, radius);
            self.cached_correlation_sum = c_sum;
            self.cached_correlation_dim = d2;
        }
        #[cfg(not(feature = "physics"))]
        {
            let _ = (active_3d, radius);
            self.cached_lyapunov = None;
            self.cached_correlation_sum = None;
            self.cached_correlation_dim = None;
        }
    }
}

fn step_lorenz_point(s: Point3D, dt: f64, sigma: f64, rho: f64, beta: f64) -> Point3D {
    let deriv = |p: Point3D| {
        Point3D::new(
            sigma * (p.y - p.x),
            p.x * (rho - p.z) - p.y,
            p.x * p.y - beta * p.z,
        )
    };
    let add =
        |a: Point3D, b: Point3D, w: f64| Point3D::new(a.x + w * b.x, a.y + w * b.y, a.z + w * b.z);
    let k1 = deriv(s);
    let k2 = deriv(add(s, k1, 0.5 * dt));
    let k3 = deriv(add(s, k2, 0.5 * dt));
    let k4 = deriv(add(s, k3, dt));
    Point3D::new(
        s.x + (dt / 6.0) * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x),
        s.y + (dt / 6.0) * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y),
        s.z + (dt / 6.0) * (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z),
    )
}

#[cfg(feature = "physics")]
pub(crate) fn compute_physics_lyapunov(sigma: f64, rho: f64, beta: f64, dt: f64) -> Option<f64> {
    use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
    use math_explorer::physics::chaos::metrics::lorenz_lyapunov;
    use math_explorer::pure_math::analysis::ode::RungeKutta4;
    use nalgebra::Vector3;

    if !sigma.is_finite() || !rho.is_finite() || !beta.is_finite() || !dt.is_finite() {
        return None;
    }
    let step = dt.clamp(0.002, 0.02);
    let sys = LorenzBuilder::new()
        .sigma(sigma)
        .rho(rho)
        .beta(beta)
        .build(LorenzState::new(1.0, 1.0, 1.0));
    let init = Vector3::new(1.0, 1.0, 1.0);
    let mut rk4 = RungeKutta4::new(&init);
    lorenz_lyapunov(&sys, &mut rk4, init, step, 60, 0.25)
        .ok()
        .filter(|v| v.is_finite())
}

#[cfg(feature = "physics")]
pub(crate) fn compute_physics_correlation(
    landmarks: &[Point3D],
    dense_trace: &[Point3D],
    radius: f64,
) -> (Option<f64>, Option<f64>) {
    use math_explorer::physics::chaos::fractals::correlation_dimension;
    use nalgebra::Vector3;

    let source = if landmarks.len() >= 8 {
        landmarks
    } else {
        dense_trace
    };
    if source.len() < 2 {
        return (None, None);
    }
    let vecs: Vec<Vector3<f64>> = source
        .iter()
        .filter(|p| p.is_finite())
        .map(|p| Vector3::new(p.x, p.y, p.z))
        .collect();
    if vecs.len() < 2 {
        return (None, None);
    }
    let eps = radius.max(0.5);
    let c_eps = correlation_dimension(&vecs, eps);
    let dense_vecs: Vec<Vector3<f64>> = if dense_trace.len() >= 64 {
        dense_trace
            .iter()
            .step_by(2)
            .filter(|p| p.is_finite())
            .map(|p| Vector3::new(p.x, p.y, p.z))
            .collect()
    } else {
        vecs
    };
    let e1 = 2.5_f64;
    let e2 = 7.5_f64;
    let c1 = correlation_dimension(&dense_vecs, e1);
    let c2 = correlation_dimension(&dense_vecs, e2);
    let d2 = if c1 > 0.0 && c2 > c1 {
        let slope = (c2.ln() - c1.ln()) / (e2 / e1).ln();
        slope.is_finite().then_some(slope)
    } else {
        c_eps.is_finite().then_some(c_eps)
    };
    (c_eps.is_finite().then_some(c_eps), d2)
}

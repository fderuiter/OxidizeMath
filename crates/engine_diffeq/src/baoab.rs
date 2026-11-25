// [cite:engine_diffeq]
use crate::system::LangevinSystem;
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, Scalar, SolverWorkspace, fill_standard_normal};
use rand_core::RngCore;

/// Physical parameters for underdamped Langevin dynamics
/// $dq = M^{-1} p \, dt$, $dp = -\nabla V(q)\,dt - \gamma p \, dt + \sqrt{2 \gamma m k_B T}\,dW_t$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaoabParams<T: Scalar> {
    /// Particle mass $m > 0$.
    pub mass: T,
    /// Friction / damping rate $\gamma \ge 0$.
    pub friction: T,
    /// Thermal energy $k_B T \ge 0$.
    pub temperature: T,
    /// Integration step size $\Delta t > 0$.
    pub dt: T,
}

/// Symmetric Leimkuhler-Matthews `B-A-O-A-B` geometric Langevin integrator for
/// $dq = m^{-1} p\,dt$, $dp = -\nabla V(q)\,dt - \gamma p\,dt + \sqrt{2 \gamma m k_B T}\,dW_t$.
///
/// Splits kick (`B`), drift (`A`), and exact Ornstein-Uhlenbeck thermostat (`O`) substeps so the
/// deterministic limit ($\gamma = 0$) reduces to symplectic Velocity Verlet while harmonic potentials
/// preserve the exact configurational Gibbs-Boltzmann variance.
/// Reference: Leimkuhler & Matthews (2013), DOI: 10.1093/amrx/abs010;
/// [`docs/math_specs/engine_diffeq.md`](../../../docs/math_specs/engine_diffeq.md).
///
/// # Stability
/// Stable on harmonic oscillators of frequency $\omega_{\max}$ for $0 < \omega_{\max} \Delta t < 2$,
/// with unconditionally contractive thermostat decay $c_1 = e^{-\gamma \Delta t} \in (0, 1]$.
///
/// # Domain of Validity
/// Requires `mass > 0`, `friction >= 0`, `temperature >= 0`, `dt > 0`, `q.len() >= dim`,
/// `p.len() >= dim`, and `SolverWorkspace` length at least `2 * dim`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaoabLangevinSolver<T: Scalar> {
    params: BaoabParams<T>,
    half_dt: T,
    half_dt_inv_mass: T,
    ou_decay: T,
    ou_noise_scale: T,
}

impl<T: Scalar> BaoabLangevinSolver<T> {
    /// Constructs a precomputed [`BaoabLangevinSolver`] after validating physical parameter bounds.
    #[verified_engine::verified]
    pub fn try_new(params: BaoabParams<T>) -> Result<Self, MathError> {
        if !(params.mass > T::ZERO && params.mass.is_finite()) {
            return Err(MathError::StrictlyOutOfBounds {
                value: params.mass.to_f64(),
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        if !(params.friction >= T::ZERO && params.friction.is_finite()) {
            return Err(MathError::OutOfBounds {
                value: params.friction.to_f64(),
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        if !(params.temperature >= T::ZERO && params.temperature.is_finite()) {
            return Err(MathError::OutOfBounds {
                value: params.temperature.to_f64(),
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        if !(params.dt > T::ZERO && params.dt.is_finite()) {
            return Err(MathError::StrictlyOutOfBounds {
                value: params.dt.to_f64(),
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        let solver = Self::new(params);
        debug_assert!(solver.half_dt > T::ZERO);
        Ok(solver)
    }

    /// Constructs a precomputed [`BaoabLangevinSolver`] from [`BaoabParams`].
    ///
    /// # Panics
    /// Panics if `mass <= 0`, `friction < 0`, `temperature < 0`, or `dt <= 0`.
    #[verified_engine::verified]
    pub fn new(params: BaoabParams<T>) -> Self {
        assert!(params.mass > T::ZERO, "BAOAB mass must be positive");
        assert!(
            params.friction >= T::ZERO,
            "BAOAB friction must be non-negative"
        );
        assert!(
            params.temperature >= T::ZERO,
            "BAOAB temperature must be non-negative"
        );
        assert!(params.dt > T::ZERO, "BAOAB dt must be positive");

        let half_dt = T::HALF * params.dt;
        let half_dt_inv_mass = half_dt / params.mass;
        let ou_decay = (-(params.friction * params.dt)).exp();
        let one_minus_c1_sq = (T::ONE - ou_decay * ou_decay).max(T::ZERO);
        let ou_noise_scale = (params.mass * params.temperature * one_minus_c1_sq).sqrt();

        Self {
            params,
            half_dt,
            half_dt_inv_mass,
            ou_decay,
            ou_noise_scale,
        }
    }

    /// Returns the underlying [`BaoabParams`].
    #[inline]
    pub const fn params(&self) -> BaoabParams<T> {
        self.params
    }

    /// Returns the minimum [`SolverWorkspace`] length (`2 * dim`) required by [`Self::step`].
    #[inline]
    pub const fn scratch_len(dim: usize) -> usize {
        2 * dim
    }

    /// Advances `(q, p)` by one symmetric `B-A-O-A-B` step drawing Gaussian impulses from `rng`.
    #[verified_engine::verified]
    pub fn step<S: LangevinSystem<T> + ?Sized, R: RngCore + ?Sized>(
        &self,
        system: &S,
        q: &mut [T],
        p: &mut [T],
        workspace: &mut SolverWorkspace<'_, T>,
        rng: &mut R,
    ) {
        let d = system.dim();
        assert!(q.len() >= d && p.len() >= d);

        let mut ws = workspace.reborrow();
        let grad = ws.take_slice(d);
        let noise = ws.take_slice(d);

        // B: half-step momentum kick
        system.grad_potential(q, grad);
        for (pi, &gi) in p.iter_mut().take(d).zip(grad.iter()) {
            *pi -= self.half_dt * gi;
        }

        // A: half-step position drift
        for (qi, &pi) in q.iter_mut().take(d).zip(p.iter()) {
            *qi += self.half_dt_inv_mass * pi;
        }

        // O: exact Ornstein-Uhlenbeck momentum thermostat
        if self.ou_noise_scale > T::ZERO {
            fill_standard_normal(noise, rng);
            for (pi, &ni) in p.iter_mut().take(d).zip(noise.iter()) {
                *pi = self.ou_decay * *pi + self.ou_noise_scale * ni;
            }
        } else if self.params.friction > T::ZERO {
            for pi in p.iter_mut().take(d) {
                *pi *= self.ou_decay;
            }
        }

        // A: second half-step position drift
        for (qi, &pi) in q.iter_mut().take(d).zip(p.iter()) {
            *qi += self.half_dt_inv_mass * pi;
        }

        // B: second half-step momentum kick
        system.grad_potential(q, grad);
        for (pi, &gi) in p.iter_mut().take(d).zip(grad.iter()) {
            *pi -= self.half_dt * gi;
        }
    }

    /// Checked companion to [`Self::step`] validating `(q, p)` lengths and `workspace` capacity (`>= 2 * dim`).
    #[verified_engine::verified]
    pub fn try_step<S: LangevinSystem<T> + ?Sized, R: RngCore + ?Sized>(
        &self,
        system: &S,
        q: &mut [T],
        p: &mut [T],
        workspace: &mut SolverWorkspace<'_, T>,
        rng: &mut R,
    ) -> Result<(), MathError> {
        let d = system.dim();
        if q.len() != d {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(d),
                actual: Dimension(q.len()),
            });
        }
        if p.len() != d {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(d),
                actual: Dimension(p.len()),
            });
        }
        let required = d.checked_mul(2).ok_or(MathError::WorkspaceTooSmall {
            required: usize::MAX,
            available: workspace.len(),
        })?;
        if workspace.len() < required {
            return Err(MathError::WorkspaceTooSmall {
                required,
                available: workspace.len(),
            });
        }
        self.step(system, q, p, workspace, rng);
        debug_assert_eq!(q.len(), d);
        Ok(())
    }
}

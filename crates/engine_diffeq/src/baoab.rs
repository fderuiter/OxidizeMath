// [cite:engine_diffeq]
use crate::error::DiffeqError;
use crate::system::LangevinSystem;
use math_commons::math_kernel::types::{Scalar, SolverWorkspace, fill_standard_normal};
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

/// Symmetric Leimkuhler-Matthews `B-A-O-A-B` geometric Langevin integrator.
///
/// Splits kick (`B`), drift (`A`), and exact Ornstein-Uhlenbeck thermostat (`O`) substeps so the
/// deterministic limit ($\gamma = 0$) reduces to symplectic Velocity Verlet while harmonic potentials
/// preserve the exact configurational Gibbs-Boltzmann variance for any stable step size $\Delta t$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaoabLangevinSolver<T: Scalar> {
    params: BaoabParams<T>,
    half_dt: T,
    half_dt_inv_mass: T,
    ou_decay: T,
    ou_noise_scale: T,
}

impl<T: Scalar> BaoabLangevinSolver<T> {
    /// Fallibly constructs a precomputed [`BaoabLangevinSolver`] from [`BaoabParams`].
    ///
    /// # Errors
    /// Returns [`DiffeqError::InvalidParameter`] if `mass <= 0`, `friction < 0`,
    /// `temperature < 0`, `dt <= 0`, or any parameter is non-finite.
    #[verified_engine::verified]
    pub fn try_new(params: BaoabParams<T>) -> Result<Self, DiffeqError> {
        if !(params.mass > T::ZERO && params.mass.is_finite()) {
            return Err(DiffeqError::InvalidParameter { parameter: "mass" });
        }
        if !(params.friction >= T::ZERO && params.friction.is_finite()) {
            return Err(DiffeqError::InvalidParameter {
                parameter: "friction",
            });
        }
        if !(params.temperature >= T::ZERO && params.temperature.is_finite()) {
            return Err(DiffeqError::InvalidParameter {
                parameter: "temperature",
            });
        }
        if !(params.dt > T::ZERO && params.dt.is_finite()) {
            return Err(DiffeqError::InvalidParameter { parameter: "dt" });
        }

        let half_dt = T::HALF * params.dt;
        let half_dt_inv_mass = half_dt / params.mass;
        let ou_decay = (-(params.friction * params.dt)).exp();
        let one_minus_c1_sq = (T::ONE - ou_decay * ou_decay).max(T::ZERO);
        let ou_noise_scale = (params.mass * params.temperature * one_minus_c1_sq).sqrt();
        debug_assert!(half_dt > T::ZERO && half_dt_inv_mass > T::ZERO);

        Ok(Self {
            params,
            half_dt,
            half_dt_inv_mass,
            ou_decay,
            ou_noise_scale,
        })
    }

    /// Constructs a precomputed [`BaoabLangevinSolver`] from [`BaoabParams`].
    ///
    /// # Panics
    /// Panics if `mass <= 0`, `friction < 0`, `temperature < 0`, or `dt <= 0`;
    /// prefer [`Self::try_new`] for fallible callers.
    #[verified_engine::verified]
    pub fn new(params: BaoabParams<T>) -> Self {
        Self::try_new(params).expect("invalid BAOAB Langevin parameters")
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

    /// Fallibly advances `(q, p)` by one symmetric `B-A-O-A-B` step drawing Gaussian impulses from `rng`.
    ///
    /// # Errors
    /// Returns [`DiffeqError::BufferTooSmall`] if `q` or `p` has length `< system.dim()`, or
    /// [`DiffeqError::Math`] if `workspace` has fewer than `2 * system.dim()` scalars.
    #[verified_engine::verified]
    pub fn try_step<S: LangevinSystem<T> + ?Sized, R: RngCore + ?Sized>(
        &self,
        system: &S,
        q: &mut [T],
        p: &mut [T],
        workspace: &mut SolverWorkspace<'_, T>,
        rng: &mut R,
    ) -> Result<(), DiffeqError> {
        let d = system.dim();
        if q.len() < d {
            return Err(DiffeqError::BufferTooSmall {
                buffer: "q",
                required: d,
                actual: q.len(),
            });
        }
        if p.len() < d {
            return Err(DiffeqError::BufferTooSmall {
                buffer: "p",
                required: d,
                actual: p.len(),
            });
        }

        let mut ws = workspace.reborrow();
        let grad = ws.try_take_slice(d)?;
        let noise = ws.try_take_slice(d)?;
        debug_assert_eq!(grad.len(), d);

        system.grad_potential(q, grad);
        for (pi, &gi) in p.iter_mut().take(d).zip(grad.iter()) {
            *pi -= self.half_dt * gi;
        }

        for (qi, &pi) in q.iter_mut().take(d).zip(p.iter()) {
            *qi += self.half_dt_inv_mass * pi;
        }

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

        for (qi, &pi) in q.iter_mut().take(d).zip(p.iter()) {
            *qi += self.half_dt_inv_mass * pi;
        }

        system.grad_potential(q, grad);
        for (pi, &gi) in p.iter_mut().take(d).zip(grad.iter()) {
            *pi -= self.half_dt * gi;
        }
        Ok(())
    }

    /// Advances `(q, p)` by one symmetric `B-A-O-A-B` step drawing Gaussian impulses from `rng`.
    ///
    /// # Panics
    /// Panics if `q`, `p`, or `workspace` is undersized; prefer [`Self::try_step`] for fallible callers.
    #[verified_engine::verified]
    pub fn step<S: LangevinSystem<T> + ?Sized, R: RngCore + ?Sized>(
        &self,
        system: &S,
        q: &mut [T],
        p: &mut [T],
        workspace: &mut SolverWorkspace<'_, T>,
        rng: &mut R,
    ) {
        self.try_step(system, q, p, workspace, rng)
            .expect("BaoabLangevinSolver::step preconditions violated");
    }
}

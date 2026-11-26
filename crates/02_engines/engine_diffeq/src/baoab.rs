// [cite:engine_diffeq]
use crate::error::DiffeqError;
use crate::system::LangevinSystem;
use math_commons::math_kernel::types::{Scalar, SolverWorkspace, fill_standard_normal};
use rand_core::RngCore;

/// Physical parameters for underdamped Langevin dynamics
/// $dq = M^{-1} p \, dt$, $dp = -\nabla V(q)\,dt - \gamma p \, dt + \sqrt{2 \gamma m k_B T}\,dW_t$.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
/// [`docs/math_specs/engine_diffeq.md`](../../../../docs/math_specs/engine_diffeq.md).
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

    /// Constructs a precomputed [`BaoabLangevinSolver`] from [`BaoabParams`], returning a typed
    /// [`MathError`](math_commons::error::MathError) when any physical parameter is out of range.
    #[verified_engine::verified]
    pub fn try_new(params: BaoabParams<T>) -> Result<Self, math_commons::error::MathError> {
        use math_commons::error::MathError;
        if !params.mass.is_finite() || params.mass <= T::ZERO {
            return Err(MathError::StrictlyOutOfBounds {
                value: params.mass.to_f64(),
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        if !params.friction.is_finite() || params.friction < T::ZERO {
            return Err(MathError::OutOfBounds {
                value: params.friction.to_f64(),
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        if !params.temperature.is_finite() || params.temperature < T::ZERO {
            return Err(MathError::OutOfBounds {
                value: params.temperature.to_f64(),
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        if !params.dt.is_finite() || params.dt <= T::ZERO {
            return Err(MathError::StrictlyOutOfBounds {
                value: params.dt.to_f64(),
                min: 0.0,
                max: f64::INFINITY,
            });
        }

        let half_dt = T::HALF * params.dt;
        let half_dt_inv_mass = half_dt / params.mass;
        let ou_decay = (-(params.friction * params.dt)).exp();
        let one_minus_c1_sq = (T::ONE - ou_decay * ou_decay).max(T::ZERO);
        let ou_noise_scale = (params.mass * params.temperature * one_minus_c1_sq).sqrt();
        debug_assert!(ou_decay.is_finite() && ou_noise_scale >= T::ZERO);

        Ok(Self {
            params,
            half_dt,
            half_dt_inv_mass,
            ou_decay,
            ou_noise_scale,
        })
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

#[cfg(feature = "serde")]
impl<T: Scalar + serde::Serialize> serde::Serialize for BaoabLangevinSolver<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.params.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Scalar + serde::Deserialize<'de>> serde::Deserialize<'de> for BaoabLangevinSolver<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let params = BaoabParams::<T>::deserialize(deserializer)?;
        if !(params.mass.is_finite() && params.mass > T::ZERO) {
            return Err(serde::de::Error::custom(
                "BAOAB mass must be finite and positive",
            ));
        }
        if !(params.friction.is_finite() && params.friction >= T::ZERO) {
            return Err(serde::de::Error::custom(
                "BAOAB friction must be finite and non-negative",
            ));
        }
        if !(params.temperature.is_finite() && params.temperature >= T::ZERO) {
            return Err(serde::de::Error::custom(
                "BAOAB temperature must be finite and non-negative",
            ));
        }
        if !(params.dt.is_finite() && params.dt > T::ZERO) {
            return Err(serde::de::Error::custom(
                "BAOAB dt must be finite and positive",
            ));
        }
        Ok(Self::new(params))
    }
}

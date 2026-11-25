//! Continuum Kardar-Parisi-Zhang nonlinear stochastic surface growth
//! ($\partial_t h = \nu \nabla^2 h + \frac{\lambda}{2}|\nabla h|^2 + \eta$).
// [cite:stat_mech]

use crate::error::StatMechError;
use crate::physics::stat_mech::edwards_wilkinson::{interface_roughness, mean_field};
use engine_pde::{
    BoundaryCondition, ByteArena, ExplicitSpdeStencilStepper, SemiImplicitFourierStepper,
    SpdeNoiseKind, StencilOperators,
};
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, StepSize};
use rand_core::RngCore;

/// 1D Kardar-Parisi-Zhang (KPZ) nonlinear interface growth solver evolving
/// $\partial_t h = \nu \partial_{xx} h + \frac{\lambda}{2}(\partial_x h)^2 + \eta(x, t)$.
///
/// Reference: Kardar, Parisi, & Zhang (1986), *Phys. Rev. Lett.* 56(9), DOI: 10.1103/PhysRevLett.56.889;
/// [`docs/math_specs/domain_physics.md`](../../../../../docs/math_specs/domain_physics.md).
///
/// # Stability
/// [`Self::step_fourier_dealiased`] treats $+\nu k^2$ implicitly and applies Orszag 2/3-rule dealiasing
/// to $(\partial_x h)^2$; explicit [`Self::step_stencil`] requires $\nu \Delta t / \Delta x^2 \le \frac{1}{2}$
/// and sufficiently small $|\lambda| \Delta t / \Delta x^2$ to avoid Cole-Hopf gradient blow-up.
///
/// # Domain of Validity
/// Requires `n >= 4` (power-of-two with `BoundaryCondition::Periodic` for [`Self::step_fourier_dealiased`]),
/// `dx > 0`, `nu > 0`, finite `lambda`, `noise_intensity >= 0`, and `dt >= 0`.
#[derive(Debug, Clone)]
pub struct KpzModel1D {
    h: Vec<f64>,
    dx: f64,
    nu: f64,
    lambda: f64,
    noise_intensity: f64,
    boundary: BoundaryCondition,
    scratch: Vec<f64>,
}

impl KpzModel1D {
    /// Creates a validated 1D KPZ solver with pre-allocated zero-allocation workspace.
    #[verified_engine::verified]
    pub fn try_new(
        n: usize,
        dx: f64,
        nu: f64,
        lambda: f64,
        noise_intensity: f64,
        boundary: BoundaryCondition,
    ) -> Result<Self, StatMechError> {
        if n < 4 {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(4),
                actual: Dimension(n),
            }
            .into());
        }
        if !nu.is_finite() || nu <= 0.0 {
            return Err(MathError::StrictlyOutOfBounds {
                value: nu,
                min: 0.0,
                max: f64::INFINITY,
            }
            .into());
        }
        if !lambda.is_finite() || !noise_intensity.is_finite() || noise_intensity < 0.0 {
            return Err(MathError::OutOfBounds {
                value: noise_intensity,
                min: 0.0,
                max: f64::INFINITY,
            }
            .into());
        }
        let _ = StencilOperators::try_new_3d(StepSize(dx), StepSize(1.0), StepSize(1.0))?;
        let fourier_len = if n.is_power_of_two() {
            SemiImplicitFourierStepper::scratch_len_1d(n)
        } else {
            0
        };
        let cap = ExplicitSpdeStencilStepper::scratch_len_1d(n).max(fourier_len);
        Ok(Self {
            h: vec![0.0; n],
            dx,
            nu,
            lambda,
            noise_intensity,
            boundary,
            scratch: vec![0.0; cap],
        })
    }

    /// Creates a 1D KPZ solver, panicking on invalid grid or physical parameters.
    #[verified_engine::verified]
    pub fn new(
        n: usize,
        dx: f64,
        nu: f64,
        lambda: f64,
        noise_intensity: f64,
        boundary: BoundaryCondition,
    ) -> Self {
        let model = Self::try_new(n, dx, nu, lambda, noise_intensity, boundary)
            .expect("Valid KpzModel1D parameters");
        assert_eq!(model.h.len(), n);
        model
    }

    /// Immutable view of the 1D interface height $h(x, t)$.
    #[inline]
    pub fn field(&self) -> &[f64] {
        &self.h
    }

    /// Mutable view of the 1D interface height $h(x, t)$.
    #[inline]
    pub fn field_mut(&mut self) -> &mut [f64] {
        &mut self.h
    }

    /// Computes the spatial mean height $\bar{h}(t)$.
    #[inline]
    pub fn mean_height(&self) -> f64 {
        mean_field(&self.h)
    }

    /// Computes the Family-Vicsek RMS interface roughness $W(L, t)$.
    #[inline]
    pub fn interface_roughness(&self) -> f64 {
        interface_roughness(&self.h)
    }

    /// Computes the spatial mean squared slope $\langle (\partial_x h)^2 \rangle$, which governs
    /// the excess nonlinear KPZ drift velocity $\partial_t \bar{h} = \frac{\lambda}{2}\langle (\partial_x h)^2 \rangle$.
    #[verified_engine::verified]
    pub fn mean_slope_squared(&self) -> f64 {
        let n = self.h.len();
        assert!(n >= 2);
        let ops = StencilOperators::new(StepSize(self.dx));
        let sum_sq: f64 = (0..n)
            .map(|x| ops.grad_sq_1d_at(&self.h, n, x, self.boundary))
            .sum();
        sum_sq / (n as f64)
    }

    /// Advances one explicit finite-difference step without heap allocations.
    #[verified_engine::verified]
    pub fn step_stencil<R: RngCore + ?Sized>(&mut self, dt: f64, rng: &mut R) {
        assert!(dt >= 0.0);
        let ops = StencilOperators::new(StepSize(self.dx));
        let stepper = ExplicitSpdeStencilStepper::new(
            ops,
            self.boundary,
            SpdeNoiseKind::Additive {
                intensity: self.noise_intensity,
            },
        );
        let nu = self.nu;
        let half_lambda = 0.5 * self.lambda;
        let mut ws = ByteArena::from_slice(&mut self.scratch);
        stepper.step_1d(
            &mut self.h,
            dt,
            |_x, _u, grad, lap| nu * lap + half_lambda * grad * grad,
            &mut ws,
            rng,
        );
    }

    /// Advances one radix-2 2/3-rule dealiased semi-implicit Fourier step without heap allocations.
    #[verified_engine::verified]
    pub fn step_fourier_dealiased<R: RngCore + ?Sized>(&mut self, dt: f64, rng: &mut R) {
        assert!(
            self.boundary == BoundaryCondition::Periodic
                && self.h.len().is_power_of_two()
                && dt >= 0.0
        );
        let stepper = SemiImplicitFourierStepper::new_1d(self.dx);
        let mut ws = ByteArena::from_slice(&mut self.scratch);
        stepper.step_kpz_1d(
            &mut self.h,
            self.nu,
            self.lambda,
            self.noise_intensity,
            dt,
            &mut ws,
            rng,
        );
    }
}

/// 2D Kardar-Parisi-Zhang nonlinear surface growth solver evolving
/// $\partial_t h = \nu \nabla^2 h + \frac{\lambda}{2}|\nabla h|^2 + \eta(x, y, t)$.
///
/// Reference: Kardar, Parisi, & Zhang (1986), *Phys. Rev. Lett.* 56(9), DOI: 10.1103/PhysRevLett.56.889;
/// [`docs/math_specs/domain_physics.md`](../../../../../docs/math_specs/domain_physics.md).
///
/// # Stability
/// [`Self::step_fourier_dealiased`] treats $+\nu |\mathbf{k}|^2$ implicitly with Orszag 2/3-rule dealiasing
/// on $|\nabla h|^2$, while explicit [`Self::step_stencil`] requires $\nu \Delta t(\Delta x^{-2} + \Delta y^{-2}) \le \frac{1}{2}$
/// and bounded nonlinear slope Courant numbers.
///
/// # Domain of Validity
/// Requires `width >= 4`, `height >= 4` (power-of-two with `BoundaryCondition::Periodic` for
/// [`Self::step_fourier_dealiased`]), `dx, dy > 0`, `nu > 0`, finite `lambda`, `noise_intensity >= 0`, and `dt >= 0`.
#[derive(Debug, Clone)]
pub struct KpzModel2D {
    h: Vec<f64>,
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    nu: f64,
    lambda: f64,
    noise_intensity: f64,
    boundary: BoundaryCondition,
    scratch: Vec<f64>,
}

impl KpzModel2D {
    /// Creates a 2D KPZ solver with pre-allocated zero-allocation workspace.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn new(
        width: usize,
        height: usize,
        dx: f64,
        dy: f64,
        nu: f64,
        lambda: f64,
        noise_intensity: f64,
        boundary: BoundaryCondition,
    ) -> Self {
        assert!(width >= 4 && height >= 4 && dx > 0.0 && dy > 0.0 && nu > 0.0);
        assert!(lambda.is_finite() && noise_intensity >= 0.0);
        let fourier_len = if width.is_power_of_two() && height.is_power_of_two() {
            SemiImplicitFourierStepper::scratch_len_2d(width, height)
        } else {
            0
        };
        let cap = ExplicitSpdeStencilStepper::scratch_len_2d(width, height).max(fourier_len);
        Self {
            h: vec![0.0; width * height],
            width,
            height,
            dx,
            dy,
            nu,
            lambda,
            noise_intensity,
            boundary,
            scratch: vec![0.0; cap],
        }
    }

    /// Immutable view of the 2D height surface $h(x, y, t)$.
    #[inline]
    pub fn field(&self) -> &[f64] {
        &self.h
    }

    /// Mutable view of the 2D height surface $h(x, y, t)$.
    #[inline]
    pub fn field_mut(&mut self) -> &mut [f64] {
        &mut self.h
    }

    /// Computes the spatial mean height $\bar{h}(t)$.
    #[inline]
    pub fn mean_height(&self) -> f64 {
        mean_field(&self.h)
    }

    /// Computes the Family-Vicsek RMS surface roughness $W(L_x, L_y, t)$.
    #[inline]
    pub fn interface_roughness(&self) -> f64 {
        interface_roughness(&self.h)
    }

    /// Advances one explicit 2D finite-difference KPZ step without heap allocations.
    #[verified_engine::verified]
    pub fn step_stencil<R: RngCore + ?Sized>(&mut self, dt: f64, rng: &mut R) {
        assert!(dt >= 0.0);
        let ops = StencilOperators::new_2d(StepSize(self.dx), StepSize(self.dy));
        let stepper = ExplicitSpdeStencilStepper::new(
            ops,
            self.boundary,
            SpdeNoiseKind::Additive {
                intensity: self.noise_intensity,
            },
        );
        let nu = self.nu;
        let half_lambda = 0.5 * self.lambda;
        let mut ws = ByteArena::from_slice(&mut self.scratch);
        stepper.step_2d(
            &mut self.h,
            self.width,
            self.height,
            dt,
            |_x, _y, _u, (gx, gy), lap| nu * lap + half_lambda * (gx * gx + gy * gy),
            &mut ws,
            rng,
        );
    }

    /// Advances one 2D radix-2 2/3-rule dealiased semi-implicit Fourier KPZ step without heap allocations.
    #[verified_engine::verified]
    pub fn step_fourier_dealiased<R: RngCore + ?Sized>(&mut self, dt: f64, rng: &mut R) {
        assert!(
            self.boundary == BoundaryCondition::Periodic
                && self.width.is_power_of_two()
                && self.height.is_power_of_two()
                && dt >= 0.0
        );
        let stepper = SemiImplicitFourierStepper::new_2d(self.dx, self.dy);
        let mut ws = ByteArena::from_slice(&mut self.scratch);
        stepper.step_kpz_2d(
            &mut self.h,
            self.width,
            self.height,
            self.nu,
            self.lambda,
            self.noise_intensity,
            dt,
            &mut ws,
            rng,
        );
    }
}

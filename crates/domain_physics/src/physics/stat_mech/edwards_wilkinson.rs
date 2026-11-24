//! Continuum Edwards-Wilkinson linear stochastic surface growth ($\partial_t h = \nu \nabla^2 h + \eta$).
// [cite:stat_mech]

use crate::error::StatMechError;
use engine_pde::{
    BoundaryCondition, ExplicitSpdeStencilStepper, SemiImplicitFourierStepper, SpdeNoiseKind,
    StencilOperators,
};
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, SolverWorkspace, StepSize};
use rand_core::RngCore;

/// Computes the spatial arithmetic mean $\bar{u} = \frac{1}{N} \sum_{i=0}^{N-1} u_i$ of a continuum field.
#[verified_engine::verified]
pub fn mean_field(u: &[f64]) -> f64 {
    if u.is_empty() {
        return 0.0;
    }
    debug_assert!(!u.is_empty());
    u.iter().copied().sum::<f64>() / (u.len() as f64)
}

/// Computes the Family-Vicsek RMS interface width (roughness)
/// $W(L, t) = \sqrt{\frac{1}{N} \sum_{i=0}^{N-1} (h_i - \bar{h})^2}$.
#[verified_engine::verified]
pub fn interface_roughness(h: &[f64]) -> f64 {
    if h.is_empty() {
        return 0.0;
    }
    debug_assert!(!h.is_empty());
    let mean = mean_field(h);
    let var = h
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / (h.len() as f64);
    var.max(0.0).sqrt()
}

/// Exact continuum steady-state squared roughness $\lim_{t \to \infty} W^2(L, t) = \frac{D L}{12 \nu}$
/// for a 1D periodic Edwards-Wilkinson interface of length $L$.
#[verified_engine::verified]
pub fn theoretical_steady_state_roughness_sq_1d(
    system_length: f64,
    nu: f64,
    noise_intensity: f64,
) -> f64 {
    assert!(system_length > 0.0 && nu > 0.0 && noise_intensity >= 0.0);
    (noise_intensity * system_length) / (12.0 * nu)
}

/// 1D Edwards-Wilkinson linear stochastic interface model $\partial_t h = \nu \partial_{xx} h + \eta$.
///
/// Reference: Edwards & Wilkinson (1982), *Proc. R. Soc. Lond. A* 381, DOI: 10.1098/rspa.1982.0056;
/// [`docs/math_specs/domain_physics.md`](../../../../../docs/math_specs/domain_physics.md).
///
/// # Stability
/// [`Self::step_fourier`] damps linear modes via $(1 + \nu k^2 \Delta t)^{-1}$ unconditionally in `dt`,
/// whereas explicit [`Self::step_stencil`] requires $\nu \Delta t / \Delta x^2 \le \frac{1}{2}$.
///
/// # Domain of Validity
/// Requires `n >= 4` (power-of-two with `BoundaryCondition::Periodic` for [`Self::step_fourier`]),
/// `dx > 0`, `nu > 0`, `noise_intensity >= 0`, and `dt >= 0`.
#[derive(Debug, Clone)]
pub struct EdwardsWilkinson1D {
    h: Vec<f64>,
    dx: f64,
    nu: f64,
    noise_intensity: f64,
    boundary: BoundaryCondition,
    scratch: Vec<f64>,
}

impl EdwardsWilkinson1D {
    /// Creates a validated 1D Edwards-Wilkinson solver with pre-allocated zero-allocation workspace.
    #[verified_engine::verified]
    pub fn try_new(
        n: usize,
        dx: f64,
        nu: f64,
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
        if !noise_intensity.is_finite() || noise_intensity < 0.0 {
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
            noise_intensity,
            boundary,
            scratch: vec![0.0; cap],
        })
    }

    /// Creates a 1D Edwards-Wilkinson solver, panicking on invalid grid or physical parameters.
    #[verified_engine::verified]
    pub fn new(
        n: usize,
        dx: f64,
        nu: f64,
        noise_intensity: f64,
        boundary: BoundaryCondition,
    ) -> Self {
        let model = Self::try_new(n, dx, nu, noise_intensity, boundary)
            .expect("Valid EdwardsWilkinson1D parameters");
        assert_eq!(model.h.len(), n);
        model
    }

    /// Immutable view of the 1D height profile $h(x, t)$.
    #[inline]
    pub fn field(&self) -> &[f64] {
        &self.h
    }

    /// Mutable view of the 1D height profile $h(x, t)$.
    #[inline]
    pub fn field_mut(&mut self) -> &mut [f64] {
        &mut self.h
    }

    /// Computes the spatial mean height $\bar{h}(t)$.
    #[inline]
    pub fn mean_height(&self) -> f64 {
        mean_field(&self.h)
    }

    /// Computes the RMS Family-Vicsek interface width $W(L, t)$.
    #[inline]
    pub fn interface_roughness(&self) -> f64 {
        interface_roughness(&self.h)
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
        let mut ws = SolverWorkspace::new(&mut self.scratch);
        stepper.step_1d(&mut self.h, dt, |_x, _u, _grad, lap| nu * lap, &mut ws, rng);
    }

    /// Advances one radix-2 semi-implicit periodic Fourier step without heap allocations.
    #[verified_engine::verified]
    pub fn step_fourier<R: RngCore + ?Sized>(&mut self, dt: f64, rng: &mut R) {
        assert!(
            self.boundary == BoundaryCondition::Periodic
                && self.h.len().is_power_of_two()
                && dt >= 0.0
        );
        let stepper = SemiImplicitFourierStepper::new_1d(self.dx);
        let mut ws = SolverWorkspace::new(&mut self.scratch);
        stepper.step_edwards_wilkinson_1d(
            &mut self.h,
            self.nu,
            self.noise_intensity,
            dt,
            &mut ws,
            rng,
        );
    }
}

/// 2D Edwards-Wilkinson linear stochastic surface model $\partial_t h = \nu \nabla^2 h + \eta$.
///
/// Reference: Edwards & Wilkinson (1982), *Proc. R. Soc. Lond. A* 381, DOI: 10.1098/rspa.1982.0056;
/// [`docs/math_specs/domain_physics.md`](../../../../../docs/math_specs/domain_physics.md).
///
/// # Stability
/// [`Self::step_fourier`] damps modes implicitly via $(1 + \nu |\mathbf{k}|^2 \Delta t)^{-1}$, whereas
/// explicit [`Self::step_stencil`] requires $\nu \Delta t(\Delta x^{-2} + \Delta y^{-2}) \le \frac{1}{2}$.
///
/// # Domain of Validity
/// Requires `width >= 4`, `height >= 4` (power-of-two with `BoundaryCondition::Periodic` for [`Self::step_fourier`]),
/// `dx, dy > 0`, `nu > 0`, `noise_intensity >= 0`, and `dt >= 0`.
#[derive(Debug, Clone)]
pub struct EdwardsWilkinson2D {
    h: Vec<f64>,
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    nu: f64,
    noise_intensity: f64,
    boundary: BoundaryCondition,
    scratch: Vec<f64>,
}

impl EdwardsWilkinson2D {
    /// Creates a 2D Edwards-Wilkinson solver with pre-allocated zero-allocation workspace.
    #[verified_engine::verified]
    pub fn new(
        width: usize,
        height: usize,
        dx: f64,
        dy: f64,
        nu: f64,
        noise_intensity: f64,
        boundary: BoundaryCondition,
    ) -> Self {
        assert!(width >= 4 && height >= 4 && dx > 0.0 && dy > 0.0 && nu > 0.0);
        assert!(noise_intensity >= 0.0);
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
            noise_intensity,
            boundary,
            scratch: vec![0.0; cap],
        }
    }

    /// Immutable view of the 2D height field $h(x, y, t)$.
    #[inline]
    pub fn field(&self) -> &[f64] {
        &self.h
    }

    /// Computes the RMS surface roughness $W(L_x, L_y, t)$.
    #[inline]
    pub fn interface_roughness(&self) -> f64 {
        interface_roughness(&self.h)
    }

    /// Advances one explicit 2D finite-difference step without heap allocations.
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
        let mut ws = SolverWorkspace::new(&mut self.scratch);
        stepper.step_2d(
            &mut self.h,
            self.width,
            self.height,
            dt,
            |_x, _y, _u, _grad, lap| nu * lap,
            &mut ws,
            rng,
        );
    }

    /// Advances one 2D radix-2 semi-implicit Fourier step without heap allocations.
    #[verified_engine::verified]
    pub fn step_fourier<R: RngCore + ?Sized>(&mut self, dt: f64, rng: &mut R) {
        assert!(
            self.boundary == BoundaryCondition::Periodic
                && self.width.is_power_of_two()
                && self.height.is_power_of_two()
                && dt >= 0.0
        );
        let stepper = SemiImplicitFourierStepper::new_2d(self.dx, self.dy);
        let mut ws = SolverWorkspace::new(&mut self.scratch);
        stepper.step_edwards_wilkinson_2d(
            &mut self.h,
            self.width,
            self.height,
            self.nu,
            self.noise_intensity,
            dt,
            &mut ws,
            rng,
        );
    }
}

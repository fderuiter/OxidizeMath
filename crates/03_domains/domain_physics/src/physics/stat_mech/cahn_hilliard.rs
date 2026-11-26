//! Hohenberg-Halperin Model B (Cahn-Hilliard-Cook) conserved spinodal decomposition SPDE
//! ($\partial_t \phi = M \nabla^2(\phi^3 - \phi - \kappa \nabla^2 \phi) + \nabla \cdot \vec{\eta}$).
// [cite:stat_mech]

use crate::error::StatMechError;
use crate::physics::stat_mech::allen_cahn::ginzburg_landau_free_energy_2d;
use crate::physics::stat_mech::edwards_wilkinson::mean_field;
use engine_pde::{
    BoundaryCondition, ExplicitSpdeStencilStepper, PdeError, SemiImplicitFourierStepper,
    SpdeNoiseKind, StencilOperators,
};
use math_commons::error::MathError;
use math_commons::math_kernel::types::{
    Dimension, SolverWorkspace, StepSize, fill_standard_normal,
};
use rand_core::RngCore;

/// 2D conserved Cahn-Hilliard-Cook (Hohenberg-Halperin Model B) phase-separation SPDE solver evolving
/// $\partial_t \phi = M \nabla^2(\phi^3 - \phi - \kappa \nabla^2 \phi) + \nabla \cdot \vec{\eta}$ via
/// 2D Biharmonic finite-difference stencils or radix-2 2/3-dealiased semi-implicit Fourier stepping.
///
/// Reference: Cahn & Hilliard (1958), DOI: 10.1063/1.1744102; Hohenberg & Halperin (1977),
/// DOI: 10.1103/RevModPhys.49.435; [`docs/math_specs/domain_physics.md`](../../../../../../docs/math_specs/domain_physics.md).
///
/// # Stability
/// Conserves spatial mean composition $\bar{\phi}$ to machine precision; [`Self::step_fourier_dealiased`]
/// treats $+M\kappa|\mathbf{k}|^4$ implicitly with Orszag 2/3 dealiasing, while [`Self::step_stencil`]
/// requires $M\kappa\,\Delta t(\Delta x^{-2} + \Delta y^{-2})^2 \le \frac{1}{8}$.
///
/// # Domain of Validity
/// Requires `width >= 4`, `height >= 4` (power-of-two with `BoundaryCondition::Periodic` for Fourier stepping),
/// `dx, dy > 0`, `mobility > 0`, `kappa > 0`, and `mobility_temp >= 0`.
#[derive(Debug, Clone)]
pub struct CahnHilliard2D {
    phi: Vec<f64>,
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    mobility: f64,
    kappa: f64,
    mobility_temp: f64,
    boundary: BoundaryCondition,
    scratch: Vec<f64>,
}

impl CahnHilliard2D {
    /// Creates a validated 2D Cahn-Hilliard solver with pre-allocated zero-allocation workspace.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_new(
        width: usize,
        height: usize,
        dx: f64,
        dy: f64,
        mobility: f64,
        kappa: f64,
        mobility_temp: f64,
        boundary: BoundaryCondition,
    ) -> Result<Self, StatMechError> {
        if width < 4 || height < 4 {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(4),
                actual: Dimension(width.min(height)),
            }
            .into());
        }
        if !mobility.is_finite() || mobility <= 0.0 || !kappa.is_finite() || kappa <= 0.0 {
            return Err(MathError::StrictlyOutOfBounds {
                value: mobility.min(kappa),
                min: 0.0,
                max: f64::INFINITY,
            }
            .into());
        }
        if !mobility_temp.is_finite() || mobility_temp < 0.0 {
            return Err(MathError::OutOfBounds {
                value: mobility_temp,
                min: 0.0,
                max: f64::INFINITY,
            }
            .into());
        }
        let _ = StencilOperators::try_new_3d(StepSize(dx), StepSize(dy), StepSize(1.0))?;
        let fourier_len = if width.is_power_of_two() && height.is_power_of_two() {
            SemiImplicitFourierStepper::scratch_len_2d(width, height)
        } else {
            0
        };
        let cap = ExplicitSpdeStencilStepper::scratch_len_2d(width, height).max(fourier_len);
        Ok(Self {
            phi: vec![0.0; width * height],
            width,
            height,
            dx,
            dy,
            mobility,
            kappa,
            mobility_temp,
            boundary,
            scratch: vec![0.0; cap],
        })
    }

    /// Creates a 2D Cahn-Hilliard solver, panicking on invalid grid or physical parameters.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn new(
        width: usize,
        height: usize,
        dx: f64,
        dy: f64,
        mobility: f64,
        kappa: f64,
        mobility_temp: f64,
        boundary: BoundaryCondition,
    ) -> Self {
        let model = Self::try_new(
            width,
            height,
            dx,
            dy,
            mobility,
            kappa,
            mobility_temp,
            boundary,
        )
        .expect("Valid CahnHilliard2D parameters");
        assert_eq!(model.phi.len(), width * height);
        model
    }

    /// Fallibly initializes `phi` around an exact conserved spatial mean `mean_phi` with zero-mean
    /// Gaussian spinodal fluctuations of standard deviation `amplitude` drawn from `rng`.
    #[verified_engine::verified]
    pub fn try_init_spinodal_perturbation<R: RngCore + ?Sized>(
        &mut self,
        mean_phi: f64,
        amplitude: f64,
        rng: &mut R,
    ) -> Result<(), StatMechError> {
        if !mean_phi.is_finite() {
            return Err(MathError::OutOfBounds {
                value: mean_phi,
                min: f64::NEG_INFINITY,
                max: f64::INFINITY,
            }
            .into());
        }
        if !amplitude.is_finite() || amplitude < 0.0 {
            return Err(MathError::OutOfBounds {
                value: amplitude,
                min: 0.0,
                max: f64::INFINITY,
            }
            .into());
        }
        let total = self.phi.len();
        let noise = &mut self.scratch[..total];
        fill_standard_normal(noise, rng);
        let sample_mean = mean_field(noise);
        for (dst, &z) in self.phi.iter_mut().zip(noise.iter()) {
            *dst = mean_phi + amplitude * (z - sample_mean);
        }
        debug_assert_eq!(self.phi.len(), total);
        debug_assert!(amplitude >= 0.0);
        Ok(())
    }

    /// Initializes `phi` around an exact conserved spatial mean `mean_phi` with zero-mean Gaussian
    /// spinodal fluctuations of standard deviation `amplitude` drawn from `rng`.
    #[verified_engine::verified]
    pub fn init_spinodal_perturbation<R: RngCore + ?Sized>(
        &mut self,
        mean_phi: f64,
        amplitude: f64,
        rng: &mut R,
    ) {
        self.try_init_spinodal_perturbation(mean_phi, amplitude, rng)
            .expect("Valid CahnHilliard2D::init_spinodal_perturbation parameters");
    }

    /// Immutable view of the conserved composition field $\phi(x, y, t)$.
    #[inline]
    pub fn field(&self) -> &[f64] {
        &self.phi
    }

    /// Mutable view of the conserved composition field $\phi(x, y, t)$.
    #[inline]
    pub fn field_mut(&mut self) -> &mut [f64] {
        &mut self.phi
    }

    /// Computes the conserved spatial mean $\bar{\phi}(t) = \frac{1}{N}\sum_{x, y} \phi(x, y, t)$.
    #[inline]
    pub fn spatial_mean(&self) -> f64 {
        mean_field(&self.phi)
    }

    /// Evaluates the Ginzburg-Landau free energy $F[\phi]$.
    #[inline]
    pub fn ginzburg_landau_energy(&self) -> f64 {
        ginzburg_landau_free_energy_2d(
            &self.phi,
            self.width,
            self.height,
            self.dx,
            self.dy,
            self.kappa,
            self.boundary,
        )
    }

    /// Fallibly advances one conserved 2D Biharmonic finite-difference step without heap allocations.
    #[verified_engine::verified]
    pub fn try_step_stencil<R: RngCore + ?Sized>(
        &mut self,
        dt: f64,
        rng: &mut R,
    ) -> Result<(), StatMechError> {
        let ops = StencilOperators::try_new_2d(StepSize(self.dx), StepSize(self.dy))?;
        let stepper = ExplicitSpdeStencilStepper::new(
            ops,
            self.boundary,
            SpdeNoiseKind::ConservedFlux {
                mobility_temp: self.mobility_temp,
            },
        );
        let mut ws = SolverWorkspace::new(&mut self.scratch);
        stepper.try_step_cahn_hilliard_2d(
            &mut self.phi,
            self.width,
            self.height,
            self.mobility,
            self.kappa,
            dt,
            &mut ws,
            rng,
        )?;
        debug_assert_eq!(self.phi.len(), self.width * self.height);
        Ok(())
    }

    /// Advances one conserved 2D Biharmonic finite-difference step (`eval_biharmonic_2d_from_laplacian`)
    /// with staggered-face conserved-flux noise without heap allocations.
    #[verified_engine::verified]
    pub fn step_stencil<R: RngCore + ?Sized>(&mut self, dt: f64, rng: &mut R) {
        self.try_step_stencil(dt, rng)
            .expect("Valid CahnHilliard2D::step_stencil parameters");
    }

    /// Fallibly advances one periodic radix-2 semi-implicit Fourier step with Orszag 2/3-rule
    /// dealiasing and conserved-flux noise without heap allocations.
    #[verified_engine::verified]
    pub fn try_step_fourier_dealiased<R: RngCore + ?Sized>(
        &mut self,
        dt: f64,
        rng: &mut R,
    ) -> Result<(), StatMechError> {
        if self.boundary != BoundaryCondition::Periodic {
            return Err(PdeError::InvalidParameter {
                parameter: "boundary (Fourier step requires Periodic)",
            }
            .into());
        }
        let stepper = SemiImplicitFourierStepper::try_new_2d(self.dx, self.dy)?;
        let mut ws = SolverWorkspace::new(&mut self.scratch);
        stepper.try_step_cahn_hilliard_2d(
            &mut self.phi,
            self.width,
            self.height,
            self.mobility,
            self.kappa,
            self.mobility_temp,
            dt,
            &mut ws,
            rng,
        )?;
        debug_assert_eq!(self.boundary, BoundaryCondition::Periodic);
        debug_assert_eq!(self.phi.len(), self.width * self.height);
        Ok(())
    }

    /// Advances one periodic radix-2 semi-implicit Fourier step with Orszag 2/3-rule dealiasing
    /// and conserved-flux noise without heap allocations.
    #[verified_engine::verified]
    pub fn step_fourier_dealiased<R: RngCore + ?Sized>(&mut self, dt: f64, rng: &mut R) {
        self.try_step_fourier_dealiased(dt, rng)
            .expect("Valid CahnHilliard2D::step_fourier_dealiased parameters");
    }
}

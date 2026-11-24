//! Hohenberg-Halperin Model B (Cahn-Hilliard-Cook) conserved spinodal decomposition SPDE
//! ($\partial_t \phi = M \nabla^2(\phi^3 - \phi - \kappa \nabla^2 \phi) + \nabla \cdot \vec{\eta}$).
// [cite:stat_mech]

use crate::error::StatMechError;
use crate::physics::stat_mech::allen_cahn::ginzburg_landau_free_energy_2d;
use crate::physics::stat_mech::edwards_wilkinson::mean_field;
use engine_pde::{
    BoundaryCondition, ExplicitSpdeStencilStepper, SemiImplicitFourierStepper, SpdeNoiseKind,
    StencilOperators,
};
use math_commons::error::MathError;
use math_commons::math_kernel::types::{
    Dimension, SolverWorkspace, StepSize, fill_standard_normal,
};
use rand_core::RngCore;

/// 2D conserved Cahn-Hilliard-Cook (Model B) phase-separation SPDE solver supporting both
/// 2D Biharmonic finite-difference stencils and radix-2 2/3-dealiased semi-implicit Fourier stepping.
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
    #[verified_engine::verified(alloc_ok = "constructor pre-allocates field and SPDE scratch")]
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

    /// Initializes `phi` around an exact conserved spatial mean `mean_phi` with zero-mean Gaussian
    /// spinodal fluctuations of standard deviation `amplitude` drawn from `rng`.
    #[verified_engine::verified]
    pub fn init_spinodal_perturbation<R: RngCore + ?Sized>(
        &mut self,
        mean_phi: f64,
        amplitude: f64,
        rng: &mut R,
    ) {
        assert!(mean_phi.is_finite() && amplitude >= 0.0);
        let total = self.phi.len();
        let noise = &mut self.scratch[..total];
        fill_standard_normal(noise, rng);
        let sample_mean = mean_field(noise);
        for (dst, &z) in self.phi.iter_mut().zip(noise.iter()) {
            *dst = mean_phi + amplitude * (z - sample_mean);
        }
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

    /// Advances one conserved 2D Biharmonic finite-difference step (`eval_biharmonic_2d_from_laplacian`)
    /// with staggered-face conserved-flux noise without heap allocations.
    #[verified_engine::verified]
    pub fn step_stencil<R: RngCore + ?Sized>(&mut self, dt: f64, rng: &mut R) {
        assert!(dt >= 0.0);
        let ops = StencilOperators::new_2d(StepSize(self.dx), StepSize(self.dy));
        let stepper = ExplicitSpdeStencilStepper::new(
            ops,
            self.boundary,
            SpdeNoiseKind::ConservedFlux {
                mobility_temp: self.mobility_temp,
            },
        );
        let mut ws = SolverWorkspace::new(&mut self.scratch);
        stepper.step_cahn_hilliard_2d(
            &mut self.phi,
            self.width,
            self.height,
            self.mobility,
            self.kappa,
            dt,
            &mut ws,
            rng,
        );
    }

    /// Advances one periodic radix-2 semi-implicit Fourier step with Orszag 2/3-rule dealiasing
    /// and conserved-flux noise without heap allocations.
    #[verified_engine::verified]
    pub fn step_fourier_dealiased<R: RngCore + ?Sized>(&mut self, dt: f64, rng: &mut R) {
        assert!(
            self.boundary == BoundaryCondition::Periodic
                && self.width.is_power_of_two()
                && self.height.is_power_of_two()
                && dt >= 0.0
        );
        let stepper = SemiImplicitFourierStepper::new_2d(self.dx, self.dy);
        let mut ws = SolverWorkspace::new(&mut self.scratch);
        stepper.step_cahn_hilliard_2d(
            &mut self.phi,
            self.width,
            self.height,
            self.mobility,
            self.kappa,
            self.mobility_temp,
            dt,
            &mut ws,
            rng,
        );
    }
}

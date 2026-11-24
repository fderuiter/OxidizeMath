//! Hohenberg-Halperin Model A (Allen-Cahn) non-conserved phase-ordering SPDE
//! ($\partial_t \phi = M(\kappa \nabla^2 \phi + \phi - \phi^3) + \eta$).
// [cite:stat_mech]

use crate::error::StatMechError;
use crate::physics::stat_mech::edwards_wilkinson::mean_field;
use engine_pde::{
    BoundaryCondition, ExplicitSpdeStencilStepper, SpdeNoiseKind, SplitStepSpdeStencilStepper,
    StencilOperators,
};
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, SolverWorkspace, StepSize, flatten_2d_index};
use rand_core::RngCore;

/// Evaluates the discrete 2D Ginzburg-Landau free-energy Lyapunov functional
/// $F_h[\phi] = \sum_{x, y} \left[\frac{1}{4}(\phi^2 - 1)^2 + \frac{\kappa}{2}|\nabla_h^+ \phi|^2\right] \Delta x \Delta y$,
/// whose discrete variational derivative $\delta F_h / \delta \phi = \phi^3 - \phi - \kappa \nabla_h^2 \phi$
/// guarantees monotonic energy decay under Allen-Cahn and Cahn-Hilliard coarsening.
#[verified_engine::verified]
pub fn ginzburg_landau_free_energy_2d(
    phi: &[f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    kappa: f64,
    bc: BoundaryCondition,
) -> f64 {
    let total = width * height;
    assert!(phi.len() >= total && dx > 0.0 && dy > 0.0 && kappa >= 0.0);
    if total == 0 {
        return 0.0;
    }
    let inv_dx = 1.0 / dx;
    let inv_dy = 1.0 / dy;
    let mut sum = 0.0;
    for y in 0..height {
        for x in 0..width {
            let idx = flatten_2d_index(x, y, width);
            let p = phi[idx];
            let well = p * p - 1.0;
            let bulk = 0.25 * well * well;
            let (_curr, _left, right, _up, down) = bc.sample_2d(phi, width, height, x, y);
            let gx = if x + 1 < width || bc == BoundaryCondition::Periodic {
                (right - p) * inv_dx
            } else {
                0.0
            };
            let gy = if y + 1 < height || bc == BoundaryCondition::Periodic {
                (down - p) * inv_dy
            } else {
                0.0
            };
            sum += bulk + 0.5 * kappa * (gx * gx + gy * gy);
        }
    }
    sum * dx * dy
}

/// 2D non-conserved Allen-Cahn (Model A) phase-ordering SPDE solver.
#[derive(Debug, Clone)]
pub struct AllenCahn2D {
    phi: Vec<f64>,
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    mobility: f64,
    kappa: f64,
    noise_intensity: f64,
    boundary: BoundaryCondition,
    scratch: Vec<f64>,
}

impl PartialEq for AllenCahn2D {
    fn eq(&self, other: &Self) -> bool {
        self.phi == other.phi
            && self.width == other.width
            && self.height == other.height
            && self.dx == other.dx
            && self.dy == other.dy
            && self.mobility == other.mobility
            && self.kappa == other.kappa
            && self.noise_intensity == other.noise_intensity
            && self.boundary == other.boundary
    }
}

impl AllenCahn2D {
    /// Creates a validated 2D Allen-Cahn solver with pre-allocated zero-allocation workspace.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_new(
        width: usize,
        height: usize,
        dx: f64,
        dy: f64,
        mobility: f64,
        kappa: f64,
        noise_intensity: f64,
        boundary: BoundaryCondition,
    ) -> Result<Self, StatMechError> {
        if width < 3 || height < 3 {
            return Err(MathError::DimensionMismatch {
                expected: Dimension(3),
                actual: Dimension(width.min(height)),
            }
            .into());
        }
        if !mobility.is_finite() || mobility <= 0.0 {
            return Err(MathError::StrictlyOutOfBounds {
                value: mobility,
                min: 0.0,
                max: f64::INFINITY,
            }
            .into());
        }
        if !kappa.is_finite()
            || kappa < 0.0
            || !noise_intensity.is_finite()
            || noise_intensity < 0.0
        {
            return Err(MathError::OutOfBounds {
                value: kappa.min(noise_intensity),
                min: 0.0,
                max: f64::INFINITY,
            }
            .into());
        }
        let _ = StencilOperators::try_new_3d(StepSize(dx), StepSize(dy), StepSize(1.0))?;
        let cap = ExplicitSpdeStencilStepper::scratch_len_2d(width, height)
            .max(SplitStepSpdeStencilStepper::scratch_len_2d(width, height));
        Ok(Self {
            phi: vec![0.0; width * height],
            width,
            height,
            dx,
            dy,
            mobility,
            kappa,
            noise_intensity,
            boundary,
            scratch: vec![0.0; cap],
        })
    }

    /// Creates a 2D Allen-Cahn solver, panicking on invalid grid or physical parameters.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn new(
        width: usize,
        height: usize,
        dx: f64,
        dy: f64,
        mobility: f64,
        kappa: f64,
        noise_intensity: f64,
        boundary: BoundaryCondition,
    ) -> Self {
        let model = Self::try_new(
            width,
            height,
            dx,
            dy,
            mobility,
            kappa,
            noise_intensity,
            boundary,
        )
        .expect("Valid AllenCahn2D parameters");
        assert_eq!(model.phi.len(), width * height);
        model
    }

    /// Immutable view of the order-parameter field $\phi(x, y, t)$.
    #[inline]
    pub fn field(&self) -> &[f64] {
        &self.phi
    }

    /// Mutable view of the order-parameter field $\phi(x, y, t)$.
    #[inline]
    pub fn field_mut(&mut self) -> &mut [f64] {
        &mut self.phi
    }

    /// Computes the spatial mean order parameter $\bar{\phi}(t)$.
    #[inline]
    pub fn spatial_mean(&self) -> f64 {
        mean_field(&self.phi)
    }

    /// Evaluates the Ginzburg-Landau free energy $F[\phi]$ on the current order-parameter state.
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

    /// Advances one explicit Euler-Maruyama step without heap allocations.
    #[verified_engine::verified]
    pub fn step_explicit<R: RngCore + ?Sized>(&mut self, dt: f64, rng: &mut R) {
        assert!(dt >= 0.0);
        let ops = StencilOperators::new_2d(StepSize(self.dx), StepSize(self.dy));
        let stepper = ExplicitSpdeStencilStepper::new(
            ops,
            self.boundary,
            SpdeNoiseKind::Additive {
                intensity: self.noise_intensity,
            },
        );
        let m = self.mobility;
        let kappa = self.kappa;
        let mut ws = SolverWorkspace::new(&mut self.scratch);
        stepper.step_2d(
            &mut self.phi,
            self.width,
            self.height,
            dt,
            |_x, _y, u, _grad, lap| m * (kappa * lap + u - u * u * u),
            &mut ws,
            rng,
        );
    }

    /// Advances one Strang operator-split step (`diffusion_substeps >= 1`) without heap allocations.
    #[verified_engine::verified]
    pub fn step_split<R: RngCore + ?Sized>(
        &mut self,
        dt: f64,
        diffusion_substeps: usize,
        rng: &mut R,
    ) {
        assert!(dt >= 0.0);
        let ops = StencilOperators::new_2d(StepSize(self.dx), StepSize(self.dy));
        let stepper = SplitStepSpdeStencilStepper::new(
            ops,
            self.boundary,
            SpdeNoiseKind::Additive {
                intensity: self.noise_intensity,
            },
            diffusion_substeps,
        );
        let m = self.mobility;
        let diffusivity = m * self.kappa;
        let mut ws = SolverWorkspace::new(&mut self.scratch);
        stepper.step_reaction_diffusion_2d(
            &mut self.phi,
            self.width,
            self.height,
            diffusivity,
            dt,
            |u| m * (u - u * u * u),
            &mut ws,
            rng,
        );
    }
}

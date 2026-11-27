// [cite:engine_pde]
mod helpers;
use crate::boundary::BoundaryCondition;
use crate::error::{PdeError, carve_f64};
use crate::fourier_stepper::check_non_neg;
use crate::spde_noise::{SpdeNoiseKind, try_apply_noise_1d, try_apply_noise_2d};
use crate::stencil::StencilOperators;
use helpers::{
    advance_cahn_hilliard_stencil_2d, advance_diffusion_substeps_2d, apply_local_reaction_2d,
};
use math_commons::math_kernel::types::flatten_2d_index;
use math_memory::{ByteArena, SolverWorkspace};
use rand_core::RngCore;
/// Explicit Euler-Maruyama finite-difference SPDE stepper evolving
/// $\partial_t u = \mathcal{N}(u, \nabla_h u, \nabla_h^2 u) + \xi$ across Periodic, Neumann, and Dirichlet
/// boundaries with additive or staggered-face conserved-flux noise.
///
/// Reference: LeVeque (2007), DOI: 10.1137/1.9780898717839;
/// [`docs/math_specs/engine_pde.md`](../../../../docs/math_specs/engine_pde.md).
///
/// # Stability
/// Second-order central diffusion requires $D\,\Delta t \sum_k \Delta x_k^{-2} \le \frac{1}{2}$, while
/// explicit 2D Biharmonic Cahn-Hilliard stepping requires $M\kappa\,\Delta t(\Delta x^{-2} + \Delta y^{-2})^2 \le \frac{1}{8}$.
///
/// # Domain of Validity
/// Requires positive finite stencil spacings (`dx, dy > 0`), `dt >= 0`, non-negative noise intensity,
/// and `SolverWorkspace` capacity [`Self::scratch_len_1d`] / [`Self::scratch_len_2d`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExplicitSpdeStencilStepper {
    /// Spatial finite-difference stencil operators.
    pub ops: StencilOperators,
    /// Boundary condition enforced on all spatial axes.
    pub boundary: BoundaryCondition,
    /// Stochastic noise structure.
    pub noise: SpdeNoiseKind,
}
impl ExplicitSpdeStencilStepper {
    /// Creates an [`ExplicitSpdeStencilStepper`] from prevalidated [`StencilOperators`].
    #[inline]
    pub const fn new(
        ops: StencilOperators,
        boundary: BoundaryCondition,
        noise: SpdeNoiseKind,
    ) -> Self {
        Self {
            ops,
            boundary,
            noise,
        }
    }
    /// Returns the scratch buffer length required for 1D SPDE steps (`2 * n`).
    #[inline]
    pub const fn scratch_len_1d(n: usize) -> usize {
        2 * n
    }
    /// Returns the scratch buffer length required for 2D SPDE steps (`5 * width * height`).
    #[inline]
    pub const fn scratch_len_2d(width: usize, height: usize) -> usize {
        5 * width * height
    }
    /// Fallibly advances a 1D SPDE $\partial_t u = \mathcal{N}(x, u, \partial_x u, \partial_{xx} u) + \xi$ in-place.
    #[verified_engine::verified]
    pub fn try_step_1d<F, R>(
        &self,
        u: &mut [f64],
        dt: f64,
        mut deterministic_rhs: F,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) -> Result<(), PdeError>
    where
        F: FnMut(usize, f64, f64, f64) -> f64,
        R: RngCore + ?Sized,
    {
        let n = u.len();
        check_non_neg("dt", dt)?;
        if n == 0 || dt == 0.0 {
            return Ok(());
        }
        let mut ws = workspace.reborrow();
        let mut cur = ws.cursor();
        let src = carve_f64(&mut cur, n)?;
        let mut noise_ws = ByteArena::new(cur.into_bytes());
        src.copy_from_slice(u);
        for x in 0..n {
            if let BoundaryCondition::Dirichlet(val) = self.boundary
                && (x == 0 || x + 1 == n)
            {
                u[x] = val;
                continue;
            }
            let grad = self.ops.try_gradient_1d_at(src, n, x, self.boundary)?;
            let lap = self.ops.try_laplacian_1d_at(src, n, x, self.boundary)?;
            u[x] = src[x] + dt * deterministic_rhs(x, src[x], grad, lap);
        }
        debug_assert_eq!(u.len(), n);
        try_apply_noise_1d(
            u,
            *self.ops.dx,
            dt,
            self.boundary,
            self.noise,
            &mut noise_ws,
            rng,
        )
    }
    /// Advances a 1D SPDE $\partial_t u = \mathcal{N}(x, u, \partial_x u, \partial_{xx} u) + \xi$ in-place; prefer [`Self::try_step_1d`].
    #[verified_engine::verified]
    pub fn step_1d<F, R>(
        &self,
        u: &mut [f64],
        dt: f64,
        deterministic_rhs: F,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) where
        F: FnMut(usize, f64, f64, f64) -> f64,
        R: RngCore + ?Sized,
    {
        self.try_step_1d(u, dt, deterministic_rhs, workspace, rng)
            .expect("ExplicitSpdeStencilStepper::step_1d failed");
    }
    /// Fallibly advances a 2D SPDE $\partial_t u = \mathcal{N}(x, y, u, \nabla u, \nabla^2 u) + \xi$ in-place.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_2d<F, R>(
        &self,
        u: &mut [f64],
        width: usize,
        height: usize,
        dt: f64,
        mut deterministic_rhs: F,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) -> Result<(), PdeError>
    where
        F: FnMut(usize, usize, f64, (f64, f64), f64) -> f64,
        R: RngCore + ?Sized,
    {
        let total = width * height;
        if u.len() < total {
            return Err(PdeError::BufferTooSmall {
                buffer: "u",
                required: total,
                actual: u.len(),
            });
        }
        check_non_neg("dt", dt)?;
        if total == 0 || dt == 0.0 {
            return Ok(());
        }
        let mut ws = workspace.reborrow();
        let mut cur = ws.cursor();
        let src = carve_f64(&mut cur, total)?;
        let mut noise_ws = ByteArena::new(cur.into_bytes());
        src.copy_from_slice(&u[..total]);
        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                if let BoundaryCondition::Dirichlet(val) = self.boundary
                    && (x == 0 || x + 1 == width || y == 0 || y + 1 == height)
                {
                    u[idx] = val;
                    continue;
                }
                let grad = self
                    .ops
                    .try_gradient_2d_at(src, width, height, x, y, self.boundary)?;
                let lap = self
                    .ops
                    .try_laplacian_2d_at(src, width, height, x, y, self.boundary)?;
                u[idx] = src[idx] + dt * deterministic_rhs(x, y, src[idx], grad, lap);
            }
        }
        debug_assert!(u.len() >= total);
        try_apply_noise_2d(
            &mut u[..total],
            width,
            height,
            *self.ops.dx,
            *self.ops.dy,
            dt,
            self.boundary,
            self.noise,
            &mut noise_ws,
            rng,
        )
    }
    /// Advances a 2D SPDE $\partial_t u = \mathcal{N}(x, y, u, \nabla u, \nabla^2 u) + \xi$ in-place; prefer [`Self::try_step_2d`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_2d<F, R>(
        &self,
        u: &mut [f64],
        width: usize,
        height: usize,
        dt: f64,
        deterministic_rhs: F,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) where
        F: FnMut(usize, usize, f64, (f64, f64), f64) -> f64,
        R: RngCore + ?Sized,
    {
        self.try_step_2d(u, width, height, dt, deterministic_rhs, workspace, rng)
            .expect("ExplicitSpdeStencilStepper::step_2d failed");
    }
    /// Fallibly advances the 2D conserved Cahn-Hilliard SPDE
    /// $\partial_t \phi = M \left[\nabla^2(\phi^3 - \phi) - \kappa \nabla^4 \phi\right] + \nabla \cdot \vec{\eta}$.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_cahn_hilliard_2d<R: RngCore + ?Sized>(
        &self,
        phi: &mut [f64],
        width: usize,
        height: usize,
        mobility: f64,
        kappa: f64,
        dt: f64,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) -> Result<(), PdeError> {
        let total = width * height;
        if phi.len() < total {
            return Err(PdeError::BufferTooSmall {
                buffer: "phi",
                required: total,
                actual: phi.len(),
            });
        }
        check_non_neg("mobility", mobility)?;
        check_non_neg("kappa", kappa)?;
        check_non_neg("dt", dt)?;
        if total == 0 || dt == 0.0 {
            return Ok(());
        }
        let mut ws = workspace.reborrow();
        let mut cur = ws.cursor();
        let src = carve_f64(&mut cur, total)?;
        let bulk_mu = carve_f64(&mut cur, total)?;
        let lap_phi = carve_f64(&mut cur, total)?;
        let mut noise_ws = ByteArena::new(cur.into_bytes());
        src.copy_from_slice(&phi[..total]);
        for i in 0..total {
            let p = src[i];
            bulk_mu[i] = p * p * p - p;
        }
        self.ops
            .try_compute_laplacian_2d(src, width, height, self.boundary, lap_phi)?;
        advance_cahn_hilliard_stencil_2d(
            &self.ops,
            self.boundary,
            src,
            bulk_mu,
            lap_phi,
            &mut phi[..total],
            width,
            height,
            mobility,
            kappa,
            dt,
        )?;
        debug_assert!(phi.len() >= total);
        try_apply_noise_2d(
            &mut phi[..total],
            width,
            height,
            *self.ops.dx,
            *self.ops.dy,
            dt,
            self.boundary,
            self.noise,
            &mut noise_ws,
            rng,
        )
    }
    /// Advances the 2D conserved Cahn-Hilliard SPDE via finite differences; prefer [`Self::try_step_cahn_hilliard_2d`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_cahn_hilliard_2d<R: RngCore + ?Sized>(
        &self,
        phi: &mut [f64],
        width: usize,
        height: usize,
        mobility: f64,
        kappa: f64,
        dt: f64,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) {
        self.try_step_cahn_hilliard_2d(phi, width, height, mobility, kappa, dt, workspace, rng)
            .expect("ExplicitSpdeStencilStepper::step_cahn_hilliard_2d failed");
    }
}
/// Operator-splitting Strang SPDE stencil stepper advancing $\partial_t u = D \nabla_h^2 u + R(u) + \xi$
/// via half-reaction ($\Delta t / 2$), `diffusion_substeps` diffusive substeps, half-reaction, and noise.
///
/// Reference: Strang (1968), DOI: 10.1137/0705041;
/// [`docs/math_specs/engine_pde.md`](../../../../docs/math_specs/engine_pde.md).
///
/// # Stability
/// Partitioning diffusion into $S = \text{diffusion\_substeps} \ge 1$ substeps relaxes the parabolic CFL
/// constraint to $D(\Delta t / S)(\Delta x^{-2} + \Delta y^{-2}) \le \frac{1}{2}$ while retaining second-order deterministic splitting.
///
/// # Domain of Validity
/// Requires positive grid spacings (`dx, dy > 0`), `diffusivity >= 0`, `dt >= 0`, `u.len() >= width * height`,
/// and `SolverWorkspace` capacity at least [`Self::scratch_len_2d`]`(width, height)`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SplitStepSpdeStencilStepper {
    /// Spatial finite-difference stencil operators.
    pub ops: StencilOperators,
    /// Boundary condition enforced during diffusive substeps.
    pub boundary: BoundaryCondition,
    /// Stochastic noise structure applied after operator splitting.
    pub noise: SpdeNoiseKind,
    /// Number of diffusive substeps executed between half-reaction kicks (`>= 1`).
    pub diffusion_substeps: usize,
}
impl SplitStepSpdeStencilStepper {
    /// Creates a [`SplitStepSpdeStencilStepper`] with `diffusion_substeps.max(1)` diffusive substeps.
    #[inline]
    pub const fn new(
        ops: StencilOperators,
        boundary: BoundaryCondition,
        noise: SpdeNoiseKind,
        diffusion_substeps: usize,
    ) -> Self {
        let subs = if diffusion_substeps == 0 {
            1
        } else {
            diffusion_substeps
        };
        Self {
            ops,
            boundary,
            noise,
            diffusion_substeps: subs,
        }
    }
    /// Returns the scratch buffer length required for 2D split-step integration (`3 * width * height`).
    #[inline]
    pub const fn scratch_len_2d(width: usize, height: usize) -> usize {
        3 * width * height
    }
    /// Fallibly advances a 2D reaction-diffusion SPDE $\partial_t u = D \nabla^2 u + R(u) + \xi$ via Strang splitting.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_reaction_diffusion_2d<F, R>(
        &self,
        u: &mut [f64],
        width: usize,
        height: usize,
        diffusivity: f64,
        dt: f64,
        mut reaction: F,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) -> Result<(), PdeError>
    where
        F: FnMut(f64) -> f64,
        R: RngCore + ?Sized,
    {
        let total = width * height;
        if u.len() < total {
            return Err(PdeError::BufferTooSmall {
                buffer: "u",
                required: total,
                actual: u.len(),
            });
        }
        check_non_neg("diffusivity", diffusivity)?;
        check_non_neg("dt", dt)?;
        if total == 0 || dt == 0.0 {
            return Ok(());
        }
        let mut ws = workspace.reborrow();
        let mut cur = ws.cursor();
        let src = carve_f64(&mut cur, total)?;
        let mut noise_ws = ByteArena::new(cur.into_bytes());
        let half_dt = 0.5 * dt;
        apply_local_reaction_2d(u, width, height, half_dt, self.boundary, &mut reaction);
        let sub_dt = dt / (self.diffusion_substeps as f64);
        advance_diffusion_substeps_2d(
            &self.ops,
            self.boundary,
            self.diffusion_substeps,
            src,
            &mut u[..total],
            width,
            height,
            diffusivity,
            sub_dt,
        )?;
        apply_local_reaction_2d(u, width, height, half_dt, self.boundary, &mut reaction);
        debug_assert!(u.len() >= total);
        try_apply_noise_2d(
            &mut u[..total],
            width,
            height,
            *self.ops.dx,
            *self.ops.dy,
            dt,
            self.boundary,
            self.noise,
            &mut noise_ws,
            rng,
        )
    }
    /// Advances a 2D reaction-diffusion SPDE via Strang splitting; prefer [`Self::try_step_reaction_diffusion_2d`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_reaction_diffusion_2d<F, R>(
        &self,
        u: &mut [f64],
        width: usize,
        height: usize,
        diffusivity: f64,
        dt: f64,
        reaction: F,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) where
        F: FnMut(f64) -> f64,
        R: RngCore + ?Sized,
    {
        self.try_step_reaction_diffusion_2d(
            u,
            width,
            height,
            diffusivity,
            dt,
            reaction,
            workspace,
            rng,
        )
        .expect("SplitStepSpdeStencilStepper::step_reaction_diffusion_2d failed");
    }
}

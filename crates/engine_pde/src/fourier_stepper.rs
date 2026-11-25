// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use crate::error::PdeError;
use crate::fft::{try_fft_1d_inplace, try_fft_2d_inplace};
use crate::fourier_kernels::{
    pack_real, try_eval_dealiased_cubic_1d, try_eval_dealiased_cubic_2d, unpack_real,
    update_cahn_hilliard_1d_modes, update_cahn_hilliard_2d_modes,
};
use crate::spde_noise::{try_inject_conserved_flux_noise_1d, try_inject_conserved_flux_noise_2d};
use math_commons::math_kernel::types::SolverWorkspace;
use rand_core::RngCore;

/// Kardar-Parisi-Zhang (KPZ) and Edwards-Wilkinson semi-implicit Fourier steppers.
pub mod kpz;

#[inline]
pub(crate) fn check_non_neg(param: &'static str, val: f64) -> Result<(), PdeError> {
    if !(val >= 0.0 && val.is_finite()) {
        return Err(PdeError::InvalidParameter { parameter: param });
    }
    Ok(())
}

#[inline]
pub(crate) fn check_radix2_min4(axis: &'static str, size: usize) -> Result<(), PdeError> {
    if size < 4 {
        return Err(PdeError::InvalidGridDimension { axis, size });
    }
    if !size.is_power_of_two() {
        return Err(PdeError::NonRadix2Dimension { axis, size });
    }
    Ok(())
}

/// `#![no_std]` radix-2 semi-implicit Fourier SPDE stepper with Orszag 2/3-rule dealiasing
/// for periodic Cahn-Hilliard, Kardar-Parisi-Zhang (KPZ), and Edwards-Wilkinson fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemiImplicitFourierStepper {
    /// Grid spacing $\Delta x > 0$.
    pub dx: f64,
    /// Secondary grid spacing $\Delta y > 0$ (used in 2D solvers).
    pub dy: f64,
}

impl SemiImplicitFourierStepper {
    /// Fallibly creates a 1D radix-2 Fourier stepper for uniform spacing `dx > 0`.
    #[verified_engine::verified]
    pub fn try_new_1d(dx: f64) -> Result<Self, PdeError> {
        if !(dx > 0.0 && dx.is_finite()) {
            return Err(PdeError::InvalidParameter { parameter: "dx" });
        }
        debug_assert!(dx > 0.0);
        Ok(Self { dx, dy: 1.0 })
    }

    /// Creates a 1D radix-2 Fourier stepper for uniform spacing `dx > 0`; prefer [`Self::try_new_1d`].
    #[verified_engine::verified]
    pub fn new_1d(dx: f64) -> Self {
        Self::try_new_1d(dx).expect("dx must be positive and finite")
    }

    /// Fallibly creates a 2D radix-2 Fourier stepper for spacings `dx > 0` and `dy > 0`.
    #[verified_engine::verified]
    pub fn try_new_2d(dx: f64, dy: f64) -> Result<Self, PdeError> {
        if !(dx > 0.0 && dx.is_finite()) {
            return Err(PdeError::InvalidParameter { parameter: "dx" });
        }
        if !(dy > 0.0 && dy.is_finite()) {
            return Err(PdeError::InvalidParameter { parameter: "dy" });
        }
        debug_assert!(dx > 0.0 && dy > 0.0);
        Ok(Self { dx, dy })
    }

    /// Creates a 2D radix-2 Fourier stepper for spacings `dx > 0` and `dy > 0`; prefer [`Self::try_new_2d`].
    #[verified_engine::verified]
    pub fn new_2d(dx: f64, dy: f64) -> Self {
        Self::try_new_2d(dx, dy).expect("dx and dy must be positive and finite")
    }

    /// Returns the scratch slice length required for 1D semi-implicit Fourier steps (`9 * n`).
    #[inline]
    pub const fn scratch_len_1d(n: usize) -> usize {
        9 * n
    }

    /// Returns the scratch slice length required for 2D semi-implicit Fourier steps (`10 * width * height + 2 * height`).
    #[inline]
    pub const fn scratch_len_2d(width: usize, height: usize) -> usize {
        10 * width * height + 2 * height
    }

    /// Fallibly advances the 1D Cahn-Hilliard SPDE
    /// $\partial_t \phi = M \partial_{xx}(\phi^3 - \phi - \kappa \partial_{xx}\phi) + \partial_x \eta$
    /// with 2/3-rule dealiasing and exact $k = 0$ mass conservation.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_cahn_hilliard_1d<R: RngCore + ?Sized>(
        &self,
        phi: &mut [f64],
        mobility: f64,
        kappa: f64,
        mobility_temp: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), PdeError> {
        let n = phi.len();
        check_radix2_min4("n", n)?;
        check_non_neg("mobility", mobility)?;
        check_non_neg("kappa", kappa)?;
        check_non_neg("dt", dt)?;
        if dt == 0.0 {
            return Ok(());
        }

        let mut ws = workspace.reborrow();
        let phi_hat = ws.try_take_slice(2 * n)?;
        let deal_buf = ws.try_take_slice(2 * n)?;
        let noise_spec = ws.try_take_slice(2 * n)?;
        let noise_phys = ws.try_take_slice(n)?;

        pack_real(phi, phi_hat, n);
        try_fft_1d_inplace(phi_hat, n, false)?;
        try_eval_dealiased_cubic_1d(phi_hat, deal_buf, n)?;

        noise_phys.fill(0.0);
        if mobility_temp > 0.0 {
            try_inject_conserved_flux_noise_1d(
                noise_phys,
                self.dx,
                dt,
                mobility_temp,
                BoundaryCondition::Periodic,
                &mut ws,
                rng,
            )?;
            pack_real(noise_phys, noise_spec, n);
            try_fft_1d_inplace(noise_spec, n, false)?;
            noise_spec[0] = 0.0;
            noise_spec[1] = 0.0;
        } else {
            noise_spec.fill(0.0);
        }

        update_cahn_hilliard_1d_modes(
            phi_hat, deal_buf, noise_spec, n, self.dx, mobility, kappa, dt,
        );
        try_fft_1d_inplace(phi_hat, n, true)?;
        unpack_real(phi_hat, phi, n);
        debug_assert_eq!(phi.len(), n);
        Ok(())
    }

    /// Advances the 1D Cahn-Hilliard SPDE with 2/3-rule dealiasing; prefer [`Self::try_step_cahn_hilliard_1d`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_cahn_hilliard_1d<R: RngCore + ?Sized>(
        &self,
        phi: &mut [f64],
        mobility: f64,
        kappa: f64,
        mobility_temp: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) {
        self.try_step_cahn_hilliard_1d(phi, mobility, kappa, mobility_temp, dt, workspace, rng)
            .expect("SemiImplicitFourierStepper::step_cahn_hilliard_1d failed");
    }

    /// Fallibly advances the 2D Cahn-Hilliard SPDE on a radix-2 `width x height` periodic grid with 2/3-rule dealiasing.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_cahn_hilliard_2d<R: RngCore + ?Sized>(
        &self,
        phi: &mut [f64],
        width: usize,
        height: usize,
        mobility: f64,
        kappa: f64,
        mobility_temp: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) -> Result<(), PdeError> {
        check_radix2_min4("width", width)?;
        check_radix2_min4("height", height)?;
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
        if dt == 0.0 {
            return Ok(());
        }

        let mut ws = workspace.reborrow();
        let phi_hat = ws.try_take_slice(2 * total)?;
        let deal_buf = ws.try_take_slice(2 * total)?;
        let noise_spec = ws.try_take_slice(2 * total)?;
        let col_scratch = ws.try_take_slice(2 * height)?;
        let noise_phys = ws.try_take_slice(total)?;

        pack_real(&phi[..total], phi_hat, total);
        try_fft_2d_inplace(phi_hat, width, height, col_scratch, false)?;
        try_eval_dealiased_cubic_2d(phi_hat, deal_buf, col_scratch, width, height)?;

        noise_phys.fill(0.0);
        if mobility_temp > 0.0 {
            try_inject_conserved_flux_noise_2d(
                noise_phys,
                width,
                height,
                self.dx,
                self.dy,
                dt,
                mobility_temp,
                BoundaryCondition::Periodic,
                &mut ws,
                rng,
            )?;
            pack_real(noise_phys, noise_spec, total);
            try_fft_2d_inplace(noise_spec, width, height, col_scratch, false)?;
            noise_spec[0] = 0.0;
            noise_spec[1] = 0.0;
        } else {
            noise_spec.fill(0.0);
        }

        update_cahn_hilliard_2d_modes(
            phi_hat, deal_buf, noise_spec, width, height, self.dx, self.dy, mobility, kappa, dt,
        );
        try_fft_2d_inplace(phi_hat, width, height, col_scratch, true)?;
        unpack_real(phi_hat, &mut phi[..total], total);
        debug_assert!(phi.len() >= total);
        Ok(())
    }

    /// Advances the 2D Cahn-Hilliard SPDE on a radix-2 `width x height` periodic grid; prefer [`Self::try_step_cahn_hilliard_2d`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_cahn_hilliard_2d<R: RngCore + ?Sized>(
        &self,
        phi: &mut [f64],
        width: usize,
        height: usize,
        mobility: f64,
        kappa: f64,
        mobility_temp: f64,
        dt: f64,
        workspace: &mut SolverWorkspace<'_, f64>,
        rng: &mut R,
    ) {
        self.try_step_cahn_hilliard_2d(
            phi,
            width,
            height,
            mobility,
            kappa,
            mobility_temp,
            dt,
            workspace,
            rng,
        )
        .expect("SemiImplicitFourierStepper::step_cahn_hilliard_2d failed");
    }
}

// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use crate::error::{PdeError, carve_f64};
use crate::fft::{try_fft_1d_inplace, try_fft_2d_inplace};
use crate::fourier_kernels::{
    pack_real, try_eval_dealiased_grad_sq_1d, try_eval_dealiased_grad_sq_2d, unpack_real,
    update_kpz_1d_modes, update_kpz_2d_modes,
};
use crate::fourier_stepper::{SemiImplicitFourierStepper, check_non_neg, check_radix2_min4};
use crate::spde_noise::{try_inject_additive_noise_1d, try_inject_additive_noise_2d};
use math_memory::{ByteArena, SolverWorkspace};
use rand_core::RngCore;

impl SemiImplicitFourierStepper {
    /// Fallibly advances the 1D Kardar-Parisi-Zhang (KPZ) SPDE
    /// $\partial_t h = \nu \partial_{xx} h + \frac{\lambda}{2}(\partial_x h)^2 + \eta(x, t)$
    /// with 2/3-rule dealiasing.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_kpz_1d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        nu: f64,
        lambda: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) -> Result<(), PdeError> {
        let n = h.len();
        check_radix2_min4("n", n)?;
        check_non_neg("nu", nu)?;
        check_non_neg("dt", dt)?;
        if dt == 0.0 {
            return Ok(());
        }

        let mut ws = workspace.reborrow();
        let mut cur = ws.cursor();
        let h_hat = carve_f64(&mut cur, 2 * n)?;
        let grad_buf = carve_f64(&mut cur, 2 * n)?;
        let noise_spec = carve_f64(&mut cur, 2 * n)?;
        let noise_phys = carve_f64(&mut cur, n)?;
        let mut noise_ws = ByteArena::new(cur.into_bytes());

        pack_real(h, h_hat, n);
        try_fft_1d_inplace(h_hat, n, false)?;
        if lambda.abs() > 0.0 {
            try_eval_dealiased_grad_sq_1d(h_hat, grad_buf, n, self.dx)?;
        } else {
            grad_buf.fill(0.0);
        }

        noise_phys.fill(0.0);
        if noise_intensity > 0.0 {
            try_inject_additive_noise_1d(
                noise_phys,
                self.dx,
                dt,
                noise_intensity,
                BoundaryCondition::Periodic,
                &mut noise_ws,
                rng,
            )?;
            pack_real(noise_phys, noise_spec, n);
            try_fft_1d_inplace(noise_spec, n, false)?;
        } else {
            noise_spec.fill(0.0);
        }

        update_kpz_1d_modes(h_hat, grad_buf, noise_spec, n, self.dx, nu, lambda, dt);
        try_fft_1d_inplace(h_hat, n, true)?;
        unpack_real(h_hat, h, n);
        debug_assert_eq!(h.len(), n);
        Ok(())
    }

    /// Advances the 1D Kardar-Parisi-Zhang (KPZ) SPDE with 2/3-rule dealiasing; prefer [`Self::try_step_kpz_1d`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_kpz_1d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        nu: f64,
        lambda: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) {
        self.try_step_kpz_1d(h, nu, lambda, noise_intensity, dt, workspace, rng)
            .expect("SemiImplicitFourierStepper::step_kpz_1d failed");
    }

    /// Fallibly advances the 2D Kardar-Parisi-Zhang (KPZ) SPDE
    /// $\partial_t h = \nu \nabla^2 h + \frac{\lambda}{2}|\nabla h|^2 + \eta(x, y, t)$
    /// on a radix-2 `width x height` periodic grid with 2/3-rule dealiasing.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_kpz_2d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        width: usize,
        height: usize,
        nu: f64,
        lambda: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) -> Result<(), PdeError> {
        check_radix2_min4("width", width)?;
        check_radix2_min4("height", height)?;
        let total = width * height;
        if h.len() < total {
            return Err(PdeError::BufferTooSmall {
                buffer: "h",
                required: total,
                actual: h.len(),
            });
        }
        check_non_neg("nu", nu)?;
        check_non_neg("dt", dt)?;
        if dt == 0.0 {
            return Ok(());
        }

        let mut ws = workspace.reborrow();
        let mut cur = ws.cursor();
        let h_hat = carve_f64(&mut cur, 2 * total)?;
        let gx_buf = carve_f64(&mut cur, 2 * total)?;
        let gy_buf = carve_f64(&mut cur, 2 * total)?;
        let noise_spec = carve_f64(&mut cur, 2 * total)?;
        let col_scratch = carve_f64(&mut cur, 2 * height)?;
        let noise_phys = carve_f64(&mut cur, total)?;
        let mut noise_ws = ByteArena::new(cur.into_bytes());

        pack_real(&h[..total], h_hat, total);
        try_fft_2d_inplace(h_hat, width, height, col_scratch, false)?;
        if lambda.abs() > 0.0 {
            try_eval_dealiased_grad_sq_2d(
                h_hat,
                gx_buf,
                gy_buf,
                col_scratch,
                width,
                height,
                self.dx,
                self.dy,
            )?;
        } else {
            gx_buf.fill(0.0);
        }

        noise_phys.fill(0.0);
        if noise_intensity > 0.0 {
            try_inject_additive_noise_2d(
                noise_phys,
                width,
                height,
                self.dx,
                self.dy,
                dt,
                noise_intensity,
                BoundaryCondition::Periodic,
                &mut noise_ws,
                rng,
            )?;
            pack_real(noise_phys, noise_spec, total);
            try_fft_2d_inplace(noise_spec, width, height, col_scratch, false)?;
        } else {
            noise_spec.fill(0.0);
        }

        update_kpz_2d_modes(
            h_hat, gx_buf, noise_spec, width, height, self.dx, self.dy, nu, lambda, dt,
        );
        try_fft_2d_inplace(h_hat, width, height, col_scratch, true)?;
        unpack_real(h_hat, &mut h[..total], total);
        debug_assert!(h.len() >= total);
        Ok(())
    }

    /// Advances the 2D Kardar-Parisi-Zhang (KPZ) SPDE on a radix-2 `width x height` grid; prefer [`Self::try_step_kpz_2d`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_kpz_2d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        width: usize,
        height: usize,
        nu: f64,
        lambda: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) {
        self.try_step_kpz_2d(
            h,
            width,
            height,
            nu,
            lambda,
            noise_intensity,
            dt,
            workspace,
            rng,
        )
        .expect("SemiImplicitFourierStepper::step_kpz_2d failed");
    }

    /// Fallibly advances the 1D linear Edwards-Wilkinson SPDE $\partial_t h = \nu \partial_{xx} h + \eta$.
    #[verified_engine::verified]
    pub fn try_step_edwards_wilkinson_1d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        nu: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) -> Result<(), PdeError> {
        self.try_step_kpz_1d(h, nu, 0.0, noise_intensity, dt, workspace, rng)
    }

    /// Advances the 1D linear Edwards-Wilkinson SPDE $\partial_t h = \nu \partial_{xx} h + \eta$; prefer [`Self::try_step_edwards_wilkinson_1d`].
    #[verified_engine::verified]
    pub fn step_edwards_wilkinson_1d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        nu: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) {
        self.try_step_edwards_wilkinson_1d(h, nu, noise_intensity, dt, workspace, rng)
            .expect("SemiImplicitFourierStepper::step_edwards_wilkinson_1d failed");
    }

    /// Fallibly advances the 2D linear Edwards-Wilkinson SPDE $\partial_t h = \nu \nabla^2 h + \eta$.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn try_step_edwards_wilkinson_2d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        width: usize,
        height: usize,
        nu: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) -> Result<(), PdeError> {
        self.try_step_kpz_2d(
            h,
            width,
            height,
            nu,
            0.0,
            noise_intensity,
            dt,
            workspace,
            rng,
        )
    }

    /// Advances the 2D linear Edwards-Wilkinson SPDE $\partial_t h = \nu \nabla^2 h + \eta$; prefer [`Self::try_step_edwards_wilkinson_2d`].
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_edwards_wilkinson_2d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        width: usize,
        height: usize,
        nu: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut (impl SolverWorkspace + ?Sized),
        rng: &mut R,
    ) {
        self.try_step_edwards_wilkinson_2d(
            h,
            width,
            height,
            nu,
            noise_intensity,
            dt,
            workspace,
            rng,
        )
        .expect("SemiImplicitFourierStepper::step_edwards_wilkinson_2d failed");
    }
}

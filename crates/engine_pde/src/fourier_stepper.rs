// [cite:engine_pde]
use crate::boundary::BoundaryCondition;
use crate::fft::{
    dealias_1d_spectrum, dealias_2d_spectrum, fft_1d_inplace, fft_2d_inplace, mode_wavenumber,
};
use crate::fourier_kernels::{
    eval_dealiased_grad_sq_2d, pack_real, unpack_real, update_cahn_hilliard_2d_modes,
    update_kpz_2d_modes,
};
use crate::spde_noise::{
    inject_additive_noise_1d, inject_additive_noise_2d, inject_conserved_flux_noise_1d,
    inject_conserved_flux_noise_2d,
};
use math_memory::SolverWorkspace;
use rand_core::RngCore;

/// `#![no_std]` radix-2 semi-implicit Fourier SPDE stepper with Orszag 2/3-rule dealiasing
/// for periodic Cahn-Hilliard ($\partial_t \phi = M \nabla^2(\phi^3 - \phi - \kappa \nabla^2 \phi) + \nabla \cdot \vec{\eta}$),
/// Kardar-Parisi-Zhang ($\partial_t h = \nu \nabla^2 h + \frac{\lambda}{2}|\nabla h|^2 + \eta$), and Edwards-Wilkinson fields.
///
/// Reference: Orszag (1972), DOI: 10.1103/PhysRevLett.28.76; Kardar, Parisi, & Zhang (1986), DOI: 10.1103/PhysRevLett.56.889;
/// [`docs/math_specs/engine_pde.md`](../../../docs/math_specs/engine_pde.md).
///
/// # Stability
/// Treats linear $+\nu |\mathbf{k}|^2$ and $+M\kappa |\mathbf{k}|^4$ operators implicitly while truncating
/// modes above $\frac{2}{3} k_{\text{Nyq}}$ to suppress cubic/quadratic pseudospectral aliasing instabilities.
///
/// # Domain of Validity
/// Requires periodic boundaries, power-of-two grid dimensions $\ge 4$, finite positive `dx > 0` and
/// `dy > 0`, `dt >= 0`, and `SolverWorkspace` capacity [`Self::scratch_len_1d`] / [`Self::scratch_len_2d`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SemiImplicitFourierStepper {
    /// Grid spacing $\Delta x > 0$.
    pub dx: f64,
    /// Secondary grid spacing $\Delta y > 0$ (used in 2D solvers).
    pub dy: f64,
}

impl SemiImplicitFourierStepper {
    /// Creates a 1D radix-2 Fourier stepper for uniform spacing `dx > 0`.
    #[verified_engine::verified]
    pub fn new_1d(dx: f64) -> Self {
        assert!(dx > 0.0 && dx.is_finite(), "dx must be positive and finite");
        Self { dx, dy: 1.0 }
    }

    /// Creates a 2D radix-2 Fourier stepper for spacings `dx > 0` and `dy > 0`.
    #[verified_engine::verified]
    pub fn new_2d(dx: f64, dy: f64) -> Self {
        assert!(dx > 0.0 && dx.is_finite(), "dx must be positive and finite");
        assert!(dy > 0.0 && dy.is_finite(), "dy must be positive and finite");
        Self { dx, dy }
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

    /// Advances the 1D Cahn-Hilliard SPDE
    /// $\partial_t \phi = M \partial_{xx}(\phi^3 - \phi - \kappa \partial_{xx}\phi) + \partial_x \eta$
    /// with 2/3-rule dealiasing and exact $k = 0$ mass conservation.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_cahn_hilliard_1d<R: RngCore + ?Sized>(
        &self,
        phi: &mut [f64],
        mobility: f64,
        kappa: f64,
        mobility_temp: f64,
        dt: f64,
        workspace: &mut impl SolverWorkspace,
        rng: &mut R,
    ) {
        let n = phi.len();
        assert!(
            n >= 4 && n.is_power_of_two(),
            "grid length n must be radix-2 >= 4"
        );
        assert!(mobility >= 0.0 && kappa >= 0.0 && dt >= 0.0);
        if dt == 0.0 {
            return;
        }

        let mut ws = workspace.reborrow();
        let phi_hat = ws.take_slice::<f64>(2 * n);
        let deal_buf = ws.take_slice::<f64>(2 * n);
        let noise_spec = ws.take_slice::<f64>(2 * n);
        let noise_phys = ws.take_slice::<f64>(n);

        pack_real(phi, phi_hat, n);
        fft_1d_inplace(phi_hat, n, false);

        deal_buf.copy_from_slice(phi_hat);
        dealias_1d_spectrum(deal_buf, n);
        fft_1d_inplace(deal_buf, n, true);
        for m in 0..n {
            let p = deal_buf[2 * m];
            deal_buf[2 * m] = p * p * p - p;
            deal_buf[2 * m + 1] = 0.0;
        }
        fft_1d_inplace(deal_buf, n, false);
        dealias_1d_spectrum(deal_buf, n);

        noise_phys.fill(0.0);
        if mobility_temp > 0.0 {
            inject_conserved_flux_noise_1d(
                noise_phys,
                self.dx,
                dt,
                mobility_temp,
                BoundaryCondition::Periodic,
                &mut ws,
                rng,
            );
            pack_real(noise_phys, noise_spec, n);
            fft_1d_inplace(noise_spec, n, false);
            noise_spec[0] = 0.0;
            noise_spec[1] = 0.0;
        } else {
            noise_spec.fill(0.0);
        }

        for m in 0..n {
            let k = mode_wavenumber(m, n, self.dx);
            let k2 = k * k;
            let denom = 1.0 + dt * mobility * kappa * k2 * k2;
            let i0 = 2 * m;
            phi_hat[i0] =
                (phi_hat[i0] - dt * mobility * k2 * deal_buf[i0] + noise_spec[i0]) / denom;
            phi_hat[i0 + 1] = (phi_hat[i0 + 1] - dt * mobility * k2 * deal_buf[i0 + 1]
                + noise_spec[i0 + 1])
                / denom;
        }
        fft_1d_inplace(phi_hat, n, true);
        unpack_real(phi_hat, phi, n);
    }

    /// Advances the 2D Cahn-Hilliard SPDE on a radix-2 `width x height` periodic grid with 2/3-rule dealiasing.
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
        workspace: &mut impl SolverWorkspace,
        rng: &mut R,
    ) {
        let total = width * height;
        assert!(width >= 4 && width.is_power_of_two() && height >= 4 && height.is_power_of_two());
        assert!(phi.len() >= total && mobility >= 0.0 && kappa >= 0.0 && dt >= 0.0);
        if dt == 0.0 {
            return;
        }

        let mut ws = workspace.reborrow();
        let phi_hat = ws.take_slice::<f64>(2 * total);
        let deal_buf = ws.take_slice::<f64>(2 * total);
        let noise_spec = ws.take_slice::<f64>(2 * total);
        let col_scratch = ws.take_slice::<f64>(2 * height);
        let noise_phys = ws.take_slice::<f64>(total);

        pack_real(&phi[..total], phi_hat, total);
        fft_2d_inplace(phi_hat, width, height, col_scratch, false);

        deal_buf.copy_from_slice(phi_hat);
        dealias_2d_spectrum(deal_buf, width, height);
        fft_2d_inplace(deal_buf, width, height, col_scratch, true);
        for idx in 0..total {
            let p = deal_buf[2 * idx];
            deal_buf[2 * idx] = p * p * p - p;
            deal_buf[2 * idx + 1] = 0.0;
        }
        fft_2d_inplace(deal_buf, width, height, col_scratch, false);
        dealias_2d_spectrum(deal_buf, width, height);

        noise_phys.fill(0.0);
        if mobility_temp > 0.0 {
            inject_conserved_flux_noise_2d(
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
            );
            pack_real(noise_phys, noise_spec, total);
            fft_2d_inplace(noise_spec, width, height, col_scratch, false);
            noise_spec[0] = 0.0;
            noise_spec[1] = 0.0;
        } else {
            noise_spec.fill(0.0);
        }

        update_cahn_hilliard_2d_modes(
            phi_hat, deal_buf, noise_spec, width, height, self.dx, self.dy, mobility, kappa, dt,
        );
        fft_2d_inplace(phi_hat, width, height, col_scratch, true);
        unpack_real(phi_hat, &mut phi[..total], total);
    }

    /// Advances the 1D Kardar-Parisi-Zhang (KPZ) SPDE
    /// $\partial_t h = \nu \partial_{xx} h + \frac{\lambda}{2}(\partial_x h)^2 + \eta(x, t)$
    /// (reducing to Edwards-Wilkinson when `lambda == 0.0`) with 2/3-rule dealiasing.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_kpz_1d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        nu: f64,
        lambda: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut impl SolverWorkspace,
        rng: &mut R,
    ) {
        let n = h.len();
        assert!(n >= 4 && n.is_power_of_two() && nu >= 0.0 && dt >= 0.0);
        if dt == 0.0 {
            return;
        }

        let mut ws = workspace.reborrow();
        let h_hat = ws.take_slice::<f64>(2 * n);
        let grad_buf = ws.take_slice::<f64>(2 * n);
        let noise_spec = ws.take_slice::<f64>(2 * n);
        let noise_phys = ws.take_slice::<f64>(n);

        pack_real(h, h_hat, n);
        fft_1d_inplace(h_hat, n, false);

        if lambda.abs() > 0.0 {
            grad_buf.copy_from_slice(h_hat);
            dealias_1d_spectrum(grad_buf, n);
            for m in 0..n {
                let k = if m == n / 2 {
                    0.0
                } else {
                    mode_wavenumber(m, n, self.dx)
                };
                let re = grad_buf[2 * m];
                let im = grad_buf[2 * m + 1];
                grad_buf[2 * m] = -k * im;
                grad_buf[2 * m + 1] = k * re;
            }
            fft_1d_inplace(grad_buf, n, true);
            for m in 0..n {
                let gx = grad_buf[2 * m];
                grad_buf[2 * m] = gx * gx;
                grad_buf[2 * m + 1] = 0.0;
            }
            fft_1d_inplace(grad_buf, n, false);
            dealias_1d_spectrum(grad_buf, n);
        } else {
            grad_buf.fill(0.0);
        }

        noise_phys.fill(0.0);
        if noise_intensity > 0.0 {
            inject_additive_noise_1d(
                noise_phys,
                self.dx,
                dt,
                noise_intensity,
                BoundaryCondition::Periodic,
                &mut ws,
                rng,
            );
            pack_real(noise_phys, noise_spec, n);
            fft_1d_inplace(noise_spec, n, false);
        } else {
            noise_spec.fill(0.0);
        }

        let half_lambda_dt = 0.5 * lambda * dt;
        for m in 0..n {
            let k = mode_wavenumber(m, n, self.dx);
            let denom = 1.0 + dt * nu * k * k;
            let i0 = 2 * m;
            h_hat[i0] = (h_hat[i0] + half_lambda_dt * grad_buf[i0] + noise_spec[i0]) / denom;
            h_hat[i0 + 1] =
                (h_hat[i0 + 1] + half_lambda_dt * grad_buf[i0 + 1] + noise_spec[i0 + 1]) / denom;
        }
        fft_1d_inplace(h_hat, n, true);
        unpack_real(h_hat, h, n);
    }

    /// Advances the 2D Kardar-Parisi-Zhang (KPZ) SPDE
    /// $\partial_t h = \nu \nabla^2 h + \frac{\lambda}{2}|\nabla h|^2 + \eta(x, y, t)$
    /// on a radix-2 `width x height` periodic grid with 2/3-rule dealiasing.
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
        workspace: &mut impl SolverWorkspace,
        rng: &mut R,
    ) {
        let total = width * height;
        assert!(width >= 4 && width.is_power_of_two() && height >= 4 && height.is_power_of_two());
        assert!(h.len() >= total && nu >= 0.0 && dt >= 0.0);
        if dt == 0.0 {
            return;
        }

        let mut ws = workspace.reborrow();
        let h_hat = ws.take_slice::<f64>(2 * total);
        let gx_buf = ws.take_slice::<f64>(2 * total);
        let gy_buf = ws.take_slice::<f64>(2 * total);
        let noise_spec = ws.take_slice::<f64>(2 * total);
        let col_scratch = ws.take_slice::<f64>(2 * height);
        let noise_phys = ws.take_slice::<f64>(total);

        pack_real(&h[..total], h_hat, total);
        fft_2d_inplace(h_hat, width, height, col_scratch, false);

        if lambda.abs() > 0.0 {
            eval_dealiased_grad_sq_2d(
                h_hat,
                gx_buf,
                gy_buf,
                col_scratch,
                width,
                height,
                self.dx,
                self.dy,
            );
        } else {
            gx_buf.fill(0.0);
        }

        noise_phys.fill(0.0);
        if noise_intensity > 0.0 {
            inject_additive_noise_2d(
                noise_phys,
                width,
                height,
                self.dx,
                self.dy,
                dt,
                noise_intensity,
                BoundaryCondition::Periodic,
                &mut ws,
                rng,
            );
            pack_real(noise_phys, noise_spec, total);
            fft_2d_inplace(noise_spec, width, height, col_scratch, false);
        } else {
            noise_spec.fill(0.0);
        }

        update_kpz_2d_modes(
            h_hat, gx_buf, noise_spec, width, height, self.dx, self.dy, nu, lambda, dt,
        );
        fft_2d_inplace(h_hat, width, height, col_scratch, true);
        unpack_real(h_hat, &mut h[..total], total);
    }

    /// Advances the 1D linear Edwards-Wilkinson SPDE $\partial_t h = \nu \partial_{xx} h + \eta$.
    #[verified_engine::verified]
    pub fn step_edwards_wilkinson_1d<R: RngCore + ?Sized>(
        &self,
        h: &mut [f64],
        nu: f64,
        noise_intensity: f64,
        dt: f64,
        workspace: &mut impl SolverWorkspace,
        rng: &mut R,
    ) {
        assert!(!h.is_empty());
        self.step_kpz_1d(h, nu, 0.0, noise_intensity, dt, workspace, rng);
    }

    /// Advances the 2D linear Edwards-Wilkinson SPDE $\partial_t h = \nu \nabla^2 h + \eta$.
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
        workspace: &mut impl SolverWorkspace,
        rng: &mut R,
    ) {
        assert!(h.len() >= width * height);
        self.step_kpz_2d(
            h,
            width,
            height,
            nu,
            0.0,
            noise_intensity,
            dt,
            workspace,
            rng,
        );
    }
}

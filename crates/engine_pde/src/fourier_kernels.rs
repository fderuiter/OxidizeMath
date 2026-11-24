// [cite:engine_pde]
use crate::fft::{dealias_2d_spectrum, fft_2d_inplace, mode_wavenumber};
use math_commons::math_kernel::types::flatten_2d_index;

#[inline]
pub(crate) fn pack_real(real: &[f64], complex_out: &mut [f64], n: usize) {
    for i in 0..n {
        complex_out[2 * i] = real[i];
        complex_out[2 * i + 1] = 0.0;
    }
}

#[inline]
pub(crate) fn unpack_real(complex_in: &[f64], real_out: &mut [f64], n: usize) {
    for i in 0..n {
        real_out[i] = complex_in[2 * i];
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_cahn_hilliard_2d_modes(
    phi_hat: &mut [f64],
    deal_buf: &[f64],
    noise_spec: &[f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    mobility: f64,
    kappa: f64,
    dt: f64,
) {
    for y in 0..height {
        let ky = mode_wavenumber(y, height, dy);
        for x in 0..width {
            let kx = mode_wavenumber(x, width, dx);
            let k2 = kx * kx + ky * ky;
            let denom = 1.0 + dt * mobility * kappa * k2 * k2;
            let i0 = 2 * flatten_2d_index(x, y, width);
            phi_hat[i0] =
                (phi_hat[i0] - dt * mobility * k2 * deal_buf[i0] + noise_spec[i0]) / denom;
            phi_hat[i0 + 1] = (phi_hat[i0 + 1] - dt * mobility * k2 * deal_buf[i0 + 1]
                + noise_spec[i0 + 1])
                / denom;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_dealiased_grad_sq_2d(
    h_hat: &[f64],
    gx_buf: &mut [f64],
    gy_buf: &mut [f64],
    col_scratch: &mut [f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
) {
    gx_buf.copy_from_slice(h_hat);
    dealias_2d_spectrum(gx_buf, width, height);
    gy_buf.copy_from_slice(gx_buf);

    for y in 0..height {
        let ky = if y == height / 2 {
            0.0
        } else {
            mode_wavenumber(y, height, dy)
        };
        for x in 0..width {
            let kx = if x == width / 2 {
                0.0
            } else {
                mode_wavenumber(x, width, dx)
            };
            let i0 = 2 * flatten_2d_index(x, y, width);
            let re = gx_buf[i0];
            let im = gx_buf[i0 + 1];
            gx_buf[i0] = -kx * im;
            gx_buf[i0 + 1] = kx * re;
            gy_buf[i0] = -ky * im;
            gy_buf[i0 + 1] = ky * re;
        }
    }

    fft_2d_inplace(gx_buf, width, height, col_scratch, true);
    fft_2d_inplace(gy_buf, width, height, col_scratch, true);
    let total = width * height;
    for idx in 0..total {
        let gx = gx_buf[2 * idx];
        let gy = gy_buf[2 * idx];
        gx_buf[2 * idx] = gx * gx + gy * gy;
        gx_buf[2 * idx + 1] = 0.0;
    }
    fft_2d_inplace(gx_buf, width, height, col_scratch, false);
    dealias_2d_spectrum(gx_buf, width, height);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_kpz_2d_modes(
    h_hat: &mut [f64],
    grad_sq_hat: &[f64],
    noise_spec: &[f64],
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    nu: f64,
    lambda: f64,
    dt: f64,
) {
    let half_lambda_dt = 0.5 * lambda * dt;
    for y in 0..height {
        let ky = mode_wavenumber(y, height, dy);
        for x in 0..width {
            let kx = mode_wavenumber(x, width, dx);
            let k2 = kx * kx + ky * ky;
            let denom = 1.0 + dt * nu * k2;
            let i0 = 2 * flatten_2d_index(x, y, width);
            h_hat[i0] = (h_hat[i0] + half_lambda_dt * grad_sq_hat[i0] + noise_spec[i0]) / denom;
            h_hat[i0 + 1] =
                (h_hat[i0 + 1] + half_lambda_dt * grad_sq_hat[i0 + 1] + noise_spec[i0 + 1]) / denom;
        }
    }
}

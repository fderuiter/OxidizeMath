// [cite:engine_pde]
use crate::error::PdeError;
use crate::fft::{
    mode_wavenumber, try_dealias_1d_spectrum, try_dealias_2d_spectrum, try_fft_1d_inplace,
    try_fft_2d_inplace,
};
use math_commons::math_kernel::types::flatten_2d_index;
use math_scalars::RealField;

#[inline]
pub(crate) fn pack_real<T: RealField>(real: &[T], complex_out: &mut [T], n: usize) {
    for i in 0..n {
        complex_out[2 * i] = real[i];
        complex_out[2 * i + 1] = T::ZERO;
    }
}

#[inline]
pub(crate) fn unpack_real<T: Copy>(complex_in: &[T], real_out: &mut [T], n: usize) {
    for i in 0..n {
        real_out[i] = complex_in[2 * i];
    }
}

pub(crate) fn try_eval_dealiased_cubic_1d<T: RealField>(
    phi_hat: &[T],
    deal_buf: &mut [T],
    n: usize,
) -> Result<(), PdeError> {
    deal_buf.copy_from_slice(phi_hat);
    try_dealias_1d_spectrum(deal_buf, n)?;
    try_fft_1d_inplace(deal_buf, n, true)?;
    for m in 0..n {
        let p = deal_buf[2 * m];
        deal_buf[2 * m] = p * p * p - p;
        deal_buf[2 * m + 1] = T::ZERO;
    }
    try_fft_1d_inplace(deal_buf, n, false)?;
    try_dealias_1d_spectrum(deal_buf, n)
}

pub(crate) fn try_eval_dealiased_cubic_2d<T: RealField>(
    phi_hat: &[T],
    deal_buf: &mut [T],
    col_scratch: &mut [T],
    width: usize,
    height: usize,
) -> Result<(), PdeError> {
    let total = width * height;
    deal_buf.copy_from_slice(phi_hat);
    try_dealias_2d_spectrum(deal_buf, width, height)?;
    try_fft_2d_inplace(deal_buf, width, height, col_scratch, true)?;
    for idx in 0..total {
        let p = deal_buf[2 * idx];
        deal_buf[2 * idx] = p * p * p - p;
        deal_buf[2 * idx + 1] = T::ZERO;
    }
    try_fft_2d_inplace(deal_buf, width, height, col_scratch, false)?;
    try_dealias_2d_spectrum(deal_buf, width, height)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_cahn_hilliard_1d_modes<T: RealField>(
    phi_hat: &mut [T],
    deal_buf: &[T],
    noise_spec: &[T],
    n: usize,
    dx: T,
    mobility: T,
    kappa: T,
    dt: T,
) {
    for m in 0..n {
        let k = mode_wavenumber(m, n, dx);
        let k2 = k * k;
        let denom = T::ONE + dt * mobility * kappa * k2 * k2;
        let i0 = 2 * m;
        phi_hat[i0] = (phi_hat[i0] - dt * mobility * k2 * deal_buf[i0] + noise_spec[i0]) / denom;
        phi_hat[i0 + 1] =
            (phi_hat[i0 + 1] - dt * mobility * k2 * deal_buf[i0 + 1] + noise_spec[i0 + 1]) / denom;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_cahn_hilliard_2d_modes<T: RealField>(
    phi_hat: &mut [T],
    deal_buf: &[T],
    noise_spec: &[T],
    width: usize,
    height: usize,
    dx: T,
    dy: T,
    mobility: T,
    kappa: T,
    dt: T,
) {
    for y in 0..height {
        let ky = mode_wavenumber(y, height, dy);
        for x in 0..width {
            let kx = mode_wavenumber(x, width, dx);
            let k2 = kx * kx + ky * ky;
            let denom = T::ONE + dt * mobility * kappa * k2 * k2;
            let i0 = 2 * flatten_2d_index(x, y, width);
            phi_hat[i0] =
                (phi_hat[i0] - dt * mobility * k2 * deal_buf[i0] + noise_spec[i0]) / denom;
            phi_hat[i0 + 1] = (phi_hat[i0 + 1] - dt * mobility * k2 * deal_buf[i0 + 1]
                + noise_spec[i0 + 1])
                / denom;
        }
    }
}

pub(crate) fn try_eval_dealiased_grad_sq_1d<T: RealField>(
    h_hat: &[T],
    grad_buf: &mut [T],
    n: usize,
    dx: T,
) -> Result<(), PdeError> {
    grad_buf.copy_from_slice(h_hat);
    try_dealias_1d_spectrum(grad_buf, n)?;
    for m in 0..n {
        let k = if m == n / 2 {
            T::ZERO
        } else {
            mode_wavenumber(m, n, dx)
        };
        let re = grad_buf[2 * m];
        let im = grad_buf[2 * m + 1];
        grad_buf[2 * m] = -k * im;
        grad_buf[2 * m + 1] = k * re;
    }
    try_fft_1d_inplace(grad_buf, n, true)?;
    for m in 0..n {
        let gx = grad_buf[2 * m];
        grad_buf[2 * m] = gx * gx;
        grad_buf[2 * m + 1] = T::ZERO;
    }
    try_fft_1d_inplace(grad_buf, n, false)?;
    try_dealias_1d_spectrum(grad_buf, n)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_eval_dealiased_grad_sq_2d<T: RealField>(
    h_hat: &[T],
    gx_buf: &mut [T],
    gy_buf: &mut [T],
    col_scratch: &mut [T],
    width: usize,
    height: usize,
    dx: T,
    dy: T,
) -> Result<(), PdeError> {
    gx_buf.copy_from_slice(h_hat);
    try_dealias_2d_spectrum(gx_buf, width, height)?;
    gy_buf.copy_from_slice(gx_buf);

    for y in 0..height {
        let ky = if y == height / 2 {
            T::ZERO
        } else {
            mode_wavenumber(y, height, dy)
        };
        for x in 0..width {
            let kx = if x == width / 2 {
                T::ZERO
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

    try_fft_2d_inplace(gx_buf, width, height, col_scratch, true)?;
    try_fft_2d_inplace(gy_buf, width, height, col_scratch, true)?;
    let total = width * height;
    for idx in 0..total {
        let gx = gx_buf[2 * idx];
        let gy = gy_buf[2 * idx];
        gx_buf[2 * idx] = gx * gx + gy * gy;
        gx_buf[2 * idx + 1] = T::ZERO;
    }
    try_fft_2d_inplace(gx_buf, width, height, col_scratch, false)?;
    try_dealias_2d_spectrum(gx_buf, width, height)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_kpz_1d_modes<T: RealField>(
    h_hat: &mut [T],
    grad_buf: &[T],
    noise_spec: &[T],
    n: usize,
    dx: T,
    nu: T,
    lambda: T,
    dt: T,
) {
    let half_lambda_dt = T::HALF * lambda * dt;
    for m in 0..n {
        let k = mode_wavenumber(m, n, dx);
        let denom = T::ONE + dt * nu * k * k;
        let i0 = 2 * m;
        h_hat[i0] = (h_hat[i0] + half_lambda_dt * grad_buf[i0] + noise_spec[i0]) / denom;
        h_hat[i0 + 1] =
            (h_hat[i0 + 1] + half_lambda_dt * grad_buf[i0 + 1] + noise_spec[i0 + 1]) / denom;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_kpz_2d_modes<T: RealField>(
    h_hat: &mut [T],
    grad_sq_hat: &[T],
    noise_spec: &[T],
    width: usize,
    height: usize,
    dx: T,
    dy: T,
    nu: T,
    lambda: T,
    dt: T,
) {
    let half_lambda_dt = T::HALF * lambda * dt;
    for y in 0..height {
        let ky = mode_wavenumber(y, height, dy);
        for x in 0..width {
            let kx = mode_wavenumber(x, width, dx);
            let k2 = kx * kx + ky * ky;
            let denom = T::ONE + dt * nu * k2;
            let i0 = 2 * flatten_2d_index(x, y, width);
            h_hat[i0] = (h_hat[i0] + half_lambda_dt * grad_sq_hat[i0] + noise_spec[i0]) / denom;
            h_hat[i0 + 1] =
                (h_hat[i0 + 1] + half_lambda_dt * grad_sq_hat[i0 + 1] + noise_spec[i0 + 1]) / denom;
        }
    }
}

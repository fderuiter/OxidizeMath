// [cite:engine_pde]
use crate::error::PdeError;
use math_commons::math_kernel::types::{Scalar, flatten_2d_index};

/// Computes the signed integer Fourier mode $\tilde{m} \in (-N/2, N/2]$ for DFT index `m` of radix-2 size `n`.
#[inline]
pub const fn signed_mode_index(m: usize, n: usize) -> isize {
    if m <= n / 2 {
        m as isize
    } else {
        (m as isize) - (n as isize)
    }
}

/// Computes the physical wavenumber $k_m = \frac{2\pi}{n \Delta x} \tilde{m}$ (zeroing Nyquist $m = n / 2$ for odd derivatives).
#[inline]
pub fn mode_wavenumber(m: usize, n: usize, dx: f64) -> f64 {
    let signed = signed_mode_index(m, n) as f64;
    (<f64 as Scalar>::TAU / ((n as f64) * dx)) * signed
}

/// Fallible zero-allocation iterative radix-2 Cooley-Tukey 1D FFT/IFFT over interleaved `(re, im)` slice `buf`.
///
/// # Errors
/// Returns [`PdeError::NonRadix2Dimension`] if `n == 0` or `!n.is_power_of_two()`, or
/// [`PdeError::BufferTooSmall`] if `buf.len() < 2 * n`.
#[verified_engine::verified]
pub fn try_fft_1d_inplace(buf: &mut [f64], n: usize, inverse: bool) -> Result<(), PdeError> {
    if n == 0 || !n.is_power_of_two() {
        return Err(PdeError::NonRadix2Dimension { axis: "n", size: n });
    }
    let required = n
        .checked_mul(2)
        .ok_or(PdeError::InvalidGridDimension { axis: "n", size: n })?;
    if buf.len() < required {
        return Err(PdeError::BufferTooSmall {
            buffer: "buf",
            required,
            actual: buf.len(),
        });
    }

    bit_reverse_permute(buf, n);
    run_cooley_tukey_butterflies(buf, n, inverse);

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for v in &mut buf[..required] {
            *v *= inv_n;
        }
    }
    debug_assert!(buf.len() >= required);
    Ok(())
}

/// Zero-allocation iterative radix-2 Cooley-Tukey 1D FFT/IFFT over interleaved `(re, im)` slice `buf` (`len >= 2 * n`).
///
/// # Panics
/// Panics if `n == 0`, `!n.is_power_of_two()`, or `buf.len() < 2 * n`; prefer [`try_fft_1d_inplace`].
#[verified_engine::verified]
pub fn fft_1d_inplace(buf: &mut [f64], n: usize, inverse: bool) {
    try_fft_1d_inplace(buf, n, inverse).expect("fft_1d_inplace preconditions violated");
}

#[verified_engine::verified]
fn run_cooley_tukey_butterflies(buf: &mut [f64], n: usize, inverse: bool) {
    debug_assert!(n.is_power_of_two() && buf.len() >= 2 * n);
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * <f64 as Scalar>::TAU / (len as f64);
        let w_step_re = <f64 as Scalar>::cos(angle);
        let w_step_im = <f64 as Scalar>::sin(angle);

        let mut base = 0;
        while base < n {
            let mut w_re = 1.0_f64;
            let mut w_im = 0.0_f64;
            for j in 0..half {
                let i0 = 2 * (base + j);
                let i1 = 2 * (base + j + half);
                let u_re = buf[i0];
                let u_im = buf[i0 + 1];
                let v_re = buf[i1] * w_re - buf[i1 + 1] * w_im;
                let v_im = buf[i1] * w_im + buf[i1 + 1] * w_re;
                buf[i0] = u_re + v_re;
                buf[i0 + 1] = u_im + v_im;
                buf[i1] = u_re - v_re;
                buf[i1 + 1] = u_im - v_im;

                let next_re = w_re * w_step_re - w_im * w_step_im;
                let next_im = w_re * w_step_im + w_im * w_step_re;
                w_re = next_re;
                w_im = next_im;
            }
            base += len;
        }
        len *= 2;
    }
}

fn bit_reverse_permute(buf: &mut [f64], n: usize) {
    let mut j = 0_usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(2 * i, 2 * j);
            buf.swap(2 * i + 1, 2 * j + 1);
        }
    }
}

/// Fallible zero-allocation radix-2 2D FFT/IFFT over interleaved `(re, im)` slice `buf`.
///
/// # Errors
/// Returns [`PdeError`] if `width` or `height` is not a power of two or `buf`/`col_scratch` is undersized.
#[verified_engine::verified]
pub fn try_fft_2d_inplace(
    buf: &mut [f64],
    width: usize,
    height: usize,
    col_scratch: &mut [f64],
    inverse: bool,
) -> Result<(), PdeError> {
    if width == 0 || !width.is_power_of_two() {
        return Err(PdeError::NonRadix2Dimension {
            axis: "width",
            size: width,
        });
    }
    if height == 0 || !height.is_power_of_two() {
        return Err(PdeError::NonRadix2Dimension {
            axis: "height",
            size: height,
        });
    }
    let cells = width
        .checked_mul(height)
        .and_then(|c| c.checked_mul(2))
        .ok_or(PdeError::InvalidGridDimension {
            axis: "grid",
            size: usize::MAX,
        })?;
    if buf.len() < cells {
        return Err(PdeError::BufferTooSmall {
            buffer: "buf",
            required: cells,
            actual: buf.len(),
        });
    }
    let req_col = height * 2;
    if col_scratch.len() < req_col {
        return Err(PdeError::BufferTooSmall {
            buffer: "col_scratch",
            required: req_col,
            actual: col_scratch.len(),
        });
    }

    for y in 0..height {
        let row_start = 2 * flatten_2d_index(0, y, width);
        let row_end = row_start + 2 * width;
        try_fft_1d_inplace(&mut buf[row_start..row_end], width, inverse)?;
    }

    for x in 0..width {
        for y in 0..height {
            let idx = 2 * flatten_2d_index(x, y, width);
            col_scratch[2 * y] = buf[idx];
            col_scratch[2 * y + 1] = buf[idx + 1];
        }
        try_fft_1d_inplace(&mut col_scratch[..req_col], height, inverse)?;
        for y in 0..height {
            let idx = 2 * flatten_2d_index(x, y, width);
            buf[idx] = col_scratch[2 * y];
            buf[idx + 1] = col_scratch[2 * y + 1];
        }
    }
    debug_assert!(buf.len() >= cells);
    Ok(())
}

/// Zero-allocation radix-2 2D FFT/IFFT over interleaved `(re, im)` slice `buf` (`len >= 2 * width * height`)
/// using `col_scratch` (`len >= 2 * height`).
///
/// # Panics
/// Panics if `width`/`height` are not powers of two or buffers are undersized; prefer [`try_fft_2d_inplace`].
#[verified_engine::verified]
pub fn fft_2d_inplace(
    buf: &mut [f64],
    width: usize,
    height: usize,
    col_scratch: &mut [f64],
    inverse: bool,
) {
    try_fft_2d_inplace(buf, width, height, col_scratch, inverse)
        .expect("fft_2d_inplace preconditions violated");
}

/// Fallibly applies the Orszag 2/3-rule spectral dealiasing mask to a 1D interleaved spectrum.
///
/// # Errors
/// Returns [`PdeError`] if `n == 0` or `spec.len() < 2 * n`.
#[verified_engine::verified]
pub fn try_dealias_1d_spectrum(spec: &mut [f64], n: usize) -> Result<(), PdeError> {
    if n == 0 {
        return Err(PdeError::InvalidGridDimension { axis: "n", size: 0 });
    }
    let required = n
        .checked_mul(2)
        .ok_or(PdeError::InvalidGridDimension { axis: "n", size: n })?;
    if spec.len() < required {
        return Err(PdeError::BufferTooSmall {
            buffer: "spec",
            required,
            actual: spec.len(),
        });
    }
    for m in 0..n {
        let k_abs = signed_mode_index(m, n).unsigned_abs();
        if 3 * k_abs >= n {
            spec[2 * m] = 0.0;
            spec[2 * m + 1] = 0.0;
        }
    }
    debug_assert!(spec.len() >= required);
    Ok(())
}

/// Applies the Orszag 2/3-rule spectral dealiasing mask to a 1D interleaved spectrum of radix-2 size `n`,
/// zeroing all modes with $3 |\tilde{m}| \ge n$ (including Nyquist).
///
/// # Panics
/// Panics if `n == 0` or `spec.len() < 2 * n`; prefer [`try_dealias_1d_spectrum`].
#[verified_engine::verified]
pub fn dealias_1d_spectrum(spec: &mut [f64], n: usize) {
    try_dealias_1d_spectrum(spec, n).expect("dealias_1d_spectrum preconditions violated");
}

/// Fallibly applies the Orszag 2/3-rule spectral dealiasing mask to a 2D interleaved spectrum.
///
/// # Errors
/// Returns [`PdeError`] if `width == 0`, `height == 0`, or `spec.len() < 2 * width * height`.
#[verified_engine::verified]
pub fn try_dealias_2d_spectrum(
    spec: &mut [f64],
    width: usize,
    height: usize,
) -> Result<(), PdeError> {
    if width == 0 {
        return Err(PdeError::InvalidGridDimension {
            axis: "width",
            size: 0,
        });
    }
    if height == 0 {
        return Err(PdeError::InvalidGridDimension {
            axis: "height",
            size: 0,
        });
    }
    let required = width
        .checked_mul(height)
        .and_then(|c| c.checked_mul(2))
        .ok_or(PdeError::InvalidGridDimension {
            axis: "grid",
            size: usize::MAX,
        })?;
    if spec.len() < required {
        return Err(PdeError::BufferTooSmall {
            buffer: "spec",
            required,
            actual: spec.len(),
        });
    }
    for y in 0..height {
        let ky_cut = 3 * signed_mode_index(y, height).unsigned_abs() >= height;
        for x in 0..width {
            let kx_cut = 3 * signed_mode_index(x, width).unsigned_abs() >= width;
            if kx_cut || ky_cut {
                let idx = 2 * flatten_2d_index(x, y, width);
                spec[idx] = 0.0;
                spec[idx + 1] = 0.0;
            }
        }
    }
    debug_assert!(spec.len() >= required);
    Ok(())
}

/// Applies the Orszag 2/3-rule spectral dealiasing mask to a 2D interleaved spectrum (`width x height`),
/// zeroing modes where $3 |\tilde{m}_x| \ge \text{width}$ or $3 |\tilde{m}_y| \ge \text{height}$.
///
/// # Panics
/// Panics if `width == 0`, `height == 0`, or `spec` is undersized; prefer [`try_dealias_2d_spectrum`].
#[verified_engine::verified]
pub fn dealias_2d_spectrum(spec: &mut [f64], width: usize, height: usize) {
    try_dealias_2d_spectrum(spec, width, height)
        .expect("dealias_2d_spectrum preconditions violated");
}

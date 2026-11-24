// [cite:engine_pde]
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

/// Zero-allocation iterative radix-2 Cooley-Tukey 1D FFT/IFFT over interleaved `(re, im)` slice `buf` (`len >= 2 * n`).
///
/// # Panics
/// Panics if `n == 0`, `!n.is_power_of_two()`, or `buf.len() < 2 * n`.
#[verified_engine::verified]
pub fn fft_1d_inplace(buf: &mut [f64], n: usize, inverse: bool) {
    assert!(
        n > 0 && n.is_power_of_two(),
        "FFT size n must be a power of two"
    );
    assert!(
        buf.len() >= 2 * n,
        "interleaved complex buffer shorter than 2 * n"
    );

    bit_reverse_permute(buf, n);

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

    if inverse {
        let inv_n = 1.0 / (n as f64);
        for v in &mut buf[..2 * n] {
            *v *= inv_n;
        }
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

/// Zero-allocation radix-2 2D FFT/IFFT over interleaved `(re, im)` slice `buf` (`len >= 2 * width * height`)
/// using `col_scratch` (`len >= 2 * height`).
#[verified_engine::verified]
pub fn fft_2d_inplace(
    buf: &mut [f64],
    width: usize,
    height: usize,
    col_scratch: &mut [f64],
    inverse: bool,
) {
    assert!(
        width > 0 && width.is_power_of_two(),
        "width must be power of two"
    );
    assert!(
        height > 0 && height.is_power_of_two(),
        "height must be power of two"
    );
    assert!(buf.len() >= 2 * width * height && col_scratch.len() >= 2 * height);

    for y in 0..height {
        let row_start = 2 * flatten_2d_index(0, y, width);
        let row_end = row_start + 2 * width;
        fft_1d_inplace(&mut buf[row_start..row_end], width, inverse);
    }

    for x in 0..width {
        for y in 0..height {
            let idx = 2 * flatten_2d_index(x, y, width);
            col_scratch[2 * y] = buf[idx];
            col_scratch[2 * y + 1] = buf[idx + 1];
        }
        fft_1d_inplace(&mut col_scratch[..2 * height], height, inverse);
        for y in 0..height {
            let idx = 2 * flatten_2d_index(x, y, width);
            buf[idx] = col_scratch[2 * y];
            buf[idx + 1] = col_scratch[2 * y + 1];
        }
    }
}

/// Applies the Orszag 2/3-rule spectral dealiasing mask to a 1D interleaved spectrum of radix-2 size `n`,
/// zeroing all modes with $3 |\tilde{m}| \ge n$ (including Nyquist).
#[verified_engine::verified]
pub fn dealias_1d_spectrum(spec: &mut [f64], n: usize) {
    assert!(n > 0 && spec.len() >= 2 * n);
    for m in 0..n {
        let k_abs = signed_mode_index(m, n).unsigned_abs();
        if 3 * k_abs >= n {
            spec[2 * m] = 0.0;
            spec[2 * m + 1] = 0.0;
        }
    }
}

/// Applies the Orszag 2/3-rule spectral dealiasing mask to a 2D interleaved spectrum (`width x height`),
/// zeroing modes where $3 |\tilde{m}_x| \ge \text{width}$ or $3 |\tilde{m}_y| \ge \text{height}$.
#[verified_engine::verified]
pub fn dealias_2d_spectrum(spec: &mut [f64], width: usize, height: usize) {
    assert!(width > 0 && height > 0 && spec.len() >= 2 * width * height);
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
}

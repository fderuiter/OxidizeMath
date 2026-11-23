//! Stack-buffered `32x32` radix-2 butterfly and Hermitian 2-row real 2D convolution helpers
//! for [`super::fft_probe`].

use super::fft_probe::{PROBE_CELLS, PROBE_SIDE};
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex;
use std::f64::consts::FRAC_1_SQRT_2;

pub(super) const HERMITIAN_COLS: usize = PROBE_SIDE / 2 + 1;
pub(super) const HERMITIAN_CELLS: usize = HERMITIAN_COLS * PROBE_SIDE;

const C1: f64 = 0.980_785_280_403_230_4;
const S1: f64 = 0.195_090_322_016_128_25;
const C2: f64 = 0.923_879_532_511_286_7;
const S2: f64 = 0.382_683_432_365_089_8;
const C3: f64 = 0.831_469_612_302_545_2;
const S3: f64 = 0.555_570_233_019_602_2;

/// Precomputed forward radix-2 twiddle factors $\cos(2\pi k/32), -\sin(2\pi k/32)$ for $k \in 0..16$.
const TWIDDLE_32_FWD: [(f64, f64); 16] = [
    (1.0, 0.0),
    (C1, -S1),
    (C2, -S2),
    (C3, -S3),
    (FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
    (S3, -C3),
    (S2, -C2),
    (S1, -C1),
    (0.0, -1.0),
    (-S1, -C1),
    (-S2, -C2),
    (-S3, -C3),
    (-FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
    (-C3, -S3),
    (-C2, -S2),
    (-C1, -S1),
];

#[inline]
pub(super) fn fft_1d_32_in_place(row: &mut [Complex<f64>], inverse: bool) {
    let mut idx = 0usize;
    while idx < PROBE_SIDE {
        let rev = ((idx as u8).reverse_bits() >> 3) as usize;
        if idx < rev {
            row.swap(idx, rev);
        }
        idx += 1;
    }
    let sign = if inverse { -1.0 } else { 1.0 };
    let mut len = 2usize;
    while len <= PROBE_SIDE {
        let half = len / 2;
        let stride = PROBE_SIDE / len;
        let mut base = 0usize;
        while base < PROBE_SIDE {
            let mut j = 0usize;
            while j < half {
                let (tw_re, raw_im) = TWIDDLE_32_FWD[j * stride];
                let tw_im = sign * raw_im;
                let i0 = base + j;
                let i1 = i0 + half;
                let u = row[i0];
                let t = row[i1];
                let v_re = t.re * tw_re - t.im * tw_im;
                let v_im = t.re * tw_im + t.im * tw_re;
                row[i0] = Complex {
                    re: u.re + v_re,
                    im: u.im + v_im,
                };
                row[i1] = Complex {
                    re: u.re - v_re,
                    im: u.im - v_im,
                };
                j += 1;
            }
            base += len;
        }
        len *= 2;
    }
}

#[inline]
fn transpose_square_32(grid: &mut [Complex<f64>; PROBE_CELLS]) {
    let mut y = 0usize;
    while y < PROBE_SIDE {
        let mut x = y + 1;
        while x < PROBE_SIDE {
            grid.swap(
                flatten_2d_index(x, y, PROBE_SIDE),
                flatten_2d_index(y, x, PROBE_SIDE),
            );
            x += 1;
        }
        y += 1;
    }
}

#[inline]
pub(super) fn fft_2d_32_in_place(grid: &mut [Complex<f64>; PROBE_CELLS], inverse: bool) {
    let mut r = 0usize;
    while r < PROBE_SIDE {
        let start = r * PROBE_SIDE;
        fft_1d_32_in_place(&mut grid[start..start + PROBE_SIDE], inverse);
        r += 1;
    }
    transpose_square_32(grid);
    let mut c = 0usize;
    while c < PROBE_SIDE {
        let start = c * PROBE_SIDE;
        fft_1d_32_in_place(&mut grid[start..start + PROBE_SIDE], inverse);
        c += 1;
    }
    transpose_square_32(grid);
}

#[verified_engine::verified]
pub(super) fn convolve_real_hermitian_32(
    state: &[f64; PROBE_CELLS],
    kernel_hat: &[Complex<f64>; PROBE_CELLS],
    tilde_cols: &mut [Complex<f64>; HERMITIAN_CELLS],
    u_out: &mut [f64; PROBE_CELLS],
) {
    let mut row = [Complex { re: 0.0, im: 0.0 }; PROBE_SIDE];
    let mut pair = 0usize;
    while pair < PROBE_SIDE / 2 {
        let r0 = 2 * pair;
        let r1 = r0 + 1;
        let s0 = &state[r0 * PROBE_SIDE..(r0 + 1) * PROBE_SIDE];
        let s1 = &state[r1 * PROBE_SIDE..(r1 + 1) * PROBE_SIDE];
        let mut x = 0usize;
        while x < PROBE_SIDE {
            row[x] = Complex {
                re: s0[x],
                im: s1[x],
            };
            x += 1;
        }
        fft_1d_32_in_place(&mut row, false);
        let mut kx = 0usize;
        while kx < HERMITIAN_COLS {
            let z = row[kx];
            let zc = row[(PROBE_SIDE - kx) & 31];
            let col_slice = &mut tilde_cols[kx * PROBE_SIDE..(kx + 1) * PROBE_SIDE];
            col_slice[r0] = Complex {
                re: 0.5 * (z.re + zc.re),
                im: 0.5 * (z.im - zc.im),
            };
            col_slice[r1] = Complex {
                re: 0.5 * (z.im + zc.im),
                im: -0.5 * (z.re - zc.re),
            };
            kx += 1;
        }
        pair += 1;
    }
    convolve_hermitian_cols_and_unpack(kernel_hat, tilde_cols, &mut row, u_out);
}

#[inline]
fn convolve_hermitian_cols_and_unpack(
    kernel_hat: &[Complex<f64>; PROBE_CELLS],
    tilde_cols: &mut [Complex<f64>; HERMITIAN_CELLS],
    row: &mut [Complex<f64>; PROBE_SIDE],
    u_out: &mut [f64; PROBE_CELLS],
) {
    let mut kx = 0usize;
    while kx < HERMITIAN_COLS {
        let col = &mut tilde_cols[kx * PROBE_SIDE..(kx + 1) * PROBE_SIDE];
        fft_1d_32_in_place(col, false);
        let mut ky = 0usize;
        while ky < PROBE_SIDE {
            let f = col[ky];
            let h = kernel_hat[flatten_2d_index(kx, ky, PROBE_SIDE)];
            col[ky] = Complex {
                re: f.re * h.re - f.im * h.im,
                im: f.re * h.im + f.im * h.re,
            };
            ky += 1;
        }
        fft_1d_32_in_place(col, true);
        kx += 1;
    }

    let mut pair = 0usize;
    while pair < PROBE_SIDE / 2 {
        let r0 = 2 * pair;
        let r1 = r0 + 1;
        let mut k = 0usize;
        while k < HERMITIAN_COLS {
            let col_slice = &tilde_cols[k * PROBE_SIDE..(k + 1) * PROBE_SIDE];
            let u0 = col_slice[r0];
            let u1 = col_slice[r1];
            row[k] = Complex {
                re: u0.re - u1.im,
                im: u0.im + u1.re,
            };
            if k > 0 && k < PROBE_SIDE / 2 {
                row[PROBE_SIDE - k] = Complex {
                    re: u0.re + u1.im,
                    im: u1.re - u0.im,
                };
            }
            k += 1;
        }
        fft_1d_32_in_place(row, true);
        let (head, tail) = u_out.split_at_mut(r1 * PROBE_SIDE);
        let dst0 = &mut head[r0 * PROBE_SIDE..(r0 + 1) * PROBE_SIDE];
        let dst1 = &mut tail[..PROBE_SIDE];
        let mut x = 0usize;
        while x < PROBE_SIDE {
            dst0[x] = row[x].re;
            dst1[x] = row[x].im;
            x += 1;
        }
        pair += 1;
    }
}

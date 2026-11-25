// [cite:mri]
//! Low-level cache-blocked transpose, shift, and shape-check kernels for 2D MRI reconstruction.

use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use nalgebra::DMatrix;
use num_complex::Complex;

#[verified_engine::verified]
pub(super) fn apply_kspace_modulation(
    buf: &mut [Complex<f64>],
    mask: &[f64],
    spikes: &[Complex<f64>],
    out_kspace: &mut [Complex<f64>],
) {
    debug_assert_eq!(buf.len(), out_kspace.len());
    for (((b, &m), &sp), out) in buf.iter_mut().zip(mask).zip(spikes).zip(out_kspace) {
        let v = *b * m + sp;
        *b = v;
        *out = v;
    }
}

pub(super) fn scale_complex_slice(slice: &mut [Complex<f64>], scale: f64) {
    for elem in slice {
        *elem *= scale;
    }
}

/// Cache-blocked `8x8` transpose of a column-major `rows x cols` slice into a column-major `cols x rows` slice.
#[verified_engine::verified]
pub(super) fn transpose_blocked(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(src.len(), dst.len());
    if rows == 0 || cols == 0 {
        return;
    }
    const TILE: usize = 8;
    let mut r0 = 0;
    for dst_band in dst.chunks_mut(cols * TILE) {
        let r_len = dst_band.len() / cols;
        for (c, src_col) in src.chunks_exact(rows).enumerate() {
            for (&val, dst_row) in src_col[r0..r0 + r_len]
                .iter()
                .zip(dst_band.chunks_exact_mut(cols))
            {
                dst_row[c] = val;
            }
        }
        r0 += r_len;
    }
}

pub(super) fn shift_2d_columns(
    src: &DMatrix<Complex<f64>>,
    dst: &mut DMatrix<Complex<f64>>,
    shift_r: usize,
    shift_c: usize,
) {
    let rows = src.nrows();
    let cols = src.ncols();
    let split_r = rows - shift_r;
    let src_slice = src.as_slice();
    let dst_slice = dst.as_mut_slice();
    for c in 0..cols {
        let dc = (c + shift_c) % cols;
        let src_col = &src_slice[c * rows..(c + 1) * rows];
        let dst_col = &mut dst_slice[dc * rows..(dc + 1) * rows];
        dst_col[shift_r..].copy_from_slice(&src_col[..split_r]);
        dst_col[..shift_r].copy_from_slice(&src_col[split_r..]);
    }
}

#[inline]
pub(super) fn check_len(expected: usize, actual: usize) -> Result<(), MathError> {
    if actual != expected {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(expected),
            actual: Dimension(actual),
        });
    }
    Ok(())
}

#[inline]
pub(super) fn check_mat_shape(
    mat: &DMatrix<Complex<f64>>,
    rows: usize,
    cols: usize,
) -> Result<(), MathError> {
    check_len(rows, mat.nrows())?;
    check_len(cols, mat.ncols())
}

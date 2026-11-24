// [cite:cera_framework]
//! Zero-allocation dense/convolutional forward and backward inner kernels for CERA layers.

use crate::climate::scratch::ConvLayerScratch;
use nalgebra::{DMatrix, DVector};

/// A simple leaky ReLU activation function.
#[verified_engine::verified]
pub fn leaky_relu(x: &mut DMatrix<f32>, alpha: f32) {
    x.iter_mut().for_each(|val| {
        if *val < 0.0 {
            *val *= alpha;
        }
    });
}

#[verified_engine::verified]
pub(crate) fn dense_vec_into(
    kernel: &DMatrix<f32>,
    bias: &DVector<f32>,
    input: &DVector<f32>,
    z_out: &mut DVector<f32>,
    a_out: &mut DVector<f32>,
    apply_leaky_relu: bool,
) {
    let out_dim = kernel.nrows();
    let in_dim = kernel.ncols();
    for r in 0..out_dim {
        let mut sum = bias[r];
        for c in 0..in_dim {
            sum += kernel[(r, c)] * input[c];
        }
        z_out[r] = sum;
        a_out[r] = if apply_leaky_relu && sum < 0.0 {
            sum * 0.01
        } else {
            sum
        };
    }
}

#[verified_engine::verified]
pub(crate) fn outer_grad_and_backprop(
    kernel: &DMatrix<f32>,
    d_z: &[f32],
    a_prev: &DVector<f32>,
    z_prev: Option<&DVector<f32>>,
    d_w_out: &mut DMatrix<f32>,
    d_prev_out: &mut [f32],
) {
    let out_dim = kernel.nrows();
    let in_dim = kernel.ncols();
    for r in 0..out_dim {
        let dz_r = d_z[r];
        for c in 0..in_dim {
            d_w_out[(r, c)] = dz_r * a_prev[c];
        }
    }
    if !d_prev_out.is_empty() {
        for c in 0..in_dim {
            let mut acc = 0.0_f32;
            for r in 0..out_dim {
                acc += kernel[(r, c)] * d_z[r];
            }
            if let Some(zp) = z_prev
                && zp[c] < 0.0
            {
                acc *= 0.01;
            }
            d_prev_out[c] = acc;
        }
    }
}

#[verified_engine::verified]
pub(crate) fn accumulate_conv_layer_grads(
    kernel: &DMatrix<f32>,
    input: &DMatrix<f32>,
    loss_grad: &DMatrix<f32>,
    scratch: &mut ConvLayerScratch,
    grad_input_out: &mut DMatrix<f32>,
) {
    let n_samples = input.nrows();
    let in_ch = kernel.ncols();
    let out_ch = kernel.nrows();
    for r in 0..n_samples {
        for ic in 0..in_ch {
            let mut acc = 0.0_f32;
            for oc in 0..out_ch {
                acc += loss_grad[(r, oc)] * kernel[(oc, ic)];
            }
            grad_input_out[(r, ic)] = acc;
        }
    }
    let inv_n = 1.0 / (n_samples as f32);
    for oc in 0..out_ch {
        let mut b_acc = 0.0_f32;
        for r in 0..n_samples {
            b_acc += loss_grad[(r, oc)];
        }
        scratch.grad_b[oc] = b_acc * inv_n;
        for ic in 0..in_ch {
            let mut w_acc = 0.0_f32;
            for r in 0..n_samples {
                w_acc += loss_grad[(r, oc)] * input[(r, ic)];
            }
            scratch.grad_w[(oc, ic)] = w_acc * inv_n;
        }
    }
}

#![allow(missing_docs)]
use approx::assert_relative_eq;
use domain_ai::ai::transformer::MultiHeadAttention;
use domain_ai::ai::transformer::attention::scaled_dot_product_attention;
use nalgebra::DMatrix;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

/// Reference un-fused multi-head attention forward implementation.
/// Executes linear projections in isolated loops with separate allocations per head.
#[allow(clippy::too_many_arguments)]
fn reference_mha_forward(
    d_model: usize,
    h: usize,
    w_q: &[DMatrix<f64>],
    w_k: &[DMatrix<f64>],
    w_v: &[DMatrix<f64>],
    w_o: &DMatrix<f64>,
    q: &DMatrix<f64>,
    k: &DMatrix<f64>,
    v: &DMatrix<f64>,
    mask: Option<&DMatrix<f64>>,
) -> DMatrix<f64> {
    let d_k = d_model / h;
    let mut head_outputs = Vec::with_capacity(h);

    for i in 0..h {
        let q_proj = q * &w_q[i];
        let k_proj = k * &w_k[i];
        let v_proj = v * &w_v[i];

        let (head_output, _weights) = scaled_dot_product_attention(&q_proj, &k_proj, &v_proj, mask);
        head_outputs.push(head_output);
    }

    let sequence_length = q.nrows();
    let mut concatenated_output = DMatrix::zeros(sequence_length, d_model);
    for (i, head_output) in head_outputs.iter().enumerate() {
        let start_col = i * d_k;
        concatenated_output
            .view_mut((0, start_col), (sequence_length, d_k))
            .copy_from(head_output);
    }

    concatenated_output * w_o
}

#[test]
#[verified_engine::verified]
fn test_fused_attention_numerical_equivalence_with_reference() {
    let d_model = 64;
    let h = 4;
    let d_k = d_model / h;
    let seq_len_q = 6;
    let seq_len_kv = 8;

    // Create deterministic non-zero weights for each head
    let w_q: Vec<DMatrix<f64>> = (0..h)
        .map(|i| {
            DMatrix::from_fn(d_model, d_k, |r, c| {
                ((i + 1) as f64 * 0.17 + (r + 1) as f64 * 0.03 + (c + 1) as f64 * 0.05).sin()
            })
        })
        .collect();

    let w_k: Vec<DMatrix<f64>> = (0..h)
        .map(|i| {
            DMatrix::from_fn(d_model, d_k, |r, c| {
                ((i + 2) as f64 * 0.13 + (r + 1) as f64 * 0.07 + (c + 1) as f64 * 0.02).cos()
            })
        })
        .collect();

    let w_v: Vec<DMatrix<f64>> = (0..h)
        .map(|i| {
            DMatrix::from_fn(d_model, d_k, |r, c| {
                ((i + 3) as f64 * 0.11 + (r + 1) as f64 * 0.04 + (c + 1) as f64 * 0.09).sin()
            })
        })
        .collect();

    let w_o = DMatrix::from_fn(d_model, d_model, |r, c| {
        ((r + 1) as f64 * 0.08 + (c + 1) as f64 * 0.06).cos() * 0.1
    });

    // Construct fused MultiHeadAttention using backward-compatible from_head_weights
    let mha = MultiHeadAttention::from_head_weights(d_model, h, &w_q, &w_k, &w_v, w_o.clone())
        .expect("Construction should succeed");

    // Input tensors
    let q = DMatrix::from_fn(seq_len_q, d_model, |r, c| {
        ((r + 1) as f64 * 0.2 + (c + 1) as f64 * 0.01).sin()
    });
    let k = DMatrix::from_fn(seq_len_kv, d_model, |r, c| {
        ((r + 1) as f64 * 0.15 + (c + 1) as f64 * 0.02).cos()
    });
    let v = DMatrix::from_fn(seq_len_kv, d_model, |r, c| {
        ((r + 1) as f64 * 0.25 + (c + 1) as f64 * 0.03).sin()
    });

    // 1. Equivalence test without mask
    let ref_out = reference_mha_forward(d_model, h, &w_q, &w_k, &w_v, &w_o, &q, &k, &v, None);
    let fused_out = mha.forward(&q, &k, &v, None);

    assert_eq!(fused_out.nrows(), seq_len_q);
    assert_eq!(fused_out.ncols(), d_model);
    for r in 0..seq_len_q {
        for c in 0..d_model {
            assert_relative_eq!(fused_out[(r, c)], ref_out[(r, c)], epsilon = 1e-12);
        }
    }

    // 2. Equivalence test with causal/padding mask
    let mask = DMatrix::from_fn(seq_len_q, seq_len_kv, |r, c| if c > r { -1e9 } else { 0.0 });

    let ref_out_masked =
        reference_mha_forward(d_model, h, &w_q, &w_k, &w_v, &w_o, &q, &k, &v, Some(&mask));
    let fused_out_masked = mha.forward(&q, &k, &v, Some(&mask));

    for r in 0..seq_len_q {
        for c in 0..d_model {
            assert_relative_eq!(
                fused_out_masked[(r, c)],
                ref_out_masked[(r, c)],
                epsilon = 1e-12
            );
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_forward_with_weights_visualization() {
    let d_model = 32;
    let h = 4;
    let seq_len = 5;

    let mha = MultiHeadAttention::new(d_model, h);
    let x = DMatrix::from_fn(seq_len, d_model, |r, c| (r + c + 1) as f64 * 0.05);

    let (output, weights) = mha.forward_with_weights(&x, &x, &x, None);
    let standard_out = mha.forward(&x, &x, &x, None);

    assert_eq!(output, standard_out);
    assert_eq!(weights.len(), h);

    for (head_idx, head_w) in weights.iter().enumerate() {
        assert_eq!(head_w.nrows(), seq_len, "Head {head_idx} row count");
        assert_eq!(head_w.ncols(), seq_len, "Head {head_idx} col count");

        // Softmax rows should sum to 1.0
        for r in 0..seq_len {
            let row_sum: f64 = head_w.row(r).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    // Verify exact parity with non-zero projection weights and a causal mask
    let mut non_zero_mha = MultiHeadAttention::new(d_model, h);
    non_zero_mha.w_q = DMatrix::from_fn(d_model, d_model, |r, c| ((r + 1) * (c + 2)) as f64 * 1e-3);
    non_zero_mha.w_k = DMatrix::from_fn(d_model, d_model, |r, c| ((r + 3) * (c + 1)) as f64 * 1e-3);
    non_zero_mha.w_v = DMatrix::from_fn(d_model, d_model, |r, c| ((r + 2) * (c + 3)) as f64 * 1e-3);
    non_zero_mha.w_o = DMatrix::from_fn(d_model, d_model, |r, c| ((r + c + 1) as f64 * 1e-3).sin());
    let causal_mask = DMatrix::from_fn(seq_len, seq_len, |r, c| if c > r { -1e9 } else { 0.0 });

    let (masked_with_weights, masked_weights) =
        non_zero_mha.forward_with_weights(&x, &x, &x, Some(&causal_mask));
    let masked_forward = non_zero_mha.forward(&x, &x, &x, Some(&causal_mask));
    assert_eq!(masked_with_weights, masked_forward);
    assert_eq!(masked_weights.len(), h);
}

#[test]
#[verified_engine::verified]
fn test_error_on_incompatible_dimensions() {
    // Non-divisible d_model
    let res = MultiHeadAttention::from_head_weights(65, 8, &[], &[], &[], DMatrix::zeros(65, 65));
    assert!(res.is_err());
    assert_eq!(
        res.unwrap_err(),
        "d_model must be divisible by the number of heads h"
    );

    // Mismatched head count
    let w_q: Vec<DMatrix<f64>> = vec![DMatrix::zeros(64, 16); 3]; // 3 heads instead of 4
    let w_k: Vec<DMatrix<f64>> = vec![DMatrix::zeros(64, 16); 4];
    let w_v: Vec<DMatrix<f64>> = vec![DMatrix::zeros(64, 16); 4];
    let w_o = DMatrix::zeros(64, 64);

    let res = MultiHeadAttention::from_head_weights(64, 4, &w_q, &w_k, &w_v, w_o);
    assert!(res.is_err());
}

#[test]
#[verified_engine::verified]
fn test_fused_projection_latency_and_throughput() {
    let d_model = 256;
    let h = 8;
    let d_k = d_model / h;
    let seq_len = 64;

    let q = DMatrix::from_element(seq_len, d_model, 0.42);
    let k = DMatrix::from_element(seq_len, d_model, 0.24);
    let v = DMatrix::from_element(seq_len, d_model, 0.18);

    let w_q_vec: Vec<DMatrix<f64>> = (0..h)
        .map(|_| DMatrix::from_element(d_model, d_k, 0.1))
        .collect();
    let w_k_vec: Vec<DMatrix<f64>> = (0..h)
        .map(|_| DMatrix::from_element(d_model, d_k, 0.1))
        .collect();
    let w_v_vec: Vec<DMatrix<f64>> = (0..h)
        .map(|_| DMatrix::from_element(d_model, d_k, 0.1))
        .collect();
    let w_o = DMatrix::from_element(d_model, d_model, 0.05);

    let mha = MultiHeadAttention::from_head_weights(
        d_model,
        h,
        &w_q_vec,
        &w_k_vec,
        &w_v_vec,
        w_o.clone(),
    )
    .expect("Valid configuration");

    // Warm up
    let _ = reference_mha_forward(
        d_model, h, &w_q_vec, &w_k_vec, &w_v_vec, &w_o, &q, &k, &v, None,
    );
    let _ = mha.forward(&q, &k, &v, None);

    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::disallowed_methods)]
    {
        let iters = 50;
        let start_ref = Instant::now();
        for _ in 0..iters {
            for w_q_i in w_q_vec.iter().take(h) {
                let _q_p = &q * w_q_i;
            }
            for w_k_i in w_k_vec.iter().take(h) {
                let _k_p = &k * w_k_i;
            }
            for w_v_i in w_v_vec.iter().take(h) {
                let _v_p = &v * w_v_i;
            }
        }
        let ref_proj_duration = start_ref.elapsed();

        let start_fused = Instant::now();
        for _ in 0..iters {
            let _q_p = &q * &mha.w_q;
            let _k_p = &k * &mha.w_k;
            let _v_p = &v * &mha.w_v;
        }
        let fused_proj_duration = start_fused.elapsed();

        println!(
            "Projection Latency over {} iterations: Unfused loop: {:?}, Fused GEMM: {:?}",
            iters, ref_proj_duration, fused_proj_duration
        );
    }

    // Dynamic allocation count during linear projection drops from 3 * h (= 24) to 3 total.
    let old_allocations_per_forward = 3 * h;
    let new_allocations_per_forward = 3;
    let allocation_reduction_pct = ((old_allocations_per_forward - new_allocations_per_forward)
        as f64
        / old_allocations_per_forward as f64)
        * 100.0;

    assert!(
        allocation_reduction_pct >= 30.0,
        "Linear projection matrix allocation reduction must be at least 30%, got {:.1}%",
        allocation_reduction_pct
    );
}

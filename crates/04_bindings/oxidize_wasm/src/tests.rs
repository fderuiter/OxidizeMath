//! Host parity, zero-copy pointer stability, and structured diagnostic tests for `oxidize_wasm`.
// [cite:engine_linalg]
// [cite:engine_diffeq]
// [cite:engine_pde]
// [cite:math_commons]

use super::*;
use approx::assert_abs_diff_eq;
use engine_linalg::{OwnedWorkspace, cholesky_solve, lu_solve, qr_solve_least_squares};
use math_diagnostics::Diagnostic;

#[test]
fn test_linalg_workspace_exact_engine_parity_and_zero_copy_in_place() {
    let mut sdk = WasmLinalgWorkspace::try_new(8).expect("workspace");
    let a_spd = [
        6.0, 2.0, 1.0, //
        2.0, 5.0, 2.0, //
        1.0, 2.0, 4.0, //
    ];
    let b = [9.0, 9.0, 7.0];
    let mut x_sdk = [0.0; 3];
    sdk.solve_cholesky(&a_spd, &b, &mut x_sdk, 3)
        .expect("SDK cholesky");

    let mut raw_ws = OwnedWorkspace::with_capacity(64);
    let mut x_ref = [0.0; 3];
    cholesky_solve(&a_spd, &b, &mut x_ref, 3, &mut raw_ws.as_workspace()).expect("ref cholesky");
    for i in 0..3 {
        assert_abs_diff_eq!(x_sdk[i], x_ref[i], epsilon = 1e-12);
    }

    // Verify zero-copy in-place LU solve using internal pointers/slices.
    let mat_ptr_before = sdk.matrix_ptr();
    let sol_ptr_before = sdk.solution_ptr();
    sdk.matrix_slice_mut(9).copy_from_slice(&a_spd);
    sdk.rhs_slice_mut(3).copy_from_slice(&b);
    sdk.solve_lu_in_place(3).expect("in-place LU");
    assert_eq!(sdk.matrix_ptr(), mat_ptr_before);
    assert_eq!(sdk.solution_ptr(), sol_ptr_before);

    let mut x_lu_ref = [0.0; 3];
    let mut piv = [0usize; 3];
    lu_solve(
        &a_spd,
        &b,
        &mut x_lu_ref,
        3,
        &mut piv,
        &mut raw_ws.as_workspace(),
    )
    .expect("ref LU");
    for (&actual, &expected) in sdk.solution_slice(3).iter().zip(x_lu_ref.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-12);
    }

    // Least-squares QR parity (`4 x 2`).
    let a_tall = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
    let b_tall = [1.0, 2.0, 2.9, 4.1];
    let mut x_qr_sdk = [0.0; 2];
    let rank = sdk
        .solve_qr_least_squares(&a_tall, 4, 2, &b_tall, &mut x_qr_sdk)
        .expect("SDK QR");
    assert_eq!(rank, 2);

    let mut x_qr_ref = [0.0; 2];
    let mut col_perm = [0usize; 2];
    qr_solve_least_squares(
        &a_tall,
        4,
        2,
        &b_tall,
        &mut x_qr_ref,
        &mut col_perm,
        &mut raw_ws.as_workspace(),
    )
    .expect("ref QR");
    assert_abs_diff_eq!(x_qr_sdk[0], x_qr_ref[0], epsilon = 1e-12);
    assert_abs_diff_eq!(x_qr_sdk[1], x_qr_ref[1], epsilon = 1e-12);
}

#[test]
fn test_structured_diagnostic_error_fields_without_panic() {
    let mut sdk = WasmLinalgWorkspace::try_new(4).expect("workspace");
    let singular = [1.0, 2.0, 2.0, 4.0];
    let b = [1.0, 2.0];
    let mut x = [0.0; 2];
    let err = sdk
        .solve_lu(&singular, &b, &mut x, 2)
        .expect_err("singular matrix must return structured WasmSdkError");
    assert_eq!(err.code, "E_LINALG_SINGULAR");
    assert_eq!(err.error_type, "LinalgError");
    assert_eq!(err.field_value("code"), Some("LINALG-004"));
    assert_eq!(
        err.metadata().get("code").map(String::as_str),
        Some("E_LINALG_SINGULAR")
    );
    assert!(err.field_value("description").is_some());

    let mut buf = WasmF64Buffer::try_new(4).expect("buffer");
    let overflow = buf
        .copy_from_slice(&[1.0; 6])
        .expect_err("overflow must fail cleanly");
    assert_eq!(overflow.code, "E_BUFFER_OVERFLOW");
    assert_eq!(overflow.field_value("expected"), Some("4"));
    assert_eq!(overflow.field_value("actual"), Some("6"));
    assert_eq!(buf.len(), 4);
    assert!(!buf.is_empty());
    assert!(buf.byte_offset() != 0);
    assert_eq!(buf.len() * core::mem::size_of::<f64>(), 32);
    buf.fill(2.5);
    assert_eq!(buf.as_slice(), &[2.5; 4]);
    let mut readback = [0.0; 4];
    buf.copy_into_slice(&mut readback).expect("copy back");
    assert_eq!(readback, [2.5; 4]);
    assert_eq!(sdk.max_dim(), 4);
    assert!(sdk.rhs_ptr() != 0);

    let mut f32_buf = WasmF32Buffer::try_new(4).expect("f32 buffer");
    assert_eq!(f32_buf.len(), 4);
    assert!(!f32_buf.is_empty());
    assert!(f32_buf.byte_offset() != 0);
    f32_buf.fill(1.5_f32);
    assert_eq!(f32_buf.as_slice(), &[1.5_f32; 4]);
    let f32_ovf = f32_buf
        .copy_from_slice(&[0.0_f32; 5])
        .expect_err("f32 overflow must fail cleanly");
    assert_eq!(f32_ovf.code, "E_BUFFER_OVERFLOW");
    let mut f32_back = [0.0_f32; 4];
    f32_buf
        .copy_into_slice(&mut f32_back)
        .expect("f32 copy back");
    assert_eq!(f32_back, [1.5_f32; 4]);
}

#[test]
fn test_sde_pde_fft_and_ast_kernels_deterministic_parity() {
    let mut sde_a = WasmSdeWorkspace::try_new(3, 3, 12345).expect("sde a");
    let mut sde_b = WasmSdeWorkspace::try_new(3, 3, 12345).expect("sde b");
    assert_eq!(sde_a.state_dim(), 3);
    assert!(sde_a.state_ptr() != 0);
    let mut st_a = [1.0, 0.5, -0.5];
    let mut st_b = st_a;
    sde_a
        .step_ou(0.0, 0.01, 2.0, 0.0, 0.3, &mut st_a)
        .expect("step a");
    sde_b
        .step_ou(0.0, 0.01, 2.0, 0.0, 0.3, &mut st_b)
        .expect("step b");
    assert_eq!(st_a, st_b);

    let mut q = [1.0, -0.5];
    let mut p = [0.0, 0.25];
    sde_a
        .step_baoab_harmonic(1.0, 0.5, 0.2, 0.01, 1.5, 10, &mut q, &mut p)
        .expect("baoab");
    assert!(q.iter().all(|v| v.is_finite()) && p.iter().all(|v| v.is_finite()));

    let mut pde = WasmPdeWorkspace::try_new_3d(8, 8, 8, 1.0, 1.0, 1.0).expect("3d pde");
    assert_eq!(pde.grid_len(), 512);
    assert!(pde.scratch_capacity() >= 512);
    let u_ptr = pde.u_ptr();
    let v_ptr = pde.v_ptr();
    pde.seed_schnakenberg_steady_state(0.1, 0.9, 0.05);
    pde.step_schnakenberg_3d_in_place(0.005, 1.0, 18.0, 0.1, 0.9, 3)
        .expect("3d schnakenberg");
    assert!(pde.u_slice().iter().all(|v| v.is_finite()));
    assert!(pde.v_slice().iter().all(|v| v.is_finite()));
    assert!(pde.u_ptr() != 0 && u_ptr != 0 && v_ptr != 0);

    let mut pde_2d = WasmPdeWorkspace::try_new_2d(8, 8, 1.0, 1.0).expect("2d pde");
    assert_eq!(pde_2d.grid_len(), 64);
    let mut sig = [
        1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, //
        0.5, 0.0, 1.5, 0.0, 2.5, 0.0, 3.5, 0.0,
    ];
    let orig = sig;
    pde_2d
        .fft_1d_interleaved(&mut sig, false)
        .expect("forward fft");
    pde_2d
        .fft_1d_interleaved(&mut sig, true)
        .expect("inverse fft");
    for i in 0..orig.len() {
        assert_abs_diff_eq!(sig[i], orig[i], epsilon = 1e-12);
    }

    let mut ast = WasmAstKernel::try_compile_3d("sin(k * x) + cos(k * y) + z").expect("ast");
    ast.set_param("k", 2.0).expect("set k");
    let mut grid = vec![0.0; 8];
    ast.eval_3d_grid(2, 1.0, &mut grid).expect("eval 3d");
    assert!(grid.iter().all(|v| v.is_finite()));
    let wgsl = ast.to_compute_wgsl(64, 2.5).expect("compute wgsl");
    assert!(wgsl.contains("@compute @workgroup_size(4, 4, 4)"));
}

use super::*;
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_thomas_solver_residual_parity_and_zero_allocations() {
    const N: usize = 64;
    let mut lower = [Complex::new(0.0, 0.0); N - 1];
    let mut main = [Complex::new(0.0, 0.0); N];
    let mut upper = [Complex::new(0.0, 0.0); N - 1];
    let mut rhs = [Complex::new(0.0, 0.0); N];

    for i in 0..N {
        let phase = i as f64 * 0.13;
        main[i] = Complex::new(2.5 + 0.2 * (phase.cos()), 0.45 + 0.1 * (phase.sin()));
        rhs[i] = Complex::new((phase * 1.7).sin(), (phase * 0.9).cos());
        if i + 1 < N {
            lower[i] = Complex::new(-0.4, 0.25);
            upper[i] = Complex::new(-0.4, -0.25);
        }
    }

    let mut c_prime = [Complex::new(0.0, 0.0); N - 1];
    let mut inv_denom = [Complex::new(0.0, 0.0); N];
    let mut scratch = [Complex::new(0.0, 0.0); N];
    let mut sol = [Complex::new(0.0, 0.0); N];
    let mut sol_in_place = rhs;
    let mut check_rhs = [Complex::new(0.0, 0.0); N];

    let ((ok_factor, ok_solve, ok_in_place, ok_apply), metrics) =
        VerifiedEngine::run_verified(|| {
            let f = factor_tridiagonal(&lower, &main, &upper, &mut c_prime, &mut inv_denom);
            let s = solve_tridiagonal_factored(
                &lower,
                &c_prime,
                &inv_denom,
                &rhs,
                &mut scratch,
                &mut sol,
            );
            let inp = solve_tridiagonal_in_place(
                &lower,
                &main,
                &upper,
                &mut sol_in_place,
                &mut c_prime,
                &mut inv_denom,
                &mut scratch,
            );
            let app = apply_tridiagonal(&lower, &main, &upper, &sol, &mut check_rhs);
            (f, s, inp, app)
        });

    assert!(ok_factor && ok_solve && ok_in_place && ok_apply);
    assert!(metrics.function_calls > 0);
    for i in 0..N {
        assert!((check_rhs[i] - rhs[i]).norm() < 1e-12);
        assert!((sol_in_place[i] - sol[i]).norm() < 1e-13);
    }
}

#[test]
#[verified_engine::verified]
fn test_thomas_edge_cases_empty_single_singular_and_non_finite() {
    let empty: [Complex<f64>; 0] = [];
    let mut empty_mut: [Complex<f64>; 0] = [];
    let mut empty_scratch: [Complex<f64>; 0] = [];
    assert!(factor_tridiagonal(
        &empty,
        &empty,
        &empty,
        &mut empty_mut,
        &mut empty_scratch
    ));
    assert!(solve_tridiagonal_factored(
        &empty,
        &empty,
        &empty,
        &empty,
        &mut empty_scratch,
        &mut empty_mut
    ));
    assert!(apply_tridiagonal(
        &empty,
        &empty,
        &empty,
        &empty,
        &mut empty_mut
    ));

    let main_one = [Complex::new(2.0, -2.0)];
    let mut inv_one = [Complex::new(0.0, 0.0)];
    let mut scratch_one = [Complex::new(0.0, 0.0)];
    let mut rhs_one = [Complex::new(4.0, 0.0)];
    assert!(solve_tridiagonal_in_place(
        &empty,
        &main_one,
        &empty,
        &mut rhs_one,
        &mut empty_mut,
        &mut inv_one,
        &mut scratch_one,
    ));
    assert!((main_one[0] * rhs_one[0] - Complex::new(4.0, 0.0)).norm() < 1e-14);

    let mut y_one = [Complex::new(0.0, 0.0)];
    assert!(apply_tridiagonal(
        &empty, &main_one, &empty, &rhs_one, &mut y_one
    ));
    assert!((y_one[0] - Complex::new(4.0, 0.0)).norm() < 1e-14);

    // Zero pivot rejected cleanly without NaN/Inf division
    let zero_main = [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
    let off_one = [Complex::new(1.0, 0.0)];
    let mut cp = [Complex::new(0.0, 0.0)];
    let mut inv2 = [Complex::new(0.0, 0.0); 2];
    assert!(!factor_tridiagonal(
        &off_one, &zero_main, &off_one, &mut cp, &mut inv2
    ));

    // Eliminated zero second pivot (main = [1, 1], lower = [1], upper = [1] -> pivot_1 = 1 - 1 = 0)
    let singular_main = [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
    assert!(!factor_tridiagonal(
        &off_one,
        &singular_main,
        &off_one,
        &mut cp,
        &mut inv2
    ));

    // Non-finite input rejected
    let nan_main = [Complex::new(f64::NAN, 0.0), Complex::new(1.0, 0.0)];
    assert!(!factor_tridiagonal(
        &off_one, &nan_main, &off_one, &mut cp, &mut inv2
    ));
}

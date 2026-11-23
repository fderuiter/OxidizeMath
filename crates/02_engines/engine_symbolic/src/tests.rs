//! Unit and zero-allocation tests for `engine_symbolic`.
// [cite:engine_symbolic]

use approx::assert_relative_eq;
use num_complex::Complex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

use crate::{Dual, DualComplex64, Jet2, SolverWorkspace};

#[global_allocator]
static TEST_ALLOC: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_dual_real_and_clarke_subgradients_zero_alloc() {
    let ((d_smooth, d_step, d_clamp_in, d_clamp_kink, d_max_kink), metrics) =
        VerifiedEngine::run_verified(|| {
            let x = Dual::variable(0.7_f64);
            let y = (x.sin() * x.exp()) / (x * x + 1.0);

            let x_thresh = Dual::variable(2.0_f64);
            let step_at_a = (x_thresh - 2.0).step();

            let x_in = Dual::variable(0.25_f64);
            let clamp_in = (x_in * 2.0).clamp(Dual::constant(-1.0), Dual::constant(1.0));

            let x_kink = Dual::variable(0.5_f64);
            let clamp_kink = (x_kink * 2.0).clamp(Dual::constant(-1.0), Dual::constant(1.0));

            let x_zero = Dual::variable(0.0_f64);
            let max_kink = (x_zero * 0.2).max(x_zero);

            (y, step_at_a, clamp_in, clamp_kink, max_kink)
        });

    assert!(metrics.function_calls > 0);
    let x0 = 0.7_f64;
    let num = x0.sin() * x0.exp();
    let den = x0 * x0 + 1.0;
    let dnum = (x0.cos() + x0.sin()) * x0.exp();
    let exact = (dnum * den - num * 2.0 * x0) / (den * den);
    assert_relative_eq!(d_smooth.eps, exact, epsilon = 1e-14);
    assert_eq!(d_step.re, 1.0);
    assert_eq!(d_step.eps, 0.0);
    assert_relative_eq!(d_clamp_in.eps, 2.0, epsilon = 1e-15);
    assert_relative_eq!(d_clamp_kink.eps, 1.0, epsilon = 1e-15);
    assert_relative_eq!(d_max_kink.eps, 0.6, epsilon = 1e-15);
}

#[test]
#[verified_engine::verified]
fn test_dual_complex_holomorphic_zero_alloc() {
    let (fz, _) = VerifiedEngine::run_verified(|| {
        let z = DualComplex64::variable(Complex::new(0.6, -0.4));
        (z * z * z + z.sin()) / z.exp()
    });

    let z0 = Complex::new(0.6, -0.4);
    let num = z0 * z0 * z0 + z0.sin();
    let dnum = Complex::new(3.0, 0.0) * z0 * z0 + z0.cos();
    let ez = z0.exp();
    let exact = (dnum - num) / ez;
    assert_relative_eq!(fz.eps.re, exact.re, epsilon = 1e-14);
    assert_relative_eq!(fz.eps.im, exact.im, epsilon = 1e-14);
}

#[test]
#[verified_engine::verified]
fn test_jet2_shallow_saddle_and_workspace_zero_alloc() {
    let mut ws = SolverWorkspace::new();
    let ((det_hess, ws_ok), _) = VerifiedEngine::run_verified(|| {
        let u = Jet2::<2>::var(1.0, 0);
        let v = Jet2::<2>::var(1.0, 1);
        let z = Jet2::<2>::constant(100.0) + (u * u - v * v) * 1.0e-4;
        ws.clear_adjoints(16);
        ws.adjoints[0] = 1.0;
        (z.duu() * z.dvv() - z.duv() * z.duv(), ws.adjoints[0] == 1.0)
    });

    assert!(ws_ok);
    assert!((det_hess - (-4.0e-8)).abs() < 1e-22);
}

#[cfg(feature = "alloc")]
#[test]
#[verified_engine::verified]
fn test_egraph_identities_caps_and_iterative_deep_tree() {
    use crate::{EGraph, MAX_EGRAPH_NODES, MAX_REWRITE_PASSES, SymFunc1, SymOp};

    let mut eg = EGraph::new();
    let x = eg.add_symbol("x").unwrap();
    let zero = eg.add(SymOp::num(0.0)).unwrap();
    let one = eg.add(SymOp::num(1.0)).unwrap();
    let two = eg.add(SymOp::num(2.0)).unwrap();

    // (x * 1 + 0) * (sin(x)^2 + cos(x)^2) -> x
    let xm1 = eg.add(SymOp::Mul([x, one])).unwrap();
    let xp0 = eg.add(SymOp::Add([xm1, zero])).unwrap();
    let sx = eg.add(SymOp::Func1(SymFunc1::Sin, x)).unwrap();
    let cx = eg.add(SymOp::Func1(SymFunc1::Cos, x)).unwrap();
    let sx2 = eg.add(SymOp::Pow([sx, two])).unwrap();
    let cx2 = eg.add(SymOp::Pow([cx, two])).unwrap();
    let pyth = eg.add(SymOp::Add([sx2, cx2])).unwrap();
    let prod = eg.add(SymOp::Mul([xp0, pyth])).unwrap();

    let passes = eg.saturate(32);
    assert!(passes <= MAX_REWRITE_PASSES);
    let extracted = eg.extract(prod);
    assert_eq!(extracted.nodes.len(), 1);
    assert_eq!(extracted.nodes[extracted.root as usize], SymOp::Symbol(0));

    // Verify 2048-node deterministic cap without panic or recursion
    let mut big = EGraph::new();
    let base = big.add_symbol("t").unwrap();
    let mut cur = base;
    for i in 1..3000 {
        let c = match big.add(SymOp::num(f64::from(i))) {
            Some(id) => id,
            None => break,
        };
        match big.add(SymOp::Add([cur, c])) {
            Some(next) => cur = next,
            None => break,
        }
    }
    assert!(big.node_count() <= MAX_EGRAPH_NODES);
    let _ = big.saturate(MAX_REWRITE_PASSES);
    let dag = big.extract(cur);
    assert!(!dag.nodes.is_empty());
}

// [cite:math_commons]
use super::*;
use std::collections::HashMap;
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_symbolic_differentiate_egraph_simplify_latex_and_glsl() {
    let expr = parse("sin(x)^2 + cos(x)^2 + x^3").expect("parse");
    let simplified = expr.simplify();
    assert_eq!(
        simplified
            .eval(&HashMap::from([("x".to_string(), 1.7)]))
            .unwrap(),
        1.0 + 1.7_f64.powi(3)
    );

    let df = expr.differentiate("x");
    let latex = df.to_latex();
    assert!(!latex.is_empty());
    let glsl_scalar = df.to_glsl_scalar().expect("valid scalar GLSL");
    assert!(!glsl_scalar.is_empty());
    let glsl_complex = df.to_glsl_complex().expect("valid complex GLSL");
    assert!(!glsl_complex.is_empty());

    let compiled_df = CompiledExpr::compile_with_vars(&df, &["x"]).expect("compile df");
    for i in -10..=10 {
        let x = i as f64 * 0.25;
        let sym_val = compiled_df.eval_slice(&[x]);
        let exact = 3.0 * x * x;
        assert!(
            (sym_val - exact).abs() < 1e-12,
            "symbolic derivative mismatch at x={x}: {sym_val} vs {exact}"
        );
    }
}

#[test]
fn test_symbolic_clarke_subgradients_match_dual_and_reverse_mode() {
    let formulas = [
        "clamp(2 * x - 1, -1, 1)",
        "max(0, 3 * x + 0.5)",
        "min(x^2, 2 * x + 1)",
        "step(x - 0.5) * x^2",
        "rect(x) * (x + 2)",
        "abs(2 * x - 0.5)",
        "atan2(sin(x), cos(x)) + exp(-0.3 * x) * cos(1.5 * x)",
    ];
    let sample_xs = [-1.5, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.4];
    let mut ws = SolverWorkspace::new();

    for src in formulas {
        let expr = parse(src).expect(src);
        let sym_df = expr.differentiate("x");
        assert!(sym_df.to_glsl_scalar().is_ok(), "GLSL failed for {src}");
        assert!(!sym_df.to_latex().is_empty(), "LaTeX empty for {src}");

        let comp = CompiledExpr::compile_with_vars(&expr, &["x"]).expect("compile f");
        let comp_df = CompiledExpr::compile_with_vars(&sym_df, &["x"]).expect("compile df");

        for &x in &sample_xs {
            let dual = comp.eval_1d_dual(Dual::variable(x), &[]);
            let mut grad = [0.0_f64; 1];
            let rev_val = comp.eval_grad_into(&[x], &mut ws, &mut grad);
            let sym_val = comp_df.eval_1d(x, &[]);

            assert!((dual.re - rev_val).abs() < 1e-12, "{src} val at {x}");
            assert!(
                (dual.eps - grad[0]).abs() < 1e-12,
                "{src} dual vs reverse at {x}: {} vs {}",
                dual.eps,
                grad[0]
            );
            assert!(
                (dual.eps - sym_val).abs() < 1e-11,
                "{src} dual vs symbolic at {x}: {} vs {}",
                dual.eps,
                sym_val
            );
        }
    }

    // Verify 0.0 spike error on step jump & bounded Clarke midpoint subgradient on clamp kinks
    let step_comp = CompiledExpr::compile_with_vars(&parse("step(x)").unwrap(), &["x"]).unwrap();
    assert_eq!(step_comp.eval_1d_dual(Dual::variable(0.0), &[]).eps, 0.0);
    let clamp_comp =
        CompiledExpr::compile_with_vars(&parse("clamp(x, -1, 1)").unwrap(), &["x"]).unwrap();
    assert_eq!(clamp_comp.eval_1d_dual(Dual::variable(1.0), &[]).eps, 0.5);
    assert_eq!(clamp_comp.eval_1d_dual(Dual::variable(-1.0), &[]).eps, 0.5);
    assert_eq!(clamp_comp.eval_1d_dual(Dual::variable(2.0), &[]).eps, 0.0);
    assert_eq!(clamp_comp.eval_1d_dual(Dual::variable(0.0), &[]).eps, 1.0);
}

#[test]
fn test_compiled_eval_grad_into_and_jet2_zero_allocations() {
    let expr = parse("a * exp(-b * x) + c * sin(w * x + p) + d * x^2").expect("parse");
    let comp =
        CompiledExpr::compile_with_vars_and_params(&expr, &["x"], &["a", "b", "c", "w", "p", "d"])
            .expect("compile");
    let surf =
        CompiledExpr::compile_with_vars(&parse("100 + 1e-4 * (u^2 - v^2)").unwrap(), &["u", "v"])
            .expect("compile surf");

    let vars = [0.7, 1.8, 0.4, 0.9, 2.1, 0.3, -0.2];
    let mut grads = [0.0_f64; 7];
    let mut ws = SolverWorkspace::new();

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((y, jet), _) = VerifiedEngine::run_verified(|| {
        let val = comp.eval_grad_into(&vars, &mut ws, &mut grads);
        let j = surf.eval_2d_jet2(Jet2::var(1.5, 0), Jet2::var(-2.0, 1), &[]);
        (val, j)
    });

    assert!(y.is_finite());
    for (slot, &g) in grads.iter().enumerate() {
        let d = comp.eval_dir_dual(&vars, slot);
        assert!((d.re - y).abs() < 1e-12);
        assert!((d.eps - g).abs() < 1e-12);
    }
    assert!((jet.hess[0][0] - 2e-4).abs() < 1e-20);
    assert!((jet.hess[1][1] + 2e-4).abs() < 1e-20);
    assert!(jet.hess[0][1].abs() < 1e-20);
}

#[test]
fn test_complex_dual_derivative_holomorphic() {
    let expr = parse("z^3 + a * sin(z)").expect("parse");
    let comp = CompiledExpr::compile_complex(&expr, &["z", "a"]).expect("compile complex");
    let z = Complex64::new(0.6, -0.35);
    let a = 1.25_f64;
    let out = comp.eval_complex_z_params_dual(DualComplex64::variable(z), &[a]);
    let ac = Complex64::new(a, 0.0);
    let exact_val = z * z * z + ac * z.sin();
    let exact_dz = Complex64::new(3.0, 0.0) * z * z + ac * z.cos();
    assert!((out.re - exact_val).norm() < 1e-12);
    assert!((out.eps - exact_dz).norm() < 1e-12);
}

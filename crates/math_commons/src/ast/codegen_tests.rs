// [cite:math_commons]
use super::*;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::process::Command;

#[test]
fn test_bookmark_ode_transpilation_exact_outputs() {
    let expr = parse("-k * y * sin(t)").expect("parse bookmark ode");
    let mut params = HashMap::new();
    params.insert("k".to_string(), 1.85);
    params.insert("t".to_string(), 99.0);
    params.insert("y".to_string(), 88.0);

    let latex = expr
        .to_latex_with_params(&params)
        .expect("calibrated latex");
    assert_eq!(latex, r"-1.85 \cdot y \cdot \sin\left(t\right)");

    let rust_fn = expr
        .to_rust_fn("rhs", &["t", "y"], &params)
        .expect("rust fn");
    assert_eq!(
        rust_fn,
        "pub fn rhs(t: f64, y: f64) -> f64 {\n    (-1.85_f64 * y) * t.sin()\n}"
    );

    let py_expr = expr.to_python_expr(&params).expect("python expr");
    assert_eq!(py_expr, "(-1.85 * y) * np.sin(t)");

    let py_ode = expr.to_python_ode_rhs(&params).expect("python ode rhs");
    assert_eq!(py_ode, "(-1.85 * y[0]) * np.sin(t)");
}

#[test]
fn test_all_14_builtin_funcs_and_operators_codegen() {
    let params = HashMap::from([("a".to_string(), -2.5), ("b".to_string(), 3.25)]);
    for (func_name, expected_rust, expected_py) in [
        ("sin", "x.sin()", "np.sin(x)"),
        ("cos", "x.cos()", "np.cos(x)"),
        ("tan", "x.tan()", "np.tan(x)"),
        ("exp", "x.exp()", "np.exp(x)"),
        ("ln", "x.ln()", "np.log(x)"),
        ("log10", "x.log10()", "np.log10(x)"),
        ("sqrt", "x.sqrt()", "np.sqrt(x)"),
        ("abs", "x.abs()", "np.abs(x)"),
        ("sinh", "x.sinh()", "np.sinh(x)"),
        ("cosh", "x.cosh()", "np.cosh(x)"),
        ("tanh", "x.tanh()", "np.tanh(x)"),
        ("asin", "x.asin()", "np.arcsin(x)"),
        ("acos", "x.acos()", "np.arccos(x)"),
        ("atan", "x.atan()", "np.arctan(x)"),
    ] {
        let expr = parse(&format!("{func_name}(x)")).expect("parse func");
        assert_eq!(expr.to_rust_expr(&params).unwrap(), expected_rust);
        assert_eq!(expr.to_python_expr(&params).unwrap(), expected_py);
        assert!(!expr.to_latex_with_params(&params).unwrap().is_empty());
    }

    let neg_pow = parse("a^2 - (-b) + +x / (pi + e + tau)").expect("parse ops");
    let latex = neg_pow.to_latex_with_params(&params).unwrap();
    assert!(latex.contains(r"{\left(-2.5\right)}^{2}"));
    let rust_code = neg_pow.to_rust_fn("eval_ops", &["x"], &params).unwrap();
    assert!(rust_code.contains("(-2.5_f64).powf(2.0_f64)"));
    assert!(rust_code.contains("std::f64::consts::PI"));
    let py_code = neg_pow.to_python_expr(&params).unwrap();
    assert!(py_code.contains("(-2.5) ** 2.0"));
    assert!(py_code.contains("(2.0 * np.pi)"));
}

#[test]
fn test_depth_64_succeeds_and_over_depth_64_rejects_iteratively() {
    let params = HashMap::from([("k".to_string(), 1.25)]);
    let exact_64 = format!("{}k", "1 + ".repeat(MAX_EVAL_DEPTH));
    let expr_64 = parse(&exact_64).expect("height 64 parses");
    assert!(expr_64.to_latex_with_params(&params).is_ok());
    assert!(expr_64.to_rust_fn("f64_chain", &["x"], &params).is_ok());
    assert!(expr_64.to_python_expr(&params).is_ok());

    let mut deep = Expr::Variable("x".to_string());
    for i in 0..100_000 {
        deep = if i % 2 == 0 {
            Expr::function(Func::Sin, vec![deep])
        } else {
            Expr::binary(BinaryOp::Add, deep, Expr::Number(1.0))
        };
    }
    assert_eq!(
        deep.to_latex_with_params(&params),
        Err(EvalError::RecursionLimitExceeded(MAX_EVAL_DEPTH))
    );
    assert_eq!(
        deep.to_rust_fn("deep_fn", &["x"], &params),
        Err(EvalError::RecursionLimitExceeded(MAX_EVAL_DEPTH))
    );
    assert_eq!(
        deep.to_python_expr(&params),
        Err(EvalError::RecursionLimitExceeded(MAX_EVAL_DEPTH))
    );
}

#[test]
fn test_validation_errors_for_nonfinite_unknown_and_bad_arity() {
    let nan_params = HashMap::from([("k".to_string(), f64::NAN)]);
    let expr = parse("k * x").unwrap();
    assert_eq!(
        expr.to_latex_with_params(&nan_params).unwrap(),
        r"k \cdot x"
    );
    assert!(matches!(
        expr.to_rust_fn("f", &["x"], &nan_params),
        Err(EvalError::DomainError(_))
    ));
    assert!(matches!(
        expr.to_python_expr(&nan_params),
        Err(EvalError::DomainError(_))
    ));

    let empty_params = HashMap::new();
    assert_eq!(
        expr.to_rust_fn("f", &["x"], &empty_params),
        Err(EvalError::UnknownVariable("k".to_string()))
    );
    assert_eq!(
        parse("x + y")
            .unwrap()
            .to_rust_fn("f", &["x"], &empty_params),
        Err(EvalError::UnknownVariable("y".to_string()))
    );

    let bad_arity = Expr::function(Func::Sin, vec![]);
    assert_eq!(
        bad_arity.to_rust_expr(&empty_params),
        Err(EvalError::WrongArgumentCount {
            func: Func::Sin,
            expected: 1,
            found: 0,
        })
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_numerical_parity_rust_and_python_against_eval_and_eval_2d() {
    let params = HashMap::from([("k".to_string(), 1.85), ("alpha".to_string(), -0.42)]);
    let expr_1d = parse(
        "sin(x) + cos(x)^2 - tan(0.3 * x) + exp(-0.2 * x) + ln(1.5 + x) + log10(2.0 + x) \
         + sqrt(1.5 + x) + abs(alpha * x) + sinh(0.2 * x) - cosh(0.1 * x) + tanh(x) \
         + asin(0.4 * x) + acos(-0.3 * x) + atan(k * x) + (pi - e + tau) / 3.0",
    )
    .expect("parse 1d parity expr");
    let expr_2d = parse("-k * y * sin(t) + alpha * (t^2 + 1.0) / (2.0 + abs(y))")
        .expect("parse 2d parity expr");

    let sample_x = [-0.8_f64, -0.25, 0.0, 0.35, 0.9];
    let sample_ty = [(0.0_f64, 1.0_f64), (0.5, 0.8), (1.25, -1.5), (3.1, 2.4)];

    let expected_1d: Vec<f64> = sample_x
        .iter()
        .map(|&x| {
            let mut vars = params.clone();
            vars.insert("x".to_string(), x);
            expr_1d.eval(&vars).expect("eval 1d")
        })
        .collect();
    let expected_2d: Vec<f64> = sample_ty
        .iter()
        .map(|&(t, y)| expr_2d.eval_2d(t, y, &params))
        .collect();

    verify_rustc_2024_parity(
        &expr_1d,
        &expr_2d,
        &params,
        &sample_x,
        &sample_ty,
        &expected_1d,
        &expected_2d,
    );
    verify_python_numpy_parity(
        &expr_1d,
        &expr_2d,
        &params,
        &sample_x,
        &sample_ty,
        &expected_1d,
        &expected_2d,
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_rustc_2024_parity(
    expr_1d: &Expr,
    expr_2d: &Expr,
    params: &HashMap<String, f64>,
    sample_x: &[f64],
    sample_ty: &[(f64, f64)],
    expected_1d: &[f64],
    expected_2d: &[f64],
) {
    static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let run_id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("ast_codegen_rs_{}_{run_id}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let src_path = dir.join("parity_check.rs");
    let bin_path = dir.join(format!("parity_check_bin{}", std::env::consts::EXE_SUFFIX));

    let fn_1d = expr_1d.to_rust_fn("eval_1d", &["x"], params).unwrap();
    let fn_2d = expr_2d.to_rust_fn("eval_2d", &["t", "y"], params).unwrap();
    let mut assertions = String::new();
    for (&x, &exp) in sample_x.iter().zip(expected_1d) {
        assertions.push_str(&format!(
            "    let v = eval_1d({x:?}_f64);\n    assert!(((v - {exp:?}_f64).abs() / {exp:?}_f64.abs().max(1.0)) < 1e-9, \"1d mismatch: {{v}} vs {exp:?}\");\n"
        ));
    }
    for (&(t, y), &exp) in sample_ty.iter().zip(expected_2d) {
        assertions.push_str(&format!(
            "    let v = eval_2d({t:?}_f64, {y:?}_f64);\n    assert!(((v - {exp:?}_f64).abs() / {exp:?}_f64.abs().max(1.0)) < 1e-9, \"2d mismatch: {{v}} vs {exp:?}\");\n"
        ));
    }
    let program = format!("{fn_1d}\n\n{fn_2d}\n\nfn main() {{\n{assertions}}}\n");
    std::fs::write(&src_path, program).expect("write rust snippet");

    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let out = Command::new(rustc)
        .args(["--edition", "2024", "-D", "warnings"])
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path)
        .output()
        .expect("invoke rustc");
    assert!(
        out.status.success(),
        "generated Rust snippet failed to compile on Rust 2024:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run_status = Command::new(&bin_path)
        .status()
        .expect("run compiled parity binary");
    assert!(
        run_status.success(),
        "compiled Rust parity binary failed assertions"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_python_numpy_parity(
    expr_1d: &Expr,
    expr_2d: &Expr,
    params: &HashMap<String, f64>,
    sample_x: &[f64],
    sample_ty: &[(f64, f64)],
    expected_1d: &[f64],
    expected_2d: &[f64],
) {
    let Some(py_cmd) = ["python3", "python"].into_iter().find(|cmd| {
        Command::new(cmd)
            .args(["-c", "import numpy"])
            .output()
            .is_ok_and(|o| o.status.success())
    }) else {
        return;
    };
    let py_1d = expr_1d.to_python_expr(params).unwrap();
    let py_2d = expr_2d.to_python_expr(params).unwrap();
    let mut script = format!(
        "import numpy as np\ndef f1(x):\n    return float({py_1d})\ndef f2(t, y):\n    return float({py_2d})\n"
    );
    for (&x, &exp) in sample_x.iter().zip(expected_1d) {
        script.push_str(&format!(
            "assert abs(f1({x:?}) - {exp:?}) / max(abs({exp:?}), 1.0) < 1e-9\n"
        ));
    }
    for (&(t, y), &exp) in sample_ty.iter().zip(expected_2d) {
        script.push_str(&format!(
            "assert abs(f2({t:?}, {y:?}) - {exp:?}) / max(abs({exp:?}), 1.0) < 1e-9\n"
        ));
    }
    let status = Command::new(py_cmd)
        .args(["-c", &script])
        .status()
        .expect("run python parity check");
    assert!(status.success(), "Python numpy parity assertions failed");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_piecewise_and_multiarg_funcs_rust_and_python_parity() {
    let expr = parse("clamp(min(x, a) + max(-0.4, atan2(x, w)), -0.7, 0.9) + step(x) - rect(x)")
        .expect("parse multiarg expr");
    let expr_2d = parse("clamp(sin(k * t) + y, -0.5, 0.8)").expect("parse 2d clamp");
    let params = HashMap::from([
        ("a".to_string(), 0.75),
        ("w".to_string(), 1.25),
        ("k".to_string(), 2.0),
    ]);
    assert!(
        expr.to_latex_with_params(&params)
            .unwrap()
            .contains(r"\mathrm{clamp}")
    );

    let xs = [-1.2_f64, -0.5, -0.2, 0.0, 0.35, 0.5, 1.4];
    let tys = [(0.0_f64, -1.0_f64), (0.6, 0.1), (1.7, 0.9)];
    let exp_1d: Vec<f64> = xs
        .iter()
        .map(|&x| {
            let mut m = params.clone();
            m.insert("x".into(), x);
            expr.eval(&m).unwrap()
        })
        .collect();
    let exp_2d: Vec<f64> = tys
        .iter()
        .map(|&(t, y)| expr_2d.eval_2d(t, y, &params))
        .collect();
    verify_rustc_2024_parity(&expr, &expr_2d, &params, &xs, &tys, &exp_1d, &exp_2d);
    verify_python_numpy_parity(&expr, &expr_2d, &params, &xs, &tys, &exp_1d, &exp_2d);
}

use super::*;
use crate::api_drift::extract_apis_in_dir;
use std::fs;
use std::path::Path;

fn write_minimal_facade_fixture(dir: &Path, include_ai_error: bool, extra_sig: &str) {
    let me = dir.join("math_explorer/src");
    fs::create_dir_all(&me).unwrap();
    fs::write(
        me.join("lib.rs"),
        "pub mod error;\npub mod prelude;\npub use diagnostics;\npub use math_commons::{ast, constants, fit, math_kernel, poi, primitives, registry, series, wigner};\npub use domain_ai::ai;\npub use nalgebra;\npub use num_complex;\npub use oxidize_core::{mesh, rng};\npub use petgraph;\npub use ::pure_math::{math_types, pure_math};\npub use rand;\n",
    )
    .unwrap();
    let err_export = if include_ai_error {
        "pub use domain_ai::error::AIError;\n"
    } else {
        ""
    };
    fs::write(me.join("error.rs"), err_export).unwrap();
    fs::write(me.join("prelude.rs"), "pub use crate::error::*;\n").unwrap();

    let ai_dir = dir.join("crates/domain_ai/src/ai");
    fs::create_dir_all(&ai_dir).unwrap();
    fs::write(
        dir.join("crates/domain_ai/src/lib.rs"),
        "pub mod ai;\npub mod error;\n",
    )
    .unwrap();
    fs::write(
        dir.join("crates/domain_ai/src/error.rs"),
        "pub enum AIError { Invalid }\n",
    )
    .unwrap();
    fs::write(
        ai_dir.join("mod.rs"),
        format!("pub use crate::error;\n{extra_sig}\n"),
    )
    .unwrap();
}

#[test]
fn detects_missing_error_reexport_and_unexported_signature_type() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_completeness_fail");
    write_minimal_facade_fixture(
        &dir,
        false,
        "mod secret { pub struct SecretToken; }\npub fn leak() -> secret::SecretToken { todo!() }",
    );
    let apis = extract_apis_in_dir(&dir);
    let violations = check_facade_completeness_in_dir(&dir, &apis);
    assert!(violations.iter().any(|v| v.contains("AIError")));
    assert!(violations.iter().any(|v| v.contains("SecretToken")));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn accepts_reexported_errors_and_public_signature_types() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_completeness_ok");
    write_minimal_facade_fixture(
        &dir,
        true,
        "pub struct PublicConfig;\npub fn run<T: Clone>(cfg: PublicConfig) -> Result<DMatrix<T>, crate::error::AIError> { todo!() }",
    );
    let apis = extract_apis_in_dir(&dir);
    let violations = check_facade_completeness_in_dir(&dir, &apis);
    assert!(
        violations.is_empty(),
        "Unexpected violations: {violations:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn detects_unreexported_submodule_error() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_submodule_error_gap");
    write_minimal_facade_fixture(&dir, true, "pub mod sub;");
    fs::write(
        dir.join("crates/domain_ai/src/ai/sub.rs"),
        "pub enum SubmoduleOptimizationError { Diverged }\n",
    )
    .unwrap();

    let apis = extract_apis_in_dir(&dir);
    let violations = check_facade_completeness_in_dir(&dir, &apis);
    assert!(
        violations.iter().any(|v| {
            v.contains("SubmoduleOptimizationError") && v.contains("domain_ai:src/ai/sub.rs")
        }),
        "Expected submodule error gap violation, got: {violations:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn rejects_duplicate_error_names_across_reachable_modules() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_duplicate_error_name");
    write_minimal_facade_fixture(&dir, true, "pub mod sub;");
    fs::write(
        dir.join("crates/domain_ai/src/ai/sub.rs"),
        "pub enum AIError { Duplicate }\n",
    )
    .unwrap();

    let apis = extract_apis_in_dir(&dir);
    let violations = check_facade_completeness_in_dir(&dir, &apis);
    assert!(
        violations
            .iter()
            .any(|v| v.contains("Duplicate reachable error name `AIError`")),
        "Expected duplicate error name violation, got: {violations:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn parses_signature_forms_on_first_filtered_suffix_and_deduplicates() {
    let forms = [
        "pub fn solve(x: f64) -> f64",
        "pub const fn dim() -> usize",
        "pub struct SolverConfig<T>",
        "pub enum SolverError",
        "pub union RawSlot",
        "pub trait Integrator",
        "pub trait Integrator { fn step(&mut self, dt: f64) }",
        "pub trait Integrator { const ORDER: usize }",
        "pub trait Integrator { type State }",
        "impl Solver { pub fn new() -> Self }",
        "impl Solver { pub const DEFAULT_STEPS: usize }",
        "impl Integrator for Solver { type State = f64 }",
        "pub const MAX_ITERS: usize",
        "pub static GLOBAL_SEED: u64",
        "pub type MatrixAlias = DMatrix<f64>",
        "pub use crate::error::AIError",
        "pub mod inner",
        "pub mod nested { pub fn inner_fn(cfg: SolverConfig<f64>) -> Result<(), AIError> }",
    ];
    for sig in forms {
        let mut attempts = 0;
        let ast = parse_signature_ast_counted(sig, &mut attempts);
        assert!(ast.is_some(), "Failed to parse signature `{sig}`");
        assert_eq!(attempts, 1, "Expected 1 syn parse attempt for `{sig}`");
    }

    let item = |s: &str| ApiItem {
        signature: s.into(),
        deprecation_expires: None,
    };
    let mut apis = BTreeMap::new();
    apis.insert(
        "domain_ai:src/a.rs".into(),
        vec![
            item("pub fn alpha() -> f64"),
            item("pub struct SharedState"),
            item("pub mod __doctest_facade {}"),
        ],
    );
    apis.insert(
        "domain_physics:src/b.rs".into(),
        vec![
            item("pub fn alpha() -> f64"),
            item("pub struct SharedState"),
        ],
    );
    apis.insert(
        "unified_verification:src/ignored.rs".into(),
        vec![item("pub fn unreachable_tool_fn()")],
    );

    let ctx = FacadeContext::default_for_sigs();
    let (parsed, total_attempts) = build_parsed_signatures_counted(&ctx, &apis);
    assert_eq!(parsed.len(), 2);
    assert_eq!(total_attempts, 2);
}

#[test]
fn detects_unreexported_tier2_engine_error_and_missing_umbrella_from() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_tier2_engine_error_gap");
    write_minimal_facade_fixture(&dir, true, "");
    for (engine_pkg, err_ident) in [
        ("engine_linalg", "LinalgKernelError"),
        ("engine_diffeq", "DiffeqStepError"),
        ("engine_pde", "PdeStencilError"),
    ] {
        let eng_dir = dir.join(format!("crates/02_engines/{engine_pkg}/src"));
        fs::create_dir_all(&eng_dir).unwrap();
        fs::write(eng_dir.join("lib.rs"), "pub mod error;\npub mod types;\n").unwrap();
        fs::write(eng_dir.join("types.rs"), "pub struct SolverWorkspace;\n").unwrap();
        fs::write(
            eng_dir.join("error.rs"),
            format!("pub enum {err_ident} {{ Bad }}\n"),
        )
        .unwrap();
    }

    let apis = extract_apis_in_dir(&dir);
    let violations = check_facade_completeness_in_dir(&dir, &apis);
    for err_ident in ["LinalgKernelError", "DiffeqStepError", "PdeStencilError"] {
        assert!(
            violations
                .iter()
                .any(|v| v.contains("Facade error gap") && v.contains(err_ident)),
            "Expected Tier 2 engine facade error gap for {err_ident}, got: {violations:?}"
        );
        assert!(
            violations.iter().any(|v| {
                v.contains("Umbrella conversion gap")
                    && v.contains("AIError")
                    && v.contains(&format!("From<{err_ident}>"))
            }),
            "Expected Tier 2 engine umbrella From<{err_ident}> gap, got: {violations:?}"
        );
    }
    assert!(
        !violations.iter().any(|v| v.contains("SolverWorkspace")),
        "Non-error engine structs must not be flagged as errors: {violations:?}"
    );

    fs::write(
        dir.join("math_explorer/src/error.rs"),
        "pub use domain_ai::error::AIError;\npub use engine_linalg::LinalgKernelError;\npub use engine_diffeq::DiffeqStepError;\npub use engine_pde::PdeStencilError;\n",
    )
    .unwrap();
    fs::write(
        dir.join("crates/domain_ai/src/error.rs"),
        "pub enum AIError { Invalid, L(#[from] LinalgKernelError), D(#[from] DiffeqStepError), P(#[from] PdeStencilError) }\n",
    )
    .unwrap();
    let apis_fixed = extract_apis_in_dir(&dir);
    let fixed_violations = check_facade_completeness_in_dir(&dir, &apis_fixed);
    assert!(
        fixed_violations.is_empty(),
        "Expected zero violations once Tier 2 errors are re-exported and converted, got: {fixed_violations:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn detects_missing_umbrella_from_for_domain_and_cross_crate_errors() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_umbrella_from_gap");
    write_minimal_facade_fixture(
        &dir,
        true,
        "pub mod sub;\npub struct OptimizationError;\npub fn step() -> Result<(), OptimizationError> { todo!() }",
    );
    fs::write(
        dir.join("crates/domain_ai/src/ai/sub.rs"),
        "pub enum AttentionKernelError { Diverged }\n",
    )
    .unwrap();
    fs::write(
        dir.join("math_explorer/src/error.rs"),
        "pub use domain_ai::error::AIError;\npub use domain_ai::ai::OptimizationError;\npub use domain_ai::ai::sub::AttentionKernelError;\n",
    )
    .unwrap();
    let conv_dir = dir.join("math_explorer/src/error");
    fs::create_dir_all(&conv_dir).unwrap();
    fs::write(
        conv_dir.join("conversions.rs"),
        "impl_error_from!(AIError => AI);\n",
    )
    .unwrap();

    let apis = extract_apis_in_dir(&dir);
    let violations = check_facade_completeness_in_dir(&dir, &apis);
    assert!(
        violations
            .iter()
            .any(|v| v.contains("AIError") && v.contains("From<AttentionKernelError>")),
        "Expected domain leaf umbrella From gap, got: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|v| v.contains("AIError") && v.contains("From<OptimizationError>")),
        "Expected cross-crate signature umbrella From gap, got: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|v| v.contains("conversions.rs") && v.contains("From<AttentionKernelError>")),
        "Expected facade Error From gap in conversions.rs, got: {violations:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn detects_unreexported_tier0_math_rng_errors_and_missing_facade_from() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_math_rng_error_gap");
    write_minimal_facade_fixture(&dir, true, "");
    let rng_dir = dir.join("crates/00_foundation/math_rng/src");
    fs::create_dir_all(&rng_dir).unwrap();
    fs::write(rng_dir.join("lib.rs"), "pub mod scalar;\npub mod qmc;\n").unwrap();
    fs::write(
        rng_dir.join("scalar.rs"),
        "mod private { pub trait Sealed {} }\npub trait RngScalar: private::Sealed {}\npub enum RngError { Zero }\n",
    )
    .unwrap();
    fs::write(rng_dir.join("qmc.rs"), "pub enum QmcError { BadDim }\n").unwrap();
    let conv_dir = dir.join("math_explorer/src/error");
    fs::create_dir_all(&conv_dir).unwrap();
    fs::write(
        conv_dir.join("conversions.rs"),
        "impl_error_from!(AIError => AI);\n",
    )
    .unwrap();

    let apis = extract_apis_in_dir(&dir);
    let violations = check_facade_completeness_in_dir(&dir, &apis);
    assert!(
        !violations.iter().any(|v| v.contains("Sealed")),
        "private::Sealed supertrait must not be flagged as unexported: {violations:?}"
    );
    for err_ident in ["RngError", "QmcError"] {
        assert!(
            violations
                .iter()
                .any(|v| v.contains("Facade error gap") && v.contains(err_ident)),
            "Expected math_rng facade error gap for {err_ident}, got: {violations:?}"
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains("conversions.rs") && v.contains(&format!("From<{err_ident}>"))),
            "Expected facade conversions.rs From<{err_ident}> gap, got: {violations:?}"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

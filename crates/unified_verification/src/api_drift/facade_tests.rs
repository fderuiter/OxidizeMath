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
    for mod_rel in DOMAIN_INNER_MODS {
        let p = dir.join(mod_rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, "pub use crate::error;\n").unwrap();
    }
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
    for mod_rel in DOMAIN_INNER_MODS {
        let p = dir.join(mod_rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, "pub use crate::error;\n").unwrap();
    }
    write_minimal_facade_fixture(&dir, true, "pub mod sub;");
    fs::write(
        dir.join("crates/domain_ai/src/ai/sub.rs"),
        "pub enum SubmoduleOptimizationError { Diverged }\n",
    )
    .unwrap();

    let apis = extract_apis_in_dir(&dir);
    let violations = check_facade_completeness_in_dir(&dir, &apis);
    assert!(
        violations
            .iter()
            .any(|v| v.contains("SubmoduleOptimizationError")
                && v.contains("crates/domain_ai/src/ai/sub.rs")),
        "Expected submodule error gap violation, got: {violations:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn rejects_duplicate_error_names_across_reachable_modules() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_duplicate_error_name");
    for mod_rel in DOMAIN_INNER_MODS {
        let p = dir.join(mod_rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, "pub use crate::error;\n").unwrap();
    }
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
        "crates/domain_ai/src/a.rs".into(),
        vec![
            item("pub fn alpha() -> f64"),
            item("pub struct SharedState"),
            item("pub mod __doctest_facade {}"),
        ],
    );
    apis.insert(
        "crates/domain_physics/src/b.rs".into(),
        vec![
            item("pub fn alpha() -> f64"),
            item("pub struct SharedState"),
        ],
    );
    apis.insert(
        "crates/unified_verification/src/ignored.rs".into(),
        vec![item("pub fn unreachable_tool_fn()")],
    );

    let (parsed, total_attempts) = build_parsed_signatures_counted(&apis);
    assert_eq!(parsed.len(), 2);
    assert_eq!(total_attempts, 2);
}

#[test]
fn detects_unreexported_tier2_engine_error_and_missing_umbrella_from() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_tier2_engine_error_gap");
    for mod_rel in DOMAIN_INNER_MODS {
        let p = dir.join(mod_rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, "pub use crate::error;\n").unwrap();
    }
    write_minimal_facade_fixture(&dir, true, "");
    let linalg_dir = dir.join("crates/02_engines/engine_linalg/src");
    fs::create_dir_all(&linalg_dir).unwrap();
    fs::write(linalg_dir.join("lib.rs"), "pub mod types;\n").unwrap();
    fs::write(
        linalg_dir.join("types.rs"),
        "pub struct SolverWorkspace;\npub enum EngineKernelError { Singular }\n",
    )
    .unwrap();

    let apis = extract_apis_in_dir(&dir);
    let violations = check_facade_completeness_in_dir(&dir, &apis);
    assert!(
        violations
            .iter()
            .any(|v| v.contains("Facade error gap") && v.contains("EngineKernelError")),
        "Expected Tier 2 engine facade error gap, got: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|v| v.contains("Umbrella conversion gap")
                && v.contains("AIError")
                && v.contains("From<EngineKernelError>")),
        "Expected Tier 2 engine umbrella From gap, got: {violations:?}"
    );
    assert!(
        !violations.iter().any(|v| v.contains("SolverWorkspace")),
        "Non-error engine structs must not be flagged as errors: {violations:?}"
    );

    fs::write(
        dir.join("math_explorer/src/error.rs"),
        "pub use domain_ai::error::AIError;\npub use engine_linalg::EngineKernelError;\n",
    )
    .unwrap();
    fs::write(
        dir.join("crates/domain_ai/src/error.rs"),
        "pub enum AIError { Invalid, Engine(#[from] EngineKernelError) }\n",
    )
    .unwrap();
    let apis_fixed = extract_apis_in_dir(&dir);
    let fixed_violations = check_facade_completeness_in_dir(&dir, &apis_fixed);
    assert!(
        fixed_violations.is_empty(),
        "Expected zero violations once Tier 2 error is re-exported and converted, got: {fixed_violations:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn detects_missing_umbrella_from_for_domain_and_cross_crate_errors() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_umbrella_from_gap");
    for mod_rel in DOMAIN_INNER_MODS {
        let p = dir.join(mod_rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, "pub use crate::error;\n").unwrap();
    }
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
fn detects_tier0_foundation_error_and_scans_child_mod_conversions() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_tier0_rng_and_child_mod");
    for mod_rel in DOMAIN_INNER_MODS {
        let p = dir.join(mod_rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, "pub use crate::error;\n").unwrap();
    }
    write_minimal_facade_fixture(&dir, true, "");
    let rng_dir = dir.join("crates/00_foundation/math_rng/src");
    fs::create_dir_all(&rng_dir).unwrap();
    fs::write(
        rng_dir.join("lib.rs"),
        "mod private { pub trait Sealed {} }\npub trait ScalarSample: private::Sealed {}\npub enum QmcKernelError { InvalidBase }\n",
    )
    .unwrap();

    let apis = extract_apis_in_dir(&dir);
    let violations = check_facade_completeness_in_dir(&dir, &apis);
    assert!(
        violations
            .iter()
            .any(|v| v.contains("Facade error gap") && v.contains("QmcKernelError")),
        "Expected Tier 0 facade error gap, got: {violations:?}"
    );
    assert!(
        violations
            .iter()
            .any(|v| v.contains("AIError") && v.contains("From<QmcKernelError>")),
        "Expected Tier 0 umbrella From gap, got: {violations:?}"
    );
    assert!(
        !violations.iter().any(|v| v.contains("Sealed")),
        "Sealed private supertrait must not trigger unexported type violation: {violations:?}"
    );

    fs::write(
        dir.join("math_explorer/src/error.rs"),
        "pub use domain_ai::error::AIError;\npub use math_rng::QmcKernelError;\n",
    )
    .unwrap();
    let ai_err_dir = dir.join("crates/domain_ai/src/error");
    fs::create_dir_all(&ai_err_dir).unwrap();
    fs::write(
        dir.join("crates/domain_ai/src/error.rs"),
        "pub enum AIError { Invalid, Qmc(QmcKernelError) }\nmod conversions;\n",
    )
    .unwrap();
    fs::write(
        ai_err_dir.join("conversions.rs"),
        "use super::*;\nimpl From<QmcKernelError> for AIError { fn from(e: QmcKernelError) -> Self { Self::Qmc(e) } }\n",
    )
    .unwrap();

    let apis_fixed = extract_apis_in_dir(&dir);
    let fixed_violations = check_facade_completeness_in_dir(&dir, &apis_fixed);
    assert!(
        fixed_violations.is_empty(),
        "Expected zero violations with child mod conversions.rs, got: {fixed_violations:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

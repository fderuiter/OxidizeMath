use super::math_contracts::audit_math_contracts_with_cache;
use crate::utils::test_utils::TempTestDir;
use oxidize_core::traceability::AstCache;
use std::fs;

fn setup_math_crate(dir: &TempTestDir, crate_rel: &str, lib_src: &str) -> Vec<String> {
    let src_dir = dir.join(crate_rel).join("src");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), lib_src).unwrap();
    vec![crate_rel.to_string()]
}

#[test]
fn test_math_contract_passes_complete_inline_and_embed_theory_items() {
    let dir = TempTestDir::new("math_contract_pass");
    fs::create_dir_all(dir.join("docs/math_specs")).unwrap();
    fs::write(dir.join("docs/math_specs/pure_math.md"), "# Spec\n").unwrap();

    let src = r#"
/// Stochastic gradient descent step $w_{t+1} = w_t - \eta g_t$.
///
/// Reference: Robbins & Monro (1951), DOI: 10.1214/aoms/1177729586;
/// [spec](../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Step sizes satisfy $0 < \eta < 2 / L$ for $L$-smooth objectives.
///
/// # Domain of Validity
/// Finite `f64` gradients and learning rates (`learning_rate.is_finite()`).
pub struct SGD {
    /// Step size.
    pub learning_rate: f64,
}

/// Non-solver helper configuration is exempt from the 4-part solver contract.
pub struct SolverConfig {
    /// Max iterations.
    pub max_iters: usize,
}

/// Range-bin beat conversion via `#[embed_theory]` (arXiv:2301.12345).
///
/// # Stability
/// Numerically well-conditioned for non-zero chirp bandwidth $B > 0$.
///
/// # Domain of Validity
/// Finite non-negative beat frequencies $f_b \ge 0$.
#[verified_engine::embed_theory("R = c f_b / (2 S)")]
pub fn beat_to_range(f_b: f64) -> f64 {
    f_b
}
"#;
    let members = setup_math_crate(&dir, "crates/pure_math", src);
    let mut cache = AstCache::new();
    let violations = audit_math_contracts_with_cache(&dir, &members, &mut cache);
    assert!(
        violations.is_empty(),
        "Expected zero violations, got: {violations:?}"
    );
    assert_eq!(cache.parse_calls(), 1);
}

#[test]
fn test_math_contract_fails_closed_for_each_missing_section_with_file_line() {
    let dir = TempTestDir::new("math_contract_missing_parts");
    let src = r#"/// Missing equation, citation, stability, and domain.
pub struct BrokenSolver;

/// Has equation $x_{n+1} = x_n$ and DOI: 10.1137/1.9780898719574.
///
/// # Stability
///
/// # Domain of Validity
/// Finite inputs.
pub struct EmptyStabilityModel;

/// Integer wrapper.
pub struct Integer(pub i64);

impl Integer {
    /// Missing contract on Integer::pow_mod.
    pub fn pow_mod(&self) -> i64 {
        self.0
    }
}
"#;
    let members = setup_math_crate(&dir, "crates/pure_math", src);
    let mut cache = AstCache::new();
    let violations = audit_math_contracts_with_cache(&dir, &members, &mut cache);
    assert_eq!(violations.len(), 3, "Got: {violations:?}");
    assert!(violations[0].contains("crates/pure_math/src/lib.rs:2:"));
    assert!(violations[0].contains("struct BrokenSolver"));
    assert!(violations[0].contains("governing LaTeX equation"));
    assert!(violations[0].contains("DOI"));
    assert!(violations[0].contains("`# Stability`"));
    assert!(violations[0].contains("`# Domain of Validity`"));

    assert!(violations[1].contains("crates/pure_math/src/lib.rs:10:"));
    assert!(violations[1].contains("struct EmptyStabilityModel"));
    assert!(violations[1].contains("`# Stability`"));
    assert!(!violations[1].contains("`# Domain of Validity`"));

    assert!(violations[2].contains("crates/pure_math/src/lib.rs:17:"));
    assert!(violations[2].contains("fn Integer::pow_mod"));
}

#[test]
fn test_math_contract_resolves_include_str_snippets_and_fails_on_missing_snippet() {
    let dir = TempTestDir::new("math_contract_include_str");
    fs::create_dir_all(dir.join("docs/math_specs/snippets")).unwrap();
    fs::write(dir.join("docs/math_specs/domain_ai.md"), "# AI Spec\n").unwrap();
    fs::write(
        dir.join("docs/math_specs/snippets/kalman.md"),
        "Kalman update $\\hat{x}_{k|k} = \\hat{x}_{k|k-1} + K_k y_k$.\n\n\
         See [AI Spec](../domain_ai.md).\n\n\
         # Stability\nJoseph-form covariance update preserves symmetry.\n\n\
         # Domain of Validity\nPositive-definite measurement noise covariance $R \\succ 0$.\n",
    )
    .unwrap();

    let ok_src = r#"
#[doc = include_str!("../../../docs/math_specs/snippets/kalman.md")]
pub struct KalmanFilter;
"#;
    let members = setup_math_crate(&dir, "crates/domain_ai", ok_src);
    let mut cache = AstCache::new();
    assert!(
        audit_math_contracts_with_cache(&dir, &members, &mut cache).is_empty(),
        "External include_str! snippet should satisfy all 4 contract sections"
    );

    let missing_src = r#"
#[doc = include_str!("../../../docs/math_specs/snippets/does_not_exist.md")]
pub struct MissingSnippetSolver;
"#;
    let bad_members = setup_math_crate(&dir, "crates/domain_physics", missing_src);
    let mut bad_cache = AstCache::new();
    let violations = audit_math_contracts_with_cache(&dir, &bad_members, &mut bad_cache);
    assert!(
        violations
            .iter()
            .any(|v| v.contains("unreadable #[doc = include_str!(...)]")),
        "Missing include_str! target must fail closed: {violations:?}"
    );
}

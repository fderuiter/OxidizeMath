//! End-to-end `rustc` tests for `#[embed_theory]` diagnostics and positive macro expansions.
#![cfg(not(target_arch = "wasm32"))]

mod support;

use support::{RustcOutcome, compile_in_workspace, compile_snippet};

fn assert_succeeded(out: &RustcOutcome) {
    assert!(
        out.success,
        "expected rustc to succeed for `{}`, stderr:\n{}",
        out.file_name, out.stderr
    );
}

const SAMPLE_TEX: &str = r#"\documentclass{article}
\begin{document}
\begin{equation}
\label{eq:ratio}
y = \frac{a}{b}
\end{equation}
\begin{equation}
\label{eq:series}
s = \sum_{i=1}^{n} x_i
\end{equation}
\end{document}
"#;

#[test]
fn mode_12_embed_theory_rejects_missing_file_and_bad_args_at_literal_span() {
    let code = r#"use verified_engine_macros::embed_theory;

#[embed_theory("papers/does_not_exist.tex", label = "eq:ratio")]
pub fn missing_tex_fn(a: f64, b: f64) -> f64 {
    a / b
}

#[embed_theory("")]
pub fn empty_tex_fn() {}

#[embed_theory("papers/sample.tex")]
pub fn missing_label_arg_fn() {}

#[embed_theory("papers/sample.tex", section = "intro")]
pub fn bad_arg_key_fn() {}
"#;
    let out = compile_in_workspace(
        "mode_12_missing_tex",
        "crates/sub",
        &[("papers/sample.tex", SAMPLE_TEX)],
        code,
    );
    out.assert_failed_without_panic();
    out.assert_has_error("LaTeX file not found: `papers/does_not_exist.tex`");
    out.assert_has_error("LaTeX file path must not be empty");
    out.assert_has_error("At least one `label = \"...\"` must be provided");
    out.assert_has_error("Unknown attribute argument, expected `label`");
    out.assert_has_span(3, 16);
    out.assert_has_span(8, 16);
    out.assert_has_span(11, 16);
    out.assert_has_span(14, 37);
}

#[test]
fn mode_13_embed_theory_rejects_missing_label_shadow_docs_and_semantic_violation() {
    let code = r#"use verified_engine_macros::embed_theory;

#[embed_theory("papers/sample.tex", label = "eq:nonexistent")]
pub fn unknown_label_fn(a: f64, b: f64) -> f64 {
    a / b
}

#[embed_theory("papers/sample.tex", label = "eq:ratio")]
/// Manual $$ y = \frac{a}{b} $$ conflicting with embedded LaTeX
pub fn shadow_doc_fn(a: f64, b: f64) -> f64 {
    a / b
}

#[embed_theory("papers/sample.tex", label = "eq:ratio")]
pub fn missing_division_fn(a: f64, b: f64) -> f64 {
    a + b
}

#[embed_theory("papers/sample.tex", label = "eq:series")]
pub fn missing_pow_or_sum_fn(x: f64) -> f64 {
    x + 1.0
}
"#;
    let out = compile_in_workspace(
        "mode_13_embed_semantic",
        "crates/sub",
        &[("papers/sample.tex", SAMPLE_TEX)],
        code,
    );
    out.assert_failed_without_panic();
    out.assert_has_error(
        "Label `eq:nonexistent` not found or not in a supported environment in `papers/sample.tex`",
    );
    out.assert_has_error("prevent shadow-mirroring");
    out.assert_has_error(
        "Semantic Integrity Violation: Missing operation in Rust implementation: expected Div based on LaTeX formula",
    );
    out.assert_has_error(
        "Semantic Integrity Violation: Missing operation in Rust implementation: expected Pow based on LaTeX formula",
    );
    out.assert_has_span(3, 45);
    out.assert_has_span(9, 1);
    out.assert_has_span(15, 8);
    out.assert_has_span(20, 8);
}

#[test]
fn positive_verified_supports_const_fn_and_bodyless_trait_signatures() {
    let valid_code = r#"use verified_engine_macros::verified;

#[verified]
pub const fn kinetic_const(mass: f64, vel: f64) -> f64 {
    0.5 * mass * vel * vel
}

pub const SAMPLE_ENERGY: f64 = kinetic_const(2.0, 3.0);

pub trait Stepper {
    #[verified]
    fn step_signature(&mut self, dt: f64) -> f64;

    #[verified]
    fn default_step(&mut self, dt: f64) -> f64 {
        self.step_signature(dt) + 1.0
    }
}
"#;
    let ok = compile_snippet("pos_const_and_trait", valid_code);
    assert_succeeded(&ok);

    let bad_const_rec = r#"use verified_engine_macros::verified;

#[verified]
pub const fn bad_const_fact(n: u64) -> u64 {
    if n <= 1 { 1 } else { n * bad_const_fact(n - 1) }
}
"#;
    let rec_err = compile_snippet("neg_const_rec", bad_const_rec);
    rec_err.assert_failed_without_panic();
    rec_err.assert_has_error(
        "Direct recursion is prohibited in #[verified] functions (NASA Power of 10 Rule 1)",
    );
    rec_err.assert_has_span(5, 32);
}

#[test]
fn positive_embed_theory_supports_top_level_nested_crates_traits_and_shared_labels() {
    let code = r#"use verified_engine_macros::{embed_theory, verified};

pub trait RatioEvaluator {
    #[embed_theory("papers/sample.tex", label = "eq:ratio")]
    fn eval_ratio_sig(&self, a: f64, b: f64) -> f64;
}

pub struct Solver;

impl RatioEvaluator for Solver {
    #[embed_theory("papers/sample.tex", label = "eq:ratio")]
    #[verified]
    fn eval_ratio_sig(&self, a: f64, b: f64) -> f64 {
        a / b
    }
}

#[embed_theory("papers/sample.tex", label = "eq:ratio")]
pub fn first_shared_label(a: f64, b: f64) -> f64 {
    a / b
}

#[embed_theory("papers/sample.tex", label = "eq:ratio")]
pub fn second_shared_label(a: f64, b: f64) -> f64 {
    (2.0 * a) / b
}
"#;
    let top_level = compile_in_workspace(
        "pos_embed_top_level",
        "math_explorer_like",
        &[("papers/sample.tex", SAMPLE_TEX)],
        code,
    );
    assert_succeeded(&top_level);

    let deeply_nested = compile_in_workspace(
        "pos_embed_deep_nested",
        "crates/domain/subcrate",
        &[("papers/sample.tex", SAMPLE_TEX)],
        code,
    );
    assert_succeeded(&deeply_nested);
}

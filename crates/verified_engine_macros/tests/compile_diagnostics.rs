//! End-to-end `rustc` diagnostic tests for `#[verified]` and `#[derive(Theory)]` failure modes.
#![cfg(not(target_arch = "wasm32"))]

mod support;

use support::compile_snippet;

#[test]
fn mode_01_verified_rejects_unsupported_items_with_spanned_plain_text() {
    let code = r#"use verified_engine_macros::verified;

#[verified]
pub struct BadStruct;

#[verified]
pub enum BadEnum {
    A,
}

#[verified]
pub mod bad_mod {}

pub struct Holder;
#[verified]
impl Holder {}
"#;
    let out = compile_snippet("mode_01_unsupported_items", code);
    out.assert_failed_without_panic();
    out.assert_has_error(
        "`#[verified]` can only be applied to functions and methods, not `struct` items",
    );
    out.assert_has_error(
        "`#[verified]` can only be applied to functions and methods, not `enum` items",
    );
    out.assert_has_error(
        "`#[verified]` can only be applied to functions and methods, not `mod` items",
    );
    out.assert_has_error(
        "`#[verified]` can only be applied to functions and methods, not `impl` blocks",
    );
    out.assert_has_span(4, 5);
    out.assert_has_span(7, 5);
    out.assert_has_span(12, 5);
    out.assert_has_span(16, 1);
}

#[test]
fn mode_02_verified_rejects_unknown_and_malformed_attribute_keys() {
    let code = r#"use verified_engine_macros::verified;

#[verified(unknown = "reason")]
pub fn bad_unknown() {}

#[verified("opt_out")]
pub fn bad_string_literal() {}

#[verified(not_opt_out = "reason")]
pub fn bad_substring_trick() {}

#[verified(opt_out)]
pub fn bad_bare_key() {}
"#;
    let out = compile_snippet("mode_02_bad_verified_keys", code);
    out.assert_failed_without_panic();
    out.assert_has_error("unknown `#[verified]` attribute key `unknown`");
    out.assert_has_error(
        "valid keys are `opt_out = \"reason\"` and `allow_recursion = \"reason\"`",
    );
    out.assert_has_error("unknown `#[verified]` attribute key `not_opt_out`");
    out.assert_has_error("expected `opt_out = \"reason\"` with a non-empty string literal");
    out.assert_has_span(3, 12);
    out.assert_has_span(6, 12);
    out.assert_has_span(9, 12);
    out.assert_has_span(12, 12);
}

#[test]
fn mode_03_verified_rejects_empty_and_whitespace_reasons() {
    let code = r#"use verified_engine_macros::verified;

#[verified(opt_out = "")]
pub fn empty_opt_out() {}

#[verified(allow_recursion = "   ")]
pub fn blank_allow_recursion() {}
"#;
    let out = compile_snippet("mode_03_empty_reasons", code);
    out.assert_failed_without_panic();
    out.assert_has_error("`opt_out` reason must not be empty");
    out.assert_has_error("`allow_recursion` reason must not be empty");
    out.assert_has_span(3, 22);
    out.assert_has_span(6, 30);
}

#[test]
fn mode_04_verified_rejects_direct_recursion_with_spanned_plain_text() {
    let code = r#"use verified_engine_macros::verified;

#[verified]
pub fn free_rec(n: u64) -> u64 {
    if n == 0 { 0 } else { free_rec(n - 1) }
}

pub struct Calc;
impl Calc {
    #[verified]
    pub fn self_path_rec(n: u64) -> u64 {
        if n == 0 { 0 } else { Self::self_path_rec(n - 1) }
    }

    #[verified]
    pub fn method_rec(&self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.method_rec(n - 1) }
    }
}
"#;
    let out = compile_snippet("mode_04_direct_recursion", code);
    out.assert_failed_without_panic();
    out.assert_has_error(
        "Direct recursion is prohibited in #[verified] functions (NASA Power of 10 Rule 1)",
    );
    out.assert_has_span(5, 28);
    out.assert_has_span(12, 32);
    out.assert_has_span(17, 32);
}

#[test]
fn mode_05_verified_rejects_over_60_statements_including_allow_recursion() {
    let mut body = String::new();
    for i in 0..61 {
        body.push_str(&format!("    let _v{i} = {i};\n"));
    }
    let code = format!(
        "use verified_engine_macros::verified;\n\n#[verified]\npub fn plain_too_long() {{\n{body}}}\n\n#[verified(allow_recursion = \"delegates to helper\")]\npub fn allow_rec_too_long() {{\n{body}}}\n"
    );
    let out = compile_snippet("mode_05_statement_limit", &code);
    out.assert_failed_without_panic();
    out.assert_has_error(
        "Function `plain_too_long` exceeds the 60-statement limit (NASA Power of 10 Rule 4): found 61 statements",
    );
    out.assert_has_error(
        "Function `allow_rec_too_long` exceeds the 60-statement limit (NASA Power of 10 Rule 4): found 61 statements",
    );
    out.assert_has_span(4, 8);
    out.assert_has_span(69, 8);
}

#[test]
fn mode_06_theory_rejects_enum_union_and_tuple_struct() {
    let code = r#"use verified_engine_macros::Theory;

#[derive(Theory)]
pub enum BadEnum {
    Variant,
}

#[derive(Theory)]
pub union BadUnion {
    pub f: f64,
    pub u: u64,
}

#[derive(Theory)]
pub struct BadTuple(pub f64);
"#;
    let out = compile_snippet("mode_06_theory_containers", code);
    out.assert_failed_without_panic();
    out.assert_has_error("`#[derive(Theory)]` can only be derived for structs, not enums");
    out.assert_has_error("`#[derive(Theory)]` can only be derived for structs, not unions");
    out.assert_has_error(
        "`#[derive(Theory)]` requires named fields and cannot be derived for tuple structs",
    );
    out.assert_has_span(4, 5);
    out.assert_has_span(9, 5);
    out.assert_has_span(15, 20);
}

#[test]
fn mode_07_theory_reports_unknown_struct_keys_and_accumulates_field_errors() {
    let code = r#"use verified_engine_macros::Theory;

#[derive(Theory)]
#[theory(author = "Lorenz")]
pub struct BadStructMeta {
    #[theory(scale = 1.0)]
    pub sigma: f64,
}
"#;
    let out = compile_snippet("mode_07_struct_and_field_errors", code);
    out.assert_failed_without_panic();
    out.assert_has_error(
        "unknown struct-level `#[theory]` key `author`; valid keys are `description`, `phonetic`, `citation`",
    );
    out.assert_has_error(
        "unknown field-level `#[theory]` key `scale`; valid keys are `min`, `max`, `step`",
    );
    out.assert_has_span(4, 10);
    out.assert_has_span(6, 14);
}

#[test]
fn mode_08_theory_rejects_unknown_field_keys_and_non_numeric_values() {
    let code = r#"use verified_engine_macros::Theory;

#[derive(Theory)]
#[theory(description = "Model")]
pub struct BadFieldMeta {
    #[theory(units = "m/s")]
    pub v: f64,
    #[theory(min = "zero", max = 10.0, step = 0.5)]
    pub x: f64,
}
"#;
    let out = compile_snippet("mode_08_bad_field_meta", code);
    out.assert_failed_without_panic();
    out.assert_has_error(
        "unknown field-level `#[theory]` key `units`; valid keys are `min`, `max`, `step`",
    );
    out.assert_has_error("expected a finite integer or float literal for `#[theory]` bound");
    out.assert_has_span(6, 14);
    out.assert_has_span(8, 20);
}

#[test]
fn mode_09_theory_rejects_overflowing_and_non_finite_literals_without_panic() {
    let code = r#"use verified_engine_macros::Theory;

#[derive(Theory)]
#[theory(description = "Overflow checks")]
pub struct OverflowBounds {
    #[theory(min = -1e999, max = 1e999, step = 0.1)]
    pub alpha: f64,
    #[theory(min = 0.0, max = f64::NAN, step = 0.1)]
    pub beta: f64,
}
"#;
    let out = compile_snippet("mode_09_non_finite_bounds", code);
    out.assert_failed_without_panic();
    out.assert_has_error("parameter bound must be a finite number (literal overflows `f64`)");
    out.assert_has_error("expected a finite integer or float literal for `#[theory]` bound");
    out.assert_has_span(6, 21);
    out.assert_has_span(6, 34);
    out.assert_has_span(8, 31);
}

#[test]
fn mode_10_theory_rejects_inverted_bounds_and_non_positive_step() {
    let code = r#"use verified_engine_macros::Theory;

#[derive(Theory)]
#[theory(description = "Range checks")]
pub struct InvalidRanges {
    #[theory(min = 10.0, max = 5.0, step = 0.5)]
    pub inverted: f64,
    #[theory(min = 0, max = 10, step = 0)]
    pub zero_step: i32,
    #[theory(min = 0.0, max = 10.0, step = -0.25)]
    pub negative_step: f64,
}
"#;
    let out = compile_snippet("mode_10_invalid_ranges", code);
    out.assert_failed_without_panic();
    out.assert_has_error("`min` (10) must not exceed `max` (5)");
    out.assert_has_error("`step` must be greater than 0.0, found 0");
    out.assert_has_error("`step` must be greater than 0.0, found -0.25");
    out.assert_has_span(6, 20);
    out.assert_has_span(8, 40);
    out.assert_has_span(10, 45);
}

#[test]
fn mode_11_theory_spans_non_numeric_field_cast_error_to_field_type() {
    let code = r#"use verified_engine_macros::Theory;

#[derive(Theory)]
#[theory(description = "Non-numeric field")]
pub struct NonNumericModel {
    #[theory(min = 0.0, max = 1.0, step = 0.1)]
    pub name: String,
}
"#;
    let out = compile_snippet("mode_11_non_numeric_field", code);
    out.assert_failed_without_panic();
    out.assert_has_error("the trait bound `String: TheoryNumeric` is not satisfied");
    out.assert_has_span(7, 15);
}

#[test]
fn mode_12_theory_validates_calibrated_default_within_bounds() {
    let bad_code = r#"use verified_engine_macros::Theory;

#[derive(Theory)]
#[theory(description = "Default checks")]
pub struct OutOfBoundsDefault {
    #[theory(min = 0.0, max = 5.0, step = 0.1, default = 12.0)]
    pub g_l: f64,
    #[theory(min = 1.0, max = 10.0, step = 1.0, default = -2.0)]
    pub i_ext: f64,
}
"#;
    let out = compile_snippet("mode_12_out_of_bounds_default", bad_code);
    out.assert_failed_without_panic();
    out.assert_has_error("`default` (12) must lie within `[0, 5]`");
    out.assert_has_error("`default` (-2) must lie within `[1, 10]`");

    let ok_code = r#"use scientific_metadata::theory::TheoryDescribable;
use verified_engine_macros::Theory;

#[derive(Theory)]
#[theory(description = "Calibrated model")]
pub struct CalibratedModel {
    #[theory(min = 0.0, max = 5.0, step = 0.1, default = 0.3)]
    pub g_l: f64,
    #[theory(min = 0.0, max = 50.0, step = 1.0)]
    pub i_ext: f64,
}

pub fn check() {
    let m = CalibratedModel { g_l: 0.3, i_ext: 10.0 };
    let p = m.theory_parameters();
    assert_eq!(p["g_l"].default, Some(0.3));
    assert_eq!(p["g_l"].effective_default(), 0.3);
    assert_eq!(p["i_ext"].default, None);
    assert_eq!(p["i_ext"].effective_default(), 25.0);
}
"#;
    let ok_out = compile_snippet("mode_12_valid_default", ok_code);
    assert!(
        ok_out.success,
        "valid calibrated default failed to compile:\n{}",
        ok_out.stderr
    );
}

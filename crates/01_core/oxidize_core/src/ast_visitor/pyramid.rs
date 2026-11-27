use syn::ItemMacro;
use syn::spanned::Spanned;

use super::AstVisitor;
use super::maturity::inspect_theory_verification_macro;
use super::vacuous::{find_stochastic_zero_init_line, span_line};

/// Structured `Tier6_Formal` proof and theorem binding extracted from `formal_proof_verification!`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormalProofMacroBinding {
    /// Registered `traceability.toml` module key (`module = "..."`).
    pub module: String,
    /// Lean 4 proof filename under `proofs/OxidizeMath/` (`proof = "..."`).
    pub proof: String,
    /// Bound Lean 4 theorem identifiers (`theorems = ["...", ...]`).
    pub theorems: Vec<String>,
    /// Whether the companion `test = { ... }` block contains at least one assertion.
    pub has_assertions: bool,
    /// 1-indexed Rust source line of the macro invocation.
    pub line: usize,
}

pub(super) fn classify_verification_macro(name: &str) -> Option<&'static str> {
    match name {
        "theory_verification" | "mms_convergence_verification" => Some("Tier1_Convergence"),
        "property_invariant_verification" => Some("Tier2_Property"),
        "golden_fixture_verification" => Some("Tier3_Golden"),
        "stochastic_signature_verification" => Some("Tier4_Stochastic"),
        "interval_enclosure_verification" => Some("Tier5_Interval"),
        "formal_proof_verification" => Some("Tier6_Formal"),
        _ => None,
    }
}

pub(super) fn record_verification_macro(visitor: &mut AstVisitor, node: &ItemMacro) {
    let Some(ident) = node.mac.path.segments.last().map(|s| &s.ident) else {
        return;
    };
    let name = ident.to_string();
    let Some(pyramid_tier) = classify_verification_macro(&name) else {
        return;
    };
    let is_stochastic = name == "stochastic_signature_verification";
    let contract_kind = if is_stochastic {
        "Stochastic"
    } else {
        "Deterministic"
    };

    visitor.has_verification_macro = true;
    let macro_line = span_line(node.span());
    if is_stochastic && let Some(zero_line) = find_stochastic_zero_init_line(&node.mac) {
        visitor.has_vacuous_bypass = true;
        let recorded = if zero_line > 0 { zero_line } else { macro_line };
        if visitor.vacuous_bypass_line.is_none() && recorded > 0 {
            visitor.vacuous_bypass_line = Some(recorded);
        }
    }

    let det_info = (!is_stochastic).then(|| inspect_theory_verification_macro(&node.mac));
    let parsed = super::attrs::parse_verification_macro_tokens(&node.mac.tokens);
    let Some(module_name) = parsed.module_name else {
        return;
    };

    if name == "formal_proof_verification" {
        visitor.formal_proof_bindings.push(FormalProofMacroBinding {
            module: module_name.clone(),
            proof: parsed.proof_path.unwrap_or_default(),
            theorems: parsed.theorems,
            has_assertions: parsed.assert_count > 0,
            line: macro_line,
        });
    }

    visitor.record_module(module_name.clone(), macro_line);
    visitor
        .module_tiers
        .insert(module_name.clone(), contract_kind.to_string());
    *visitor
        .tier_counts
        .entry(pyramid_tier.to_string())
        .or_insert(0) += 1;

    record_macro_maturity_and_counts(
        visitor,
        module_name,
        macro_line,
        det_info,
        parsed.assert_count,
    );
}

fn record_macro_maturity_and_counts(
    visitor: &mut AstVisitor,
    module_name: String,
    macro_line: usize,
    det_info: Option<super::maturity::TheoryVerificationInfo>,
    assert_count: usize,
) {
    if let Some(info) = det_info {
        if info.has_non_empty_constants {
            visitor.has_golden_fixture = true;
        }
        if info.has_non_empty_test {
            visitor.non_vacuous_deterministic_modules.push(module_name);
            visitor.has_seeded_rng = true;
        } else {
            visitor
                .vacuous_deterministic_modules
                .push((module_name, macro_line));
        }
    } else {
        visitor.stochastic_modules.push(module_name);
    }

    visitor.semantic_integrity_funcs += 1;
    visitor.total_funcs += 1;
    visitor.active_funcs += 1;
    visitor.verified_funcs += 1;
    visitor.test_funcs += 1;
    visitor.verified_test_funcs += 1;
    visitor.total_asserts += assert_count;
    visitor.verified_asserts += assert_count;
    visitor.test_asserts += assert_count;
    visitor.verified_test_asserts += assert_count;
    visitor.maturity_asserts += assert_count;
}

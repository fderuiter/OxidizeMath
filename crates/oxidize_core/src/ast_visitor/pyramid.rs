use syn::ItemMacro;
use syn::spanned::Spanned;

use super::{AstVisitor, find_stochastic_zero_init_line, span_line};

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
    let Some(tier_name) = classify_verification_macro(&name) else {
        return;
    };

    let macro_line = span_line(node.span());
    if name == "stochastic_signature_verification"
        && let Some(zero_line) = find_stochastic_zero_init_line(&node.mac)
    {
        visitor.has_vacuous_bypass = true;
        let recorded = if zero_line > 0 { zero_line } else { macro_line };
        if visitor.vacuous_bypass_line.is_none() && recorded > 0 {
            visitor.vacuous_bypass_line = Some(recorded);
        }
    }

    let tokens = node.mac.tokens.to_string();
    if let Some(idx) = tokens.find("module = \"") {
        let start = idx + 10;
        if let Some(end) = tokens[start..].find('"') {
            let module_name = &tokens[start..start + end];
            visitor.record_module(module_name.to_string(), macro_line);
            visitor
                .module_tiers
                .insert(module_name.to_string(), tier_name.to_string());
            *visitor
                .tier_counts
                .entry(tier_name.to_string())
                .or_insert(0) += 1;
            visitor.semantic_integrity_funcs += 1;
            visitor.total_funcs += 1;
            visitor.verified_funcs += 1;
            let assert_count = tokens.matches("assert").count().max(3);
            visitor.total_asserts += assert_count;
            visitor.verified_asserts += assert_count;
        }
    }
}

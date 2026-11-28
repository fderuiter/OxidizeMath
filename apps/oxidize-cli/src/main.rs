#![doc = include_str!("../README.md")]

use oxidizemath::prelude::{MaturityTier, TheoryDescribable};

fn summary_banner() -> String {
    let fit_tier = oxidizemath::fit::FitModel::Linear.maturity_tier();
    format!(
        "oxidize-cli v{} (core fit kernel tier: {fit_tier:?})\n\
         Run domain demos via: cargo run --package oxidize-cli --features <domain> --example <name>\n\
         See apps/oxidize-cli/examples/README.md for all 32 registered examples.",
        env!("CARGO_PKG_VERSION")
    )
}

fn main() {
    println!("{}", summary_banner());
    debug_assert_eq!(
        oxidizemath::fit::FitModel::Linear.maturity_tier(),
        MaturityTier::Validated
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_banner_reports_version_and_catalog_hint() {
        let text = summary_banner();
        assert!(text.contains("oxidize-cli v"));
        assert!(text.contains("--package oxidize-cli"));
        let tier = oxidizemath::fit::FitModel::Linear.maturity_tier();
        assert!(text.contains(&format!("tier: {tier:?}")));
    }
}

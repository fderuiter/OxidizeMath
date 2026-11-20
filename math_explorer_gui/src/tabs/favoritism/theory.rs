#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// GUI [`TheoryDescribable`] wrapper for `domain_applied::applied::favoritism::UnifiedFavoritismModel`.
#[derive(Default, Clone, Debug)]
pub struct FavoritismTheory;

impl TheoryDescribable for FavoritismTheory {
    fn theory_description(&self) -> String {
        "Unified Favoritism Theory (UFT) aggregates proximity integration, exponential contact decay, \
         practical and emotional gift valuations, weighted personality traits, and crisis-intervention \
         multipliers via Clenshaw-Curtis quadrature into a comparative parental favoritism score."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Unified Favoritism Theory comparative quadrature score".to_string()
    }

    fn theory_citation(&self) -> String {
        "Unified Favoritism Theory (UFT): Quantitative Inheritance & Behavioral Strategy Model, OxidizeMath Applied Compendium."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut descs = HashMap::new();
        descs.insert(
            "proximity_quadrature".to_string(),
            "Clenshaw-Curtis integration of visit utility attenuated by geographic separation."
                .to_string(),
        );
        descs.insert(
            "contact_decay".to_string(),
            "Exponential attenuation of call frequency over elapsed days since last contact."
                .to_string(),
        );
        descs.insert(
            "crisis_multiplier".to_string(),
            "Nonlinear amplification applied to crisis interventions and birth-order weight."
                .to_string(),
        );
        descs
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"F = B_{\text{order}} \cdot M_{\text{crisis}} \int_0^T \frac{C(t)\,\left(G_p + G_e + P\right)}{1 + \alpha x_0}\,dt"
                .to_string(),
        )
    }
}

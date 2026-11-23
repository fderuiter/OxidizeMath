#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// GUI [`TheoryDescribable`] metadata for [`super::CannibalismTool`].
#[derive(Default, Clone, Debug)]
pub struct CannibalismTheory;

impl TheoryDescribable for CannibalismTheory {
    fn theory_description(&self) -> String {
        "Age-structured McKendrick-von Foerster transport PDE with nonlocal cannibalistic \
         mortality and boundary renewal n(t, 0) = b(t). Below the Diekmann Hopf threshold \
         cohorts relax toward the analytical equilibrium A*; above threshold, adult predation \
         on vulnerable juveniles carves diagonal missing-generation troughs across the (t, a) \
         Lexis plane, while Tribolium callow pupa cannibalism drives period-doubling chaos."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "McKendrick von Foerster age structured cannibalism partial differential equation and Tribolium bifurcation lab"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Diekmann, O., Nisbet, R. M., Gurney, W. S. C., & van den Bosch, F. (1986). Simple mathematical models for cannibalism: A critique and a new approach. Mathematical Biosciences, 78(1), 21-46; Costantino, R. F., Desharnais, R. A., Cushing, J. M., & Dennis, B. (1997). Chaotic dynamics in an insect population. Science, 275(5298), 389-391."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut descs = HashMap::new();
        descs.insert(
            "mckendrick_transport".to_string(),
            "First-order hyperbolic age transport along characteristic lines da/dt = 1 with exponential survival."
                .to_string(),
        );
        descs.insert(
            "diekmann_hopf".to_string(),
            "Supercritical Hopf bifurcation into single-generation cohort waves driven by adult-on-juvenile predation."
                .to_string(),
        );
        descs.insert(
            "tribolium_lpa".to_string(),
            "Nonlinear Larva-Pupa-Adult flour beetle stage dynamics exhibiting 2-cycles, invariant loops, and chaos."
                .to_string(),
        );
        descs
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\frac{\partial n}{\partial t} + \frac{\partial n}{\partial a} = -\left(\mu_0(a) + c \int_0^\infty \phi(a, a')\,n(t, a')\,da'\right) n(t, a), \quad n(t, 0) = \int_0^\infty \beta(a, N)\,n(t, a)\,da"
                .to_string(),
        )
    }
}

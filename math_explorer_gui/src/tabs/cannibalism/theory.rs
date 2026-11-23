#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// GUI [`TheoryDescribable`] metadata for age-structured cannibalism PDE, delay, and *Tribolium* models.
#[derive(Default, Clone, Debug)]
pub struct CannibalismTheory;

impl TheoryDescribable for CannibalismTheory {
    fn theory_description(&self) -> String {
        "Age-structured cannibalism couples McKendrick-von Foerster advection along characteristics \
         with Diekmann vulnerability and Ricker/Holling saturation kernels, van den Bosch juvenile-adult \
         maturation delay dynamics generating dwarf-and-giant Hopf cohort waves, stage-structured \
         Tribolium callow predation period-doubling cascades, and guarded two-dimensional cannibal-normal ODEs."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Age-structured McKendrick von Foerster cannibalism PDE, Hopf cohort cycles, and Tribolium bifurcation lab"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Diekmann et al. (1986), Simple mathematical models for cannibalism, Math. Biosci. 78(1), 21-46; \
         van den Bosch et al. (1988), Cannibalism as a Life Boat Mechanism, J. Math. Biol. 26, 619-633; \
         Costantino et al. (1997), Chaotic dynamics in an insect population, Science 275(5298), 389-391."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut descs = HashMap::new();
        descs.insert(
            "mckendrick_pde".to_string(),
            "Characteristic upwind discretization of the McKendrick-von Foerster transport PDE \
             with boundary renewal quadrature."
                .to_string(),
        );
        descs.insert(
            "diekmann_saturation".to_string(),
            "Age-specific cannibalism mortality modulated by vulnerable cohort biomass and \
             Ricker or Holling functional response saturation."
                .to_string(),
        );
        descs.insert(
            "hopf_cohorts".to_string(),
            "Maturation delay across juvenile age [0, alpha] interacting with adult cannibalism \
             pressure to trigger sustained dwarf-and-giant cohort waves."
                .to_string(),
        );
        descs.insert(
            "tribolium_lpa".to_string(),
            "Nonlinear Larva-Pupa-Adult flour beetle stage dynamics exhibiting 2-cycles, invariant loops, and chaos."
                .to_string(),
        );
        descs.insert(
            "lumped_ode".to_string(),
            "Runge-Kutta 4 integration of the guarded two-compartment cannibal and normal population ODE."
                .to_string(),
        );
        descs
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\frac{\partial n}{\partial t} + \frac{\partial n}{\partial a} = -\left(\nu(a) + c(t)\,\Phi(c(t))\,C(a)\right)n(t,a), \quad n(t,0)=\int_0^{a_{\max}}\beta(a,N)\,n(t,a)\,da"
                .to_string(),
        )
    }
}

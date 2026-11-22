#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// GUI [`TheoryDescribable`] metadata for age-structured cannibalism PDE and delay models.
#[derive(Default, Clone, Debug)]
pub struct CannibalismTheory;

impl TheoryDescribable for CannibalismTheory {
    fn theory_description(&self) -> String {
        "Age-structured cannibalism couples McKendrick-von Foerster advection along characteristics \
         with Diekmann vulnerability and Ricker saturation kernels, van den Bosch juvenile-adult \
         maturation delay dynamics generating dwarf-and-giant Hopf cohort cycles, and a lumped \
         two-dimensional cannibal-normal ODE approximation."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Age-structured McKendrick von Foerster cannibalism PDE and Hopf cohort cycles".to_string()
    }

    fn theory_citation(&self) -> String {
        "Diekmann et al. (1986), The Dynamics of Cannibalism, Theor. Popul. Biol. 30, 99-121; \
         van den Bosch et al. (1988), Cannibalism as a Life Boat Mechanism, J. Math. Biol. 26, 619-633."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut descs = HashMap::new();
        descs.insert(
            "mckendrick_pde".to_string(),
            "Characteristic upwind discretization of the McKendrick-von Foerster transport PDE \
             with trapezoidal renewal boundary quadrature."
                .to_string(),
        );
        descs.insert(
            "diekmann_saturation".to_string(),
            "Age-specific cannibalism mortality modulated by vulnerable cohort biomass and \
             Ricker functional response saturation."
                .to_string(),
        );
        descs.insert(
            "hopf_cohorts".to_string(),
            "Maturation delay across juvenile age [0, alpha] interacting with adult cannibalism \
             pressure to trigger sustained dwarf-and-giant cohort waves."
                .to_string(),
        );
        descs.insert(
            "lumped_ode".to_string(),
            "Runge-Kutta 4 integration of the two-compartment cannibal and normal population ODE."
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

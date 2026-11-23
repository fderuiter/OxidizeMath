use crate::epidemiology::compartmental::{SIRModel, SIRState};
use pure_math::pure_math::analysis::ode::Solver;
pub use pure_math::pure_math::analysis::stochastic::{
    GillespieSolver, StochasticError, StochasticSystem,
};

impl<S: Solver<SIRState>> StochasticSystem<SIRState> for SIRModel<S> {
    #[verified_engine::verified]
    fn propensities(&self, state: &SIRState, out: &mut Vec<f64>) {
        // Reaction 0: Infection (S + I -> 2I)
        // Rate: beta * S * I / N
        let infection_rate = self.beta() * state.s * state.i / self.n();

        // Reaction 1: Recovery (I -> R)
        // Rate: gamma * I
        let recovery_rate = self.gamma() * state.i;

        out.push(infection_rate);
        out.push(recovery_rate);
    }

    #[verified_engine::verified]
    fn react(&self, state: &mut SIRState, reaction_index: usize) -> Result<(), StochasticError> {
        match reaction_index {
            0 => {
                // Infection: S decreases by 1, I increases by 1
                state.s -= 1.0;
                state.i += 1.0;
                Ok(())
            }
            1 => {
                // Recovery: I decreases by 1, R increases by 1
                state.i -= 1.0;
                state.r += 1.0;
                Ok(())
            }
            _ => Err(StochasticError::InvalidReactionIndex(reaction_index)),
        }
    }
}

/// Calculates the probability of extinction given R0 and initial cases.
///
/// Based on Branching Process theory: $P_{ext} = (1/R_0)^{I_0}$ if $R_0 > 1$.
#[verified_engine::verified]
pub fn probability_of_extinction(r0: f64, initial_cases: f64) -> f64 {
    if r0 <= 1.0 {
        1.0
    } else {
        (1.0 / r0).powf(initial_cases)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EpidemiologyDomainError;

    #[test]
    #[verified_engine::verified]
    fn test_extinction_probability() {
        let r0 = 0.5;
        let i0 = 10.0;
        assert_eq!(probability_of_extinction(r0, i0), 1.0);

        let r0_high = 2.0;
        let i0_one = 1.0;
        // P = 1/2
        assert!(
            (probability_of_extinction(r0_high, i0_one) - 0.5).abs()
                < math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_gillespie_solver_deterministic() -> Result<(), EpidemiologyDomainError> {
        // Seeded RNG for deterministic test
        let rng = oxidize_core::rng::OxidizeRng::default();
        let mut solver = GillespieSolver::new(rng);

        let n = 100.0;
        let i0 = 10.0;
        // High beta, low gamma -> likely infection
        let model = SIRModel::new(n, i0, 2.0, 0.1)?;

        let mut state = *model.state(); // Working copy of state
        let initial_s = state.s;
        let initial_i = state.i;

        // Take one step
        let dt = solver.step(&model, &mut state)?;

        assert!(dt > 0.0, "Time step should be positive");
        assert!(dt.is_finite());

        // Check conservation of population
        assert_eq!(state.s + state.i + state.r, n);

        // State should have changed by exactly 1 individual (integer steps)
        let s_diff = (state.s - initial_s).abs();
        let i_diff = (state.i - initial_i).abs();

        assert!(s_diff <= 1.0);
        assert!(i_diff >= 1.0); // I changes in both infection and recovery

        let reaction_err: EpidemiologyDomainError = model.react(&mut state, 99).unwrap_err().into();
        diagnostics::DiagnosticBus::new().emit_diagnostic(&reaction_err);
        assert!(std::error::Error::source(&reaction_err).is_some());

        Ok(())
    }
}

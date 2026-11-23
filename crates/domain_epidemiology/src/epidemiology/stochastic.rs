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
        math_commons::math_kernel::soft_float::powf(1.0 / r0, initial_cases)
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
        const STEPS: usize = 25;
        const PINNED_DIGEST: &str =
            "7f63345f136b4408acb97177239bd33c1e57be8e3024fa4ae592c7c8cb490f0d";
        let n = 100.0;
        let i0 = 10.0;
        let model = SIRModel::new(n, i0, 2.0, 0.1)?;

        let run_trajectory =
            |seed: u64| -> Result<(Vec<[f64; 4]>, String), EpidemiologyDomainError> {
                let mut solver = GillespieSolver::new(oxidize_core::rng::OxidizeRng::new(seed));
                let mut state = *model.state();
                let mut trace = Vec::with_capacity(STEPS);
                let mut digest = oxidize_core::rng::TrajectoryDigest::new();
                for step_idx in 0..STEPS {
                    let prev_s = state.s;
                    let prev_i = state.i;
                    let dt = solver.step(&model, &mut state)?;
                    assert!(
                        dt > 0.0 && dt.is_finite(),
                        "step {step_idx} dt must be positive finite"
                    );
                    assert_eq!(state.s + state.i + state.r, n);
                    if step_idx == 0 {
                        assert!((state.s - prev_s).abs() <= 1.0);
                        assert!((state.i - prev_i).abs() >= 1.0);
                    }
                    let row = [dt, state.s, state.i, state.r];
                    digest.update_f64_slice(&row);
                    trace.push(row);
                }
                Ok((trace, digest.finish_hex()))
            };

        let (trace_a, digest_a) = run_trajectory(0xE91D_0042)?;
        let (trace_b, digest_b) = run_trajectory(0xE91D_0042)?;
        let (trace_other, digest_other) = run_trajectory(0xE91D_0043)?;
        assert_eq!(trace_a, trace_b);
        assert_eq!(digest_a, digest_b);
        assert_ne!(trace_a, trace_other);
        assert_ne!(digest_a, digest_other);
        assert_eq!(digest_a, PINNED_DIGEST);

        let mut err_state = *model.state();
        let reaction_err: EpidemiologyDomainError =
            model.react(&mut err_state, 99).unwrap_err().into();
        diagnostics::DiagnosticBus::new().emit_diagnostic(&reaction_err);
        assert!(std::error::Error::source(&reaction_err).is_some());

        Ok(())
    }
}

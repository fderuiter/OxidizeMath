//! Gray-Scott autocatalytic reaction-diffusion kinetics.

use super::reaction::{ReactionKinetics, add_reaction_batch_2d, reaction_2d_slice};
use crate::biology::reaction_diffusion::{ChemicalState, ReactionModel};
use verified_engine::Theory;

/// Gray-Scott reaction kinetics modeling cubic autocatalysis with substrate feed and catalyst kill.
///
/// ## Equations
///
/// $$ \frac{du}{dt} = -u v^2 + F(1 - u) $$
/// $$ \frac{dv}{dt} = u v^2 - (F + k) v $$
///
/// Where:
/// - $u$: Substrate concentration replenished toward $1.0$ at feed rate $F$.
/// - $v$: Autocatalyst concentration decaying at rate $F + k$.
/// - $F$: Dimensionless feed rate (`f`).
/// - $k$: Dimensionless kill/removal rate (`k`).
#[derive(Debug, Clone, Copy, PartialEq, Theory)]
#[theory(
    description = "Gray-Scott cubic autocatalytic reaction kinetics generating mitosis, coral, and soliton patterns.",
    citation = "Pearson, J. E. (1993). Complex patterns in a simple system. Science, 261(5118), 189-192."
)]
pub struct GrayScottKinetics {
    /// Dimensionless substrate feed rate ($F$).
    #[theory(min = 0.0, max = 0.12, step = 0.001)]
    pub f: f64,
    /// Dimensionless autocatalyst kill rate ($k$).
    #[theory(min = 0.0, max = 0.12, step = 0.001)]
    pub k: f64,
}

impl GrayScottKinetics {
    /// Creates a new Gray-Scott kinetics model with feed rate `f` and kill rate `k`.
    #[verified_engine::verified]
    pub fn new(f: f64, k: f64) -> Self {
        Self { f, k }
    }
}

impl Default for GrayScottKinetics {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            f: 0.0367,
            k: 0.0649,
        }
    }
}

impl ReactionKinetics<2> for GrayScottKinetics {
    #[verified_engine::verified]
    fn reaction(&self, concentrations: [f64; 2]) -> [f64; 2] {
        let u = concentrations[0];
        let v = concentrations[1];
        let uv_sq = u * v * v;
        let reaction_u = -uv_sq + self.f * (1.0 - u);
        let reaction_v = uv_sq - (self.f + self.k) * v;
        [reaction_u, reaction_v]
    }
}

impl ReactionModel for GrayScottKinetics {
    #[verified_engine::verified]
    fn reaction(&self, concentrations: &[f64], rates: &mut [f64]) {
        reaction_2d_slice(self, concentrations, rates);
    }

    #[verified_engine::verified]
    fn add_reaction_batch(&self, state: &ChemicalState, rates: &mut ChemicalState) {
        add_reaction_batch_2d(self, state, rates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::scientific_metadata::theory::TheoryDescribable;

    #[test]
    #[verified_engine::verified]
    fn test_gray_scott_equilibrium_and_autocatalysis() {
        let gs = GrayScottKinetics::new(0.055, 0.062);
        let eq = <GrayScottKinetics as ReactionKinetics<2>>::reaction(&gs, [1.0, 0.0]);
        assert!(eq[0].abs() < 1e-15);
        assert!(eq[1].abs() < 1e-15);

        let rates = <GrayScottKinetics as ReactionKinetics<2>>::reaction(&gs, [0.5, 0.25]);
        let expected_u = -0.5 * 0.0625 + 0.055 * 0.5;
        let expected_v = 0.5 * 0.0625 - (0.055 + 0.062) * 0.25;
        assert!((rates[0] - expected_u).abs() < 1e-15);
        assert!((rates[1] - expected_v).abs() < 1e-15);
    }

    #[test]
    #[verified_engine::verified]
    fn test_gray_scott_reaction_model_and_theory() {
        let gs = GrayScottKinetics::default();
        assert!(gs.theory_citation().contains("Pearson"));
        assert!(gs.theory_description().contains("Gray-Scott"));

        let mut single = [0.0; 2];
        ReactionModel::reaction(&gs, &[0.5, 0.25], &mut single);
        let direct = <GrayScottKinetics as ReactionKinetics<2>>::reaction(&gs, [0.5, 0.25]);
        assert_eq!(single, direct);

        let mut state = ChemicalState::new(2, 2);
        state.species_mut(0).copy_from_slice(&[1.0, 0.5]);
        state.species_mut(1).copy_from_slice(&[0.0, 0.25]);
        let mut out = ChemicalState::new(2, 2);
        gs.add_reaction_batch(&state, &mut out);
        assert!(out.species(0)[0].abs() < 1e-15);
        assert!(out.species(1)[0].abs() < 1e-15);
        assert!((out.species(0)[1] - direct[0]).abs() < 1e-15);
        assert!((out.species(1)[1] - direct[1]).abs() < 1e-15);
    }
}

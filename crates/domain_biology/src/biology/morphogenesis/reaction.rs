use crate::biology::reaction_diffusion::ReactionModel;
use verified_engine::Theory;

/// Defines the reaction kinetics for an N-component reaction-diffusion system.
pub trait ReactionKinetics<const N: usize = 2> {
    /// Calculates the reaction rates for the given species concentrations.
    ///
    /// # Arguments
    /// * `concentrations` - Array of concentrations for each species.
    ///
    /// # Returns
    /// An array `[dC_1/dt, ..., dC_N/dt]` representing the reaction terms.
    #[verified_engine::verified]
    fn reaction(&self, concentrations: [f64; N]) -> [f64; N];
}

/// Schnakenberg kinetics (often used for Turing patterns).
///
/// This model is famous for generating spot-like patterns (like leopard spots).
///
/// ## Equations
///
/// $$ \frac{du}{dt} = a - u + u^2 v $$
/// $$ \frac{dv}{dt} = b - u^2 v $$
///
/// Where:
/// - $a$: Production rate of the activator.
/// - $b$: Production rate of the inhibitor.
/// - $u^2 v$: Non-linear autocatalysis term (Activator requires Inhibitor to grow, but consumes it).
#[derive(Debug, Clone, Copy, Theory)]
#[theory(
    description = "Schnakenberg kinetics (often used for Turing patterns).",
    citation = "Schnakenberg, J. (1979). Simple chemical reaction systems with limit cycle behaviour. Journal of theoretical biology, 81(3), 389-400."
)]
pub struct SchnakenbergKinetics {
    /// Production rate of activator ($a$).
    #[theory(min = 0.0, max = 1.0, step = 0.01)]
    pub a: f64,
    /// Production rate of inhibitor ($b$).
    #[theory(min = 0.0, max = 2.0, step = 0.01)]
    pub b: f64,
}

impl SchnakenbergKinetics {
    /// Creates a new Schnakenberg kinetics model.
    #[verified_engine::verified]
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }
}

impl Default for SchnakenbergKinetics {
    #[verified_engine::verified]
    fn default() -> Self {
        Self { a: 0.01, b: 0.05 }
    }
}

impl ReactionKinetics<2> for SchnakenbergKinetics {
    #[verified_engine::verified]
    fn reaction(&self, concentrations: [f64; 2]) -> [f64; 2] {
        let u = concentrations[0];
        let v = concentrations[1];
        let uv_sq = u * u * v;
        let reaction_u = self.a - u + uv_sq;
        let reaction_v = self.b - uv_sq;
        [reaction_u, reaction_v]
    }
}

pub(crate) use crate::biology::reaction_diffusion::ChemicalState;

#[verified_engine::verified]
pub(crate) fn reaction_2d_slice<K: ReactionKinetics<2>>(
    kinetics: &K,
    concentrations: &[f64],
    rates: &mut [f64],
) {
    if concentrations.len() < 2 || rates.len() < 2 {
        return;
    }
    let out = kinetics.reaction([concentrations[0], concentrations[1]]);
    rates[0] = out[0];
    rates[1] = out[1];
}

#[verified_engine::verified]
pub(crate) fn add_reaction_batch_2d<K: ReactionKinetics<2>>(
    kinetics: &K,
    state: &ChemicalState,
    rates: &mut ChemicalState,
) {
    if state.num_species() < 2 {
        return;
    }
    let u_vec = state.species(0);
    let v_vec = state.species(1);
    let n = state.grid_size();
    let (rates_u, rates_v) = rates.grid.as_mut_slice().split_at_mut(n);
    for i in 0..n {
        let out = kinetics.reaction([u_vec[i], v_vec[i]]);
        rates_u[i] += out[0];
        rates_v[i] += out[1];
    }
}

impl ReactionModel for SchnakenbergKinetics {
    #[verified_engine::verified]
    fn reaction(&self, concentrations: &[f64], rates: &mut [f64]) {
        reaction_2d_slice(self, concentrations, rates);
    }

    #[verified_engine::verified]
    fn add_reaction_batch(&self, state: &ChemicalState, rates: &mut ChemicalState) {
        add_reaction_batch_2d(self, state, rates);
    }
}

/// Gray-Scott autocatalytic reaction kinetics ($U + 2V \to 3V$).
///
/// Generates self-replicating spot mitosis, labyrinthine coral, and soliton wave patterns.
///
/// ## Equations
///
/// $$ \frac{du}{dt} = -u v^2 + f(1 - u) $$
/// $$ \frac{dv}{dt} = u v^2 - (f + k)v $$
///
/// Where:
/// - $f$: Dimensionless feed rate replenishing substrate $u$.
/// - $k$: Dimensionless kill rate removing autocatalyst $v$.
#[derive(Debug, Clone, Copy, PartialEq, Theory)]
#[theory(
    description = "Gray-Scott autocatalytic reaction kinetics generating spot mitosis and coral morphologies.",
    citation = "Pearson, J. E. (1993). Complex patterns in a simple system. Science, 261(5118), 189-192."
)]
pub struct GrayScottKinetics {
    /// Feed rate ($f$) replenishing substrate $u$.
    #[theory(min = 0.0, max = 0.1, step = 0.001)]
    pub f: f64,
    /// Kill rate ($k$) draining autocatalyst $v$.
    #[theory(min = 0.0, max = 0.1, step = 0.001)]
    pub k: f64,
}

impl GrayScottKinetics {
    /// Creates a new Gray-Scott kinetics model with feed rate `f` and kill rate `k`.
    #[verified_engine::verified]
    pub fn new(f: f64, k: f64) -> Self {
        Self { f, k }
    }

    /// Canonical Pearson (1993) spot-division (`Mitosis`) regime (`f = 0.0367`, `k = 0.0649`).
    #[verified_engine::verified]
    pub fn mitosis() -> Self {
        Self {
            f: 0.0367,
            k: 0.0649,
        }
    }

    /// Canonical Pearson (1993) dendritic branching (`Coral`) regime (`f = 0.0545`, `k = 0.0620`).
    #[verified_engine::verified]
    pub fn coral() -> Self {
        Self {
            f: 0.0545,
            k: 0.0620,
        }
    }
}

impl Default for GrayScottKinetics {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::mitosis()
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
        assert_eq!(gs, GrayScottKinetics::mitosis());
        assert_ne!(gs, GrayScottKinetics::coral());
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

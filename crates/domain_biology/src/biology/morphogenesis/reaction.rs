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

    /// Evaluates reaction rates at linear grid cell `index`, defaulting to [`Self::reaction`].
    #[inline]
    #[verified_engine::verified]
    fn reaction_at(&self, _index: usize, concentrations: [f64; N]) -> [f64; N] {
        self.reaction(concentrations)
    }
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
///
/// Reference: Schnakenberg (1979), DOI: `10.1016/0022-5193(79)90002-4`; see
/// [`docs/math_specs/domain_biology.md`](../../../../../docs/math_specs/domain_biology.md).
///
/// # Stability
/// Exhibits a unique homogeneous steady state $(u^*, v^*) = (a + b, b / (a + b)^2)$; cubic $u^2 v$
/// stiffness requires bounded $\Delta t$ so explicit updates do not overshoot positive orthant bounds.
///
/// # Domain of Validity
/// Non-negative concentrations $u, v \ge 0$ and strictly positive feed rates $a \ge 0$, $b > 0$
/// with $a + b > 0$.
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
    #[inline(always)]
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

#[inline]
#[verified_engine::verified]
fn reaction_2d_slice<K: ReactionKinetics<2>>(
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
fn add_reaction_batch_2d<K: ReactionKinetics<2>>(
    kinetics: &K,
    state: &crate::biology::reaction_diffusion::ChemicalState,
    rates: &mut crate::biology::reaction_diffusion::ChemicalState,
) {
    if state.num_species() < 2 || rates.num_species() < 2 {
        return;
    }
    let u_vec = state.species(0);
    let v_vec = state.species(1);
    let rates_stride = rates.grid_size();
    let n = state.grid_size().min(rates_stride);
    let (rates_u, rates_v) = rates.grid.as_mut_slice().split_at_mut(rates_stride);
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
    fn add_reaction_batch(
        &self,
        state: &crate::biology::reaction_diffusion::ChemicalState,
        rates: &mut crate::biology::reaction_diffusion::ChemicalState,
    ) {
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
///
/// Reference: Pearson (1993), DOI: `10.1126/science.261.5118.189`; see
/// [`docs/math_specs/domain_biology.md`](../../../../../docs/math_specs/domain_biology.md).
///
/// # Stability
/// Preserves $u, v \in [0, 1]$ analytically; explicit time-stepping requires $\Delta t \lesssim 1$
/// alongside the diffusive CFL constraint $\Delta t \le h^2 / (2 d \max(D_u, D_v))$.
///
/// # Domain of Validity
/// Dimensionless concentrations $u, v \in [0, 1]$ and feed/kill rates $f, k \in [0, 0.1]$.
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
    #[inline(always)]
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
    fn add_reaction_batch(
        &self,
        state: &crate::biology::reaction_diffusion::ChemicalState,
        rates: &mut crate::biology::reaction_diffusion::ChemicalState,
    ) {
        add_reaction_batch_2d(self, state, rates);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biology::reaction_diffusion::ChemicalState;
    use scientific_metadata::theory::TheoryDescribable;

    #[test]
    #[verified_engine::verified]
    fn test_gray_scott_trivial_equilibrium_and_autocatalysis() {
        let mitosis = GrayScottKinetics::mitosis();
        assert_eq!(GrayScottKinetics::default(), mitosis);
        let eq = ReactionKinetics::reaction(&mitosis, [1.0, 0.0]);
        assert!(eq[0].abs() < 1e-15 && eq[1].abs() < 1e-15);

        let coral = GrayScottKinetics::coral();
        let active = ReactionKinetics::reaction(&coral, [0.5, 0.25]);
        assert!((active[0] - (-0.5 * 0.0625 + coral.f * 0.5)).abs() < 1e-15);
        assert!((active[1] - (0.5 * 0.0625 - (coral.f + coral.k) * 0.25)).abs() < 1e-15);
        assert!(coral.theory_citation().contains("Pearson"));
    }

    #[test]
    #[verified_engine::verified]
    fn test_reaction_model_batch_matches_pointwise_rates() {
        let gs = GrayScottKinetics::new(0.030, 0.060);
        let mut state = ChemicalState::new(2, 3);
        state.species_mut(0).copy_from_slice(&[1.0, 0.5, 0.75]);
        state.species_mut(1).copy_from_slice(&[0.0, 0.25, 0.2]);
        let mut rates = ChemicalState::new(2, 3);
        gs.add_reaction_batch(&state, &mut rates);
        for i in 0..3 {
            let mut single = [0.0; 2];
            <GrayScottKinetics as ReactionModel>::reaction(
                &gs,
                &[state.species(0)[i], state.species(1)[i]],
                &mut single,
            );
            assert!((rates.species(0)[i] - single[0]).abs() < 1e-15);
            assert!((rates.species(1)[i] - single[1]).abs() < 1e-15);
        }
    }
}

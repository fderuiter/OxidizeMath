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
#[derive(Debug, Clone, Copy, PartialEq, Theory)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

impl ReactionModel for SchnakenbergKinetics {
    #[verified_engine::verified]
    fn reaction(&self, concentrations: &[f64], rates: &mut [f64]) {
        if concentrations.len() < 2 || rates.len() < 2 {
            return;
        }
        // We can safely assume Schnakenberg is 2D
        let inp = [concentrations[0], concentrations[1]];
        let out = <Self as ReactionKinetics<2>>::reaction(self, inp);
        rates[0] = out[0];
        rates[1] = out[1];
    }

    #[verified_engine::verified]
    fn add_reaction_batch(
        &self,
        state: &crate::biology::reaction_diffusion::ChemicalState,
        rates: &mut crate::biology::reaction_diffusion::ChemicalState,
    ) {
        if state.num_species() < 2 {
            return;
        }

        let u_vec = state.species(0);
        let v_vec = state.species(1);

        let n = state.grid_size();

        // Split mutable borrow to access both rate vectors simultaneously
        let (rates_u, rates_v) = rates.grid.as_mut_slice().split_at_mut(n);

        // Vectorized loop: Access memory linearly, enabling prefetch and SIMD
        for i in 0..n {
            let inp = [u_vec[i], v_vec[i]];
            let out = <Self as ReactionKinetics<2>>::reaction(self, inp);
            rates_u[i] += out[0];
            rates_v[i] += out[1];
        }
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        if concentrations.len() < 2 || rates.len() < 2 {
            return;
        }
        let inp = [concentrations[0], concentrations[1]];
        let out = <Self as ReactionKinetics<2>>::reaction(self, inp);
        rates[0] = out[0];
        rates[1] = out[1];
    }

    #[verified_engine::verified]
    fn add_reaction_batch(
        &self,
        state: &crate::biology::reaction_diffusion::ChemicalState,
        rates: &mut crate::biology::reaction_diffusion::ChemicalState,
    ) {
        if state.num_species() < 2 {
            return;
        }
        let u_vec = state.species(0);
        let v_vec = state.species(1);
        let n = state.grid_size();
        let (rates_u, rates_v) = rates.grid.as_mut_slice().split_at_mut(n);
        for i in 0..n {
            let inp = [u_vec[i], v_vec[i]];
            let out = <Self as ReactionKinetics<2>>::reaction(self, inp);
            rates_u[i] += out[0];
            rates_v[i] += out[1];
        }
    }
}

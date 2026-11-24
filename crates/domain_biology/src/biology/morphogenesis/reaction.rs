use crate::biology::reaction_diffusion::ReactionModel;
use verified_engine::Theory;

/// Defines local reaction kinetics $\mathbf{R}(\mathbf{c}) = [R_1(\mathbf{c}), \dots, R_N(\mathbf{c})]$ for an $N$-component reaction-diffusion system (`@turing1952chemical`, `spec:biology_v1`).
///
/// - **Numerical stability**: Pointwise polynomial evaluation must avoid division by zero and remain Lipschitz-bounded on compact concentration intervals.
/// - **Domain of validity**: Valid for finite non-negative species concentrations $c_i \ge 0$ ($i = 1, \dots, N$).
pub trait ReactionKinetics<const N: usize = 2> {
    /// Evaluates instantaneous reaction rates $\dot{c}_i = R_i(c_1, \dots, c_N)$ (`@turing1952chemical`, `spec:biology_v1`).
    ///
    /// - **Numerical stability**: Accumulates at most $O(N \varepsilon_{\text{mach}})$ rounding error for low-degree mass-action polynomials.
    /// - **Domain of validity**: Requires finite concentrations $c_i \ge 0$ within the physical state box of the chosen kinetic law.
    ///
    /// # Arguments
    /// * `concentrations` - Array of concentrations for each species.
    ///
    /// # Returns
    /// An array `[dC_1/dt, ..., dC_N/dt]` representing the reaction terms.
    #[verified_engine::verified]
    fn reaction(&self, concentrations: [f64; N]) -> [f64; N];

    /// Evaluates spatially indexed reaction rates $\dot{c}_i(x_k) = R_i(x_k, c_1, \dots, c_N)$ at linear grid cell `index`, defaulting to [`Self::reaction`] (`@turing1952chemical`, `spec:biology_v1`).
    ///
    /// - **Numerical stability**: Identical $O(N \varepsilon_{\text{mach}})$ floating-point rounding bound to [`Self::reaction`] unless overridden with spatially varying parameters.
    /// - **Domain of validity**: Valid for finite species concentrations $c_i \ge 0$ and linear cell index $0 \le \text{index} < N_{\text{grid}}$.
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
/// - **Citation**: `@schnakenberg1979simple` (`spec:biology_v1`).
/// - **Numerical stability**: Cubic nonlinearity $u^2 v$ restricts explicit time steps to $\Delta t < \min(1, (2 u_{\max} v_{\max} + 1)^{-1})$.
/// - **Domain of validity**: Non-negative feed rates $a \ge 0, b > 0$ with positive steady state $(u^*, v^*) = (a + b, b / (a + b)^2)$.
#[derive(Debug, Clone, Copy, Theory)]
#[theory(
    description = "Schnakenberg kinetics (often used for Turing patterns).",
    citation = "schnakenberg1979simple"
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
    /// Creates a Schnakenberg kinetics model with rates $(a, b)$ governing $\dot{u} = a - u + u^2 v, \dot{v} = b - u^2 v$ (`@schnakenberg1979simple`, `spec:biology_v1`).
    /// Numerical stability: exact `f64` construction; explicit solvers require $\Delta t$ bounded by local Jacobian spectral radius. Domain of validity: finite $a \in [0, 1], b \in (0, 2]$.
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
/// - **Citation**: `@pearson1993complex` (`spec:biology_v1`).
/// - **Numerical stability**: Explicit integration remains stable for $\Delta t \le 1.0$ when $u, v \in [0, 1]$ and $f, k \in [0, 0.1]$.
/// - **Domain of validity**: Dimensionless concentrations $u, v \in [0, 1]$ and feed/kill rates $f, k \in [0, 0.1]$.
#[derive(Debug, Clone, Copy, PartialEq, Theory)]
#[theory(
    description = "Gray-Scott autocatalytic reaction kinetics generating spot mitosis and coral morphologies.",
    citation = "pearson1993complex"
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
    /// Creates a Gray-Scott model for $\dot{u} = -u v^2 + f(1-u), \dot{v} = u v^2 - (f+k)v$ (`@pearson1993complex`, `spec:biology_v1`).
    /// Numerical stability: exact `f64` parameter capture; stable under RK4/Euler for $\Delta t \le 1$. Domain of validity: finite $f, k \in [0, 0.1]$.
    #[verified_engine::verified]
    pub fn new(f: f64, k: f64) -> Self {
        Self { f, k }
    }

    /// Canonical Pearson spot-division (`Mitosis`) regime $(f, k) = (0.0367, 0.0649)$ (`@pearson1993complex`, `spec:biology_v1`).
    /// Numerical stability: well-conditioned away from saddle-node fold for $\Delta t \le 1.0$. Domain of validity: concentrations $(u, v) \in [0, 1]^2$ with $D_u / D_v = 2$.
    #[verified_engine::verified]
    pub fn mitosis() -> Self {
        Self {
            f: 0.0367,
            k: 0.0649,
        }
    }

    /// Canonical Pearson dendritic branching (`Coral`) regime $(f, k) = (0.0545, 0.0620)$ (`@pearson1993complex`, `spec:biology_v1`).
    /// Numerical stability: bounded cubic reaction rates on $[0, 1]^2$ for $\Delta t \le 1.0$. Domain of validity: concentrations $(u, v) \in [0, 1]^2$ with $D_u / D_v = 2$.
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

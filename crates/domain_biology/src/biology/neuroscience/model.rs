//! Mathematical model of the Hodgkin-Huxley equations.

use super::types::{HodgkinHuxleyParameters, HodgkinHuxleyState};
use pure_math::pure_math::analysis::ode::OdeSystem;
use verified_engine::Theory;

/// The Hodgkin-Huxley system of differential equations.
///
/// This struct implements the `OdeSystem` trait, defining the time derivatives
/// for the membrane potential $V$ and the gating variables $n, m, h$.
///
/// # Equations
///
/// The system is governed by the following set of coupled ordinary differential equations:
///
/// 1.  **Membrane Potential ($V$):**
///     $$ C_m \frac{dV}{dt} = I_{ext} - (I_{Na} + I_{K} + I_{L}) $$
///     Where:
///     *   $I_{Na} = g_{Na} m^3 h (V - E_{Na})$ (Sodium current)
///     *   $I_{K} = g_{K} n^4 (V - E_{K})$ (Potassium current)
///     *   $I_{L} = g_{L} (V - E_{L})$ (Leak current)
///
/// 2.  **Gating Variables ($x \in \{n, m, h\}$):**
///     $$ \frac{dx}{dt} = \alpha_x(V)(1 - x) - \beta_x(V)x $$
///     The rate constants $\alpha_x$ and $\beta_x$ are determined by the [`GatingKinetics`](crate::biology::neuroscience::kinetics::GatingKinetics) strategy.
#[derive(Debug, Clone, Theory)]
#[theory(
    description = "The Hodgkin-Huxley model describes the initiation and propagation of action potentials in neurons using a set of non-linear differential equations.",
    citation = "A quantitative description of membrane current and its application to conduction and excitation in nerve (Hodgkin & Huxley, 1952)"
)]
pub struct HodgkinHuxleyModel {
    /// Parameters of the model (conductances, potentials, kinetics).
    pub params: HodgkinHuxleyParameters,
    /// External current injection ($I_{ext}$) in $\mu A/cm^2$.
    /// This acts as the driving force for the neuron (e.g., synaptic input or electrode).
    pub i_ext: f64,
}

impl HodgkinHuxleyModel {
    /// Creates a new Hodgkin-Huxley model instance.
    ///
    /// # Arguments
    /// * `params` - The biophysical parameters (conductances, etc.).
    /// * `i_ext` - The external current being applied at this moment.
    #[verified_engine::verified]
    pub fn new(params: HodgkinHuxleyParameters, i_ext: f64) -> Self {
        Self { params, i_ext }
    }
}

impl OdeSystem<HodgkinHuxleyState> for HodgkinHuxleyModel {
    /// Computes the time derivative of the state vector.
    ///
    /// # Returns
    /// A `HodgkinHuxleyState` representing $[\frac{dV}{dt}, \frac{dn}{dt}, \frac{dm}{dt}, \frac{dh}{dt}]$.
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &HodgkinHuxleyState) -> HodgkinHuxleyState {
        // Unpack parameters
        let p = &self.params;
        let k = &p.kinetics;

        let v = state.v;
        let n = state.n;
        let m = state.m;
        let h = state.h;

        // Calculate ionic currents
        // Sodium Current: I_Na = g_Na * m^3 * h * (V - E_Na)
        let i_na = p.g_na * m.powi(3) * h * (v - p.e_na);

        // Potassium Current: I_K = g_K * n^4 * (V - E_K)
        let i_k = p.g_k * n.powi(4) * (v - p.e_k);

        // Leak Current: I_L = g_L * (V - E_L)
        let i_l = p.g_l * (v - p.e_l);

        // Membrane Potential Derivative (assuming unit capacitance Cm = 1.0 uF/cm^2)
        // dV/dt = I_ext - (I_Na + I_K + I_L)
        let dv_dt = self.i_ext - i_na - i_k - i_l;

        // Gating variable derivatives
        // dx/dt = alpha_x(V) * (1 - x) - beta_x(V) * x
        let dn_dt = k.alpha_n(v, p.v_rest) * (1.0 - n) - k.beta_n(v, p.v_rest) * n;
        let dm_dt = k.alpha_m(v, p.v_rest) * (1.0 - m) - k.beta_m(v, p.v_rest) * m;
        let dh_dt = k.alpha_h(v, p.v_rest) * (1.0 - h) - k.beta_h(v, p.v_rest) * h;

        HodgkinHuxleyState {
            v: dv_dt,
            n: dn_dt,
            m: dm_dt,
            h: dh_dt,
        }
    }
}

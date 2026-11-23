//! Kinetics strategy for ion channel gating variables.
//!
//! This module defines the `GatingKinetics` trait and its standard implementation `StandardKinetics`,
//! which governs the opening and closing rates of ion channels in the Hodgkin-Huxley model.

/// Defines the voltage-dependent rate constants for gating variables.
///
/// Implement this trait to define custom ion channel dynamics (e.g., for different neuron types).
/// Each method returns a rate constant ($\alpha$ or $\beta$) which determines the time derivative
/// of the gating variable $x$ according to:
/// $$ \frac{dx}{dt} = \alpha_x(V) (1 - x) - \beta_x(V) x $$
pub trait GatingKinetics: Send + Sync + std::fmt::Debug {
    /// Rate constant $\alpha_n$ for Potassium activation ($n$).
    ///
    /// Represents the rate at which closed Potassium channels open (activate).
    #[verified_engine::verified]
    fn alpha_n(&self, v: f64, v_rest: f64) -> f64;

    /// Rate constant $\beta_n$ for Potassium activation ($n$).
    ///
    /// Represents the rate at which open Potassium channels close (deactivate).
    #[verified_engine::verified]
    fn beta_n(&self, v: f64, v_rest: f64) -> f64;

    /// Rate constant $\alpha_m$ for Sodium activation ($m$).
    ///
    /// Represents the rate at which closed Sodium channels open (activate).
    #[verified_engine::verified]
    fn alpha_m(&self, v: f64, v_rest: f64) -> f64;

    /// Rate constant $\beta_m$ for Sodium activation ($m$).
    ///
    /// Represents the rate at which open Sodium channels close (deactivate).
    #[verified_engine::verified]
    fn beta_m(&self, v: f64, v_rest: f64) -> f64;

    /// Rate constant $\alpha_h$ for Sodium inactivation ($h$).
    ///
    /// Represents the rate at which inactivated Sodium channels recover (become closed but active).
    /// Note that for inactivation, "alpha" typically refers to the recovery from inactivation.
    #[verified_engine::verified]
    fn alpha_h(&self, v: f64, v_rest: f64) -> f64;

    /// Rate constant $\beta_h$ for Sodium inactivation ($h$).
    ///
    /// Represents the rate at which open Sodium channels inactivate.
    #[verified_engine::verified]
    fn beta_h(&self, v: f64, v_rest: f64) -> f64;
}

/// Standard Hodgkin-Huxley kinetics (Squid Giant Axon).
///
/// This implementation uses the original equations from Hodgkin & Huxley (1952).
/// It handles numerical singularities where the denominator approaches zero using
/// L'Hôpital's rule approximations.
///
/// # Example
///
/// ```rust
/// # use domain_biology::__doctest_facade as math_explorer;
/// use math_explorer::biology::neuroscience::kinetics::{GatingKinetics, StandardKinetics};
///
/// let kinetics = StandardKinetics::default();
/// let v_rest = -65.0;
/// let v_curr = -55.0; // Depolarized by 10mV
///
/// // Calculate opening rate for Potassium channels
/// let alpha_n = kinetics.alpha_n(v_curr, v_rest);
/// println!("Alpha n: {:.4}", alpha_n);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StandardKinetics;

impl GatingKinetics for StandardKinetics {
    /// Potassium activation rate $\alpha_n$.
    ///
    /// $$ \alpha_n = \frac{0.01 (10 - (V - V_{rest}))}{\exp(0.1(10 - (V - V_{rest}))) - 1} $$
    ///
    /// # Numerical Stability
    /// When $V = V_{rest} + 10$, the denominator approaches zero ($e^0 - 1 = 0$).
    /// We use L'Hôpital's rule to evaluate the limit:
    /// $$ \lim_{x \to 0} \frac{0.01 x}{e^{0.1 x} - 1} = \frac{0.01}{0.1} = 0.1 $$
    #[verified_engine::verified]
    fn alpha_n(&self, v: f64, v_rest: f64) -> f64 {
        let x = 10.0 - (v - v_rest);
        if x.abs() < math_commons::registry::TOLERANCE_STANDARD {
            0.1 // Limit as x -> 0
        } else {
            0.01 * x / ((0.1 * x).exp() - 1.0)
        }
    }

    /// Potassium deactivation rate $\beta_n$.
    ///
    /// $$ \beta_n = 0.125 \exp\left(-\frac{V - V_{rest}}{80}\right) $$
    #[verified_engine::verified]
    fn beta_n(&self, v: f64, v_rest: f64) -> f64 {
        let dv = v - v_rest;
        0.125 * (-dv / 80.0).exp()
    }

    /// Sodium activation rate $\alpha_m$.
    ///
    /// $$ \alpha_m = \frac{0.1 (25 - (V - V_{rest}))}{\exp(0.1(25 - (V - V_{rest}))) - 1} $$
    ///
    /// # Numerical Stability
    /// When $V = V_{rest} + 25$, the denominator approaches zero.
    /// We use L'Hôpital's rule to evaluate the limit:
    /// $$ \lim_{x \to 0} \frac{0.1 x}{e^{0.1 x} - 1} = \frac{0.1}{0.1} = 1.0 $$
    #[verified_engine::verified]
    fn alpha_m(&self, v: f64, v_rest: f64) -> f64 {
        let dv = v - v_rest;
        let x = 25.0 - dv;
        if x.abs() < math_commons::registry::TOLERANCE_STANDARD {
            1.0
        } else {
            0.1 * x / ((0.1 * x).exp() - 1.0)
        }
    }

    /// Sodium deactivation rate $\beta_m$.
    ///
    /// $$ \beta_m = 4.0 \exp\left(-\frac{V - V_{rest}}{18}\right) $$
    #[verified_engine::verified]
    fn beta_m(&self, v: f64, v_rest: f64) -> f64 {
        let dv = v - v_rest;
        4.0 * (-dv / 18.0).exp()
    }

    /// Sodium recovery from inactivation rate $\alpha_h$.
    ///
    /// $$ \alpha_h = 0.07 \exp\left(-\frac{V - V_{rest}}{20}\right) $$
    #[verified_engine::verified]
    fn alpha_h(&self, v: f64, v_rest: f64) -> f64 {
        let dv = v - v_rest;
        0.07 * (-dv / 20.0).exp()
    }

    /// Sodium inactivation rate $\beta_h$.
    ///
    /// $$ \beta_h = \frac{1}{\exp(3.0 - 0.1(V - V_{rest})) + 1} $$
    #[verified_engine::verified]
    fn beta_h(&self, v: f64, v_rest: f64) -> f64 {
        let dv = v - v_rest;
        1.0 / ((3.0 - 0.1 * dv).exp() + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_alpha_n_singularity() {
        let kinetics = StandardKinetics;
        let v_rest = -65.0;
        // Singularity point: v = v_rest + 10.0
        let v_singular = v_rest + 10.0;
        let result = kinetics.alpha_n(v_singular, v_rest);

        // Expected limit: 0.1
        assert!(
            (result - 0.1).abs() < math_commons::registry::TOLERANCE_STANDARD,
            "Expected 0.1, got {}",
            result
        );
        assert!(result.is_finite(), "Result should be finite");
    }

    #[test]
    #[verified_engine::verified]
    fn test_alpha_m_singularity() {
        let kinetics = StandardKinetics;
        let v_rest = -65.0;
        // Singularity point: v = v_rest + 25.0
        let v_singular = v_rest + 25.0;
        let result = kinetics.alpha_m(v_singular, v_rest);

        // Expected limit: 1.0
        assert!(
            (result - 1.0).abs() < math_commons::registry::TOLERANCE_STANDARD,
            "Expected 1.0, got {}",
            result
        );
        assert!(result.is_finite(), "Result should be finite");
    }
}

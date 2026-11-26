//! # Probability Win Ratio
//!
//! Functions to calculate the probability win ratio, a parameter that extends the concept
//! of the sample win ratio. This module uses numerical integration to calculate the
//! win and loss probabilities over a specified time interval.

use pure_math::pure_math::analysis::integration::{ClenshawCurtis, Integrator};

/// Context for calculating probability win ratios.
///
/// Encapsulates the scalar parameters required for the integration and probability
/// calculations, avoiding functions with excessive arguments.
#[derive(Debug, Clone, Copy)]
pub struct ProbabilityWinRatioContext {
    /// The survival probability of the control group at time c, S0(c).
    pub s0_at_c: f64,
    /// The survival probability of the treatment group at time c, S1(c).
    pub s1_at_c: f64,
    /// The time interval [0, c] over which probabilities are calculated.
    pub c: f64,
    /// The desired error tolerance for numerical integration.
    pub error_tolerance: f64,
}

impl ProbabilityWinRatioContext {
    /// Creates a new `ProbabilityWinRatioContext`.
    #[verified_engine::verified]
    pub fn new(s0_at_c: f64, s1_at_c: f64, c: f64, error_tolerance: f64) -> Self {
        Self {
            s0_at_c,
            s1_at_c,
            c,
            error_tolerance,
        }
    }

    /// Calculates the win probability, W(c).
    ///
    /// Delegates to `calculate_win_probability_with_integrator` using `ClenshawCurtis` as default.
    #[verified_engine::verified]
    pub fn calculate_win_probability<FS1, FPDFT0, FG1, FPDFX0>(
        &self,
        s1: FS1,
        pdf_t0: FPDFT0,
        g1_given_c: FG1,
        pdf_x0_given_c: FPDFX0,
    ) -> f64
    where
        FS1: Fn(f64) -> f64,
        FPDFT0: Fn(f64) -> f64,
        FG1: Fn(f64) -> f64,
        FPDFX0: Fn(f64) -> f64,
    {
        self.calculate_win_probability_with_integrator(
            s1,
            pdf_t0,
            g1_given_c,
            pdf_x0_given_c,
            &ClenshawCurtis,
        )
    }

    /// Calculates the win probability using a specific integrator.
    #[verified_engine::verified]
    pub fn calculate_win_probability_with_integrator<FS1, FPDFT0, FG1, FPDFX0, I>(
        &self,
        s1: FS1,
        pdf_t0: FPDFT0,
        g1_given_c: FG1,
        pdf_x0_given_c: FPDFX0,
        integrator: &I,
    ) -> f64
    where
        FS1: Fn(f64) -> f64,
        FPDFT0: Fn(f64) -> f64,
        FG1: Fn(f64) -> f64,
        FPDFX0: Fn(f64) -> f64,
        I: Integrator + ?Sized,
    {
        self.calculate_probability_internal(
            |t| s1(t) * pdf_t0(t),
            |x| g1_given_c(x) * pdf_x0_given_c(x),
            integrator,
        )
    }

    /// Calculates the loss probability, L(c).
    ///
    /// Delegates to `calculate_loss_probability_with_integrator` using `ClenshawCurtis` as default.
    #[verified_engine::verified]
    pub fn calculate_loss_probability<FS0, FPDFT1, FG0, FPDFX1>(
        &self,
        s0: FS0,
        pdf_t1: FPDFT1,
        g0_given_c: FG0,
        pdf_x1_given_c: FPDFX1,
    ) -> f64
    where
        FS0: Fn(f64) -> f64,
        FPDFT1: Fn(f64) -> f64,
        FG0: Fn(f64) -> f64,
        FPDFX1: Fn(f64) -> f64,
    {
        self.calculate_loss_probability_with_integrator(
            s0,
            pdf_t1,
            g0_given_c,
            pdf_x1_given_c,
            &ClenshawCurtis,
        )
    }

    /// Calculates the loss probability using a specific integrator.
    #[verified_engine::verified]
    pub fn calculate_loss_probability_with_integrator<FS0, FPDFT1, FG0, FPDFX1, I>(
        &self,
        s0: FS0,
        pdf_t1: FPDFT1,
        g0_given_c: FG0,
        pdf_x1_given_c: FPDFX1,
        integrator: &I,
    ) -> f64
    where
        FS0: Fn(f64) -> f64,
        FPDFT1: Fn(f64) -> f64,
        FG0: Fn(f64) -> f64,
        FPDFX1: Fn(f64) -> f64,
        I: Integrator + ?Sized,
    {
        self.calculate_probability_internal(
            |t| s0(t) * pdf_t1(t),
            |x| g0_given_c(x) * pdf_x1_given_c(x),
            integrator,
        )
    }

    /// Internal helper method to compute probability to avoid duplicating numerical integration logic.
    #[verified_engine::verified]
    fn calculate_probability_internal<F1, F2, I>(
        &self,
        integrand1: F1,
        integrand2: F2,
        integrator: &I,
    ) -> f64
    where
        F1: Fn(f64) -> f64,
        F2: Fn(f64) -> f64,
        I: Integrator + ?Sized,
    {
        let integral1 = integrator
            .integrate(integrand1, 0.0, self.c, self.error_tolerance)
            .value;

        let integral2 = integrator
            .integrate(integrand2, 0.0, self.c, self.error_tolerance)
            .value;

        integral1 - self.s0_at_c * self.s1_at_c * integral2
    }
}

/// Calculates the probability win ratio, PR(c).
///
/// ## Formula
///
/// PR(c) = W(c) / L(c)
///
/// ## Returns
///
/// The probability win ratio as a `f64`. Returns `f64::INFINITY` if the loss probability is zero.
#[verified_engine::verified]
pub fn calculate_probability_win_ratio(win_probability: f64, loss_probability: f64) -> f64 {
    if loss_probability == 0.0 {
        return f64::INFINITY;
    }
    win_probability / loss_probability
}

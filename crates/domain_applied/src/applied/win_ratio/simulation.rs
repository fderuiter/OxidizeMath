//! # Mathematics of the Simulation Study
//!
//! This module implements the formulas used in the paper's simulation study.
//! It includes joint survival functions, the win ratio parameter, and derived
//! marginal and conditional survival functions and their PDFs.

pub use super::power_sweep::{
    MAX_ARM_SIZE, MAX_SWEEP_REPLICATES, MAX_SWEEP_STEPS, PowerSweepConfig, PowerSweepReport,
    PowerSweepWorkspace, SampleSizePowerPoint, run_power_sweep_into, run_weibull_power_sweep,
    simulate_weibull_trial_into,
};
pub use super::weibull_sampler::SimulatedSubject;

/// Represents a Bivariate Survival Model (Weibull-like) used in the simulation.
///
/// The joint survival function is defined as:
/// S(t, x) = exp( - (lambda1 * t + lambda2 * x)^alpha )
#[derive(Debug, Clone, Copy)]
pub struct BivariateWeibullModel {
    /// Scale parameter for the first event (T).
    pub lambda1: f64,
    /// Scale parameter for the second event (X).
    pub lambda2: f64,
    /// Shape parameter (correlation).
    pub alpha: f64,
}

impl BivariateWeibullModel {
    /// Creates a new `BivariateWeibullModel`.
    #[verified_engine::verified]
    pub fn new(lambda1: f64, lambda2: f64, alpha: f64) -> Self {
        Self {
            lambda1,
            lambda2,
            alpha,
        }
    }

    /// Joint survival function S(t, x).
    #[verified_engine::verified]
    pub fn joint_survival(&self, t: f64, x: f64) -> f64 {
        (-(self.lambda1 * t + self.lambda2 * x).powf(self.alpha)).exp()
    }

    /// Marginal survival function for T, S(t).
    #[verified_engine::verified]
    pub fn marginal_survival_t(&self, t: f64) -> f64 {
        self.joint_survival(t, 0.0)
    }

    /// PDF for T, f(t).
    #[verified_engine::verified]
    pub fn pdf_t(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let l1_alpha = self.lambda1.powf(self.alpha);
        // f(t) = - dS(t)/dt = alpha * lambda1 * (lambda1 * t)^(alpha-1) * S(t)
        //      = alpha * lambda1^alpha * t^(alpha-1) * S(t)
        self.alpha * l1_alpha * t.powf(self.alpha - 1.0) * self.marginal_survival_t(t)
    }

    /// Conditional survival function for X given T > c, G(x|c) = S(c, x) / S(c, 0).
    #[verified_engine::verified]
    pub fn conditional_survival_x_given_t(&self, x: f64, c: f64) -> f64 {
        let s_c_x = self.joint_survival(c, x);
        let s_c_0 = self.marginal_survival_t(c);
        if s_c_0 == 0.0 { 0.0 } else { s_c_x / s_c_0 }
    }

    /// Conditional PDF for X given T > c, f(x|c).
    #[verified_engine::verified]
    pub fn pdf_x_given_t(&self, x: f64, c: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let term = self.lambda1 * c + self.lambda2 * x;
        if term < 0.0 {
            return 0.0;
        }
        let g = self.conditional_survival_x_given_t(x, c);
        // f(x|c) = - dG(x|c)/dx
        // G(x|c) = exp( - (lambda1*c + lambda2*x)^alpha ) / S(c,0)
        // dG/dx = G * (- alpha * (lambda1*c + lambda2*x)^(alpha-1) * lambda2)
        // f = alpha * lambda2 * (term)^(alpha-1) * G
        self.alpha * self.lambda2 * term.powf(self.alpha - 1.0) * g
    }
}

/// Parameters for the simulation study.
#[derive(Debug, Clone, Copy)]
pub struct SimulationParams {
    /// Lambda1 parameter.
    pub lambda1: f64,
    /// Lambda2 parameter.
    pub lambda2: f64,
    /// Alpha parameter (correlation).
    pub alpha: f64,
    /// Theta parameter (treatment effect).
    pub theta: f64,
}

impl SimulationParams {
    /// Creates a new `SimulationParams`.
    #[verified_engine::verified]
    pub fn new(lambda1: f64, lambda2: f64, alpha: f64, theta: f64) -> Self {
        Self {
            lambda1,
            lambda2,
            alpha,
            theta,
        }
    }

    /// Returns the model for the control group.
    #[verified_engine::verified]
    pub fn control_model(&self) -> BivariateWeibullModel {
        BivariateWeibullModel::new(self.lambda1, self.lambda2, self.alpha)
    }

    /// Returns the model for the treatment group.
    #[verified_engine::verified]
    pub fn treatment_model(&self) -> BivariateWeibullModel {
        BivariateWeibullModel::new(
            self.theta * self.lambda1,
            self.theta * self.lambda2,
            self.alpha,
        )
    }
}

// --- Joint Survival Functions ---

/// Joint survival function for the control group, S0(t,x).
#[verified_engine::verified]
pub fn joint_survival_function_control(t: f64, x: f64, params: &SimulationParams) -> f64 {
    params.control_model().joint_survival(t, x)
}

/// Joint survival function for the treatment group, S1(t,x).
#[verified_engine::verified]
pub fn joint_survival_function_treatment(t: f64, x: f64, params: &SimulationParams) -> f64 {
    params.treatment_model().joint_survival(t, x)
}

// --- Win Ratio Parameter ---

/// Win ratio parameter in the absence of censoring, PR_W.
#[verified_engine::verified]
pub fn win_ratio_parameter(params: &SimulationParams) -> f64 {
    // This depends on the specific relationship between control and treatment models defined by theta.
    1.0 / params.theta.powf(params.alpha)
}

// --- Derived Functions for T (Fatal Event) ---

/// Marginal survival function for T in the control group, S0(t).
#[verified_engine::verified]
pub fn marginal_survival_t_control(t: f64, params: &SimulationParams) -> f64 {
    params.control_model().marginal_survival_t(t)
}

/// Marginal survival function for T in the treatment group, S1(t).
#[verified_engine::verified]
pub fn marginal_survival_t_treatment(t: f64, params: &SimulationParams) -> f64 {
    params.treatment_model().marginal_survival_t(t)
}

/// PDF for T in the control group, f0(t).
#[verified_engine::verified]
pub fn pdf_t_control(t: f64, params: &SimulationParams) -> f64 {
    params.control_model().pdf_t(t)
}

/// PDF for T in the treatment group, f1(t).
#[verified_engine::verified]
pub fn pdf_t_treatment(t: f64, params: &SimulationParams) -> f64 {
    params.treatment_model().pdf_t(t)
}

// --- Derived Functions for X (Non-Fatal Event) given T > c ---

/// Conditional survival function for X in the control group, G0(x|c).
#[verified_engine::verified]
pub fn conditional_survival_x_given_t_control(x: f64, c: f64, params: &SimulationParams) -> f64 {
    params.control_model().conditional_survival_x_given_t(x, c)
}

/// Conditional survival function for X in the treatment group, G1(x|c).
#[verified_engine::verified]
pub fn conditional_survival_x_given_t_treatment(x: f64, c: f64, params: &SimulationParams) -> f64 {
    params
        .treatment_model()
        .conditional_survival_x_given_t(x, c)
}

/// Conditional PDF for X in the control group, f0(x|c).
#[verified_engine::verified]
pub fn pdf_x_given_t_control(x: f64, c: f64, params: &SimulationParams) -> f64 {
    params.control_model().pdf_x_given_t(x, c)
}

/// Conditional PDF for X in the treatment group, f1(x|c).
#[verified_engine::verified]
pub fn pdf_x_given_t_treatment(x: f64, c: f64, params: &SimulationParams) -> f64 {
    params.treatment_model().pdf_x_given_t(x, c)
}

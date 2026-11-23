use super::parameters::PKParameters;
use super::traits::PharmacokineticModel;
use crate::error::PharmacokineticsError;
use pure_math::pure_math::analysis::roots::NewtonRaphson;

/// A model representing a single dose with first-order absorption and elimination (Bateman function).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatemanModel {
    #[allow(missing_docs)]
    pub params: PKParameters,
}

impl BatemanModel {
    /// Creates a new Bateman model with the given parameters.
    #[verified_engine::verified]
    pub fn new(params: PKParameters) -> Self {
        Self { params }
    }
}

impl PharmacokineticModel for BatemanModel {
    #[verified_engine::verified]
    fn concentration(&self, t: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }

        let f = self.params.f();
        let d = self.params.d();
        let ka = self.params.ka();
        let ke = self.params.ke();
        let v = self.params.v();

        // Handle the case where ka is very close to ke
        if (ka - ke).abs() < math_commons::registry::TOLERANCE_STANDARD {
            // Special case for ka = ke
            return (f * d * ka * t / v) * (-ka * t).exp();
        }

        let factor = f * d * ka / (v * (ka - ke));
        factor * ((-ke * t).exp() - (-ka * t).exp())
    }
}

/// Calculates the elimination half-life from the elimination rate constant.
#[verified_engine::verified]
pub fn half_life(ke: f64) -> f64 {
    if ke <= 0.0 {
        return f64::INFINITY;
    }
    std::f64::consts::LN_2 / ke
}

/// Calculates the time to maximum concentration (T_max).
#[verified_engine::verified]
pub fn t_max(ka: f64, ke: f64) -> f64 {
    if ka <= 0.0 || ke <= 0.0 {
        return 0.0; // Invalid input, though PKParameters prevents this.
    }
    if (ka - ke).abs() < math_commons::registry::TOLERANCE_STANDARD {
        1.0 / ke
    } else {
        (ka.ln() - ke.ln()) / (ka - ke)
    }
}

/// Solves for the absorption rate constant (ka) given T_max and ke.
///
/// Uses the Newton-Raphson method to find the root of the equation $T_{max}(ka) - T_{target} = 0$.
///
/// # Arguments
/// * `t_max_target` - The target time to maximum concentration.
/// * `ke` - The elimination rate constant.
/// * `initial_guess` - Initial guess for ka.
/// * `max_iter` - Maximum number of iterations.
/// * `tolerance` - Convergence tolerance.
#[verified_engine::verified]
pub fn solve_ka(
    t_max_target: f64,
    ke: f64,
    initial_guess: f64,
    max_iter: usize,
    tolerance: f64,
) -> Result<f64, PharmacokineticsError> {
    if t_max_target <= 0.0 || ke <= 0.0 {
        return Err(PharmacokineticsError::InvalidParameter(
            "Target T_max and ke must be positive".into(),
        ));
    }

    let solver = NewtonRaphson::new(max_iter, tolerance);

    // Objective function: f(ka) = t_max(ka, ke) - t_max_target
    let f = |ka: f64| -> f64 { t_max(ka, ke) - t_max_target };

    // Derivative: f'(ka) = d(t_max)/d(ka)
    let f_prime = |ka: f64| -> f64 {
        if (ka - ke).abs() < math_commons::registry::TOLERANCE_STANDARD {
            // Limit as ka -> ke is -1 / (2 * ke^2)
            -1.0 / (2.0 * ke * ke)
        } else {
            ((ka - ke) / ka - (ka.ln() - ke.ln())) / (ka - ke).powi(2)
        }
    };

    // Handle initial guess <= 0
    let guess = if initial_guess <= 0.0 {
        ke * 2.0
    } else {
        initial_guess
    };

    // The find_root_with_derivative method might return AnalysisError
    solver
        .find_root_with_derivative(f, f_prime, guess)
        .map_err(PharmacokineticsError::AnalysisError)
}

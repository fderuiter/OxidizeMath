use super::parameters::PKParameters;
use super::traits::PharmacokineticModel;
use crate::error::PharmacokineticsError;
use engine_symbolic::Dual;
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{DualArrayState, OdeSystem};
use pure_math::pure_math::analysis::roots::NewtonRaphson;

/// A model representing a single dose with first-order absorption and elimination (Bateman function).
#[derive(Debug, Clone, Copy)]
pub struct BatemanModel {
    #[allow(missing_docs)]
    pub params: PKParameters,
}

impl BatemanModel {
    /// Creates a new Bateman model with the given parameters.
    #[verified_engine::verified]
    pub fn new(params: PKParameters) -> Self {
        debug_assert!(params.d() >= 0.0);
        debug_assert!(params.v() > 0.0);
        Self { params }
    }

    /// Returns a copy of this Bateman model with its dose updated via [`PKParameters::with_dose`].
    #[verified_engine::verified]
    pub fn with_dose(&self, dose_mg: f64) -> Result<Self, PharmacokineticsError> {
        let updated = self.params.with_dose(dose_mg)?;
        debug_assert!(updated.d() >= 0.0);
        debug_assert_eq!(updated.v(), self.params.v());
        Ok(Self { params: updated })
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

/// Calibrates `(ka, ke)` in $\text{h}^{-1}$ from target $T_{\max}$ (hours) and elimination half-life $t_{1/2}$ (hours).
///
/// Selects a convex monotone starting point so [`solve_ka`] converges strictly inside $k_a > 0$.
#[verified_engine::verified]
pub fn calibrate_ka_from_half_life(
    t_max_target: f64,
    half_life_h: f64,
    max_iter: usize,
    tolerance: f64,
) -> Result<(f64, f64), PharmacokineticsError> {
    if !t_max_target.is_finite()
        || t_max_target <= 0.0
        || !half_life_h.is_finite()
        || half_life_h <= 0.0
    {
        return Err(PharmacokineticsError::InvalidParameter(
            "t_max_target and half_life_h must be positive finite values".to_string(),
        ));
    }
    let ke = std::f64::consts::LN_2 / half_life_h;
    let inv_ke = 1.0 / ke;
    let initial_guess =
        if (t_max_target - inv_ke).abs() < math_commons::registry::TOLERANCE_STANDARD {
            return Ok((ke, ke));
        } else if t_max_target < inv_ke {
            ke
        } else {
            (ke * (-ke * t_max_target).exp()).max(1e-9)
        };
    let ka = solve_ka(t_max_target, ke, initial_guess, max_iter, tolerance)?;
    debug_assert!(ka > 0.0);
    debug_assert!(ke > 0.0);
    Ok((ka, ke))
}

/// Augmented one-compartment oral ADME differential system `[A_g, C_p, k_e]` matching [`BatemanModel`].
///
/// Models gut depot drug mass $A_g$ ($\text{mg}$), central plasma concentration $C_p$ ($\text{mg/L}$),
/// and subject-specific elimination rate constant $k_e$ ($\text{h}^{-1}$) for joint state-parameter
/// estimation and closed-loop dosing:
/// $$\dot{A}_g = -k_a A_g + u_{\text{in}}, \qquad \dot{C}_p = \frac{k_a}{V} A_g - k_e C_p, \qquad \dot{k}_e = 0$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatemanCompartmentDynamics {
    /// First-order gut absorption rate constant $k_a$ ($\text{h}^{-1}$).
    pub ka: f64,
    /// Apparent central volume of distribution $V$ ($\text{L}$).
    pub volume: f64,
    /// Bioavailable continuous oral input rate $u_{\text{in}} = F \cdot R_{\text{in}}$ ($\text{mg/h}$).
    pub input_rate_mg_per_h: f64,
}

impl BatemanCompartmentDynamics {
    /// Constructs a bolus-driven oral ADME compartment system (`input_rate_mg_per_h = 0.0`).
    #[must_use]
    #[verified_engine::verified]
    pub const fn new(ka: f64, volume: f64) -> Self {
        Self {
            ka,
            volume,
            input_rate_mg_per_h: 0.0,
        }
    }

    /// Constructs a controlled oral ADME compartment system with continuous bioavailable input rate.
    #[must_use]
    #[verified_engine::verified]
    pub const fn with_input_rate(ka: f64, volume: f64, input_rate_mg_per_h: f64) -> Self {
        Self {
            ka,
            volume,
            input_rate_mg_per_h,
        }
    }
}

impl OdeSystem<Vector3<f64>> for BatemanCompartmentDynamics {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &Vector3<f64>) -> Vector3<f64> {
        let ag = state.x;
        let cp = state.y;
        let ke = state.z;
        let inv_v = 1.0 / self.volume.max(1.0e-3);
        Vector3::new(
            -self.ka * ag + self.input_rate_mg_per_h,
            self.ka * inv_v * ag - ke * cp,
            0.0,
        )
    }
}

impl OdeSystem<DualArrayState<3>> for BatemanCompartmentDynamics {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &DualArrayState<3>) -> DualArrayState<3> {
        let [ag, cp, ke] = state.0;
        let inv_v = 1.0 / self.volume.max(1.0e-3);
        let d_ag = -ag * self.ka + Dual::constant(self.input_rate_mg_per_h);
        let d_cp = ag * (self.ka * inv_v) - ke * cp;
        DualArrayState([d_ag, d_cp, Dual::ZERO])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pure_math::pure_math::analysis::ode::rk4_step_dual_jacobian;

    #[test]
    #[verified_engine::verified]
    fn test_bateman_compartment_dynamics_matches_closed_form_and_dual_jacobian() {
        let params = PKParameters::new(0.85, 105.0, 1.10, 0.13, 25.0).unwrap();
        let model = BatemanModel::new(params);
        let dyn_pk = BatemanCompartmentDynamics::new(1.10, 25.0);

        let mut st = [0.85 * 105.0, 0.0, 0.13];
        let dt = 0.05;
        let mut step = 1;
        while step <= 20 {
            let (ns, _) = rk4_step_dual_jacobian(&dyn_pk, 0.0, &st, dt);
            st = ns;
            let exact = model.concentration(step as f64 * dt);
            assert!((st[1] - exact).abs() < 1.0e-6);
            step += 1;
        }

        let probe = [42.0, 2.15, 0.16];
        let (_, rhs_jac) =
            pure_math::pure_math::analysis::ode::ode_rhs_dual_jacobian(&dyn_pk, 0.0, &probe);
        assert!((rhs_jac[0][0] - (-1.10)).abs() < 1.0e-15);
        assert!((rhs_jac[1][0] - (1.10 / 25.0)).abs() < 1.0e-15);
        assert!((rhs_jac[1][1] - (-0.16)).abs() < 1.0e-15);
        assert!((rhs_jac[1][2] - (-2.15)).abs() < 1.0e-15);

        let (_, jac) = rk4_step_dual_jacobian(&dyn_pk, 0.0, &probe, 0.5);
        let eps = 1.0e-5;
        let mut col = 0;
        while col < 3 {
            let mut xp = probe;
            let mut xm = probe;
            xp[col] += eps;
            xm[col] -= eps;
            let (fp, _) = rk4_step_dual_jacobian(&dyn_pk, 0.0, &xp, 0.5);
            let (fm, _) = rk4_step_dual_jacobian(&dyn_pk, 0.0, &xm, 0.5);
            let mut row = 0;
            while row < 3 {
                let fd = (fp[row] - fm[row]) / (2.0 * eps);
                assert!((jac[row][col] - fd).abs() < 1.0e-8);
                row += 1;
            }
            col += 1;
        }
    }
}

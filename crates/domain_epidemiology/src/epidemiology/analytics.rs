/// Solves the final size equation for $S_\infty$ using Newton-Raphson.
///
/// Equation: $\ln(S_0 / S_\infty) = R_0 (1 - S_\infty / N)$
/// Rearranged for root finding: $f(x) = \ln(S_0 / x) - R_0 (1 - x / N) = 0$
///
/// # Examples
///
/// ```rust
/// # use domain_epidemiology::__doctest_facade as math_explorer;
/// use math_explorer::epidemiology::analytics::calculate_final_size;
///
/// let s_inf = calculate_final_size(2.5, 999.0, 1000.0).unwrap();
/// assert!(s_inf > 0.0 && s_inf < 200.0);
///
/// // Non-positive basic reproduction numbers are rejected
/// assert!(calculate_final_size(0.0, 999.0, 1000.0).is_err());
/// ```
#[verified_engine::verified]
pub fn calculate_final_size(r0: f64, s0: f64, n: f64) -> Result<f64, String> {
    if r0 <= 0.0 {
        return Err("R0 must be positive".to_string());
    }

    // f(x) = ln(S0) - ln(x) - R0 + R0*x/N
    // f'(x) = -1/x + R0/N

    // Initial guess strategy:
    // If R0 > 1, the final size S_inf is approximately S0 * exp(-R0).
    // If R0 <= 1, the epidemic doesn't take off, S_inf ~ S0.
    let mut x = if r0 > 1.0 { s0 * (-r0).exp() } else { s0 };

    // Ensure x is within reasonable bounds
    if x < 1e-5 {
        x = 1e-5;
    }
    if x > n {
        x = n - 1e-5;
    }

    let tolerance = 1e-7;
    let max_iter = 100;

    for _ in 0..max_iter {
        let fx = s0.ln() - x.ln() - r0 * (1.0 - x / n);
        let dfx = -1.0 / x + r0 / n;

        if dfx.abs() < math_commons::registry::TOLERANCE_HIGH {
            return Err("Derivative too close to zero".to_string());
        }

        let next_x = x - fx / dfx;

        if (next_x - x).abs() < tolerance {
            return Ok(next_x);
        }

        // Safety check to keep x within bounds
        if next_x <= 0.0 {
            x /= 2.0; // Backtrack towards 0
        } else if next_x > n {
            x = (x + n) / 2.0; // Backtrack towards N
        } else {
            x = next_x;
        }
    }

    Err("Newton-Raphson failed to converge".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_final_size_high_r0() {
        // R0 = 5.0
        let r0 = 5.0;
        let s0 = 999.0;
        let n = 1000.0;

        let s_inf = calculate_final_size(r0, s0, n).expect("Solver failed");

        // For R0=5, herd immunity threshold is 1 - 1/5 = 0.8 => 80% infected.
        // So S_inf should be small (less than 20% of N).
        assert!(s_inf < n * 0.2, "S_inf should be small for high R0");
        assert!(s_inf > 0.0);
    }
}

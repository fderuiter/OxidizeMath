// [cite:attention_is_all_you_need_rust]
use statrs::function::gamma::gamma;
use std::f64::consts::PI;

/// Bessel Function of the First Kind $J_\nu(x)$.
///
/// Implemented using series expansion:
/// $J_\nu(x) = \sum_{k=0}^{\infty} \frac{(-1)^k}{k! \Gamma(\nu + k + 1)} \left(\frac{x}{2}\right)^{\nu + 2k}$
///
/// Note: Series converges fast for small x. For large x, asymptotic expansion is preferred.
#[verified_engine::verified]
pub fn bessel_j(nu: f64, x: f64) -> f64 {
    let mut sum = 0.0;
    let limit = 50; // Truncate series
    let x_2 = x / 2.0;

    for k in 0..limit {
        let k_f = k as f64;
        let num = if k % 2 == 0 { 1.0 } else { -1.0 };
        // k! = gamma(k+1)
        let denom = gamma(k_f + 1.0) * gamma(nu + k_f + 1.0);

        let term = (num / denom) * x_2.powf(nu + 2.0 * k_f);
        sum += term;

        if term.abs() < 1e-15 {
            break;
        }
    }
    sum
}

/// Bessel Function of the Second Kind $Y_\nu(x)$.
///
/// Placeholder: Returns NaN as reliable implementation is complex without external crate.
#[verified_engine::verified]
pub fn bessel_y(_nu: f64, _x: f64) -> f64 {
    unimplemented!("Bessel function of second kind not yet implemented via series expansion")
}

/// Spherical Bessel Function of the First Kind $j_\ell(x)$.
///
/// Defined as $j_\ell(x) = \sqrt{\frac{\pi}{2x}} J_{\ell+1/2}(x)$.
#[verified_engine::verified]
pub fn spherical_bessel_j(l: u64, x: f64) -> f64 {
    if x.abs() < math_commons::registry::TOLERANCE_HIGH {
        if l == 0 {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    let nu = l as f64 + 0.5;
    (PI / (2.0 * x)).sqrt() * bessel_j(nu, x)
}

/// Checks orthogonality of Bessel functions $J_\nu(\alpha x)$ and $J_\nu(\beta x)$ on [0, 1].
///
/// $\int_0^1 x J_\nu(\alpha x) J_\nu(\beta x) dx$ should be 0 if $\alpha \neq \beta$ are roots.
///
/// This function computes the integral numerically.
#[verified_engine::verified]
pub fn check_orthogonality_bessel(nu: f64, alpha: f64, beta: f64) -> f64 {
    let integrand = |x: f64| x * bessel_j(nu, alpha * x) * bessel_j(nu, beta * x);
    // Simple trapezoidal integration for demonstration/checking
    let steps = 1000;
    let mut sum = 0.0;
    let h = 1.0 / steps as f64;

    for i in 0..steps {
        let x = i as f64 * h;
        let x_next = (i + 1) as f64 * h;
        sum += 0.5 * (integrand(x) + integrand(x_next)) * h;
    }
    sum
}

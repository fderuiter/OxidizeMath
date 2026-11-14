// [cite:attention_is_all_you_need_rust]
use statrs::function::gamma::gamma;
use std::f64::consts::{EULER_GAMMA, PI};

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

const EULER_MASCHERONI: f64 = EULER_GAMMA;

/// Bessel Function of the Second Kind $Y_\nu(x)$ (Neumann function).
///
/// Implemented using Neumann series expansion for integer orders and trigonometric
/// relation $Y_\nu(x) = \frac{J_\nu(x)\cos(\nu\pi) - J_{-\nu}(x)}{\sin(\nu\pi)}$ for non-integer orders.
#[verified_engine::verified]
pub fn bessel_y(_nu: f64, _x: f64) -> f64 {
    let nu = _nu;
    let x = _x;

    if x.is_nan() || nu.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() || nu.is_infinite() {
        return f64::NAN;
    }

    let is_integer = (nu - nu.round()).abs() < 1e-10;
    if is_integer {
        let n_raw = nu.round() as i64;
        let n = n_raw.unsigned_abs() as usize;
        let x_2 = x / 2.0;
        let limit = 50;

        // First finite sum for n >= 1
        let mut sum1 = 0.0;
        for k in 0..n {
            let k_f = k as f64;
            let num = gamma((n - k) as f64);
            let denom = gamma(k_f + 1.0);
            sum1 += (num / denom) * x_2.powf(2.0 * k_f - n as f64);
        }
        let term1 = -sum1 / PI;

        // Logarithmic term
        let term2 = (2.0 / PI) * (x_2.ln() + EULER_MASCHERONI) * bessel_j(n as f64, x);

        // Infinite series with harmonic numbers
        let mut sum3 = 0.0;
        let mut h_k = 0.0;
        let mut h_nk = 0.0;
        for j in 1..=n {
            h_nk += 1.0 / j as f64;
        }

        for k in 0..limit {
            let k_f = k as f64;
            let hk = h_k + h_nk;
            let num = if k % 2 == 0 { 1.0 } else { -1.0 };
            let denom = gamma(k_f + 1.0) * gamma((n + k) as f64 + 1.0);
            let term = (num * hk / denom) * x_2.powf(2.0 * k_f + n as f64);
            sum3 += term;

            h_k += 1.0 / (k + 1) as f64;
            h_nk += 1.0 / (n + k + 1) as f64;

            if k > 0 && term.abs() < 1e-15 {
                break;
            }
        }
        let term3 = -sum3 / PI;

        let y_n = term1 + term2 + term3;
        if n_raw < 0 && n % 2 == 1 { -y_n } else { y_n }
    } else {
        let j_nu = bessel_j(nu, x);
        let j_neg_nu = bessel_j(-nu, x);
        (j_nu * (nu * PI).cos() - j_neg_nu) / (nu * PI).sin()
    }
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

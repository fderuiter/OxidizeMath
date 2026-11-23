//! `#![no_std]` regularized incomplete gamma ($P(a, x)$, $Q(a, x)$) and incomplete beta ($I_x(a, b)$).

use super::gamma::{ln_beta, ln_gamma};
use crate::error::ProbError;

const MAX_SERIES_ITERS: usize = 500;
const CF_EPS: f64 = 1.0e-15;
const CF_TINY: f64 = 1.0e-300;

#[cfg_attr(feature = "verified", verified_engine::verified)]
fn lower_gamma_series(a: f64, x: f64) -> Result<f64, ProbError> {
    debug_assert!(a > 0.0 && x > 0.0);
    let mut ap = a;
    let mut term = 1.0 / a;
    let mut sum = term;
    for _ in 0..MAX_SERIES_ITERS {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() <= sum.abs() * CF_EPS {
            let log_prefactor = a * libm::log(x) - x - ln_gamma(a);
            if log_prefactor < -745.0 {
                return Ok(0.0);
            }
            return Ok((sum * libm::exp(log_prefactor)).clamp(0.0, 1.0));
        }
    }
    Err(ProbError::NumericalFailure {
        reason: "incomplete gamma series failed to converge",
    })
}

#[cfg_attr(feature = "verified", verified_engine::verified)]
fn upper_gamma_cf(a: f64, x: f64) -> Result<f64, ProbError> {
    debug_assert!(a > 0.0 && x > 0.0);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / CF_TINY;
    let mut d = if b.abs() < CF_TINY {
        1.0 / CF_TINY
    } else {
        1.0 / b
    };
    let mut h = d;
    for i in 1..=MAX_SERIES_ITERS {
        let idx = i as f64;
        let an = -idx * (idx - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < CF_TINY {
            d = CF_TINY;
        }
        c = b + an / c;
        if c.abs() < CF_TINY {
            c = CF_TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= CF_EPS {
            let log_prefactor = a * libm::log(x) - x - ln_gamma(a);
            if log_prefactor < -745.0 {
                return Ok(0.0);
            }
            return Ok((h * libm::exp(log_prefactor)).clamp(0.0, 1.0));
        }
    }
    Err(ProbError::NumericalFailure {
        reason: "incomplete gamma continued fraction failed to converge",
    })
}

/// Evaluates the regularized lower incomplete gamma function $P(a, x) = \frac{\gamma(a, x)}{\Gamma(a)}$.
///
/// # Example
///
/// ```rust
/// # use engine_prob::__doctest_facade as math_explorer;
/// use math_explorer::engine_prob::regularized_lower_gamma;
///
/// // P(1, x) = 1 - exp(-x)
/// let p = regularized_lower_gamma(1.0, 2.0).unwrap();
/// assert!((p - (1.0 - (-2.0_f64).exp())).abs() < 1e-14);
/// ```
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn regularized_lower_gamma(a: f64, x: f64) -> Result<f64, ProbError> {
    if !a.is_finite() || a <= 0.0 {
        return Err(ProbError::InvalidParameter {
            name: "a",
            value: a,
        });
    }
    if x.is_nan() || x < 0.0 {
        return Err(ProbError::InvalidParameter {
            name: "x",
            value: x,
        });
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if x.is_infinite() {
        return Ok(1.0);
    }
    let val = if x < a + 1.0 {
        lower_gamma_series(a, x)?
    } else {
        (1.0 - upper_gamma_cf(a, x)?).clamp(0.0, 1.0)
    };
    debug_assert!((0.0..=1.0).contains(&val));
    Ok(val)
}

/// Evaluates the regularized upper incomplete gamma function $Q(a, x) = 1 - P(a, x)$.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn regularized_upper_gamma(a: f64, x: f64) -> Result<f64, ProbError> {
    if !a.is_finite() || a <= 0.0 {
        return Err(ProbError::InvalidParameter {
            name: "a",
            value: a,
        });
    }
    if x.is_nan() || x < 0.0 {
        return Err(ProbError::InvalidParameter {
            name: "x",
            value: x,
        });
    }
    if x == 0.0 {
        return Ok(1.0);
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    let val = if x < a + 1.0 {
        (1.0 - lower_gamma_series(a, x)?).clamp(0.0, 1.0)
    } else {
        upper_gamma_cf(a, x)?
    };
    debug_assert!((0.0..=1.0).contains(&val));
    Ok(val)
}

#[cfg_attr(feature = "verified", verified_engine::verified)]
fn betacf(a: f64, b: f64, x: f64) -> Result<f64, ProbError> {
    debug_assert!(a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&x));
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < CF_TINY {
        d = CF_TINY;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_SERIES_ITERS {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa_even = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa_even * d;
        if d.abs() < CF_TINY {
            d = CF_TINY;
        }
        c = 1.0 + aa_even / c;
        if c.abs() < CF_TINY {
            c = CF_TINY;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa_odd = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa_odd * d;
        if d.abs() < CF_TINY {
            d = CF_TINY;
        }
        c = 1.0 + aa_odd / c;
        if c.abs() < CF_TINY {
            c = CF_TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= CF_EPS {
            return Ok(h);
        }
    }
    Err(ProbError::NumericalFailure {
        reason: "incomplete beta continued fraction failed to converge",
    })
}

/// Evaluates the regularized incomplete beta function $I_x(a, b)$ for $a > 0, b > 0, x \in [0, 1]$.
///
/// Uses symmetry $I_x(a, b) = 1 - I_{1 - x}(b, a)$ when $x > \frac{a + 1}{a + b + 2}$ for rapid
/// modified-Lentz continued-fraction convergence.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> Result<f64, ProbError> {
    if !a.is_finite() || a <= 0.0 {
        return Err(ProbError::InvalidParameter {
            name: "a",
            value: a,
        });
    }
    if !b.is_finite() || b <= 0.0 {
        return Err(ProbError::InvalidParameter {
            name: "b",
            value: b,
        });
    }
    if !x.is_finite() || !(0.0..=1.0).contains(&x) {
        return Err(ProbError::InvalidProbability { value: x });
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }
    let log_bt = a * libm::log(x) + b * libm::log1p(-x) - ln_beta(a, b);
    let bt = if log_bt < -745.0 {
        0.0
    } else {
        libm::exp(log_bt)
    };
    let pivot = (a + 1.0) / (a + b + 2.0);
    let val = if x < pivot {
        (bt * betacf(a, b, x)? / a).clamp(0.0, 1.0)
    } else {
        (1.0 - bt * betacf(b, a, 1.0 - x)? / b).clamp(0.0, 1.0)
    };
    debug_assert!((0.0..=1.0).contains(&val));
    Ok(val)
}

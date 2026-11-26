/// Legendre Polynomial $P_\ell(x)$ computed via recurrence relation.
///
/// $(n+1)P_{n+1}(x) = (2n+1)x P_n(x) - n P_{n-1}(x)$
#[verified_engine::verified]
pub fn legendre_p(l: u64, x: f64) -> f64 {
    if l == 0 {
        return 1.0;
    }
    if l == 1 {
        return x;
    }

    let mut p_prev = 1.0; // P_0
    let mut p_curr = x; // P_1

    for n in 1..l {
        let n_f = n as f64;
        let p_next = ((2.0 * n_f + 1.0) * x * p_curr - n_f * p_prev) / (n_f + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

/// Associated Legendre Polynomial $P_\ell^m(x)$ for $-\ell \le m \le \ell$.
///
/// Uses upward recurrence in $\ell$ from the closed-form diagonal $P_{|m|}^{|m|}(x)$
/// with the Condon-Shortley phase $(-1)^m$, and applies
/// $P_\ell^{-|m|}(x) = (-1)^{|m|} \frac{(\ell - |m|)!}{(\ell + |m|)!} P_\ell^{|m|}(x)$ for $m < 0$.
#[verified_engine::verified]
pub fn legendre_p_associated(l: u64, m: i64, x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    let abs_m = m.unsigned_abs();
    if abs_m > l {
        return 0.0;
    }
    let xc = x.clamp(-1.0, 1.0);
    let p_pos = legendre_p_associated_nonneg(l, abs_m, xc);
    if m >= 0 {
        return p_pos;
    }
    let sign = if abs_m % 2 == 1 { -1.0 } else { 1.0 };
    sign * factorial_ratio_minus_plus(l, abs_m) * p_pos
}

#[verified_engine::verified]
fn factorial_ratio_minus_plus(l: u64, abs_m: u64) -> f64 {
    let mut ratio = 1.0;
    for k in (l - abs_m + 1)..=(l + abs_m) {
        ratio /= k as f64;
    }
    ratio
}

#[verified_engine::verified]
fn legendre_p_associated_nonneg(l: u64, m: u64, x: f64) -> f64 {
    let mut p_mm = 1.0;
    if m > 0 {
        let somx2 = (1.0 - x * x).max(0.0).sqrt();
        let mut fact = 1.0;
        for _ in 1..=m {
            p_mm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if l == m {
        return p_mm;
    }
    let p_m1m = x * (2.0 * m as f64 + 1.0) * p_mm;
    if l == m + 1 {
        return p_m1m;
    }
    let mut p_prev = p_mm;
    let mut p_curr = p_m1m;
    let m_f = m as f64;
    for ll in (m + 2)..=l {
        let ll_f = ll as f64;
        let p_next = (x * (2.0 * ll_f - 1.0) * p_curr - (ll_f + m_f - 1.0) * p_prev) / (ll_f - m_f);
        p_prev = p_curr;
        p_curr = p_next;
    }
    p_curr
}

/// Real orthonormal tesseral spherical harmonic $Y_{\ell m}(\theta, \phi)$ evaluated at `cos_theta` and `phi`.
///
/// Orthonormal on $S^2$: $\int_0^{2\pi}\int_{-1}^1 Y_{\ell m} Y_{\ell' m'} \, d(\cos\theta)\, d\phi = \delta_{\ell\ell'}\delta_{mm'}$.
#[verified_engine::verified]
pub fn real_spherical_harmonic(l: u64, m: i64, cos_theta: f64, phi: f64) -> f64 {
    let abs_m = m.unsigned_abs();
    if abs_m > l || !cos_theta.is_finite() || !phi.is_finite() {
        return 0.0;
    }
    let norm_sq =
        ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI)) * factorial_ratio_minus_plus(l, abs_m);
    let p_lm = legendre_p_associated(l, abs_m as i64, cos_theta);
    if m == 0 {
        return norm_sq.sqrt() * p_lm;
    }
    let cs_cancel = if abs_m % 2 == 1 { -1.0 } else { 1.0 };
    let angular = if m > 0 {
        (abs_m as f64 * phi).cos()
    } else {
        (abs_m as f64 * phi).sin()
    };
    std::f64::consts::SQRT_2 * cs_cancel * norm_sq.sqrt() * p_lm * angular
}

/// Checks orthogonality of Legendre polynomials on [-1, 1].
/// $\int_{-1}^1 P_\ell(x) P_k(x) dx = \frac{2}{2\ell+1} \delta_{\ell k}$
#[verified_engine::verified]
pub fn check_orthogonality_legendre(l: u64, k: u64) -> f64 {
    let steps = 1000;
    let h = 2.0 / steps as f64; // Range is 2 (-1 to 1)
    let mut sum = 0.0;

    let integrand = |x: f64| legendre_p(l, x) * legendre_p(k, x);

    for i in 0..steps {
        let x = -1.0 + i as f64 * h;
        let x_next = -1.0 + (i + 1) as f64 * h;
        sum += 0.5 * (integrand(x) + integrand(x_next)) * h;
    }
    sum
}

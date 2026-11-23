/// Hermite Polynomial $H_n(x)$ (Physicists' convention).
///
/// Recurrence: $H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x)$
/// $H_0(x) = 1$, $H_1(x) = 2x$.
#[verified_engine::verified]
pub fn hermite(n: u64, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }

    let mut h_prev = 1.0;
    let mut h_curr = 2.0 * x;

    for k in 1..n {
        let k_f = k as f64;
        let h_next = 2.0 * x * h_curr - 2.0 * k_f * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    h_curr
}

/// Laguerre Polynomial $L_n(x)$.
///
/// Recurrence: $(n+1)L_{n+1}(x) = (2n+1-x)L_n(x) - nL_{n-1}(x)$
/// $L_0(x) = 1$, $L_1(x) = 1 - x$.
#[verified_engine::verified]
pub fn laguerre(n: u64, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 1.0 - x;
    }

    let mut l_prev = 1.0;
    let mut l_curr = 1.0 - x;

    for k in 1..n {
        let k_f = k as f64;
        let l_next = ((2.0 * k_f + 1.0 - x) * l_curr - k_f * l_prev) / (k_f + 1.0);
        l_prev = l_curr;
        l_curr = l_next;
    }
    l_curr
}

/// Generalized (Associated) Laguerre Polynomial $L_n^{(\alpha)}(x)$ for radial hydrogenic eigenstates.
///
/// Satisfies the upward three-term recurrence:
/// $(k+1)L_{k+1}^{(\alpha)}(x) = (2k + 1 + \alpha - x)L_k^{(\alpha)}(x) - (k + \alpha)L_{k-1}^{(\alpha)}(x)$
/// with $L_0^{(\alpha)}(x) = 1$ and $L_1^{(\alpha)}(x) = 1 + \alpha - x$.
#[verified_engine::verified]
pub fn associated_laguerre(n: u64, alpha: f64, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 1.0 + alpha - x;
    }

    let mut l_prev = 1.0;
    let mut l_curr = 1.0 + alpha - x;

    for k in 1..n {
        let k_f = k as f64;
        let l_next =
            ((2.0 * k_f + 1.0 + alpha - x) * l_curr - (k_f + alpha) * l_prev) / (k_f + 1.0);
        l_prev = l_curr;
        l_curr = l_next;
    }
    l_curr
}

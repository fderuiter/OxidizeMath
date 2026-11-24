//! Zero-allocation discrete Shannon & Rényi entropies, divergences, and joint mutual information.

use super::ShannonError;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::registry::{TOLERANCE_FAST, TOLERANCE_HIGH};

/// Preallocated scratch workspace for computing mutual information of discrete joint distributions.
#[derive(Clone, Debug)]
pub struct JointMiWorkspace {
    row_marginals: Vec<f64>,
    col_marginals: Vec<f64>,
}

impl JointMiWorkspace {
    /// Allocates marginal buffers supporting joint tables up to `max_rows` by `max_cols`.
    #[must_use]
    pub fn new(max_rows: usize, max_cols: usize) -> Self {
        Self {
            row_marginals: vec![0.0; max_rows],
            col_marginals: vec![0.0; max_cols],
        }
    }

    /// Resizes marginal buffers if needed outside verified zero-allocation regions.
    pub fn ensure_capacity(&mut self, rows: usize, cols: usize) {
        if self.row_marginals.len() < rows {
            self.row_marginals.resize(rows, 0.0);
        }
        if self.col_marginals.len() < cols {
            self.col_marginals.resize(cols, 0.0);
        }
    }
}

/// Validates that `probs` is a non-empty, finite discrete probability distribution summing to `1.0`.
#[verified_engine::verified]
pub fn validate_pmf(probs: &[f64]) -> Result<(), ShannonError> {
    if probs.is_empty() {
        return Err(ShannonError::EmptyDistribution);
    }
    let mut sum = 0.0;
    for &p in probs {
        if !p.is_finite() || p < -TOLERANCE_HIGH || p > 1.0 + TOLERANCE_FAST {
            return Err(ShannonError::InvalidProbability { value: p });
        }
        sum += p.max(0.0);
    }
    if (sum - 1.0).abs() > TOLERANCE_FAST {
        return Err(ShannonError::UnnormalizedDistribution { sum });
    }
    Ok(())
}

/// Computes discrete Shannon entropy $H(P) = -\sum_i p_i \log_2 p_i$ in bits (`0 log2 0 = 0`).
#[verified_engine::verified]
pub fn categorical_entropy(probs: &[f64]) -> Result<f64, ShannonError> {
    validate_pmf(probs)?;
    let mut entropy = 0.0;
    for &raw_p in probs {
        let p = raw_p.max(0.0);
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    Ok(entropy.max(0.0))
}

/// Computes Rényi entropy $H_\alpha(P) = \frac{1}{1-\alpha}\log_2 \sum_i p_i^\alpha$ in bits for $\alpha \in [0, \infty]$.
///
/// Handles Hartley entropy ($\alpha = 0$), the Shannon limit ($\alpha \to 1$), and min-entropy ($\alpha = +\infty$).
#[verified_engine::verified]
pub fn renyi_entropy(probs: &[f64], alpha: f64) -> Result<f64, ShannonError> {
    validate_pmf(probs)?;
    if alpha.is_nan() || alpha < 0.0 {
        return Err(ShannonError::InvalidRenyiOrder { alpha });
    }
    if alpha.is_infinite() {
        let mut max_p = 0.0_f64;
        for &p in probs {
            max_p = max_p.max(p);
        }
        return Ok((-max_p.log2()).max(0.0));
    }
    if alpha <= TOLERANCE_HIGH {
        let mut support = 0usize;
        for &p in probs {
            if p > TOLERANCE_HIGH {
                support += 1;
            }
        }
        return Ok((support as f64).log2().max(0.0));
    }
    if (alpha - 1.0).abs() <= TOLERANCE_FAST {
        return categorical_entropy(probs);
    }

    let mut max_p = 0.0_f64;
    for &p in probs {
        max_p = max_p.max(p);
    }
    let mut scaled_sum = 0.0;
    for &raw_p in probs {
        let p = raw_p.max(0.0);
        if p > 0.0 {
            scaled_sum += (p / max_p).powf(alpha);
        }
    }
    let log2_power_sum = alpha * max_p.log2() + scaled_sum.log2();
    Ok((log2_power_sum / (1.0 - alpha)).max(0.0))
}

/// Computes Kullback-Leibler divergence $D_{\mathrm{KL}}(P \parallel Q) = \sum_i p_i \log_2(p_i / q_i)$ in bits.
///
/// Returns [`ShannonError::SupportMismatch`] if $p_i > 0$ while $q_i = 0$.
#[verified_engine::verified]
pub fn kl_divergence(p: &[f64], q: &[f64]) -> Result<f64, ShannonError> {
    validate_pmf(p)?;
    validate_pmf(q)?;
    if p.len() != q.len() {
        return Err(ShannonError::DimensionMismatch {
            expected: p.len(),
            actual: q.len(),
        });
    }
    let mut kl = 0.0;
    for (idx, (&raw_p, &raw_q)) in p.iter().zip(q.iter()).enumerate() {
        let pi = raw_p.max(0.0);
        let qi = raw_q.max(0.0);
        if pi > TOLERANCE_HIGH {
            if qi <= 0.0 {
                return Err(ShannonError::SupportMismatch { index: idx });
            }
            kl += pi * (pi / qi).log2();
        }
    }
    Ok(kl.max(0.0))
}

/// Computes symmetric base-2 Jensen-Shannon divergence $\mathrm{JSD}(P \parallel Q) \in [0, 1]$ in a single pass.
#[verified_engine::verified]
pub fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> Result<f64, ShannonError> {
    validate_pmf(p)?;
    validate_pmf(q)?;
    if p.len() != q.len() {
        return Err(ShannonError::DimensionMismatch {
            expected: p.len(),
            actual: q.len(),
        });
    }
    let mut jsd = 0.0;
    for (&raw_p, &raw_q) in p.iter().zip(q.iter()) {
        let pi = raw_p.max(0.0);
        let qi = raw_q.max(0.0);
        let mi = 0.5 * (pi + qi);
        if pi > 0.0 {
            jsd += 0.5 * pi * (pi / mi).log2();
        }
        if qi > 0.0 {
            jsd += 0.5 * qi * (qi / mi).log2();
        }
    }
    Ok(jsd.clamp(0.0, 1.0))
}

/// Computes joint mutual information $I(X; Y)$ in bits from a row-major `n_rows x n_cols` joint PMF.
#[verified_engine::verified]
pub fn joint_mutual_information(
    joint: &[f64],
    n_rows: usize,
    n_cols: usize,
    workspace: &mut JointMiWorkspace,
) -> Result<f64, ShannonError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(ShannonError::EmptyDistribution);
    }
    let expected = n_rows * n_cols;
    if joint.len() != expected {
        return Err(ShannonError::DimensionMismatch {
            expected,
            actual: joint.len(),
        });
    }
    if workspace.row_marginals.len() < n_rows {
        return Err(ShannonError::WorkspaceTooSmall {
            required: n_rows,
            actual: workspace.row_marginals.len(),
        });
    }
    if workspace.col_marginals.len() < n_cols {
        return Err(ShannonError::WorkspaceTooSmall {
            required: n_cols,
            actual: workspace.col_marginals.len(),
        });
    }
    validate_pmf(joint)?;

    let row_m = &mut workspace.row_marginals[..n_rows];
    let col_m = &mut workspace.col_marginals[..n_cols];
    row_m.fill(0.0);
    col_m.fill(0.0);

    for r in 0..n_rows {
        for c in 0..n_cols {
            let p_rc = joint[flatten_2d_index(c, r, n_cols)].max(0.0);
            row_m[r] += p_rc;
            col_m[c] += p_rc;
        }
    }

    let mut mi = 0.0;
    for r in 0..n_rows {
        let px = row_m[r];
        if px <= 0.0 {
            continue;
        }
        for c in 0..n_cols {
            let p_rc = joint[flatten_2d_index(c, r, n_cols)].max(0.0);
            let py = col_m[c];
            if p_rc > 0.0 && py > 0.0 {
                mi += p_rc * (p_rc / (px * py)).log2();
            }
        }
    }
    Ok(mi.max(0.0))
}

// [cite:algorithmic_information_rust]

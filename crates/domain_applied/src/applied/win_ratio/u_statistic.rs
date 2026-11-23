//! Pocock/Bebu-Lachin two-sample U-statistic variance estimation for unmatched win ratios.

use statrs::distribution::{ContinuousCDF, Normal};

use super::censoring::TierCount;
use super::pair_comparison::{ComparisonResult, WinRatioAnalysis, WinRatioStats};

/// Borrowed per-subject win and loss count slices for two-sample U-statistic variance projection.
#[derive(Debug, Clone, Copy)]
pub struct SubjectMarginals<'a> {
    /// Per-subject win counts for treatment arm (`len >= n_treatment`).
    pub row_wins: &'a [u32],
    /// Per-subject loss counts for treatment arm (`len >= n_treatment`).
    pub row_losses: &'a [u32],
    /// Per-subject treatment-win counts against each control subject (`len >= n_control`).
    pub col_wins: &'a [u32],
    /// Per-subject treatment-loss counts against each control subject (`len >= n_control`).
    pub col_losses: &'a [u32],
}

/// Preallocated scratch buffers for zero-allocation unmatched U-statistic evaluation.
#[derive(Debug, Clone, Default)]
pub struct UnmatchedScratch {
    /// Per-subject win counts for treatment arm (`len >= n_treatment`).
    pub row_wins: Vec<u32>,
    /// Per-subject loss counts for treatment arm (`len >= n_treatment`).
    pub row_losses: Vec<u32>,
    /// Per-subject treatment-win counts against each control subject (`len >= n_control`).
    pub col_wins: Vec<u32>,
    /// Per-subject treatment-loss counts against each control subject (`len >= n_control`).
    pub col_losses: Vec<u32>,
    /// Per-tier win/loss/tie counters (`len >= num_tiers`).
    pub tier_counts: Vec<TierCount>,
}

impl UnmatchedScratch {
    /// Preallocates scratch slices sized for up to `max_n1` treatment subjects,
    /// `max_n2` control subjects, and `max_tiers` prioritized endpoints.
    #[must_use]
    pub fn with_capacity(max_n1: usize, max_n2: usize, max_tiers: usize) -> Self {
        Self {
            row_wins: vec![0; max_n1],
            row_losses: vec![0; max_n1],
            col_wins: vec![0; max_n2],
            col_losses: vec![0; max_n2],
            tier_counts: vec![TierCount::default(); max_tiers],
        }
    }

    /// Borrows the first `n1` treatment and `n2` control marginal slices.
    #[verified_engine::verified]
    #[must_use]
    pub fn marginals(&self, n1: usize, n2: usize) -> SubjectMarginals<'_> {
        SubjectMarginals {
            row_wins: &self.row_wins[..n1],
            row_losses: &self.row_losses[..n1],
            col_wins: &self.col_wins[..n2],
            col_losses: &self.col_losses[..n2],
        }
    }
}

/// Summary of an unmatched two-sample win-ratio analysis using Pocock/Bebu-Lachin U-statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnmatchedWinRatioSummary {
    /// Number of subjects in the treatment group ($n_1$).
    pub n_treatment: usize,
    /// Number of subjects in the control group ($n_2$).
    pub n_control: usize,
    /// Total pairwise comparisons ($n_1 \times n_2$).
    pub total_pairs: usize,
    /// Total treatment wins ($N_W$).
    pub wins: usize,
    /// Total treatment losses ($N_L$).
    pub losses: usize,
    /// Total pairs tied across all composite tiers ($N_T$).
    pub ties: usize,
    /// Estimated unmatched win ratio ($N_W / N_L$).
    pub win_ratio: f64,
    /// Dong et al. win odds accounting for ties: $(N_W + 0.5 N_T) / (N_L + 0.5 N_T)$.
    pub win_odds: f64,
    /// Buyse net treatment benefit: $(N_W - N_L) / (n_1 n_2)$.
    pub net_benefit: f64,
    /// Lower confidence bound on the win ratio at level $1 - \alpha$.
    pub ci_low: f64,
    /// Upper confidence bound on the win ratio at level $1 - \alpha$.
    pub ci_high: f64,
    /// Two-sample U-statistic standard error of $\ln(\widehat{WR})$.
    pub log_wr_se: f64,
    /// Asymptotic standard normal Z-statistic testing $H_0: WR = 1$.
    pub z_score: f64,
    /// Two-sided p-value testing $H_0: WR = 1$.
    pub p_value: f64,
}

impl UnmatchedWinRatioSummary {
    /// Projects core confidence-interval and significance fields to [`WinRatioStats`].
    #[verified_engine::verified]
    #[must_use]
    pub fn as_stats(&self) -> WinRatioStats {
        WinRatioStats {
            win_ratio: self.win_ratio,
            ci_low: self.ci_low,
            ci_high: self.ci_high,
            p_value: self.p_value,
        }
    }
}

fn accumulate_log_wr_variance(
    n1: usize,
    n2: usize,
    p_w: f64,
    p_l: f64,
    m: &SubjectMarginals<'_>,
) -> f64 {
    debug_assert!(n1 >= 2 && n2 >= 2 && p_w > 0.0 && p_l > 0.0);
    let inv_n2 = 1.0 / (n2 as f64);
    let mut sum_psi1_sq = 0.0;
    for i in 0..n1 {
        let psi =
            (f64::from(m.row_wins[i]) * inv_n2) / p_w - (f64::from(m.row_losses[i]) * inv_n2) / p_l;
        sum_psi1_sq += psi * psi;
    }

    let inv_n1 = 1.0 / (n1 as f64);
    let mut sum_psi2_sq = 0.0;
    for j in 0..n2 {
        let psi =
            (f64::from(m.col_wins[j]) * inv_n1) / p_w - (f64::from(m.col_losses[j]) * inv_n1) / p_l;
        sum_psi2_sq += psi * psi;
    }

    let denom1 = (n1 * (n1 - 1)) as f64;
    let denom2 = (n2 * (n2 - 1)) as f64;
    (sum_psi1_sq / denom1) + (sum_psi2_sq / denom2)
}

fn accumulate_net_benefit_variance(
    n1: usize,
    n2: usize,
    net_benefit: f64,
    m: &SubjectMarginals<'_>,
) -> f64 {
    let inv_n2 = 1.0 / (n2 as f64);
    let mut s1 = 0.0;
    for i in 0..n1 {
        let diff = (f64::from(m.row_wins[i]) - f64::from(m.row_losses[i])) * inv_n2 - net_benefit;
        s1 += diff * diff;
    }
    let inv_n1 = 1.0 / (n1 as f64);
    let mut s2 = 0.0;
    for j in 0..n2 {
        let diff = (f64::from(m.col_wins[j]) - f64::from(m.col_losses[j])) * inv_n1 - net_benefit;
        s2 += diff * diff;
    }
    (s1 / ((n1 * (n1 - 1)) as f64)) + (s2 / ((n2 * (n2 - 1)) as f64))
}

#[verified_engine::verified]
fn degenerate_inference(
    n1: usize,
    n2: usize,
    wins: usize,
    losses: usize,
    ties: usize,
    net_benefit: f64,
    m: &SubjectMarginals<'_>,
) -> Option<(f64, f64, f64, f64, f64)> {
    let normal = Normal::new(0.0, 1.0).ok()?;
    let win_ratio = if losses == 0 { f64::INFINITY } else { 0.0 };
    let var_net = accumulate_net_benefit_variance(n1, n2, net_benefit, m);
    let (z_score, p_value) = if var_net > 0.0 {
        let z = net_benefit / var_net.sqrt();
        (z, (2.0 * (1.0 - normal.cdf(z.abs()))).clamp(0.0, 1.0))
    } else if wins > 0 {
        (f64::INFINITY, 0.0)
    } else {
        (f64::NEG_INFINITY, 0.0)
    };
    let wo_num = wins as f64 + 0.5 * (ties as f64);
    let wo_den = losses as f64 + 0.5 * (ties as f64);
    let ci_low = if losses == 0 { wo_num / 0.5 } else { 0.0 };
    let ci_high = if losses == 0 {
        f64::INFINITY
    } else {
        0.5 / wo_den.max(0.5)
    };
    Some((win_ratio, ci_low, ci_high, z_score, p_value))
}

/// Computes Pocock/Bebu-Lachin two-sample U-statistic inference from pre-accumulated
/// subject-level win and loss counts without allocating heap memory.
#[verified_engine::verified]
pub fn compute_unmatched_u_statistics(
    n1: usize,
    n2: usize,
    wins: usize,
    losses: usize,
    m: &SubjectMarginals<'_>,
    alpha: f64,
) -> Option<UnmatchedWinRatioSummary> {
    let valid = n1 >= 2
        && n2 >= 2
        && (wins > 0 || losses > 0)
        && alpha.is_finite()
        && alpha > 0.0
        && alpha < 1.0
        && m.row_wins.len() >= n1
        && m.row_losses.len() >= n1
        && m.col_wins.len() >= n2
        && m.col_losses.len() >= n2;
    if !valid {
        return None;
    }
    let total_pairs = n1.checked_mul(n2)?;
    let untied = wins.checked_add(losses)?;
    if untied > total_pairs {
        return None;
    }
    let ties = total_pairs - untied;
    let total_f = total_pairs as f64;
    let net_benefit = (wins as f64 - losses as f64) / total_f;
    let wo_den = losses as f64 + 0.5 * (ties as f64);
    let win_odds = if wo_den == 0.0 {
        f64::INFINITY
    } else {
        (wins as f64 + 0.5 * (ties as f64)) / wo_den
    };

    let (win_ratio, ci_low, ci_high, log_wr_se, z_score, p_value) = if wins == 0 || losses == 0 {
        let (wr, lo, hi, z, p) = degenerate_inference(n1, n2, wins, losses, ties, net_benefit, m)?;
        (wr, lo, hi, 0.0, z, p)
    } else {
        finite_log_wr_inference(n1, n2, wins, losses, total_f, m, alpha)?
    };

    Some(UnmatchedWinRatioSummary {
        n_treatment: n1,
        n_control: n2,
        total_pairs,
        wins,
        losses,
        ties,
        win_ratio,
        win_odds,
        net_benefit,
        ci_low,
        ci_high,
        log_wr_se,
        z_score,
        p_value,
    })
}

#[verified_engine::verified]
fn finite_log_wr_inference(
    n1: usize,
    n2: usize,
    wins: usize,
    losses: usize,
    total_f: f64,
    m: &SubjectMarginals<'_>,
    alpha: f64,
) -> Option<(f64, f64, f64, f64, f64, f64)> {
    let normal = Normal::new(0.0, 1.0).ok()?;
    let win_ratio = wins as f64 / losses as f64;
    let var_ln_wr =
        accumulate_log_wr_variance(n1, n2, wins as f64 / total_f, losses as f64 / total_f, m);
    let log_wr_se = var_ln_wr.max(0.0).sqrt();
    if log_wr_se == 0.0 {
        let (z, p) = if wins == losses {
            (0.0, 1.0)
        } else if wins > losses {
            (f64::INFINITY, 0.0)
        } else {
            (f64::NEG_INFINITY, 0.0)
        };
        return Some((win_ratio, win_ratio, win_ratio, 0.0, z, p));
    }
    let ln_wr = win_ratio.ln();
    let z_crit = normal.inverse_cdf(1.0 - alpha / 2.0);
    let z = ln_wr / log_wr_se;
    let p = (2.0 * (1.0 - normal.cdf(z.abs()))).clamp(0.0, 1.0);
    Some((
        win_ratio,
        (ln_wr - z_crit * log_wr_se).exp(),
        (ln_wr + z_crit * log_wr_se).exp(),
        log_wr_se,
        z,
        p,
    ))
}

fn reset_unmatched_scratch(scratch: &mut UnmatchedScratch, n1: usize, n2: usize, n_tiers: usize) {
    scratch.row_wins[..n1].fill(0);
    scratch.row_losses[..n1].fill(0);
    scratch.col_wins[..n2].fill(0);
    scratch.col_losses[..n2].fill(0);
    for (k, tier) in scratch.tier_counts.iter_mut().enumerate().take(n_tiers) {
        *tier = TierCount {
            tier_index: k,
            wins: 0,
            losses: 0,
            ties_remaining: 0,
        };
    }
}

impl<T> WinRatioAnalysis<T> {
    /// Evaluates an unmatched all-pairs composite comparison into preallocated scratch buffers
    /// without dynamic heap allocation, returning Pocock/Bebu-Lachin U-statistic inference.
    #[verified_engine::verified]
    pub fn evaluate_unmatched_into<S1, S2>(
        &self,
        group1: &[S1],
        group2: &[S2],
        alpha: f64,
        scratch: &mut UnmatchedScratch,
    ) -> Option<UnmatchedWinRatioSummary>
    where
        S1: AsRef<[T]>,
        S2: AsRef<[T]>,
    {
        let (n1, n2, n_tiers) = (group1.len(), group2.len(), self.num_tiers());
        if scratch.row_wins.len() < n1
            || scratch.row_losses.len() < n1
            || scratch.col_wins.len() < n2
            || scratch.col_losses.len() < n2
            || scratch.tier_counts.len() < n_tiers
        {
            return None;
        }
        reset_unmatched_scratch(scratch, n1, n2, n_tiers);

        let (mut total_wins, mut total_losses) = (0usize, 0usize);
        for (i, subj1) in group1.iter().enumerate() {
            let s1 = subj1.as_ref();
            for (j, subj2) in group2.iter().enumerate() {
                match self.compare_subjects_with_tier(s1, subj2.as_ref()) {
                    (ComparisonResult::Win, Some(k)) => {
                        total_wins += 1;
                        scratch.row_wins[i] += 1;
                        scratch.col_wins[j] += 1;
                        scratch.tier_counts[k].wins += 1;
                    }
                    (ComparisonResult::Loss, Some(k)) => {
                        total_losses += 1;
                        scratch.row_losses[i] += 1;
                        scratch.col_losses[j] += 1;
                        scratch.tier_counts[k].losses += 1;
                    }
                    _ => {}
                }
            }
        }
        let mut remaining = n1.saturating_mul(n2);
        for tier in scratch.tier_counts.iter_mut().take(n_tiers) {
            remaining = remaining.saturating_sub(tier.wins + tier.losses);
            tier.ties_remaining = remaining;
        }
        let marginals = scratch.marginals(n1, n2);
        compute_unmatched_u_statistics(n1, n2, total_wins, total_losses, &marginals, alpha)
    }

    /// Evaluates an unmatched all-pairs composite comparison at significance level `alpha`
    /// (e.g. `0.05` for 95% CIs), returning the U-statistic summary and per-tier counts.
    #[verified_engine::verified]
    pub fn evaluate_unmatched<S1, S2>(
        &self,
        group1: &[S1],
        group2: &[S2],
        alpha: f64,
    ) -> Option<(UnmatchedWinRatioSummary, Vec<TierCount>)>
    where
        S1: AsRef<[T]>,
        S2: AsRef<[T]>,
    {
        let n_tiers = self.num_tiers();
        let mut scratch = UnmatchedScratch::with_capacity(group1.len(), group2.len(), n_tiers);
        let summary = self.evaluate_unmatched_into(group1, group2, alpha, &mut scratch)?;
        scratch.tier_counts.truncate(n_tiers);
        Some((summary, scratch.tier_counts))
    }
}

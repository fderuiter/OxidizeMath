use crate::applied::clinical_trials::types::ClinicalTrialError;
use statrs::distribution::{ContinuousCDF, Normal};
use std::marker::PhantomData;

pub use super::censoring::{CensoredOutcome, RightCensoredComparator, TierCount};
pub use super::u_statistic::{
    SubjectMarginals, UnmatchedScratch, UnmatchedWinRatioSummary, compute_unmatched_u_statistics,
};

/// Per-tier win, loss, and unresolved tie counts in a hierarchical Win Ratio evaluation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TierOutcomeCounts {
    /// Number of pairs won by the treatment group at this tier.
    pub wins: usize,
    /// Number of pairs lost by the treatment group at this tier.
    pub losses: usize,
    /// Number of pairs remaining tied after evaluating this tier.
    pub ties: usize,
}

impl TierOutcomeCounts {
    /// Constructs a new per-tier outcome count record.
    #[verified_engine::verified]
    pub fn new(wins: usize, losses: usize, ties: usize) -> Self {
        Self { wins, losses, ties }
    }

    /// Returns the number of pairs decided (won or lost) at this tier.
    #[verified_engine::verified]
    pub fn decided(&self) -> usize {
        self.wins + self.losses
    }
}

/// Outcome of comparing a treated subject against a control subject at a single endpoint tier.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ComparisonResult {
    /// Treated subject achieved a strictly better outcome than the control subject.
    Win,
    /// Treated subject achieved a strictly worse outcome than the control subject.
    Loss,
    /// Pair could not be distinguished at this tier and advances to the next tier.
    Tie,
}

/// Defines how to compare two outcomes of type T.
pub trait OutcomeComparator<T: ?Sized> {
    /// Compares treated outcome `a` against control outcome `b` at a single tier.
    #[verified_engine::verified]
    fn compare(&self, a: &T, b: &T) -> ComparisonResult;
}

/// Strategy: Higher value is better (e.g., survival time, quality of life score).
///
/// If `a > b`, outcome `a` is a Win.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HigherIsBetter;

impl<T: PartialOrd> OutcomeComparator<T> for HigherIsBetter {
    #[verified_engine::verified]
    fn compare(&self, a: &T, b: &T) -> ComparisonResult {
        if a > b {
            ComparisonResult::Win
        } else if a < b {
            ComparisonResult::Loss
        } else {
            ComparisonResult::Tie
        }
    }
}

/// Strategy: Lower value is better (e.g., number of hospitalizations, symptom severity).
///
/// If `a < b`, outcome `a` is a Win.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LowerIsBetter;

impl<T: PartialOrd> OutcomeComparator<T> for LowerIsBetter {
    #[verified_engine::verified]
    fn compare(&self, a: &T, b: &T) -> ComparisonResult {
        if a < b {
            ComparisonResult::Win
        } else if a > b {
            ComparisonResult::Loss
        } else {
            ComparisonResult::Tie
        }
    }
}

/// Strategy: Higher value is better, but only if the difference exceeds a threshold (`a - b > threshold`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ThresholdComparator {
    /// Minimum indifference margin $\tau \ge 0$ required to declare a win or loss.
    pub threshold: f64,
}

impl ThresholdComparator {
    /// Creates a [`ThresholdComparator`] with indifference margin `threshold`.
    #[verified_engine::verified]
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl OutcomeComparator<f64> for ThresholdComparator {
    #[verified_engine::verified]
    fn compare(&self, a: &f64, b: &f64) -> ComparisonResult {
        if a - b > self.threshold {
            ComparisonResult::Win
        } else if b - a > self.threshold {
            ComparisonResult::Loss
        } else {
            ComparisonResult::Tie
        }
    }
}

/// Analysis context for hierarchical Win Ratio comparisons.
///
/// Allows configuring a specific comparison strategy for each outcome level in the hierarchy.
pub struct WinRatioAnalysis<T> {
    strategies: Vec<Box<dyn OutcomeComparator<T>>>,
    _marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for WinRatioAnalysis<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WinRatioAnalysis")
            .field("tiers", &self.strategies.len())
            .finish()
    }
}

#[verified_engine::verified]
fn validate_tier_inputs(
    strategy_count: usize,
    group1_len: usize,
    group2_len: usize,
    out_len: usize,
) -> Result<(), ClinicalTrialError> {
    if strategy_count == 0 {
        return Err(ClinicalTrialError::InvalidData(
            "At least one outcome comparator tier is required.".to_string(),
        ));
    }
    if group1_len == 0 || group2_len == 0 {
        return Err(ClinicalTrialError::InsufficientSampleSize {
            required: 1,
            actual: group1_len.min(group2_len),
        });
    }
    if out_len < strategy_count {
        return Err(ClinicalTrialError::InvalidData(
            "Tier accumulator slice is smaller than configured tier count.".to_string(),
        ));
    }
    Ok(())
}

/// A strategy for pairing subjects between two groups for comparison.
pub trait PairingStrategy<T> {
    /// Evaluates total `(wins, losses)` across pairs formed between `group1` and `group2`.
    #[verified_engine::verified]
    fn evaluate(
        &self,
        analysis: &WinRatioAnalysis<T>,
        group1: &[Vec<T>],
        group2: &[Vec<T>],
    ) -> (i32, i32);

    /// Evaluates hierarchical per-tier win, loss, and tie counts into a preallocated slice.
    #[verified_engine::verified]
    fn evaluate_tiers_into(
        &self,
        analysis: &WinRatioAnalysis<T>,
        group1: &[Vec<T>],
        group2: &[Vec<T>],
        tier_counts: &mut [TierOutcomeCounts],
    ) -> Result<(), ClinicalTrialError>;
}

/// Unmatched Pair comparison (All-Pairs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UnmatchedPairing;

impl<T> PairingStrategy<T> for UnmatchedPairing {
    #[verified_engine::verified]
    fn evaluate(
        &self,
        analysis: &WinRatioAnalysis<T>,
        group1: &[Vec<T>],
        group2: &[Vec<T>],
    ) -> (i32, i32) {
        let mut wins = 0;
        let mut losses = 0;
        for subj1 in group1 {
            for subj2 in group2 {
                match analysis.compare_subjects(subj1, subj2) {
                    ComparisonResult::Win => wins += 1,
                    ComparisonResult::Loss => losses += 1,
                    ComparisonResult::Tie => (),
                }
            }
        }
        (wins, losses)
    }

    #[verified_engine::verified]
    fn evaluate_tiers_into(
        &self,
        analysis: &WinRatioAnalysis<T>,
        group1: &[Vec<T>],
        group2: &[Vec<T>],
        tier_counts: &mut [TierOutcomeCounts],
    ) -> Result<(), ClinicalTrialError> {
        let n_tiers = analysis.tier_count();
        validate_tier_inputs(n_tiers, group1.len(), group2.len(), tier_counts.len())?;
        tier_counts[..n_tiers].fill(TierOutcomeCounts::default());
        for subj1 in group1 {
            for subj2 in group2 {
                let _ =
                    analysis.compare_subjects_into_tiers(subj1, subj2, &mut tier_counts[..n_tiers]);
            }
        }
        Ok(())
    }
}

/// Matched Pair comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatchedPairing;

impl<T> PairingStrategy<T> for MatchedPairing {
    #[verified_engine::verified]
    fn evaluate(
        &self,
        analysis: &WinRatioAnalysis<T>,
        group1: &[Vec<T>],
        group2: &[Vec<T>],
    ) -> (i32, i32) {
        assert_eq!(
            group1.len(),
            group2.len(),
            "Groups must be of equal length for matched pairs."
        );
        let mut wins = 0;
        let mut losses = 0;
        for (subj1, subj2) in group1.iter().zip(group2.iter()) {
            match analysis.compare_subjects(subj1, subj2) {
                ComparisonResult::Win => wins += 1,
                ComparisonResult::Loss => losses += 1,
                ComparisonResult::Tie => (),
            }
        }
        (wins, losses)
    }

    #[verified_engine::verified]
    fn evaluate_tiers_into(
        &self,
        analysis: &WinRatioAnalysis<T>,
        group1: &[Vec<T>],
        group2: &[Vec<T>],
        tier_counts: &mut [TierOutcomeCounts],
    ) -> Result<(), ClinicalTrialError> {
        let n_tiers = analysis.tier_count();
        validate_tier_inputs(n_tiers, group1.len(), group2.len(), tier_counts.len())?;
        if group1.len() != group2.len() {
            return Err(ClinicalTrialError::InvalidData(format!(
                "Groups must be of equal length for matched pairs, got {} and {}.",
                group1.len(),
                group2.len()
            )));
        }
        tier_counts[..n_tiers].fill(TierOutcomeCounts::default());
        for (subj1, subj2) in group1.iter().zip(group2.iter()) {
            let _ = analysis.compare_subjects_into_tiers(subj1, subj2, &mut tier_counts[..n_tiers]);
        }
        Ok(())
    }
}

impl<T> WinRatioAnalysis<T> {
    /// Creates a new analysis with no strategies.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Returns the number of configured hierarchical comparison tiers.
    #[verified_engine::verified]
    pub fn tier_count(&self) -> usize {
        self.strategies.len()
    }

    /// Adds a comparison strategy for the next outcome level.
    #[verified_engine::verified]
    pub fn add_strategy(mut self, strategy: Box<dyn OutcomeComparator<T>>) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Returns the number of configured priority tiers in this analysis.
    #[verified_engine::verified]
    #[must_use]
    pub fn num_tiers(&self) -> usize {
        self.tier_count()
    }

    /// Compares two subjects tier-by-tier, returning both the decision and the resolving tier index.
    #[verified_engine::verified]
    pub fn compare_subjects_with_tier(
        &self,
        subject1: &[T],
        subject2: &[T],
    ) -> (ComparisonResult, Option<usize>) {
        let limit = self
            .strategies
            .len()
            .min(subject1.len())
            .min(subject2.len());

        for i in 0..limit {
            let res = self.strategies[i].compare(&subject1[i], &subject2[i]);
            if res != ComparisonResult::Tie {
                return (res, Some(i));
            }
        }
        (ComparisonResult::Tie, None)
    }

    /// Compares two subjects outcome-by-outcome using the configured strategies.
    #[verified_engine::verified]
    pub fn compare_subjects(&self, subject1: &[T], subject2: &[T]) -> ComparisonResult {
        self.compare_subjects_with_tier(subject1, subject2).0
    }

    /// Compares two subjects hierarchically while recording per-tier win, loss, and tie counts.
    #[verified_engine::verified]
    pub fn compare_subjects_into_tiers(
        &self,
        subject1: &[T],
        subject2: &[T],
        tier_counts: &mut [TierOutcomeCounts],
    ) -> ComparisonResult {
        let n_tiers = self.strategies.len().min(tier_counts.len());
        for (i, counts) in tier_counts.iter_mut().enumerate().take(n_tiers) {
            if i >= subject1.len() || i >= subject2.len() {
                counts.ties += 1;
                continue;
            }
            let res = self.strategies[i].compare(&subject1[i], &subject2[i]);
            match res {
                ComparisonResult::Win => {
                    counts.wins += 1;
                    return ComparisonResult::Win;
                }
                ComparisonResult::Loss => {
                    counts.losses += 1;
                    return ComparisonResult::Loss;
                }
                ComparisonResult::Tie => {
                    counts.ties += 1;
                }
            }
        }
        ComparisonResult::Tie
    }

    /// Evaluates pairs using the provided pairing strategy.
    #[verified_engine::verified]
    pub fn evaluate_pairs<P: PairingStrategy<T>>(
        &self,
        group1: &[Vec<T>],
        group2: &[Vec<T>],
        strategy: &P,
    ) -> (i32, i32) {
        strategy.evaluate(self, group1, group2)
    }

    /// Evaluates per-tier win, loss, and tie counts into a caller-supplied preallocated slice.
    #[verified_engine::verified]
    pub fn evaluate_tiers_into<P: PairingStrategy<T>>(
        &self,
        group1: &[Vec<T>],
        group2: &[Vec<T>],
        strategy: &P,
        tier_counts: &mut [TierOutcomeCounts],
    ) -> Result<(), ClinicalTrialError> {
        strategy.evaluate_tiers_into(self, group1, group2, tier_counts)
    }

    /// Evaluates fallible per-tier win, loss, and tie waterfall counts across two subject groups.
    #[verified_engine::verified]
    pub fn evaluate_tiers<P: PairingStrategy<T>>(
        &self,
        group1: &[Vec<T>],
        group2: &[Vec<T>],
        strategy: &P,
    ) -> Result<Vec<TierOutcomeCounts>, ClinicalTrialError> {
        let mut counts = vec![TierOutcomeCounts::default(); self.strategies.len()];
        self.evaluate_tiers_into(group1, group2, strategy, &mut counts)?;
        Ok(counts)
    }
}

impl<T: PartialOrd + 'static> Default for WinRatioAnalysis<T> {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new()
    }
}

/// Calculates the raw Win Ratio `WR = N_wins / N_losses` (`f64::INFINITY` if `losses == 0`).
#[verified_engine::verified]
pub fn calculate_win_ratio(wins: i32, losses: i32) -> f64 {
    if losses == 0 {
        f64::INFINITY
    } else {
        wins as f64 / losses as f64
    }
}

/// Statistical results for a Win Ratio analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WinRatioStats {
    /// The calculated win ratio ($N_W / N_L$).
    pub win_ratio: f64,
    /// Lower bound of the 95% Confidence Interval.
    pub ci_low: f64,
    /// Upper bound of the 95% Confidence Interval.
    pub ci_high: f64,
    /// Two-sided P-value testing the null hypothesis that $WR = 1$.
    pub p_value: f64,
}

/// Calculates 95% Confidence Intervals and P-values for the Win Ratio (`None` if total pairs is zero).
///
/// Uses the normal approximation with standard error $SE = \sqrt{p(1-p)/N}$ where $p$ is the win proportion.
#[verified_engine::verified]
pub fn calculate_statistics(wins: i32, losses: i32) -> Option<WinRatioStats> {
    let total_pairs = wins + losses;
    if total_pairs == 0 {
        return None;
    }

    let p_win = wins as f64 / total_pairs as f64;
    let se = (p_win * (1.0 - p_win) / total_pairs as f64).sqrt();

    if se == 0.0 {
        let win_ratio = calculate_win_ratio(wins, losses);
        return Some(WinRatioStats {
            win_ratio,
            ci_low: if p_win == 1.0 { win_ratio } else { 0.0 },
            ci_high: if p_win == 1.0 {
                f64::INFINITY
            } else {
                win_ratio
            },
            p_value: 0.0,
        });
    }

    let ci_low_p = (p_win - 1.96 * se).max(0.0);
    let ci_high_p = (p_win + 1.96 * se).min(1.0);

    let ci_low = if ci_low_p == 1.0 {
        f64::INFINITY
    } else {
        ci_low_p / (1.0 - ci_low_p)
    };

    let ci_high = if ci_high_p == 1.0 {
        f64::INFINITY
    } else {
        ci_high_p / (1.0 - ci_high_p)
    };

    let z_score = (p_win - 0.5) / se;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = 2.0 * (1.0 - normal.cdf(z_score.abs()));

    Some(WinRatioStats {
        win_ratio: calculate_win_ratio(wins, losses),
        ci_low,
        ci_high,
        p_value,
    })
}

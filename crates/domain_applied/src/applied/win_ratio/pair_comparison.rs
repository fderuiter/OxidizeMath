use statrs::distribution::{ContinuousCDF, Normal};
use std::marker::PhantomData;

#[derive(Debug, PartialEq, Clone, Copy)]
#[allow(missing_docs)]
pub enum ComparisonResult {
    #[allow(missing_docs)]
    Win,
    #[allow(missing_docs)]
    Loss,
    #[allow(missing_docs)]
    Tie,
}

/// Defines how to compare two outcomes of type T.
pub trait OutcomeComparator<T: ?Sized> {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn compare(&self, a: &T, b: &T) -> ComparisonResult;
}

/// Strategy: Higher value is better (e.g., survival time, quality of life score).
///
/// If `a > b`, outcome `a` is a Win.
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

/// Strategy: Higher value is better, but only if the difference exceeds a threshold.
///
/// If `a - b > threshold`, outcome `a` is a Win.
/// This is useful when small differences are considered clinically negligible (ties).
///
/// Note: This strategy requires `T` to be `f64` (or convertible).
pub struct ThresholdComparator {
    #[allow(missing_docs)]
    pub threshold: f64,
}

impl ThresholdComparator {
    #[allow(missing_docs)]
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

/// A strategy for pairing subjects between two groups for comparison.
pub trait PairingStrategy<T> {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn evaluate(
        &self,
        analysis: &WinRatioAnalysis<T>,
        group1: &[Vec<T>],
        group2: &[Vec<T>],
    ) -> (i32, i32);
}

/// Unmatched Pair comparison (All-Pairs).
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
}

/// Matched Pair comparison.
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

    /// Adds a comparison strategy for the next outcome level.
    #[verified_engine::verified]
    pub fn add_strategy(mut self, strategy: Box<dyn OutcomeComparator<T>>) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Compares two subjects outcome-by-outcome using the configured strategies.
    #[verified_engine::verified]
    pub fn compare_subjects(&self, subject1: &[T], subject2: &[T]) -> ComparisonResult {
        // Zip the outcomes with the strategies.
        // If there are more outcomes than strategies, the extra outcomes are ignored (or we could panic/error).
        // If there are fewer outcomes, we stop early.
        let limit = self
            .strategies
            .len()
            .min(subject1.len())
            .min(subject2.len());

        for i in 0..limit {
            let res = self.strategies[i].compare(&subject1[i], &subject2[i]);
            if res != ComparisonResult::Tie {
                return res;
            }
        }
        ComparisonResult::Tie
    }

    /// Evaluates pairs using the provided pairing strategy.
    pub fn evaluate_pairs<P: PairingStrategy<T>>(
        &self,
        group1: &[Vec<T>],
        group2: &[Vec<T>],
        strategy: &P,
    ) -> (i32, i32) {
        strategy.evaluate(self, group1, group2)
    }
}

impl<T: PartialOrd + 'static> Default for WinRatioAnalysis<T> {
    #[verified_engine::verified]
    fn default() -> Self {
        // Default behavior mimics the old `compare_outcomes`: HigherIsBetter for everything.
        // However, since we don't know how many levels there are, we can't pre-populate.
        // So the default behavior here is technically "No comparison".
        // To truly mimic `compare_outcomes`, we need a strategy that applies to ALL levels dynamically,
        // but our struct stores a `Vec` of strategies per level.
        // So `Default` here is just an empty analysis.
        Self::new()
    }
}

/// Calculates the raw Win Ratio.
///
/// $$ WR = \frac{N_{wins}}{N_{losses}} $$
///
/// Returns `f64::INFINITY` if losses are zero.
#[verified_engine::verified]
pub fn calculate_win_ratio(wins: i32, losses: i32) -> f64 {
    if losses == 0 {
        f64::INFINITY
    } else {
        wins as f64 / losses as f64
    }
}

/// Statistical results for a Win Ratio analysis.
#[derive(Debug, Clone, Copy)]
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

/// Calculates 95% Confidence Intervals and P-values for the Win Ratio.
///
/// Uses the normal approximation for the log-win-ratio.
///
/// # Returns
///
/// `None` if the total number of pairs is zero.
///
/// # Mathematical Details
///
/// The standard error is estimated as:
/// $$ SE = \sqrt{\frac{p(1-p)}{N}} $$
/// Where $p$ is the proportion of wins among untied pairs (simplified).
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

//! Preallocated zero-allocation scratch buffers and flat-stride evaluators for `WinRatioAnalysis`.

use super::pair_comparison::{
    ComparisonResult, MatchedPairing, UnmatchedPairing, WinRatioAnalysis, WinRatioStats,
    calculate_statistics,
};
use math_commons::math_kernel::types::flatten_2d_index;

/// Hierarchical win/loss/tie breakdown across two prioritized clinical endpoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WinRatioBreakdown {
    /// Wins adjudicated on Tier 1 (e.g., fatal survival).
    pub tier1_wins: i32,
    /// Losses adjudicated on Tier 1 (e.g., fatal survival).
    pub tier1_losses: i32,
    /// Wins adjudicated on Tier 2 (e.g., non-fatal hospitalization) after Tier-1 ties.
    pub tier2_wins: i32,
    /// Losses adjudicated on Tier 2 (e.g., non-fatal hospitalization) after Tier-1 ties.
    pub tier2_losses: i32,
    /// Pairs remaining tied across all hierarchy levels.
    pub ties: i32,
    /// Asymptotic Win Ratio confidence interval and two-sided p-value.
    pub stats: Option<WinRatioStats>,
}

impl WinRatioBreakdown {
    /// Total wins across all hierarchy tiers ($N_W = N_{W,1} + N_{W,2}$).
    #[verified_engine::verified]
    pub fn total_wins(&self) -> i32 {
        self.tier1_wins + self.tier2_wins
    }

    /// Total losses across all hierarchy tiers ($N_L = N_{L,1} + N_{L,2}$).
    #[verified_engine::verified]
    pub fn total_losses(&self) -> i32 {
        self.tier1_losses + self.tier2_losses
    }
}

impl<T> WinRatioAnalysis<T> {
    /// Evaluates all-pairs unmatched Win Ratio on contiguous flat-stride slices (`n_levels` per subject)
    /// without allocating per-subject `Vec<T>` buffers.
    #[verified_engine::verified]
    pub fn evaluate_flat_unmatched(
        &self,
        group1_flat: &[T],
        group2_flat: &[T],
        n_levels: usize,
    ) -> (i32, i32) {
        let _ = UnmatchedPairing;
        if n_levels == 0 {
            return (0, 0);
        }
        let mut wins = 0;
        let mut losses = 0;
        for subj1 in group1_flat.chunks_exact(n_levels) {
            for subj2 in group2_flat.chunks_exact(n_levels) {
                match self.compare_subjects(subj1, subj2) {
                    ComparisonResult::Win => wins += 1,
                    ComparisonResult::Loss => losses += 1,
                    ComparisonResult::Tie => {}
                }
            }
        }
        (wins, losses)
    }

    /// Evaluates 1:1 matched-pair Win Ratio on contiguous flat-stride slices (`n_levels` per subject)
    /// without allocating per-subject `Vec<T>` buffers.
    #[verified_engine::verified]
    pub fn evaluate_flat_matched(
        &self,
        group1_flat: &[T],
        group2_flat: &[T],
        n_levels: usize,
    ) -> (i32, i32) {
        let _ = MatchedPairing;
        if n_levels == 0 {
            return (0, 0);
        }
        let mut wins = 0;
        let mut losses = 0;
        for (subj1, subj2) in group1_flat
            .chunks_exact(n_levels)
            .zip(group2_flat.chunks_exact(n_levels))
        {
            match self.compare_subjects(subj1, subj2) {
                ComparisonResult::Win => wins += 1,
                ComparisonResult::Loss => losses += 1,
                ComparisonResult::Tie => {}
            }
        }
        (wins, losses)
    }
}

/// Preallocated workspace for zero-allocation hierarchical Win Ratio and 2D reversal regime evaluation.
#[derive(Debug, Clone)]
pub struct WinRatioScratch {
    max_subjects_per_arm: usize,
    max_cols: usize,
    max_rows: usize,
    treatment_flat: Vec<f64>,
    control_flat: Vec<f64>,
    wr_grid: Vec<f64>,
    hr_grid: Vec<f64>,
}

impl WinRatioScratch {
    /// Preallocates two-tier subject buffers (`2 * max_subjects_per_arm` each) and a 2D `max_cols * max_rows` regime grid.
    #[verified_engine::verified]
    pub fn new(max_subjects_per_arm: usize, max_cols: usize, max_rows: usize) -> Self {
        let subj_cap = max_subjects_per_arm.saturating_mul(2);
        let grid_cap = max_cols.saturating_mul(max_rows);
        Self {
            max_subjects_per_arm,
            max_cols,
            max_rows,
            treatment_flat: vec![0.0; subj_cap],
            control_flat: vec![0.0; subj_cap],
            wr_grid: vec![1.0; grid_cap],
            hr_grid: vec![1.0; grid_cap],
        }
    }

    /// Mutable view of preallocated flat two-tier `[tier1, tier2, ...]` treatment outcomes (`2 * n_treat`).
    #[verified_engine::verified]
    pub fn treatment_two_tier_mut(&mut self, n_treat: usize) -> &mut [f64] {
        let len = n_treat.min(self.max_subjects_per_arm) * 2;
        &mut self.treatment_flat[..len]
    }

    /// Mutable view of preallocated flat two-tier `[tier1, tier2, ...]` control outcomes (`2 * n_ctrl`).
    #[verified_engine::verified]
    pub fn control_two_tier_mut(&mut self, n_ctrl: usize) -> &mut [f64] {
        let len = n_ctrl.min(self.max_subjects_per_arm) * 2;
        &mut self.control_flat[..len]
    }

    /// Immutable view of flat two-tier treatment outcomes (`2 * n_treat`).
    #[verified_engine::verified]
    pub fn treatment_two_tier(&self, n_treat: usize) -> &[f64] {
        let len = n_treat.min(self.max_subjects_per_arm) * 2;
        &self.treatment_flat[..len]
    }

    /// Immutable view of flat two-tier control outcomes (`2 * n_ctrl`).
    #[verified_engine::verified]
    pub fn control_two_tier(&self, n_ctrl: usize) -> &[f64] {
        let len = n_ctrl.min(self.max_subjects_per_arm) * 2;
        &self.control_flat[..len]
    }

    /// Evaluates prioritized two-tier (`HigherIsBetter`) breakdown on the currently staged treatment/control slices.
    #[verified_engine::verified]
    pub fn evaluate_staged_two_tier(
        &self,
        n_treat: usize,
        n_ctrl: usize,
        matched: bool,
    ) -> WinRatioBreakdown {
        let t_slice = self.treatment_two_tier(n_treat);
        let c_slice = self.control_two_tier(n_ctrl);
        evaluate_two_tier_slices(t_slice, c_slice, matched)
    }

    /// Stores a `(win_ratio, hazard_ratio)` pair into the preallocated 2D regime grid at `(col, row)`.
    #[verified_engine::verified]
    pub fn set_regime_cell(&mut self, col: usize, row: usize, win_ratio: f64, hazard_ratio: f64) {
        if col < self.max_cols && row < self.max_rows {
            let idx = flatten_2d_index(col, row, self.max_cols);
            self.wr_grid[idx] = win_ratio;
            self.hr_grid[idx] = hazard_ratio;
        }
    }

    /// Reads `(win_ratio, hazard_ratio)` at `(col, row)`.
    #[verified_engine::verified]
    pub fn regime_cell(&self, col: usize, row: usize) -> (f64, f64) {
        let idx = flatten_2d_index(
            col.min(self.max_cols.saturating_sub(1)),
            row.min(self.max_rows.saturating_sub(1)),
            self.max_cols,
        );
        (self.wr_grid[idx], self.hr_grid[idx])
    }
}

fn evaluate_two_tier_slices(
    treat_flat: &[f64],
    ctrl_flat: &[f64],
    matched: bool,
) -> WinRatioBreakdown {
    let mut tier1_wins = 0;
    let mut tier1_losses = 0;
    let mut tier2_wins = 0;
    let mut tier2_losses = 0;
    let mut ties = 0;

    let treat_pairs = treat_flat.as_chunks::<2>().0;
    let ctrl_pairs = ctrl_flat.as_chunks::<2>().0;
    if matched {
        for (t, c) in treat_pairs.iter().zip(ctrl_pairs.iter()) {
            accumulate_two_tier_pair(
                t,
                c,
                &mut tier1_wins,
                &mut tier1_losses,
                &mut tier2_wins,
                &mut tier2_losses,
                &mut ties,
            );
        }
    } else {
        for t in treat_pairs {
            for c in ctrl_pairs {
                accumulate_two_tier_pair(
                    t,
                    c,
                    &mut tier1_wins,
                    &mut tier1_losses,
                    &mut tier2_wins,
                    &mut tier2_losses,
                    &mut ties,
                );
            }
        }
    }

    let stats = calculate_statistics(tier1_wins + tier2_wins, tier1_losses + tier2_losses);
    WinRatioBreakdown {
        tier1_wins,
        tier1_losses,
        tier2_wins,
        tier2_losses,
        ties,
        stats,
    }
}

#[inline]
fn accumulate_two_tier_pair(
    t: &[f64],
    c: &[f64],
    t1_w: &mut i32,
    t1_l: &mut i32,
    t2_w: &mut i32,
    t2_l: &mut i32,
    ties: &mut i32,
) {
    if t[0] > c[0] {
        *t1_w += 1;
    } else if t[0] < c[0] {
        *t1_l += 1;
    } else if t[1] > c[1] {
        *t2_w += 1;
    } else if t[1] < c[1] {
        *t2_l += 1;
    } else {
        *ties += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::pair_comparison::{HigherIsBetter, UnmatchedPairing};
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_win_ratio_scratch_matches_analysis_with_zero_alloc() {
        let analysis = WinRatioAnalysis::new()
            .add_strategy(Box::new(HigherIsBetter))
            .add_strategy(Box::new(HigherIsBetter));

        let treat_vecs = vec![vec![36.0, 4.0], vec![36.0, 36.0], vec![28.0, 3.0]];
        let ctrl_vecs = vec![vec![8.0, 8.0], vec![10.0, 10.0], vec![36.0, 36.0]];
        let expected_unmatched =
            analysis.evaluate_pairs(&treat_vecs, &ctrl_vecs, &UnmatchedPairing);
        let expected_matched = analysis.evaluate_pairs(&treat_vecs, &ctrl_vecs, &MatchedPairing);

        let mut scratch = WinRatioScratch::new(20, 4, 4);
        let ((flat_unmatched, flat_matched, breakdown), _cert) =
            verified_engine::engine::VerifiedEngine::run_verified(|| {
                let t_buf = scratch.treatment_two_tier_mut(3);
                t_buf.copy_from_slice(&[36.0, 4.0, 36.0, 36.0, 28.0, 3.0]);
                let c_buf = scratch.control_two_tier_mut(3);
                c_buf.copy_from_slice(&[8.0, 8.0, 10.0, 10.0, 36.0, 36.0]);

                let u = analysis.evaluate_flat_unmatched(
                    scratch.treatment_two_tier(3),
                    scratch.control_two_tier(3),
                    2,
                );
                let m = analysis.evaluate_flat_matched(
                    scratch.treatment_two_tier(3),
                    scratch.control_two_tier(3),
                    2,
                );
                let b = scratch.evaluate_staged_two_tier(3, 3, false);
                scratch.set_regime_cell(1, 2, b.stats.unwrap().win_ratio, 3.86);
                (u, m, b)
            });

        assert_eq!(flat_unmatched, expected_unmatched);
        assert_eq!(flat_matched, expected_matched);
        assert_eq!(
            (breakdown.total_wins(), breakdown.total_losses()),
            expected_unmatched
        );
        assert_relative_eq!(scratch.regime_cell(1, 2).1, 3.86, epsilon = 1e-12);
    }
}

// [cite:win_ratio]

//! Lockstep counterfactual twin slice budgeting (`4ms + 4ms` inside `8ms` WASM frame cap).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::time::Duration;

/// Cooperative frame budget (`8ms`) per browser animation tick on `wasm32`.
pub const WASM_FRAME_SLICE_BUDGET: Duration = Duration::from_millis(8);

/// Per-branch slice budget (`4ms`) inside a `TwinRunner` pair (`4ms + 4ms = 8ms`).
pub const WASM_TWIN_BRANCH_SLICE_BUDGET: Duration = Duration::from_millis(4);

/// Tracks split `4ms + 4ms` execution budgets across `Branch A` and `Branch B`
/// inside a single cooperative simulation worker task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwinSliceBudget {
    total_budget: Duration,
    branch_budget: Duration,
    elapsed_a: Duration,
    elapsed_b: Duration,
    completed_pairs: usize,
}

impl TwinSliceBudget {
    /// Standard `wasm32` `8ms` total slice split evenly into `4ms + 4ms`.
    #[must_use]
    pub const fn wasm_default() -> Self {
        Self {
            total_budget: WASM_FRAME_SLICE_BUDGET,
            branch_budget: WASM_TWIN_BRANCH_SLICE_BUDGET,
            elapsed_a: Duration::ZERO,
            elapsed_b: Duration::ZERO,
            completed_pairs: 0,
        }
    }

    /// Splits an arbitrary cooperative frame `total_budget` into two equal half-budgets.
    #[must_use]
    pub fn from_total(total_budget: Duration) -> Self {
        Self {
            total_budget,
            branch_budget: total_budget / 2,
            elapsed_a: Duration::ZERO,
            elapsed_b: Duration::ZERO,
            completed_pairs: 0,
        }
    }

    /// Total slice cap (`8ms` on `wasm32`).
    #[must_use]
    pub const fn total_budget(&self) -> Duration {
        self.total_budget
    }

    /// Half-slice cap (`4ms` on `wasm32`) allocated to each twin branch.
    #[must_use]
    pub const fn branch_budget(&self) -> Duration {
        self.branch_budget
    }

    /// Cumulative time spent stepping `Branch A` in the current slice.
    #[must_use]
    pub const fn elapsed_branch_a(&self) -> Duration {
        self.elapsed_a
    }

    /// Cumulative time spent stepping `Branch B` in the current slice.
    #[must_use]
    pub const fn elapsed_branch_b(&self) -> Duration {
        self.elapsed_b
    }

    /// Number of lockstep `(Branch A, Branch B)` step pairs completed in this slice.
    #[must_use]
    pub const fn completed_pairs(&self) -> usize {
        self.completed_pairs
    }

    /// Returns `true` when another lockstep pair may execute without exceeding either
    /// the `4ms` per-branch cap or the `8ms` combined frame budget (always allowing pair `0`).
    #[must_use]
    pub fn can_advance_pair(&self, total_elapsed: Duration) -> bool {
        if self.completed_pairs == 0 {
            return true;
        }
        self.elapsed_a < self.branch_budget
            && self.elapsed_b < self.branch_budget
            && (self.elapsed_a + self.elapsed_b) < self.total_budget
            && total_elapsed < self.total_budget
    }

    /// Records the measured execution durations of one lockstep `(Branch A, Branch B)` step pair.
    pub fn record_pair(&mut self, branch_a: Duration, branch_b: Duration) {
        self.elapsed_a = self.elapsed_a.saturating_add(branch_a);
        self.elapsed_b = self.elapsed_b.saturating_add(branch_b);
        self.completed_pairs = self.completed_pairs.saturating_add(1);
    }
}

/// Lockstep coordinator pairing an unperturbed baseline (`branch_a`) and active (`branch_b`) state.
#[derive(Clone, Debug)]
pub struct TwinRunner<T> {
    /// Frozen-parameter baseline shadow branch (`Branch A`).
    pub branch_a: Option<T>,
    /// Active perturbed branch (`Branch B`).
    pub branch_b: T,
}

impl<T> TwinRunner<T> {
    /// Wraps an initial un-forked primary simulation state (`Branch B`).
    #[must_use]
    pub const fn new(branch_b: T) -> Self {
        Self {
            branch_a: None,
            branch_b,
        }
    }

    /// Returns `true` when a counterfactual shadow twin (`Branch A`) is active.
    #[must_use]
    pub const fn is_forked(&self) -> bool {
        self.branch_a.is_some()
    }

    /// Swaps `Branch A` and `Branch B` in place when forked.
    pub fn swap_branches(&mut self) -> bool {
        if let Some(ref mut a) = self.branch_a {
            std::mem::swap(a, &mut self.branch_b);
            true
        } else {
            false
        }
    }

    /// Clears the shadow twin (`Branch A`), keeping `Branch B`.
    pub fn clear_fork(&mut self) {
        self.branch_a = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_twin_slice_budget_splits_8ms_as_4ms_plus_4ms() {
        let mut budget = TwinSliceBudget::wasm_default();
        assert_eq!(budget.total_budget(), Duration::from_millis(8));
        assert_eq!(budget.branch_budget(), Duration::from_millis(4));
        assert!(budget.can_advance_pair(Duration::ZERO));

        budget.record_pair(Duration::from_millis(2), Duration::from_millis(2));
        assert_eq!(budget.completed_pairs(), 1);
        assert!(budget.can_advance_pair(Duration::from_millis(4)));

        budget.record_pair(Duration::from_millis(2), Duration::from_millis(1));
        assert_eq!(budget.completed_pairs(), 2);
        assert!(
            !budget.can_advance_pair(Duration::from_millis(7)),
            "Branch A reached its 4ms half-slice cap so cooperative worker must yield"
        );
    }
}

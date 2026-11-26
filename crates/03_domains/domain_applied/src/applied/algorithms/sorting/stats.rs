use std::fmt::Debug;

/// Tracks the operational cost of a sorting algorithm.
///
/// This structure allows for empirical verification of theoretical bounds.
#[derive(Debug, Clone, Default)]
pub struct SortingStats {
    /// Number of comparisons performed (e.g., `A[i] > A[j]`).
    pub comparisons: u64,
    /// Number of swaps performed.
    pub swaps: u64,
    /// Number of array writes/assignments (for non-swapping sorts like Merge Sort).
    pub assignments: u64,
}

/// Result of a sorting operation.
pub struct SortingResult<T> {
    /// The sorted data.
    pub sorted_data: Vec<T>,
    /// The statistics collected during the sort.
    pub stats: SortingStats,
}

//! # Sorting Algorithms
//!
//! A collection of classic and specialized sorting algorithms, instrumented to provide
//! operational statistics (comparisons, swaps, assignments).
//!
//! This module is designed not just to sort data, but to *teach* and *analyze* sorting behavior.
//! All algorithms return a [`SortingResult`] containing the sorted data and a [`SortingStats`]
//! struct detailed performance metrics.
//!
//! ## Algorithm Comparison
//!
//! | Algorithm | Best Time | Avg Time | Worst Time | Space | Stable? | Constraints |
//! |-----------|-----------|----------|------------|-------|---------|-------------|
//! | [`bubble_sort`] | O(n) | O(n^2) | O(n^2) | O(1) | Yes | None |
//! | [`insertion_sort`] | O(n) | O(n^2) | O(n^2) | O(1) | Yes | None |
//! | [`heap_sort`] | O(n log n) | O(n log n) | O(n log n) | O(1) | No | None |
//! | [`merge_sort`] | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | None |
//! | [`quick_sort`] | O(n log n) | O(n log n) | O(n^2) | O(log n) | No | None |
//! | [`radix_sort`] | O(nk) | O(nk) | O(nk) | O(n+k) | Yes | `u64` only |
//!
//! *Note: k is the number of digits/bits.*
//!
//! ## Example
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::algorithms::sorting::{merge_sort, SortingResult};
//!
//! let data = vec![5, 2, 9, 1, 5, 6];
//!
//! // Perform the sort
//! let result: SortingResult<i32> = merge_sort(&data);
//!
//! assert_eq!(result.sorted_data, vec![1, 2, 5, 5, 6, 9]);
//!
//! // Inspect the cost
//! println!("Comparisons: {}", result.stats.comparisons);
//! println!("Assignments: {}", result.stats.assignments);
//! ```

#[allow(missing_docs)]
pub mod divide_conquer;
#[allow(missing_docs)]
pub mod elementary;
#[allow(missing_docs)]
pub mod heap;
#[allow(missing_docs)]
pub mod linear;
#[allow(missing_docs)]
pub mod stats;
#[allow(missing_docs)]
pub mod strategy;
#[allow(missing_docs)]
pub mod theory;

pub use divide_conquer::{MergeSorter, QuickSorter, merge_sort, quick_sort};
pub use elementary::{BubbleSorter, InsertionSorter, bubble_sort, insertion_sort};
pub use heap::{HeapSorter, heap_sort};
pub use linear::{RadixSorter, radix_sort};
pub use stats::{SortingResult, SortingStats};
pub use strategy::Sorter;
pub use theory::{information_theoretic_bound, stirling_approximation_ln_factorial};

// [cite:algorithmic_information_rust]

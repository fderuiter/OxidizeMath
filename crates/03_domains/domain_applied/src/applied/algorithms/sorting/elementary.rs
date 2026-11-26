use super::stats::{SortingResult, SortingStats};
use super::strategy::Sorter;

/// Strategy implementation for Bubble Sort.
#[derive(Debug, Default, Clone, Copy)]
pub struct BubbleSorter;

impl<T: Ord + Clone> Sorter<T> for BubbleSorter {
    #[verified_engine::verified]
    fn sort(&self, data: &[T]) -> SortingResult<T> {
        let mut sorted_data = data.to_vec();
        let mut stats = SortingStats::default();
        let n = sorted_data.len();

        if n < 2 {
            return SortingResult { sorted_data, stats };
        }

        for i in 0..n {
            let mut swapped = false;
            // Optimization: the last i elements are already in place
            for j in 0..n - 1 - i {
                stats.comparisons += 1;
                if sorted_data[j] > sorted_data[j + 1] {
                    sorted_data.swap(j, j + 1);
                    stats.swaps += 1;
                    swapped = true;
                }
            }
            if !swapped {
                break;
            }
        }

        SortingResult { sorted_data, stats }
    }
}

/// Bubble Sort
///
/// Repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order.
///
/// # Mathematical Analysis
///
/// * **Comparisons**: $\sum_{i=1}^{n-1} i = \frac{n(n-1)}{2} = \Theta(n^2)$.
/// * **Swaps (Worst Case)**: Reverse order implies every comparison leads to a swap: $\frac{n(n-1)}{2} = \Theta(n^2)$.
/// * **Swaps (Best Case)**: Sorted order implies 0 swaps.
/// * **Stability**: Yes.
#[verified_engine::verified]
pub fn bubble_sort<T: Ord + Clone>(data: &[T]) -> SortingResult<T> {
    BubbleSorter.sort(data)
}

/// Strategy implementation for Insertion Sort.
#[derive(Debug, Default, Clone, Copy)]
pub struct InsertionSorter;

impl<T: Ord + Clone> Sorter<T> for InsertionSorter {
    #[verified_engine::verified]
    fn sort(&self, data: &[T]) -> SortingResult<T> {
        let mut sorted_data = data.to_vec();
        let mut stats = SortingStats::default();
        let n = sorted_data.len();

        for i in 1..n {
            let mut j = i;

            // ⚡ Bolt Optimization:
            // Replaced the `.clone()` based shift with `swap`.
            // For heap-allocated types (e.g. String), cloning inside this hot loop causes severe allocation overhead.
            // Using swap fulfills the 'ping-pong' memory strategy, ensuring zero allocations.
            while j > 0 {
                stats.comparisons += 1;
                if sorted_data[j - 1] > sorted_data[j] {
                    sorted_data.swap(j - 1, j);
                    stats.swaps += 1;
                    j -= 1;
                } else {
                    break;
                }
            }
        }

        SortingResult { sorted_data, stats }
    }
}

/// Insertion Sort
///
/// Builds the final sorted array one item at a time.
///
/// # Mathematical Analysis
///
/// * **Worst Case**: Reverse order. To insert the k-th element, we shift k-1 elements.
///   Total time: $\sum_{k=2}^{n} (k-1) = \Theta(n^2)$.
/// * **Best Case**: Sorted order. 1 comparison per element, 0 swaps. $\Theta(n)$.
/// * **Stability**: Yes.
#[verified_engine::verified]
pub fn insertion_sort<T: Ord + Clone>(data: &[T]) -> SortingResult<T> {
    InsertionSorter.sort(data)
}

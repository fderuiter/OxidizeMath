use super::stats::{SortingResult, SortingStats};
use super::strategy::Sorter;

/// Strategy implementation for Merge Sort.
#[derive(Debug, Default, Clone, Copy)]
pub struct MergeSorter;

impl<T: Ord + Clone> Sorter<T> for MergeSorter {
    #[verified_engine::verified]
    fn sort(&self, data: &[T]) -> SortingResult<T> {
        let mut stats = SortingStats::default();
        let mut sorted_data = data.to_vec();
        let n = sorted_data.len();

        if n > 1 {
            let mut temp = sorted_data.clone();
            let mut width = 1;
            let mut in_temp = false;

            while width < n {
                let mut i = 0;
                while i < n {
                    let left = i;
                    let mid = std::cmp::min(i + width, n);
                    let right = std::cmp::min(i + 2 * width, n);

                    if in_temp {
                        merge(&mut sorted_data, &mut temp, left, mid, right, &mut stats);
                    } else {
                        merge(&mut temp, &mut sorted_data, left, mid, right, &mut stats);
                    }
                    i += 2 * width;
                }
                width *= 2;
                in_temp = !in_temp;
            }

            if in_temp {
                sorted_data = temp;
            }
        }

        SortingResult { sorted_data, stats }
    }
}

/// Merge Sort
///
/// Divides the array into two halves, recursively sorts them, and merges the sorted halves.
///
/// # Mathematical Analysis
///
/// **Recurrence Relation**: $T(n) = 2T(n/2) + \Theta(n)$.
///
/// Using the **Master Theorem** for $T(n) = aT(n/b) + f(n)$:
/// * $a = 2$ (subproblems)
/// * $b = 2$ (factor of reduction)
/// * $f(n) = n$ (merge cost)
/// * $\log_b a = \log_2 2 = 1$.
/// * Since $f(n) = \Theta(n^{\log_b a})$, we are in Case 2.
/// * Result: $T(n) = \Theta(n \log n)$.
///
/// **Space Complexity**: $O(n)$ auxiliary space.
#[verified_engine::verified]
pub fn merge_sort<T: Ord + Clone>(data: &[T]) -> SortingResult<T> {
    MergeSorter.sort(data)
}

#[verified_engine::verified]
fn merge<T: Ord + Clone>(
    arr: &mut [T],
    temp: &mut [T],
    left: usize,
    mid: usize,
    right: usize,
    stats: &mut SortingStats,
) {
    let mut i = left;
    let mut j = mid;
    let mut k = left;

    while i < mid && j < right {
        stats.comparisons += 1;
        if temp[i] <= temp[j] {
            std::mem::swap(&mut arr[k], &mut temp[i]);
            stats.assignments += 1;
            i += 1;
        } else {
            std::mem::swap(&mut arr[k], &mut temp[j]);
            stats.assignments += 1;
            j += 1;
        }
        k += 1;
    }

    while i < mid {
        std::mem::swap(&mut arr[k], &mut temp[i]);
        stats.assignments += 1;
        i += 1;
        k += 1;
    }
    while j < right {
        std::mem::swap(&mut arr[k], &mut temp[j]);
        stats.assignments += 1;
        j += 1;
        k += 1;
    }
}

/// Strategy implementation for Quick Sort.
#[derive(Debug, Default, Clone, Copy)]
pub struct QuickSorter;

impl<T: Ord + Clone> Sorter<T> for QuickSorter {
    #[verified_engine::verified]
    fn sort(&self, data: &[T]) -> SortingResult<T> {
        let mut sorted_data = data.to_vec();
        let mut stats = SortingStats::default();
        if !sorted_data.is_empty() {
            let n = sorted_data.len();
            let mut stack = Vec::with_capacity(n); // pre-allocate maximum depth
            stack.push((0, n - 1));

            while let Some((low, high)) = stack.pop() {
                if low < high {
                    let p = partition(&mut sorted_data, low, high, &mut stats);
                    if p > 0 && p - 1 > low {
                        stack.push((low, p - 1));
                    }
                    if p + 1 < high {
                        stack.push((p + 1, high));
                    }
                }
            }
        }
        SortingResult { sorted_data, stats }
    }
}

/// Quick Sort
///
/// Selects a pivot and partitions the array.
///
/// # Probabilistic Analysis
///
/// * **Best Case**: Pivot splits array in half. $O(n \log n)$.
/// * **Worst Case**: Pivot is min or max (unbalanced). Recurrence $T(n) = T(n-1) + \Theta(n) \implies O(n^2)$.
/// * **Average Case**: Expected depth is $O(\log n)$.
///
/// **Space Complexity**: $O(\log n)$ stack space (average).
#[verified_engine::verified]
pub fn quick_sort<T: Ord + Clone>(data: &[T]) -> SortingResult<T> {
    QuickSorter.sort(data)
}

#[verified_engine::verified]
fn partition<T: Ord + Clone>(
    arr: &mut [T],
    low: usize,
    high: usize,
    stats: &mut SortingStats,
) -> usize {
    // To prevent O(n^2) behavior on sorted/reverse-sorted data,
    // we swap the middle element to the end and use it as pivot (Lomuto scheme).
    let mid = low + (high - low) / 2;
    arr.swap(mid, high);
    stats.swaps += 1;

    let pivot_index = high;
    let mut i = low;
    for j in low..high {
        stats.comparisons += 1;
        if arr[j] < arr[pivot_index] {
            if i != j {
                arr.swap(i, j);
                stats.swaps += 1;
            }
            i += 1;
        }
    }
    arr.swap(i, high);
    stats.swaps += 1;
    i
}

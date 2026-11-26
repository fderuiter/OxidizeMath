use super::stats::{SortingResult, SortingStats};
use super::strategy::Sorter;

/// Strategy implementation for Heap Sort.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeapSorter;

impl<T: Ord + Clone> Sorter<T> for HeapSorter {
    #[verified_engine::verified]
    fn sort(&self, data: &[T]) -> SortingResult<T> {
        let mut sorted_data = data.to_vec();
        let mut stats = SortingStats::default();
        let n = sorted_data.len();

        // Build heap (rearrange array)
        for i in (0..n / 2).rev() {
            heapify(&mut sorted_data, n, i, &mut stats);
        }

        // One by one extract an element from heap
        for i in (1..n).rev() {
            // Move current root to end
            sorted_data.swap(0, i);
            stats.swaps += 1;

            // call max heapify on the reduced heap
            heapify(&mut sorted_data, i, 0, &mut stats);
        }

        SortingResult { sorted_data, stats }
    }
}

/// Heap Sort
///
/// Uses a Binary Heap.
///
/// # Mathematical Steps
///
/// 1. **Heapify**: Convert array to Max-Heap. $O(n)$.
/// 2. **Sort**: Extract max `n` times. Each extraction is $O(\log n)$ (sift down).
///    Total: $O(n \log n)$.
///
/// Total Complexity: $O(n) + O(n \log n) = \Theta(n \log n)$.
/// Space: $O(1)$ (in-place).
#[verified_engine::verified]
pub fn heap_sort<T: Ord + Clone>(data: &[T]) -> SortingResult<T> {
    HeapSorter.sort(data)
}

#[verified_engine::verified]
fn heapify<T: Ord + Clone>(arr: &mut [T], n: usize, mut i: usize, stats: &mut SortingStats) {
    loop {
        let mut largest = i;
        let l = 2 * i + 1;
        let r = 2 * i + 2;

        if l < n {
            stats.comparisons += 1;
            if arr[l] > arr[largest] {
                largest = l;
            }
        }

        if r < n {
            stats.comparisons += 1;
            if arr[r] > arr[largest] {
                largest = r;
            }
        }

        if largest != i {
            arr.swap(i, largest);
            stats.swaps += 1;
            i = largest;
        } else {
            break;
        }
    }
}

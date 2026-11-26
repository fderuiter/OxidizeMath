//! Integration tests for test sorting.
use domain_applied::applied::algorithms::sorting::*;

// Helper to test strategy composition
fn run_sort<S: Sorter<i32>>(sorter: S, data: &[i32]) -> SortingResult<i32> {
    sorter.sort(data)
}

#[test]
#[verified_engine::verified]
fn test_strategy_pattern_composability() {
    let data = vec![5, 1, 4, 2, 8];
    let expected = vec![1, 2, 4, 5, 8];

    // Test dependency injection
    let merge_result = run_sort(MergeSorter, &data);
    assert_eq!(merge_result.sorted_data, expected);

    let quick_result = run_sort(QuickSorter, &data);
    assert_eq!(quick_result.sorted_data, expected);

    let bubble_result = run_sort(BubbleSorter, &data);
    assert_eq!(bubble_result.sorted_data, expected);
}

#[test]
#[verified_engine::verified]
fn test_information_theoretic_bound() {
    // Test small values
    let n = 10;
    let bound = information_theoretic_bound(n);
    // n * log2(n) - 1.44 * n
    // 10 * 3.32 - 14.4 = 33.2 - 14.4 = 18.8
    // log2(10!) = log2(3628800) approx 21.79
    // Stirling approximation is an approximation.
    // The function implements n log n - n log e

    assert!(bound > 0.0);
    assert!(bound < (n as f64) * (n as f64).log2());
}

#[test]
#[verified_engine::verified]
fn test_bubble_sort() {
    let data = vec![5, 1, 4, 2, 8];
    let result = bubble_sort(&data);
    assert_eq!(result.sorted_data, vec![1, 2, 4, 5, 8]);
    // Bubble sort on random data has comparisons.
    assert!(result.stats.comparisons > 0);
    assert!(result.stats.swaps > 0);

    // Test sorted (Best case)
    let sorted = vec![1, 2, 3, 4, 5];
    let result_sorted = bubble_sort(&sorted);
    assert_eq!(result_sorted.stats.swaps, 0);
    // Comparisons should be n-1? Wait, with optimization it's O(n) comparisons if it breaks early.
    // My implementation has `!swapped` check.
    // Inner loop runs n-1 times first pass. swapped is false. break.
    // So comparisons = n-1 = 4.
    assert_eq!(result_sorted.stats.comparisons, 4);

    // Test reverse (Worst case)
    let reverse = vec![5, 4, 3, 2, 1];
    let result_rev = bubble_sort(&reverse);
    // Comparisons: 4 + 3 + 2 + 1 = 10
    assert_eq!(result_rev.stats.comparisons, 10);
    // Swaps: 10
    assert_eq!(result_rev.stats.swaps, 10);
}

#[test]
#[verified_engine::verified]
fn test_insertion_sort() {
    let data = vec![5, 1, 4, 2, 8];
    let result = insertion_sort(&data);
    assert_eq!(result.sorted_data, vec![1, 2, 4, 5, 8]);

    // Test sorted
    let sorted = vec![1, 2, 3, 4, 5];
    let result_sorted = insertion_sort(&sorted);
    // 1 comparison per element starting from index 1.
    // index 1: 1 comp. index 2: 1 comp...
    // Total n-1 comparisons.
    assert_eq!(result_sorted.stats.comparisons, 4);
}

#[test]
#[verified_engine::verified]
fn test_merge_sort() {
    let data = vec![38, 27, 43, 3, 9, 82, 10];
    let result = merge_sort(&data);
    assert_eq!(result.sorted_data, vec![3, 9, 10, 27, 38, 43, 82]);
    // Merge sort always does O(n log n) comparisons roughly.
    assert!(result.stats.comparisons > 0);
}

#[test]
#[verified_engine::verified]
fn test_quick_sort() {
    let data = vec![10, 7, 8, 9, 1, 5];
    let result = quick_sort(&data);
    assert_eq!(result.sorted_data, vec![1, 5, 7, 8, 9, 10]);
}

#[test]
#[verified_engine::verified]
fn test_heap_sort() {
    let data = vec![12, 11, 13, 5, 6, 7];
    let result = heap_sort(&data);
    assert_eq!(result.sorted_data, vec![5, 6, 7, 11, 12, 13]);
}

#[test]
#[verified_engine::verified]
fn test_radix_sort() {
    let data = vec![170, 45, 75, 90, 802, 24, 2, 66];
    let result = radix_sort(&data);
    assert_eq!(result.sorted_data, vec![2, 24, 45, 66, 75, 90, 170, 802]);
}

#[test]
#[verified_engine::verified]
fn test_duplicate_elements() {
    let data = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
    let expected = vec![1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9];

    assert_eq!(bubble_sort(&data).sorted_data, expected);
    assert_eq!(insertion_sort(&data).sorted_data, expected);
    assert_eq!(merge_sort(&data).sorted_data, expected);
    assert_eq!(quick_sort(&data).sorted_data, expected);
    assert_eq!(heap_sort(&data).sorted_data, expected);
}

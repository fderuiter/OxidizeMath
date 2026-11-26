use super::stats::{SortingResult, SortingStats};
use super::strategy::Sorter;

/// Strategy implementation for Radix Sort.
#[derive(Debug, Default, Clone, Copy)]
pub struct RadixSorter;

impl Sorter<u64> for RadixSorter {
    #[verified_engine::verified]
    fn sort(&self, data: &[u64]) -> SortingResult<u64> {
        let mut sorted_data = data.to_vec();
        let mut stats = SortingStats::default();

        if sorted_data.is_empty() {
            return SortingResult { sorted_data, stats };
        }

        // Safe unwrap: sorted_data is not empty (checked above)
        let max_val = *sorted_data.iter().max().expect("sorted_data is not empty");
        // Comparisons to find max? technically yes, but usually considered O(n) overhead not part of sort loop logic strictly.
        // We'll count them for rigorousness.
        stats.comparisons += sorted_data.len() as u64 - 1;

        let n = sorted_data.len();
        // ⚡ Bolt Optimization:
        // Hoisted the `output` buffer allocation outside of the digit-processing loop.
        // Reusing a single buffer avoids multiple large heap allocations during sort.
        let mut output = vec![0; n];

        // ⚡ Bolt Optimization:
        // Switched Radix Sort from Base-10 (division/modulo) to Base-256 (bitwise ops).
        // This reduces the number of passes required from 20 (for u64) to 8,
        // and replaces expensive division with fast bit shifts.
        let mut shift = 0;
        // Check `shift < 64` to avoid `max_val >> 64` which causes panic or infinite loop in Rust.
        while shift < 64 && (max_val >> shift) > 0 {
            counting_sort_for_radix(&sorted_data, &mut output, shift, &mut stats);
            std::mem::swap(&mut sorted_data, &mut output);
            shift += 8;
        }

        SortingResult { sorted_data, stats }
    }
}

/// Radix Sort (LSD)
///
/// Sorts integers by processing individual digits.
///
/// # Mathematical Analysis
///
/// * $n$ = number of elements.
/// * $b$ = base (256).
/// * $d$ = number of digits (max value width).
///
/// Time: $d \times O(n + b)$.
/// If $d$ is constant and $b$ is small, this is linear time $O(n)$.
///
/// # constraints
/// Only works for non-negative integers (`u64`) in this implementation.
#[verified_engine::verified]
pub fn radix_sort(data: &[u64]) -> SortingResult<u64> {
    RadixSorter.sort(data)
}

#[verified_engine::verified]
fn counting_sort_for_radix(arr: &[u64], output: &mut [u64], shift: u32, stats: &mut SortingStats) {
    let n = arr.len();
    let mut count = [0; 256];

    // Store count of occurrences in count[]
    for &x in arr.iter() {
        let idx = ((x >> shift) & 0xFF) as usize;
        count[idx] += 1;
        stats.assignments += 1; // Read/Inc
    }

    // Change count[i] so that count[i] now contains actual
    // position of this digit in output[]
    for i in 1..256 {
        count[i] += count[i - 1];
    }

    // Build the output array
    for i in (0..n).rev() {
        let x = arr[i];
        let idx = ((x >> shift) & 0xFF) as usize;
        output[count[idx] - 1] = x;
        count[idx] -= 1;
        stats.assignments += 1; // Move to output
    }

    // Copy the output array to arr[], so that arr[] now
    // contains sorted numbers according to current digit
}

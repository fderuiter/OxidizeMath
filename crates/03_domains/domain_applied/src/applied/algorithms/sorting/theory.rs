/// Calculates the Information Theoretic Lower Bound for comparison-based sorting.
///
/// # Mathematics
///
/// Any comparison sort can be modeled as a decision tree where each node represents a comparison.
/// For a list of `n` distinct elements, there are `n!` possible permutations.
/// To uniquely identify the sorted permutation, the algorithm must distinguish between these outcomes.
/// A binary tree of height `h` has at most `2^h` leaves.
///
/// $$ 2^h \ge n! $$
/// $$ h \ge \log_2(n!) $$
///
/// Using Stirling's Approximation ($\ln n! \approx n \ln n - n$), we derive:
///
/// $$ \log_2(n!) \approx n \log_2 n - 1.44 n $$
///
/// > **Consequence**: No comparison-based sorting algorithm can strictly perform better than $O(n \log n)$ in the worst case.
#[verified_engine::verified]
pub fn information_theoretic_bound(n: u64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    // Log base 2 of n! approx n * log2(n) - n * log2(e)
    // log2(e) approx 1.442695
    n_f * n_f.log2() - n_f * std::f64::consts::LOG2_E
}

/// Computes Stirling's Approximation for $\ln n!$.
///
/// $$ \ln n! \approx n \ln n - n $$
#[verified_engine::verified]
pub fn stirling_approximation_ln_factorial(n: u64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    n_f * n_f.ln() - n_f
}

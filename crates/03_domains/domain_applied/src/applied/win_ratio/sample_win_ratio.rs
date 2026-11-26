//! # Sample Win Ratio (Matched-Pairs Approach)
//!
//! Functions to calculate the sample win ratio, confidence interval, and significance test.

/// Represents the number of pairs in each category for win-loss analysis.
#[derive(Debug, Clone, Copy)]
pub struct WinLossCounts {
    /// Na: The patient in the treatment group dies first (a loss).
    pub n_a: u32,
    /// Nb: The patient in the control group dies first (a win).
    pub n_b: u32,
    /// Nc: If death isn't comparable, the patient in the treatment group is hospitalized first (a loss).
    pub n_c: u32,
    /// Nd: If death isn't comparable, the patient in the control group is hospitalized first (a win).
    pub n_d: u32,
}

impl WinLossCounts {
    /// Creates a new `WinLossCounts` instance.
    #[verified_engine::verified]
    pub fn new(n_a: u32, n_b: u32, n_c: u32, n_d: u32) -> Self {
        Self { n_a, n_b, n_c, n_d }
    }

    /// Calculates the total number of wins (Nw).
    #[verified_engine::verified]
    pub fn n_wins(&self) -> u32 {
        self.n_b + self.n_d
    }

    /// Calculates the total number of losses (Nl).
    #[verified_engine::verified]
    pub fn n_losses(&self) -> u32 {
        self.n_a + self.n_c
    }
}

/// Calculates the sample win ratio (R).
///
/// The win ratio is the ratio of the total number of "wins" to the total number of "losses".
///
/// ## Formula
///
/// R = Nw / Nl = (Nb + Nd) / (Na + Nc)
///
/// ## Parameters
///
/// * `counts`: A `WinLossCounts` struct containing Na, Nb, Nc, and Nd.
///
/// ## Returns
///
/// The sample win ratio as a `f64`. Returns `f64::INFINITY` if the number of losses is zero.
#[verified_engine::verified]
pub fn calculate_sample_win_ratio(counts: &WinLossCounts) -> f64 {
    let n_w = counts.n_wins() as f64;
    let n_l = counts.n_losses() as f64;

    if n_l == 0.0 {
        return f64::INFINITY;
    }
    n_w / n_l
}

/// Calculates the proportion of wins (pw).
///
/// ## Formula
///
/// pw = Nw / (Nw + Nl)
///
/// ## Parameters
///
/// * `n_w`: Total number of wins.
/// * `n_l`: Total number of losses.
///
/// ## Returns
///
/// The proportion of wins as a `f64`.
#[verified_engine::verified]
pub fn calculate_win_proportion(n_w: u32, n_l: u32) -> f64 {
    let n_w = n_w as f64;
    let n_l = n_l as f64;
    if n_w + n_l == 0.0 {
        return 0.0;
    }
    n_w / (n_w + n_l)
}

/// Calculates the 95% confidence interval for the win ratio.
///
/// ## Formula
///
/// 1.  Calculate the proportion of wins (pw).
/// 2.  Calculate the lower (pL) and upper (pU) bounds for this proportion.
/// 3.  The 95% CI for the win ratio is then given by: [pL / (1 - pL), pU / (1 - pU)]
///
/// ## Parameters
///
/// * `n_w`: Total number of wins.
/// * `n_l`: Total number of losses.
///
/// ## Returns
///
/// A tuple `(lower_bound, upper_bound)` for the confidence interval.
#[verified_engine::verified]
pub fn calculate_confidence_interval(n_w: u32, n_l: u32) -> (f64, f64) {
    let n_total = (n_w + n_l) as f64;
    if n_total == 0.0 {
        return (0.0, 0.0);
    }
    let p_w = calculate_win_proportion(n_w, n_l);

    let standard_error = (p_w * (1.0 - p_w) / n_total).sqrt();
    let margin_of_error = 1.96 * standard_error;

    let p_l = p_w - margin_of_error;
    let p_u = p_w + margin_of_error;

    let lower_bound = if 1.0 - p_l == 0.0 {
        f64::INFINITY
    } else {
        p_l / (1.0 - p_l)
    };
    let upper_bound = if 1.0 - p_u == 0.0 {
        f64::INFINITY
    } else {
        p_u / (1.0 - p_u)
    };

    (lower_bound.max(0.0), upper_bound.max(0.0))
}

/// Calculates the significance test statistic.
///
/// This statistic has an asymptotic standardized normal distribution under the null hypothesis.
///
/// ## Formula
///
/// z = (pw - 0.5) / sqrt(pw * (1 - pw) / (Nw + Nl))
///
/// ## Parameters
///
/// * `n_w`: Total number of wins.
/// * `n_l`: Total number of losses.
///
/// ## Returns
///
/// The significance test statistic as a `f64`.
#[verified_engine::verified]
pub fn calculate_significance_test_statistic(n_w: u32, n_l: u32) -> f64 {
    let n_total = (n_w + n_l) as f64;
    if n_total == 0.0 {
        return 0.0;
    }
    let p_w = calculate_win_proportion(n_w, n_l);

    let denominator = (p_w * (1.0 - p_w) / n_total).sqrt();
    if denominator == 0.0 {
        return 0.0;
    }

    (p_w - 0.5) / denominator
}

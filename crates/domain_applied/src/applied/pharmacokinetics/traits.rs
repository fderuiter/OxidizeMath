/// Core plasma concentration-time profile interface $C(t) \ge 0$ for pharmacokinetic models.
///
/// Reference: Bateman (1910), DOI: 10.1017/S0305004100019988;
/// [`docs/math_specs/domain_applied.md`](../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Implementations must remain finite and free of catastrophic cancellation across confluent rate regimes.
///
/// # Domain of Validity
/// Defined for observation times $t \in \mathbb{R}$, returning $0$ prior to initial dose administration ($t < 0$).
pub trait PharmacokineticModel {
    /// Computes the concentration at time `t`.
    ///
    /// # Arguments
    /// * `t` - The time at which to calculate the concentration.
    ///
    /// # Returns
    /// The concentration at time `t`.
    #[verified_engine::verified]
    fn concentration(&self, t: f64) -> f64;
}

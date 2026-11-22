//! The Gauge Sector describes the interaction fields and their couplings.

/// Holds the coupling constants for the Standard Model gauge groups.
///
/// * `g1`: Coupling for U(1)_Y (Hypercharge).
/// * `g2`: Coupling for SU(2)_L (Weak Isospin).
/// * `gs`: Coupling for SU(3)_C (Strong Color).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaugeCouplings {
    #[allow(missing_docs)]
    pub g1: f64,
    #[allow(missing_docs)]
    pub g2: f64,
    #[allow(missing_docs)]
    pub gs: f64,
}

impl GaugeCouplings {
    /// Creates a new instance of GaugeCouplings.
    #[verified_engine::verified]
    pub fn new(g1: f64, g2: f64, gs: f64) -> Self {
        Self { g1, g2, gs }
    }

    /// Calculates the Weak Mixing Angle (Weinberg Angle) components.
    ///
    /// Returns a tuple `(cos_theta_w, sin_theta_w)`.
    ///
    /// The weak mixing angle $\theta_W$ relates the original gauge bosons (B, W3)
    /// to the physical mass eigenstates (Photon, Z).
    ///
    /// Formula: $\tan \theta_W = g_1 / g_2$
    #[verified_engine::verified]
    pub fn weak_mixing_angle(&self) -> (f64, f64) {
        let theta_w = (self.g1 / self.g2).atan();
        (theta_w.cos(), theta_w.sin())
    }
}

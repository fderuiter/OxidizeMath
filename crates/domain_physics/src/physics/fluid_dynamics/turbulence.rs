// [cite:generative_turbulence]
//! Turbulence Modeling concepts.
//!
//! Implements basic Reynolds-Averaged Navier-Stokes (RANS) concepts.

use nalgebra::Matrix3;

/// Reynolds Decomposition of a variable.
///
/// $$u = \bar{u} + u'$$
#[derive(Debug, Clone, Copy)]
pub struct ReynoldsDecomposition {
    #[allow(missing_docs)]
    pub mean: f64,
    #[allow(missing_docs)]
    pub fluctuating: f64,
}

impl ReynoldsDecomposition {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(mean: f64, fluctuating: f64) -> Self {
        Self { mean, fluctuating }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn instantaneous(&self) -> f64 {
        self.mean + self.fluctuating
    }
}

/// Represents the Reynolds Stress Tensor ($-\rho \overline{u'_i u'_j}$).
///
/// In RANS equations, this tensor represents the transfer of momentum due to turbulent fluctuations.
#[derive(Debug, Clone)]
pub struct ReynoldsStressTensor {
    /// The stress tensor matrix (symmetric).
    pub tensor: Matrix3<f64>,
}

impl ReynoldsStressTensor {
    /// Constructs a Reynolds Stress Tensor from velocity fluctuation correlations.
    ///
    /// $$R_{ij} = -\rho \langle u'_i u'_j \rangle$$
    ///
    /// * `rho`: Fluid density.
    /// * `correlations`: Matrix where $C_{ij} = \langle u'_i u'_j \rangle$.
    #[verified_engine::verified]
    pub fn from_fluctuations(rho: f64, correlations: Matrix3<f64>) -> Self {
        Self {
            tensor: correlations * -rho,
        }
    }
}

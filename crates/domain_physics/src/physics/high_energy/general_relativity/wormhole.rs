//! Zero-tidal-force traversable Morris-Thorne wormhole (`StaticMetric<3>`) and 3D embeddings.

use nalgebra::{SMatrix, SVector};
use pure_math::pure_math::tensor::{StaticMetric, TensorError};

use crate::error::HighEnergyError;

/// Equatorial slice of a zero-redshift Morris-Thorne traversable wormhole in proper radial
/// coordinates $(ct, l, \varphi)$ where $l \in (-\infty, +\infty)$ crosses the throat at $l = 0$.
///
/// Line element:
/// $$ds^2 = -d(ct)^2 + dl^2 + \left(b_0^2 + l^2\right) d\varphi^2$$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct MorrisThorneWormhole {
    /// Minimum areal radius $b_0 > 0$ at the wormhole throat $l = 0$.
    pub throat_radius: f64,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for MorrisThorneWormhole {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Wire {
            throat_radius: f64,
        }
        let wire = Wire::deserialize(deserializer)?;
        Self::new(wire.throat_radius).map_err(serde::de::Error::custom)
    }
}

impl MorrisThorneWormhole {
    /// Constructs a traversable [`MorrisThorneWormhole`] with throat radius $b_0 > 0$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if `throat_radius <= 0` or non-finite.
    #[verified_engine::verified]
    pub fn new(throat_radius: f64) -> Result<Self, HighEnergyError> {
        if !throat_radius.is_finite() || throat_radius <= 0.0 {
            return Err(HighEnergyError::InvalidRadius {
                radius: throat_radius,
                limit: 0.0,
            });
        }
        Ok(Self { throat_radius })
    }

    /// Circumferential (areal) radius $r(l) = \sqrt{b_0^2 + l^2}$ at proper distance $l$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if `l` or `self.throat_radius` is invalid.
    #[verified_engine::verified]
    pub fn areal_radius(&self, l: f64) -> Result<f64, HighEnergyError> {
        if !self.throat_radius.is_finite() || self.throat_radius <= 0.0 || !l.is_finite() {
            return Err(HighEnergyError::InvalidRadius {
                radius: l,
                limit: 0.0,
            });
        }
        Ok((self.throat_radius * self.throat_radius + l * l).sqrt())
    }

    /// Isometric Euclidean vertical embedding height $z(l) = b_0 \operatorname{arsinh}(l / b_0)$.
    ///
    /// Satisfies $(dr/dl)^2 + (dz/dl)^2 = 1$ across both sheets $l > 0$ and $l < 0$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if `l` or `self.throat_radius` is invalid.
    #[verified_engine::verified]
    pub fn embedding_z(&self, l: f64) -> Result<f64, HighEnergyError> {
        if !self.throat_radius.is_finite() || self.throat_radius <= 0.0 || !l.is_finite() {
            return Err(HighEnergyError::InvalidRadius {
                radius: l,
                limit: 0.0,
            });
        }
        Ok(self.throat_radius * (l / self.throat_radius).asinh())
    }

    /// Maps wormhole coordinates $(l, \varphi)$ to 3D Euclidean embedding coordinates $[x, y, z]$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if `l`, `phi`, or `throat_radius` is invalid.
    #[verified_engine::verified]
    pub fn embedding_xyz(&self, l: f64, phi: f64) -> Result<[f64; 3], HighEnergyError> {
        if !phi.is_finite() {
            return Err(HighEnergyError::InvalidRadius {
                radius: phi,
                limit: 0.0,
            });
        }
        let r = self.areal_radius(l)?;
        let z = self.embedding_z(l)?;
        let (sin_p, cos_p) = phi.sin_cos();
        Ok([r * cos_p, r * sin_p, z])
    }

    /// Evaluates the centrifugal barrier $V_{\text{eff}}(l) = \frac{\ell^2}{2(b_0^2 + l^2)}$
    /// governing proper radial motion $\frac{1}{2}\dot{l}^2 + V_{\text{eff}}(l) = \frac{1}{2}(e^2 - \kappa)$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidRadius`] if inputs are non-finite.
    #[verified_engine::verified]
    pub fn effective_potential(
        &self,
        l: f64,
        specific_angular_momentum: f64,
    ) -> Result<f64, HighEnergyError> {
        if !specific_angular_momentum.is_finite() {
            return Err(HighEnergyError::InvalidRadius {
                radius: specific_angular_momentum,
                limit: 0.0,
            });
        }
        let r = self.areal_radius(l)?;
        Ok((specific_angular_momentum * specific_angular_momentum) / (2.0 * r * r))
    }
}

impl StaticMetric<3> for MorrisThorneWormhole {
    #[verified_engine::verified]
    fn metric_at(&self, point: &SVector<f64, 3>) -> Result<SMatrix<f64, 3, 3>, TensorError> {
        if !self.is_regular_at(point) {
            return Err(TensorError::SingularMetric);
        }
        let l = point[1];
        let r2 = self.throat_radius * self.throat_radius + l * l;
        Ok(SMatrix::<f64, 3, 3>::new(
            -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, r2,
        ))
    }

    #[verified_engine::verified]
    fn inverse_metric_at(
        &self,
        point: &SVector<f64, 3>,
    ) -> Result<SMatrix<f64, 3, 3>, TensorError> {
        if !self.is_regular_at(point) {
            return Err(TensorError::SingularMetric);
        }
        let l = point[1];
        let r2 = self.throat_radius * self.throat_radius + l * l;
        Ok(SMatrix::<f64, 3, 3>::new(
            -1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0 / r2,
        ))
    }

    #[verified_engine::verified]
    fn christoffel_at(
        &self,
        point: &SVector<f64, 3>,
    ) -> Result<[SMatrix<f64, 3, 3>; 3], TensorError> {
        if !self.is_regular_at(point) {
            return Err(TensorError::SingularMetric);
        }
        let l = point[1];
        let r2 = self.throat_radius * self.throat_radius + l * l;
        let gamma_t = SMatrix::<f64, 3, 3>::zeros();
        let mut gamma_l = SMatrix::<f64, 3, 3>::zeros();
        gamma_l[(2, 2)] = -l;
        let mut gamma_phi = SMatrix::<f64, 3, 3>::zeros();
        let ratio = l / r2;
        gamma_phi[(1, 2)] = ratio;
        gamma_phi[(2, 1)] = ratio;
        Ok([gamma_t, gamma_l, gamma_phi])
    }

    #[verified_engine::verified]
    fn is_regular_at(&self, point: &SVector<f64, 3>) -> bool {
        self.throat_radius.is_finite()
            && self.throat_radius > 0.0
            && point.iter().all(|v| v.is_finite())
    }
}

use std::f64::consts::PI;

/// Represents the 1D Heat Equation parameters.
/// $\frac{\partial u}{\partial t} = \kappa \frac{\partial^2 u}{\partial x^2}$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HeatEquation1D {
    #[allow(missing_docs)]
    pub diffusivity: f64, // \kappa
}

impl HeatEquation1D {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(diffusivity: f64) -> Self {
        Self { diffusivity }
    }

    /// Computes a single separated variable mode.
    /// $u(x, t) = (A \cos \lambda x + B \sin \lambda x) e^{-\lambda^2 \kappa t}$
    #[verified_engine::verified]
    pub fn separated_mode(
        &self,
        lambda: f64,
        params: (f64, f64), // A, B
        x: f64,
        t: f64,
    ) -> f64 {
        let (a, b) = params;
        let spatial = a * (lambda * x).cos() + b * (lambda * x).sin();
        let temporal = (-lambda * lambda * self.diffusivity * t).exp();
        spatial * temporal
    }
}

/// Represents a heat distribution solution on a rod of length L with zero boundary conditions (ends at temp 0).
/// $u(x, t) = \sum_{n=1}^\infty B_n \sin(n \pi x / L) e^{-(n \pi / L)^2 \kappa t}$
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HeatRodSolution {
    #[allow(missing_docs)]
    pub length: f64,
    #[allow(missing_docs)]
    pub diffusivity: f64,
    #[allow(missing_docs)]
    pub coefficients: Vec<f64>, // B_n coefficients for n=1, 2, ...
}

impl HeatRodSolution {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn evaluate(&self, x: f64, t: f64) -> f64 {
        let mut u = 0.0;
        let k = self.diffusivity;
        let l = self.length;

        for (n_idx, &bn) in self.coefficients.iter().enumerate() {
            let n = (n_idx + 1) as f64;
            let lambda = n * PI / l;

            u += bn * (lambda * x).sin() * (-lambda * lambda * k * t).exp();
        }
        u
    }
}

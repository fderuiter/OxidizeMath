/// Represents the 2D Laplace Equation.
/// $\nabla^2 u = 0$
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LaplaceEquation2D;

impl LaplaceEquation2D {
    /// Computes a single separated variable mode in Cartesian coordinates.
    /// $u(x, y) = (A \cos \lambda x + B \sin \lambda x)(C \cosh \lambda y + D \sinh \lambda y)$
    #[verified_engine::verified]
    pub fn separated_mode_cartesian(
        lambda: f64,
        x_params: (f64, f64), // A, B
        y_params: (f64, f64), // C, D
        x: f64,
        y: f64,
    ) -> f64 {
        let (a, b) = x_params;
        let (c, d) = y_params;

        let x_part = a * (lambda * x).cos() + b * (lambda * x).sin();
        let y_part = c * (lambda * y).cosh() + d * (lambda * y).sinh();

        x_part * y_part
    }

    /// Poisson's equation source term check.
    /// Returns $\rho(r)$ given $\nabla^2 u = \rho$.
    /// This is just a placeholder for the concept.
    #[verified_engine::verified]
    pub fn poisson_source(laplacian_u: f64) -> f64 {
        laplacian_u
    }
}

/// Solves Laplace equation on a rectangular domain with Dirichlet boundary conditions using Fourier series.
/// Currently assumes $u(0,y)=0, u(a,y)=0, u(x,0)=0, u(x,b)=f(x)$.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RectangularLaplaceSolver {
    #[allow(missing_docs)]
    pub width: f64, // a
    #[allow(missing_docs)]
    pub height: f64, // b
    #[allow(missing_docs)]
    pub coefficients: Vec<f64>, // A_n coefficients
}

impl RectangularLaplaceSolver {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        let mut u = 0.0;
        let a = self.width;
        // let b = self.height;

        for (n_idx, &an) in self.coefficients.iter().enumerate() {
            let n = (n_idx + 1) as f64;
            let lambda = n * std::f64::consts::PI / a;

            // Typical solution form for these BCs: sin(n pi x / a) * sinh(n pi y / a)
            // The coefficient A_n usually absorbs the sinh(n pi b / a) term.
            u += an * (lambda * x).sin() * (lambda * y).sinh();
        }
        u
    }
}

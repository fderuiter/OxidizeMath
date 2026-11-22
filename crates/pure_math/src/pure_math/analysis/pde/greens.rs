use crate::pure_math::analysis::integration::{ClenshawCurtis, Integrator};

/// Represents a 1D Green's Function Solver for inhomogeneous problems.
/// $Ly = f(x)$ => $y(x) = \int G(x, \xi) f(\xi) d\xi$
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GreenFunctionSolver1D<I: Integrator> {
    integrator: I,
}

impl Default for GreenFunctionSolver1D<ClenshawCurtis> {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            integrator: ClenshawCurtis,
        }
    }
}

impl GreenFunctionSolver1D<ClenshawCurtis> {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<I: Integrator> GreenFunctionSolver1D<I> {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn with_integrator(integrator: I) -> Self {
        Self { integrator }
    }

    /// Solves for y(x) at a specific point x given the Green's function G and source f.
    ///
    /// # Arguments
    /// * `x` - The point at which to evaluate the solution.
    /// * `domain` - The integration domain (start, end).
    /// * `g` - The Green's function G(x, xi). Takes (x, xi).
    /// * `f` - The source function f(xi).
    #[verified_engine::verified]
    pub fn solve_at<G, F>(&self, x: f64, domain: (f64, f64), g: G, source: F) -> f64
    where
        G: Fn(f64, f64) -> f64,
        F: Fn(f64) -> f64,
    {
        let (min, max) = domain;
        // integrand = G(x, xi) * f(xi)
        let integrand = |xi: f64| g(x, xi) * source(xi);

        let result =
            self.integrator
                .integrate(integrand, min, max, math_commons::registry::TOLERANCE_FAST);
        result.value
    }
}

/// Helper for constructing Method of Images solutions.
///
/// For a problem like Poisson's equation $\nabla^2 u = \rho$, the Green's function can be
/// constructed using image charges.
///
/// Example: Infinite half-space z > 0.
/// $G(r, r') = \frac{1}{4\pi |r - r'|} - \frac{1}{4\pi |r - r'_{image}|}$
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MethodOfImages;

impl MethodOfImages {
    /// Returns the Green's function value for a 3D half-space (z > 0) with Dirichlet BC (G=0 at z=0).
    /// $r$ and $r_prime$ are vectors [x, y, z].
    #[verified_engine::verified]
    pub fn half_space_greens_function(r: [f64; 3], r_prime: [f64; 3]) -> f64 {
        let dist = |a: [f64; 3], b: [f64; 3]| -> f64 {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            let dz = a[2] - b[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };

        // Image source at (x', y', -z')
        let r_image = [r_prime[0], r_prime[1], -r_prime[2]];

        let d_source = dist(r, r_prime);
        let d_image = dist(r, r_image);

        if d_source.abs() < math_commons::registry::TOLERANCE_STANDARD {
            return f64::INFINITY; // Singularity
        }

        (1.0 / d_source - 1.0 / d_image) / (4.0 * std::f64::consts::PI)
    }
}

use crate::pure_math::analysis::integration::{ClenshawCurtis, Integrator};

/// 1D homogeneous Dirichlet Poisson boundary-value solver $u(x) = \int_0^L G(x, \xi) f(\xi)\,d\xi$ for $-u''(x) = f(x)$ with $G(x, \xi) = \frac{1}{L}\min(x, \xi)(L - \max(x, \xi))$.
///
/// Reference: Evans (2010), *Partial Differential Equations*, DOI: 10.1090/gsm/019; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Non-negative symmetric kernel $G(x, \xi) \ge 0$ guarantees discrete maximum-principle preservation without matrix inversion conditioning loss.
///
/// # Domain of Validity
/// Valid on bounded intervals $(0, L)$ with $L > 0$, homogeneous boundary conditions $u(0) = u(L) = 0$, and integrable forcing $f$.
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
    /// Creates a 1D Green's function solver backed by [`ClenshawCurtis`] quadrature.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }
}

impl<I: Integrator> GreenFunctionSolver1D<I> {
    /// Creates a 1D Green's function solver with a custom numerical [`Integrator`].
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

/// Classification of Second-Order Linear PDEs.
/// Based on $B^2 - 4AC$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdeClassification {
    /// Variant `Elliptic` (Elliptic).
    Elliptic,
    /// Variant `Parabolic` (Parabolic).
    Parabolic,
    /// Variant `Hyperbolic` (Hyperbolic).
    Hyperbolic,
    /// Variant `Undefined` (Undefined).
    Undefined, // If A=B=C=0 or degenerate
}

/// Represents the coefficients of a general 2nd order linear PDE:
/// $A u_{xx} + B u_{xy} + C u_{yy} + D u_x + E u_y + F u = G$
pub struct SecondOrderLinearPde2D {
    /// Value for `a` (a).
    pub a: f64,
    /// Value for `b` (b).
    pub b: f64,
    /// Value for `c` (c).
    pub c: f64,
    // Lower order terms don't affect classification
}

impl SecondOrderLinearPde2D {
    /// Classifies the PDE as elliptic, parabolic, or hyperbolic from its discriminant $B^2 - 4AC$.
    #[verified_engine::verified]
    pub fn classify(&self) -> PdeClassification {
        let discriminant = self.b * self.b - 4.0 * self.a * self.c;
        if discriminant < -math_commons::registry::TOLERANCE_STANDARD {
            PdeClassification::Elliptic
        } else if discriminant > math_commons::registry::TOLERANCE_STANDARD {
            PdeClassification::Hyperbolic
        } else {
            PdeClassification::Parabolic
        }
    }
}

/// Strategy trait for evaluating boundary conditions.
///
/// Adheres to the Strategy Pattern and Open/Closed Principle, allowing
/// users to implement custom boundary condition types (e.g., Robin) without
/// modifying the core enum.
pub trait BoundaryCondition {
    /// Evaluates the boundary condition at the given spatial or spatio-temporal coordinates.
    #[verified_engine::verified]
    fn evaluate_boundary(&self, x: f64, y: f64) -> f64;
}

/// Dirichlet boundary condition: specifies the value $u = f$ on the boundary.
pub struct DirichletBoundary {
    /// Value for `function` (function).
    pub function: Box<dyn Fn(f64, f64) -> f64>,
}

impl BoundaryCondition for DirichletBoundary {
    #[verified_engine::verified]
    fn evaluate_boundary(&self, x: f64, y: f64) -> f64 {
        (self.function)(x, y)
    }
}

/// Neumann boundary condition: specifies the normal derivative $\frac{\partial u}{\partial n} = g$.
pub struct NeumannBoundary {
    /// Value for `function` (function).
    pub function: Box<dyn Fn(f64, f64) -> f64>,
}

impl BoundaryCondition for NeumannBoundary {
    #[verified_engine::verified]
    fn evaluate_boundary(&self, x: f64, y: f64) -> f64 {
        (self.function)(x, y)
    }
}

/// Submodule providing `fused_stepper` types, traits, and algorithms.
pub mod fused_stepper;
/// Submodule providing `greens` types, traits, and algorithms.
pub mod greens;
/// Submodule providing `heat` types, traits, and algorithms.
pub mod heat;
/// Submodule providing `laplace` types, traits, and algorithms.
pub mod laplace;
/// Zero-allocation, `no_std`-compatible complex Thomas tridiagonal matrix factorization and sweeps.
pub mod tridiagonal;
/// Submodule providing `wave` types, traits, and algorithms.
pub mod wave;
/// Zero-allocation 2D Verlet wave tank, biharmonic Chladni plate, 2D FFT spectra, and sand drift.
pub mod wave_2d;
/// Pre-allocated 2D FDTD scalar and matched-Drude metamaterial wave solver.
pub mod wave_fdtd;

// [cite:graph_parameters_rust]

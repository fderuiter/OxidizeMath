/// Classification of Second-Order Linear PDEs.
/// Based on $B^2 - 4AC$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PdeClassification {
    #[allow(missing_docs)]
    Elliptic,
    #[allow(missing_docs)]
    Parabolic,
    #[allow(missing_docs)]
    Hyperbolic,
    #[allow(missing_docs)]
    Undefined, // If A=B=C=0 or degenerate
}

/// Represents the coefficients of a general 2nd order linear PDE:
/// $A u_{xx} + B u_{xy} + C u_{yy} + D u_x + E u_y + F u = G$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SecondOrderLinearPde2D {
    #[allow(missing_docs)]
    pub a: f64,
    #[allow(missing_docs)]
    pub b: f64,
    #[allow(missing_docs)]
    pub c: f64,
    // Lower order terms don't affect classification
}

impl SecondOrderLinearPde2D {
    #[allow(missing_docs)]
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
    #[allow(missing_docs)]
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
    #[allow(missing_docs)]
    pub function: Box<dyn Fn(f64, f64) -> f64>,
}

impl BoundaryCondition for NeumannBoundary {
    #[verified_engine::verified]
    fn evaluate_boundary(&self, x: f64, y: f64) -> f64 {
        (self.function)(x, y)
    }
}

#[allow(missing_docs)]
pub mod fused_stepper;
#[allow(missing_docs)]
pub mod greens;
#[allow(missing_docs)]
pub mod heat;
#[allow(missing_docs)]
pub mod laplace;
#[allow(missing_docs)]
pub mod wave;
/// Zero-allocation 2D Verlet wave tank, biharmonic Chladni plate, 2D FFT spectra, and sand drift.
pub mod wave_2d;

// [cite:graph_parameters_rust]

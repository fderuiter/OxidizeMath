pub mod buckled_flow;
pub mod cotangent_mesh;
/// Submodule providing `heat_equation` types, traits, and algorithms.
pub mod heat_equation;
/// Intrinsic hyperbolic geometry on the Poincaré and Beltrami-Klein disks via $\mathrm{SU}(1,1)$.
pub mod hyperbolic;
/// Submodule providing `mean_curvature_flow` types, traits, and algorithms.
pub mod mean_curvature_flow;
/// Submodule providing `operators` types, traits, and algorithms.
pub mod operators;
/// Submodule providing `surface` types, traits, and algorithms.
pub mod surface;
mod surface_ad;

pub use cotangent_mesh::CotangentLaplacian;

// [cite:modular_polynomials_review]

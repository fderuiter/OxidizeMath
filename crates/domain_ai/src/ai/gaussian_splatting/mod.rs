//! # 3D Gaussian Splatting
//!
//! A rasterization-based technique for real-time rendering of radiance fields.
//!
//! Unlike **NeRF** (which uses ray-marching to query a neural network at many points along a ray),
//! **Gaussian Splatting** represents the scene as a collection of explicitly stored 3D Gaussians.
//! These Gaussians are projected onto the 2D image plane ("splatted") and blended to form the image.
//!
//! ##  The Rendering Pipeline
//!
//! ```mermaid
//! graph TD
//!     subgraph Scene
//!     G3D[3D Gaussians] -->|View Transform| Cam[Camera Space]
//!     end
//!
//!     subgraph Projection
//!     Cam -->|Jacobian J| Cov2D[2D Covariance]
//!     Cov2D -->|Filter| G2D[2D Gaussians]
//!     end
//!
//!     subgraph Rasterization
//!     G2D -->|Sort by Depth| Sorted[Sorted Splats]
//!     Sorted -->|Alpha Blend| Image[Final Pixel Color]
//!     end
//!
//!     subgraph "Adaptive Density Control"
//!     Image -->|Loss Gradient| Grad[Positional Gradients]
//!     Grad -->|Thresholds| Action{Action?}
//!     Action -->|Large Grad + Large Scale| Split[Split Gaussian]
//!     Action -->|Large Grad + Small Scale| Clone[Clone Gaussian]
//!     Action -->|Low Opacity| Prune[Prune Gaussian]
//!     Split --> G3D
//!     Clone --> G3D
//!     Prune --> G3D
//!     end
//!
//!     style Scene fill:#e1f5fe,stroke:#01579b
//!     style Projection fill:#fff3e0,stroke:#e65100
//!     style Rasterization fill:#f3e5f5,stroke:#4a148c
//!     style Action fill:#ffebee,stroke:#b71c1c
//! ```
//!
//! ##  Quick Start: Blending Splats
//!
//! Create a scene with two Gaussians (Red and Green) and simulate the blending process at a specific pixel.
//!
//! ```rust
//! # use domain_ai::__doctest_facade as math_explorer;
//! use math_explorer::ai::gaussian_splatting::{Gaussian2D, rendering};
//! use math_explorer::nalgebra::{Point2, Vector3, Matrix2};
//!
//! // 1. Define pre-projected 2D Gaussians
//! // In a full pipeline, these come from projecting Gaussian3D objects.
//!
//! // Red Gaussian at (0,0) with high opacity
//! let red_splat = Gaussian2D {
//!     mean: Point2::new(0.0, 0.0),
//!     conic: Matrix2::from_diagonal_element(-0.5), // Standard covariance
//!     opacity: 0.8,
//!     color: Vector3::new(1.0, 0.0, 0.0),
//!     depth: 1.0, // Closer
//! };
//!
//! // Green Gaussian at (0,0) (behind red)
//! let green_splat = Gaussian2D {
//!     mean: Point2::new(0.0, 0.0),
//!     conic: Matrix2::from_diagonal_element(-0.5),
//!     opacity: 0.8,
//!     color: Vector3::new(0.0, 1.0, 0.0),
//!     depth: 2.0, // Further
//! };
//!
//! // 2. Sort by depth (Front-to-Back for some algorithms, Back-to-Front for standard alpha blending)
//! // This module uses a specific blending implementation (see rendering::blend_gaussians)
//! let scene = vec![red_splat, green_splat];
//!
//! // 3. Render pixel at (0,0)
//! let pixel_color = rendering::blend_gaussians(&scene, &Point2::new(0.0, 0.0));
//!
//! println!("Pixel Color: {:.2}, {:.2}, {:.2}", pixel_color.x, pixel_color.y, pixel_color.z);
//! // Expect a mix dominated by Red.
//! ```
//!
//! ## Modules
//!
//! *   **[structs]**: Core data structures (`Gaussian3D`, `Gaussian2D`, `Scene`).
//! *   **[projection]**: Math for projecting 3D covariance to 2D using the Jacobian of the view transform.
//! *   **[rendering]**: Alpha blending and opacity evaluation logic.
//! *   **[optimization]**: Adaptive density control strategies (Split vs Clone).

#[allow(missing_docs)]
pub mod optimization;
#[allow(missing_docs)]
pub mod projection;
#[allow(missing_docs)]
pub mod rendering;
#[allow(missing_docs)]
pub mod structs;

pub use structs::{Gaussian2D, Gaussian3D, Scene};

// [cite:gaussian_splatting]

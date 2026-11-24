#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod attractors;
pub mod bifurcation;
pub mod fractals;
pub mod hamiltonian;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

crate::declare_simulation_tab!(
    /// Interactive Chaos Theory exploration suite coupling spatial parameter atlases to live
    /// dynamical state orbits across [`fractals::FractalViewer`], [`bifurcation::BifurcationDiagram`],
    /// [`attractors::AttractorPlotter`], and [`hamiltonian::HamiltonianChaosLab`].
    ///
    /// - **Fractal Viewer**: Links the Mandelbrot parameter plane to live `160x120` Julia set hover
    ///   insets, critical orbit overlays, and progressive click/`Enter` pinning or teleportation while preserving per-mode viewports.
    /// - **Bifurcation Diagram**: Offloads logistic sweeps into cooperative `8ms` `BackgroundWorker`
    ///   slices with `logistic_lyapunov` stability overlays and click-linked cobweb/$x_n$ inspectors.
    /// - **Attractor Plotter**: Streams 3D Lorenz trajectories alongside live `lorenz_lyapunov` and
    ///   zero-allocation `correlation_dimension_in_place` readouts plus a 2D $(\rho, \sigma)$ regime map.
    /// - **Hamiltonian Chaos Lab**: Pairs zero-allocation symplectic solvers (`StormerVerlet`,
    ///   `Yoshida4`, `GaussLegendre4`) with cubic-Hermite Poincaré sections and `RungeKutta4` drift comparisons.
    pub struct ChaosTab,
    name: "Chaos Theory",
    domain: "Physics",
    slug: "chaos",
    order: 13,
);

// [cite:graph_parameters_rust]

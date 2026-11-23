// @explorer_feature = "physics"

use crate::tabs::ExplorerTab;
use eframe::egui;

pub mod attractors;
pub mod bifurcation;
pub mod fractals;
pub mod hamiltonian;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

/// Interactive Chaos Theory exploration suite coupling spatial parameter atlases to live
/// dynamical state orbits across [`fractals::FractalViewer`], [`bifurcation::BifurcationDiagram`],
/// [`attractors::AttractorPlotter`], and [`hamiltonian::HamiltonianChaosLab`].
///
/// - **Fractal Viewer**: Links the Mandelbrot parameter plane to live `96x72` Julia set hover
///   insets and progressive click/`Enter` teleportation while preserving per-mode viewports.
/// - **Bifurcation Diagram**: Offloads logistic sweeps into cooperative `8ms` `BackgroundWorker`
///   slices with `logistic_lyapunov` stability overlays and click-linked cobweb/$x_n$ inspectors.
/// - **Attractor Plotter**: Streams 3D Lorenz trajectories alongside live `lorenz_lyapunov` and
///   zero-allocation `correlation_dimension_in_place` readouts plus a 2D $(\rho, \sigma)$ regime map.
/// - **Hamiltonian Chaos Lab**: Pairs zero-allocation symplectic solvers (`StormerVerlet`,
///   `Yoshida4`, `GaussLegendre4`) with cubic-Hermite Poincaré sections and `RungeKutta4` drift comparisons.
pub struct ChaosTab {
    framework: crate::framework::SimulationFramework,
}

impl Default for ChaosTab {
    fn default() -> Self {
        Self {
            framework: crate::framework::SimulationFramework::new("chaos"),
        }
    }
}

impl ExplorerTab for ChaosTab {
    fn name(&self) -> &'static str {
        "Chaos Theory"
    }

    fn show_inside(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.framework.show_inside(ui, "chaos");
    }

    fn state_revision(&self) -> u64 {
        self.framework.state_revision()
    }

    fn save_state(&self) -> Option<String> {
        self.framework.save_state()
    }

    fn load_state(&mut self, state: &str) {
        self.framework.load_state(state);
    }

    fn select_tool_by_name(&mut self, name: &str) -> bool {
        self.framework.select_tool_by_name(name)
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.framework.focused_canvas_id()
    }

    fn framework(&self) -> Option<&crate::framework::SimulationFramework> {
        Some(&self.framework)
    }

    fn framework_mut(&mut self) -> Option<&mut crate::framework::SimulationFramework> {
        Some(&mut self.framework)
    }
}

// [cite:graph_parameters_rust]

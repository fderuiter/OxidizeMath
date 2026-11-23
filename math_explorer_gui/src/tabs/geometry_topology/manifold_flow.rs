//! Interactive 3D Manifold Flow Studio (`ManifoldFlowStudio`) stepping flat-buffer
//! `MeanCurvatureFlow`, `HeatEquationSolver`, and `laplace_beltrami` diagnostics on
//! `Plot3DViewport` with CFL clamping, `FiniteStateGuard` rollback, and watertight
//! `.stl`, `.ply`, and `.obj` 3D exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:pure_math]

use super::custom_surface::{CustomSurfaceSandbox, SurfaceType};
use crate::camera::Camera3D;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolMetadata};
use eframe::egui;
use math_explorer::pure_math::differential_geometry::mean_curvature_flow::{
    DiscreteSurface, MeanCurvatureFlow,
};
use nalgebra::Point3;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

#[path = "manifold_flow/export.rs"]
mod export;
#[path = "manifold_flow/preset.rs"]
pub mod preset;
#[path = "manifold_flow/simulation.rs"]
mod simulation;
#[path = "manifold_flow/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "manifold_flow/tests.rs"]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
#[path = "manifold_flow/wasm_tests.rs"]
mod wasm_tests;

/// Active geometric PDE evolution law in [`ManifoldFlowStudio`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ManifoldFlowMode {
    /// Extrinsic Mean Curvature Flow $\partial_t \mathbf{r} = \vec{H}$ evolving 3D vertices.
    MeanCurvature,
    /// Intrinsic Laplace-Beltrami heat diffusion $\partial_t u = \alpha \Delta_S u$.
    HeatDiffusion,
}

impl ManifoldFlowMode {
    /// Human-readable mode label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::MeanCurvature => "Mean Curvature Flow (dr/dt = H n)",
            Self::HeatDiffusion => "Laplace-Beltrami Heat Diffusion (du/dt = alpha Delta_S u)",
        }
    }
}

/// Interactive 3D Manifold Flow & Geometric Diffusion Studio.
pub struct ManifoldFlowStudio {
    /// Selected PDE evolution mode (`MeanCurvature` or `HeatDiffusion`).
    pub mode: ManifoldFlowMode,
    /// Active parametric or custom surface topology.
    pub surface_type: SurfaceType,
    /// Shared custom heightfield and parametric formula sandbox.
    pub custom_sandbox: CustomSurfaceSandbox,
    /// Sphere radius parameter (`0.3..=4.0`).
    pub sphere_radius: f64,
    /// Torus major radius (`1.2..=4.5`).
    pub torus_major_radius: f64,
    /// Torus minor radius (`0.25..=1.8`).
    pub torus_minor_radius: f64,
    /// Klein bottle scale radius (`0.6..=3.5`).
    pub klein_radius: f64,
    /// Radial ripple perturbation amplitude applied to initial `MeanCurvatureFlow` geometry (`0.0..=0.5`).
    pub initial_perturbation: f64,
    /// Grid resolution along $u$ (`8..=48`).
    pub u_res: usize,
    /// Grid resolution along $v$ (`8..=48`).
    pub v_res: usize,
    /// Requested integration time step $\Delta t$ before CFL clamping.
    pub dt: f64,
    /// Latest computed CFL stability ceiling $\Delta t_{\text{CFL}}$.
    pub cfl_dt: f64,
    /// Effective clamped time step $\min(\Delta t, \Delta t_{\text{CFL}})$.
    pub effective_dt: f64,
    /// Thermal diffusivity $\alpha$ for `HeatDiffusion` (`0.005..=1.0`).
    pub alpha: f64,
    /// Number of explicit sub-steps advanced per animation frame (`1..=16`).
    pub substeps_per_frame: usize,
    /// Probe coordinate `(u, v)` in normalized `[0, 1]^2` parameter space for `laplace_beltrami`.
    pub probe_uv_frac: [f64; 2],
    /// Whether continuous playback is active.
    pub is_playing: bool,
    /// Render wireframe edges over shaded triangles.
    pub show_wireframe: bool,
    /// Shared 3D orbital camera.
    pub camera: Camera3D,
    /// Zero-allocation Mean Curvature Flow solver.
    pub mcf: MeanCurvatureFlow,
    /// Flat temperature buffer (`u_res * v_res`) for `HeatEquationSolver`.
    pub temp_grid: Vec<f64>,
    /// Pre-allocated scratch buffer (`u_res * v_res`) for `HeatEquationSolver`.
    pub temp_scratch: Vec<f64>,
    /// Cached static surface vertex positions (`u_res * v_res`) for `HeatDiffusion` rendering.
    pub static_points: Vec<Point3<f64>>,
    /// Elapsed simulation time $t$ and total executed steps.
    pub sim_time: f64,
    /// Executed solver step count since reset.
    pub step_count: usize,
    /// Initial surface area or total heat at $t = 0$.
    pub initial_metric: f64,
    /// Current surface area or total heat at $t$.
    pub current_metric: f64,
    /// Live `laplace_beltrami` operator value at `probe_uv_frac`.
    pub probe_laplace_beltrami: f64,
    /// Time history `[t, normalized_metric]` for CSV/SVG export and diagnostics.
    pub metric_history: Vec<[f64; 2]>,
    /// Recovery status note when `FiniteStateGuard` rolls back a non-finite step.
    pub recovery_note: Option<String>,
    pub(crate) backup_points: Vec<Point3<f64>>,
    pub(crate) backup_temps: Vec<f64>,
    pub(crate) state_rev: u64,
}

impl Default for ManifoldFlowStudio {
    fn default() -> Self {
        let placeholder = DiscreteSurface::new(vec![Point3::origin(); 4], 2, 2, false, false);
        let mut studio = Self {
            mode: ManifoldFlowMode::MeanCurvature,
            surface_type: SurfaceType::Torus,
            custom_sandbox: CustomSurfaceSandbox::default(),
            sphere_radius: 1.8,
            torus_major_radius: 2.0,
            torus_minor_radius: 0.75,
            klein_radius: 1.5,
            initial_perturbation: 0.18,
            u_res: 24,
            v_res: 20,
            dt: 0.008,
            cfl_dt: 0.01,
            effective_dt: 0.008,
            alpha: 0.08,
            substeps_per_frame: 2,
            probe_uv_frac: [0.35, 0.45],
            is_playing: false,
            show_wireframe: true,
            camera: Camera3D::new(0.55, 0.58, 0.92),
            mcf: MeanCurvatureFlow::new(placeholder),
            temp_grid: Vec::new(),
            temp_scratch: Vec::new(),
            static_points: Vec::new(),
            sim_time: 0.0,
            step_count: 0,
            initial_metric: 1.0,
            current_metric: 1.0,
            probe_laplace_beltrami: 0.0,
            metric_history: Vec::with_capacity(256),
            recovery_note: None,
            backup_points: Vec::new(),
            backup_temps: Vec::new(),
            state_rev: 1,
        };
        studio.reset_simulation();
        studio
    }
}

impl FiniteStateGuard for ManifoldFlowStudio {
    fn is_finite_state(&self) -> bool {
        self.dt.is_finite()
            && self.alpha.is_finite()
            && self.current_metric.is_finite()
            && self
                .mcf
                .surface
                .as_slice()
                .iter()
                .all(|p| p.coords.iter().all(|c| c.is_finite()))
            && self.temp_grid.iter().all(|v| v.is_finite())
    }
}

impl TheoryDescribable for ManifoldFlowStudio {
    fn theory_description(&self) -> String {
        format!(
            "{} on {:?} (${}\\times{}$ flat buffer, $\\Delta t_{{\\mathrm{{eff}}}} = {:.4} \\le \\Delta t_{{\\mathrm{{CFL}}}} = {:.4}$, $\\Delta_S f = {:.4}$). Mean curvature flow evolves along steepest area descent $\\partial_t \\mathbf{{r}} = H\\mathbf{{n}}$, while intrinsic heat flow diffuses scalar temperature via the metric Laplace-Beltrami operator.",
            self.mode.label(),
            self.surface_type,
            self.u_res,
            self.v_res,
            self.effective_dt,
            self.cfl_dt,
            self.probe_laplace_beltrami
        )
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\frac{\partial \mathbf{r}}{\partial t} = H\mathbf{n} = \frac{1}{2}\Delta_S \mathbf{r}, \qquad \frac{\partial u}{\partial t} = \alpha \Delta_S u".to_string(),
        )
    }

    fn phonetic_description(&self) -> String {
        format!(
            "Manifold flow studio running {} on {:?} at time {:.3}.",
            self.mode.label(),
            self.surface_type,
            self.sim_time
        )
    }

    fn theory_citation(&self) -> String {
        "Huisken, G. (1984). Flow by mean curvature of convex surfaces into spheres. J. Differential Geometry; Meyer, M., Desbrun, M., Schröder, P., & Barr, A. H. (2003).".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "cfl".to_string(),
            "Explicit parabolic geometric flows require dt <= C * min(ds^2) for numerical stability.".to_string(),
        );
        map
    }
}

impl InteractiveTool for ManifoldFlowStudio {
    fn name(&self) -> &'static str {
        "Manifold Flow Studio"
    }

    fn domain(&self) -> &'static str {
        "geometry_topology"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash as _, Hasher as _};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.state_rev.hash(&mut h);
        self.custom_sandbox
            .state_fingerprint(self.surface_type)
            .hash(&mut h);
        self.camera.hash_state(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.parameters_are_valid() && self.is_finite_state()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.clamp_model_parameters();
        true
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot().with_3d_solid()
    }

    fn export_colored_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::ColoredMesh, String>> {
        Some(self.build_watertight_colored_mesh())
    }

    fn export_mesh_3d(&self) -> Option<Result<oxidize_core::mesh::Mesh, String>> {
        Some(self.build_watertight_colored_mesh().map(|cm| cm.mesh))
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.collect_export_primitives()
    }

    fn step_simulation(&mut self) {
        self.advance_substeps(self.substeps_per_frame.clamp(1, 16));
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_studio_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_studio_state(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        if self.is_playing {
            self.advance_substeps(self.substeps_per_frame.clamp(1, 16));
            ui.ctx().request_repaint();
        }
        self.render_side_panel(ui);
        self.render_viewport_panel(ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    ToolMetadata {
        id: "ManifoldFlowStudio",
        display_name: "Manifold Flow Studio",
        domain: "geometry_topology",
        tags: &["mean-curvature-flow", "heat-equation", "laplace-beltrami", "manifold", "3d"],
        build: || Box::new(ManifoldFlowStudio::default()),
    }
}

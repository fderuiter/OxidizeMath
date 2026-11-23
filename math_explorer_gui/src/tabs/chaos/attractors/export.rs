//! Cold-path rotation-minimizing tube extrusion for [`super::AttractorPlotter`] 3D exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AttractorMode, AttractorPlotter, CompiledFlowJob};
use math_commons::math_kernel::colormap::mapped_gradient;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
use nalgebra::Vector3;
use oxidize_core::mesh::{
    extrude_polyline_tube, ColoredMesh, Mesh, Point3D, VertexColor, MAX_TUBE_RADIAL_SEGMENTS,
};

/// Minimum swept tube radius for 3D Lorenz trajectory export.
pub const MIN_TUBE_RADIUS: f64 = 0.05;
/// Maximum swept tube radius for 3D Lorenz trajectory export.
pub const MAX_TUBE_RADIUS: f64 = 2.00;
/// Default swept tube radius for 3D Lorenz trajectory export.
pub const DEFAULT_TUBE_RADIUS: f64 = 0.35;
/// Maximum trajectory points extruded in a single cold-path export.
const MAX_EXPORT_TRAJECTORY_POINTS: usize = 10_000;
/// Radial segments around the swept tube cross-section (`<= MAX_TUBE_RADIAL_SEGMENTS`).
const DEFAULT_TUBE_RADIAL_SEGMENTS: usize = 16;
/// Preview RK4 steps synthesized on cold export when history has not yet accumulated `< 2` points.
const COLD_PREVIEW_STEPS: usize = 320;

fn centered_point3d(p: Vector3<f64>, mode: AttractorMode) -> Point3D {
    let z_offset = match mode {
        AttractorMode::Lorenz => 25.0,
        AttractorMode::Custom => 0.0,
    };
    Point3D::new(p.x as f32, p.y as f32, (p.z - z_offset) as f32)
}

impl AttractorPlotter {
    fn collect_trajectory_points_3d(&self) -> Result<Vec<Point3D>, String> {
        if self.diverged() || !self.is_finite_state() {
            return Err(
                "Cannot export 3D trajectory while attractor solver state is non-finite."
                    .to_string(),
            );
        }
        let mode = self.mode;
        let mut pts: Vec<Point3D> = self
            .history
            .iter()
            .filter(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite())
            .take(MAX_EXPORT_TRAJECTORY_POINTS)
            .map(|&v| centered_point3d(v, mode))
            .collect();

        if pts.len() < 2 {
            pts.clear();
            let start = self.system.state.vec;
            let dt = self.dt.clamp(0.001, 0.05);
            pts.push(centered_point3d(start, mode));
            match mode {
                AttractorMode::Lorenz => {
                    let mut preview = LorenzBuilder::new()
                        .sigma(self.system.sigma)
                        .rho(self.system.rho)
                        .beta(self.system.beta)
                        .build(LorenzState::new(start.x, start.y, start.z));
                    for _ in 0..COLD_PREVIEW_STEPS {
                        preview.step(dt);
                        let v = preview.state.vec;
                        if !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite() {
                            break;
                        }
                        pts.push(centered_point3d(v, mode));
                    }
                }
                AttractorMode::Custom => {
                    let job = CompiledFlowJob::from_bank(&self.flow_bank);
                    let mut curr = start;
                    for _ in 0..COLD_PREVIEW_STEPS {
                        curr = job.rk4_step(curr, dt);
                        if !curr.x.is_finite() || !curr.y.is_finite() || !curr.z.is_finite() {
                            break;
                        }
                        pts.push(centered_point3d(curr, mode));
                    }
                }
            }
        }
        if pts.len() < 2 {
            return Err("Trajectory requires at least 2 finite 3D points to extrude.".to_string());
        }
        Ok(pts)
    }

    pub(super) fn build_trajectory_mesh_3d(&self) -> Result<Mesh, String> {
        if !self.tube_radius.is_finite() {
            return Err("Tube radius must be finite.".to_string());
        }
        let pts = self.collect_trajectory_points_3d()?;
        let radius = self.tube_radius.clamp(MIN_TUBE_RADIUS, MAX_TUBE_RADIUS) as f32;
        let radial = DEFAULT_TUBE_RADIAL_SEGMENTS.min(MAX_TUBE_RADIAL_SEGMENTS);
        extrude_polyline_tube(&pts, radius, radial, false).map_err(|e| e.to_string())
    }

    pub(super) fn build_trajectory_colored_mesh_3d(&self) -> Result<ColoredMesh, String> {
        let mesh = self.build_trajectory_mesh_3d()?;
        let radial = DEFAULT_TUBE_RADIAL_SEGMENTS.min(MAX_TUBE_RADIAL_SEGMENTS);
        let ring_count = mesh.vertices.len().saturating_sub(2) / radial;
        let max_idx = ring_count.saturating_sub(1).max(1) as f64;
        let mut colors = Vec::with_capacity(mesh.vertices.len());
        for ring_i in 0..ring_count {
            let rgb = mapped_gradient(ring_i as f64, 0.0, max_idx);
            let vc = VertexColor::new(rgb[0], rgb[1], rgb[2]);
            for _ in 0..radial {
                colors.push(vc);
            }
        }
        let start_rgb = mapped_gradient(0.0, 0.0, 1.0);
        let end_rgb = mapped_gradient(1.0, 0.0, 1.0);
        while colors.len() + 1 < mesh.vertices.len() {
            colors.push(VertexColor::new(start_rgb[0], start_rgb[1], start_rgb[2]));
        }
        if colors.len() < mesh.vertices.len() {
            colors.push(VertexColor::new(end_rgb[0], end_rgb[1], end_rgb[2]));
        }
        Ok(ColoredMesh::new(mesh, colors))
    }

    pub(super) fn build_attractor_datasets_2d(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut points: Vec<[f64; 2]> = self.history.iter().map(|p| self.project(*p)).collect();
        if points.is_empty() && self.is_finite_state() {
            points.push(self.project(self.system.state.vec));
        }
        let mut out = vec![("Trajectory".to_string(), points)];
        let view = self.sync_branch.effective_view(self.step_count);
        if view.fork_active {
            let shadow_pts: Vec<[f64; 2]> = view
                .shadow_history
                .iter()
                .map(|p| self.project(*p))
                .collect();
            let t_k = view.fork_time.unwrap_or(0.0);
            out.push((format!("Shadow Baseline @ t_k={t_k:.3}"), shadow_pts));
            if let Some(meta) = self.attractor_branch_metadata() {
                out.push(crate::framework::build_branch_metadata_dataset_2d(&meta));
            }
        }
        out
    }

    pub(super) fn build_attractor_datasets_3d(&self) -> Vec<(String, Vec<[f64; 3]>)> {
        let mut points: Vec<[f64; 3]> = self.history.iter().map(|p| [p.x, p.y, p.z]).collect();
        if points.is_empty() && self.is_finite_state() {
            let v = self.system.state.vec;
            points.push([v.x, v.y, v.z]);
        }
        let mut out = vec![("Trajectory".to_string(), points)];
        let view = self.sync_branch.effective_view(self.step_count);
        if view.fork_active {
            let shadow_pts: Vec<[f64; 3]> = view
                .shadow_history
                .iter()
                .map(|p| [p.x, p.y, p.z])
                .collect();
            let t_k = view.fork_time.unwrap_or(0.0);
            out.push((format!("Shadow Baseline @ t_k={t_k:.3}"), shadow_pts));
            if let Some(meta) = self.attractor_branch_metadata() {
                let anchor = view.fork_vec.map_or([0.0; 3], |v| [v.x, v.y, v.z]);
                out.push(crate::framework::build_branch_metadata_dataset_3d(
                    &meta, anchor,
                ));
            }
        }
        out
    }

    pub(super) fn build_attractor_export_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        let view = self.sync_branch.effective_view(self.step_count);
        if !view.fork_active {
            return egui_plot::get_last_plot_primitives();
        }
        let palette = crate::app::SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
        let mut prims = Vec::new();
        let shadow_2d: Vec<[f64; 2]> = view
            .shadow_history
            .iter()
            .map(|p| self.project(*p))
            .collect();
        prims.extend(crate::framework::build_dashed_export_primitives(
            "Shadow Baseline",
            &shadow_2d,
            palette.series_color(1),
            1.6,
        ));
        let primary_2d: Vec<[f64; 2]> = self.history.iter().map(|p| self.project(*p)).collect();
        if primary_2d.len() >= 2 {
            prims.push(egui_plot::ExportPrimitive::Polyline {
                name: "Trajectory".to_string(),
                points: primary_2d,
                color: palette.series_color(0),
                stroke_width: 1.8,
                closed: false,
                fill: None,
            });
        }
        if let Some(fv) = view.fork_vec {
            prims.push(egui_plot::ExportPrimitive::Points {
                name: "Fork @ t_k".to_string(),
                points: vec![self.project(fv)],
                radius: 5.0,
                color: palette.accent,
            });
        }
        prims
    }
}

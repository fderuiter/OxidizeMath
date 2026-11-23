#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{SculptBrush, VrSourceMode, MAX_VR_COORD_ABS};
use super::{VietorisRipsTool, MAX_VIETORIS_RIPS_POINTS};
use math_explorer::pure_math::statistics::tda::{Point2D, Point3D};

impl VietorisRipsTool {
    /// Projects a 3D world landmark into `Plot3DViewport` 2D coordinates and viewer depth.
    pub fn project_3d_with_depth(&self, pt: Point3D) -> ([f64; 2], f64) {
        let centered = [
            pt.x - self.view_center_3d[0],
            pt.y - self.view_center_3d[1],
            pt.z - self.view_center_3d[2],
        ];
        let [rx, ry, rz] = self.camera.rotate_to_camera_space(&centered);
        let z = (self.camera.zoom as f64).max(0.01);
        ([rx * z, ry * z], rz)
    }

    /// Projects a 3D world landmark into `Plot3DViewport` 2D plot coordinates.
    pub fn project_3d(&self, pt: Point3D) -> [f64; 2] {
        self.project_3d_with_depth(pt).0
    }

    /// Exact orthogonal inverse of [`Self::project_3d_with_depth`] at fixed camera-space depth.
    pub fn unproject_3d(&self, plot_xy: [f64; 2], cam_depth: f64) -> Point3D {
        let z = (self.camera.zoom as f64).max(0.01);
        let x_cam = plot_xy[0] / z;
        let y_cam = plot_xy[1] / z;
        let (sy, cy) = (self.camera.yaw as f64).sin_cos();
        let (sp, cp) = (self.camera.pitch as f64).sin_cos();

        let y1 = y_cam * cp + cam_depth * sp;
        let z1 = -y_cam * sp + cam_depth * cp;
        let x0 = x_cam * cy + y1 * sy;
        let y0 = -x_cam * sy + y1 * cy;

        Point3D::new(
            (x0 + self.view_center_3d[0]).clamp(-MAX_VR_COORD_ABS, MAX_VR_COORD_ABS),
            (y0 + self.view_center_3d[1]).clamp(-MAX_VR_COORD_ABS, MAX_VR_COORD_ABS),
            (z1 + self.view_center_3d[2]).clamp(-MAX_VR_COORD_ABS, MAX_VR_COORD_ABS),
        )
    }

    /// Applies a single 2D tactile brush gesture at `plot_xy`, returning `true` if vertices changed.
    pub fn apply_sculpt_2d(
        &mut self,
        brush: SculptBrush,
        plot_xy: [f64; 2],
        hit_radius: f64,
        continuing_drag: bool,
    ) -> bool {
        if !plot_xy[0].is_finite() || !plot_xy[1].is_finite() || !hit_radius.is_finite() {
            return false;
        }
        let clamped = Point2D::new(plot_xy[0].clamp(-50.0, 50.0), plot_xy[1].clamp(-50.0, 50.0));
        let r2 = hit_radius.max(0.05).powi(2);
        match brush {
            SculptBrush::OrbitOrPan => false,
            SculptBrush::Move => self.sculpt_move_2d(clamped, r2, continuing_drag),
            SculptBrush::Add => self.sculpt_add_2d(clamped, r2, continuing_drag),
            SculptBrush::Erase => self.sculpt_erase_2d(clamped, r2),
        }
    }

    fn sculpt_move_2d(&mut self, target: Point2D, r2: f64, continuing_drag: bool) -> bool {
        if !continuing_drag {
            self.active_drag_vertex = self.nearest_2d_vertex(target, r2);
        }
        if let Some(idx) = self.active_drag_vertex {
            if idx < self.points.len() && self.points[idx] != target {
                self.points[idx] = target;
                self.update_complex();
                return true;
            }
        }
        false
    }

    fn sculpt_add_2d(&mut self, target: Point2D, r2: f64, continuing_drag: bool) -> bool {
        if self.points.len() >= MAX_VIETORIS_RIPS_POINTS {
            return false;
        }
        let min_spacing_sq = if continuing_drag { r2 * 0.35 } else { 1e-6 };
        if self.nearest_2d_vertex(target, min_spacing_sq).is_some() {
            return false;
        }
        self.points.push(target);
        self.update_complex();
        true
    }

    fn sculpt_erase_2d(&mut self, target: Point2D, r2: f64) -> bool {
        if self.points.len() <= 1 {
            return false;
        }
        if let Some(idx) = self.nearest_2d_vertex(target, r2) {
            self.points.remove(idx);
            self.active_drag_vertex = None;
            self.update_complex();
            return true;
        }
        false
    }

    fn nearest_2d_vertex(&self, target: Point2D, max_dist_sq: f64) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (i, p) in self.points.iter().enumerate() {
            let d2 = (p.x - target.x).powi(2) + (p.y - target.y).powi(2);
            if d2 <= max_dist_sq && best.is_none_or(|(_, b)| d2 < b) {
                best = Some((i, d2));
            }
        }
        best.map(|(idx, _)| idx)
    }

    /// Applies a 3D tactile brush gesture at projected `plot_xy` in `Plot3DViewport`.
    pub fn apply_sculpt_3d_projected(
        &mut self,
        brush: SculptBrush,
        plot_xy: [f64; 2],
        hit_radius: f64,
        continuing_drag: bool,
    ) -> bool {
        if !plot_xy[0].is_finite() || !plot_xy[1].is_finite() || !hit_radius.is_finite() {
            return false;
        }
        let r2 = hit_radius.max(0.1).powi(2);
        match brush {
            SculptBrush::OrbitOrPan => false,
            SculptBrush::Move => self.sculpt_move_3d(plot_xy, r2, continuing_drag),
            SculptBrush::Add => self.sculpt_add_3d(plot_xy, r2, continuing_drag),
            SculptBrush::Erase => self.sculpt_erase_3d(plot_xy, r2),
        }
    }

    fn sculpt_move_3d(&mut self, plot_xy: [f64; 2], r2: f64, continuing_drag: bool) -> bool {
        if !continuing_drag {
            self.active_drag_vertex = self.nearest_3d_projected(plot_xy, r2).map(|(i, _)| i);
        }
        let Some(idx) = self.active_drag_vertex else {
            return false;
        };
        if idx >= self.points_3d.len() {
            return false;
        }
        let (_, depth) = self.project_3d_with_depth(self.points_3d[idx]);
        let updated = self.unproject_3d(plot_xy, depth);
        if self.points_3d[idx] == updated {
            return false;
        }
        self.mode = VrSourceMode::Points3D;
        self.chaos.playing = false;
        self.points_3d[idx] = updated;
        self.update_complex();
        true
    }

    fn sculpt_add_3d(&mut self, plot_xy: [f64; 2], r2: f64, continuing_drag: bool) -> bool {
        if self.points_3d.len() >= MAX_VIETORIS_RIPS_POINTS {
            return false;
        }
        let min_spacing_sq = if continuing_drag { r2 * 0.45 } else { 1e-6 };
        if self.nearest_3d_projected(plot_xy, min_spacing_sq).is_some() {
            return false;
        }
        let depth = self
            .nearest_3d_projected(plot_xy, f64::INFINITY)
            .map(|(_, d)| d)
            .unwrap_or(0.0);
        let inserted = self.unproject_3d(plot_xy, depth);
        self.mode = VrSourceMode::Points3D;
        self.chaos.playing = false;
        self.points_3d.push(inserted);
        self.update_complex();
        true
    }

    fn sculpt_erase_3d(&mut self, plot_xy: [f64; 2], r2: f64) -> bool {
        if self.points_3d.len() <= 1 {
            return false;
        }
        if let Some((idx, _)) = self.nearest_3d_projected(plot_xy, r2) {
            self.mode = VrSourceMode::Points3D;
            self.chaos.playing = false;
            self.points_3d.remove(idx);
            self.active_drag_vertex = None;
            self.update_complex();
            return true;
        }
        false
    }

    fn nearest_3d_projected(&self, plot_xy: [f64; 2], max_dist_sq: f64) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64, f64)> = None;
        for (i, &p) in self.points_3d.iter().enumerate() {
            let (proj, depth) = self.project_3d_with_depth(p);
            let d2 = (proj[0] - plot_xy[0]).powi(2) + (proj[1] - plot_xy[1]).powi(2);
            if d2 <= max_dist_sq && best.is_none_or(|(_, b, _)| d2 < b) {
                best = Some((i, d2, depth));
            }
        }
        best.map(|(idx, _, depth)| (idx, depth))
    }

    pub(super) fn handle_canvas_sculpt(
        &mut self,
        plot_ui: &egui_plot::PlotUi<'_>,
        is_3d: bool,
    ) -> Option<[f64; 2]> {
        let ptr = plot_ui.pointer_coordinate().map(|p| [p.x, p.y]);
        if self.brush == SculptBrush::OrbitOrPan {
            self.active_drag_vertex = None;
            return None;
        }
        let ctx = plot_ui.ctx().clone();
        let pid = plot_ui.plot_id();
        let (pressed, down) =
            ctx.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down()));
        if is_3d {
            ctx.data_mut(|d| {
                if down {
                    d.insert_temp(pid.with("dragging_callout"), true);
                } else {
                    d.remove_temp::<bool>(pid.with("dragging_callout"));
                }
            });
        }
        if !down {
            self.active_drag_vertex = None;
            return ptr;
        }
        if let Some(xy) = ptr {
            let continuing = !pressed;
            let r = if is_3d {
                self.brush_radius * 2.5
            } else {
                self.brush_radius
            };
            if is_3d {
                self.apply_sculpt_3d_projected(self.brush, xy, r, continuing);
            } else {
                self.apply_sculpt_2d(self.brush, xy, r, continuing);
            }
        }
        ptr
    }
}

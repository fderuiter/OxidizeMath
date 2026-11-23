//! Filtration builders, geometry caches, and zero-allocation workspace sweeps for `VietorisRipsTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{VrSourceMode, MAX_VR_RADIUS};
use super::{CloudSource, VietorisRipsTool, FILTRATION_RADII_STEPS, MAX_CHAOS_LANDMARKS};
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::tda::core::{NdPointCloud, Point2D, PointCloud};
use math_explorer::pure_math::statistics::tda::{
    compute_rips_persistence_from_distances, vietoris_rips_from_distance_matrix, Point3D,
    PointCloud3D, SimplicialComplex, MAX_FILTRATION_TRIANGLES,
};

impl VietorisRipsTool {
    pub(super) fn clear_cache(&mut self) {
        self.cached_triangles.clear();
        self.cached_edges.clear();
        self.cached_vertices.clear();
        self.cached_triangles_3d.clear();
        self.cached_edges_3d.clear();
        self.barcode_intervals.clear();
        self.betti = (0, 0);
    }

    pub(super) fn rebuild_from_2d(
        &mut self,
        source: CloudSource,
        explicit_pts: Vec<Point2D>,
        radius: f64,
    ) -> Result<(), String> {
        let pts = match &source {
            CloudSource::Custom2D(custom) => custom.clone(),
            _ => explicit_pts,
        };
        let cloud = PointCloud::new(pts.clone()).map_err(|e| format!("{e}"))?;
        let dist = cloud.distance_matrix();
        let persistence =
            compute_rips_persistence_from_distances(&dist).map_err(|e| format!("{e}"))?;
        let (complex, truncated) = self.build_complex_slice(&dist, radius)?;

        self.mode = VrSourceMode::Points2D;
        self.source = source;
        self.points = pts;
        self.distance_matrix = dist;
        self.max_slider_radius = (persistence.max_radius * 1.05).max(15.0).max(radius);
        self.persistence = persistence;
        self.radius = radius;
        self.sync_barcode_intervals_from_rips();
        self.apply_2d_complex(complex, truncated);
        self.refresh_url_budget_flag();
        self.bump_revision();
        Ok(())
    }

    pub(super) fn rebuild_from_nd(
        &mut self,
        cloud: NdPointCloud,
        axis_x: usize,
        axis_y: usize,
        radius: f64,
    ) -> Result<(), String> {
        let projected = cloud
            .project_2d(axis_x, axis_y)
            .map_err(|e| format!("{e}"))?;
        let dist = cloud.distance_matrix();
        let persistence =
            compute_rips_persistence_from_distances(&dist).map_err(|e| format!("{e}"))?;
        let (complex, truncated) = self.build_complex_slice(&dist, radius)?;

        self.mode = VrSourceMode::Points2D;
        self.source = CloudSource::CustomNd {
            cloud,
            axis_x,
            axis_y,
        };
        self.points = projected.points;
        self.distance_matrix = dist;
        self.max_slider_radius = (persistence.max_radius * 1.05).max(15.0).max(radius);
        self.persistence = persistence;
        self.radius = radius;
        self.sync_barcode_intervals_from_rips();
        self.apply_2d_complex(complex, truncated);
        self.refresh_url_budget_flag();
        self.bump_revision();
        Ok(())
    }

    /// Loads validated 3D points (`1..=MAX_CHAOS_LANDMARKS`), computes exact persistence, and updates 3D caches.
    pub fn load_validated_3d_points(
        &mut self,
        pts: Vec<Point3D>,
        radius: f64,
    ) -> Result<(), String> {
        if pts.len() > MAX_CHAOS_LANDMARKS {
            return Err(format!(
                "3D landmark count ({}) exceeds MAX_CHAOS_LANDMARKS ({MAX_CHAOS_LANDMARKS})",
                pts.len()
            ));
        }
        self.rebuild_from_3d(VrSourceMode::Points3D, pts, radius, true)
    }

    /// Rebuilds landmarks from `self.chaos` (`Lorenz3D` or `Takens3D`) and refreshes 3D filtration.
    pub fn rebuild_chaos_landmarks(&mut self) -> Result<(), String> {
        let pts = self.chaos.build_landmarks(self.mode)?;
        self.rebuild_from_3d(self.mode, pts, self.radius, true)
    }

    pub(super) fn refresh_sculpted_3d_points(&mut self) -> Result<(), String> {
        let pts = self.points_3d.clone();
        self.rebuild_from_3d(VrSourceMode::Points3D, pts, self.radius, false)
    }

    fn rebuild_from_3d(
        &mut self,
        mode: VrSourceMode,
        pts: Vec<Point3D>,
        radius: f64,
        recenter: bool,
    ) -> Result<(), String> {
        let cloud = PointCloud3D::new(pts.clone()).map_err(|e| format!("{e}"))?;
        let dist = cloud.distance_matrix();
        let persistence =
            compute_rips_persistence_from_distances(&dist).map_err(|e| format!("{e}"))?;
        let (complex, truncated) = self.build_complex_slice(&dist, radius)?;

        self.mode = mode;
        self.points_3d = pts;
        if recenter {
            self.update_view_center_3d();
        }
        self.distance_matrix = dist;
        self.max_slider_radius = (persistence.max_radius * 1.05)
            .max(MAX_VR_RADIUS)
            .max(radius);
        self.persistence = persistence;
        self.radius = radius;
        self.workspace
            .load_points_3d(&self.points_3d)
            .map_err(|e| format!("{e:?}"))?;
        self.fill_filtration_radii(self.points_3d.len());
        self.sync_barcode_intervals_from_rips();
        self.apply_3d_complex(complex, truncated);
        self.refresh_url_budget_flag();
        self.bump_revision();
        Ok(())
    }

    /// Evaluates active Betti numbers and 24-step persistence barcode in-place with zero heap allocations.
    #[cfg(test)]
    pub fn recompute_filtration_zero_alloc(
        &mut self,
        radius: f64,
    ) -> Result<(usize, usize), String> {
        let n = if self.mode.is_3d() {
            self.workspace
                .load_points_3d(&self.points_3d)
                .map_err(|e| format!("{e:?}"))?;
            self.points_3d.len()
        } else {
            self.workspace
                .load_points_2d(&self.points)
                .map_err(|e| format!("{e:?}"))?;
            self.points.len()
        };
        self.fill_filtration_radii(n);
        self.workspace
            .compute_persistence_into(n, &self.filtration_radii, &mut self.barcode_intervals)
            .map_err(|e| format!("{e:?}"))?;
        let step = self
            .workspace
            .evaluate_radius(n, radius)
            .map_err(|e| format!("{e:?}"))?;
        self.radius = radius;
        self.betti = (step.beta0, step.beta1);
        Ok(self.betti)
    }

    pub(super) fn build_complex_slice(
        &mut self,
        dist: &[Vec<f64>],
        radius: f64,
    ) -> Result<(SimplicialComplex, bool), String> {
        #[cfg(test)]
        {
            self.complex_builds += 1;
        }
        vietoris_rips_from_distance_matrix(dist, radius, MAX_FILTRATION_TRIANGLES)
            .map_err(|e| e.to_string())
    }

    fn sync_barcode_intervals_from_rips(&mut self) {
        self.barcode_intervals.clear();
        for iv in &self.persistence.intervals {
            if (iv.death > iv.birth || iv.is_essential) && iv.birth.is_finite() {
                if let Ok(mapped) = iv.to_persistence_interval() {
                    self.barcode_intervals.push(mapped);
                }
            }
        }
    }

    fn apply_2d_complex(&mut self, complex: SimplicialComplex, truncated: bool) {
        self.complex = complex;
        self.complex_truncated = truncated;
        self.betti = self.persistence.betti_numbers_at(self.radius);
        self.cached_triangles_3d.clear();
        self.cached_edges_3d.clear();
        self.refresh_plot_geometry();
        self.error_msg = None;
    }

    fn apply_3d_complex(&mut self, complex: SimplicialComplex, truncated: bool) {
        self.complex = complex;
        self.complex_truncated = truncated;
        self.betti = self.persistence.betti_numbers_at(self.radius);
        self.cached_triangles_3d = self
            .complex
            .get_simplices(2)
            .into_iter()
            .filter_map(|t| {
                Some([
                    *self.points_3d.get(t.vertices[0])?,
                    *self.points_3d.get(t.vertices[1])?,
                    *self.points_3d.get(t.vertices[2])?,
                ])
            })
            .collect();
        self.cached_edges_3d = self
            .complex
            .get_simplices(1)
            .into_iter()
            .filter_map(|e| {
                Some([
                    *self.points_3d.get(e.vertices[0])?,
                    *self.points_3d.get(e.vertices[1])?,
                ])
            })
            .collect();
        self.refresh_projected_3d_export_geometry();
        self.chaos.refresh_invariants(&self.points_3d, self.radius);
        self.error_msg = None;
    }

    pub(super) fn refresh_projected_3d_export_geometry(&mut self) {
        self.cached_triangles = self
            .cached_triangles_3d
            .iter()
            .map(|&[a, b, c]| {
                let (pa, pb, pc) = (self.project_3d(a), self.project_3d(b), self.project_3d(c));
                [
                    PlotPoint::new(pa[0], pa[1]),
                    PlotPoint::new(pb[0], pb[1]),
                    PlotPoint::new(pc[0], pc[1]),
                ]
            })
            .collect();
        self.cached_edges = self
            .cached_edges_3d
            .iter()
            .map(|&[a, b]| {
                let (pa, pb) = (self.project_3d(a), self.project_3d(b));
                [PlotPoint::new(pa[0], pa[1]), PlotPoint::new(pb[0], pb[1])]
            })
            .collect();
        self.cached_vertices = self
            .points_3d
            .iter()
            .map(|&p| {
                let xy = self.project_3d(p);
                PlotPoint::new(xy[0], xy[1])
            })
            .collect();
    }

    pub(super) fn refresh_plot_geometry(&mut self) {
        self.cached_triangles = self
            .complex
            .get_simplices(2)
            .into_iter()
            .filter_map(|t| {
                let (p0, p1, p2) = (
                    self.points.get(t.vertices[0])?,
                    self.points.get(t.vertices[1])?,
                    self.points.get(t.vertices[2])?,
                );
                Some([
                    PlotPoint::new(p0.x, p0.y),
                    PlotPoint::new(p1.x, p1.y),
                    PlotPoint::new(p2.x, p2.y),
                ])
            })
            .collect();
        self.cached_edges = self
            .complex
            .get_simplices(1)
            .into_iter()
            .filter_map(|e| {
                let (p0, p1) = (
                    self.points.get(e.vertices[0])?,
                    self.points.get(e.vertices[1])?,
                );
                Some([PlotPoint::new(p0.x, p0.y), PlotPoint::new(p1.x, p1.y)])
            })
            .collect();
        self.cached_vertices = self
            .points
            .iter()
            .map(|p| PlotPoint::new(p.x, p.y))
            .collect();
    }

    pub(super) fn update_complex(&mut self) {
        self.bump_revision();
        let active_len = if self.mode.is_3d() {
            self.points_3d.len()
        } else {
            self.points.len()
        };
        if active_len == 0 || self.distance_matrix.len() != active_len {
            self.clear_cache();
            self.error_msg = Some("Error building point cloud: empty point cloud".to_string());
            return;
        }
        let dist = self.distance_matrix.clone();
        match self.build_complex_slice(&dist, self.radius) {
            Ok((complex, truncated)) => {
                if self.mode.is_3d() {
                    self.apply_3d_complex(complex, truncated);
                } else {
                    self.apply_2d_complex(complex, truncated);
                }
            }
            Err(e) => {
                self.clear_cache();
                self.error_msg = Some(format!("Error building complex: {e}"));
            }
        }
    }

    fn fill_filtration_radii(&mut self, n: usize) {
        let mut max_d = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.workspace.distance(n, i, j);
                if d > max_d {
                    max_d = d;
                }
            }
        }
        let max_scale = if self.mode.is_3d() {
            (max_d * 0.75).max(self.radius * 1.15).clamp(4.0, 28.0)
        } else {
            (max_d * 0.85).max(self.radius * 1.15).clamp(2.0, 15.0)
        };
        let denom = (FILTRATION_RADII_STEPS - 1) as f64;
        for (idx, slot) in self.filtration_radii.iter_mut().enumerate() {
            *slot = max_scale * (idx as f64 / denom);
        }
    }

    pub(super) fn update_view_center_3d(&mut self) {
        if self.points_3d.is_empty() {
            self.view_center_3d = [0.0, 0.0, 0.0];
            return;
        }
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in &self.points_3d {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
            min_z = min_z.min(p.z);
            max_z = max_z.max(p.z);
        }
        self.view_center_3d = [
            0.5 * (min_x + max_x),
            0.5 * (min_y + max_y),
            0.5 * (min_z + max_z),
        ];
    }
}

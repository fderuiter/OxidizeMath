#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::VrSourceMode;
use super::{FILTRATION_RADII_STEPS, VietorisRipsTool};
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::tda::{
    Point2D, Point3D, PointCloud, PointCloud3D, SimplicialComplex, vietoris_rips_complex,
    vietoris_rips_complex_3d,
};

impl VietorisRipsTool {
    pub(super) fn clear_cache(&mut self) {
        self.cached_triangles.clear();
        self.cached_edges.clear();
        self.cached_vertices.clear();
        self.cached_triangles_3d.clear();
        self.cached_edges_3d.clear();
        self.barcode_intervals.clear();
        self.beta_0 = 0;
        self.beta_1 = 0;
    }

    /// Loads validated 2D points, computes exact filtration + 24-step barcode, and updates caches.
    pub fn load_validated_2d_points(
        &mut self,
        pts: Vec<Point2D>,
        radius: f64,
    ) -> Result<(), String> {
        let cloud = PointCloud::new(pts.clone()).map_err(|e| format!("{e:?}"))?;
        let complex = self.compute_complex_2d(&cloud, radius)?;
        self.mode = VrSourceMode::Points2D;
        self.radius = radius;
        self.points = pts;
        self.revision = self.revision.wrapping_add(1).max(1);
        self.apply_2d_complex_and_barcode(complex)?;
        Ok(())
    }

    /// Loads validated 3D points, computes exact filtration + 24-step barcode, and updates caches.
    pub fn load_validated_3d_points(
        &mut self,
        pts: Vec<Point3D>,
        radius: f64,
    ) -> Result<(), String> {
        let cloud = PointCloud3D::new(pts.clone()).map_err(|e| format!("{e:?}"))?;
        let complex = self.compute_complex_3d(&cloud, radius)?;
        self.mode = VrSourceMode::Points3D;
        self.radius = radius;
        self.points_3d = pts;
        self.update_view_center_3d();
        self.revision = self.revision.wrapping_add(1).max(1);
        self.apply_3d_complex_and_barcode(complex)?;
        Ok(())
    }

    /// Rebuilds landmarks from `self.chaos` (`Lorenz3D` or `Takens3D`) and refreshes filtration.
    pub fn rebuild_chaos_landmarks(&mut self) -> Result<(), String> {
        let pts = self.chaos.build_landmarks(self.mode)?;
        let cloud = PointCloud3D::new(pts.clone()).map_err(|e| format!("{e:?}"))?;
        let complex = self.compute_complex_3d(&cloud, self.radius)?;
        self.points_3d = pts;
        self.update_view_center_3d();
        self.revision = self.revision.wrapping_add(1).max(1);
        self.apply_3d_complex_and_barcode(complex)?;
        Ok(())
    }

    /// Evaluates active Betti numbers and 24-step persistence barcode in-place with zero heap allocations.
    pub fn recompute_filtration_zero_alloc(&mut self, radius: f64) -> Result<(usize, usize), String> {
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
        self.beta_0 = step.beta0;
        self.beta_1 = step.beta1;
        Ok((step.beta0, step.beta1))
    }

    pub(super) fn update_complex(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
        if self.mode.is_3d() {
            self.update_complex_3d_state();
        } else {
            self.update_complex_2d_state();
        }
    }

    fn update_complex_2d_state(&mut self) {
        match PointCloud::new(self.points.clone()) {
            Ok(cloud) => match self.compute_complex_2d(&cloud, self.radius) {
                Ok(complex) => {
                    if let Err(e) = self.apply_2d_complex_and_barcode(complex) {
                        self.clear_cache();
                        self.error_msg = Some(e);
                    }
                }
                Err(e) => {
                    self.clear_cache();
                    self.error_msg = Some(format!("Error building complex: {e}"));
                }
            },
            Err(e) => {
                self.clear_cache();
                self.error_msg = Some(format!("Error building point cloud: {e}"));
            }
        }
    }

    fn update_complex_3d_state(&mut self) {
        match PointCloud3D::new(self.points_3d.clone()) {
            Ok(cloud) => match self.compute_complex_3d(&cloud, self.radius) {
                Ok(complex) => {
                    if let Err(e) = self.apply_3d_complex_and_barcode(complex) {
                        self.clear_cache();
                        self.error_msg = Some(e);
                    }
                }
                Err(e) => {
                    self.clear_cache();
                    self.error_msg = Some(format!("Error building 3D complex: {e}"));
                }
            },
            Err(e) => {
                self.clear_cache();
                self.error_msg = Some(format!("Error building 3D point cloud: {e}"));
            }
        }
    }

    fn compute_complex_2d(
        &mut self,
        cloud: &PointCloud,
        radius: f64,
    ) -> Result<SimplicialComplex, String> {
        #[cfg(test)]
        {
            self.complex_builds += 1;
        }
        vietoris_rips_complex(cloud, radius).map_err(|e| format!("{e:?}"))
    }

    fn compute_complex_3d(
        &mut self,
        cloud: &PointCloud3D,
        radius: f64,
    ) -> Result<SimplicialComplex, String> {
        #[cfg(test)]
        {
            self.complex_builds += 1;
        }
        vietoris_rips_complex_3d(cloud, radius).map_err(|e| format!("{e:?}"))
    }

    fn apply_2d_complex_and_barcode(&mut self, complex: SimplicialComplex) -> Result<(), String> {
        let _ = self.recompute_filtration_zero_alloc(self.radius)?;

        self.cached_triangles = self
            .workspace
            .active_triangles()
            .iter()
            .map(|&[i, j, k]| {
                let (p0, p1, p2) = (&self.points[i], &self.points[j], &self.points[k]);
                [
                    PlotPoint::new(p0.x, p0.y),
                    PlotPoint::new(p1.x, p1.y),
                    PlotPoint::new(p2.x, p2.y),
                ]
            })
            .collect();
        self.cached_edges = self
            .workspace
            .active_edges()
            .iter()
            .map(|&[i, j]| {
                let (p0, p1) = (&self.points[i], &self.points[j]);
                [PlotPoint::new(p0.x, p0.y), PlotPoint::new(p1.x, p1.y)]
            })
            .collect();
        self.cached_vertices = self
            .points
            .iter()
            .map(|p| PlotPoint::new(p.x, p.y))
            .collect();
        self.cached_triangles_3d.clear();
        self.cached_edges_3d.clear();
        self.complex = complex;
        self.error_msg = None;
        Ok(())
    }

    fn apply_3d_complex_and_barcode(&mut self, complex: SimplicialComplex) -> Result<(), String> {
        let _ = self.recompute_filtration_zero_alloc(self.radius)?;

        self.cached_triangles_3d = self
            .workspace
            .active_triangles()
            .iter()
            .map(|&[i, j, k]| [self.points_3d[i], self.points_3d[j], self.points_3d[k]])
            .collect();
        self.cached_edges_3d = self
            .workspace
            .active_edges()
            .iter()
            .map(|&[i, j]| [self.points_3d[i], self.points_3d[j]])
            .collect();
        self.cached_triangles.clear();
        self.cached_edges.clear();
        self.cached_vertices.clear();
        self.chaos.refresh_invariants(&self.points_3d, self.radius);
        self.complex = complex;
        self.error_msg = None;
        Ok(())
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

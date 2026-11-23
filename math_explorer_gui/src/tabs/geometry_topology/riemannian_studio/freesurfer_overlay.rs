#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! FreeSurfer cortical morphometry (`Surface`, `cortical_thickness_batch`, `cortical_thickness`,
//! `bayesian_classification`, `internal_energy`, `external_energy`, `evolve_surface`) and
//! mass-univariate group GLM (`estimate_beta`, `t_statistic`) overlays.

use super::mesh_gen::{cortical_shell_point_and_thickness, ManifoldGeometry};
use super::state::{FreeSurferOverlayKind, ManifoldKind, RiemannianStudioConfig};
use math_explorer::applied::freesurfer::{
    bayesian_classification, cortical_thickness, cortical_thickness_batch, estimate_beta,
    evolve_surface, external_energy, internal_energy, t_statistic, Surface,
};
use nalgebra::{DMatrix, DVector};

/// Cached FreeSurfer morphometry and multi-subject GLM maps across the active 3D manifold.
pub struct FreeSurferMorphometryState {
    /// Inner white-matter boundary surface.
    pub white_surface: Surface,
    /// Outer pial boundary surface (deformable via `evolve_surface`).
    pub pial_surface: Surface,
    /// Vertex-wise symmetric white-to-pial thickness (mm) via `cortical_thickness_batch`.
    pub thickness_mm: Vec<f64>,
    /// Single-vertex cursor probe thickness from `cortical_thickness` (mm).
    pub cursor_probe_thickness_mm: f64,
    /// Mass-univariate group contrast `t_statistic` values per vertex.
    pub glm_t_stat: Vec<f64>,
    /// Mass-univariate group contrast effect size `estimate_beta` values per vertex.
    pub glm_beta_contrast: Vec<f64>,
    /// Bayesian tissue class index (`0 = CSF`, `1 = GM`, `2 = WM`) mapped to scalar `[0.0, 2.0]`.
    pub bayesian_class_map: Vec<f64>,
    /// Active pial `internal_energy` readout.
    pub pial_internal_energy: f64,
    /// Active pial `external_energy` readout.
    pub pial_external_energy: f64,
}

impl FreeSurferMorphometryState {
    /// Recomputes white/pial surfaces, `cortical_thickness_batch`, Bayesian tissue priors,
    /// surface energies, and 12-subject group GLM contrast maps (`estimate_beta` and `t_statistic`).
    #[must_use]
    pub fn build(
        geom: &ManifoldGeometry,
        cfg: &RiemannianStudioConfig,
        cursor_vertex: usize,
    ) -> Self {
        let n = geom.positions.len();
        let mut white_verts = Vec::with_capacity(n);
        let mut pial_verts = Vec::with_capacity(n);

        for idx in 0..n {
            if cfg.manifold == ManifoldKind::FoldedCorticalShell {
                let (u, v) = geom.uvs[idx];
                let (w, p, _t_mm) = cortical_shell_point_and_thickness(u, v, cfg.sulcal_depth);
                // Scale into millimeter-equivalent morphometry coordinates (~28.57x) so 1 unit = 1 mm.
                white_verts.push([w[0] * 28.57, w[1] * 28.57, w[2] * 28.57]);
                pial_verts.push([p[0] * 28.57, p[1] * 28.57, p[2] * 28.57]);
            } else {
                let pos = geom.positions[idx];
                let nrm = geom.normals[idx];
                let h_mod = (geom.mean_curvature[idx] - 0.75).clamp(-0.5, 0.8);
                let thick_mm = (2.45 + 0.65 * h_mod).clamp(1.4, 3.8);
                let w_mm = [pos[0] * 25.0, pos[1] * 25.0, pos[2] * 25.0];
                let p_mm = [
                    w_mm[0] + nrm[0] * thick_mm,
                    w_mm[1] + nrm[1] * thick_mm,
                    w_mm[2] + nrm[2] * thick_mm,
                ];
                white_verts.push(w_mm);
                pial_verts.push(p_mm);
            }
        }

        let white_surface = Surface {
            vertices: white_verts,
        };
        let pial_surface = Surface {
            vertices: pial_verts,
        };

        let mut state = Self {
            white_surface,
            pial_surface,
            thickness_mm: Vec::new(),
            cursor_probe_thickness_mm: 0.0,
            glm_t_stat: Vec::new(),
            glm_beta_contrast: Vec::new(),
            bayesian_class_map: Vec::new(),
            pial_internal_energy: 0.0,
            pial_external_energy: 0.0,
        };
        state.recompute_metrics(geom, cfg, cursor_vertex);
        state
    }

    /// Re-evaluates thickness, energies, Bayesian segmentation, and GLM contrast after pial evolution.
    pub fn recompute_metrics(
        &mut self,
        geom: &ManifoldGeometry,
        cfg: &RiemannianStudioConfig,
        cursor_vertex: usize,
    ) {
        self.thickness_mm = cortical_thickness_batch(&self.white_surface, &self.pial_surface);
        let probe = cursor_vertex.min(self.white_surface.vertices.len().saturating_sub(1));
        self.cursor_probe_thickness_mm = if self.white_surface.vertices.is_empty() {
            0.0
        } else {
            cortical_thickness(
                &self.white_surface.vertices[probe],
                &self.pial_surface.vertices[probe],
                &self.white_surface,
                &self.pial_surface,
            )
        };

        self.pial_internal_energy = internal_energy(&self.pial_surface, 0.35, 0.15);
        self.pial_external_energy = external_energy(&self.pial_surface, cfg.fs_grad_strength);

        self.bayesian_class_map = compute_bayesian_tissue_map(geom, &self.thickness_mm);
        let (betas, t_stats) = compute_cohort_glm(geom, &self.thickness_mm);
        self.glm_beta_contrast = betas;
        self.glm_t_stat = t_stats;
    }

    /// Evolves the deformable pial surface one step via `freesurfer::evolve_surface` and updates morphometry.
    pub fn step_pial_evolution(
        &mut self,
        geom: &ManifoldGeometry,
        cfg: &RiemannianStudioConfig,
        cursor_vertex: usize,
    ) {
        evolve_surface(&mut self.pial_surface, cfg.dt * 4.0, cfg.fs_grad_strength);
        self.recompute_metrics(geom, cfg, cursor_vertex);
    }

    /// Updates the single-vertex `cortical_thickness` probe when the geodesic cursor moves.
    pub fn refresh_cursor_probe(&mut self, cursor_vertex: usize) {
        if self.white_surface.vertices.is_empty() {
            self.cursor_probe_thickness_mm = 0.0;
            return;
        }
        let idx = cursor_vertex.min(self.white_surface.vertices.len() - 1);
        self.cursor_probe_thickness_mm = cortical_thickness(
            &self.white_surface.vertices[idx],
            &self.pial_surface.vertices[idx],
            &self.white_surface,
            &self.pial_surface,
        );
    }

    /// Copies the selected FreeSurfer overlay into `out`.
    pub fn write_overlay(&self, kind: FreeSurferOverlayKind, out: &mut [f64]) {
        let src = match kind {
            FreeSurferOverlayKind::CorticalThickness => &self.thickness_mm,
            FreeSurferOverlayKind::GlmTStatistic => &self.glm_t_stat,
            FreeSurferOverlayKind::GlmBetaContrast => &self.glm_beta_contrast,
            FreeSurferOverlayKind::BayesianTissueClass => &self.bayesian_class_map,
        };
        let len = out.len().min(src.len());
        out[..len].copy_from_slice(&src[..len]);
    }
}

fn compute_bayesian_tissue_map(geom: &ManifoldGeometry, thickness_mm: &[f64]) -> Vec<f64> {
    // Priors and Gaussian T1 likelihood parameters for [CSF, Gray Matter, White Matter].
    let priors = [0.20, 0.45, 0.35];
    let means = [32.0, 76.0, 108.0];
    let std_devs = [8.0, 10.0, 9.0];
    let mut classes = Vec::with_capacity(geom.positions.len());

    for (idx, pos) in geom.positions.iter().enumerate() {
        let thick = thickness_mm.get(idx).copied().unwrap_or(2.5);
        // Synthetic T1 intensity modulated smoothly by local cortical thickness and sulcal depth.
        let t1_intensity = 42.0 + 22.0 * thick + 14.0 * pos[2];
        let mut best_class = 1_usize;
        let mut best_post = f64::NEG_INFINITY;
        for c in 0..3 {
            let z = (t1_intensity - means[c]) / std_devs[c];
            let likelihood = (-0.5 * z * z).exp() / std_devs[c];
            let post = bayesian_classification(likelihood, priors[c]);
            if post > best_post {
                best_post = post;
                best_class = c;
            }
        }
        classes.push(best_class as f64);
    }
    classes
}

fn compute_cohort_glm(geom: &ManifoldGeometry, thickness_mm: &[f64]) -> (Vec<f64>, Vec<f64>) {
    // 12-subject design matrix X: column 0 = intercept (1.0), column 1 = group indicator (-0.5 control, +0.5 patient).
    let n_subj = 12_usize;
    let mut x_data = Vec::with_capacity(n_subj * 2);
    for s in 0..n_subj {
        x_data.push(1.0);
        x_data.push(if s < 6 { -0.5 } else { 0.5 });
    }
    let design = DMatrix::from_row_slice(n_subj, 2, &x_data);
    let contrast = DVector::from_vec(vec![0.0, 1.0]);

    let n_verts = geom.positions.len();
    let mut betas = Vec::with_capacity(n_verts);
    let mut t_stats = Vec::with_capacity(n_verts);

    for (v, pos) in geom.positions.iter().enumerate() {
        let base_t = thickness_mm.get(v).copied().unwrap_or(2.5);
        // Focal cortical thinning effect in fronto-temporal lobe region (x > 0.2) vs mild thickening anteriorly.
        let focal_effect = -0.42 * (-2.2 * ((pos[0] - 0.65).powi(2) + pos[1].powi(2))).exp()
            + 0.18 * (-2.5 * ((pos[0] + 0.65).powi(2) + (pos[2] - 0.35).powi(2))).exp();
        let mut y_vals = Vec::with_capacity(n_subj);
        for s in 0..n_subj {
            let group = if s < 6 { -0.5 } else { 0.5 };
            // Deterministic subject-by-vertex biological residual (~0.06 mm SD).
            let phase = ((s + 1) * 13 + (v + 3) * 7) as f64;
            let residual = 0.055 * phase.sin() + 0.025 * (1.7 * phase).cos();
            y_vals.push(base_t + group * focal_effect + residual);
        }
        let y_vec = DVector::from_vec(y_vals);
        let (beta_val, t_val) = if let Ok(beta_vec) = estimate_beta(&design, &y_vec) {
            let resid = &y_vec - &design * &beta_vec;
            let dof = (n_subj - 2) as f64;
            let sigma_sq = (resid.dot(&resid) / dof).max(1e-12);
            let b1 = beta_vec.get(1).copied().unwrap_or(0.0);
            let t = t_statistic(&contrast, &beta_vec, &design, sigma_sq).unwrap_or(0.0);
            (b1, t)
        } else {
            (0.0, 0.0)
        };
        betas.push(if beta_val.is_finite() { beta_val } else { 0.0 });
        t_stats.push(if t_val.is_finite() {
            t_val.clamp(-25.0, 25.0)
        } else {
            0.0
        });
    }
    (betas, t_stats)
}

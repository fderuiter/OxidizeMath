#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Synthetic genus-0 folded/inflated cortical hemisphere generator, deformable surface
//! evolution driver, and cached morphometry orchestrator.

#[path = "volume_glm.rs"]
mod volume_glm;

use super::CorticalMorphometryTool;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::InteractiveTool;
use math_explorer::applied::freesurfer::{
    cortical_thickness_map_into, evolve_surface_deformable, external_energy_volume,
    internal_energy, mrf_segment_volume, MriVolume, Surface, ThicknessWorkspace,
};
use nalgebra::Vector3;
use std::f64::consts::{PI, TAU};
use std::hash::{Hash as _, Hasher as _};
#[cfg(test)]
pub(super) use volume_glm::evaluate_cohort_glm;
#[cfg(not(test))]
use volume_glm::evaluate_cohort_glm;
use volume_glm::{sample_gm_posteriors_on_cortex, synthesize_mri_and_truth, tissue_classes};

pub(super) const MRI_DIMS: [usize; 3] = [16, 16, 16];
pub(super) const MRI_ORIGIN_MM: [f64; 3] = [-45.0, -45.0, -45.0];
pub(super) const MRI_VOXEL_MM: f64 = 6.0;
pub(super) const SEMI_AXES_MM: [f64; 3] = [21.0, 27.0, 19.5];

#[derive(Default)]
pub(super) struct CorticalPhantomCache {
    pub(super) compute_key: Option<u64>,
    pub(super) proj_key: Option<(u64, u64, u8, u8, u64)>,
    pub(super) white_surface: Option<Surface>,
    pub(super) pial_surface: Option<Surface>,
    pub(super) white_inflated: Vec<[f64; 3]>,
    pub(super) pial_inflated: Vec<[f64; 3]>,
    pub(super) sulcal_depth_mm: Vec<f64>,
    pub(super) thickness_mm: Vec<f64>,
    pub(super) thickness_workspace: ThicknessWorkspace,
    pub(super) glm_t_stats: Vec<f64>,
    pub(super) gm_posteriors: Vec<f64>,
    pub(super) volume: Option<MriVolume>,
    pub(super) mrf_labels: Vec<u8>,
    pub(super) ml_accuracy: f64,
    pub(super) mrf_accuracy: f64,
    pub(super) mean_thickness_mm: f64,
    pub(super) internal_energy_val: f64,
    pub(super) external_energy_val: f64,
    pub(super) total_energy: f64,
    pub(super) peak_abs_t_stat: f64,
    pub(super) glm_warning: Option<String>,
}

impl CorticalPhantomCache {
    pub(super) fn invalidate(&mut self) {
        self.compute_key = None;
        self.proj_key = None;
    }

    pub(super) fn evaluate_thickness(&mut self, white: &Surface, pial: &Surface) {
        let pair_count = white.vertices.len().min(pial.vertices.len());
        self.thickness_workspace
            .ensure_capacity(white.vertices.len().max(pial.vertices.len()));
        self.thickness_mm.resize(pair_count, 0.0);
        let _ = cortical_thickness_map_into(
            white,
            pial,
            &mut self.thickness_workspace,
            &mut self.thickness_mm,
        );
    }

    pub(super) fn are_meshes_finite(&self) -> bool {
        let surface_ok = |s: &Option<Surface>| {
            s.as_ref().is_none_or(|surf| {
                surf.vertices
                    .iter()
                    .all(|v| v[0].is_finite() && v[1].is_finite() && v[2].is_finite())
            })
        };
        surface_ok(&self.white_surface) && surface_ok(&self.pial_surface)
    }
}

pub(super) fn canonical_bits(v: f64) -> u64 {
    if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

pub(super) fn gyral_and_atrophy_pattern(u: f64, v: f64) -> (f64, f64) {
    let lat = v.sin().powi(2);
    let gyral = 0.48 * (3.0 * u + 1.35 * v.cos()).cos()
        + 0.32 * (5.0 * u - 2.15 * v).sin()
        + 0.20 * (2.0 * u + 3.4 * v).cos();
    let du = (u - 2.15).sin();
    let dv = v - 1.55;
    let atrophy_focus = (-2.6 * (du * du + dv * dv)).exp() * lat;
    (lat * gyral, atrophy_focus)
}

fn hemisphere_point(u: f64, v: f64, radial_offset_mm: f64) -> [f64; 3] {
    let (sv, cv) = v.sin_cos();
    let (su, cu) = u.sin_cos();
    let dir = Vector3::new(
        SEMI_AXES_MM[0] * sv * cu,
        SEMI_AXES_MM[1] * sv * su,
        SEMI_AXES_MM[2] * cv,
    );
    let unit = dir
        .try_normalize(1e-12)
        .unwrap_or_else(|| Vector3::new(0.0, 0.0, 1.0));
    let pos = dir + unit * radial_offset_mm;
    [pos.x, pos.y, pos.z]
}

/// Builds closed genus-0 welded-pole triangle faces with outward counter-clockwise winding.
pub(super) fn build_polar_faces(nu: usize, nv: usize) -> Vec<[usize; 3]> {
    let ring = |j: usize, i: usize| -> usize { 2 + (j - 1) * nu + (i % nu) };
    let mut faces = Vec::with_capacity(2 * (nv - 1) * nu);
    for i in 0..nu {
        faces.push([0, ring(1, i), ring(1, i + 1)]);
    }
    for j in 1..(nv - 1) {
        for i in 0..nu {
            let (r00, r10, r11, r01) = (
                ring(j, i),
                ring(j + 1, i),
                ring(j + 1, i + 1),
                ring(j, i + 1),
            );
            faces.push([r00, r10, r11]);
            faces.push([r00, r11, r01]);
        }
    }
    for i in 0..nu {
        faces.push([1, ring(nv - 1, i + 1), ring(nv - 1, i)]);
    }
    faces
}

struct GeneratedSurfaces {
    white: Surface,
    pial: Surface,
    white_inflated: Vec<[f64; 3]>,
    pial_inflated: Vec<[f64; 3]>,
    sulcal_depth_mm: Vec<f64>,
    atrophy_weights: Vec<f64>,
}

fn generate_cortical_surfaces(
    nu: usize,
    nv: usize,
    sulcal_amp_mm: f64,
    base_thick_mm: f64,
) -> GeneratedSurfaces {
    let n_verts = 2 + (nv - 1) * nu;
    let mut white_v = Vec::with_capacity(n_verts);
    let mut pial_v = Vec::with_capacity(n_verts);
    let mut white_inf = Vec::with_capacity(n_verts);
    let mut pial_inf = Vec::with_capacity(n_verts);
    let mut sulcal = Vec::with_capacity(n_verts);
    let mut atrophy = Vec::with_capacity(n_verts);

    let mut push_uv = |u: f64, v: f64| {
        let (fold, focus) = gyral_and_atrophy_pattern(u, v);
        let w_off = sulcal_amp_mm * fold;
        let local_t = (base_thick_mm * (1.0 + 0.26 * fold)).max(0.9);
        white_v.push(hemisphere_point(u, v, w_off));
        pial_v.push(hemisphere_point(u, v, w_off + local_t));
        white_inf.push(hemisphere_point(u, v, 0.0));
        pial_inf.push(hemisphere_point(u, v, base_thick_mm));
        sulcal.push(-w_off);
        atrophy.push(focus);
    };

    push_uv(0.0, 0.0);
    push_uv(0.0, PI);
    for j in 1..nv {
        let v = PI * (j as f64 / nv as f64);
        for i in 0..nu {
            push_uv(TAU * (i as f64 / nu as f64), v);
        }
    }

    let faces = build_polar_faces(nu, nv);
    GeneratedSurfaces {
        white: Surface::new(white_v, faces.clone()),
        pial: Surface::new(pial_v, faces),
        white_inflated: white_inf,
        pial_inflated: pial_inf,
        sulcal_depth_mm: sulcal,
        atrophy_weights: atrophy,
    }
}

impl CorticalMorphometryTool {
    pub(super) fn compute_fingerprint(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.u_resolution.hash(&mut h);
        self.v_resolution.hash(&mut h);
        canonical_bits(self.sulcal_depth_mm).hash(&mut h);
        canonical_bits(self.base_thickness_mm).hash(&mut h);
        canonical_bits(self.alpha).hash(&mut h);
        canonical_bits(self.beta).hash(&mut h);
        canonical_bits(self.sigma).hash(&mut h);
        canonical_bits(self.learning_rate).hash(&mut h);
        self.evolution_steps.hash(&mut h);
        canonical_bits(self.potts_beta).hash(&mut h);
        canonical_bits(self.mri_noise_level).hash(&mut h);
        canonical_bits(self.atrophy_effect_mm).hash(&mut h);
        self.collinear_glm_design.hash(&mut h);
        h.finish().max(1)
    }

    pub(super) fn full_state_fingerprint(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.compute_fingerprint().hash(&mut h);
        (self.overlay_mode as u8).hash(&mut h);
        (self.surface_mode as u8).hash(&mut h);
        canonical_bits(self.inflation).hash(&mut h);
        h.finish().max(1)
    }

    fn evolve_white_surface_steps(&mut self, white: &mut Surface, volume: &MriVolume) {
        for step in 0..self.evolution_steps.min(25) {
            let snapshot = white.vertices.clone();
            evolve_surface_deformable(
                white,
                volume,
                self.alpha,
                self.beta,
                self.sigma,
                self.learning_rate,
            );
            if white
                .vertices
                .iter()
                .any(|p| !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite())
            {
                white.vertices = snapshot;
                self.diagnostic_report = Some(DiagnosticReport::new(
                    step as f64,
                    step,
                    "white_surface.vertices",
                    "Non-finite vertex encountered during deformable evolution; rolled back to last finite mesh.",
                ));
                break;
            }
        }
    }

    pub(super) fn ensure_evaluated(&mut self) {
        let key = self.compute_fingerprint();
        if self.cache.compute_key == Some(key)
            && self.cache.white_surface.is_some()
            && self.cache.pial_surface.is_some()
        {
            return;
        }
        let (nu, nv) = (
            self.u_resolution.clamp(12, 48),
            self.v_resolution.clamp(12, 36),
        );
        let mut gen =
            generate_cortical_surfaces(nu, nv, self.sulcal_depth_mm, self.base_thickness_mm);
        let (volume, truth) = synthesize_mri_and_truth(
            self.sulcal_depth_mm,
            self.base_thickness_mm,
            self.mri_noise_level,
        );
        let classes = tissue_classes();
        let (mut ml_labels, mut mrf_labels) = (vec![0_u8; truth.len()], vec![0_u8; truth.len()]);
        let _ = mrf_segment_volume(&volume, &classes, 0.0, 0, &mut ml_labels);
        let _ = mrf_segment_volume(&volume, &classes, self.potts_beta, 5, &mut mrf_labels);
        let acc = |pred: &[u8]| -> f64 {
            pred.iter().zip(&truth).filter(|(a, b)| a == b).count() as f64
                / truth.len().max(1) as f64
        };
        self.diagnostic_report = None;
        self.evolve_white_surface_steps(&mut gen.white, &volume);
        self.cache.evaluate_thickness(&gen.white, &gen.pial);
        let e_int = internal_energy(&gen.white, self.alpha, self.beta);
        let e_ext = external_energy_volume(&gen.white, &volume, self.sigma);
        let gm_post = sample_gm_posteriors_on_cortex(
            &gen.white,
            &gen.pial,
            &volume,
            &mrf_labels,
            self.potts_beta,
        );
        let glm = evaluate_cohort_glm(
            &self.cache.thickness_mm,
            &gen.atrophy_weights,
            self.atrophy_effect_mm,
            self.collinear_glm_design,
        );
        self.store_evaluation_cache(
            key,
            (gen, volume, gm_post),
            (acc(&ml_labels), acc(&mrf_labels), mrf_labels),
            (e_int, e_ext, glm),
        );
    }

    fn store_evaluation_cache(
        &mut self,
        key: u64,
        geom: (GeneratedSurfaces, MriVolume, Vec<f64>),
        mrf: (f64, f64, Vec<u8>),
        stats: (f64, f64, (Vec<f64>, f64, Option<String>)),
    ) {
        let (gen, volume, gm_post) = geom;
        self.cache.mean_thickness_mm = self.cache.thickness_mm.iter().sum::<f64>()
            / self.cache.thickness_mm.len().max(1) as f64;
        self.cache.white_surface = Some(gen.white);
        self.cache.pial_surface = Some(gen.pial);
        self.cache.white_inflated = gen.white_inflated;
        self.cache.pial_inflated = gen.pial_inflated;
        self.cache.sulcal_depth_mm = gen.sulcal_depth_mm;
        self.cache.gm_posteriors = gm_post;
        self.cache.ml_accuracy = mrf.0;
        self.cache.mrf_accuracy = mrf.1;
        self.cache.mrf_labels = mrf.2;
        self.cache.volume = Some(volume);
        self.cache.internal_energy_val = stats.0;
        self.cache.external_energy_val = stats.1;
        self.cache.total_energy = stats.0 + stats.1;
        self.cache.glm_t_stats = (stats.2).0;
        self.cache.peak_abs_t_stat = (stats.2).1;
        self.cache.glm_warning = (stats.2).2;
        self.cache.compute_key = Some(key);
    }

    pub(super) fn step_deformable_guarded(&mut self) {
        if !self.has_finite_parameters() {
            FiniteStateGuard::clamp_parameters(self);
            return;
        }
        self.evolution_steps = if self.evolution_steps >= 25 {
            0
        } else {
            self.evolution_steps + 1
        };
        self.ensure_evaluated();
    }
}

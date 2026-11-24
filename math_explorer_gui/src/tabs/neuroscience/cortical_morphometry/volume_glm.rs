#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Synthetic 3D T1 MRI volume generator, Potts MRF gray-matter posterior sampler,
//! and multi-subject general linear model (GLM) cohort contrast evaluator.

use super::{MRI_DIMS, MRI_ORIGIN_MM, MRI_VOXEL_MM, SEMI_AXES_MM, gyral_and_atrophy_pattern};
use math_explorer::applied::freesurfer::{
    GlmDesignQr, MriVolume, Surface, TissueClass, mrf_voxel_posteriors,
};
use nalgebra::{DMatrix, DVector, Vector3};
use std::f64::consts::TAU;

fn deterministic_voxel_noise(ix: usize, iy: usize, iz: usize) -> f64 {
    let hash_u01 = |seed: u64| -> f64 {
        let mut z = (ix as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((iy as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
            .wrapping_add((iz as u64).wrapping_mul(0x94D0_49BB_1331_11EB))
            .wrapping_add(seed);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    };
    let sum3 = hash_u01(0x2545_F491_4F6C_DD1D)
        + hash_u01(0x517C_C1B7_2722_0A95)
        + hash_u01(0x9E37_79B1_85EB_CA87);
    (sum3 - 1.5) * 2.0
}

pub(super) fn synthesize_mri_and_truth(
    sulcal_amp_mm: f64,
    base_thick_mm: f64,
    noise_level: f64,
) -> (MriVolume, Vec<u8>) {
    let total = MRI_DIMS[0] * MRI_DIMS[1] * MRI_DIMS[2];
    let mut voxels = Vec::with_capacity(total);
    let mut truth = Vec::with_capacity(total);

    for iz in 0..MRI_DIMS[2] {
        for iy in 0..MRI_DIMS[1] {
            for ix in 0..MRI_DIMS[0] {
                let pos = Vector3::new(
                    MRI_ORIGIN_MM[0] + ix as f64 * MRI_VOXEL_MM,
                    MRI_ORIGIN_MM[1] + iy as f64 * MRI_VOXEL_MM,
                    MRI_ORIGIN_MM[2] + iz as f64 * MRI_VOXEL_MM,
                );
                let norm_ellip = Vector3::new(
                    pos.x / SEMI_AXES_MM[0],
                    pos.y / SEMI_AXES_MM[1],
                    pos.z / SEMI_AXES_MM[2],
                );
                let r_norm = norm_ellip.norm();
                let v = if r_norm > 1e-9 {
                    (norm_ellip.z / r_norm).clamp(-1.0, 1.0).acos()
                } else {
                    0.0
                };
                let u = norm_ellip.y.atan2(norm_ellip.x).rem_euclid(TAU);
                let (fold, _) = gyral_and_atrophy_pattern(u, v);
                let signed_dist_mm = (r_norm - 1.0) * 22.5 - sulcal_amp_mm * fold;
                let ribbon_mm = (base_thick_mm * 4.2).max(9.5);
                let (label, clean_t1) = if signed_dist_mm < 0.0 {
                    (2_u8, 110.0)
                } else if signed_dist_mm <= ribbon_mm {
                    (1_u8, 75.0)
                } else {
                    (0_u8, 30.0)
                };
                truth.push(label);
                let noisy = (clean_t1 + noise_level * deterministic_voxel_noise(ix, iy, iz))
                    .clamp(2.0, 160.0);
                voxels.push(noisy);
            }
        }
    }
    let vol = MriVolume::new(MRI_DIMS, MRI_ORIGIN_MM, MRI_VOXEL_MM, voxels)
        .expect("16^3 volume matches voxel count");
    (vol, truth)
}

pub(super) fn tissue_classes() -> [TissueClass; 3] {
    [
        TissueClass::new(30.0, 11.0, 0.45),
        TissueClass::new(75.0, 11.0, 0.25),
        TissueClass::new(110.0, 11.0, 0.30),
    ]
}

fn neighbor_label_counts(dims: [usize; 3], labels: &[u8], ijk: [usize; 3]) -> [u8; 3] {
    let [nx, ny, nz] = dims;
    let [ix, iy, iz] = ijk;
    let mut counts = [0_u8; 3];
    let mut add = |x: usize, y: usize, z: usize| {
        let k = usize::from(labels[(z * ny + y) * nx + x]);
        if k < 3 {
            counts[k] = counts[k].saturating_add(1);
        }
    };
    if ix > 0 {
        add(ix - 1, iy, iz);
    }
    if ix + 1 < nx {
        add(ix + 1, iy, iz);
    }
    if iy > 0 {
        add(ix, iy - 1, iz);
    }
    if iy + 1 < ny {
        add(ix, iy + 1, iz);
    }
    if iz > 0 {
        add(ix, iy, iz - 1);
    }
    if iz + 1 < nz {
        add(ix, iy, iz + 1);
    }
    counts
}

pub(super) fn sample_gm_posteriors_on_cortex(
    white: &Surface,
    pial: &Surface,
    volume: &MriVolume,
    labels: &[u8],
    potts_beta: f64,
) -> Vec<f64> {
    let classes = tissue_classes();
    let [nx, ny, nz] = volume.dims;
    let mut post = [0.0_f64; 3];
    white
        .vertices
        .iter()
        .zip(&pial.vertices)
        .map(|(w, p)| {
            let mid = [
                0.5 * (w[0] + p[0]),
                0.5 * (w[1] + p[1]),
                0.5 * (w[2] + p[2]),
            ];
            let to_idx = |coord: f64, orig: f64, dim: usize| -> usize {
                (((coord - orig) / volume.voxel_size).round() as isize).clamp(0, dim as isize - 1)
                    as usize
            };
            let ix = to_idx(mid[0], volume.origin[0], nx);
            let iy = to_idx(mid[1], volume.origin[1], ny);
            let iz = to_idx(mid[2], volume.origin[2], nz);
            let counts = neighbor_label_counts(volume.dims, labels, [ix, iy, iz]);
            let intensity = volume.sample_trilinear(&mid);
            let _ = mrf_voxel_posteriors(intensity, &classes, &counts, potts_beta, &mut post);
            post[1]
        })
        .collect()
}

fn build_cohort_design_matrix(ages: &[f64], collinear_design: bool) -> DMatrix<f64> {
    let mut x_rows = Vec::with_capacity(ages.len() * 3);
    for (s, &age) in ages.iter().enumerate() {
        let is_patient = if s >= 4 { 1.0 } else { 0.0 };
        let third_col = if collinear_design {
            2.0 * is_patient
        } else {
            age
        };
        x_rows.extend_from_slice(&[1.0, is_patient, third_col]);
    }
    DMatrix::from_row_slice(ages.len(), 3, &x_rows)
}

pub(crate) fn evaluate_cohort_glm(
    thickness_mm: &[f64],
    atrophy_weights: &[f64],
    atrophy_effect_mm: f64,
    collinear_design: bool,
) -> (Vec<f64>, f64, Option<String>) {
    let ages = [-1.2, -0.4, 0.3, 1.1, -0.9, -0.2, 0.5, 1.0];
    let n_subj = ages.len();
    let x = build_cohort_design_matrix(&ages, collinear_design);
    let singular_msg = "GLM design matrix XᵀX is singular (collinear predictors): t-statistic contrast cannot be estimated.";
    let Ok(design_qr) = GlmDesignQr::try_new(&x) else {
        return (
            vec![0.0; thickness_mm.len()],
            0.0,
            Some(singular_msg.to_string()),
        );
    };
    let contrast = DVector::from_vec(vec![0.0, -1.0, 0.0]);
    let subj_jitter = [-0.09, 0.06, -0.04, 0.07, -0.08, 0.05, 0.09, -0.06];
    let dof = (n_subj - 3) as f64;

    let mut t_map = Vec::with_capacity(thickness_mm.len());
    let mut peak_abs = 0.0_f64;
    for (idx, (&base_t, &w)) in thickness_mm.iter().zip(atrophy_weights).enumerate() {
        let phase = ((idx % 7) as f64 - 3.0) * 0.012;
        let y = DVector::from_fn(n_subj, |s, _| {
            let group_loss = if s >= 4 { atrophy_effect_mm * w } else { 0.0 };
            (base_t - group_loss - 0.06 * ages[s] + subj_jitter[s] + phase * (s as f64 - 3.5))
                .max(0.5)
        });
        let Ok(beta) = design_qr.estimate_beta(&y) else {
            return (
                vec![0.0; thickness_mm.len()],
                0.0,
                Some(singular_msg.to_string()),
            );
        };
        let resid = &y - &x * &beta;
        let sigma2 = (resid.norm_squared() / dof).max(1e-12);
        let Ok(t_val) = design_qr.t_statistic(&contrast, &beta, sigma2) else {
            return (
                vec![0.0; thickness_mm.len()],
                0.0,
                Some(singular_msg.to_string()),
            );
        };
        peak_abs = peak_abs.max(t_val.abs());
        t_map.push(t_val);
    }
    (t_map, peak_abs, None)
}

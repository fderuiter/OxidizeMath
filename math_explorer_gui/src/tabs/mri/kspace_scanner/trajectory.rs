//! 2D k-Space trajectory generators, chaotic [`math_explorer::physics::chaos::lorenz::LorenzSystem`] projection, and physical gradient mapping.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{KSpaceMaskFilter, KSpaceTrajectory, MriKSpaceScannerState};
use super::{DEFAULT_FOV_METERS, KSPACE_CENTER, KSPACE_DIM, KSPACE_LEN};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
use math_explorer::physics::mri::proton::GYROMAGNETIC_RATIO;
use math_explorer::physics::mri::scanner::{accumulated_phase, k_space_coordinate};
use nalgebra::Vector3;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use std::f64::consts::PI;

/// Physical k-space coordinate and nuclear spin phase readout for a centered `(row, col)` bin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinScannerPhysics {
    /// Physical k-space vector $(k_x, k_y, k_z)$ in cycles/meter ($\text{m}^{-1}$).
    pub k_vec_per_m: Vector3<f64>,
    /// Magnitude $|\vec{k}|$ in cycles/meter ($\text{m}^{-1}$).
    pub k_mag_per_m: f64,
    /// Accumulated spin phase $\phi(\vec{r}_{\text{ref}}, t)$ in radians at the FOV quarter-diagonal voxel.
    pub reference_phase_rad: f64,
}

/// Converts normalized spatial-frequency coordinates $(u_x, u_y) \in [-0.5, 0.5]^2$ (cycles/pixel)
/// into a physical gradient time-integral $\int_0^t \vec{G}(\tau)\,d\tau$ ($\text{T}\cdot\text{s/m}$).
#[must_use]
pub fn gradient_integral_for_normalized_k(ux: f64, uy: f64) -> Vector3<f64> {
    let k_scale = (KSPACE_DIM as f64) / DEFAULT_FOV_METERS;
    let target_kx = ux * k_scale;
    let target_ky = uy * k_scale;
    let inv_gamma_over_2pi = (2.0 * PI) / GYROMAGNETIC_RATIO;
    Vector3::new(
        target_kx * inv_gamma_over_2pi,
        target_ky * inv_gamma_over_2pi,
        0.0,
    )
}

/// Evaluates [`k_space_coordinate`] and [`accumulated_phase`] for a centered k-space bin `(row, col)`.
#[must_use]
pub fn evaluate_bin_scanner_physics(row: usize, col: usize) -> BinScannerPhysics {
    let uy = (row.min(KSPACE_DIM - 1) as f64 - KSPACE_CENTER as f64) / (KSPACE_DIM as f64);
    let ux = (col.min(KSPACE_DIM - 1) as f64 - KSPACE_CENTER as f64) / (KSPACE_DIM as f64);
    let grad_integral = gradient_integral_for_normalized_k(ux, uy);
    let k_vec = k_space_coordinate(grad_integral);
    let ref_voxel = Vector3::new(0.25 * DEFAULT_FOV_METERS, 0.25 * DEFAULT_FOV_METERS, 0.0);
    let phase = accumulated_phase(ref_voxel, grad_integral);
    BinScannerPhysics {
        k_vec_per_m: k_vec,
        k_mag_per_m: k_vec.norm(),
        reference_phase_rad: phase,
    }
}

fn mark_bin_from_normalized_k(mask: &mut [bool], ux: f64, uy: f64) {
    let grad_integral = gradient_integral_for_normalized_k(ux, uy);
    let k_phys = k_space_coordinate(grad_integral);
    let bin_per_m = DEFAULT_FOV_METERS;
    let dc = KSPACE_CENTER as f64;
    let col = (dc + k_phys.x * bin_per_m).round() as isize;
    let row = (dc + k_phys.y * bin_per_m).round() as isize;
    if (0..KSPACE_DIM as isize).contains(&row) && (0..KSPACE_DIM as isize).contains(&col) {
        mask[flatten_2d_index(col as usize, row as usize, KSPACE_DIM)] = true;
    }
}

fn sample_cartesian_lines(stride: usize) -> Vec<bool> {
    let mut mask = vec![false; KSPACE_LEN];
    let s = stride.max(1);
    let dc_rem = KSPACE_CENTER % s;
    for row in 0..KSPACE_DIM {
        if row % s == dc_rem {
            let uy = (row as f64 - KSPACE_CENTER as f64) / (KSPACE_DIM as f64);
            for col in 0..KSPACE_DIM {
                let ux = (col as f64 - KSPACE_CENTER as f64) / (KSPACE_DIM as f64);
                mark_bin_from_normalized_k(&mut mask, ux, uy);
            }
        }
    }
    mask
}

fn sample_radial_spokes(spoke_count: usize) -> Vec<bool> {
    let mut mask = vec![false; KSPACE_LEN];
    let count = spoke_count.max(1);
    let samples_per_spoke = KSPACE_DIM * 2;
    for s in 0..count {
        let angle = PI * (s as f64) / (count as f64);
        let (sin_a, cos_a) = angle.sin_cos();
        for step in 0..=samples_per_spoke {
            let r = -0.5 + (step as f64) / (samples_per_spoke as f64);
            mark_bin_from_normalized_k(&mut mask, r * cos_a, r * sin_a);
        }
    }
    mask
}

fn sample_archimedean_spiral(turns: f64) -> Vec<bool> {
    let mut mask = vec![false; KSPACE_LEN];
    let total_steps = 4096_usize;
    let theta_max = 2.0 * PI * turns.clamp(2.0, 32.0);
    for i in 0..=total_steps {
        let frac = (i as f64) / (total_steps as f64);
        let radius = 0.5 * frac;
        let theta = theta_max * frac;
        let (sin_t, cos_t) = theta.sin_cos();
        mark_bin_from_normalized_k(&mut mask, radius * cos_t, radius * sin_t);
    }
    mask
}

fn sample_lorenz_chaotic(steps: usize, rho: f64) -> Vec<bool> {
    let mut mask = vec![false; KSPACE_LEN];
    let mut lorenz = LorenzBuilder::new()
        .sigma(10.0)
        .rho(rho.clamp(10.0, 60.0))
        .beta(8.0 / 3.0)
        .build(LorenzState::new(1.0, 1.0, 20.0));

    // Warm up onto the strange attractor before recording k-space samples.
    for _ in 0..80 {
        lorenz.step(0.01);
    }
    // Always acquire the 3x3 DC calibration core so bulk magnetization is anchored.
    for dr in -1_isize..=1 {
        for dc in -1_isize..=1 {
            let r = (KSPACE_CENTER as isize + dr) as usize;
            let c = (KSPACE_CENTER as isize + dc) as usize;
            mask[flatten_2d_index(c, r, KSPACE_DIM)] = true;
        }
    }
    let z_center = (rho - 1.0).max(10.0);
    for idx in 0..steps {
        lorenz.step(0.01);
        let v = lorenz.state.vec;
        if !v.x.is_finite() || !v.y.is_finite() || !v.z.is_finite() {
            break;
        }
        let ux_xy = (v.x / 44.0).clamp(-0.5, 0.49);
        let uy_xy = (v.y / 56.0).clamp(-0.5, 0.49);
        mark_bin_from_normalized_k(&mut mask, ux_xy, uy_xy);

        // Golden-angle rotated (x, z) projection fills orthogonal butterfly quadrants incoherently.
        let angle = (idx as f64) * 2.399_963_229_728_653;
        let (sin_a, cos_a) = angle.sin_cos();
        let rx = v.x / 44.0;
        let rz = (v.z - z_center) / 52.0;
        let ux_rot = (rx * cos_a - rz * sin_a).clamp(-0.5, 0.49);
        let uy_rot = (rx * sin_a + rz * cos_a).clamp(-0.5, 0.49);
        mark_bin_from_normalized_k(&mut mask, ux_rot, uy_rot);
    }
    mask
}

/// Builds the base centered `64x64` boolean mask from trajectory + frequency-domain filter settings.
#[must_use]
pub fn build_parametric_mask(state: &MriKSpaceScannerState) -> Vec<bool> {
    let mut mask = match state.trajectory {
        KSpaceTrajectory::FullCartesian => vec![true; KSPACE_LEN],
        KSpaceTrajectory::CartesianLines => sample_cartesian_lines(state.cartesian_stride),
        KSpaceTrajectory::RadialSpokes => sample_radial_spokes(state.radial_spokes),
        KSpaceTrajectory::ArchimedeanSpiral => sample_archimedean_spiral(state.spiral_turns),
        KSpaceTrajectory::LorenzChaotic => sample_lorenz_chaotic(state.lorenz_steps, state.lorenz_rho),
    };
    apply_mask_filter(&mut mask, state);
    mask
}

fn apply_mask_filter(mask: &mut [bool], state: &MriKSpaceScannerState) {
    let r2_cutoff = (state.cutoff_radius * state.cutoff_radius) as isize;
    match state.mask_filter {
        KSpaceMaskFilter::TrajectoryOnly => {}
        KSpaceMaskFilter::Keyhole => {
            for row in 0..KSPACE_DIM {
                let dr = row as isize - KSPACE_CENTER as isize;
                for col in 0..KSPACE_DIM {
                    let dc = col as isize - KSPACE_CENTER as isize;
                    if dr * dr + dc * dc > r2_cutoff {
                        mask[flatten_2d_index(col, row, KSPACE_DIM)] = false;
                    }
                }
            }
        }
        KSpaceMaskFilter::HighPass => {
            for row in 0..KSPACE_DIM {
                let dr = row as isize - KSPACE_CENTER as isize;
                for col in 0..KSPACE_DIM {
                    let dc = col as isize - KSPACE_CENTER as isize;
                    if dr * dr + dc * dc <= r2_cutoff {
                        mask[flatten_2d_index(col, row, KSPACE_DIM)] = false;
                    }
                }
            }
        }
        KSpaceMaskFilter::SeededRandom => {
            let mut rng = OxidizeRng::new(state.rng_seed);
            let keep = state.random_keep_ratio.clamp(0.05, 1.0);
            for active in mask.iter_mut() {
                if *active && rng.gen_range(0.0_f64..1.0_f64) > keep {
                    *active = false;
                }
            }
        }
    }
}

/// Applies a circular `Paint` or `Erase` brush stroke at `(center_row, center_col)`, returning `true` if `mask` changed.
pub fn apply_brush_stroke(
    mask: &mut [bool],
    center_row: usize,
    center_col: usize,
    radius: usize,
    paint_value: bool,
) -> bool {
    let r_lim = (radius.saturating_sub(1)) as isize;
    let r2 = r_lim * r_lim;
    let cr = center_row as isize;
    let cc = center_col as isize;
    let mut changed = false;

    for dr in -r_lim..=r_lim {
        let row = cr + dr;
        if !(0..KSPACE_DIM as isize).contains(&row) {
            continue;
        }
        for dc in -r_lim..=r_lim {
            let col = cc + dc;
            if !(0..KSPACE_DIM as isize).contains(&col) || dr * dr + dc * dc > r2 {
                continue;
            }
            let idx = flatten_2d_index(col as usize, row as usize, KSPACE_DIM);
            if mask[idx] != paint_value {
                mask[idx] = paint_value;
                changed = true;
            }
        }
    }
    changed
}

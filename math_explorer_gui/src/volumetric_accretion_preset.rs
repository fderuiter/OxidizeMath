#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Relativistic Shakura-Sunyaev accretion disk & bipolar corona `Gaussian3D` preset builder.
// [cite:gaussian_splatting]
// [cite:general_relativity_geodesics]

use math_explorer::ai::gaussian_splatting::{Gaussian3D, Scene};
use math_explorer::physics::high_energy::general_relativity::SchwarzschildBlackHole;
use nalgebra::{Point3, UnitQuaternion, Vector3};

/// Normalized Schwarzschild radius $R_s$ in world units (`[-1.8, 1.8]^3`) for the accretion preset.
pub const ACCRETION_WORLD_RS: f64 = 0.14;

/// Builds a 320-splat ($\le 384$) tilted relativistic accretion disk and coronal ring outside $3 R_s$.
pub fn build_relativistic_accretion_disk_scene() -> (Scene, Vec<f64>) {
    let bh = SchwarzschildBlackHole::from_schwarzschild_radius(ACCRETION_WORLD_RS)
        .expect("positive accretion world R_s");
    let r_isco = bh.isco().unwrap_or(3.0 * ACCRETION_WORLD_RS);
    let ring_count = 8usize;
    let per_ring = 36usize;
    let corona_count = 32usize;
    let total = ring_count * per_ring + corona_count;

    let mut gaussians = Vec::with_capacity(total);
    let mut grads = Vec::with_capacity(total);
    let disk_tilt = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.24);

    for ring_idx in 0..ring_count {
        let u = (ring_idx as f64) / ((ring_count - 1) as f64);
        let r = r_isco * 1.04 + u * (1.70 - r_isco * 1.04);
        let temp = ((r_isco / r).powf(0.75) * (1.0 - (r_isco / r).sqrt().min(0.96)).powf(0.25))
            .clamp(0.12, 1.0);
        let thickness = 0.022 + 0.038 * u;
        let tang_scale = (std::f64::consts::TAU * r / (per_ring as f64)) * 0.72;
        let rad_scale = 0.068 + 0.045 * u;

        for seg_idx in 0..per_ring {
            let spiral_twist = 0.35 * (r / r_isco).ln();
            let phi = (seg_idx as f64) * std::f64::consts::TAU / (per_ring as f64) + spiral_twist;
            let (sin_p, cos_p) = phi.sin_cos();
            let spiral_arm = 0.5 + 0.5 * (2.0 * (phi - spiral_twist)).cos();
            let y_warp = 0.018 * (phi * 2.0).sin() * u;
            let local_pos = Vector3::new(r * cos_p, y_warp, r * sin_p);
            let world_pos = disk_tilt * local_pos;
            let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -phi);
            let rot = disk_tilt * yaw;
            let color = Vector3::new(
                (0.98 - 0.14 * (1.0 - temp)).clamp(0.25, 1.0),
                (0.36 + 0.54 * temp + 0.08 * spiral_arm).clamp(0.18, 1.0),
                (0.14 + 0.76 * temp * temp).clamp(0.08, 1.0),
            );
            let opacity = (0.34 + 0.48 * temp + 0.10 * spiral_arm).clamp(0.18, 0.92);
            gaussians.push(Gaussian3D {
                mean: Point3::from(world_pos),
                scale: Vector3::new(rad_scale, thickness, tang_scale),
                rotation: rot,
                opacity,
                color,
            });
            grads.push((0.06 + 0.28 * temp).clamp(0.02, 0.5));
        }
    }

    append_coronal_jet_splats(&disk_tilt, r_isco, corona_count, &mut gaussians, &mut grads);
    (Scene { gaussians }, grads)
}

fn append_coronal_jet_splats(
    disk_tilt: &UnitQuaternion<f64>,
    r_isco: f64,
    corona_count: usize,
    gaussians: &mut Vec<Gaussian3D>,
    grads: &mut Vec<f64>,
) {
    let half = corona_count / 2;
    for idx in 0..corona_count {
        let sign = if idx < half { 1.0 } else { -1.0 };
        let k = idx % half.max(1);
        let frac = (k as f64 + 0.5) / (half.max(1) as f64);
        let z_jet = sign * (0.28 + 1.05 * frac);
        let r_funnel = r_isco * (0.65 + 0.95 * frac);
        let phi = (k as f64) * 1.15;
        let (sin_p, cos_p) = phi.sin_cos();
        let local_pos = Vector3::new(r_funnel * cos_p, z_jet, r_funnel * sin_p);
        let world_pos = disk_tilt * local_pos;
        gaussians.push(Gaussian3D {
            mean: Point3::from(world_pos),
            scale: Vector3::new(0.075, 0.14, 0.075),
            rotation: *disk_tilt,
            opacity: (0.42 * (1.0 - 0.55 * frac)).clamp(0.12, 0.55),
            color: Vector3::new(0.38, 0.72, 0.98),
        });
        grads.push(0.19);
    }
}

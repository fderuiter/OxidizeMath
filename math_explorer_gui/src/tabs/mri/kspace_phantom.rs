#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::kspace_types::{KSpaceMaskPreset, PhantomPreset};
use eframe::egui::Color32;
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::mri::phantom::fill_shepp_logan_grid;
use std::f64::consts::PI;

#[cfg(feature = "biology")]
use math_explorer::biology::diffusion::FiniteDifference2D;
#[cfg(feature = "biology")]
use math_explorer::biology::morphogenesis::{DiffusionCoeff, SchnakenbergKinetics, TuringSystem};
#[cfg(feature = "biology")]
use math_explorer::math_kernel::types::{Dimension, StepSize};

/// Pre-allocated phantom generator coupling analytical Shepp-Logan ellipses and `TuringSystem`.
pub struct PhantomEngine {
    size: usize,
    active_preset: PhantomPreset,
    #[cfg(feature = "biology")]
    turing: Option<TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>>,
}

impl PhantomEngine {
    /// Creates and initializes a `size x size` phantom engine for `preset`.
    pub fn new(size: usize, preset: PhantomPreset) -> Self {
        let mut engine = Self {
            size,
            active_preset: preset,
            #[cfg(feature = "biology")]
            turing: None,
        };
        engine.rebuild(size, preset);
        engine
    }

    /// Rebuilds internal reaction-diffusion buffers if `size` or `preset` changed, then populates tissues.
    pub fn rebuild(&mut self, size: usize, preset: PhantomPreset) {
        self.size = size;
        self.active_preset = preset;
        #[cfg(feature = "biology")]
        {
            self.turing = match preset {
                PhantomPreset::SheppLogan => None,
                PhantomPreset::TuringSpots => Some(build_warmed_turing_system(size, false)),
                PhantomPreset::TuringStripes => Some(build_warmed_turing_system(size, true)),
            };
        }
    }

    /// Populates pre-allocated `(rho, t1, t2)` slices of length `size * size` without heap allocation.
    pub fn write_tissues_into(&self, rho: &mut [f64], t1: &mut [f64], t2: &mut [f64]) {
        match self.active_preset {
            PhantomPreset::SheppLogan => {
                fill_shepp_logan_grid(self.size, self.size, rho, t1, t2);
            }
            PhantomPreset::TuringSpots | PhantomPreset::TuringStripes => {
                #[cfg(feature = "biology")]
                if let Some(ref sys) = self.turing {
                    map_uv_to_mri_tissues(self.size, sys.u(), sys.v(), rho, t1, t2);
                    return;
                }
                let stripes = self.active_preset == PhantomPreset::TuringStripes;
                fill_analytical_pattern_fallback(self.size, stripes, rho, t1, t2);
            }
        }
    }

    /// Advances the live `TuringSystem` by one CFL-stable reaction-diffusion step with zero allocation.
    pub fn step_live_turing(&mut self, rho: &mut [f64], t1: &mut [f64], t2: &mut [f64]) {
        #[cfg(feature = "biology")]
        if let Some(ref mut sys) = self.turing {
            sys.step(0.015);
            map_uv_to_mri_tissues(self.size, sys.u(), sys.v(), rho, t1, t2);
        }
        #[cfg(not(feature = "biology"))]
        {
            let _ = (rho, t1, t2);
        }
    }
}

#[cfg(feature = "biology")]
fn build_warmed_turing_system(
    size: usize,
    stripes: bool,
) -> TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D> {
    let (a, b, d_u, d_v) = if stripes {
        (0.14, 0.86, 1.0, 24.0)
    } else {
        (0.10, 0.90, 1.0, 32.0)
    };
    let kinetics = SchnakenbergKinetics { a, b };
    let diffusion = FiniteDifference2D::new(
        Dimension(size),
        Dimension(size),
        StepSize(2.0),
        StepSize(2.0),
    );
    let mut sys = TuringSystem::new_with_kinetics(
        Dimension(size * size),
        DiffusionCoeff(d_u),
        DiffusionCoeff(d_v),
        kinetics,
        diffusion,
    );
    seed_turing_perturbation(&mut sys, size, stripes);
    for _ in 0..240 {
        sys.step(0.015);
    }
    sys
}

#[cfg(feature = "biology")]
fn seed_turing_perturbation(
    sys: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    size: usize,
    stripes: bool,
) {
    let u_eq = sys.kinetics.a + sys.kinetics.b;
    let v_eq = sys.kinetics.b / (u_eq * u_eq);
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ (u64::from(stripes) << 32);
    for r in 0..size {
        let fy = (r as f64) / (size as f64) * 2.0 * PI;
        for c in 0..size {
            let fx = (c as f64) / (size as f64) * 2.0 * PI;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let jitter = (((state >> 32) as u32 as f64) / (u32::MAX as f64) - 0.5) * 0.08;
            let mode = if stripes {
                0.16 * (5.0 * fx + 1.5 * fy.sin()).cos() + 0.08 * (4.0 * fy).sin()
            } else {
                0.18 * (4.0 * fx).cos() * (4.0 * fy).cos() + 0.09 * (6.0 * fx + 5.0 * fy).sin()
            };
            let idx = r * size + c;
            sys.u_mut()[idx] = (u_eq + mode + jitter).max(0.05);
            sys.v_mut()[idx] = (v_eq - 0.5 * mode - jitter).max(0.05);
        }
    }
}

#[cfg(feature = "biology")]
fn map_uv_to_mri_tissues(
    size: usize,
    u: &[f64],
    v: &[f64],
    out_rho: &mut [f64],
    out_t1: &mut [f64],
    out_t2: &mut [f64],
) {
    let (mut u_min, mut u_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut v_min, mut v_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for (&ui, &vi) in u.iter().zip(v.iter()) {
        if ui.is_finite() {
            u_min = u_min.min(ui);
            u_max = u_max.max(ui);
        }
        if vi.is_finite() {
            v_min = v_min.min(vi);
            v_max = v_max.max(vi);
        }
    }
    let inv_u = if u_max - u_min > 1e-9 {
        1.0 / (u_max - u_min)
    } else {
        0.0
    };
    let inv_v = if v_max - v_min > 1e-9 {
        1.0 / (v_max - v_min)
    } else {
        0.0
    };
    let half = (size as f64) * 0.5;
    let r_lim_sq = (half * 0.90).powi(2);

    for r in 0..size {
        let dy = (r as f64) + 0.5 - half;
        for c in 0..size {
            let dx = (c as f64) + 0.5 - half;
            let idx = r * size + c;
            if dx * dx + dy * dy > r_lim_sq {
                out_rho[idx] = 0.0;
                out_t1[idx] = 2.40;
                out_t2[idx] = 0.32;
                continue;
            }
            let un = ((u[idx] - u_min) * inv_u).clamp(0.0, 1.0);
            let vn = ((v[idx] - v_min) * inv_v).clamp(0.0, 1.0);
            // Activator foci: higher density, short T1 (0.38s), short T2 (0.042s)
            // Inhibitor matrix: moderate density, long T1 (2.10s), long T2 (0.30s)
            out_rho[idx] = (0.58 + 0.34 * un + 0.08 * vn).clamp(0.1, 1.0);
            out_t1[idx] = 2.10 - 1.72 * un;
            out_t2[idx] = 0.30 - 0.258 * un;
        }
    }
}

fn fill_analytical_pattern_fallback(
    size: usize,
    stripes: bool,
    out_rho: &mut [f64],
    out_t1: &mut [f64],
    out_t2: &mut [f64],
) {
    let half = (size as f64) * 0.5;
    let r_lim_sq = (half * 0.90).powi(2);
    for r in 0..size {
        let dy = (r as f64) + 0.5 - half;
        let fy = (r as f64) / (size as f64) * 2.0 * PI;
        for c in 0..size {
            let dx = (c as f64) + 0.5 - half;
            let fx = (c as f64) / (size as f64) * 2.0 * PI;
            let idx = r * size + c;
            if dx * dx + dy * dy > r_lim_sq {
                out_rho[idx] = 0.0;
                out_t1[idx] = 2.40;
                out_t2[idx] = 0.32;
                continue;
            }
            let wave = if stripes {
                (5.0 * fx + 1.2 * fy.sin()).cos()
            } else {
                (4.0 * fx).cos() * (4.0 * fy).cos()
            };
            let un = usize::from(wave > 0.15) as f64;
            out_rho[idx] = 0.62 + 0.33 * un;
            out_t1[idx] = 2.10 - 1.72 * un;
            out_t2[idx] = 0.30 - 0.258 * un;
        }
    }
}

/// Populates a centered `(R/2, C/2)` k-space sampling mask without heap allocation.
pub fn build_preset_mask_into(
    preset: KSpaceMaskPreset,
    size: usize,
    cutoff: f64,
    spokes: usize,
    stride: usize,
    out_mask: &mut [f64],
) {
    let center = (size / 2) as f64;
    let max_r = center.max(1.0);
    let r_thresh = cutoff.clamp(0.05, 1.0) * max_r;
    let spoke_count = spokes.clamp(4, 64);
    let line_stride = stride.clamp(1, 8);

    for r in 0..size {
        let dy = (r as f64) - center;
        for c in 0..size {
            let dx = (c as f64) - center;
            let rad = (dx * dx + dy * dy).sqrt();
            let keep = match preset {
                KSpaceMaskPreset::Full => true,
                KSpaceMaskPreset::LowPass => rad <= r_thresh,
                KSpaceMaskPreset::HighPass => rad >= r_thresh,
                KSpaceMaskPreset::Cartesian => {
                    (r % line_stride == 0) || (r.abs_diff(size / 2) <= 2)
                }
                KSpaceMaskPreset::Radial => point_on_radial_spoke(dx, dy, rad, spoke_count),
            };
            out_mask[flatten_2d_index(c, r, size)] = if keep { 1.0 } else { 0.0 };
        }
    }
}

fn point_on_radial_spoke(dx: f64, dy: f64, rad: f64, spoke_count: usize) -> bool {
    if rad <= 2.0 {
        return true;
    }
    let step_angle = PI / (spoke_count as f64);
    let mut theta = dy.atan2(dx);
    if theta < 0.0 {
        theta += PI;
    }
    let nearest_idx = (theta / step_angle).round();
    let nearest_theta = nearest_idx * step_angle;
    let perp_dist = rad * (theta - nearest_theta).sin().abs();
    perp_dist <= 0.75
}

/// Maps normalized scalar field values into a pre-allocated [`Color32`] slice using high-contrast-aware colormaps.
pub fn colorize_scalar_field_into(values: &[f64], high_contrast: bool, out_rgba: &mut [Color32]) {
    let (mut min_v, mut max_v) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in values {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() || max_v < min_v {
        min_v = 0.0;
        max_v = 1.0;
    }
    let inv_span = if max_v - min_v > 1e-9 {
        1.0 / (max_v - min_v)
    } else {
        0.0
    };
    for (&val, px) in values.iter().zip(out_rgba.iter_mut()) {
        let norm = if val.is_finite() {
            ((val - min_v) * inv_span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let rgb = if high_contrast {
            heatmap_color_high_contrast(norm)
        } else {
            heatmap_color(norm)
        };
        *px = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
    }
}

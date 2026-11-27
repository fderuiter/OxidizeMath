//! Caustic envelope extraction (nephroid and cusp caustics) and detector intensity histograms.
//!
//! Extracts envelope focal loci where adjacent rays in a traced ray family intersect, and accumulates
//! transverse detector screen intensity profiles into pre-allocated zero-allocation buffers.

// [cite:optics]

use crate::physics::optics::ray_tracing::{RayFanWorkspace, TracedRayMeta};

/// Single extracted focal tangency point along an optical caustic envelope.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CausticPoint {
    /// Planar optical-bench coordinates $(x, y)$ of the envelope intersection.
    pub position: [f64; 2],
    /// Local geometric concentration weight proportional to inverse ray separation.
    pub intensity: f64,
    /// Vacuum wavelength $\lambda$ in nanometers of the contributing ray pair.
    pub wavelength_nm: f64,
}

/// Pre-allocated buffer holding extracted caustic envelope loci and a detector intensity histogram.
pub struct CausticEnvelopeBuffer {
    points: Vec<CausticPoint>,
    active_len: usize,
    detector_bins: Vec<f64>,
    detector_y_min: f64,
    detector_y_max: f64,
}

impl CausticEnvelopeBuffer {
    /// Allocates capacity for `max_caustic_points` focal intersections and `detector_bin_count` histogram bins.
    #[must_use]
    pub fn new(
        max_caustic_points: usize,
        detector_bin_count: usize,
        detector_y_min: f64,
        detector_y_max: f64,
    ) -> Self {
        Self {
            points: vec![CausticPoint::default(); max_caustic_points.max(1)],
            active_len: 0,
            detector_bins: vec![0.0; detector_bin_count.max(4)],
            detector_y_min,
            detector_y_max: detector_y_max.max(detector_y_min + 1.0e-3),
        }
    }

    /// Updates the transverse $(y_{\min}, y_{\max})$ span of the detector histogram without allocating.
    pub fn set_detector_span(&mut self, y_min: f64, y_max: f64) {
        self.detector_y_min = y_min;
        self.detector_y_max = y_max.max(y_min + 1.0e-3);
    }

    /// Slice of valid caustic envelope points extracted by the most recent [`extract_caustic_envelope`] call.
    #[must_use]
    pub fn points(&self) -> &[CausticPoint] {
        &self.points[..self.active_len]
    }

    /// Detector screen intensity histogram bins spanning `[detector_y_min, detector_y_max]`.
    #[must_use]
    pub fn detector_bins(&self) -> &[f64] {
        &self.detector_bins
    }

    /// Transverse $y$ center coordinate corresponding to histogram `bin_index`.
    #[must_use]
    pub fn bin_center_y(&self, bin_index: usize) -> f64 {
        let n = self.detector_bins.len().max(1) as f64;
        let frac = ((bin_index as f64) + 0.5) / n;
        self.detector_y_min + frac * (self.detector_y_max - self.detector_y_min)
    }
}

/// Closed-form coffee-cup nephroid caustic point for a concave circular mirror of `radius` at angle `phi_rad`.
///
/// For horizontal $+x$ rays reflecting inside $x = R\cos\phi, y = R\sin\phi$ ($\phi \in (-\pi/2, \pi/2)$):
/// $$x_c(\phi) = \frac{R}{4}(3\cos\phi - \cos 3\phi), \qquad y_c(\phi) = R\sin^3\phi$$
#[verified_engine::verified]
#[must_use]
pub fn nephroid_caustic_point(radius: f64, phi_rad: f64) -> [f64; 2] {
    let (s, c) = phi_rad.sin_cos();
    let c3 = (3.0 * phi_rad).cos();
    [0.25 * radius * (3.0 * c - c3), radius * s * s * s]
}

/// Exact axial cusp x-coordinate $x_{\text{cusp}} = R / 2$ of the circular mirror nephroid caustic.
#[verified_engine::verified]
#[must_use]
pub fn coffee_cup_cusp_x(radius: f64) -> f64 {
    0.5 * radius
}

/// Paraxial focal length $f = \frac{R}{2(n - 1)}$ of a symmetric biconvex lens in the thin-lens limit.
#[verified_engine::verified]
#[must_use]
pub fn thin_biconvex_paraxial_focal_length(radius_curv: f64, refractive_index: f64) -> f64 {
    radius_curv / (2.0 * (refractive_index - 1.0).max(1.0e-9))
}

fn intersect_forward_rays(
    r0: &TracedRayMeta,
    r1: &TracedRayMeta,
    max_extent: f64,
) -> Option<CausticPoint> {
    let dx = r1.exit_origin[0] - r0.exit_origin[0];
    let dy = r1.exit_origin[1] - r0.exit_origin[1];
    let cross = r0.exit_dir[0] * r1.exit_dir[1] - r0.exit_dir[1] * r1.exit_dir[0];
    if cross.abs() < 1.0e-6 {
        return None;
    }
    let t0 = (dx * r1.exit_dir[1] - dy * r1.exit_dir[0]) / cross;
    let t1 = (dx * r0.exit_dir[1] - dy * r0.exit_dir[0]) / cross;
    if t0 <= 1.0e-6 || t1 <= 1.0e-6 {
        return None;
    }
    let px = r0.exit_origin[0] + t0 * r0.exit_dir[0];
    let py = r0.exit_origin[1] + t0 * r0.exit_dir[1];
    if px.abs() > max_extent || py.abs() > max_extent {
        return None;
    }
    let sep = (dx * dx + dy * dy).sqrt().max(1.0e-4);
    let weight = 0.5 * (r0.power + r1.power) / (sep + cross.abs());
    Some(CausticPoint {
        position: [px, py],
        intensity: weight.min(250.0),
        wavelength_nm: 0.5 * (r0.wavelength_nm + r1.wavelength_nm),
    })
}

fn intersect_segments(a0: [f64; 2], a1: [f64; 2], b0: [f64; 2], b1: [f64; 2]) -> Option<[f64; 2]> {
    let dax = a1[0] - a0[0];
    let day = a1[1] - a0[1];
    let dbx = b1[0] - b0[0];
    let dby = b1[1] - b0[1];
    let denom = dax * dby - day * dbx;
    if denom.abs() < 1.0e-10 {
        return None;
    }
    let qx = b0[0] - a0[0];
    let qy = b0[1] - a0[1];
    let ta = (qx * dby - qy * dbx) / denom;
    let tb = (qx * day - qy * dax) / denom;
    if (0.0..=1.0).contains(&ta) && (0.0..=1.0).contains(&tb) {
        Some([a0[0] + ta * dax, a0[1] + ta * day])
    } else {
        None
    }
}

fn accumulate_detector_hits(
    vertices: &[[f64; 2]],
    power: f64,
    detector_x: f64,
    y_min: f64,
    y_max: f64,
    bins: &mut [f64],
) {
    let inv_span = 1.0 / (y_max - y_min).max(1.0e-9);
    let nb = bins.len();
    for seg in vertices.windows(2) {
        let (p0, p1) = (seg[0], seg[1]);
        let dx0 = p0[0] - detector_x;
        let dx1 = p1[0] - detector_x;
        if (dx0 <= 0.0 && dx1 >= 0.0) || (dx0 >= 0.0 && dx1 <= 0.0) {
            let denom = (p1[0] - p0[0]).abs();
            let t = if denom > 1.0e-12 {
                (detector_x - p0[0]) / (p1[0] - p0[0])
            } else {
                0.5
            };
            let hit_y = p0[1] + t.clamp(0.0, 1.0) * (p1[1] - p0[1]);
            let norm = (hit_y - y_min) * inv_span;
            if (0.0..1.0).contains(&norm) {
                let idx = ((norm * (nb as f64)) as usize).min(nb - 1);
                bins[idx] += power;
            }
        }
    }
}

/// Extracts caustic envelope points from adjacent rays in `workspace` and fills the detector histogram.
///
/// Operates strictly in-place inside `out` with zero heap allocations.
#[verified_engine::verified]
pub fn extract_caustic_envelope(
    workspace: &RayFanWorkspace,
    detector_x: f64,
    max_bench_extent: f64,
    out: &mut CausticEnvelopeBuffer,
) -> usize {
    out.detector_bins.fill(0.0);
    let rays = workspace.rays();
    let y_min = out.detector_y_min;
    let y_max = out.detector_y_max;
    for (idx, ray) in rays.iter().enumerate() {
        accumulate_detector_hits(
            workspace.ray_vertices(idx),
            ray.power,
            detector_x,
            y_min,
            y_max,
            &mut out.detector_bins,
        );
    }

    let cap = out.points.len();
    let mut count = 0usize;
    let bound = max_bench_extent.max(1.0);
    for i in 0..rays.len().saturating_sub(1) {
        if count >= cap {
            break;
        }
        let r0 = &rays[i];
        let r1 = &rays[i + 1];
        if r0.spectral_band != r1.spectral_band {
            continue;
        }
        if r0.point_len > 4 && r1.point_len > 4 {
            let v0 = workspace.ray_vertices(i);
            let v1 = workspace.ray_vertices(i + 1);
            let seg_limit = v0.len().min(v1.len()).saturating_sub(1);
            for s in 0..seg_limit {
                if count >= cap {
                    break;
                }
                if let Some(pos) = intersect_segments(v0[s], v0[s + 1], v1[s], v1[s + 1])
                    && pos[0].abs() <= bound
                    && pos[1].abs() <= bound
                {
                    out.points[count] = CausticPoint {
                        position: pos,
                        intensity: 10.0 * (r0.power + r1.power),
                        wavelength_nm: r0.wavelength_nm,
                    };
                    count += 1;
                }
            }
        } else if let Some(cp) = intersect_forward_rays(r0, r1, bound) {
            out.points[count] = cp;
            count += 1;
        }
    }
    out.active_len = count;
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::optics::ray_tracing::{
        OpticalBenchScene, PolarizationMode, RayFanConfig, trace_ray_fan,
    };

    #[test]
    #[verified_engine::verified]
    fn coffee_cup_nephroid_cusp_matches_half_radius() {
        let radius = 2.0;
        let mut ws = RayFanWorkspace::new(128, 8);
        let cfg = RayFanConfig {
            scene: OpticalBenchScene::CoffeeCupMirror { radius },
            ray_count: 101,
            beam_half_width: 0.96 * radius,
            source_x: -2.5,
            source_y: 0.0,
            launch_angle_rad: 0.0,
            wavelength_nm: 550.0,
            polychromatic: false,
            polarization: PolarizationMode::Unpolarized,
            bench_half_extent: 3.0,
        };
        trace_ray_fan(&cfg, &mut ws);
        let mut buf = CausticEnvelopeBuffer::new(128, 64, -2.0, 2.0);
        let count = extract_caustic_envelope(&ws, 1.0, 3.0, &mut buf);
        assert!(count > 80);

        let paraxial = buf
            .points()
            .iter()
            .min_by(|a, b| a.position[1].abs().total_cmp(&b.position[1].abs()))
            .expect("paraxial caustic intersection exists");
        let expected_cusp = coffee_cup_cusp_x(radius);
        assert!(
            (paraxial.position[0] - expected_cusp).abs() < 1.0e-3,
            "cusp x = {}, expected {}",
            paraxial.position[0],
            expected_cusp
        );
        let detector_sum: f64 = buf.detector_bins().iter().sum();
        assert!(detector_sum > 10.0);
    }

    #[test]
    #[verified_engine::verified]
    fn spherical_biconvex_lens_extracts_aberration_cusp_caustic() {
        use crate::physics::optics::refraction::GlassMaterial;
        let mut ws = RayFanWorkspace::new(96, 8);
        let cfg = RayFanConfig {
            scene: OpticalBenchScene::SphericalBiconvexLens {
                radius_curv: 1.8,
                half_thickness: 0.24,
                aperture_radius: 0.95,
                glass: GlassMaterial::Bk7Crown,
            },
            ray_count: 61,
            beam_half_width: 0.85,
            source_x: -2.0,
            source_y: 0.0,
            launch_angle_rad: 0.0,
            wavelength_nm: 550.0,
            polychromatic: false,
            polarization: PolarizationMode::Unpolarized,
            bench_half_extent: 4.0,
        };
        trace_ray_fan(&cfg, &mut ws);
        let mut buf = CausticEnvelopeBuffer::new(128, 64, -2.0, 2.0);
        let count = extract_caustic_envelope(&ws, 1.6, 4.0, &mut buf);
        assert!(count > 40);
        let paraxial_f = thin_biconvex_paraxial_focal_length(1.8, 1.5168);
        let marginal_min_x = buf
            .points()
            .iter()
            .map(|p| p.position[0])
            .fold(f64::INFINITY, f64::min);
        assert!(marginal_min_x < paraxial_f);
    }
}

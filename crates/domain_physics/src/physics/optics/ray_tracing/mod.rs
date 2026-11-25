//! Zero-allocation RK4 eikonal graded-index (GRIN) integration and discrete optical bench ray tracing.
//!
//! Integrates canonical eikonal ray equations $\frac{d\mathbf{r}}{d\tau} = \mathbf{p}$,
//! $\frac{d\mathbf{p}}{d\tau} = \frac{1}{2}\nabla(n^2)$ using `RungeKutta4<ArrayState<4>>`, alongside
//! exact multi-surface ray tracing across spherical lenses, circular mirrors, and dispersive prisms.

// [cite:optics]

mod surfaces;
mod workspace;

pub use workspace::{RayFanWorkspace, TracedRayMeta};

use pure_math::pure_math::analysis::ode::{ArrayState, OdeSystem};
use serde::{Deserialize, Serialize};

use super::refraction::GlassMaterial;
use surfaces::{RayStepCtx, trace_single_ray};

/// Continuous graded-index (GRIN) refractive index profiles $n(x, y)$.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GrinMedium {
    /// Luneburg lens $n(r) = \sqrt{2 - (r/R)^2}$ for $r \le R$, focusing parallel rays onto rim $r = R$.
    Luneburg {
        /// Outer metamaterial sphere radius $R > 0$.
        radius: f64,
    },
    /// Maxwell fisheye lens $n(r) = \frac{2 n_0}{1 + (r/R)^2}$, imaging interior points antipodally.
    MaxwellFisheye {
        /// Characteristic fisheye scale radius $R > 0$.
        radius: f64,
        /// Refractive index at $r = R$ ($n(0) = 2 n_0$).
        n0: f64,
    },
    /// Regularized Eaton $180^\circ$ retroreflector $n(r) = \sqrt{2R / r_{\text{eff}} - 1}$ for $r \le R$.
    EatonRetroreflector {
        /// Outer retroreflector radius $R > 0$.
        radius: f64,
        /// Dimensionless core regularization ratio $\epsilon \in (0, 0.5]$.
        core_eps: f64,
    },
    /// Parabolic GRIN slab waveguide $n^2(y) = n_0^2(1 - g^2 y^2)$ with sinusoidal period $2\pi/g$.
    ParabolicWaveguide {
        /// Axial refractive index $n_0$ at $y = 0$.
        n0: f64,
        /// Quadratic focusing constant $g > 0$.
        gradient_g: f64,
    },
}

impl GrinMedium {
    /// Human-readable label for the graded-index profile.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Luneburg { .. } => "Luneburg Metamaterial Lens",
            Self::MaxwellFisheye { .. } => "Maxwell Fisheye Sphere",
            Self::EatonRetroreflector { .. } => "Eaton 180° Retroreflector",
            Self::ParabolicWaveguide { .. } => "Parabolic GRIN Waveguide",
        }
    }

    /// Evaluates refractive index $n(x, y)$ at planar coordinates `(x, y)`.
    #[verified_engine::verified]
    #[must_use]
    pub fn refractive_index_at(self, x: f64, y: f64) -> f64 {
        let r2 = x * x + y * y;
        match self {
            Self::Luneburg { radius } => {
                let inv_r2 = 1.0 / (radius * radius).max(1.0e-12);
                if r2 >= radius * radius {
                    1.0
                } else {
                    (2.0 - r2 * inv_r2).max(1.0).sqrt()
                }
            }
            Self::MaxwellFisheye { radius, n0 } => {
                let u = r2 / (radius * radius).max(1.0e-12);
                (2.0 * n0 / (1.0 + u)).max(1.0e-6)
            }
            Self::EatonRetroreflector { radius, core_eps } => {
                if r2 >= radius * radius {
                    1.0
                } else {
                    let eps = core_eps.clamp(0.03, 0.5) * radius;
                    let r_eff = (r2 + eps * eps).sqrt();
                    (2.0 * radius / r_eff - 1.0).max(1.0).sqrt()
                }
            }
            Self::ParabolicWaveguide { n0, gradient_g } => {
                let n_sq = n0 * n0 * (1.0 - gradient_g * gradient_g * y * y).max(0.08);
                n_sq.sqrt()
            }
        }
    }

    /// Evaluates analytic half-gradient $\frac{1}{2}\nabla(n^2) = n\nabla n$ for canonical eikonal flow.
    #[must_use]
    pub fn grad_half_n_sq(self, x: f64, y: f64) -> [f64; 2] {
        let r2 = x * x + y * y;
        match self {
            Self::Luneburg { radius } => {
                let r_max2 = (radius * radius).max(1.0e-12);
                if r2 >= r_max2 {
                    [0.0, 0.0]
                } else {
                    [-x / r_max2, -y / r_max2]
                }
            }
            Self::MaxwellFisheye { radius, n0 } => {
                let r_max2 = (radius * radius).max(1.0e-12);
                let denom = (1.0 + r2 / r_max2).powi(3) * r_max2;
                let factor = -8.0 * n0 * n0 / denom.max(1.0e-15);
                [factor * x, factor * y]
            }
            Self::EatonRetroreflector { radius, core_eps } => {
                if r2 >= radius * radius {
                    [0.0, 0.0]
                } else {
                    let eps = core_eps.clamp(0.03, 0.5) * radius;
                    let r_eff3 = (r2 + eps * eps).powf(1.5).max(1.0e-15);
                    let factor = -radius / r_eff3;
                    [factor * x, factor * y]
                }
            }
            Self::ParabolicWaveguide { n0, gradient_g } => {
                if 1.0 - gradient_g * gradient_g * y * y <= 0.08 {
                    [0.0, 0.0]
                } else {
                    [0.0, -n0 * n0 * gradient_g * gradient_g * y]
                }
            }
        }
    }
}

/// Canonical 2D eikonal ODE system $\dot{\mathbf{r}} = \mathbf{p}$, $\dot{\mathbf{p}} = \frac{1}{2}\nabla(n^2)$.
#[derive(Debug, Clone, Copy)]
pub struct EikonalRaySystem {
    /// Underlying continuous graded-index profile.
    pub medium: GrinMedium,
}

impl OdeSystem<ArrayState<4>> for EikonalRaySystem {
    fn derivative(&self, _tau: f64, state: &ArrayState<4>) -> ArrayState<4> {
        let g = self.medium.grad_half_n_sq(state.0[0], state.0[1]);
        ArrayState([state.0[2], state.0[3], g[0], g[1]])
    }

    fn derivative_in_place(&self, _tau: f64, state: &ArrayState<4>, out: &mut ArrayState<4>) {
        let g = self.medium.grad_half_n_sq(state.0[0], state.0[1]);
        out.0 = [state.0[2], state.0[3], g[0], g[1]];
    }
}

/// Input polarization state used to weight Fresnel surface reflections and transmissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PolarizationMode {
    /// Equal $s$ and $p$ incoherent power average.
    #[default]
    Unpolarized,
    /// Transverse-electric ($s$-polarized) field perpendicular to the plane of incidence.
    SPolarized,
    /// Transverse-magnetic ($p$-polarized) field parallel to the plane of incidence (zero reflection at $\theta_B$).
    PPolarized,
}

impl PolarizationMode {
    /// Human-readable label for UI selectors.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unpolarized => "Unpolarized",
            Self::SPolarized => "TE (s-Polarized)",
            Self::PPolarized => "TM (p-Polarized)",
        }
    }
}

/// Optical bench scene topology for geometric ray-fan tracing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OpticalBenchScene {
    /// Concave circular mirror of radius $R$ forming a coffee-cup nephroid caustic with cusp at $x = R/2$.
    CoffeeCupMirror {
        /// Mirror circle radius $R > 0$.
        radius: f64,
    },
    /// Thick spherical biconvex dielectric lens producing longitudinal spherical aberration and cusp caustics.
    SphericalBiconvexLens {
        /// Spherical surface radius of curvature $R > 0$.
        radius_curv: f64,
        /// Axial half-thickness of the lens at $y = 0$.
        half_thickness: f64,
        /// Transverse semi-aperture height.
        aperture_radius: f64,
        /// Dispersive glass material.
        glass: GlassMaterial,
    },
    /// Triangular dispersive prism separating polychromatic light by Cauchy refractive index $n(\lambda)$.
    DispersivePrism {
        /// Prism apex angle $\alpha$ in radians.
        apex_angle_rad: f64,
        /// Slant side length of the isosceles prism.
        side_length: f64,
        /// In-plane rotation angle of the prism in radians.
        tilt_rad: f64,
        /// Dispersive glass material.
        glass: GlassMaterial,
    },
    /// Planar dielectric boundary at $x = x_0$ demonstrating Snell refraction, Brewster $\theta_B$, and TIR.
    PlanarDielectric {
        /// Interface x-coordinate.
        interface_x: f64,
        /// Incident refractive index $n_1$.
        n1: f64,
        /// Transmitted refractive index $n_2$.
        n2: f64,
    },
    /// Continuous graded-index metamaterial integrated via 4th-order Runge-Kutta (`RungeKutta4`).
    ContinuousGrin {
        /// Graded-index profile $n(x, y)$.
        medium: GrinMedium,
        /// Canonical eikonal step size $\Delta\tau$.
        step_tau: f64,
        /// Number of RK4 integration steps per ray.
        steps: usize,
    },
}

/// Configuration for a collimated or tilted ray fan on the 2D optical bench.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RayFanConfig {
    /// Optical element or continuous GRIN medium placed on the bench.
    pub scene: OpticalBenchScene,
    /// Number of transverse spatial rays in the incident fan.
    pub ray_count: usize,
    /// Half-width of the incident beam bundle.
    pub beam_half_width: f64,
    /// Launch plane x-coordinate.
    pub source_x: f64,
    /// Beam center y-offset.
    pub source_y: f64,
    /// Launch angle $\theta_0$ relative to $+x$ in radians.
    pub launch_angle_rad: f64,
    /// Central vacuum wavelength $\lambda$ in nanometers.
    pub wavelength_nm: f64,
    /// When true, expands each spatial ray into a 5-band visible spectrum ($420..=680\,\text{nm}$).
    pub polychromatic: bool,
    /// Polarization state for Fresnel power accounting.
    pub polarization: PolarizationMode,
    /// Half-extent of the optical bench bounding box.
    pub bench_half_extent: f64,
}

const POLY_BANDS_NM: [f64; 5] = [420.0, 480.0, 540.0, 600.0, 670.0];

/// Traces a ray fan across the configured optical bench scene into `workspace` with zero heap allocations.
#[verified_engine::verified]
pub fn trace_ray_fan(config: &RayFanConfig, workspace: &mut RayFanWorkspace) -> usize {
    let bands: &[f64] = if config.polychromatic {
        &POLY_BANDS_NM
    } else {
        std::slice::from_ref(&config.wavelength_nm)
    };
    let max_rays = workspace.max_rays;
    let max_pts = workspace.max_points_per_ray;
    let spatial_n = (config.ray_count.max(1))
        .min(max_rays / bands.len().max(1))
        .max(1);
    let (sin_a, cos_a) = config.launch_angle_rad.sin_cos();
    let dir0 = [cos_a, sin_a];
    let normal_perp = [-sin_a, cos_a];

    let mut out_idx = 0usize;
    for (b_idx, &wl_nm) in bands.iter().enumerate() {
        for r_idx in 0..spatial_n {
            if out_idx >= max_rays {
                break;
            }
            let u = if spatial_n <= 1 {
                0.0
            } else {
                2.0 * (r_idx as f64) / ((spatial_n - 1) as f64) - 1.0
            };
            let offset = u * config.beam_half_width;
            let origin = [
                config.source_x + offset * normal_perp[0],
                config.source_y + offset * normal_perp[1],
            ];
            let p_start = out_idx * max_pts;
            let mut meta = TracedRayMeta {
                point_start: p_start,
                point_len: 0,
                wavelength_nm: wl_nm,
                power: 1.0,
                exit_origin: origin,
                exit_dir: dir0,
                spectral_band: b_idx as u8,
                total_internal_reflection: false,
            };
            let ctx = RayStepCtx {
                cfg: config,
                origin,
                dir0,
                wl_nm,
            };
            {
                let points_buf = workspace.arena.region_mut(workspace.points);
                let p_slice = &mut points_buf[p_start..p_start + max_pts];
                trace_single_ray(&ctx, &mut workspace.rk4, p_slice, &mut meta);
            }
            workspace.write_meta(out_idx, &meta);
            out_idx += 1;
        }
    }
    workspace.active_rays = out_idx;
    out_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn luneburg_lens_focuses_parallel_bundle_onto_rim_within_1e6() {
        let mut ws = RayFanWorkspace::new(16, 256);
        let cfg = RayFanConfig {
            scene: OpticalBenchScene::ContinuousGrin {
                medium: GrinMedium::Luneburg { radius: 1.0 },
                step_tau: 0.01,
                steps: 180,
            },
            ray_count: 9,
            beam_half_width: 0.85,
            source_x: -1.5,
            source_y: 0.0,
            launch_angle_rad: 0.0,
            wavelength_nm: 550.0,
            polychromatic: false,
            polarization: PolarizationMode::Unpolarized,
            bench_half_extent: 2.0,
        };
        let n = trace_ray_fan(&cfg, &mut ws);
        assert_eq!(n, 9);
        for (i, meta) in ws.rays().enumerate() {
            let rim_err =
                ((meta.exit_origin[0] - 1.0).powi(2) + meta.exit_origin[1].powi(2)).sqrt();
            assert!(rim_err < 1.0e-6, "ray {i} Luneburg rim error = {rim_err}");
        }
    }

    #[test]
    #[verified_engine::verified]
    fn parabolic_waveguide_and_eaton_retroreflector_match_analytic() {
        let mut ws = RayFanWorkspace::new(8, 220);
        let (n0, g, y0, dt, steps) = (1.5, 0.5, 0.4, 0.01, 150usize);
        let wg = RayFanConfig {
            scene: OpticalBenchScene::ContinuousGrin {
                medium: GrinMedium::ParabolicWaveguide { n0, gradient_g: g },
                step_tau: dt,
                steps,
            },
            ray_count: 1,
            beam_half_width: 0.0,
            source_x: -1.5,
            source_y: y0,
            launch_angle_rad: 0.0,
            wavelength_nm: 550.0,
            polychromatic: false,
            polarization: PolarizationMode::Unpolarized,
            bench_half_extent: 4.0,
        };
        trace_ray_fan(&wg, &mut ws);
        let end_pt = ws.ray_vertices(0)[steps];
        let tau = (steps as f64) * dt;
        let expected_y = y0 * (n0 * g * tau).cos();
        assert!((end_pt[1] - expected_y).abs() < 1.0e-6);

        let eaton = RayFanConfig {
            scene: OpticalBenchScene::ContinuousGrin {
                medium: GrinMedium::EatonRetroreflector {
                    radius: 1.0,
                    core_eps: 0.05,
                },
                step_tau: 0.015,
                steps: 210,
            },
            source_y: 0.55,
            ..wg
        };
        trace_ray_fan(&eaton, &mut ws);
        assert!(ws.ray(0).exit_dir[0] < -0.85);
    }

    #[test]
    #[verified_engine::verified]
    fn prism_maxwell_fisheye_and_biconvex_lens_refract_and_focus() {
        let mut ws = RayFanWorkspace::new(16, 220);
        let prism = RayFanConfig {
            scene: OpticalBenchScene::DispersivePrism {
                apex_angle_rad: 55.0_f64.to_radians(),
                side_length: 1.5,
                tilt_rad: 0.0,
                glass: GlassMaterial::Sf11Flint,
            },
            ray_count: 1,
            beam_half_width: 0.0,
            source_x: -1.4,
            source_y: -0.45,
            launch_angle_rad: 0.38,
            wavelength_nm: 550.0,
            polychromatic: true,
            polarization: PolarizationMode::Unpolarized,
            bench_half_extent: 2.0,
        };
        assert_eq!(trace_ray_fan(&prism, &mut ws), 5);
        assert!(ws.ray(0).exit_dir[1] < ws.ray(4).exit_dir[1] - 0.05);

        let fisheye = RayFanConfig {
            scene: OpticalBenchScene::ContinuousGrin {
                medium: GrinMedium::MaxwellFisheye {
                    radius: 1.0,
                    n0: 1.0,
                },
                step_tau: 0.02,
                steps: 180,
            },
            source_x: -1.0,
            source_y: 0.0,
            launch_angle_rad: 0.25,
            polychromatic: false,
            ..prism
        };
        trace_ray_fan(&fisheye, &mut ws);
        let min_antipodal = ws
            .ray_vertices(0)
            .iter()
            .map(|p| ((p[0] - 1.0).powi(2) + p[1].powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
        assert!(min_antipodal < 0.025);
    }
}

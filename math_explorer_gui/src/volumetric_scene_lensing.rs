#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Zero-allocation thin-lens `Gaussian3D` projection, $1.5 R_s$ shadow masking, relativistic
//! beaming, Christoffel connection telemetry, and NFW vs. Burkert caustic polyline builders.
// [cite:gaussian_splatting]
// [cite:general_relativity_geodesics]
// [cite:dwarf_galaxy_empirical_dependencies]

use crate::camera::Camera3D;
use math_explorer::ai::gaussian_splatting::projection::{
    LensedProjectionOutcome, LensedSplatImageSpec, SplatCameraProjection, project_scene_lensed_into,
};
use math_explorer::constants::{C, SOLAR_MASS};
use math_explorer::physics::astrophysics::{
    HaloProfile, evaluate_relativistic_beaming, halo_lens_deflection_and_jacobian,
    schwarzschild_thin_lens_images, tangential_critical_radius_kpc,
    validate_christoffel_radial_connection,
};
use math_explorer::physics::high_energy::general_relativity::SchwarzschildBlackHole;
use nalgebra::{Point2, Vector2, Vector3};
use serde::{Deserialize, Serialize};

use super::{
    VolumetricSplatWorkspace, covariance_ellipse_ring, presets::ACCRETION_WORLD_RS,
    rasterize_splats_with_pan_into, splat_stroke_color,
};

/// Number of vertices in closed circular critical/shadow/caustic overlay rings.
pub const CAUSTIC_RING_POINTS: usize = 65;

/// Gravitational thin-lens model applied to projected 3D Gaussian splats.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplatLensMode {
    /// Standard unlensed perspective EWA splat projection.
    #[default]
    Off,
    /// Schwarzschild black hole dual-image Einstein ring with $1.5 R_s$ photon-sphere shadow.
    Schwarzschild,
    /// Navarro-Frenk-White (`NfwCusp`) cold dark-matter halo cusp shear & critical arcs.
    NfwCusp,
    /// Burkert (`BurkertCore`) constant-density cored dark-matter halo shear.
    BurkertCore,
}

impl SplatLensMode {
    /// Ordered list of lens modes for UI combo boxes.
    pub const ALL: [Self; 4] = [
        Self::Off,
        Self::Schwarzschild,
        Self::NfwCusp,
        Self::BurkertCore,
    ];

    /// Human-readable lens model label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off (Unlensed Perspective)",
            Self::Schwarzschild => "Schwarzschild Black Hole (Dual Einstein Ring)",
            Self::NfwCusp => "NFW Cusp Dark-Matter Halo",
            Self::BurkertCore => "Burkert Core Dark-Matter Halo",
        }
    }
}

/// Serializable thin-lens, relativistic beaming, and caustic overlay configuration for `AiTab`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SplatLensConfig {
    /// Active gravitational lens operator.
    pub mode: SplatLensMode,
    /// Interactive 2D viewport coordinates `[x, y]` of the gravitational lens center.
    pub center: [f64; 2],
    /// When `true`, pointer drag on the splat canvas repositions `center` instead of orbiting.
    pub drag_lens_on_canvas: bool,
    /// Effective Einstein radius $\theta_E$ in viewport units for `Schwarzschild` mode.
    pub einstein_radius: f64,
    /// Viewport Schwarzschild horizon scale $R_s$ (shadow masked inside $1.5 R_s$).
    pub viewport_rs: f64,
    /// Logarithmic halo virial mass $\log_{10}(M_{200}/M_\odot)$ for `NfwCusp` / `BurkertCore`.
    pub halo_log10_mass: f64,
    /// Characteristic halo scale/core radius $r_s$ or $r_0$ in viewport units.
    pub halo_scale_radius: f64,
    /// Logarithmic critical surface density $\log_{10}(\Sigma_{\text{crit}})$ controlling caustic size.
    pub log10_sigma_crit: f64,
    /// Peak inner orbital speed fraction $v_{\text{ISCO}} / c \in [0, 0.99]$ for Doppler/synchrotron beaming.
    pub peak_orbital_beta: f64,
    /// Ambient coronal magnetic energy density $U_B$ ($\text{J}\,\text{m}^{-3}$) for `synchrotron_power`.
    pub magnetic_u_b: f64,
    /// Overlays `NfwCusp` vs. `BurkertCore` critical/caustic curves and $1.5 R_s$ shadow horizon.
    pub show_caustic_overlay: bool,
}

impl Default for SplatLensConfig {
    fn default() -> Self {
        Self {
            mode: SplatLensMode::Off,
            center: [0.0, 0.0],
            drag_lens_on_canvas: false,
            einstein_radius: 0.62,
            viewport_rs: 0.16,
            halo_log10_mass: 11.4,
            halo_scale_radius: 0.55,
            log10_sigma_crit: 8.15,
            peak_orbital_beta: 0.42,
            magnetic_u_b: 1.5,
            show_caustic_overlay: true,
        }
    }
}

impl SplatLensConfig {
    /// Returns `true` when identical to [`Self::default`] for `#[serde(skip_serializing_if)]`.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Returns `true` when all numeric lens parameters are finite and within safe bounds.
    pub fn is_valid(&self) -> bool {
        self.center[0].is_finite()
            && (-6.0..=6.0).contains(&self.center[0])
            && self.center[1].is_finite()
            && (-6.0..=6.0).contains(&self.center[1])
            && self.einstein_radius.is_finite()
            && (0.05..=3.0).contains(&self.einstein_radius)
            && self.viewport_rs.is_finite()
            && (0.02..=1.5).contains(&self.viewport_rs)
            && self.halo_log10_mass.is_finite()
            && (8.0..=14.0).contains(&self.halo_log10_mass)
            && self.halo_scale_radius.is_finite()
            && (0.08..=4.0).contains(&self.halo_scale_radius)
            && self.log10_sigma_crit.is_finite()
            && (6.0..=11.0).contains(&self.log10_sigma_crit)
            && self.peak_orbital_beta.is_finite()
            && (0.0..=0.995).contains(&self.peak_orbital_beta)
            && self.magnetic_u_b.is_finite()
            && (0.0..=50.0).contains(&self.magnetic_u_b)
    }

    /// Clamps any out-of-range or non-finite numeric fields in-place to safe physical bounds.
    pub fn clamp_in_place(&mut self) {
        let def = Self::default();
        self.center[0] = if self.center[0].is_finite() {
            self.center[0].clamp(-6.0, 6.0)
        } else {
            def.center[0]
        };
        self.center[1] = if self.center[1].is_finite() {
            self.center[1].clamp(-6.0, 6.0)
        } else {
            def.center[1]
        };
        self.einstein_radius = if self.einstein_radius.is_finite() {
            self.einstein_radius.clamp(0.05, 3.0)
        } else {
            def.einstein_radius
        };
        self.viewport_rs = if self.viewport_rs.is_finite() {
            self.viewport_rs.clamp(0.02, 1.5)
        } else {
            def.viewport_rs
        };
        self.halo_log10_mass = if self.halo_log10_mass.is_finite() {
            self.halo_log10_mass.clamp(8.0, 14.0)
        } else {
            def.halo_log10_mass
        };
        self.halo_scale_radius = if self.halo_scale_radius.is_finite() {
            self.halo_scale_radius.clamp(0.08, 4.0)
        } else {
            def.halo_scale_radius
        };
        self.log10_sigma_crit = if self.log10_sigma_crit.is_finite() {
            self.log10_sigma_crit.clamp(6.0, 11.0)
        } else {
            def.log10_sigma_crit
        };
        self.peak_orbital_beta = if self.peak_orbital_beta.is_finite() {
            self.peak_orbital_beta.clamp(0.0, 0.995)
        } else {
            def.peak_orbital_beta
        };
        self.magnetic_u_b = if self.magnetic_u_b.is_finite() {
            self.magnetic_u_b.clamp(0.0, 50.0)
        } else {
            def.magnetic_u_b
        };
    }

    /// Feeds all lens fields into a `Hasher` for render-cache invalidation and URL revision tracking.
    pub fn hash_state(&self, h: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        self.mode.hash(h);
        self.center[0].to_bits().hash(h);
        self.center[1].to_bits().hash(h);
        self.drag_lens_on_canvas.hash(h);
        self.einstein_radius.to_bits().hash(h);
        self.viewport_rs.to_bits().hash(h);
        self.halo_log10_mass.to_bits().hash(h);
        self.halo_scale_radius.to_bits().hash(h);
        self.log10_sigma_crit.to_bits().hash(h);
        self.peak_orbital_beta.to_bits().hash(h);
        self.magnetic_u_b.to_bits().hash(h);
        self.show_caustic_overlay.hash(h);
    }

    /// Preset lens configuration tailored to [`super::presets::VolumetricSplatPreset::RelativisticAccretionDisk`].
    pub fn relativistic_disk_preset() -> Self {
        Self {
            mode: SplatLensMode::Schwarzschild,
            drag_lens_on_canvas: true,
            einstein_radius: 0.68,
            viewport_rs: 0.16,
            peak_orbital_beta: 0.48,
            magnetic_u_b: 2.0,
            show_caustic_overlay: true,
            ..Self::default()
        }
    }

    /// Builds the `NfwCusp` and `BurkertCore` halo profiles at matched mass & scale for comparison.
    pub fn matched_halos(&self) -> (HaloProfile, HaloProfile, f64) {
        let m200 = 10.0_f64.powf(self.halo_log10_mass.clamp(8.0, 14.0));
        let scale = self.halo_scale_radius.clamp(0.08, 4.0);
        let sigma_crit = 10.0_f64.powf(self.log10_sigma_crit.clamp(6.0, 11.0));
        let nfw = HaloProfile::NfwCusp {
            virial_mass_solar: m200,
            scale_radius_kpc: scale,
            concentration: 12.0,
        };
        let burk = HaloProfile::BurkertCore {
            virial_mass_solar: m200,
            core_radius_kpc: scale,
            concentration: 12.0,
        };
        (nfw, burk, sigma_crit)
    }
}

/// Frame-level relativistic lensing & beaming diagnostics displayed in `AiTab`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LensedFrameTelemetry {
    /// Number of lensed 2D Gaussian images emitted after shadow masking (`<= 2 * MAX_DYNAMIC_SPLATS`).
    pub emitted_images: usize,
    /// Number of rays occluded inside the $1.5 R_s$ photon-sphere shadow.
    pub shadow_masked_rays: usize,
    /// Photon-sphere shadow radius $1.5 R_s$ in viewport units.
    pub shadow_radius: f64,
    /// Tangential critical radius for `NfwCusp` (if supercritical).
    pub nfw_critical_radius: Option<f64>,
    /// Tangential critical radius for `BurkertCore` (if supercritical).
    pub burkert_critical_radius: Option<f64>,
    /// Peak boosted splat radiance multiplier across the frame.
    pub max_radiance_boost: f64,
    /// Equatorial Christoffel radial connection $\Gamma^r_{00}(6 R_s)$ from `christoffel_symbols_fixed`.
    pub christoffel_gamma_r00: f64,
}

/// Projects and rasterizes `ws.scene` through `lens` in-place with zero heap allocations.
#[allow(clippy::too_many_arguments)]
pub fn project_and_rasterize_lensed_workspace(
    ws: &mut VolumetricSplatWorkspace,
    camera: &Camera3D,
    camera_distance: f64,
    viewport_half_extent: f64,
    pan: [f64; 2],
    scale_multiplier: f64,
    max_rings: usize,
    lens: &SplatLensConfig,
) -> LensedFrameTelemetry {
    let view_matrix = camera.view_matrix(camera_distance);
    let focal = camera_distance * (camera.zoom as f64);
    let proj = SplatCameraProjection {
        view_matrix,
        focal: (focal, focal),
        scale_multiplier,
    };
    let bh = SchwarzschildBlackHole::new(1.0e8 * SOLAR_MASS).expect("valid lens black hole");
    let cam_forward = Vector3::new(
        view_matrix[(2, 0)],
        view_matrix[(2, 1)],
        view_matrix[(2, 2)],
    );
    let (nfw_halo, burk_halo, sigma_crit) = lens.matched_halos();
    let mut masked_rays = 0usize;
    let mut max_boost = 1.0_f64;
    let lens_center = Vector2::new(lens.center[0], lens.center[1]);

    project_scene_lensed_into(
        &ws.scene,
        &proj,
        |_idx, g, _cam_vec, beta_pt| {
            let rad_w = compute_splat_beaming_weight(&bh, &g.mean.coords, &cam_forward, lens);
            max_boost = max_boost.max(rad_w);
            evaluate_lens_mode_outcome(
                &bh,
                lens,
                lens_center,
                &nfw_halo,
                &burk_halo,
                sigma_crit,
                beta_pt,
                rad_w,
                &mut masked_rays,
            )
        },
        &mut ws.projected_splats,
    );

    populate_ellipse_overlays(ws, max_rings);
    rasterize_splats_with_pan_into(
        &ws.projected_splats,
        viewport_half_extent,
        pan,
        &mut ws.pixel_buffer,
    );

    let shadow_r = if lens.mode == SplatLensMode::Schwarzschild {
        1.5 * lens.viewport_rs.max(0.0)
    } else {
        0.0
    };
    let (gamma_r00, _) = validate_christoffel_radial_connection(&bh, 6.0).unwrap_or((0.0, 0.0));
    LensedFrameTelemetry {
        emitted_images: ws.projected_splats.len(),
        shadow_masked_rays: masked_rays,
        shadow_radius: shadow_r,
        nfw_critical_radius: tangential_critical_radius_kpc(None, &nfw_halo, sigma_crit, 6.0),
        burkert_critical_radius: tangential_critical_radius_kpc(None, &burk_halo, sigma_crit, 6.0),
        max_radiance_boost: max_boost,
        christoffel_gamma_r00: gamma_r00,
    }
}

pub(crate) fn compute_splat_beaming_weight(
    bh: &SchwarzschildBlackHole,
    world_pos: &Vector3<f64>,
    cam_forward: &Vector3<f64>,
    lens: &SplatLensConfig,
) -> f64 {
    if lens.mode == SplatLensMode::Off || lens.peak_orbital_beta <= 1.0e-4 {
        return 1.0;
    }
    let r_cyl = (world_pos.x * world_pos.x + world_pos.z * world_pos.z)
        .sqrt()
        .max(1.0e-4);
    let r_3d = world_pos.norm().max(1.001 * ACCRETION_WORLD_RS);
    let r_over_rs = (r_3d / ACCRETION_WORLD_RS).max(1.001);
    let tang_dir = Vector3::new(-world_pos.z / r_cyl, 0.0, world_pos.x / r_cyl);
    let los_cos = (-tang_dir.dot(cam_forward)).clamp(-1.0, 1.0);
    let kep_falloff = (3.0 / r_over_rs).clamp(0.05, 1.0).sqrt();
    let speed_m_s = (lens.peak_orbital_beta.clamp(0.0, 0.998) * kep_falloff) * C;
    match evaluate_relativistic_beaming(bh, r_over_rs, speed_m_s, los_cos, lens.magnetic_u_b) {
        Ok(sample) => sample.radiance_weight,
        Err(_) => 1.0,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_lens_mode_outcome(
    bh: &SchwarzschildBlackHole,
    lens: &SplatLensConfig,
    lens_center: Vector2<f64>,
    nfw_halo: &HaloProfile,
    burk_halo: &HaloProfile,
    sigma_crit: f64,
    beta_pt: Point2<f64>,
    radiance_weight: f64,
    masked_rays: &mut usize,
) -> LensedProjectionOutcome {
    let rel = Vector2::new(beta_pt.x - lens_center.x, beta_pt.y - lens_center.y);
    match lens.mode {
        SplatLensMode::Off => LensedProjectionOutcome {
            primary: Some(LensedSplatImageSpec {
                mean: beta_pt,
                lens_jacobian: nalgebra::Matrix2::identity(),
                radiance_weight: 1.0,
                opacity_weight: 1.0,
            }),
            secondary: None,
        },
        SplatLensMode::Schwarzschild => {
            evaluate_schwarzschild_outcome(bh, lens, lens_center, rel, radiance_weight, masked_rays)
        }
        SplatLensMode::NfwCusp | SplatLensMode::BurkertCore => {
            let halo = if lens.mode == SplatLensMode::NfwCusp {
                nfw_halo
            } else {
                burk_halo
            };
            let (defl, jac, sample) =
                halo_lens_deflection_and_jacobian(rel, None, halo, sigma_crit);
            let deflected = Point2::new(beta_pt.x + defl.x, beta_pt.y + defl.y);
            LensedProjectionOutcome {
                primary: Some(LensedSplatImageSpec {
                    mean: deflected,
                    lens_jacobian: jac,
                    radiance_weight,
                    opacity_weight: sample.magnification.abs().sqrt().clamp(0.45, 1.35),
                }),
                secondary: None,
            }
        }
    }
}

fn evaluate_schwarzschild_outcome(
    bh: &SchwarzschildBlackHole,
    lens: &SplatLensConfig,
    lens_center: Vector2<f64>,
    rel: Vector2<f64>,
    radiance_weight: f64,
    masked_rays: &mut usize,
) -> LensedProjectionOutcome {
    let Ok(dual) = schwarzschild_thin_lens_images(bh, rel, lens.einstein_radius, lens.viewport_rs)
    else {
        return LensedProjectionOutcome {
            primary: None,
            secondary: None,
        };
    };
    if dual.primary.is_none() {
        *masked_rays += 1;
    }
    if dual.secondary.is_none() {
        *masked_rays += 1;
    }
    LensedProjectionOutcome {
        primary: dual.primary.map(|img| LensedSplatImageSpec {
            mean: Point2::new(lens_center.x + img.offset.x, lens_center.y + img.offset.y),
            lens_jacobian: img.jacobian,
            radiance_weight,
            opacity_weight: img.magnification_abs.sqrt().clamp(0.35, 1.35),
        }),
        secondary: dual.secondary.map(|img| LensedSplatImageSpec {
            mean: Point2::new(lens_center.x + img.offset.x, lens_center.y + img.offset.y),
            lens_jacobian: img.jacobian,
            radiance_weight: radiance_weight * 0.85,
            opacity_weight: img.magnification_abs.sqrt().clamp(0.22, 1.15),
        }),
    }
}

fn populate_ellipse_overlays(ws: &mut VolumetricSplatWorkspace, max_rings: usize) {
    ws.ellipse_rings.clear();
    ws.splat_centers.clear();
    let ring_limit = max_rings
        .min(ws.ellipse_rings.capacity())
        .min(ws.projected_splats.len());
    for splat in ws.projected_splats.iter().take(ring_limit) {
        if ws.splat_centers.len() < ws.splat_centers.capacity() {
            ws.splat_centers.push([splat.mean.x, splat.mean.y]);
        }
        let neg_two_conic = -2.0 * splat.conic;
        if let Some(cov2d) = neg_two_conic.try_inverse() {
            ws.ellipse_rings.push((
                covariance_ellipse_ring(splat.mean, &cov2d),
                splat_stroke_color(&splat.color, splat.opacity),
            ));
        }
    }
}

/// Builds a stack-allocated closed circle polyline centered at `center` with `radius`.
pub fn build_circle_ring(center: [f64; 2], radius: f64) -> [[f64; 2]; CAUSTIC_RING_POINTS] {
    let r = radius.max(0.0);
    let mut pts = [[0.0_f64; 2]; CAUSTIC_RING_POINTS];
    let segs = (CAUSTIC_RING_POINTS - 1) as f64;
    for (idx, slot) in pts.iter_mut().enumerate() {
        let theta = (idx as f64) * std::f64::consts::TAU / segs;
        let (sin_t, cos_t) = theta.sin_cos();
        *slot = [center[0] + r * cos_t, center[1] + r * sin_t];
    }
    pts
}

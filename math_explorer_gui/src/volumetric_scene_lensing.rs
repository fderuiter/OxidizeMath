#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Zero-allocation thin-lens `Gaussian3D` projection, $1.5 R_s$ shadow masking, relativistic
//! beaming, Christoffel connection telemetry, and NFW vs. Burkert caustic polyline builders.
// [cite:gaussian_splatting]
// [cite:general_relativity_geodesics]
// [cite:dwarf_galaxy_empirical_dependencies]

use crate::camera::Camera3D;
use math_explorer::ai::gaussian_splatting::projection::{
    project_scene_lensed_into, LensedProjectionOutcome, LensedSplatImageSpec, SplatCameraProjection,
};
use math_explorer::constants::si::{C, SOLAR_MASS};
use math_explorer::physics::astrophysics::{
    evaluate_relativistic_beaming, halo_lens_deflection_and_jacobian,
    schwarzschild_thin_lens_images, tangential_critical_radius_kpc,
    validate_christoffel_radial_connection, HaloProfile, SchwarzschildLensImage,
};
use math_explorer::physics::high_energy::general_relativity::SchwarzschildBlackHole;
use math_scalars::{Kiloparsecs, ViewportLength};
use nalgebra::{Point2, Vector2, Vector3};
use serde::{Deserialize, Serialize};

use super::{
    covariance_ellipse_ring, presets::ACCRETION_WORLD_RS, rasterize_splats_with_pan_into,
    splat_stroke_color, VolumetricSplatWorkspace,
};

/// Number of vertices in closed circular critical/shadow/caustic overlay rings.
pub const CAUSTIC_RING_POINTS: usize = 65;

/// Default galactic field-of-view scale (`8.0 kpc` per viewport unit; `[-2.5, 2.5]` spans `[-20, 20] kpc`).
pub const DEFAULT_VIEWPORT_KPC_SCALE: f64 = 8.0;

const fn default_viewport_kpc_scale() -> f64 {
    DEFAULT_VIEWPORT_KPC_SCALE
}

fn in_range(val: f64, lo: f64, hi: f64) -> bool {
    val.is_finite() && (lo..=hi).contains(&val)
}

fn clamp_or(val: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(lo, hi)
    } else {
        fallback
    }
}

fn safe_kpc_scale(kpc_per_viewport: f64) -> f64 {
    clamp_or(kpc_per_viewport, 0.1, 25.0, DEFAULT_VIEWPORT_KPC_SCALE)
}

/// Converts a 1D viewport length (`ViewportFrame`) into physical `Kiloparsecs` (`GalacticKpcFrame`).
#[must_use]
pub fn viewport_to_kpc(vp: ViewportLength, kpc_per_viewport: f64) -> Kiloparsecs {
    vp.to_kiloparsecs(safe_kpc_scale(kpc_per_viewport))
}

/// Converts a physical `Kiloparsecs` (`GalacticKpcFrame`) length back into `ViewportLength` (`ViewportFrame`).
#[must_use]
pub fn kpc_to_viewport(kpc: Kiloparsecs, kpc_per_viewport: f64) -> ViewportLength {
    kpc.to_viewport(safe_kpc_scale(kpc_per_viewport))
}

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
    /// Physical kiloparsecs per viewport unit (`kpc / vp`) bridging `ViewportFrame` and `GalacticKpcFrame`.
    #[serde(default = "default_viewport_kpc_scale")]
    pub viewport_kpc_scale: f64,
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
            viewport_kpc_scale: DEFAULT_VIEWPORT_KPC_SCALE,
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
        in_range(self.center[0], -6.0, 6.0)
            && in_range(self.center[1], -6.0, 6.0)
            && in_range(self.einstein_radius, 0.05, 3.0)
            && in_range(self.viewport_rs, 0.02, 1.5)
            && in_range(self.halo_log10_mass, 8.0, 14.0)
            && in_range(self.halo_scale_radius, 0.08, 4.0)
            && in_range(self.viewport_kpc_scale, 0.1, 25.0)
            && in_range(self.log10_sigma_crit, 6.0, 11.0)
            && in_range(self.peak_orbital_beta, 0.0, 0.995)
            && in_range(self.magnetic_u_b, 0.0, 50.0)
    }

    /// Clamps any out-of-range or non-finite numeric fields in-place to safe physical bounds.
    pub fn clamp_in_place(&mut self) {
        let def = Self::default();
        self.center[0] = clamp_or(self.center[0], -6.0, 6.0, def.center[0]);
        self.center[1] = clamp_or(self.center[1], -6.0, 6.0, def.center[1]);
        self.einstein_radius = clamp_or(self.einstein_radius, 0.05, 3.0, def.einstein_radius);
        self.viewport_rs = clamp_or(self.viewport_rs, 0.02, 1.5, def.viewport_rs);
        self.halo_log10_mass = clamp_or(self.halo_log10_mass, 8.0, 14.0, def.halo_log10_mass);
        self.halo_scale_radius = clamp_or(self.halo_scale_radius, 0.08, 4.0, def.halo_scale_radius);
        self.viewport_kpc_scale =
            clamp_or(self.viewport_kpc_scale, 0.1, 25.0, def.viewport_kpc_scale);
        self.log10_sigma_crit = clamp_or(self.log10_sigma_crit, 6.0, 11.0, def.log10_sigma_crit);
        self.peak_orbital_beta =
            clamp_or(self.peak_orbital_beta, 0.0, 0.995, def.peak_orbital_beta);
        self.magnetic_u_b = clamp_or(self.magnetic_u_b, 0.0, 50.0, def.magnetic_u_b);
    }

    /// Feeds all lens fields into a `Hasher` for render-cache invalidation and URL revision tracking.
    pub fn hash_state(&self, h: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        (
            self.mode,
            self.drag_lens_on_canvas,
            self.show_caustic_overlay,
        )
            .hash(h);
        [
            self.center[0],
            self.center[1],
            self.einstein_radius,
            self.viewport_rs,
            self.halo_log10_mass,
            self.halo_scale_radius,
            self.viewport_kpc_scale,
            self.log10_sigma_crit,
            self.peak_orbital_beta,
            self.magnetic_u_b,
        ]
        .map(f64::to_bits)
        .hash(h);
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

    /// Builds the `NfwCusp` and `BurkertCore` halo profiles at matched mass & physical `kpc` scale for comparison.
    pub fn matched_halos(&self) -> (HaloProfile, HaloProfile, f64) {
        let m200 = 10.0_f64.powf(self.halo_log10_mass.clamp(8.0, 14.0));
        let scale_vp = ViewportLength::new(self.halo_scale_radius.clamp(0.08, 4.0));
        let scale_kpc = viewport_to_kpc(scale_vp, self.viewport_kpc_scale);
        let sigma_crit = 10.0_f64.powf(self.log10_sigma_crit.clamp(6.0, 11.0));
        let nfw = HaloProfile::NfwCusp {
            virial_mass_solar: m200,
            scale_radius_kpc: scale_kpc,
            concentration: 12.0,
        };
        let burk = HaloProfile::BurkertCore {
            virial_mass_solar: m200,
            core_radius_kpc: scale_kpc,
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
    let bh = SchwarzschildBlackHole::new(1.0e8 * SOLAR_MASS.get()).expect("valid lens black hole");
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
    build_frame_telemetry(
        &bh,
        lens,
        (&nfw_halo, &burk_halo, sigma_crit),
        (ws.projected_splats.len(), masked_rays, max_boost),
    )
}

fn build_frame_telemetry(
    bh: &SchwarzschildBlackHole,
    lens: &SplatLensConfig,
    (nfw_halo, burk_halo, sigma_crit): (&HaloProfile, &HaloProfile, f64),
    (emitted_images, shadow_masked_rays, max_radiance_boost): (usize, usize, f64),
) -> LensedFrameTelemetry {
    let shadow_radius = if lens.mode == SplatLensMode::Schwarzschild {
        1.5 * lens.viewport_rs.max(0.0)
    } else {
        0.0
    };
    let (gamma_r00, _) = validate_christoffel_radial_connection(bh, 6.0).unwrap_or((0.0, 0.0));
    let r_search_kpc = viewport_to_kpc(ViewportLength::new(6.0), lens.viewport_kpc_scale);
    let crit_vp = |h: &HaloProfile| {
        tangential_critical_radius_kpc(None, h, sigma_crit, r_search_kpc)
            .map(|r_kpc| kpc_to_viewport(r_kpc, lens.viewport_kpc_scale).get())
    };
    LensedFrameTelemetry {
        emitted_images,
        shadow_masked_rays,
        shadow_radius,
        nfw_critical_radius: crit_vp(nfw_halo),
        burkert_critical_radius: crit_vp(burk_halo),
        max_radiance_boost,
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
    let speed_m_s = (lens.peak_orbital_beta.clamp(0.0, 0.998) * kep_falloff) * C.get();
    evaluate_relativistic_beaming(bh, r_over_rs, speed_m_s, los_cos, lens.magnetic_u_b)
        .map_or(1.0, |sample| sample.radiance_weight)
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
            let rel_kpc =
                rel.map(|v| viewport_to_kpc(ViewportLength::new(v), lens.viewport_kpc_scale));
            let (defl_kpc, jac, sample) =
                halo_lens_deflection_and_jacobian(rel_kpc, None, halo, sigma_crit);
            let defl_vp = defl_kpc.map(|d| kpc_to_viewport(d, lens.viewport_kpc_scale));
            let deflected = Point2::new(beta_pt.x + defl_vp.x.get(), beta_pt.y + defl_vp.y.get());
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
    *masked_rays += usize::from(dual.primary.is_none()) + usize::from(dual.secondary.is_none());
    let map_img = |img: SchwarzschildLensImage, rad: f64, lo: f64, hi: f64| LensedSplatImageSpec {
        mean: Point2::new(lens_center.x + img.offset.x, lens_center.y + img.offset.y),
        lens_jacobian: img.jacobian,
        radiance_weight: rad,
        opacity_weight: img.magnification_abs.sqrt().clamp(lo, hi),
    };
    LensedProjectionOutcome {
        primary: dual
            .primary
            .map(|img| map_img(img, radiance_weight, 0.35, 1.35)),
        secondary: dual
            .secondary
            .map(|img| map_img(img, radiance_weight * 0.85, 0.22, 1.15)),
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

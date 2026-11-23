//! Refractive index distributions, Snell boundary laws, and canonical 2D optical presets.
//!
//! Provides analytical graded-index (Luneburg, Maxwell fish-eye), curved dielectric (biconvex
//! droplet), diffractive aperture (double-slit), and negative-index metamaterial (Veselago flat
//! superlens) media that configure [`WaveField2D`] in place without dynamic heap allocation.

use pure_math::pure_math::analysis::pde::wave::{
    WaveField2D, WaveFieldConfig2D, WaveSource2D, WaveSourceKind,
};
use serde::{Deserialize, Serialize};

// [cite:optics]

/// Canonical 2D wave and geometric-optics benchmark presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum OpticsPreset {
    /// Coherent plane wave diffracting through a two-slit conducting screen.
    DoubleSlit,
    /// Graded-index Luneburg sphere $n(r) = \sqrt{2 - (r/R)^2}$ focusing plane waves on its rim.
    #[default]
    LuneburgLens,
    /// Maxwell fish-eye graded-index lens $n(r) = 2 / (1 + (r/R)^2)$ imaging antipodal points.
    MaxwellFishEye,
    /// Spherical/circular dielectric droplet ($n = 1.50$) forming a spherical-aberration caustic cusp.
    BiconvexDroplet,
    /// Flat Veselago negative-index metamaterial slab ($n = -1.0$) with internal and external foci.
    VeselagoLens,
}

impl OpticsPreset {
    /// Ordered list of all supported optical presets.
    pub const ALL: [Self; 5] = [
        Self::DoubleSlit,
        Self::LuneburgLens,
        Self::MaxwellFishEye,
        Self::BiconvexDroplet,
        Self::VeselagoLens,
    ];

    /// Human-readable display label for UI controls and accessibility descriptions.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::DoubleSlit => "Double-Slit Aperture",
            Self::LuneburgLens => "Luneburg GRIN Lens",
            Self::MaxwellFishEye => "Maxwell Fish-Eye",
            Self::BiconvexDroplet => "Biconvex Droplet Caustic",
            Self::VeselagoLens => "Veselago Negative-Index Lens",
        }
    }
}

/// Physical parameters describing a 2D optical medium and coherent driving source.
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::optics::refraction::{
///     OpticalMediumConfig, OpticsPreset, snell_refracted_angle,
/// };
///
/// let medium = OpticalMediumConfig {
///     preset: OpticsPreset::LuneburgLens,
///     ..OpticalMediumConfig::default()
/// };
/// let [cx, cy] = medium.lens_center();
/// assert!((medium.refractive_index_at(cx, cy) - std::f64::consts::SQRT_2).abs() < 1e-12);
///
/// let neg_theta = snell_refracted_angle(1.0, -1.0, 0.4).expect("transmitted");
/// assert!((neg_theta + 0.4).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OpticalMediumConfig {
    /// Selected optical element or aperture preset.
    pub preset: OpticsPreset,
    /// Physical domain width $L_x > 0$.
    pub domain_width: f64,
    /// Physical domain height $L_y > 0$.
    pub domain_height: f64,
    /// Coherent source driving frequency $f > 0$.
    pub frequency: f64,
    /// Center-to-center slit separation $d$ for [`OpticsPreset::DoubleSlit`].
    pub slit_separation: f64,
    /// Slit opening width $a$ (or normalized beam/aperture scale).
    pub aperture_width: f64,
    /// Characteristic lens radius $R$ for graded-index and droplet lenses.
    pub lens_radius: f64,
    /// Dielectric refractive index $n > 1$ of [`OpticsPreset::BiconvexDroplet`].
    pub droplet_index: f64,
    /// Metamaterial refractive index $n < 0$ of [`OpticsPreset::VeselagoLens`].
    pub veselago_index: f64,
}

impl Default for OpticalMediumConfig {
    fn default() -> Self {
        Self {
            preset: OpticsPreset::LuneburgLens,
            domain_width: 1.0,
            domain_height: 1.0,
            frequency: 14.0,
            slit_separation: 0.18,
            aperture_width: 0.04,
            lens_radius: 0.22,
            droplet_index: 1.50,
            veselago_index: -1.0,
        }
    }
}

impl OpticalMediumConfig {
    /// Returns the `[x_c, y_c]` center of circular/spherical optical elements.
    #[inline]
    #[verified_engine::verified]
    pub fn lens_center(&self) -> [f64; 2] {
        let cx_frac = match self.preset {
            OpticsPreset::LuneburgLens => 0.46,
            OpticsPreset::BiconvexDroplet => 0.42,
            OpticsPreset::DoubleSlit
            | OpticsPreset::MaxwellFishEye
            | OpticsPreset::VeselagoLens => 0.50,
        };
        [cx_frac * self.domain_width, 0.50 * self.domain_height]
    }

    /// Returns the `[x_min, x_max]` slab boundaries for [`OpticsPreset::VeselagoLens`].
    #[inline]
    #[verified_engine::verified]
    pub fn veselago_slab_bounds(&self) -> [f64; 2] {
        [0.38 * self.domain_width, 0.62 * self.domain_width]
    }

    /// Returns the conducting barrier x-coordinate for [`OpticsPreset::DoubleSlit`].
    #[inline]
    #[verified_engine::verified]
    pub fn slit_wall_x(&self) -> f64 {
        0.28 * self.domain_width
    }

    /// Returns the point-source `[x_s, y_s]` coordinates for point-driven presets.
    #[inline]
    #[verified_engine::verified]
    pub fn source_position(&self) -> [f64; 2] {
        let [cx, cy] = self.lens_center();
        match self.preset {
            OpticsPreset::MaxwellFishEye => [cx - self.lens_radius, cy],
            OpticsPreset::VeselagoLens => [0.26 * self.domain_width, cy],
            OpticsPreset::DoubleSlit
            | OpticsPreset::LuneburgLens
            | OpticsPreset::BiconvexDroplet => [0.14 * self.domain_width, cy],
        }
    }

    /// Evaluates the signed refractive index $n(x, y)$ at physical coordinates `(x, y)`.
    #[inline]
    #[verified_engine::verified]
    pub fn refractive_index_at(&self, x: f64, y: f64) -> f64 {
        let [cx, cy] = self.lens_center();
        let r_hat_sq = ((x - cx) * (x - cx) + (y - cy) * (y - cy))
            / (self.lens_radius * self.lens_radius).max(1e-8);
        match self.preset {
            OpticsPreset::DoubleSlit => 1.0,
            OpticsPreset::LuneburgLens => {
                if r_hat_sq <= 1.0 {
                    (2.0 - r_hat_sq).sqrt()
                } else {
                    1.0
                }
            }
            OpticsPreset::MaxwellFishEye => (2.0 / (1.0 + r_hat_sq)).max(0.45),
            OpticsPreset::BiconvexDroplet => {
                if r_hat_sq <= 1.0 {
                    self.droplet_index.max(1.05)
                } else {
                    1.0
                }
            }
            OpticsPreset::VeselagoLens => {
                let [xa, xb] = self.veselago_slab_bounds();
                let y_pad = 0.08 * self.domain_height;
                if x >= xa && x <= xb && y >= y_pad && y <= self.domain_height - y_pad {
                    self.veselago_index.min(-0.2)
                } else {
                    1.0
                }
            }
        }
    }

    /// Evaluates the analytical gradient $\nabla(n^2)(x, y)$ in smooth graded-index regions.
    #[inline]
    #[verified_engine::verified]
    pub fn grad_n_squared(&self, x: f64, y: f64) -> [f64; 2] {
        let [cx, cy] = self.lens_center();
        let inv_r2 = 1.0 / (self.lens_radius * self.lens_radius).max(1e-8);
        let dx = x - cx;
        let dy = y - cy;
        let u = (dx * dx + dy * dy) * inv_r2;
        match self.preset {
            OpticsPreset::LuneburgLens => {
                if u <= 1.0 {
                    [-2.0 * dx * inv_r2, -2.0 * dy * inv_r2]
                } else {
                    [0.0, 0.0]
                }
            }
            OpticsPreset::MaxwellFishEye => {
                let n_val = 2.0 / (1.0 + u);
                if n_val > 0.45 {
                    let denom = (1.0 + u) * (1.0 + u) * (1.0 + u);
                    let factor = -16.0 * inv_r2 / denom;
                    [factor * dx, factor * dy]
                } else {
                    [0.0, 0.0]
                }
            }
            OpticsPreset::DoubleSlit
            | OpticsPreset::BiconvexDroplet
            | OpticsPreset::VeselagoLens => [0.0, 0.0],
        }
    }

    /// Identifies discrete refractive body membership (`0` = exterior, `1` = interior) for Snell jumps.
    #[inline]
    #[verified_engine::verified]
    pub fn discrete_region_id(&self, x: f64, y: f64) -> u8 {
        match self.preset {
            OpticsPreset::BiconvexDroplet => {
                let [cx, cy] = self.lens_center();
                let dx = x - cx;
                let dy = y - cy;
                u8::from(dx * dx + dy * dy <= self.lens_radius * self.lens_radius)
            }
            OpticsPreset::VeselagoLens => {
                let [xa, xb] = self.veselago_slab_bounds();
                let y_pad = 0.08 * self.domain_height;
                u8::from(x >= xa && x <= xb && y >= y_pad && y <= self.domain_height - y_pad)
            }
            OpticsPreset::DoubleSlit
            | OpticsPreset::LuneburgLens
            | OpticsPreset::MaxwellFishEye => 0,
        }
    }

    /// Returns the unit interface normal at `(x, y)` oriented along `incoming_dir`.
    #[inline]
    #[verified_engine::verified]
    pub fn interface_normal(&self, x: f64, y: f64, incoming_dir: [f64; 2]) -> [f64; 2] {
        let raw = match self.preset {
            OpticsPreset::BiconvexDroplet => {
                let [cx, cy] = self.lens_center();
                let rx = x - cx;
                let ry = y - cy;
                let norm = (rx * rx + ry * ry).sqrt().max(1e-12);
                [rx / norm, ry / norm]
            }
            OpticsPreset::VeselagoLens => [1.0, 0.0],
            OpticsPreset::DoubleSlit
            | OpticsPreset::LuneburgLens
            | OpticsPreset::MaxwellFishEye => [1.0, 0.0],
        };
        let dot = raw[0] * incoming_dir[0] + raw[1] * incoming_dir[1];
        if dot >= 0.0 { raw } else { [-raw[0], -raw[1]] }
    }

    /// Evaluates whether physical point `(x, y)` lies inside an opaque conducting barrier.
    #[inline]
    #[verified_engine::verified]
    pub fn is_obstacle_at(&self, x: f64, y: f64, grid_dx: f64) -> bool {
        if self.preset != OpticsPreset::DoubleSlit {
            return false;
        }
        let wall_x = self.slit_wall_x();
        let half_wall = (0.75 * grid_dx).max(0.008 * self.domain_width);
        if (x - wall_x).abs() > half_wall {
            return false;
        }
        let cy = 0.5 * self.domain_height;
        let half_sep = 0.5 * self.slit_separation;
        let half_ap = (0.5 * self.aperture_width).max(grid_dx);
        let in_upper = (y - (cy + half_sep)).abs() <= half_ap;
        let in_lower = (y - (cy - half_sep)).abs() <= half_ap;
        !(in_upper || in_lower)
    }

    /// Constructs the canonical coherent [`WaveSource2D`] matching this preset on `cfg`.
    #[verified_engine::verified]
    pub fn default_wave_source(&self, cfg: &WaveFieldConfig2D) -> WaveSource2D {
        let margin = cfg.sponge_width + 4;
        match self.preset {
            OpticsPreset::DoubleSlit
            | OpticsPreset::LuneburgLens
            | OpticsPreset::BiconvexDroplet => {
                let x_idx = ((0.14 * self.domain_width / cfg.dx).round() as usize)
                    .clamp(margin, cfg.width.saturating_sub(margin + 1));
                WaveSource2D {
                    kind: WaveSourceKind::PlaneWave {
                        x_index: x_idx,
                        y_min: margin,
                        y_max: cfg.height.saturating_sub(margin + 1),
                    },
                    frequency: self.frequency,
                    amplitude: 110.0,
                    phase: 0.0,
                }
            }
            OpticsPreset::MaxwellFishEye | OpticsPreset::VeselagoLens => {
                let [sx, sy] = self.source_position();
                let gx = ((sx / cfg.dx).round() as usize)
                    .clamp(margin, cfg.width.saturating_sub(margin + 1));
                let gy = ((sy / cfg.dy).round() as usize)
                    .clamp(margin, cfg.height.saturating_sub(margin + 1));
                WaveSource2D {
                    kind: WaveSourceKind::PointSource { x: gx, y: gy },
                    frequency: self.frequency,
                    amplitude: 180.0,
                    phase: 0.0,
                }
            }
        }
    }

    /// Populates refractive indices, obstacle masks, and Drude tuning on `field` in place.
    #[verified_engine::verified]
    pub fn apply_to_wave_field(&self, field: &mut WaveField2D) {
        field.set_reference_frequency(self.frequency);
        field.configure_medium_in_place(|gx, gy, cfg, n_out, obs_out| {
            let x = (gx as f64 + 0.5) * cfg.dx;
            let y = (gy as f64 + 0.5) * cfg.dy;
            *n_out = self.refractive_index_at(x, y);
            *obs_out = u8::from(self.is_obstacle_at(x, y, cfg.dx));
        });
    }
}

/// Computes the transmitted Snell angle $\theta_2$ in radians, supporting negative-index media ($n_2 < 0$).
///
/// Returns `None` when total internal reflection occurs ($|(n_1 / n_2) \sin\theta_1| > 1$).
#[inline]
#[verified_engine::verified]
pub fn snell_refracted_angle(n1: f64, n2: f64, incident_angle_rad: f64) -> Option<f64> {
    if n2.abs() < 1e-12 {
        return None;
    }
    let sin_t2 = (n1 / n2) * incident_angle_rad.sin();
    if sin_t2.abs() > 1.0 {
        None
    } else {
        Some(sin_t2.asin())
    }
}

/// Computes the unit refracted (or total-internally-reflected) ray direction across an interface.
///
/// `normal_into_medium` must be unit-length and oriented into the destination medium
/// ($\hat{\mathbf{s}}_1 \cdot \hat{\mathbf{N}} \ge 0$). When $n_1 / n_2 < 0$ (Veselago interface),
/// the tangential component reverses sign across the normal while forward energy flow continues.
#[inline]
#[verified_engine::verified]
pub fn snell_refract_vector(
    incident_unit: [f64; 2],
    normal_into_medium: [f64; 2],
    n1: f64,
    n2: f64,
) -> [f64; 2] {
    let cos_t1 =
        incident_unit[0] * normal_into_medium[0] + incident_unit[1] * normal_into_medium[1];
    let tang_x = incident_unit[0] - cos_t1 * normal_into_medium[0];
    let tang_y = incident_unit[1] - cos_t1 * normal_into_medium[1];
    let ratio = n1 / n2;
    let t2_x = ratio * tang_x;
    let t2_y = ratio * tang_y;
    let sin2_t2 = t2_x * t2_x + t2_y * t2_y;
    if sin2_t2 > 1.0 {
        // Total internal reflection
        let rx = incident_unit[0] - 2.0 * cos_t1 * normal_into_medium[0];
        let ry = incident_unit[1] - 2.0 * cos_t1 * normal_into_medium[1];
        let norm = (rx * rx + ry * ry).sqrt().max(1e-12);
        [rx / norm, ry / norm]
    } else {
        let cos_t2 = (1.0 - sin2_t2).sqrt();
        let tx = t2_x + cos_t2 * normal_into_medium[0];
        let ty = t2_y + cos_t2 * normal_into_medium[1];
        let norm = (tx * tx + ty * ty).sqrt().max(1e-12);
        [tx / norm, ty / norm]
    }
}

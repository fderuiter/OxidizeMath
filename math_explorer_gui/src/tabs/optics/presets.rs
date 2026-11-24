#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Shared WCAG-compliant optics colormaps, scene selectors, and federated starter gallery presets.

// [cite:optics]

use eframe::egui::Color32;
use math_explorer::math_kernel::colormap::{
    cyclic_cosine_palette_high_contrast, ensure_dark_canvas_contrast, heatmap_color,
    heatmap_color_high_contrast,
};
use math_explorer::physics::optics::{
    DiffractionAperture, GlassMaterial, GrinMedium, OpticalBenchScene,
};
use serde::{Deserialize, Serialize};

use super::diffraction_tool::FraunhoferDiffractionTool;
use super::ray_bench_tool::CausticsRayBenchTool;

pub(crate) fn clamp_finite(val: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

/// WCAG-compliant colormaps for ray dispersion, caustic density, and 2D diffraction irradiance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OpticsColormap {
    /// Perceptually monotonic thermal ramp (`heatmap_color`).
    #[default]
    PerceptualThermal,
    /// Wavelength-mapped cosine spectrum with dark-canvas contrast floor.
    SpectralDispersion,
    /// High-contrast luminance ramp enforcing WCAG >= 4.5:1 foreground contrast.
    WcagHighContrast,
}

impl OpticsColormap {
    /// Human-readable label for UI combo boxes.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PerceptualThermal => "Perceptual Thermal",
            Self::SpectralDispersion => "Spectral Dispersion",
            Self::WcagHighContrast => "WCAG High Contrast",
        }
    }

    /// Maps normalized intensity `t` in `[0, 1]` and vacuum `wavelength_nm` to `Color32`.
    #[must_use]
    pub fn sample_irradiance(self, t: f64, wavelength_nm: f64) -> Color32 {
        let u = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if u <= 1.0e-5 {
            return Color32::BLACK;
        }
        let rgb = match self {
            Self::PerceptualThermal => heatmap_color(u),
            Self::SpectralDispersion => {
                let phase = ((700.0 - wavelength_nm.clamp(380.0, 700.0)) / 360.0).clamp(0.0, 0.9);
                let base = cyclic_cosine_palette_high_contrast(phase);
                let gain = u.sqrt();
                [
                    ((base[0] as f64) * gain).round() as u8,
                    ((base[1] as f64) * gain).round() as u8,
                    ((base[2] as f64) * gain).round() as u8,
                ]
            }
            Self::WcagHighContrast => {
                let hi = ensure_dark_canvas_contrast(heatmap_color_high_contrast(u));
                let gain = (0.20 + 0.80 * u.sqrt()).clamp(0.0, 1.0);
                [
                    ((hi[0] as f64) * gain).round() as u8,
                    ((hi[1] as f64) * gain).round() as u8,
                    ((hi[2] as f64) * gain).round() as u8,
                ]
            }
        };
        Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    }

    /// High-contrast stroke color for a ray of `wavelength_nm` drawn against a dark optical bench.
    #[must_use]
    pub fn ray_stroke_color(self, wavelength_nm: f64) -> Color32 {
        let rgb = match self {
            Self::PerceptualThermal => ensure_dark_canvas_contrast(heatmap_color(0.78)),
            Self::SpectralDispersion => {
                let phase = ((700.0 - wavelength_nm.clamp(380.0, 700.0)) / 360.0).clamp(0.0, 0.9);
                ensure_dark_canvas_contrast(cyclic_cosine_palette_high_contrast(phase))
            }
            Self::WcagHighContrast => {
                ensure_dark_canvas_contrast(heatmap_color_high_contrast(0.88))
            }
        };
        Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    }
}

/// Selectable optical bench scene topology for [`CausticsRayBenchTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RayBenchSceneChoice {
    /// Concave semicircular mirror generating a coffee-cup nephroid caustic.
    CoffeeCupNephroid,
    /// Thick spherical biconvex lens exhibiting spherical aberration and axial cusp caustics.
    #[default]
    SphericalBiconvexLens,
    /// Continuous Luneburg metamaterial sphere focusing parallel rays onto its rim.
    LuneburgMetamaterial,
    /// Continuous Maxwell fisheye sphere imaging interior rays antipodally.
    MaxwellFisheye,
    /// Continuous Eaton $180^\circ$ retroreflector bending rays back toward the source.
    EatonRetroreflector,
    /// Parabolic graded-index slab waveguide supporting sinusoidal ray oscillations.
    ParabolicGrinFiber,
    /// Dispersive triangular prism separating polychromatic light via Cauchy's equation.
    DispersivePrism,
    /// Planar dielectric boundary demonstrating Snell refraction, Brewster $\theta_B$, and TIR.
    PlanarFresnelInterface,
}

impl RayBenchSceneChoice {
    /// Human-readable scene title.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::CoffeeCupNephroid => "Coffee-Cup Nephroid Mirror",
            Self::SphericalBiconvexLens => "Spherical Biconvex Lens",
            Self::LuneburgMetamaterial => "Luneburg Metamaterial Lens",
            Self::MaxwellFisheye => "Maxwell Fisheye Sphere",
            Self::EatonRetroreflector => "Eaton 180° Retroreflector",
            Self::ParabolicGrinFiber => "Parabolic GRIN Waveguide",
            Self::DispersivePrism => "Dispersive Triangular Prism",
            Self::PlanarFresnelInterface => "Planar Fresnel / TIR Interface",
        }
    }

    /// Converts UI parameters into the `domain_physics` [`OpticalBenchScene`].
    #[must_use]
    pub fn build_scene(
        self,
        scale: f64,
        prism_apex_deg: f64,
        glass: GlassMaterial,
        interface_n2: f64,
    ) -> OpticalBenchScene {
        let r = scale.clamp(0.4, 2.4);
        match self {
            Self::CoffeeCupNephroid => OpticalBenchScene::CoffeeCupMirror { radius: r },
            Self::SphericalBiconvexLens => OpticalBenchScene::SphericalBiconvexLens {
                radius_curv: 1.35 * r,
                half_thickness: 0.18 * r,
                aperture_radius: 0.72 * r,
                glass,
            },
            Self::LuneburgMetamaterial => OpticalBenchScene::ContinuousGrin {
                medium: GrinMedium::Luneburg { radius: r },
                step_tau: 0.025,
                steps: 170,
            },
            Self::MaxwellFisheye => OpticalBenchScene::ContinuousGrin {
                medium: GrinMedium::MaxwellFisheye { radius: r, n0: 1.0 },
                step_tau: 0.025,
                steps: 180,
            },
            Self::EatonRetroreflector => OpticalBenchScene::ContinuousGrin {
                medium: GrinMedium::EatonRetroreflector {
                    radius: r,
                    core_eps: 0.08,
                },
                step_tau: 0.02,
                steps: 190,
            },
            Self::ParabolicGrinFiber => OpticalBenchScene::ContinuousGrin {
                medium: GrinMedium::ParabolicWaveguide {
                    n0: 1.55,
                    gradient_g: 0.65 / r.max(0.4),
                },
                step_tau: 0.03,
                steps: 160,
            },
            Self::DispersivePrism => OpticalBenchScene::DispersivePrism {
                apex_angle_rad: prism_apex_deg.clamp(25.0, 75.0).to_radians(),
                side_length: 1.25 * r,
                tilt_rad: 0.0,
                glass,
            },
            Self::PlanarFresnelInterface => OpticalBenchScene::PlanarDielectric {
                interface_x: 0.0,
                n1: 1.5168,
                n2: interface_n2.clamp(1.0, 2.5),
            },
        }
    }
}

/// Secondary analysis plot mode in [`CausticsRayBenchTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RaySecondaryPlot {
    /// Polarized Fresnel reflectance $R_s(\theta_i), R_p(\theta_i)$ with Brewster & critical angles.
    #[default]
    FresnelPolarization,
    /// Transverse detector screen intensity histogram $I(y)$ at $x = x_{\text{det}}$.
    DetectorCausticHistogram,
}

impl RaySecondaryPlot {
    /// Human-readable plot label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::FresnelPolarization => "Fresnel Polarization R_s / R_p",
            Self::DetectorCausticHistogram => "Detector Caustic Histogram",
        }
    }
}

/// Aperture selector for [`FraunhoferDiffractionTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ApertureChoice {
    /// Single rectangular slit of width $a$.
    SingleSlit,
    /// Periodic $N$-slit diffraction grating.
    MultiSlitGrating,
    /// Circular aperture Airy disk ($2 J_1(x)/x$).
    #[default]
    CircularAiry,
}

impl ApertureChoice {
    /// Human-readable aperture name.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SingleSlit => "Single Slit (sinc²)",
            Self::MultiSlitGrating => "Multi-Slit Grating",
            Self::CircularAiry => "Circular Aperture (Airy Disk)",
        }
    }

    /// Converts UI parameters into [`DiffractionAperture`].
    #[must_use]
    pub fn build_aperture(
        self,
        slit_width_um: f64,
        slit_pitch_um: f64,
        slit_count: usize,
        airy_diameter_um: f64,
    ) -> DiffractionAperture {
        match self {
            Self::SingleSlit => DiffractionAperture::SingleSlit { slit_width_um },
            Self::MultiSlitGrating => DiffractionAperture::MultiSlitGrating {
                slit_width_um,
                slit_pitch_um: slit_pitch_um.max(slit_width_um),
                slit_count,
            },
            Self::CircularAiry => DiffractionAperture::CircularAiry {
                diameter_um: airy_diameter_um,
            },
        }
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CausticsRayBenchTool",
        display_name: "Caustics & Ray Bench",
        domain: "optics",
        tags: &["optics", "caustics", "refraction", "fresnel", "luneburg", "grin"],
        build: || Box::new(CausticsRayBenchTool::default()),
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "FraunhoferDiffractionTool",
        display_name: "Fraunhofer Diffraction Lab",
        domain: "optics",
        tags: &["optics", "diffraction", "fraunhofer", "airy", "grating", "bessel"],
        build: || Box::new(FraunhoferDiffractionTool::default()),
    }
}

const PRESET_COFFEE_CUP_NEPHROID: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Coffee-Cup Nephroid","domain":"optics","timestamp":"2026-11-23T00:00:00Z"},"#,
    r#""config":{"scene":"CoffeeCupNephroid","glass":"Bk7Crown","polarization":"Unpolarized","colormap":"PerceptualThermal","#,
    r#""secondary_plot":"DetectorCausticHistogram","ray_count":45,"beam_half_width":1.52,"source_x":-2.2,"source_y":0.0,"#,
    r#""launch_angle_rad":0.0,"wavelength_nm":560.0,"polychromatic":false,"element_scale":1.6,"prism_apex_deg":55.0,"#,
    r#""interface_n2":1.0,"detector_x":0.8,"show_caustic_envelope":true}}"#
);

const PRESET_LUNEBURG_METAMATERIAL: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Luneburg Metamaterial","domain":"optics","timestamp":"2026-11-23T00:00:00Z"},"#,
    r#""config":{"scene":"LuneburgMetamaterial","glass":"Bk7Crown","polarization":"Unpolarized","colormap":"WcagHighContrast","#,
    r#""secondary_plot":"DetectorCausticHistogram","ray_count":27,"beam_half_width":1.15,"source_x":-2.1,"source_y":0.0,"#,
    r#""launch_angle_rad":0.0,"wavelength_nm":532.0,"polychromatic":false,"element_scale":1.3,"prism_apex_deg":55.0,"#,
    r#""interface_n2":1.0,"detector_x":1.3,"show_caustic_envelope":true}}"#
);

const PRESET_DISPERSIVE_PRISM: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Dispersive Prism","domain":"optics","timestamp":"2026-11-23T00:00:00Z"},"#,
    r#""config":{"scene":"DispersivePrism","glass":"Sf11Flint","polarization":"PPolarized","colormap":"SpectralDispersion","#,
    r#""secondary_plot":"FresnelPolarization","ray_count":5,"beam_half_width":0.08,"source_x":-1.8,"source_y":-0.48,"#,
    r#""launch_angle_rad":0.38,"wavelength_nm":550.0,"polychromatic":true,"element_scale":1.2,"prism_apex_deg":55.0,"#,
    r#""interface_n2":1.78,"detector_x":1.6,"show_caustic_envelope":false}}"#
);

const PRESET_FRAUNHOFER_GRATING: &str = concat!(
    r#"{"metadata":{"schema_version":"1.0","title":"Fraunhofer Grating","domain":"optics","timestamp":"2026-11-23T00:00:00Z"},"#,
    r#""config":{"aperture":"MultiSlitGrating","colormap":"SpectralDispersion","wavelength_nm":589.0,"slit_width_um":2.2,"#,
    r#""slit_pitch_um":8.8,"slit_count":6,"airy_diameter_um":14.0,"screen_distance_m":1.2,"half_angle_rad":0.24,"#,
    r#""log_intensity_view":false,"show_critical_markers":true}}"#
);

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "optics_coffee_cup_nephroid_starter",
        title: "Coffee-Cup Nephroid",
        description: "Reflect collimated rays inside a circular mirror to reveal the nephroid caustic cusp at R/2.",
        category: "Geometric Optics & Caustics",
        icon: "☕",
        domain: "optics",
        tool_name: Some("CausticsRayBenchTool"),
        preset_generator: || PRESET_COFFEE_CUP_NEPHROID,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "optics_luneburg_metamaterial_starter",
        title: "Luneburg Metamaterial",
        description: "Integrate RK4 eikonal rays across n(r)=sqrt(2-(r/R)^2) focusing a plane wave onto the sphere rim.",
        category: "Geometric Optics & Caustics",
        icon: "🔮",
        domain: "optics",
        tool_name: Some("CausticsRayBenchTool"),
        preset_generator: || PRESET_LUNEBURG_METAMATERIAL,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "optics_dispersive_prism_starter",
        title: "Dispersive Prism",
        description: "Refract polychromatic visible rays through SF11 dense flint glass via Cauchy dispersion and Fresnel polarization.",
        category: "Geometric Optics & Caustics",
        icon: "🌈",
        domain: "optics",
        tool_name: Some("CausticsRayBenchTool"),
        preset_generator: || PRESET_DISPERSIVE_PRISM,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "optics_fraunhofer_grating_starter",
        title: "Fraunhofer Grating",
        description: "Resolve N-slit principal diffraction maxima, sinc² single-slit envelope minima, and missing orders.",
        category: "Wave Diffraction",
        icon: "📊",
        domain: "optics",
        tool_name: Some("FraunhoferDiffractionTool"),
        preset_generator: || PRESET_FRAUNHOFER_GRATING,
    }
}

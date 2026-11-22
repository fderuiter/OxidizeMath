#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Starter preset JSON envelope for Luneburg GRIN lens focusing and ray caustics.
pub const PRESET_OPTICS_LUNEBURG: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Luneburg Lens Caustics",
    "domain": "optics_waves",
    "timestamp": "2026-01-01T00:00:00Z"
  },
  "config": {
    "preset": "LuneburgLens",
    "grid_size": 128,
    "frequency": 15.5,
    "slit_separation": 0.18,
    "aperture_width": 0.04,
    "lens_radius": 0.22,
    "detector_x": 0.68,
    "cfl_scale": 0.95,
    "show_rays": true,
    "show_irradiance": false,
    "ray_count": 15,
    "adc_bits": 12
  }
}"#;

/// Starter preset JSON envelope for double-slit diffraction and calibrated detector fringes.
pub const PRESET_OPTICS_DOUBLE_SLIT: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Double-Slit Sensor Interference",
    "domain": "optics_waves",
    "timestamp": "2026-01-01T00:00:00Z"
  },
  "config": {
    "preset": "DoubleSlit",
    "grid_size": 128,
    "frequency": 15.0,
    "slit_separation": 0.18,
    "aperture_width": 0.035,
    "lens_radius": 0.22,
    "detector_x": 0.74,
    "cfl_scale": 0.95,
    "show_rays": false,
    "show_irradiance": true,
    "ray_count": 13,
    "adc_bits": 12
  }
}"#;

/// Starter preset JSON envelope for a flat negative-index Veselago metamaterial superlens.
pub const PRESET_OPTICS_VESELAGO: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Veselago Negative-Index Superlens",
    "domain": "optics_waves",
    "timestamp": "2026-01-01T00:00:00Z"
  },
  "config": {
    "preset": "VeselagoLens",
    "grid_size": 128,
    "frequency": 13.0,
    "slit_separation": 0.18,
    "aperture_width": 0.04,
    "lens_radius": 0.22,
    "detector_x": 0.74,
    "cfl_scale": 0.95,
    "show_rays": true,
    "show_irradiance": false,
    "ray_count": 15,
    "adc_bits": 14
  }
}"#;

/// Theoretical metadata provider for [`super::tool::OpticsWaveTool`].
#[derive(Clone, Debug, Default)]
pub struct OpticsWaveTheory;

impl TheoryDescribable for OpticsWaveTheory {
    fn theory_description(&self) -> String {
        "2D Finite-Difference Time-Domain (FDTD) inhomogeneous wave propagation coupled with \
         RK4 Hamiltonian geometric ray tracing, negative-index Drude metamaterial refraction, \
         caustic cusp detection, and calibrated optical detector plane metrics."
            .to_string()
    }

    fn phonetic_description(&self) -> String {
        "Two dimensional wave optics and Hamiltonian ray caustics: n squared over c naught squared \
         times second time derivative of u equals Laplacian of u plus harmonic source S, paired \
         with Snell refraction and Lambertian detector irradiance."
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Luneburg, R. K. (1944). Mathematical Theory of Optics; Veselago, V. G. (1968). \
         The Electrodynamics of Substances with Simultaneously Negative Values of epsilon and mu. \
         Soviet Physics Uspekhi, 10(4), 509-514; Born, M., & Wolf, E. (1999). Principles of Optics."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "fdtd_wave_2d".to_string(),
            "Centered leapfrog 2D wave solver with quadratic sponge boundaries and matched-Drude metamaterials."
                .to_string(),
        );
        map.insert(
            "hamiltonian_caustics".to_string(),
            "RK4 Hamiltonian ray bundle tracer locating geometric caustic folds via ray-tube Jacobian sign flips."
                .to_string(),
        );
        map.insert(
            "detector_sensing".to_string(),
            "Virtual detector line reporting wave irradiance contrast, Lambertian intensity, and LWS range resolution."
                .to_string(),
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\frac{n(x,y)^2}{c_0^2}\frac{\partial^2 u}{\partial t^2} = \nabla^2 u + S(x,y,t),\quad \frac{d\mathbf{r}}{d\tau}=\mathbf{p},\quad \frac{d\mathbf{p}}{d\tau}=\frac{1}{2}\nabla(n^2)"
                .to_string(),
        )
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "optics_luneburg_caustics",
        title: "Luneburg Lens Caustics",
        description: "n(r)=√(2−(r/R)²): Plane waves bend continuously inside a graded-index sphere onto a aberration-free rim focus.",
        category: "Wave & Ray Optics",
        icon: "🔮",
        domain: "optics_waves",
        tool_name: Some("OpticsWaveTool"),
        preset_generator: || PRESET_OPTICS_LUNEBURG,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "optics_double_slit_sensor",
        title: "Double-Slit Sensor Interference",
        description: "Δy = λL/d: Coherent diffraction fringes measured along a movable detector plane with Lambertian and LWS readouts.",
        category: "Diffraction & Sensing",
        icon: "🌈",
        domain: "optics_waves",
        tool_name: Some("OpticsWaveTool"),
        preset_generator: || PRESET_OPTICS_DOUBLE_SLIT,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "optics_veselago_superlens",
        title: "Veselago Negative-Index Superlens",
        description: "n = −1: Matched-Drude metamaterial slab reverses Snell refraction to form internal and external foci.",
        category: "Metamaterials",
        icon: "🔬",
        domain: "optics_waves",
        tool_name: Some("OpticsWaveTool"),
        preset_generator: || PRESET_OPTICS_VESELAGO,
    }
}

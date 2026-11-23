#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Curated starter presets for [`super::workbench::FourierOpticsWorkbenchTool`] in the Starter Gallery.

// [cite:graph_parameters_rust]

use crate::starter_gallery::StarterPresetMetadata;

/// Starter preset 1: Arago-Poisson bright spot behind an opaque circular disk at Fresnel number N_F = 2.0.
pub const PRESET_ARAGO_BRIGHT_SPOT: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Fourier Optics Workbench",
    "domain": "optics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "HelmholtzDiffraction",
    "aperture": "OpaqueCircularDisk",
    "display": "IntensityXY",
    "wavelength_nm": 500.0,
    "pixel_pitch_um": 8.0,
    "propagation_mm": 16.384,
    "carpet_max_mm": 40.0,
    "feature_scale_um": 128.0,
    "slit_pitch_um": 96.0,
    "slit_count": 2,
    "duty_cycle": 0.5,
    "focal_length_mm": 14.0,
    "zernike_n": 0,
    "zernike_m": 0,
    "zernike_rms_waves": 0.0,
    "gs_target": "OffAxisRingAndSpot",
    "gs_lensless": false,
    "gs_iterations": 50,
    "gs_seed": 200142514494,
    "high_contrast_colormap": false,
    "is_playing": false
  }
}"#;

/// Starter preset 2: Periodic Ronchi grating fractal Talbot carpet I(x, z) across [0, z_T].
pub const PRESET_TALBOT_CARPET: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Fourier Optics Workbench",
    "domain": "optics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "HelmholtzDiffraction",
    "aperture": "RonchiGrating",
    "display": "AxialCarpetXZ",
    "wavelength_nm": 500.0,
    "pixel_pitch_um": 8.0,
    "propagation_mm": 20.48,
    "carpet_max_mm": 40.96,
    "feature_scale_um": 128.0,
    "slit_pitch_um": 128.0,
    "slit_count": 2,
    "duty_cycle": 0.5,
    "focal_length_mm": 14.0,
    "zernike_n": 0,
    "zernike_m": 0,
    "zernike_rms_waves": 0.0,
    "gs_target": "OffAxisRingAndSpot",
    "gs_lensless": false,
    "gs_iterations": 50,
    "gs_seed": 200142514494,
    "high_contrast_colormap": false,
    "is_playing": false
  }
}"#;

/// Starter preset 3: Thin quadratic lens focus degraded by Zernike primary spherical aberration Z_4^0.
pub const PRESET_ZERNIKE_ABERRATED_LENS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Fourier Optics Workbench",
    "domain": "optics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "HelmholtzDiffraction",
    "aperture": "ThinQuadraticLens",
    "display": "IntensityXY",
    "wavelength_nm": 532.0,
    "pixel_pitch_um": 8.0,
    "propagation_mm": 12.0,
    "carpet_max_mm": 24.0,
    "feature_scale_um": 240.0,
    "slit_pitch_um": 96.0,
    "slit_count": 2,
    "duty_cycle": 0.5,
    "focal_length_mm": 12.0,
    "zernike_n": 4,
    "zernike_m": 0,
    "zernike_rms_waves": 0.45,
    "gs_target": "OffAxisRingAndSpot",
    "gs_lensless": false,
    "gs_iterations": 50,
    "gs_seed": 200142514494,
    "high_contrast_colormap": false,
    "is_playing": false
  }
}"#;

/// Starter preset 4: Gerchberg-Saxton phase-only SLM hologram breaking conjugate twin-image symmetry.
pub const PRESET_GERCHBERG_SAXTON_HOLOGRAM: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Fourier Optics Workbench",
    "domain": "optics",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "GerchbergSaxtonHolography",
    "aperture": "OpaqueCircularDisk",
    "display": "IntensityXY",
    "wavelength_nm": 532.0,
    "pixel_pitch_um": 8.0,
    "propagation_mm": 18.0,
    "carpet_max_mm": 40.0,
    "feature_scale_um": 128.0,
    "slit_pitch_um": 96.0,
    "slit_count": 2,
    "duty_cycle": 0.5,
    "focal_length_mm": 14.0,
    "zernike_n": 0,
    "zernike_m": 0,
    "zernike_rms_waves": 0.0,
    "gs_target": "OffAxisRingAndSpot",
    "gs_lensless": false,
    "gs_iterations": 50,
    "gs_seed": 200142514494,
    "high_contrast_colormap": false,
    "is_playing": false
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "optics_arago_bright_spot_starter",
        title: "Arago-Poisson Bright Spot",
        description: "Constructive on-axis diffraction peak inside the geometric shadow of an opaque circular disk at Fresnel number N_F = 2.0.",
        category: "Wave Diffraction",
        icon: "🎯",
        domain: "optics",
        tool_name: Some("FourierOpticsWorkbenchTool"),
        preset_generator: || PRESET_ARAGO_BRIGHT_SPOT,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "optics_talbot_carpet_starter",
        title: "Fractal Talbot Self-Imaging Carpet",
        description: "Near-field axial (x, z) intensity carpet of a periodic Ronchi grating across full and fractional Talbot revivals.",
        category: "Self-Imaging",
        icon: "🧬",
        domain: "optics",
        tool_name: Some("FourierOpticsWorkbenchTool"),
        preset_generator: || PRESET_TALBOT_CARPET,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "optics_zernike_lens_starter",
        title: "Zernike-Aberrated Focusing Lens",
        description: "Thin quadratic focusing lens perturbed by primary spherical wavefront aberration (n = 4, m = 0) at focal plane z = f.",
        category: "Wavefront Aberrations",
        icon: "🔍",
        domain: "optics",
        tool_name: Some("FourierOpticsWorkbenchTool"),
        preset_generator: || PRESET_ZERNIKE_ABERRATED_LENS,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "optics_gerchberg_saxton_hologram_starter",
        title: "Gerchberg-Saxton Phase Hologram",
        description: "Iterative unit-modulus SLM phase retrieval suppressing conjugate twin-image leakage well below the 0.707 real-scalar floor.",
        category: "Phase Holography",
        icon: "🔮",
        domain: "optics",
        tool_name: Some("FourierOpticsWorkbenchTool"),
        preset_generator: || PRESET_GERCHBERG_SAXTON_HOLOGRAM,
    }
}
